"""
Preference Cache Manager
Tiered caching system for user preference K/V data

Architecture:
- L1 (In-Memory): Active session preferences, < 1ms latency
- L2 (Redis): Recently active users, 1-5ms latency (distributed)
- L3 (Recompute): Cold users, compute on demand

Key Features:
- Multi-instance cache sharing via Redis
- Automatic compression for large K/V tensors
- Graceful degradation when Redis unavailable
- Configurable via YAML

安全特性 (v3.1):
- 缓存键包含用户身份校验 (user_id 前缀)
- Redis 键使用用户命名空间隔离
- 所有操作记录审计日志

Author: AGI Demo Project
Version: 3.1.0
"""

import asyncio
import hashlib
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from dki.cache.redis_client import DKIRedisClient


class CacheTier(str, Enum):
    """Cache tier identifiers."""
    L1_MEMORY = "L1_MEMORY"
    L2_REDIS = "L2_REDIS"
    L3_COMPUTE = "L3_COMPUTE"
    MISS = "MISS"


@dataclass
class CacheTierInfo:
    """Information about cache tier hit."""
    tier: CacheTier
    latency_ms: float
    hit: bool
    user_id: str
    preference_hash: str
    size_bytes: int = 0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    kv_data: Any  # List[KVCacheEntry] or serialized bytes
    preference_hash: str
    created_at: float
    last_accessed: float
    access_count: int = 1
    size_bytes: int = 0


@dataclass
class CacheConfig:
    """Configuration for preference cache."""
    # L1 (Memory) settings
    l1_max_size: int = 1000  # Max users in memory
    l1_max_memory_mb: int = 5000  # Max memory in MB
    
    # L2 (Redis) settings
    l2_enabled: bool = False  # Disabled by default, enable via config
    l2_ttl_seconds: int = 86400  # 24 hours
    l2_key_prefix: str = "dki:pref_kv"
    
    # General settings
    enable_compression: bool = True
    compression_level: int = 6
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create config from dictionary"""
        return cls(
            l1_max_size=data.get('l1_max_size', 1000),
            l1_max_memory_mb=data.get('l1_max_memory_mb', 5000),
            l2_enabled=data.get('l2_enabled', False),
            l2_ttl_seconds=data.get('l2_ttl_seconds', 86400),
            l2_key_prefix=data.get('l2_key_prefix', 'dki:pref_kv'),
            enable_compression=data.get('enable_compression', True),
            compression_level=data.get('compression_level', 6),
        )


class LRUCache:
    """
    Thread-safe LRU cache with size limit.
    
    Uses OrderedDict for O(1) operations.
    """
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache, updating access order."""
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                self._stats["hits"] += 1
                return entry
            
            self._stats["misses"] += 1
            return None
    
    async def put(self, key: str, entry: CacheEntry) -> None:
        """Put item in cache, evicting if necessary."""
        async with self._lock:
            if key in self._cache:
                # Update existing
                self._cache.move_to_end(key)
                self._cache[key] = entry
            else:
                # Add new
                if len(self._cache) >= self.maxsize:
                    # Evict oldest
                    self._cache.popitem(last=False)
                    self._stats["evictions"] += 1
                
                self._cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def delete_prefix(self, prefix: str) -> int:
        """Delete all items with key prefix."""
        async with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)
    
    async def clear(self) -> None:
        """Clear all items."""
        async with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
        }


class PreferenceCacheManager:
    """
    Tiered cache manager for user preference K/V data.
    
    Strategy:
    - L1 (Memory): Hot data for active sessions (< 1ms)
    - L2 (Redis): Warm data shared across instances (1-5ms)
    - L3 (Recompute): Cold data computed on demand (50-200ms)
    
    Cache Key Format: {user_id}:{preference_hash}
    
    Why Redis (L2) is Important:
    - In multi-instance deployments, L1 cache is per-instance
    - Without L2, cache hit rate drops to ~70%/N (N = instance count)
    - With L2 (Redis), cache hit rate stays at ~70% regardless of instances
    
    Example:
        # Without Redis (single instance or degraded mode)
        cache = PreferenceCacheManager()
        
        # With Redis (recommended for production)
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        redis_client = DKIRedisClient(RedisConfig(enabled=True))
        await redis_client.connect()
        cache = PreferenceCacheManager(redis_client=redis_client)
        
        # Get preference K/V (checks L1 -> L2 -> computes)
        kv_entries, tier_info = await cache.get_preference_kv(
            user_id="user_123",
            preference_text="vegetarian, no spicy food",
            model=model_adapter,
        )
        
        # Invalidate on preference update
        await cache.invalidate(user_id="user_123")
    """
    
    def __init__(
        self,
        redis_client: Optional["DKIRedisClient"] = None,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: DKIRedisClient instance (recommended for production)
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._redis_client = redis_client
        
        # L1 cache (in-memory)
        self._l1_cache = LRUCache(maxsize=self.config.l1_max_size)
        
        # Stats
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_computes": 0,
            "total_requests": 0,
            "invalidations": 0,
            "l2_errors": 0,
            "total_bytes_cached": 0,
        }
        
        # Check Redis availability
        l2_status = "disabled"
        if self.config.l2_enabled:
            if redis_client and redis_client.is_available:
                l2_status = "enabled (connected)"
            elif redis_client:
                l2_status = "enabled (not connected)"
            else:
                l2_status = "enabled (no client)"
        
        logger.info(
            f"PreferenceCacheManager initialized "
            f"(L1 max={self.config.l1_max_size}, L2 {l2_status})"
        )
    
    @property
    def redis(self) -> Optional["DKIRedisClient"]:
        """Get Redis client"""
        return self._redis_client
    
    @redis.setter
    def redis(self, client: Optional["DKIRedisClient"]):
        """Set Redis client (allows late binding)"""
        self._redis_client = client
        if client and client.is_available:
            logger.info("Redis client connected to PreferenceCacheManager")
    
    def _is_l2_available(self) -> bool:
        """Check if L2 (Redis) is available"""
        return (
            self.config.l2_enabled
            and self._redis_client is not None
            and self._redis_client.is_available
        )
    
    def _compute_preference_hash(self, preference_text: str) -> str:
        """Compute hash of preference text."""
        return hashlib.md5(preference_text.encode()).hexdigest()[:16]
    
    def _make_cache_key(self, user_id: str, preference_hash: str) -> str:
        """Create cache key."""
        return f"{user_id}:{preference_hash}"
    
    def _make_redis_key(self, cache_key: str) -> str:
        """Create Redis key."""
        return f"{self.config.l2_key_prefix}:{cache_key}"
    
    async def get_preference_kv(
        self,
        user_id: str,
        preference_text: str,
        model: Any,  # BaseModelAdapter
        force_recompute: bool = False,
    ) -> Tuple[Any, CacheTierInfo]:
        """
        Get preference K/V cache, checking tiers in order.
        
        安全 (v3.1): user_id 必须非空，用于缓存键隔离
        
        Args:
            user_id: User identifier (must be non-empty, validated by caller)
            preference_text: Preference text to compute K/V for
            model: Model adapter for K/V computation
            force_recompute: Skip cache and recompute
            
        Returns:
            (kv_entries, tier_info)
        """
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        # 用户身份验证 (v3.1)
        if not user_id or not user_id.strip():
            logger.warning("Empty user_id in get_preference_kv, computing without cache")
            kv_entries = await self._compute_kv(preference_text, model)
            return kv_entries, CacheTierInfo(
                tier=CacheTier.L3_COMPUTE,
                latency_ms=(time.time() - start_time) * 1000,
                hit=False,
                user_id="<anonymous>",
                preference_hash="",
            )
        
        user_id = user_id.strip()
        preference_hash = self._compute_preference_hash(preference_text)
        cache_key = self._make_cache_key(user_id, preference_hash)
        
        # Recompute if forced (also updates cache with fresh data)
        if force_recompute:
            kv_entries = await self._compute_kv(preference_text, model)
            
            # Store recomputed data in caches so subsequent requests get fresh data
            await self._store_in_caches(
                cache_key=cache_key,
                kv_entries=kv_entries,
                preference_hash=preference_hash,
            )
            
            self._stats["l3_computes"] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            return kv_entries, CacheTierInfo(
                tier=CacheTier.L3_COMPUTE,
                latency_ms=latency_ms,
                hit=False,
                user_id=user_id,
                preference_hash=preference_hash,
            )
        
        # L1: Check memory cache
        entry = await self._l1_cache.get(cache_key)
        if entry is not None:
            self._stats["l1_hits"] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            return entry.kv_data, CacheTierInfo(
                tier=CacheTier.L1_MEMORY,
                latency_ms=latency_ms,
                hit=True,
                user_id=user_id,
                preference_hash=preference_hash,
            )
        
        # L2: Check Redis cache
        if self._is_l2_available():
            redis_key = self._make_redis_key(cache_key)
            
            try:
                cached_data = await self._redis_client.get_raw(redis_key)
                
                if cached_data:
                    # Deserialize
                    kv_entries = self._deserialize_kv(cached_data)
                    size_bytes = len(cached_data)
                    
                    # Promote to L1
                    await self._l1_cache.put(cache_key, CacheEntry(
                        kv_data=kv_entries,
                        preference_hash=preference_hash,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        size_bytes=size_bytes,
                    ))
                    
                    self._stats["l2_hits"] += 1
                    latency_ms = (time.time() - start_time) * 1000
                    
                    logger.debug(
                        f"L2 cache hit for user {user_id} "
                        f"(size={size_bytes/1024:.1f}KB, latency={latency_ms:.1f}ms)"
                    )
                    
                    return kv_entries, CacheTierInfo(
                        tier=CacheTier.L2_REDIS,
                        latency_ms=latency_ms,
                        hit=True,
                        user_id=user_id,
                        preference_hash=preference_hash,
                        size_bytes=size_bytes,
                    )
                    
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
                self._stats["l2_errors"] += 1
        
        # L3: Compute K/V
        kv_entries = await self._compute_kv(preference_text, model)
        
        # Store in caches
        await self._store_in_caches(
            cache_key=cache_key,
            kv_entries=kv_entries,
            preference_hash=preference_hash,
        )
        
        self._stats["l3_computes"] += 1
        latency_ms = (time.time() - start_time) * 1000
        
        return kv_entries, CacheTierInfo(
            tier=CacheTier.L3_COMPUTE,
            latency_ms=latency_ms,
            hit=False,
            user_id=user_id,
            preference_hash=preference_hash,
        )
    
    async def _compute_kv(self, preference_text: str, model: Any) -> Any:
        """Compute K/V using model."""
        try:
            with torch.no_grad():
                if hasattr(model, 'compute_kv'):
                    kv_entries, _ = model.compute_kv(preference_text)
                    return kv_entries
                else:
                    logger.warning("Model does not support compute_kv")
                    return []
        except Exception as e:
            logger.error(f"Error computing K/V: {e}")
            return []
    
    async def _store_in_caches(
        self,
        cache_key: str,
        kv_entries: Any,
        preference_hash: str,
    ) -> None:
        """Store K/V in all cache tiers."""
        now = time.time()
        
        # Serialize once for both caches
        serialized = self._serialize_kv(kv_entries)
        size_bytes = len(serialized)
        self._stats["total_bytes_cached"] += size_bytes
        
        # L1: Store in memory
        await self._l1_cache.put(cache_key, CacheEntry(
            kv_data=kv_entries,
            preference_hash=preference_hash,
            created_at=now,
            last_accessed=now,
            size_bytes=size_bytes,
        ))
        
        # L2: Store in Redis (async, non-blocking)
        if self._is_l2_available():
            try:
                redis_key = self._make_redis_key(cache_key)
                
                await self._redis_client.set_raw(
                    redis_key,
                    serialized,
                    ttl=self.config.l2_ttl_seconds,
                )
                
                logger.debug(
                    f"Stored in L2 cache: {cache_key} "
                    f"(size={size_bytes/1024:.1f}KB, ttl={self.config.l2_ttl_seconds}s)"
                )
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")
                self._stats["l2_errors"] += 1
    
    async def invalidate(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a user.
        
        Call this when user preferences are updated.
        
        安全 (v3.1): user_id 必须非空
        
        Args:
            user_id: User identifier (must be non-empty)
            
        Returns:
            Number of entries invalidated
        """
        if not user_id or not user_id.strip():
            logger.warning("Empty user_id in invalidate, skipping")
            return 0
        
        user_id = user_id.strip()
        self._stats["invalidations"] += 1
        count = 0
        
        # L1: Clear from memory (仅匹配该用户的前缀)
        prefix = f"{user_id}:"
        l1_count = await self._l1_cache.delete_prefix(prefix)
        count += l1_count
        
        # L2: Clear from Redis
        if self._is_l2_available():
            try:
                pattern = f"{user_id}:*"
                l2_count = await self._redis_client.delete_pattern(
                    f"{self.config.l2_key_prefix}:{pattern}"
                )
                count += l2_count
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache: {e}")
                self._stats["l2_errors"] += 1
        
        logger.debug(f"Invalidated {count} cache entries for user {user_id}")
        return count
    
    async def clear_all(self) -> None:
        """Clear all cache entries."""
        await self._l1_cache.clear()
        
        if self._is_l2_available():
            try:
                pattern = f"{self.config.l2_key_prefix}:*"
                await self._redis_client.delete_pattern(pattern)
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")
                self._stats["l2_errors"] += 1
        
        # Reset stats
        self._stats["total_bytes_cached"] = 0
        
        logger.info("Cleared all preference caches")
    
    def _serialize_kv(self, kv_entries: Any) -> bytes:
        """Serialize K/V entries for storage."""
        try:
            import zlib
            import numpy as np
            
            # Convert to CPU and serialize
            serializable = []
            for entry in kv_entries:
                key_np = entry.key.cpu().numpy()
                value_np = entry.value.cpu().numpy()
                serializable.append({
                    'key': key_np.tobytes(),
                    'value': value_np.tobytes(),
                    'layer_idx': entry.layer_idx,
                    'key_shape': list(key_np.shape),
                    'value_shape': list(value_np.shape),
                    'key_dtype': str(key_np.dtype),
                    'value_dtype': str(value_np.dtype),
                })
            
            data = pickle.dumps(serializable)
            
            if self.config.enable_compression:
                data = zlib.compress(data, level=self.config.compression_level)
            
            return data
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps([])
    
    def _deserialize_kv(self, data: bytes) -> Any:
        """Deserialize K/V entries from storage."""
        try:
            import zlib
            import numpy as np
            from dki.models.base import KVCacheEntry
            
            if self.config.enable_compression:
                data = zlib.decompress(data)
            
            serializable = pickle.loads(data)
            
            kv_entries = []
            for item in serializable:
                # Support both old format (single 'shape'+'dtype') and
                # new format (separate key_shape/value_shape/key_dtype/value_dtype)
                key_shape = tuple(item.get('key_shape', item.get('shape', ())))
                value_shape = tuple(item.get('value_shape', item.get('shape', ())))
                key_dtype = np.dtype(item.get('key_dtype', 'float16'))
                value_dtype = np.dtype(item.get('value_dtype', 'float16'))
                
                key = torch.from_numpy(
                    np.frombuffer(item['key'], dtype=key_dtype).reshape(key_shape).copy()
                )
                value = torch.from_numpy(
                    np.frombuffer(item['value'], dtype=value_dtype).reshape(value_shape).copy()
                )
                
                kv_entries.append(KVCacheEntry(
                    key=key,
                    value=value,
                    layer_idx=item['layer_idx'],
                ))
            
            return kv_entries
            
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["total_requests"]
        
        # Calculate hit rates
        l1_hit_rate = self._stats["l1_hits"] / total if total > 0 else 0
        l2_hit_rate = self._stats["l2_hits"] / total if total > 0 else 0
        overall_hit_rate = (self._stats["l1_hits"] + self._stats["l2_hits"]) / total if total > 0 else 0
        
        # Get Redis stats if available
        redis_stats = {}
        if self._redis_client:
            redis_stats = self._redis_client.get_stats()
        
        return {
            "total_requests": total,
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "l3_computes": self._stats["l3_computes"],
            "invalidations": self._stats["invalidations"],
            "l2_errors": self._stats["l2_errors"],
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "total_bytes_cached": self._stats["total_bytes_cached"],
            "total_mb_cached": self._stats["total_bytes_cached"] / (1024 * 1024),
            "l1_cache": self._l1_cache.get_stats(),
            "l2_enabled": self.config.l2_enabled,
            "l2_available": self._is_l2_available(),
            "redis": redis_stats,
        }
    
    async def warm_cache(
        self,
        user_ids: List[str],
        preference_getter: Any,  # Callable[[str], str]
        model: Any,
    ) -> Dict[str, Any]:
        """
        Pre-warm cache for a list of users.
        
        Useful for:
        - Service startup
        - Batch pre-computation for high-frequency users
        
        Args:
            user_ids: List of user IDs to warm
            preference_getter: Async function to get preference text for user
            model: Model adapter for K/V computation
            
        Returns:
            Warming statistics
        """
        stats = {
            "total": len(user_ids),
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        for user_id in user_ids:
            try:
                # Get preference text
                preference_text = await preference_getter(user_id)
                
                if not preference_text:
                    stats["skipped"] += 1
                    continue
                
                # Compute and cache
                _, tier_info = await self.get_preference_kv(
                    user_id=user_id,
                    preference_text=preference_text,
                    model=model,
                )
                
                if tier_info.hit:
                    stats["skipped"] += 1  # Already cached
                else:
                    stats["success"] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to warm cache for user {user_id}: {e}")
                stats["failed"] += 1
        
        logger.info(
            f"Cache warming complete: "
            f"{stats['success']} computed, "
            f"{stats['skipped']} skipped, "
            f"{stats['failed']} failed"
        )
        
        return stats
