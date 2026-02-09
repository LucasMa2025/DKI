"""
Preference Cache Manager
Tiered caching system for user preference K/V data

Architecture:
- L1 (In-Memory): Active session preferences, < 1ms latency
- L2 (Redis): Recently active users, 1-5ms latency
- L3 (Recompute): Cold users, compute on demand

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
import hashlib
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger


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
    l2_enabled: bool = True
    l2_ttl_seconds: int = 86400  # 24 hours
    l2_key_prefix: str = "dki:pref_kv"
    
    # General settings
    enable_compression: bool = True
    compression_level: int = 6


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
    - L1 (Memory): Hot data for active sessions
    - L2 (Redis): Warm data for recently active users
    - L3 (Recompute): Cold data computed on demand
    
    Cache Key Format: {user_id}:{preference_hash}
    
    Example:
        cache = PreferenceCacheManager(redis_client=redis)
        
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
        redis_client: Optional[Any] = None,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client (async or sync)
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.redis = redis_client
        
        # L1 cache (in-memory)
        self._l1_cache = LRUCache(maxsize=self.config.l1_max_size)
        
        # Stats
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_computes": 0,
            "total_requests": 0,
            "invalidations": 0,
        }
        
        logger.info(
            f"PreferenceCacheManager initialized "
            f"(L1 max={self.config.l1_max_size}, "
            f"L2 enabled={self.config.l2_enabled})"
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
        
        Args:
            user_id: User identifier
            preference_text: Preference text to compute K/V for
            model: Model adapter for K/V computation
            force_recompute: Skip cache and recompute
            
        Returns:
            (kv_entries, tier_info)
        """
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        preference_hash = self._compute_preference_hash(preference_text)
        cache_key = self._make_cache_key(user_id, preference_hash)
        
        # Skip cache if forced
        if force_recompute:
            kv_entries = await self._compute_kv(preference_text, model)
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
        if self.config.l2_enabled and self.redis:
            redis_key = self._make_redis_key(cache_key)
            
            try:
                cached_data = await self._redis_get(redis_key)
                
                if cached_data:
                    # Deserialize
                    kv_entries = self._deserialize_kv(cached_data)
                    
                    # Promote to L1
                    await self._l1_cache.put(cache_key, CacheEntry(
                        kv_data=kv_entries,
                        preference_hash=preference_hash,
                        created_at=time.time(),
                        last_accessed=time.time(),
                    ))
                    
                    self._stats["l2_hits"] += 1
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return kv_entries, CacheTierInfo(
                        tier=CacheTier.L2_REDIS,
                        latency_ms=latency_ms,
                        hit=True,
                        user_id=user_id,
                        preference_hash=preference_hash,
                    )
                    
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
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
        
        # L1: Store in memory
        await self._l1_cache.put(cache_key, CacheEntry(
            kv_data=kv_entries,
            preference_hash=preference_hash,
            created_at=now,
            last_accessed=now,
        ))
        
        # L2: Store in Redis
        if self.config.l2_enabled and self.redis:
            try:
                redis_key = self._make_redis_key(cache_key)
                serialized = self._serialize_kv(kv_entries)
                
                await self._redis_setex(
                    redis_key,
                    self.config.l2_ttl_seconds,
                    serialized,
                )
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")
    
    async def invalidate(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a user.
        
        Call this when user preferences are updated.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of entries invalidated
        """
        self._stats["invalidations"] += 1
        count = 0
        
        # L1: Clear from memory
        prefix = f"{user_id}:"
        l1_count = await self._l1_cache.delete_prefix(prefix)
        count += l1_count
        
        # L2: Clear from Redis
        if self.config.l2_enabled and self.redis:
            try:
                pattern = f"{self.config.l2_key_prefix}:{user_id}:*"
                l2_count = await self._redis_delete_pattern(pattern)
                count += l2_count
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache: {e}")
        
        logger.debug(f"Invalidated {count} cache entries for user {user_id}")
        return count
    
    async def clear_all(self) -> None:
        """Clear all cache entries."""
        await self._l1_cache.clear()
        
        if self.config.l2_enabled and self.redis:
            try:
                pattern = f"{self.config.l2_key_prefix}:*"
                await self._redis_delete_pattern(pattern)
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")
        
        logger.info("Cleared all preference caches")
    
    def _serialize_kv(self, kv_entries: Any) -> bytes:
        """Serialize K/V entries for storage."""
        try:
            import zlib
            
            # Convert to CPU and serialize
            serializable = []
            for entry in kv_entries:
                serializable.append({
                    'key': entry.key.cpu().numpy().tobytes(),
                    'value': entry.value.cpu().numpy().tobytes(),
                    'layer_idx': entry.layer_idx,
                    'shape': list(entry.key.shape),
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
                shape = tuple(item['shape'])
                key = torch.from_numpy(
                    np.frombuffer(item['key'], dtype=np.float16).reshape(shape)
                )
                value = torch.from_numpy(
                    np.frombuffer(item['value'], dtype=np.float16).reshape(shape)
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
    
    # Redis helper methods (handle both async and sync clients)
    
    async def _redis_get(self, key: str) -> Optional[bytes]:
        """Get from Redis."""
        if hasattr(self.redis, 'get'):
            result = self.redis.get(key)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return None
    
    async def _redis_setex(self, key: str, ttl: int, value: bytes) -> None:
        """Set with expiration in Redis."""
        if hasattr(self.redis, 'setex'):
            result = self.redis.setex(key, ttl, value)
            if asyncio.iscoroutine(result):
                await result
    
    async def _redis_delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        count = 0
        
        if hasattr(self.redis, 'scan_iter'):
            # Async redis
            if hasattr(self.redis, 'scan'):
                cursor = 0
                while True:
                    result = self.redis.scan(cursor, match=pattern, count=100)
                    if asyncio.iscoroutine(result):
                        cursor, keys = await result
                    else:
                        cursor, keys = result
                    
                    for key in keys:
                        del_result = self.redis.delete(key)
                        if asyncio.iscoroutine(del_result):
                            await del_result
                        count += 1
                    
                    if cursor == 0:
                        break
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["total_requests"]
        
        return {
            "total_requests": total,
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "l3_computes": self._stats["l3_computes"],
            "invalidations": self._stats["invalidations"],
            "l1_hit_rate": self._stats["l1_hits"] / total if total > 0 else 0,
            "l2_hit_rate": self._stats["l2_hits"] / total if total > 0 else 0,
            "overall_hit_rate": (self._stats["l1_hits"] + self._stats["l2_hits"]) / total if total > 0 else 0,
            "l1_cache": self._l1_cache.get_stats(),
        }
