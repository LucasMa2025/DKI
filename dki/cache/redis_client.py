"""
Redis Client Wrapper for DKI
Provides async Redis client with connection pool and health checks

Features:
- Connection pooling for high concurrency
- Automatic reconnection
- Health check support
- Configurable via YAML

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
import pickle
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from loguru import logger

try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool, Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    ConnectionPool = None
    Redis = None


@dataclass
class RedisConfig:
    """
    Redis configuration
    
    Can be loaded from config.yaml or passed directly
    """
    # Connection settings
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    
    # Connection pool settings
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    
    # Key settings
    key_prefix: str = "dki"
    default_ttl: int = 86400  # 24 hours
    
    # Compression settings
    enable_compression: bool = True
    compression_level: int = 6
    compression_threshold: int = 1024  # Only compress if > 1KB
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', False),
            host=data.get('host', 'localhost'),
            port=data.get('port', 6379),
            password=data.get('password', ''),
            db=data.get('db', 0),
            max_connections=data.get('max_connections', 50),
            socket_timeout=data.get('socket_timeout', 5.0),
            socket_connect_timeout=data.get('socket_connect_timeout', 5.0),
            retry_on_timeout=data.get('retry_on_timeout', True),
            key_prefix=data.get('key_prefix', 'dki'),
            default_ttl=data.get('default_ttl', 86400),
            enable_compression=data.get('enable_compression', True),
            compression_level=data.get('compression_level', 6),
            compression_threshold=data.get('compression_threshold', 1024),
            health_check_interval=data.get('health_check_interval', 30),
        )


class DKIRedisClient:
    """
    DKI Redis Client
    
    Async Redis client with:
    - Connection pooling
    - Automatic compression
    - Health monitoring
    - Graceful degradation
    
    Usage:
    ```python
    # Initialize
    config = RedisConfig(enabled=True, host="localhost")
    client = DKIRedisClient(config)
    await client.connect()
    
    # Use
    await client.set("key", {"data": "value"}, ttl=3600)
    data = await client.get("key")
    
    # Cleanup
    await client.close()
    ```
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis client
        
        Args:
            config: Redis configuration
        """
        self.config = config or RedisConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._connected = False
        self._health_task: Optional[asyncio.Task] = None
        
        # Stats
        self._stats = {
            "connections": 0,
            "disconnections": 0,
            "operations": 0,
            "errors": 0,
            "compressions": 0,
            "decompressions": 0,
        }
    
    @property
    def is_available(self) -> bool:
        """Check if Redis is available and enabled"""
        return REDIS_AVAILABLE and self.config.enabled and self._connected
    
    @property
    def client(self) -> Optional[Redis]:
        """Get the underlying Redis client"""
        return self._client
    
    async def connect(self) -> bool:
        """
        Connect to Redis
        
        Returns:
            True if connected successfully
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not installed. Install with: pip install redis")
            return False
        
        if not self.config.enabled:
            logger.info("Redis is disabled in configuration")
            return False
        
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password or None,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=False,  # We handle binary data
            )
            
            # Create client
            self._client = Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            
            self._connected = True
            self._stats["connections"] += 1
            
            # Start health check
            if self.config.health_check_interval > 0:
                self._health_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(
                f"Connected to Redis at {self.config.host}:{self.config.port} "
                f"(db={self.config.db}, pool_size={self.config.max_connections})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            self._stats["errors"] += 1
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        if self._client:
            await self._client.close()
        
        if self._pool:
            await self._pool.disconnect()
        
        self._connected = False
        self._stats["disconnections"] += 1
        logger.info("Redis connection closed")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self._client:
                    await self._client.ping()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                self._connected = False
                
                # Try to reconnect
                try:
                    await self.connect()
                except Exception:
                    pass
    
    def _make_key(self, key: str) -> str:
        """Create full key with prefix"""
        return f"{self.config.key_prefix}:{key}"
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold"""
        if (
            self.config.enable_compression
            and len(data) > self.config.compression_threshold
        ):
            compressed = zlib.compress(data, level=self.config.compression_level)
            self._stats["compressions"] += 1
            # Add marker to indicate compression
            return b'\x01' + compressed
        return b'\x00' + data
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data if compressed"""
        if not data:
            return data
        
        marker = data[0:1]
        payload = data[1:]
        
        if marker == b'\x01':
            self._stats["decompressions"] += 1
            return zlib.decompress(payload)
        return payload
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            Deserialized value or None
        """
        if not self.is_available:
            return None
        
        try:
            full_key = self._make_key(key)
            data = await self._client.get(full_key)
            
            if data is None:
                return None
            
            self._stats["operations"] += 1
            
            # Decompress and deserialize
            decompressed = self._decompress(data)
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.warning(f"Redis GET error for key {key}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in Redis
        
        Args:
            key: Cache key (without prefix)
            value: Value to store (will be pickled)
            ttl: Time to live in seconds (default from config)
            
        Returns:
            True if successful
        """
        if not self.is_available:
            return False
        
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.config.default_ttl
            
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = self._compress(serialized)
            
            await self._client.setex(full_key, ttl, compressed)
            
            self._stats["operations"] += 1
            return True
            
        except Exception as e:
            logger.warning(f"Redis SET error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from Redis
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            True if deleted
        """
        if not self.is_available:
            return False
        
        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            self._stats["operations"] += 1
            return result > 0
            
        except Exception as e:
            logger.warning(f"Redis DELETE error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching pattern
        
        Args:
            pattern: Key pattern (without prefix, supports *)
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            count = 0
            
            # Use SCAN to find keys (safer than KEYS for large datasets)
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match=full_pattern,
                    count=100,
                )
                
                if keys:
                    await self._client.delete(*keys)
                    count += len(keys)
                
                if cursor == 0:
                    break
            
            self._stats["operations"] += 1
            return count
            
        except Exception as e:
            logger.warning(f"Redis DELETE_PATTERN error for {pattern}: {e}")
            self._stats["errors"] += 1
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.is_available:
            return False
        
        try:
            full_key = self._make_key(key)
            return await self._client.exists(full_key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL of key in seconds"""
        if not self.is_available:
            return -1
        
        try:
            full_key = self._make_key(key)
            return await self._client.ttl(full_key)
        except Exception as e:
            logger.warning(f"Redis TTL error for key {key}: {e}")
            return -1
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key"""
        if not self.is_available:
            return False
        
        try:
            full_key = self._make_key(key)
            return await self._client.expire(full_key, ttl)
        except Exception as e:
            logger.warning(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def get_raw(self, key: str) -> Optional[bytes]:
        """Get raw bytes from Redis (no deserialization)"""
        if not self.is_available:
            return None
        
        try:
            full_key = self._make_key(key)
            data = await self._client.get(full_key)
            if data:
                return self._decompress(data)
            return None
        except Exception as e:
            logger.warning(f"Redis GET_RAW error for key {key}: {e}")
            return None
    
    async def set_raw(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set raw bytes in Redis (no serialization)"""
        if not self.is_available:
            return False
        
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.config.default_ttl
            compressed = self._compress(value)
            await self._client.setex(full_key, ttl, compressed)
            return True
        except Exception as e:
            logger.warning(f"Redis SET_RAW error for key {key}: {e}")
            return False
    
    async def ping(self) -> bool:
        """Ping Redis server"""
        if not self._client:
            return False
        
        try:
            await self._client.ping()
            return True
        except Exception:
            return False
    
    async def info(self) -> Dict[str, Any]:
        """Get Redis server info"""
        if not self.is_available:
            return {}
        
        try:
            info = await self._client.info()
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_connections_received": info.get("total_connections_received"),
                "total_commands_processed": info.get("total_commands_processed"),
            }
        except Exception as e:
            logger.warning(f"Redis INFO error: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "enabled": self.config.enabled,
            "connected": self._connected,
            "host": f"{self.config.host}:{self.config.port}",
            "db": self.config.db,
            **self._stats,
        }


# Singleton instance for global access
_global_redis_client: Optional[DKIRedisClient] = None


async def get_redis_client(config: Optional[RedisConfig] = None) -> DKIRedisClient:
    """
    Get or create global Redis client
    
    Args:
        config: Redis configuration (only used on first call)
        
    Returns:
        DKIRedisClient instance
    """
    global _global_redis_client
    
    if _global_redis_client is None:
        _global_redis_client = DKIRedisClient(config)
        await _global_redis_client.connect()
    
    return _global_redis_client


async def close_redis_client():
    """Close global Redis client"""
    global _global_redis_client
    
    if _global_redis_client:
        await _global_redis_client.close()
        _global_redis_client = None
