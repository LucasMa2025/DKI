"""
DKI Cache System
Provides tiered caching for user preferences and K/V data

This module provides:
- PreferenceCacheManager: Tiered cache for user preference K/V
- NonVectorizedDataHandler: Handler for non-vectorized message data
- DKIRedisClient: Redis client for distributed caching
- RedisConfig: Redis configuration

Architecture:
- L1 (Memory): Per-instance hot cache, < 1ms latency
- L2 (Redis): Distributed warm cache, 1-5ms latency
- L3 (Recompute): Cold data, 50-200ms latency

Why Redis:
- Multi-instance deployments need shared cache
- Without Redis: cache hit rate = 70%/N (N = instances)
- With Redis: cache hit rate = 70% (constant)
"""

from dki.cache.preference_cache import (
    PreferenceCacheManager,
    CacheTierInfo,
    CacheConfig,
    CacheTier,
)
from dki.cache.non_vectorized_handler import NonVectorizedDataHandler
from dki.cache.redis_client import (
    DKIRedisClient,
    RedisConfig,
    get_redis_client,
    close_redis_client,
    REDIS_AVAILABLE,
)

__all__ = [
    # Preference Cache
    "PreferenceCacheManager",
    "CacheTierInfo",
    "CacheConfig",
    "CacheTier",
    # Non-Vectorized Handler
    "NonVectorizedDataHandler",
    # Redis
    "DKIRedisClient",
    "RedisConfig",
    "get_redis_client",
    "close_redis_client",
    "REDIS_AVAILABLE",
]
