"""
DKI Cache System
Provides tiered caching for user preferences and K/V data

This module provides:
- PreferenceCacheManager: Tiered cache for user preference K/V
- NonVectorizedDataHandler: Handler for non-vectorized message data
- DKIRedisClient: Redis client for distributed caching
- RedisConfig: Redis configuration
- UserIsolationContext: User-level isolation context (v3.1)
- IsolatedPreferenceCacheManager: User-isolated preference cache (v3.1)

Architecture:
- L1 (Memory): Per-instance hot cache, < 1ms latency
- L2 (Redis): Distributed warm cache, 1-5ms latency
- L3 (Recompute): Cold data, 50-200ms latency

安全架构 (v3.1):
- 用户级物理分区: 每个用户独立 OrderedDict
- HMAC 签名缓存键: 防止键名猜测/构造攻击
- 推理上下文隔离: InferenceContextGuard 确保 K/V 不残留
- 审计日志: 所有缓存操作可追溯

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

# 用户隔离组件 (v3.1)
try:
    from dki.cache.user_isolation import (
        UserIsolationContext,
        UserIsolationConfig,
        UserScopedCacheStore,
        CacheKeySigner,
        InferenceContextGuard,
        IsolatedPreferenceCacheManager,
        CacheAuditLog,
        CacheAccessRecord,
    )
    USER_ISOLATION_AVAILABLE = True
except ImportError:
    USER_ISOLATION_AVAILABLE = False

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
    # User Isolation (v3.1)
    "USER_ISOLATION_AVAILABLE",
    "UserIsolationContext",
    "UserIsolationConfig",
    "UserScopedCacheStore",
    "CacheKeySigner",
    "InferenceContextGuard",
    "IsolatedPreferenceCacheManager",
    "CacheAuditLog",
    "CacheAccessRecord",
]
