"""
API Dependencies
FastAPI dependency injection for DKI API

v8.2: 增加 DKIPlugin 全局实例支持
安全 (v3.1): 增加用户隔离组件的依赖注入

Author: AGI Demo Project
Version: 8.2.0
"""

from typing import Any, Optional

from loguru import logger

# Global instances (initialized on startup)
_dki_system: Optional[Any] = None
_dki_plugin_instance: Optional[Any] = None  # v8.2: DKIPlugin 实例
_user_adapter: Optional[Any] = None
_preference_cache: Optional[Any] = None
_non_vectorized_handler: Optional[Any] = None
_isolated_preference_cache: Optional[Any] = None
_startup_time: float = 0


def init_dependencies(
    dki_system: Any = None,
    user_adapter: Any = None,
    preference_cache: Any = None,
    non_vectorized_handler: Any = None,
    isolated_preference_cache: Any = None,
    dki_plugin: Any = None,
) -> None:
    """
    Initialize global dependencies.
    
    Called during application startup.
    
    v8.2: 支持 dki_plugin 参数, 用于 /v1/dki/chat 端点
    安全 (v3.1): 支持 IsolatedPreferenceCacheManager 注入
    """
    import time
    global _dki_system, _dki_plugin_instance, _user_adapter, _preference_cache
    global _non_vectorized_handler, _isolated_preference_cache, _startup_time
    
    _dki_system = dki_system
    _dki_plugin_instance = dki_plugin
    _user_adapter = user_adapter
    _preference_cache = preference_cache
    _non_vectorized_handler = non_vectorized_handler
    _isolated_preference_cache = isolated_preference_cache
    _startup_time = time.time()
    
    logger.info(
        f"API dependencies initialized "
        f"(dki_plugin={'yes' if dki_plugin else 'no'}, "
        f"isolated_cache={'yes' if isolated_preference_cache else 'no'})"
    )


def get_dki_system():
    """
    Get DKI system instance.
    
    FastAPI dependency for DKI system access.
    仅用于 /api/chat (legacy web demo endpoint).
    """
    global _dki_system
    if _dki_system is None:
        # Lazy initialization
        from dki.core.dki_system import DKISystem
        _dki_system = DKISystem()
    
    return _dki_system


def get_dki_plugin_dep():
    """
    Get DKI plugin instance (v8.2).
    
    FastAPI dependency for /v1/dki/* endpoints.
    优先返回 DKIPlugin, 降级返回 DKISystem.
    """
    global _dki_plugin_instance, _dki_system
    if _dki_plugin_instance is not None:
        return _dki_plugin_instance
    # 降级: 返回 DKISystem
    return get_dki_system()


def get_user_adapter():
    """
    Get user data adapter instance.
    
    FastAPI dependency for user data access.
    Note: Demo UI 已独立为 demo/app.py, 此处仅用于实验系统 web 接口。
    """
    global _user_adapter
    if _user_adapter is None:
        # Lazy initialization - 无外部配置时使用 ExampleAdapter (仅开发/实验)
        from dki.adapters import ExampleAdapter

        _user_adapter = ExampleAdapter()
        logger.warning(
            "Using ExampleAdapter (no external adapter configured). "
            "Demo UI should use demo/app.py with its own persistence."
        )
    
    return _user_adapter


def get_preference_cache():
    """
    Get preference cache manager instance.
    
    FastAPI dependency for preference caching.
    """
    global _preference_cache
    if _preference_cache is None:
        # Lazy initialization
        from dki.cache import PreferenceCacheManager

        _preference_cache = PreferenceCacheManager()
    
    return _preference_cache


def get_isolated_preference_cache():
    """
    Get isolated preference cache manager instance.
    
    FastAPI dependency for user-isolated preference caching (v3.1).
    Returns IsolatedPreferenceCacheManager if available, otherwise None.
    """
    global _isolated_preference_cache
    if _isolated_preference_cache is None:
        try:
            from dki.cache.user_isolation import IsolatedPreferenceCacheManager
            _isolated_preference_cache = IsolatedPreferenceCacheManager()
            logger.info("IsolatedPreferenceCacheManager initialized")
        except ImportError:
            logger.warning("IsolatedPreferenceCacheManager not available")
            return None
    
    return _isolated_preference_cache


def get_non_vectorized_handler():
    """
    Get non-vectorized data handler instance.
    
    FastAPI dependency for handling non-vectorized data.
    """
    global _non_vectorized_handler
    if _non_vectorized_handler is None:
        # Lazy initialization
        from dki.cache import NonVectorizedDataHandler
        from dki.core.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()
        _non_vectorized_handler = NonVectorizedDataHandler(embedding_service)
    
    return _non_vectorized_handler


def get_startup_time() -> float:
    """Get application startup time."""
    return _startup_time


async def cleanup_dependencies() -> None:
    """
    Cleanup dependencies on shutdown.
    
    Called during application shutdown.
    """
    global _dki_system, _dki_plugin_instance, _user_adapter, _preference_cache
    global _non_vectorized_handler, _isolated_preference_cache
    
    # Disconnect user adapter
    if _user_adapter is not None:
        try:
            await _user_adapter.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting user adapter: {e}")
    
    # Clear preference cache
    if _preference_cache is not None:
        try:
            await _preference_cache.clear_all()
        except Exception as e:
            logger.error(f"Error clearing preference cache: {e}")
    
    # Clear isolated preference cache (v3.1)
    if _isolated_preference_cache is not None:
        try:
            from dki.cache.user_isolation import UserScopedCacheStore
            l1_store = getattr(_isolated_preference_cache, '_l1_store', None)
            if l1_store and isinstance(l1_store, UserScopedCacheStore):
                await l1_store.clear_all()
        except Exception as e:
            logger.error(f"Error clearing isolated preference cache: {e}")
    
    # Clear non-vectorized handler cache
    if _non_vectorized_handler is not None:
        try:
            await _non_vectorized_handler.clear_cache()
        except Exception as e:
            logger.error(f"Error clearing non-vectorized handler cache: {e}")
    
    _dki_system = None
    _dki_plugin_instance = None
    _user_adapter = None
    _preference_cache = None
    _non_vectorized_handler = None
    _isolated_preference_cache = None
    
    logger.info("API dependencies cleaned up")
