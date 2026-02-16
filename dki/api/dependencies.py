"""
API Dependencies
FastAPI dependency injection for DKI API

安全 (v3.1): 增加用户隔离组件的依赖注入

Author: AGI Demo Project
Version: 3.1.0
"""

from typing import Any, Optional

from loguru import logger

# Global instances (initialized on startup)
_dki_system: Optional[Any] = None
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
) -> None:
    """
    Initialize global dependencies.
    
    Called during application startup.
    
    安全 (v3.1): 支持 IsolatedPreferenceCacheManager 注入
    """
    import time
    global _dki_system, _user_adapter, _preference_cache, _non_vectorized_handler
    global _isolated_preference_cache, _startup_time
    
    _dki_system = dki_system
    _user_adapter = user_adapter
    _preference_cache = preference_cache
    _non_vectorized_handler = non_vectorized_handler
    _isolated_preference_cache = isolated_preference_cache
    _startup_time = time.time()
    
    logger.info(
        f"API dependencies initialized "
        f"(isolated_cache={'yes' if isolated_preference_cache else 'no'})"
    )


def get_dki_system():
    """
    Get DKI system instance.
    
    FastAPI dependency for DKI system access.
    """
    global _dki_system
    if _dki_system is None:
        # Lazy initialization
        from dki.core.dki_system import DKISystem
        _dki_system = DKISystem()
    
    return _dki_system


def get_user_adapter():
    """
    Get user data adapter instance.
    
    FastAPI dependency for user data access.
    """
    global _user_adapter
    if _user_adapter is None:
        # Lazy initialization with in-memory adapter
        from dki.adapters import InMemoryUserDataAdapter

        _user_adapter = InMemoryUserDataAdapter()
        logger.warning("Using in-memory adapter (no external adapter configured)")
    
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
    global _dki_system, _user_adapter, _preference_cache, _non_vectorized_handler
    global _isolated_preference_cache
    
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
    _user_adapter = None
    _preference_cache = None
    _non_vectorized_handler = None
    _isolated_preference_cache = None
    
    logger.info("API dependencies cleaned up")
