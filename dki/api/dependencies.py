"""
API Dependencies
FastAPI dependency injection for DKI API

Author: AGI Demo Project
Version: 1.0.0
"""

from typing import Any, Optional

from loguru import logger

# Global instances (initialized on startup)
_dki_system: Optional[Any] = None
_user_adapter: Optional[Any] = None
_preference_cache: Optional[Any] = None
_non_vectorized_handler: Optional[Any] = None
_startup_time: float = 0


def init_dependencies(
    dki_system: Any = None,
    user_adapter: Any = None,
    preference_cache: Any = None,
    non_vectorized_handler: Any = None,
) -> None:
    """
    Initialize global dependencies.
    
    Called during application startup.
    """
    import time
    global _dki_system, _user_adapter, _preference_cache, _non_vectorized_handler, _startup_time
    
    _dki_system = dki_system
    _user_adapter = user_adapter
    _preference_cache = preference_cache
    _non_vectorized_handler = non_vectorized_handler
    _startup_time = time.time()
    
    logger.info("API dependencies initialized")


def get_dki_system():
    """
    Get DKI system instance.
    
    FastAPI dependency for DKI system access.
    """
    if _dki_system is None:
        # Lazy initialization
        from dki.core.dki_system import DKISystem
        global _dki_system
        _dki_system = DKISystem()
    
    return _dki_system


def get_user_adapter():
    """
    Get user data adapter instance.
    
    FastAPI dependency for user data access.
    """
    if _user_adapter is None:
        # Lazy initialization with in-memory adapter
        from dki.adapters import InMemoryUserDataAdapter
        global _user_adapter
        _user_adapter = InMemoryUserDataAdapter()
        logger.warning("Using in-memory adapter (no external adapter configured)")
    
    return _user_adapter


def get_preference_cache():
    """
    Get preference cache manager instance.
    
    FastAPI dependency for preference caching.
    """
    if _preference_cache is None:
        # Lazy initialization
        from dki.cache import PreferenceCacheManager
        global _preference_cache
        _preference_cache = PreferenceCacheManager()
    
    return _preference_cache


def get_non_vectorized_handler():
    """
    Get non-vectorized data handler instance.
    
    FastAPI dependency for handling non-vectorized data.
    """
    if _non_vectorized_handler is None:
        # Lazy initialization
        from dki.cache import NonVectorizedDataHandler
        from dki.core.embedding_service import EmbeddingService
        global _non_vectorized_handler
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
    
    logger.info("API dependencies cleaned up")
