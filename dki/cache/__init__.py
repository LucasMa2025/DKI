"""
DKI Cache System
Provides tiered caching for user preferences and K/V data

This module provides:
- PreferenceCacheManager: Tiered cache for user preference K/V
- NonVectorizedDataHandler: Handler for non-vectorized message data
"""

from dki.cache.preference_cache import PreferenceCacheManager, CacheTierInfo
from dki.cache.non_vectorized_handler import NonVectorizedDataHandler

__all__ = [
    "PreferenceCacheManager",
    "CacheTierInfo",
    "NonVectorizedDataHandler",
]
