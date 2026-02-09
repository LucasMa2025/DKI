"""
DKI User Data Adapters
Standardized adapters for connecting to external user data sources

This module provides:
- IUserDataAdapter: Abstract interface for all adapters
- Built-in implementations: PostgreSQL, MySQL, MongoDB, Redis, REST API
- Data models: UserProfile, ChatMessage, UserPreference
"""

from dki.adapters.base import (
    IUserDataAdapter,
    UserProfile,
    ChatMessage,
    UserPreference,
    AdapterConfig,
)
from dki.adapters.postgresql_adapter import PostgreSQLUserDataAdapter
from dki.adapters.mysql_adapter import MySQLUserDataAdapter
from dki.adapters.mongodb_adapter import MongoDBUserDataAdapter
from dki.adapters.redis_adapter import RedisUserDataAdapter
from dki.adapters.rest_adapter import RESTAPIUserDataAdapter
from dki.adapters.memory_adapter import InMemoryUserDataAdapter
from dki.adapters.factory import AdapterFactory

__all__ = [
    # Interface and Models
    "IUserDataAdapter",
    "UserProfile",
    "ChatMessage",
    "UserPreference",
    "AdapterConfig",
    # Implementations
    "PostgreSQLUserDataAdapter",
    "MySQLUserDataAdapter",
    "MongoDBUserDataAdapter",
    "RedisUserDataAdapter",
    "RESTAPIUserDataAdapter",
    "InMemoryUserDataAdapter",
    # Factory
    "AdapterFactory",
]
