"""
Demo Store - Independent Persistence Layer

Ref: AGA persistence design patterns:
- Exception hierarchy
- Health check with detailed status
- Statistics interface
- Connection lifecycle management
- Factory with optional dependency detection
- Async support for PostgreSQL backends

Author: AGI Demo Project
Version: 3.1.0
"""

from demo.store.base import (
    IChatStore,
    StoreError,
    StoreConnectionError,
    StoreOperationError,
    StoreNotConnectedError,
)
from demo.store.connection import DemoDBConfig, DemoDBManager, PoolStats
from demo.store.models import DemoBase, DemoUser, DemoSession, DemoMessage, DemoPreference
from demo.store.factory import create_chat_store, create_async_chat_store
from demo.store.bm25_mixin import BM25Mixin

# Optional: BaseChatStore for custom subclasses
try:
    from demo.store.base_impl import BaseChatStore
except ImportError:
    BaseChatStore = None  # type: ignore

# Optional: concrete stores (may fail if dependencies missing)
try:
    from demo.store.sqlite_store import SQLiteChatStore
except ImportError:
    SQLiteChatStore = None  # type: ignore

try:
    from demo.store.postgres_store import PostgresChatStore
except ImportError:
    PostgresChatStore = None  # type: ignore

try:
    from demo.store.pgvector_store import PgVectorChatStore
except ImportError:
    PgVectorChatStore = None  # type: ignore

# Optional: async stores (require sqlalchemy.ext.asyncio + asyncpg)
try:
    from demo.store.connection import AsyncDemoDBManager
except ImportError:
    AsyncDemoDBManager = None  # type: ignore

try:
    from demo.store.async_base_impl import AsyncBaseChatStore
except ImportError:
    AsyncBaseChatStore = None  # type: ignore

try:
    from demo.store.async_postgres_store import AsyncPostgresChatStore
except ImportError:
    AsyncPostgresChatStore = None  # type: ignore

try:
    from demo.store.async_pgvector_store import AsyncPgVectorChatStore
except ImportError:
    AsyncPgVectorChatStore = None  # type: ignore

__all__ = [
    # Abstract
    "IChatStore",
    # Exceptions
    "StoreError",
    "StoreConnectionError",
    "StoreOperationError",
    "StoreNotConnectedError",
    # Config & Connection
    "DemoDBConfig",
    "DemoDBManager",
    "AsyncDemoDBManager",
    "PoolStats",
    # Models
    "DemoBase",
    "DemoUser",
    "DemoSession",
    "DemoMessage",
    "DemoPreference",
    # Factory
    "create_chat_store",
    "create_async_chat_store",
    # Mixins
    "BM25Mixin",
    # Sync Implementations (optional)
    "BaseChatStore",
    "SQLiteChatStore",
    "PostgresChatStore",
    "PgVectorChatStore",
    # Async Implementations (optional)
    "AsyncBaseChatStore",
    "AsyncPostgresChatStore",
    "AsyncPgVectorChatStore",
]
