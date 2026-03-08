"""
Chat Store Factory

Creates the appropriate IChatStore implementation based on config.
Ref: AGA persistence factory pattern (optional dependency detection).

Supports both sync and async store creation:
- create_chat_store(): sync factory (SQLite, or sync PostgreSQL)
- create_async_chat_store(): async factory (PostgreSQL/pgvector with asyncpg)

Author: AGI Demo Project
Version: 3.1.0
"""

from loguru import logger

from demo.store.base import IChatStore, StoreError
from demo.store.connection import DemoDBConfig


def create_chat_store(config: DemoDBConfig, **kwargs) -> IChatStore:
    """
    Factory function to create the appropriate sync IChatStore.

    Args:
        config: database configuration
        **kwargs: additional arguments (e.g. embedding_service for pgvector)

    Returns:
        IChatStore instance (already connected)

    Raises:
        StoreError: if backend is unsupported or dependency missing
    """
    backend = config.backend

    if backend == "sqlite":
        from demo.store.sqlite_store import SQLiteChatStore
        store = SQLiteChatStore(config)

    elif backend == "postgresql":
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise StoreError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            )
        from demo.store.postgres_store import PostgresChatStore
        store = PostgresChatStore(config)

    elif backend == "pgvector":
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise StoreError(
                "pgvector backend requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            )
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
        except ImportError:
            logger.warning(
                "pgvector Python package not found. "
                "Install with: pip install pgvector. "
                "Falling back to PostgresChatStore without vector search."
            )
            from demo.store.postgres_store import PostgresChatStore
            store = PostgresChatStore(config)
            store.connect()
            return store

        from demo.store.pgvector_store import PgVectorChatStore
        store = PgVectorChatStore(
            config,
            embedding_service=kwargs.get("embedding_service"),
        )

    else:
        raise StoreError(
            f"Unsupported backend: {backend}. "
            f"Supported: sqlite, postgresql, pgvector"
        )

    store.connect()
    logger.info(f"Chat store created: {store.__class__.__name__} (backend={backend})")
    return store


async def create_async_chat_store(config: DemoDBConfig, **kwargs) -> IChatStore:
    """
    Async factory function to create async IChatStore (PostgreSQL/pgvector).

    For SQLite, falls back to sync SQLiteChatStore.
    For PostgreSQL/pgvector, creates async stores using asyncpg driver.

    Args:
        config: database configuration
        **kwargs: additional arguments (e.g. embedding_service for pgvector)

    Returns:
        IChatStore instance (already connected)

    Raises:
        StoreError: if backend is unsupported or dependency missing
    """
    backend = config.backend

    if backend == "sqlite":
        # SQLite: use sync store (aiosqlite has limitations with ORM)
        logger.info("SQLite backend: using synchronous store")
        from demo.store.sqlite_store import SQLiteChatStore
        store = SQLiteChatStore(config)
        store.connect()
        logger.info(f"Chat store created: SQLiteChatStore (backend=sqlite)")
        return store

    elif backend == "postgresql":
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise StoreError(
                "Async PostgreSQL backend requires asyncpg. "
                "Install with: pip install asyncpg"
            )
        from demo.store.async_postgres_store import AsyncPostgresChatStore
        store = AsyncPostgresChatStore(config)

    elif backend == "pgvector":
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise StoreError(
                "Async pgvector backend requires asyncpg. "
                "Install with: pip install asyncpg"
            )
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
        except ImportError:
            logger.warning(
                "pgvector Python package not found. "
                "Falling back to AsyncPostgresChatStore without vector search."
            )
            from demo.store.async_postgres_store import AsyncPostgresChatStore
            store = AsyncPostgresChatStore(config)
            await store.connect()
            return store

        from demo.store.async_pgvector_store import AsyncPgVectorChatStore
        store = AsyncPgVectorChatStore(
            config,
            embedding_service=kwargs.get("embedding_service"),
        )

    else:
        raise StoreError(
            f"Unsupported backend: {backend}. "
            f"Supported: sqlite, postgresql, pgvector"
        )

    await store.connect()
    logger.info(f"Async chat store created: {store.__class__.__name__} (backend={backend})")
    return store
