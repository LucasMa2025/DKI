"""
Demo Database Connection Manager

Independent from DKI/dki/database/connection.py.
Supports SQLite / PostgreSQL / pgvector backends.

Ref: AGA persistence patterns:
- SQLite WAL mode for concurrency
- Retry on busy (ref: AGA SQLiteAdapter._retry_on_busy)
- Pool statistics
- Connection health check
- Async support for PostgreSQL (asyncpg)

Author: AGI Demo Project
Version: 3.1.0
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator, Dict, Any
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session as SQLSession
from sqlalchemy.pool import StaticPool
from loguru import logger

from demo.store.models import DemoBase
from demo.store.base import StoreConnectionError

# Optional async imports
try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncSession,
        async_sessionmaker,
    )
    _HAS_ASYNC_SA = True
except ImportError:
    _HAS_ASYNC_SA = False


# ==================== Retry Config (ref: AGA SQLiteAdapter) ====================

MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 0.1  # seconds


@dataclass
class DemoDBConfig:
    """Demo database configuration"""
    # Backend type: sqlite | postgresql | pgvector
    backend: str = "sqlite"

    # SQLite
    sqlite_path: str = "./data/demo.db"

    # PostgreSQL / pgvector
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "dki_demo"
    pg_username: str = ""
    pg_password: str = ""

    # pgvector
    pgvector_enabled: bool = False
    embedding_dim: int = 768

    # Connection pool
    pool_size: int = 5
    max_overflow: int = 10

    # SQLite specific (ref: AGA persistence)
    enable_wal: bool = True
    busy_timeout_ms: int = 5000

    # Debug
    echo: bool = False

    def get_connection_url(self) -> str:
        """Get SQLAlchemy connection URL (synchronous)"""
        if self.backend == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        elif self.backend in ("postgresql", "pgvector"):
            return (
                f"postgresql://{self.pg_username}:{self.pg_password}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def get_async_connection_url(self) -> str:
        """Get SQLAlchemy async connection URL (asyncpg driver)"""
        if self.backend == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        elif self.backend in ("postgresql", "pgvector"):
            return (
                f"postgresql+asyncpg://{self.pg_username}:{self.pg_password}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @classmethod
    def from_dict(cls, data: dict) -> "DemoDBConfig":
        """Create config from dict"""
        return cls(
            backend=data.get("backend", "sqlite"),
            sqlite_path=data.get("sqlite_path", "./data/demo.db"),
            pg_host=data.get("pg_host", "localhost"),
            pg_port=data.get("pg_port", 5432),
            pg_database=data.get("pg_database", "dki_demo"),
            pg_username=data.get("pg_username", ""),
            pg_password=data.get("pg_password", ""),
            pgvector_enabled=data.get("pgvector_enabled", False),
            embedding_dim=data.get("embedding_dim", 768),
            pool_size=data.get("pool_size", 5),
            max_overflow=data.get("max_overflow", 10),
            enable_wal=data.get("enable_wal", True),
            busy_timeout_ms=data.get("busy_timeout_ms", 5000),
            echo=data.get("echo", False),
        )


@dataclass
class PoolStats:
    """Connection pool statistics (ref: AGA persistence.pool.PoolStats)"""
    total_sessions: int = 0
    active_sessions: int = 0
    error_count: int = 0
    retry_count: int = 0
    total_acquire_time_ms: float = 0.0

    @property
    def avg_acquire_time_ms(self) -> float:
        return self.total_acquire_time_ms / max(1, self.total_sessions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": self.active_sessions,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "avg_acquire_time_ms": round(self.avg_acquire_time_ms, 2),
        }


class DemoDBManager:
    """
    Demo database connection manager.

    Similar to DKI DatabaseManager but fully independent.
    Enhanced with: WAL mode, retry on busy, pool stats (ref: AGA persistence).
    """

    def __init__(self, config: DemoDBConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
        self._connected = False
        self._stats = PoolStats()

    @property
    def engine(self):
        return self._engine

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> PoolStats:
        return self._stats

    def init_database(self) -> None:
        """Initialize database connection and create tables"""
        url = self.config.get_connection_url()

        try:
            if self.config.backend == "sqlite":
                # Ensure directory exists
                db_dir = Path(self.config.sqlite_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

                self._engine = create_engine(
                    url,
                    echo=self.config.echo,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": self.config.busy_timeout_ms / 1000.0,
                    },
                    poolclass=StaticPool,
                )

                # SQLite: enable WAL mode + foreign keys (ref: AGA SQLiteAdapter)
                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    if self.config.enable_wal:
                        cursor.execute("PRAGMA journal_mode=WAL")
                        cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.close()
            else:
                # PostgreSQL
                self._engine = create_engine(
                    url,
                    echo=self.config.echo,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                )

            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )

            # Create tables
            DemoBase.metadata.create_all(self._engine)

            # pgvector: initialize extension
            if self.config.backend == "pgvector" and self.config.pgvector_enabled:
                self._init_pgvector_extension()

            self._connected = True
            logger.info(
                f"Demo database initialized "
                f"(backend={self.config.backend}, "
                f"url={self._mask_url(url)})"
            )
        except Exception as e:
            self._connected = False
            raise StoreConnectionError(f"Failed to initialize database: {e}") from e

    def _init_pgvector_extension(self) -> None:
        """Initialize pgvector extension"""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                dim = self.config.embedding_dim
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'demo_messages'
                            AND column_name = 'embedding_vector'
                        ) THEN
                            ALTER TABLE demo_messages
                            ADD COLUMN embedding_vector vector({dim});
                        END IF;
                    END $$;
                """))

                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS ix_demo_messages_embedding_vector
                    ON demo_messages
                    USING hnsw (embedding_vector vector_cosine_ops)
                """))

            logger.info(f"pgvector extension initialized (dim={dim})")
        except Exception as e:
            logger.warning(f"Failed to initialize pgvector: {e}")

    def get_session(self) -> SQLSession:
        """Get database session"""
        if self._session_factory is None:
            from demo.store.base import StoreNotConnectedError
            raise StoreNotConnectedError(
                "Database not initialized. Call init_database() first."
            )
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[SQLSession, None, None]:
        """
        Database session context manager (auto commit/rollback).

        Enhanced with retry on busy (ref: AGA SQLiteAdapter._retry_on_busy).
        """
        start_time = time.perf_counter()
        self._stats.total_sessions += 1
        self._stats.active_sessions += 1

        session = self.get_session()
        try:
            yield session
            self._commit_with_retry(session)
        except Exception as e:
            session.rollback()
            self._stats.error_count += 1
            logger.error(f"Demo DB session error: {e}")
            raise
        finally:
            session.close()
            self._stats.active_sessions -= 1
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.total_acquire_time_ms += elapsed

    def _commit_with_retry(self, session: SQLSession) -> None:
        """
        Commit with retry on busy/locked (ref: AGA SQLiteAdapter._retry_on_busy).
        Only relevant for SQLite; PostgreSQL commits normally.
        """
        if self.config.backend != "sqlite":
            session.commit()
            return

        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                session.commit()
                return
            except Exception as e:
                err_str = str(e).lower()
                if "locked" in err_str or "busy" in err_str:
                    last_error = e
                    self._stats.retry_count += 1
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(
                        f"DB busy, retrying commit in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})"
                    )
                    time.sleep(delay)
                else:
                    raise

        # All retries exhausted
        if last_error:
            raise last_error

    def dispose(self) -> None:
        """Close connection pool"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Demo database connection disposed")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics (ref: AGA persistence.pool)"""
        return {
            "backend": self.config.backend,
            "connected": self._connected,
            **self._stats.to_dict(),
        }

    def _mask_url(self, url: str) -> str:
        """Mask password in URL"""
        if "://" in url and "@" in url:
            prefix, rest = url.split("://", 1)
            if "@" in rest:
                creds, host = rest.rsplit("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    return f"{prefix}://{user}:***@{host}"
        return url


# ==================== Async Database Manager ====================

class AsyncDemoDBManager:
    """
    Async database connection manager for PostgreSQL backends.

    Uses sqlalchemy.ext.asyncio with asyncpg driver.
    For SQLite, use the synchronous DemoDBManager instead.

    Ref: AGA PostgresAdapter (asyncpg-based async operations).
    """

    def __init__(self, config: DemoDBConfig):
        if not _HAS_ASYNC_SA:
            raise ImportError(
                "Async SQLAlchemy support requires sqlalchemy>=1.4 with "
                "sqlalchemy.ext.asyncio. Install: pip install sqlalchemy[asyncio]"
            )
        self.config = config
        self._engine = None
        self._session_factory = None
        self._connected = False
        self._stats = PoolStats()

    @property
    def engine(self):
        return self._engine

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> PoolStats:
        return self._stats

    async def init_database(self) -> None:
        """Initialize async database connection and create tables."""
        url = self.config.get_async_connection_url()

        try:
            self._engine = create_async_engine(
                url,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
            )

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Create tables using sync connection (run_sync)
            async with self._engine.begin() as conn:
                await conn.run_sync(DemoBase.metadata.create_all)

            # pgvector extension
            if self.config.backend == "pgvector" and self.config.pgvector_enabled:
                await self._init_pgvector_extension()

            self._connected = True
            logger.info(
                f"Async demo database initialized "
                f"(backend={self.config.backend}, "
                f"url={self._mask_url(url)})"
            )
        except Exception as e:
            self._connected = False
            raise StoreConnectionError(f"Failed to initialize async database: {e}") from e

    async def _init_pgvector_extension(self) -> None:
        """Initialize pgvector extension (async)."""
        try:
            dim = self.config.embedding_dim
            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'demo_messages'
                            AND column_name = 'embedding_vector'
                        ) THEN
                            ALTER TABLE demo_messages
                            ADD COLUMN embedding_vector vector({dim});
                        END IF;
                    END $$;
                """))
                await conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS ix_demo_messages_embedding_vector
                    ON demo_messages
                    USING hnsw (embedding_vector vector_cosine_ops)
                """))
            logger.info(f"pgvector extension initialized (dim={dim})")
        except Exception as e:
            logger.warning(f"Failed to initialize pgvector (async): {e}")

    @asynccontextmanager
    async def session_scope(self):
        """
        Async database session context manager (auto commit/rollback).
        """
        start_time = time.perf_counter()
        self._stats.total_sessions += 1
        self._stats.active_sessions += 1

        if self._session_factory is None:
            from demo.store.base import StoreNotConnectedError
            raise StoreNotConnectedError(
                "Async database not initialized. Call init_database() first."
            )

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._stats.error_count += 1
                logger.error(f"Async demo DB session error: {e}")
                raise
            finally:
                self._stats.active_sessions -= 1
                elapsed = (time.perf_counter() - start_time) * 1000
                self._stats.total_acquire_time_ms += elapsed

    async def dispose(self) -> None:
        """Close async connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Async demo database connection disposed")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "backend": f"{self.config.backend} (async)",
            "connected": self._connected,
            **self._stats.to_dict(),
        }

    def _mask_url(self, url: str) -> str:
        """Mask password in URL."""
        if "://" in url and "@" in url:
            prefix, rest = url.split("://", 1)
            if "@" in rest:
                creds, host = rest.rsplit("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    return f"{prefix}://{user}:***@{host}"
        return url
