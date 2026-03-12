"""
Experiment Database Connection Manager

从 demo/store/connection.py 复制并简化, 仅保留 SQLite 支持。
使用独立的 dki.db (默认 ./data/dki.db)。

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store/connection.py 3.1.0)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session as SQLSession
from sqlalchemy.pool import StaticPool
from loguru import logger

from dki.experiment.store.models import DemoBase
from dki.experiment.store.base import StoreConnectionError


# Retry config
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 0.1


@dataclass
class ExperimentDBConfig:
    """实验系统数据库配置 (仅 SQLite)"""
    backend: str = "sqlite"
    sqlite_path: str = "./data/dki.db"

    # SQLite specific
    enable_wal: bool = True
    busy_timeout_ms: int = 5000

    # Debug
    echo: bool = False

    def get_connection_url(self) -> str:
        return f"sqlite:///{self.sqlite_path}"

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentDBConfig":
        return cls(
            backend=data.get("backend", "sqlite"),
            sqlite_path=data.get("sqlite_path", "./data/dki.db"),
            enable_wal=data.get("enable_wal", True),
            busy_timeout_ms=data.get("busy_timeout_ms", 5000),
            echo=data.get("echo", False),
        )


@dataclass
class PoolStats:
    """Connection pool statistics"""
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


class ExperimentDBManager:
    """实验系统数据库连接管理器 (仅 SQLite)"""

    def __init__(self, config: ExperimentDBConfig):
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
        """Initialize database connection and create tables.
        
        包含 schema 迁移: 如果 dki.db 中的 demo_* 表由旧版 init_db.sql 创建,
        可能缺少新增列 (如 password_hash)。create_all() 只创建不存在的表,
        不会为已存在的表添加新列, 因此需要显式 ALTER TABLE。
        """
        url = self.config.get_connection_url()

        try:
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

            # SQLite: enable WAL mode + foreign keys
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                if self.config.enable_wal:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )

            # Create tables (demo_users, demo_sessions, demo_messages, demo_preferences)
            DemoBase.metadata.create_all(self._engine)

            # Schema migration: 补全旧表缺失的列
            self._migrate_schema()

            self._connected = True
            logger.info(
                f"Experiment database initialized "
                f"(backend=sqlite, path={self.config.sqlite_path})"
            )
        except Exception as e:
            self._connected = False
            raise StoreConnectionError(f"Failed to initialize database: {e}") from e

    def _migrate_schema(self) -> None:
        """检测并补全旧表缺失的列 (SQLite ALTER TABLE ADD COLUMN)。
        
        场景: dki.db 由 init_db.sql 创建, demo_users 表缺少 password_hash 等新列。
        create_all() 使用 CREATE TABLE IF NOT EXISTS, 不会修改已存在的表。
        """
        if self._engine is None:
            return

        # 定义需要检查的 (table_name, column_name, column_type, default) 列表
        # 仅列出可能缺失的列 (旧 init_db.sql 未定义但 ORM 模型需要的)
        migrations = [
            ("demo_users", "password_hash", "TEXT", None),
        ]

        with self._engine.connect() as conn:
            for table_name, col_name, col_type, default in migrations:
                try:
                    # 检查表是否存在
                    result = conn.execute(
                        text(f"SELECT name FROM sqlite_master WHERE type='table' AND name=:tbl"),
                        {"tbl": table_name},
                    )
                    if result.fetchone() is None:
                        continue  # 表不存在, create_all() 会创建完整表

                    # 检查列是否存在
                    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                    existing_columns = {row[1] for row in result}

                    if col_name not in existing_columns:
                        alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                        if default is not None:
                            alter_sql += f" DEFAULT {default}"
                        conn.execute(text(alter_sql))
                        conn.commit()
                        logger.info(
                            f"Schema migration: added column {table_name}.{col_name} ({col_type})"
                        )
                except Exception as e:
                    logger.warning(f"Schema migration warning ({table_name}.{col_name}): {e}")

    def get_session(self) -> SQLSession:
        if self._session_factory is None:
            from dki.experiment.store.base import StoreNotConnectedError
            raise StoreNotConnectedError(
                "Database not initialized. Call init_database() first."
            )
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[SQLSession, None, None]:
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
            logger.error(f"Experiment DB session error: {e}")
            raise
        finally:
            session.close()
            self._stats.active_sessions -= 1
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.total_acquire_time_ms += elapsed

    def _commit_with_retry(self, session: SQLSession) -> None:
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

        if last_error:
            raise last_error

    def dispose(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Experiment database connection disposed")

    def get_pool_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.config.backend,
            "connected": self._connected,
            **self._stats.to_dict(),
        }
