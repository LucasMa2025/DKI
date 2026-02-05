"""
Database Connection Manager for DKI System
Handles SQLite connections and session management
"""

import os
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session as SQLSession
from sqlalchemy.pool import StaticPool
from loguru import logger

from dki.database.models import Base


class DatabaseManager:
    """Database connection and session manager."""
    
    _instance: Optional['DatabaseManager'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        db_path: str = "./data/dki.db",
        echo: bool = False,
        pool_size: int = 5,
    ):
        if self._initialized:
            return
            
        self._db_path = db_path
        self._echo = echo
        self._pool_size = pool_size
        self._engine = None
        self._session_factory = None
        self._initialized = True
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure data directory exists
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        db_url = f"sqlite:///{self._db_path}"
        self._engine = create_engine(
            db_url,
            echo=self._echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
        # Create session factory
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
        
        # Create tables
        Base.metadata.create_all(self._engine)
        logger.info(f"Database initialized at {self._db_path}")
    
    @property
    def engine(self):
        """Get SQLAlchemy engine."""
        return self._engine
    
    def get_session(self) -> SQLSession:
        """Get a new database session."""
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[SQLSession, None, None]:
        """Context manager for database sessions with automatic commit/rollback."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_script(self, script_path: str) -> None:
        """Execute SQL script file."""
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()
        
        with self.session_scope() as session:
            for statement in script.split(';'):
                statement = statement.strip()
                if statement:
                    try:
                        session.execute(statement)
                    except Exception as e:
                        logger.warning(f"Statement execution failed: {e}")
        
        logger.info(f"Executed script: {script_path}")
    
    def drop_all(self) -> None:
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All tables dropped")
    
    def reset(self) -> None:
        """Reset database (drop and recreate all tables)."""
        self.drop_all()
        Base.metadata.create_all(self._engine)
        logger.info("Database reset completed")
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        if cls._instance is not None:
            if cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance._initialized = False
            cls._instance = None


def get_db() -> Generator[SQLSession, None, None]:
    """Dependency for FastAPI to get database session."""
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
