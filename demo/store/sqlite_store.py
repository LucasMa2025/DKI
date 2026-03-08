"""
SQLiteChatStore - SQLite implementation of IChatStore

Inherits BaseChatStore for shared CRUD logic.
Only implements: connect, disconnect, health_check, get_statistics, search_messages.

Author: AGI Demo Project
Version: 3.0.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import text

from demo.store.base_impl import BaseChatStore
from demo.store.connection import DemoDBConfig
from demo.store.models import DemoMessage, DemoSession


class SQLiteChatStore(BaseChatStore):
    """SQLite-backed chat store with BM25 search."""

    def __init__(self, config: DemoDBConfig):
        super().__init__(config)

    # ============ Lifecycle ============

    def connect(self) -> None:
        self.db_manager.init_database()
        logger.info(f"SQLiteChatStore connected: {self.config.sqlite_path}")

    def disconnect(self) -> None:
        self.db_manager.dispose()
        logger.info("SQLiteChatStore disconnected")

    def health_check(self) -> Dict[str, Any]:
        try:
            with self.db_manager.session_scope() as session:
                session.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "adapter": "SQLiteChatStore",
                "backend": "sqlite",
                "path": self.config.sqlite_path,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "adapter": "SQLiteChatStore",
                "error": str(e),
            }

    def get_statistics(self) -> Dict[str, Any]:
        try:
            with self.db_manager.session_scope() as session:
                from demo.store.models import DemoUser, DemoSession, DemoPreference
                user_count = session.query(DemoUser).count()
                session_count = session.query(DemoSession).filter(
                    DemoSession.is_active == True
                ).count()
                msg_count = session.query(DemoMessage).count()
                pref_count = session.query(DemoPreference).filter(
                    DemoPreference.is_active == True
                ).count()

            return {
                "adapter": "SQLiteChatStore",
                "backend": "sqlite",
                "user_count": user_count,
                "session_count": session_count,
                "message_count": msg_count,
                "preference_count": pref_count,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {"adapter": "SQLiteChatStore", "error": str(e)}

    # ============ Search (backend-specific) ============

    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """BM25 search on SQLite (in-memory scoring). 过滤已软删除会话的消息。"""
        self._check_connected()

        with self.db_manager.session_scope() as session:
            q = session.query(DemoMessage).join(
                DemoSession, DemoMessage.session_id == DemoSession.id
            ).filter(
                DemoMessage.user_id == user_id,
                DemoSession.is_active == True,
            )
            if session_id:
                q = q.filter(DemoMessage.session_id == session_id)

            candidates = q.order_by(DemoMessage.created_at.desc()).limit(500).all()

            if not candidates:
                return []

            scored = self._bm25_score(query, candidates)
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for msg, score in scored[:limit]:
                if score > 0:
                    session.expunge(msg)
                    results.append(msg)

            return results
