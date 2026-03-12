"""
SQLiteChatStore - 实验系统 SQLite 实现

从 demo/store/sqlite_store.py 复制, 修改导入路径。

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store/sqlite_store.py 3.0.0)
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import text

from dki.experiment.store.base_impl import BaseChatStore
from dki.experiment.store.connection import ExperimentDBConfig
from dki.experiment.store.models import DemoMessage, DemoSession


class SQLiteChatStore(BaseChatStore):
    """SQLite-backed chat store with BM25 search."""

    def __init__(self, config: ExperimentDBConfig):
        super().__init__(config)

    # ============ Lifecycle ============

    def connect(self) -> None:
        self.db_manager.init_database()
        logger.info(f"Experiment SQLiteChatStore connected: {self.config.sqlite_path}")

    def disconnect(self) -> None:
        self.db_manager.dispose()
        logger.info("Experiment SQLiteChatStore disconnected")

    def health_check(self) -> Dict[str, Any]:
        try:
            with self.db_manager.session_scope() as session:
                session.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "adapter": "ExperimentSQLiteChatStore",
                "backend": "sqlite",
                "path": self.config.sqlite_path,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "adapter": "ExperimentSQLiteChatStore",
                "error": str(e),
            }

    def get_statistics(self) -> Dict[str, Any]:
        try:
            with self.db_manager.session_scope() as session:
                from dki.experiment.store.models import DemoUser, DemoPreference
                user_count = session.query(DemoUser).count()
                session_count = session.query(DemoSession).filter(
                    DemoSession.is_active == True
                ).count()
                msg_count = session.query(DemoMessage).count()
                pref_count = session.query(DemoPreference).filter(
                    DemoPreference.is_active == True
                ).count()

            return {
                "adapter": "ExperimentSQLiteChatStore",
                "backend": "sqlite",
                "user_count": user_count,
                "session_count": session_count,
                "message_count": msg_count,
                "preference_count": pref_count,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {"adapter": "ExperimentSQLiteChatStore", "error": str(e)}

    # ============ Search (backend-specific) ============

    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """BM25 search on SQLite (in-memory scoring)."""
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
