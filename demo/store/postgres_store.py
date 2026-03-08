"""
PostgresChatStore - PostgreSQL implementation of IChatStore

Inherits BaseChatStore for shared CRUD logic.
Adds: ILIKE pre-filtering before BM25 scoring for better performance.

Author: AGI Demo Project
Version: 3.0.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import or_, text

from demo.store.base_impl import BaseChatStore
from demo.store.connection import DemoDBConfig
from demo.store.models import DemoMessage, DemoSession


class PostgresChatStore(BaseChatStore):
    """PostgreSQL-backed chat store with ILIKE + BM25 search."""

    def __init__(self, config: DemoDBConfig):
        super().__init__(config)

    # ============ Lifecycle ============

    def connect(self) -> None:
        self.db_manager.init_database()
        logger.info(
            f"PostgresChatStore connected: "
            f"{self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database}"
        )

    def disconnect(self) -> None:
        self.db_manager.dispose()
        logger.info("PostgresChatStore disconnected")

    def health_check(self) -> Dict[str, Any]:
        try:
            with self.db_manager.session_scope() as session:
                result = session.execute(text("SELECT version()"))
                pg_version = result.scalar()
            return {
                "status": "healthy",
                "adapter": "PostgresChatStore",
                "backend": "postgresql",
                "pg_version": pg_version,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "adapter": "PostgresChatStore",
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

                # PostgreSQL specific: database size
                db_size = session.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )).scalar()

            return {
                "adapter": "PostgresChatStore",
                "backend": "postgresql",
                "user_count": user_count,
                "session_count": session_count,
                "message_count": msg_count,
                "preference_count": pref_count,
                "database_size": db_size,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {"adapter": "PostgresChatStore", "error": str(e)}

    # ============ Search (backend-specific: ILIKE + BM25) ============

    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """ILIKE pre-filtering + BM25 scoring on PostgreSQL. 过滤已软删除会话的消息。"""
        self._check_connected()

        keywords = self._extract_keywords(query)

        with self.db_manager.session_scope() as session:
            q = session.query(DemoMessage).join(
                DemoSession, DemoMessage.session_id == DemoSession.id
            ).filter(
                DemoMessage.user_id == user_id,
                DemoSession.is_active == True,
            )
            if session_id:
                q = q.filter(DemoMessage.session_id == session_id)

            # PostgreSQL ILIKE pre-filtering
            if keywords:
                conditions = [
                    DemoMessage.content.ilike(f"%{kw}%")
                    for kw in keywords
                ]
                q = q.filter(or_(*conditions))

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
