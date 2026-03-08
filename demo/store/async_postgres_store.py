"""
AsyncPostgresChatStore - Async PostgreSQL implementation of IChatStore

Async counterpart of PostgresChatStore.
Uses sqlalchemy.ext.asyncio with asyncpg driver for non-blocking I/O.

Author: AGI Demo Project
Version: 3.1.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import or_, text, select

from demo.store.async_base_impl import AsyncBaseChatStore
from demo.store.connection import DemoDBConfig
from demo.store.models import DemoMessage, DemoUser, DemoSession, DemoPreference


class AsyncPostgresChatStore(AsyncBaseChatStore):
    """Async PostgreSQL-backed chat store with ILIKE + BM25 search."""

    def __init__(self, config: DemoDBConfig):
        super().__init__(config)

    # ============ Lifecycle ============

    async def connect(self) -> None:
        await self.db_manager.init_database()
        logger.info(
            f"AsyncPostgresChatStore connected: "
            f"{self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database}"
        )

    async def disconnect(self) -> None:
        await self.db_manager.dispose()
        logger.info("AsyncPostgresChatStore disconnected")

    def health_check(self) -> Dict[str, Any]:
        """Sync health check (for compatibility). Use a_health_check() for async."""
        return {
            "status": "healthy" if self.db_manager.is_connected else "disconnected",
            "adapter": "AsyncPostgresChatStore",
            "backend": "postgresql (async)",
            "pool": self.db_manager.get_pool_stats(),
        }

    async def a_health_check(self) -> Dict[str, Any]:
        """Async health check."""
        try:
            async with self.db_manager.session_scope() as session:
                result = await session.execute(text("SELECT version()"))
                pg_version = result.scalar()
            return {
                "status": "healthy",
                "adapter": "AsyncPostgresChatStore",
                "backend": "postgresql (async)",
                "pg_version": pg_version,
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "adapter": "AsyncPostgresChatStore",
                "error": str(e),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Sync statistics (basic). Use a_get_statistics() for full async."""
        return {
            "adapter": "AsyncPostgresChatStore",
            "backend": "postgresql (async)",
            "pool": self.db_manager.get_pool_stats(),
        }

    async def a_get_statistics(self) -> Dict[str, Any]:
        """Async statistics with full DB queries."""
        try:
            async with self.db_manager.session_scope() as session:
                user_count = (await session.execute(
                    select(DemoUser).with_only_columns(DemoUser.id)
                )).all()
                session_count = (await session.execute(
                    select(DemoSession)
                    .filter(DemoSession.is_active == True)
                    .with_only_columns(DemoSession.id)
                )).all()
                msg_count_result = await session.execute(
                    text("SELECT COUNT(*) FROM demo_messages")
                )
                pref_count_result = await session.execute(
                    text("SELECT COUNT(*) FROM demo_preferences WHERE is_active = true")
                )
                db_size_result = await session.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                ))

            return {
                "adapter": "AsyncPostgresChatStore",
                "backend": "postgresql (async)",
                "user_count": len(user_count),
                "session_count": len(session_count),
                "message_count": msg_count_result.scalar() or 0,
                "preference_count": pref_count_result.scalar() or 0,
                "database_size": db_size_result.scalar(),
                "pool": self.db_manager.get_pool_stats(),
            }
        except Exception as e:
            return {"adapter": "AsyncPostgresChatStore", "error": str(e)}

    # ============ Search (async: ILIKE + BM25) ============

    async def a_search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """Async ILIKE pre-filtering + BM25 scoring on PostgreSQL. 过滤已软删除会话的消息。"""
        self._check_connected()

        keywords = self._extract_keywords(query)

        async with self.db_manager.session_scope() as session:
            stmt = (
                select(DemoMessage)
                .join(DemoSession, DemoMessage.session_id == DemoSession.id)
                .filter(DemoMessage.user_id == user_id)
                .filter(DemoSession.is_active == True)
            )
            if session_id:
                stmt = stmt.filter(DemoMessage.session_id == session_id)

            # PostgreSQL ILIKE pre-filtering
            if keywords:
                conditions = [
                    DemoMessage.content.ilike(f"%{kw}%")
                    for kw in keywords
                ]
                stmt = stmt.filter(or_(*conditions))

            stmt = stmt.order_by(DemoMessage.created_at.desc()).limit(500)
            result = await session.execute(stmt)
            candidates = list(result.scalars().all())

            if not candidates:
                return []

            scored = self._bm25_score(query, candidates)
            scored.sort(key=lambda x: x[1], reverse=True)

            return [msg for msg, score in scored[:limit] if score > 0]
