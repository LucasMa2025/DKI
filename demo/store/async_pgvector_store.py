"""
AsyncPgVectorChatStore - Async PostgreSQL + pgvector implementation

Async counterpart of PgVectorChatStore.
Extends AsyncPostgresChatStore with vector similarity search (hybrid: BM25 + cosine).

Author: AGI Demo Project
Version: 3.1.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import text, select

from demo.store.async_postgres_store import AsyncPostgresChatStore
from demo.store.connection import DemoDBConfig
from demo.store.models import DemoMessage


class AsyncPgVectorChatStore(AsyncPostgresChatStore):
    """Async PostgreSQL + pgvector chat store with hybrid search."""

    def __init__(self, config: DemoDBConfig, embedding_service=None):
        """
        Args:
            config: database configuration
            embedding_service: optional embedding service with encode(text) -> List[float]
        """
        super().__init__(config)
        self._embedding_service = embedding_service

    # ============ Lifecycle (override for pgvector info) ============

    async def connect(self) -> None:
        await self.db_manager.init_database()
        logger.info(
            f"AsyncPgVectorChatStore connected: "
            f"{self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database} "
            f"(embedding_dim={self.config.embedding_dim})"
        )

    def health_check(self) -> Dict[str, Any]:
        result = super().health_check()
        result["adapter"] = "AsyncPgVectorChatStore"
        result["backend"] = "pgvector (async)"
        result["embedding_dim"] = self.config.embedding_dim
        result["has_embedding_service"] = self._embedding_service is not None
        return result

    async def a_health_check(self) -> Dict[str, Any]:
        result = await super().a_health_check()
        result["adapter"] = "AsyncPgVectorChatStore"
        result["backend"] = "pgvector (async)"
        result["embedding_dim"] = self.config.embedding_dim
        result["has_embedding_service"] = self._embedding_service is not None
        return result

    async def a_get_statistics(self) -> Dict[str, Any]:
        result = await super().a_get_statistics()
        result["adapter"] = "AsyncPgVectorChatStore"
        result["backend"] = "pgvector (async)"

        # Count messages with embeddings
        try:
            async with self.db_manager.session_scope() as session:
                embedded_count = await session.execute(text(
                    "SELECT COUNT(*) FROM demo_messages WHERE embedding_vector IS NOT NULL"
                ))
                result["embedded_message_count"] = embedded_count.scalar() or 0
        except Exception:
            result["embedded_message_count"] = "N/A"

        return result

    # ============ Embedding ============

    def _get_embedding(self, text_str: str) -> Optional[List[float]]:
        """Get embedding vector for text."""
        if self._embedding_service is None:
            return None
        try:
            return self._embedding_service.encode(text_str)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    # ============ Override add_message to store embedding ============

    async def a_add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        """Add message with optional embedding vector (async)."""
        msg = await super().a_add_message(session_id, user_id, role, content, metadata)

        # Store embedding (best effort)
        if self._embedding_service is not None:
            try:
                embedding = self._get_embedding(content)
                if embedding:
                    async with self.db_manager.session_scope() as session:
                        await session.execute(text(
                            "UPDATE demo_messages SET embedding_vector = :vec WHERE id = :mid"
                        ), {"vec": str(embedding), "mid": msg.id})
            except Exception as e:
                logger.warning(f"Failed to store embedding for message {msg.id}: {e}")

        return msg

    # ============ Search (async hybrid: BM25 + vector) ============

    async def a_search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """Async hybrid search: BM25 + vector similarity."""
        self._check_connected()

        # Try vector search first
        vector_results = await self._vector_search(user_id, query, limit * 2, session_id)

        # BM25 search (ILIKE + BM25 from parent)
        bm25_results = await super().a_search_messages(user_id, query, limit * 2, session_id)

        if not vector_results:
            return bm25_results[:limit]

        if not bm25_results:
            return vector_results[:limit]

        # Merge results (reciprocal rank fusion)
        return self._rrf_merge(bm25_results, vector_results, limit)

    async def _vector_search(
        self, user_id: str, query: str,
        limit: int, session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """Async vector similarity search using pgvector. 过滤已软删除会话的消息。"""
        embedding = self._get_embedding(query)
        if embedding is None:
            return []

        try:
            async with self.db_manager.session_scope() as session:
                # Build filter — JOIN demo_sessions 过滤已删除会话
                filters = (
                    "m.user_id = :uid AND m.embedding_vector IS NOT NULL "
                    "AND s.is_active = true"
                )
                params = {"uid": user_id, "vec": str(embedding), "lim": limit}

                if session_id:
                    filters += " AND m.session_id = :sid"
                    params["sid"] = session_id

                sql = f"""
                    SELECT m.id, m.session_id, m.user_id, m.role, m.content,
                           m.metadata, m.created_at,
                           1 - (m.embedding_vector <=> :vec::vector) as similarity
                    FROM demo_messages m
                    JOIN demo_sessions s ON m.session_id = s.id
                    WHERE {filters}
                    ORDER BY m.embedding_vector <=> :vec::vector
                    LIMIT :lim
                """

                rows = (await session.execute(text(sql), params)).fetchall()

                results = []
                for row in rows:
                    result = await session.execute(
                        select(DemoMessage).filter(DemoMessage.id == row[0])
                    )
                    msg = result.scalars().first()
                    if msg:
                        results.append(msg)

                return results
        except Exception as e:
            logger.warning(f"Async vector search failed, falling back to BM25: {e}")
            return []

    def _rrf_merge(
        self,
        bm25_results: List[DemoMessage],
        vector_results: List[DemoMessage],
        limit: int,
        k: int = 60,
    ) -> List[DemoMessage]:
        """
        Reciprocal Rank Fusion to merge BM25 and vector results.

        score = 1/(k + rank_bm25) + 1/(k + rank_vector)
        """
        scores: Dict[str, float] = {}
        msg_map: Dict[str, DemoMessage] = {}

        for rank, msg in enumerate(bm25_results):
            scores[msg.id] = scores.get(msg.id, 0) + 1.0 / (k + rank + 1)
            msg_map[msg.id] = msg

        for rank, msg in enumerate(vector_results):
            scores[msg.id] = scores.get(msg.id, 0) + 1.0 / (k + rank + 1)
            msg_map[msg.id] = msg

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        return [msg_map[mid] for mid in sorted_ids[:limit]]
