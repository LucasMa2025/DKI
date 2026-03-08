"""
PgVectorChatStore - PostgreSQL + pgvector implementation of IChatStore

Extends PostgresChatStore with vector similarity search (hybrid: BM25 + cosine).

Author: AGI Demo Project
Version: 3.0.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import or_, text

from demo.store.postgres_store import PostgresChatStore
from demo.store.connection import DemoDBConfig
from demo.store.models import DemoMessage


class PgVectorChatStore(PostgresChatStore):
    """PostgreSQL + pgvector chat store with hybrid search."""

    def __init__(self, config: DemoDBConfig, embedding_service=None):
        """
        Args:
            config: database configuration
            embedding_service: optional embedding service with encode(text) -> List[float]
        """
        super().__init__(config)
        self._embedding_service = embedding_service

    # ============ Lifecycle (override for pgvector info) ============

    def connect(self) -> None:
        self.db_manager.init_database()
        logger.info(
            f"PgVectorChatStore connected: "
            f"{self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database} "
            f"(embedding_dim={self.config.embedding_dim})"
        )

    def health_check(self) -> Dict[str, Any]:
        result = super().health_check()
        result["adapter"] = "PgVectorChatStore"
        result["backend"] = "pgvector"
        result["embedding_dim"] = self.config.embedding_dim
        result["has_embedding_service"] = self._embedding_service is not None
        return result

    def get_statistics(self) -> Dict[str, Any]:
        result = super().get_statistics()
        result["adapter"] = "PgVectorChatStore"
        result["backend"] = "pgvector"

        # Count messages with embeddings
        try:
            with self.db_manager.session_scope() as session:
                embedded_count = session.execute(text(
                    "SELECT COUNT(*) FROM demo_messages WHERE embedding_vector IS NOT NULL"
                )).scalar() or 0
                result["embedded_message_count"] = embedded_count
        except Exception:
            result["embedded_message_count"] = "N/A"

        return result

    # ============ Embedding ============

    def _get_embedding(self, text_str: str) -> Optional[List[float]]:
        """Get embedding vector for text"""
        if self._embedding_service is None:
            return None
        try:
            return self._embedding_service.encode(text_str)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    # ============ Override add_message to store embedding ============

    def add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        """Add message with optional embedding vector."""
        msg = super().add_message(session_id, user_id, role, content, metadata)

        # Async-store embedding (best effort)
        if self._embedding_service is not None:
            try:
                embedding = self._get_embedding(content)
                if embedding:
                    with self.db_manager.session_scope() as session:
                        session.execute(text(
                            "UPDATE demo_messages SET embedding_vector = :vec WHERE id = :mid"
                        ), {"vec": str(embedding), "mid": msg.id})
            except Exception as e:
                logger.warning(f"Failed to store embedding for message {msg.id}: {e}")

        return msg

    # ============ Search (hybrid: BM25 + vector) ============

    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """Hybrid search: BM25 + vector similarity."""
        self._check_connected()

        # Try vector search first
        vector_results = self._vector_search(user_id, query, limit * 2, session_id)

        # BM25 search (ILIKE + BM25 from parent)
        bm25_results = super().search_messages(user_id, query, limit * 2, session_id)

        if not vector_results:
            return bm25_results[:limit]

        if not bm25_results:
            return vector_results[:limit]

        # Merge results (reciprocal rank fusion)
        return self._rrf_merge(bm25_results, vector_results, limit)

    def _vector_search(
        self, user_id: str, query: str,
        limit: int, session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """Vector similarity search using pgvector. 过滤已软删除会话的消息。"""
        embedding = self._get_embedding(query)
        if embedding is None:
            return []

        try:
            with self.db_manager.session_scope() as session:
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

                rows = session.execute(text(sql), params).fetchall()

                results = []
                for row in rows:
                    msg = session.query(DemoMessage).filter(
                        DemoMessage.id == row[0]
                    ).first()
                    if msg:
                        session.expunge(msg)
                        results.append(msg)

                return results
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to BM25: {e}")
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
