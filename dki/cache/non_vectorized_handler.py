"""
Non-Vectorized Data Handler
Handles message data that doesn't have pre-computed embeddings

Strategies:
- Lazy: Real-time embedding computation (for small datasets)
- Batch: Pre-compute embeddings in batches
- Hybrid: BM25 initial filtering + embedding reranking

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from dki.adapters.base import ChatMessage


class SearchStrategy(str, Enum):
    """Search strategy for non-vectorized data."""
    LAZY = "lazy"  # Real-time embedding computation
    BATCH = "batch"  # Pre-computed embeddings
    HYBRID = "hybrid"  # BM25 + embedding reranking


@dataclass
class HandlerConfig:
    """Configuration for non-vectorized data handler."""
    # Strategy selection
    default_strategy: SearchStrategy = SearchStrategy.HYBRID
    
    # Thresholds for automatic strategy selection
    lazy_max_messages: int = 100  # Use lazy for < 100 messages
    batch_trigger_count: int = 1000  # Trigger batch for > 1000 messages
    
    # BM25 settings
    bm25_candidates_multiplier: int = 4  # top_k * multiplier = BM25 candidates
    bm25_min_candidates: int = 20
    
    # Embedding cache
    cache_embeddings: bool = True
    cache_max_size: int = 100000  # Max cached embeddings
    cache_ttl_hours: int = 168  # 7 days
    
    # Batch computation
    batch_size: int = 100
    max_concurrent_batches: int = 4


@dataclass
class SearchResult:
    """Search result with relevance score."""
    message: ChatMessage
    score: float
    strategy_used: SearchStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingCache:
    """
    Simple embedding cache with LRU eviction.
    
    Key: message_id or content_hash
    Value: embedding vector
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        async with self._lock:
            if key in self._cache:
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    async def put(self, key: str, embedding: List[float]) -> None:
        """Put embedding in cache."""
        async with self._lock:
            if key in self._cache:
                # Update existing
                self._access_order.remove(key)
                self._access_order.append(key)
            else:
                # Add new
                if len(self._cache) >= self.max_size:
                    # Evict oldest
                    oldest_key = self._access_order.pop(0)
                    del self._cache[oldest_key]
                
                self._access_order.append(key)
            
            self._cache[key] = embedding
    
    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


class NonVectorizedDataHandler:
    """
    Handler for non-vectorized message data.
    
    Provides multiple strategies for searching messages
    that don't have pre-computed embeddings.
    
    Example:
        handler = NonVectorizedDataHandler(embedding_service)
        
        # Search with automatic strategy selection
        results = await handler.search_relevant_messages(
            messages=messages,
            query="What did we discuss about food?",
            top_k=5,
        )
        
        # Force specific strategy
        results = await handler.search_relevant_messages(
            messages=messages,
            query="food preferences",
            top_k=5,
            strategy=SearchStrategy.HYBRID,
        )
    """
    
    def __init__(
        self,
        embedding_service: Any,  # EmbeddingService
        config: Optional[HandlerConfig] = None,
    ):
        """
        Initialize handler.
        
        Args:
            embedding_service: Service for computing embeddings
            config: Handler configuration
        """
        self.embedding_service = embedding_service
        self.config = config or HandlerConfig()
        
        # Embedding cache
        self._embedding_cache = EmbeddingCache(
            max_size=self.config.cache_max_size
        )
        
        # BM25 index (lazy initialized)
        self._bm25_index = None
        self._bm25_messages: List[ChatMessage] = []
        
        # Stats
        self._stats = {
            "lazy_searches": 0,
            "batch_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "embeddings_computed": 0,
        }
        
        logger.info(f"NonVectorizedDataHandler initialized (strategy={self.config.default_strategy})")
    
    def _select_strategy(self, message_count: int) -> SearchStrategy:
        """Select optimal strategy based on message count."""
        if message_count <= self.config.lazy_max_messages:
            return SearchStrategy.LAZY
        elif message_count >= self.config.batch_trigger_count:
            return SearchStrategy.BATCH
        else:
            return SearchStrategy.HYBRID
    
    async def search_relevant_messages(
        self,
        messages: List[ChatMessage],
        query: str,
        top_k: int = 5,
        strategy: Optional[SearchStrategy] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant messages.
        
        Args:
            messages: List of messages to search
            query: Search query
            top_k: Number of results to return
            strategy: Search strategy (auto-selected if None)
            
        Returns:
            List of SearchResult sorted by relevance
        """
        if not messages:
            return []
        
        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(len(messages))
        
        # Execute search
        if strategy == SearchStrategy.LAZY:
            return await self._lazy_search(messages, query, top_k)
        elif strategy == SearchStrategy.BATCH:
            return await self._batch_search(messages, query, top_k)
        else:  # HYBRID
            return await self._hybrid_search(messages, query, top_k)
    
    async def _lazy_search(
        self,
        messages: List[ChatMessage],
        query: str,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Lazy search: compute embeddings on-demand.
        
        Best for small datasets (< 100 messages).
        """
        self._stats["lazy_searches"] += 1
        
        # Compute query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Score all messages
        scored_messages = []
        
        for msg in messages:
            # Check cache first
            cache_key = msg.message_id or msg.content_hash()
            cached_embedding = await self._embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                msg_embedding = cached_embedding
                self._stats["cache_hits"] += 1
            else:
                # Compute embedding
                msg_embedding = self.embedding_service.embed(msg.content)
                self._stats["embeddings_computed"] += 1
                
                # Cache if enabled
                if self.config.cache_embeddings:
                    await self._embedding_cache.put(cache_key, msg_embedding)
            
            # Compute similarity
            score = self._cosine_similarity(query_embedding, msg_embedding)
            scored_messages.append((msg, score))
        
        # Sort and return top_k
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                message=msg,
                score=score,
                strategy_used=SearchStrategy.LAZY,
            )
            for msg, score in scored_messages[:top_k]
        ]
    
    async def _batch_search(
        self,
        messages: List[ChatMessage],
        query: str,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Batch search: use pre-computed embeddings.
        
        Assumes embeddings are already computed or will be batch-computed.
        """
        self._stats["batch_searches"] += 1
        
        # Ensure all messages have embeddings
        messages_without_embeddings = [
            msg for msg in messages
            if msg.embedding is None and msg.message_id not in self._embedding_cache
        ]
        
        if messages_without_embeddings:
            # Batch compute missing embeddings
            await self.batch_precompute_embeddings(messages_without_embeddings)
        
        # Now do lazy search (all embeddings should be cached)
        return await self._lazy_search(messages, query, top_k)
    
    async def _hybrid_search(
        self,
        messages: List[ChatMessage],
        query: str,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Hybrid search: BM25 initial filtering + embedding reranking.
        
        Best for medium to large datasets (100-10000 messages).
        
        Flow:
        1. BM25 quick filter to get top-N candidates
        2. Compute embeddings only for candidates
        3. Rerank by embedding similarity
        """
        self._stats["hybrid_searches"] += 1
        
        # Calculate BM25 candidates
        bm25_candidates = max(
            self.config.bm25_min_candidates,
            top_k * self.config.bm25_candidates_multiplier
        )
        
        # Step 1: BM25 filtering
        candidates = await self._bm25_filter(messages, query, bm25_candidates)
        
        if not candidates:
            return []
        
        # Step 2: Embedding reranking
        return await self._lazy_search(candidates, query, top_k)
    
    async def _bm25_filter(
        self,
        messages: List[ChatMessage],
        query: str,
        top_n: int,
    ) -> List[ChatMessage]:
        """Filter messages using BM25."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed, falling back to simple search")
            return messages[:top_n]
        
        # Tokenize messages
        tokenized_messages = [
            self._tokenize(msg.content) for msg in messages
        ]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_messages)
        
        # Score query
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        # Get top-N candidates
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_n]
        
        return [messages[i] for i in scored_indices]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Basic tokenization: lowercase, split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def batch_precompute_embeddings(
        self,
        messages: List[ChatMessage],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Batch pre-compute embeddings for messages.
        
        Args:
            messages: Messages to compute embeddings for
            progress_callback: Optional callback(computed, total)
            
        Returns:
            Number of embeddings computed
        """
        computed = 0
        total = len(messages)
        
        # Process in batches
        for i in range(0, total, self.config.batch_size):
            batch = messages[i:i + self.config.batch_size]
            texts = [msg.content for msg in batch]
            
            # Batch compute embeddings
            if hasattr(self.embedding_service, 'embed_batch'):
                embeddings = self.embedding_service.embed_batch(texts)
            else:
                # Fallback to individual computation
                embeddings = [self.embedding_service.embed(t) for t in texts]
            
            # Cache embeddings
            for msg, embedding in zip(batch, embeddings):
                cache_key = msg.message_id or msg.content_hash()
                await self._embedding_cache.put(cache_key, embedding)
                computed += 1
                self._stats["embeddings_computed"] += 1
            
            if progress_callback:
                progress_callback(computed, total)
        
        logger.info(f"Pre-computed {computed} embeddings")
        return computed
    
    async def clear_cache(self) -> None:
        """Clear embedding cache."""
        await self._embedding_cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            **self._stats,
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self.config.cache_max_size,
        }


# Convenience function for creating handler
def create_non_vectorized_handler(
    embedding_service: Any,
    strategy: str = "hybrid",
    cache_embeddings: bool = True,
    **kwargs,
) -> NonVectorizedDataHandler:
    """
    Create a non-vectorized data handler.
    
    Args:
        embedding_service: Embedding service
        strategy: Default strategy ("lazy", "batch", "hybrid")
        cache_embeddings: Whether to cache embeddings
        **kwargs: Additional config options
        
    Returns:
        Configured handler
    """
    config = HandlerConfig(
        default_strategy=SearchStrategy(strategy),
        cache_embeddings=cache_embeddings,
        **kwargs,
    )
    
    return NonVectorizedDataHandler(
        embedding_service=embedding_service,
        config=config,
    )
