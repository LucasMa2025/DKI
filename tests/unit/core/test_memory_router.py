"""
Unit tests for Memory Router.

Tests FAISS-based semantic memory retrieval:
- Memory indexing
- Semantic search
- Top-k retrieval
- Routing stability
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.fake_embeddings import FakeEmbeddingService
from fixtures.sample_memories import SAMPLE_MEMORIES, SAMPLE_QUERIES

from dki.core.memory_router import MemoryRouter, MemorySearchResult


class TestMemoryRouter:
    """Tests for MemoryRouter."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create fake embedding service."""
        return FakeEmbeddingService(embedding_dim=384)
    
    @pytest.fixture
    def router(self, embedding_service):
        """Create empty router."""
        return MemoryRouter(embedding_service)
    
    @pytest.fixture
    def router_with_memories(self, embedding_service):
        """Create router with sample memories."""
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES:
            router.add_memory(
                memory_id=mem['id'],
                content=mem['content'],
                metadata=mem['metadata'],
            )
        
        return router
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.index is None  # Lazy initialization
        assert len(router._memories) == 0
    
    def test_add_memory(self, router):
        """Test adding memory."""
        router.add_memory("mem_1", "Test memory content")
        
        assert len(router._memories) == 1
        assert "mem_1" in router._memories
    
    def test_add_memory_with_embedding(self, router, embedding_service):
        """Test adding memory with pre-computed embedding."""
        embedding = embedding_service.embed("Test content")
        
        router.add_memory(
            memory_id="mem_1",
            content="Test content",
            embedding=embedding,
        )
        
        assert len(router._memories) == 1
    
    def test_add_memory_with_metadata(self, router):
        """Test adding memory with metadata."""
        router.add_memory(
            memory_id="mem_1",
            content="Test content",
            metadata={"category": "test", "importance": "high"},
        )
        
        memory = router.get_memory("mem_1")
        assert memory['metadata']['category'] == "test"
    
    def test_add_memories_batch(self, router):
        """Test batch adding memories."""
        memories = [
            {"id": f"mem_{i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        
        count = router.add_memories(memories)
        
        assert count == 5
        assert len(router._memories) == 5
    
    def test_search_empty_router(self, router):
        """Test search on empty router."""
        results = router.search("test query")
        assert len(results) == 0
    
    def test_search_basic(self, router_with_memories):
        """Test basic search."""
        results = router_with_memories.search("今晚吃什么？", top_k=3, threshold=-1.0)
        
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, MemorySearchResult) for r in results)
    
    def test_search_returns_sorted(self, router_with_memories):
        """Test that search returns sorted results."""
        results = router_with_memories.search("推荐餐厅", top_k=5, threshold=-1.0)
        
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_search_with_threshold(self, router_with_memories):
        """Test search with similarity threshold."""
        results = router_with_memories.search(
            "完全不相关的查询 xyz abc",
            top_k=5,
            threshold=0.9,  # High threshold
        )
        
        # Should filter out low-score results
        for r in results:
            assert r.score >= 0.9
    
    def test_search_top_k_limit(self, router_with_memories):
        """Test top_k limit."""
        results = router_with_memories.search("测试", top_k=2, threshold=-1.0)
        assert len(results) <= 2
    
    def test_get_memory(self, router_with_memories):
        """Test getting memory by ID."""
        memory = router_with_memories.get_memory("mem_food_1")
        
        assert memory is not None
        assert "素食" in memory['content']
    
    def test_get_nonexistent_memory(self, router_with_memories):
        """Test getting non-existent memory."""
        memory = router_with_memories.get_memory("nonexistent")
        assert memory is None
    
    def test_remove_memory(self, router_with_memories):
        """Test removing memory."""
        initial_count = len(router_with_memories._memories)
        
        result = router_with_memories.remove_memory("mem_food_1")
        
        assert result is True
        assert len(router_with_memories._memories) == initial_count - 1
    
    def test_remove_nonexistent_memory(self, router_with_memories):
        """Test removing non-existent memory."""
        result = router_with_memories.remove_memory("nonexistent")
        assert result is False
    
    def test_clear(self, router_with_memories):
        """Test clearing router."""
        router_with_memories.clear()
        
        assert len(router_with_memories._memories) == 0
        assert router_with_memories.index is None
    
    def test_get_stats(self, router_with_memories):
        """Test getting statistics."""
        stats = router_with_memories.get_stats()
        
        assert stats['total_memories'] == len(SAMPLE_MEMORIES)
        assert stats['embedding_dim'] == 384
        assert stats['index_type'] == "faiss"


class TestRouterStability:
    """Tests for router stability and determinism."""
    
    @pytest.fixture
    def router(self):
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES:
            router.add_memory(mem['id'], mem['content'])
        
        return router
    
    def test_search_deterministic(self, router):
        """Test that search is deterministic."""
        query = "今晚吃什么？"
        
        results_1 = router.search(query, top_k=3)
        results_2 = router.search(query, top_k=3)
        
        assert len(results_1) == len(results_2)
        
        for r1, r2 in zip(results_1, results_2):
            assert r1.memory_id == r2.memory_id
            assert abs(r1.score - r2.score) < 1e-6
    
    def test_search_order_stable(self, router):
        """Test that search order is stable."""
        query = "推荐活动"
        
        # Run multiple times
        orders = []
        for _ in range(5):
            results = router.search(query, top_k=3)
            orders.append([r.memory_id for r in results])
        
        # All orders should be identical
        for order in orders[1:]:
            assert order == orders[0]


class TestMemorySearchResult:
    """Tests for MemorySearchResult dataclass."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = MemorySearchResult(
            memory_id="mem_1",
            content="Test content",
            score=0.85,
            metadata={"key": "value"},
        )
        
        assert result.memory_id == "mem_1"
        assert result.score == 0.85
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = MemorySearchResult(
            memory_id="mem_1",
            content="Test content",
            score=0.85,
            metadata={"key": "value"},
        )
        
        d = result.to_dict()
        
        assert d['memory_id'] == "mem_1"
        assert d['score'] == 0.85
        assert d['metadata']['key'] == "value"


class TestRouterRebuild:
    """Tests for router index rebuilding."""
    
    @pytest.fixture
    def router(self):
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        return MemoryRouter(embedding_service)
    
    def test_rebuild_index(self, router):
        """Test index rebuilding."""
        # Add memories
        for i in range(5):
            router.add_memory(f"mem_{i}", f"Content {i}")
        
        # Remove some
        router.remove_memory("mem_1")
        router.remove_memory("mem_3")
        
        # Rebuild
        router.rebuild_index()
        
        # Should have 3 memories
        assert len(router._memories) == 3
        
        # Search should still work
        results = router.search("Content", top_k=5, threshold=-1.0)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
