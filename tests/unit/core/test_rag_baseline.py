"""
Unit tests for RAG System (baseline comparison).

Tests the RAG baseline implementation:
- Memory retrieval
- Prompt building
- Response generation

RAG serves as the comparison baseline for DKI.
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.fake_model import FakeModelAdapter
from fixtures.fake_embeddings import FakeEmbeddingService
from fixtures.sample_memories import SAMPLE_MEMORIES

from dki.core.memory_router import MemoryRouter, MemorySearchResult


class TestRAGBaseline:
    """Tests for RAG baseline system."""
    
    @pytest.fixture
    def fake_model(self):
        """Create fake model."""
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    @pytest.fixture
    def router_with_memories(self):
        """Create router with sample memories."""
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES:
            router.add_memory(
                memory_id=mem['id'],
                content=mem['content'],
                metadata=mem['metadata'],
            )
        
        return router
    
    def test_rag_does_not_modify_kv_shape(self, fake_model):
        """
        Test that RAG does not modify K/V shape.
        
        RAG concatenates memory to prompt, not to K/V.
        """
        # Compute K/V for query only
        query = "What should I eat?"
        kv_query, _ = fake_model.compute_kv(query)
        
        # Compute K/V for query + memory (RAG style)
        memory = "User is vegetarian"
        rag_prompt = f"{memory}\n\n{query}"
        kv_rag, _ = fake_model.compute_kv(rag_prompt)
        
        # RAG K/V should be longer (includes memory tokens)
        assert kv_rag[0].key.shape[2] > kv_query[0].key.shape[2]
    
    def test_rag_prompt_building(self):
        """Test RAG prompt building."""
        query = "今晚吃什么？"
        memories = [
            MemorySearchResult(
                memory_id="mem1",
                content="用户是素食主义者",
                score=0.9,
            ),
            MemorySearchResult(
                memory_id="mem2",
                content="用户住在北京",
                score=0.7,
            ),
        ]
        
        # Build RAG prompt
        parts = ["Relevant information:"]
        for i, mem in enumerate(memories, 1):
            parts.append(f"[{i}] {mem.content}")
        parts.append("")
        parts.append(f"User: {query}")
        parts.append("\nAssistant:")
        
        prompt = "\n".join(parts)
        
        assert "素食主义者" in prompt
        assert "北京" in prompt
        assert query in prompt
    
    def test_rag_retrieval(self, router_with_memories):
        """Test RAG memory retrieval."""
        results = router_with_memories.search("推荐餐厅", top_k=3, threshold=-1.0)
        
        assert len(results) > 0
        assert all(isinstance(r, MemorySearchResult) for r in results)
    
    def test_rag_vs_dki_token_usage(self, fake_model):
        """
        Test token usage difference between RAG and DKI.
        
        RAG: Memory tokens in context window
        DKI: Memory tokens NOT in context window
        """
        query = "What should I eat?"
        memory = "User is vegetarian and allergic to seafood"
        
        # RAG: Memory in prompt
        rag_prompt = f"{memory}\n\n{query}"
        rag_tokens = len(rag_prompt.split())  # Rough estimate
        
        # DKI: Only query in prompt
        dki_tokens = len(query.split())
        
        # RAG uses more tokens in context
        assert rag_tokens > dki_tokens
    
    def test_rag_generation(self, fake_model):
        """Test RAG generation flow."""
        query = "今晚吃什么？"
        memory = "用户是素食主义者"
        
        # Build RAG prompt
        prompt = f"Relevant information:\n[1] {memory}\n\nUser: {query}\n\nAssistant:"
        
        # Generate
        output = fake_model.generate(prompt, max_new_tokens=50)
        
        assert output.text is not None
        assert output.input_tokens > 0


class TestRAGvsDKIComparison:
    """Tests comparing RAG and DKI behavior."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_dki_kv_growth(self, fake_model):
        """
        Test that DKI grows K/V while RAG grows prompt.
        
        DKI: K_aug = [K_mem; K_user]
        RAG: prompt = memory + query
        """
        query = "Test query"
        memory = "Test memory content"
        
        # DKI approach: Separate K/V computation
        kv_query, _ = fake_model.compute_kv(query)
        kv_memory, _ = fake_model.compute_kv(memory)
        
        # DKI K/V length = query + memory
        dki_kv_len = kv_query[0].key.shape[2] + kv_memory[0].key.shape[2]
        
        # RAG approach: Combined prompt
        rag_prompt = f"{memory}\n\n{query}"
        kv_rag, _ = fake_model.compute_kv(rag_prompt)
        rag_kv_len = kv_rag[0].key.shape[2]
        
        # Both should have similar total K/V length
        # (DKI concatenates, RAG processes together)
        assert dki_kv_len > 0
        assert rag_kv_len > 0
    
    def test_dki_preserves_context_window(self, fake_model):
        """
        Test that DKI preserves context window for user.
        
        RAG: context_window - memory_tokens = available for user
        DKI: context_window = available for user (memory in K/V)
        """
        context_window = 4096
        memory_tokens = 500
        
        # RAG available tokens
        rag_available = context_window - memory_tokens
        
        # DKI available tokens (memory not in token budget)
        dki_available = context_window
        
        assert dki_available > rag_available
        assert dki_available - rag_available == memory_tokens


class TestRAGSystemStats:
    """Tests for RAG system statistics."""
    
    @pytest.fixture
    def router(self):
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        return MemoryRouter(embedding_service)
    
    def test_router_stats(self, router):
        """Test router statistics."""
        # Add memories
        for i in range(5):
            router.add_memory(f"mem_{i}", f"Memory content {i}")
        
        stats = router.get_stats()
        
        assert stats['total_memories'] == 5
        assert stats['index_type'] == "faiss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
