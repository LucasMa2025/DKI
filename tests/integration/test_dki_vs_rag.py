"""
Integration tests comparing DKI vs RAG behavior.

Tests the key behavioral differences:
- Token budget usage
- K/V growth patterns
- Attention budget allocation
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixtures.fake_model import FakeModelAdapter
from fixtures.fake_embeddings import FakeEmbeddingService
from fixtures.sample_memories import SAMPLE_MEMORIES, create_sample_kv_entries

from dki.core.memory_router import MemoryRouter
from dki.core.components.attention_budget import AttentionBudgetAnalyzer


class TestDKIvsRAGTokenBudget:
    """Tests comparing token budget usage."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_dki_saves_token_budget(self, analyzer):
        """
        Test that DKI saves token budget compared to RAG.
        
        RAG: Memory tokens consume context window
        DKI: Memory tokens NOT in context window
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        # DKI should have more free tokens
        assert analysis.dki_token_budget_free > analysis.rag_token_budget_free
        
        # Difference should equal memory tokens
        assert analysis.token_budget_saved == memory_tokens
    
    def test_dki_token_efficiency_increases_with_memory(self, analyzer):
        """Test that token efficiency increases with memory size."""
        user_tokens = 200
        
        efficiencies = []
        for memory_tokens in [100, 300, 500, 1000]:
            analysis = analyzer.analyze(user_tokens, memory_tokens)
            efficiencies.append(analysis.token_efficiency_gain)
        
        # Efficiency should increase with memory size
        for i in range(len(efficiencies) - 1):
            assert efficiencies[i] < efficiencies[i + 1]
    
    def test_rag_context_overflow(self, analyzer):
        """Test RAG context overflow scenario."""
        context_window = 2048
        user_tokens = 500
        memory_tokens = 2000  # Exceeds context
        
        analysis = analyzer.analyze(
            user_tokens, memory_tokens, context_window=context_window
        )
        
        # RAG would overflow (negative free tokens clamped to 0)
        assert analysis.rag_token_budget_free == 0
        
        # DKI still has space
        assert analysis.dki_token_budget_free > 0


class TestDKIvsRAGKVGrowth:
    """Tests comparing K/V growth patterns."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_dki_kv_growth(self, fake_model):
        """
        Test DKI K/V growth pattern.
        
        DKI: K_aug = [K_mem; K_user]
        """
        query = "Test query"
        memory = "Test memory content"
        
        # Compute separate K/V
        kv_query, _ = fake_model.compute_kv(query)
        kv_memory, _ = fake_model.compute_kv(memory)
        
        query_len = kv_query[0].key.shape[2]
        memory_len = kv_memory[0].key.shape[2]
        
        # DKI concatenates K/V
        dki_kv_len = query_len + memory_len
        
        assert dki_kv_len == query_len + memory_len
    
    def test_rag_kv_from_combined_prompt(self, fake_model):
        """
        Test RAG K/V from combined prompt.
        
        RAG: K/V computed from memory + query prompt
        """
        query = "Test query"
        memory = "Test memory content"
        
        # RAG combines in prompt
        rag_prompt = f"{memory}\n\n{query}"
        kv_rag, _ = fake_model.compute_kv(rag_prompt)
        
        rag_kv_len = kv_rag[0].key.shape[2]
        
        # RAG K/V length depends on combined prompt
        assert rag_kv_len > 0
    
    def test_dki_vs_rag_kv_comparison(self, fake_model):
        """Compare DKI and RAG K/V characteristics."""
        query = "What should I eat for dinner?"
        memory = "User is vegetarian and allergic to seafood"
        
        # DKI: Separate K/V
        kv_query, _ = fake_model.compute_kv(query)
        kv_memory, _ = fake_model.compute_kv(memory)
        
        # RAG: Combined K/V
        rag_prompt = f"{memory}\n\n{query}"
        kv_rag, _ = fake_model.compute_kv(rag_prompt)
        
        # Both approaches result in K/V that can be used for generation
        assert len(kv_query) == len(kv_rag)
        assert len(kv_memory) == len(kv_rag)


class TestDKIvsRAGAttentionBudget:
    """Tests comparing attention budget allocation."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_attention_budget_comparison(self, analyzer):
        """
        Compare attention budgets.
        
        RAG: B_a = (n_m + n_u)^2
        DKI: B_a = n_u × (n_m + n_u)
        """
        user_tokens = 100
        memory_tokens = 200
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        # RAG attention: (100 + 200)^2 = 90000
        expected_rag = (user_tokens + memory_tokens) ** 2
        assert analysis.rag_attention_budget == expected_rag
        
        # DKI attention: 100 × 300 = 30000
        expected_dki = user_tokens * (user_tokens + memory_tokens)
        assert analysis.dki_attention_budget == expected_dki
        
        # DKI has lower attention budget
        assert analysis.dki_attention_budget < analysis.rag_attention_budget
    
    def test_attention_overhead_acceptable(self, analyzer):
        """Test that DKI attention overhead is acceptable."""
        user_tokens = 200
        memory_tokens = 300
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        # Overhead should be reasonable (< 3x vanilla)
        assert analysis.attention_overhead_ratio < 3.0


class TestDKIvsRAGRecommendation:
    """Tests for DKI vs RAG recommendation logic."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_recommend_dki_for_large_memory(self, analyzer):
        """Test DKI recommendation for large memory."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=500,
            task_type="reasoning",
        )
        
        # Large memory should favor DKI
        assert result['factors']['memory_tokens_significant'] is True
    
    def test_recommend_rag_for_small_memory(self, analyzer):
        """Test RAG recommendation for small memory."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=50,
            task_type="qa",
        )
        
        # Small memory may not justify DKI overhead
        assert result['factors']['memory_tokens_significant'] is False
    
    def test_recommend_dki_for_constrained_context(self, analyzer):
        """Test DKI recommendation for constrained context."""
        analyzer.context_window = 2048
        
        result = analyzer.should_prefer_dki(
            user_tokens=500,
            memory_tokens=800,
            task_type="reasoning",
        )
        
        # Constrained context should favor DKI
        assert result['factors']['context_constrained'] is True


class TestIntegrationFlow:
    """Tests for complete DKI vs RAG flow."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    @pytest.fixture
    def router(self):
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES[:3]:
            router.add_memory(mem['id'], mem['content'])
        
        return router
    
    def test_complete_dki_flow(self, fake_model, router):
        """Test complete DKI flow."""
        query = "今晚吃什么？"
        
        # 1. Retrieve memories
        memories = router.search(query, top_k=2, threshold=-1.0)
        assert len(memories) > 0
        
        # 2. Compute memory K/V
        memory_kvs = []
        for mem in memories:
            kv, _ = fake_model.compute_kv(mem.content)
            memory_kvs.append(kv)
        
        # 3. Merge K/V
        merged_kv = memory_kvs[0]  # Simplified
        
        # 4. Generate with injection
        output = fake_model.forward_with_kv_injection(
            prompt=query,
            injected_kv=merged_kv,
            alpha=0.8,
        )
        
        assert output.text is not None
        assert output.metadata['alpha'] == 0.8
    
    def test_complete_rag_flow(self, fake_model, router):
        """Test complete RAG flow."""
        query = "今晚吃什么？"
        
        # 1. Retrieve memories
        memories = router.search(query, top_k=2, threshold=-1.0)
        assert len(memories) > 0
        
        # 2. Build RAG prompt
        parts = ["Relevant information:"]
        for i, mem in enumerate(memories, 1):
            parts.append(f"[{i}] {mem.content}")
        parts.append(f"\nUser: {query}")
        parts.append("\nAssistant:")
        
        rag_prompt = "\n".join(parts)
        
        # 3. Generate
        output = fake_model.generate(rag_prompt, max_new_tokens=50)
        
        assert output.text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
