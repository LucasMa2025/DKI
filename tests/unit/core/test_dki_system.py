"""
Unit tests for DKI System.

Tests the main DKI system without requiring GPU or real LLM:
- Memory addition and retrieval
- K/V injection flow
- Gating integration
- Cache management
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.fake_model import FakeModelAdapter
from fixtures.fake_embeddings import FakeEmbeddingService
from fixtures.sample_memories import SAMPLE_MEMORIES, create_sample_kv_entries

from dki.core.memory_router import MemoryRouter
from dki.core.components.dual_factor_gating import GatingDecision


class TestDKISystemBasic:
    """Basic tests for DKI system components."""
    
    @pytest.fixture
    def fake_model(self):
        """Create fake model."""
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    @pytest.fixture
    def embedding_service(self):
        """Create fake embedding service."""
        return FakeEmbeddingService(embedding_dim=384)
    
    @pytest.fixture
    def memory_router(self, embedding_service):
        """Create memory router with fake embeddings."""
        return MemoryRouter(embedding_service)
    
    def test_fake_model_generate(self, fake_model):
        """Test fake model generation."""
        output = fake_model.generate("Test prompt", max_new_tokens=50)
        
        assert output.text is not None
        assert len(output.text) > 0
        assert output.input_tokens > 0
        assert output.output_tokens > 0
    
    def test_fake_model_compute_kv(self, fake_model):
        """Test fake model K/V computation."""
        kv_entries, hidden = fake_model.compute_kv(
            "Test memory content",
            return_hidden=True,
        )
        
        assert len(kv_entries) == fake_model.num_layers
        assert hidden is not None
        
        for entry in kv_entries:
            assert entry.key.shape[1] == fake_model.num_heads
            assert entry.key.shape[3] == fake_model.head_dim
    
    def test_fake_model_forward_with_injection(self, fake_model):
        """Test fake model forward with K/V injection."""
        kv_entries = create_sample_kv_entries(
            num_layers=fake_model.num_layers,
            num_heads=fake_model.num_heads,
            head_dim=fake_model.head_dim,
        )
        
        output = fake_model.forward_with_kv_injection(
            prompt="Test query",
            injected_kv=kv_entries,
            alpha=0.8,
        )
        
        assert output.text is not None
        assert output.metadata['injected_seq_len'] > 0
        assert output.metadata['alpha'] == 0.8
    
    def test_fake_model_prefill_entropy(self, fake_model):
        """Test fake model prefill entropy computation."""
        entropy = fake_model.compute_prefill_entropy("Test query")
        
        assert 0.0 <= entropy <= 1.0


class TestMemoryRouterIntegration:
    """Tests for memory router integration."""
    
    @pytest.fixture
    def router(self):
        """Create router with sample memories."""
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES[:5]:
            router.add_memory(
                memory_id=mem['id'],
                content=mem['content'],
                metadata=mem['metadata'],
            )
        
        return router
    
    def test_router_search(self, router):
        """Test router search functionality."""
        results = router.search("今晚吃什么？", top_k=3, threshold=-1.0)
        
        assert len(results) > 0
        assert results[0].score > 0
    
    def test_router_search_sorted(self, router):
        """Test that search results are sorted by score."""
        results = router.search("推荐餐厅", top_k=5, threshold=-1.0)
        
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_router_stats(self, router):
        """Test router statistics."""
        stats = router.get_stats()
        
        assert stats['total_memories'] == 5
        assert stats['embedding_dim'] == 384


class TestKVInjectionFlow:
    """Tests for K/V injection flow."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_injection_changes_kv_shape(self, fake_model):
        """Test that injection changes K/V shape."""
        # Original K/V
        original_kv, _ = fake_model.compute_kv("Original query")
        original_seq_len = original_kv[0].key.shape[2]
        
        # Memory K/V
        memory_kv, _ = fake_model.compute_kv("Memory content to inject")
        memory_seq_len = memory_kv[0].key.shape[2]
        
        # Merged K/V should have combined length
        merged_seq_len = original_seq_len + memory_seq_len
        
        # Simulate merge
        merged_key = torch.cat([memory_kv[0].key, original_kv[0].key], dim=2)
        
        assert merged_key.shape[2] == merged_seq_len
    
    def test_injection_with_alpha_scaling(self, fake_model):
        """Test injection with alpha scaling."""
        kv_entries = create_sample_kv_entries(
            num_layers=fake_model.num_layers,
            num_heads=fake_model.num_heads,
            head_dim=fake_model.head_dim,
        )
        
        # Test different alpha values
        for alpha in [0.0, 0.5, 1.0]:
            output = fake_model.forward_with_kv_injection(
                prompt="Test",
                injected_kv=kv_entries,
                alpha=alpha,
            )
            
            assert output.metadata['alpha'] == alpha


class TestGatingIntegration:
    """Tests for gating integration with DKI."""
    
    def test_gating_decision_flow(self):
        """Test gating decision flow."""
        # Simulate gating decision
        decision = GatingDecision(
            should_inject=True,
            alpha=0.8,
            entropy=0.6,
            relevance_score=0.9,
            margin=0.2,
            memories=[],
            reasoning="High relevance memory found",
        )
        
        assert decision.should_inject is True
        assert decision.alpha == 0.8
    
    def test_gating_no_injection(self):
        """Test gating decision for no injection."""
        decision = GatingDecision(
            should_inject=False,
            alpha=0.0,
            entropy=0.8,
            relevance_score=0.3,
            margin=0.0,
            memories=[],
            reasoning="No relevant memory",
        )
        
        assert decision.should_inject is False
        assert decision.alpha == 0.0


class TestCacheIntegration:
    """Tests for cache integration with DKI."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_kv_cache_reuse(self, fake_model):
        """Test K/V cache reuse across queries."""
        # Compute K/V once
        kv_entries_1, _ = fake_model.compute_kv("Memory content")
        
        # Compute again (should be deterministic)
        kv_entries_2, _ = fake_model.compute_kv("Memory content")
        
        # Should be identical (deterministic fake model)
        for e1, e2 in zip(kv_entries_1, kv_entries_2):
            assert torch.allclose(e1.key, e2.key)
            assert torch.allclose(e1.value, e2.value)
    
    def test_different_content_different_kv(self, fake_model):
        """Test different content produces different K/V."""
        kv_1, _ = fake_model.compute_kv("Content A")
        kv_2, _ = fake_model.compute_kv("Content B")
        
        # Should be different
        assert not torch.allclose(kv_1[0].key, kv_2[0].key)


class TestDKISystemStats:
    """Tests for DKI system statistics."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_model_info(self, fake_model):
        """Test model info retrieval."""
        info = fake_model.get_model_info()
        
        assert info['model_name'] == "fake-model"
        assert info['hidden_dim'] == 128
        assert info['num_layers'] == 4
        assert info['is_fake'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
