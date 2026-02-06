"""
Integration tests for K/V injection flow.

Tests the complete injection pipeline:
- Memory retrieval
- K/V computation
- Projection
- Scaling
- Injection
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
from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling
from dki.core.components.query_conditioned_projection import QueryConditionedProjection
from dki.models.base import KVCacheEntry


class TestKVInjectionPipeline:
    """Tests for complete K/V injection pipeline."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    @pytest.fixture
    def projection(self):
        return QueryConditionedProjection(hidden_dim=128, rank=16, dropout=0.0)
    
    @pytest.fixture
    def router(self):
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        for mem in SAMPLE_MEMORIES[:3]:
            router.add_memory(mem['id'], mem['content'])
        
        return router
    
    def test_retrieval_to_injection(self, fake_model, router, mis):
        """Test flow from retrieval to injection."""
        query = "今晚吃什么？"
        
        # Step 1: Retrieve
        memories = router.search(query, top_k=2, threshold=-1.0)
        assert len(memories) > 0
        
        # Step 2: Compute K/V for each memory
        all_kv = []
        for mem in memories:
            kv, _ = fake_model.compute_kv(mem.content)
            all_kv.append(kv)
        
        # Step 3: Merge K/V
        merged_kv = self._merge_kv(all_kv)
        
        # Step 4: Apply alpha scaling
        alpha = 0.7
        scaled_kv = []
        for entry in merged_kv:
            key, value = mis.scale_kv_values(entry.key, entry.value, alpha)
            scaled_kv.append(KVCacheEntry(key=key, value=value, layer_idx=entry.layer_idx))
        
        # Step 5: Inject
        output = fake_model.forward_with_kv_injection(
            prompt=query,
            injected_kv=scaled_kv,
            alpha=alpha,
        )
        
        assert output.text is not None
        assert output.metadata['injected_seq_len'] > 0
    
    def test_projection_in_pipeline(self, fake_model, projection):
        """Test projection integration in pipeline."""
        # Compute memory hidden states
        memory_text = "用户是素食主义者"
        _, hidden_mem = fake_model.compute_kv(memory_text, return_hidden=True)
        
        # Compute user hidden states
        query_text = "推荐餐厅"
        _, hidden_user = fake_model.compute_kv(query_text, return_hidden=True)
        
        # Apply projection
        projected = projection(hidden_mem.squeeze(0), hidden_user.squeeze(0))
        
        assert projected.shape == hidden_mem.squeeze(0).shape
    
    def test_alpha_scaling_effect(self, fake_model, mis):
        """Test effect of different alpha values."""
        kv_entries = create_sample_kv_entries(
            num_layers=4,
            num_heads=4,
            head_dim=32,
        )
        
        alphas = [0.0, 0.3, 0.7, 1.0]
        results = []
        
        for alpha in alphas:
            scaled_kv = []
            for entry in kv_entries:
                key, value = mis.scale_kv_values(entry.key, entry.value, alpha)
                scaled_kv.append(KVCacheEntry(key=key, value=value, layer_idx=entry.layer_idx))
            
            # Check value norms
            value_norm = sum(e.value.norm().item() for e in scaled_kv)
            results.append(value_norm)
        
        # Value norms should increase with alpha
        assert results[0] == 0.0  # alpha=0 zeros out
        assert results[-1] > results[1]  # alpha=1 > alpha=0.3
    
    def _merge_kv(self, kv_list):
        """Merge K/V from multiple memories."""
        if not kv_list:
            return []
        
        num_layers = len(kv_list[0])
        merged = []
        
        for layer_idx in range(num_layers):
            keys = [kv[layer_idx].key for kv in kv_list]
            values = [kv[layer_idx].value for kv in kv_list]
            
            merged_key = torch.cat(keys, dim=2)
            merged_value = torch.cat(values, dim=2)
            
            merged.append(KVCacheEntry(
                key=merged_key,
                value=merged_value,
                layer_idx=layer_idx,
            ))
        
        return merged


class TestKVMerging:
    """Tests for K/V merging from multiple memories."""
    
    def test_merge_two_memories(self):
        """Test merging K/V from two memories."""
        kv_1 = create_sample_kv_entries(num_layers=2, seq_len=5, seed=1)
        kv_2 = create_sample_kv_entries(num_layers=2, seq_len=7, seed=2)
        
        # Merge
        merged = []
        for e1, e2 in zip(kv_1, kv_2):
            merged_key = torch.cat([e1.key, e2.key], dim=2)
            merged_value = torch.cat([e1.value, e2.value], dim=2)
            merged.append(KVCacheEntry(
                key=merged_key,
                value=merged_value,
                layer_idx=e1.layer_idx,
            ))
        
        # Check merged length
        assert merged[0].key.shape[2] == 5 + 7
    
    def test_merge_preserves_layers(self):
        """Test that merging preserves layer structure."""
        kv_1 = create_sample_kv_entries(num_layers=4, seq_len=5)
        kv_2 = create_sample_kv_entries(num_layers=4, seq_len=5)
        
        # Merge
        merged = []
        for e1, e2 in zip(kv_1, kv_2):
            merged_key = torch.cat([e1.key, e2.key], dim=2)
            merged_value = torch.cat([e1.value, e2.value], dim=2)
            merged.append(KVCacheEntry(
                key=merged_key,
                value=merged_value,
                layer_idx=e1.layer_idx,
            ))
        
        assert len(merged) == 4
        for i, entry in enumerate(merged):
            assert entry.layer_idx == i


class TestInjectionWithCache:
    """Tests for injection with caching."""
    
    @pytest.fixture
    def fake_model(self):
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    def test_cached_kv_reuse(self, fake_model):
        """Test that cached K/V can be reused."""
        memory_text = "用户是素食主义者"
        
        # Compute K/V (would be cached in real system)
        kv_1, _ = fake_model.compute_kv(memory_text)
        kv_2, _ = fake_model.compute_kv(memory_text)
        
        # Should be identical (deterministic)
        for e1, e2 in zip(kv_1, kv_2):
            assert torch.allclose(e1.key, e2.key)
            assert torch.allclose(e1.value, e2.value)
    
    def test_injection_with_different_queries(self, fake_model):
        """Test injection with same memory, different queries."""
        memory_text = "用户是素食主义者"
        kv_memory, _ = fake_model.compute_kv(memory_text)
        
        queries = ["今晚吃什么？", "推荐餐厅", "有什么好吃的？"]
        
        for query in queries:
            output = fake_model.forward_with_kv_injection(
                prompt=query,
                injected_kv=kv_memory,
                alpha=0.8,
            )
            
            assert output.text is not None
            assert output.metadata['injected_seq_len'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
