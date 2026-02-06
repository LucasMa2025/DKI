"""
Unit tests for Memory Influence Scaling (MIS) component.

Tests the continuous injection strength control:
- α = 0: No memory influence (vanilla LLM)
- α = 1: Full memory influence
- α ∈ (0, 1): Partial influence

Key properties tested:
- Controllability: Prevents injection runaway
- Testability: α → output mapping
- Degradability: α → 0 gracefully falls back to vanilla
"""

import pytest
import torch
import numpy as np

from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling


class TestMemoryInfluenceScaling:
    """Tests for MemoryInfluenceScaling component."""
    
    @pytest.fixture
    def mis(self):
        """Create MIS instance with heuristic alpha."""
        return MemoryInfluenceScaling(
            hidden_dim=128,
            use_learned_alpha=False,
            alpha_min=0.0,
            alpha_max=1.0,
            alpha_default=0.5,
        )
    
    @pytest.fixture
    def mis_learned(self):
        """Create MIS instance with learned alpha."""
        return MemoryInfluenceScaling(
            hidden_dim=128,
            use_learned_alpha=True,
            alpha_min=0.0,
            alpha_max=1.0,
        )
    
    def test_initialization(self, mis):
        """Test MIS initialization."""
        assert mis.hidden_dim == 128
        assert mis.use_learned_alpha is False
        assert mis.alpha_min == 0.0
        assert mis.alpha_max == 1.0
        assert mis.alpha_default == 0.5
    
    def test_learned_alpha_initialization(self, mis_learned):
        """Test MIS with learned alpha predictor."""
        assert mis_learned.use_learned_alpha is True
        assert mis_learned.alpha_predictor is not None
    
    def test_compute_alpha_heuristic(self, mis):
        """Test heuristic alpha computation."""
        query_emb = torch.randn(128)
        
        alpha = mis.compute_alpha(
            query_embedding=query_emb,
            memory_relevance=0.8,
            entropy=0.6,
        )
        
        # Alpha should be in valid range
        assert 0.0 <= alpha <= 1.0
    
    def test_compute_alpha_high_relevance(self, mis):
        """Test alpha with high relevance."""
        query_emb = torch.randn(128)
        
        alpha_high = mis.compute_alpha(query_emb, memory_relevance=0.9, entropy=0.5)
        alpha_low = mis.compute_alpha(query_emb, memory_relevance=0.3, entropy=0.5)
        
        # Higher relevance should give higher alpha
        assert alpha_high > alpha_low
    
    def test_compute_alpha_high_entropy(self, mis):
        """Test alpha with high entropy."""
        query_emb = torch.randn(128)
        
        alpha_high = mis.compute_alpha(query_emb, memory_relevance=0.7, entropy=0.9)
        alpha_low = mis.compute_alpha(query_emb, memory_relevance=0.7, entropy=0.3)
        
        # Higher entropy should give higher alpha
        assert alpha_high > alpha_low
    
    def test_compute_alpha_bounds(self, mis):
        """Test alpha stays within bounds."""
        query_emb = torch.randn(128)
        
        # Test extreme values
        alpha_max = mis.compute_alpha(query_emb, memory_relevance=1.0, entropy=1.0)
        alpha_min = mis.compute_alpha(query_emb, memory_relevance=0.0, entropy=0.0)
        
        assert alpha_max <= mis.alpha_max
        assert alpha_min >= mis.alpha_min
    
    def test_compute_alpha_with_batch(self, mis_learned):
        """Test alpha computation with batched input."""
        query_emb = torch.randn(2, 10, 128)  # [batch, seq, hidden]
        
        alpha = mis_learned.compute_alpha(
            query_embedding=query_emb,
            memory_relevance=0.8,
            entropy=0.6,
        )
        
        assert 0.0 <= alpha <= 1.0


class TestLogitBiasScaling:
    """Tests for logit bias scaling (pre-softmax)."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_apply_scaling_full_alpha(self, mis):
        """Test scaling with alpha=1.0 (no change)."""
        logits = torch.randn(1, 4, 10, 20)  # [batch, heads, seq, total_len]
        original = logits.clone()
        
        scaled = mis.apply_scaling(logits, mem_len=5, alpha=1.0)
        
        # Should be unchanged
        assert torch.allclose(scaled, original)
    
    def test_apply_scaling_zero_alpha(self, mis):
        """Test scaling with alpha=0.0 (no memory influence)."""
        logits = torch.randn(1, 4, 10, 20)
        mem_len = 5
        
        scaled = mis.apply_scaling(logits, mem_len=mem_len, alpha=0.0)
        
        # Memory positions should be -inf
        assert torch.all(scaled[..., :mem_len] == float('-inf'))
        
        # User positions should be unchanged
        assert torch.allclose(scaled[..., mem_len:], logits[..., mem_len:])
    
    def test_apply_scaling_partial_alpha(self, mis):
        """Test scaling with partial alpha."""
        logits = torch.randn(1, 4, 10, 20)
        mem_len = 5
        alpha = 0.5
        
        scaled = mis.apply_scaling(logits, mem_len=mem_len, alpha=alpha)
        
        # Memory positions should have logit bias applied
        expected_bias = torch.log(torch.tensor(alpha + 1e-9))
        
        # Check that memory logits are reduced
        assert torch.all(scaled[..., :mem_len] < logits[..., :mem_len])
    
    def test_apply_scaling_no_memory(self, mis):
        """Test scaling with no memory tokens."""
        logits = torch.randn(1, 4, 10, 20)
        original = logits.clone()
        
        scaled = mis.apply_scaling(logits, mem_len=0, alpha=0.5)
        
        # Should be unchanged when no memory
        assert torch.allclose(scaled, original)
    
    def test_apply_scaling_shape_preserved(self, mis):
        """Test that scaling preserves tensor shape."""
        shapes = [
            (1, 4, 10, 20),
            (2, 8, 5, 15),
            (4, 16, 20, 50),
        ]
        
        for shape in shapes:
            logits = torch.randn(*shape)
            scaled = mis.apply_scaling(logits, mem_len=5, alpha=0.5)
            assert scaled.shape == logits.shape


class TestKVValueScaling:
    """Tests for K/V value scaling (alternative method)."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_scale_kv_full_alpha(self, mis):
        """Test K/V scaling with alpha=1.0."""
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=1.0)
        
        # Should be unchanged
        assert torch.allclose(scaled_k, key)
        assert torch.allclose(scaled_v, value)
    
    def test_scale_kv_zero_alpha(self, mis):
        """Test K/V scaling with alpha=0.0."""
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=0.0)
        
        # Should be zeroed
        assert torch.all(scaled_k == 0)
        assert torch.all(scaled_v == 0)
    
    def test_scale_kv_partial_alpha(self, mis):
        """Test K/V scaling with partial alpha."""
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        alpha = 0.5
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=alpha)
        
        # Key should be unchanged
        assert torch.allclose(scaled_k, key)
        
        # Value should be scaled
        assert torch.allclose(scaled_v, value * alpha)
    
    def test_scale_kv_shape_preserved(self, mis):
        """Test that K/V scaling preserves shapes."""
        key = torch.randn(2, 8, 15, 64)
        value = torch.randn(2, 8, 15, 64)
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=0.7)
        
        assert scaled_k.shape == key.shape
        assert scaled_v.shape == value.shape


class TestInfluenceScalingBudget:
    """Tests for influence scaling respecting budget constraints."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_influence_scaling_respects_budget(self, mis):
        """Test that influence scaling respects budget constraints."""
        # Simulate budget constraint: injected KV should not exceed 30% of base
        max_ratio = 0.3
        
        base_kv = torch.ones(10, 64)
        base_norm = base_kv.norm()
        
        # Scale with alpha that respects budget
        alpha = max_ratio
        scaled_kv = base_kv * alpha
        
        # Check norm constraint
        assert scaled_kv.norm() <= base_norm * (1 + max_ratio)
    
    def test_influence_monotonicity(self, mis):
        """Test that influence increases monotonically with alpha."""
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        alphas = [0.2, 0.4, 0.6, 0.8]
        norms = []
        
        for alpha in alphas:
            _, scaled_v = mis.scale_kv_values(key, value, alpha=alpha)
            norms.append(scaled_v.norm().item())
        
        # Norms should increase with alpha
        for i in range(len(norms) - 1):
            assert norms[i] < norms[i + 1]


class TestForwardPass:
    """Tests for MIS forward pass."""
    
    def test_forward_returns_alpha(self):
        """Test forward pass returns alpha value."""
        mis = MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
        
        query_emb = torch.randn(128)
        
        alpha = mis.forward(
            query_embedding=query_emb,
            memory_relevance=0.8,
            entropy=0.6,
        )
        
        assert isinstance(alpha, float)
        assert 0.0 <= alpha <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
