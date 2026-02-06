"""
Behavior tests for influence monotonicity.

Tests theoretical properties:
- Influence increases monotonically with similarity
- Influence increases monotonically with alpha
- Influence is bounded
"""

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixtures.fake_model import FakeModelAdapter
from fixtures.sample_memories import create_sample_kv_entries

from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling


class TestInfluenceMonotonicity:
    """Tests for influence monotonicity properties."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_influence_monotonic_with_similarity(self, mis):
        """
        Test that influence increases monotonically with similarity.
        
        Property: Higher similarity → Higher alpha → More influence
        """
        query_emb = torch.randn(128)
        entropy = 0.6  # Fixed
        
        similarities = torch.linspace(0.1, 0.9, 10).tolist()
        alphas = []
        
        for sim in similarities:
            alpha = mis.compute_alpha(query_emb, memory_relevance=sim, entropy=entropy)
            alphas.append(alpha)
        
        # Alpha should increase with similarity
        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1], \
                f"Monotonicity violated: alpha[{i}]={alphas[i]} > alpha[{i+1}]={alphas[i+1]}"
    
    def test_influence_monotonic_with_entropy(self, mis):
        """
        Test that influence increases monotonically with entropy.
        
        Property: Higher entropy (uncertainty) → Higher alpha
        """
        query_emb = torch.randn(128)
        similarity = 0.8  # Fixed
        
        entropies = torch.linspace(0.1, 0.9, 10).tolist()
        alphas = []
        
        for entropy in entropies:
            alpha = mis.compute_alpha(query_emb, memory_relevance=similarity, entropy=entropy)
            alphas.append(alpha)
        
        # Alpha should increase with entropy
        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1], \
                f"Monotonicity violated: alpha[{i}]={alphas[i]} > alpha[{i+1}]={alphas[i+1]}"
    
    def test_influence_bounded(self, mis):
        """
        Test that influence is bounded within [0, 1].
        
        Property: 0 ≤ α ≤ 1 for all inputs
        """
        query_emb = torch.randn(128)
        
        # Test extreme values
        test_cases = [
            (0.0, 0.0),  # Min similarity, min entropy
            (1.0, 1.0),  # Max similarity, max entropy
            (0.0, 1.0),  # Min similarity, max entropy
            (1.0, 0.0),  # Max similarity, min entropy
        ]
        
        for sim, entropy in test_cases:
            alpha = mis.compute_alpha(query_emb, memory_relevance=sim, entropy=entropy)
            assert 0.0 <= alpha <= 1.0, \
                f"Alpha out of bounds: {alpha} for sim={sim}, entropy={entropy}"
    
    def test_influence_scaling_monotonic(self, mis):
        """
        Test that K/V scaling is monotonic with alpha.
        
        Property: Higher alpha → Larger scaled values
        """
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
        norms = []
        
        for alpha in alphas:
            _, scaled_v = mis.scale_kv_values(key, value, alpha)
            norms.append(scaled_v.norm().item())
        
        # Norms should increase with alpha
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i + 1], \
                f"Scaling monotonicity violated: norm[{i}]={norms[i]} > norm[{i+1}]={norms[i+1]}"


class TestInfluenceGracefulDegradation:
    """Tests for graceful degradation property."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_alpha_zero_recovers_vanilla(self, mis):
        """
        Test that α → 0 recovers vanilla LLM behavior.
        
        Property: With α = 0, memory has no influence
        """
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=0.0)
        
        # With alpha=0, values should be zero
        assert torch.all(scaled_v == 0), "Alpha=0 should zero out values"
    
    def test_alpha_one_full_influence(self, mis):
        """
        Test that α = 1 gives full memory influence.
        
        Property: With α = 1, memory is fully used
        """
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        scaled_k, scaled_v = mis.scale_kv_values(key, value, alpha=1.0)
        
        # With alpha=1, values should be unchanged
        assert torch.allclose(scaled_v, value), "Alpha=1 should preserve values"
    
    def test_continuous_degradation(self, mis):
        """
        Test continuous degradation from α=1 to α=0.
        
        Property: Influence decreases smoothly
        """
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        alphas = torch.linspace(1.0, 0.0, 11).tolist()
        norms = []
        
        for alpha in alphas:
            _, scaled_v = mis.scale_kv_values(key, value, alpha)
            norms.append(scaled_v.norm().item())
        
        # Norms should decrease continuously
        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1], \
                f"Degradation not continuous: norm[{i}]={norms[i]} < norm[{i+1}]={norms[i+1]}"


class TestInfluenceConsistency:
    """Tests for influence consistency properties."""
    
    @pytest.fixture
    def mis(self):
        return MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    def test_deterministic_alpha(self, mis):
        """
        Test that alpha computation is deterministic.
        
        Property: Same inputs → Same alpha
        """
        query_emb = torch.randn(128)
        similarity = 0.8
        entropy = 0.6
        
        alpha_1 = mis.compute_alpha(query_emb, similarity, entropy)
        alpha_2 = mis.compute_alpha(query_emb, similarity, entropy)
        
        assert alpha_1 == alpha_2, "Alpha should be deterministic"
    
    def test_scaling_deterministic(self, mis):
        """
        Test that K/V scaling is deterministic.
        
        Property: Same inputs → Same scaled outputs
        """
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        alpha = 0.7
        
        k1, v1 = mis.scale_kv_values(key, value, alpha)
        k2, v2 = mis.scale_kv_values(key, value, alpha)
        
        assert torch.allclose(k1, k2), "Key scaling should be deterministic"
        assert torch.allclose(v1, v2), "Value scaling should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
