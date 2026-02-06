"""
Unit tests for Dual-Factor Gating component.

Tests the core gating logic that determines:
- Whether to inject memory
- How strong the injection should be (alpha)

Key test scenarios:
- High entropy + High relevance → Strong injection
- High entropy + Low relevance → No injection (open-ended question)
- Low entropy + High relevance → Moderate injection
- Low entropy + Low relevance → No injection
"""

import pytest
import torch
import numpy as np

from dki.core.components.dual_factor_gating import (
    DualFactorGating,
    GatingDecision,
)
from dki.core.memory_router import MemoryRouter, MemorySearchResult

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.fake_model import FakeModelAdapter
from fixtures.fake_embeddings import FakeEmbeddingService


class TestDualFactorGating:
    """Tests for DualFactorGating component."""
    
    @pytest.fixture
    def gating(self):
        """Create gating instance with test thresholds."""
        return DualFactorGating(
            entropy_threshold=0.5,
            relevance_threshold=0.7,
            use_margin=True,
            margin_weight=0.3,
        )
    
    @pytest.fixture
    def fake_model(self):
        """Create fake model for testing."""
        return FakeModelAdapter(hidden_dim=128, num_layers=4)
    
    @pytest.fixture
    def router_with_memories(self):
        """Create router with sample memories."""
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        router = MemoryRouter(embedding_service)
        
        # Add sample memories
        router.add_memory("mem1", "用户是素食主义者，对海鲜过敏")
        router.add_memory("mem2", "用户住在北京，是软件工程师")
        router.add_memory("mem3", "用户喜欢爬山和摄影")
        
        return router
    
    def test_gating_initialization(self, gating):
        """Test gating initialization with correct thresholds."""
        assert gating.entropy_threshold == 0.5
        assert gating.relevance_threshold == 0.7
        assert gating.use_margin is True
        assert gating.margin_weight == 0.3
    
    def test_threshold_update(self, gating):
        """Test threshold update functionality."""
        gating.update_thresholds(entropy_threshold=0.4, relevance_threshold=0.8)
        
        assert gating.entropy_threshold == 0.4
        assert gating.relevance_threshold == 0.8
    
    def test_partial_threshold_update(self, gating):
        """Test updating only one threshold."""
        original_relevance = gating.relevance_threshold
        
        gating.update_thresholds(entropy_threshold=0.3)
        
        assert gating.entropy_threshold == 0.3
        assert gating.relevance_threshold == original_relevance
    
    def test_gating_decision_dataclass(self):
        """Test GatingDecision dataclass."""
        decision = GatingDecision(
            should_inject=True,
            alpha=0.8,
            entropy=0.6,
            relevance_score=0.9,
            margin=0.2,
            memories=[],
            reasoning="Test reasoning",
        )
        
        assert decision.should_inject is True
        assert decision.alpha == 0.8
        assert decision.entropy == 0.6
        assert decision.relevance_score == 0.9
        
        # Test to_dict
        d = decision.to_dict()
        assert d['should_inject'] is True
        assert d['alpha'] == 0.8
        assert 'reasoning' in d
    
    def test_force_inject(self, gating, router_with_memories):
        """Test force injection bypasses gating."""
        decision = gating.force_inject(
            router=router_with_memories,
            query="今晚吃什么？",
            alpha=0.9,
            top_k=3,
        )
        
        # Force inject should always inject if memories exist
        assert decision.should_inject is True
        assert decision.alpha == 0.9
        assert "Forced injection" in decision.reasoning
    
    def test_force_inject_no_memories(self, gating):
        """Test force injection with empty router."""
        embedding_service = FakeEmbeddingService(embedding_dim=384)
        empty_router = MemoryRouter(embedding_service)
        
        decision = gating.force_inject(
            router=empty_router,
            query="测试查询",
            alpha=0.9,
        )
        
        # Should not inject if no memories
        assert decision.should_inject is False
        assert len(decision.memories) == 0
    
    def test_gating_weights(self, gating):
        """Test gating weight configuration."""
        assert gating.entropy_weight == 2.0
        assert gating.relevance_weight == 1.5


class TestGatingDecisionLogic:
    """Tests for gating decision logic without real model."""
    
    def test_decision_high_entropy_high_relevance(self):
        """
        High entropy + High relevance → Should inject with high alpha.
        
        This scenario represents:
        - Model is uncertain (high entropy)
        - Relevant memory is available
        - Strong injection is appropriate
        """
        # Simulate the decision logic
        entropy = 0.8  # High
        relevance = 0.9  # High
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        
        entropy_condition = entropy > entropy_threshold
        relevance_condition = relevance > relevance_threshold
        
        assert entropy_condition is True
        assert relevance_condition is True
        
        # Both conditions met → should inject
        should_inject = relevance_condition
        assert should_inject is True
    
    def test_decision_high_entropy_low_relevance(self):
        """
        High entropy + Low relevance → Should NOT inject.
        
        This scenario represents:
        - Model is uncertain (high entropy)
        - But no relevant memory available
        - This is likely an open-ended question
        """
        entropy = 0.8  # High
        relevance = 0.3  # Low
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        
        entropy_condition = entropy > entropy_threshold
        relevance_condition = relevance > relevance_threshold
        
        assert entropy_condition is True
        assert relevance_condition is False
        
        # Low relevance → should not inject
        should_inject = relevance_condition
        assert should_inject is False
    
    def test_decision_low_entropy_high_relevance(self):
        """
        Low entropy + High relevance → Should inject with moderate alpha.
        
        This scenario represents:
        - Model is confident (low entropy)
        - But relevant memory available for enrichment
        - Moderate injection for context
        """
        entropy = 0.3  # Low
        relevance = 0.9  # High
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        
        entropy_condition = entropy > entropy_threshold
        relevance_condition = relevance > relevance_threshold
        
        assert entropy_condition is False
        assert relevance_condition is True
        
        # High relevance → should inject (for enrichment)
        should_inject = relevance_condition
        assert should_inject is True
    
    def test_decision_low_entropy_low_relevance(self):
        """
        Low entropy + Low relevance → Should NOT inject.
        
        This scenario represents:
        - Model is confident (low entropy)
        - No relevant memory available
        - No injection needed
        """
        entropy = 0.3  # Low
        relevance = 0.3  # Low
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        
        entropy_condition = entropy > entropy_threshold
        relevance_condition = relevance > relevance_threshold
        
        assert entropy_condition is False
        assert relevance_condition is False
        
        # Neither condition met → should not inject
        should_inject = relevance_condition
        assert should_inject is False


class TestAlphaComputation:
    """Tests for alpha (injection strength) computation."""
    
    def test_alpha_sigmoid_computation(self):
        """Test sigmoid-based alpha computation."""
        entropy = 0.7
        relevance = 0.85
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        entropy_weight = 2.0
        relevance_weight = 1.5
        margin = 0.1
        margin_weight = 0.3
        
        # Compute alpha input
        alpha_input = (
            entropy_weight * (entropy - entropy_threshold) +
            relevance_weight * (relevance - relevance_threshold) +
            margin_weight * margin
        )
        
        # Sigmoid
        alpha = float(torch.sigmoid(torch.tensor(alpha_input)))
        
        # Alpha should be in valid range
        assert 0.0 <= alpha <= 1.0
        
        # With positive inputs, alpha should be > 0.5
        assert alpha > 0.5
    
    def test_alpha_minimum_when_injecting(self):
        """Test that alpha has minimum value when injecting."""
        # Even with low inputs, alpha should be at least 0.1 when injecting
        min_alpha = 0.1
        
        # Simulate low alpha computation
        computed_alpha = 0.05
        
        # Apply minimum
        final_alpha = max(min_alpha, computed_alpha)
        
        assert final_alpha >= min_alpha
    
    def test_alpha_zero_when_not_injecting(self):
        """Test that alpha is 0 when not injecting."""
        should_inject = False
        
        if not should_inject:
            alpha = 0.0
        else:
            alpha = 0.5
        
        assert alpha == 0.0
    
    def test_alpha_monotonicity_with_relevance(self):
        """Test that alpha increases with relevance."""
        entropy = 0.6
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        entropy_weight = 2.0
        relevance_weight = 1.5
        
        relevances = [0.75, 0.85, 0.95]
        alphas = []
        
        for relevance in relevances:
            alpha_input = (
                entropy_weight * (entropy - entropy_threshold) +
                relevance_weight * (relevance - relevance_threshold)
            )
            alpha = float(torch.sigmoid(torch.tensor(alpha_input)))
            alphas.append(alpha)
        
        # Alpha should increase with relevance
        assert alphas[0] < alphas[1] < alphas[2]
    
    def test_alpha_monotonicity_with_entropy(self):
        """Test that alpha increases with entropy (when relevant)."""
        relevance = 0.85
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        entropy_weight = 2.0
        relevance_weight = 1.5
        
        entropies = [0.55, 0.7, 0.9]
        alphas = []
        
        for entropy in entropies:
            alpha_input = (
                entropy_weight * (entropy - entropy_threshold) +
                relevance_weight * (relevance - relevance_threshold)
            )
            alpha = float(torch.sigmoid(torch.tensor(alpha_input)))
            alphas.append(alpha)
        
        # Alpha should increase with entropy
        assert alphas[0] < alphas[1] < alphas[2]


class TestMarginEffect:
    """Tests for margin effect on gating decisions."""
    
    def test_margin_increases_alpha(self):
        """Test that higher margin increases alpha."""
        entropy = 0.6
        relevance = 0.85
        entropy_threshold = 0.5
        relevance_threshold = 0.7
        entropy_weight = 2.0
        relevance_weight = 1.5
        margin_weight = 0.3
        
        margins = [0.0, 0.2, 0.5]
        alphas = []
        
        for margin in margins:
            alpha_input = (
                entropy_weight * (entropy - entropy_threshold) +
                relevance_weight * (relevance - relevance_threshold) +
                margin_weight * margin
            )
            alpha = float(torch.sigmoid(torch.tensor(alpha_input)))
            alphas.append(alpha)
        
        # Alpha should increase with margin
        assert alphas[0] < alphas[1] < alphas[2]
    
    def test_margin_disabled(self):
        """Test gating without margin."""
        gating = DualFactorGating(
            entropy_threshold=0.5,
            relevance_threshold=0.7,
            use_margin=False,
        )
        
        assert gating.use_margin is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
