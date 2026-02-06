"""
Behavior tests for budget enforcement.

Tests theoretical properties:
- Token budget constraints
- Attention budget constraints
- Budget reallocation hypothesis
"""

import pytest

from dki.core.components.attention_budget import AttentionBudgetAnalyzer


class TestTokenBudgetEnforcement:
    """Tests for token budget enforcement."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_rag_token_budget_constraint(self, analyzer):
        """
        Test RAG token budget constraint.
        
        Property: RAG token budget = context_window - (memory + user)
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        expected_free = 4096 - (user_tokens + memory_tokens)
        assert analysis.rag_token_budget_free == expected_free
    
    def test_dki_token_budget_constraint(self, analyzer):
        """
        Test DKI token budget constraint.
        
        Property: DKI token budget = context_window - user (memory not in budget)
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        expected_free = 4096 - user_tokens
        assert analysis.dki_token_budget_free == expected_free
    
    def test_token_budget_non_negative(self, analyzer):
        """
        Test that token budget is non-negative.
        
        Property: Free tokens ≥ 0
        """
        # Test overflow scenario
        user_tokens = 3000
        memory_tokens = 2000
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        assert analysis.rag_token_budget_free >= 0
        assert analysis.dki_token_budget_free >= 0
    
    def test_token_savings_equals_memory(self, analyzer):
        """
        Test that DKI token savings equals memory tokens.
        
        Property: token_saved = memory_tokens
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        assert analysis.token_budget_saved == memory_tokens


class TestAttentionBudgetEnforcement:
    """Tests for attention budget enforcement."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_rag_attention_budget_formula(self, analyzer):
        """
        Test RAG attention budget formula.
        
        Property: B_a^RAG = (n_m + n_u)^2
        """
        user_tokens = 100
        memory_tokens = 200
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        expected = (user_tokens + memory_tokens) ** 2
        assert analysis.rag_attention_budget == expected
    
    def test_dki_attention_budget_formula(self, analyzer):
        """
        Test DKI attention budget formula.
        
        Property: B_a^DKI = n_u × (n_m + n_u)
        """
        user_tokens = 100
        memory_tokens = 200
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        expected = user_tokens * (user_tokens + memory_tokens)
        assert analysis.dki_attention_budget == expected
    
    def test_dki_attention_less_than_rag(self, analyzer):
        """
        Test that DKI attention budget is less than RAG.
        
        Property: B_a^DKI < B_a^RAG (when memory > 0)
        """
        user_tokens = 100
        memory_tokens = 200
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        assert analysis.dki_attention_budget < analysis.rag_attention_budget
    
    def test_attention_overhead_bounded(self, analyzer):
        """
        Test that attention overhead is bounded.
        
        Property: Overhead ratio should be reasonable
        """
        test_cases = [
            (100, 100),
            (200, 300),
            (500, 1000),
        ]
        
        for user_tokens, memory_tokens in test_cases:
            analysis = analyzer.analyze(user_tokens, memory_tokens)
            
            # Overhead should be less than memory_tokens/user_tokens + 1
            max_overhead = memory_tokens / user_tokens + 1
            assert analysis.attention_overhead_ratio <= max_overhead


class TestBudgetReallocationHypothesis:
    """Tests for Budget Reallocation Hypothesis."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_token_efficiency_increases_with_memory(self, analyzer):
        """
        Test that token efficiency increases with memory size.
        
        Property: More memory → Higher token efficiency gain
        """
        user_tokens = 200
        
        efficiencies = []
        for memory_tokens in [100, 300, 500, 800]:
            analysis = analyzer.analyze(user_tokens, memory_tokens)
            efficiencies.append(analysis.token_efficiency_gain)
        
        # Efficiency should increase
        for i in range(len(efficiencies) - 1):
            assert efficiencies[i] < efficiencies[i + 1]
    
    def test_dki_recommended_for_reasoning(self, analyzer):
        """
        Test DKI recommendation for reasoning tasks.
        
        Property: Reasoning tasks should favor DKI
        """
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=400,
            task_type="reasoning",
        )
        
        assert result['factors']['reasoning_task'] is True
    
    def test_dki_recommended_for_constrained_context(self, analyzer):
        """
        Test DKI recommendation for constrained context.
        
        Property: Constrained context should favor DKI
        """
        analyzer.context_window = 2048
        
        result = analyzer.should_prefer_dki(
            user_tokens=500,
            memory_tokens=800,
            task_type="reasoning",
        )
        
        # RAG free = 2048 - 1300 = 748 < 1000
        assert result['factors']['context_constrained'] is True


class TestBudgetConsistency:
    """Tests for budget calculation consistency."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_budget_sum_consistency(self, analyzer):
        """
        Test budget sum consistency.
        
        Property: used + free = context_window
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis = analyzer.analyze(user_tokens, memory_tokens)
        
        # RAG
        rag_total = analysis.rag_token_budget_used + analysis.rag_token_budget_free
        assert rag_total == 4096 or analysis.rag_token_budget_free == 0
        
        # DKI
        dki_total = analysis.dki_token_budget_used + analysis.dki_token_budget_free
        assert dki_total == 4096
    
    def test_deterministic_analysis(self, analyzer):
        """
        Test that analysis is deterministic.
        
        Property: Same inputs → Same analysis
        """
        user_tokens = 200
        memory_tokens = 500
        
        analysis_1 = analyzer.analyze(user_tokens, memory_tokens)
        analysis_2 = analyzer.analyze(user_tokens, memory_tokens)
        
        assert analysis_1.rag_attention_budget == analysis_2.rag_attention_budget
        assert analysis_1.dki_attention_budget == analysis_2.dki_attention_budget


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
