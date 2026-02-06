"""
Unit tests for Attention Budget Analyzer.

Tests the core DKI vs RAG budget analysis:
- Token Budget: Hard constraint (truncation causes info loss)
- Attention Budget: Soft constraint (increased computation, no info loss)

Key insight from paper:
∂TaskSuccess/∂B_t^free > ∂Latency/∂B_a
"""

import pytest
import time

from dki.core.components.attention_budget import (
    AttentionBudgetAnalyzer,
    BudgetAnalysis,
    LatencyBreakdown,
    LatencyTimer,
)


class TestBudgetAnalysis:
    """Tests for BudgetAnalysis dataclass."""
    
    def test_budget_analysis_creation(self):
        """Test BudgetAnalysis dataclass creation."""
        analysis = BudgetAnalysis(
            context_window=4096,
            user_tokens=100,
            memory_tokens=200,
            rag_token_budget_used=300,
            rag_token_budget_free=3796,
            rag_attention_budget=90000,
            dki_token_budget_used=100,
            dki_token_budget_free=3996,
            dki_attention_budget=30000,
            token_budget_saved=200,
            attention_budget_increase=20000,
            token_efficiency_gain=0.05,
            attention_overhead_ratio=2.0,
        )
        
        assert analysis.context_window == 4096
        assert analysis.user_tokens == 100
        assert analysis.memory_tokens == 200
    
    def test_budget_analysis_to_dict(self):
        """Test BudgetAnalysis serialization."""
        analysis = BudgetAnalysis(
            context_window=4096,
            user_tokens=100,
            memory_tokens=200,
            rag_token_budget_used=300,
            rag_token_budget_free=3796,
            rag_attention_budget=90000,
            dki_token_budget_used=100,
            dki_token_budget_free=3996,
            dki_attention_budget=30000,
            token_budget_saved=200,
            attention_budget_increase=20000,
            token_efficiency_gain=0.05,
            attention_overhead_ratio=2.0,
        )
        
        d = analysis.to_dict()
        
        assert 'context_window' in d
        assert 'rag' in d
        assert 'dki' in d
        assert 'comparison' in d


class TestAttentionBudgetAnalyzer:
    """Tests for AttentionBudgetAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default context window."""
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.context_window == 4096
        assert len(analyzer._history) == 0
    
    def test_analyze_basic(self, analyzer):
        """Test basic budget analysis."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
        )
        
        # RAG uses both user and memory tokens
        assert analysis.rag_token_budget_used == 300
        
        # DKI only uses user tokens in token budget
        assert analysis.dki_token_budget_used == 100
        
        # Token savings
        assert analysis.token_budget_saved == 200
    
    def test_analyze_rag_attention_budget(self, analyzer):
        """Test RAG attention budget calculation."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
        )
        
        # RAG attention: (n_m + n_u)^2 = 300^2 = 90000
        expected_rag_attention = (100 + 200) ** 2
        assert analysis.rag_attention_budget == expected_rag_attention
    
    def test_analyze_dki_attention_budget(self, analyzer):
        """Test DKI attention budget calculation."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
        )
        
        # DKI attention: n_u × (n_m + n_u) = 100 × 300 = 30000
        expected_dki_attention = 100 * (100 + 200)
        assert analysis.dki_attention_budget == expected_dki_attention
    
    def test_analyze_token_efficiency(self, analyzer):
        """Test token efficiency gain calculation."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=400,
        )
        
        # Token efficiency = memory_tokens / context_window
        expected_efficiency = 400 / 4096
        assert abs(analysis.token_efficiency_gain - expected_efficiency) < 0.001
    
    def test_analyze_attention_overhead(self, analyzer):
        """Test attention overhead ratio calculation."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
        )
        
        # Attention overhead = (dki_attention - vanilla_attention) / vanilla_attention
        # vanilla_attention = user_tokens^2 = 10000
        # dki_attention = 30000
        # overhead = (30000 - 10000) / 10000 = 2.0
        expected_overhead = (100 * 300 - 100 * 100) / (100 * 100)
        assert abs(analysis.attention_overhead_ratio - expected_overhead) < 0.001
    
    def test_analyze_free_token_budget(self, analyzer):
        """Test free token budget calculation."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
        )
        
        # RAG free: 4096 - 300 = 3796
        assert analysis.rag_token_budget_free == 3796
        
        # DKI free: 4096 - 100 = 3996
        assert analysis.dki_token_budget_free == 3996
    
    def test_analyze_with_custom_context(self, analyzer):
        """Test analysis with custom context window."""
        analysis = analyzer.analyze(
            user_tokens=100,
            memory_tokens=200,
            context_window=2048,
        )
        
        assert analysis.context_window == 2048
        assert analysis.rag_token_budget_free == 2048 - 300
    
    def test_analyze_history_tracking(self, analyzer):
        """Test that analysis is tracked in history."""
        analyzer.analyze(user_tokens=100, memory_tokens=200)
        analyzer.analyze(user_tokens=150, memory_tokens=300)
        
        assert len(analyzer._history) == 2
    
    def test_clear_history(self, analyzer):
        """Test clearing history."""
        analyzer.analyze(user_tokens=100, memory_tokens=200)
        analyzer.clear()
        
        assert len(analyzer._history) == 0


class TestShouldPreferDKI:
    """Tests for DKI recommendation logic."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_recommend_dki_large_memory(self, analyzer):
        """Test DKI recommendation with large memory."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=500,  # > 200 threshold
            task_type="reasoning",
        )
        
        assert result['factors']['memory_tokens_significant'] is True
    
    def test_recommend_dki_constrained_context(self, analyzer):
        """Test DKI recommendation with constrained context."""
        analyzer.context_window = 2048
        
        result = analyzer.should_prefer_dki(
            user_tokens=500,
            memory_tokens=800,
            task_type="reasoning",
        )
        
        # RAG free = 2048 - 1300 = 748 < 1000
        assert result['factors']['context_constrained'] is True
    
    def test_recommend_dki_reasoning_task(self, analyzer):
        """Test DKI recommendation for reasoning task."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=300,
            task_type="reasoning",
        )
        
        assert result['factors']['reasoning_task'] is True
    
    def test_recommend_rag_small_memory(self, analyzer):
        """Test RAG recommendation with small memory."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=100,  # < 200 threshold
            task_type="qa",
        )
        
        assert result['factors']['memory_tokens_significant'] is False
    
    def test_recommend_rag_creative_task(self, analyzer):
        """Test RAG recommendation for creative task."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=300,
            task_type="creative",
        )
        
        assert result['factors']['reasoning_task'] is False
    
    def test_recommendation_score(self, analyzer):
        """Test recommendation score calculation."""
        result = analyzer.should_prefer_dki(
            user_tokens=200,
            memory_tokens=500,
            task_type="reasoning",
        )
        
        # Score is sum of True factors
        score = sum(result['factors'].values())
        assert result['score'] == score
    
    def test_recommendation_threshold(self, analyzer):
        """Test recommendation threshold (score >= 3)."""
        # High score scenario
        result_high = analyzer.should_prefer_dki(
            user_tokens=500,
            memory_tokens=1000,
            task_type="reasoning",
        )
        
        # Low score scenario
        result_low = analyzer.should_prefer_dki(
            user_tokens=100,
            memory_tokens=50,
            task_type="creative",
        )
        
        # High score should recommend DKI
        if result_high['score'] >= 3:
            assert result_high['recommend_dki'] is True
        
        # Low score should not recommend DKI
        if result_low['score'] < 3:
            assert result_low['recommend_dki'] is False


class TestLatencyTracking:
    """Tests for latency tracking."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_latency_breakdown_creation(self):
        """Test LatencyBreakdown creation."""
        breakdown = LatencyBreakdown(
            router_ms=5.0,
            gating_ms=3.0,
            kv_compute_ms=10.0,
            prefill_ms=50.0,
            total_ms=70.0,
        )
        
        assert breakdown.router_ms == 5.0
        assert breakdown.total_ms == 70.0
    
    def test_latency_breakdown_to_dict(self):
        """Test LatencyBreakdown serialization."""
        breakdown = LatencyBreakdown(
            router_ms=5.0,
            total_ms=70.0,
            cache_hit=True,
            cache_tier="l1",
        )
        
        d = breakdown.to_dict()
        
        assert d['router_ms'] == 5.0
        assert d['cache_hit'] is True
        assert d['cache_tier'] == "l1"
    
    def test_record_latency(self, analyzer):
        """Test recording latency."""
        breakdown = LatencyBreakdown(total_ms=100.0)
        analyzer.record_latency(breakdown)
        
        assert len(analyzer._latency_history) == 1
    
    def test_average_latency(self, analyzer):
        """Test average latency calculation."""
        analyzer.record_latency(LatencyBreakdown(total_ms=100.0, router_ms=10.0))
        analyzer.record_latency(LatencyBreakdown(total_ms=200.0, router_ms=20.0))
        
        avg = analyzer.get_average_latency()
        
        assert avg['total_ms'] == 150.0
        assert avg['router_ms'] == 15.0
    
    def test_cache_hit_rate(self, analyzer):
        """Test cache hit rate calculation."""
        analyzer.record_latency(LatencyBreakdown(cache_hit=True))
        analyzer.record_latency(LatencyBreakdown(cache_hit=True))
        analyzer.record_latency(LatencyBreakdown(cache_hit=False))
        
        avg = analyzer.get_average_latency()
        
        assert abs(avg['cache_hit_rate'] - 2/3) < 0.001


class TestLatencyTimer:
    """Tests for LatencyTimer context manager."""
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with LatencyTimer() as timer:
            time.sleep(0.01)  # 10ms
        
        assert timer.breakdown.total_ms >= 10.0
    
    def test_timer_stages(self):
        """Test timing individual stages."""
        with LatencyTimer() as timer:
            timer.start_stage("router")
            time.sleep(0.005)
            timer.end_stage()
            
            timer.start_stage("prefill")
            time.sleep(0.01)
            timer.end_stage()
        
        assert timer.breakdown.router_ms >= 5.0
        assert timer.breakdown.prefill_ms >= 10.0
    
    def test_timer_auto_end_stage(self):
        """Test automatic stage ending."""
        with LatencyTimer() as timer:
            timer.start_stage("router")
            time.sleep(0.005)
            # Start new stage without ending previous
            timer.start_stage("prefill")
            time.sleep(0.005)
            timer.end_stage()
        
        # Both stages should be recorded
        assert timer.breakdown.router_ms >= 5.0
        assert timer.breakdown.prefill_ms >= 5.0
    
    def test_timer_cache_info(self):
        """Test setting cache info."""
        with LatencyTimer() as timer:
            timer.set_cache_info(hit=True, tier="l1")
        
        assert timer.breakdown.cache_hit is True
        assert timer.breakdown.cache_tier == "l1"


class TestAnalyzerStats:
    """Tests for analyzer statistics."""
    
    @pytest.fixture
    def analyzer(self):
        return AttentionBudgetAnalyzer(context_window=4096)
    
    def test_empty_stats(self, analyzer):
        """Test stats with no history."""
        stats = analyzer.get_stats()
        assert stats['count'] == 0
    
    def test_stats_with_history(self, analyzer):
        """Test stats with analysis history."""
        analyzer.analyze(user_tokens=100, memory_tokens=200)
        analyzer.analyze(user_tokens=150, memory_tokens=300)
        
        stats = analyzer.get_stats()
        
        assert stats['count'] == 2
        assert 'avg_token_efficiency_gain' in stats
        assert 'avg_attention_overhead' in stats
        assert 'avg_memory_tokens' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
