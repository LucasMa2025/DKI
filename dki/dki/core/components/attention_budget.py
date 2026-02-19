"""
Attention Budget Analysis for DKI System
Implements the Attention Budget Reallocation Hypothesis from Paper Section 3.2

Key Insight from Paper:
- Token Budget (B_t): Hard constraint - truncation causes information loss
- Attention Budget (B_a): Soft constraint - increased computation, no information loss

Hypothesis:
∂TaskSuccess/∂B_t^free > ∂Latency/∂B_a

In reasoning-intensive tasks, the marginal benefit of releasing token budget
exceeds the marginal cost of increased attention budget.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
from loguru import logger


@dataclass
class BudgetAnalysis:
    """
    Analysis of token and attention budgets.
    
    Compares RAG vs DKI budget utilization.
    """
    # Token budget
    context_window: int
    user_tokens: int
    memory_tokens: int
    
    # RAG budget usage
    rag_token_budget_used: int  # n_m + n_u
    rag_token_budget_free: int  # context_window - (n_m + n_u)
    rag_attention_budget: int   # (n_m + n_u)^2
    
    # DKI budget usage
    dki_token_budget_used: int  # n_u (memory not in token budget)
    dki_token_budget_free: int  # context_window - n_u
    dki_attention_budget: int   # n_u × (n_m + n_u)
    
    # Comparison
    token_budget_saved: int     # DKI saves this many tokens
    attention_budget_increase: int  # DKI adds this much attention
    
    # Ratios
    token_efficiency_gain: float  # % more tokens available
    attention_overhead_ratio: float  # % attention increase
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'context_window': self.context_window,
            'user_tokens': self.user_tokens,
            'memory_tokens': self.memory_tokens,
            'rag': {
                'token_budget_used': self.rag_token_budget_used,
                'token_budget_free': self.rag_token_budget_free,
                'attention_budget': self.rag_attention_budget,
            },
            'dki': {
                'token_budget_used': self.dki_token_budget_used,
                'token_budget_free': self.dki_token_budget_free,
                'attention_budget': self.dki_attention_budget,
            },
            'comparison': {
                'token_budget_saved': self.token_budget_saved,
                'attention_budget_increase': self.attention_budget_increase,
                'token_efficiency_gain': self.token_efficiency_gain,
                'attention_overhead_ratio': self.attention_overhead_ratio,
            },
        }


@dataclass
class LatencyBreakdown:
    """Breakdown of DKI latency components."""
    router_ms: float = 0.0
    gating_ms: float = 0.0
    kv_compute_ms: float = 0.0
    kv_load_ms: float = 0.0
    projection_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    
    # Cache info
    cache_hit: bool = False
    cache_tier: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'router_ms': self.router_ms,
            'gating_ms': self.gating_ms,
            'kv_compute_ms': self.kv_compute_ms,
            'kv_load_ms': self.kv_load_ms,
            'projection_ms': self.projection_ms,
            'prefill_ms': self.prefill_ms,
            'decode_ms': self.decode_ms,
            'total_ms': self.total_ms,
            'cache_hit': self.cache_hit,
            'cache_tier': self.cache_tier,
        }


class AttentionBudgetAnalyzer:
    """
    Analyzer for Attention Budget Reallocation Hypothesis.
    
    From Paper Section 3.2:
    
    Definition 2 (Budget Reallocation):
    - Token Budget B_t: Maximum tokens in context window
    - Attention Budget B_a = n_q × n_k: Computational cost of attention
    
    For RAG: B_t^used = n_m + n_u, B_a = (n_m + n_u)^2
    For DKI: B_t^used = n_u, B_a = n_u × (n_m + n_u)
    
    Hypothesis (Cognitive Bandwidth Release):
    In reasoning-intensive tasks, the marginal benefit of releasing token budget
    exceeds the marginal cost of increased attention budget.
    """
    
    def __init__(self, context_window: int = 4096):
        self.context_window = context_window
        self._history: List[BudgetAnalysis] = []
        self._latency_history: List[LatencyBreakdown] = []
    
    def analyze(
        self,
        user_tokens: int,
        memory_tokens: int,
        context_window: Optional[int] = None,
    ) -> BudgetAnalysis:
        """
        Analyze token and attention budgets for RAG vs DKI.
        
        Args:
            user_tokens: Number of user input tokens (n_u)
            memory_tokens: Number of memory tokens (n_m)
            context_window: Context window size (default: self.context_window)
            
        Returns:
            BudgetAnalysis with comparison metrics
        """
        ctx_window = context_window or self.context_window
        
        # RAG budget
        rag_token_used = memory_tokens + user_tokens
        rag_token_free = max(0, ctx_window - rag_token_used)
        rag_attention = (memory_tokens + user_tokens) ** 2
        
        # DKI budget (memory not in token budget)
        dki_token_used = user_tokens
        dki_token_free = max(0, ctx_window - user_tokens)
        dki_attention = user_tokens * (memory_tokens + user_tokens)
        
        # Comparison
        token_saved = memory_tokens
        attention_increase = dki_attention - (user_tokens ** 2)  # vs vanilla
        
        # Ratios
        token_efficiency = token_saved / ctx_window if ctx_window > 0 else 0.0
        attention_overhead = (
            attention_increase / (user_tokens ** 2) 
            if user_tokens > 0 else 0.0
        )
        
        analysis = BudgetAnalysis(
            context_window=ctx_window,
            user_tokens=user_tokens,
            memory_tokens=memory_tokens,
            rag_token_budget_used=rag_token_used,
            rag_token_budget_free=rag_token_free,
            rag_attention_budget=rag_attention,
            dki_token_budget_used=dki_token_used,
            dki_token_budget_free=dki_token_free,
            dki_attention_budget=dki_attention,
            token_budget_saved=token_saved,
            attention_budget_increase=attention_increase,
            token_efficiency_gain=token_efficiency,
            attention_overhead_ratio=attention_overhead,
        )
        
        self._history.append(analysis)
        return analysis
    
    def record_latency(self, breakdown: LatencyBreakdown) -> None:
        """Record latency breakdown for analysis."""
        self._latency_history.append(breakdown)
    
    def get_average_latency(self) -> Dict[str, float]:
        """Get average latency breakdown."""
        if not self._latency_history:
            return {}
        
        n = len(self._latency_history)
        return {
            'router_ms': sum(l.router_ms for l in self._latency_history) / n,
            'gating_ms': sum(l.gating_ms for l in self._latency_history) / n,
            'kv_compute_ms': sum(l.kv_compute_ms for l in self._latency_history) / n,
            'kv_load_ms': sum(l.kv_load_ms for l in self._latency_history) / n,
            'projection_ms': sum(l.projection_ms for l in self._latency_history) / n,
            'prefill_ms': sum(l.prefill_ms for l in self._latency_history) / n,
            'decode_ms': sum(l.decode_ms for l in self._latency_history) / n,
            'total_ms': sum(l.total_ms for l in self._latency_history) / n,
            'cache_hit_rate': sum(1 for l in self._latency_history if l.cache_hit) / n,
        }
    
    def should_prefer_dki(
        self,
        user_tokens: int,
        memory_tokens: int,
        task_type: str = "reasoning",
    ) -> Dict[str, Any]:
        """
        Recommend whether to use DKI based on budget analysis.
        
        From Paper Section 7.3:
        Consider DKI when:
        - Memory tokens > 200
        - Context window < 8K
        - Multi-turn dialogue > 3 turns
        - Need to inject > 3 memory items
        - Require fine-grained injection control
        
        Args:
            user_tokens: Number of user tokens
            memory_tokens: Number of memory tokens
            task_type: Type of task (reasoning, creative, qa, etc.)
            
        Returns:
            Recommendation with reasoning
        """
        analysis = self.analyze(user_tokens, memory_tokens)
        
        # Decision factors
        factors = {
            'memory_tokens_significant': memory_tokens > 200,
            'context_constrained': analysis.rag_token_budget_free < 1000,
            'token_efficiency_high': analysis.token_efficiency_gain > 0.1,
            'attention_overhead_acceptable': analysis.attention_overhead_ratio < 2.0,
            'reasoning_task': task_type in ['reasoning', 'math', 'planning', 'coding'],
        }
        
        # Score
        score = sum(factors.values())
        recommend_dki = score >= 3
        
        # Reasoning
        if recommend_dki:
            reasoning = "DKI recommended: "
            reasons = []
            if factors['memory_tokens_significant']:
                reasons.append(f"significant memory ({memory_tokens} tokens)")
            if factors['context_constrained']:
                reasons.append(f"constrained context ({analysis.rag_token_budget_free} free)")
            if factors['token_efficiency_high']:
                reasons.append(f"high token efficiency ({analysis.token_efficiency_gain:.1%})")
            if factors['reasoning_task']:
                reasons.append(f"reasoning-intensive task ({task_type})")
            reasoning += ", ".join(reasons)
        else:
            reasoning = "RAG may be sufficient: "
            if not factors['memory_tokens_significant']:
                reasoning += f"small memory ({memory_tokens} tokens), "
            if not factors['context_constrained']:
                reasoning += f"ample context ({analysis.rag_token_budget_free} free), "
            if not factors['reasoning_task']:
                reasoning += f"non-reasoning task ({task_type})"
        
        return {
            'recommend_dki': recommend_dki,
            'score': score,
            'factors': factors,
            'reasoning': reasoning,
            'analysis': analysis.to_dict(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        if not self._history:
            return {'count': 0}
        
        return {
            'count': len(self._history),
            'avg_token_efficiency_gain': sum(
                a.token_efficiency_gain for a in self._history
            ) / len(self._history),
            'avg_attention_overhead': sum(
                a.attention_overhead_ratio for a in self._history
            ) / len(self._history),
            'avg_memory_tokens': sum(
                a.memory_tokens for a in self._history
            ) / len(self._history),
            'latency': self.get_average_latency(),
        }
    
    def clear(self) -> None:
        """Clear history."""
        self._history.clear()
        self._latency_history.clear()


class LatencyTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.breakdown = LatencyBreakdown()
        self._current_stage: Optional[str] = None
        self._stage_start: float = 0.0
        self._total_start: float = 0.0
    
    def __enter__(self):
        self._total_start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.breakdown.total_ms = (time.perf_counter() - self._total_start) * 1000
    
    def start_stage(self, stage: str) -> None:
        """Start timing a stage."""
        if self._current_stage:
            self.end_stage()
        self._current_stage = stage
        self._stage_start = time.perf_counter()
    
    def end_stage(self) -> None:
        """End timing current stage."""
        if self._current_stage:
            elapsed = (time.perf_counter() - self._stage_start) * 1000
            stage_attr = f"{self._current_stage}_ms"
            if hasattr(self.breakdown, stage_attr):
                setattr(self.breakdown, stage_attr, elapsed)
            self._current_stage = None
    
    def set_cache_info(self, hit: bool, tier: str) -> None:
        """Set cache hit information."""
        self.breakdown.cache_hit = hit
        self.breakdown.cache_tier = tier
