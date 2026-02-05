"""
Dual-Factor Gating for DKI System
Combines uncertainty and relevance for injection decisions
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from dki.core.memory_router import MemoryRouter, MemorySearchResult
from dki.models.base import BaseModelAdapter
from dki.config.config_loader import ConfigLoader


@dataclass
class GatingDecision:
    """Gating decision result."""
    should_inject: bool
    alpha: float
    entropy: float
    relevance_score: float
    margin: float
    memories: List[MemorySearchResult]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'should_inject': self.should_inject,
            'alpha': self.alpha,
            'entropy': self.entropy,
            'relevance_score': self.relevance_score,
            'margin': self.margin,
            'memories': [m.to_dict() for m in self.memories],
            'reasoning': self.reasoning,
        }


class DualFactorGating:
    """
    Dual-Factor Gating: Inject = f(Uncertainty, Relevance)
    
    Single-factor gating based only on model uncertainty is insufficient.
    High uncertainty may indicate an inherently open-ended question
    rather than a need for memory.
    
    IMPORTANT: Entropy as Heuristic Proxy
    -------------------------------------
    We use attention entropy as a HEURISTIC proxy for model uncertainty,
    NOT as a rigorous uncertainty estimator. This distinction is critical:
    
    - Factual QA: High entropy likely indicates epistemic uncertainty
    - Brainstorming/Creative: High entropy is inherent task property
    - Summarization: Distributed attention is appropriate, not uncertainty
    
    Limitations:
    - Entropy conflates epistemic and aleatoric uncertainty
    - Task-specific calibration may be needed
    - Alternative signals (logit variance, layer disagreement) may be more robust
    
    Decision logic:
    - High entropy + High relevance → Inject with high α
    - High entropy + Low relevance → Don't inject (open-ended question)
    - Low entropy + High relevance → Inject with moderate α (clarification)
    - Low entropy + Low relevance → Don't inject (model is confident)
    """
    
    def __init__(
        self,
        entropy_threshold: Optional[float] = None,
        relevance_threshold: Optional[float] = None,
        use_margin: bool = True,
        margin_weight: float = 0.3,
    ):
        config = ConfigLoader().config
        
        self.entropy_threshold = entropy_threshold or config.dki.gating.entropy_threshold
        self.relevance_threshold = relevance_threshold or config.dki.gating.relevance_threshold
        self.use_margin = use_margin
        self.margin_weight = margin_weight or config.dki.gating.margin_weight
        
        # Alpha computation weights
        self.entropy_weight = 2.0
        self.relevance_weight = 1.5
    
    def compute_prefill_entropy(
        self,
        model: BaseModelAdapter,
        query: str,
        layer_idx: int = 3,
    ) -> float:
        """
        Compute prefill-stage entropy (no decoding).
        
        Args:
            model: Model adapter
            query: Input query
            layer_idx: Which layer's attention to use
            
        Returns:
            Normalized entropy value (0-1)
        """
        entropy = model.compute_prefill_entropy(query, layer_idx=layer_idx)
        
        # Normalize entropy to [0, 1] range
        # Typical entropy values vary by model; we use a soft sigmoid normalization
        normalized = 1.0 / (1.0 + torch.exp(torch.tensor(-entropy / 10.0)))
        
        return float(normalized)
    
    def should_inject(
        self,
        model: BaseModelAdapter,
        query: str,
        router: MemoryRouter,
        top_k: int = 5,
    ) -> GatingDecision:
        """
        Determine whether to inject memory and compute injection strength.
        
        Args:
            model: Model adapter for entropy computation
            query: User query
            router: Memory router for retrieval
            top_k: Number of memories to retrieve
            
        Returns:
            GatingDecision with injection decision and metadata
        """
        # Factor 1: Uncertainty (Prefill-level, no decode)
        entropy = self.compute_prefill_entropy(model, query)
        
        # Factor 2: Memory relevance from router
        top_results = router.search(query, top_k=top_k)
        
        if not top_results:
            return GatingDecision(
                should_inject=False,
                alpha=0.0,
                entropy=entropy,
                relevance_score=0.0,
                margin=0.0,
                memories=[],
                reasoning="No relevant memories found",
            )
        
        sim_top1 = top_results[0].score
        
        # Compute margin between top-1 and top-2 (confidence in retrieval)
        margin = 0.0
        if len(top_results) > 1:
            margin = top_results[0].score - top_results[1].score
        
        # Dual-factor decision
        entropy_condition = entropy > self.entropy_threshold
        relevance_condition = sim_top1 > self.relevance_threshold
        
        # Determine injection
        if relevance_condition:
            if entropy_condition:
                # High entropy + High relevance → Strong injection
                should_inject = True
                reasoning = "High uncertainty with relevant memory available"
            else:
                # Low entropy + High relevance → Moderate injection for enrichment
                should_inject = True
                reasoning = "Relevant memory available for context enrichment"
        else:
            if entropy_condition:
                # High entropy + Low relevance → Don't inject
                should_inject = False
                reasoning = "High uncertainty but no relevant memory (open-ended question)"
            else:
                # Low entropy + Low relevance → Don't inject
                should_inject = False
                reasoning = "Model confident and no relevant memory needed"
        
        # Compute continuous alpha if injecting
        if should_inject:
            # Sigmoid-based alpha computation
            alpha_input = (
                self.entropy_weight * (entropy - self.entropy_threshold) +
                self.relevance_weight * (sim_top1 - self.relevance_threshold)
            )
            
            if self.use_margin:
                alpha_input += self.margin_weight * margin
            
            alpha = float(torch.sigmoid(torch.tensor(alpha_input)))
            
            # Ensure minimum alpha when injecting
            alpha = max(0.1, alpha)
        else:
            alpha = 0.0
        
        return GatingDecision(
            should_inject=should_inject,
            alpha=alpha,
            entropy=entropy,
            relevance_score=sim_top1,
            margin=margin,
            memories=top_results,
            reasoning=reasoning,
        )
    
    def force_inject(
        self,
        router: MemoryRouter,
        query: str,
        alpha: float = 1.0,
        top_k: int = 5,
    ) -> GatingDecision:
        """
        Force injection without gating check.
        
        Args:
            router: Memory router
            query: User query
            alpha: Fixed injection strength
            top_k: Number of memories
            
        Returns:
            GatingDecision with forced injection
        """
        top_results = router.search(query, top_k=top_k)
        
        return GatingDecision(
            should_inject=len(top_results) > 0,
            alpha=alpha,
            entropy=0.5,  # Default
            relevance_score=top_results[0].score if top_results else 0.0,
            margin=0.0,
            memories=top_results,
            reasoning="Forced injection (gating bypassed)",
        )
    
    def update_thresholds(
        self,
        entropy_threshold: Optional[float] = None,
        relevance_threshold: Optional[float] = None,
    ) -> None:
        """Update gating thresholds (for A/B testing)."""
        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
        if relevance_threshold is not None:
            self.relevance_threshold = relevance_threshold
        
        logger.info(f"Updated thresholds: entropy={self.entropy_threshold}, relevance={self.relevance_threshold}")
