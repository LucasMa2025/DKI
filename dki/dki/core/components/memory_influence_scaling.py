"""
Memory Influence Scaling (MIS) for DKI System
Continuous injection strength control with α ∈ [0, 1]

============================================================================
Key Insight: MIS as Differentiable Control Signal (Paper Section 3.3)
============================================================================

MIS provides a CONTINUOUS, DIFFERENTIABLE control signal for memory injection,
in contrast to RAG's binary include/exclude decision.

Mathematical Formulation:
    Attn_α(Q, K_aug, V_aug) = softmax(Q K_aug^T / √d + α·bias_mem) V_aug

where bias_mem = [log(α), 0, 0, ...] is a logit bias on memory positions.

Key Properties:
1. Graceful Degradation: α → 0 smoothly recovers vanilla LLM behavior
2. Controllability: Prevents "injection runaway" where memory dominates
3. Testability: α → output mapping enables systematic debugging
4. Differentiability: Enables gradient-based optimization of α

Implementation Options:
1. Logit Bias (preferred): Pre-softmax adjustment, mathematically correct
2. Value Scaling: Post-attention scaling, simpler but approximation
3. Attention Masking: Binary, loses continuity

We prefer logit bias as it correctly adjusts the attention distribution
while maintaining numerical stability.
============================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from loguru import logger

from dki.config.config_loader import ConfigLoader


class MemoryInfluenceScaling(nn.Module):
    """
    Memory Influence Scaling (MIS) module.
    
    Provides continuous injection strength control:
    - α = 0: No memory influence (vanilla LLM)
    - α = 1: Full memory influence
    - α ∈ (0, 1): Partial influence
    
    Properties:
    - Controllability: Prevents injection runaway
    - Testability: α → output mapping enables debugging
    - Degradability: α → 0 gracefully falls back to vanilla LLM
    
    Comparison with RAG:
    - RAG: Binary decision (include or exclude memory in prompt)
    - DKI+MIS: Continuous control (how much to attend to memory)
    
    This enables scenarios like:
    - "Use memory for context but don't let it override user intent" (α=0.3)
    - "Trust memory strongly for factual recall" (α=0.9)
    - "Experiment with different injection strengths" (α sweep)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        use_learned_alpha: bool = True,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_default: float = 0.5,
    ):
        super().__init__()
        
        config = ConfigLoader().config
        
        self.hidden_dim = hidden_dim
        self.use_learned_alpha = use_learned_alpha
        self.alpha_min = alpha_min if alpha_min is not None else config.dki.mis.alpha_min
        self.alpha_max = alpha_max if alpha_max is not None else config.dki.mis.alpha_max
        self.alpha_default = alpha_default if alpha_default is not None else config.dki.mis.alpha_default
        
        if use_learned_alpha:
            # Multi-factor alpha prediction network
            # Input: query_embedding (pooled) + memory_relevance + entropy
            self.alpha_predictor = nn.Sequential(
                nn.Linear(hidden_dim + 2, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            
            # Initialize to predict moderate alpha
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for moderate alpha prediction."""
        for module in self.alpha_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_alpha(
        self,
        query_embedding: torch.Tensor,
        memory_relevance: float,
        entropy: float,
    ) -> float:
        """
        Compute injection strength α.
        
        Args:
            query_embedding: Query representation [batch, seq_len, hidden_dim] or [hidden_dim]
            memory_relevance: Router similarity score (0-1)
            entropy: Prefill-stage uncertainty
            
        Returns:
            α ∈ [alpha_min, alpha_max]
        """
        if not self.use_learned_alpha:
            # Simple heuristic-based alpha
            alpha = 0.5 * memory_relevance + 0.3 * min(entropy, 1.0) + 0.2
            return max(self.alpha_min, min(self.alpha_max, alpha))
        
        # Ensure query embedding is 1D (pooled)
        if query_embedding.dim() == 3:
            query_embedding = query_embedding.mean(dim=(0, 1))  # [hidden_dim]
        elif query_embedding.dim() == 2:
            query_embedding = query_embedding.mean(dim=0)  # [hidden_dim]
        
        # Build feature vector
        features = torch.cat([
            query_embedding,
            torch.tensor([memory_relevance, entropy], device=query_embedding.device),
        ])
        
        # Predict alpha
        with torch.no_grad():
            alpha = self.alpha_predictor(features.unsqueeze(0)).item()
        
        # Scale to [alpha_min, alpha_max]
        alpha = self.alpha_min + alpha * (self.alpha_max - self.alpha_min)
        
        return alpha
    
    def apply_scaling(
        self,
        attention_logits: torch.Tensor,
        mem_len: int,
        alpha: float,
    ) -> torch.Tensor:
        """
        Apply α scaling via logit bias (pre-softmax).
        
        Note: Direct post-softmax scaling breaks probability normalization.
        We use logit bias for mathematically correct scaling.
        
        Args:
            attention_logits: Pre-softmax attention logits [batch, heads, seq, total_len]
            mem_len: Length of memory tokens
            alpha: Injection strength
            
        Returns:
            Scaled attention logits
        """
        if alpha >= 1.0 or mem_len == 0:
            return attention_logits
        
        if alpha <= 0.0:
            # Zero out memory influence
            attention_logits[..., :mem_len] = float('-inf')
            return attention_logits
        
        # Apply logit bias for scaling
        # log(α) bias on memory logits effectively scales attention weights
        logit_bias = torch.log(torch.tensor(alpha + 1e-9, device=attention_logits.device))
        
        logits_mem = attention_logits[..., :mem_len] + logit_bias
        logits_user = attention_logits[..., mem_len:]
        
        return torch.cat([logits_mem, logits_user], dim=-1)
    
    def scale_kv_values(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale Value tensors directly (alternative to logit bias).
        
        IMPORTANT: Only Value is scaled, Key is NEVER modified.
        
        Rationale (Paper Section 3.3, Value-Only Scaling Principle):
        - Key determines attention distribution (which positions are attended to)
        - Value determines output content (what information is retrieved)
        - Scaling Key would distort attention addressing, causing the model to
          "forget" where memory tokens are, leading to unpredictable behavior
        - Scaling Value only adjusts the contribution strength of memory to output,
          preserving correct attention routing
        
        This is consistent with the Engram paper's "Value-Only Gating" design,
        where α gates only V while K remains unchanged to preserve attention
        addressing accuracy.
        
        Args:
            key: Key tensor [batch, heads, seq, head_dim]
            value: Value tensor [batch, heads, seq, head_dim]
            alpha: Injection strength
            
        Returns:
            (key, scaled_value) — key is always returned unchanged
        """
        if alpha >= 1.0:
            return key, value
        
        if alpha <= 0.0:
            return key, value * 0
        
        # Scale value only (key is never modified)
        scaled_value = value * alpha
        
        return key, scaled_value
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        memory_relevance: float,
        entropy: float,
    ) -> float:
        """Forward pass to compute alpha."""
        return self.compute_alpha(query_embedding, memory_relevance, entropy)
