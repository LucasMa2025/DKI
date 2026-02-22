"""
DKI Injection Strategies

提供不同的注入策略实现:
- stable: 稳定策略 (偏好 K/V + 历史 Suffix Prompt)
- full_attention: 研究策略 (偏好 K/V + 历史 K/V)
- engram_inspired: Engram-Inspired 增强策略 (上下文门控 + 卷积精化 + 层级注入)
"""

from dki.core.injection.full_attention_injector import (
    FullAttentionInjector,
    FullAttentionConfig,
    PositionMode,
    InjectionResult,
)

from dki.core.injection.engram_inspired_injector import (
    EngramInspiredFullAttentionInjector,
    EngramInspiredConfig,
    EngramInjectionResult,
    ContextAwareGating,
    ConvolutionRefinement,
    LayerInjectionPolicy,
    GatingMode,
    RefinementMode,
    RMSNorm,
)

__all__ = [
    # Original Full-Attention
    "FullAttentionInjector",
    "FullAttentionConfig",
    "PositionMode",
    "InjectionResult",
    # Engram-Inspired
    "EngramInspiredFullAttentionInjector",
    "EngramInspiredConfig",
    "EngramInjectionResult",
    "ContextAwareGating",
    "ConvolutionRefinement",
    "LayerInjectionPolicy",
    "GatingMode",
    "RefinementMode",
    "RMSNorm",
]
