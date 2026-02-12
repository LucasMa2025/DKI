"""
DKI Injection Strategies

提供不同的注入策略实现:
- stable: 稳定策略 (偏好 K/V + 历史 Suffix Prompt)
- full_attention: 研究策略 (偏好 K/V + 历史 K/V)
"""

from dki.core.injection.full_attention_injector import (
    FullAttentionInjector,
    FullAttentionConfig,
    PositionMode,
    InjectionResult,
)

__all__ = [
    "FullAttentionInjector",
    "FullAttentionConfig",
    "PositionMode",
    "InjectionResult",
]
