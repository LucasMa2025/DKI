"""
DKI Attention Module
FlashAttention-3/2 integration for optimized K/V injection

This module provides:
- FlashAttentionConfig: Configuration for FlashAttention
- FlashAttentionBackend: Backend detection and selection
- KVInjectionOptimizer: Optimized K/V injection using FlashAttention
- AttentionProfiler: Performance profiling

Architecture:
- FA3 (H100+): Full FlashAttention-3 with async + warp specialization
- FA2 (A100/4090): FlashAttention-2 fallback
- Standard: PyTorch native attention (compatibility mode)

Performance:
- FA3 vs Standard: ~70% latency reduction, ~40% memory reduction
- FA2 vs Standard: ~50% latency reduction, ~30% memory reduction

Author: AGI Demo Project
Version: 1.0.0
"""

from dki.attention.config import FlashAttentionConfig
from dki.attention.backend import (
    FlashAttentionBackend,
    FLASH_ATTN_AVAILABLE,
    FLASH_ATTN_VERSION,
)
from dki.attention.kv_injection import KVInjectionOptimizer
from dki.attention.profiler import AttentionProfiler

__all__ = [
    # Config
    "FlashAttentionConfig",
    # Backend
    "FlashAttentionBackend",
    "FLASH_ATTN_AVAILABLE",
    "FLASH_ATTN_VERSION",
    # Optimizer
    "KVInjectionOptimizer",
    # Profiler
    "AttentionProfiler",
]
