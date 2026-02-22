"""
DKI Architecture Documentation
Explains the key architectural differences and design decisions

This module serves as living documentation for the DKI system architecture,
explaining key concepts from the paper and their implementation.
"""

# ===========================================================================
# DKI vs Cross-Attention: A Fundamental Distinction
# ===========================================================================
#
# DKI is often confused with Cross-Attention, but they are fundamentally
# different mechanisms. This distinction is critical for understanding
# when and why to use DKI.
#
# Cross-Attention (e.g., in Encoder-Decoder Transformers):
# --------------------------------------------------------
# - Separate learned parameters: W_q^cross, W_k^cross, W_v^cross
# - Dedicated cross-attention layers (typically every N layers)
# - Requires training: Cross-attention weights must be learned
# - Architecture-specific: Cannot be added to decoder-only models post-hoc
# - Source: External encoder output (different semantic space)
#
# Mathematical formulation:
#     Attn_cross(Q, K_enc, V_enc) = softmax(Q W_q^cross (K_enc W_k^cross)^T / √d) V_enc W_v^cross
#
# DKI (Dynamic KV Injection):
# ---------------------------
# - Reuses existing self-attention parameters: W_k, W_v
# - Injects into existing self-attention (no new layers)
# - Training-free: Works with frozen decoder-only models
# - Architecture-agnostic: Works with any transformer
# - Source: Same model's K/V computation (same semantic space)
# - α-controllable: Continuous injection strength
# - Graceful degradation: α → 0 recovers vanilla behavior
#
# Mathematical formulation:
#     K_aug = [K_mem; K_user]  (concatenation)
#     V_aug = [V_mem; V_user]
#     Attn_dki(Q, K_aug, V_aug) = softmax(Q K_aug^T / √d) V_aug
#
#     With MIS (Memory Influence Scaling):
#     Attn_dki(Q, K_aug, V_aug, α) = softmax(Q K_aug^T / √d + α·bias_mem) V_aug
#
# ===========================================================================


# ===========================================================================
# Attention Budget Reallocation Hypothesis
# ===========================================================================
#
# From Paper Section 3.2:
#
# Definition (Budget Types):
# - Token Budget (B_t): Hard constraint - context window size
#   Exceeding B_t causes truncation → information loss
#
# - Attention Budget (B_a = n_q × n_k): Soft constraint - computational cost
#   Exceeding B_a increases latency but no information loss
#
# Budget Comparison:
#
# RAG:
#     B_t^used = n_m + n_u  (memory + user tokens)
#     B_a = (n_m + n_u)^2   (quadratic in total length)
#
# DKI:
#     B_t^used = n_u        (user tokens only, memory not in prompt)
#     B_a = n_u × (n_m + n_u)  (linear in user length)
#
# Key Insight:
# DKI trades attention budget for token budget. This is beneficial when:
# 1. Context window is constrained
# 2. Memory is large relative to user query
# 3. Task requires reasoning over freed token budget
#
# Hypothesis (Cognitive Bandwidth Release):
# In reasoning-intensive tasks, the marginal benefit of releasing token budget
# exceeds the marginal cost of increased attention budget:
#     ∂TaskSuccess/∂B_t^free > ∂Latency/∂B_a
#
# ===========================================================================


# ===========================================================================
# Memory Hierarchy and Amortization
# ===========================================================================
#
# From Paper Section 7.4:
#
# DKI memory footprint scales with ACTIVE memories, not total corpus size.
# This enables a tiered caching strategy:
#
# ┌─────────────────────────────────────────────────────────────┐
# │                    DKI Memory Hierarchy                      │
# ├─────────────────────────────────────────────────────────────┤
# │  L1: GPU HBM (Hot)                                          │
# │  ├── Top-k most recently used memories                      │
# │  ├── Uncompressed FP16                                      │
# │  └── Capacity: 5-10 memories per session                    │
# │                                                             │
# │  L2: CPU RAM (Warm)                                         │
# │  ├── Session-active memories                                │
# │  ├── Compressed (2-4× via GEAR-style compression)           │
# │  └── Capacity: 50-100 memories per session                  │
# │                                                             │
# │  L3: NVMe SSD (Cold)                                        │
# │  ├── All session memories                                   │
# │  ├── Quantized INT8 + Compressed (8×)                       │
# │  └── Capacity: Unlimited                                    │
# │                                                             │
# │  L4: Recompute on Demand                                    │
# │  ├── Store only text + routing vectors                      │
# │  └── Recompute K/V when promoted to L3                      │
# └─────────────────────────────────────────────────────────────┘
#
# Memory Amortization:
# Session KV Cache transforms DKI from a STATELESS memory injection
# mechanism into a STATEFUL temporal operator:
#
#     Amortized Cost = (C_compute + (T-1)·C_load) / T → C_load as T → ∞
#
# This makes DKI MORE EFFICIENT than RAG for long conversations,
# as RAG must re-process memory tokens on every turn.
#
# ===========================================================================


# ===========================================================================
# Design Invariants
# ===========================================================================
#
# From Paper Section 5:
#
# 1. Storage Model-Agnostic
#    Memory is stored as text + routing vectors (embeddings).
#    K/V representations are computed on-demand by the target model.
#    This ensures compatibility across different LLMs.
#
# 2. Injection Model-Consistent
#    K/V for injection MUST come from the same model being used for generation.
#    Cross-model K/V injection would cause semantic space mismatch.
#
# 3. Session Cache Disposable
#    Session KV cache is an inference-time optimization only.
#    It can be cleared without affecting correctness.
#    Persistent memory is always stored as text.
#
# 4. Graceful Degradation
#    When α → 0, DKI smoothly recovers vanilla LLM behavior.
#    This provides a safety net for uncertain injection decisions.
#
# 5. Audit Logging
#    All injection decisions are logged for compliance and debugging.
#    Includes: query, memories used, α value, gating reasoning.
#
# ===========================================================================


# ===========================================================================
# Query-Conditioned Projection: Memory-Centric Design
# ===========================================================================
#
# From Paper Section 3.4.2:
#
# The projection is MEMORY-CENTRIC, not query-centric:
# - Query only MODULATES, never re-encodes memory semantics
# - This differs from cross-attention where query actively re-weights
#
# Structural Constraints (implicit regularization):
# 1. γ bounded: ||γ||_2 bounded by initialization and training dynamics
# 2. β zero-mean: Initialized to zero, shifts should remain small
# 3. Residual dominance: X_proj ≈ X_mem + ε·f(X_mem, X_user)
#
# These constraints prevent "hallucinating" new memory content;
# the projection can only adjust EMPHASIS of existing information.
#
# FiLM-style modulation:
#     X_proj = γ(X_user) ⊙ X_mem_low + β(X_user)
#
# where X_mem_low is a low-rank projection of memory embeddings.
#
# ===========================================================================


# ===========================================================================
# Entropy as Heuristic Proxy
# ===========================================================================
#
# From Paper Section 3.5.2:
#
# We use attention entropy as a HEURISTIC proxy for model uncertainty,
# NOT as a rigorous uncertainty estimator. This distinction is critical:
#
# When entropy works well:
# - Factual QA: High entropy likely indicates epistemic uncertainty
# - Knowledge retrieval: Distributed attention suggests need for grounding
#
# When entropy may mislead:
# - Brainstorming/Creative: High entropy is inherent task property
# - Summarization: Distributed attention is appropriate, not uncertainty
# - Open-ended questions: High entropy doesn't mean memory is needed
#
# Limitations:
# - Entropy conflates epistemic and aleatoric uncertainty
# - Task-specific calibration may be needed
# - Alternative signals (logit variance, layer disagreement) may be more robust
#
# This is why we use DUAL-FACTOR gating (entropy × relevance) rather than
# single-factor entropy-based gating.
#
# ===========================================================================


def get_architecture_summary() -> str:
    """Return a summary of DKI architecture for documentation."""
    return """
    DKI (Dynamic KV Injection) Architecture Summary
    ================================================
    
    Core Mechanism:
    - Inject pre-computed K/V representations into self-attention
    - No prompt tokens consumed, preserving context window
    - Continuous α control for injection strength
    
    Key Components:
    1. Memory Router: Semantic retrieval of relevant memories
    2. Memory Influence Scaling (MIS): Continuous α ∈ [0, 1] control
    3. Query-Conditioned Projection (QCP): FiLM-style adaptive projection
    4. Dual-Factor Gating: Uncertainty × Relevance decision
    5. Tiered KV Cache: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
    6. Position Remapper: RoPE/ALiBi compatibility
    
    Design Principles:
    - Training-free: Works with frozen decoder-only models
    - Architecture-agnostic: Compatible with any transformer
    - Graceful degradation: α → 0 recovers vanilla behavior
    - Audit logging: All decisions logged for compliance
    
    When to Use DKI vs RAG:
    - DKI: Large memories, constrained context, multi-turn, fine control
    - RAG: Simple retrieval, short memories, single-turn
    """


def get_comparison_table() -> str:
    """Return a comparison table of DKI vs RAG vs Cross-Attention."""
    return """
    ┌─────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │        Feature          │       RAG        │       DKI        │  Cross-Attention │
    ├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Token Budget            │ Consumed         │ Preserved        │ N/A              │
    │ Attention Budget        │ (n_m + n_u)²     │ n_u × (n_m + n_u)│ n_q × n_enc      │
    │ Training Required       │ No               │ No               │ Yes              │
    │ Architecture Change     │ No               │ No               │ Yes              │
    │ Injection Control       │ Binary           │ Continuous (α)   │ Learned          │
    │ Graceful Degradation    │ Remove from prompt│ α → 0           │ Mask cross-attn  │
    │ Multi-turn Efficiency   │ Re-encode each turn│ Cache K/V      │ Re-encode        │
    │ Decoder-Only Compatible │ Yes              │ Yes              │ No               │
    └─────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
    """
