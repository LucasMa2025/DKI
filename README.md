# DKI - Dynamic KV Injection

> Attention-Level Memory Augmentation for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[简体中文](README_CN.md) | English

## 📖 Overview

DKI (Dynamic KV Injection) is a novel approach to memory augmentation for Large Language Models that injects memory content at the attention level rather than the token level.

Unlike traditional RAG (Retrieval-Augmented Generation) which consumes context window tokens, DKI computes Key-Value representations of memory content and injects them directly into the attention mechanism, **preserving the full context window for user input**.

### Key Features

-   **🧠 Attention-Level Injection**: Memory injected via K/V, not prompt tokens
-   **🎚️ Memory Influence Scaling (MIS)**: Continuous α ∈ [0, 1] control
-   **🔄 Query-Conditioned Projection**: FiLM-style memory-centric transformation
-   **🚦 Dual-Factor Gating**: Uncertainty × Relevance for smart injection decisions
-   **💾 Tiered KV Cache**: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
-   **📊 Attention Budget Analysis**: Token vs Attention budget tracking
-   **🔌 Multi-Engine Support**: vLLM, LLaMA, DeepSeek, GLM
-   **✅ Graceful Degradation**: α → 0 smoothly recovers vanilla LLM behavior

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Dynamic KV Injection System                         │
├─────────────────────────────────────────────────────────────────────────┤
│  User Query                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  1. Memory Router (FAISS + Sentence Embedding)                  │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  2. Dual-Factor Gating (Entropy × Relevance)                    │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                    ┌───────────┴───────────┐                            │
│                    ▼                       ▼                            │
│           ┌──────────────┐    ┌────────────────────────────────┐        │
│           │ Vanilla LLM  │    │ 3. Session KV Cache            │        │
│           │ (fallback)   │    │ + Query-Conditioned Projection │        │
│           └──────────────┘    └─────────────┬──────────────────┘        │
│                                             ▼                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  4. Memory Influence Scaling (α control)                        │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  5. LLM with KV Injection → Generate Response                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd DKI

# Setup (creates venv, installs dependencies, initializes DB)
# Windows:
scripts\setup.bat

# Linux/Mac:
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Start Web UI

```bash
# Windows:
scripts\start.bat web

# Linux/Mac:
./scripts/start.sh web
```

Open http://localhost:8080 in your browser.

### Python Usage

```python
from dki import DKISystem

# Initialize
dki = DKISystem()

# Add memories
dki.add_memory(
    session_id="user_001",
    content="User prefers vegetarian food and is allergic to seafood"
)
dki.add_memory(
    session_id="user_001",
    content="User lives in Beijing and enjoys hiking"
)

# Chat with memory injection
response = dki.chat(
    query="Recommend a restaurant for lunch",
    session_id="user_001"
)

print(response.text)
# Output considers vegetarian preference without explicit prompt mention

print(f"Alpha: {response.gating_decision.alpha}")
print(f"Memories used: {len(response.memories_used)}")
print(f"Latency: {response.latency_ms}ms")
```

## 📁 Project Structure

```
DKI/
├── config/
│   └── config.yaml           # Main configuration
├── dki/
│   ├── core/
│   │   ├── dki_system.py     # Main DKI system
│   │   ├── rag_system.py     # RAG baseline
│   │   ├── memory_router.py  # FAISS-based retrieval
│   │   ├── embedding_service.py
│   │   ├── architecture.py   # Architecture documentation
│   │   └── components/
│   │       ├── memory_influence_scaling.py
│   │       ├── query_conditioned_projection.py
│   │       ├── dual_factor_gating.py
│   │       ├── session_kv_cache.py
│   │       ├── tiered_kv_cache.py    # L1/L2/L3/L4 memory hierarchy
│   │       ├── attention_budget.py   # Budget analysis
│   │       └── position_remapper.py
│   ├── models/
│   │   ├── factory.py        # Model factory
│   │   ├── base.py           # Base adapter
│   │   ├── vllm_adapter.py
│   │   ├── llama_adapter.py
│   │   ├── deepseek_adapter.py
│   │   └── glm_adapter.py
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── connection.py     # DB connection manager
│   │   └── repository.py     # Repository pattern
│   ├── experiment/
│   │   ├── runner.py         # Experiment runner
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── data_generator.py # Test data generation
│   └── web/
│       └── app.py            # FastAPI + Web UI
├── scripts/
│   ├── init_db.sql           # Database schema
│   ├── setup.bat/.sh         # Setup scripts
│   └── start.bat/.sh         # Start scripts
├── data/                      # Experiment data
├── experiment_results/        # Experiment outputs
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
# Model Engine
model:
    default_engine: "vllm" # vllm, llama, deepseek, glm
    engines:
        vllm:
            model_name: "Qwen/Qwen2-7B-Instruct"
            tensor_parallel_size: 1

# DKI Settings
dki:
    gating:
        entropy_threshold: 0.5
        relevance_threshold: 0.7
    cache:
        max_size: 100
        strategy: "weighted" # lru, lfu, weighted

# RAG Settings
rag:
    top_k: 5
    similarity_threshold: 0.5
```

## 📊 Experiments

### Generate Test Data

```bash
python -m dki.experiment.data_generator
```

### Run Comparison Experiment

```python
from dki.experiment.runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()
config = ExperimentConfig(
    name="DKI vs RAG Comparison",
    modes=["dki", "rag", "baseline"],
    datasets=["persona_chat", "memory_qa"],
    max_samples=100
)

results = runner.run_experiment(config)
```

### Alpha Sensitivity Analysis

```python
results = runner.run_alpha_sensitivity(
    alpha_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
```

## 📈 API Reference

### REST API

| Endpoint                     | Method | Description                |
| ---------------------------- | ------ | -------------------------- |
| `/api/chat`                  | POST   | Chat with DKI/RAG/Baseline |
| `/api/memory`                | POST   | Add memory                 |
| `/api/memories/{session_id}` | GET    | Get session memories       |
| `/api/search`                | POST   | Search memories            |
| `/api/stats`                 | GET    | Get system statistics      |
| `/api/experiment/run`        | POST   | Run experiment             |

### Chat Request

```json
{
    "query": "Recommend a restaurant",
    "session_id": "user_001",
    "mode": "dki",
    "force_alpha": 0.7,
    "max_new_tokens": 256,
    "temperature": 0.7
}
```

### Chat Response

```json
{
    "response": "Based on your preference for vegetarian food...",
    "mode": "dki",
    "session_id": "user_001",
    "latency_ms": 156.3,
    "memories_used": [...],
    "alpha": 0.72,
    "cache_hit": true
}
```

## 🔬 Research Background

DKI addresses a fundamental limitation of RAG: retrieved content consumes context window capacity, creating a trade-off between memory content and user input space.

**RAG Paradigm:**

```
[Retrieved Content (consumed)] [User Input (remaining)]
Token Budget: B_t^used = n_m + n_u
Attention Budget: B_a = (n_m + n_u)²
```

**DKI Paradigm:**

```
[User Input (full budget available)]
     ↑ Memory injected via K/V (not in token budget)
Token Budget: B_t^used = n_u (memory free!)
Attention Budget: B_a = n_u × (n_m + n_u)
```

### DKI vs Cross-Attention

DKI is NOT equivalent to Cross-Attention:

| Feature       | DKI                  | Cross-Attention                          |
| ------------- | -------------------- | ---------------------------------------- |
| Parameters    | Reuses W_k, W_v      | Separate W_q^cross, W_k^cross, W_v^cross |
| Training      | Training-free        | Requires training                        |
| Architecture  | No modification      | Dedicated layers                         |
| Compatibility | Any decoder-only LLM | Encoder-decoder only                     |
| Control       | Continuous α         | Learned weights                          |

### Design Invariants

1. **Storage Model-Agnostic**: Store only original text + routing vectors
2. **Injection Model-Consistent**: K/V computed with target model parameters
3. **Session Cache Disposable**: Inference-time enhancement, not persistent
4. **Graceful Degradation**: α → 0 falls back to vanilla LLM
5. **Audit Logging**: All injection decisions logged for compliance

### Memory Hierarchy (Tiered KV Cache)

```
┌────────────────────────────────────────────────────┐
│  L1: GPU HBM (Hot)     - Uncompressed FP16         │
│  L2: CPU RAM (Warm)    - Compressed (2-4×)         │
│  L3: NVMe SSD (Cold)   - Quantized INT8 (8×)       │
│  L4: Text Only         - Recompute on demand       │
└────────────────────────────────────────────────────┘
```

Memory footprint scales with ACTIVE memories, not total corpus size.

## 📚 Detailed Documentation

### Core Concepts

#### 1. Attention Budget Reallocation Hypothesis

**Hypothesis Statement**: In reasoning-intensive tasks, the marginal benefit of releasing token budget exceeds the marginal cost of increased attention computation.

**Mathematical Formulation**:

```
∂TaskSuccess/∂B_t^free > ∂Latency/∂B_a
```

**Intuition**:

-   Token budget is a **hard constraint** (truncation causes information loss)
-   Attention budget is a **soft constraint** (increased computation, but no information loss)
-   For tasks requiring deep reasoning chains (multi-step math, complex planning), preserving token budget for reasoning steps provides greater utility than attention latency cost

#### 2. Memory Influence Scaling (MIS)

Continuous strength control α ∈ [0, 1]:

| α Value     | Behavior                          | Use Case               |
| ----------- | --------------------------------- | ---------------------- |
| 0.0         | No memory influence (vanilla LLM) | Safe fallback          |
| 0.2-0.4     | Gentle injection                  | Exploratory dialogue   |
| **0.4-0.7** | **Optimal range**                 | **Most scenarios**     |
| 0.8-1.0     | Strong injection                  | High-confidence memory |

**Implementation**: Scaling applied via pre-softmax logit bias:

```python
logit_bias = torch.log(torch.tensor(alpha + 1e-9))
logits_mem_scaled = logits_mem + logit_bias
```

#### 3. Query-Conditioned Projection (QCP)

Query-dependent memory projection using FiLM-style modulation:

```python
# Generate modulation parameters
gamma = gamma_net(query_context)  # Scale
beta = beta_net(query_context)    # Shift

# Low-rank projection with FiLM modulation
X_mem_low = X_mem @ W_mem
X_mem_modulated = X_mem_low * gamma + beta
X_mem_proj = proj_out(X_mem_modulated)

# Residual connection
return layer_norm(X_mem + X_mem_proj)
```

**Key Principle**: Projection is **memory-centric**, not query-centric—query only modulates, never re-encodes memory semantics.

#### 4. Dual-Factor Gating

**Design Decision**: Injection decision is **relevance-driven**; uncertainty only **modulates the upper bound of α**.

```python
# Factor 1: Memory relevance (PRIMARY)
inject = similarity_top1 > threshold_relevance

# Factor 2: Entropy modulates α ceiling (not decision)
alpha_max = entropy_floor + (entropy_ceiling - entropy_floor) * entropy

# Continuous strength
alpha = min(alpha_base, alpha_max)
```

**Important Note**: We use attention entropy as a **heuristic proxy** for model uncertainty, not a rigorous uncertainty estimator.

### Comparison with RAG

| Dimension          | RAG                                 | DKI                                       |
| ------------------ | ----------------------------------- | ----------------------------------------- |
| Injection Level    | Token domain (prompt concatenation) | Attention domain (K/V injection)          |
| Injection Control  | None (hard concatenation)           | Continuous (α ∈ [0, 1])                   |
| Prompt Consumption | Yes                                 | No                                        |
| Context Window     | Consumed by retrieved content       | Fully available to user                   |
| Prompt Engineering | Required                            | Simplified                                |
| Interpretability   | High (visible in prompt)            | Medium (requires attention visualization) |
| Ecosystem Maturity | High (LangChain, LlamaIndex, etc.)  | Low (emerging)                            |
| Debugging          | Straightforward                     | Requires specialized tools                |

### Applicable Scenarios

#### ✅ Recommended for DKI

1. **Long Context + Constrained Window**

    - User input > 2500 tokens
    - Context window < 8K tokens
    - Need to inject > 200 tokens of memory

2. **Multi-turn Dialogue**

    - > 3 turns of interaction
    - Persistent personalization needed
    - Session cache significantly reduces latency

3. **Fine-grained Control Requirements**

    - Need dynamic memory strength adjustment
    - Different memories need different weights
    - Auditable injection decisions required

4. **Real-time Interactive Systems**
    - Latency sensitive (after multiple turns)
    - Fast response time needed
    - Token billing is expensive

#### ⚠️ Use DKI with Caution

1. **Simple Factual QA**

    - Single-turn queries
    - Ample context window
    - RAG is simpler and more direct

2. **First-turn Latency Critical**

    - First interaction must be fastest
    - Cannot accept gating overhead
    - Cold start performance critical

3. **High Audit Requirements**

    - Need to display full retrieved content
    - Regulators require visible prompts
    - Attention visualization insufficient

4. **Rapid Prototyping**
    - Using mature RAG ecosystem
    - Need fast iteration
    - No deep customization needed

### Performance Benchmarks

Based on DeepSeek-V3 7B experiments (n=500):

| Metric                 | RAG    | DKI        | Change     |
| ---------------------- | ------ | ---------- | ---------- |
| Memory Recall          | 87.3%  | 86.2%      | -1.1%      |
| First Turn Latency     | 78.8ms | 92.4ms     | +17.3%     |
| **Subsequent Latency** | 76.1ms | **42.8ms** | **-43.7%** |
| Cache Hit Rate         | N/A    | 69.7%      | -          |
| Hallucination Rate     | 10.2%  | 10.4%      | +0.2%      |

**Key Findings**:

-   DKI has 17.3% overhead on first turn, but 43.7% reduction on subsequent turns
-   After 3 turns total latency: DKI < RAG (178.2ms vs 228.9ms)
-   Memory recall only drops 1.1%, hallucination rate comparable

### Failure Modes and Mitigation

| Failure Mode           | Symptoms                                     | Detection                             | Mitigation                            |
| ---------------------- | -------------------------------------------- | ------------------------------------- | ------------------------------------- |
| Memory Confusion       | Multiple similar memories cause mixed output | Compute inter-memory similarity       | Keep only top-1 when similarity > 0.9 |
| Temporal Inconsistency | Old memory conflicts with current context    | Check memory timestamp vs query tense | Add time decay factor to α            |
| Projection Overfitting | Good training, poor test performance         | Monitor validation vs training loss   | Increase dropout, data augmentation   |
| Cache Thrashing        | Hit rate < 30%, latency increases            | Monitor cache hit rate                | Increase cache size or use LFU        |
| Bias Amplification     | Memory injection amplifies model biases      | Monitor output diversity metrics      | Implement diversity-aware routing     |

### Advanced Features

#### Tiered KV Cache

```python
from dki.core.components.tiered_kv_cache import TieredKVCache

cache = TieredKVCache(
    l1_size_gb=8,      # GPU
    l2_size_gb=32,     # CPU RAM
    l3_size_gb=128,    # SSD
    compression_l2=4,   # L2 compression ratio
    compression_l3=8    # L3 compression ratio
)

# Automatic tiered management
kv_pair = cache.get_or_compute(memory_id, compute_fn)
```

#### Attention Budget Tracking

```python
from dki.core.components.attention_budget import AttentionBudgetTracker

tracker = AttentionBudgetTracker()
with tracker.track_request(session_id):
    response = dki.chat(query, session_id)

# Get budget usage
budget = tracker.get_budget_usage(session_id)
print(f"Token budget used: {budget.token_used}/{budget.token_total}")
print(f"Attention FLOPs: {budget.attention_flops}")
```

### FAQ

**Q: Does DKI require retraining the model?**  
A: No. DKI is an inference-time enhancement using frozen model parameters. Only the α predictor and QCP are optional small networks (if training is needed).

**Q: Can DKI be combined with existing RAG systems?**  
A: Yes. You can use RAG for initial retrieval, then use DKI for injection. This is a hybrid approach.

**Q: What about memory footprint?**  
A: Each 200-token memory is ~100MB (uncompressed). Using tiered cache and GEAR compression can achieve 8× compression.

**Q: Which position encoding schemes are supported?**  
A: Currently RoPE and ALiBi are supported. For other schemes, see the implementation in `position_remapper.py`.

**Q: How to debug injection decisions?**  
A: Enable audit logging (`audit_logging: true` in `config.yaml`), all injection decisions will be logged including memory_ids, α values, and gating reasons.

**Q: Production deployment recommendations?**  
A:

1. Start with small cache (size=50) and monitor hit rate
2. Use Weighted cache strategy
3. Set reasonable α ceiling (0.7)
4. Monitor hallucination rate
5. A/B test against RAG

### Roadmap

-   [x] Core DKI implementation
-   [x] vLLM adapter
-   [x] Experiment framework
-   [x] LLaMA/DeepSeek/GLM adapters
-   [x] FlashAttention-3 integration
-   [ ] Multi-modal extension (image memory)
-   [ ] Distributed deployment support
-   [ ] Attention visualization tools
-   [ ] LangChain/LlamaIndex integration

### Acknowledgments

This project is inspired by the following research:

-   RAG ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401))
-   RETRO ([Borgeaud et al., 2022](https://arxiv.org/abs/2112.04426))
-   Self-RAG ([Asai et al., 2023](https://arxiv.org/abs/2310.11511))
-   FiLM ([Perez et al., 2018](https://arxiv.org/abs/1709.07871))
-   FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))

## 📄 Related Papers

This project is based on the paper "Dynamic KV Injection: Attention-Level Memory Augmentation for Large Language Models".

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

---

**DKI** - Rethinking Memory Augmentation at the Attention Level
