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

## 📄 Related Papers

This project is based on the paper "Dynamic KV Injection: Attention-Level Memory Augmentation for Large Language Models".

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

---

**DKI** - Rethinking Memory Augmentation at the Attention Level
