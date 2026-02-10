# DKI - Dynamic KV Injection

> Attention-Level User Memory Plugin for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[简体中文](README_CN.md) | English

## 📖 Overview

DKI (Dynamic KV Injection) is an **LLM attention-level plugin** that dynamically injects user preferences and session history via Attention Hooks during inference, enabling cross-session personalized memory.

### What DKI Is

DKI is an **LLM plugin** designed specifically for **user-level memory**:

-   **Attention Hook Mechanism**: Injects K/V at the attention level via PyTorch Hooks, not prompt concatenation
-   **Configuration-Driven Adapter**: Automatically reads from upstream application databases, no code changes required
-   **Hybrid Injection Strategy**: Preference K/V injection (negative position) + History suffix prompt (positive position)

**Core Workflow**:

```
Upstream App → Pass user_id + raw input → DKI Plugin
    ↓
DKI reads upstream app database via config-driven adapter
    ↓
Preferences → K/V injection (negative pos) | History → suffix prompt (positive pos)
    ↓
Call LLM inference → Return response
```

### What DKI Is NOT

-   **Not RAG**: DKI uses K/V injection, not prompt concatenation, doesn't consume token budget
-   **Not Knowledge Base Retrieval**: DKI focuses on user-level memory, use RAG for external knowledge
-   **No Interface Implementation Required**: Configuration-driven, upstream apps only pass user_id and raw input

### Why This Scope Matters

This focused scope enables:

1. **Short preferences** (50-200 tokens) → reduced position encoding risks, cacheable
2. **User-owned data** → simplified privacy considerations
3. **Session-coherent** → effective K/V caching
4. **Stable preferences** → high cache reuse rate

### Key Features

-   **🧠 Attention Hook Injection**: Injects K/V at attention level via PyTorch Hooks, not prompt tokens
-   **🔀 Hybrid Injection Strategy**: Preferences (K/V negative position) + History (suffix prompt positive position)
-   **🔧 Configuration-Driven Adapter**: SQLAlchemy dynamic table mapping, no interface implementation required
-   **🎚️ Memory Influence Scaling (MIS)**: Continuous α ∈ [0, 1] control
-   **🔄 Query-Conditioned Projection**: FiLM-style memory-centric transformation
-   **🚦 Dual-Factor Gating**: Relevance-driven decision, entropy-modulated strength
-   **💾 Tiered KV Cache**: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
-   **📊 Monitoring API**: Statistics, injection logs, health checks
-   **🔌 Multi-Engine Support**: vLLM, LLaMA, DeepSeek, GLM
-   **✅ Graceful Degradation**: α → 0 smoothly recovers vanilla LLM behavior

## 🏗️ Architecture

### Core Architecture: LLM Plugin Mode

DKI operates as an **attention-level plugin** for LLMs, implementing K/V injection via PyTorch Hook mechanism:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DKI Plugin Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Upstream Application (Chat UI / Customer Service / Other Apps) │    │
│  │  └── Only needs to pass: user_id + raw user input               │    │
│  │     (No RAG, No Prompt Engineering, No Interface Implementation)│    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DKI Plugin                                                     │    │
│  │  ├── Config-Driven Adapter (SQLAlchemy dynamic table mapping)   │    │
│  │  │   └── Reads upstream app database (preferences + history)    │    │
│  │  ├── Preference Processing → K/V Injection (negative pos, Hook) │    │
│  │  ├── History Processing → Suffix Prompt (positive pos)          │    │
│  │  └── Monitoring API (stats/logs/health)                         │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LLM Engine (vLLM / LLaMA / DeepSeek / GLM)                     │    │
│  │  └── Inference with K/V Injection                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Injection Strategy

DKI uses a **layered injection approach** that mirrors human cognition:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DKI Hybrid Injection Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 1: User Preferences (K/V Injection - Attention Hook)     │    │
│  │  ├── Content: Dietary, style, interests                         │    │
│  │  ├── Position: Negative (conceptually "before" user input)      │    │
│  │  ├── Mechanism: PyTorch Hook modifies Attention K/V             │    │
│  │  ├── Influence: Implicit, background (like personality)         │    │
│  │  └── α: 0.3-0.5 (lower, for subtle influence)                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 2: Session History (Suffix Prompt)                       │    │
│  │  ├── Content: Recent conversation turns                         │    │
│  │  ├── Position: After user query (positive positions)            │    │
│  │  ├── Mechanism: Standard token concatenation                    │    │
│  │  ├── Influence: Explicit, citable (like memory)                 │    │
│  │  └── Prompt: Trust-establishing guidance                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 3: Current Query (Standard Input)                        │    │
│  │  └── Primary focus of attention                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DKI Data Flow                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  User Query + user_id + session_id                                      │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  1. Config-Driven Adapter reads upstream app database           │    │
│  │     ├── Preferences table → Preference list                     │    │
│  │     └── Messages table → Relevant history (vector/BM25/keyword) │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  2. Preference Processing                                       │    │
│  │     ├── Format preference text                                  │    │
│  │     ├── Compute/cache K/V representation                        │    │
│  │     └── Prepare Attention Hook                                  │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  3. History Processing                                          │    │
│  │     └── Format as suffix prompt (with trust guidance)           │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  4. LLM Inference (with K/V Injection)                          │    │
│  │     ├── Attention Hook injects preference K/V (negative pos)    │    │
│  │     └── Input = query + history suffix (positive pos)           │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  5. Return response + Record monitoring data                    │    │
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

### Upstream Application Integration (Recommended)

DKI as an LLM plugin, upstream applications only need to:

1. **Provide adapter config file** - Specify database connection and field mapping
2. **Remove RAG/Prompt engineering code** - DKI handles it automatically
3. **Pass user_id + raw input** - No prompt construction needed

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

# 1. Initialize LLM adapter
model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")

# 2. Create DKI plugin from config (recommended)
# Upstream apps only need to provide config, no interface implementation
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",  # DB connection + field mapping
)

# 3. Call DKI - only pass user_id and raw input
# DKI will automatically:
# - Read user preferences from upstream app DB → K/V injection
# - Retrieve relevant history messages → suffix prompt
response = await dki.chat(
    query="Recommend a restaurant for tonight",  # Raw input, no prompt construction
    user_id="user_001",   # User ID (DKI uses to read preferences and history)
    session_id="session_001",  # Session ID (DKI uses to read session history)
)

print(response.text)
# Output considers:
# - Vegetarian preference (implicit, from K/V injection)
# - Previous restaurant visit (explicit, from history suffix)
# - Beijing location (implicit, from K/V injection)

# Monitoring data
print(f"Injection enabled: {response.metadata.injection_enabled}")
print(f"Alpha: {response.metadata.alpha}")
print(f"Preference tokens: {response.metadata.preference_tokens}")
print(f"History tokens: {response.metadata.history_tokens}")
print(f"Cache hit: {response.metadata.preference_cache_hit}")
print(f"Latency: {response.metadata.latency_ms}ms")
```

### Adapter Configuration Example

Create `config/adapter_config.yaml` to specify how to connect to upstream app database:

```yaml
user_adapter:
    # Database connection (connects to upstream app's database)
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        port: 5432
        database: my_app_db # Upstream app's database
        username: user
        password: pass

    # Preferences table mapping (maps to upstream app's table structure)
    preferences:
        table: user_preferences # Upstream app's table name
        fields:
            user_id: user_id # Upstream app's field name
            preference_text: content
            preference_type: type
            priority: priority

    # Messages table mapping
    messages:
        table: chat_messages
        fields:
            message_id: id
            session_id: session_id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at

    # Vector search config (supports dynamic vector processing)
    vector_search:
        type: dynamic # pgvector | faiss | dynamic
        dynamic:
            strategy: hybrid # lazy | batch | hybrid (BM25 + embedding)
```

### Example Chat UI

DKI provides a Vue3 + Element Plus example Chat UI to demonstrate DKI integration:

```bash
# Start both frontend and backend
python start_dev.py

# Start backend only
python start_dev.py backend

# Start frontend only
python start_dev.py frontend
```

**UI Features**:

-   🔐 User login/registration
-   💬 Chat interface with Markdown rendering
-   ⚙️ User preference management (CRUD)
-   📊 Session history management
-   📈 System statistics monitoring (requires admin password)
-   🎨 Light/dark theme toggle

**Note**: Chat UI is an **example application** to demonstrate DKI integration. DKI's adapter reads the Chat UI's database to get user preferences and history messages.

### Monitoring API

DKI provides monitoring API to view internal working status:

```python
# Get statistics
stats = dki.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Injection rate: {stats['injection_rate']:.2%}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg latency: {stats['avg_latency_ms']:.1f}ms")

# Get injection logs
logs = dki.get_injection_logs(limit=10)
for log in logs:
    print(f"[{log['timestamp']}] alpha={log['alpha']:.2f}, enabled={log['injection_enabled']}")
```

REST API endpoints:

-   `GET /v1/dki/info` - Get DKI plugin status
-   `POST /v1/dki/chat` - DKI enhanced chat (upstream apps call this endpoint)

## 📁 Project Structure

```
DKI/
├── config/
│   ├── config.yaml              # Main configuration
│   └── adapter_config.example.yaml  # Adapter config example
├── dki/
│   ├── core/
│   │   ├── dki_plugin.py        # ⭐ DKI Plugin Core
│   │   ├── dki_system.py        # DKI System wrapper
│   │   ├── memory_router.py     # FAISS-based retrieval
│   │   ├── embedding_service.py
│   │   └── components/
│   │       ├── memory_influence_scaling.py   # MIS component
│   │       ├── query_conditioned_projection.py  # QCP component
│   │       ├── dual_factor_gating.py         # Dual-factor gating
│   │       ├── hybrid_injector.py            # Hybrid injector
│   │       ├── session_kv_cache.py
│   │       ├── tiered_kv_cache.py    # L1/L2/L3/L4 memory hierarchy
│   │       └── position_remapper.py  # Position encoding remapping
│   ├── adapters/
│   │   ├── base.py              # Adapter base class
│   │   └── config_driven_adapter.py  # ⭐ Config-driven adapter
│   ├── api/
│   │   └── dki_routes.py        # ⭐ DKI API routes
│   ├── models/
│   │   ├── factory.py           # Model factory
│   │   ├── base.py              # Base adapter
│   │   ├── vllm_adapter.py
│   │   ├── llama_adapter.py
│   │   ├── deepseek_adapter.py
│   │   └── glm_adapter.py
│   ├── cache/
│   │   └── non_vectorized_handler.py  # Dynamic vector processing
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── connection.py        # DB connection manager
│   ├── experiment/
│   │   ├── runner.py            # Experiment runner
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── data_generator.py    # Test data generation
│   └── web/
│       └── app.py               # FastAPI application
├── ui/                          # Vue3 Example Frontend UI
│   ├── src/
│   │   ├── views/               # Page components
│   │   ├── components/          # Common components
│   │   ├── stores/              # Pinia state management
│   │   ├── services/            # API services
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── Integration_Guide.md     # Integration guide
│   └── Dynamic_Vector_Search.md # Dynamic vector search docs
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── scripts/
│   ├── setup.bat/.sh            # Setup scripts
│   └── start.bat/.sh            # Start scripts
├── start_dev.py                 # Development startup script
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### DKI Main Configuration

Edit `config/config.yaml`:

```yaml
# Model Engine
model:
    default_engine: "vllm" # vllm, llama, deepseek, glm
    engines:
        vllm:
            model_name: "Qwen/Qwen2-7B-Instruct"
            tensor_parallel_size: 1

# DKI Plugin Settings
dki:
    enabled: true
    version: "2.5"

    # Hybrid Injection Strategy
    hybrid_injection:
        enabled: true
        language: "cn" # en | cn

        # Preferences: K/V injection (Attention Hook, negative position)
        preference:
            enabled: true
            position_strategy: "negative"
            alpha: 0.4 # Lower for background influence
            max_tokens: 200

        # History: Suffix prompt (positive position)
        history:
            enabled: true
            method: "suffix_prompt"
            max_tokens: 2000
            max_messages: 10

    # Gating: Relevance-driven, entropy-modulated
    gating:
        relevance_threshold: 0.7
        entropy_ceiling: 1.0
        entropy_floor: 0.5

    # Safety
    safety:
        max_alpha: 0.8
        fallback_on_error: true
        audit_logging: true
```

### Adapter Configuration (Core)

Create `config/adapter_config.yaml` to configure how to connect to upstream app database:

```yaml
user_adapter:
    # Database connection - connects to upstream app's database
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        port: 5432
        database: my_app_db # Upstream app's database name
        username: user
        password: pass
        pool_size: 5

    # Preferences table mapping - maps to upstream app's table structure
    preferences:
        table: user_preferences # Upstream app's preference table name
        fields:
            user_id: user_id # Field mapping: DKI field -> upstream app field
            preference_text: content
            preference_type: type
            priority: priority
            created_at: created_at
        filters:
            is_active: true # Additional filter conditions

    # Messages table mapping
    messages:
        table: chat_messages # Upstream app's message table name
        fields:
            message_id: id
            session_id: session_id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at
            embedding: embedding # Vector field (optional)

    # Vector search configuration
    vector_search:
        enabled: true
        type: dynamic # pgvector | faiss | dynamic | none

        # Dynamic vector processing (used when no pre-computed vectors)
        dynamic:
            strategy: hybrid # lazy | batch | hybrid
            # hybrid = BM25 initial filtering + embedding reranking

        # Retrieval parameters
        top_k: 10
        similarity_threshold: 0.5

    # Cache configuration
    cache_enabled: true
    cache_ttl: 300 # 5 minutes
```

### Configuration Notes

**Core philosophy of adapter configuration**:

-   Upstream apps **don't need to implement any interface**
-   Just provide config file specifying database connection and field mapping
-   DKI uses SQLAlchemy to dynamically reflect table structure
-   Supports PostgreSQL, MySQL, SQLite and other mainstream databases

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

### DKI Plugin API

| Endpoint       | Method | Description                                 |
| -------------- | ------ | ------------------------------------------- |
| `/v1/dki/chat` | POST   | DKI enhanced chat (upstream apps call this) |
| `/v1/dki/info` | GET    | Get DKI plugin status                       |

### Monitoring API

| Endpoint         | Method | Description           |
| ---------------- | ------ | --------------------- |
| `/api/stats`     | GET    | Get system statistics |
| `/api/stats/dki` | GET    | Get DKI statistics    |
| `/api/health`    | GET    | Health check          |

### DKI Chat Request

Upstream apps only need to pass `user_id` and raw input:

```json
{
    "query": "Recommend a restaurant",
    "user_id": "user_001",
    "session_id": "session_001",
    "temperature": 0.7,
    "max_tokens": 512
}
```

### DKI Chat Response

```json
{
    "id": "dki-abc12345",
    "text": "Based on your preference for vegetarian food...",
    "input_tokens": 128,
    "output_tokens": 256,
    "dki_metadata": {
        "injection_enabled": true,
        "alpha": 0.4,
        "preference_tokens": 85,
        "history_tokens": 320,
        "cache_hit": true,
        "cache_tier": "memory",
        "latency_ms": 156.3
    },
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Based on your preference..."
            },
            "finish_reason": "stop"
        }
    ],
    "created": 1707523200
}
```

## 🔬 Research Background

### DKI's Positioning: User-Level Memory System

Unlike RAG which targets **external knowledge** (documents, databases, web content), DKI is designed specifically for **user-level memory**:

| Dimension            | RAG                            | DKI                                                     |
| -------------------- | ------------------------------ | ------------------------------------------------------- |
| **Target Data**      | External knowledge bases       | User preferences, session history                       |
| **Data Size**        | Large (thousands of documents) | Small to Medium (prefs 50-200, history 100-4000 tokens) |
| **Update Frequency** | Batch updates                  | Real-time per session                                   |
| **Privacy**          | Shared knowledge               | User-owned data                                         |
| **Caching**          | Document-level                 | User-level (high reuse)                                 |

> **Note**: DKI token count depends on session complexity and relevance. Preference injection should stay short (50-200 tokens), while history injection can extend to 4000+ tokens as needed. For long histories, enable relevance filtering to optimize performance.

This focused scope is **intentional** and enables DKI's key advantages.

### Token Budget Analysis

DKI addresses a fundamental limitation of RAG: retrieved content consumes context window capacity.

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

### Hybrid Injection Rationale

Why use different strategies for preferences vs history?

| Memory Type     | Characteristics                               | Strategy                          | Reason                                      |
| --------------- | --------------------------------------------- | --------------------------------- | ------------------------------------------- |
| **Preferences** | Short (50-200 tokens), stable, abstract       | K/V injection (negative position) | Low OOD risk, cacheable, implicit influence |
| **History**     | Variable (100-4000 tokens), dynamic, concrete | Suffix prompt (positive position) | Zero OOD risk, citable, explicit reference  |

This layered approach:

-   Minimizes OOD risk (preferences are short)
-   Enables history citation (visible in prompt)
-   Reduces hallucination (trust-establishing prompts)

### Token Count and Performance Impact

DKI's supported token range depends on session complexity and relevance:

| Token Range | Use Case                              | Latency Impact | VRAM (7B model) |
| ----------- | ------------------------------------- | -------------- | --------------- |
| 100-500     | Simple prefs + short history          | < 10%          | ~250MB          |
| 500-2000    | Medium complexity sessions            | 10-30%         | ~1GB            |
| 2000-4000   | Complex multi-turn sessions           | 30-50%         | ~2GB            |
| 4000+       | Long sessions (filtering recommended) | > 50%          | > 2GB           |

**Performance Optimization Tips**:

-   For long histories, enable `search_relevant_history` for relevance filtering
-   Keep preferences short (50-200 tokens) to leverage K/V caching
-   History uses suffix prompts, length can be dynamically adjusted

**VRAM Comparison with RAG**:

-   For the same token count, DKI and RAG have **similar VRAM usage**
-   DKI may use more VRAM due to caching multi-turn history
-   However, DKI's K/V cache is reusable, no recomputation for subsequent requests

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

1. **Personalized Assistants**

    - User preferences need to persist across sessions
    - Implicit personalization (no explicit mention in prompt)
    - Multi-turn dialogue with context continuity

2. **Customer Service Systems**

    - User profile + conversation history
    - Consistent experience across sessions
    - Privacy-sensitive user data

3. **Educational Applications**

    - Learning preferences + progress history
    - Adaptive responses based on user level
    - Long-term user modeling

4. **Health/Wellness Assistants**
    - Health profile + consultation history
    - Sensitive personal data
    - Consistent medical context

#### ⚠️ Use RAG Instead

1. **External Knowledge Retrieval**

    - Document search and QA
    - Public knowledge bases
    - Frequently updated content

2. **First-turn Latency Critical**

    - First interaction must be fastest
    - Cannot accept gating overhead
    - Cold start performance critical

3. **High Audit Requirements**

    - Need to display full retrieved content
    - Regulators require visible prompts
    - Attention visualization insufficient

4. **Rapid Prototyping**
    - Using mature RAG ecosystem (LangChain, LlamaIndex)
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
A: No. DKI is an inference-time enhancement using frozen model parameters, injecting K/V via Attention Hooks.

**Q: What's the difference between DKI and RAG?**  
A:

-   **RAG**: Concatenates retrieved content at token level, consumes context window
-   **DKI**: Injects K/V at attention level, doesn't consume token budget
-   They are complementary: RAG handles external knowledge, DKI handles user-level memory

**Q: What changes does the upstream app need to make?**  
A:

1. Provide adapter config file (specify database connection and field mapping)
2. Remove RAG/Prompt engineering code
3. Pass `user_id` and raw input when calling DKI API

**Q: What if the upstream app's database doesn't have vector indexes?**  
A: DKI supports dynamic vector processing, just configure `vector_search.type: dynamic`. Three strategies supported:

-   `lazy`: Compute embedding on-demand
-   `batch`: Batch pre-computation
-   `hybrid`: BM25 initial filtering + embedding reranking (recommended)

**Q: What's the difference between preference and history injection?**  
A:

-   **Preferences**: K/V injection at negative positions via Attention Hook, implicit influence, cached
-   **History**: Suffix prompt at positive positions, standard token concatenation, explicit reference, dynamic

**Q: How to integrate DKI into existing systems?**  
A:

```python
from dki.core.dki_plugin import DKIPlugin

# Create from config file (recommended)
dki = await DKIPlugin.from_config(
    model_adapter=your_model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# Only pass user_id and raw input when calling
response = await dki.chat(
    query="User's raw input",
    user_id="user_123",
    session_id="session_456",
)
```

**Q: Does DKI need to consider distributed deployment?**  
A: No. DKI as an LLM plugin only reads user config and message data to complete injection. Distributed deployment is the responsibility of the LLM engine and upstream application. DKI itself is stateless (except for preference K/V cache) and can scale horizontally with LLM instances.

**Q: Production deployment recommendations?**  
A:

1. Enable hybrid injection
2. Set preference α to 0.4 (conservative)
3. Configure adapter to connect to upstream app's database
4. Monitor injection rate and latency
5. Adjust alpha and cache strategy based on metrics

### Latest Optimizations (v2.5)

#### Memory Trigger

Detects memory-related signals in user input, deciding "what to remember":

```python
# Supports 5 trigger types
- META_COGNITIVE: "what we just discussed", "you said earlier"
- STATE_CHANGE: "I changed my mind", "let me add"
- LONG_TERM_VALUE: "please remember I like...", "I'm vegetarian"
- RECALL_REQUEST: "what did we discuss recently"
- OPINION_QUERY: "do you have new thoughts"

# Rules are configurable, can be enhanced with classifier later
trigger = MemoryTrigger(language="auto")
result = trigger.detect("What did we just talk about?")
```

#### Reference Resolver

Parses referential expressions in user input, determines history recall scope:

```python
# Recall turns are externally configurable
resolver = ReferenceResolver(config=ReferenceResolverConfig(
    last_few_turns=5,    # "just now" recalls 5 turns
    recent_turns=20,     # "recently" recalls 20 turns
))

# Supports runtime dynamic updates
dki.update_reference_resolver_config(
    just_now_turns=3,
    recently_turns=15,
)
```

#### Why Rolling Summary Is Not Needed

Unlike ChatGPT/Claude/Grok, DKI **does not need** Rolling Summary:

| Approach           | Reason                                  | DKI Alternative                            |
| ------------------ | --------------------------------------- | ------------------------------------------ |
| RAG+Prompt         | Context window limit, needs compression | K/V injection doesn't consume context      |
| Rolling Summary    | Information loss from compression       | Memory Trigger for precise recall          |
| Summary Generation | Extra LLM call overhead                 | Reference Resolver for on-demand retrieval |

### Roadmap

**Completed**:

-   [x] Core DKI implementation (Attention Hook K/V injection)
-   [x] vLLM/LLaMA/DeepSeek/GLM adapters
-   [x] Hybrid injection strategy (preference K/V + history suffix)
-   [x] Config-driven adapter (SQLAlchemy dynamic table mapping)
-   [x] Dynamic vector processing (BM25 + Embedding hybrid search)
-   [x] Preference K/V cache (memory level)
-   [x] Monitoring API (stats/logs/health)
-   [x] Vue3 Example Frontend UI
-   [x] Experiment framework
-   [x] Memory Trigger
-   [x] Reference Resolver (configurable recall turns)

**In Progress**:

-   [ ] Stance State Machine
-   [ ] Classifier-enhanced Memory Trigger

**Future Work**:

-   [ ] Redis distributed cache integration
-   [ ] Attention visualization tools
-   [ ] Multi-modal extension (image/audio memory)
-   [ ] LangChain/LlamaIndex integration

---

## 🔮 Future Work Directions

### 1. Redis Distributed Cache Integration ⭐ Recommended Priority

**Current State**: Preference K/V cache is memory-only, effective for single instance.

**Optimization Goal**: Integrate Redis for cross-instance shared caching.

**Why Redis Integration Is the Most Important Optimization**:

One of DKI's core advantages is **preference K/V cache reuse**—after computing K/V on the first turn, subsequent requests use the cache directly, reducing latency by 43.7%. However, the current memory cache has a critical limitation: **only effective for single instance**.

In production environments, LLM services are typically multi-instance deployments (load balanced). If user requests are routed to different instances, cache misses occur, and DKI's core advantage is significantly diminished:

| Deployment Mode | Cache Hit Rate | DKI Advantage      |
| --------------- | -------------- | ------------------ |
| Single Instance | ~70%           | Full benefit       |
| 2 Instances     | ~35%           | Halved             |
| 4 Instances     | ~17.5%         | Greatly reduced    |
| N Instances     | ~70%/N         | Nearly ineffective |

**After Redis Integration**: Regardless of instance count, cache hit rate remains ~70%, DKI advantage fully preserved.

**Core Value**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Current Architecture (Single-Instance Cache)         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LLM Instance 1          LLM Instance 2          LLM Instance 3         │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐        │
│  │ DKI Plugin  │         │ DKI Plugin  │         │ DKI Plugin  │        │
│  │ ┌─────────┐ │         │ ┌─────────┐ │         │ ┌─────────┐ │        │
│  │ │ Memory  │ │         │ │ Memory  │ │         │ │ Memory  │ │        │
│  │ │ user_001│ │         │ │ user_002│ │         │ │ user_003│ │        │
│  │ └─────────┘ │         │ └─────────┘ │         │ └─────────┘ │        │
│  └─────────────┘         └─────────────┘         └─────────────┘        │
│                                                                         │
│  Problems:                                                              │
│  - user_001 request to Instance 2 = cache miss, K/V recomputation       │
│  - Cache hit rate decreases with more instances                         │
│  - Cannot achieve true horizontal scaling                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Optimized Architecture (Redis Distributed Cache)     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LLM Instance 1          LLM Instance 2          LLM Instance 3         │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐        │
│  │ DKI Plugin  │         │ DKI Plugin  │         │ DKI Plugin  │        │
│  │ ┌─────────┐ │         │ ┌─────────┐ │         │ ┌─────────┐ │        │
│  │ │ L1 Mem  │ │         │ │ L1 Mem  │ │         │ │ L1 Mem  │ │        │
│  │ │ (Hot)   │ │         │ │ (Hot)   │ │         │ │ (Hot)   │ │        │
│  │ └────┬────┘ │         │ └────┬────┘ │         │ └────┬────┘ │        │
│  └──────┼──────┘         └──────┼──────┘         └──────┼──────┘        │
│         │                       │                       │               │
│         └───────────────────────┼───────────────────────┘               │
│                                 ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │      Redis Cluster      │                          │
│                    │  ┌────────────────────┐ │                          │
│                    │  │L2 Distributed Cache│ │                          │
│                    │  │user_001, user_002  │ │                          │
│                    │  │user_003, ...       │ │                          │
│                    │  └────────────────────┘ │                          │
│                    └─────────────────────────┘                          │
│                                                                         │
│  Benefits:                                                              │
│  - Any instance can hit cache                                           │
│  - Cache hit rate unaffected by instance count                          │
│  - True horizontal scaling support                                      │
│  - Cache persistence, survives restarts                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation Plan**:

```python
# Configuration example
cache:
  type: tiered  # memory | redis | tiered
  tiered:
    l1:
      type: memory
      max_size_mb: 512
      ttl: 300  # 5 minutes
    l2:
      type: redis
      host: redis-cluster.example.com
      port: 6379
      password: ${REDIS_PASSWORD}
      db: 0
      ttl: 3600  # 1 hour
      key_prefix: "dki:kv:"
      serialization: msgpack  # compressed serialization
```

**Feasibility Assessment**:

| Dimension            | Rating      | Notes                                                                      |
| -------------------- | ----------- | -------------------------------------------------------------------------- |
| Technical Complexity | ⭐⭐ Medium | Redis client mature, main work is K/V Tensor serialization                 |
| Performance Impact   | ⭐⭐ Medium | Network latency ~1-5ms, but avoids K/V recomputation (~50-200ms)           |
| Benefit              | ⭐⭐⭐ High | Essential for multi-instance deployment, significant cache hit improvement |
| Dependencies         | ⭐ Low      | Only redis-py, optional dependency                                         |

**Key Technical Points**:

1. **K/V Tensor Serialization**: Use `msgpack` + `numpy` compression to reduce network transfer
2. **Cache Invalidation Strategy**: Active invalidation on preference change, TTL as fallback
3. **Hot Data Local Cache**: L1 memory cache for high-frequency users, reduce Redis access

### 2. Attention Visualization Tools

**Goal**: Debug K/V injection effects, understand injection impact on attention distribution.

**Design Plan**:

```python
# Visualization API
from dki.visualization import AttentionVisualizer

visualizer = AttentionVisualizer(dki_plugin)

# Generate attention heatmap
heatmap = visualizer.generate_heatmap(
    query="Recommend a restaurant",
    user_id="user_001",
    layer_indices=[0, 12, 24],  # Visualize specific layers
)

# Compare before/after injection
comparison = visualizer.compare_injection(
    query="Recommend a restaurant",
    user_id="user_001",
    show_diff=True,
)

# Export as HTML report
visualizer.export_report("attention_analysis.html")
```

**Feasibility Assessment**:

| Dimension            | Rating      | Notes                                                     |
| -------------------- | ----------- | --------------------------------------------------------- |
| Technical Complexity | ⭐⭐ Medium | Need to hook attention weights, visualization libs mature |
| Performance Impact   | ⭐⭐⭐ High | Only enabled during debugging, disabled in production     |
| Benefit              | ⭐⭐ Medium | Valuable for debugging and paper presentation             |
| Dependencies         | ⭐ Low      | matplotlib, plotly as optional dependencies               |

### 3. Multi-Modal Extension (Image/Audio Memory)

**Goal**: Support images and audio as part of user memory.

**Design Plan**:

```python
# Multi-modal preferences
preferences = [
    {"type": "text", "content": "likes minimalist style"},
    {"type": "image", "content": "user_avatar.jpg", "embedding": [...]},
    {"type": "audio", "content": "voice_sample.wav", "embedding": [...]},
]

# Multi-modal K/V injection
# Images/audio first converted to embedding via encoder, then K/V computed
```

**Feasibility Assessment**:

| Dimension            | Rating      | Notes                                                         |
| -------------------- | ----------- | ------------------------------------------------------------- |
| Technical Complexity | ⭐⭐⭐ High | Needs multi-modal encoders, K/V computation method adjustment |
| Performance Impact   | ⭐⭐⭐ High | Image/audio encoding overhead is large                        |
| Benefit              | ⭐⭐ Medium | Valuable for specific scenarios (e.g., virtual assistants)    |
| Dependencies         | ⭐⭐⭐ High | Requires CLIP, Whisper, etc.                                  |

**Recommendation**: Long-term goal, lower priority.

### 4. LangChain/LlamaIndex Integration

**Goal**: Package DKI as LangChain/LlamaIndex module to expand ecosystem.

**Design Plan**:

```python
# LangChain integration
from langchain_dki import DKIMemory

memory = DKIMemory(
    adapter_config_path="config/adapter_config.yaml",
)

chain = ConversationChain(
    llm=llm,
    memory=memory,  # DKI as Memory module
)

# LlamaIndex integration
from llama_index_dki import DKIRetriever

retriever = DKIRetriever(
    adapter_config_path="config/adapter_config.yaml",
)

query_engine = index.as_query_engine(
    retriever=retriever,  # DKI as Retriever
)
```

**Feasibility Assessment**:

| Dimension            | Rating      | Notes                                           |
| -------------------- | ----------- | ----------------------------------------------- |
| Technical Complexity | ⭐⭐ Medium | Need to adapt LangChain/LlamaIndex interfaces   |
| Performance Impact   | ⭐ Low      | Wrapper layer only, no extra overhead           |
| Benefit              | ⭐⭐⭐ High | Expand user base, lower adoption barrier        |
| Dependencies         | ⭐⭐ Medium | langchain, llama-index as optional dependencies |

**Recommendation**: Medium priority, implement after core features stabilize.

### Priority Ranking

| Priority | Optimization Direction           | Reason                                      |
| -------- | -------------------------------- | ------------------------------------------- |
| P1       | Redis Distributed Cache          | Essential for multi-instance, clear benefit |
| P2       | Attention Visualization          | Valuable for debugging and papers           |
| P3       | LangChain/LlamaIndex Integration | Expand ecosystem, but not core              |
| P4       | Multi-Modal Extension            | High complexity, specific scenarios         |

### Additional Value of Redis Integration

Beyond solving multi-instance caching, Redis integration brings additional value:

1. **Cache Persistence**

    - Cache survives service restarts
    - Reduces K/V recomputation on cold start

2. **Cache Warming**

    - Pre-compute K/V for high-frequency users
    - Batch import historical user preferences

3. **Cache Monitoring**

    - Redis provides rich monitoring metrics
    - Analyze cache hit rate, memory usage, etc.

4. **Cache Eviction Strategies**

    - Mature LRU/LFU strategies
    - Automatic cache capacity management

5. **Cross-Service Sharing**
    - Multiple DKI instances share the same cache
    - Can even share across different LLM services (if using same model)

**Cost Analysis**:

| Resource        | Estimate         | Notes                       |
| --------------- | ---------------- | --------------------------- |
| Redis Memory    | ~100MB/10K users | Assuming ~10KB K/V per user |
| Network Latency | 1-5ms            | Within LAN                  |
| Ops Cost        | Low              | Redis ops is mature         |

**ROI Analysis**:

```
Assumptions:
- 4 instance deployment
- Current cache hit rate ~17.5% (70%/4)
- Post-Redis cache hit rate ~70%

Benefits:
- Cache hit rate improves 4x
- Subsequent turn latency reduces ~40%
- Overall throughput improves ~30%

Costs:
- Redis instance ~$50/month (cloud service)
- Development effort ~2-3 person-days
```

### Example Frontend UI

DKI provides a **Vue3** based example frontend UI to demonstrate DKI integration:

-   **Vue 3 + TypeScript**: Type-safe modern frontend development
-   **Vite**: Fast development server and build tool
-   **Pinia**: Vue3 official state management
-   **Element Plus**: Enterprise-grade UI component library

UI Features:

-   Chat interface (with DKI metadata badges)
-   User preference management panel
-   Session history browser
-   System statistics dashboard (requires password)

**Note**: Chat UI is an example application, DKI's adapter reads its database to get user preferences and history messages.

### Acknowledgments

This project is inspired by the following research:

-   RAG ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401))
-   RETRO ([Borgeaud et al., 2022](https://arxiv.org/abs/2112.04426))
-   Self-RAG ([Asai et al., 2023](https://arxiv.org/abs/2310.11511))
-   FiLM ([Perez et al., 2018](https://arxiv.org/abs/1709.07871))
-   FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))

## 📄 Related Papers

This project is based on the paper "Dynamic KV Injection: An Attention-Level User Memory System for Large Language Models".

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

---

**DKI** - Rethinking Memory Augmentation at the Attention Level
