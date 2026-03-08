# DKI - Dynamic KV Injection

> Attention-Level User Memory Plugin for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-4.0.0-green.svg)]()

[简体中文](README_CN.md) | English

## 📖 Overview

DKI (Dynamic KV Injection) is an **LLM attention-level plugin** that dynamically injects user preferences and session history via Attention Hooks during inference, enabling cross-session personalized memory.

### What DKI Is

DKI is an **LLM plugin** designed specifically for **user-level memory**:

-   **Attention Hook Mechanism**: Injects K/V at the attention level via PyTorch Hooks, not prompt concatenation
-   **Configuration-Driven Adapter**: Automatically reads from upstream application databases, no code changes required
-   **Hybrid Injection Strategy**: Preference K/V injection (negative position) + History suffix prompt (positive position)
-   **Minimal Integration** (v4.0): 3-line integration, dynamic routing, message management

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
-   **🔀 Recall v4 Memory Recall**: Multi-signal retrieval + dynamic summary + fact supplementation (primary), stable hybrid injection as automatic fallback
-   **🔧 Configuration-Driven Adapter**: SQLAlchemy dynamic table mapping, no interface implementation required
-   **🔐 User-Level Isolation**: HMAC-signed cache keys + UserIsolationContext + post-inference K/V cleanup
-   **🎚️ Memory Influence Scaling (MIS)**: Continuous α ∈ [0, 1] control
-   **🔄 Query-Conditioned Projection**: FiLM-style memory-centric transformation
-   **🚦 Dual-Factor Gating**: Relevance-driven decision, entropy-modulated strength
-   **💾 Tiered KV Cache**: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
-   **📊 Monitoring API**: Statistics, injection logs, health checks
-   **🔌 Multi-Engine Support**: vLLM, SGLang, LLaMA, DeepSeek, GLM (all with streaming)
-   **✅ Graceful Degradation**: recall_v4 → stable → plain LLM, three-tier fallback
-   **🚀 Minimal Integration (v4.0)**: 3-line integration, FastAPI Middleware, dynamic routing, message management
-   **🔀 Dynamic Routing (v4.0)**: Auto-switch between RAG and DKI, five-dimension scoring model
-   **📝 Message Management (v4.0)**: DKI handles message and preference persistence internally
-   **⚡ Streaming Generation (v4.0)**: All model adapters support async streaming output

## 🏗️ Architecture

### v4.0 Integration Architecture

DKI v4.0 provides three integration levels, from simple to advanced, inspired by the AGA plugin architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DKI v4.0 Integration Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: One-Line Factory (Minimal Integration, Recommended)           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  from dki.integration import create_plugin                      │    │
│  │  dki = await create_plugin(adapter_config_path="config/a.yaml") │    │
│  │  response = await dki.chat("Recommend a restaurant", ...)       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Level 2: FastAPI Middleware (Web App Integration)                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  from dki.integration import DKIMiddleware                      │    │
│  │  app.add_middleware(DKIMiddleware, adapter_config_path="...")   │    │
│  │  # Auto-init, lifecycle management, dependency injection        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Level 3: EnhancedDKIPlugin (Full Control)                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  EnhancedDKIPlugin(dki_plugin, rag_system, store, config)       │    │
│  │  ├── dynamic_router: RAG ↔ DKI auto-switching                   │    │
│  │  ├── message_management: message + preference persistence       │    │
│  │  └── unified lifecycle management                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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
│  │  DKI Integration Layer (v4.0)                                   │    │
│  │  ├── create_plugin() — one-line creation                        │    │
│  │  ├── DKIMiddleware — FastAPI auto-integration                   │    │
│  │  ├── EnhancedDKIPlugin — dynamic routing + message management   │    │
│  │  │   ├── ConversationRouter — RAG/DKI 5-dimension scoring       │    │
│  │  │   └── MessageManagement — auto message/preference write      │    │
│  │  └── DKIPlugin — core plugin                                    │    │
│  │      ├── Config-Driven Adapter (SQLAlchemy dynamic table map)   │    │
│  │      ├── Preference → K/V Injection (negative pos, Hook)        │    │
│  │      ├── History → Suffix Prompt (positive pos)                 │    │
│  │      ├── Streaming Generation (chat_stream)                     │    │
│  │      └── Monitoring API (stats/logs/health)                     │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LLM Engine (vLLM / SGLang / LLaMA / DeepSeek / GLM)            │    │
│  │  └── Inference with K/V Injection (sync/async/streaming)        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dynamic Routing (v4.0)

DKI v4.0's `ConversationRouter` uses a five-dimension scoring model to auto-switch between RAG and DKI:

```
Score_DKI = w₁·S_history + w₂·S_preference + w₃·S_trigger
          + w₄·S_session_depth + w₅·S_cross_session

Route = DKI  if Score_DKI > θ_dki
      = RAG  if Score_DKI < θ_rag
      = DKI  otherwise (with reduced confidence)
```

| Dimension                | RAG Advantage      | DKI Advantage                     |
| ------------------------ | ------------------ | --------------------------------- |
| Session Length           | 1-3 turns (short)  | 5+ turns (long, cross-session)    |
| Memory Type              | External knowledge | User preferences + history        |
| Personalization          | None/weak          | Strong (preference K/V injection) |
| First Interaction        | ★ Strong           | Weak (no history to recall)       |
| Cross-Session Continuity | Weak               | ★ Strong (cross-session memory)   |

### Injection Strategy Selection

DKI v4.0 uses **recall_v4** as the primary strategy with **stable** as the fallback:

| Strategy                | Status        | Use Case          | Context Usage | Stability  |
| ----------------------- | ------------- | ----------------- | ------------- | ---------- |
| **recall_v4** (default) | ✅ Primary    | Long history      | Dynamic       | ⭐⭐⭐⭐⭐ |
| **stable** (fallback)   | ✅ Fallback   | recall_v4 failure | Medium        | ⭐⭐⭐⭐⭐ |
| **full_attention**      | ⚠️ Deprecated | Research only     | Minimal       | ⭐⭐⭐     |

**Fallback Mechanism**: When recall_v4 fails, the system automatically falls back to stable; if stable also fails, it degrades to plain LLM inference.

### Recall v4 Memory Recall Strategy (Recommended)

**Production-recommended strategy**. Simulates human memory recall through multi-signal retrieval, dynamic history construction, and application-layer fact supplementation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  DKI Recall v4 Memory Recall Architecture               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: Multi-Signal Recall                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  User Input → [Keywords+Weights] + [Anaphora] + [Vector Sim]    │    │
│  │            →  Weighted Merge + Normalization → Message List     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ↓                                              │
│  Phase 2: Dynamic History Construction                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Iterate messages:                                              │    │
│  │    Over threshold → [SUMMARY] + trace_id (traceable)            │    │
│  │    Under threshold → Original message                           │    │
│  │  + Recent N turns of complete conversation                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ↓                                              │
│  Phase 3: Model-Adaptive Assembly + Inference                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  [History Suffix] + [Trust+Reasoning Constraints] + [Pref K/V]  │    │
│  │  + [Query] → LLM Inference → Detect retrieve_fact call          │    │
│  │  → Fact supplementation (chunked offset+limit) → Continue       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

> 📖 **Complete Deployment Guide**: See [DKI+AGA Complete Deployment Guide](docs/DKI_AGA_Complete_Deployment_Guide.md)

### Installation

```bash
cd DKI

# Windows:
scripts\setup.bat

# Linux/Mac:
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Integration (v4.0 — Minimal Integration)

DKI v4.0 provides three integration levels:

#### Level 1: One-Line Factory (Recommended)

```python
from dki.integration import create_plugin

# 1. Create DKI plugin (auto-manages model, cache, adapter)
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")

# 2. Regular chat
response = await dki.chat(
    query="Recommend a restaurant for tonight",
    user_id="user_001",
    session_id="session_001",
)
print(response.text)

# 3. Streaming chat
async for chunk in dki.chat_stream(
    query="Recommend a good book",
    user_id="user_001",
    session_id="session_001",
):
    if chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\nDone! Latency: {chunk['latency_ms']:.1f}ms")
```

#### Level 2: FastAPI Middleware (Web App Integration)

```python
from fastapi import FastAPI
from dki.integration import DKIMiddleware, get_dki_plugin

app = FastAPI()

# One line to add DKI support (auto-init, lifecycle management)
app.add_middleware(
    DKIMiddleware,
    adapter_config_path="config/adapter_config.yaml",
)

@app.post("/chat")
async def chat(query: str, user_id: str, session_id: str):
    dki = await get_dki_plugin(request)
    response = await dki.chat(query=query, user_id=user_id, session_id=session_id)
    return {"text": response.text, "metadata": response.metadata}
```

#### Level 3: EnhancedDKIPlugin (Full Control + Dynamic Routing + Message Management)

```python
from dki.integration import create_plugin
from dki.integration.enhanced_plugin import (
    EnhancedDKIPluginConfig,
    DynamicRouterConfig,
    MessageManagementConfig,
)

# Enable dynamic routing + message management
dki = await create_plugin(
    adapter_config_path="config/adapter_config.yaml",
    enhanced_config=EnhancedDKIPluginConfig(
        dynamic_router=DynamicRouterConfig(
            enabled=True,
            dki_threshold=0.45,
            rag_threshold=0.25,
        ),
        message_management=MessageManagementConfig(
            enabled=True,
            auto_write_messages=True,
            auto_write_preferences=True,
        ),
    ),
    rag_system=your_rag_system,  # RAG system instance
    store=your_store,            # Data persistence instance
)

# Call chat — auto-routes to RAG or DKI
response = await dki.chat(
    query="Recommend a restaurant for tonight",
    user_id="user_001",
    session_id="session_001",
)
# Messages auto-persisted, no upstream app handling needed
```

#### Legacy Integration (Still Supported)

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)
response = await dki.chat(
    query="Recommend a restaurant",
    user_id="user_001",
    session_id="session_001",
)
```

### Adapter Configuration Example

Create `config/adapter_config.yaml` to specify how to connect to upstream app database:

```yaml
user_adapter:
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        port: 5432
        database: my_app_db
        username: user
        password: pass

    preferences:
        table: user_preferences
        fields:
            user_id: user_id
            preference_text: content
            preference_type: type
            priority: priority

    messages:
        table: chat_messages
        fields:
            message_id: id
            session_id: session_id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at
        content_json_key: null # Supports JSON content extraction

    vector_search:
        type: dynamic
        dynamic:
            strategy: hybrid # BM25 + embedding
```

### Example Chat UI

DKI provides a Vue3 + Element Plus example Chat UI:

```bash
# Start both frontend and backend
python start_dev.py

# Start backend only
python start_dev.py backend

# Start frontend only
python start_dev.py frontend
```

**UI Features**:

-   🔐 User login/registration/password management
-   💬 Chat interface with Markdown rendering
-   ⚡ Streaming/regular generation mode toggle
-   📍 Conversation anchor navigation (jump to specific turns)
-   ⬆️⬇️ Dynamic scroll buttons (top/bottom)
-   ⚙️ User preference management (CRUD)
-   📊 Session history management
-   📈 System statistics monitoring
-   🎨 Light/dark theme toggle

### Monitoring API

```python
# Get statistics
stats = dki.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Injection rate: {stats['injection_rate']:.2%}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

# Get enhanced stats (v4.0)
if hasattr(dki, 'get_stats'):
    enhanced_stats = dki.get_stats()
    print(f"DKI routes: {enhanced_stats.get('enhanced', {}).get('dki_routes', 0)}")
    print(f"RAG routes: {enhanced_stats.get('enhanced', {}).get('rag_routes', 0)}")
```

REST API Endpoints:

| Endpoint              | Method | Description                  |
| --------------------- | ------ | ---------------------------- |
| `/v1/dki/chat`        | POST   | DKI enhanced chat            |
| `/v1/dki/chat/stream` | POST   | DKI streaming chat (v4.0)    |
| `/v1/dki/info`        | GET    | Get DKI plugin status        |
| `/api/health`         | GET    | Health check                 |
| `/api/health/detail`  | GET    | Detailed health check (v4.0) |
| `/api/stats`          | GET    | Get system statistics        |

## 📁 Project Structure

```
DKI/
├── config/                              # Configuration files
│   ├── config.yaml                      # ⭐ Main configuration
│   ├── adapter_config.example.yaml      # ⭐ Adapter config example
│   ├── memory_trigger.yaml              # Memory Trigger config
│   └── reference_resolver.yaml          # Reference Resolver config
│
├── dki/                                 # Core code directory
│   ├── __init__.py                      # v4.0 entry (create_plugin, DKIMiddleware)
│   │
│   ├── integration/                     # ⭐ v4.0 Integration Layer (new)
│   │   ├── __init__.py                  # Integration entry
│   │   ├── factory.py                   # ⭐ create_plugin() factory
│   │   ├── middleware.py                # ⭐ DKIMiddleware (FastAPI)
│   │   └── enhanced_plugin.py           # ⭐ EnhancedDKIPlugin (routing+messaging)
│   │
│   ├── core/                            # ⭐ Core modules
│   │   ├── dki_plugin.py                # ⭐ DKI Plugin Core (entry point)
│   │   ├── dki_system.py                # DKI System wrapper (experimental)
│   │   ├── rag_system.py                # RAG System (async+streaming, v4.0)
│   │   ├── conversation_router.py       # ⭐ RAG/DKI dynamic routing (v4.0)
│   │   ├── plugin_manager.py            # Multi-instance management (v4.0)
│   │   ├── exceptions.py                # ⭐ Structured exception hierarchy (v4.0)
│   │   ├── rate_limiter.py              # Rate limiting/circuit breaking (v4.0)
│   │   ├── memory_router.py             # FAISS-based vector retrieval
│   │   ├── recall/                      # ⭐ Recall v4 memory recall
│   │   └── components/                  # Core algorithm components
│   │       ├── memory_influence_scaling.py    # MIS
│   │       ├── query_conditioned_projection.py  # QCP
│   │       ├── dual_factor_gating.py          # Dual-Factor Gating
│   │       ├── memory_trigger.py              # Memory trigger detection
│   │       ├── reference_resolver.py          # Reference resolver
│   │       ├── tiered_kv_cache.py             # L1/L2/L3/L4 tiered cache
│   │       └── ...
│   │
│   ├── adapters/                        # External data adapters
│   │   ├── config_driven_adapter.py     # ⭐ Config-driven adapter (core)
│   │   ├── postgresql_adapter.py        # PostgreSQL
│   │   ├── mysql_adapter.py             # MySQL
│   │   └── ...
│   │
│   ├── models/                          # LLM model adapters
│   │   ├── factory.py                   # Model factory (namespace support)
│   │   ├── base.py                      # Base adapter (sync/async/streaming)
│   │   ├── vllm_adapter.py              # vLLM adapter (streaming)
│   │   ├── sglang_adapter.py            # SGLang adapter (streaming)
│   │   ├── llama_adapter.py             # LLaMA adapter (streaming)
│   │   ├── deepseek_adapter.py          # DeepSeek adapter (streaming)
│   │   └── glm_adapter.py              # GLM adapter (streaming)
│   │
│   ├── cache/                           # Cache system
│   │   ├── preference_cache.py          # Preference cache (L1+L2)
│   │   ├── redis_client.py              # Redis distributed cache
│   │   └── non_vectorized_handler.py    # Dynamic vector processing
│   │
│   ├── config/                          # Configuration loading
│   │   └── config_loader.py             # YAML config loader (hot-reload)
│   │
│   ├── attention/                       # FlashAttention integration
│   └── experiment/                      # Experiment system
│
├── demo/                                # ⭐ Example application
│   ├── app.py                           # FastAPI app (uses integration layer)
│   ├── api/                             # API routes
│   │   ├── auth.py                      # Auth (login/register/password)
│   │   └── chat.py                      # Chat (regular + streaming)
│   └── store/                           # Data persistence
│       ├── base.py                      # IChatStore interface
│       └── base_impl.py                 # SQLite/PostgreSQL implementation
│
├── ui/                                  # Vue3 Example Frontend
│   └── src/
│       ├── views/
│       │   ├── ChatView.vue             # Chat (streaming+anchors+scroll)
│       │   ├── LoginView.vue            # Login/Register
│       │   └── ProfileView.vue          # User profile management
│       ├── components/
│       │   └── SettingsDialog.vue       # Settings (streaming toggle)
│       └── stores/
│           └── settings.ts              # Settings state (streamingEnabled)
│
├── tests/                               # Tests
│   └── unit/
│       ├── test_integration_layer.py    # Integration layer tests (v4.0)
│       ├── test_p0p1_exceptions.py      # Exception hierarchy tests
│       ├── test_p1_rate_limiter.py      # Rate limiting tests
│       ├── test_streaming_chat.py       # Streaming chat tests
│       ├── test_rag_system_v6.py        # RAG system tests
│       └── ...
│
├── scripts/                             # Scripts
│   ├── setup.bat / setup.sh             # Setup scripts
│   ├── init_db.sql                      # SQLite database init
│   ├── init_db_postgresql.sql           # PostgreSQL database init
│   └── start_vllm_with_tools.sh         # vLLM Function Calling startup
│
├── docs/                                # Documentation
├── examples/                            # Examples
│   └── vllm_function_calling_web_search.py  # vLLM native web search example
│
├── start_dev.py                         # Dev startup script
├── requirements.txt                     # Python dependencies
├── QUICKSTART.md                        # Quick start guide
├── README_CN.md                         # Chinese documentation
└── README.md                            # English documentation
```

## 📊 Project Status

| Module                         | Status     | Description                                          |
| ------------------------------ | ---------- | ---------------------------------------------------- |
| DKI Core Plugin                | ✅ Done    | K/V injection, hybrid strategy, gating               |
| v4.0 Integration Layer         | ✅ Done    | create_plugin, DKIMiddleware, EnhancedDKIPlugin      |
| Dynamic Routing                | ✅ Done    | RAG/DKI 5-dimension auto-switching                   |
| Message Management             | ✅ Done    | Message + preference persistence                     |
| Streaming Generation           | ✅ Done    | All model adapters support async streaming           |
| Structured Exceptions          | ✅ Done    | Three-tier degradation chain                         |
| Rate Limiting/Circuit Breaking | ✅ Done    | Externalized config, hot-reload                      |
| Recall v4 Memory Recall        | ✅ Done    | Multi-signal retrieval + dynamic summary + fact call |
| Config-Driven Adapter          | ✅ Done    | SQLAlchemy dynamic table mapping                     |
| Redis Distributed Cache        | ✅ Done    | L1+L2 cache                                          |
| FlashAttention                 | ✅ Done    | FA3/FA2 auto-detection                               |
| User Management                | ✅ Done    | Register/login/password change/recovery              |
| Vue3 Example UI                | ✅ Done    | Chat/streaming/anchors/preferences/stats             |
| Unit Tests                     | ✅ Done    | Core component test coverage                         |
| Attention Heatmap              | 🔄 Planned | Debug attention weight visualization                 |
| File Upload/Skills             | 🔄 Planned | Text file upload and skill support                   |
| LangChain/LlamaIndex           | 📋 TBD     | Ecosystem integration                                |

## ⚙️ Configuration

### DKI Main Configuration

Edit `config/config.yaml`:

```yaml
# Model Engine
model:
    default_engine: "vllm" # vllm, sglang, llama, deepseek, glm
    engines:
        vllm:
            model_name: "Qwen/Qwen2-7B-Instruct"
            tensor_parallel_size: 1

# DKI Plugin Settings
dki:
    enabled: true
    version: "4.0"
    injection_strategy: "recall_v4"

    recall:
        enabled: true
        strategy: "summary_with_fact_call"
        signals:
            keyword_enabled: true
            vector_enabled: true
        budget:
            generation_reserve: 512
            min_recent_turns: 2
            max_recent_turns: 5
        fact_call:
            enabled: true
            max_rounds: 3

    hybrid_injection:
        enabled: true
        language: "cn"
        preference:
            enabled: true
            position_strategy: "negative"
            alpha: 0.4
            max_tokens: 200
        history:
            enabled: true
            method: "suffix_prompt"
            max_tokens: 2000

    gating:
        relevance_threshold: 0.7
        entropy_ceiling: 1.0
        entropy_floor: 0.5

    safety:
        max_alpha: 0.8
        fallback_on_error: true
        audit_logging: true

# v4.0: Rate Limiting
rate_limit:
    enabled: true
    max_rpm: 30
    max_concurrent: 3
    burst_size: 5

# v4.0: Circuit Breaking
circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 30
```

### v4.0 Enhanced Configuration

Enable dynamic routing and message management in Demo App config:

```yaml
# demo config
dki:
    config_path: "config/config.yaml"
    dynamic_router:
        enabled: true
        dki_threshold: 0.45
        rag_threshold: 0.25
        default_mode: "dki"
    message_management:
        enabled: true
        auto_write_messages: true
        auto_write_preferences: true
```

## 🔬 Research Background

### DKI's Positioning: User-Level Memory System

| Dimension            | RAG                            | DKI                                                     |
| -------------------- | ------------------------------ | ------------------------------------------------------- |
| **Target Data**      | External knowledge bases       | User preferences, session history                       |
| **Data Size**        | Large (thousands of documents) | Small to Medium (prefs 50-200, history 100-4000 tokens) |
| **Update Frequency** | Batch updates                  | Real-time per session                                   |
| **Privacy**          | Shared knowledge               | User-owned data                                         |
| **Caching**          | Document-level                 | User-level (high reuse)                                 |

### Token Budget Analysis

**RAG Paradigm:**

```
[Retrieved Content (consumed)] [User Input (remaining)]
Token Budget: B_t^used = n_m + n_u
```

**DKI Paradigm:**

```
[User Input (full budget available)]
     ↑ Memory injected via K/V (not in token budget)
Token Budget: B_t^used = n_u (memory free!)
```

### Performance Benchmarks

Based on DeepSeek-V3 7B experiments (n=500):

| Metric                 | RAG    | DKI        | Change     |
| ---------------------- | ------ | ---------- | ---------- |
| Memory Recall          | 87.3%  | 86.2%      | -1.1%      |
| First Turn Latency     | 78.8ms | 92.4ms     | +17.3%     |
| **Subsequent Latency** | 76.1ms | **42.8ms** | **-43.7%** |
| Cache Hit Rate         | N/A    | 69.7%      | -          |

### Design Invariants

1. **Storage Model-Agnostic**: Store only original text + routing vectors
2. **Injection Model-Consistent**: K/V computed with target model parameters
3. **Session Cache Disposable**: Inference-time enhancement, not persistent
4. **Three-Tier Graceful Degradation**: recall_v4 → stable → plain LLM
5. **User-Level Isolation**: HMAC-signed cache keys + post-inference K/V cleanup
6. **Audit Logging**: All injection decisions logged for compliance

## 📚 FAQ

**Q: Does DKI require retraining the model?**  
A: No. DKI is an inference-time enhancement using frozen model parameters.

**Q: What's the difference between DKI and RAG?**  
A: RAG concatenates retrieved content at token level, consuming context window; DKI injects K/V at attention level, doesn't consume token budget. They are complementary.

**Q: What is dynamic routing in v4.0?**  
A: `ConversationRouter` uses a five-dimension scoring model (history depth, preference match, trigger signals, session depth, cross-session correlation) to automatically decide whether to use RAG or DKI for the current query.

**Q: How to integrate DKI v4.0?**  
A: The simplest way requires only 3 lines:

```python
from dki.integration import create_plugin
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")
response = await dki.chat("User input", user_id="u1", session_id="s1")
```

**Q: Does v4.0 support streaming generation?**  
A: Yes. All model adapters (vLLM, SGLang, LLaMA, DeepSeek, GLM) implement `async_stream_generate`, accessible via `dki.chat_stream()`.

**Q: What changes does the upstream app need?**  
A: Provide an adapter config file, call `create_plugin()` to create the plugin, then call `dki.chat()`. That's it.

**Q: Production deployment recommendations?**  
A:

1. Enable Redis distributed cache
2. Enable rate limiting and circuit breaking
3. Configure dynamic routing (RAG + DKI complementary)
4. Monitor injection rate and latency
5. Adjust alpha and cache strategy based on metrics

## 📄 Related Papers

This project is based on the paper "Dynamic KV Injection: An Attention-Level User Memory System for Large Language Models".

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

---

**DKI v4.0** - Rethinking Memory Augmentation at the Attention Level
