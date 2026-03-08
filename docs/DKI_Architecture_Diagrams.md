# DKI 系统架构图与流程图 / DKI System Architecture & Flow Diagrams

本文档提供 DKI 系统的完整可视化架构图、注入流程图和数据流图。

## 1. 系统全景架构 / System Overview

### 1.1 整体架构 (Complete Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DKI System Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Application Layer (External / Upstream)                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │
│  │  │  Chat UI    │  │  Customer   │  │  Education  │  │  Any App    │           │  │
│  │  │  (Vue3)     │  │  Service    │  │  Platform   │  │  with DB    │           │  │
│  │  │  [Example]  │  │  [External] │  │  [External] │  │  [External] │           │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │  │
│  │         └────────────────┴────────────────┴────────────────┘                  │  │
│  │                                 │                                             │  │
│  │              Pass: user_id + raw input (No RAG, No Prompt Eng.)               │  │
│  └─────────────────────────────────┬─────────────────────────────────────────────┘  │
│                                    ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                         DKI Plugin Layer (Core)                               │  │
│  │                                                                               │  │
│  │  ┌─ Preprocessing ────────────────────────────────────────────────────────┐   │  │
│  │  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐             │   │  │
│  │  │  │ Memory        │   │ Reference     │   │ Dual-Factor   │             │   │  │
│  │  │  │ Trigger       │   │ Resolver      │   │ Gating        │             │   │  │
│  │  │  │ (What to      │   │ (Scope of     │   │ (Whether to   │             │   │  │
│  │  │  │  remember)    │   │  recall)      │   │  inject)      │             │   │  │
│  │  │  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘             │   │  │
│  │  └──────────┼───────────────────┼───────────────────┼─────────────────────┘   │  │
│  │             └───────────────────┼───────────────────┘                         │  │
│  │                                 ▼                                             │  │
│  │  ┌─ Injection Engine ─────────────────────────────────────────────────────┐   │  │
│  │  │                                                                        │   │  │
│  │  │  Strategy Selector: config → "stable" | "full_attention"               │   │  │
│  │  │                                                                        │   │  │
│  │  │  ┌─ Stable (Default) ──────────────────────────────────────┐           │   │  │
│  │  │  │  ┌───────────────┐        ┌─────────────────────────┐   │           │   │  │
│  │  │  │  │ Preference    │        │ History Suffix          │   │           │   │  │
│  │  │  │  │ K/V Injection │        │ Prompt Injector         │   │           │   │  │
│  │  │  │  │ (Attn Hook)   │        │ (Token Concat)          │   │           │   │  │
│  │  │  │  │ neg position  │        │ pos position            │   │           │   │  │
│  │  │  │  └───────────────┘        └─────────────────────────┘   │           │   │  │
│  │  │  └─────────────────────────────────────────────────────────┘           │   │  │
│  │  │                                                                        │   │  │
│  │  │  ┌─ Full Attention (Research) ─────────────────────────────┐           │   │  │
│  │  │  │  ┌───────────────┐        ┌─────────────────────────┐   │           │   │  │
│  │  │  │  │ Preference    │        │ History K/V             │   │           │   │  │
│  │  │  │  │ K/V Injection │        │ Injection               │   │           │   │  │
│  │  │  │  │ neg pos -100  │        │ neg pos -500            │   │           │   │  │
│  │  │  │  └───────────────┘        └─────────────────────────┘   │           │   │  │
│  │  │  └─────────────────────────────────────────────────────────┘           │   │  │
│  │  │                                                                        │   │  │
│  │  │  ┌─ Core Components ───────────────────────────────────────┐           │   │  │
│  │  │  │  MIS │ QCP │ Position Remapper │ FlashAttention Opt.    │           │   │  │
│  │  │  └─────────────────────────────────────────────────────────┘           │   │  │
│  │  └────────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                               │  │
│  │  ┌─ Supporting Services ──────────────────────────────────────────────────┐   │  │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │   │  │
│  │  │  │ Config-Driven │  │ Preference    │  │ Monitoring    │               │   │  │
│  │  │  │ Adapter       │  │ Cache         │  │ API           │               │   │  │
│  │  │  │ (SQLAlchemy)  │  │ (L1+L2)       │  │ (Stats/Logs)  │               │   │  │
│  │  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘               │   │  │
│  │  │          │                  │                   │                      │   │  │
│  │  │  ┌───────┴───────┐  ┌──────┴────────┐  ┌──────┴────────┐               │   │  │
│  │  │  │ JSON Content │   │ Redis (L2)    │  │ Visualization │               │   │  │
│  │  │  │ Extractor    │   │ Client        │  │ Routes        │               │   │  │
│  │  │  └───────────────┘  └───────────────┘  └───────────────┘               │   │  │
│  │  └────────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                │
│  ┌─────────────────────────────────┼─────────────────────────────────────────────┐  │
│  │                         Data Layer                                            │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │  │
│  │  │ Upstream App  │  │ Redis         │  │ GPU Memory    │  │ DKI Internal  │   │  │
│  │  │ Database      │  │ (L2 Cache)    │  │ (L1 Cache)    │  │ DB (Example)  │   │  │
│  │  │ (Prefs+Msgs)  │  │               │  │               │  │               │   │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM Engine Layer                                      │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │  │
│  │  │   vLLM    │  │  LLaMA    │  │  DeepSeek │  │   GLM     │  │  Others   │    │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖关系 (Module Dependency Graph)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Module Dependency Graph                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                                 ┌──────────────┐                                    │
│                                 │  main.py     │                                    │
│                                 │  start_dev   │                                    │
│                                 └──────┬───────┘                                    │
│                                        │                                            │
│                          ┌─────────────┼─────────────┐                              │
│                          ▼             ▼             ▼                              │
│                 ┌──────────────┐ ┌──────────┐ ┌──────────────┐                      │
│                 │ DKI Plugin   │ │ Web App  │ │ UI (Vue3)    │                      │
│                 │ (dki_plugin) │ │ (FastAPI)│ │ Example App  │                      │
│                 └──────┬───────┘ └─────┬────┘ └──────────────┘                      │
│                        │               │                                            │
│         ┌──────────────┼───────────────┼──────────────────┐                         │
│         │              │               │                  │                         │
│         ▼              ▼               ▼                  ▼                         │
│  ┌─────────────┐ ┌───────────┐ ┌────────────┐ ┌──────────────────┐                  │
│  │ Memory      │ │ Reference │ │ API Routes │ │ Full Attention   │                  │
│  │ Trigger     │ │ Resolver  │ │ (dki/stats │ │ Injector         │                  │
│  │             │ │           │ │  /viz/auth)│ │ (Research)       │                  │
│  └──────┬──────┘ └─────┬─────┘ └─────┬──────┘ └─────────┬────────┘                  │
│         │              │             │                  │                           │
│         │              │             │                  │                           │
│         └──────────────┼─────────────┘                  │                           │
│                        ▼                                │                           │
│  ┌─────────────────────────────────────┐                │                           │
│  │         Core Components             │                │                           │
│  │  ┌─────┐ ┌─────┐ ┌────────┐         │                │                           │
│  │  │ MIS │ │ QCP │ │ D.F.   │         │                │                           │
│  │  │     │ │     │ │ Gating │         │                │                           │
│  │  └──┬──┘ └──┬──┘ └───┬────┘         │                │                           │
│  │     │       │        │              │                │                           │
│  │  ┌──┴───────┴────────┴──────┐       │                │                           │
│  │  │ Hybrid Injector          │       │                │                           │
│  │  │ (preference_kv + suffix) │       │                │                           │
│  │  └──────────────────────────┘       │                │                           │
│  └─────────────┬───────────────────────┘                │                           │
│                │                                        │                           │
│       ┌────────┼────────────────────────────────────────┘                           │
│       │        │                                                                    │
│       ▼        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐                 │
│  │              Shared Infrastructure                             │                 │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │                 │
│  │  │ Config-Driven│  │ Preference   │  │ FlashAttn    │          │                 │
│  │  │ Adapter      │  │ Cache (L1+L2)│  │ Integration  │          │                 │
│  │  │ + JSON Parse │  │ + Redis      │  │ (FA3/FA2)    │          │                 │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │                 │
│  │         │                 │                 │                  │                 │
│  │  ┌──────┴───────┐  ┌──────┴─────┐   ┌───────┴──────┐           │                 │
│  │  │ SQLAlchemy   │  │ Redis      │   │ torch/flash  │           │                 │
│  │  │ (Dynamic     │  │ Client     │   │ _attn        │           │                 │
│  │  │  Mapping)    │  │            │   │              │           │                 │
│  │  └──────────────┘  └────────────┘   └──────────────┘           │                 │
│  └────────────────────────────────────────────────────────────────┘                 │
│                │                                                                    │
│                ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐                 │
│  │              LLM Model Adapters                                │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │                 │
│  │  │ vLLM     │ │ LLaMA    │ │ DeepSeek │ │ GLM      │           │                 │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │           │                 │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │                 │
│  └────────────────────────────────────────────────────────────────┘                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 注入流程图 / Injection Flow Diagrams

### 2.1 完整注入流程 (Complete Injection Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DKI Complete Injection Flow                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────┐                                                                  │
│  │ User Input    │   query = "最近我们聊了什么？"                                     │
│  │ + user_id     │   user_id = "user_001"                                           │
│  │ + session_id  │   session_id = "sess_042"                                        │
│  └───────┬───────┘                                                                  │
│          │                                                                          │
│          ▼                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 1: Memory Trigger Detection                                          │      │
│  │                                                                           │      │
│  │  Input: "最近我们聊了什么？"                                                │      │
│  │  Pattern Match:                                                           │      │
│  │    ├── META_COGNITIVE: ❌                                                 │      │
│  │    ├── STATE_CHANGE: ❌                                                   │      │
│  │    ├── LONG_TERM_VALUE: ❌                                                │      │
│  │    ├── RECALL_REQUEST: ✅ → "最近...聊了什么"                              │      │
│  │    └── OPINION_QUERY: ❌                                                  │      │
│  │                                                                           │      │
│  │  Result: trigger_type=RECALL_REQUEST, should_expand_recall=True           │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 2: Reference Resolver                                                │      │
│  │                                                                           │      │
│  │  Input: "最近我们聊了什么？" + trigger=RECALL_REQUEST                       │      │
│  │  Resolution:                                                              │      │
│  │    ├── "最近" → RECENT pattern detected                                   │      │
│  │    └── recall_turns = config.recent_turns (default: 10)                   │      │
│  │                                                                           │      │
│  │  Result: resolved_scope = {type: "recent", turns: 10}                     │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 3: External Data Retrieval (Config-Driven Adapter)                   │      │
│  │                                                                           │      │
│  │  3a. Read User Preferences                                                │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐      │      │
│  │  │ SELECT * FROM user_preferences WHERE user_id = 'user_001'       │      │      │
│  │  │ → [{type: "diet", content: "素食主义者"},                        │      │      │
│  │  │    {type: "style", content: "喜欢简洁回答"}]                     │      │      │
│  │  │                                                                 │      │      │
│  │  │ If content is JSON + json_key configured:                       │      │      │
│  │  │   → Parse JSON, extract specified key value                     │      │      │
│  │  └─────────────────────────────────────────────────────────────────┘      │      │
│  │                                                                           │      │
│  │  3b. Retrieve Relevant History (scope: 10 turns)                          │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐      │      │
│  │  │ Strategy: vector_search.type                                    │      │      │
│  │  │  ├── pgvector:  SQL vector similarity search                    │      │      │
│  │  │  ├── faiss:     FAISS index search                              │      │      │
│  │  │  ├── dynamic:                                                   │      │      │
│  │  │  │   ├── lazy:  On-demand embedding + cosine similarity         │      │      │
│  │  │  │   ├── batch: Pre-computed embeddings                         │      │      │
│  │  │  │   └── hybrid: BM25 top-20 → embedding rerank → top-10        │      │      │
│  │  │  └── none: Latest N messages by timestamp                       │      │      │
│  │  └─────────────────────────────────────────────────────────────────┘      │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│          ┌───────┴───────┐                                                          │
│          │               │                                                          │
│          ▼               ▼                                                          │
│  ┌──────────────┐  ┌──────────────────────────────┐                                 │
│  │ Step 4a:     │  │ Step 4b:                     │                                 │
│  │ Preference   │  │ History Processing           │                                 │
│  │ K/V Inject   │  │                              │                                 │
│  │              │  │ ┌──────────────────────────┐ │                                 │
│  │ Check Cache: │  │ │ Format as suffix prompt  │ │                                 │
│  │ L1 (Memory)  │  │ │                          │ │                                 │
│  │  ↓ miss      │  │ │ [Previous conversations: │ │                                 │
│  │ L2 (Redis)   │  │ │  User: 推荐一家餐厅       │ │                                 │
│  │  ↓ miss      │  │ │  AI: 根据您的素食偏好...  │ │                                 │
│  │ L3 (Compute) │  │ │  User: 谢谢,下次推荐...   │ │                                 │
│  │              │  │ │  ...                     │ │                                 │
│  │ Tokenize →   │  │ │  Based on the above      │ │                                 │
│  │ Compute K/V  │  │ │  context, please respond │ │                                 │
│  │ → Cache L1+L2│  │ │  naturally.]             │ │                                 │
│  │              │  │ └──────────────────────────┘ │                                 │
│  │ Position:    │  │                              │                                 │
│  │  negative    │  │ Position: positive (after    │                                 │
│  │ α: 0.3-0.5  │  │  user query)                  │                                 │
│  │ No Context ↑ │  │ Uses Context ↑               │                                 │
│  └──────┬───────┘  └──────────────┬───────────────┘                                 │
│         │                         │                                                 │
│         └─────────┬───────────────┘                                                 │
│                   ▼                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 5: Dual-Factor Gating Decision                                       │      │
│  │                                                                           │      │
│  │  Factor 1 (Primary): Relevance Score                                      │      │
│  │    similarity = cosine(query_emb, memory_emb) = 0.82                      │      │
│  │    inject = 0.82 > threshold(0.7) → True                                  │      │
│  │                                                                           │      │
│  │  Factor 2 (Modulator): Entropy-based α ceiling                            │      │
│  │    entropy = attention_entropy(model) = 0.6                               │      │
│  │    α_max = 0.5 + (1.0 - 0.5) × 0.6 = 0.8                                  │      │
│  │    α = min(0.4, 0.8) = 0.4                                                │      │
│  │                                                                           │      │
│  │  Decision: inject=True, α=0.4                                             │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 6: LLM Inference (with Injection)                                    │      │
│  │                                                                           │      │
│  │  ┌─ Attention Hook Registration ─────────────────────────────────────┐    │      │
│  │  │                                                                   │    │      │
│  │  │  def attention_hook(module, input, output):                       │    │      │
│  │  │      K, V = output.key, output.value                              │    │      │
│  │  │      K_new = concat([K_preference, K], dim=seq_len)               │    │      │
│  │  │      V_new = concat([V_preference, V], dim=seq_len)               │    │      │
│  │  │      # Apply MIS alpha scaling                                    │    │      │
│  │  │      output = α * attention(Q, K_new, V_new) + (1-α) * output     │    │      │
│  │  │      return output                                                │    │      │
│  │  │                                                                   │    │      │
│  │  └───────────────────────────────────────────────────────────────────┘    │      │
│  │                                                                           │      │
│  │  Final Input to LLM:                                                      │      │
│  │    [query_tokens] + [history_suffix_tokens]                               │      │
│  │                                                                           │      │
│  │  Hidden Injection (via Hook):                                             │      │
│  │    preference_kv → prepended to K,V at negative positions                 │      │
│  │                                                                           │      │
│  │  FlashAttention Optimization:                                             │      │
│  │    GPU >= Hopper → FA3 | GPU >= Ampere → FA2 | else → Standard            │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐      │
│  │ Step 7: Response & Monitoring                                             │      │
│  │                                                                           │      │
│  │  ├── Record injection log (for monitoring API)                            │      │
│  │  │   {timestamp, user_id, alpha, injection_enabled, latency_ms,           │      │
│  │  │    preference_tokens, history_tokens, cache_hit, strategy}             │      │
│  │  ├── Record visualization data (for debug UI)                             │      │
│  │  │   {flow_steps, token_distribution, attention_weights}                  │      │
│  │  └── Return response + DKI metadata                                       │      │
│  └───────────────┬───────────────────────────────────────────────────────────┘      │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌───────────────┐                                                                  │
│  │ Response:     │   text = "最近我们主要讨论了餐厅推荐、旅行计划..."                  │
│  │ + metadata    │   dki_metadata = {alpha: 0.4, cache_hit: true, ...}              │
│  └───────────────┘                                                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 注入策略对比 (Injection Strategy Comparison)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                  Injection Strategy Comparison: Stable vs Full Attention            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ═══ Stable Strategy (Production) ═══                                               │
│                                                                                     │
│  Token Position Layout:                                                             │
│  ◄───── Negative Positions ─────►│◄────────── Positive Positions ──────────────►    │
│                                  │                                                  │
│  ┌───────────────────────────────┼───────────────────────────────────────────────┐  │
│  │                               │                                               │  │
│  │  ┌─────────────────────────┐  │  ┌─────────────┐  ┌───────────────────────┐   │  │
│  │  │  Preference K/V         │  │  │  User Query │  │  History Suffix       │   │  │
│  │  │  (Attention Hook)       │  │  │  (Raw Input)│  │  (Trust Guidance)     │   │  │
│  │  │  α = 0.3-0.5            │  │  │  α = 1.0    │  │  α = 1.0              │   │  │
│  │  │  50-200 tokens          │  │  │  10-100 tok │  │  100-2000 tokens      │   │  │
│  │  │  ❌ No Context Usae    │  │   │  ✅ Context │  │  ✅ Context Usage    │   │  │
│  │  │  ✅ Cacheable          │  │   │             │  │  ❌ Dynamic          │   │  │
│  │  └─────────────────────────┘  │  └─────────────┘  └───────────────────────┘   │  │
│  │                               │                                               │  │
│  └───────────────────────────────┴───────────────────────────────────────────────┘  │
│                                                                                     │
│  ═══ Full Attention Strategy (Research) ═══                                         │
│                                                                                     │
│  Token Position Layout:                                                             │
│  ◄──────────────── All Negative Positions ──────────────────►│◄── Positive ──►     │
│                                                               │                     │
│  ┌────────────────────────────────────────────────────────────┼───────────────────┐ │
│  │                                                            │                   │ │
│  │  ┌────────────────────┐  ┌────────────────────┐            │  ┌─────────────┐  │ │
│  │  │  History K/V       │  │  Preference K/V    │            │  │  User Query │  │ │
│  │  │  (Attention Hook)  │  │  (Attention Hook)  │            │  │  + Global   │  │ │
│  │  │  pos: -500 ~ -101  │  │  pos: -100 ~ -1    │            │  │  Indication │  │ │
│  │  │  α = 0.3           │  │  α = 0.4           │            │  │  α = 1.0    │  │ │
│  │  │  100-400 tokens    │  │  50-200 tokens     │            │  │  +3-5 token │  │ │
│  │  │  ❌ No Context    │  │  ❌ No Context     │            │  │  ✅ Context │  │ │
│  │  └────────────────────┘  └────────────────────┘            │  └─────────────┘  │ │
│  │                                                            │                   │ │
│  └────────────────────────────────────────────────────────────┴───────────────────┘ │
│                                                                                     │
│  ═══ Comparison Summary ═══                                                         │
│                                                                                     │
│  ┌──────────────────┬──────────────────────┬──────────────────────────┐             │
│  │ Dimension        │ Stable               │ Full Attention           │             │
│  ├──────────────────┼──────────────────────┼──────────────────────────┤             │
│  │ Context Usage    │ Medium (suffix)      │ Minimal (3-5 tokens)     │             │
│  │ OOD Risk         │ Low (prefs only)     │ Higher (history too)     │             │
│  │ Citability       │ History citable      │ History implicit         │             │
│  │ Cache Efficiency │ Prefs only           │ Both cacheable           │             │
│  │ Stability        │ ⭐⭐⭐⭐⭐          │ ⭐⭐⭐                │             │
│  │ Research Value   │ ⭐⭐                 │ ⭐⭐⭐⭐⭐            │            │
│  │ Hallucination    │ Lower                │ Needs validation         │             │
│  │ Recommended For  │ Production           │ Research / Experiment    │             │
│  └──────────────────┴──────────────────────┴──────────────────────────┘             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 数据适配器工作流程 (Adapter Workflow)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Config-Driven Adapter Workflow                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────┐                                                              │
│  │ adapter_config.yml│  Database type, table names, field mappings,                 │
│  │ (User Provided)   │  vector search config, JSON content key                      │
│  └────────┬──────────┘                                                              │
│           │                                                                         │
│           ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐             │
│  │  ConfigDrivenAdapter.__init__()                                    │             │
│  │                                                                    │             │
│  │  1. Parse YAML config                                              │             │
│  │  2. Build SQLAlchemy connection string                             │             │
│  │  3. Create Engine + Session                                        │             │
│  │  4. Dynamic Table Reflection (no hardcoded schema)                 │             │
│  │     table = Table(config.table_name, metadata, autoload=True)      │             │
│  │  5. Create TableMapping objects (field_map + content_json_key)     │             │
│  └────────────────────────────────────────────────────────────────────┘             │
│                                                                                     │
│  ┌── Query Flow ──────────────────────────────────────────────────────┐             │
│  │                                                                    │             │
│  │  get_user_preferences(user_id)                                     │             │
│  │  ┌────────────────────────────────────────────────────┐            │             │
│  │  │  1. SELECT from preferences_table                  │            │             │
│  │  │     WHERE user_id_field = :user_id                 │            │             │
│  │  │     AND filter_conditions                          │            │             │
│  │  │  2. For each row:                                  │            │             │
│  │  │     content = row[content_field]                   │            │             │
│  │  │     if content_json_key:                           │            │             │
│  │  │       content = _extract_json_content(content, key)│            │             │
│  │  │     → UserPreference(content=content, ...)         │            │             │
│  │  └────────────────────────────────────────────────────┘            │             │
│  │                                                                    │             │
│  │  search_relevant_history(user_id, query, session_id, limit)        │             │
│  │  ┌────────────────────────────────────────────────────┐            │             │
│  │  │  1. SELECT from messages_table                     │            │             │
│  │  │     WHERE user_id_field = :user_id                 │            │             │
│  │  │     AND session_id_field = :session_id             │            │             │
│  │  │     ORDER BY timestamp DESC LIMIT :limit           │            │             │
│  │  │  2. Vector search strategy (if configured):        │            │             │
│  │  │     ├── pgvector: WHERE embedding <-> :q < :thres  │            │             │
│  │  │     ├── dynamic.hybrid:                            │            │             │
│  │  │     │   BM25(query, messages) → top_20             │            │             │
│  │  │     │   → embed(top_20) → cosine rerank → top_k    │            │             │
│  │  │     └── dynamic.lazy: embed_on_demand → cosine     │            │             │
│  │  │  3. JSON content extraction (if json_key set)      │            │             │
│  │  └────────────────────────────────────────────────────┘            │             │
│  └────────────────────────────────────────────────────────────────────┘             │
│                                                                                     │
│  ┌── JSON Content Extraction ─────────────────────────────────────────┐             │
│  │                                                                    │             │
│  │  _extract_json_content(raw_content, json_key)                      │             │
│  │                                                                    │             │
│  │  Input:  raw = '{"text":"Hello","model":"gpt-4","tokens":50}'      │             │
│  │          key = "text"                                              │             │
│  │                                                                    │             │
│  │  Process:                                                          │             │
│  │  ┌──────────────────────────────────────────────┐                  │             │
│  │  │ 1. json.loads(raw) → dict                    │                  │             │
│  │  │ 2. Split key by "." → ["text"]               │                  │             │
│  │  │ 3. Navigate: dict["text"] → "Hello"          │                  │             │
│  │  │ 4. Return "Hello"                            │                  │             │
│  │  │                                              │                  │             │
│  │  │ Nested: key = "choices.0.message.content"    │                  │             │
│  │  │ → dict["choices"][0]["message"]["content"]   │                  │             │
│  │  │                                              │                  │             │
│  │  │ Fallback: If parse fails → return raw_content│                  │             │
│  │  └──────────────────────────────────────────────┘                  │             │
│  └────────────────────────────────────────────────────────────────────┘             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 缓存层次结构 (Cache Hierarchy)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Preference K/V Cache Hierarchy                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Request for user_001 preference K/V                                                │
│       │                                                                             │
│       ▼                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  L1: GPU Memory (LRU Cache)                                                 │    │
│  │  ├── Speed: ~0.1ms                                                          │    │
│  │  ├── Capacity: ~1,000 users (configurable)                                  │    │
│  │  ├── Format: Raw FP16 Tensors                                               │    │
│  │  └── Eviction: LRU                                                          │    │
│  │                                                                             │    │
│  │  ┌─ HIT ────────────────────────┐  ┌─ MISS ─────────────────────────────┐   │    │
│  │  │ Return cached K/V tensors    │  │ Proceed to L2                      │   │    │
│  │  │ Latency: ~0.1ms              │  │                                    │   │    │
│  │  └──────────────────────────────┘  └──────────────────┬─────────────────┘   │    │
│  └───────────────────────────────────────────────────────┼─────────────────────┘    │
│                                                          ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  L2: Redis (Distributed Cache)     [Optional, config: redis.enabled=true]   │    │
│  │  ├── Speed: ~1-5ms (network)                                                │    │
│  │  ├── Capacity: ~100,000 users                                               │    │
│  │  ├── Format: msgpack serialized tensors                                     │    │
│  │  ├── Sharing: Cross-instance                                                │    │
│  │  └── TTL: Configurable (default 1h)                                         │    │
│  │                                                                             │    │
│  │  ┌─ HIT ────────────────────────┐  ┌─ MISS ─────────────────────────────┐   │    │
│  │  │ Deserialize → promote to L1  │  │ Proceed to L3 (Compute)            │   │    │
│  │  │ Latency: ~1-5ms              │  │                                    │   │    │
│  │  └──────────────────────────────┘  └──────────────────┬─────────────────┘   │    │
│  └───────────────────────────────────────────────────────┼─────────────────────┘    │
│                                                          ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  L3: Compute (Recompute on Demand)                                          │    │
│  │  ├── Speed: ~50-200ms                                                       │    │
│  │  ├── Process:                                                               │    │
│  │  │   1. Read preference text from adapter                                   │    │
│  │  │   2. Tokenize preference text                                            │    │
│  │  │   3. model.compute_kv(tokens) → K, V tensors                             │    │
│  │  │   4. Store in L1 (always) + L2 (if Redis enabled)                        │    │
│  │  └── No capacity limit (compute on demand)                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  ═══ Cache Hit Rate Analysis ═══                                                    │
│                                                                                     │
│  Single Instance (no Redis):                                                        │
│  ┌──────────────────────────────────────────────┐                                   │
│  │  L1 Hit: ~70%    L3 Compute: ~30%            │                                   │
│  │  Avg Latency: 0.7×0.1 + 0.3×100 ≈ 30ms       │                                   │
│  └──────────────────────────────────────────────┘                                   │
│                                                                                     │
│  Multi Instance (N=4, no Redis):                                                    │
│  ┌──────────────────────────────────────────────┐                                   │
│  │  L1 Hit: ~17.5%  L3 Compute: ~82.5%          │                                   │
│  │  Avg Latency: 0.175×0.1 + 0.825×100 ≈ 82ms   │                                   │
│  └──────────────────────────────────────────────┘                                   │
│                                                                                     │
│  Multi Instance (N=4, with Redis):                                                  │
│  ┌──────────────────────────────────────────────┐                                   │
│  │  L1 Hit: ~50%  L2 Hit: ~20%  L3: ~30%        │                                   │
│  │  Avg: 0.5×0.1 + 0.2×3 + 0.3×100 ≈ 30.6ms     │                                   │
│  └──────────────────────────────────────────────┘                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Attention 计算详解 (Attention Computation Detail)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    DKI Attention Computation                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Standard LLM Attention:                                                            │
│  ┌──────────────────────────────────────────────────────────────┐                   │
│  │  Q, K, V = W_q(X), W_k(X), W_v(X)                            │                   │
│  │  Attn = softmax(Q·K^T / √d) · V                              │                   │
│  │                                                              │                   │
│  │  Size: [batch, heads, seq_len, head_dim]                     │                   │
│  └──────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
│  DKI-Injected Attention (Stable Strategy):                                          │
│  ┌──────────────────────────────────────────────────────────────┐                   │
│  │                                                              │                   │
│  │  1. Compute preference K/V:                                  │                   │
│  │     K_pref = W_k(preference_tokens)  # from cache or compute │                   │
│  │     V_pref = W_v(preference_tokens)                          │                   │
│  │                                                              │                   │
│  │  2. Prepare input tokens:                                    │                   │
│  │     input = [user_query_tokens] + [history_suffix_tokens]    │                   │
│  │     K_input, V_input = W_k(input), W_v(input)                │                   │
│  │     Q = W_q(input)                                           │                   │
│  │                                                              │                   │
│  │  3. Concatenate (via Attention Hook):                        │                   │
│  │     K_full = [K_pref | K_input]  # dim=seq_len               │                   │
│  │     V_full = [V_pref | V_input]  # dim=seq_len               │                   │
│  │                                                              │                   │
│  │  4. Compute attention with position remapping:               │                   │
│  │     positions = [-n_pref, ..., -1, 0, 1, ..., n_input]       │                   │
│  │     Attn_raw = softmax(Q·K_full^T / √d + pos_bias)           │                   │
│  │                                                              │                   │
│  │  5. Apply Memory Influence Scaling (MIS):                    │                   │
│  │     Attn_final = α · Attn_raw_with_pref +                    │                   │
│  │                  (1-α) · Attn_raw_without_pref               │                   │
│  │                                                              │                   │
│  │  6. Output:                                                  │                   │
│  │     output = Attn_final · V_full                             │                   │
│  │                                                              │                   │
│  │  FlashAttention Optimization (if available):                 │                   │
│  │     Uses flash_attn_func for steps 4-6 with:                 │                   │
│  │     - IO-aware memory tiling                                 │                   │
│  │     - Kernel fusion                                          │                   │
│  │     - FP8 (FA3 on Hopper GPUs)                               │                   │
│  └──────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
│  Token Budget Comparison:                                                           │
│  ┌──────────────────────────────────────────────────────────────┐                   │
│  │                                                              │                   │
│  │  RAG:   [retrieved_context | user_query]                     │                   │
│  │         ◄── 500 tokens ──►◄── 100 ──►                        │                   │
│  │         Total context used: 600 tokens                       │                   │
│  │                                                              │                   │
│  │  DKI:   [user_query | history_suffix]                        │                   │
│  │         ◄── 100 ──►◄── 200 tokens ──►                        │                   │
│  │         + preference K/V via hook (not in context)           │                   │
│  │         Total context used: 300 tokens                       │                   │
│  │         (Preferences are FREE - no context cost)             │                   │
│  │                                                              │                   │
│  └──────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. 关于显式后缀是否占用 Context 的说明

### 3.1 结论

**是的，历史后缀（显式后缀）占用 Context 窗口。偏好 K/V 注入不占用。**

### 3.2 详细对比

| 注入方式            | 位置     | 是否占用 Context              | 原因                                           |
| ------------------- | -------- | ----------------------------- | ---------------------------------------------- |
| 偏好 K/V 注入       | 负位置   | ❌ 不占用                     | 通过 Attention Hook 直接修改 K/V，不增加 token |
| 历史后缀注入        | 正位置   | ✅ 占用                       | 作为 token 拼接到输入中                        |
| Full Attention 策略 | 全负位置 | ❌ 不占用 (仅 3-5 token 指示) | 历史也通过 K/V 注入                            |

### 3.3 Token 预算估算 (Context Window = 8192)

| 组件         | Token 数      | 占比 (Stable) | 占比 (Full Attn)           |
| ------------ | ------------- | ------------- | -------------------------- |
| 系统提示词   | ~200          | 2.4%          | 2.4%                       |
| 用户查询     | ~100          | 1.2%          | 1.2%                       |
| 历史后缀     | ~500          | 6.1%          | 0%                         |
| 全局指示     | 0             | 0%            | 0.06% (5 tokens)           |
| 可用于生成   | ~7392 / ~7687 | 90.2%         | 93.8%                      |
| **偏好 K/V** | **~100**      | **0% (免费)** | **0% (免费)**              |
| **历史 K/V** | **N/A**       | **N/A**       | **0% (免费, ~300 tokens)** |

## 4. DKI 与 AGA 协同工作时序 (Co-deployment with AGA)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    DKI + AGA Co-deployment Timeline                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Time →                                                                             │
│  ═══════════════════════════════════════════════════════════════════                │
│                                                                                     │
│  Phase 1: Input Processing                                                          │
│  ┌──────────────────────────────────────────────────────────────┐                   │
│  │ User Input: "推荐一本关于机器学习的书"                         │                   │
│  │ + user_id, session_id                                        │                   │
│  └────────────────────────┬─────────────────────────────────────┘                   │
│                           │                                                         │
│  Phase 2: DKI Activation (PRE-INFERENCE)                                            │
│  ┌────────────────────────┴─────────────────────────────────────┐                   │
│  │ [DKI] Attention Hook Registration                            │                   │
│  │  ├── Read user preferences → K/V injection                   │                   │
│  │  ├── Retrieve relevant history → suffix prompt               │                   │
│  │  └── Prepare attention hook for K/V prepend                  │                   │
│  │                                                              │                   │
│  │  Hook Type: Attention Layer (K/V modification)               │                   │
│  │  Trigger: Before model.forward()                             │                   │
│  └────────────────────────┬─────────────────────────────────────┘                   │
│                           │                                                         │
│  Phase 3: LLM Inference Start                                                       │
│  ┌────────────────────────┴─────────────────────────────────────┐                   │
│  │ model.forward(input_tokens + history_suffix)                 │                   │
│  │                                                              │                   │
│  │  At each Attention Layer:                                    │                   │
│  │  ├── [DKI Hook] Prepend preference K/V to K, V               │                   │
│  │  └── Compute attention with injected K/V                     │                   │
│  │                                                              │                   │
│  └────────────────────────┬─────────────────────────────────────┘                   │
│                           │                                                         │
│  Phase 4: AGA Activation (DURING INFERENCE, High Entropy)                           │
│  ┌────────────────────────┴─────────────────────────────────────┐                   │
│  │ [AGA] FFN Hook                                               │                   │
│  │  ├── Monitor attention entropy at each layer                 │                   │
│  │  ├── If entropy > threshold (model uncertain):               │                   │
│  │  │   ├── Retrieve relevant knowledge                         │                   │
│  │  │   ├── Inject via FFN output modification                  │                   │
│  │  │   └── α = f(entropy, relevance)                           │                   │
│  │  └── If entropy < threshold: bypass (no injection)           │                   │
│  │                                                              │                   │
│  │  Hook Type: FFN Output Layer (residual modification)         │                   │
│  │  Trigger: During model.forward(), per-layer                  │                   │
│  └────────────────────────┬─────────────────────────────────────┘                   │
│                           │                                                         │
│  Phase 5: Output                                                                    │
│  ┌────────────────────────┴─────────────────────────────────────┐                   │
│  │ Response with:                                               │                   │
│  │  ├── DKI: User-personalized context                          │                   │
│  │  └── AGA: Knowledge-grounded facts (if high-entropy)         │                   │
│  └──────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
│  ═══ No Conflict Analysis ═══                                                       │
│  ┌──────────────────────────────────────────────────────────────┐                   │
│  │  DKI: Modifies K, V in Attention layers (pre-inference)      │                   │
│  │  AGA: Modifies FFN output (during inference)                 │                   │
│  │                                                              │                   │
│  │  These operate on DIFFERENT components:                      │                   │
│  │  - DKI → Attention K/V (what the model "sees")               │                   │
│  │  - AGA → FFN output (what the model "knows")                 │                   │
│  │                                                              │                   │
│  │  Conclusion: ✅ No conflict, complementary design            │                   │
│  └──────────────────────────────────────────────────────────────┘                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 5. 相关文档

-   [集成指南 / Integration Guide](Integration_Guide.md)
-   [FlashAttention-3 集成方案](FlashAttention3_Integration.md)
-   [动态向量检索说明](Dynamic_Vector_Search.md)
-   [用户记忆注入完整系统方案](DKI_用户记忆注入完整系统方案.md)
-   [DKI 演化路径的思考](DKI演化路径的思考.md)
