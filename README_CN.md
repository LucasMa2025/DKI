# DKI - Dynamic KV Injection 动态键值注入--大语言模型用户级记忆系统

> 大型语言模型的注意力层级用户记忆插件

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-4.0.0-green.svg)]()

[English](README.md) | 简体中文

## 📖 概述

DKI (Dynamic KV Injection，动态键值注入) 是一个**LLM 注意力层级插件**，通过 Attention Hook 在推理时动态注入用户偏好和会话历史，实现跨会话的个性化记忆。

### DKI 是什么

DKI 是一个 **LLM 插件**，专为**用户级记忆**设计：

-   **注意力 Hook 机制**：通过 PyTorch Hook 在注意力层级注入 K/V，而非 prompt 拼接
-   **配置驱动适配器**：自动读取上层应用的数据库，无需修改上层应用代码
-   **混合注入策略**：偏好 K/V 注入（负位置）+ 历史后缀提示词（正位置）
-   **极简集成** (v4.0)：3 行代码集成，支持动态路由和消息管理

**核心工作流**：

```
上层应用 → 传递 user_id + 原始输入 → DKI 插件
    ↓
DKI 通过配置驱动适配器读取上层应用数据库
    ↓
偏好 → K/V 注入 (负位置) | 历史 → 后缀提示词 (正位置)
    ↓
调用 LLM 推理 → 返回响应
```

### DKI 不是什么

-   **不是 RAG**：DKI 使用 K/V 注入而非 prompt 拼接，不消耗 token 预算
-   **不是知识库检索**：DKI 专注于用户级记忆，外部知识请使用 RAG
-   **不需要上层应用实现接口**：配置驱动，上层应用只需传递 user_id 和原始输入

### 为什么需要这个定位

这种聚焦的范围带来以下优势：

1. **短偏好**（50-200 tokens）→ 降低位置编码风险，可缓存
2. **用户自有数据** → 简化隐私考量
3. **会话连贯** → 有效的 K/V 缓存
4. **稳定偏好** → 高缓存复用率

### 核心特性

-   **🧠 注意力 Hook 注入**：通过 PyTorch Hook 在注意力层级注入 K/V，而非 prompt token
-   **🔀 Recall v4 记忆召回**：多信号检索 + 动态摘要 + 事实补充（主策略），stable 混合注入自动回退
-   **🔧 配置驱动适配器**：SQLAlchemy 动态表映射，无需上层应用实现接口
-   **🔐 用户级隔离**：HMAC 签名缓存键 + UserIsolationContext + 推理后 K/V 清理
-   **🎚️ 记忆影响缩放（MIS）**：连续的 α ∈ [0, 1] 强度控制
-   **🔄 查询条件投影**：FiLM 风格的记忆中心变换
-   **🚦 双因子门控**：相关性驱动决策，熵调制强度
-   **💾 分层 KV 缓存**：L1(GPU) → L2(CPU) → L3(SSD) → L4(重计算)
-   **📊 监控 API**：统计数据、注入日志、健康检查
-   **🔌 多引擎支持**：vLLM、SGLang、LLaMA、DeepSeek、GLM（均支持流式生成）
-   **✅ 优雅降级**：recall_v4 → stable → 纯 LLM，三级回退
-   **🚀 极简集成 (v4.0)**：3 行代码集成、FastAPI Middleware、动态路由、消息管理
-   **🔀 动态路由 (v4.0)**：自动在 RAG 和 DKI 之间切换，五维评分模型
-   **📝 消息管理 (v4.0)**：DKI 内部完成消息写入和偏好写入
-   **⚡ 流式生成 (v4.0)**：所有模型适配器支持异步流式输出

## 🏗️ 架构

### v4.0 集成架构

DKI v4.0 提供三层集成模式，由简到繁，参考 AGA 插件架构模式：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DKI v4.0 集成架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: One-Line Factory (极简集成, 推荐)                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  from dki.integration import create_plugin                      │    │
│  │  dki = await create_plugin(adapter_config_path="config/a.yaml") │    │
│  │  response = await dki.chat("推荐餐厅", user_id="u1", ...)        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Level 2: FastAPI Middleware (Web 应用集成)                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  from dki.integration import DKIMiddleware                      │    │
│  │  app.add_middleware(DKIMiddleware, adapter_config_path="...")   │    │
│  │  # 自动初始化、生命周期管理、依赖注入                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Level 3: EnhancedDKIPlugin (完全控制)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  EnhancedDKIPlugin(dki_plugin, rag_system, store, config)       │    │
│  │  ├── dynamic_router: RAG ↔ DKI 自动切换                         │    │
│  │  ├── message_management: 消息写入 + 偏好写入                     │    │
│  │  └── 统一生命周期管理                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心架构：LLM 插件模式

DKI 作为 LLM 的**注意力层级插件**，通过 PyTorch Hook 机制实现 K/V 注入：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DKI 插件架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  上层应用 (Chat UI / 客服系统 / 其他应用)                         │    │
│  │  └── 只需传递: user_id + 原始用户输入                             │    │
│  │      (无需 RAG, 无需 Prompt 工程, 无需实现接口)                   │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DKI 集成层 (v4.0)                                              │    │
│  │  ├── create_plugin() — 一行代码创建                              │    │
│  │  ├── DKIMiddleware — FastAPI 自动集成                           │    │
│  │  ├── EnhancedDKIPlugin — 动态路由 + 消息管理                     │    │
│  │  │   ├── ConversationRouter — RAG/DKI 五维评分路由               │    │
│  │  │   └── MessageManagement — 消息/偏好自动写入                   │    │
│  │  └── DKIPlugin — 核心插件                                        │    │
│  │      ├── 配置驱动适配器 (SQLAlchemy 动态表映射)                   │    │
│  │      ├── 偏好处理 → K/V 注入 (负位置, Attention Hook)             │    │
│  │      ├── 历史处理 → 后缀提示词 (正位置)                           │    │
│  │      ├── 流式生成 (chat_stream)                                  │    │
│  │      └── 监控 API (统计/日志/健康检查)                            │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LLM 引擎 (vLLM / SGLang / LLaMA / DeepSeek / GLM)              │    │
│  │  └── 带 K/V 注入的推理 (支持同步/异步/流式)                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 动态路由 (v4.0)

DKI v4.0 的 `ConversationRouter` 基于五维评分模型自动在 RAG 和 DKI 之间切换：

```
Score_DKI = w₁·S_history + w₂·S_preference + w₃·S_trigger
          + w₄·S_session_depth + w₅·S_cross_session

Route = DKI  if Score_DKI > θ_dki
      = RAG  if Score_DKI < θ_rag
      = DKI  otherwise (with reduced confidence)
```

| 维度         | RAG 优势    | DKI 优势            |
| ------------ | ----------- | ------------------- |
| 会话长度     | 1-3 轮 (短) | 5+ 轮 (长, 跨会话)  |
| 记忆类型     | 外部知识库  | 用户偏好 + 会话历史 |
| 个性化       | 无/弱       | 强 (偏好 K/V 注入)  |
| 首次交互     | ★ 强        | 弱 (无历史可召回)   |
| 跨会话连续性 | 弱          | ★ 强 (跨会话记忆)   |

### 注入策略选择

DKI v4.0 采用 **recall_v4** 作为主策略，**stable** 作为回退方案：

| 策略                 | 状态        | 适用场景       | Context 占用 | 稳定性     |
| -------------------- | ----------- | -------------- | ------------ | ---------- |
| **recall_v4** (默认) | ✅ 主策略   | 长历史场景     | 动态管理     | ⭐⭐⭐⭐⭐ |
| **stable** (回退)    | ✅ 回退方案 | recall_v4 失败 | 中等         | ⭐⭐⭐⭐⭐ |
| **full_attention**   | ⚠️ 已弃用   | 仅研究         | 极小         | ⭐⭐⭐     |

**回退机制**: 当 recall_v4 执行失败时（如组件未初始化、召回错误），系统自动回退到 stable 策略，使用混合注入（偏好 K/V + 历史后缀提示词）。如果 stable 也失败，则降级到无注入的纯 LLM 推理。

```yaml
# config.yaml
dki:
    injection_strategy: "recall_v4" # recall_v4 (推荐) | stable (回退)
```

### 混合注入策略 (Stable) — 回退方案

**回退策略**，当 recall_v4 失败时自动激活。采用**分层注入方式**，模拟人类认知结构：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  DKI 混合注入架构 (Stable)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 1: 用户偏好 (K/V 注入 - Attention Hook)                   │    │
│  │  ├── 内容: 饮食偏好、风格、兴趣                                   │    │
│  │  ├── 位置: 负位置 (概念上在用户输入"之前")                         │    │
│  │  ├── 机制: PyTorch Hook 修改 Attention K/V                       │    │
│  │  ├── 影响: 隐式的、背景性的 (类似人格特质)                         │    │
│  │  └── α: 0.3-0.5 (较低，用于微妙影响)                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 2: 会话历史 (后缀提示词)                                  │    │
│  │  ├── 内容: 最近的对话轮次                                        │    │
│  │  ├── 位置: 用户查询之后 (正位置)                                  │    │
│  │  ├── 机制: 标准 token 拼接                                       │    │
│  │  ├── 影响: 显式的、可引用的 (类似记忆)                            │    │
│  │  └── 提示: 可信度引导提示                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 3: 当前查询 (标准输入)                                    │    │
│  │  └── 注意力的主要焦点                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Full Attention / Engram 启发策略 (⚠️ 已弃用)

> **弃用说明**: Full Attention 策略和 Engram 启发注入策略已被弃用。这些策略将偏好和历史全部通过 K/V 注入以实现接近零的上下文占用，但在**长历史场景下存在根本性限制**：
>
> 1. **K/V 注入容量有限**：随着对话历史增长（数十到数百轮），K/V token 数量急剧增加，超出有效注意力范围
> 2. **无法显式引用**：通过 K/V 注入的历史无法被模型显式引用或推理
> 3. **OOD 风险**：在负位置大量注入 K/V 会导致严重的训练分布偏移
> 4. **事实准确性差**：模型无法从 K/V 注入的历史中提取具体事实（日期、价格等）
>
> **替代方案**: 使用 **Recall v4 记忆召回策略**（见下文），通过多信号检索 + 动态摘要 + 应用层事实补充，为长历史场景提供稳定可靠的记忆召回能力。

### Recall v4 记忆召回策略 (推荐)

**生产推荐策略**。模拟人类记忆召回过程，通过多信号检索、动态历史构造和应用层事实补充，在长历史场景下提供稳定可靠的记忆能力。核心实现位于 `dki/core/recall/`。

> **⚠️ 重要区分: DKI Recall v4 的 Summary ≠ Rolling Summary**
>
> DKI 的摘要机制与 ChatGPT/Claude/Grok 等 AI 系统使用的 Rolling Summary 有**本质区别**：
>
> | 维度         | DKI Recall v4 Summary                          | Rolling Summary (ChatGPT/Claude/Grok) |
> | ------------ | ---------------------------------------------- | ------------------------------------- |
> | **触发时机** | 按需触发，仅当召回消息超过阈值时               | 持续运行，每 N 轮后自动执行           |
> | **作用范围** | 单消息粒度，仅摘要单条长消息                   | 全局性，压缩整个对话历史              |
> | **可追溯性** | 每条摘要携带 `trace_id` → 可回溯原始消息       | 原始消息被丢弃，不可恢复              |
> | **摘要策略** | 默认抽取式 (jieba TextRank)，无需额外 LLM 调用 | 需要 LLM 调用进行生成式摘要           |
> | **信息损失** | 极小 — 原始消息始终保留，摘要只是"指针"        | 显著 — 原始上下文永久丢失             |
> | **事实恢复** | `retrieve_fact` 调用可按需获取原始消息分块     | 无法恢复丢失的细节                    |
> | **设计目的** | 适配上下文预算，同时保留召回能力               | 压缩以适应上下文窗口                  |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  DKI Recall v4 记忆召回架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 多信号召回                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  用户输入 → [关键词+权重] + [指代解析] + [向量相似度]              │    │
│  │           →  加权合并 + 归一化  →  关联消息列表                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ↓                                              │
│  Phase 2: 动态 History 构造                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  遍历消息列表:                                                   │    │
│  │    超阈值 → [SUMMARY] + trace_id (可追溯)                        │    │
│  │    阈值内 → 原始消息                                             │    │
│  │  + 最近 N 轮完整对话                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ↓                                              │
│  Phase 3: 模型适配 + 推理                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  [History 后缀] + [可信+推理限定提示] + [偏好 K/V 注入] + [Query] │    │
│  │  → LLM 推理 → 检测 retrieve_fact 调用                            │    │
│  │  → 事实补充 (支持分块 offset+limit) → 继续推理                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  优势:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ✅ 长历史场景下稳定可靠                                         │    │
│  │  ✅ 事实可追溯 (trace_id → 原始消息)                             │    │
│  │  ✅ 动态上下文预算管理                                           │    │
│  │  ✅ 多模型支持 (DeepSeek, GLM, Generic)                         │    │
│  │  ✅ 偏好仍通过 K/V 注入 (复用现有基础设施)                        │    │
│  │  ✅ 默认抽取式摘要 (无需额外 LLM 调用)                           │    │
│  │  ✅ retrieve_fact 支持按需细节恢复                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**配置示例**:

```yaml
dki:
    injection_strategy: "recall_v4"

    recall:
        enabled: true
        strategy: "summary_with_fact_call"

        signals:
            keyword_enabled: true
            keyword_topk: 5
            keyword_method: "tfidf"
            vector_enabled: true
            vector_top_k: 10

        budget:
            generation_reserve: 512
            min_recent_turns: 2
            max_recent_turns: 5

        summary:
            per_message_threshold: 200
            strategy: "extractive" # extractive (jieba TextRank) | llm

        fact_call:
            enabled: true
            max_rounds: 3
            max_fact_tokens: 800
```

**回退机制**:

```
recall_v4 执行失败
    ↓ 自动回退
stable (混合注入: 偏好 K/V + 历史后缀提示词)
    ↓ 如果也失败
纯 LLM 推理 (无注入)
```

#### 为什么不需要 Rolling Summary

与 ChatGPT/Claude/Grok 不同，DKI **不需要** Rolling Summary：

| 方案            | 原因                     | DKI 替代方案                |
| --------------- | ------------------------ | --------------------------- |
| RAG+Prompt      | 上下文窗口限制，需要压缩 | K/V 注入不消耗上下文        |
| Rolling Summary | 压缩导致信息丢失         | Memory Trigger 精准召回     |
| 摘要生成        | 额外的 LLM 调用开销      | Reference Resolver 按需检索 |

**核心洞察**: DKI 的 Recall v4 摘要是一种**单消息粒度、按需触发、可追溯**的操作 — 而非全局性的、持续的、有损压缩。原始消息始终被保留，当模型需要具体细节时，可以通过 `retrieve_fact` 按需获取。这与 Rolling Summary 永久丢弃原始上下文有着本质区别。

## 🚀 快速开始

> 📖 **完整部署指南**：请参阅 [DKI+AGA 完整部署指南](docs/DKI_AGA_Complete_Deployment_Guide.md)

### 安装

```bash
cd DKI

# Windows:
scripts\setup.bat

# Linux/Mac:
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 集成方式 (v4.0 — 极简集成)

DKI v4.0 提供三层集成模式：

#### Level 1: One-Line Factory（最简集成，推荐）

```python
from dki.integration import create_plugin

# 1. 创建 DKI 插件 (自动管理模型、缓存、适配器)
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")

# 2. 普通聊天
response = await dki.chat(
    query="今晚想找一家餐厅，有什么新推荐吗？",
    user_id="user_001",
    session_id="session_001",
)
print(response.text)

# 3. 流式聊天
async for chunk in dki.chat_stream(
    query="推荐一本好书",
    user_id="user_001",
    session_id="session_001",
):
    if chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\n完成! 延迟: {chunk['latency_ms']:.1f}ms")
```

#### Level 2: FastAPI Middleware（Web 应用集成）

```python
from fastapi import FastAPI
from dki.integration import DKIMiddleware, get_dki_plugin

app = FastAPI()

# 一行添加 DKI 支持 (自动初始化、生命周期管理)
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

#### Level 3: EnhancedDKIPlugin（完全控制 + 动态路由 + 消息管理）

```python
from dki.integration import create_plugin
from dki.integration.enhanced_plugin import (
    EnhancedDKIPluginConfig,
    DynamicRouterConfig,
    MessageManagementConfig,
)

# 启用动态路由 + 消息管理
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
    rag_system=your_rag_system,  # RAG 系统实例
    store=your_store,            # 数据持久化实例
)

# 调用 chat — 自动路由到 RAG 或 DKI
response = await dki.chat(
    query="今晚想找一家餐厅",
    user_id="user_001",
    session_id="session_001",
)
# 消息自动写入数据库，无需上层应用处理
```

#### 旧版集成方式 (仍然支持)

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)
response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_001",
    session_id="session_001",
)
```

### 适配器配置示例

创建 `config/adapter_config.yaml`，指定如何连接上层应用的数据库：

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
        content_json_key: null # 支持 JSON 内容提取

    vector_search:
        type: dynamic
        dynamic:
            strategy: hybrid # BM25 + embedding
```

### Chat UI 示例界面

DKI 提供了一个基于 Vue3 + Element Plus 的示例 Chat UI：

```bash
# 同时启动前后端
python start_dev.py

# 仅启动后端
python start_dev.py backend

# 仅启动前端
python start_dev.py frontend
```

**UI 功能特性**：

-   🔐 用户登录/注册/密码管理
-   💬 支持 Markdown 渲染的聊天界面
-   ⚡ 流式/普通生成模式切换
-   📍 对话锚点导航（快速跳转到指定对话轮次）
-   ⬆️⬇️ 动态滚动按钮（回到顶部/底部）
-   ⚙️ 用户偏好管理（增删改查）
-   📊 会话历史管理
-   📈 系统统计监控
-   🎨 浅色/深色主题切换

### 监控 API

```python
# 获取统计数据
stats = dki.get_stats()
print(f"总请求数: {stats['total_requests']}")
print(f"注入率: {stats['injection_rate']:.2%}")
print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")

# 获取增强统计 (v4.0)
if hasattr(dki, 'get_stats'):
    enhanced_stats = dki.get_stats()
    print(f"DKI 路由次数: {enhanced_stats.get('enhanced', {}).get('dki_routes', 0)}")
    print(f"RAG 路由次数: {enhanced_stats.get('enhanced', {}).get('rag_routes', 0)}")
```

REST API 端点：

| 端点                  | 方法 | 描述                |
| --------------------- | ---- | ------------------- |
| `/v1/dki/chat`        | POST | DKI 增强聊天        |
| `/v1/dki/chat/stream` | POST | DKI 流式聊天 (v4.0) |
| `/v1/dki/info`        | GET  | 获取 DKI 插件状态   |
| `/api/health`         | GET  | 健康检查            |
| `/api/health/detail`  | GET  | 详细健康检查 (v4.0) |
| `/api/stats`          | GET  | 获取系统统计信息    |

## 📁 项目结构

```
DKI/
├── config/                              # 配置文件目录
│   ├── config.yaml                      # ⭐ 主配置文件
│   ├── adapter_config.example.yaml      # ⭐ 适配器配置示例
│   ├── memory_trigger.yaml              # Memory Trigger 配置
│   └── reference_resolver.yaml          # Reference Resolver 配置
│
├── dki/                                 # 核心代码目录
│   ├── __init__.py                      # v4.0 入口 (create_plugin, DKIMiddleware)
│   │
│   ├── integration/                     # ⭐ v4.0 集成层 (新)
│   │   ├── __init__.py                  # 集成层入口
│   │   ├── factory.py                   # ⭐ create_plugin() 工厂函数
│   │   ├── middleware.py                # ⭐ DKIMiddleware (FastAPI)
│   │   └── enhanced_plugin.py           # ⭐ EnhancedDKIPlugin (路由+消息管理)
│   │
│   ├── core/                            # ⭐ 核心模块
│   │   ├── __init__.py
│   │   ├── dki_plugin.py                # ⭐ DKI 插件核心 (入口)
│   │   ├── dki_system.py                # DKI 系统封装 (实验用)
│   │   ├── rag_system.py                # RAG 系统 (异步+流式, v4.0)
│   │   ├── conversation_router.py       # ⭐ RAG/DKI 动态路由 (v4.0)
│   │   ├── plugin_manager.py            # 多实例管理 (v4.0)
│   │   ├── exceptions.py                # ⭐ 结构化异常体系 (v4.0)
│   │   ├── rate_limiter.py              # 限流/熔断 (v4.0)
│   │   ├── memory_router.py             # 基于 FAISS 的向量检索
│   │   ├── embedding_service.py         # 嵌入计算服务
│   │   │
│   │   ├── injection/                   # 注入策略
│   │   │   └── full_attention_injector.py  # Full Attention (研究, 已弃用)
│   │   │
│   │   ├── recall/                      # ⭐ Recall v4 记忆召回
│   │   │   └── ...
│   │   │
│   │   └── components/                  # 核心算法组件
│   │       ├── memory_influence_scaling.py    # MIS
│   │       ├── query_conditioned_projection.py  # QCP
│   │       ├── dual_factor_gating.py          # 双因子门控
│   │       ├── memory_trigger.py              # 记忆触发检测
│   │       ├── reference_resolver.py          # 指代解析器
│   │       ├── tiered_kv_cache.py             # L1/L2/L3/L4 分层缓存
│   │       └── ...
│   │
│   ├── adapters/                        # 外部数据适配器
│   │   ├── config_driven_adapter.py     # ⭐ 配置驱动适配器 (核心)
│   │   ├── postgresql_adapter.py        # PostgreSQL
│   │   ├── mysql_adapter.py             # MySQL
│   │   └── ...
│   │
│   ├── models/                          # LLM 模型适配器
│   │   ├── factory.py                   # 模型工厂 (支持命名空间)
│   │   ├── base.py                      # 基础适配器 (同步/异步/流式)
│   │   ├── vllm_adapter.py              # vLLM 适配器 (流式)
│   │   ├── sglang_adapter.py            # SGLang 适配器 (流式)
│   │   ├── llama_adapter.py             # LLaMA 适配器 (流式)
│   │   ├── deepseek_adapter.py          # DeepSeek 适配器 (流式)
│   │   └── glm_adapter.py              # GLM 适配器 (流式)
│   │
│   ├── cache/                           # 缓存系统
│   │   ├── preference_cache.py          # 偏好缓存管理 (L1+L2)
│   │   ├── redis_client.py              # Redis 分布式缓存
│   │   └── non_vectorized_handler.py    # 动态向量处理
│   │
│   ├── config/                          # 配置加载
│   │   └── config_loader.py             # YAML 配置加载器 (热重载)
│   │
│   ├── attention/                       # FlashAttention 集成
│   │   └── ...
│   │
│   └── experiment/                      # 实验系统
│       └── ...
│
├── demo/                                # ⭐ 示例应用
│   ├── app.py                           # FastAPI 应用 (使用集成层)
│   ├── api/                             # API 路由
│   │   ├── auth.py                      # 认证 (登录/注册/密码管理)
│   │   └── chat.py                      # 聊天 (普通+流式)
│   └── store/                           # 数据持久化
│       ├── base.py                      # IChatStore 接口
│       └── base_impl.py                 # SQLite/PostgreSQL 实现
│
├── ui/                                  # Vue3 示例前端
│   └── src/
│       ├── views/
│       │   ├── ChatView.vue             # 聊天 (流式+锚点+滚动)
│       │   ├── LoginView.vue            # 登录/注册
│       │   └── ProfileView.vue          # 用户资料管理
│       ├── components/
│       │   └── SettingsDialog.vue       # 设置 (含流式开关)
│       └── stores/
│           └── settings.ts              # 设置状态 (含 streamingEnabled)
│
├── tests/                               # 测试
│   └── unit/
│       ├── test_integration_layer.py    # 集成层测试 (v4.0)
│       ├── test_p0p1_exceptions.py      # 异常体系测试
│       ├── test_p1_rate_limiter.py      # 限流/熔断测试
│       ├── test_p1_config_reload.py     # 配置热重载测试
│       ├── test_user_management.py      # 用户管理测试
│       ├── test_streaming_chat.py       # 流式聊天测试
│       ├── test_rag_system_v6.py        # RAG 系统测试
│       └── ...
│
├── scripts/                             # 脚本
│   ├── setup.bat / setup.sh             # 安装脚本
│   ├── init_db.sql                      # SQLite 数据库初始化
│   ├── init_db_postgresql.sql           # PostgreSQL 数据库初始化
│   └── start_vllm_with_tools.sh         # vLLM Function Calling 启动
│
├── docs/                                # 文档
│   ├── DKI_AGA_Complete_Deployment_Guide.md  # 完整部署指南
│   ├── 文件上传与Skills支持方案.md             # 文件上传方案 (v4.0)
│   └── ...
│
├── examples/                            # 示例
│   └── vllm_function_calling_web_search.py  # vLLM 原生网络搜索示例
│
├── start_dev.py                         # 开发启动脚本
├── requirements.txt                     # Python 依赖
├── QUICKSTART.md                        # 快速开始
├── README_CN.md                         # 中文文档
└── README.md                            # 英文文档
```

## 📊 项目状态

| 模块                 | 状态      | 说明                                            |
| -------------------- | --------- | ----------------------------------------------- |
| DKI 核心插件         | ✅ 完成   | K/V 注入、混合策略、门控决策                    |
| v4.0 集成层          | ✅ 完成   | create_plugin、DKIMiddleware、EnhancedDKIPlugin |
| 动态路由             | ✅ 完成   | RAG/DKI 五维评分自动切换                        |
| 消息管理             | ✅ 完成   | 消息写入 + 偏好写入                             |
| 流式生成             | ✅ 完成   | 所有模型适配器支持异步流式                      |
| 结构化异常           | ✅ 完成   | 三级降级链                                      |
| 限流/熔断            | ✅ 完成   | 外置配置、热重载                                |
| Recall v4 记忆召回   | ✅ 完成   | 多信号检索 + 动态摘要 + 事实补充                |
| 配置驱动适配器       | ✅ 完成   | SQLAlchemy 动态表映射                           |
| Redis 分布式缓存     | ✅ 完成   | L1+L2 缓存                                      |
| FlashAttention       | ✅ 完成   | FA3/FA2 自动检测                                |
| 用户管理             | ✅ 完成   | 注册/登录/密码修改/找回                         |
| Vue3 示例 UI         | ✅ 完成   | 聊天/流式/锚点/偏好/统计                        |
| 单元测试             | ✅ 完成   | 核心组件测试覆盖                                |
| 注意力热力图         | 🔄 规划中 | 调试用注意力权重可视化                          |
| 文件上传/Skills      | 🔄 规划中 | 文本文件上传和技能支持                          |
| LangChain/LlamaIndex | 📋 待定   | 生态集成                                        |

## ⚙️ 配置

### DKI 主配置

编辑 `config/config.yaml`:

```yaml
# 模型引擎
model:
    default_engine: "vllm" # vllm, sglang, llama, deepseek, glm
    engines:
        vllm:
            model_name: "Qwen/Qwen2-7B-Instruct"
            tensor_parallel_size: 1

# DKI 插件设置
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

# v4.0: 限流配置
rate_limit:
    enabled: true
    max_rpm: 30
    max_concurrent: 3
    burst_size: 5

# v4.0: 熔断配置
circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 30
```

### v4.0 增强配置

在 Demo App 配置中启用动态路由和消息管理：

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

## 🔬 研究背景

### DKI 的定位：用户级记忆系统

| 维度         | RAG            | DKI                                         |
| ------------ | -------------- | ------------------------------------------- |
| **目标数据** | 外部知识库     | 用户偏好、会话历史                          |
| **数据规模** | 大（数千文档） | 小到中（偏好 50-200，历史 100-4000 tokens） |
| **更新频率** | 批量更新       | 每会话实时更新                              |
| **隐私**     | 共享知识       | 用户自有数据                                |
| **缓存**     | 文档级         | 用户级（高复用）                            |

### Token 预算分析

**RAG 范式：**

```
[检索内容（已消耗）] [用户输入（剩余空间）]
Token预算: B_t^used = n_m + n_u
```

**DKI 范式：**

```
[用户输入（完整预算可用）]
     ↑ 记忆通过K/V注入（不占用token预算）
Token预算: B_t^used = n_u（记忆免费！）
```

### 性能基准

基于 DeepSeek-V3 7B 的模拟实验（n=500）：

| 指标         | RAG    | DKI        | 变化       |
| ------------ | ------ | ---------- | ---------- |
| 记忆召回率   | 87.3%  | 86.2%      | -1.1%      |
| 首轮延迟     | 78.8ms | 92.4ms     | +17.3%     |
| **后续延迟** | 76.1ms | **42.8ms** | **-43.7%** |
| 缓存命中率   | N/A    | 69.7%      | -          |

### 设计不变量

1. **存储模型无关**：仅存储原始文本 + 路由向量
2. **注入模型一致**：使用目标模型参数计算 K/V
3. **会话缓存可丢弃**：推理时增强，非持久化记忆
4. **三级优雅降级**：recall_v4 → stable → 纯 LLM
5. **用户级隔离**：HMAC 签名缓存键 + 推理后 K/V 清理
6. **审计日志**：所有注入决策均记录

## 📚 常见问题

**Q: DKI 需要重新训练模型吗？**  
A: 不需要。DKI 是推理时增强，使用冻结的模型参数。

**Q: DKI 和 RAG 有什么区别？**  
A: RAG 在 token 层级拼接检索内容，消耗上下文窗口；DKI 在注意力层级注入 K/V，不消耗 token 预算。它们是互补的。

**Q: v4.0 的动态路由是什么？**  
A: `ConversationRouter` 基于五维评分模型（历史深度、偏好匹配、触发信号、会话深度、跨会话关联）自动决定使用 RAG 还是 DKI 处理当前查询。

**Q: 如何集成 DKI v4.0？**  
A: 最简方式只需 3 行代码：

```python
from dki.integration import create_plugin
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")
response = await dki.chat("用户输入", user_id="u1", session_id="s1")
```

**Q: v4.0 支持流式生成吗？**  
A: 支持。所有模型适配器（vLLM、SGLang、LLaMA、DeepSeek、GLM）均已实现 `async_stream_generate`，通过 `dki.chat_stream()` 调用。

**Q: 上层应用需要做什么修改？**  
A: 提供适配器配置文件，调用 `create_plugin()` 创建插件，然后调用 `dki.chat()` 即可。

**Q: 生产环境部署建议？**  
A:

1. 启用 Redis 分布式缓存
2. 启用限流和熔断
3. 配置动态路由（RAG + DKI 互补）
4. 监控注入率和延迟
5. 根据指标调整 alpha 和缓存策略

## 📄 许可证

MIT 许可证 - 详见 LICENSE 文件

## 🤝 贡献

欢迎贡献！请先阅读我们的贡献指南。

---

**DKI v4.0** - 在注意力层级重新思考记忆增强
