# DKI - Dynamic KV Injection 动态键值注入--大语言模型用户级记忆系统

> 大型语言模型的注意力层级用户记忆插件

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

## 📖 概述

DKI (Dynamic KV Injection，动态键值注入) 是一个**LLM 注意力层级插件**，通过 Attention Hook 在推理时动态注入用户偏好和会话历史，实现跨会话的个性化记忆。

### DKI 是什么

DKI 是一个 **LLM 插件**，专为**用户级记忆**设计：

-   **注意力 Hook 机制**：通过 PyTorch Hook 在注意力层级注入 K/V，而非 prompt 拼接
-   **配置驱动适配器**：自动读取上层应用的数据库，无需修改上层应用代码
-   **混合注入策略**：偏好 K/V 注入（负位置）+ 历史后缀提示词（正位置）

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

### 为什么这个定位很重要

这种聚焦的范围带来以下优势：

1. **短偏好**（50-200 tokens）→ 降低位置编码风险，可缓存
2. **用户自有数据** → 简化隐私考量
3. **会话连贯** → 有效的 K/V 缓存
4. **稳定偏好** → 高缓存复用率

### 核心特性

-   **🧠 注意力 Hook 注入**：通过 PyTorch Hook 在注意力层级注入 K/V，而非 prompt token
-   **🔀 混合注入策略**：偏好（K/V 负位置）+ 历史（后缀提示词正位置）
-   **🔧 配置驱动适配器**：SQLAlchemy 动态表映射，无需上层应用实现接口
-   **🎚️ 记忆影响缩放（MIS）**：连续的 α ∈ [0, 1] 强度控制
-   **🔄 查询条件投影**：FiLM 风格的记忆中心变换
-   **🚦 双因子门控**：相关性驱动决策，熵调制强度
-   **💾 分层 KV 缓存**：L1(GPU) → L2(CPU) → L3(SSD) → L4(重计算)
-   **📊 监控 API**：统计数据、注入日志、健康检查
-   **🔌 多引擎支持**：vLLM、LLaMA、DeepSeek、GLM
-   **✅ 优雅降级**：α → 0 平滑回退到普通 LLM 行为

## 🏗️ 架构

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
│  │  DKI 插件                                                       │    │
│  │  ├── 配置驱动适配器 (SQLAlchemy 动态表映射)                       │    │
│  │  │   └── 读取上层应用数据库 (用户偏好 + 历史消息)                  │    │
│  │  ├── 偏好处理 → K/V 注入 (负位置, Attention Hook)                 │    │
│  │  ├── 历史处理 → 后缀提示词 (正位置)                               │    │
│  │  └── 监控 API (统计/日志/健康检查)                                │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LLM 引擎 (vLLM / LLaMA / DeepSeek / GLM)                       │    │
│  │  └── 带 K/V 注入的推理                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 混合注入策略

DKI 使用**分层注入方法**，模拟人类认知：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DKI 混合注入架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  第1层：用户偏好（K/V 注入 - Attention Hook）                     │    │
│  │  ├── 内容：饮食、风格、兴趣                                       │    │
│  │  ├── 位置：负位置（概念上在用户输入"之前"）                        │    │
│  │  ├── 机制：PyTorch Hook 修改 Attention 的 K/V                    │    │
│  │  ├── 影响：隐式、背景（如同人格）                                 │    │
│  │  └── α：0.3-0.5（较低，用于微妙影响）                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  第2层：会话历史（后缀提示词）                                     │    │
│  │  ├── 内容：近期对话轮次                                           │    │
│  │  ├── 位置：用户查询之后（正位置）                                  │    │
│  │  ├── 机制：标准 token 拼接                                        │    │
│  │  ├── 影响：显式、可引用（如同记忆）                                │    │
│  │  └── 提示词：建立信任的引导                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  第3层：当前查询（标准输入）                                       │    │
│  │  └── 注意力的主要焦点                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DKI 数据流                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  用户查询 + user_id + session_id                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  1. 配置驱动适配器读取上层应用数据库                               │    │
│  │     ├── 用户偏好表 → 偏好列表                                     │    │
│  │     └── 消息表 → 相关历史 (向量检索/BM25/关键词)                   │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  2. 偏好处理                                                     │    │
│  │     ├── 格式化偏好文本                                            │    │
│  │     ├── 计算/缓存 K/V 表示                                        │    │
│  │     └── 准备 Attention Hook                                      │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  3. 历史处理                                                     │    │
│  │     └── 格式化为后缀提示词 (带信任引导)                            │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  4. LLM 推理 (带 K/V 注入)                                       │    │
│  │     ├── Attention Hook 注入偏好 K/V (负位置)                     │    │
│  │     └── 输入 = 查询 + 历史后缀 (正位置)                           │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  5. 返回响应 + 记录监控数据                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
cd DKI

# 安装（创建虚拟环境、安装依赖、初始化数据库）
# Windows:
scripts\setup.bat

# Linux/Mac:
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 上层应用集成 (推荐方式)

DKI 作为 LLM 插件，上层应用只需：

1. **提供适配器配置文件** - 指定数据库连接和字段映射
2. **删除 RAG/Prompt 工程代码** - DKI 自动处理
3. **传递 user_id + 原始输入** - 无需任何 prompt 构造

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

# 1. 初始化 LLM 适配器
model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")

# 2. 从配置文件创建 DKI 插件 (推荐)
# 上层应用只需提供配置文件，无需实现任何接口
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",  # 指定数据库连接和字段映射
)

# 3. 调用 DKI - 只需传递 user_id 和原始输入
# DKI 会自动:
# - 通过适配器读取上层应用数据库的用户偏好 → K/V 注入
# - 通过适配器检索相关历史消息 → 后缀提示词
response = await dki.chat(
    query="今晚想找一家餐厅，有什么新推荐吗？",  # 原始用户输入，无需任何 prompt 构造
    user_id="user_001",   # 用户标识 (DKI 用于读取偏好和历史)
    session_id="session_001",  # 会话标识 (DKI 用于读取会话历史)
)

print(response.text)
# 输出会考虑：
# - 素食偏好（隐式，来自 K/V 注入）
# - 之前的餐厅访问（显式，来自历史后缀）
# - 北京位置（隐式，来自 K/V 注入）

# 监控数据
print(f"注入启用: {response.metadata.injection_enabled}")
print(f"Alpha值: {response.metadata.alpha}")
print(f"偏好 tokens: {response.metadata.preference_tokens}")
print(f"历史 tokens: {response.metadata.history_tokens}")
print(f"缓存命中: {response.metadata.preference_cache_hit}")
print(f"延迟: {response.metadata.latency_ms}ms")
```

### 适配器配置示例

创建 `config/adapter_config.yaml`，指定如何连接上层应用的数据库：

```yaml
user_adapter:
    # 数据库连接 (连接到上层应用的数据库)
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        port: 5432
        database: my_app_db # 上层应用的数据库
        username: user
        password: pass

    # 用户偏好表映射 (映射到上层应用的表结构)
    preferences:
        table: user_preferences # 上层应用的表名
        fields:
            user_id: user_id # 上层应用的字段名
            preference_text: content
            preference_type: type
            priority: priority

    # 消息表映射
    messages:
        table: chat_messages
        fields:
            message_id: id
            session_id: session_id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at

    # 向量检索配置 (支持动态向量处理)
    vector_search:
        type: dynamic # pgvector | faiss | dynamic
        dynamic:
            strategy: hybrid # lazy | batch | hybrid (BM25 + embedding)
```

### Chat UI 示例界面

DKI 提供了一个基于 Vue3 + Element Plus 的示例 Chat UI，用于演示 DKI 集成：

```bash
# 同时启动前后端
python start_dev.py

# 仅启动后端
python start_dev.py backend

# 仅启动前端
python start_dev.py frontend
```

**UI 功能特性**：

-   🔐 用户登录/注册
-   💬 支持 Markdown 渲染的聊天界面
-   ⚙️ 用户偏好管理（增删改查）
-   📊 会话历史管理
-   📈 系统统计监控（需管理员密码）
-   🎨 浅色/深色主题切换

**注意**：Chat UI 是一个**示例应用**，用于演示 DKI 集成。DKI 的适配器会读取 Chat UI 的数据库来获取用户偏好和历史消息。

**统计页面鉴权**：
统计页面使用简单密码保护（非生产标准），可在配置中设置：

```bash
# 环境变量
VITE_STATS_PASSWORD=your_password
```

### 监控 API

DKI 提供监控 API 用于查看内部工作状态：

```python
# 获取统计数据
stats = dki.get_stats()
print(f"总请求数: {stats['total_requests']}")
print(f"注入率: {stats['injection_rate']:.2%}")
print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
print(f"平均延迟: {stats['avg_latency_ms']:.1f}ms")

# 获取注入日志
logs = dki.get_injection_logs(limit=10)
for log in logs:
    print(f"[{log['timestamp']}] alpha={log['alpha']:.2f}, enabled={log['injection_enabled']}")
```

REST API 端点：

-   `GET /v1/dki/info` - 获取 DKI 插件状态
-   `POST /v1/dki/chat` - DKI 增强聊天 (上层应用调用此接口)

## 📁 项目结构

```
DKI/
├── config/
│   ├── config.yaml              # 主配置文件
│   └── adapter_config.example.yaml  # 适配器配置示例
├── dki/
│   ├── core/
│   │   ├── dki_plugin.py        # ⭐ DKI 插件核心
│   │   ├── dki_system.py        # DKI 系统封装
│   │   ├── memory_router.py     # 基于FAISS的检索
│   │   ├── embedding_service.py
│   │   └── components/
│   │       ├── memory_influence_scaling.py   # MIS 组件
│   │       ├── query_conditioned_projection.py  # QCP 组件
│   │       ├── dual_factor_gating.py         # 双因子门控
│   │       ├── hybrid_injector.py            # 混合注入器
│   │       ├── session_kv_cache.py
│   │       ├── tiered_kv_cache.py    # L1/L2/L3/L4 内存层次
│   │       └── position_remapper.py  # 位置编码重映射
│   ├── adapters/
│   │   ├── base.py              # 适配器基类
│   │   └── config_driven_adapter.py  # ⭐ 配置驱动适配器
│   ├── api/
│   │   └── dki_routes.py        # ⭐ DKI API 路由
│   ├── models/
│   │   ├── factory.py           # 模型工厂
│   │   ├── base.py              # 基础适配器
│   │   ├── vllm_adapter.py
│   │   ├── llama_adapter.py
│   │   ├── deepseek_adapter.py
│   │   └── glm_adapter.py
│   ├── cache/
│   │   └── non_vectorized_handler.py  # 动态向量处理
│   ├── database/
│   │   ├── models.py            # SQLAlchemy模型
│   │   └── connection.py        # 数据库连接管理
│   ├── experiment/
│   │   ├── runner.py            # 实验运行器
│   │   ├── metrics.py           # 评估指标
│   │   └── data_generator.py    # 测试数据生成
│   └── web/
│       └── app.py               # FastAPI 应用
├── ui/                          # Vue3 示例前端界面
│   ├── src/
│   │   ├── views/               # 页面组件
│   │   ├── components/          # 通用组件
│   │   ├── stores/              # Pinia 状态管理
│   │   ├── services/            # API 服务
│   │   └── types/               # TypeScript 类型
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── Integration_Guide.md     # 集成指南
│   └── Dynamic_Vector_Search.md # 动态向量检索说明
├── tests/
│   ├── unit/                    # 单元测试
│   └── integration/             # 集成测试
├── scripts/
│   ├── setup.bat/.sh            # 安装脚本
│   └── start.bat/.sh            # 启动脚本
├── start_dev.py                 # 开发启动脚本
├── requirements.txt
└── README.md
```

## ⚙️ 配置

### DKI 主配置

编辑 `config/config.yaml`:

```yaml
# 模型引擎
model:
    default_engine: "vllm" # vllm, llama, deepseek, glm
    engines:
        vllm:
            model_name: "Qwen/Qwen2-7B-Instruct"
            tensor_parallel_size: 1

# DKI 插件设置
dki:
    enabled: true
    version: "2.5"

    # 混合注入策略
    hybrid_injection:
        enabled: true
        language: "cn" # en | cn

        # 偏好：K/V 注入（Attention Hook，负位置）
        preference:
            enabled: true
            position_strategy: "negative"
            alpha: 0.4 # 较低用于背景影响
            max_tokens: 200

        # 历史：后缀提示词（正位置）
        history:
            enabled: true
            method: "suffix_prompt"
            max_tokens: 2000
            max_messages: 10

    # 门控：相关性驱动，熵调制
    gating:
        relevance_threshold: 0.7
        entropy_ceiling: 1.0
        entropy_floor: 0.5

    # 安全设置
    safety:
        max_alpha: 0.8
        fallback_on_error: true
        audit_logging: true
```

### 适配器配置 (核心)

创建 `config/adapter_config.yaml`，配置如何连接上层应用的数据库：

```yaml
user_adapter:
    # 数据库连接 - 连接到上层应用的数据库
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        port: 5432
        database: my_app_db # 上层应用的数据库名
        username: user
        password: pass
        pool_size: 5

    # 用户偏好表映射 - 映射到上层应用的表结构
    preferences:
        table: user_preferences # 上层应用的偏好表名
        fields:
            user_id: user_id # 字段映射: DKI字段 -> 上层应用字段
            preference_text: content
            preference_type: type
            priority: priority
            created_at: created_at
        filters:
            is_active: true # 额外过滤条件

    # 消息表映射
    messages:
        table: chat_messages # 上层应用的消息表名
        fields:
            message_id: id
            session_id: session_id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at
            embedding: embedding # 向量字段 (可选)

    # 向量检索配置
    vector_search:
        enabled: true
        type: dynamic # pgvector | faiss | dynamic | none

        # 动态向量处理 (无预计算向量时使用)
        dynamic:
            strategy: hybrid # lazy | batch | hybrid
            # hybrid = BM25 初筛 + embedding 重排序

        # 检索参数
        top_k: 10
        similarity_threshold: 0.5

    # 缓存配置
    cache_enabled: true
    cache_ttl: 300 # 5 分钟
```

### 配置说明

**适配器配置的核心理念**：

-   上层应用**无需实现任何接口**
-   只需提供配置文件，指定数据库连接和字段映射
-   DKI 使用 SQLAlchemy 动态反射表结构
-   支持 PostgreSQL、MySQL、SQLite 等主流数据库

## 📊 实验

### 生成测试数据

```bash
python -m dki.experiment.data_generator
```

### 运行对比实验

```python
from dki.experiment.runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()
config = ExperimentConfig(
    name="DKI vs RAG 对比",
    modes=["dki", "rag", "baseline"],
    datasets=["persona_chat", "memory_qa"],
    max_samples=100
)

results = runner.run_experiment(config)
```

### Alpha 敏感性分析

```python
results = runner.run_alpha_sensitivity(
    alpha_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
```

## 📈 API 参考

### DKI 插件 API

| 端点           | 方法 | 描述                              |
| -------------- | ---- | --------------------------------- |
| `/v1/dki/chat` | POST | DKI 增强聊天 (上层应用调用此接口) |
| `/v1/dki/info` | GET  | 获取 DKI 插件状态                 |

### 监控 API

| 端点             | 方法 | 描述              |
| ---------------- | ---- | ----------------- |
| `/api/stats`     | GET  | 获取系统统计信息  |
| `/api/stats/dki` | GET  | 获取 DKI 统计信息 |
| `/api/health`    | GET  | 健康检查          |

### DKI 聊天请求

上层应用只需传递 `user_id` 和原始输入：

```json
{
    "query": "推荐一家餐厅",
    "user_id": "user_001",
    "session_id": "session_001",
    "temperature": 0.7,
    "max_tokens": 512
}
```

### DKI 聊天响应

```json
{
    "id": "dki-abc12345",
    "text": "根据您偏好素食的特点...",
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
                "content": "根据您偏好素食的特点..."
            },
            "finish_reason": "stop"
        }
    ],
    "created": 1707523200
}
```

## 🔬 研究背景

### DKI 的定位：用户级记忆系统

与 RAG 针对**外部知识**（文档、数据库、网页内容）不同，DKI 专为**用户级记忆**设计：

| 维度         | RAG            | DKI                                         |
| ------------ | -------------- | ------------------------------------------- |
| **目标数据** | 外部知识库     | 用户偏好、会话历史                          |
| **数据规模** | 大（数千文档） | 小到中（偏好 50-200，历史 100-4000 tokens） |
| **更新频率** | 批量更新       | 每会话实时更新                              |
| **隐私**     | 共享知识       | 用户自有数据                                |
| **缓存**     | 文档级         | 用户级（高复用）                            |

> **注意**：DKI 的 token 数量取决于会话复杂度和相关性。偏好注入建议保持短小（50-200 tokens），而历史注入可根据需要扩展至 4000+ tokens。长历史建议启用相关性过滤以优化性能。

这种聚焦的范围是**有意为之**的，它使 DKI 的核心优势得以实现。

### Token 预算分析

DKI 解决了 RAG 的一个根本性限制：检索到的内容会消耗上下文窗口容量。

**RAG 范式：**

```
[检索内容（已消耗）] [用户输入（剩余空间）]
Token预算: B_t^used = n_m + n_u
注意力预算: B_a = (n_m + n_u)²
```

**DKI 范式：**

```
[用户输入（完整预算可用）]
     ↑ 记忆通过K/V注入（不占用token预算）
Token预算: B_t^used = n_u（记忆免费！）
注意力预算: B_a = n_u × (n_m + n_u)
```

### DKI vs 交叉注意力

DKI **不等同于**交叉注意力：

| 特性   | DKI              | 交叉注意力                             |
| ------ | ---------------- | -------------------------------------- |
| 参数   | 重用 W_k, W_v    | 独立的 W_q^cross, W_k^cross, W_v^cross |
| 训练   | 无需训练         | 需要训练                               |
| 架构   | 无需修改         | 专用层                                 |
| 兼容性 | 任何仅解码器 LLM | 仅编码器-解码器                        |
| 控制   | 连续 α           | 学习权重                               |

### 混合注入策略的原理

为什么对偏好和历史使用不同的策略？

| 记忆类型 | 特点                                | 策略                 | 原因                          |
| -------- | ----------------------------------- | -------------------- | ----------------------------- |
| **偏好** | 短（50-200 tokens），稳定，抽象     | K/V 注入（负位置）   | OOD 风险低，可缓存，隐式影响  |
| **历史** | 可变（100-4000 tokens），动态，具体 | 后缀提示词（正位置） | 零 OOD 风险，可引用，显式参考 |

这种分层方法：

-   最小化 OOD 风险（偏好很短）
-   支持历史引用（在提示词中可见）
-   减少幻觉（建立信任的提示词）

### Token 数量与性能影响

DKI 支持的 token 范围取决于会话复杂度和相关性：

| Token 范围 | 适用场景             | 延迟影响 | 显存占用（7B 模型） |
| ---------- | -------------------- | -------- | ------------------- |
| 100-500    | 简单偏好 + 短历史    | < 10%    | ~250MB              |
| 500-2000   | 中等复杂度会话       | 10-30%   | ~1GB                |
| 2000-4000  | 复杂多轮会话         | 30-50%   | ~2GB                |
| 4000+      | 长期会话（建议过滤） | > 50%    | > 2GB               |

**性能优化建议**：

-   对于长历史，启用 `search_relevant_history` 进行相关性过滤
-   偏好保持短小（50-200 tokens），利用 K/V 缓存
-   历史使用后缀提示词，可根据需要动态调整长度

**与 RAG 的显存对比**：

-   相同 token 数量下，DKI 和 RAG 显存占用**基本相同**
-   DKI 可能因缓存多轮历史而占用更多显存
-   但 DKI 的 K/V 缓存可复用，后续请求无需重计算

### 设计不变量

1. **存储模型无关**：仅存储原始文本 + 路由向量
2. **注入模型一致**：使用目标模型参数计算 K/V
3. **会话缓存可丢弃**：推理时增强，非持久化记忆
4. **优雅降级**：α → 0 回退到普通 LLM
5. **审计日志**：所有注入决策均记录以符合合规要求

### 内存层次（分层 KV 缓存）

```
┌───────────────────────────────────────────────────┐
│  L1: GPU HBM (热数据)   - 未压缩 FP16              │
│  L2: CPU RAM (温数据)   - 压缩 (2-4×)              │
│  L3: NVMe SSD (冷数据)  - 量化 INT8 (8×)           │
│  L4: 仅文本            - 按需重计算                │
└───────────────────────────────────────────────────┘
```

内存占用与**活跃记忆**数量成比例，而非总语料库大小。

## 📄 许可证

MIT 许可证 - 详见 LICENSE 文件

## 🤝 贡献

欢迎贡献！请先阅读我们的贡献指南。

---

## 📚 详细文档

### 核心概念

#### 1. 注意力预算重分配假设

**假设陈述**：在推理密集型任务中，释放 token 预算的边际收益超过增加注意力计算的边际成本。

**数学表述**：

```
∂TaskSuccess/∂B_t^free > ∂Latency/∂B_a
```

**直觉**：

-   Token 预算是**硬约束**（截断导致信息丢失）
-   注意力预算是**软约束**（增加计算，但无信息丢失）
-   对于需要深度推理链的任务（多步骤数学、复杂规划），保留用于推理步骤的 token 预算提供的效用大于注意力延迟成本

#### 2. 记忆影响缩放（MIS）

连续强度控制 α ∈ [0, 1]：

| α 值        | 行为                   | 用例           |
| ----------- | ---------------------- | -------------- |
| 0.0         | 无记忆影响（普通 LLM） | 安全降级       |
| 0.2-0.4     | 温和注入               | 探索性对话     |
| **0.4-0.7** | **最优范围**           | **大多数场景** |
| 0.8-1.0     | 强注入                 | 高置信度记忆   |

**实现**：通过预 softmax 的 logit 偏置应用缩放：

```python
logit_bias = torch.log(torch.tensor(alpha + 1e-9))
logits_mem_scaled = logits_mem + logit_bias
```

#### 3. 查询条件投影（QCP）

使用 FiLM 风格的调制进行查询依赖的记忆投影：

```python
# 生成调制参数
gamma = gamma_net(query_context)  # 缩放
beta = beta_net(query_context)    # 偏移

# 低秩投影与FiLM调制
X_mem_low = X_mem @ W_mem
X_mem_modulated = X_mem_low * gamma + beta
X_mem_proj = proj_out(X_mem_modulated)

# 残差连接
return layer_norm(X_mem + X_mem_proj)
```

**关键原则**：投影是**记忆中心的**，而非查询中心的——查询仅调制，不重新编码记忆语义。

#### 4. 双因子门控

**设计决策**：注入决策由**相关性驱动**；不确定性仅**调制 α 的上限**。

```python
# 因子1: 记忆相关性（主要）
inject = similarity_top1 > threshold_relevance

# 因子2: 熵调制α上限（非决策）
alpha_max = entropy_floor + (entropy_ceiling - entropy_floor) * entropy

# 连续强度
alpha = min(alpha_base, alpha_max)
```

**重要说明**：我们使用注意力熵作为模型不确定性的**启发式代理**，而非严格的不确定性估计器。

### 与 RAG 的对比

| 维度       | RAG                            | DKI                  |
| ---------- | ------------------------------ | -------------------- |
| 注入层级   | Token 域（提示词拼接）         | 注意力域（K/V 注入） |
| 注入控制   | 无（硬拼接）                   | 连续（α ∈ [0, 1]）   |
| 提示词消耗 | 是                             | 否                   |
| 上下文窗口 | 被检索内容消耗                 | 完全可用于用户       |
| 提示工程   | 必需                           | 简化                 |
| 可解释性   | 高（提示词可见）               | 中（需注意力可视化） |
| 生态成熟度 | 高（LangChain、LlamaIndex 等） | 低（新兴）           |
| 调试       | 直接                           | 需专用工具           |

### 适用场景

#### ✅ 推荐使用 DKI 的场景

1. **个性化助手**

    - 用户偏好需要跨会话持久化
    - 隐式个性化（无需在提示词中显式提及）
    - 多轮对话需要上下文连续性

2. **客服系统**

    - 用户画像 + 对话历史
    - 跨会话一致体验
    - 隐私敏感的用户数据

3. **教育应用**

    - 学习偏好 + 进度历史
    - 基于用户水平的自适应响应
    - 长期用户建模

4. **健康/养生助手**
    - 健康档案 + 咨询历史
    - 敏感个人数据
    - 一致的医疗上下文

#### ⚠️ 建议使用 RAG 的场景

1. **外部知识检索**

    - 文档搜索和问答
    - 公共知识库
    - 频繁更新的内容

2. **首轮延迟极度敏感**

    - 首次交互必须最快
    - 无法接受门控开销
    - 冷启动性能关键

3. **高审计要求**

    - 需要显示完整检索内容
    - 监管机构要求可见提示词
    - 注意力可视化不足

4. **快速原型开发**
    - 使用成熟 RAG 生态系统（LangChain、LlamaIndex）
    - 需要快速迭代
    - 无需深度定制

### 性能基准

基于 DeepSeek-V3 7B 的模拟实验（n=500）：

| 指标         | RAG    | DKI        | 变化       |
| ------------ | ------ | ---------- | ---------- |
| 记忆召回率   | 87.3%  | 86.2%      | -1.1%      |
| 首轮延迟     | 78.8ms | 92.4ms     | +17.3%     |
| **后续延迟** | 76.1ms | **42.8ms** | **-43.7%** |
| 缓存命中率   | N/A    | 69.7%      | -          |
| 幻觉率       | 10.2%  | 10.4%      | +0.2%      |

**关键发现**：

-   DKI 在首轮有 17.3%的开销，但后续轮次降低 43.7%
-   3 轮后总延迟：DKI < RAG（178.2ms vs 228.9ms）
-   记忆召回仅降低 1.1%，幻觉率相当

### 故障模式与缓解

| 故障模式   | 症状                     | 检测                       | 缓解                      |
| ---------- | ------------------------ | -------------------------- | ------------------------- |
| 记忆混淆   | 多个相似记忆导致混合输出 | 计算记忆间相似度           | 相似度>0.9 时仅保留 top-1 |
| 时间不一致 | 旧记忆与当前上下文冲突   | 检查记忆时间戳 vs 查询时态 | 为 α 添加时间衰减因子     |
| 投影过拟合 | 训练好但测试差           | 监控验证 vs 训练损失       | 增加 dropout，数据增强    |
| 缓存抖动   | 命中率<30%，延迟增加     | 监控缓存命中率             | 增加缓存大小或使用 LFU    |
| 偏见放大   | 记忆注入放大模型偏见     | 监控输出多样性指标         | 实现多样性感知路由        |

### 高级特性

#### 分层 KV 缓存

```python
from dki.core.components.tiered_kv_cache import TieredKVCache

cache = TieredKVCache(
    l1_size_gb=8,      # GPU
    l2_size_gb=32,     # CPU RAM
    l3_size_gb=128,    # SSD
    compression_l2=4,   # L2压缩率
    compression_l3=8    # L3压缩率
)

# 自动分层管理
kv_pair = cache.get_or_compute(memory_id, compute_fn)
```

#### 注意力预算追踪

```python
from dki.core.components.attention_budget import AttentionBudgetTracker

tracker = AttentionBudgetTracker()
with tracker.track_request(session_id):
    response = dki.chat(query, session_id)

# 获取预算使用情况
budget = tracker.get_budget_usage(session_id)
print(f"Token预算使用: {budget.token_used}/{budget.token_total}")
print(f"注意力FLOPs: {budget.attention_flops}")
```

### 📄 相关论文

本项目基于论文《Dynamic KV Injection: An Attention-Level User Memory System for Large Language Models》实现。

### 常见问题

**Q: DKI 需要重新训练模型吗？**  
A: 不需要。DKI 是推理时增强，使用冻结的模型参数，通过 Attention Hook 注入 K/V。

**Q: DKI 和 RAG 有什么区别？**  
A:

-   **RAG**：在 token 层级拼接检索内容，消耗上下文窗口
-   **DKI**：在注意力层级注入 K/V，不消耗 token 预算
-   它们是互补的：RAG 处理外部知识，DKI 处理用户级记忆

**Q: 上层应用需要做什么修改？**  
A:

1. 提供适配器配置文件（指定数据库连接和字段映射）
2. 删除 RAG/Prompt 工程代码
3. 调用 DKI API 时传递 `user_id` 和原始输入

**Q: 如果上层应用的数据库没有向量索引怎么办？**  
A: DKI 支持动态向量处理，配置 `vector_search.type: dynamic` 即可。支持三种策略：

-   `lazy`：按需计算 embedding
-   `batch`：批量预计算
-   `hybrid`：BM25 初筛 + embedding 重排序（推荐）

**Q: 偏好注入和历史注入有什么区别？**  
A:

-   **偏好**：K/V 注入在负位置，通过 Attention Hook 实现，隐式影响，可缓存
-   **历史**：后缀提示词在正位置，标准 token 拼接，显式参考，动态变化

**Q: 如何将 DKI 集成到现有系统？**  
A:

```python
from dki.core.dki_plugin import DKIPlugin

# 从配置文件创建 (推荐)
dki = await DKIPlugin.from_config(
    model_adapter=your_model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 调用时只需传递 user_id 和原始输入
response = await dki.chat(
    query="用户的原始输入",
    user_id="user_123",
    session_id="session_456",
)
```

**Q: DKI 需要考虑分布式部署吗？**  
A: 不需要。DKI 作为 LLM 插件，只负责读取用户配置和消息数据完成注入。分布式部署是 LLM 引擎和上层应用的责任。DKI 本身是无状态的（除了偏好 K/V 缓存），可以随 LLM 实例水平扩展。

**Q: 生产环境部署建议？**  
A:

1. 启用混合注入
2. 偏好 α 设置为 0.4（保守）
3. 配置适配器连接到上层应用的数据库
4. 监控注入率和延迟
5. 根据指标调整 alpha 和缓存策略

### 最新优化 (v2.5)

#### Memory Trigger (记忆触发器)

检测用户输入中的记忆相关信号，决定"记住什么"：

```python
# 支持 5 种触发类型
- META_COGNITIVE: "我们刚刚讨论的"、"之前你说过"
- STATE_CHANGE: "我改变想法了"、"补充一点"
- LONG_TERM_VALUE: "请记住我喜欢..."、"我是素食主义者"
- RECALL_REQUEST: "最近我们聊了什么"
- OPINION_QUERY: "你有新看法吗"

# 规则可配置，后续可用分类器增强
trigger = MemoryTrigger(language="auto")
result = trigger.detect("我们刚才聊了什么？")
```

#### Reference Resolver (指代解析器)

解析用户输入中的指代表达，确定历史召回范围：

```python
# 召回轮数可外置配置
resolver = ReferenceResolver(config=ReferenceResolverConfig(
    last_few_turns=5,    # "刚刚" 召回 5 轮
    recent_turns=20,     # "最近" 召回 20 轮
))

# 支持运行时动态更新
dki.update_reference_resolver_config(
    just_now_turns=3,
    recently_turns=15,
)
```

#### 为什么不需要 Rolling Summary

与 ChatGPT/Claude/Grok 等系统不同，DKI **不需要** Rolling Summary：

| 方案            | 原因                     | DKI 替代方案                |
| --------------- | ------------------------ | --------------------------- |
| RAG+Prompt      | 上下文窗口限制，需要压缩 | K/V 注入不占用上下文        |
| Rolling Summary | 信息压缩导致丢失         | Memory Trigger 精准召回     |
| 摘要生成        | 额外 LLM 调用开销        | Reference Resolver 按需检索 |

### 路线图

**已完成**：

-   [x] 核心 DKI 实现 (Attention Hook K/V 注入)
-   [x] vLLM/LLaMA/DeepSeek/GLM 适配器
-   [x] 混合注入策略（偏好 K/V + 历史后缀）
-   [x] 配置驱动适配器（SQLAlchemy 动态表映射）
-   [x] 动态向量处理（BM25 + Embedding 混合搜索）
-   [x] 偏好 K/V 缓存（内存级）
-   [x] 监控 API（统计/日志/健康检查）
-   [x] Vue3 示例前端 UI
-   [x] 实验框架
-   [x] Memory Trigger (记忆触发器)
-   [x] Reference Resolver (指代解析器，召回轮数可配置)

**进行中**：

-   [ ] Stance State Machine (观点状态机)
-   [ ] 分类器增强 Memory Trigger

**未来工作**：

-   [ ] Redis 分布式缓存集成
-   [ ] 注意力可视化工具
-   [ ] 多模态扩展（图像/音频记忆）
-   [ ] LangChain/LlamaIndex 集成

---

## 🔮 未来工作方向

### 1. Redis 分布式缓存集成 ⭐ 推荐优先实施

**当前状态**：偏好 K/V 缓存仅在内存中，单实例有效。

**优化目标**：集成 Redis 实现跨实例共享缓存。

**为什么 Redis 集成是最重要的优化**：

DKI 的核心优势之一是**偏好 K/V 缓存复用**——首轮计算 K/V 后，后续请求直接使用缓存，延迟降低 43.7%。但当前的内存缓存有一个致命问题：**只在单实例有效**。

在生产环境中，LLM 服务通常是多实例部署（负载均衡）。如果用户的请求被路由到不同实例，缓存就会失效，DKI 的核心优势将大打折扣：

| 部署模式 | 缓存命中率 | DKI 优势 |
| -------- | ---------- | -------- |
| 单实例   | ~70%       | 完整发挥 |
| 2 实例   | ~35%       | 减半     |
| 4 实例   | ~17.5%     | 大幅削弱 |
| N 实例   | ~70%/N     | 几乎无效 |

**Redis 集成后**：无论多少实例，缓存命中率始终保持 ~70%，DKI 优势完整发挥。

**核心价值**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    当前架构 (单实例缓存)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LLM Instance 1          LLM Instance 2          LLM Instance 3         │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐        │
│  │ DKI Plugin  │         │ DKI Plugin  │         │ DKI Plugin  │        │
│  │ ┌─────────┐ │         │ ┌─────────┐ │         │ ┌─────────┐ │        │
│  │ │ 内存缓存 │ │         │ │ 内存缓存 │ │        │ │ 内存缓存 │ │        │
│  │ │ user_001│ │         │ │ user_002 │ │        │ │ user_003│ │        │
│  │ └─────────┘ │         │ └─────────┘ │         │ └─────────┘ │        │
│  └─────────────┘         └─────────────┘         └─────────────┘        │
│                                                                         │
│  问题：                                                                  │
│  - user_001 请求到 Instance 2 时，缓存未命中，需重新计算 K/V               │
│  - 缓存命中率随实例数增加而下降                                            │
│  - 无法实现真正的水平扩展                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    优化架构 (Redis 分布式缓存)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LLM Instance 1          LLM Instance 2          LLM Instance 3         │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐        │
│  │ DKI Plugin  │         │ DKI Plugin  │         │ DKI Plugin  │        │
│  │ ┌─────────┐ │         │ ┌─────────┐ │         │ ┌─────────┐ │        │
│  │ │ L1 内存  │ │        │ │ L1 内存  │ │         │ │ L1 内存 │ │        │
│  │ │ (热数据) │ │        │ │ (热数据) │ │         │ │ (热数据) │ │        │
│  │ └────┬────┘ │         │ └────┬────┘ │         │ └────┬────┘ │        │
│  └──────┼──────┘         └──────┼──────┘         └──────┼──────┘        │
│         │                       │                       │               │
│         └───────────────────────┼───────────────────────┘               │
│                                 ▼                                       │
│                    ┌──────────────────────────┐                         │
│                    │      Redis Cluster       │                         │
│                    │  ┌─────────────────────┐ │                         │
│                    │  │ L2 分布式缓存        │ │                         │
│                    │  │ user_001, user_002  │ │                         │
│                    │  │ user_003, ...       │ │                         │
│                    │  └─────────────────────┘ │                         │
│                    └──────────────────────────┘                         │
│                                                                         │
│  优势：                                                                  │
│  - 任意实例都能命中缓存                                                   │
│  - 缓存命中率不受实例数影响                                               │
│  - 支持真正的水平扩展                                                     │
│  - 缓存持久化，重启不丢失                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**实现方案**：

```python
# 配置示例
cache:
  type: tiered  # memory | redis | tiered
  tiered:
    l1:
      type: memory
      max_size_mb: 512
      ttl: 300  # 5 分钟
    l2:
      type: redis
      host: redis-cluster.example.com
      port: 6379
      password: ${REDIS_PASSWORD}
      db: 0
      ttl: 3600  # 1 小时
      key_prefix: "dki:kv:"
      serialization: msgpack  # 压缩序列化
```

**可行性评估**：

| 维度       | 评估      | 说明                                             |
| ---------- | --------- | ------------------------------------------------ |
| 技术复杂度 | ⭐⭐ 中等 | Redis 客户端成熟，主要工作是序列化 K/V Tensor    |
| 性能影响   | ⭐⭐ 中等 | 网络延迟 ~1-5ms，但避免了 K/V 重计算 (~50-200ms) |
| 收益       | ⭐⭐⭐ 高 | 多实例部署必需，缓存命中率提升显著               |
| 依赖       | ⭐ 低     | 仅需 redis-py，可选依赖                          |

**关键技术点**：

1. **K/V Tensor 序列化**：使用 `msgpack` + `numpy` 压缩，减少网络传输
2. **缓存失效策略**：偏好变更时主动失效，TTL 兜底
3. **热点数据本地缓存**：L1 内存缓存高频用户，减少 Redis 访问

### 2. 注意力可视化工具

**目标**：调试 K/V 注入效果，理解注入对注意力分布的影响。

**设计方案**：

```python
# 可视化 API
from dki.visualization import AttentionVisualizer

visualizer = AttentionVisualizer(dki_plugin)

# 生成注意力热力图
heatmap = visualizer.generate_heatmap(
    query="推荐一家餐厅",
    user_id="user_001",
    layer_indices=[0, 12, 24],  # 可视化特定层
)

# 对比注入前后
comparison = visualizer.compare_injection(
    query="推荐一家餐厅",
    user_id="user_001",
    show_diff=True,
)

# 导出为 HTML 报告
visualizer.export_report("attention_analysis.html")
```

**可行性评估**：

| 维度       | 评估      | 说明                               |
| ---------- | --------- | ---------------------------------- |
| 技术复杂度 | ⭐⭐ 中等 | 需要 Hook 注意力权重，可视化库成熟 |
| 性能影响   | ⭐⭐⭐ 高 | 仅调试时启用，生产环境关闭         |
| 收益       | ⭐⭐ 中等 | 调试和论文展示有价值               |
| 依赖       | ⭐ 低     | matplotlib, plotly 等可选依赖      |

### 3. 多模态扩展（图像/音频记忆）

**目标**：支持图像和音频作为用户记忆的一部分。

**设计方案**：

```python
# 多模态偏好
preferences = [
    {"type": "text", "content": "喜欢简约风格"},
    {"type": "image", "content": "user_avatar.jpg", "embedding": [...]},
    {"type": "audio", "content": "voice_sample.wav", "embedding": [...]},
]

# 多模态 K/V 注入
# 图像/音频先通过编码器转换为 embedding，再计算 K/V
```

**可行性评估**：

| 维度       | 评估      | 说明                                 |
| ---------- | --------- | ------------------------------------ |
| 技术复杂度 | ⭐⭐⭐ 高 | 需要多模态编码器，K/V 计算方式需调整 |
| 性能影响   | ⭐⭐⭐ 高 | 图像/音频编码开销大                  |
| 收益       | ⭐⭐ 中等 | 特定场景有价值（如虚拟助手）         |
| 依赖       | ⭐⭐⭐ 高 | 需要 CLIP、Whisper 等模型            |

**建议**：作为长期目标，优先级较低。

### 4. LangChain/LlamaIndex 集成

**目标**：将 DKI 包装为 LangChain/LlamaIndex 模块，扩大生态。

**设计方案**：

```python
# LangChain 集成
from langchain_dki import DKIMemory

memory = DKIMemory(
    adapter_config_path="config/adapter_config.yaml",
)

chain = ConversationChain(
    llm=llm,
    memory=memory,  # DKI 作为 Memory 模块
)

# LlamaIndex 集成
from llama_index_dki import DKIRetriever

retriever = DKIRetriever(
    adapter_config_path="config/adapter_config.yaml",
)

query_engine = index.as_query_engine(
    retriever=retriever,  # DKI 作为 Retriever
)
```

**可行性评估**：

| 维度       | 评估      | 说明                                |
| ---------- | --------- | ----------------------------------- |
| 技术复杂度 | ⭐⭐ 中等 | 需要适配 LangChain/LlamaIndex 接口  |
| 性能影响   | ⭐ 低     | 仅封装层，无额外开销                |
| 收益       | ⭐⭐⭐ 高 | 扩大用户群，降低使用门槛            |
| 依赖       | ⭐⭐ 中等 | langchain, llama-index 作为可选依赖 |

**建议**：优先级中等，待核心功能稳定后实施。

### 优先级排序

| 优先级 | 优化方向                  | 原因                     |
| ------ | ------------------------- | ------------------------ |
| P1     | Redis 分布式缓存          | 多实例部署必需，收益明确 |
| P2     | 注意力可视化              | 调试和论文展示有价值     |
| P3     | LangChain/LlamaIndex 集成 | 扩大生态，但非核心功能   |
| P4     | 多模态扩展                | 技术复杂度高，特定场景   |

### Redis 集成的额外价值

除了解决多实例缓存问题，Redis 集成还带来以下价值：

1. **缓存持久化**

    - 服务重启后缓存不丢失
    - 减少冷启动时的 K/V 重计算

2. **缓存预热**

    - 可以提前计算高频用户的 K/V
    - 批量导入历史用户偏好

3. **缓存监控**

    - Redis 提供丰富的监控指标
    - 可以分析缓存命中率、内存使用等

4. **缓存淘汰策略**

    - LRU/LFU 等成熟策略
    - 自动管理缓存容量

5. **跨服务共享**
    - 多个 DKI 实例共享同一缓存
    - 甚至可以跨不同 LLM 服务共享（如果使用相同模型）

**成本分析**：

| 资源       | 估算          | 说明                 |
| ---------- | ------------- | -------------------- |
| Redis 内存 | ~100MB/万用户 | 假设每用户 K/V ~10KB |
| 网络延迟   | 1-5ms         | 局域网内             |
| 运维成本   | 低            | Redis 运维成熟       |

**ROI 分析**：

```
假设：
- 4 实例部署
- 当前缓存命中率 ~17.5% (70%/4)
- Redis 后缓存命中率 ~70%

收益：
- 缓存命中率提升 4x
- 后续轮次延迟降低 ~40%
- 整体吞吐量提升 ~30%

成本：
- Redis 实例 ~$50/月 (云服务)
- 开发工作量 ~2-3 人天
```

### 示例前端 UI

DKI 提供了一个基于 **Vue3** 的示例前端 UI，用于演示 DKI 集成：

-   **Vue 3 + TypeScript**：类型安全的现代前端开发
-   **Vite**：快速的开发服务器和构建工具
-   **Pinia**：Vue3 官方状态管理
-   **Element Plus**：企业级 UI 组件库

UI 功能：

-   聊天界面（显示 DKI 元数据徽章）
-   用户偏好管理面板
-   会话历史浏览
-   系统统计仪表板（需密码）

**注意**：Chat UI 是示例应用，DKI 的适配器会读取其数据库来获取用户偏好和历史消息。

### 致谢

本项目受到以下研究的启发：

-   RAG ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401))
-   RETRO ([Borgeaud et al., 2022](https://arxiv.org/abs/2112.04426))
-   Self-RAG ([Asai et al., 2023](https://arxiv.org/abs/2310.11511))
-   FiLM ([Perez et al., 2018](https://arxiv.org/abs/1709.07871))
-   FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))

---

**DKI** - 在注意力层级重新思考记忆增强
