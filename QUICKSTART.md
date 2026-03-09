# DKI Quick Start Guide

快速开始使用 DKI (Dynamic KV Injection) v4.0 - LLM 注意力层级用户记忆插件。

## 📋 Prerequisites

-   Python 3.10+
-   CUDA 11.8+ (for GPU support)
-   16GB+ RAM
-   8GB+ VRAM (for 7B models)

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd DKI

# Windows
scripts\setup.bat

# Linux/Mac
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Step 2: Start Development Servers

```bash
# Start both backend and example Chat UI
python start_dev.py
```

### Step 3: Open Example Chat UI

Open http://localhost:3000 in your browser.

## 💻 Integration (v4.0 — 极简集成)

DKI v4.0 提供三层集成模式，由简到繁：

### Level 1: One-Line Factory（最简集成，推荐）

上层应用只需 **3 行代码** 即可集成 DKI：

```python
from dki.integration import create_plugin

# 1. 创建 DKI 插件 (自动管理模型、缓存、适配器)
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")

# 2. 调用 chat — 只需传递 user_id 和原始输入
response = await dki.chat(
    query="今晚想找一家餐厅",
    user_id="user_001",
    session_id="session_001",
)

print(f"Response: {response.text}")
print(f"Injection Enabled: {response.metadata.injection_enabled}")
print(f"Alpha: {response.metadata.alpha:.2f}")
print(f"Latency: {response.metadata.latency_ms:.0f}ms")
```

### Level 2: FastAPI Middleware（Web 应用推荐）

```python
from fastapi import FastAPI, Depends
from dki.integration import DKIMiddleware, get_dki_plugin

app = FastAPI()

# 一行添加 DKI 支持 (自动管理生命周期)
app.add_middleware(
    DKIMiddleware,
    adapter_config_path="config/adapter_config.yaml",
)

@app.post("/chat")
async def chat(query: str, user_id: str, dki=Depends(get_dki_plugin)):
    response = await dki.chat(query=query, user_id=user_id, session_id="s1")
    return {"text": response.text, "metadata": response.metadata}
```

### Level 3: Full Control（启用动态路由 + 消息管理）

```python
from dki.integration import create_plugin

# 启用全部增强功能
dki = await create_plugin(
    adapter_config_path="config/adapter_config.yaml",
    engine="vllm",
    model_name="Qwen/Qwen3-8B",
    dynamic_router=True,        # 自动在 RAG 和 DKI 之间切换
    message_management=True,    # DKI 内部完成消息读写
    rag_system=my_rag,          # RAG 系统实例
    store=my_store,             # 数据持久化层
)

# 调用方式不变 — DKI 内部自动路由
response = await dki.chat(
    query="什么是量子计算？",  # → 自动路由到 RAG (知识检索型)
    user_id="user_001",
    session_id="session_001",
)

response = await dki.chat(
    query="推荐一家适合我的餐厅",  # → 自动路由到 DKI (个性化型)
    user_id="user_001",
    session_id="session_001",
)
```

### Level 4: Closed-Source Model（闭源模型，v4.1）

DKI v4.1 支持闭源模型（OpenAI、DeepSeek API、GLM API、Moonshot 等），只需在 `config.yaml` 中配置即可，集成代码完全不变：

```yaml
# config/config.yaml
model:
    default_engine: "closed_source"
    engines:
        closed_source:
            enabled: true
            model_name: "deepseek-chat"           # 或 gpt-4o, glm-4 等
            api_key: "${DEEPSEEK_API_KEY}"         # 从环境变量读取
            api_base: "https://api.deepseek.com/v1"
            max_model_len: 32768
            timeout: 120.0
            max_retries: 2
```

```python
from dki.integration import create_plugin

# 与开源模型完全相同的 3 行代码
# DKI 自动检测闭源模型 → 强制走 RAG 路由 → API 调用
dki = await create_plugin(adapter_config_path="config/adapter_config.yaml")

response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_001",
    session_id="session_001",
)
print(response.text)

# 流式也支持 (SSE)
async for chunk in dki.chat_stream(
    query="推荐一本好书",
    user_id="user_001",
    session_id="session_001",
):
    if chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
```

> **注意**：闭源模型不支持 K/V 注入（无法访问模型内部），DKI 会自动使用 RAG 路由（prompt 拼接方式）。用户偏好和历史仍然通过适配器读取，但以 prompt 形式拼接而非 K/V 注入。

### 向后兼容：直接使用 DKIPlugin

v4.0 完全向后兼容，原有的 `DKIPlugin` 接口不变：

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

# 1. 初始化 LLM 适配器
model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")

# 2. 从配置文件创建 DKI 插件
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 3. 调用 DKI
response = await dki.chat(
    query="今晚想找一家餐厅",
    user_id="user_001",
    session_id="session_001",
)
```

## 🔧 Adapter Configuration

创建 `config/adapter_config.yaml` 配置如何连接上层应用的数据库：

```yaml
user_adapter:
    # 数据库连接 (连接到上层应用的数据库)
    database:
        type: postgresql # postgresql | mysql | sqlite
        host: localhost
        database: my_app_db
        username: user
        password: pass

    # 偏好表映射
    preferences:
        table: user_preferences
        fields:
            user_id: user_id
            preference_text: content
            preference_type: type

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

    # 向量检索 (支持动态向量处理)
    vector_search:
        type: dynamic
        dynamic:
            strategy: hybrid # BM25 + embedding
```

## 🌊 Streaming Chat

DKI v4.0 支持流式生成，所有模型适配器（vLLM、SGLang、LLaMA、DeepSeek、GLM）均已实现原生流式支持：

```python
# 流式生成
async for chunk in dki.chat_stream(
    query="推荐一家餐厅",
    user_id="user_001",
    session_id="session_001",
):
    if chunk["type"] == "metadata":
        print(f"Metadata: {chunk['metadata']}")
    elif chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\n\nTotal tokens: {chunk['output_tokens']}")
```

REST API 流式端点：

```bash
curl -X POST http://localhost:8080/v1/dki/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐一家餐厅",
    "user_id": "user_001",
    "session_id": "session_001"
  }'
```

## 🎛️ Control Injection Strength

```python
# 强制指定 alpha 值
response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_001",
    session_id="session_001",
    force_alpha=0.8,  # 强注入
)

# 查看注入详情
print(f"Alpha: {response.metadata.alpha}")
print(f"Gating Decision: {response.metadata.gating_decision}")
```

## 📊 Monitoring

```python
# 获取统计数据
stats = dki.get_stats()
print(f"Total Requests: {stats['total_requests']}")
print(f"Injection Rate: {stats['injection_rate']:.2%}")
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg Latency: {stats['avg_latency_ms']:.1f}ms")

# EnhancedDKIPlugin 额外统计
if "enhanced" in stats:
    print(f"DKI Routes: {stats['enhanced']['dki_routes']}")
    print(f"RAG Routes: {stats['enhanced']['rag_routes']}")

# 获取注入日志
logs = dki.get_injection_logs(limit=5)
for log in logs:
    print(f"[{log['timestamp']}] alpha={log['alpha']:.2f}")
```

## 🌐 REST API Usage

### DKI Chat (上层应用调用此接口)

```bash
curl -X POST http://localhost:8080/v1/dki/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐一家餐厅",
    "user_id": "user_001",
    "session_id": "session_001"
  }'
```

### DKI Streaming Chat

```bash
curl -X POST http://localhost:8080/v1/dki/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐一家餐厅",
    "user_id": "user_001",
    "session_id": "session_001"
  }'
```

### Get DKI Status

```bash
curl http://localhost:8080/v1/dki/info
```

### Health Check (Detailed)

```bash
curl http://localhost:8080/api/health/detail
```

### Get Stats

```bash
curl http://localhost:8080/api/stats
```

## ❓ Troubleshooting

### CUDA Out of Memory

-   Use smaller model (3B instead of 7B)
-   Enable 8-bit quantization: `load_in_8bit: true`
-   Reduce `max_model_len` in config

### Slow First Response

-   First turn computes K/V cache for preferences
-   Subsequent turns use cached K/V
-   This is expected behavior

### No Injection

Check metadata:

```python
response = await dki.chat(query, user_id, session_id)
print(f"Injection Enabled: {response.metadata.injection_enabled}")
print(f"Preferences Count: {response.metadata.preferences_count}")
print(f"History Count: {response.metadata.relevant_history_count}")
```

确保:

1. 适配器配置正确连接到数据库
2. 数据库中有该用户的偏好或历史数据
3. 字段映射正确

## 📚 Key Concepts

### DKI vs RAG

| 特性         | DKI              | RAG                 |
| ------------ | ---------------- | ------------------- |
| 注入层级     | 注意力层级 (K/V) | Token 层级 (prompt) |
| Token 消耗   | 不消耗           | 消耗上下文窗口      |
| 适用场景     | 用户级记忆       | 外部知识库          |
| 上层应用改动 | 只需传 user_id   | 需要 prompt 工程    |
| 闭源模型     | 不支持 (v4.1 自动降级到 RAG) | ✅ 支持 |

### v4.0/v4.1 Integration Layer

| 集成模式              | 适用场景            | 代码量   |
| --------------------- | ------------------- | -------- |
| One-Line Factory      | 快速集成、脚本      | 3 行     |
| FastAPI Middleware    | Web 应用            | 5 行     |
| Full Control          | 动态路由 + 消息管理 | 10-15 行 |
| Closed-Source (v4.1)  | 闭源 API 模型       | 3 行     |
| DKIPlugin (直接)      | 完全控制、高级定制  | 10-20 行 |

### Hybrid Injection

-   **偏好**: K/V 注入 (负位置, Attention Hook)
    -   隐式影响，如同人格
    -   可缓存，高复用
-   **历史**: 后缀提示词 (正位置)
    -   显式参考，可引用
    -   动态变化

## 📚 Next Steps

1. Read the full [README.md](README.md)
2. Check [Integration Guide](docs/Integration_Guide.md)
3. Explore the [DKI Paper](../DKIPaper/DKI_Paper_v2.md)
4. Run experiments with different models
5. Customize adapter config for your database

---

Happy experimenting with DKI! 🚀
