# DKI Quick Start Guide

快速开始使用 DKI (Dynamic KV Injection) - LLM 注意力层级用户记忆插件。

## 📋 Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ VRAM (for 7B models)

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

## 💻 Integration Example (Recommended)

DKI 作为 LLM 插件，上层应用只需传递 `user_id` 和原始输入：

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models.vllm_adapter import VLLMAdapter

# 1. 初始化 LLM 适配器
model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")

# 2. 从配置文件创建 DKI 插件
# 配置文件指定如何连接上层应用的数据库
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 3. 调用 DKI - 只需传递 user_id 和原始输入
# DKI 会自动:
# - 通过适配器读取用户偏好 → K/V 注入 (Attention Hook)
# - 通过适配器检索相关历史 → 后缀提示词
response = await dki.chat(
    query="今晚想找一家餐厅",  # 原始输入，无需任何 prompt 构造
    user_id="user_001",
    session_id="session_001",
)

print(f"Response: {response.text}")
print(f"Injection Enabled: {response.metadata.injection_enabled}")
print(f"Alpha: {response.metadata.alpha:.2f}")
print(f"Preference Tokens: {response.metadata.preference_tokens}")
print(f"History Tokens: {response.metadata.history_tokens}")
print(f"Cache Hit: {response.metadata.preference_cache_hit}")
print(f"Latency: {response.metadata.latency_ms:.0f}ms")
```

## 🔧 Adapter Configuration

创建 `config/adapter_config.yaml` 配置如何连接上层应用的数据库：

```yaml
user_adapter:
  # 数据库连接 (连接到上层应用的数据库)
  database:
    type: postgresql  # postgresql | mysql | sqlite
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
      strategy: hybrid  # BM25 + embedding
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

### Get DKI Status

```bash
curl http://localhost:8080/v1/dki/info
```

### Get Stats

```bash
curl http://localhost:8080/api/stats
```

## ❓ Troubleshooting

### CUDA Out of Memory

- Use smaller model (3B instead of 7B)
- Enable 8-bit quantization: `load_in_8bit: true`
- Reduce `max_model_len` in config

### Slow First Response

- First turn computes K/V cache for preferences
- Subsequent turns use cached K/V
- This is expected behavior

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

| 特性 | DKI | RAG |
|------|-----|-----|
| 注入层级 | 注意力层级 (K/V) | Token 层级 (prompt) |
| Token 消耗 | 不消耗 | 消耗上下文窗口 |
| 适用场景 | 用户级记忆 | 外部知识库 |
| 上层应用改动 | 只需传 user_id | 需要 prompt 工程 |

### Hybrid Injection

- **偏好**: K/V 注入 (负位置, Attention Hook)
  - 隐式影响，如同人格
  - 可缓存，高复用
- **历史**: 后缀提示词 (正位置)
  - 显式参考，可引用
  - 动态变化

## 📚 Next Steps

1. Read the full [README.md](README.md)
2. Check [Integration Guide](docs/Integration_Guide.md)
3. Explore the [DKI Paper](../DKIPaper/DKI_Paper_v2.md)
4. Run experiments with different models
5. Customize adapter config for your database

---

Happy experimenting with DKI! 🚀
