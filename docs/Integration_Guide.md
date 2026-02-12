# DKI 集成指南

## 概述

DKI 是 LLM 的插件，上层应用集成 DKI 只需要：

1. **提供适配器配置文件** - 指定数据库连接和字段映射
2. **删除 RAG/Prompt 工程代码** - 移除用户偏好和历史对话拼接逻辑
3. **传递 user_id + 原始输入** - DKI 自动处理其余工作

## 集成前后对比

### 集成前 (传统 RAG/Prompt 方式)

```python
# 上层应用需要做大量工作
async def handle_chat(user_id: str, message: str):
    # 1. 读取用户偏好
    preferences = await db.get_user_preferences(user_id)
    
    # 2. 检索相关历史
    history = await vector_db.search(message, user_id)
    
    # 3. 构造 Prompt (RAG 方式)
    prompt = f"""
    用户偏好:
    {format_preferences(preferences)}
    
    相关历史:
    {format_history(history)}
    
    用户问题: {message}
    
    请根据以上信息回答用户问题。
    """
    
    # 4. 调用 LLM
    response = await llm.generate(prompt)
    
    return response
```

**问题**:
- 偏好和历史占用 Token Budget
- 需要维护复杂的 Prompt 工程
- 检索和拼接逻辑耦合在业务代码中

### 集成后 (使用 DKI)

```python
# 上层应用只需传递 user_id 和原始输入
async def handle_chat(user_id: str, message: str, session_id: str):
    # 直接调用 DKI，传递原始输入
    response = await dki.chat(
        query=message,      # 原始用户输入，不含任何 prompt 构造
        user_id=user_id,    # 用户标识
        session_id=session_id,
    )
    
    return response.text
```

**优势**:
- 偏好通过 K/V 注入，不占用 Token Budget
- 无需维护 Prompt 工程
- 检索和注入逻辑由 DKI 处理

## 集成步骤

### Step 1: 提供适配器配置文件

创建 `config/adapter_config.yaml`，告诉 DKI 如何连接您的数据库：

```yaml
user_adapter:
  # 数据库连接
  database:
    type: postgresql  # postgresql | mysql | sqlite
    host: localhost
    port: 5432
    database: your_app_db
    username: your_user
    password: your_password
  
  # 用户偏好表映射
  # 告诉 DKI 您的偏好表结构
  preferences:
    table: user_preferences  # 您的表名
    fields:
      user_id: user_id       # 您的字段名
      preference_text: content
      preference_type: type
      priority: priority
    filters:
      is_active: true
  
  # 消息表映射
  # 告诉 DKI 您的消息表结构
  messages:
    table: chat_messages
    fields:
      message_id: id
      session_id: session_id
      user_id: user_id
      role: role
      content: content
      timestamp: created_at
  
  # 向量检索配置
  vector_search:
    type: dynamic  # 动态向量处理 (无需预计算向量)
    dynamic:
      strategy: hybrid  # BM25 + embedding 混合检索
```

### Step 2: 删除 RAG/Prompt 工程代码

移除以下代码：
- 用户偏好读取和格式化
- 历史消息检索和拼接
- Prompt 模板构造

### Step 3: 传递 user_id + 原始输入

修改您的聊天接口：

```python
# 修改前
async def chat(user_id: str, message: str):
    # ... 复杂的 RAG/Prompt 逻辑 ...
    pass

# 修改后
async def chat(user_id: str, message: str, session_id: str):
    response = await dki.chat(
        query=message,        # 原始用户输入
        user_id=user_id,      # 用户标识
        session_id=session_id,
    )
    return response.text
```

## user_id 标识符说明

### 什么是 user_id

`user_id` 是上层应用中用户的唯一标识符，DKI 使用它来：

1. **读取用户偏好** - 从您的 `user_preferences` 表中读取该用户的偏好
2. **检索历史消息** - 从您的 `chat_messages` 表中检索该用户的相关历史

### user_id 的要求

- **唯一性**: 每个用户必须有唯一的 `user_id`
- **一致性**: 同一用户的 `user_id` 在所有请求中必须一致
- **格式**: 字符串类型，可以是 UUID、数字 ID 或其他格式

### 示例

```python
# 示例 1: 使用数据库自增 ID
user_id = str(user.id)  # "12345"

# 示例 2: 使用 UUID
user_id = str(user.uuid)  # "550e8400-e29b-41d4-a716-446655440000"

# 示例 3: 使用用户名
user_id = user.username  # "john_doe"
```

### session_id 说明

`session_id` 是会话的唯一标识符，用于：

1. **限定历史检索范围** - 只检索当前会话的历史
2. **会话隔离** - 不同会话的历史不会混淆

如果不提供 `session_id`，DKI 会使用 `user_id` 作为默认值。

## API 接口

### POST /v1/dki/chat

DKI 增强聊天接口。

**请求**:
```json
{
  "query": "推荐一家餐厅",
  "user_id": "user_123",
  "session_id": "session_456",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**响应**:
```json
{
  "id": "dki-abc123",
  "text": "根据您的偏好（素食主义者），推荐...",
  "input_tokens": 50,
  "output_tokens": 150,
  "dki_metadata": {
    "injection_enabled": true,
    "alpha": 0.4,
    "preference_tokens": 30,
    "history_tokens": 100,
    "cache_hit": true,
    "cache_tier": "memory",
    "latency_ms": 145.2
  },
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "根据您的偏好..."},
      "finish_reason": "stop"
    }
  ],
  "created": 1707580800
}
```

## 动态向量检索方案

### 可行性分析

DKI 的动态向量检索方案是**完全可行的**，原因如下：

1. **与 RAG 类似的流程**
   - RAG: 检索 → 拼接到 Prompt → LLM 推理
   - DKI: 检索 → K/V 注入 + 后缀提示词 → LLM 推理
   - 检索部分的流程是相同的

2. **成熟的技术栈**
   - BM25: 经典的文本检索算法，性能稳定
   - Embedding: OpenAI/本地模型，效果成熟
   - 混合检索: BM25 初筛 + Embedding 重排序，业界最佳实践

3. **缓存优化**
   - Embedding 缓存: 避免重复计算
   - 偏好 K/V 缓存: 偏好稳定，可长期复用

### 历史消息关联检索方式

DKI 使用三种策略检索相关历史：

#### 1. Lazy 策略 (实时计算)

```
消息数量 < 100 时使用

流程:
1. 计算查询的 embedding
2. 遍历所有消息，计算 embedding
3. 计算余弦相似度
4. 返回 top-k
```

#### 2. Hybrid 策略 (BM25 + Embedding)

```
消息数量 100-1000 时使用 (推荐)

流程:
1. BM25 快速筛选 top-N 候选 (N = top_k × 4)
2. 只对候选消息计算 embedding
3. 余弦相似度重排序
4. 返回 top-k

优势:
- BM25 快速过滤无关内容
- 只对候选计算 embedding，减少计算量
- 结合关键词匹配和语义相似度
```

#### 3. Batch 策略 (预计算)

```
消息数量 > 1000 时使用

流程:
1. 后台定时预计算所有消息的 embedding
2. 检索时直接使用缓存的 embedding
3. 计算余弦相似度
4. 返回 top-k
```

### 性能对比

| 策略 | 100 条消息 | 500 条消息 | 2000 条消息 |
|------|-----------|-----------|------------|
| lazy | 5s | 25s | 100s |
| hybrid | 2s | 3s | 5s |
| batch (预计算后) | 0.1s | 0.1s | 0.1s |

## 完整集成示例

### Python 后端集成

```python
from dki.core.dki_plugin import DKIPlugin

# 1. 创建 DKI 插件 (从配置文件)
dki = await DKIPlugin.from_config(
    model_adapter=vllm_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 2. 在您的聊天接口中使用
@app.post("/chat")
async def chat(request: ChatRequest):
    # 只传递 user_id 和原始输入
    response = await dki.chat(
        query=request.message,  # 原始用户输入
        user_id=request.user_id,
        session_id=request.session_id,
    )
    
    return {
        "text": response.text,
        "dki_metadata": response.metadata.to_dict(),
    }
```

### HTTP API 集成

```bash
# 调用 DKI API
curl -X POST http://localhost:8080/v1/dki/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐一家餐厅",
    "user_id": "user_123",
    "session_id": "session_456"
  }'
```

## 常见问题

### Q: 我的数据库没有向量字段，能用 DKI 吗？

可以。DKI 支持动态向量处理，会自动计算 embedding 并缓存。

### Q: user_id 必须是什么格式？

任意字符串格式，只要在您的系统中唯一即可。

### Q: 如何监控 DKI 的工作状态？

使用监控 API：
- `GET /v1/health` - 健康检查
- `GET /v1/stats` - 统计数据
- `GET /v1/logs` - 注入日志

### Q: DKI 会修改我的数据库吗？

不会。DKI 只读取数据，不会写入或修改您的数据库。
