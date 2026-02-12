# DKI 使用指南

## 概述

DKI (Dynamic KV Injection) 是 LLM 的插件，用于在注意力层级注入用户记忆。

**核心定位**: DKI 是 LLM 的插件，不是独立的聊天服务。

## 架构说明

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DKI 作为 LLM 插件的架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  上层应用 (第三方系统)                                                       │
│  ├── 自己的用户系统                                                         │
│  ├── 自己的消息数据库                                                       │
│  └── 自己的 Chat UI                                                        │
│       │                                                                     │
│       │ 用户输入 (原始文本) + user_id + session_id                          │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DKI 插件                                                           │   │
│  │  1. 通过适配器读取上层应用的数据库                                    │   │
│  │     ├── 用户偏好 → K/V 注入 (负位置)                                 │   │
│  │     └── 历史消息 → 后缀提示词 (正位置)                               │   │
│  │  2. 调用 LLM 推理                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       │ 返回响应                                                            │
│       ▼                                                                     │
│  上层应用                                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 组件说明

### 1. DKI 插件 (DKIPlugin)

核心组件，接收原始用户输入，执行注入后调用 LLM。

```python
from dki.core.dki_plugin import DKIPlugin

# 创建插件
dki = DKIPlugin(
    model_adapter=vllm_adapter,      # LLM 适配器
    user_data_adapter=pg_adapter,    # 外部数据适配器
)

# 调用
response = await dki.chat(
    query="推荐一家餐厅",  # 原始用户输入
    user_id="user_123",
    session_id="session_456",
)
```

### 2. 外部数据适配器 (IUserDataAdapter)

读取上层应用数据库的接口。

**注意**: 这是读取外部系统的数据，不是 DKI 自己的数据存储。

```python
from dki.adapters.base import IUserDataAdapter

class MyAppAdapter(IUserDataAdapter):
    """连接我的应用数据库"""
    
    async def get_user_preferences(self, user_id: str):
        # 从我的应用的 user_preferences 表读取
        return await self.db.fetch(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user_id
        )
    
    async def search_relevant_history(self, user_id: str, query: str):
        # 从我的应用的 messages 表检索相关历史
        return await self.vector_search(query, user_id)
```

### 3. 监控 API

提供 DKI 工作数据的监控接口。

```
GET /v1/health    # 健康检查
GET /v1/stats     # 统计数据
GET /v1/logs      # 注入日志
```

### 4. 示例应用

演示如何使用 DKI 插件，自带简单数据存储。

```python
from dki.example_app import ExampleAppService

# 创建示例服务
service = ExampleAppService(model_adapter=vllm_adapter)

# 注册用户 (示例应用功能)
service.register_user("user_001", "张三")

# 设置偏好 (示例应用功能)
service.add_preference("user_001", "dietary", "素食主义者")

# 聊天 (调用 DKI 插件)
response = await service.chat(
    session_id="session_001",
    user_id="user_001",
    message="推荐一家餐厅",
)
```

## 使用方式

### 方式 1: 作为 Python 库

```python
from dki.core.dki_plugin import DKIPlugin
from dki.adapters.postgresql_adapter import PostgreSQLAdapter
from dki.models.vllm_adapter import VLLMAdapter

# 1. 创建 LLM 适配器
model = VLLMAdapter(
    base_url="http://localhost:8001/v1",
    model_name="deepseek-coder-7b",
)

# 2. 创建外部数据适配器 (连接上层应用的数据库)
adapter = PostgreSQLAdapter(
    connection_string="postgresql://user:pass@host:5432/app_db",
    users_table="users",
    messages_table="chat_messages",
    preferences_table="user_preferences",
)

# 3. 创建 DKI 插件
dki = DKIPlugin(
    model_adapter=model,
    user_data_adapter=adapter,
)

# 4. 在上层应用中调用
async def handle_user_message(user_id: str, session_id: str, message: str):
    response = await dki.chat(
        query=message,  # 原始用户输入
        user_id=user_id,
        session_id=session_id,
    )
    return response.text
```

### 方式 2: 使用示例应用

```bash
# 启动示例应用
python -m dki.example_app.main

# API 端点
# POST /api/sessions/{session_id}/chat  # 聊天
# GET  /v1/stats                        # DKI 统计
# GET  /v1/health                       # 健康检查
```

## 数据流

```
用户输入 "推荐一家餐厅"
    │
    ▼
DKI 插件
    │
    ├── 1. 读取外部数据 (通过适配器)
    │   ├── adapter.get_user_preferences("user_001")
    │   │   → ["素食主义者", "不吃辣", "预算200元"]
    │   └── adapter.search_relevant_history("user_001", "餐厅")
    │       → ["上次推荐了素心斋", "用户说很满意"]
    │
    ├── 2. DKI 注入处理
    │   ├── 偏好 → K/V 计算 → 负位置注入
    │   └── 历史 → 格式化 → 正位置后缀
    │
    ├── 3. LLM 推理
    │   └── model.forward_with_kv_injection(...)
    │
    └── 4. 返回响应
        └── "推荐素心斋，环境安静，人均150元..."
```

## 与 RAG 的区别

| 特性 | RAG | DKI |
|------|-----|-----|
| 注入级别 | Token 级别 | K/V 级别 |
| Token 消耗 | 消耗上下文窗口 | 偏好不消耗 |
| 注入位置 | 正位置 (Prompt 前缀) | 偏好负位置 + 历史正位置 |
| 控制方式 | 无 | α 连续控制 |

## 配置

```yaml
# config.yaml

dki:
  enabled: true
  
  # 混合注入策略
  hybrid_injection:
    enabled: true
    language: "cn"
    
    # 偏好注入 (K/V 负位置)
    preference:
      enabled: true
      alpha: 0.4
      max_tokens: 100
    
    # 历史注入 (后缀提示词)
    history:
      enabled: true
      max_tokens: 500
      max_messages: 10

# 外部数据适配器
user_adapter:
  type: "postgresql"
  connection_string: "postgresql://user:pass@host:5432/app_db"
  users_table: "users"
  messages_table: "messages"
  preferences_table: "user_preferences"
```

## 监控

### 统计数据

```bash
curl http://localhost:8080/v1/stats
```

```json
{
  "total_requests": 1000,
  "injection_enabled_count": 850,
  "injection_rate": 0.85,
  "cache_hits": 700,
  "cache_hit_rate": 0.82,
  "avg_latency_ms": 150.5,
  "avg_alpha": 0.42
}
```

### 注入日志

```bash
curl http://localhost:8080/v1/logs?limit=10
```

```json
{
  "logs": [
    {
      "request_id": "abc123",
      "timestamp": "2026-02-10T10:30:00Z",
      "injection_enabled": true,
      "alpha": 0.4,
      "preference_tokens": 50,
      "history_tokens": 200,
      "latency_ms": 145.2
    }
  ]
}
```

## 常见问题

### Q: DKI 和 RAG 有什么区别？

DKI 在 K/V 级别注入，偏好不消耗 Token Budget；RAG 在 Token 级别注入，消耗上下文窗口。

### Q: 适配器是做什么的？

适配器用于读取上层应用的数据库，获取用户偏好和历史消息。DKI 不存储数据，只读取。

### Q: 示例应用和 DKI 插件的关系？

示例应用演示如何使用 DKI 插件。生产环境中，上层应用应该有自己的后端系统，通过适配器让 DKI 读取数据。

### Q: 如何监控 DKI 的工作状态？

使用监控 API：`/v1/stats`、`/v1/logs`、`/v1/health`。
