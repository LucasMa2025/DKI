# DKI Database 模块说明书 — `models.py`

## 1. 文件概述

| 属性 | 值 |
|------|------|
| **文件路径** | `DKI/dki/database/models.py` |
| **模块名称** | `dki.database.models` |
| **职责** | 定义所有 SQLAlchemy ORM 模型，映射数据库表结构 |
| **行数** | ~413 行 |
| **模型数量** | 10 个 ORM 模型 |
| **表数量** | 10 张表 |
| **依赖** | `sqlalchemy`, `json`, `datetime` |

## 2. 模块定位

`models.py` 定义了 DKI 系统的 **完整数据模型层**，涵盖以下业务域:

```
┌─────────────────────────────────────────────────────────────┐
│                     DKI 数据模型                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─── 核心业务 ───────────────────────────────────────┐      │
│  │  Session → Memory → KVCache                        │      │
│  │  Session → Conversation                            │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌─── 用户管理 ───────────────────────────────────────┐      │
│  │  DemoUser → UserPreference                         │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌─── 实验系统 ───────────────────────────────────────┐      │
│  │  Experiment → ExperimentResult                     │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌─── 运维辅助 ───────────────────────────────────────┐      │
│  │  AuditLog    ModelRegistry                         │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 3. 通用设计模式

### 3.1 JSON 字段封装

所有模型中的 JSON 类型字段均使用 `Text` 列 + getter/setter 方法实现:

```python
# 数据库列 (Text 类型存储 JSON 字符串)
_metadata = Column('metadata', Text, default='{}')

# Getter: JSON 字符串 → Python dict
def get_metadata(self) -> Dict[str, Any]:
    return json.loads(self._metadata) if self._metadata else {}

# Setter: Python dict → JSON 字符串
def set_metadata(self, value: Dict[str, Any]) -> None:
    self._metadata = json.dumps(value, ensure_ascii=False)
```

**设计原因**: SQLite 不原生支持 JSON 类型，使用 Text + JSON 序列化是通用方案。`ensure_ascii=False` 确保中文字符正确存储。

### 3.2 hybrid_property

部分模型使用 SQLAlchemy 的 `hybrid_property` 实现属性级别的序列化/反序列化:

```python
@hybrid_property
def memory_ids(self) -> List[str]:
    return json.loads(self._memory_ids) if self._memory_ids else []

@memory_ids.setter
def memory_ids(self, value: List[str]):
    self._memory_ids = json.dumps(value)
```

**优势**: 可在 Python 层面作为 `List[str]` 使用，同时在数据库层面存储为 JSON 字符串。

### 3.3 软删除

大部分模型使用 `is_active` 字段实现软删除:

```python
is_active = Column(Boolean, default=True)
```

Repository 层的 `delete()` 方法将 `is_active` 设为 `False`，而非物理删除记录。

### 3.4 时间戳

所有模型都包含 `created_at` 时间戳，部分模型包含 `updated_at`:

```python
created_at = Column(DateTime, default=datetime.utcnow)
updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**注意**: 使用 `datetime.utcnow` 而非 `datetime.now`，确保时间一致性。

---

## 4. 模型详解

### 4.1 Session (会话)

| 属性 | 值 |
|------|------|
| **表名** | `sessions` |
| **主键** | `id` (String, 64) |
| **关联** | `memories` (一对多), `conversations` (一对多) |
| **级联删除** | 是 (`cascade="all, delete-orphan"`) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 会话唯一标识 (格式: `sess_xxxx`) |
| `user_id` | String(64) | — | ✅ | 所属用户 ID |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `updated_at` | DateTime | default=utcnow, onupdate | — | 最后更新时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |
| `is_active` | Boolean | default=True | — | 是否活跃 (软删除) |

#### 关系

```
Session (1) ──┬──→ (N) Memory     (cascade: all, delete-orphan)
              └──→ (N) Conversation (cascade: all, delete-orphan)
```

#### to_dict() 输出示例

```json
{
  "id": "sess_a1b2c3d4e5f6g7h8",
  "user_id": "user_123",
  "created_at": "2026-02-16T10:30:00",
  "updated_at": "2026-02-16T11:45:00",
  "metadata": {"title": "关于美食的对话"},
  "is_active": true,
  "memory_count": 5
}
```

#### 使用场景

- **DKISystem**: 创建/获取会话，关联记忆和对话
- **RAGSystem**: 创建/获取会话，记录对话历史
- **session_routes**: Web UI 会话管理
- **ExperimentRunner**: 实验会话管理

---

### 4.2 Memory (记忆)

| 属性 | 值 |
|------|------|
| **表名** | `memories` |
| **主键** | `id` (String, 64) |
| **外键** | `session_id → sessions.id` (CASCADE) |
| **关联** | `session` (多对一), `kv_caches` (一对多) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 记忆唯一标识 (格式: `mem_xxxx`) |
| `session_id` | String(64) | FK (CASCADE) | ✅ | 所属会话 ID |
| `content` | Text | NOT NULL | — | 记忆文本内容 |
| `embedding` | LargeBinary | — | — | 序列化的 numpy 向量 (float32) |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `updated_at` | DateTime | default=utcnow, onupdate | — | 最后更新时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |
| `is_active` | Boolean | default=True | — | 是否活跃 (软删除) |

#### embedding 字段存储格式

```python
# 存储: numpy array → bytes
memory.embedding = embedding_array.tobytes()

# 读取: bytes → numpy array
embedding = np.frombuffer(memory.embedding, dtype=np.float32)
```

**注意**: 
- 向量维度取决于 Embedding 模型 (通常为 384 或 768 维)
- 存储为 `float32` 格式，每维 4 字节
- 384 维向量占用 384 × 4 = 1,536 字节

#### 关系

```
Session (1) ──→ (N) Memory (1) ──→ (N) KVCache
```

#### 使用场景

- **DKISystem**: 存储用户记忆、检索语义相似记忆
- **RAGSystem**: 存储和检索用于 RAG 的记忆文本
- **MemoryRouter**: FAISS 向量索引的持久化来源
- **ExperimentRunner**: 为实验会话写入 persona 记忆

---

### 4.3 Conversation (对话)

| 属性 | 值 |
|------|------|
| **表名** | `conversations` |
| **主键** | `id` (String, 64) |
| **外键** | `session_id → sessions.id` (CASCADE) |
| **关联** | `session` (多对一) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 对话唯一标识 (格式: `conv_xxxx`) |
| `session_id` | String(64) | FK (CASCADE), NOT NULL | ✅ | 所属会话 ID |
| `role` | String(16) | NOT NULL | — | 角色: `"user"` 或 `"assistant"` |
| `content` | Text | NOT NULL | — | 对话内容 |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `injection_mode` | String(16) | — | — | 注入模式: `"rag"`, `"dki"`, `"none"` |
| `injection_alpha` | Float | — | — | DKI 注入强度 α 值 |
| `_memory_ids` | Text | — | — | JSON 数组: 关联的记忆 ID 列表 |
| `latency_ms` | Float | — | — | 响应延迟 (毫秒) |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### memory_ids hybrid_property

```python
@hybrid_property
def memory_ids(self) -> List[str]:
    """获取关联的记忆 ID 列表"""
    return json.loads(self._memory_ids) if self._memory_ids else []

@memory_ids.setter
def memory_ids(self, value: List[str]):
    """设置关联的记忆 ID 列表"""
    self._memory_ids = json.dumps(value)
```

#### to_dict() 输出示例

```json
{
  "id": "conv_a1b2c3d4e5f6g7h8",
  "session_id": "sess_x1y2z3w4",
  "role": "assistant",
  "content": "根据您的偏好，我推荐这家素食餐厅...",
  "created_at": "2026-02-16T10:31:00",
  "injection_mode": "dki",
  "injection_alpha": 0.4,
  "memory_ids": ["mem_aaa", "mem_bbb"],
  "latency_ms": 245.6,
  "metadata": {"model": "deepseek-7b"}
}
```

#### 使用场景

- **DKISystem.chat()**: 记录用户输入和助手回复，包含注入元数据
- **RAGSystem.chat()**: 记录对话，标记注入模式为 `"rag"`
- **session_routes**: 获取会话的对话历史
- **ConversationRepository.get_recent()**: 获取最近 N 轮对话用于上下文

---

### 4.4 KVCache (K/V 缓存)

| 属性 | 值 |
|------|------|
| **表名** | `kv_cache` |
| **主键** | `id` (String, 64) |
| **外键** | `memory_id → memories.id` (CASCADE) |
| **唯一约束** | `(memory_id, model_name, layer_idx)` |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 缓存唯一标识 |
| `memory_id` | String(64) | FK (CASCADE), NOT NULL | ✅ | 关联的记忆 ID |
| `model_name` | String(128) | NOT NULL | — | 模型名称 |
| `layer_idx` | Integer | NOT NULL | — | Transformer 层索引 |
| `key_cache` | LargeBinary | — | — | 序列化的 Key 张量 |
| `value_cache` | LargeBinary | — | — | 序列化的 Value 张量 |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `last_accessed` | DateTime | default=utcnow | — | 最后访问时间 |
| `access_count` | Integer | default=1 | — | 访问计数 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### 唯一索引

```python
__table_args__ = (
    Index('ix_kv_cache_unique', 'memory_id', 'model_name', 'layer_idx', unique=True),
)
```

**含义**: 同一记忆、同一模型、同一层只能有一条 K/V 缓存记录。当模型切换或更新时，需要重新计算 K/V。

#### 存储格式

```
Key/Value 张量 → tobytes() → LargeBinary 列

张量形状: [num_heads, seq_len, head_dim]
数据类型: float16 (典型)
存储大小: num_heads × seq_len × head_dim × 2 bytes
```

#### 使用场景

- **DKISystem**: 缓存记忆对应的 K/V 对，避免重复计算
- **TieredKVCache L3**: SSD 层持久化存储

---

### 4.5 Experiment (实验)

| 属性 | 值 |
|------|------|
| **表名** | `experiments` |
| **主键** | `id` (String, 64) |
| **关联** | `results` (一对多) |
| **级联删除** | 是 |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 实验唯一标识 (格式: `exp_xxxx`) |
| `name` | String(128) | NOT NULL | — | 实验名称 |
| `description` | Text | — | — | 实验描述 |
| `_config` | Text | NOT NULL | — | JSON: 实验配置 |
| `status` | String(16) | default='pending' | ✅ | 状态: `pending/running/completed/failed` |
| `created_at` | DateTime | default=utcnow | ✅ | 创建时间 |
| `started_at` | DateTime | — | — | 开始时间 |
| `completed_at` | DateTime | — | — | 完成时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### 状态机

```
pending ──→ running ──→ completed
                   └──→ failed
```

#### config hybrid_property

```python
@hybrid_property
def config(self) -> Dict[str, Any]:
    return json.loads(self._config) if self._config else {}

@config.setter
def config(self, value: Dict[str, Any]):
    self._config = json.dumps(value, ensure_ascii=False)
```

#### 使用场景

- **ExperimentRunner**: 创建实验记录、更新状态、存储结果

---

### 4.6 ExperimentResult (实验结果)

| 属性 | 值 |
|------|------|
| **表名** | `experiment_results` |
| **主键** | `id` (String, 64) |
| **外键** | `experiment_id → experiments.id` (CASCADE) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 结果唯一标识 (格式: `res_xxxx`) |
| `experiment_id` | String(64) | FK (CASCADE), NOT NULL | ✅ | 所属实验 ID |
| `mode` | String(16) | NOT NULL | ✅ | 模式: `"rag"`, `"dki"`, `"baseline"` |
| `dataset` | String(64) | NOT NULL | — | 数据集名称 |
| `_metrics` | Text | NOT NULL | — | JSON: 评估指标 |
| `sample_count` | Integer | — | — | 样本数量 |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### metrics 数据示例

```json
{
  "memory_recall": 0.85,
  "hallucination_rate": 0.12,
  "bleu": 0.34,
  "rouge_1": 0.45,
  "rouge_l": 0.41,
  "latency": {
    "mean_ms": 245.6,
    "p50_ms": 230.0,
    "p95_ms": 380.0,
    "p99_ms": 520.0
  }
}
```

---

### 4.7 AuditLog (审计日志)

| 属性 | 值 |
|------|------|
| **表名** | `audit_logs` |
| **主键** | `id` (Integer, 自增) |
| **外键** | 无 (session_id 为逻辑外键) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | Integer | PK, AUTO | ✅ | 自增主键 |
| `session_id` | String(64) | — | ✅ | 关联会话 ID |
| `action` | String(64) | NOT NULL | — | 操作类型 |
| `_memory_ids` | Text | — | — | JSON: 关联的记忆 ID 列表 |
| `alpha` | Float | — | — | 注入 α 值 |
| `mode` | String(16) | — | — | 模式: `"rag"`, `"dki"`, `"none"` |
| `created_at` | DateTime | default=utcnow | ✅ | 创建时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### 常见 action 类型

| action 值 | 说明 | 记录内容 |
|-----------|------|----------|
| `"chat"` | 聊天请求 | alpha, mode, memory_ids |
| `"memory_add"` | 添加记忆 | memory_ids |
| `"memory_delete"` | 删除记忆 | memory_ids |
| `"preference_set"` | 设置偏好 | metadata 中包含偏好详情 |
| `"injection"` | K/V 注入 | alpha, memory_ids, mode |

#### 设计特点

- **自增主键**: 唯一使用 Integer 自增主键的模型（其他模型使用 UUID 字符串）
- **无外键约束**: `session_id` 不设外键，避免日志写入失败影响主流程
- **仅追加**: 审计日志不支持更新或删除

---

### 4.8 ModelRegistry (模型注册表)

| 属性 | 值 |
|------|------|
| **表名** | `model_registry` |
| **主键** | `id` (String, 64) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 模型唯一标识 |
| `engine` | String(16) | NOT NULL | ✅ | 引擎: `"vllm"`, `"llama"`, `"deepseek"`, `"glm"` |
| `model_name` | String(128) | NOT NULL | — | 模型名称/路径 |
| `_config` | Text | — | — | JSON: 模型配置 |
| `is_active` | Boolean | default=True | — | 是否可用 |
| `created_at` | DateTime | default=utcnow | — | 注册时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### 使用场景

- 记录系统中可用的模型信息
- 支持多模型切换和管理

---

### 4.9 DemoUser (演示用户)

| 属性 | 值 |
|------|------|
| **表名** | `demo_users` |
| **主键** | `id` (String, 64) |
| **唯一约束** | `username` |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 用户唯一标识 (格式: `user_xxxx`) |
| `username` | String(64) | NOT NULL, UNIQUE | ✅ | 用户名 |
| `display_name` | String(128) | — | — | 显示名称 |
| `email` | String(128) | — | — | 邮箱 |
| `avatar` | String(256) | — | — | 头像 URL |
| `is_active` | Boolean | default=True | — | 是否活跃 |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `last_login_at` | DateTime | — | — | 最后登录时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### 设计说明

```
演示系统简化认证:
- 只查询用户账号，不验证密码
- 登录时查询用户，如不存在则创建 (get_or_create)
- 确保测试过程中的偏好及会话历史可管理
```

#### to_dict() 输出格式

注意 `to_dict()` 使用 **camelCase** 命名（前端友好）:

```json
{
  "id": "user_a1b2c3d4e5f6g7h8",
  "username": "alice",
  "displayName": "Alice",
  "email": "alice@example.com",
  "avatar": null,
  "isActive": true,
  "createdAt": "2026-02-16T10:00:00",
  "lastLoginAt": "2026-02-16T11:30:00",
  "metadata": {}
}
```

#### 使用场景

- **auth_routes**: 用户登录/注册
- **ExperimentRunner**: 实验用户管理 (`_setup_experiment_users`)

---

### 4.10 UserPreference (用户偏好)

| 属性 | 值 |
|------|------|
| **表名** | `user_preferences` |
| **主键** | `id` (String, 64) |
| **逻辑外键** | `user_id → demo_users.id` (无数据库外键约束) |

#### 字段定义

| 字段名 | 类型 | 约束 | 索引 | 说明 |
|--------|------|------|------|------|
| `id` | String(64) | PK | ✅ | 偏好唯一标识 (格式: `pref_xxxx`) |
| `user_id` | String(64) | NOT NULL | ✅ | 所属用户 ID |
| `preference_text` | Text | NOT NULL | — | 偏好文本内容 |
| `preference_type` | String(32) | default='general' | — | 类型 (见下表) |
| `priority` | Integer | default=5 | — | 优先级 (0-10, 越高越重要) |
| `category` | String(64) | — | — | 可选分类标签 |
| `is_active` | Boolean | default=True | — | 是否活跃 |
| `created_at` | DateTime | default=utcnow | — | 创建时间 |
| `updated_at` | DateTime | default=utcnow, onupdate | — | 最后更新时间 |
| `_metadata` | Text | default='{}' | — | JSON 扩展元数据 |

#### preference_type 枚举值

| 类型 | 说明 | 示例 |
|------|------|------|
| `general` | 通用偏好 | "我喜欢简洁的回答" |
| `style` | 回答风格 | "请用正式语气回复" |
| `technical` | 技术偏好 | "我使用 Python 和 PyTorch" |
| `format` | 格式偏好 | "请用 Markdown 格式" |
| `domain` | 领域偏好 | "我对机器学习感兴趣" |
| `dietary` | 饮食偏好 | "我喜欢素食，对海鲜过敏" |
| `hobby` | 兴趣爱好 | "我喜欢户外运动" |
| `profession` | 职业信息 | "我是软件工程师" |
| `pet` | 宠物信息 | "我养了两只猫" |
| `lifestyle` | 生活方式 | "我喜欢安静的居家生活" |
| `other` | 其他 | — |

#### 在 DKI 注入中的角色

```
UserPreference (数据库)
    ↓ 加载
DKISystem._load_user_preferences_from_db()
    ↓ 转换为
HybridDKIInjector.UserPreference (内存)
    ↓ 编码
model.compute_kv(preference_text)
    ↓ 注入
K/V 负位置注入 (α=0.3-0.5)
```

#### 使用场景

- **DKISystem**: 加载用户偏好用于 K/V 注入
- **DKIPlugin**: 通过 `InjectionPlanner` 格式化偏好文本
- **preference_routes**: Web UI 偏好管理 (CRUD)
- **ExperimentRunner**: 为实验用户写入偏好

---

## 5. 表关系总览

### 5.1 外键关系

| 子表 | 外键字段 | 父表 | 删除策略 |
|------|----------|------|----------|
| `memories` | `session_id` | `sessions` | CASCADE |
| `conversations` | `session_id` | `sessions` | CASCADE |
| `kv_cache` | `memory_id` | `memories` | CASCADE |
| `experiment_results` | `experiment_id` | `experiments` | CASCADE |

### 5.2 逻辑关系 (无外键约束)

| 表 | 字段 | 逻辑关联 | 说明 |
|------|------|----------|------|
| `sessions` | `user_id` | `demo_users.id` | 用户-会话关联 |
| `user_preferences` | `user_id` | `demo_users.id` | 用户-偏好关联 |
| `audit_logs` | `session_id` | `sessions.id` | 日志-会话关联 |

**设计原因**: 逻辑外键不设数据库约束，避免关联表操作失败影响主流程（尤其是审计日志）。

### 5.3 级联删除链

```
删除 Session:
  sessions ──CASCADE──→ memories ──CASCADE──→ kv_cache
  sessions ──CASCADE──→ conversations

删除 Experiment:
  experiments ──CASCADE──→ experiment_results
```

## 6. 索引策略

| 表 | 索引 | 类型 | 用途 |
|------|------|------|------|
| `sessions` | `user_id` | 普通 | 按用户查询会话 |
| `memories` | `session_id` | 普通 | 按会话查询记忆 |
| `conversations` | `session_id` | 普通 | 按会话查询对话 |
| `kv_cache` | `memory_id` | 普通 | 按记忆查询缓存 |
| `kv_cache` | `(memory_id, model_name, layer_idx)` | 唯一 | 防止重复缓存 |
| `experiments` | `status` | 普通 | 按状态筛选实验 |
| `experiments` | `created_at` | 普通 | 按时间排序 |
| `experiment_results` | `experiment_id` | 普通 | 按实验查询结果 |
| `experiment_results` | `mode` | 普通 | 按模式筛选 |
| `audit_logs` | `session_id` | 普通 | 按会话查询日志 |
| `audit_logs` | `created_at` | 普通 | 按时间排序 |
| `model_registry` | `engine` | 普通 | 按引擎筛选 |
| `demo_users` | `username` | 唯一 | 用户名唯一性 |
| `user_preferences` | `user_id` | 普通 | 按用户查询偏好 |
