# DKI Database 模块说明书 — `repository.py`

## 1. 文件概述

| 属性 | 值 |
|------|------|
| **文件路径** | `DKI/dki/database/repository.py` |
| **模块名称** | `dki.database.repository` |
| **职责** | 实现 Repository 模式，为所有 ORM 模型提供 CRUD 操作 |
| **行数** | ~589 行 |
| **Repository 数量** | 7 个 (+ 1 个基类) |
| **依赖** | `sqlalchemy`, `numpy`, `loguru`, `dki.database.models` |

## 2. 模块定位

`repository.py` 实现了 **Repository 模式 (仓储模式)**，将数据访问逻辑封装在独立的类中，使上层业务代码无需直接操作 SQLAlchemy Session。

### 2.1 设计优势

```
传统方式 (直接操作 ORM):
  DKISystem → db.query(Memory).filter(...).all()  ← 业务代码与 ORM 耦合

Repository 模式:
  DKISystem → MemoryRepository.get_by_session(session_id)  ← 解耦
```

| 优势 | 说明 |
|------|------|
| **解耦** | 业务逻辑不依赖 ORM 实现细节 |
| **可测试** | 可 Mock Repository 进行单元测试 |
| **一致性** | 统一的 ID 生成、错误处理 |
| **可维护** | 数据库变更只影响 Repository 层 |

### 2.2 调用链

```
┌──────────────────────────────────────────────────────────────┐
│  上层调用方                                                    │
│  DKISystem / RAGSystem / ExperimentRunner / API Routes        │
├──────────────────────────────────────────────────────────────┤
│  DatabaseManager.session_scope()                              │
│  → 产生 SQLAlchemy Session                                    │
├──────────────────────────────────────────────────────────────┤
│  ★ Repository (repository.py)                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │SessionRepo   │ │MemoryRepo    │ │ConversationRepo      │  │
│  │ExperimentRepo│ │AuditLogRepo  │ │DemoUserRepo          │  │
│  │UserPrefRepo  │ │              │ │                      │  │
│  └──────────────┘ └──────────────┘ └──────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│  ORM Models (models.py)                                       │
├──────────────────────────────────────────────────────────────┤
│  SQLite (./data/dki.db)                                       │
└──────────────────────────────────────────────────────────────┘
```

## 3. 基类：`BaseRepository`

```python
class BaseRepository:
    def __init__(self, db: SQLSession):
        self.db = db
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        return f"{prefix}{uuid.uuid4().hex[:16]}"
```

### 3.1 职责

- 持有 SQLAlchemy Session 引用
- 提供统一的 ID 生成方法

### 3.2 ID 生成格式

| 前缀 | 示例 | 使用者 |
|------|------|--------|
| `sess_` | `sess_a1b2c3d4e5f6g7h8` | SessionRepository |
| `mem_` | `mem_a1b2c3d4e5f6g7h8` | MemoryRepository |
| `conv_` | `conv_a1b2c3d4e5f6g7h8` | ConversationRepository |
| `exp_` | `exp_a1b2c3d4e5f6g7h8` | ExperimentRepository |
| `res_` | `res_a1b2c3d4e5f6g7h8` | ExperimentRepository (结果) |
| `user_` | `user_a1b2c3d4e5f6g7h8` | DemoUserRepository |
| `pref_` | `pref_a1b2c3d4e5f6g7h8` | UserPreferenceRepository |

**ID 格式**: `{prefix}{uuid4_hex[:16]}`，总长度为 `len(prefix) + 16`。

### 3.3 使用模式

所有 Repository 的使用方式一致:

```python
db_manager = DatabaseManager()

with db_manager.session_scope() as db:
    repo = SomeRepository(db)
    # ... 执行 CRUD 操作 ...
    # commit 由 session_scope 自动处理
```

---

## 4. SessionRepository

### 4.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `sessions` |
| **主要调用方** | `DKISystem`, `RAGSystem`, `session_routes`, `ExperimentRunner` |

### 4.2 方法列表

#### `create(session_id, user_id, metadata) → Session`

创建新会话。

```
参数:
  session_id: Optional[str]  — 指定 ID 或自动生成 (sess_xxxx)
  user_id: Optional[str]     — 所属用户 ID
  metadata: Optional[Dict]   — 扩展元数据

流程:
  1. 创建 Session 实例 (ID 自动生成或使用指定值)
  2. 如有 metadata, 调用 set_metadata() 序列化为 JSON
  3. db.add() + db.flush()
  4. 返回 Session 实例

注意: flush() 而非 commit(), 事务由 session_scope 管理
```

#### `get(session_id) → Optional[Session]`

按 ID 查询会话。

```
SQL 等价:
  SELECT * FROM sessions WHERE id = :session_id LIMIT 1
```

#### `get_or_create(session_id, user_id) → Tuple[Session, bool]`

获取已有会话或创建新会话。

```
返回: (session, created)
  created = True  → 新创建
  created = False → 已存在

流程:
  1. 调用 get(session_id)
  2. 如果存在 → 返回 (session, False)
  3. 如果不存在 → 调用 create() → 返回 (new_session, True)
```

**使用场景**: `DKISystem.chat()` 和 `RAGSystem.chat()` 中确保会话存在。

#### `list_by_user(user_id, limit=100) → List[Session]`

按用户 ID 列出活跃会话。

```
SQL 等价:
  SELECT * FROM sessions
  WHERE user_id = :user_id AND is_active = True
  ORDER BY updated_at DESC
  LIMIT :limit
```

#### `update(session_id, **kwargs) → Optional[Session]`

更新会话属性。

```
流程:
  1. 获取会话
  2. 遍历 kwargs, 使用 setattr 更新属性
  3. 更新 updated_at 时间戳
  4. flush()
```

#### `delete(session_id) → bool`

软删除会话 (`is_active = False`)。

---

## 5. MemoryRepository

### 5.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `memories` |
| **主要调用方** | `DKISystem`, `RAGSystem`, `MemoryRouter`, `ExperimentRunner` |
| **特殊能力** | 向量 (embedding) 存取 |

### 5.2 方法列表

#### `create(session_id, content, embedding, memory_id, metadata) → Memory`

创建新记忆。

```
参数:
  session_id: str                    — 所属会话 ID
  content: str                       — 记忆文本
  embedding: Optional[np.ndarray]    — 向量表示 (float32)
  memory_id: Optional[str]           — 指定 ID 或自动生成
  metadata: Optional[Dict]           — 扩展元数据

流程:
  1. 创建 Memory 实例
  2. 如有 embedding: memory.embedding = embedding.tobytes()
  3. 如有 metadata: memory.set_metadata(metadata)
  4. db.add() + db.flush()
```

**embedding 序列化**:

```python
# numpy array → bytes (用于存储)
memory.embedding = embedding.tobytes()

# 存储大小计算:
# 384 维 float32 → 384 × 4 = 1,536 bytes
# 768 维 float32 → 768 × 4 = 3,072 bytes
```

#### `get(memory_id) → Optional[Memory]`

按 ID 查询记忆。

#### `get_by_session(session_id, limit=100, include_inactive=False) → List[Memory]`

按会话查询记忆列表。

```
SQL 等价:
  SELECT * FROM memories
  WHERE session_id = :session_id
    [AND is_active = True]          -- 当 include_inactive=False
  ORDER BY created_at DESC
  LIMIT :limit
```

#### `get_embedding(memory_id) → Optional[np.ndarray]`

获取记忆的向量表示。

```
流程:
  1. 获取 Memory 对象
  2. 如果 embedding 存在:
     return np.frombuffer(memory.embedding, dtype=np.float32)
  3. 否则返回 None

注意: 固定使用 float32 类型反序列化
```

#### `update_embedding(memory_id, embedding) → bool`

更新记忆的向量表示。

```
流程:
  1. 获取 Memory 对象
  2. memory.embedding = embedding.tobytes()
  3. 更新 updated_at
  4. flush()
```

#### `search_by_content(session_id, query) → List[Memory]`

基于文本内容的模糊搜索。

```
SQL 等价:
  SELECT * FROM memories
  WHERE session_id = :session_id
    AND is_active = True
    AND content LIKE '%query%'     -- 不区分大小写 (ilike)
```

**注意**: 这是简单的文本匹配，不是语义搜索。语义搜索由 `MemoryRouter` (FAISS) 实现。

#### `delete(memory_id) → bool`

软删除记忆。

#### `bulk_create(session_id, memories_data) → List[Memory]`

批量创建记忆。

```
参数:
  session_id: str
  memories_data: List[Dict[str, Any]]
    每个 dict 包含: content, embedding (可选), id (可选), metadata (可选)

流程:
  遍历 memories_data, 逐条调用 create()

注意: 内部逐条调用 create(), 非真正的 bulk insert
```

---

## 6. ConversationRepository

### 6.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `conversations` |
| **主要调用方** | `DKISystem`, `RAGSystem`, `session_routes` |
| **特殊能力** | 支持按轮次 (turn) 获取最近对话 |

### 6.2 方法列表

#### `create(session_id, role, content, injection_mode, injection_alpha, memory_ids, latency_ms, metadata) → Conversation`

创建对话记录。

```
参数:
  session_id: str          — 所属会话 ID
  role: str                — "user" 或 "assistant"
  content: str             — 对话内容
  injection_mode: str      — "rag", "dki", "none"
  injection_alpha: float   — DKI 注入 α 值
  memory_ids: List[str]    — 关联的记忆 ID 列表
  latency_ms: float        — 响应延迟
  metadata: Dict           — 扩展元数据

典型调用 (DKISystem):
  conversation_repo.create(
      session_id=session_id,
      role="assistant",
      content=response_text,
      injection_mode="dki",
      injection_alpha=0.4,
      memory_ids=["mem_aaa", "mem_bbb"],
      latency_ms=245.6,
  )
```

#### `get_by_session(session_id, limit=100) → List[Conversation]`

获取会话的完整对话历史（按时间正序）。

```
SQL 等价:
  SELECT * FROM conversations
  WHERE session_id = :session_id
  ORDER BY created_at ASC
  LIMIT :limit
```

#### `get_recent(session_id, n_turns=10, limit=None) → List[Conversation]`

获取最近 N 轮对话。

```
参数:
  session_id: str
  n_turns: int = 10        — 对话轮次数 (每轮 = user + assistant)
  limit: Optional[int]     — 替代参数 (兼容性)

实际限制条数:
  actual_limit = limit if limit is not None else n_turns * 2

流程:
  1. 查询最近 actual_limit 条记录 (倒序)
     ORDER BY created_at DESC, id DESC
  2. 反转列表 (变为时间正序)
  3. 返回

排序说明:
  - 主排序: created_at DESC (最新的在前)
  - 次排序: id DESC (同一秒内的消息按 ID 排序)
  - 因为 id 是按时间顺序生成的 (conv_xxxx), 可保证顺序正确

返回顺序: 从旧到新 (经过 reversed())
```

**流程图:**

```
数据库中的对话 (按时间):
  [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8]

get_recent(n_turns=2) → actual_limit = 4

Step 1: 查询 (倒序, LIMIT 4):
  [conv_8, conv_7, conv_6, conv_5]

Step 2: 反转:
  [conv_5, conv_6, conv_7, conv_8]

返回: 最近 2 轮对话 (4 条消息, 时间正序)
```

**使用场景**:
- `DKISystem.get_session_history()`: 获取最近对话用于历史注入
- `RAGSystem._get_session_history()`: 获取对话上下文
- `session_routes`: Web UI 加载对话历史

---

## 7. ExperimentRepository

### 7.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `experiments` + `experiment_results` |
| **主要调用方** | `ExperimentRunner` |

### 7.2 方法列表

#### `create(name, config, description, experiment_id) → Experiment`

创建实验记录。

```
参数:
  name: str                — 实验名称
  config: Dict[str, Any]   — 实验配置 (会序列化为 JSON)
  description: str          — 实验描述
  experiment_id: str        — 指定 ID 或自动生成

流程:
  1. 创建 Experiment 实例
  2. experiment.config = config  → 触发 hybrid_property setter → JSON 序列化
  3. db.add() + db.flush()
```

#### `get(experiment_id) → Optional[Experiment]`

按 ID 查询实验。

#### `list_all(status=None, limit=100) → List[Experiment]`

列出实验记录。

```
SQL 等价:
  SELECT * FROM experiments
  [WHERE status = :status]
  ORDER BY created_at DESC
  LIMIT :limit
```

#### `update_status(experiment_id, status) → Optional[Experiment]`

更新实验状态，自动记录时间戳。

```
流程:
  1. 获取实验
  2. 更新 status
  3. 如果 status == 'running':
       experiment.started_at = utcnow()
  4. 如果 status in ('completed', 'failed'):
       experiment.completed_at = utcnow()
  5. flush()

状态转换:
  pending → running   (记录 started_at)
  running → completed (记录 completed_at)
  running → failed    (记录 completed_at)
```

#### `add_result(experiment_id, mode, dataset, metrics, sample_count, metadata) → ExperimentResult`

添加实验结果。

```
参数:
  experiment_id: str       — 所属实验 ID
  mode: str                — "rag", "dki", "baseline"
  dataset: str             — 数据集名称
  metrics: Dict[str, Any]  — 评估指标 (序列化为 JSON)
  sample_count: int         — 样本数量
  metadata: Dict            — 扩展元数据

典型调用:
  experiment_repo.add_result(
      experiment_id="exp_xxxx",
      mode="dki",
      dataset="persona_chat",
      metrics={
          "memory_recall": 0.85,
          "hallucination_rate": 0.12,
          "bleu": 0.34,
          "latency": {"mean_ms": 245.6}
      },
      sample_count=100,
  )
```

#### `get_results(experiment_id) → List[ExperimentResult]`

获取实验的所有结果。

---

## 8. AuditLogRepository

### 8.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `audit_logs` |
| **主要调用方** | `DKISystem`, `RAGSystem` |
| **特点** | 仅追加，不支持更新/删除 |

### 8.2 方法列表

#### `log(action, session_id, memory_ids, alpha, mode, metadata) → AuditLog`

创建审计日志条目。

```
参数:
  action: str              — 操作类型 (如 "chat", "injection", "memory_add")
  session_id: str          — 关联会话 ID
  memory_ids: List[str]    — 关联的记忆 ID 列表
  alpha: float             — 注入 α 值
  mode: str                — 模式 ("rag", "dki", "none")
  metadata: Dict           — 扩展元数据

典型调用 (DKISystem.chat):
  audit_repo.log(
      action="chat",
      session_id=session_id,
      memory_ids=[m.id for m in used_memories],
      alpha=gating_result.alpha,
      mode="dki",
      metadata={
          "query": user_query,
          "response_length": len(response),
          "latency_ms": latency,
      },
  )
```

#### `get_by_session(session_id, limit=100) → List[AuditLog]`

获取会话的审计日志（按时间倒序）。

```
SQL 等价:
  SELECT * FROM audit_logs
  WHERE session_id = :session_id
  ORDER BY created_at DESC
  LIMIT :limit
```

---

## 9. DemoUserRepository

### 9.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `demo_users` |
| **主要调用方** | `auth_routes`, `ExperimentRunner` |
| **特点** | 简化认证 (无密码验证) |

### 9.2 方法列表

#### `create(username, user_id, display_name, email) → DemoUser`

创建演示用户。

```
参数:
  username: str            — 用户名 (唯一)
  user_id: str             — 指定 ID 或自动生成 (user_xxxx)
  display_name: str        — 显示名称 (默认 = username)
  email: str               — 邮箱
```

#### `get(user_id) → Optional[DemoUser]`

按 ID 查询用户。

#### `get_by_username(username) → Optional[DemoUser]`

按用户名查询用户。

```
SQL 等价:
  SELECT * FROM demo_users WHERE username = :username LIMIT 1
```

#### `get_or_create(username, display_name, email) → Tuple[DemoUser, bool]`

获取已有用户或创建新用户。

```
返回: (user, created)
  created = True  → 新创建
  created = False → 已存在 (同时更新 last_login_at)

流程:
  1. 调用 get_by_username(username)
  2. 如果存在:
     - 更新 last_login_at = utcnow()
     - flush()
     - 返回 (user, False)
  3. 如果不存在:
     - 调用 create()
     - 返回 (new_user, True)
```

**使用场景**:

```python
# auth_routes.py 登录流程
user, created = user_repo.get_or_create(
    username=request.username,
    display_name=request.username,
)
if created:
    logger.info(f"New user registered: {request.username}")
```

#### `list_all(limit=100) → List[DemoUser]`

列出所有活跃用户（按最后登录时间倒序）。

#### `update(user_id, **kwargs) → Optional[DemoUser]`

更新用户属性。

#### `delete(user_id) → bool`

软删除用户。

---

## 10. UserPreferenceRepository

### 10.1 概述

| 属性 | 值 |
|------|------|
| **操作表** | `user_preferences` |
| **主要调用方** | `preference_routes`, `ExperimentRunner`, `DKISystem` |
| **特点** | 支持按类型、优先级筛选和排序 |

### 10.2 方法列表

#### `create(user_id, preference_text, preference_type, priority, category, preference_id) → UserPreference`

创建用户偏好。

```
参数:
  user_id: str                — 所属用户 ID
  preference_text: str        — 偏好文本
  preference_type: str        — 类型 (默认 "general")
  priority: int               — 优先级 0-10 (默认 5)
  category: str               — 分类标签
  preference_id: str          — 指定 ID 或自动生成

典型调用:
  pref_repo.create(
      user_id="user_xxxx",
      preference_text="我喜欢素食，对海鲜过敏",
      preference_type="dietary",
      priority=10,
  )
```

#### `get(preference_id) → Optional[UserPreference]`

按 ID 查询偏好。

#### `get_by_user(user_id, preference_type=None, active_only=True) → List[UserPreference]`

获取用户的所有偏好。

```
参数:
  user_id: str                    — 用户 ID
  preference_type: Optional[str]  — 按类型筛选 (如 "dietary")
  active_only: bool               — 是否只返回活跃偏好 (默认 True)

SQL 等价:
  SELECT * FROM user_preferences
  WHERE user_id = :user_id
    [AND preference_type = :preference_type]
    [AND is_active = True]
  ORDER BY priority DESC, created_at DESC

排序: 优先级高的在前, 同优先级按创建时间倒序
```

**使用场景**:

```python
# DKISystem._load_user_preferences_from_db()
preferences = pref_repo.get_by_user(user_id=user_id)
for pref in preferences:
    # 转换为 HybridDKIInjector.UserPreference
    user_pref = UserPreference(
        text=pref.preference_text,
        type=pref.preference_type,
        priority=pref.priority,
    )
```

#### `update(preference_id, **kwargs) → Optional[UserPreference]`

更新偏好属性。

```
流程:
  1. 获取偏好
  2. 遍历 kwargs, 使用 setattr 更新属性
  3. 更新 updated_at
  4. flush()
```

#### `delete(preference_id) → bool`

软删除偏好。

#### `list_all(limit=100) → List[UserPreference]`

列出所有活跃偏好（按创建时间倒序）。

---

## 11. 典型使用流程

### 11.1 DKISystem.chat() 数据库交互流程

```
DKISystem.chat(query, session_id)
  │
  ├── 1. 获取/创建会话
  │      with db_manager.session_scope() as db:
  │          session_repo = SessionRepository(db)
  │          session, created = session_repo.get_or_create(session_id, user_id)
  │
  ├── 2. 加载用户偏好
  │      with db_manager.session_scope() as db:
  │          pref_repo = UserPreferenceRepository(db)
  │          preferences = pref_repo.get_by_user(user_id)
  │
  ├── 3. 获取历史对话
  │      with db_manager.session_scope() as db:
  │          conv_repo = ConversationRepository(db)
  │          history = conv_repo.get_recent(session_id, n_turns=10)
  │
  ├── 4. 检索相关记忆
  │      with db_manager.session_scope() as db:
  │          memory_repo = MemoryRepository(db)
  │          memories = memory_repo.get_by_session(session_id)
  │
  ├── 5. [DKI 注入 + LLM 推理]
  │
  ├── 6. 记录用户消息
  │      with db_manager.session_scope() as db:
  │          conv_repo = ConversationRepository(db)
  │          conv_repo.create(
  │              session_id=session_id,
  │              role="user",
  │              content=query,
  │          )
  │
  ├── 7. 记录助手回复
  │      conv_repo.create(
  │          session_id=session_id,
  │          role="assistant",
  │          content=response_text,
  │          injection_mode="dki",
  │          injection_alpha=alpha,
  │          memory_ids=used_memory_ids,
  │          latency_ms=latency,
  │      )
  │
  └── 8. 记录审计日志
         with db_manager.session_scope() as db:
             audit_repo = AuditLogRepository(db)
             audit_repo.log(
                 action="chat",
                 session_id=session_id,
                 memory_ids=used_memory_ids,
                 alpha=alpha,
                 mode="dki",
             )
```

### 11.2 ExperimentRunner 数据库交互流程

```
ExperimentRunner.run_comparison()
  │
  ├── 1. 创建实验记录
  │      experiment_repo.create(name, config)
  │
  ├── 2. 更新状态为 running
  │      experiment_repo.update_status(exp_id, "running")
  │
  ├── 3. 设置实验用户 (_setup_experiment_users)
  │      ├── demo_user_repo.get_or_create(username)
  │      └── pref_repo.create(user_id, preference_text, ...)
  │
  ├── 4. 对每个样本运行实验
  │      ├── session_repo.create(session_id, user_id)
  │      ├── memory_repo.create(session_id, persona_content)
  │      ├── [DKI/RAG 推理]
  │      └── conv_repo.create(session_id, role, content, ...)
  │
  ├── 5. 添加实验结果
  │      experiment_repo.add_result(exp_id, mode, dataset, metrics)
  │
  └── 6. 更新状态为 completed
         experiment_repo.update_status(exp_id, "completed")
```

### 11.3 用户登录流程 (auth_routes)

```
POST /api/auth/login
  │
  ├── 1. 接收 LoginRequest (username, password)
  │
  ├── 2. 查询或创建用户
  │      with db_manager.session_scope() as db:
  │          user_repo = DemoUserRepository(db)
  │          user, created = user_repo.get_or_create(username)
  │          │
  │          ├── 已存在: 更新 last_login_at
  │          └── 不存在: 创建新用户
  │
  ├── 3. 生成 Token (内存存储)
  │      token = uuid4().hex
  │      _tokens_db[token] = user_info
  │
  └── 4. 返回 LoginResponse (token, user)
```

### 11.4 偏好管理流程 (preference_routes)

```
POST /api/preferences/
  │
  ├── 1. 验证用户 Token (require_auth)
  │
  ├── 2. 创建偏好
  │      with db_manager.session_scope() as db:
  │          pref_repo = UserPreferenceRepository(db)
  │          pref = pref_repo.create(
  │              user_id=request.user_id,
  │              preference_text=request.preference_text,
  │              preference_type=request.preference_type,
  │              priority=request.priority,
  │          )
  │
  └── 3. 返回 PreferenceResponse

GET /api/preferences/?user_id=xxx
  │
  ├── 1. 验证用户 Token
  │
  ├── 2. 查询偏好
  │      preferences = pref_repo.get_by_user(user_id)
  │
  └── 3. 返回 List[PreferenceResponse]
```

## 12. 事务管理说明

### 12.1 flush vs commit

| 操作 | 说明 | 使用场景 |
|------|------|----------|
| `flush()` | 将变更写入数据库但不提交事务 | Repository 内部 (所有方法) |
| `commit()` | 提交事务，使变更永久生效 | `session_scope()` 正常退出时 |
| `rollback()` | 回滚事务，撤销所有变更 | `session_scope()` 异常时 |

**设计原则**: Repository 方法只调用 `flush()`，事务的 `commit/rollback` 由 `session_scope()` 统一管理。这确保了:
- 多个 Repository 操作可以在同一事务中执行
- 任何一步失败都会回滚整个事务

### 12.2 示例

```python
with db_manager.session_scope() as db:
    session_repo = SessionRepository(db)
    conv_repo = ConversationRepository(db)
    audit_repo = AuditLogRepository(db)
    
    # 以下三个操作在同一事务中
    session_repo.create(session_id="sess_1", user_id="user_1")  # flush
    conv_repo.create(session_id="sess_1", role="user", content="hello")  # flush
    audit_repo.log(action="chat", session_id="sess_1")  # flush
    
    # with 块结束时自动 commit
    # 如果任何一步抛出异常，所有操作都会 rollback
```

## 13. 注意事项与潜在问题

### 13.1 embedding 类型固定

`MemoryRepository.get_embedding()` 固定使用 `np.float32` 反序列化:

```python
return np.frombuffer(memory.embedding, dtype=np.float32)
```

如果存储时使用了其他 dtype (如 `float16`)，反序列化会产生错误结果。建议在 metadata 中记录 dtype 信息。

### 13.2 bulk_create 非真正批量

`MemoryRepository.bulk_create()` 内部逐条调用 `create()`，每次都会 `flush()`。对于大量数据，性能可能不理想。如需高性能批量插入，应使用 SQLAlchemy 的 `bulk_save_objects()` 或 `bulk_insert_mappings()`。

### 13.3 软删除一致性

软删除 (`is_active=False`) 的记录在某些查询中可能被意外包含:
- `get()` 方法不检查 `is_active`，会返回已删除的记录
- `get_by_session()` 默认排除已删除记录 (`include_inactive=False`)
- 建议在所有查询中明确处理 `is_active` 条件

### 13.4 ConversationRepository.get_recent 排序

使用 `id` 作为次要排序键依赖于 ID 的生成顺序。由于 ID 使用 UUID 随机生成 (`uuid4().hex[:16]`)，其字典序不一定与创建顺序一致。在高并发场景下，同一秒内创建的多条记录可能排序不正确。
