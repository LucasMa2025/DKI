# DKI Database 模块说明书 — `__init__.py`

## 1. 文件概述

| 属性 | 值 |
|------|------|
| **文件路径** | `DKI/dki/database/__init__.py` |
| **模块名称** | `dki.database` |
| **职责** | 数据库包入口，统一导出所有 ORM 模型、Repository 和连接管理器 |
| **行数** | ~48 行 |

## 2. 模块定位

`dki.database` 是 DKI 系统的 **持久化层 (Persistence Layer)**，为上层模块提供统一的数据库访问接口。该包遵循 **Repository 模式**，将数据访问逻辑与业务逻辑分离。

### 2.1 架构层级

```
┌─────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI Routes)                                  │
│  auth_routes / session_routes / preference_routes            │
├─────────────────────────────────────────────────────────────┤
│  Core Layer                                                  │
│  DKISystem / RAGSystem / DKIPlugin / ExperimentRunner        │
├─────────────────────────────────────────────────────────────┤
│  ★ Database Layer (dki.database)  ← 本模块                   │
│  ┌──────────────┬──────────────┬──────────────┐              │
│  │ connection.py│  models.py   │ repository.py│              │
│  │ (连接管理)    │  (ORM 模型)  │  (CRUD 操作) │              │
│  └──────────────┴──────────────┴──────────────┘              │
├─────────────────────────────────────────────────────────────┤
│  Storage (SQLite)                                            │
│  ./data/dki.db                                               │
└─────────────────────────────────────────────────────────────┘
```

## 3. 导出清单

### 3.1 ORM 模型 (来自 `models.py`)

| 导出名 | 类型 | 对应表名 | 说明 |
|--------|------|----------|------|
| `Base` | `DeclarativeBase` | — | SQLAlchemy 声明基类 |
| `Session` | ORM Model | `sessions` | 用户会话 |
| `Memory` | ORM Model | `memories` | 用户记忆 |
| `Conversation` | ORM Model | `conversations` | 对话历史 |
| `KVCache` | ORM Model | `kv_cache` | K/V 缓存持久化 |
| `Experiment` | ORM Model | `experiments` | 实验记录 |
| `ExperimentResult` | ORM Model | `experiment_results` | 实验结果 |
| `AuditLog` | ORM Model | `audit_logs` | 审计日志 |
| `ModelRegistry` | ORM Model | `model_registry` | 模型注册表 |
| `DemoUser` | ORM Model | `demo_users` | 演示用户 |
| `UserPreference` | ORM Model | `user_preferences` | 用户偏好 |

### 3.2 Repository (来自 `repository.py`)

| 导出名 | 操作表 | 说明 |
|--------|--------|------|
| `SessionRepository` | `sessions` | 会话 CRUD |
| `MemoryRepository` | `memories` | 记忆 CRUD + 向量存取 |
| `ConversationRepository` | `conversations` | 对话历史 CRUD |
| `ExperimentRepository` | `experiments` + `experiment_results` | 实验管理 |
| `AuditLogRepository` | `audit_logs` | 审计日志记录 |
| `DemoUserRepository` | `demo_users` | 演示用户管理 |
| `UserPreferenceRepository` | `user_preferences` | 用户偏好管理 |

### 3.3 连接管理 (来自 `connection.py`)

| 导出名 | 类型 | 说明 |
|--------|------|------|
| `DatabaseManager` | Singleton | 数据库连接和会话管理器 |

## 4. 使用方式

### 4.1 基本使用

```python
from dki.database import (
    DatabaseManager,
    SessionRepository,
    MemoryRepository,
    Session,
    Memory,
)

# 获取数据库管理器 (单例)
db_manager = DatabaseManager(db_path="./data/dki.db")

# 使用 session_scope 上下文管理器
with db_manager.session_scope() as db:
    session_repo = SessionRepository(db)
    memory_repo = MemoryRepository(db)
    
    # 创建会话
    session = session_repo.create(user_id="user_123")
    
    # 创建记忆
    memory = memory_repo.create(
        session_id=session.id,
        content="用户喜欢素食",
    )
```

### 4.2 跨模块调用关系

```
dki.core.dki_system.DKISystem
  └── 使用: SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository

dki.core.rag_system.RAGSystem
  └── 使用: SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository

dki.experiment.runner.ExperimentRunner
  └── 使用: ExperimentRepository, DemoUserRepository, UserPreferenceRepository

dki.api.auth_routes
  └── 使用: DemoUserRepository

dki.api.session_routes
  └── 使用: SessionRepository, ConversationRepository

dki.api.preference_routes
  └── 使用: UserPreferenceRepository (通过 DatabaseManager)

dki.web.app
  └── 使用: DatabaseManager, SessionRepository, MemoryRepository, ExperimentRepository
```

## 5. 数据库 ER 图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  demo_users  │     │   sessions   │     │  experiments  │
│──────────────│     │──────────────│     │──────────────│
│ id (PK)      │     │ id (PK)      │     │ id (PK)      │
│ username     │◄────│ user_id (FK) │     │ name         │
│ display_name │     │ is_active    │     │ status       │
│ ...          │     │ ...          │     │ ...          │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       │              ┌─────┴──────┐        ┌─────┴──────────┐
       │              │            │        │                │
       │     ┌────────▼──┐  ┌─────▼──────┐ │ ┌──────────────▼─┐
       │     │ memories   │  │conversations│ │ │experiment_results│
       │     │────────────│  │────────────│ │ │────────────────│
       │     │ id (PK)    │  │ id (PK)    │ │ │ id (PK)        │
       │     │ session_id │  │ session_id │ │ │ experiment_id  │
       │     │ content    │  │ role       │ │ │ mode           │
       │     │ embedding  │  │ content    │ │ │ metrics        │
       │     │ ...        │  │ ...        │ │ │ ...            │
       │     └─────┬──────┘  └────────────┘ │ └────────────────┘
       │           │                        │
       │     ┌─────▼──────┐                 │
       │     │  kv_cache   │                │
       │     │────────────│                 │
       │     │ id (PK)    │                 │
       │     │ memory_id  │                 │
       │     │ model_name │                 │
       │     │ layer_idx  │   ┌─────────────┴───┐
       │     │ key_cache  │   │  audit_logs      │
       │     │ value_cache│   │─────────────────│
       │     │ ...        │   │ id (PK, auto)   │
       │     └────────────┘   │ session_id      │
       │                      │ action          │
       ▼                      │ ...             │
┌──────────────────┐          └─────────────────┘
│ user_preferences │
│──────────────────│    ┌──────────────────┐
│ id (PK)          │    │  model_registry  │
│ user_id          │    │──────────────────│
│ preference_text  │    │ id (PK)          │
│ preference_type  │    │ engine           │
│ priority         │    │ model_name       │
│ ...              │    │ config           │
└──────────────────┘    │ ...              │
                        └──────────────────┘
```

## 6. 注意事项

- **单例模式**: `DatabaseManager` 采用单例模式，全局只有一个数据库连接实例
- **SQLite**: 当前使用 SQLite 作为存储后端，适合演示和小规模部署
- **StaticPool**: 使用 `StaticPool` 连接池，适合 SQLite 的单线程写入特性
- **外键约束**: 通过 PRAGMA 启用 SQLite 的外键约束支持
- **软删除**: 大部分模型使用 `is_active` 字段实现软删除，而非物理删除
