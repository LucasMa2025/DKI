# DKI Database 模块说明书 — `connection.py`

## 1. 文件概述

| 属性 | 值 |
|------|------|
| **文件路径** | `DKI/dki/database/connection.py` |
| **模块名称** | `dki.database.connection` |
| **职责** | 数据库连接管理、会话工厂、表初始化 |
| **行数** | ~167 行 |
| **核心类** | `DatabaseManager` (单例) |
| **辅助函数** | `get_db()` (FastAPI 依赖注入) |
| **依赖** | `sqlalchemy`, `loguru`, `dki.database.models.Base` |

## 2. 模块定位

`connection.py` 是 DKI 数据库层的 **连接管理核心**，负责：
1. 创建和管理 SQLAlchemy Engine
2. 提供 Session 工厂和上下文管理器
3. 自动建表 (DDL)
4. 为 FastAPI 提供依赖注入函数

### 2.1 在系统中的位置

```
┌─────────────────────────────────────────────┐
│  上层调用方                                   │
│  DKISystem / RAGSystem / ExperimentRunner    │
│  FastAPI Routes (auth / session / preference)│
├─────────────────────────────────────────────┤
│  ★ DatabaseManager (connection.py)           │
│  ┌────────────┐  ┌──────────────┐            │
│  │ Engine     │  │ SessionMaker │            │
│  │ (SQLite)   │  │ (Factory)    │            │
│  └─────┬──────┘  └──────┬───────┘            │
│        │                │                    │
│        ▼                ▼                    │
│  ┌─────────────────────────────┐             │
│  │  SQLite File (./data/dki.db)│             │
│  └─────────────────────────────┘             │
└─────────────────────────────────────────────┘
```

## 3. 核心类：`DatabaseManager`

### 3.1 类签名

```python
class DatabaseManager:
    _instance: Optional['DatabaseManager'] = None  # 单例实例
    
    def __init__(
        self,
        db_path: str = "./data/dki.db",
        echo: bool = False,
        pool_size: int = 5,
    )
```

### 3.2 设计模式：单例 (Singleton)

`DatabaseManager` 使用 `__new__` + `_initialized` 标志实现单例模式，确保全局只有一个数据库连接实例。

```
首次调用:
  __new__() → 创建实例, _initialized = False
  __init__() → _initialized == False → 执行初始化 → _initialized = True

后续调用:
  __new__() → 返回已有实例
  __init__() → _initialized == True → 直接 return (跳过初始化)
```

**关键代码:**

```python
def __new__(cls, *args, **kwargs):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
        cls._instance._initialized = False
    return cls._instance

def __init__(self, db_path, echo, pool_size):
    if self._initialized:
        return          # ← 跳过重复初始化
    # ... 初始化逻辑 ...
    self._initialized = True
```

### 3.3 初始化流程

```
DatabaseManager.__init__()
  │
  ├── 1. 检查 _initialized → 如果已初始化则直接返回
  │
  ├── 2. 保存配置参数
  │      db_path, echo, pool_size
  │
  ├── 3. _init_database()
  │      │
  │      ├── 3.1 创建数据目录 (mkdir -p)
  │      │      Path(db_path).parent.mkdir(parents=True, exist_ok=True)
  │      │
  │      ├── 3.2 创建 SQLAlchemy Engine
  │      │      create_engine(
  │      │          "sqlite:///./data/dki.db",
  │      │          echo=False,
  │      │          connect_args={"check_same_thread": False},
  │      │          poolclass=StaticPool,
  │      │      )
  │      │
  │      ├── 3.3 注册 SQLite PRAGMA 事件监听器
  │      │      @event.listens_for(engine, "connect")
  │      │      → PRAGMA foreign_keys=ON
  │      │
  │      ├── 3.4 创建 Session 工厂
  │      │      sessionmaker(
  │      │          bind=engine,
  │      │          autocommit=False,
  │      │          autoflush=False,
  │      │      )
  │      │
  │      └── 3.5 建表 (DDL)
  │             Base.metadata.create_all(engine)
  │             → 自动创建所有 ORM 模型对应的表
  │
  └── 4. _initialized = True
```

### 3.4 关键配置说明

| 配置项 | 值 | 说明 |
|--------|------|------|
| `check_same_thread` | `False` | 允许 SQLite 跨线程使用（FastAPI 异步环境必需） |
| `poolclass` | `StaticPool` | 使用静态连接池，所有请求共享同一连接（适合 SQLite） |
| `autocommit` | `False` | 手动控制事务提交 |
| `autoflush` | `False` | 手动控制刷新，避免意外写入 |
| `PRAGMA foreign_keys=ON` | — | SQLite 默认不启用外键约束，需手动开启 |

## 4. 主要方法

### 4.1 `get_session() → SQLSession`

```python
def get_session(self) -> SQLSession:
    """获取新的数据库会话"""
    return self._session_factory()
```

- **用途**: 创建一个新的 SQLAlchemy Session 对象
- **注意**: 调用方需自行管理 commit/rollback/close
- **推荐**: 优先使用 `session_scope()` 上下文管理器

### 4.2 `session_scope() → Generator[SQLSession]`

```python
@contextmanager
def session_scope(self) -> Generator[SQLSession, None, None]:
    session = self.get_session()
    try:
        yield session
        session.commit()      # ← 正常退出自动提交
    except Exception as e:
        session.rollback()    # ← 异常自动回滚
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()       # ← 无论如何都关闭会话
```

**流程图:**

```
with db_manager.session_scope() as session:
    # 执行数据库操作
    │
    ├── 正常执行完毕 → session.commit() → session.close()
    │
    └── 抛出异常 → session.rollback() → session.close() → re-raise
```

**使用示例:**

```python
db_manager = DatabaseManager()

with db_manager.session_scope() as db:
    repo = SessionRepository(db)
    session = repo.create(user_id="user_123")
    # commit 在 with 块结束时自动调用
```

### 4.3 `execute_script(script_path: str) → None`

```python
def execute_script(self, script_path: str) -> None:
```

- **用途**: 执行外部 SQL 脚本文件
- **流程**:
  1. 检查文件是否存在
  2. 读取 SQL 文件内容
  3. 按 `;` 分割为多条 SQL 语句
  4. 在 `session_scope` 中逐条执行
  5. 单条语句失败不影响其他语句（仅记录 warning）

### 4.4 `drop_all() → None`

```python
def drop_all(self) -> None:
    """删除所有表 (危险操作)"""
    Base.metadata.drop_all(self._engine)
```

- **用途**: 删除数据库中所有表
- **警告**: 此操作不可逆，仅用于开发/测试

### 4.5 `reset() → None`

```python
def reset(self) -> None:
    """重置数据库 (删除并重建所有表)"""
    self.drop_all()
    Base.metadata.create_all(self._engine)
```

- **用途**: 清空数据库并重建表结构
- **场景**: 开发环境数据库迁移、测试前清理

### 4.6 `get_instance() → DatabaseManager` (类方法)

```python
@classmethod
def get_instance(cls) -> 'DatabaseManager':
    if cls._instance is None or not cls._instance._initialized:
        return cls()        # ← 使用默认参数创建
    return cls._instance    # ← 返回已有实例
```

- **用途**: 获取单例实例
- **注意**: 如果实例不存在，会使用 **默认参数** (`db_path="./data/dki.db"`) 创建

### 4.7 `reset_instance() → None` (类方法)

```python
@classmethod
def reset_instance(cls) -> None:
    if cls._instance is not None:
        if cls._instance._engine is not None:
            cls._instance._engine.dispose()  # ← 释放连接池
        cls._instance._initialized = False
        cls._instance = None
```

- **用途**: 重置单例实例（释放连接池 + 清空实例引用）
- **场景**: 单元测试 teardown、配置变更后重新初始化

## 5. 辅助函数：`get_db()`

```python
def get_db() -> Generator[SQLSession, None, None]:
    """FastAPI 依赖注入函数"""
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

### 5.1 用途

为 FastAPI 路由提供数据库会话的依赖注入:

```python
from fastapi import Depends
from dki.database.connection import get_db

@router.get("/sessions")
def list_sessions(db=Depends(get_db)):
    repo = SessionRepository(db)
    return repo.list_by_user(user_id)
```

### 5.2 与 `session_scope()` 的区别

| 特性 | `session_scope()` | `get_db()` |
|------|-------------------|------------|
| **使用场景** | 非 Web 环境 (Core / Experiment) | FastAPI 路由 |
| **调用方式** | `with db_manager.session_scope() as db:` | `Depends(get_db)` |
| **事务管理** | 相同 (auto-commit / rollback) | 相同 |
| **生命周期** | 由 `with` 块控制 | 由 FastAPI 请求生命周期控制 |

## 6. 线程安全性分析

| 组件 | 线程安全 | 说明 |
|------|----------|------|
| `DatabaseManager` 单例 | ⚠️ 非严格线程安全 | `__new__` 中无锁保护，但实际使用中通常在启动时初始化 |
| `Engine` | ✅ 线程安全 | SQLAlchemy Engine 本身是线程安全的 |
| `Session` | ❌ 非线程安全 | 每次调用 `get_session()` 创建新实例，不跨线程共享 |
| `StaticPool` | ✅ 适合 SQLite | SQLite 单写多读，StaticPool 确保连接复用 |

## 7. 调用方汇总

| 调用方 | 使用方式 | 说明 |
|--------|----------|------|
| `DKISystem.__init__` | `DatabaseManager(db_path=config.database.path)` | 核心系统初始化 |
| `RAGSystem.__init__` | `DatabaseManager(db_path=config.database.path)` | RAG 系统初始化 |
| `ExperimentRunner` | `DatabaseManager(db_path=config.database.path)` | 实验运行器 |
| `auth_routes` | `DatabaseManager(db_path=config.database.path)` | 用户认证 API |
| `session_routes` | `DatabaseManager(db_path=config.database.path)` | 会话管理 API |
| `preference_routes` | `DatabaseManager(db_path=config.database.path)` | 偏好管理 API |
| `web/app.py` | `get_db()` + `DatabaseManager()` | FastAPI 应用 |

## 8. 配置来源

数据库路径通过 `config.yaml` 配置:

```yaml
database:
  path: "./data/dki.db"
  echo: false
```

对应的配置加载:

```python
config = ConfigLoader().config
db_manager = DatabaseManager(
    db_path=config.database.path,
    echo=config.database.echo,
)
```
