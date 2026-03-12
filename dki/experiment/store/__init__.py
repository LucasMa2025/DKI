"""
Experiment Store - 实验系统独立持久化层

从 demo/store 复制并简化, 仅保留 SQLite 支持。
使用独立的 dki.db (不与 demo.db 共享)。

设计原则:
- ORM 模型与 demo 完全一致 (demo_users/sessions/messages/preferences 表)
- 通过 DemoBase.metadata.create_all() 自动建表
- ConfigDrivenAdapter 通过相同的表映射读取数据

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store 3.1.0)
"""

from dki.experiment.store.base import (
    IChatStore,
    StoreError,
    StoreConnectionError,
    StoreOperationError,
    StoreNotConnectedError,
)
from dki.experiment.store.connection import ExperimentDBConfig, ExperimentDBManager, PoolStats
from dki.experiment.store.models import DemoBase, DemoUser, DemoSession, DemoMessage, DemoPreference
from dki.experiment.store.factory import create_experiment_store
from dki.experiment.store.bm25_mixin import BM25Mixin

# BaseChatStore for subclasses
try:
    from dki.experiment.store.base_impl import BaseChatStore
except ImportError:
    BaseChatStore = None  # type: ignore

# SQLite store
try:
    from dki.experiment.store.sqlite_store import SQLiteChatStore
except ImportError:
    SQLiteChatStore = None  # type: ignore

__all__ = [
    # Abstract
    "IChatStore",
    # Exceptions
    "StoreError",
    "StoreConnectionError",
    "StoreOperationError",
    "StoreNotConnectedError",
    # Config & Connection
    "ExperimentDBConfig",
    "ExperimentDBManager",
    "PoolStats",
    # Models
    "DemoBase",
    "DemoUser",
    "DemoSession",
    "DemoMessage",
    "DemoPreference",
    # Factory
    "create_experiment_store",
    # Mixins
    "BM25Mixin",
    # Implementations
    "BaseChatStore",
    "SQLiteChatStore",
]
