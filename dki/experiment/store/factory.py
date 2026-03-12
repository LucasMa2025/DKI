"""
Experiment Store Factory — 仅 SQLite

从 demo/store/factory.py 简化, 仅保留 SQLite 支持。

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store/factory.py 3.1.0)
"""

from loguru import logger

from dki.experiment.store.base import IChatStore, StoreError
from dki.experiment.store.connection import ExperimentDBConfig


def create_experiment_store(config: ExperimentDBConfig, **kwargs) -> IChatStore:
    """
    创建实验系统 SQLite Store (已连接)。

    Args:
        config: 数据库配置

    Returns:
        IChatStore 实例 (已连接)
    """
    if config.backend != "sqlite":
        raise StoreError(
            f"Experiment store only supports SQLite, got: {config.backend}"
        )

    from dki.experiment.store.sqlite_store import SQLiteChatStore
    store = SQLiteChatStore(config)
    store.connect()
    logger.info(f"Experiment store created: SQLiteChatStore (path={config.sqlite_path})")
    return store
