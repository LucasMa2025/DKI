"""
DKI Bridge for Experiment System — ConfigDrivenAdapter 配置生成

将实验系统的 demo_* 表映射为 DKI Plugin 可读取的格式。
使用独立的 dki.db (不与 demo.db 共享)。

核心原理 (与 demo/dki_bridge.py 完全一致):
- ExperimentRunner 管理数据 (写入 demo_messages, demo_preferences 等表)
- DKI Plugin 通过 ConfigDrivenAdapter 只读访问这些表
- 两者共享同一个数据库, 但职责分离 (写 vs 读)

Author: AGI Demo Project — Experiment System Refactoring
Version: 2.0.0
"""

from typing import Any, Dict

from loguru import logger

from dki.experiment.store.connection import ExperimentDBConfig


def build_experiment_adapter_config(db_config: ExperimentDBConfig) -> Dict[str, Any]:
    """
    构建实验系统的 ConfigDrivenAdapter 配置

    参数:
        db_config: 实验系统数据库配置 (SQLite)

    返回:
        ConfigDrivenAdapter 配置字典 (可直接传给 create_plugin(adapter_config=...))
    """
    database_config = {
        "type": "sqlite",
        "database": db_config.sqlite_path,
    }

    config: Dict[str, Any] = {
        "database": database_config,

        # ★ 偏好表映射 (demo_preferences → IUserDataAdapter.get_user_preferences)
        "preferences": {
            "table": "demo_preferences",
            "fields": {
                "user_id": "user_id",
                "preference_id": "id",
                "preference_text": "preference_text",
                "preference_type": "preference_type",
                "priority": "priority",
            },
            "filters": {"is_active": True},
            "order_by": "priority",
            "order_desc": True,
        },

        # ★ 消息表映射 (demo_messages → IUserDataAdapter.search_relevant_history)
        "messages": {
            "table": "demo_messages",
            "fields": {
                "message_id": "id",
                "session_id": "session_id",
                "user_id": "user_id",
                "role": "role",
                "content": "content",
                "timestamp": "created_at",
            },
            "order_by": "created_at",
            "order_desc": True,
        },

        # ★ 用户表映射
        "users": {
            "table": "demo_users",
            "fields": {
                "user_id": "id",
                "username": "username",
            },
        },

        # ★ 会话表映射
        "sessions": {
            "table": "demo_sessions",
            "fields": {
                "session_id": "id",
                "user_id": "user_id",
            },
            "filters": {"is_active": True},
        },

        # 向量检索配置 (BM25-only)
        "vector_search": {
            "enabled": True,
            "type": "dynamic",
        },

        # 缓存
        "cache_enabled": True,
        "cache_ttl": 300,
    }

    logger.info(
        f"Built experiment ConfigDrivenAdapter config "
        f"(backend=sqlite, path={db_config.sqlite_path}, tables=demo_*)"
    )

    return config
