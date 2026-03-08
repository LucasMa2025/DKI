"""
DKI Bridge — ConfigDrivenAdapter 配置生成

将 Demo 的数据库表映射为 DKI Plugin 可读取的格式。
DKI Plugin 通过 ConfigDrivenAdapter 只读访问 Demo 的数据库。

核心原理:
- Demo App 管理数据 (写入 demo_messages, demo_preferences 等表)
- DKI Plugin 通过 ConfigDrivenAdapter 读取这些表
- 两者共享同一个数据库, 但职责分离 (写 vs 读)

Author: AGI Demo Project
Version: 2.0.0
"""

from typing import Any, Dict, Optional

from loguru import logger

from demo.store.connection import DemoDBConfig


def build_adapter_config(db_config: DemoDBConfig) -> Dict[str, Any]:
    """
    构建 ConfigDrivenAdapter 配置
    
    将 Demo 的数据库表映射为 DKI Plugin 可读取的格式
    
    参数:
        db_config: Demo 数据库配置
    
    返回:
        ConfigDrivenAdapter 配置字典 (可直接传给 ConfigDrivenAdapter.from_dict)
    """
    # 数据库连接配置
    if db_config.backend == "sqlite":
        database_config = {
            "type": "sqlite",
            "database": db_config.sqlite_path,
        }
    else:
        database_config = {
            "type": "postgresql",
            "host": db_config.pg_host,
            "port": db_config.pg_port,
            "database": db_config.pg_database,
            "username": db_config.pg_username,
            "password": db_config.pg_password,
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
        
        # 向量检索配置 (默认 BM25-only)
        "vector_search": {
            "enabled": True,
            "type": "dynamic",  # BM25-only (无 vector_index_config)
        },
        
        # 缓存
        "cache_enabled": True,
        "cache_ttl": 300,
    }
    
    # pgvector 模式: 添加 vector_index_config
    if db_config.backend == "pgvector" and db_config.pgvector_enabled:
        config["messages"]["fields"]["embedding"] = "embedding_vector"
        config["vector_search"] = {
            "enabled": True,
            "type": "dynamic",
            "vector_index_config": {
                "core": {
                    "index_type": "HNSW",
                    "dimension": db_config.embedding_dim,
                    "similarity_metric": "cosine",
                },
                "embedding": {
                    "api_type": "local",
                    "model_name": "all-MiniLM-L6-v2",
                    "normalization": True,
                },
                "retrieval": {"top_k": 10},
            },
        }
    
    logger.info(
        f"Built ConfigDrivenAdapter config for Demo "
        f"(backend={db_config.backend}, "
        f"pgvector={'enabled' if db_config.pgvector_enabled else 'disabled'})"
    )
    
    return config
