"""
Demo Application Configuration

独立于 DKI/dki/config/config_loader.py 的配置系统。
支持从环境变量、YAML 文件或字典加载配置。

Author: AGI Demo Project
Version: 2.0.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class DemoServerConfig:
    """Demo 服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False
    workers: int = 1
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class DemoDKIConfig:
    """DKI Plugin 集成配置"""
    language: str = "cn"
    # 模型配置 (继承主 config.yaml)
    config_path: str = "config/config.yaml"
    # 适配器配置 (由 dki_bridge.py 自动生成, 指向 Demo 数据库)
    # 如果提供了 adapter_config_path, 则使用文件配置
    adapter_config_path: Optional[str] = None
    # Redis 配置
    enable_redis: bool = True
    redis_url: str = "redis://:hello2006@127.0.0.1:6379/0"


@dataclass
class DemoConfig:
    """Demo 应用主配置"""
    # 服务器
    server: DemoServerConfig = field(default_factory=DemoServerConfig)
    
    # 数据库 (独立于实验系统)
    # 使用 DemoDBConfig (from demo.store.connection)
    # 这里只存储原始配置字典, 由 app.py 解析为 DemoDBConfig
    # 默认使用 sqlite (开箱即用, 无需额外配置)
    # 切换到 postgresql/pgvector 只需修改 backend 字段
    db: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "sqlite",
        "sqlite_path": "./data/demo.db",
        "pg_host": "localhost",
        "pg_port": 5432,
        "pg_database": "dkidemo",
        "pg_username": "postgres",
        "pg_password": "pg_2025",
        "pgvector_enabled": True,
        "embedding_dim": 768,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
    })
    
    # DKI Plugin 集成
    dki: DemoDKIConfig = field(default_factory=DemoDKIConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemoConfig":
        """从字典创建配置"""
        config = cls()
        
        # 服务器配置
        if "server" in data:
            srv = data["server"]
            config.server = DemoServerConfig(
                host=srv.get("host", "0.0.0.0"),
                port=srv.get("port", 8080),
                reload=srv.get("reload", False),
                workers=srv.get("workers", 1),
                cors_origins=srv.get("cors_origins", ["*"]),
            )
        
        # 数据库配置
        if "database" in data:
            config.db = data["database"]
        
        # DKI 配置
        if "dki_plugin" in data:
            dki = data["dki_plugin"]
            config.dki = DemoDKIConfig(
                language=dki.get("language", "cn"),
                config_path=dki.get("config_path", "config/config.yaml"),
                adapter_config_path=dki.get("adapter_config_path"),
                enable_redis=dki.get("enable_redis", True),
                redis_url=dki.get("redis_url", "redis://:hello2006@127.0.0.1:6379/0"),
            )
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DemoConfig":
        """从 YAML 文件加载配置"""
        import yaml
        
        if not Path(yaml_path).exists():
            logger.warning(f"Config file not found: {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # 支持 demo: 前缀或直接配置
        if "demo" in data:
            data = data["demo"]
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> "DemoConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 服务器
        config.server.host = os.getenv("DEMO_HOST", "0.0.0.0")
        config.server.port = int(os.getenv("DEMO_PORT", "8080"))
        
        # 数据库 (默认 sqlite)
        backend = os.getenv("DEMO_DB_BACKEND", "sqlite")
        config.db = {
            "backend": backend,
            "sqlite_path": os.getenv("DEMO_DB_SQLITE_PATH", "./data/demo.db"),
            "pg_host": os.getenv("DEMO_DB_PG_HOST", "localhost"),
            "pg_port": int(os.getenv("DEMO_DB_PG_PORT", "5432")),
            "pg_database": os.getenv("DEMO_DB_PG_DATABASE", "dkidemo"),
            "pg_username": os.getenv("DEMO_DB_PG_USERNAME", "postgres"),
            "pg_password": os.getenv("DEMO_DB_PG_PASSWORD", "pg_2025"),
            "pgvector_enabled": os.getenv("DEMO_DB_PGVECTOR_ENABLED", "true").lower() == "true",
            "embedding_dim": int(os.getenv("DEMO_DB_EMBEDDING_DIM", "768")),
            "pool_size": int(os.getenv("DEMO_DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DEMO_DB_MAX_OVERFLOW", "20")),
            "echo": os.getenv("DEMO_DB_ECHO", "false").lower() == "true",
        }
        
        # DKI
        config.dki.language = os.getenv("DEMO_DKI_LANGUAGE", "cn")
        # DKI 配置文件路径: 优先 DEMO_DKI_CONFIG_PATH > DKI_CONFIG_PATH > 默认
        config.dki.config_path = os.getenv(
            "DEMO_DKI_CONFIG_PATH",
            os.getenv("DKI_CONFIG_PATH", "config/config.yaml")
        )
        config.dki.adapter_config_path = os.getenv("DEMO_DKI_ADAPTER_CONFIG_PATH")
        config.dki.enable_redis = os.getenv("DEMO_DKI_ENABLE_REDIS", "true").lower() == "true"
        config.dki.redis_url = os.getenv("DEMO_DKI_REDIS_URL", "redis://:hello2006@127.0.0.1:6379/0")
        
        return config


def load_demo_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
) -> DemoConfig:
    """
    加载 Demo 配置
    
    优先级: config_path > 环境变量 > 默认值
    """
    # 尝试从文件加载
    if config_path and Path(config_path).exists():
        logger.info(f"Loading demo config from: {config_path}")
        return DemoConfig.from_yaml(config_path)
    
    # 尝试默认路径
    default_paths = [
        "config/demo.yaml",
        "demo.yaml",
        "../config/demo.yaml",
    ]
    for path in default_paths:
        if Path(path).exists():
            logger.info(f"Loading demo config from: {path}")
            return DemoConfig.from_yaml(path)
    
    # 从环境变量加载
    if use_env:
        logger.info("Loading demo config from environment variables")
        return DemoConfig.from_env()
    
    # 使用默认值
    logger.info("Using default demo config")
    return DemoConfig()
