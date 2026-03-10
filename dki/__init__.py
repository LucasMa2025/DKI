"""
DKI - Dynamic KV Injection System
Attention-Level Memory Augmentation for Large Language Models

集成方式 (v4.0):
    # 极简集成 (推荐)
    from dki.integration import create_plugin
    dki = await create_plugin(adapter_config_path="config/adapter.yaml")
    response = await dki.chat("推荐餐厅", user_id="u1", session_id="s1")
    
    # FastAPI Middleware
    from dki.integration import DKIMiddleware
    app.add_middleware(DKIMiddleware, adapter_config_path="config/adapter.yaml")

Author: AGI Demo Project
Version: 8.0.0
"""

from dki.core.dki_system import DKISystem
from dki.core.rag_system import RAGSystem
from dki.config.config_loader import ConfigLoader

# v4.0: Integration Layer (极简化集成入口)
try:
    from dki.integration import create_plugin, create_plugin_from_dict, DKIMiddleware
except ImportError:
    # 允许在最小依赖下运行 (不依赖 FastAPI)
    pass

__version__ = "8.0.0"
__all__ = [
    # 核心系统
    "DKISystem",
    "RAGSystem",
    "ConfigLoader",
    # v4.0: 极简集成入口
    "create_plugin",
    "create_plugin_from_dict",
    "DKIMiddleware",
]
