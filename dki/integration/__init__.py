"""
DKI Integration Layer — 极简化集成入口

三层集成模式 (由简到繁):

Level 1: One-Line Factory (最简集成)
    from dki.integration import create_plugin
    dki = await create_plugin(adapter_config_path="config/adapter.yaml")
    response = await dki.chat("推荐餐厅", user_id="u1", session_id="s1")

Level 2: FastAPI Middleware (Web 应用推荐)
    from dki.integration import DKIMiddleware
    app.add_middleware(DKIMiddleware, adapter_config_path="config/adapter.yaml")

Level 3: Full Control (高级用法)
    from dki.integration import create_plugin
    dki = await create_plugin(
        adapter_config_path="config/adapter.yaml",
        dynamic_router=True,
        message_management=True,
    )

核心设计原则:
- 上层应用只需知道 "我的数据在哪里" 和 "调用 chat()"
- 所有内部组件 (ModelFactory, ConfigDrivenAdapter, CacheConfig) 自动管理
- 向后兼容: 现有 DKIPlugin 接口不变

Author: AGI Demo Project
Version: 4.0.0
"""

from dki.integration.factory import create_plugin, create_plugin_from_dict
from dki.integration.middleware import DKIMiddleware, get_dki_plugin
from dki.integration.enhanced_plugin import EnhancedDKIPlugin

__all__ = [
    # Level 1: One-Line Factory
    "create_plugin",
    "create_plugin_from_dict",
    # Level 2: FastAPI Middleware
    "DKIMiddleware",
    "get_dki_plugin",
    # Level 3: Enhanced Plugin (with routing + message management)
    "EnhancedDKIPlugin",
]
