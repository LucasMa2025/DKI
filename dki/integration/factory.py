"""
DKI Integration Factory — 极简化插件创建

提供两种工厂方法:
1. create_plugin(): 从配置文件创建 (推荐, 类似 AGA 的 from_config)
2. create_plugin_from_dict(): 从配置字典创建

核心设计:
- 上层应用只需提供 adapter_config_path (数据库映射配置)
- ModelFactory, ConfigDrivenAdapter, Redis, CacheConfig 全部自动管理
- 支持可选的 dynamic_router 和 message_management 增强功能
- 支持可选的 RAG 系统 (用于动态路由)
- 支持可选的 Store (用于消息管理)

使用方式:
```python
# 最简 (3 行集成)
from dki.integration import create_plugin
dki = await create_plugin(adapter_config_path="config/adapter.yaml")
response = await dki.chat("推荐餐厅", user_id="u1", session_id="s1")

# 启用动态路由 + 消息管理
dki = await create_plugin(
    adapter_config_path="config/adapter.yaml",
    dynamic_router=True,
    message_management=True,
    store=my_chat_store,
)
```

Author: AGI Demo Project
Version: 4.0.0
"""

import os
from typing import Any, Dict, Optional, Union

from loguru import logger

from dki.core.dki_plugin import DKIPlugin
from dki.models.base import BaseModelAdapter
from dki.integration.enhanced_plugin import (
    EnhancedDKIPlugin,
    EnhancedPluginConfig,
    DynamicRouterConfig,
    MessageManagementConfig,
)


async def create_plugin(
    # ============ 必需: 数据适配器配置 ============
    adapter_config_path: Optional[str] = None,
    adapter_config: Optional[Union[str, Dict[str, Any]]] = None,
    
    # ============ 可选: 模型适配器 ============
    model_adapter: Optional[BaseModelAdapter] = None,
    engine: Optional[str] = None,
    model_name: Optional[str] = None,
    
    # ============ 可选: DKI 全局配置 ============
    config_path: Optional[str] = None,
    config: Optional[Any] = None,
    language: str = "cn",
    
    # ============ 可选: Redis ============
    enable_redis: Optional[bool] = None,
    redis_url: Optional[str] = None,
    
    # ============ 可选: 增强功能 ============
    dynamic_router: Union[bool, Dict[str, Any], DynamicRouterConfig] = False,
    message_management: Union[bool, Dict[str, Any], MessageManagementConfig] = False,
    
    # ============ 可选: 外部组件 (增强功能需要) ============
    rag_system: Optional[Any] = None,
    store: Optional[Any] = None,
    
    # ============ 可选: 高级配置 ============
    memory_trigger_config: Optional[Dict[str, Any]] = None,
    reference_resolver_config: Optional[Dict[str, Any]] = None,
    
    **kwargs,
) -> Union[EnhancedDKIPlugin, DKIPlugin]:
    """
    创建 DKI Plugin (极简化工厂方法)
    
    这是 DKI 集成的推荐入口。上层应用只需提供数据库映射配置,
    其余组件 (模型、缓存、Redis) 全部自动管理。
    
    Args:
        adapter_config_path: 适配器配置文件路径 (YAML)
        adapter_config: 适配器配置字典 (与 adapter_config_path 二选一)
        model_adapter: 模型适配器 (可选, 不提供则自动创建)
        engine: 模型引擎 (可选, 不提供则从 config.yaml 读取)
        model_name: 模型名称 (可选, 不提供则从 config.yaml 读取)
        config_path: DKI 全局配置路径 (可选)
        config: DKI 全局配置对象 (可选)
        language: 语言 ("cn" | "en")
        enable_redis: 是否启用 Redis (可选, 不提供则从配置读取)
        redis_url: Redis URL (可选)
        dynamic_router: 动态路由 (True=启用默认配置, dict=自定义配置)
        message_management: 消息管理 (True=启用默认配置, dict=自定义配置)
        rag_system: RAG 系统实例 (动态路由需要)
        store: IChatStore 实例 (消息管理需要)
        memory_trigger_config: Memory Trigger 配置
        reference_resolver_config: Reference Resolver 配置
        
    Returns:
        EnhancedDKIPlugin (如果启用了增强功能) 或 DKIPlugin
        
    Examples:
        # 最简集成
        dki = await create_plugin(adapter_config_path="config/adapter.yaml")
        
        # 指定模型
        dki = await create_plugin(
            adapter_config_path="config/adapter.yaml",
            engine="vllm",
            model_name="Qwen/Qwen3-8B",
        )
        
        # 启用全部增强功能
        dki = await create_plugin(
            adapter_config_path="config/adapter.yaml",
            dynamic_router=True,
            message_management=True,
            rag_system=my_rag,
            store=my_store,
        )
    """
    logger.info("Creating DKI Plugin via integration factory...")
    
    # ============ Step 1: 初始化全局配置 ============
    if config_path:
        os.environ.setdefault("DKI_CONFIG_PATH", config_path)
    
    from dki.config.config_loader import ConfigLoader
    config_loader = ConfigLoader(config_path)
    global_config = config if config else config_loader.config
    
    # ============ Step 2: 创建模型适配器 ============
    if model_adapter is None:
        from dki.models.factory import ModelFactory
        model_adapter = ModelFactory.get_or_create(
            engine=engine,
            model_name=model_name,
        )
        logger.info(f"Model adapter auto-created: {model_adapter.model_name}")
    
    # ============ Step 3: 处理 Redis 配置 ============
    redis_config_dict = None
    if redis_url:
        redis_config_dict = {"url": redis_url, "enabled": True}
    
    # ============ Step 4: 创建核心 DKI Plugin ============
    dki_plugin = await DKIPlugin.from_config(
        model_adapter=model_adapter,
        adapter_config=adapter_config,
        adapter_config_path=adapter_config_path,
        config=global_config,
        language=language,
        memory_trigger_config=memory_trigger_config,
        reference_resolver_config=reference_resolver_config,
        enable_redis=enable_redis,
        redis_config=redis_config_dict,
    )
    
    logger.info("Core DKI Plugin created successfully")
    
    # ============ Step 5: 判断是否需要增强包装 ============
    need_enhanced = bool(dynamic_router) or bool(message_management)
    
    if not need_enhanced:
        logger.info("DKI Plugin ready (standard mode)")
        return dki_plugin
    
    # ============ Step 6: 构建增强配置 ============
    dr_config = _resolve_router_config(dynamic_router)
    mm_config = _resolve_message_config(message_management)
    
    enhanced_config = EnhancedPluginConfig(
        dynamic_router=dr_config,
        message_management=mm_config,
    )
    
    # ============ Step 7: 创建增强插件 ============
    enhanced = EnhancedDKIPlugin(
        dki_plugin=dki_plugin,
        rag_system=rag_system,
        store=store,
        config=enhanced_config,
    )
    
    features = []
    if dr_config.enabled:
        features.append("dynamic_router")
    if mm_config.enabled:
        features.append("message_management")
    
    logger.info(f"EnhancedDKIPlugin ready (features={features})")
    return enhanced


async def create_plugin_from_dict(
    config_dict: Dict[str, Any],
    model_adapter: Optional[BaseModelAdapter] = None,
    rag_system: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Union[EnhancedDKIPlugin, DKIPlugin]:
    """
    从配置字典创建 DKI Plugin
    
    适用于配置已经在内存中的场景 (如从 YAML/JSON 加载后)。
    
    Args:
        config_dict: 完整配置字典, 格式:
            {
                "adapter_config_path": "config/adapter.yaml",  # 或 "adapter_config": {...}
                "engine": "vllm",  # 可选
                "model_name": "Qwen/Qwen3-8B",  # 可选
                "config_path": "config/config.yaml",  # 可选
                "language": "cn",  # 可选
                "enable_redis": false,  # 可选
                "redis_url": "redis://localhost:6379",  # 可选
                "dynamic_router": {"enabled": true, ...},  # 可选
                "message_management": {"enabled": true, ...},  # 可选
            }
        model_adapter: 模型适配器 (可选)
        rag_system: RAG 系统 (可选)
        store: IChatStore (可选)
        
    Returns:
        DKI Plugin 实例
    """
    return await create_plugin(
        adapter_config_path=config_dict.get("adapter_config_path"),
        adapter_config=config_dict.get("adapter_config"),
        model_adapter=model_adapter,
        engine=config_dict.get("engine"),
        model_name=config_dict.get("model_name"),
        config_path=config_dict.get("config_path"),
        language=config_dict.get("language", "cn"),
        enable_redis=config_dict.get("enable_redis"),
        redis_url=config_dict.get("redis_url"),
        dynamic_router=config_dict.get("dynamic_router", False),
        message_management=config_dict.get("message_management", False),
        rag_system=rag_system,
        store=store,
        memory_trigger_config=config_dict.get("memory_trigger"),
        reference_resolver_config=config_dict.get("reference_resolver"),
    )


# ============================================================
# 内部辅助
# ============================================================

def _resolve_router_config(
    value: Union[bool, Dict[str, Any], DynamicRouterConfig],
) -> DynamicRouterConfig:
    """解析动态路由配置"""
    if isinstance(value, DynamicRouterConfig):
        return value
    if isinstance(value, dict):
        cfg = DynamicRouterConfig.from_dict(value)
        cfg.enabled = value.get("enabled", True)
        return cfg
    if value is True:
        return DynamicRouterConfig(enabled=True)
    return DynamicRouterConfig(enabled=False)


def _resolve_message_config(
    value: Union[bool, Dict[str, Any], MessageManagementConfig],
) -> MessageManagementConfig:
    """解析消息管理配置"""
    if isinstance(value, MessageManagementConfig):
        return value
    if isinstance(value, dict):
        cfg = MessageManagementConfig.from_dict(value)
        cfg.enabled = value.get("enabled", True)
        return cfg
    if value is True:
        return MessageManagementConfig(enabled=True)
    return MessageManagementConfig(enabled=False)
