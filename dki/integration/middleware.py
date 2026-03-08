"""
DKI FastAPI Middleware — 自动集成层

提供两种 FastAPI 集成方式:

方式 1: Middleware (自动生命周期管理)
    from dki.integration import DKIMiddleware
    app.add_middleware(
        DKIMiddleware,
        adapter_config_path="config/adapter.yaml",
    )

方式 2: Dependency Injection (手动获取)
    from dki.integration import get_dki_plugin
    
    @app.post("/chat")
    async def chat(dki = Depends(get_dki_plugin)):
        response = await dki.chat("推荐餐厅", user_id="u1", session_id="s1")

核心设计:
- Middleware 在 startup 时自动创建 DKI Plugin, shutdown 时自动关闭
- 通过 app.state.dki_plugin 暴露给路由
- 支持所有 create_plugin() 的参数
- 向后兼容: 如果 app.state.dki_plugin 已存在, 不覆盖

Author: AGI Demo Project
Version: 4.0.0
"""

from typing import Any, Dict, Optional, Union

from loguru import logger

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    from fastapi import Depends, Request as FastAPIRequest
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def get_dki_plugin(request: "FastAPIRequest"):
    """
    FastAPI Dependency — 从 app.state 获取 DKI Plugin
    
    用法:
    ```python
    from dki.integration import get_dki_plugin
    
    @app.post("/chat")
    async def chat(dki = Depends(get_dki_plugin)):
        response = await dki.chat("推荐餐厅", user_id="u1", session_id="s1")
    ```
    
    Returns:
        EnhancedDKIPlugin 或 DKIPlugin 实例
        
    Raises:
        RuntimeError: 如果 DKI Plugin 未初始化
    """
    plugin = getattr(request.app.state, "dki_plugin", None)
    if plugin is None:
        raise RuntimeError(
            "DKI Plugin not initialized. "
            "Please add DKIMiddleware to your app or manually set app.state.dki_plugin."
        )
    return plugin


if FASTAPI_AVAILABLE:
    class DKIMiddleware(BaseHTTPMiddleware):
        """
        DKI FastAPI Middleware — 自动管理 DKI Plugin 生命周期
        
        功能:
        1. 在 app startup 时自动创建 DKI Plugin
        2. 在 app shutdown 时自动关闭 DKI Plugin
        3. 通过 app.state.dki_plugin 暴露给路由
        4. 请求级别的错误处理和日志
        
        用法:
        ```python
        from dki.integration import DKIMiddleware
        
        app = FastAPI()
        app.add_middleware(
            DKIMiddleware,
            adapter_config_path="config/adapter.yaml",
            dynamic_router=True,
            message_management=True,
        )
        ```
        
        注意:
        - Middleware 会在第一个请求到达时检查 DKI Plugin 是否已初始化
        - 如果 app.state.dki_plugin 已存在 (如手动创建), 不会覆盖
        - 建议配合 app.on_event("startup") 使用, 确保 Plugin 在请求前就绑定
        """
        
        def __init__(
            self,
            app,
            # ============ create_plugin 参数 ============
            adapter_config_path: Optional[str] = None,
            adapter_config: Optional[Union[str, Dict[str, Any]]] = None,
            model_adapter: Optional[Any] = None,
            engine: Optional[str] = None,
            model_name: Optional[str] = None,
            config_path: Optional[str] = None,
            language: str = "cn",
            enable_redis: Optional[bool] = None,
            redis_url: Optional[str] = None,
            dynamic_router: Union[bool, Dict[str, Any]] = False,
            message_management: Union[bool, Dict[str, Any]] = False,
            rag_system: Optional[Any] = None,
            store: Optional[Any] = None,
            # ============ Middleware 特有参数 ============
            auto_init: bool = True,
            state_key: str = "dki_plugin",
        ):
            """
            Args:
                app: FastAPI/Starlette app
                adapter_config_path: 适配器配置路径
                adapter_config: 适配器配置字典
                model_adapter: 模型适配器
                engine: 模型引擎
                model_name: 模型名称
                config_path: DKI 全局配置路径
                language: 语言
                enable_redis: 是否启用 Redis
                redis_url: Redis URL
                dynamic_router: 动态路由配置
                message_management: 消息管理配置
                rag_system: RAG 系统
                store: IChatStore
                auto_init: 是否在第一个请求时自动初始化
                state_key: app.state 中的键名
            """
            super().__init__(app)
            
            self._plugin_kwargs = {
                "adapter_config_path": adapter_config_path,
                "adapter_config": adapter_config,
                "model_adapter": model_adapter,
                "engine": engine,
                "model_name": model_name,
                "config_path": config_path,
                "language": language,
                "enable_redis": enable_redis,
                "redis_url": redis_url,
                "dynamic_router": dynamic_router,
                "message_management": message_management,
                "rag_system": rag_system,
                "store": store,
            }
            self._auto_init = auto_init
            self._state_key = state_key
            self._initialized = False
            
            # 注册 startup/shutdown 事件
            self._register_lifecycle_events(app)
        
        def _register_lifecycle_events(self, app):
            """注册生命周期事件"""
            original_startup = getattr(app, '_dki_original_startup', None)
            original_shutdown = getattr(app, '_dki_original_shutdown', None)
            
            # 避免重复注册
            if hasattr(app, '_dki_middleware_registered'):
                return
            app._dki_middleware_registered = True
            
            @app.on_event("startup")
            async def _dki_middleware_startup():
                """DKI Middleware: 自动初始化 Plugin"""
                if self._auto_init:
                    await self._ensure_initialized(app)
            
            @app.on_event("shutdown")
            async def _dki_middleware_shutdown():
                """DKI Middleware: 自动关闭 Plugin"""
                plugin = getattr(app.state, self._state_key, None)
                if plugin and hasattr(plugin, 'close'):
                    try:
                        await plugin.close()
                        logger.info("DKI Plugin closed by middleware")
                    except Exception as e:
                        logger.warning(f"Error closing DKI Plugin: {e}")
        
        async def _ensure_initialized(self, app):
            """确保 DKI Plugin 已初始化"""
            if self._initialized:
                return
            
            # 检查是否已手动创建
            existing = getattr(app.state, self._state_key, None)
            if existing is not None:
                logger.info(
                    f"DKI Plugin already exists at app.state.{self._state_key}, "
                    f"middleware will not override"
                )
                self._initialized = True
                return
            
            # 自动创建
            try:
                from dki.integration.factory import create_plugin
                
                plugin = await create_plugin(**self._plugin_kwargs)
                setattr(app.state, self._state_key, plugin)
                self._initialized = True
                logger.info(f"DKI Plugin auto-initialized by middleware → app.state.{self._state_key}")
            except Exception as e:
                logger.error(f"DKI Middleware failed to initialize plugin: {e}")
                logger.warning("DKI Plugin will be unavailable. Requests to DKI endpoints will fail.")
        
        async def dispatch(self, request: Request, call_next):
            """
            请求处理 (透传, 不做额外处理)
            
            DKI Middleware 的主要价值在于生命周期管理,
            不在请求级别做额外处理 (避免性能开销)。
            """
            # 延迟初始化 (如果 startup 事件未触发)
            if not self._initialized and self._auto_init:
                await self._ensure_initialized(request.app)
            
            response = await call_next(request)
            return response

else:
    # FastAPI 不可用时的占位
    class DKIMiddleware:
        """DKI Middleware (FastAPI not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DKIMiddleware requires FastAPI/Starlette. "
                "Install with: pip install fastapi"
            )
