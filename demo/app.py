"""
Demo App — DKI Plugin 的独立上层应用

启动方式:
    uvicorn demo.app:app --port 8080

    或使用 create_demo_app() 工厂函数:
    app = create_demo_app(config_path="config/demo.yaml")

核心设计:
1. 独立的 FastAPI 应用 (不依赖 dki/web/app.py)
2. 独立的持久化层 (demo/store/)
3. 通过 dki.integration 极简化集成 DKI Plugin
4. 支持动态路由 (RAG/DKI 自动切换) 和消息管理

v4.0 重构:
- 使用 dki.integration.create_plugin() 替代手动创建
- 支持 EnhancedDKIPlugin (dynamic_router + message_management)
- 向后兼容: 所有现有 API 不变

Author: AGI Demo Project
Version: 4.0.0
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from demo.config import DemoConfig, load_demo_config
from demo.store.connection import DemoDBConfig, DemoDBManager
from demo.store.factory import create_chat_store, create_async_chat_store
from demo.dki_bridge import build_adapter_config

from demo.api.auth import create_auth_router
from demo.api.chat import create_chat_router
from demo.api.sessions import create_session_router
from demo.api.preferences import create_preference_router
from demo.api.messages import create_message_router
from dki.api.stats_routes import create_stats_router


def create_demo_app(
    config_path: Optional[str] = None,
    config: Optional[DemoConfig] = None,
) -> FastAPI:
    """
    创建 Demo FastAPI 应用
    
    参数:
        config_path: 配置文件路径 (YAML)
        config: 配置对象 (优先于 config_path)
    
    返回:
        配置完成的 FastAPI 应用
    """
    # 加载配置
    demo_config = config or load_demo_config(config_path)
    
    app = FastAPI(
        title="DKI Demo",
        description=(
            "DKI Plugin 的独立上层应用。\n\n"
            "Demo App 管理自己的数据 (消息、偏好、会话)，\n"
            "通过 dki_plugin.chat() 与 DKI 交互。\n"
            "DKI Plugin 通过 ConfigDrivenAdapter 只读访问 Demo 数据库。"
        ),
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=demo_config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ============ Startup ============
    
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info("DKI Demo App starting...")
        logger.info("=" * 60)
        
        # Step 1: 创建 Demo 持久化层
        # PostgreSQL/pgvector 使用异步 store, SQLite 使用同步 store
        db_config = DemoDBConfig.from_dict(demo_config.db)
        if db_config.backend in ("postgresql", "pgvector"):
            try:
                store = await create_async_chat_store(db_config)
                logger.info(f"✅ Demo store initialized (async, backend={db_config.backend})")
            except ImportError as e:
                logger.warning(f"Async store unavailable ({e}), falling back to sync")
                store = create_chat_store(db_config)
                logger.info(f"✅ Demo store initialized (sync fallback, backend={db_config.backend})")
        else:
            store = create_chat_store(db_config)
            logger.info(f"✅ Demo store initialized (backend={db_config.backend})")
        app.state.store = store
        
        # Step 2: 创建 DKI Plugin (v4.0: 使用 integration layer 极简化创建)
        adapter_config = build_adapter_config(db_config)
        
        try:
            from dki.integration import create_plugin
            
            # 从 demo_config 读取增强功能配置
            dki_conf = demo_config.dki
            dynamic_router_cfg = getattr(dki_conf, 'dynamic_router', None)
            message_mgmt_cfg = getattr(dki_conf, 'message_management', None)
            
            dki_plugin = await create_plugin(
                # 数据适配器配置 (指向 Demo 数据库)
                adapter_config=adapter_config,
                # DKI 全局配置路径
                config_path=os.getenv("DKI_CONFIG_PATH", dki_conf.config_path),
                # 语言
                language=dki_conf.language,
                # Redis
                enable_redis=dki_conf.enable_redis,
                redis_url=dki_conf.redis_url if dki_conf.enable_redis else None,
                # 增强功能 (可选, 从配置读取)
                dynamic_router=dynamic_router_cfg or False,
                message_management=message_mgmt_cfg or False,
                # 消息管理需要 store
                store=store if message_mgmt_cfg else None,
            )
            app.state.dki_plugin = dki_plugin
            logger.info("✅ DKI Plugin initialized (via integration layer)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DKI Plugin: {e}")
            logger.warning("Demo App will start without DKI Plugin (chat will be unavailable)")
            app.state.dki_plugin = None
        
        # Step 4: 透传 DKI 可视化路由 (可选)
        _mount_visualization_routes(app)
        
        logger.info("=" * 60)
        logger.info(f"DKI Demo App ready at http://{demo_config.server.host}:{demo_config.server.port}")
        logger.info("=" * 60)
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("DKI Demo App shutting down...")
        
        # 关闭 DKI Plugin (v4.0: 支持 EnhancedDKIPlugin)
        dki_plugin = getattr(app.state, "dki_plugin", None)
        if dki_plugin and hasattr(dki_plugin, 'close'):
            try:
                await dki_plugin.close()
                logger.info("DKI Plugin closed")
            except Exception as e:
                logger.warning(f"Error closing DKI Plugin: {e}")
        
        # 关闭持久化层 (支持同步和异步)
        store = getattr(app.state, "store", None)
        if store:
            if hasattr(store, 'a_get_user'):
                # Async store
                await store.disconnect()
            else:
                store.disconnect()
        
        logger.info("DKI Demo App stopped")
    
    # ============ Routes ============
    
    # 认证
    app.include_router(create_auth_router())
    
    # 对话 (★ 核心: 调用 dki_plugin.chat())
    app.include_router(create_chat_router())
    
    # 会话管理
    app.include_router(create_session_router())
    
    # 偏好管理
    app.include_router(create_preference_router())
    
    # 消息搜索
    app.include_router(create_message_router())
    
    # ============ Health Check (P1-5: Enhanced) ============
    
    @app.get("/api/health")
    async def health():
        """基础健康检查 (用于负载均衡器探活)"""
        store = getattr(app.state, "store", None)
        dki_plugin = getattr(app.state, "dki_plugin", None)
        store_info = "disconnected"
        if store:
            if hasattr(store, 'a_health_check'):
                hc = await store.a_health_check()
                store_info = hc.get("status", "connected")
            else:
                try:
                    hc = store.health_check()
                    store_info = hc.get("status", "connected")
                except Exception:
                    store_info = "error"
        return {
            "status": "ok",
            "version": "4.0.0",
            "store": store_info,
            "dki_plugin": "ready" if dki_plugin else "unavailable",
        }
    
    @app.get("/api/health/detail")
    async def health_detail():
        """
        P1-5: 详细健康检查 (含组件状态、统计信息)
        
        用于运维监控面板, 返回各组件的详细状态。
        """
        import time as _time
        store = getattr(app.state, "store", None)
        dki_plugin = getattr(app.state, "dki_plugin", None)
        
        result = {
            "status": "ok",
            "version": "4.0.0",
            "timestamp": _time.time(),
            "components": {},
        }
        
        # Store health
        if store:
            try:
                if hasattr(store, 'a_health_check'):
                    store_health = await store.a_health_check()
                else:
                    store_health = store.health_check()
                result["components"]["store"] = store_health
            except Exception as e:
                result["components"]["store"] = {"status": "error", "error": str(e)}
        else:
            result["components"]["store"] = {"status": "not_initialized"}
        
        # Store statistics
        if store:
            try:
                if hasattr(store, 'a_get_statistics'):
                    stats = await store.a_get_statistics()
                elif hasattr(store, 'get_statistics'):
                    stats = store.get_statistics()
                else:
                    stats = {}
                result["components"]["store_stats"] = stats
            except Exception as e:
                result["components"]["store_stats"] = {"error": str(e)}
        
        # DKI Plugin health
        if dki_plugin:
            try:
                dki_stats = dki_plugin.get_stats()
                result["components"]["dki_plugin"] = {
                    "status": "ready",
                    "stats": dki_stats,
                }
            except Exception as e:
                result["components"]["dki_plugin"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            result["components"]["dki_plugin"] = {"status": "not_initialized"}
        
        # Overall status
        component_statuses = []
        for comp in result["components"].values():
            s = comp.get("status", "unknown") if isinstance(comp, dict) else "unknown"
            component_statuses.append(s)
        
        if any(s in ("error", "disconnected") for s in component_statuses):
            result["status"] = "degraded"
        elif any(s == "not_initialized" for s in component_statuses):
            result["status"] = "partial"
        
        return result
    
    # 统计数据 (/api/stats, /api/stats/dki, /api/stats/cache)
    # 注意: stats_routes 中的 /api/health 会被上方已注册的 /api/health 覆盖 (FastAPI 先注册优先)
    app.include_router(create_stats_router())
    
    # ============ Static Files (Vue3 前端) ============
    
    _mount_static_files(app)
    
    return app


## _create_dki_plugin 已被 dki.integration.create_plugin() 替代 (v4.0)
## 保留此注释用于版本追溯


def _mount_visualization_routes(app: FastAPI) -> None:
    """透传 DKI 可视化路由 (复用 DKI 内部的可视化模块)"""
    try:
        from dki.api.visualization_routes import create_visualization_router
        viz_router = create_visualization_router()
        app.include_router(viz_router)
        logger.info("✅ DKI Visualization routes mounted")
    except ImportError:
        logger.info("DKI Visualization routes not available (skipped)")
    except Exception as e:
        logger.warning(f"Failed to mount visualization routes: {e}")


def _mount_static_files(app: FastAPI) -> None:
    """挂载 Vue3 前端静态文件 (如果存在)"""
    # 尝试多个可能的路径
    ui_dist_paths = [
        Path("ui/dist"),
        Path("../ui/dist"),
        Path("DKI/ui/dist"),
    ]
    
    for ui_path in ui_dist_paths:
        if ui_path.exists() and (ui_path / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="static")
            logger.info(f"✅ Vue3 frontend mounted from: {ui_path}")
            return
    
    logger.info("Vue3 frontend dist not found (API-only mode)")


# ============ 默认应用实例 ============
# 方式 1: uvicorn demo.app:app (直接引用模块级变量)
# 方式 2: uvicorn demo.app:get_app --factory (工厂模式, 推荐)
# 方式 3: python main.py demo (通过 main.py 创建, 支持 CLI 参数)
#
# 注意: 通过 main.py demo 启动时, main.py 直接调用 create_demo_app(config=...)
# 并传给 uvicorn.run(app, ...), 不会使用此处的 app 变量。
# 因此这里使用延迟创建, 避免 import 时触发不必要的初始化。

def get_app():
    """工厂函数: 用于 uvicorn demo.app:get_app --factory"""
    return create_demo_app()


# ============ CLI 启动 ============

def main():
    """CLI 启动入口"""
    import uvicorn
    
    config = load_demo_config()
    
    uvicorn.run(
        "demo.app:get_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=config.server.workers,
    )


if __name__ == "__main__":
    main()
