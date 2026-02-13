"""
FastAPI Web Application for DKI System
Provides REST API and Web UI for testing

Updated to include:
- OpenAI-compatible API endpoints
- DKI-specific injection API
- User preference management
- Integrated with new adapter and cache systems
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from dki.core.dki_system import DKISystem
from dki.core.rag_system import RAGSystem
from dki.database.connection import get_db, DatabaseManager
from dki.database.repository import SessionRepository, MemoryRepository, ExperimentRepository
from dki.experiment.runner import ExperimentRunner, ExperimentConfig
from dki.experiment.data_generator import ExperimentDataGenerator
from dki.config.config_loader import ConfigLoader
from dki.api.routes import create_api_router
from dki.api.auth_routes import create_auth_router
from dki.api.session_routes import create_session_router
from dki.api.preference_routes import create_preference_router
from dki.api.stats_routes import create_stats_router
from dki.api.visualization_routes import create_visualization_router
from dki.api.dki_routes import create_dki_router, set_dki_plugin
from dki.api.dependencies import init_dependencies, cleanup_dependencies
from dki.api.visualization_routes import record_visualization
from dki.api.stats_routes import record_dki_request
from dki.adapters import AdapterFactory, AdapterConfig, AdapterType
from dki.cache import (
    PreferenceCacheManager,
    CacheConfig,
    DKIRedisClient,
    RedisConfig,
    REDIS_AVAILABLE,
)


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "dki"  # dki, rag, baseline
    force_alpha: Optional[float] = None
    max_new_tokens: int = 256
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    mode: str
    session_id: str
    latency_ms: float
    memories_used: List[Dict[str, Any]]
    alpha: Optional[float] = None
    cache_hit: bool = False
    metadata: Dict[str, Any] = {}


class MemoryRequest(BaseModel):
    session_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    memory_id: str
    session_id: str
    content: str


class ExperimentRequest(BaseModel):
    name: str
    description: str = ""
    modes: List[str] = ["dki", "rag", "baseline"]
    datasets: List[str] = ["persona_chat", "memory_qa"]
    max_samples: int = 50


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    config = ConfigLoader().config
    
    app = FastAPI(
        title="DKI System",
        description="Dynamic KV Injection - Attention-Level Memory Augmentation",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware for Vue3 frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize systems (lazy)
    _systems = {}
    _redis_client = None  # Redis 客户端 (全局)
    _preference_cache = None  # 偏好缓存管理器 (全局)
    
    def get_dki_system() -> DKISystem:
        if 'dki' not in _systems:
            _systems['dki'] = DKISystem()
        return _systems['dki']
    
    def get_rag_system() -> RAGSystem:
        if 'rag' not in _systems:
            _systems['rag'] = RAGSystem()
        return _systems['rag']
    
    # Initialize user data adapter from config
    def get_user_adapter():
        if 'user_adapter' not in _systems:
            adapter_config = getattr(config, 'user_adapter', None)
            if adapter_config:
                _systems['user_adapter'] = AdapterFactory.create_from_dict(
                    adapter_config.__dict__ if hasattr(adapter_config, '__dict__') else {}
                )
            else:
                _systems['user_adapter'] = AdapterFactory.create_memory()
        return _systems['user_adapter']
    
    def get_preference_cache() -> Optional[PreferenceCacheManager]:
        """获取偏好缓存管理器 (支持 Redis)"""
        nonlocal _preference_cache
        return _preference_cache
    
    def get_redis_client() -> Optional[DKIRedisClient]:
        """获取 Redis 客户端"""
        nonlocal _redis_client
        return _redis_client
    
    # Initialize dependencies for API routes
    @app.on_event("startup")
    async def startup_event():
        """Initialize dependencies on startup."""
        nonlocal _redis_client, _preference_cache
        
        dki = get_dki_system()
        adapter = get_user_adapter()
        
        # Connect adapter
        try:
            await adapter.connect()
        except Exception as e:
            logger.warning(f"Failed to connect user adapter: {e}")
        
        # ============ 初始化 Redis 缓存 (与生产系统一致) ============
        # 从配置文件加载 Redis 配置
        import yaml
        config_loader = ConfigLoader()
        try:
            with open(config_loader._config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
        except Exception:
            raw_config = {}
        
        redis_config_data = raw_config.get('redis', {})
        cache_config_data = raw_config.get('preference_cache', {})
        
        # 初始化 Redis 客户端
        if REDIS_AVAILABLE and redis_config_data.get('enabled', False):
            redis_config = RedisConfig.from_dict(redis_config_data)
            _redis_client = DKIRedisClient(redis_config)
            
            try:
                connected = await _redis_client.connect()
                if connected:
                    logger.info(
                        f"✅ Redis connected: {redis_config.host}:{redis_config.port} "
                        f"(db={redis_config.db})"
                    )
                else:
                    logger.warning("⚠️ Redis connection failed, falling back to L1 only")
                    _redis_client = None
            except Exception as e:
                logger.warning(f"⚠️ Redis connection error: {e}")
                _redis_client = None
        else:
            if not REDIS_AVAILABLE:
                logger.info("Redis library not installed, using L1 cache only")
            else:
                logger.info("Redis disabled in configuration, using L1 cache only")
        
        # 初始化偏好缓存管理器 (支持 Redis L2 缓存)
        cache_config = CacheConfig.from_dict(cache_config_data)
        if _redis_client and _redis_client.is_available:
            cache_config.l2_enabled = True
        
        _preference_cache = PreferenceCacheManager(
            redis_client=_redis_client,
            config=cache_config,
        )
        
        logger.info(
            f"Preference cache initialized: "
            f"L1={cache_config.l1_max_size}, "
            f"L2={'Redis' if cache_config.l2_enabled else 'disabled'}"
        )
        
        # Initialize API dependencies
        init_dependencies(
            dki_system=dki,
            user_adapter=adapter,
        )
        
        # 设置 DKI 插件 (用于 /v1/dki/chat 端点)
        set_dki_plugin(dki)
        
        logger.info("DKI System started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        nonlocal _redis_client
        
        # 关闭 Redis 连接
        if _redis_client:
            await _redis_client.close()
            logger.info("Redis connection closed")
        
        await cleanup_dependencies()
        logger.info("DKI System stopped")
    
    # Include OpenAI-compatible API routes
    api_router = create_api_router()
    app.include_router(api_router)
    
    # Include authentication routes
    auth_router = create_auth_router()
    app.include_router(auth_router)
    
    # Include session management routes
    session_router = create_session_router()
    app.include_router(session_router)
    
    # Include preference management routes
    preference_router = create_preference_router()
    app.include_router(preference_router)
    
    # Include stats routes
    stats_router = create_stats_router()
    app.include_router(stats_router)
    
    # Include visualization routes
    visualization_router = create_visualization_router()
    app.include_router(visualization_router)
    
    # Include DKI plugin routes (核心 DKI 聊天 API)
    dki_router = create_dki_router()
    app.include_router(dki_router)
    
    # Note: /api/health and /api/stats are already defined in stats_routes.py
    # Do not duplicate them here
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Chat endpoint supporting DKI, RAG, and baseline modes.
        Also records visualization and statistics data.
        """
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        try:
            if request.mode == "dki":
                dki = get_dki_system()
                response = dki.chat(
                    query=request.query,
                    session_id=session_id,
                    user_id="demo_user",  # Demo 使用默认用户
                    force_alpha=request.force_alpha,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                
                # 记录可视化数据
                try:
                    hybrid_info = response.metadata.get("hybrid_injection", {})
                    preference_tokens = hybrid_info.get("preference_tokens", 0)
                    history_tokens = hybrid_info.get("history_tokens", 0)
                    lb = response.latency_breakdown
                    
                    viz_data = {
                        "request_id": f"chat-dki-{uuid.uuid4().hex[:8]}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "mode": "dki",
                        "query": request.query,
                        "user_id": "demo_user",
                        "session_id": session_id,
                        "injection_enabled": response.gating_decision.should_inject if response.gating_decision else False,
                        "alpha": response.gating_decision.alpha if response.gating_decision else 0.0,
                        "preference_tokens": preference_tokens,
                        "history_tokens": history_tokens,
                        "query_tokens": max(0, response.input_tokens - preference_tokens - history_tokens),
                        "total_tokens": response.input_tokens,
                        "cache_hit": response.cache_hit,
                        "cache_tier": response.cache_tier or "none",
                        "latency_ms": response.latency_ms,
                        "preference_text": hybrid_info.get("preference_text", ""),
                        "history_suffix_text": hybrid_info.get("history_suffix_text", ""),
                        "history_messages": hybrid_info.get("history_messages", []),
                        "final_input": hybrid_info.get("final_input", request.query),
                        "rag_prompt_text": "",
                        "rag_context_text": "",
                        "adapter_latency_ms": (lb.router_ms if lb else 0),
                        "injection_latency_ms": ((lb.kv_compute_ms + lb.projection_ms) if lb else 0),
                        "inference_latency_ms": ((lb.prefill_ms + lb.decode_ms) if lb else 0),
                    }
                    record_visualization(viz_data)
                except Exception as viz_err:
                    logger.warning(f"Failed to record visualization for /api/chat DKI: {viz_err}")
                
                # 记录统计数据
                try:
                    record_dki_request(
                        cache_tier=response.cache_tier or "L3",
                        alpha=response.gating_decision.alpha if response.gating_decision else 0.0,
                        injected=response.gating_decision.should_inject if response.gating_decision else False,
                    )
                except Exception:
                    pass
                
                return ChatResponse(
                    response=response.text,
                    mode="dki",
                    session_id=session_id,
                    latency_ms=response.latency_ms,
                    memories_used=[m.to_dict() for m in response.memories_used],
                    alpha=response.gating_decision.alpha if response.gating_decision else None,
                    cache_hit=response.cache_hit,
                    metadata=response.metadata,
                )
                
            elif request.mode == "rag":
                rag = get_rag_system()
                response = rag.chat(
                    query=request.query,
                    session_id=session_id,
                    user_id="demo_user",
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                
                # 记录 RAG 可视化 (基本信息)
                try:
                    rag_prompt_info = {}
                    if response.prompt_info:
                        rag_prompt_info = response.prompt_info.to_dict()
                    
                    viz_data = {
                        "request_id": f"chat-rag-{uuid.uuid4().hex[:8]}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "mode": "rag",
                        "query": request.query,
                        "user_id": "demo_user",
                        "session_id": session_id,
                        "injection_enabled": False,
                        "alpha": 0.0,
                        "preference_tokens": 0,
                        "history_tokens": len(response.memories_used) * 50,  # 估算
                        "query_tokens": response.input_tokens,
                        "total_tokens": response.input_tokens,
                        "cache_hit": False,
                        "cache_tier": "none",
                        "latency_ms": response.latency_ms,
                        "preference_text": "",
                        "history_suffix_text": rag_prompt_info.get("history_text", ""),
                        "history_messages": rag_prompt_info.get("history_messages", []),
                        "final_input": rag_prompt_info.get("final_prompt", request.query),
                        "rag_prompt_text": rag_prompt_info.get("final_prompt", ""),
                        "rag_context_text": rag_prompt_info.get("retrieved_context", ""),
                        "adapter_latency_ms": 0,
                        "injection_latency_ms": 0,
                        "inference_latency_ms": response.latency_ms,
                    }
                    record_visualization(viz_data)
                except Exception as viz_err:
                    logger.warning(f"Failed to record visualization for /api/chat RAG: {viz_err}")
                
                return ChatResponse(
                    response=response.text,
                    mode="rag",
                    session_id=session_id,
                    latency_ms=response.latency_ms,
                    memories_used=[m.to_dict() for m in response.memories_used],
                    metadata=response.metadata,
                )
                
            else:  # baseline
                dki = get_dki_system()
                output = dki.model.generate(
                    prompt=request.query,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                return ChatResponse(
                    response=output.text,
                    mode="baseline",
                    session_id=session_id,
                    latency_ms=output.latency_ms,
                    memories_used=[],
                )
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/memory", response_model=MemoryResponse)
    async def add_memory(request: MemoryRequest):
        """Add memory to both DKI and RAG systems."""
        try:
            dki = get_dki_system()
            rag = get_rag_system()
            
            memory_id = dki.add_memory(
                session_id=request.session_id,
                content=request.content,
                metadata=request.metadata,
            )
            
            # Also add to RAG for comparison
            rag.add_memory(
                session_id=request.session_id,
                content=request.content,
                memory_id=memory_id,
                metadata=request.metadata,
            )
            
            return MemoryResponse(
                memory_id=memory_id,
                session_id=request.session_id,
                content=request.content,
            )
            
        except Exception as e:
            logger.error(f"Add memory error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/memories/{session_id}")
    async def get_memories(session_id: str):
        """Get all memories for a session."""
        try:
            db_manager = DatabaseManager()
            with db_manager.session_scope() as db:
                memory_repo = MemoryRepository(db)
                memories = memory_repo.get_by_session(session_id)
                return {
                    "session_id": session_id,
                    "memories": [m.to_dict() for m in memories],
                }
        except Exception as e:
            logger.error(f"Get memories error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/search")
    async def search_memories(query: str, session_id: Optional[str] = None, top_k: int = 5):
        """Search memories."""
        try:
            dki = get_dki_system()
            results = dki.search_memories(query, top_k=top_k)
            return {
                "query": query,
                "results": [r.to_dict() for r in results],
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Note: /api/stats is defined in stats_routes.py (camelCase format for Vue frontend)
    # Use /api/stats/detailed for the raw system stats if needed
    @app.get("/api/stats/detailed")
    async def get_detailed_stats():
        """Get detailed system statistics (raw format)."""
        try:
            dki = get_dki_system()
            rag = get_rag_system()
            
            # 获取缓存统计
            cache_stats = {}
            if _preference_cache:
                cache_stats = _preference_cache.get_stats()
            
            return {
                "dki": dki.get_stats(),
                "rag": rag.get_stats(),
                "cache": cache_stats,
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/cache/stats")
    async def get_cache_stats():
        """
        获取缓存统计信息
        
        包括:
        - L1 (Memory) 缓存状态
        - L2 (Redis) 缓存状态 (如果启用)
        - 命中率统计
        """
        try:
            if not _preference_cache:
                return {
                    "status": "not_initialized",
                    "message": "Preference cache not initialized",
                }
            
            stats = _preference_cache.get_stats()
            
            # 添加 Redis 服务器信息
            if _redis_client and _redis_client.is_available:
                redis_info = await _redis_client.info()
                stats["redis_server"] = redis_info
            
            return {
                "status": "ok",
                "cache": stats,
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/redis/status")
    async def get_redis_status():
        """
        获取 Redis 连接状态
        
        用于验证 Redis 是否正常工作
        """
        try:
            if not REDIS_AVAILABLE:
                return {
                    "status": "unavailable",
                    "reason": "Redis library not installed",
                    "install_command": "pip install redis",
                }
            
            if not _redis_client:
                return {
                    "status": "disabled",
                    "reason": "Redis is disabled in configuration or failed to connect",
                }
            
            if not _redis_client.is_available:
                return {
                    "status": "disconnected",
                    "reason": "Redis client exists but not connected",
                }
            
            # 测试连接
            ping_ok = await _redis_client.ping()
            info = await _redis_client.info()
            client_stats = _redis_client.get_stats()
            
            return {
                "status": "connected" if ping_ok else "error",
                "ping": ping_ok,
                "server_info": info,
                "client_stats": client_stats,
            }
        except Exception as e:
            logger.error(f"Redis status error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    @app.post("/api/experiment/generate-data")
    async def generate_experiment_data():
        """Generate experiment data."""
        try:
            generator = ExperimentDataGenerator("./data")
            generator.generate_all()
            generator.generate_alpha_sensitivity_data()
            return {"status": "success", "message": "Experiment data generated"}
        except Exception as e:
            logger.error(f"Generate data error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/experiment/run")
    async def run_experiment(request: ExperimentRequest):
        """Run an experiment."""
        try:
            runner = ExperimentRunner()
            config = ExperimentConfig(
                name=request.name,
                description=request.description,
                modes=request.modes,
                datasets=request.datasets,
                max_samples=request.max_samples,
            )
            results = runner.run_experiment(config)
            return results
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/experiments")
    async def list_experiments():
        """List all experiments."""
        try:
            db_manager = DatabaseManager()
            with db_manager.session_scope() as db:
                exp_repo = ExperimentRepository(db)
                experiments = exp_repo.list_all()
                return {
                    "experiments": [e.to_dict() for e in experiments],
                }
        except Exception as e:
            logger.error(f"List experiments error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Web UI Route
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main web UI."""
        return get_index_html()
    
    return app


def get_index_html() -> str:
    """Return the main HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DKI System - Dynamic KV Injection</title>
    <style>
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-tertiary: #252540;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 20px;
        }
        
        .panel-title {
            font-size: 1.2rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--accent-primary);
            border-radius: 2px;
        }
        
        /* Mode Selector */
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .mode-btn {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .mode-btn:hover {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }
        
        .mode-btn.active {
            border-color: var(--accent-primary);
            background: var(--accent-primary);
            color: white;
        }
        
        .mode-btn.dki.active { background: var(--accent-primary); }
        .mode-btn.rag.active { background: var(--accent-success); }
        .mode-btn.baseline.active { background: var(--accent-warning); }
        
        /* Chat Area */
        .chat-container {
            height: 500px;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
        }
        
        .message.user {
            background: var(--accent-primary);
            margin-left: auto;
        }
        
        .message.assistant {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
        }
        
        .message-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 8px;
            display: flex;
            gap: 15px;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* Input Area */
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
        }
        
        .input-area input:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        
        .input-area button {
            padding: 12px 24px;
            background: var(--accent-primary);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1rem;
            transition: background 0.3s;
        }
        
        .input-area button:hover {
            background: var(--accent-secondary);
        }
        
        /* Memory Panel */
        .memory-input {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .memory-input textarea {
            padding: 12px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            font-family: inherit;
            resize: vertical;
            min-height: 80px;
        }
        
        .memory-input textarea:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        
        .memory-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .memory-item {
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        
        /* Stats Panel */
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-card {
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-primary);
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        /* Alpha Slider */
        .alpha-control {
            margin: 15px 0;
        }
        
        .alpha-control label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-secondary);
        }
        
        .alpha-control input[type="range"] {
            width: 100%;
            -webkit-appearance: none;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
        }
        
        .alpha-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--accent-primary);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .alpha-value {
            text-align: center;
            font-size: 1.2rem;
            color: var(--accent-primary);
            margin-top: 5px;
        }
        
        /* Experiment Panel */
        .experiment-btn {
            width: 100%;
            padding: 12px;
            margin-bottom: 10px;
            background: var(--bg-tertiary);
            border: 2px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .experiment-btn:hover {
            border-color: var(--accent-primary);
            background: var(--accent-primary);
        }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--bg-tertiary);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 DKI System</h1>
            <p>Dynamic KV Injection - Attention-Level Memory Augmentation</p>
        </header>
        
        <div class="main-grid">
            <!-- Memory Panel -->
            <div class="panel">
                <div class="panel-title">Memory Store</div>
                
                <div class="memory-input">
                    <textarea id="memoryInput" placeholder="Enter memory content..."></textarea>
                    <button class="experiment-btn" onclick="addMemory()">Add Memory</button>
                </div>
                
                <div class="panel-title" style="margin-top: 20px;">Memories</div>
                <div class="memory-list" id="memoryList">
                    <div class="memory-item">No memories yet. Add some above!</div>
                </div>
                
                <div class="alpha-control">
                    <label>Force Alpha (DKI mode):</label>
                    <input type="range" id="alphaSlider" min="0" max="100" value="50" oninput="updateAlpha()">
                    <div class="alpha-value" id="alphaValue">α = 0.50</div>
                </div>
            </div>
            
            <!-- Chat Panel -->
            <div class="panel">
                <div class="panel-title">Conversation</div>
                
                <div class="mode-selector">
                    <button class="mode-btn dki active" onclick="setMode('dki')">DKI Mode</button>
                    <button class="mode-btn rag" onclick="setMode('rag')">RAG Mode</button>
                    <button class="mode-btn baseline" onclick="setMode('baseline')">Baseline</button>
                </div>
                
                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div>Welcome! I'm the DKI system. Add some memories and start chatting!</div>
                        <div class="message-meta">
                            <span class="meta-item">Mode: DKI</span>
                        </div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px; color: var(--text-secondary);">Generating response...</p>
                </div>
                
                <div class="input-area">
                    <input type="text" id="queryInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <!-- Stats & Experiments Panel -->
            <div class="panel">
                <div class="panel-title">Statistics</div>
                
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="statLatency">0</div>
                        <div class="stat-label">Avg Latency (ms)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statMemories">0</div>
                        <div class="stat-label">Memories Used</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statCacheHit">0%</div>
                        <div class="stat-label">Cache Hit Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statAlpha">-</div>
                        <div class="stat-label">Avg Alpha</div>
                    </div>
                </div>
                
                <div class="panel-title" style="margin-top: 20px;">Injection Info</div>
                <button class="experiment-btn" style="background: var(--accent-primary); color: white; border-color: var(--accent-primary);" onclick="viewInjectionInfo()">
                    🔍 查看注入信息 (View Injection)
                </button>
                
                <div class="panel-title" style="margin-top: 20px;">Experiments</div>
                
                <button class="experiment-btn" onclick="generateData()">Generate Test Data</button>
                <button class="experiment-btn" onclick="runExperiment()">Run Comparison</button>
                <button class="experiment-btn" onclick="runAlphaSensitivity()">α Sensitivity Test</button>
                
                <div class="panel-title" style="margin-top: 20px;">Session</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">
                    Session ID: <span id="sessionId">-</span>
                </div>
                <button class="experiment-btn" style="margin-top: 10px;" onclick="newSession()">New Session</button>
            </div>
        </div>
    </div>
    
    <!-- Injection Info Modal -->
    <div id="injectionModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; overflow:auto;">
        <div style="max-width:900px; margin:30px auto; background:var(--bg-secondary); border-radius:12px; border:1px solid var(--border-color); max-height:90vh; overflow:auto;">
            <div style="padding:20px; border-bottom:1px solid var(--border-color); display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; background:var(--bg-secondary); z-index:1;">
                <h2 style="margin:0; background:linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">🔍 注入信息 (Injection Info)</h2>
                <div>
                    <button onclick="copyInjectionText()" style="padding:8px 16px; background:var(--accent-success); color:white; border:none; border-radius:6px; cursor:pointer; font-family:inherit; margin-right:8px;">📋 复制</button>
                    <button onclick="closeInjectionModal()" style="padding:8px 16px; background:var(--bg-tertiary); color:var(--text-primary); border:2px solid var(--border-color); border-radius:6px; cursor:pointer; font-family:inherit;">✕ 关闭</button>
                </div>
            </div>
            <div id="injectionContent" style="padding:20px;">
                <p style="color:var(--text-secondary);">暂无注入信息，请先发送一条消息。</p>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentMode = 'dki';
        let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        let useForceAlpha = false;
        let forceAlphaValue = 0.5;
        let stats = { latencies: [], memoriesUsed: 0, cacheHits: 0, totalQueries: 0, alphas: [] };
        
        document.getElementById('sessionId').textContent = sessionId;
        
        // Mode selection
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.mode-btn.' + mode).classList.add('active');
        }
        
        // Alpha control
        function updateAlpha() {
            const slider = document.getElementById('alphaSlider');
            forceAlphaValue = slider.value / 100;
            document.getElementById('alphaValue').textContent = 'α = ' + forceAlphaValue.toFixed(2);
        }
        
        // Add memory
        async function addMemory() {
            const content = document.getElementById('memoryInput').value.trim();
            if (!content) return;
            
            try {
                const response = await fetch('/api/memory', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, content: content })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    addMemoryToList(data.content);
                    document.getElementById('memoryInput').value = '';
                }
            } catch (error) {
                console.error('Add memory error:', error);
            }
        }
        
        function addMemoryToList(content) {
            const list = document.getElementById('memoryList');
            if (list.children[0].textContent.includes('No memories')) {
                list.innerHTML = '';
            }
            const item = document.createElement('div');
            item.className = 'memory-item';
            item.textContent = content;
            list.appendChild(item);
        }
        
        // Send message
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;
            
            // Add user message
            addMessage('user', query);
            input.value = '';
            
            // Show loading
            document.getElementById('loading').classList.add('active');
            
            try {
                const requestBody = {
                    query: query,
                    session_id: sessionId,
                    mode: currentMode,
                    max_new_tokens: 256,
                    temperature: 0.7
                };
                
                // Add force alpha for DKI mode if checkbox would be checked
                if (currentMode === 'dki') {
                    requestBody.force_alpha = forceAlphaValue;
                }
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                
                // Add assistant message
                addMessage('assistant', data.response, data);
                
                // Update stats
                updateStats(data);
                
            } catch (error) {
                console.error('Chat error:', error);
                addMessage('assistant', 'Error: ' + error.message, {});
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }
        
        function addMessage(role, content, meta = {}) {
            const container = document.getElementById('chatContainer');
            const msg = document.createElement('div');
            msg.className = 'message ' + role;
            
            let metaHtml = '';
            if (role === 'assistant' && meta.mode) {
                metaHtml = '<div class="message-meta">';
                metaHtml += '<span class="meta-item">Mode: ' + meta.mode + '</span>';
                if (meta.latency_ms) {
                    metaHtml += '<span class="meta-item">Latency: ' + meta.latency_ms.toFixed(0) + 'ms</span>';
                }
                if (meta.alpha !== undefined && meta.alpha !== null) {
                    metaHtml += '<span class="meta-item">α: ' + meta.alpha.toFixed(2) + '</span>';
                }
                if (meta.cache_hit) {
                    metaHtml += '<span class="meta-item">Cache Hit ✓</span>';
                }
                if (meta.memories_used && meta.memories_used.length > 0) {
                    metaHtml += '<span class="meta-item">Memories: ' + meta.memories_used.length + '</span>';
                }
                metaHtml += '</div>';
            }
            
            msg.innerHTML = '<div>' + content + '</div>' + metaHtml;
            container.appendChild(msg);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateStats(data) {
            stats.totalQueries++;
            if (data.latency_ms) {
                stats.latencies.push(data.latency_ms);
            }
            if (data.memories_used) {
                stats.memoriesUsed += data.memories_used.length;
            }
            if (data.cache_hit) {
                stats.cacheHits++;
            }
            if (data.alpha !== undefined && data.alpha !== null) {
                stats.alphas.push(data.alpha);
            }
            
            // Update display
            const avgLatency = stats.latencies.length > 0 
                ? stats.latencies.reduce((a, b) => a + b) / stats.latencies.length 
                : 0;
            document.getElementById('statLatency').textContent = avgLatency.toFixed(0);
            document.getElementById('statMemories').textContent = stats.memoriesUsed;
            document.getElementById('statCacheHit').textContent = 
                stats.totalQueries > 0 ? ((stats.cacheHits / stats.totalQueries) * 100).toFixed(0) + '%' : '0%';
            document.getElementById('statAlpha').textContent = 
                stats.alphas.length > 0 
                    ? (stats.alphas.reduce((a, b) => a + b) / stats.alphas.length).toFixed(2) 
                    : '-';
        }
        
        // Session management
        function newSession() {
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            document.getElementById('sessionId').textContent = sessionId;
            document.getElementById('chatContainer').innerHTML = '';
            document.getElementById('memoryList').innerHTML = '<div class="memory-item">No memories yet. Add some above!</div>';
            stats = { latencies: [], memoriesUsed: 0, cacheHits: 0, totalQueries: 0, alphas: [] };
            updateStats({});
        }
        
        // Experiments
        async function generateData() {
            try {
                const response = await fetch('/api/experiment/generate-data', { method: 'POST' });
                const data = await response.json();
                alert('Data generated successfully!');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function runExperiment() {
            alert('Starting experiment... This may take a while. Check console for progress.');
            try {
                const response = await fetch('/api/experiment/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: 'Web UI Experiment',
                        modes: ['dki', 'rag', 'baseline'],
                        datasets: ['persona_chat'],
                        max_samples: 10
                    })
                });
                const data = await response.json();
                console.log('Experiment results:', data);
                alert('Experiment completed! Check console for results.');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function runAlphaSensitivity() {
            alert('Running α sensitivity analysis...');
            // This would typically call the alpha sensitivity endpoint
            alert('Feature coming soon!');
        }
        
        // ============ Injection Info Viewer ============
        let lastInjectionData = null;
        
        async function viewInjectionInfo() {
            const modal = document.getElementById('injectionModal');
            const content = document.getElementById('injectionContent');
            
            modal.style.display = 'block';
            content.innerHTML = '<p style="color:var(--text-secondary);">加载中...</p>';
            
            try {
                // Fetch latest visualization data
                const response = await fetch('/v1/dki/visualization/latest');
                
                if (response.ok) {
                    const data = await response.json();
                    lastInjectionData = data;
                    content.innerHTML = renderInjectionInfo(data);
                } else if (response.status === 404) {
                    content.innerHTML = '<p style="color:var(--text-secondary);">暂无注入信息。请先发送一条消息，系统会自动记录注入过程。</p>';
                } else {
                    content.innerHTML = '<p style="color:#f56c6c;">获取数据失败: ' + response.statusText + '</p>';
                }
            } catch (error) {
                content.innerHTML = '<p style="color:#f56c6c;">请求失败: ' + error.message + '</p>';
            }
        }
        
        function renderInjectionInfo(data) {
            let html = '';
            
            // 基本信息
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px;">';
            html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">📌 基本信息</h3>';
            html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.85rem;">';
            html += '<div>请求 ID: <span style="color:var(--accent-primary)">' + (data.request_id || '-') + '</span></div>';
            html += '<div>时间: <span style="color:var(--text-secondary)">' + (data.timestamp || '-') + '</span></div>';
            html += '<div>用户 ID: <span style="color:var(--text-secondary)">' + (data.user_id || '-') + '</span></div>';
            html += '<div>会话 ID: <span style="color:var(--text-secondary)">' + (data.session_id || '-') + '</span></div>';
            html += '<div>总延迟: <span style="color:var(--accent-warning)">' + (data.total_latency_ms?.toFixed(1) || 0) + ' ms</span></div>';
            html += '<div>注入开销: <span style="color:var(--accent-warning)">' + (data.injection_overhead_ms?.toFixed(1) || 0) + ' ms</span></div>';
            html += '</div></div>';
            
            // Token 分布
            const dist = data.token_distribution || {};
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px;">';
            html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">📊 Token 分布</h3>';
            html += '<div style="display:flex; gap:20px; font-size:0.85rem;">';
            html += '<div style="text-align:center; flex:1;"><div style="font-size:1.5rem; font-weight:bold; color:#3b82f6;">' + (dist.query || 0) + '</div><div style="color:var(--text-secondary);">查询</div></div>';
            html += '<div style="text-align:center; flex:1;"><div style="font-size:1.5rem; font-weight:bold; color:#10b981;">' + (dist.preference || 0) + '</div><div style="color:var(--text-secondary);">偏好 (K/V)</div></div>';
            html += '<div style="text-align:center; flex:1;"><div style="font-size:1.5rem; font-weight:bold; color:#f59e0b;">' + (dist.history || 0) + '</div><div style="color:var(--text-secondary);">历史 (后缀)</div></div>';
            html += '<div style="text-align:center; flex:1;"><div style="font-size:1.5rem; font-weight:bold; color:var(--accent-primary);">' + (dist.total || 0) + '</div><div style="color:var(--text-secondary);">总计</div></div>';
            html += '</div></div>';
            
            // 原始查询
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px;">';
            html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">💬 原始查询</h3>';
            html += '<pre style="background:var(--bg-primary); padding:10px; border-radius:6px; color:var(--text-primary); white-space:pre-wrap; word-break:break-all; max-height:100px; overflow:auto;">' + escapeHtml(data.original_query || '') + '</pre>';
            html += '</div>';
            
            // DKI 偏好注入
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid #10b981;">';
            html += '<h3 style="margin:0 0 10px 0; color:#10b981;">🧠 DKI 偏好注入 (K/V, 负位置)</h3>';
            html += '<pre style="background:var(--bg-primary); padding:10px; border-radius:6px; color:var(--text-primary); white-space:pre-wrap; word-break:break-all; max-height:200px; overflow:auto;">' + escapeHtml(data.preference_text || '(无偏好注入)') + '</pre>';
            html += '</div>';
            
            // DKI 历史后缀
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid #f59e0b;">';
            html += '<h3 style="margin:0 0 10px 0; color:#f59e0b;">📜 DKI 历史后缀 (Suffix, 正位置)</h3>';
            html += '<pre style="background:var(--bg-primary); padding:10px; border-radius:6px; color:var(--text-primary); white-space:pre-wrap; word-break:break-all; max-height:300px; overflow:auto;">' + escapeHtml(data.history_suffix_text || '(无历史后缀)') + '</pre>';
            html += '</div>';
            
            // 历史消息
            if (data.history_messages && data.history_messages.length > 0) {
                html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px;">';
                html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">💬 历史消息 (' + data.history_messages.length + ' 条)</h3>';
                for (const msg of data.history_messages) {
                    const isUser = msg.role === 'user';
                    html += '<div style="padding:8px 12px; margin-bottom:6px; border-radius:8px; background:' + (isUser ? 'rgba(99,102,241,0.2)' : 'var(--bg-primary)') + '; font-size:0.85rem;">';
                    html += '<span style="color:' + (isUser ? 'var(--accent-primary)' : 'var(--accent-success)') + '; font-weight:bold;">' + (isUser ? '用户' : '助手') + ':</span> ';
                    html += escapeHtml(msg.content || '');
                    html += '</div>';
                }
                html += '</div>';
            }
            
            // 最终输入预览
            html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid var(--accent-primary);">';
            html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">📄 最终输入预览</h3>';
            html += '<pre style="background:var(--bg-primary); padding:10px; border-radius:6px; color:var(--text-primary); white-space:pre-wrap; word-break:break-all; max-height:400px; overflow:auto;">' + escapeHtml(data.final_input_preview || '') + '</pre>';
            html += '</div>';
            
            // 流程步骤
            if (data.flow_steps && data.flow_steps.length > 0) {
                html += '<div style="background:var(--bg-tertiary); padding:15px; border-radius:8px; margin-bottom:15px;">';
                html += '<h3 style="margin:0 0 10px 0; color:var(--accent-primary);">⚡ 注入流程步骤</h3>';
                for (const step of data.flow_steps) {
                    const statusColor = step.status === 'completed' ? '#10b981' : step.status === 'skipped' ? '#94a3b8' : '#f59e0b';
                    html += '<div style="display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px solid var(--border-color);">';
                    html += '<span style="min-width:20px; color:' + statusColor + ';">' + (step.status === 'completed' ? '✓' : step.status === 'skipped' ? '○' : '●') + '</span>';
                    html += '<span style="flex:1;">' + escapeHtml(step.step_name) + '</span>';
                    html += '<span style="color:var(--text-secondary); font-size:0.8rem;">' + step.duration_ms.toFixed(1) + 'ms</span>';
                    html += '</div>';
                }
                html += '</div>';
            }
            
            return html;
        }
        
        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
        
        function closeInjectionModal() {
            document.getElementById('injectionModal').style.display = 'none';
        }
        
        function copyInjectionText() {
            if (!lastInjectionData) {
                alert('无注入数据可复制');
                return;
            }
            
            let text = '=== DKI 注入信息 ===\\n';
            text += '请求 ID: ' + (lastInjectionData.request_id || '-') + '\\n';
            text += '时间: ' + (lastInjectionData.timestamp || '-') + '\\n';
            text += '\\n== 偏好注入 (K/V) ==\\n';
            text += (lastInjectionData.preference_text || '(无)') + '\\n';
            text += '\\n== 历史后缀 (Suffix) ==\\n';
            text += (lastInjectionData.history_suffix_text || '(无)') + '\\n';
            text += '\\n== 最终输入 ==\\n';
            text += (lastInjectionData.final_input_preview || '(无)') + '\\n';
            
            navigator.clipboard.writeText(text).then(() => {
                alert('已复制到剪贴板');
            }).catch(() => {
                // Fallback
                const textarea = document.createElement('textarea');
                textarea.value = text;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('已复制到剪贴板');
            });
        }
        
        // Close modal on click outside
        document.getElementById('injectionModal').addEventListener('click', function(e) {
            if (e.target === this) closeInjectionModal();
        });
        
        // Close modal on Escape
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeInjectionModal();
        });
    </script>
</body>
</html>'''


# Run server
def run_server():
    """Run the web server."""
    import uvicorn
    
    config = ConfigLoader().config
    app = create_app()
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


if __name__ == "__main__":
    run_server()
