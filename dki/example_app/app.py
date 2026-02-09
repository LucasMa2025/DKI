"""
Example Application FastAPI App

示例应用 FastAPI 应用

职责: 提供示例 Chat UI 的后端 API
- 用户管理 API
- 会话管理 API
- 聊天 API (调用 DKI 插件)
- 偏好管理 API

注意:
- 这是示例应用，演示如何使用 DKI 插件
- 生产环境应使用自己的后端系统

Author: AGI Demo Project
Version: 2.0.0
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from dki.example_app.service import ExampleAppService
from dki.api.monitoring_routes import (
    create_monitoring_router,
    set_startup_time,
    set_dki_plugin,
    set_user_adapter,
)


# ============ Request/Response Models ============

class RegisterUserRequest(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str] = None


class AddPreferenceRequest(BaseModel):
    preference_type: str
    preference_text: str
    priority: int = 0


class UpdatePreferenceRequest(BaseModel):
    preference_text: Optional[str] = None
    priority: Optional[int] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    use_dki: bool = True
    force_alpha: Optional[float] = None
    max_new_tokens: int = 512
    temperature: float = 0.7


class ChatResponseModel(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    dki: Dict[str, Any] = Field(default_factory=dict)


# ============ App Factory ============

def create_example_app(
    service: Optional[ExampleAppService] = None,
    title: str = "DKI Example Application",
    description: str = "示例应用，演示如何使用 DKI 插件",
) -> FastAPI:
    """
    创建示例应用
    
    Args:
        service: ExampleAppService 实例
        title: 应用标题
        description: 应用描述
        
    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title=title,
        description=description,
        version="2.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 服务实例
    _service: Optional[ExampleAppService] = service
    
    @app.on_event("startup")
    async def startup():
        nonlocal _service
        
        set_startup_time(time.time())
        
        if _service:
            await _service.initialize()
            
            # 设置监控 API 的依赖
            if _service.dki_plugin:
                set_dki_plugin(_service.dki_plugin)
            set_user_adapter(_service.adapter)
        
        logger.info("Example application started")
    
    @app.on_event("shutdown")
    async def shutdown():
        if _service:
            await _service.shutdown()
        logger.info("Example application stopped")
    
    def get_service() -> ExampleAppService:
        if _service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return _service
    
    # ============ 用户 API ============
    
    @app.post("/api/users", tags=["Users"])
    async def register_user(request: RegisterUserRequest):
        """注册用户"""
        service = get_service()
        return service.register_user(
            user_id=request.user_id,
            username=request.username,
            display_name=request.display_name,
        )
    
    @app.get("/api/users/{user_id}", tags=["Users"])
    async def get_user(user_id: str):
        """获取用户信息"""
        service = get_service()
        user = service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    # ============ 偏好 API ============
    
    @app.get("/api/users/{user_id}/preferences", tags=["Preferences"])
    async def get_preferences(user_id: str):
        """获取用户偏好"""
        service = get_service()
        return await service.get_preferences(user_id)
    
    @app.post("/api/users/{user_id}/preferences", tags=["Preferences"])
    async def add_preference(user_id: str, request: AddPreferenceRequest):
        """添加用户偏好"""
        service = get_service()
        return service.add_preference(
            user_id=user_id,
            preference_type=request.preference_type,
            preference_text=request.preference_text,
            priority=request.priority,
        )
    
    @app.put("/api/users/{user_id}/preferences/{preference_id}", tags=["Preferences"])
    async def update_preference(
        user_id: str,
        preference_id: str,
        request: UpdatePreferenceRequest,
    ):
        """更新用户偏好"""
        service = get_service()
        success = service.update_preference(
            user_id=user_id,
            preference_id=preference_id,
            preference_text=request.preference_text,
            priority=request.priority,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Preference not found")
        return {"status": "ok"}
    
    @app.delete("/api/users/{user_id}/preferences/{preference_id}", tags=["Preferences"])
    async def delete_preference(user_id: str, preference_id: str):
        """删除用户偏好"""
        service = get_service()
        success = service.delete_preference(user_id, preference_id)
        if not success:
            raise HTTPException(status_code=404, detail="Preference not found")
        return {"status": "ok"}
    
    # ============ 会话 API ============
    
    @app.get("/api/users/{user_id}/sessions", tags=["Sessions"])
    async def get_sessions(
        user_id: str,
        limit: int = Query(default=10, ge=1, le=100),
    ):
        """获取用户会话列表"""
        service = get_service()
        return await service.get_sessions(user_id, limit=limit)
    
    @app.post("/api/users/{user_id}/sessions", tags=["Sessions"])
    async def create_session(user_id: str, request: CreateSessionRequest):
        """创建会话"""
        service = get_service()
        return service.create_session(user_id=user_id, title=request.title)
    
    @app.get("/api/sessions/{session_id}/history", tags=["Sessions"])
    async def get_session_history(
        session_id: str,
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """获取会话历史"""
        service = get_service()
        return await service.get_session_history(session_id, limit=limit)
    
    # ============ 聊天 API ============
    
    @app.post("/api/sessions/{session_id}/chat", tags=["Chat"], response_model=ChatResponseModel)
    async def chat(
        session_id: str,
        user_id: str = Query(..., description="用户 ID"),
        request: ChatRequest = ...,
    ):
        """
        聊天
        
        调用 DKI 插件进行增强聊天:
        1. DKI 通过适配器读取用户偏好
        2. DKI 通过适配器检索相关历史
        3. DKI 执行 K/V 注入
        4. DKI 调用 LLM 推理
        """
        service = get_service()
        
        response = await service.chat(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            use_dki=request.use_dki,
            force_alpha=request.force_alpha,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        
        return ChatResponseModel(
            message_id=response.message_id,
            session_id=response.session_id,
            role=response.role,
            content=response.content,
            timestamp=response.timestamp,
            dki={
                "enabled": response.dki_enabled,
                "alpha": response.dki_alpha,
                "preference_tokens": response.dki_preference_tokens,
                "history_tokens": response.dki_history_tokens,
                "cache_hit": response.dki_cache_hit,
                "latency_ms": response.dki_latency_ms,
            },
        )
    
    # ============ 统计 API ============
    
    @app.get("/api/stats", tags=["Stats"])
    async def get_stats():
        """获取统计数据"""
        service = get_service()
        return service.get_stats()
    
    # ============ 监控 API ============
    
    # 包含 DKI 监控 API
    monitoring_router = create_monitoring_router()
    app.include_router(monitoring_router)
    
    return app
