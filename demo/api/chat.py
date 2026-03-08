"""
Demo Chat API — ★ 核心对话路由

完整的 Demo 对话流程:
1. 验证用户身份
2. 确保会话存在 (auto-create)
3. 将用户消息写入 Demo DB ← 上层应用的职责
4. 调用 dki_plugin.chat()    ← DKI 只读取数据并推理
5. 将助手回复写入 Demo DB   ← 上层应用的职责
6. 返回响应

支持同步和异步 store 操作。

Author: AGI Demo Project
Version: 2.1.0
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from demo.api.deps import get_store, get_dki_plugin, require_auth, is_async_store
from demo.store.base import IChatStore


# ============ Request/Response Models ============

class ChatSendRequest(BaseModel):
    """对话请求 (与前端 /v1/dki/chat 请求格式兼容)"""
    query: str = Field(..., description="原始用户输入")
    user_id: str = Field(..., description="用户标识")
    session_id: Optional[str] = Field(None, description="会话标识")
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(2048, ge=1, le=8192)
    force_alpha: Optional[float] = Field(None, ge=0, le=1)
    use_hybrid: bool = True


class DKIMetadataResponse(BaseModel):
    """DKI 元数据"""
    injection_enabled: bool = False
    alpha: float = 0.0
    preference_tokens: int = 0
    history_tokens: int = 0
    cache_hit: bool = False
    cache_tier: str = "none"
    latency_ms: float = 0
    retrieval_mode: str = "unknown"
    preferences_count: int = 0
    relevant_history_count: int = 0


class ChatSendResponse(BaseModel):
    """对话响应 (与前端 DKIChatResponse 格式兼容)"""
    id: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    dki_metadata: DKIMetadataResponse = Field(default_factory=DKIMetadataResponse)
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    created: int


# ============ Router ============

def create_chat_router() -> APIRouter:
    """创建对话路由"""
    
    router = APIRouter(tags=["Chat"])
    
    @router.post(
        "/v1/dki/chat",
        response_model=ChatSendResponse,
        summary="DKI 增强对话",
        description="Demo 对话端点: 写入消息 → 调用 DKI Plugin → 写入回复",
    )
    async def dki_chat(
        request: ChatSendRequest,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
        dki_plugin = Depends(get_dki_plugin),
    ):
        """
        ★ Demo 对话核心流程
        
        1. 验证用户身份 (token → user_id)
        2. 确保会话存在
        3. 写入用户消息到 Demo DB
        4. 调用 dki_plugin.chat()
        5. 写入助手回复到 Demo DB
        6. 返回响应
        """
        _async = is_async_store(store)

        # ============ Step 1: 验证用户身份 ============
        verified_user_id = user["id"]
        
        # Cross-validate request user_id with token user
        if request.user_id and request.user_id != verified_user_id:
            logger.warning(
                f"User ID mismatch: request={request.user_id}, "
                f"token={verified_user_id}. Using token user_id."
            )
        
        # ============ Step 2: 确保会话存在 ============
        session_id = request.session_id or verified_user_id
        
        if _async:
            demo_session = await store.a_get_session(session_id)
        else:
            demo_session = store.get_session(session_id)

        if not demo_session:
            # Auto-create session
            if _async:
                demo_session = await store.a_create_session(
                    user_id=verified_user_id,
                    title="New Chat",
                    session_id=session_id,
                )
            else:
                demo_session = store.create_session(
                    user_id=verified_user_id,
                    title="New Chat",
                    session_id=session_id,
                )
            logger.info(f"Auto-created session: {session_id}")
        
        # ============ Step 3: ★ 写入用户消息到 Demo DB ============
        if _async:
            await store.a_add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="user",
                content=request.query,
            )
        else:
            store.add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="user",
                content=request.query,
            )
        
        # ============ Step 4: ★ 调用 DKI Plugin ============
        try:
            response = await dki_plugin.chat(
                query=request.query,
                user_id=verified_user_id,
                session_id=session_id,
                force_alpha=request.force_alpha,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception as e:
            import traceback
            logger.error(f"DKI Plugin chat error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"DKI chat error: {str(e)}")
        
        # ============ Step 5: ★ 写入助手回复到 Demo DB ============
        meta = response.metadata
        msg_metadata = {
            "injection_enabled": meta.injection_enabled,
            "alpha": meta.alpha,
            "latency_ms": meta.latency_ms,
            "retrieval_mode": meta.retrieval_mode,
        }
        if _async:
            await store.a_add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="assistant",
                content=response.text,
                metadata=msg_metadata,
            )
        else:
            store.add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="assistant",
                content=response.text,
                metadata=msg_metadata,
            )
        
        # ============ Step 6: 构建响应 ============
        request_id = meta.request_id if hasattr(meta, 'request_id') and meta.request_id else f"demo-{uuid.uuid4().hex[:8]}"
        
        dki_metadata = DKIMetadataResponse(
            injection_enabled=meta.injection_enabled,
            alpha=meta.alpha,
            preference_tokens=meta.preference_tokens,
            history_tokens=meta.history_tokens,
            cache_hit=meta.preference_cache_hit,
            cache_tier=meta.preference_cache_tier or "none",
            latency_ms=meta.latency_ms,
            retrieval_mode=meta.retrieval_mode,
            preferences_count=meta.preferences_count,
            relevant_history_count=meta.relevant_history_count,
        )
        
        return ChatSendResponse(
            id=request_id,
            text=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            dki_metadata=dki_metadata,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": "stop",
            }],
            created=int(datetime.now(timezone.utc).timestamp()),
        )
    
    @router.post(
        "/v1/dki/chat/stream",
        summary="DKI 流式对话",
        description="SSE 流式对话端点: 逐 token 返回生成结果",
    )
    async def dki_chat_stream(
        request: ChatSendRequest,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
        dki_plugin = Depends(get_dki_plugin),
    ):
        """
        ★ 流式对话端点 (SSE)
        
        返回 Server-Sent Events 流:
        - event: token  → data: {"content": "..."}
        - event: metadata → data: {DKI 元数据}
        - event: done   → data: {"text": "完整文本"}
        - event: error  → data: {"error": "错误信息"}
        """
        _async = is_async_store(store)
        verified_user_id = user["id"]
        
        # 确保会话存在
        session_id = request.session_id or verified_user_id
        if _async:
            demo_session = await store.a_get_session(session_id)
        else:
            demo_session = store.get_session(session_id)
        
        if not demo_session:
            if _async:
                await store.a_create_session(
                    user_id=verified_user_id,
                    title="New Chat",
                    session_id=session_id,
                )
            else:
                store.create_session(
                    user_id=verified_user_id,
                    title="New Chat",
                    session_id=session_id,
                )
        
        # 写入用户消息
        if _async:
            await store.a_add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="user",
                content=request.query,
            )
        else:
            store.add_message(
                session_id=session_id,
                user_id=verified_user_id,
                role="user",
                content=request.query,
            )
        
        async def event_generator() -> AsyncIterator[str]:
            """SSE 事件生成器"""
            full_text = ""
            try:
                # 检查 DKI Plugin 是否支持流式
                if not hasattr(dki_plugin, 'chat_stream'):
                    # 回退到非流式
                    response = await dki_plugin.chat(
                        query=request.query,
                        user_id=verified_user_id,
                        session_id=session_id,
                        force_alpha=request.force_alpha,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                    full_text = response.text
                    yield f"event: token\ndata: {json.dumps({'content': response.text}, ensure_ascii=False)}\n\n"
                    
                    meta = response.metadata
                    yield f"event: metadata\ndata: {json.dumps({'injection_enabled': meta.injection_enabled, 'alpha': meta.alpha, 'latency_ms': meta.latency_ms, 'retrieval_mode': meta.retrieval_mode}, ensure_ascii=False)}\n\n"
                else:
                    # 流式生成
                    async for chunk in dki_plugin.chat_stream(
                        query=request.query,
                        user_id=verified_user_id,
                        session_id=session_id,
                        force_alpha=request.force_alpha,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ):
                        chunk_type = chunk.get("type", "token")
                        
                        if chunk_type == "token":
                            content = chunk.get("content", "")
                            full_text += content
                            yield f"event: token\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                        
                        elif chunk_type == "metadata":
                            yield f"event: metadata\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        
                        elif chunk_type == "done":
                            full_text = chunk.get("text", full_text)
                            yield f"event: done\ndata: {json.dumps({'text': full_text, 'input_tokens': chunk.get('input_tokens', 0), 'output_tokens': chunk.get('output_tokens', 0)}, ensure_ascii=False)}\n\n"
                        
                        elif chunk_type == "error":
                            yield f"event: error\ndata: {json.dumps({'error': chunk.get('error', 'Unknown error')}, ensure_ascii=False)}\n\n"
                            return
                
                # 写入助手回复
                if full_text:
                    if _async:
                        await store.a_add_message(
                            session_id=session_id,
                            user_id=verified_user_id,
                            role="assistant",
                            content=full_text,
                        )
                    else:
                        store.add_message(
                            session_id=session_id,
                            user_id=verified_user_id,
                            role="assistant",
                            content=full_text,
                        )
                
                # 发送完成事件
                yield f"event: done\ndata: {json.dumps({'text': full_text}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"DKI stream error: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    @router.get("/v1/dki/info", summary="DKI 插件信息")
    async def dki_info(dki_plugin = Depends(get_dki_plugin)):
        """获取 DKI Plugin 信息"""
        try:
            stats = dki_plugin.get_stats()
            return {
                "status": "ready",
                "version": "2.1.0",
                "type": "DKIPlugin",
                "stats": stats,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    return router
