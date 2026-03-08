"""
Demo Sessions API — 会话管理路由

支持同步和异步 store 操作。

Author: AGI Demo Project
Version: 2.1.0
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from demo.api.deps import get_store, require_auth, is_async_store
from demo.store.base import IChatStore


# ============ Request/Response Models ============

class SessionCreate(BaseModel):
    title: str = "New Chat"


class SessionUpdate(BaseModel):
    title: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: str
    user_id: Optional[str] = None
    message_count: int = 0
    created_at: str
    updated_at: str
    preview: Optional[str] = None


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    dki_metadata: Optional[dict] = None


# ============ Helpers ============

def _get_preview(messages) -> Optional[str]:
    """Get preview from first user message."""
    user_msgs = [m for m in messages if m.role == "user"]
    if user_msgs:
        content = user_msgs[0].content
        return content[:50] + "..." if len(content) > 50 else content
    return None


# ============ Router ============

def create_session_router() -> APIRouter:
    """创建会话管理路由"""
    
    router = APIRouter(prefix="/api/sessions", tags=["Sessions"])
    
    @router.get("", response_model=List[SessionResponse])
    async def list_sessions(
        limit: int = Query(default=50, ge=1, le=100),
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """列出用户会话"""
        _async = is_async_store(store)

        if _async:
            sessions = await store.a_list_sessions(user["id"], limit=limit)
        else:
            sessions = store.list_sessions(user["id"], limit=limit)
        
        result = []
        for s in sessions:
            if _async:
                msg_count = await store.a_get_message_count(s.id)
            else:
                msg_count = store.get_message_count(s.id)
            
            preview = None
            if msg_count > 0:
                if _async:
                    messages = await store.a_get_messages(s.id, limit=10)
                else:
                    messages = store.get_messages(s.id, limit=10)
                preview = _get_preview(messages)
            
            result.append(SessionResponse(
                id=s.id,
                title=s.title or s.id,
                user_id=s.user_id,
                message_count=msg_count,
                created_at=s.created_at.isoformat() if s.created_at else "",
                updated_at=s.updated_at.isoformat() if s.updated_at else "",
                preview=preview,
            ))
        
        return result
    
    @router.post("", response_model=SessionResponse)
    async def create_session(
        request: SessionCreate,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """创建新会话"""
        if is_async_store(store):
            session = await store.a_create_session(
                user_id=user["id"],
                title=request.title,
            )
        else:
            session = store.create_session(
                user_id=user["id"],
                title=request.title,
            )
        
        logger.info(f"Created session: {session.id} for user: {user['id']}")
        
        return SessionResponse(
            id=session.id,
            title=session.title or request.title,
            user_id=session.user_id,
            message_count=0,
            created_at=session.created_at.isoformat() if session.created_at else "",
            updated_at=session.updated_at.isoformat() if session.updated_at else "",
            preview=None,
        )
    
    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """获取会话详情"""
        _async = is_async_store(store)

        if _async:
            session = await store.a_get_session(session_id)
        else:
            session = store.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if _async:
            msg_count = await store.a_get_message_count(session_id)
        else:
            msg_count = store.get_message_count(session_id)
        
        preview = None
        if msg_count > 0:
            if _async:
                messages = await store.a_get_messages(session_id, limit=10)
            else:
                messages = store.get_messages(session_id, limit=10)
            preview = _get_preview(messages)
        
        return SessionResponse(
            id=session.id,
            title=session.title or session_id,
            user_id=session.user_id,
            message_count=msg_count,
            created_at=session.created_at.isoformat() if session.created_at else "",
            updated_at=session.updated_at.isoformat() if session.updated_at else "",
            preview=preview,
        )
    
    @router.patch("/{session_id}", response_model=SessionResponse)
    async def update_session(
        session_id: str,
        request: SessionUpdate,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """更新会话 (重命名等)"""
        _async = is_async_store(store)

        if _async:
            session = await store.a_get_session(session_id)
        else:
            session = store.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        kwargs = {}
        if request.title is not None:
            kwargs["title"] = request.title
        
        if _async:
            updated = await store.a_update_session(session_id, **kwargs)
            msg_count = await store.a_get_message_count(session_id)
        else:
            updated = store.update_session(session_id, **kwargs)
            msg_count = store.get_message_count(session_id)
        
        return SessionResponse(
            id=updated.id,
            title=updated.title or session_id,
            user_id=updated.user_id,
            message_count=msg_count,
            created_at=updated.created_at.isoformat() if updated.created_at else "",
            updated_at=updated.updated_at.isoformat() if updated.updated_at else "",
            preview=None,
        )
    
    @router.delete("/{session_id}")
    async def delete_session(
        session_id: str,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """删除会话 (软删除)"""
        _async = is_async_store(store)

        if _async:
            session = await store.a_get_session(session_id)
        else:
            session = store.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if _async:
            await store.a_delete_session(session_id)
        else:
            store.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
        
        return {"status": "deleted", "session_id": session_id}
    
    @router.get("/{session_id}/messages", response_model=List[MessageResponse])
    async def get_session_messages(
        session_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """获取会话消息"""
        _async = is_async_store(store)

        if _async:
            session = await store.a_get_session(session_id)
        else:
            session = store.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if _async:
            messages = await store.a_get_messages(session_id, limit=limit)
        else:
            messages = store.get_messages(session_id, limit=limit)
        
        return [
            MessageResponse(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                timestamp=m.created_at.isoformat() if m.created_at else "",
                dki_metadata=m.get_metadata() if m.get_metadata() != {} else None,
            )
            for m in messages
        ]
    
    return router
