"""
Demo Messages API — 消息管理路由 (补充)

提供独立的消息查询和搜索 API。
注意: 消息写入主要通过 chat.py 的对话流程完成。
支持同步和异步 store 操作。

Author: AGI Demo Project
Version: 2.1.0
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from demo.api.deps import get_store, require_auth, is_async_store
from demo.store.base import IChatStore


# ============ Response Models ============

class MessageSearchResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    score: Optional[float] = None


# ============ Router ============

def create_message_router() -> APIRouter:
    """创建消息管理路由"""
    
    router = APIRouter(prefix="/api/messages", tags=["Messages"])
    
    @router.get("/search", response_model=List[MessageSearchResponse])
    async def search_messages(
        q: str = Query(..., description="搜索查询"),
        session_id: Optional[str] = Query(default=None),
        limit: int = Query(default=10, ge=1, le=50),
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """
        搜索用户消息 (BM25)
        
        用于前端消息搜索功能。
        """
        if is_async_store(store):
            messages = await store.a_search_messages(
                user_id=user["id"],
                query=q,
                limit=limit,
                session_id=session_id,
            )
        else:
            messages = store.search_messages(
                user_id=user["id"],
                query=q,
                limit=limit,
                session_id=session_id,
            )
        
        return [
            MessageSearchResponse(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                timestamp=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ]
    
    return router
