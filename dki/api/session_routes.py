"""
Session Management Routes
FastAPI routes for chat session management

Author: AGI Demo Project
Version: 1.0.0
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from dki.api.auth_routes import require_auth, get_current_user


# Simple in-memory session store for demo purposes
_sessions_db = {}
_messages_db = {}


class SessionCreate(BaseModel):
    title: str = "新对话"


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


def create_session_router() -> APIRouter:
    """Create session management router."""
    
    router = APIRouter(prefix="/api/sessions", tags=["Sessions"])
    
    @router.get("", response_model=List[SessionResponse])
    async def list_sessions(
        limit: int = Query(default=50, ge=1, le=100),
        user: dict = Depends(require_auth),
    ):
        """
        List all sessions for the current user.
        """
        user_sessions = [
            s for s in _sessions_db.values()
            if s.get("user_id") == user["id"]
        ]
        
        # Sort by updated_at descending
        user_sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return [
            SessionResponse(
                id=s["id"],
                title=s["title"],
                user_id=s.get("user_id"),
                message_count=s.get("message_count", 0),
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                preview=s.get("preview"),
            )
            for s in user_sessions[:limit]
        ]
    
    @router.post("", response_model=SessionResponse)
    async def create_session(
        request: SessionCreate,
        user: dict = Depends(require_auth),
    ):
        """
        Create a new chat session.
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()
        
        session = {
            "id": session_id,
            "title": request.title,
            "user_id": user["id"],
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
            "preview": None,
        }
        
        _sessions_db[session_id] = session
        _messages_db[session_id] = []
        
        logger.info(f"Created session: {session_id} for user: {user['id']}")
        
        return SessionResponse(**session)
    
    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Get a specific session.
        """
        session = _sessions_db.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.get("user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return SessionResponse(**session)
    
    @router.patch("/{session_id}", response_model=SessionResponse)
    async def update_session(
        session_id: str,
        request: SessionUpdate,
        user: dict = Depends(require_auth),
    ):
        """
        Update a session (e.g., rename).
        """
        session = _sessions_db.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.get("user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if request.title is not None:
            session["title"] = request.title
        
        session["updated_at"] = datetime.now().isoformat()
        
        return SessionResponse(**session)
    
    @router.delete("/{session_id}")
    async def delete_session(
        session_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Delete a session and its messages.
        """
        session = _sessions_db.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.get("user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        del _sessions_db[session_id]
        if session_id in _messages_db:
            del _messages_db[session_id]
        
        logger.info(f"Deleted session: {session_id}")
        
        return {"status": "deleted", "session_id": session_id}
    
    @router.get("/{session_id}/messages", response_model=List[MessageResponse])
    async def get_session_messages(
        session_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        user: dict = Depends(require_auth),
    ):
        """
        Get all messages in a session.
        """
        session = _sessions_db.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.get("user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        messages = _messages_db.get(session_id, [])
        
        return [
            MessageResponse(
                id=m["id"],
                session_id=m["session_id"],
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
                dki_metadata=m.get("dki_metadata"),
            )
            for m in messages[-limit:]
        ]
    
    return router


def add_message_to_session(
    session_id: str,
    role: str,
    content: str,
    dki_metadata: Optional[dict] = None,
) -> dict:
    """
    Add a message to a session (called from chat endpoint).
    """
    if session_id not in _sessions_db:
        return None
    
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    
    message = {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": now,
        "dki_metadata": dki_metadata,
    }
    
    if session_id not in _messages_db:
        _messages_db[session_id] = []
    
    _messages_db[session_id].append(message)
    
    # Update session
    session = _sessions_db[session_id]
    session["message_count"] = len(_messages_db[session_id])
    session["updated_at"] = now
    if role == "user":
        session["preview"] = content[:50] + "..." if len(content) > 50 else content
    
    return message
