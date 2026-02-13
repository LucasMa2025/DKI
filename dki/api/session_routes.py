"""
Session Management Routes
FastAPI routes for chat session management

Author: AGI Demo Project
Version: 1.0.0

修正: 使用实际的 SQLite 数据库而非内存存储
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from dki.api.auth_routes import require_auth, get_current_user
from dki.database.connection import DatabaseManager
from dki.database.repository import SessionRepository, ConversationRepository
from dki.config.config_loader import ConfigLoader


# Database manager singleton
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager."""
    global _db_manager
    if _db_manager is None:
        config = ConfigLoader().config
        _db_manager = DatabaseManager(
            db_path=config.database.path,
            echo=config.database.echo,
        )
    return _db_manager


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
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            conv_repo = ConversationRepository(db)
            
            sessions = session_repo.list_by_user(user["id"], limit=limit)
            
            result = []
            for s in sessions:
                # Get message count and preview
                messages = conv_repo.get_by_session(s.id, limit=1)
                message_count = len(conv_repo.get_by_session(s.id, limit=1000))
                preview = None
                if messages:
                    # Get the first user message as preview
                    user_msgs = [m for m in conv_repo.get_by_session(s.id, limit=10) if m.role == 'user']
                    if user_msgs:
                        preview = user_msgs[0].content[:50] + "..." if len(user_msgs[0].content) > 50 else user_msgs[0].content
                
                result.append(SessionResponse(
                    id=s.id,
                    title=getattr(s, 'title', None) or s.id,
                    user_id=s.user_id,
                    message_count=message_count,
                    created_at=s.created_at.isoformat() if s.created_at else datetime.now().isoformat(),
                    updated_at=s.updated_at.isoformat() if s.updated_at else datetime.now().isoformat(),
                    preview=preview,
                ))
            
            return result
    
    @router.post("", response_model=SessionResponse)
    async def create_session(
        request: SessionCreate,
        user: dict = Depends(require_auth),
    ):
        """
        Create a new chat session.
        """
        db_manager = get_db_manager()
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            
            # Create session with title in metadata
            session = session_repo.create(
                session_id=session_id,
                user_id=user["id"],
                metadata={"title": request.title},
            )
            
            logger.info(f"Created session: {session_id} for user: {user['id']}")
            
            return SessionResponse(
                id=session.id,
                title=request.title,
                user_id=session.user_id,
                message_count=0,
                created_at=session.created_at.isoformat() if session.created_at else now.isoformat(),
                updated_at=session.updated_at.isoformat() if session.updated_at else now.isoformat(),
                preview=None,
            )
    
    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Get a specific session.
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            conv_repo = ConversationRepository(db)
            
            session = session_repo.get(session_id)
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            if session.user_id != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Get message count
            messages = conv_repo.get_by_session(session_id, limit=1000)
            message_count = len(messages)
            
            # Get preview
            preview = None
            user_msgs = [m for m in messages if m.role == 'user']
            if user_msgs:
                preview = user_msgs[0].content[:50] + "..." if len(user_msgs[0].content) > 50 else user_msgs[0].content
            
            # Get title from metadata
            title = session_id
            if hasattr(session, 'metadata') and session.metadata:
                title = session.metadata.get('title', session_id)
            
            return SessionResponse(
                id=session.id,
                title=title,
                user_id=session.user_id,
                message_count=message_count,
                created_at=session.created_at.isoformat() if session.created_at else datetime.now().isoformat(),
                updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now().isoformat(),
                preview=preview,
            )
    
    @router.patch("/{session_id}", response_model=SessionResponse)
    async def update_session(
        session_id: str,
        request: SessionUpdate,
        user: dict = Depends(require_auth),
    ):
        """
        Update a session (e.g., rename).
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            
            session = session_repo.get(session_id)
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            if session.user_id != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Update title in metadata
            if request.title is not None:
                metadata = session.metadata or {}
                metadata['title'] = request.title
                session_repo.update(session_id, metadata=metadata)
            
            # Get updated session
            session = session_repo.get(session_id)
            title = session_id
            if hasattr(session, 'metadata') and session.metadata:
                title = session.metadata.get('title', session_id)
            
            return SessionResponse(
                id=session.id,
                title=title,
                user_id=session.user_id,
                message_count=0,
                created_at=session.created_at.isoformat() if session.created_at else datetime.now().isoformat(),
                updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now().isoformat(),
                preview=None,
            )
    
    @router.delete("/{session_id}")
    async def delete_session(
        session_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Delete a session and its messages.
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            
            session = session_repo.get(session_id)
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            if session.user_id != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Soft delete
            session_repo.delete(session_id)
            
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
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            conv_repo = ConversationRepository(db)
            
            session = session_repo.get(session_id)
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            if session.user_id != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            messages = conv_repo.get_by_session(session_id, limit=limit)
            
            return [
                MessageResponse(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.created_at.isoformat() if m.created_at else datetime.now().isoformat(),
                    dki_metadata={
                        "injection_mode": m.injection_mode,
                        "injection_alpha": m.injection_alpha,
                        "latency_ms": m.latency_ms,
                    } if m.injection_mode else None,
                )
                for m in messages
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
    
    Note: This function is kept for backward compatibility.
    The actual message logging is now handled by dki_system._log_conversation()
    """
    db_manager = get_db_manager()
    
    with db_manager.session_scope() as db:
        session_repo = SessionRepository(db)
        conv_repo = ConversationRepository(db)
        
        # Ensure session exists
        session = session_repo.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found, creating...")
            session_repo.create(session_id=session_id)
        
        # Create conversation entry
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        conv = conv_repo.create(
            session_id=session_id,
            role=role,
            content=content,
            injection_mode=dki_metadata.get("injection_mode") if dki_metadata else None,
            injection_alpha=dki_metadata.get("injection_alpha") if dki_metadata else None,
            latency_ms=dki_metadata.get("latency_ms") if dki_metadata else None,
        )
        
        return {
            "id": conv.id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": conv.created_at.isoformat() if conv.created_at else datetime.now().isoformat(),
            "dki_metadata": dki_metadata,
        }
