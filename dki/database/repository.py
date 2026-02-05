"""
Repository Pattern Implementation for DKI System
Provides CRUD operations for database models
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session as SQLSession
from sqlalchemy import desc
from loguru import logger

from dki.database.models import (
    Session, Memory, Conversation, KVCache, 
    Experiment, ExperimentResult, AuditLog
)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, db: SQLSession):
        self.db = db
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}{uuid.uuid4().hex[:16]}"


class SessionRepository(BaseRepository):
    """Repository for Session operations."""
    
    def create(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        session = Session(
            id=session_id or self.generate_id("sess_"),
            user_id=user_id,
        )
        if metadata:
            session.metadata = metadata
        
        self.db.add(session)
        self.db.flush()
        logger.debug(f"Created session: {session.id}")
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.db.query(Session).filter(Session.id == session_id).first()
    
    def get_or_create(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Tuple[Session, bool]:
        """Get existing session or create new one."""
        session = self.get(session_id)
        if session:
            return session, False
        return self.create(session_id=session_id, user_id=user_id), True
    
    def list_by_user(self, user_id: str, limit: int = 100) -> List[Session]:
        """List sessions by user ID."""
        return (
            self.db.query(Session)
            .filter(Session.user_id == user_id, Session.is_active == True)
            .order_by(desc(Session.updated_at))
            .limit(limit)
            .all()
        )
    
    def update(self, session_id: str, **kwargs) -> Optional[Session]:
        """Update session attributes."""
        session = self.get(session_id)
        if session:
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = datetime.utcnow()
            self.db.flush()
        return session
    
    def delete(self, session_id: str) -> bool:
        """Soft delete session."""
        session = self.get(session_id)
        if session:
            session.is_active = False
            self.db.flush()
            return True
        return False


class MemoryRepository(BaseRepository):
    """Repository for Memory operations."""
    
    def create(
        self,
        session_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Create a new memory."""
        memory = Memory(
            id=memory_id or self.generate_id("mem_"),
            session_id=session_id,
            content=content,
        )
        
        if embedding is not None:
            memory.embedding = embedding.tobytes()
        
        if metadata:
            memory.metadata = metadata
        
        self.db.add(memory)
        self.db.flush()
        logger.debug(f"Created memory: {memory.id}")
        return memory
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID."""
        return self.db.query(Memory).filter(Memory.id == memory_id).first()
    
    def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[Memory]:
        """Get all memories for a session."""
        query = self.db.query(Memory).filter(Memory.session_id == session_id)
        
        if not include_inactive:
            query = query.filter(Memory.is_active == True)
        
        return query.order_by(desc(Memory.created_at)).limit(limit).all()
    
    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Get memory embedding as numpy array."""
        memory = self.get(memory_id)
        if memory and memory.embedding:
            return np.frombuffer(memory.embedding, dtype=np.float32)
        return None
    
    def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update memory embedding."""
        memory = self.get(memory_id)
        if memory:
            memory.embedding = embedding.tobytes()
            memory.updated_at = datetime.utcnow()
            self.db.flush()
            return True
        return False
    
    def search_by_content(self, session_id: str, query: str) -> List[Memory]:
        """Simple text search in memory content."""
        return (
            self.db.query(Memory)
            .filter(
                Memory.session_id == session_id,
                Memory.is_active == True,
                Memory.content.ilike(f"%{query}%")
            )
            .all()
        )
    
    def delete(self, memory_id: str) -> bool:
        """Soft delete memory."""
        memory = self.get(memory_id)
        if memory:
            memory.is_active = False
            self.db.flush()
            return True
        return False
    
    def bulk_create(
        self,
        session_id: str,
        memories_data: List[Dict[str, Any]],
    ) -> List[Memory]:
        """Bulk create memories."""
        memories = []
        for data in memories_data:
            memory = self.create(
                session_id=session_id,
                content=data.get('content', ''),
                embedding=data.get('embedding'),
                memory_id=data.get('id'),
                metadata=data.get('metadata'),
            )
            memories.append(memory)
        return memories


class ConversationRepository(BaseRepository):
    """Repository for Conversation operations."""
    
    def create(
        self,
        session_id: str,
        role: str,
        content: str,
        injection_mode: Optional[str] = None,
        injection_alpha: Optional[float] = None,
        memory_ids: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation entry."""
        conversation = Conversation(
            id=self.generate_id("conv_"),
            session_id=session_id,
            role=role,
            content=content,
            injection_mode=injection_mode,
            injection_alpha=injection_alpha,
            latency_ms=latency_ms,
        )
        
        if memory_ids:
            conversation.memory_ids = memory_ids
        
        if metadata:
            conversation.metadata = metadata
        
        self.db.add(conversation)
        self.db.flush()
        return conversation
    
    def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Conversation]:
        """Get conversation history for a session."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.created_at)
            .limit(limit)
            .all()
        )
    
    def get_recent(
        self,
        session_id: str,
        n_turns: int = 10,
    ) -> List[Conversation]:
        """Get recent conversation turns."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(desc(Conversation.created_at))
            .limit(n_turns * 2)  # Each turn has user + assistant
            .all()[::-1]  # Reverse to get chronological order
        )


class ExperimentRepository(BaseRepository):
    """Repository for Experiment operations."""
    
    def create(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            id=experiment_id or self.generate_id("exp_"),
            name=name,
            description=description,
        )
        experiment.config = config
        
        self.db.add(experiment)
        self.db.flush()
        logger.info(f"Created experiment: {experiment.id}")
        return experiment
    
    def get(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def list_all(self, status: Optional[str] = None, limit: int = 100) -> List[Experiment]:
        """List experiments."""
        query = self.db.query(Experiment)
        
        if status:
            query = query.filter(Experiment.status == status)
        
        return query.order_by(desc(Experiment.created_at)).limit(limit).all()
    
    def update_status(
        self,
        experiment_id: str,
        status: str,
    ) -> Optional[Experiment]:
        """Update experiment status."""
        experiment = self.get(experiment_id)
        if experiment:
            experiment.status = status
            if status == 'running':
                experiment.started_at = datetime.utcnow()
            elif status in ('completed', 'failed'):
                experiment.completed_at = datetime.utcnow()
            self.db.flush()
        return experiment
    
    def add_result(
        self,
        experiment_id: str,
        mode: str,
        dataset: str,
        metrics: Dict[str, Any],
        sample_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """Add result to experiment."""
        result = ExperimentResult(
            id=self.generate_id("res_"),
            experiment_id=experiment_id,
            mode=mode,
            dataset=dataset,
            sample_count=sample_count,
        )
        result.metrics = metrics
        
        if metadata:
            result.metadata = metadata
        
        self.db.add(result)
        self.db.flush()
        return result
    
    def get_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Get all results for an experiment."""
        return (
            self.db.query(ExperimentResult)
            .filter(ExperimentResult.experiment_id == experiment_id)
            .all()
        )


class AuditLogRepository(BaseRepository):
    """Repository for AuditLog operations."""
    
    def log(
        self,
        action: str,
        session_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Create an audit log entry."""
        log_entry = AuditLog(
            session_id=session_id,
            action=action,
            alpha=alpha,
            mode=mode,
        )
        
        if memory_ids:
            log_entry.memory_ids = memory_ids
        
        if metadata:
            log_entry.metadata = metadata
        
        self.db.add(log_entry)
        self.db.flush()
        return log_entry
    
    def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Get audit logs for a session."""
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.session_id == session_id)
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
            .all()
        )
