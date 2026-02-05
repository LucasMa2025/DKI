"""
Database Models for DKI System
SQLAlchemy ORM models for persistent storage
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime,
    Boolean, LargeBinary, ForeignKey, Index, event
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class Session(Base):
    """Session model for tracking user sessions."""
    
    __tablename__ = 'sessions'
    
    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    _metadata = Column('metadata', Text, default='{}')
    is_active = Column(Boolean, default=True)
    
    # Relationships
    memories = relationship("Memory", back_populates="session", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata,
            'is_active': self.is_active,
            'memory_count': len(self.memories) if self.memories else 0,
        }


class Memory(Base):
    """Memory model for storing user memories."""
    
    __tablename__ = 'memories'
    
    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey('sessions.id', ondelete='CASCADE'), index=True)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary)  # Serialized numpy array
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    _metadata = Column('metadata', Text, default='{}')
    is_active = Column(Boolean, default=True)
    
    # Relationships
    session = relationship("Session", back_populates="memories")
    kv_caches = relationship("KVCache", back_populates="memory", cascade="all, delete-orphan")
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata,
            'is_active': self.is_active,
        }


class Conversation(Base):
    """Conversation model for storing chat history."""
    
    __tablename__ = 'conversations'
    
    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    injection_mode = Column(String(16))  # 'rag', 'dki', 'none'
    injection_alpha = Column(Float)
    _memory_ids = Column('memory_ids', Text)  # JSON array
    latency_ms = Column(Float)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    session = relationship("Session", back_populates="conversations")
    
    @hybrid_property
    def memory_ids(self) -> List[str]:
        return json.loads(self._memory_ids) if self._memory_ids else []
    
    @memory_ids.setter
    def memory_ids(self, value: List[str]):
        self._memory_ids = json.dumps(value)
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'injection_mode': self.injection_mode,
            'injection_alpha': self.injection_alpha,
            'memory_ids': self.memory_ids,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata,
        }


class KVCache(Base):
    """KV Cache model for storing computed Key-Value representations."""
    
    __tablename__ = 'kv_cache'
    
    id = Column(String(64), primary_key=True)
    memory_id = Column(String(64), ForeignKey('memories.id', ondelete='CASCADE'), nullable=False, index=True)
    model_name = Column(String(128), nullable=False)
    layer_idx = Column(Integer, nullable=False)
    key_cache = Column(LargeBinary)  # Serialized tensor
    value_cache = Column(LargeBinary)  # Serialized tensor
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    memory = relationship("Memory", back_populates="kv_caches")
    
    # Unique constraint
    __table_args__ = (
        Index('ix_kv_cache_unique', 'memory_id', 'model_name', 'layer_idx', unique=True),
    )
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)


class Experiment(Base):
    """Experiment model for tracking experiment runs."""
    
    __tablename__ = 'experiments'
    
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    description = Column(Text)
    _config = Column('config', Text, nullable=False)  # JSON string
    status = Column(String(16), default='pending', index=True)  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")
    
    @hybrid_property
    def config(self) -> Dict[str, Any]:
        return json.loads(self._config) if self._config else {}
    
    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = json.dumps(value, ensure_ascii=False)
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
        }


class ExperimentResult(Base):
    """Experiment result model for storing experiment outcomes."""
    
    __tablename__ = 'experiment_results'
    
    id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64), ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False, index=True)
    mode = Column(String(16), nullable=False, index=True)  # 'rag', 'dki', 'baseline'
    dataset = Column(String(64), nullable=False)
    _metrics = Column('metrics', Text, nullable=False)  # JSON string
    sample_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    
    @hybrid_property
    def metrics(self) -> Dict[str, Any]:
        return json.loads(self._metrics) if self._metrics else {}
    
    @metrics.setter
    def metrics(self, value: Dict[str, Any]):
        self._metrics = json.dumps(value, ensure_ascii=False)
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'mode': self.mode,
            'dataset': self.dataset,
            'metrics': self.metrics,
            'sample_count': self.sample_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata,
        }


class AuditLog(Base):
    """Audit log model for tracking system actions."""
    
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    action = Column(String(64), nullable=False)
    _memory_ids = Column('memory_ids', Text)  # JSON array
    alpha = Column(Float)
    mode = Column(String(16))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    _metadata = Column('metadata', Text, default='{}')
    
    @hybrid_property
    def memory_ids(self) -> List[str]:
        return json.loads(self._memory_ids) if self._memory_ids else []
    
    @memory_ids.setter
    def memory_ids(self, value: List[str]):
        self._memory_ids = json.dumps(value)
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)


class ModelRegistry(Base):
    """Model registry for tracking available models."""
    
    __tablename__ = 'model_registry'
    
    id = Column(String(64), primary_key=True)
    engine = Column(String(16), nullable=False, index=True)  # vllm, llama, deepseek, glm
    model_name = Column(String(128), nullable=False)
    _config = Column('config', Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    _metadata = Column('metadata', Text, default='{}')
    
    @hybrid_property
    def config(self) -> Dict[str, Any]:
        return json.loads(self._config) if self._config else {}
    
    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = json.dumps(value, ensure_ascii=False)
    
    @hybrid_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, ensure_ascii=False)
