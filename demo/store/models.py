"""
Demo ORM Models — 独立于 DKI/dki/database/models.py

独立的 declarative_base，不与实验系统共享。
表名使用 demo_ 前缀，避免与实验系统冲突。

Author: AGI Demo Project
Version: 2.0.0
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime,
    Boolean, LargeBinary, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, relationship

DemoBase = declarative_base()


def _utcnow():
    """UTC now (timezone-aware)"""
    return datetime.now(timezone.utc)


class DemoUser(DemoBase):
    """
    Demo 用户模型
    
    用户管理:
    - 支持注册 (用户名 + 密码 + 可选邮箱)
    - 支持密码修改和重置
    - 密码使用 SHA-256 哈希存储 (演示级别)
    - 向后兼容: password_hash 为空时允许任意密码登录 (demo mode)
    """
    __tablename__ = 'demo_users'
    
    id = Column(String(64), primary_key=True)
    username = Column(String(64), nullable=False, unique=True, index=True)
    display_name = Column(String(128))
    email = Column(String(128), index=True)
    avatar = Column(String(256))
    password_hash = Column(String(128), nullable=True)  # SHA-256 hash, NULL = demo mode
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    last_login_at = Column(DateTime)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    sessions = relationship("DemoSession", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("DemoPreference", back_populates="user", cascade="all, delete-orphan")
    
    def get_metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """SHA-256 密码哈希 (演示级别, 生产环境应使用 bcrypt)"""
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def set_password(self, password: str) -> None:
        """设置密码"""
        self.password_hash = self.hash_password(password)
    
    def verify_password(self, password: str) -> bool:
        """
        验证密码
        
        - password_hash 为空: demo mode, 任意密码通过
        - password_hash 非空: 验证 SHA-256 哈希
        """
        if not self.password_hash:
            return True  # demo mode: 无密码时任意密码通过
        return self.password_hash == self.hash_password(password)
    
    @property
    def has_password(self) -> bool:
        """是否已设置密码"""
        return bool(self.password_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'username': self.username,
            'displayName': self.display_name,
            'email': self.email,
            'avatar': self.avatar,
            'isActive': self.is_active,
            'hasPassword': self.has_password,
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'lastLoginAt': self.last_login_at.isoformat() if self.last_login_at else None,
            'metadata': self.get_metadata(),
        }


class DemoSession(DemoBase):
    """
    Demo 会话模型
    
    每个用户可以有多个会话，每个会话包含多条消息。
    """
    __tablename__ = 'demo_sessions'
    
    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey('demo_users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = Column(String(256), default='New Chat')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    user = relationship("DemoUser", back_populates="sessions")
    messages = relationship("DemoMessage", back_populates="session", cascade="all, delete-orphan")
    
    def get_metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.get_metadata(),
        }


class DemoMessage(DemoBase):
    """
    Demo 消息模型
    
    存储对话消息，包括用户消息和助手回复。
    embedding 字段预留给 pgvector 支持。
    """
    __tablename__ = 'demo_messages'
    
    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey('demo_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)  # 预留 pgvector 支持 (SQLite 下为 NULL)
    _metadata = Column('metadata', Text, default='{}')
    created_at = Column(DateTime, default=_utcnow, index=True)
    
    # Relationships
    session = relationship("DemoSession", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('ix_demo_messages_user_created', 'user_id', 'created_at'),
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'role': self.role,
            'content': self.content,
            'metadata': self.get_metadata(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class DemoPreference(DemoBase):
    """
    Demo 用户偏好模型
    
    存储用户偏好，用于 DKI K/V 注入。
    与 IUserDataAdapter.get_user_preferences 对齐。
    """
    __tablename__ = 'demo_preferences'
    
    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey('demo_users.id', ondelete='CASCADE'), nullable=False, index=True)
    preference_text = Column(Text, nullable=False)
    preference_type = Column(String(32), default='general')
    priority = Column(Integer, default=5)  # 0-10, higher = more important
    category = Column(String(64), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    _metadata = Column('metadata', Text, default='{}')
    
    # Relationships
    user = relationship("DemoUser", back_populates="preferences")
    
    def get_metadata(self) -> Dict[str, Any]:
        return json.loads(self._metadata) if self._metadata else {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = json.dumps(value, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preference_text': self.preference_text,
            'preference_type': self.preference_type,
            'priority': self.priority,
            'category': self.category,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.get_metadata(),
        }
