"""
Experiment ORM Models — 与 demo/store/models.py 完全一致

独立的 declarative_base，不与 demo 共享 metadata。
表名使用 demo_ 前缀 (与 demo 一致, 但写入不同的 db 文件)。

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store/models.py 2.0.0)
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
    """用户模型"""
    __tablename__ = 'demo_users'

    id = Column(String(64), primary_key=True)
    username = Column(String(64), nullable=False, unique=True, index=True)
    display_name = Column(String(128))
    email = Column(String(128), index=True)
    avatar = Column(String(256))
    password_hash = Column(String(128), nullable=True)
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
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def set_password(self, password: str) -> None:
        self.password_hash = self.hash_password(password)

    def verify_password(self, password: str) -> bool:
        if not self.password_hash:
            return True
        return self.password_hash == self.hash_password(password)

    @property
    def has_password(self) -> bool:
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
    """会话模型"""
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
    """消息模型"""
    __tablename__ = 'demo_messages'

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey('demo_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)
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
    """用户偏好模型"""
    __tablename__ = 'demo_preferences'

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey('demo_users.id', ondelete='CASCADE'), nullable=False, index=True)
    preference_text = Column(Text, nullable=False)
    preference_type = Column(String(32), default='general')
    priority = Column(Integer, default=5)
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
