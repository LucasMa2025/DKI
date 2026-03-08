"""
BaseChatStore - Shared CRUD Implementation

Extracts common user/session/preference/message CRUD logic from
SQLiteChatStore and PostgresChatStore to eliminate code duplication.

Only backend-specific logic (search, health_check, connect, etc.)
needs to be implemented by subclasses.

Author: AGI Demo Project
Version: 3.0.0
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import func

from demo.store.base import IChatStore, StoreOperationError, StoreNotConnectedError
from demo.store.bm25_mixin import BM25Mixin
from demo.store.connection import DemoDBManager, DemoDBConfig
from demo.store.models import DemoUser, DemoSession, DemoMessage, DemoPreference


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseChatStore(IChatStore, BM25Mixin):
    """
    Base implementation of IChatStore with shared CRUD logic.

    Subclasses only need to implement:
    - connect() / disconnect()
    - health_check() / get_statistics()
    - search_messages() (backend-specific search)
    """

    def __init__(self, config: DemoDBConfig):
        self.config = config
        self.db_manager = DemoDBManager(config)
        self.__init_bm25__()

    # ============ User Management ============

    def create_user(
        self, username: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> DemoUser:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = DemoUser(
                id=str(uuid.uuid4()),
                username=username,
                display_name=display_name or username,
                email=email,
                created_at=_now(),
                last_login_at=_now(),
            )
            if password:
                user.set_password(password)
            session.add(user)
            session.flush()
            session.expunge(user)
            return user

    def get_user(self, user_id: str) -> Optional[DemoUser]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(DemoUser.id == user_id).first()
            if user:
                session.expunge(user)
            return user

    def get_user_by_username(self, username: str) -> Optional[DemoUser]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(
                DemoUser.username == username
            ).first()
            if user:
                session.expunge(user)
            return user

    def get_or_create_user(
        self, username: str,
        display_name: Optional[str] = None,
    ) -> Tuple[DemoUser, bool]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(
                DemoUser.username == username
            ).first()
            if user:
                session.expunge(user)
                return user, False

            user = DemoUser(
                id=str(uuid.uuid4()),
                username=username,
                display_name=display_name or username,
                created_at=_now(),
                last_login_at=_now(),
            )
            session.add(user)
            session.flush()
            session.expunge(user)
            return user, True

    def get_user_by_email(self, email: str) -> Optional[DemoUser]:
        """通过邮箱获取用户"""
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(
                DemoUser.email == email,
                DemoUser.is_active == True,
            ).first()
            if user:
                session.expunge(user)
            return user

    def update_user(self, user_id: str, **kwargs) -> Optional[DemoUser]:
        """
        更新用户信息
        
        支持字段: display_name, email, avatar, password (自动哈希)
        """
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(DemoUser.id == user_id).first()
            if not user:
                return None
            
            # 特殊处理 password 字段
            password = kwargs.pop('password', None)
            if password:
                user.set_password(password)
            
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            session.flush()
            session.expunge(user)
            return user

    def update_user_login(self, user_id: str) -> None:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            user = session.query(DemoUser).filter(DemoUser.id == user_id).first()
            if user:
                user.last_login_at = _now()

    def list_users(self, limit: int = 100) -> List[DemoUser]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            users = session.query(DemoUser).limit(limit).all()
            for u in users:
                session.expunge(u)
            return users

    # ============ Session Management ============

    def create_session(
        self, user_id: str,
        title: str = "New Chat",
        session_id: Optional[str] = None,
    ) -> DemoSession:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            demo_session = DemoSession(
                id=session_id or str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                created_at=_now(),
                updated_at=_now(),
                is_active=True,
            )
            session.add(demo_session)
            session.flush()
            session.expunge(demo_session)
            return demo_session

    def get_session(self, session_id: str) -> Optional[DemoSession]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            demo_session = session.query(DemoSession).filter(
                DemoSession.id == session_id
            ).first()
            if demo_session:
                session.expunge(demo_session)
            return demo_session

    def list_sessions(
        self, user_id: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> List[DemoSession]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            query = session.query(DemoSession).filter(
                DemoSession.user_id == user_id
            )
            if active_only:
                query = query.filter(DemoSession.is_active == True)
            sessions = query.order_by(
                DemoSession.updated_at.desc()
            ).limit(limit).all()
            for s in sessions:
                session.expunge(s)
            return sessions

    def update_session(self, session_id: str, **kwargs) -> Optional[DemoSession]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            demo_session = session.query(DemoSession).filter(
                DemoSession.id == session_id
            ).first()
            if not demo_session:
                return None
            for key, value in kwargs.items():
                if hasattr(demo_session, key):
                    setattr(demo_session, key, value)
            demo_session.updated_at = _now()
            session.flush()
            session.expunge(demo_session)
            return demo_session

    def delete_session(self, session_id: str) -> bool:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            demo_session = session.query(DemoSession).filter(
                DemoSession.id == session_id
            ).first()
            if not demo_session:
                return False
            demo_session.is_active = False
            demo_session.updated_at = _now()
            return True

    # ============ Message Management ============

    def add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            msg = DemoMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                created_at=_now(),
            )
            if metadata:
                msg.set_metadata(metadata)
            session.add(msg)
            session.flush()

            # Update session timestamp
            demo_session = session.query(DemoSession).filter(
                DemoSession.id == session_id
            ).first()
            if demo_session:
                demo_session.updated_at = _now()

            session.expunge(msg)
            return msg

    def get_messages(
        self, session_id: str,
        limit: int = 100,
    ) -> List[DemoMessage]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            messages = session.query(DemoMessage).filter(
                DemoMessage.session_id == session_id
            ).order_by(
                DemoMessage.created_at.asc()
            ).limit(limit).all()
            for m in messages:
                session.expunge(m)
            return messages

    def get_message_count(self, session_id: str) -> int:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            return session.query(func.count(DemoMessage.id)).filter(
                DemoMessage.session_id == session_id
            ).scalar() or 0

    # ============ Preference Management ============

    def add_preference(
        self, user_id: str,
        preference_text: str,
        preference_type: str = "general",
        priority: int = 5,
        category: Optional[str] = None,
    ) -> DemoPreference:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            pref = DemoPreference(
                id=str(uuid.uuid4()),
                user_id=user_id,
                preference_text=preference_text,
                preference_type=preference_type,
                priority=priority,
                category=category,
                is_active=True,
                created_at=_now(),
                updated_at=_now(),
            )
            session.add(pref)
            session.flush()
            session.expunge(pref)
            return pref

    def get_preferences(
        self, user_id: str,
        preference_types: Optional[List[str]] = None,
        active_only: bool = True,
    ) -> List[DemoPreference]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            query = session.query(DemoPreference).filter(
                DemoPreference.user_id == user_id
            )
            if active_only:
                query = query.filter(DemoPreference.is_active == True)
            if preference_types:
                query = query.filter(
                    DemoPreference.preference_type.in_(preference_types)
                )
            prefs = query.order_by(
                DemoPreference.priority.desc()
            ).all()
            for p in prefs:
                session.expunge(p)
            return prefs

    def get_preference(self, preference_id: str) -> Optional[DemoPreference]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            pref = session.query(DemoPreference).filter(
                DemoPreference.id == preference_id
            ).first()
            if pref:
                session.expunge(pref)
            return pref

    def update_preference(self, preference_id: str, **kwargs) -> Optional[DemoPreference]:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            pref = session.query(DemoPreference).filter(
                DemoPreference.id == preference_id
            ).first()
            if not pref:
                return None
            for key, value in kwargs.items():
                if hasattr(pref, key):
                    setattr(pref, key, value)
            pref.updated_at = _now()
            session.flush()
            session.expunge(pref)
            return pref

    def delete_preference(self, preference_id: str) -> bool:
        self._check_connected()
        with self.db_manager.session_scope() as session:
            pref = session.query(DemoPreference).filter(
                DemoPreference.id == preference_id
            ).first()
            if not pref:
                return False
            pref.is_active = False
            pref.updated_at = _now()
            return True

    # ============ Internal Helpers ============

    def _check_connected(self) -> None:
        """Check if database is connected"""
        if not self.db_manager.is_connected:
            raise StoreNotConnectedError(
                "Store not connected. Call connect() first."
            )
