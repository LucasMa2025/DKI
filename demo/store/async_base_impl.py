"""
AsyncBaseChatStore - Async Shared CRUD Implementation

Async counterpart of BaseChatStore (base_impl.py).
Uses sqlalchemy.ext.asyncio for non-blocking database operations.

Only PostgreSQL backends use async; SQLite remains synchronous.

Author: AGI Demo Project
Version: 3.1.0
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import func, select

from demo.store.base import IChatStore, StoreOperationError, StoreNotConnectedError
from demo.store.bm25_mixin import BM25Mixin
from demo.store.connection import AsyncDemoDBManager, DemoDBConfig
from demo.store.models import DemoUser, DemoSession, DemoMessage, DemoPreference


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AsyncBaseChatStore(IChatStore, BM25Mixin):
    """
    Async base implementation of IChatStore with shared CRUD logic.

    Uses AsyncDemoDBManager + sqlalchemy.ext.asyncio.
    Subclasses only need to implement:
    - connect() / disconnect()
    - health_check() / get_statistics()
    - search_messages() (backend-specific search)
    """

    def __init__(self, config: DemoDBConfig):
        self.config = config
        self.db_manager = AsyncDemoDBManager(config)
        self.__init_bm25__()

    # ============ Sync wrappers for IChatStore interface ============
    # IChatStore defines sync methods. We provide both sync (raise) and async versions.
    # The API layer should call the async versions directly.

    def _sync_not_supported(self, method_name: str):
        raise StoreOperationError(
            f"Sync method '{method_name}' not supported on async store. "
            f"Use 'a_{method_name}()' instead."
        )

    # ============ User Management (async) ============

    async def a_create_user(
        self, username: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> DemoUser:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            user = DemoUser(
                id=str(uuid.uuid4()),
                username=username,
                display_name=display_name or username,
                email=email,
                created_at=_now(),
                last_login_at=_now(),
            )
            session.add(user)
            await session.flush()
            return user

    async def a_get_user(self, user_id: str) -> Optional[DemoUser]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoUser).filter(DemoUser.id == user_id)
            )
            return result.scalars().first()

    async def a_get_user_by_username(self, username: str) -> Optional[DemoUser]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoUser).filter(DemoUser.username == username)
            )
            return result.scalars().first()

    async def a_get_or_create_user(
        self, username: str,
        display_name: Optional[str] = None,
    ) -> Tuple[DemoUser, bool]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoUser).filter(DemoUser.username == username)
            )
            user = result.scalars().first()
            if user:
                return user, False

            user = DemoUser(
                id=str(uuid.uuid4()),
                username=username,
                display_name=display_name or username,
                created_at=_now(),
                last_login_at=_now(),
            )
            session.add(user)
            await session.flush()
            return user, True

    async def a_update_user_login(self, user_id: str) -> None:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoUser).filter(DemoUser.id == user_id)
            )
            user = result.scalars().first()
            if user:
                user.last_login_at = _now()

    async def a_list_users(self, limit: int = 100) -> List[DemoUser]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoUser).limit(limit)
            )
            return list(result.scalars().all())

    # ============ Session Management (async) ============

    async def a_create_session(
        self, user_id: str,
        title: str = "New Chat",
        session_id: Optional[str] = None,
    ) -> DemoSession:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            demo_session = DemoSession(
                id=session_id or str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                created_at=_now(),
                updated_at=_now(),
                is_active=True,
            )
            session.add(demo_session)
            await session.flush()
            return demo_session

    async def a_get_session(self, session_id: str) -> Optional[DemoSession]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoSession).filter(DemoSession.id == session_id)
            )
            return result.scalars().first()

    async def a_list_sessions(
        self, user_id: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> List[DemoSession]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            stmt = select(DemoSession).filter(DemoSession.user_id == user_id)
            if active_only:
                stmt = stmt.filter(DemoSession.is_active == True)
            stmt = stmt.order_by(DemoSession.updated_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def a_update_session(self, session_id: str, **kwargs) -> Optional[DemoSession]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoSession).filter(DemoSession.id == session_id)
            )
            demo_session = result.scalars().first()
            if not demo_session:
                return None
            for key, value in kwargs.items():
                if hasattr(demo_session, key):
                    setattr(demo_session, key, value)
            demo_session.updated_at = _now()
            await session.flush()
            return demo_session

    async def a_delete_session(self, session_id: str) -> bool:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoSession).filter(DemoSession.id == session_id)
            )
            demo_session = result.scalars().first()
            if not demo_session:
                return False
            demo_session.is_active = False
            demo_session.updated_at = _now()
            return True

    # ============ Message Management (async) ============

    async def a_add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
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
            await session.flush()

            # Update session timestamp
            result = await session.execute(
                select(DemoSession).filter(DemoSession.id == session_id)
            )
            demo_session = result.scalars().first()
            if demo_session:
                demo_session.updated_at = _now()

            return msg

    async def a_get_messages(
        self, session_id: str,
        limit: int = 100,
    ) -> List[DemoMessage]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoMessage)
                .filter(DemoMessage.session_id == session_id)
                .order_by(DemoMessage.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def a_get_message_count(self, session_id: str) -> int:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(func.count(DemoMessage.id))
                .filter(DemoMessage.session_id == session_id)
            )
            return result.scalar() or 0

    # ============ Preference Management (async) ============

    async def a_add_preference(
        self, user_id: str,
        preference_text: str,
        preference_type: str = "general",
        priority: int = 5,
        category: Optional[str] = None,
    ) -> DemoPreference:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
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
            await session.flush()
            return pref

    async def a_get_preferences(
        self, user_id: str,
        preference_types: Optional[List[str]] = None,
        active_only: bool = True,
    ) -> List[DemoPreference]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            stmt = select(DemoPreference).filter(DemoPreference.user_id == user_id)
            if active_only:
                stmt = stmt.filter(DemoPreference.is_active == True)
            if preference_types:
                stmt = stmt.filter(DemoPreference.preference_type.in_(preference_types))
            stmt = stmt.order_by(DemoPreference.priority.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def a_get_preference(self, preference_id: str) -> Optional[DemoPreference]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoPreference).filter(DemoPreference.id == preference_id)
            )
            return result.scalars().first()

    async def a_update_preference(self, preference_id: str, **kwargs) -> Optional[DemoPreference]:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoPreference).filter(DemoPreference.id == preference_id)
            )
            pref = result.scalars().first()
            if not pref:
                return None
            for key, value in kwargs.items():
                if hasattr(pref, key):
                    setattr(pref, key, value)
            pref.updated_at = _now()
            await session.flush()
            return pref

    async def a_delete_preference(self, preference_id: str) -> bool:
        self._check_connected()
        async with self.db_manager.session_scope() as session:
            result = await session.execute(
                select(DemoPreference).filter(DemoPreference.id == preference_id)
            )
            pref = result.scalars().first()
            if not pref:
                return False
            pref.is_active = False
            pref.updated_at = _now()
            return True

    # ============ Sync IChatStore interface (delegates to async) ============
    # These are required by IChatStore ABC. They delegate to async versions.

    def create_user(self, username, display_name=None, email=None):
        self._sync_not_supported("create_user")

    def get_user(self, user_id):
        self._sync_not_supported("get_user")

    def get_user_by_username(self, username):
        self._sync_not_supported("get_user_by_username")

    def get_or_create_user(self, username, display_name=None):
        self._sync_not_supported("get_or_create_user")

    def update_user_login(self, user_id):
        self._sync_not_supported("update_user_login")

    def list_users(self, limit=100):
        self._sync_not_supported("list_users")

    def create_session(self, user_id, title="New Chat", session_id=None):
        self._sync_not_supported("create_session")

    def get_session(self, session_id):
        self._sync_not_supported("get_session")

    def list_sessions(self, user_id, limit=50, active_only=True):
        self._sync_not_supported("list_sessions")

    def update_session(self, session_id, **kwargs):
        self._sync_not_supported("update_session")

    def delete_session(self, session_id):
        self._sync_not_supported("delete_session")

    def add_message(self, session_id, user_id, role, content, metadata=None):
        self._sync_not_supported("add_message")

    def get_messages(self, session_id, limit=100):
        self._sync_not_supported("get_messages")

    def get_message_count(self, session_id):
        self._sync_not_supported("get_message_count")

    def search_messages(self, user_id, query, limit=5, session_id=None):
        self._sync_not_supported("search_messages")

    def add_preference(self, user_id, preference_text, preference_type="general",
                       priority=5, category=None):
        self._sync_not_supported("add_preference")

    def get_preferences(self, user_id, preference_types=None, active_only=True):
        self._sync_not_supported("get_preferences")

    def get_preference(self, preference_id):
        self._sync_not_supported("get_preference")

    def update_preference(self, preference_id, **kwargs):
        self._sync_not_supported("update_preference")

    def delete_preference(self, preference_id):
        self._sync_not_supported("delete_preference")

    # ============ Internal Helpers ============

    def _check_connected(self) -> None:
        """Check if database is connected."""
        if not self.db_manager.is_connected:
            raise StoreNotConnectedError(
                "Async store not connected. Call connect() first."
            )
