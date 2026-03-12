"""
IChatStore — 实验系统持久化层抽象接口

从 demo/store/base.py 复制, 修改导入路径。

Author: AGI Demo Project
Version: 1.0.0 (forked from demo/store/base.py 3.0.0)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from dki.experiment.store.models import DemoUser, DemoSession, DemoMessage, DemoPreference


# ==================== 异常类体系 ====================

class StoreError(Exception):
    """持久化层基础异常"""
    pass


class StoreConnectionError(StoreError):
    """连接错误"""
    pass


class StoreOperationError(StoreError):
    """操作错误"""
    pass


class StoreNotConnectedError(StoreError):
    """未连接错误"""
    pass


# ==================== 抽象基类 ====================

class IChatStore(ABC):
    """实验系统持久化层抽象接口"""

    # ============ 用户管理 ============

    @abstractmethod
    def create_user(
        self, username: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> DemoUser:
        ...

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[DemoUser]:
        ...

    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[DemoUser]:
        ...

    def get_user_by_email(self, email: str) -> Optional[DemoUser]:
        return None

    @abstractmethod
    def get_or_create_user(
        self, username: str,
        display_name: Optional[str] = None,
    ) -> Tuple[DemoUser, bool]:
        ...

    def update_user(self, user_id: str, **kwargs) -> Optional[DemoUser]:
        return None

    @abstractmethod
    def update_user_login(self, user_id: str) -> None:
        ...

    @abstractmethod
    def list_users(self, limit: int = 100) -> List[DemoUser]:
        ...

    # ============ 会话管理 ============

    @abstractmethod
    def create_session(
        self, user_id: str,
        title: str = "New Chat",
        session_id: Optional[str] = None,
    ) -> DemoSession:
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[DemoSession]:
        ...

    @abstractmethod
    def list_sessions(
        self, user_id: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> List[DemoSession]:
        ...

    @abstractmethod
    def update_session(self, session_id: str, **kwargs) -> Optional[DemoSession]:
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        ...

    # ============ 消息管理 ============

    @abstractmethod
    def add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        ...

    @abstractmethod
    def get_messages(
        self, session_id: str,
        limit: int = 100,
    ) -> List[DemoMessage]:
        ...

    @abstractmethod
    def get_message_count(self, session_id: str) -> int:
        ...

    @abstractmethod
    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        ...

    # ============ 偏好管理 ============

    @abstractmethod
    def add_preference(
        self, user_id: str,
        preference_text: str,
        preference_type: str = "general",
        priority: int = 5,
        category: Optional[str] = None,
    ) -> DemoPreference:
        ...

    @abstractmethod
    def get_preferences(
        self, user_id: str,
        preference_types: Optional[List[str]] = None,
        active_only: bool = True,
    ) -> List[DemoPreference]:
        ...

    @abstractmethod
    def get_preference(self, preference_id: str) -> Optional[DemoPreference]:
        ...

    @abstractmethod
    def update_preference(self, preference_id: str, **kwargs) -> Optional[DemoPreference]:
        ...

    @abstractmethod
    def delete_preference(self, preference_id: str) -> bool:
        ...

    # ============ 生命周期 ============

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    def is_connected(self) -> bool:
        try:
            result = self.health_check()
            return result.get("status") == "healthy"
        except Exception:
            return False

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        ...

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "adapter": self.__class__.__name__,
            "status": "unknown",
        }
