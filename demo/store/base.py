"""
IChatStore — Demo 持久化层抽象接口

参考 AGA persistence 层设计:
- 完善的异常类体系 (StoreError, ConnectionError, etc.)
- health_check 返回 Dict (包含详细状态信息)
- 统计接口 (get_statistics)
- 连接管理生命周期 (connect/disconnect/is_connected)

设计要点:
1. 与 IUserDataAdapter 的读取接口对齐 (方便未来 DKI Plugin 集成)
2. 增加写入方法 (IUserDataAdapter 是只读的)
3. 支持多种后端 (SQLite / PostgreSQL / pgvector)

未来规划:
  DKI Plugin 可通过配置启用消息管理功能时,
  其内部的持久化接口将与 IChatStore 对齐,
  上层应用只需提供配置即可完成消息管理的委托。

接口对齐关系:
  IChatStore.add_message()     ↔ 未来 DKIPlugin.add_message()
  IChatStore.get_messages()    ↔ IUserDataAdapter.get_session_history()
  IChatStore.search_messages() ↔ IUserDataAdapter.search_relevant_history()
  IChatStore.add_preference()  ↔ 未来 DKIPlugin.add_preference()
  IChatStore.get_preferences() ↔ IUserDataAdapter.get_user_preferences()

Author: AGI Demo Project
Version: 3.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from demo.store.models import DemoUser, DemoSession, DemoMessage, DemoPreference


# ==================== 异常类体系 (参考 AGA persistence.base) ====================

class StoreError(Exception):
    """持久化层基础异常"""
    pass


class StoreConnectionError(StoreError):
    """连接错误 (数据库不可达、认证失败等)"""
    pass


class StoreOperationError(StoreError):
    """操作错误 (查询失败、约束冲突等)"""
    pass


class StoreNotConnectedError(StoreError):
    """未连接错误 (在连接建立前调用操作)"""
    pass


# ==================== 抽象基类 ====================

class IChatStore(ABC):
    """
    Demo 持久化层抽象接口
    
    参考 AGA PersistenceAdapter 设计:
    - 统一的生命周期管理 (connect/disconnect/is_connected)
    - 详细的健康检查 (health_check 返回 Dict)
    - 统计接口 (get_statistics)
    """
    
    # ============ 用户管理 ============
    
    @abstractmethod
    def create_user(
        self, username: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> DemoUser:
        """创建用户"""
        ...
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[DemoUser]:
        """获取用户"""
        ...
    
    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[DemoUser]:
        """通过用户名获取用户"""
        ...
    
    def get_user_by_email(self, email: str) -> Optional[DemoUser]:
        """通过邮箱获取用户 (默认实现: 不支持)"""
        return None
    
    @abstractmethod
    def get_or_create_user(
        self, username: str,
        display_name: Optional[str] = None,
    ) -> Tuple[DemoUser, bool]:
        """获取或创建用户, 返回 (user, created)"""
        ...
    
    def update_user(self, user_id: str, **kwargs) -> Optional[DemoUser]:
        """更新用户信息 (display_name, email, avatar, password 等)"""
        return None
    
    @abstractmethod
    def update_user_login(self, user_id: str) -> None:
        """更新用户登录时间"""
        ...
    
    @abstractmethod
    def list_users(self, limit: int = 100) -> List[DemoUser]:
        """列出所有用户"""
        ...
    
    # ============ 会话管理 ============
    
    @abstractmethod
    def create_session(
        self, user_id: str,
        title: str = "New Chat",
        session_id: Optional[str] = None,
    ) -> DemoSession:
        """创建会话"""
        ...
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[DemoSession]:
        """获取会话"""
        ...
    
    @abstractmethod
    def list_sessions(
        self, user_id: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> List[DemoSession]:
        """列出用户会话"""
        ...
    
    @abstractmethod
    def update_session(self, session_id: str, **kwargs) -> Optional[DemoSession]:
        """更新会话 (title, metadata 等)"""
        ...
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """软删除会话"""
        ...
    
    # ============ 消息管理 (读+写) ============
    
    @abstractmethod
    def add_message(
        self, session_id: str, user_id: str,
        role: str, content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemoMessage:
        """
        添加消息
        
        ★ 与未来 DKI Plugin.add_message() 对齐
        """
        ...
    
    @abstractmethod
    def get_messages(
        self, session_id: str,
        limit: int = 100,
    ) -> List[DemoMessage]:
        """
        获取会话消息 (按时间正序)
        
        ★ 与 IUserDataAdapter.get_session_history() 对齐
        """
        ...
    
    @abstractmethod
    def get_message_count(self, session_id: str) -> int:
        """获取会话消息数量"""
        ...
    
    @abstractmethod
    def search_messages(
        self, user_id: str, query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[DemoMessage]:
        """
        检索相关消息 (BM25)
        
        ★ 与 IUserDataAdapter.search_relevant_history() 对齐
        """
        ...
    
    # ============ 偏好管理 (读+写) ============
    
    @abstractmethod
    def add_preference(
        self, user_id: str,
        preference_text: str,
        preference_type: str = "general",
        priority: int = 5,
        category: Optional[str] = None,
    ) -> DemoPreference:
        """
        添加偏好
        
        ★ 与未来 DKI Plugin.add_preference() 对齐
        """
        ...
    
    @abstractmethod
    def get_preferences(
        self, user_id: str,
        preference_types: Optional[List[str]] = None,
        active_only: bool = True,
    ) -> List[DemoPreference]:
        """
        获取用户偏好
        
        ★ 与 IUserDataAdapter.get_user_preferences() 对齐
        """
        ...
    
    @abstractmethod
    def get_preference(self, preference_id: str) -> Optional[DemoPreference]:
        """获取单个偏好"""
        ...
    
    @abstractmethod
    def update_preference(self, preference_id: str, **kwargs) -> Optional[DemoPreference]:
        """更新偏好"""
        ...
    
    @abstractmethod
    def delete_preference(self, preference_id: str) -> bool:
        """软删除偏好"""
        ...
    
    # ============ 生命周期 (参考 AGA PersistenceAdapter) ============
    
    @abstractmethod
    def connect(self) -> None:
        """初始化数据库连接"""
        ...
    
    @abstractmethod
    def disconnect(self) -> None:
        """关闭数据库连接"""
        ...
    
    def is_connected(self) -> bool:
        """
        检查连接状态
        
        默认实现: 尝试 health_check
        子类可覆盖以提供更高效的实现
        """
        try:
            result = self.health_check()
            return result.get("status") == "healthy"
        except Exception:
            return False
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查 (参考 AGA PersistenceAdapter.health_check)
        
        Returns:
            健康状态信息字典, 至少包含:
            - status: "healthy" | "degraded" | "disconnected"
            - adapter: 适配器类型名称
            - 其他适配器特定信息
        """
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息 (参考 AGA PersistenceAdapter.get_statistics)
        
        Returns:
            统计信息字典, 包含:
            - adapter: 适配器类型
            - user_count: 用户数量
            - session_count: 会话数量
            - message_count: 消息数量
            - preference_count: 偏好数量
        
        默认实现: 返回基本信息, 子类可覆盖
        """
        return {
            "adapter": self.__class__.__name__,
            "status": "unknown",
        }