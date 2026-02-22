"""
Example Adapter - 示例适配器

用于 DKI 示例 Chat UI 的内存适配器
演示如何实现 IUserDataAdapter 接口

注意:
- 这是一个示例实现，仅用于演示和测试
- 生产环境应使用 PostgreSQL/MySQL 等适配器连接真实数据库
- 数据存储在内存中，重启后丢失

Author: AGI Demo Project
Version: 2.0.0
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    UserPreference,
    ChatMessage,
)


@dataclass
class ExampleDataStore:
    """
    示例数据存储
    
    模拟上层应用的数据库
    """
    # 用户表
    users: Dict[str, UserProfile] = field(default_factory=dict)
    
    # 偏好表
    preferences: Dict[str, List[UserPreference]] = field(default_factory=dict)
    
    # 消息表
    messages: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    
    # 会话表
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ExampleAdapter(IUserDataAdapter):
    """
    示例适配器
    
    用于 DKI 示例 Chat UI，演示如何实现 IUserDataAdapter 接口
    
    使用方式:
    ```python
    # 创建适配器
    adapter = ExampleAdapter()
    await adapter.connect()
    
    # 添加示例数据 (模拟上层应用写入)
    adapter.add_user("user_001", "张三")
    adapter.add_preference("user_001", "dietary", "素食主义者，不吃辣")
    adapter.add_message("session_001", "user_001", "user", "推荐一家餐厅")
    
    # DKI 读取数据
    preferences = await adapter.get_user_preferences("user_001")
    history = await adapter.search_relevant_history("user_001", "餐厅")
    ```
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config or AdapterConfig(adapter_type=AdapterType.MEMORY))
        
        # 内存数据存储 (模拟上层应用的数据库)
        self._store = ExampleDataStore()
    
    async def connect(self) -> None:
        """连接 (内存适配器无需实际连接)"""
        self._connected = True
        logger.info("ExampleAdapter connected (in-memory)")
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        logger.info("ExampleAdapter disconnected")
    
    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected
    
    # ============ 读取接口 (DKI 调用) ============
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self._store.users.get(user_id)
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """获取用户偏好"""
        preferences = self._store.preferences.get(user_id, [])
        
        # 过滤
        result = []
        for pref in preferences:
            if not pref.is_active:
                continue
            if not include_expired and pref.is_expired():
                continue
            if preference_types and pref.preference_type not in preference_types:
                continue
            result.append(pref)
        
        # 按优先级排序
        result.sort(key=lambda p: p.priority, reverse=True)
        return result
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """获取会话历史"""
        messages = self._store.messages.get(session_id, [])
        
        # 过滤
        result = []
        for msg in messages:
            if before and msg.timestamp >= before:
                continue
            if after and msg.timestamp <= after:
                continue
            result.append(msg)
        
        # 按时间排序，取最近的
        result.sort(key=lambda m: m.timestamp)
        return result[-limit:]
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        检索相关历史
        
        简单实现: 关键词匹配
        生产环境应使用向量检索
        """
        # 收集所有消息
        all_messages = []
        
        if session_id:
            # 仅搜索指定会话
            all_messages = self._store.messages.get(session_id, [])
        else:
            # 搜索用户的所有会话
            for sid, messages in self._store.messages.items():
                for msg in messages:
                    if msg.user_id == user_id:
                        all_messages.append(msg)
        
        # 简单关键词匹配
        query_words = set(query.lower().split())
        scored_messages = []
        
        for msg in all_messages:
            content_words = set(msg.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_messages.append((overlap, msg))
        
        # 按相关性排序
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        return [msg for _, msg in scored_messages[:limit]]
    
    # ============ 写入接口 (上层应用调用，非 DKI) ============
    
    def add_user(
        self,
        user_id: str,
        username: str,
        display_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        添加用户 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        user = UserProfile(
            user_id=user_id,
            username=username,
            display_name=display_name or username,
            preferences=preferences or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._store.users[user_id] = user
        return user
    
    def add_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_text: str,
        priority: int = 0,
        category: Optional[str] = None,
    ) -> UserPreference:
        """
        添加用户偏好 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        pref = UserPreference(
            user_id=user_id,
            preference_id=uuid.uuid4().hex[:8],
            preference_type=preference_type,
            preference_text=preference_text,
            priority=priority,
            category=category,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        if user_id not in self._store.preferences:
            self._store.preferences[user_id] = []
        self._store.preferences[user_id].append(pref)
        
        return pref
    
    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        添加消息 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        msg = ChatMessage(
            message_id=uuid.uuid4().hex[:8],
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        
        if session_id not in self._store.messages:
            self._store.messages[session_id] = []
        self._store.messages[session_id].append(msg)
        
        # 更新会话
        if session_id not in self._store.sessions:
            self._store.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "message_count": 0,
            }
        self._store.sessions[session_id]["message_count"] += 1
        self._store.sessions[session_id]["updated_at"] = datetime.utcnow()
        
        return msg
    
    def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建会话 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        sid = session_id or uuid.uuid4().hex[:8]
        session = {
            "session_id": sid,
            "user_id": user_id,
            "title": title or f"会话 {sid}",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0,
            "is_active": True,
        }
        self._store.sessions[sid] = session
        self._store.messages[sid] = []
        return session
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """获取用户会话列表"""
        sessions = []
        for session in self._store.sessions.values():
            if session["user_id"] != user_id:
                continue
            if active_only and not session.get("is_active", True):
                continue
            sessions.append(session)
        
        # 按更新时间排序
        sessions.sort(key=lambda s: s.get("updated_at", datetime.min), reverse=True)
        return sessions[:limit]
    
    def update_preference(
        self,
        user_id: str,
        preference_id: str,
        preference_text: Optional[str] = None,
        priority: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """
        更新偏好 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        preferences = self._store.preferences.get(user_id, [])
        for pref in preferences:
            if pref.preference_id == preference_id:
                if preference_text is not None:
                    pref.preference_text = preference_text
                if priority is not None:
                    pref.priority = priority
                if is_active is not None:
                    pref.is_active = is_active
                pref.updated_at = datetime.utcnow()
                return True
        return False
    
    def delete_preference(self, user_id: str, preference_id: str) -> bool:
        """
        删除偏好 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        preferences = self._store.preferences.get(user_id, [])
        for i, pref in enumerate(preferences):
            if pref.preference_id == preference_id:
                preferences.pop(i)
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        total_messages = sum(len(msgs) for msgs in self._store.messages.values())
        total_preferences = sum(len(prefs) for prefs in self._store.preferences.values())
        
        return {
            "users_count": len(self._store.users),
            "sessions_count": len(self._store.sessions),
            "messages_count": total_messages,
            "preferences_count": total_preferences,
        }
    
    def clear(self):
        """清空所有数据"""
        self._store = ExampleDataStore()
