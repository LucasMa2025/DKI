"""
External User Data Adapter Interface

外部用户数据适配器接口

职责: 读取上层应用的数据库，获取用户偏好和历史消息
注意: 这是读取外部系统的数据，不是 DKI 自己的数据存储

DKI 作为 LLM 插件，需要从上层应用获取:
1. 用户偏好 (用于 K/V 注入)
2. 历史消息 (用于后缀提示词)

上层应用可能是:
- 第三方 Chat 系统
- 企业内部应用
- 任何需要 LLM 增强的系统

Author: AGI Demo Project
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib

from loguru import logger


class AdapterType(str, Enum):
    """支持的适配器类型"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    REST_API = "rest_api"
    SQLITE = "sqlite"
    MEMORY = "memory"  # 仅用于示例/测试
    CUSTOM = "custom"


@dataclass
class AdapterConfig:
    """
    外部数据适配器配置
    
    用于连接上层应用的数据库
    """
    adapter_type: AdapterType = AdapterType.MEMORY
    
    # 数据库连接
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    
    # 连接字符串 (替代单独参数)
    connection_string: Optional[str] = None
    
    # 表名/集合名 (上层应用的表结构)
    schema: str = "public"
    users_table: str = "users"
    messages_table: str = "messages"
    preferences_table: str = "user_preferences"
    sessions_table: str = "sessions"
    
    # REST API 配置
    base_url: str = ""
    api_key: str = ""
    timeout: int = 30
    
    # 连接池
    pool_size: int = 5
    max_overflow: int = 10
    
    # 缓存
    enable_cache: bool = True
    cache_ttl: int = 300  # 5 分钟
    
    # 额外选项
    options: Dict[str, Any] = field(default_factory=dict)
    
    def get_connection_string(self) -> str:
        """构建连接字符串"""
        if self.connection_string:
            return self.connection_string
        
        if self.adapter_type == AdapterType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.adapter_type == AdapterType.MYSQL:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.adapter_type == AdapterType.MONGODB:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.adapter_type == AdapterType.REDIS:
            if self.password:
                return f"redis://:{self.password}@{self.host}:{self.port}/0"
            return f"redis://{self.host}:{self.port}/0"
        elif self.adapter_type == AdapterType.SQLITE:
            return f"sqlite:///{self.database}"
        else:
            return ""


@dataclass
class UserProfile:
    """
    用户画像数据结构
    
    来源: 上层应用的 users 表
    """
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    
    # 用户偏好 (结构化数据)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # 用户设置 (UI 偏好、语言等)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    
    # 状态
    is_active: bool = True
    
    def get_preference_text(self) -> str:
        """将偏好转换为文本格式 (用于 DKI 注入)"""
        if not self.preferences:
            return ""
        
        lines = []
        for key, value in self.preferences.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "email": self.email,
            "preferences": self.preferences,
            "settings": self.settings,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        for field_name in ["created_at", "updated_at", "last_active_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ChatMessage:
    """
    聊天消息数据结构
    
    来源: 上层应用的 messages 表
    """
    message_id: str
    session_id: str
    user_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    
    # 可选的 embedding (用于向量检索)
    embedding: Optional[List[float]] = None
    
    # 消息元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Token 数量 (如果可用)
    token_count: Optional[int] = None
    
    # 父消息 ID (用于线程)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "parent_id": self.parent_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def content_hash(self) -> str:
        """获取内容哈希 (用于缓存)"""
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class UserPreference:
    """
    用户偏好数据结构
    
    来源: 上层应用的 user_preferences 表
    用途: DKI K/V 注入
    """
    user_id: str
    preference_text: str
    preference_type: str  # "dietary" | "communication" | "interests" | "custom"
    
    # 偏好 ID
    preference_id: Optional[str] = None
    
    # 优先级 (越高越重要)
    priority: int = 0
    
    # 分类
    category: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # 过期时间 (可选)
    expires_at: Optional[datetime] = None
    
    # 状态
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """检查偏好是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preference_id": self.preference_id,
            "preference_text": self.preference_text,
            "preference_type": self.preference_type,
            "priority": self.priority,
            "category": self.category,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        for field_name in ["created_at", "updated_at", "expires_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class IUserDataAdapter(ABC):
    """
    外部用户数据适配器接口
    
    职责: 读取上层应用的数据库，获取用户偏好和历史消息
    
    注意:
    - 这是读取外部系统的数据，不是 DKI 自己的数据存储
    - 上层应用负责数据的写入和管理
    - DKI 只读取数据用于注入
    
    实现示例:
    ```python
    class MyAppAdapter(IUserDataAdapter):
        '''连接我的应用数据库'''
        
        async def get_user_preferences(self, user_id: str) -> List[UserPreference]:
            # 从我的应用的 user_preferences 表读取
            rows = await self.db.fetch(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user_id
            )
            return [UserPreference.from_dict(row) for row in rows]
    ```
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """
        初始化适配器
        
        Args:
            config: 适配器配置 (连接上层应用的数据库)
        """
        self.config = config or AdapterConfig()
        self._connected = False
        self._connection = None
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> None:
        """
        建立与上层应用数据库的连接
        
        在使用适配器前必须调用
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        关闭与上层应用数据库的连接
        """
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        从上层应用的数据库获取用户画像
        
        Args:
            user_id: 用户标识
            
        Returns:
            UserProfile 或 None
        """
        pass
    
    @abstractmethod
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """
        从上层应用的数据库获取用户偏好
        
        这是 DKI K/V 注入的数据来源
        
        Args:
            user_id: 用户标识
            preference_types: 过滤偏好类型
            include_expired: 是否包含过期偏好
            
        Returns:
            用户偏好列表 (按优先级排序)
        """
        pass
    
    @abstractmethod
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """
        从上层应用的数据库获取会话历史
        
        Args:
            session_id: 会话标识
            limit: 最大消息数
            before: 获取此时间之前的消息
            after: 获取此时间之后的消息
            
        Returns:
            消息列表 (按时间顺序)
        """
        pass
    
    @abstractmethod
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        从上层应用的数据库检索与查询相关的历史消息
        
        这是 DKI 历史后缀注入的数据来源
        
        Args:
            user_id: 用户标识
            query: 搜索查询
            limit: 最大结果数
            session_id: 可选的会话过滤
            
        Returns:
            相关消息列表 (按相关性排序)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        检查与上层应用数据库的连接是否健康
        
        Returns:
            True 如果健康
        """
        pass
    
    # ============ 可选方法 (带默认实现) ============
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        获取用户的会话列表
        
        Args:
            user_id: 用户标识
            limit: 最大会话数
            active_only: 仅返回活跃会话
            
        Returns:
            会话信息列表
        """
        logger.warning(f"{self.__class__.__name__} 未实现 get_user_sessions")
        return []
    
    async def get_preference_text(
        self,
        user_id: str,
        max_tokens: int = 100,
    ) -> str:
        """
        获取格式化的偏好文本 (用于 DKI 注入)
        
        Args:
            user_id: 用户标识
            max_tokens: 最大 token 数 (近似)
            
        Returns:
            格式化的偏好文本
        """
        preferences = await self.get_user_preferences(user_id)
        if not preferences:
            return ""
        
        # 按优先级排序并格式化
        preferences.sort(key=lambda p: p.priority, reverse=True)
        
        lines = []
        estimated_tokens = 0
        
        for pref in preferences:
            if pref.is_expired():
                continue
            
            line = f"- {pref.preference_type}: {pref.preference_text}"
            line_tokens = len(line.split()) * 1.3  # 粗略估算
            
            if estimated_tokens + line_tokens > max_tokens:
                break
            
            lines.append(line)
            estimated_tokens += line_tokens
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.config.adapter_type}, connected={self._connected})"
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
