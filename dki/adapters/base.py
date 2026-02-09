"""
Base User Data Adapter Interface and Data Models
Defines the standard interface for connecting DKI to external user data sources

Author: AGI Demo Project
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib

from loguru import logger


class AdapterType(str, Enum):
    """Supported adapter types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    REST_API = "rest_api"
    MEMORY = "memory"
    SQLITE = "sqlite"
    CUSTOM = "custom"


@dataclass
class AdapterConfig:
    """
    Configuration for user data adapter.
    
    Supports multiple connection types with unified configuration.
    """
    adapter_type: AdapterType = AdapterType.MEMORY
    
    # Database connection
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    
    # Connection string (alternative to individual params)
    connection_string: Optional[str] = None
    
    # Schema/Collection names
    schema: str = "public"
    users_table: str = "users"
    messages_table: str = "messages"
    preferences_table: str = "user_preferences"
    
    # REST API specific
    base_url: str = ""
    api_key: str = ""
    timeout: int = 30
    
    # Connection pool
    pool_size: int = 5
    max_overflow: int = 10
    
    # Cache settings
    enable_cache: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)
    
    def get_connection_string(self) -> str:
        """Build connection string from config."""
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
        else:
            return ""


@dataclass
class UserProfile:
    """
    User profile data structure.
    
    Contains user identity and configuration information.
    """
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    
    # User preferences as structured data
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # User settings (UI preferences, language, etc.)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    
    def get_preference_text(self) -> str:
        """
        Convert preferences to text format for DKI injection.
        
        Returns:
            Formatted preference text
        """
        if not self.preferences:
            return ""
        
        lines = []
        for key, value in self.preferences.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Create from dictionary."""
        # Parse datetime fields
        for field_name in ["created_at", "updated_at", "last_active_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ChatMessage:
    """
    Chat message data structure.
    
    Represents a single message in conversation history.
    """
    message_id: str
    session_id: str
    user_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    
    # Optional embedding for vector search
    embedding: Optional[List[float]] = None
    
    # Message metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Token count (if available)
    token_count: Optional[int] = None
    
    # Parent message ID (for threading)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Create from dictionary."""
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def content_hash(self) -> str:
        """Get hash of message content for caching."""
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class UserPreference:
    """
    User preference data structure.
    
    Represents a specific user preference for DKI injection.
    """
    user_id: str
    preference_text: str
    preference_type: str  # "dietary" | "communication" | "interests" | "custom"
    
    # Preference ID (optional)
    preference_id: Optional[str] = None
    
    # Priority for ordering (higher = more important)
    priority: int = 0
    
    # Category for grouping
    category: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Expiration (optional)
    expires_at: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if preference has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Create from dictionary."""
        for field_name in ["created_at", "updated_at", "expires_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class IUserDataAdapter(ABC):
    """
    User Data Adapter Interface.
    
    All external data sources must implement this interface to integrate with DKI.
    
    This interface provides:
    - User profile retrieval
    - Session history access
    - User preference management
    - Relevant history search
    
    Example Implementation:
        class MyDatabaseAdapter(IUserDataAdapter):
            async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
                # Fetch from your database
                ...
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """
        Initialize adapter with configuration.
        
        Args:
            config: Adapter configuration
        """
        self.config = config or AdapterConfig()
        self._connected = False
        self._connection = None
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to data source.
        
        Should be called before any data operations.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to data source.
        
        Should be called when adapter is no longer needed.
        """
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by user ID.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserProfile if found, None otherwise
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
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            before: Get messages before this timestamp
            after: Get messages after this timestamp
            
        Returns:
            List of messages in chronological order
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
        Get user preferences.
        
        Args:
            user_id: User identifier
            preference_types: Filter by preference types
            include_expired: Include expired preferences
            
        Returns:
            List of user preferences sorted by priority
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
        Search for messages relevant to a query.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
            session_id: Optional session filter
            
        Returns:
            Relevant messages sorted by relevance
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if data source is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    # Optional methods with default implementations
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get user's sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum sessions to return
            active_only: Only return active sessions
            
        Returns:
            List of session info dictionaries
        """
        logger.warning(f"{self.__class__.__name__} does not implement get_user_sessions")
        return []
    
    async def save_message(
        self,
        message: ChatMessage,
    ) -> bool:
        """
        Save a message to the data source.
        
        Args:
            message: Message to save
            
        Returns:
            True if saved successfully
        """
        logger.warning(f"{self.__class__.__name__} does not implement save_message")
        return False
    
    async def update_user_preference(
        self,
        preference: UserPreference,
    ) -> bool:
        """
        Update or create a user preference.
        
        Args:
            preference: Preference to update/create
            
        Returns:
            True if successful
        """
        logger.warning(f"{self.__class__.__name__} does not implement update_user_preference")
        return False
    
    async def delete_user_preference(
        self,
        user_id: str,
        preference_id: str,
    ) -> bool:
        """
        Delete a user preference.
        
        Args:
            user_id: User identifier
            preference_id: Preference identifier
            
        Returns:
            True if deleted successfully
        """
        logger.warning(f"{self.__class__.__name__} does not implement delete_user_preference")
        return False
    
    async def get_preference_text(
        self,
        user_id: str,
        max_tokens: int = 100,
    ) -> str:
        """
        Get formatted preference text for DKI injection.
        
        Args:
            user_id: User identifier
            max_tokens: Maximum tokens (approximate)
            
        Returns:
            Formatted preference text
        """
        preferences = await self.get_user_preferences(user_id)
        if not preferences:
            return ""
        
        # Sort by priority and format
        preferences.sort(key=lambda p: p.priority, reverse=True)
        
        lines = []
        estimated_tokens = 0
        
        for pref in preferences:
            if pref.is_expired():
                continue
            
            line = f"- {pref.preference_type}: {pref.preference_text}"
            line_tokens = len(line.split()) * 1.3  # Rough estimate
            
            if estimated_tokens + line_tokens > max_tokens:
                break
            
            lines.append(line)
            estimated_tokens += line_tokens
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.config.adapter_type}, connected={self._connected})"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
