"""
In-Memory User Data Adapter
Simple in-memory adapter for testing and development

Author: AGI Demo Project
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    ChatMessage,
    UserPreference,
)


class InMemoryUserDataAdapter(IUserDataAdapter):
    """
    In-memory adapter for user data.
    
    Stores all data in memory. Useful for:
    - Testing
    - Development
    - Demo purposes
    - Single-instance deployments
    
    Note: Data is lost when the process terminates.
    
    Example:
        adapter = InMemoryUserDataAdapter()
        await adapter.connect()
        
        # Add some test data
        await adapter.set_user_profile(UserProfile(
            user_id="user_123",
            username="testuser",
            preferences={"dietary": "vegetarian"},
        ))
        
        profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize in-memory adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.MEMORY:
            self.config.adapter_type = AdapterType.MEMORY
        
        # In-memory storage
        self._users: Dict[str, UserProfile] = {}
        self._messages: Dict[str, List[ChatMessage]] = {}  # session_id -> messages
        self._preferences: Dict[str, List[UserPreference]] = {}  # user_id -> preferences
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
    
    async def connect(self) -> None:
        """Initialize in-memory storage (no-op)."""
        self._connected = True
        logger.info("In-memory adapter initialized")
    
    async def disconnect(self) -> None:
        """Clear in-memory storage."""
        self._users.clear()
        self._messages.clear()
        self._preferences.clear()
        self._user_sessions.clear()
        self._connected = False
        logger.info("In-memory adapter cleared")
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from memory."""
        return self._users.get(user_id)
    
    async def set_user_profile(self, profile: UserProfile) -> bool:
        """Set user profile in memory."""
        self._users[profile.user_id] = profile
        return True
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """Get session conversation history from memory."""
        messages = self._messages.get(session_id, [])
        
        # Apply filters
        filtered = []
        for msg in messages:
            if before and msg.timestamp and msg.timestamp >= before:
                continue
            if after and msg.timestamp and msg.timestamp <= after:
                continue
            filtered.append(msg)
        
        # Sort by timestamp and limit
        filtered.sort(key=lambda m: m.timestamp or datetime.min)
        return filtered[-limit:]
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """Get user preferences from memory."""
        preferences = self._preferences.get(user_id, [])
        
        now = datetime.utcnow()
        filtered = []
        
        for pref in preferences:
            # Filter by type
            if preference_types and pref.preference_type not in preference_types:
                continue
            
            # Filter expired
            if not include_expired and pref.expires_at and pref.expires_at < now:
                continue
            
            # Filter inactive
            if not pref.is_active:
                continue
            
            filtered.append(pref)
        
        # Sort by priority
        filtered.sort(key=lambda p: p.priority, reverse=True)
        return filtered
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Search for relevant messages in memory."""
        query_lower = query.lower()
        all_messages = []
        
        if session_id:
            # Search in specific session
            all_messages = self._messages.get(session_id, [])
        else:
            # Search in all user sessions
            session_ids = self._user_sessions.get(user_id, [])
            for sid in session_ids:
                messages = self._messages.get(sid, [])
                all_messages.extend([m for m in messages if m.user_id == user_id])
        
        # Simple relevance scoring
        scored = []
        for msg in all_messages:
            content_lower = msg.content.lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                scored.append((msg, score))
        
        # Sort by score and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, _ in scored[:limit]]
    
    async def health_check(self) -> bool:
        """Check adapter health (always healthy for in-memory)."""
        return self._connected
    
    async def save_message(self, message: ChatMessage) -> bool:
        """Save a message to memory."""
        if message.session_id not in self._messages:
            self._messages[message.session_id] = []
        
        # Check if message already exists
        existing_idx = None
        for i, msg in enumerate(self._messages[message.session_id]):
            if msg.message_id == message.message_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self._messages[message.session_id][existing_idx] = message
        else:
            self._messages[message.session_id].append(message)
        
        # Track session for user
        if message.user_id not in self._user_sessions:
            self._user_sessions[message.user_id] = []
        if message.session_id not in self._user_sessions[message.user_id]:
            self._user_sessions[message.user_id].append(message.session_id)
        
        return True
    
    async def update_user_preference(self, preference: UserPreference) -> bool:
        """Update or create a user preference in memory."""
        if not preference.preference_id:
            preference.preference_id = f"pref_{uuid.uuid4().hex[:16]}"
        
        if preference.user_id not in self._preferences:
            self._preferences[preference.user_id] = []
        
        # Check if preference already exists
        existing_idx = None
        for i, pref in enumerate(self._preferences[preference.user_id]):
            if pref.preference_id == preference.preference_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self._preferences[preference.user_id][existing_idx] = preference
        else:
            self._preferences[preference.user_id].append(preference)
        
        return True
    
    async def delete_user_preference(
        self,
        user_id: str,
        preference_id: str,
    ) -> bool:
        """Delete a user preference from memory."""
        if user_id not in self._preferences:
            return False
        
        original_len = len(self._preferences[user_id])
        self._preferences[user_id] = [
            p for p in self._preferences[user_id]
            if p.preference_id != preference_id
        ]
        
        return len(self._preferences[user_id]) < original_len
    
    # Additional utility methods for testing
    
    def clear_all(self) -> None:
        """Clear all data."""
        self._users.clear()
        self._messages.clear()
        self._preferences.clear()
        self._user_sessions.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "users": len(self._users),
            "sessions": len(self._messages),
            "total_messages": sum(len(msgs) for msgs in self._messages.values()),
            "users_with_preferences": len(self._preferences),
            "total_preferences": sum(len(prefs) for prefs in self._preferences.values()),
        }
    
    async def bulk_add_messages(
        self,
        session_id: str,
        messages: List[ChatMessage],
    ) -> int:
        """Bulk add messages to a session."""
        count = 0
        for msg in messages:
            msg.session_id = session_id
            if await self.save_message(msg):
                count += 1
        return count
    
    async def bulk_add_preferences(
        self,
        user_id: str,
        preferences: List[UserPreference],
    ) -> int:
        """Bulk add preferences for a user."""
        count = 0
        for pref in preferences:
            pref.user_id = user_id
            if await self.update_user_preference(pref):
                count += 1
        return count
