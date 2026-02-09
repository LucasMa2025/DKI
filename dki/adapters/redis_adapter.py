"""
Redis User Data Adapter
Connects DKI to Redis for user data caching and storage

Author: AGI Demo Project
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    ChatMessage,
    UserPreference,
)


class RedisUserDataAdapter(IUserDataAdapter):
    """
    Redis adapter for user data.
    
    Uses Redis as a fast cache/store for user data.
    Supports both redis-py async and sync modes.
    
    Data Structure:
    - user:{user_id} -> Hash (user profile)
    - user:{user_id}:preferences -> List (preferences)
    - session:{session_id}:messages -> List (messages)
    - user:{user_id}:sessions -> Set (session IDs)
    
    Example:
        config = AdapterConfig(
            adapter_type=AdapterType.REDIS,
            host="localhost",
            port=6379,
            password="secret",
        )
        
        async with RedisUserDataAdapter(config) as adapter:
            profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize Redis adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.REDIS:
            self.config.adapter_type = AdapterType.REDIS
        
        if self.config.port == 5432:  # Default PostgreSQL port
            self.config.port = 6379
        
        self._client = None
        self._use_async = True
        
        # Key prefixes
        self._user_prefix = "user"
        self._session_prefix = "session"
        self._pref_suffix = "preferences"
        self._msg_suffix = "messages"
    
    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._connected:
            return
        
        try:
            import redis.asyncio as aioredis
            
            connection_string = self.config.get_connection_string()
            if connection_string:
                self._client = aioredis.from_url(
                    connection_string,
                    decode_responses=True,
                )
            else:
                self._client = aioredis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password or None,
                    decode_responses=True,
                )
            
            # Test connection
            await self._client.ping()
            
            self._connected = True
            self._use_async = True
            logger.info(f"Connected to Redis (async): {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("redis.asyncio not available, falling back to sync redis")
            self._use_async = False
            
            try:
                import redis
                
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password or None,
                    decode_responses=True,
                )
                
                # Test connection
                self._client.ping()
                
                self._connected = True
                logger.info(f"Connected to Redis (sync): {self.config.host}:{self.config.port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if not self._connected:
            return
        
        try:
            if self._use_async and self._client:
                await self._client.close()
            elif self._client:
                self._client.close()
            
            self._connected = False
            self._client = None
            logger.info("Disconnected from Redis")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    def _user_key(self, user_id: str) -> str:
        """Get user profile key."""
        return f"{self._user_prefix}:{user_id}"
    
    def _preferences_key(self, user_id: str) -> str:
        """Get user preferences key."""
        return f"{self._user_prefix}:{user_id}:{self._pref_suffix}"
    
    def _messages_key(self, session_id: str) -> str:
        """Get session messages key."""
        return f"{self._session_prefix}:{session_id}:{self._msg_suffix}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        """Get user sessions set key."""
        return f"{self._user_prefix}:{user_id}:sessions"
    
    async def _execute(self, method: str, *args, **kwargs) -> Any:
        """Execute Redis command."""
        if not self._connected:
            await self.connect()
        
        cmd = getattr(self._client, method)
        
        if self._use_async:
            return await cmd(*args, **kwargs)
        else:
            return cmd(*args, **kwargs)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from Redis."""
        try:
            key = self._user_key(user_id)
            data = await self._execute("hgetall", key)
            
            if not data:
                return None
            
            # Parse JSON fields
            for field in ['preferences', 'settings', 'metadata']:
                if field in data and isinstance(data[field], str):
                    try:
                        data[field] = json.loads(data[field])
                    except json.JSONDecodeError:
                        data[field] = {}
            
            # Parse datetime fields
            for field in ['created_at', 'updated_at', 'last_active_at']:
                if field in data and data[field]:
                    try:
                        data[field] = datetime.fromisoformat(data[field])
                    except (ValueError, TypeError):
                        data[field] = None
            
            # Parse boolean
            if 'is_active' in data:
                data['is_active'] = data['is_active'] in ('true', 'True', '1', True)
            
            data['user_id'] = user_id
            
            return UserProfile.from_dict(data)
            
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """Get session conversation history from Redis."""
        try:
            key = self._messages_key(session_id)
            
            # Get all messages (Redis list)
            raw_messages = await self._execute("lrange", key, 0, -1)
            
            if not raw_messages:
                return []
            
            messages = []
            for raw in raw_messages:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    
                    # Parse timestamp
                    if 'timestamp' in data and isinstance(data['timestamp'], str):
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    msg = ChatMessage.from_dict(data)
                    
                    # Apply time filters
                    if before and msg.timestamp and msg.timestamp >= before:
                        continue
                    if after and msg.timestamp and msg.timestamp <= after:
                        continue
                    
                    messages.append(msg)
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error parsing message: {e}")
                    continue
            
            # Sort by timestamp and limit
            messages.sort(key=lambda m: m.timestamp or datetime.min)
            return messages[-limit:]
            
        except Exception as e:
            logger.error(f"Error fetching session history: {e}")
            return []
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """Get user preferences from Redis."""
        try:
            key = self._preferences_key(user_id)
            
            # Get all preferences (Redis list)
            raw_prefs = await self._execute("lrange", key, 0, -1)
            
            if not raw_prefs:
                return []
            
            preferences = []
            now = datetime.utcnow()
            
            for raw in raw_prefs:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    
                    # Parse datetime fields
                    for field in ['created_at', 'updated_at', 'expires_at']:
                        if field in data and isinstance(data[field], str):
                            try:
                                data[field] = datetime.fromisoformat(data[field])
                            except (ValueError, TypeError):
                                data[field] = None
                    
                    # Parse boolean
                    if 'is_active' in data:
                        data['is_active'] = data['is_active'] in ('true', 'True', '1', True)
                    
                    pref = UserPreference.from_dict(data)
                    
                    # Filter by type
                    if preference_types and pref.preference_type not in preference_types:
                        continue
                    
                    # Filter expired
                    if not include_expired and pref.expires_at and pref.expires_at < now:
                        continue
                    
                    # Filter inactive
                    if not pref.is_active:
                        continue
                    
                    preferences.append(pref)
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error parsing preference: {e}")
                    continue
            
            # Sort by priority
            preferences.sort(key=lambda p: p.priority, reverse=True)
            return preferences
            
        except Exception as e:
            logger.error(f"Error fetching user preferences: {e}")
            return []
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        Search for relevant messages.
        
        Note: Redis doesn't have built-in full-text search.
        This implementation does simple substring matching.
        For production, consider using RediSearch module.
        """
        try:
            query_lower = query.lower()
            all_messages = []
            
            if session_id:
                # Search in specific session
                messages = await self.get_session_history(session_id, limit=1000)
                all_messages.extend(messages)
            else:
                # Get all user sessions and search
                sessions_key = self._user_sessions_key(user_id)
                session_ids = await self._execute("smembers", sessions_key)
                
                for sid in (session_ids or []):
                    messages = await self.get_session_history(sid, limit=100)
                    all_messages.extend([m for m in messages if m.user_id == user_id])
            
            # Simple relevance scoring
            scored = []
            for msg in all_messages:
                content_lower = msg.content.lower()
                if query_lower in content_lower:
                    # Count occurrences as simple relevance score
                    score = content_lower.count(query_lower)
                    scored.append((msg, score))
            
            # Sort by score and return top results
            scored.sort(key=lambda x: x[1], reverse=True)
            return [msg for msg, _ in scored[:limit]]
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            result = await self._execute("ping")
            return result in (True, "PONG", b"PONG")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_message(self, message: ChatMessage) -> bool:
        """Save a message to Redis."""
        try:
            key = self._messages_key(message.session_id)
            
            data = message.to_dict()
            # Convert datetime to string
            if data.get('timestamp'):
                data['timestamp'] = data['timestamp']
            
            await self._execute("rpush", key, json.dumps(data, default=str))
            
            # Also track session for user
            sessions_key = self._user_sessions_key(message.user_id)
            await self._execute("sadd", sessions_key, message.session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False
    
    async def update_user_preference(self, preference: UserPreference) -> bool:
        """Update or create a user preference in Redis."""
        try:
            import uuid
            
            if not preference.preference_id:
                preference.preference_id = f"pref_{uuid.uuid4().hex[:16]}"
            
            key = self._preferences_key(preference.user_id)
            
            # Get existing preferences
            raw_prefs = await self._execute("lrange", key, 0, -1)
            
            # Find and update or append
            updated = False
            new_prefs = []
            
            pref_data = preference.to_dict()
            
            for raw in (raw_prefs or []):
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    if data.get('preference_id') == preference.preference_id:
                        new_prefs.append(json.dumps(pref_data, default=str))
                        updated = True
                    else:
                        new_prefs.append(raw)
                except Exception:
                    new_prefs.append(raw)
            
            if not updated:
                new_prefs.append(json.dumps(pref_data, default=str))
            
            # Replace list
            await self._execute("delete", key)
            if new_prefs:
                await self._execute("rpush", key, *new_prefs)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            return False
    
    async def set_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile to Redis."""
        try:
            key = self._user_key(profile.user_id)
            
            data = {
                "username": profile.username or "",
                "display_name": profile.display_name or "",
                "email": profile.email or "",
                "preferences": json.dumps(profile.preferences),
                "settings": json.dumps(profile.settings),
                "metadata": json.dumps(profile.metadata),
                "created_at": profile.created_at.isoformat() if profile.created_at else "",
                "updated_at": datetime.utcnow().isoformat(),
                "last_active_at": profile.last_active_at.isoformat() if profile.last_active_at else "",
                "is_active": str(profile.is_active),
            }
            
            await self._execute("hset", key, mapping=data)
            
            # Set TTL if configured
            if self.config.cache_ttl > 0:
                await self._execute("expire", key, self.config.cache_ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            return False
