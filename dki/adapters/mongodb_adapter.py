"""
MongoDB User Data Adapter
Connects DKI to MongoDB databases for user data

Author: AGI Demo Project
Version: 1.0.0
"""

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


class MongoDBUserDataAdapter(IUserDataAdapter):
    """
    MongoDB adapter for user data.
    
    Connects to MongoDB to fetch user data.
    Supports both motor (async) and pymongo (sync) drivers.
    
    Example:
        config = AdapterConfig(
            adapter_type=AdapterType.MONGODB,
            host="localhost",
            port=27017,
            database="chat_db",
            username="user",
            password="pass",
        )
        
        async with MongoDBUserDataAdapter(config) as adapter:
            profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize MongoDB adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.MONGODB:
            self.config.adapter_type = AdapterType.MONGODB
        
        if self.config.port == 5432:  # Default PostgreSQL port
            self.config.port = 27017
        
        self._client = None
        self._db = None
        self._use_motor = True
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._connected:
            return
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            connection_string = self.config.get_connection_string()
            if not connection_string:
                connection_string = f"mongodb://{self.config.host}:{self.config.port}"
            
            self._client = AsyncIOMotorClient(
                connection_string,
                maxPoolSize=self.config.pool_size,
            )
            self._db = self._client[self.config.database]
            self._connected = True
            self._use_motor = True
            logger.info(f"Connected to MongoDB (motor): {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("motor not installed, falling back to pymongo")
            self._use_motor = False
            
            try:
                from pymongo import MongoClient
                
                connection_string = self.config.get_connection_string()
                if not connection_string:
                    connection_string = f"mongodb://{self.config.host}:{self.config.port}"
                
                self._client = MongoClient(
                    connection_string,
                    maxPoolSize=self.config.pool_size,
                )
                self._db = self._client[self.config.database]
                self._connected = True
                logger.info(f"Connected to MongoDB (pymongo): {self.config.host}:{self.config.port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if not self._connected:
            return
        
        try:
            if self._client:
                self._client.close()
            
            self._connected = False
            self._client = None
            self._db = None
            logger.info("Disconnected from MongoDB")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {e}")
    
    def _get_collection(self, name: str):
        """Get MongoDB collection."""
        return self._db[name]
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from MongoDB."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self._get_collection(self.config.users_table)
            
            if self._use_motor:
                doc = await collection.find_one({"_id": user_id})
            else:
                doc = collection.find_one({"_id": user_id})
            
            if doc:
                # Map MongoDB _id to user_id
                data = {
                    "user_id": doc.get("_id") or doc.get("user_id"),
                    "username": doc.get("username"),
                    "display_name": doc.get("display_name"),
                    "email": doc.get("email"),
                    "preferences": doc.get("preferences", {}),
                    "settings": doc.get("settings", {}),
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "last_active_at": doc.get("last_active_at"),
                    "is_active": doc.get("is_active", True),
                }
                
                return UserProfile.from_dict(data)
            
            return None
            
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
        """Get session conversation history."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self._get_collection(self.config.messages_table)
            
            query = {"session_id": session_id}
            
            if before or after:
                query["timestamp"] = {}
                if before:
                    query["timestamp"]["$lt"] = before
                if after:
                    query["timestamp"]["$gt"] = after
            
            if self._use_motor:
                cursor = collection.find(query).sort("timestamp", 1).limit(limit)
                docs = await cursor.to_list(length=limit)
            else:
                docs = list(collection.find(query).sort("timestamp", 1).limit(limit))
            
            messages = []
            for doc in docs:
                data = {
                    "message_id": str(doc.get("_id") or doc.get("message_id")),
                    "session_id": doc.get("session_id"),
                    "user_id": doc.get("user_id"),
                    "role": doc.get("role"),
                    "content": doc.get("content"),
                    "timestamp": doc.get("timestamp") or doc.get("created_at"),
                    "embedding": doc.get("embedding"),
                    "metadata": doc.get("metadata", {}),
                    "token_count": doc.get("token_count"),
                    "parent_id": doc.get("parent_id"),
                }
                messages.append(ChatMessage.from_dict(data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching session history: {e}")
            return []
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """Get user preferences."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self._get_collection(self.config.preferences_table)
            
            query = {
                "user_id": user_id,
                "is_active": True,
            }
            
            if preference_types:
                query["preference_type"] = {"$in": preference_types}
            
            if not include_expired:
                query["$or"] = [
                    {"expires_at": None},
                    {"expires_at": {"$gt": datetime.utcnow()}},
                ]
            
            if self._use_motor:
                cursor = collection.find(query).sort([("priority", -1), ("created_at", -1)])
                docs = await cursor.to_list(length=100)
            else:
                docs = list(collection.find(query).sort([("priority", -1), ("created_at", -1)]))
            
            preferences = []
            for doc in docs:
                data = {
                    "preference_id": str(doc.get("_id") or doc.get("preference_id")),
                    "user_id": doc.get("user_id"),
                    "preference_text": doc.get("preference_text"),
                    "preference_type": doc.get("preference_type"),
                    "priority": doc.get("priority", 0),
                    "category": doc.get("category"),
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "expires_at": doc.get("expires_at"),
                    "is_active": doc.get("is_active", True),
                }
                preferences.append(UserPreference.from_dict(data))
            
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
        """Search for relevant messages using text search or regex."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self._get_collection(self.config.messages_table)
            
            # Try text search first (requires text index)
            try:
                search_query = {
                    "user_id": user_id,
                    "$text": {"$search": query},
                }
                
                if session_id:
                    search_query["session_id"] = session_id
                
                if self._use_motor:
                    cursor = collection.find(
                        search_query,
                        {"score": {"$meta": "textScore"}}
                    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
                    docs = await cursor.to_list(length=limit)
                else:
                    docs = list(collection.find(
                        search_query,
                        {"score": {"$meta": "textScore"}}
                    ).sort([("score", {"$meta": "textScore"})]).limit(limit))
                    
            except Exception:
                # Fallback to regex search
                search_query = {
                    "user_id": user_id,
                    "content": {"$regex": query, "$options": "i"},
                }
                
                if session_id:
                    search_query["session_id"] = session_id
                
                if self._use_motor:
                    cursor = collection.find(search_query).sort("timestamp", -1).limit(limit)
                    docs = await cursor.to_list(length=limit)
                else:
                    docs = list(collection.find(search_query).sort("timestamp", -1).limit(limit))
            
            messages = []
            for doc in docs:
                data = {
                    "message_id": str(doc.get("_id") or doc.get("message_id")),
                    "session_id": doc.get("session_id"),
                    "user_id": doc.get("user_id"),
                    "role": doc.get("role"),
                    "content": doc.get("content"),
                    "timestamp": doc.get("timestamp") or doc.get("created_at"),
                    "embedding": doc.get("embedding"),
                    "metadata": doc.get("metadata", {}),
                    "token_count": doc.get("token_count"),
                    "parent_id": doc.get("parent_id"),
                }
                messages.append(ChatMessage.from_dict(data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check MongoDB connection health."""
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                return False
        
        try:
            if self._use_motor:
                await self._client.admin.command('ping')
            else:
                self._client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_message(self, message: ChatMessage) -> bool:
        """Save a message to MongoDB."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self._get_collection(self.config.messages_table)
            
            doc = {
                "_id": message.message_id,
                "session_id": message.session_id,
                "user_id": message.user_id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "embedding": message.embedding,
                "metadata": message.metadata,
                "token_count": message.token_count,
                "parent_id": message.parent_id,
            }
            
            if self._use_motor:
                await collection.replace_one(
                    {"_id": message.message_id},
                    doc,
                    upsert=True
                )
            else:
                collection.replace_one(
                    {"_id": message.message_id},
                    doc,
                    upsert=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False
    
    async def update_user_preference(self, preference: UserPreference) -> bool:
        """Update or create a user preference."""
        if not self._connected:
            await self.connect()
        
        try:
            import uuid
            collection = self._get_collection(self.config.preferences_table)
            
            if not preference.preference_id:
                preference.preference_id = f"pref_{uuid.uuid4().hex[:16]}"
            
            doc = {
                "_id": preference.preference_id,
                "user_id": preference.user_id,
                "preference_text": preference.preference_text,
                "preference_type": preference.preference_type,
                "priority": preference.priority,
                "category": preference.category,
                "metadata": preference.metadata,
                "created_at": preference.created_at or datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "expires_at": preference.expires_at,
                "is_active": preference.is_active,
            }
            
            if self._use_motor:
                await collection.replace_one(
                    {"_id": preference.preference_id},
                    doc,
                    upsert=True
                )
            else:
                collection.replace_one(
                    {"_id": preference.preference_id},
                    doc,
                    upsert=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            return False
