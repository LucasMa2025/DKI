"""
MySQL User Data Adapter
Connects DKI to MySQL databases for user data

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


class MySQLUserDataAdapter(IUserDataAdapter):
    """
    MySQL adapter for user data.
    
    Connects to MySQL databases to fetch user data.
    Supports both aiomysql (async) and pymysql (sync) drivers.
    
    Example:
        config = AdapterConfig(
            adapter_type=AdapterType.MYSQL,
            host="localhost",
            port=3306,
            database="chat_db",
            username="user",
            password="pass",
        )
        
        async with MySQLUserDataAdapter(config) as adapter:
            profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize MySQL adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.MYSQL:
            self.config.adapter_type = AdapterType.MYSQL
        
        if self.config.port == 5432:  # Default PostgreSQL port
            self.config.port = 3306
        
        self._pool = None
        self._use_aiomysql = True
    
    async def connect(self) -> None:
        """Establish connection pool to MySQL."""
        if self._connected:
            return
        
        try:
            import aiomysql
            
            self._pool = await aiomysql.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                db=self.config.database,
                minsize=1,
                maxsize=self.config.pool_size,
                autocommit=True,
            )
            self._connected = True
            self._use_aiomysql = True
            logger.info(f"Connected to MySQL (aiomysql): {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("aiomysql not installed, falling back to pymysql")
            self._use_aiomysql = False
            
            try:
                import pymysql
                from dbutils.pooled_db import PooledDB
                
                self._pool = PooledDB(
                    creator=pymysql,
                    maxconnections=self.config.pool_size,
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                    autocommit=True,
                )
                self._connected = True
                logger.info(f"Connected to MySQL (pymysql): {self.config.host}:{self.config.port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to MySQL: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if not self._connected:
            return
        
        try:
            if self._use_aiomysql and self._pool:
                self._pool.close()
                await self._pool.wait_closed()
            elif self._pool:
                self._pool.close()
            
            self._connected = False
            self._pool = None
            logger.info("Disconnected from MySQL")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MySQL: {e}")
    
    async def _execute_query(
        self,
        query: str,
        params: tuple = (),
        fetch_one: bool = False,
        fetch_all: bool = True,
    ) -> Any:
        """Execute a query and return results."""
        if not self._connected:
            await self.connect()
        
        # Convert $N placeholders to %s for MySQL
        import re
        query = re.sub(r'\$\d+', '%s', query)
        
        if self._use_aiomysql:
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(query, params)
                    if fetch_one:
                        return await cur.fetchone()
                    elif fetch_all:
                        return await cur.fetchall()
                    else:
                        return cur.rowcount
        else:
            conn = self._pool.connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query, params)
                    if fetch_one:
                        return cur.fetchone()
                    elif fetch_all:
                        return cur.fetchall()
                    else:
                        return cur.rowcount
            finally:
                conn.close()
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from MySQL."""
        query = f"""
            SELECT 
                id as user_id,
                username,
                display_name,
                email,
                preferences,
                settings,
                metadata,
                created_at,
                updated_at,
                last_active_at,
                is_active
            FROM {self.config.users_table}
            WHERE id = %s
        """
        
        try:
            row = await self._execute_query(query, (user_id,), fetch_one=True)
            
            if row:
                data = dict(row)
                
                import json
                for field in ['preferences', 'settings', 'metadata']:
                    if isinstance(data.get(field), str):
                        data[field] = json.loads(data[field])
                    elif data.get(field) is None:
                        data[field] = {}
                
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
        conditions = ["session_id = %s"]
        params = [session_id]
        
        if before:
            conditions.append("created_at < %s")
            params.append(before)
        
        if after:
            conditions.append("created_at > %s")
            params.append(after)
        
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        query = f"""
            SELECT 
                id as message_id,
                session_id,
                user_id,
                role,
                content,
                created_at as timestamp,
                embedding,
                metadata,
                token_count,
                parent_id
            FROM {self.config.messages_table}
            WHERE {where_clause}
            ORDER BY created_at ASC
            LIMIT %s
        """
        
        try:
            rows = await self._execute_query(query, tuple(params), fetch_all=True)
            
            messages = []
            for row in rows:
                data = dict(row)
                
                import json
                if isinstance(data.get('metadata'), str):
                    data['metadata'] = json.loads(data['metadata'])
                elif data.get('metadata') is None:
                    data['metadata'] = {}
                
                if data.get('embedding') and isinstance(data['embedding'], (bytes, bytearray)):
                    import numpy as np
                    data['embedding'] = np.frombuffer(bytes(data['embedding']), dtype=np.float32).tolist()
                
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
        conditions = ["user_id = %s", "is_active = 1"]
        params = [user_id]
        
        if preference_types:
            placeholders = ", ".join(["%s"] * len(preference_types))
            conditions.append(f"preference_type IN ({placeholders})")
            params.extend(preference_types)
        
        if not include_expired:
            conditions.append("(expires_at IS NULL OR expires_at > %s)")
            params.append(datetime.utcnow())
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id as preference_id,
                user_id,
                preference_text,
                preference_type,
                priority,
                category,
                metadata,
                created_at,
                updated_at,
                expires_at,
                is_active
            FROM {self.config.preferences_table}
            WHERE {where_clause}
            ORDER BY priority DESC, created_at DESC
        """
        
        try:
            rows = await self._execute_query(query, tuple(params), fetch_all=True)
            
            preferences = []
            for row in rows:
                data = dict(row)
                
                import json
                if isinstance(data.get('metadata'), str):
                    data['metadata'] = json.loads(data['metadata'])
                elif data.get('metadata') is None:
                    data['metadata'] = {}
                
                # Convert is_active from int to bool
                data['is_active'] = bool(data.get('is_active', True))
                
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
        """Search for relevant messages using FULLTEXT or LIKE."""
        conditions = ["user_id = %s"]
        params = [user_id]
        
        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)
        
        # Try FULLTEXT search first
        try:
            conditions.append("MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)")
            params.append(query)
            params.append(limit)
            
            where_clause = " AND ".join(conditions)
            
            search_query = f"""
                SELECT 
                    id as message_id,
                    session_id,
                    user_id,
                    role,
                    content,
                    created_at as timestamp,
                    embedding,
                    metadata,
                    token_count,
                    parent_id,
                    MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
                FROM {self.config.messages_table}
                WHERE {where_clause}
                ORDER BY relevance DESC
                LIMIT %s
            """
            # Add query again for relevance calculation
            params.insert(-1, query)
            
            rows = await self._execute_query(search_query, tuple(params), fetch_all=True)
            
        except Exception:
            # Fallback to LIKE search
            conditions = conditions[:-1]  # Remove FULLTEXT condition
            params = params[:-2]  # Remove query and limit
            
            conditions.append("content LIKE %s")
            params.append(f"%{query}%")
            params.append(limit)
            
            where_clause = " AND ".join(conditions)
            
            search_query = f"""
                SELECT 
                    id as message_id,
                    session_id,
                    user_id,
                    role,
                    content,
                    created_at as timestamp,
                    embedding,
                    metadata,
                    token_count,
                    parent_id
                FROM {self.config.messages_table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            
            rows = await self._execute_query(search_query, tuple(params), fetch_all=True)
        
        try:
            messages = []
            for row in rows:
                data = dict(row)
                data.pop('relevance', None)
                
                import json
                if isinstance(data.get('metadata'), str):
                    data['metadata'] = json.loads(data['metadata'])
                elif data.get('metadata') is None:
                    data['metadata'] = {}
                
                messages.append(ChatMessage.from_dict(data))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check MySQL connection health."""
        try:
            result = await self._execute_query("SELECT 1 as health", fetch_one=True)
            return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Import for type hints
try:
    import aiomysql
except ImportError:
    aiomysql = None

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    pymysql = None
