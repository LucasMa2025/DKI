"""
PostgreSQL User Data Adapter
Connects DKI to PostgreSQL databases for user data

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


class PostgreSQLUserDataAdapter(IUserDataAdapter):
    """
    PostgreSQL adapter for user data.
    
    Connects to PostgreSQL databases to fetch:
    - User profiles
    - Session history
    - User preferences
    
    Supports both asyncpg (async) and psycopg2 (sync) drivers.
    
    Example:
        config = AdapterConfig(
            adapter_type=AdapterType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="chat_db",
            username="user",
            password="pass",
        )
        
        async with PostgreSQLUserDataAdapter(config) as adapter:
            profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize PostgreSQL adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.POSTGRESQL:
            self.config.adapter_type = AdapterType.POSTGRESQL
        
        self._pool = None
        self._use_asyncpg = True  # Prefer asyncpg
    
    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        if self._connected:
            return
        
        try:
            import asyncpg
            
            connection_string = self.config.get_connection_string()
            self._pool = await asyncpg.create_pool(
                connection_string,
                min_size=1,
                max_size=self.config.pool_size,
            )
            self._connected = True
            self._use_asyncpg = True
            logger.info(f"Connected to PostgreSQL (asyncpg): {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("asyncpg not installed, falling back to psycopg2")
            self._use_asyncpg = False
            
            try:
                import psycopg2
                from psycopg2 import pool
                
                self._pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.config.pool_size,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                )
                self._connected = True
                logger.info(f"Connected to PostgreSQL (psycopg2): {self.config.host}:{self.config.port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if not self._connected:
            return
        
        try:
            if self._use_asyncpg and self._pool:
                await self._pool.close()
            elif self._pool:
                self._pool.closeall()
            
            self._connected = False
            self._pool = None
            logger.info("Disconnected from PostgreSQL")
            
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {e}")
    
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
        
        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                if fetch_one:
                    return await conn.fetchrow(query, *params)
                elif fetch_all:
                    return await conn.fetch(query, *params)
                else:
                    return await conn.execute(query, *params)
        else:
            # Sync fallback with psycopg2
            conn = self._pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    if fetch_one:
                        row = cur.fetchone()
                        if row:
                            columns = [desc[0] for desc in cur.description]
                            return dict(zip(columns, row))
                        return None
                    elif fetch_all:
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description]
                        return [dict(zip(columns, row)) for row in rows]
                    else:
                        conn.commit()
                        return cur.rowcount
            finally:
                self._pool.putconn(conn)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from PostgreSQL."""
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
            FROM {self.config.schema}.{self.config.users_table}
            WHERE id = $1
        """
        
        try:
            row = await self._execute_query(query, (user_id,), fetch_one=True)
            
            if row:
                # Handle dict conversion for asyncpg Record
                if hasattr(row, 'items'):
                    data = dict(row.items())
                else:
                    data = dict(row)
                
                # Parse JSON fields if they're strings
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
        conditions = ["session_id = $1"]
        params = [session_id]
        param_idx = 2
        
        if before:
            conditions.append(f"created_at < ${param_idx}")
            params.append(before)
            param_idx += 1
        
        if after:
            conditions.append(f"created_at > ${param_idx}")
            params.append(after)
            param_idx += 1
        
        where_clause = " AND ".join(conditions)
        
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
            FROM {self.config.schema}.{self.config.messages_table}
            WHERE {where_clause}
            ORDER BY created_at ASC
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        try:
            rows = await self._execute_query(query, tuple(params), fetch_all=True)
            
            messages = []
            for row in rows:
                if hasattr(row, 'items'):
                    data = dict(row.items())
                else:
                    data = dict(row)
                
                # Parse metadata if string
                import json
                if isinstance(data.get('metadata'), str):
                    data['metadata'] = json.loads(data['metadata'])
                elif data.get('metadata') is None:
                    data['metadata'] = {}
                
                # Convert embedding from bytes if needed
                if data.get('embedding') and isinstance(data['embedding'], (bytes, memoryview)):
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
        conditions = ["user_id = $1", "is_active = true"]
        params = [user_id]
        param_idx = 2
        
        if preference_types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(preference_types)))
            conditions.append(f"preference_type IN ({placeholders})")
            params.extend(preference_types)
            param_idx += len(preference_types)
        
        if not include_expired:
            conditions.append(f"(expires_at IS NULL OR expires_at > ${param_idx})")
            params.append(datetime.utcnow())
            param_idx += 1
        
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
            FROM {self.config.schema}.{self.config.preferences_table}
            WHERE {where_clause}
            ORDER BY priority DESC, created_at DESC
        """
        
        try:
            rows = await self._execute_query(query, tuple(params), fetch_all=True)
            
            preferences = []
            for row in rows:
                if hasattr(row, 'items'):
                    data = dict(row.items())
                else:
                    data = dict(row)
                
                import json
                if isinstance(data.get('metadata'), str):
                    data['metadata'] = json.loads(data['metadata'])
                elif data.get('metadata') is None:
                    data['metadata'] = {}
                
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
        """
        Search for relevant messages.
        
        Uses PostgreSQL full-text search if available,
        otherwise falls back to ILIKE pattern matching.
        """
        conditions = ["user_id = $1"]
        params = [user_id]
        param_idx = 2
        
        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1
        
        # Try full-text search first
        try:
            # Check if ts_vector column exists
            check_query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{self.config.schema}' 
                AND table_name = '{self.config.messages_table}'
                AND column_name = 'content_tsv'
            """
            result = await self._execute_query(check_query, fetch_all=True)
            
            if result:
                # Use full-text search
                conditions.append(f"content_tsv @@ plainto_tsquery('english', ${param_idx})")
                params.append(query)
                param_idx += 1
                
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
                        ts_rank(content_tsv, plainto_tsquery('english', ${param_idx - 1})) as rank
                    FROM {self.config.schema}.{self.config.messages_table}
                    WHERE {where_clause}
                    ORDER BY rank DESC
                    LIMIT ${param_idx}
                """
                params.append(limit)
            else:
                raise Exception("No ts_vector column")
                
        except Exception:
            # Fallback to ILIKE search
            conditions.append(f"content ILIKE ${param_idx}")
            params.append(f"%{query}%")
            param_idx += 1
            
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
                FROM {self.config.schema}.{self.config.messages_table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx}
            """
            params.append(limit)
        
        try:
            rows = await self._execute_query(search_query, tuple(params), fetch_all=True)
            
            messages = []
            for row in rows:
                if hasattr(row, 'items'):
                    data = dict(row.items())
                else:
                    data = dict(row)
                
                # Remove rank field if present
                data.pop('rank', None)
                
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
        """Check PostgreSQL connection health."""
        try:
            result = await self._execute_query("SELECT 1", fetch_one=True)
            return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_message(self, message: ChatMessage) -> bool:
        """Save a message to PostgreSQL."""
        import json
        
        query = f"""
            INSERT INTO {self.config.schema}.{self.config.messages_table}
            (id, session_id, user_id, role, content, created_at, embedding, metadata, token_count, parent_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                token_count = EXCLUDED.token_count
        """
        
        try:
            embedding_bytes = None
            if message.embedding:
                import numpy as np
                embedding_bytes = np.array(message.embedding, dtype=np.float32).tobytes()
            
            await self._execute_query(
                query,
                (
                    message.message_id,
                    message.session_id,
                    message.user_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    embedding_bytes,
                    json.dumps(message.metadata),
                    message.token_count,
                    message.parent_id,
                ),
                fetch_all=False,
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False
    
    async def update_user_preference(self, preference: UserPreference) -> bool:
        """Update or create a user preference."""
        import json
        import uuid
        
        if not preference.preference_id:
            preference.preference_id = f"pref_{uuid.uuid4().hex[:16]}"
        
        query = f"""
            INSERT INTO {self.config.schema}.{self.config.preferences_table}
            (id, user_id, preference_text, preference_type, priority, category, metadata, created_at, updated_at, expires_at, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                preference_text = EXCLUDED.preference_text,
                preference_type = EXCLUDED.preference_type,
                priority = EXCLUDED.priority,
                category = EXCLUDED.category,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at,
                expires_at = EXCLUDED.expires_at,
                is_active = EXCLUDED.is_active
        """
        
        try:
            now = datetime.utcnow()
            await self._execute_query(
                query,
                (
                    preference.preference_id,
                    preference.user_id,
                    preference.preference_text,
                    preference.preference_type,
                    preference.priority,
                    preference.category,
                    json.dumps(preference.metadata),
                    preference.created_at or now,
                    now,
                    preference.expires_at,
                    preference.is_active,
                ),
                fetch_all=False,
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            return False
