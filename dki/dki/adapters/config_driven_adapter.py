"""
Configuration-Driven User Data Adapter

配置驱动的用户数据适配器

核心设计理念:
- 上层应用无需实现任何接口
- 只需提供配置文件，指定数据库连接和字段映射
- DKI 自动连接上层应用的数据库，读取用户偏好和历史消息

配置示例:
```yaml
user_adapter:
  database:
    type: postgresql  # postgresql | mysql | sqlite
    host: localhost
    port: 5432
    database: my_app_db
    username: user
    password: pass
  
  # 用户偏好表映射
  preferences:
    table: user_preferences
    fields:
      user_id: user_id          # 用户 ID 字段
      preference_text: content  # 偏好内容字段
      preference_type: type     # 偏好类型字段
      priority: priority        # 优先级字段 (可选)
      created_at: created_at    # 创建时间字段 (可选)
    filters:
      is_active: true           # 额外过滤条件
  
  # 消息表映射
  messages:
    table: chat_messages
    fields:
      message_id: id
      session_id: session_id
      user_id: user_id
      role: role
      content: content
      timestamp: created_at
      embedding: embedding      # 向量字段 (可选)
    
  # 向量检索配置
  vector_search:
    enabled: true
    type: pgvector  # pgvector | faiss | dynamic
    embedding_field: embedding
    embedding_dim: 1536
    # 如果 type=dynamic，使用动态向量处理
    dynamic:
      strategy: hybrid  # lazy | batch | hybrid
      embedding_model: text-embedding-ada-002
```

Author: AGI Demo Project
Version: 2.0.0
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    Text,
    create_engine,
    select,
    and_,
    or_,
    desc,
    asc,
    func,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    UserPreference,
    ChatMessage,
)


class DatabaseType(str, Enum):
    """支持的数据库类型"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class VectorSearchType(str, Enum):
    """向量检索类型"""
    PGVECTOR = "pgvector"       # PostgreSQL pgvector 扩展
    FAISS = "faiss"             # FAISS 索引
    DYNAMIC = "dynamic"         # 动态向量处理 (无预计算向量)
    NONE = "none"               # 不使用向量检索


@dataclass
class DatabaseConfig:
    """数据库连接配置"""
    type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    
    # 连接池
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    # SSL
    ssl_enabled: bool = False
    ssl_ca: Optional[str] = None
    
    def get_async_url(self) -> str:
        """获取异步连接 URL"""
        if self.type == DatabaseType.POSTGRESQL:
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.MYSQL:
            return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.SQLITE:
            return f"sqlite+aiosqlite:///{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class FieldMapping:
    """字段映射配置"""
    # 源字段名 (上层应用数据库中的字段名)
    source: str
    # 目标字段名 (DKI 内部使用的字段名)
    target: str
    # 字段类型
    type: str = "string"  # string | int | float | datetime | bool | json | vector
    # 是否必需
    required: bool = False
    # 默认值
    default: Any = None


@dataclass
class TableMapping:
    """表映射配置"""
    # 表名
    table: str
    # 字段映射
    fields: Dict[str, str]  # target_field -> source_field
    # 额外过滤条件
    filters: Dict[str, Any] = field(default_factory=dict)
    # 排序字段
    order_by: Optional[str] = None
    order_desc: bool = True
    # JSON 内容解析 key (用于从 JSON 字符串中提取实际内容)
    # 支持嵌套 key，如 "text", "data.text", "choices.0.text"
    content_json_key: Optional[str] = None


@dataclass
class VectorSearchConfig:
    """向量检索配置"""
    enabled: bool = True
    type: VectorSearchType = VectorSearchType.DYNAMIC
    
    # 向量字段 (如果数据库中有预计算向量)
    embedding_field: Optional[str] = None
    embedding_dim: int = 1536
    
    # 动态向量处理配置
    dynamic_strategy: str = "hybrid"  # lazy | batch | hybrid
    embedding_model: str = "text-embedding-ada-002"
    embedding_api_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    
    # 检索参数
    top_k: int = 10
    similarity_threshold: float = 0.5


@dataclass
class ConfigDrivenAdapterConfig:
    """
    配置驱动适配器的完整配置
    
    上层应用只需提供此配置，无需实现任何接口
    """
    # 数据库连接
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # 用户偏好表映射
    preferences: Optional[TableMapping] = None
    
    # 消息表映射
    messages: Optional[TableMapping] = None
    
    # 用户表映射 (可选)
    users: Optional[TableMapping] = None
    
    # 会话表映射 (可选)
    sessions: Optional[TableMapping] = None
    
    # 向量检索配置
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 分钟
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigDrivenAdapterConfig":
        """
        从字典创建配置
        
        支持两种格式:
        1. 完整格式 (adapter_config.yaml): 包含 database, preferences, messages 等嵌套结构
        2. 简化格式 (config.yaml 中的 user_adapter): 扁平化结构
        """
        config = cls()
        
        # 检测配置格式
        # 简化格式: "type" 在顶层，且 "database" 不是字典 (可能是字符串表示数据库名)
        is_simplified = "type" in data and not isinstance(data.get("database"), dict)
        
        if is_simplified:
            # 简化格式 (从主配置文件 config.yaml)
            db_type = data.get("type", "postgresql")
            if db_type == "memory":
                # 内存模式，不需要数据库连接
                logger.info("Using memory adapter mode")
                return config
            
            config.database = DatabaseConfig(
                type=DatabaseType(db_type),
                host=data.get("host", "localhost"),
                port=data.get("port", 5432),
                database=data.get("database", ""),
                username=data.get("username", ""),
                password=data.get("password", ""),
                pool_size=data.get("pool_size", 5),
            )
            
            # 简化格式的表映射
            config.preferences = TableMapping(
                table=data.get("preferences_table", "user_preferences"),
                fields={
                    "user_id": "user_id",
                    "preference_id": "id",
                    "preference_text": "content",
                    "preference_type": "type",
                },
                content_json_key=data.get("preferences_content_json_key"),
            )
            
            config.messages = TableMapping(
                table=data.get("messages_table", "messages"),
                fields={
                    "message_id": "id",
                    "session_id": "session_id",
                    "user_id": "user_id",
                    "role": "role",
                    "content": "content",
                    "timestamp": "created_at",
                },
                content_json_key=data.get("messages_content_json_key"),
            )
            
            config.users = TableMapping(
                table=data.get("users_table", "users"),
                fields={
                    "user_id": "id",
                    "username": "username",
                },
            )
            
            config.cache_enabled = data.get("enable_cache", True)
            config.cache_ttl = data.get("cache_ttl", 300)
            
            return config
        
        # 完整格式 (adapter_config.yaml)
        # 数据库配置
        if "database" in data:
            db_data = data["database"]
            config.database = DatabaseConfig(
                type=DatabaseType(db_data.get("type", "postgresql")),
                host=db_data.get("host", "localhost"),
                port=db_data.get("port", 5432),
                database=db_data.get("database", ""),
                username=db_data.get("username", ""),
                password=db_data.get("password", ""),
                pool_size=db_data.get("pool_size", 5),
                max_overflow=db_data.get("max_overflow", 10),
            )
        
        # 偏好表映射
        if "preferences" in data:
            pref_data = data["preferences"]
            config.preferences = TableMapping(
                table=pref_data.get("table", "user_preferences"),
                fields=pref_data.get("fields", {}),
                filters=pref_data.get("filters", {}),
                order_by=pref_data.get("order_by"),
                order_desc=pref_data.get("order_desc", True),
                content_json_key=pref_data.get("content_json_key"),
            )
        
        # 消息表映射
        if "messages" in data:
            msg_data = data["messages"]
            config.messages = TableMapping(
                table=msg_data.get("table", "messages"),
                fields=msg_data.get("fields", {}),
                filters=msg_data.get("filters", {}),
                order_by=msg_data.get("order_by", "timestamp"),
                order_desc=msg_data.get("order_desc", True),
                content_json_key=msg_data.get("content_json_key"),
            )
        
        # 用户表映射
        if "users" in data:
            user_data = data["users"]
            config.users = TableMapping(
                table=user_data.get("table", "users"),
                fields=user_data.get("fields", {}),
                filters=user_data.get("filters", {}),
            )
        
        # 会话表映射
        if "sessions" in data:
            sess_data = data["sessions"]
            config.sessions = TableMapping(
                table=sess_data.get("table", "sessions"),
                fields=sess_data.get("fields", {}),
                filters=sess_data.get("filters", {}),
            )
        
        # 向量检索配置
        if "vector_search" in data:
            vs_data = data["vector_search"]
            config.vector_search = VectorSearchConfig(
                enabled=vs_data.get("enabled", True),
                type=VectorSearchType(vs_data.get("type", "dynamic")),
                embedding_field=vs_data.get("embedding_field"),
                embedding_dim=vs_data.get("embedding_dim", 1536),
                dynamic_strategy=vs_data.get("dynamic", {}).get("strategy", "hybrid"),
                embedding_model=vs_data.get("dynamic", {}).get("embedding_model", "text-embedding-ada-002"),
                embedding_api_url=vs_data.get("dynamic", {}).get("api_url"),
                embedding_api_key=vs_data.get("dynamic", {}).get("api_key"),
                top_k=vs_data.get("top_k", 10),
                similarity_threshold=vs_data.get("similarity_threshold", 0.5),
            )
        
        # 缓存配置
        config.cache_enabled = data.get("cache_enabled", True)
        config.cache_ttl = data.get("cache_ttl", 300)
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ConfigDrivenAdapterConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("user_adapter", data))


class ConfigDrivenAdapter(IUserDataAdapter):
    """
    配置驱动的用户数据适配器
    
    核心特性:
    1. 配置驱动 - 上层应用只需提供配置，无需实现接口
    2. SQLAlchemy 动态表映射 - 根据配置动态连接任意表结构
    3. 向量检索集成 - 支持 pgvector、FAISS、动态向量处理
    
    使用方式:
    ```python
    # 方式 1: 从配置字典创建
    config = ConfigDrivenAdapterConfig.from_dict({
        "database": {
            "type": "postgresql",
            "host": "localhost",
            "database": "my_app_db",
            "username": "user",
            "password": "pass",
        },
        "preferences": {
            "table": "user_preferences",
            "fields": {
                "user_id": "user_id",
                "preference_text": "content",
                "preference_type": "type",
            },
        },
        "messages": {
            "table": "chat_messages",
            "fields": {
                "message_id": "id",
                "session_id": "session_id",
                "user_id": "user_id",
                "role": "role",
                "content": "content",
                "timestamp": "created_at",
            },
        },
    })
    adapter = ConfigDrivenAdapter(config)
    
    # 方式 2: 从 YAML 文件创建
    adapter = ConfigDrivenAdapter.from_yaml("config/adapter.yaml")
    
    # 连接并使用
    await adapter.connect()
    preferences = await adapter.get_user_preferences("user_123")
    history = await adapter.search_relevant_history("user_123", "餐厅推荐")
    ```
    """
    
    def __init__(self, config: ConfigDrivenAdapterConfig):
        """
        初始化适配器
        
        Args:
            config: 适配器配置
        """
        super().__init__()
        self.adapter_config = config
        
        # SQLAlchemy 引擎和会话
        self._engine: Optional[AsyncEngine] = None
        self._session_factory = None
        
        # 动态表对象
        self._metadata = MetaData()
        self._tables: Dict[str, Table] = {}
        
        # 动态向量处理器
        self._vector_handler = None
        self._embedding_service = None
        
        # 缓存
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        logger.info(f"ConfigDrivenAdapter initialized (db_type={config.database.type})")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ConfigDrivenAdapter":
        """从 YAML 文件创建适配器"""
        config = ConfigDrivenAdapterConfig.from_yaml(yaml_path)
        return cls(config)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigDrivenAdapter":
        """从字典创建适配器"""
        config = ConfigDrivenAdapterConfig.from_dict(data)
        return cls(config)
    
    async def connect(self) -> None:
        """建立数据库连接"""
        try:
            # 创建异步引擎
            self._engine = create_async_engine(
                self.adapter_config.database.get_async_url(),
                pool_size=self.adapter_config.database.pool_size,
                max_overflow=self.adapter_config.database.max_overflow,
                pool_timeout=self.adapter_config.database.pool_timeout,
                echo=False,
            )
            
            # 创建会话工厂
            self._session_factory = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # 反射表结构
            await self._reflect_tables()
            
            # 初始化向量处理器
            await self._init_vector_handler()
            
            self._connected = True
            logger.info("ConfigDrivenAdapter connected to database")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        
        self._connected = False
        logger.info("ConfigDrivenAdapter disconnected")
    
    async def _reflect_tables(self) -> None:
        """反射表结构"""
        async with self._engine.begin() as conn:
            # 反射偏好表
            if self.adapter_config.preferences:
                await conn.run_sync(
                    lambda sync_conn: self._reflect_table(
                        sync_conn,
                        self.adapter_config.preferences.table,
                        "preferences"
                    )
                )
            
            # 反射消息表
            if self.adapter_config.messages:
                await conn.run_sync(
                    lambda sync_conn: self._reflect_table(
                        sync_conn,
                        self.adapter_config.messages.table,
                        "messages"
                    )
                )
            
            # 反射用户表
            if self.adapter_config.users:
                await conn.run_sync(
                    lambda sync_conn: self._reflect_table(
                        sync_conn,
                        self.adapter_config.users.table,
                        "users"
                    )
                )
            
            # 反射会话表
            if self.adapter_config.sessions:
                await conn.run_sync(
                    lambda sync_conn: self._reflect_table(
                        sync_conn,
                        self.adapter_config.sessions.table,
                        "sessions"
                    )
                )
    
    def _reflect_table(self, conn, table_name: str, key: str) -> None:
        """反射单个表"""
        try:
            table = Table(
                table_name,
                self._metadata,
                autoload_with=conn,
            )
            self._tables[key] = table
            logger.debug(f"Reflected table: {table_name} -> {key}")
        except Exception as e:
            logger.warning(f"Failed to reflect table {table_name}: {e}")
    
    async def _init_vector_handler(self) -> None:
        """初始化向量处理器"""
        vs_config = self.adapter_config.vector_search
        
        if not vs_config.enabled:
            return
        
        if vs_config.type == VectorSearchType.DYNAMIC:
            # 使用动态向量处理
            from dki.cache.non_vectorized_handler import (
                NonVectorizedDataHandler,
                HandlerConfig,
                SearchStrategy,
            )
            
            # 创建 embedding 服务
            self._embedding_service = await self._create_embedding_service()
            
            if self._embedding_service:
                handler_config = HandlerConfig(
                    default_strategy=SearchStrategy(vs_config.dynamic_strategy),
                    cache_embeddings=True,
                )
                self._vector_handler = NonVectorizedDataHandler(
                    embedding_service=self._embedding_service,
                    config=handler_config,
                )
                logger.info(f"Dynamic vector handler initialized (strategy={vs_config.dynamic_strategy})")
    
    async def _create_embedding_service(self):
        """创建 embedding 服务"""
        vs_config = self.adapter_config.vector_search
        
        # 简单的 embedding 服务封装
        class EmbeddingService:
            def __init__(self, model: str, api_url: Optional[str], api_key: Optional[str]):
                self.model = model
                self.api_url = api_url or "https://api.openai.com/v1"
                self.api_key = api_key
            
            def embed(self, text: str) -> List[float]:
                """计算单个文本的 embedding"""
                # 这里应该调用实际的 embedding API
                # 为了演示，返回模拟向量
                import hashlib
                import random
                
                # 基于文本哈希生成伪随机向量 (仅用于演示)
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                random.seed(seed)
                return [random.gauss(0, 1) for _ in range(1536)]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                """批量计算 embedding"""
                return [self.embed(t) for t in texts]
        
        return EmbeddingService(
            model=vs_config.embedding_model,
            api_url=vs_config.embedding_api_url,
            api_key=vs_config.embedding_api_key,
        )
    
    def _get_field(self, mapping: TableMapping, target_field: str) -> str:
        """获取映射后的字段名"""
        return mapping.fields.get(target_field, target_field)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        if not self.adapter_config.users or "users" not in self._tables:
            return None
        
        mapping = self.adapter_config.users
        table = self._tables["users"]
        
        user_id_field = self._get_field(mapping, "user_id")
        
        async with self._session_factory() as session:
            stmt = select(table).where(
                table.c[user_id_field] == user_id
            )
            
            result = await session.execute(stmt)
            row = result.fetchone()
            
            if not row:
                return None
            
            return self._row_to_user_profile(row, mapping)
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """获取用户偏好"""
        if not self.adapter_config.preferences or "preferences" not in self._tables:
            return []
        
        # 检查缓存
        cache_key = f"prefs:{user_id}:{preference_types}:{include_expired}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        mapping = self.adapter_config.preferences
        table = self._tables["preferences"]
        
        user_id_field = self._get_field(mapping, "user_id")
        
        async with self._session_factory() as session:
            # 构建查询
            conditions = [table.c[user_id_field] == user_id]
            
            # 添加类型过滤
            if preference_types:
                type_field = self._get_field(mapping, "preference_type")
                if type_field in table.c:
                    conditions.append(table.c[type_field].in_(preference_types))
            
            # 添加额外过滤条件
            for filter_field, filter_value in mapping.filters.items():
                if filter_field in table.c:
                    conditions.append(table.c[filter_field] == filter_value)
            
            stmt = select(table).where(and_(*conditions))
            
            # 排序
            if mapping.order_by and mapping.order_by in table.c:
                if mapping.order_desc:
                    stmt = stmt.order_by(desc(table.c[mapping.order_by]))
                else:
                    stmt = stmt.order_by(asc(table.c[mapping.order_by]))
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            preferences = [
                self._row_to_preference(row, mapping)
                for row in rows
            ]
            
            # 缓存结果
            self._set_cached(cache_key, preferences)
            
            return preferences
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """获取会话历史"""
        if not self.adapter_config.messages or "messages" not in self._tables:
            return []
        
        mapping = self.adapter_config.messages
        table = self._tables["messages"]
        
        session_id_field = self._get_field(mapping, "session_id")
        timestamp_field = self._get_field(mapping, "timestamp")
        
        async with self._session_factory() as session:
            conditions = [table.c[session_id_field] == session_id]
            
            # 时间过滤
            if before and timestamp_field in table.c:
                conditions.append(table.c[timestamp_field] < before)
            if after and timestamp_field in table.c:
                conditions.append(table.c[timestamp_field] > after)
            
            # 添加额外过滤条件
            for filter_field, filter_value in mapping.filters.items():
                if filter_field in table.c:
                    conditions.append(table.c[filter_field] == filter_value)
            
            stmt = select(table).where(and_(*conditions))
            
            # 排序
            if timestamp_field in table.c:
                stmt = stmt.order_by(desc(table.c[timestamp_field]))
            
            stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            messages = [
                self._row_to_message(row, mapping)
                for row in rows
            ]
            
            # 按时间正序返回
            messages.reverse()
            
            return messages
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        检索相关历史消息
        
        根据配置使用不同的检索策略:
        1. pgvector - 使用 PostgreSQL 向量扩展
        2. dynamic - 使用动态向量处理 (BM25 + embedding)
        3. none - 简单的关键词匹配
        """
        if not self.adapter_config.messages or "messages" not in self._tables:
            return []
        
        vs_config = self.adapter_config.vector_search
        
        if vs_config.type == VectorSearchType.PGVECTOR:
            return await self._search_with_pgvector(user_id, query, limit, session_id)
        elif vs_config.type == VectorSearchType.DYNAMIC and self._vector_handler:
            return await self._search_with_dynamic_handler(user_id, query, limit, session_id)
        else:
            return await self._search_with_keywords(user_id, query, limit, session_id)
    
    async def _search_with_pgvector(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """使用 pgvector 进行向量检索"""
        mapping = self.adapter_config.messages
        table = self._tables["messages"]
        vs_config = self.adapter_config.vector_search
        
        user_id_field = self._get_field(mapping, "user_id")
        embedding_field = vs_config.embedding_field or self._get_field(mapping, "embedding")
        
        if embedding_field not in table.c:
            logger.warning(f"Embedding field {embedding_field} not found, falling back to keyword search")
            return await self._search_with_keywords(user_id, query, limit, session_id)
        
        # 计算查询向量
        if self._embedding_service:
            query_embedding = self._embedding_service.embed(query)
        else:
            return await self._search_with_keywords(user_id, query, limit, session_id)
        
        async with self._session_factory() as session:
            conditions = [table.c[user_id_field] == user_id]
            
            if session_id:
                session_id_field = self._get_field(mapping, "session_id")
                conditions.append(table.c[session_id_field] == session_id)
            
            # pgvector 相似度查询
            # 注意: 这需要 pgvector 扩展已安装
            from sqlalchemy import text
            
            query_vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            stmt = text(f"""
                SELECT *, ({embedding_field} <=> :query_vec::vector) as distance
                FROM {mapping.table}
                WHERE {user_id_field} = :user_id
                {"AND " + self._get_field(mapping, "session_id") + " = :session_id" if session_id else ""}
                ORDER BY distance
                LIMIT :limit
            """)
            
            params = {
                "query_vec": query_vec_str,
                "user_id": user_id,
                "limit": limit,
            }
            if session_id:
                params["session_id"] = session_id
            
            result = await session.execute(stmt, params)
            rows = result.fetchall()
            
            return [
                self._row_to_message(row, mapping)
                for row in rows
            ]
    
    async def _search_with_dynamic_handler(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """使用动态向量处理器进行检索"""
        # 先获取用户的所有消息
        all_messages = await self._get_user_messages(user_id, session_id, limit=200)
        
        if not all_messages:
            return []
        
        # 使用动态向量处理器检索
        from dki.cache.non_vectorized_handler import SearchResult
        
        results: List[SearchResult] = await self._vector_handler.search_relevant_messages(
            messages=all_messages,
            query=query,
            top_k=limit,
        )
        
        return [r.message for r in results]
    
    async def _search_with_keywords(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """使用关键词匹配进行检索"""
        mapping = self.adapter_config.messages
        table = self._tables["messages"]
        
        user_id_field = self._get_field(mapping, "user_id")
        content_field = self._get_field(mapping, "content")
        
        # 提取查询关键词
        keywords = query.lower().split()
        
        async with self._session_factory() as session:
            conditions = [table.c[user_id_field] == user_id]
            
            if session_id:
                session_id_field = self._get_field(mapping, "session_id")
                conditions.append(table.c[session_id_field] == session_id)
            
            # 关键词匹配
            if keywords and content_field in table.c:
                keyword_conditions = [
                    func.lower(table.c[content_field]).contains(kw)
                    for kw in keywords[:5]  # 最多使用 5 个关键词
                ]
                conditions.append(or_(*keyword_conditions))
            
            stmt = select(table).where(and_(*conditions)).limit(limit * 2)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            messages = [
                self._row_to_message(row, mapping)
                for row in rows
            ]
            
            # 简单的相关性排序
            def relevance_score(msg: ChatMessage) -> int:
                content_lower = msg.content.lower()
                return sum(1 for kw in keywords if kw in content_lower)
            
            messages.sort(key=relevance_score, reverse=True)
            
            return messages[:limit]
    
    async def _get_user_messages(
        self,
        user_id: str,
        session_id: Optional[str],
        limit: int = 200,
    ) -> List[ChatMessage]:
        """获取用户的消息"""
        mapping = self.adapter_config.messages
        table = self._tables["messages"]
        
        user_id_field = self._get_field(mapping, "user_id")
        timestamp_field = self._get_field(mapping, "timestamp")
        
        async with self._session_factory() as session:
            conditions = [table.c[user_id_field] == user_id]
            
            if session_id:
                session_id_field = self._get_field(mapping, "session_id")
                conditions.append(table.c[session_id_field] == session_id)
            
            stmt = select(table).where(and_(*conditions))
            
            if timestamp_field in table.c:
                stmt = stmt.order_by(desc(table.c[timestamp_field]))
            
            stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            return [
                self._row_to_message(row, mapping)
                for row in rows
            ]
    
    async def health_check(self) -> bool:
        """健康检查"""
        if not self._engine:
            return False
        
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    # ============ JSON 解析方法 ============
    
    def _extract_json_content(self, raw_content: str, json_key: Optional[str]) -> str:
        """
        从 JSON 字符串中提取实际内容
        
        场景: 上层应用直接存储 AI 原始响应，content 字段可能是 JSON 字符串
        例如: '{"text": "推荐川菜馆", "model": "gpt-4", "tokens": 100}'
        
        Args:
            raw_content: 原始内容 (可能是 JSON 字符串，也可能是普通文本)
            json_key: JSON key 路径，支持嵌套 (如 "text", "data.text", "choices.0.text")
            
        Returns:
            提取后的文本内容，如果解析失败则返回原始内容
        """
        if not json_key:
            return raw_content
        
        if not raw_content:
            return raw_content
        
        # 尝试解析 JSON
        try:
            import json
            data = json.loads(raw_content)
            
            # 支持嵌套 key，如 "data.text" 或 "choices.0.text"
            keys = json_key.split(".")
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list):
                    # 支持数组索引，如 "choices.0.text"
                    try:
                        index = int(key)
                        value = value[index] if 0 <= index < len(value) else None
                    except ValueError:
                        value = None
                else:
                    value = None
                
                if value is None:
                    # key 不存在，返回原始内容
                    logger.debug(
                        f"JSON key '{json_key}' not found in content, using raw content"
                    )
                    return raw_content
            
            # 确保返回字符串
            if isinstance(value, str):
                return value
            elif value is not None:
                # 如果提取的值不是字符串，转换为字符串
                return str(value)
            else:
                return raw_content
                
        except json.JSONDecodeError:
            # 不是有效的 JSON，返回原始内容
            # 这是正常情况，不需要警告
            return raw_content
        except Exception as e:
            # 其他错误，返回原始内容
            logger.debug(f"Failed to extract JSON content: {e}, using raw content")
            return raw_content
    
    # ============ 数据转换方法 ============
    
    def _row_to_preference(self, row, mapping: TableMapping) -> UserPreference:
        """将数据库行转换为 UserPreference"""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        # 获取原始 preference_text
        raw_text = row_dict.get(self._get_field(mapping, "preference_text"), "")
        
        # 如果配置了 content_json_key，尝试从 JSON 中提取实际内容
        preference_text = self._extract_json_content(raw_text, mapping.content_json_key)
        
        return UserPreference(
            user_id=row_dict.get(self._get_field(mapping, "user_id"), ""),
            preference_id=str(row_dict.get(self._get_field(mapping, "preference_id"), "")),
            preference_text=preference_text,
            preference_type=row_dict.get(self._get_field(mapping, "preference_type"), "general"),
            priority=row_dict.get(self._get_field(mapping, "priority"), 0),
            created_at=row_dict.get(self._get_field(mapping, "created_at")),
            updated_at=row_dict.get(self._get_field(mapping, "updated_at")),
        )
    
    def _row_to_message(self, row, mapping: TableMapping) -> ChatMessage:
        """将数据库行转换为 ChatMessage"""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        # 获取原始 content
        raw_content = row_dict.get(self._get_field(mapping, "content"), "")
        
        # 如果配置了 content_json_key，尝试从 JSON 中提取实际内容
        # 这处理了上层应用直接存储 AI 原始响应的场景
        content = self._extract_json_content(raw_content, mapping.content_json_key)
        
        return ChatMessage(
            message_id=str(row_dict.get(self._get_field(mapping, "message_id"), "")),
            session_id=str(row_dict.get(self._get_field(mapping, "session_id"), "")),
            user_id=row_dict.get(self._get_field(mapping, "user_id"), ""),
            role=row_dict.get(self._get_field(mapping, "role"), "user"),
            content=content,
            timestamp=row_dict.get(self._get_field(mapping, "timestamp"), datetime.utcnow()),
            embedding=row_dict.get(self._get_field(mapping, "embedding")),
        )
    
    def _row_to_user_profile(self, row, mapping: TableMapping) -> UserProfile:
        """将数据库行转换为 UserProfile"""
        row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
        
        return UserProfile(
            user_id=row_dict.get(self._get_field(mapping, "user_id"), ""),
            username=row_dict.get(self._get_field(mapping, "username")),
            display_name=row_dict.get(self._get_field(mapping, "display_name")),
            email=row_dict.get(self._get_field(mapping, "email")),
            created_at=row_dict.get(self._get_field(mapping, "created_at")),
            updated_at=row_dict.get(self._get_field(mapping, "updated_at")),
        )
    
    # ============ 缓存方法 ============
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.adapter_config.cache_enabled:
            return None
        
        if key not in self._cache:
            return None
        
        # 检查过期
        timestamp = self._cache_timestamps.get(key)
        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self.adapter_config.cache_ttl:
                del self._cache[key]
                del self._cache_timestamps[key]
                return None
        
        return self._cache[key]
    
    def _set_cached(self, key: str, value: Any) -> None:
        """设置缓存"""
        if not self.adapter_config.cache_enabled:
            return
        
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.utcnow()
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        stats = {
            "connected": self._connected,
            "database_type": self.adapter_config.database.type.value,
            "tables_reflected": list(self._tables.keys()),
            "cache_size": len(self._cache),
            "vector_search_type": self.adapter_config.vector_search.type.value,
        }
        
        if self._vector_handler:
            stats["vector_handler_stats"] = self._vector_handler.get_stats()
        
        return stats
