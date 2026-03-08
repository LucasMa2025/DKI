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
class VectorIndexCoreConfig:
    """向量索引核心配置"""
    index_type: str = "HNSW"           # HNSW | FLAT | IVF
    dimension: int = 768               # 向量维度
    vector_data_type: str = "float32"  # float32 | float16
    similarity_metric: str = "cosine"  # cosine | l2 | ip (inner product)


@dataclass
class VectorIndexEmbeddingConfig:
    """向量索引 Embedding 配置"""
    api_type: str = "local"            # openai | local | aliyun | pinecone | custom
    model_name: str = "all-MiniLM-L6-v2"
    api_endpoint: Optional[str] = None  # 远程 API 地址
    api_key: Optional[str] = None       # API 密钥 (本地模型填 "local")
    normalization: bool = True          # 是否归一化向量


@dataclass
class VectorIndexRetrievalConfig:
    """向量索引检索配置"""
    top_k: int = 10
    index_file_path: Optional[str] = None  # 本地索引文件路径


@dataclass
class VectorIndexMetadataConfig:
    """向量索引元数据配置"""
    id_mapping_table: str = "vector_id_mapping"
    primary_key: str = "vector_id"


@dataclass
class VectorIndexConfig:
    """
    完整的向量索引配置
    
    当外部消息系统提供此配置时，DKI 启用语义检索 (BM25 + Embedding)。
    未提供此配置时，DKI 仅使用 BM25 召回。
    
    配置示例:
    ```yaml
    vector_index_config:
      core:
        index_type: HNSW
        dimension: 768
        vector_data_type: float32
        similarity_metric: cosine
      embedding:
        api_type: openai
        model_name: text-embedding-ada-002
        api_endpoint: https://api.openai.com/v1/embeddings
        api_key: your-api-key
        normalization: true
      retrieval:
        top_k: 10
        index_file_path: ./vector_index.index
      metadata:
        id_mapping_table: vector_id_mapping
        primary_key: vector_id
    ```
    """
    core: VectorIndexCoreConfig = field(default_factory=VectorIndexCoreConfig)
    embedding: VectorIndexEmbeddingConfig = field(default_factory=VectorIndexEmbeddingConfig)
    retrieval: VectorIndexRetrievalConfig = field(default_factory=VectorIndexRetrievalConfig)
    metadata: VectorIndexMetadataConfig = field(default_factory=VectorIndexMetadataConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorIndexConfig":
        """从字典创建配置"""
        config = cls()
        if "core" in data:
            c = data["core"]
            config.core = VectorIndexCoreConfig(
                index_type=c.get("index_type", "HNSW"),
                dimension=c.get("dimension", 768),
                vector_data_type=c.get("vector_data_type", "float32"),
                similarity_metric=c.get("similarity_metric", "cosine"),
            )
        if "embedding" in data:
            e = data["embedding"]
            config.embedding = VectorIndexEmbeddingConfig(
                api_type=e.get("api_type", "local"),
                model_name=e.get("model_name", "all-MiniLM-L6-v2"),
                api_endpoint=e.get("api_endpoint"),
                api_key=e.get("api_key"),
                normalization=e.get("normalization", True),
            )
        if "retrieval" in data:
            r = data["retrieval"]
            config.retrieval = VectorIndexRetrievalConfig(
                top_k=r.get("top_k", 10),
                index_file_path=r.get("index_file_path"),
            )
        if "metadata" in data:
            m = data["metadata"]
            config.metadata = VectorIndexMetadataConfig(
                id_mapping_table=m.get("id_mapping_table", "vector_id_mapping"),
                primary_key=m.get("primary_key", "vector_id"),
            )
        return config


@dataclass
class VectorSearchConfig:
    """
    向量检索配置
    
    核心逻辑:
    - 如果提供了 vector_index_config → 启用语义检索 (BM25 + Embedding)
    - 如果未提供 vector_index_config → 仅使用 BM25 召回
    - type 字段控制已有预计算向量的检索方式 (pgvector/faiss)
    - type=dynamic + vector_index_config → DKI 内部做 BM25 + Embedding
    """
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
    
    # ============ v7.0: 向量索引配置 (核心新增) ============
    # 外部消息系统提供此配置时才启用语义检索
    # 未提供时仅使用 BM25 召回
    vector_index_config: Optional[VectorIndexConfig] = None
    
    @property
    def has_vector_capability(self) -> bool:
        """
        判断是否具备向量检索能力
        
        条件:
        1. vector_search.enabled = True
        2. 提供了 vector_index_config (含 embedding 配置)
        3. 或 type=pgvector 且有 embedding_field (上层 DB 已有预计算向量)
        """
        if not self.enabled:
            return False
        # 情况 1: 有完整的 vector_index_config
        if self.vector_index_config is not None:
            return True
        # 情况 2: pgvector 模式且有 embedding_field (上层已有预计算向量)
        if self.type == VectorSearchType.PGVECTOR and self.embedding_field:
            return True
        return False


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
            
            # v7.0: 解析 vector_index_config
            vic = None
            if "vector_index_config" in vs_data:
                vic = VectorIndexConfig.from_dict(vs_data["vector_index_config"])
            
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
                vector_index_config=vic,
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
        self._bm25_only_mode = False  # v7.0: 无 vector_index_config 时降级为 BM25-only
        
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
    
    def _build_active_session_join(
        self,
        msg_table: "Table",
        mapping: "TableMapping",
    ):
        """
        构建 JOIN sessions 表 + is_active 过滤的条件。
        
        当 sessions 表已配置且已反射时，返回 (join_clause, filter_condition)；
        否则返回 (None, None)，调用方无需添加任何额外条件。
        
        设计原则: 软删除会话后，所有消息检索路径都应排除已删除会话的消息。
        这是一个通用辅助方法，供 _get_user_messages / _search_with_keywords /
        _search_with_pgvector / get_session_history 共用。
        """
        if (
            not self.adapter_config.sessions
            or "sessions" not in self._tables
        ):
            return None, None
        
        sess_table = self._tables["sessions"]
        sess_mapping = self.adapter_config.sessions
        
        # 消息表的 session_id 字段
        msg_session_id_field = self._get_field(mapping, "session_id")
        # 会话表的 session_id 字段 (主键)
        sess_id_field = self._get_field(sess_mapping, "session_id")
        
        if msg_session_id_field not in msg_table.c or sess_id_field not in sess_table.c:
            logger.debug(
                f"Cannot build session join: "
                f"msg.{msg_session_id_field} or sess.{sess_id_field} not found"
            )
            return None, None
        
        # JOIN 条件
        join_clause = msg_table.c[msg_session_id_field] == sess_table.c[sess_id_field]
        
        # is_active 过滤 — 优先使用 sessions mapping 的 filters 配置,
        # 否则默认检查 is_active 字段
        active_filter = None
        if "is_active" in sess_table.c:
            active_filter = sess_table.c["is_active"] == True
        
        return join_clause, active_filter
    
    async def _init_vector_handler(self) -> None:
        """
        初始化向量处理器
        
        v7.0 核心逻辑:
        - 有 vector_index_config → 创建真实 EmbeddingService + NonVectorizedDataHandler
        - 无 vector_index_config 且 type=PGVECTOR + embedding_field → 仅创建 EmbeddingService (query embedding)
        - 无 vector_index_config 且 type=DYNAMIC → 仅 BM25 (不创建向量处理器)
        - 无 vector_index_config 且 type=NONE → 纯关键词
        """
        vs_config = self.adapter_config.vector_search
        
        if not vs_config.enabled:
            logger.info("Vector search disabled, using keyword-only retrieval")
            return
        
        # ============ v7.0: 基于 vector_index_config 判断是否启用语义检索 ============
        if not vs_config.has_vector_capability:
            logger.info(
                "No vector_index_config provided and no pgvector embedding_field, "
                "falling back to BM25-only retrieval"
            )
            # 仍然创建 BM25-only handler (不需要 embedding_service)
            if vs_config.type == VectorSearchType.DYNAMIC:
                self._bm25_only_mode = True
                logger.info("BM25-only mode enabled for DYNAMIC type (no vector_index_config)")
            return
        
        # ============ 有向量能力: 创建 EmbeddingService ============
        self._embedding_service = await self._create_embedding_service()
        
        if not self._embedding_service:
            logger.warning("Failed to create embedding service, falling back to BM25-only")
            self._bm25_only_mode = True
            return
        
        # ============ DYNAMIC 模式: 创建 NonVectorizedDataHandler ============
        if vs_config.type == VectorSearchType.DYNAMIC:
            from dki.cache.non_vectorized_handler import (
                NonVectorizedDataHandler,
                HandlerConfig,
                SearchStrategy,
            )
            
            handler_config = HandlerConfig(
                default_strategy=SearchStrategy(vs_config.dynamic_strategy),
                cache_embeddings=True,
            )
            self._vector_handler = NonVectorizedDataHandler(
                embedding_service=self._embedding_service,
                config=handler_config,
            )
            logger.info(
                f"Dynamic vector handler initialized "
                f"(strategy={vs_config.dynamic_strategy}, "
                f"dim={vs_config.vector_index_config.core.dimension if vs_config.vector_index_config else vs_config.embedding_dim})"
            )
        elif vs_config.type == VectorSearchType.PGVECTOR:
            logger.info(
                f"PGVECTOR mode: using pre-computed embeddings from DB "
                f"(field={vs_config.embedding_field}, dim={vs_config.embedding_dim})"
            )
    
    async def _create_embedding_service(self):
        """
        v7.0: 基于 vector_index_config 创建真实 Embedding 服务
        
        路由逻辑:
        1. vector_index_config.embedding.api_type == "local" → LocalEmbeddingService
        2. vector_index_config.embedding.api_type == "openai"/"aliyun"/... → RemoteEmbeddingService
        3. 回退: 使用旧的 embedding_model + embedding_api_url 配置
        """
        vs_config = self.adapter_config.vector_search
        vic = vs_config.vector_index_config
        
        # ============ 优先使用 vector_index_config ============
        if vic:
            emb_cfg = vic.embedding
            dimension = vic.core.dimension
            normalize = emb_cfg.normalization
            
            if emb_cfg.api_type == "local" or (emb_cfg.api_key and emb_cfg.api_key.lower() == "local"):
                # 本地模型
                return self._create_local_embedding_service(
                    model_name=emb_cfg.model_name,
                    dimension=dimension,
                    normalize=normalize,
                )
            else:
                # 远程 API (openai / aliyun / pinecone / custom)
                return self._create_remote_embedding_service(
                    api_type=emb_cfg.api_type,
                    model_name=emb_cfg.model_name,
                    api_endpoint=emb_cfg.api_endpoint,
                    api_key=emb_cfg.api_key,
                    dimension=dimension,
                    normalize=normalize,
                )
        
        # ============ 回退: 使用旧配置 (向后兼容) ============
        if vs_config.embedding_api_url or vs_config.embedding_api_key:
            return self._create_remote_embedding_service(
                api_type="openai",
                model_name=vs_config.embedding_model,
                api_endpoint=vs_config.embedding_api_url,
                api_key=vs_config.embedding_api_key,
                dimension=vs_config.embedding_dim,
                normalize=True,
            )
        
        # 本地模型回退
        return self._create_local_embedding_service(
            model_name=vs_config.embedding_model,
            dimension=vs_config.embedding_dim,
            normalize=True,
        )
    
    def _create_remote_embedding_service(
        self,
        api_type: str,
        model_name: str,
        api_endpoint: Optional[str],
        api_key: Optional[str],
        dimension: int,
        normalize: bool,
    ):
        """创建远程 Embedding 服务"""
        
        class RemoteEmbeddingService:
            """
            调用远程 Embedding API
            
            支持 OpenAI 兼容格式的 API (OpenAI / Azure / 阿里云 / 自建服务)
            """
            
            def __init__(self, api_type: str, model: str, api_url: str,
                         api_key: str, dim: int, normalize: bool):
                self.api_type = api_type
                self.model = model
                self.api_url = api_url
                self.api_key = api_key
                self.dim = dim
                self.normalize = normalize
            
            def embed(self, text: str) -> List[float]:
                """计算单个文本的 embedding"""
                import httpx
                
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                headers["Content-Type"] = "application/json"
                
                try:
                    resp = httpx.post(
                        self.api_url,
                        headers=headers,
                        json={"input": text, "model": self.model},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # OpenAI 兼容格式: {"data": [{"embedding": [...]}]}
                    embedding = data["data"][0]["embedding"]
                    
                    if self.normalize:
                        embedding = self._normalize_vector(embedding)
                    
                    return embedding
                except Exception as e:
                    logger.error(f"Remote embedding failed: {e}")
                    raise
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                """批量计算 embedding"""
                import httpx
                
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                headers["Content-Type"] = "application/json"
                
                try:
                    resp = httpx.post(
                        self.api_url,
                        headers=headers,
                        json={"input": texts, "model": self.model},
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    
                    embeddings = [d["embedding"] for d in data["data"]]
                    
                    if self.normalize:
                        embeddings = [self._normalize_vector(e) for e in embeddings]
                    
                    return embeddings
                except Exception as e:
                    logger.error(f"Remote batch embedding failed: {e}")
                    raise
            
            @staticmethod
            def _normalize_vector(vec: List[float]) -> List[float]:
                import math
                norm = math.sqrt(sum(x * x for x in vec))
                if norm == 0:
                    return vec
                return [x / norm for x in vec]
        
        # 默认 API endpoint
        default_endpoints = {
            "openai": "https://api.openai.com/v1/embeddings",
            "aliyun": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        }
        url = api_endpoint or default_endpoints.get(api_type, "https://api.openai.com/v1/embeddings")
        
        logger.info(f"Creating RemoteEmbeddingService (api_type={api_type}, model={model_name}, dim={dimension})")
        
        return RemoteEmbeddingService(
            api_type=api_type,
            model=model_name,
            api_url=url,
            api_key=api_key or "",
            dim=dimension,
            normalize=normalize,
        )
    
    def _create_local_embedding_service(
        self,
        model_name: str,
        dimension: int,
        normalize: bool,
    ):
        """创建本地 Embedding 服务 (sentence-transformers)"""
        
        class LocalEmbeddingService:
            """
            使用本地 sentence-transformers 模型
            
            DKI 自行加载模型，不依赖上层应用
            """
            
            def __init__(self, model_name: str, dim: int, normalize: bool):
                self.model_name = model_name
                self.dim = dim
                self.normalize = normalize
                self._model = None
            
            def _ensure_model(self):
                """延迟加载模型"""
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self.model_name)
                        logger.info(f"Loaded local embedding model: {self.model_name}")
                    except ImportError:
                        logger.error(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        )
                        raise
            
            def embed(self, text: str) -> List[float]:
                """计算单个文本的 embedding"""
                self._ensure_model()
                embedding = self._model.encode(
                    text, normalize_embeddings=self.normalize
                )
                return embedding.tolist()
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                """批量计算 embedding"""
                self._ensure_model()
                embeddings = self._model.encode(
                    texts, normalize_embeddings=self.normalize
                )
                return embeddings.tolist()
        
        logger.info(f"Creating LocalEmbeddingService (model={model_name}, dim={dimension})")
        
        return LocalEmbeddingService(
            model_name=model_name,
            dim=dimension,
            normalize=normalize,
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
        """获取会话历史 (过滤已软删除会话)"""
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
            
            # JOIN sessions 表过滤已软删除会话
            join_clause, active_filter = self._build_active_session_join(table, mapping)
            
            stmt = select(table)
            if join_clause is not None:
                sess_table = self._tables["sessions"]
                stmt = stmt.select_from(table.join(sess_table, join_clause))
                if active_filter is not None:
                    conditions.append(active_filter)
            
            stmt = stmt.where(and_(*conditions))
            
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
        
        v7.0 路由逻辑 (核心变更):
        1. 有 vector_index_config + PGVECTOR → pgvector 向量检索
        2. 有 vector_index_config + DYNAMIC + vector_handler → BM25 + Embedding 混合检索
        3. 无 vector_index_config (BM25-only) → 仅 BM25 召回
        4. type=NONE 或 disabled → 简单关键词匹配
        
        关键原则: 外部消息系统未提供向量化配置时，不做语义检索
        """
        if not self.adapter_config.messages or "messages" not in self._tables:
            return []
        
        vs_config = self.adapter_config.vector_search
        
        # ============ v7.0: 检查向量能力 ============
        if not vs_config.enabled:
            return await self._search_with_keywords(user_id, query, limit, session_id)
        
        # 有向量能力: 使用对应的向量检索
        if vs_config.has_vector_capability:
            if vs_config.type == VectorSearchType.PGVECTOR:
                return await self._search_with_pgvector(user_id, query, limit, session_id)
            elif vs_config.type == VectorSearchType.DYNAMIC and self._vector_handler:
                return await self._search_with_dynamic_handler(user_id, query, limit, session_id)
        
        # 无向量能力但 type=DYNAMIC: 使用 BM25-only
        if self._bm25_only_mode or vs_config.type == VectorSearchType.DYNAMIC:
            return await self._search_with_bm25_only(user_id, query, limit, session_id)
        
        # 最终回退: 关键词匹配
        return await self._search_with_keywords(user_id, query, limit, session_id)
    
    async def _search_with_pgvector(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """使用 pgvector 进行向量检索 (过滤已软删除会话)"""
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
            from sqlalchemy import text
            
            query_vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # 构建 SQL — 如果有 sessions 表则 JOIN 过滤已删除会话
            session_id_col = self._get_field(mapping, "session_id")
            has_session_join = (
                self.adapter_config.sessions
                and "sessions" in self._tables
            )
            
            if has_session_join:
                sess_mapping = self.adapter_config.sessions
                sess_table_name = sess_mapping.table
                sess_id_field = self._get_field(sess_mapping, "session_id")
                
                join_clause = f"JOIN {sess_table_name} s ON m.{session_id_col} = s.{sess_id_field}"
                active_clause = "AND s.is_active = true" if "is_active" in self._tables["sessions"].c else ""
                
                stmt = text(f"""
                    SELECT m.*, (m.{embedding_field} <=> :query_vec::vector) as distance
                    FROM {mapping.table} m
                    {join_clause}
                    WHERE m.{user_id_field} = :user_id
                    {active_clause}
                    {"AND m." + session_id_col + " = :session_id" if session_id else ""}
                    ORDER BY distance
                    LIMIT :limit
                """)
            else:
                stmt = text(f"""
                    SELECT *, ({embedding_field} <=> :query_vec::vector) as distance
                    FROM {mapping.table}
                    WHERE {user_id_field} = :user_id
                    {"AND " + session_id_col + " = :session_id" if session_id else ""}
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
    
    async def _search_with_bm25_only(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """
        v7.0: 仅使用 BM25 进行检索 (无向量化配置时的降级路径)
        
        v7.1 改进:
        - BM25 分词改用 jieba (有效匹配 "纠正", "作者" 等中文词)
        - 过滤 score=0 的消息 (避免返回无关消息)
        - 当 BM25 无结果时回退到最近消息 (保证基本可用)
        
        与 _search_with_keywords 的区别:
        - BM25 使用 TF-IDF 加权的词频匹配，考虑文档长度归一化
        - _search_with_keywords 仅做简单的关键词包含匹配
        
        BM25 更适合作为无 Embedding 时的主要召回策略
        """
        # 获取用户消息
        all_messages = await self._get_user_messages(user_id, session_id, limit=200)
        
        if not all_messages:
            return []
        
        # 使用 BM25 评分
        scored_messages = self._bm25_score(query, all_messages)
        
        # 按 BM25 分数排序, 过滤 score=0 的消息 (v7.1)
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        relevant = [(msg, score) for msg, score in scored_messages if score > 0]
        
        if relevant:
            logger.debug(
                f"BM25 recall: {len(relevant)} messages with score > 0 "
                f"(top score={relevant[0][1]:.3f})"
            )
            return [msg for msg, score in relevant[:limit]]
        
        # BM25 无结果时回退: 返回最近的消息 (按时间倒序, 已由 _get_user_messages 排序)
        logger.info(
            f"BM25 recall: no messages scored > 0 for query '{query[:50]}...', "
            f"falling back to {min(limit, 5)} most recent messages"
        )
        return all_messages[:min(limit, 5)]
    
    # ============ BM25 中文停用词表 (高频无信息量词) ============
    _CN_STOPWORDS = frozenset({
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        '什么', '吗', '呢', '吧', '啊', '哦', '嗯', '呀', '哈', '哪', '嘛',
        '可以', '没', '还', '对', '把', '让', '被', '从', '给', '用', '但',
        '而', '又', '所以', '因为', '如果', '这个', '那个', '怎么', '为什么',
        '哪个', '多少', '几', '谁', '怎样', '这样', '那样',
    })
    
    def _bm25_score(
        self,
        query: str,
        messages: List["ChatMessage"],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[tuple]:
        """
        BM25 评分 (v7.1: 改进中文分词 + 停用词过滤)
        
        改进点:
        1. 优先使用 jieba 分词 (比单字+bigram 更准确)
        2. 过滤中文停用词 (避免高频无信息量词稀释权重)
        3. 保留 bigram 作为 jieba 不可用时的回退
        
        参数:
        - k1: 词频饱和参数 (默认 1.5)
        - b: 文档长度归一化参数 (默认 0.75)
        
        返回: [(message, score), ...]
        """
        import math
        import re
        
        # 尝试使用 jieba 分词 (更准确的中文分词)
        try:
            import jieba
            _jieba_available = True
        except ImportError:
            _jieba_available = False
        
        def tokenize(text: str) -> List[str]:
            """
            中英文混合分词 (v7.1 改进)
            
            策略:
            - 有 jieba: jieba 分词 + 英文单词 + 停用词过滤
            - 无 jieba: 单字 + bigram + 英文单词 + 停用词过滤
            """
            tokens = []
            text_lower = text.lower()
            
            # 英文单词 (保持不变)
            en_tokens = re.findall(r'[a-zA-Z0-9]+', text_lower)
            tokens.extend(en_tokens)
            
            if _jieba_available:
                # jieba 分词: 产出有语义的词组 (如 "纠正", "作者", "挪威")
                cn_text = re.sub(r'[a-zA-Z0-9]+', ' ', text_lower)  # 移除英文
                words = jieba.lcut(cn_text)
                for w in words:
                    w = w.strip()
                    if len(w) >= 1 and any('\u4e00' <= c <= '\u9fff' for c in w):
                        # 过滤停用词
                        if w not in self._CN_STOPWORDS:
                            tokens.append(w)
            else:
                # 回退: 单字 + bigram (过滤停用词)
                cn_chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
                for i in range(len(cn_chars)):
                    if cn_chars[i] not in self._CN_STOPWORDS:
                        tokens.append(cn_chars[i])
                    if i + 1 < len(cn_chars):
                        bigram = cn_chars[i] + cn_chars[i + 1]
                        if bigram not in self._CN_STOPWORDS:
                            tokens.append(bigram)
            
            return tokens
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return [(msg, 0.0) for msg in messages]
        
        # 文档分词
        doc_tokens_list = [tokenize(msg.content) for msg in messages]
        
        # 计算平均文档长度
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)
        
        # 计算 IDF
        N = len(messages)
        idf = {}
        for qt in set(query_tokens):
            df = sum(1 for dt in doc_tokens_list if qt in dt)
            idf[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        # 计算每个文档的 BM25 分数
        results = []
        for msg, doc_tokens in zip(messages, doc_tokens_list):
            score = 0.0
            dl = len(doc_tokens)
            
            # 词频统计
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                score += idf.get(qt, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            
            results.append((msg, score))
        
        return results
    
    async def _search_with_keywords(
        self,
        user_id: str,
        query: str,
        limit: int,
        session_id: Optional[str],
    ) -> List[ChatMessage]:
        """使用关键词匹配进行检索 (过滤已软删除会话)"""
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
            
            # JOIN sessions 表过滤已软删除会话
            join_clause, active_filter = self._build_active_session_join(table, mapping)
            
            stmt = select(table)
            if join_clause is not None:
                sess_table = self._tables["sessions"]
                stmt = stmt.select_from(table.join(sess_table, join_clause))
                if active_filter is not None:
                    conditions.append(active_filter)
            
            stmt = stmt.where(and_(*conditions)).limit(limit * 2)
            
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
        """获取用户的消息 (过滤已软删除会话)"""
        mapping = self.adapter_config.messages
        table = self._tables["messages"]
        
        user_id_field = self._get_field(mapping, "user_id")
        timestamp_field = self._get_field(mapping, "timestamp")
        
        async with self._session_factory() as session:
            conditions = [table.c[user_id_field] == user_id]
            
            if session_id:
                session_id_field = self._get_field(mapping, "session_id")
                conditions.append(table.c[session_id_field] == session_id)
            
            # JOIN sessions 表过滤已软删除会话
            join_clause, active_filter = self._build_active_session_join(table, mapping)
            
            stmt = select(table)
            if join_clause is not None:
                sess_table = self._tables["sessions"]
                stmt = stmt.select_from(table.join(sess_table, join_clause))
                if active_filter is not None:
                    conditions.append(active_filter)
            
            stmt = stmt.where(and_(*conditions))
            
            if timestamp_field in table.c:
                stmt = stmt.order_by(desc(table.c[timestamp_field]))
            
            stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            return [
                self._row_to_message(row, mapping)
                for row in rows
            ]
    
    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[ChatMessage]:
        """
        获取用户最近的消息 (跨会话, 按时间降序后反转为正序)
        
        直接复用 _get_user_messages (session_id=None) 实现跨会话近轮获取
        """
        messages = await self._get_user_messages(
            user_id=user_id,
            session_id=None,  # 跨会话
            limit=limit,
        )
        # _get_user_messages 返回时间降序, 反转为正序 (最旧在前)
        messages.reverse()
        return messages
    
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
