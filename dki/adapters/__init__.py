"""
DKI Adapters - 外部数据适配器

核心设计:
- 配置驱动: 上层应用只需提供配置文件，无需实现任何接口
- SQLAlchemy 动态表映射: 根据配置动态连接任意表结构
- 向量检索集成: 支持 pgvector、FAISS、动态向量处理

使用方式:
```python
# 方式 1: 从配置文件创建 (推荐)
from dki.adapters import ConfigDrivenAdapter

adapter = ConfigDrivenAdapter.from_yaml("config/adapter_config.yaml")
await adapter.connect()

# 方式 2: 从配置字典创建
adapter = ConfigDrivenAdapter.from_dict({
    "database": {"type": "postgresql", "host": "localhost", ...},
    "preferences": {"table": "user_preferences", "fields": {...}},
    "messages": {"table": "chat_messages", "fields": {...}},
})
await adapter.connect()
```

Author: AGI Demo Project
Version: 2.0.0
"""

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    UserPreference,
    ChatMessage,
)

from dki.adapters.config_driven_adapter import (
    ConfigDrivenAdapter,
    ConfigDrivenAdapterConfig,
    DatabaseConfig,
    DatabaseType,
    TableMapping,
    VectorSearchConfig,
    VectorSearchType,
)

from dki.adapters.example_adapter import ExampleAdapter

__all__ = [
    # 基础接口
    "IUserDataAdapter",
    "AdapterConfig",
    "AdapterType",
    "UserProfile",
    "UserPreference",
    "ChatMessage",
    
    # 配置驱动适配器 (推荐)
    "ConfigDrivenAdapter",
    "ConfigDrivenAdapterConfig",
    "DatabaseConfig",
    "DatabaseType",
    "TableMapping",
    "VectorSearchConfig",
    "VectorSearchType",
    
    # 示例适配器 (仅用于演示)
    "ExampleAdapter",
]
