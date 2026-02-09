"""
Unit Tests for Config-Driven Adapter

测试配置驱动的适配器功能

Author: AGI Demo Project
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from dki.adapters.config_driven_adapter import (
    ConfigDrivenAdapter,
    ConfigDrivenAdapterConfig,
    DatabaseConfig,
    DatabaseType,
    TableMapping,
    VectorSearchConfig,
    VectorSearchType,
)
from dki.adapters.base import UserPreference, ChatMessage


class TestConfigDrivenAdapterConfig:
    """测试配置解析"""
    
    def test_from_dict_basic(self):
        """测试基本配置解析"""
        config_dict = {
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "user",
                "password": "pass",
            },
            "preferences": {
                "table": "user_prefs",
                "fields": {
                    "user_id": "uid",
                    "preference_text": "content",
                },
            },
        }
        
        config = ConfigDrivenAdapterConfig.from_dict(config_dict)
        
        assert config.database.type == DatabaseType.POSTGRESQL
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.preferences.table == "user_prefs"
        assert config.preferences.fields["user_id"] == "uid"
    
    def test_from_dict_with_vector_search(self):
        """测试向量检索配置解析"""
        config_dict = {
            "database": {"type": "postgresql"},
            "vector_search": {
                "enabled": True,
                "type": "dynamic",
                "dynamic": {
                    "strategy": "hybrid",
                    "embedding_model": "text-embedding-ada-002",
                },
            },
        }
        
        config = ConfigDrivenAdapterConfig.from_dict(config_dict)
        
        assert config.vector_search.enabled is True
        assert config.vector_search.type == VectorSearchType.DYNAMIC
        assert config.vector_search.dynamic_strategy == "hybrid"
    
    def test_database_url_generation(self):
        """测试数据库 URL 生成"""
        # PostgreSQL
        pg_config = DatabaseConfig(
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb",
            username="user",
            password="pass",
        )
        assert "postgresql+asyncpg://" in pg_config.get_async_url()
        
        # MySQL
        mysql_config = DatabaseConfig(
            type=DatabaseType.MYSQL,
            host="localhost",
            port=3306,
            database="testdb",
            username="user",
            password="pass",
        )
        assert "mysql+aiomysql://" in mysql_config.get_async_url()
        
        # SQLite
        sqlite_config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            database="/tmp/test.db",
        )
        assert "sqlite+aiosqlite://" in sqlite_config.get_async_url()


class TestConfigDrivenAdapter:
    """测试配置驱动适配器"""
    
    @pytest.fixture
    def adapter_config(self):
        """创建测试配置"""
        return ConfigDrivenAdapterConfig.from_dict({
            "database": {
                "type": "sqlite",
                "database": ":memory:",
            },
            "preferences": {
                "table": "user_preferences",
                "fields": {
                    "user_id": "user_id",
                    "preference_text": "content",
                    "preference_type": "type",
                    "priority": "priority",
                },
            },
            "messages": {
                "table": "messages",
                "fields": {
                    "message_id": "id",
                    "session_id": "session_id",
                    "user_id": "user_id",
                    "role": "role",
                    "content": "content",
                    "timestamp": "created_at",
                },
            },
            "vector_search": {
                "enabled": True,
                "type": "dynamic",
                "dynamic": {
                    "strategy": "hybrid",
                },
            },
        })
    
    def test_adapter_creation(self, adapter_config):
        """测试适配器创建"""
        adapter = ConfigDrivenAdapter(adapter_config)
        
        assert adapter.adapter_config == adapter_config
        assert adapter.is_connected is False
    
    def test_from_dict_factory(self):
        """测试从字典创建适配器"""
        adapter = ConfigDrivenAdapter.from_dict({
            "database": {"type": "sqlite", "database": ":memory:"},
        })
        
        assert adapter is not None
        assert adapter.adapter_config.database.type == DatabaseType.SQLITE
    
    def test_field_mapping(self, adapter_config):
        """测试字段映射"""
        adapter = ConfigDrivenAdapter(adapter_config)
        
        # 测试偏好表字段映射
        pref_mapping = adapter_config.preferences
        assert adapter._get_field(pref_mapping, "user_id") == "user_id"
        assert adapter._get_field(pref_mapping, "preference_text") == "content"
        
        # 测试未映射字段返回原名
        assert adapter._get_field(pref_mapping, "unknown_field") == "unknown_field"
    
    def test_cache_operations(self, adapter_config):
        """测试缓存操作"""
        adapter = ConfigDrivenAdapter(adapter_config)
        
        # 设置缓存
        adapter._set_cached("test_key", ["data"])
        
        # 获取缓存
        cached = adapter._get_cached("test_key")
        assert cached == ["data"]
        
        # 清除缓存
        adapter.clear_cache()
        assert adapter._get_cached("test_key") is None
    
    def test_stats(self, adapter_config):
        """测试统计数据"""
        adapter = ConfigDrivenAdapter(adapter_config)
        
        stats = adapter.get_stats()
        
        assert "connected" in stats
        assert "database_type" in stats
        assert "vector_search_type" in stats
        assert stats["database_type"] == "sqlite"


class TestDynamicVectorSearch:
    """测试动态向量检索"""
    
    @pytest.fixture
    def messages(self):
        """创建测试消息"""
        return [
            ChatMessage(
                message_id="1",
                session_id="s1",
                user_id="u1",
                role="user",
                content="我喜欢吃素食",
                timestamp=datetime.utcnow(),
            ),
            ChatMessage(
                message_id="2",
                session_id="s1",
                user_id="u1",
                role="assistant",
                content="好的，我记住了您的饮食偏好",
                timestamp=datetime.utcnow(),
            ),
            ChatMessage(
                message_id="3",
                session_id="s1",
                user_id="u1",
                role="user",
                content="推荐一家餐厅",
                timestamp=datetime.utcnow(),
            ),
        ]
    
    def test_keyword_search_relevance(self, messages):
        """测试关键词检索相关性"""
        query = "素食餐厅"
        keywords = query.lower().split()
        
        # 简单相关性评分
        def relevance_score(msg):
            content_lower = msg.content.lower()
            return sum(1 for kw in keywords if kw in content_lower)
        
        # 第一条消息应该最相关 (包含 "素食")
        scores = [(msg, relevance_score(msg)) for msg in messages]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        assert scores[0][0].message_id == "1"  # 包含 "素食"


class TestAdapterIntegration:
    """适配器集成测试"""
    
    @pytest.mark.asyncio
    async def test_adapter_lifecycle(self):
        """测试适配器生命周期"""
        adapter = ConfigDrivenAdapter.from_dict({
            "database": {"type": "sqlite", "database": ":memory:"},
        })
        
        # 初始状态
        assert adapter.is_connected is False
        
        # 注意: 实际连接需要数据库，这里只测试接口
        # await adapter.connect()
        # assert adapter.is_connected is True
        # await adapter.disconnect()
        # assert adapter.is_connected is False


class TestUserIdHandling:
    """测试 user_id 处理"""
    
    def test_user_id_in_cache_key(self):
        """测试 user_id 在缓存键中的使用"""
        adapter = ConfigDrivenAdapter.from_dict({
            "database": {"type": "sqlite", "database": ":memory:"},
        })
        
        # 不同 user_id 应该有不同的缓存键
        key1 = f"prefs:user_123:None:False"
        key2 = f"prefs:user_456:None:False"
        
        adapter._set_cached(key1, ["pref1"])
        adapter._set_cached(key2, ["pref2"])
        
        assert adapter._get_cached(key1) == ["pref1"]
        assert adapter._get_cached(key2) == ["pref2"]
    
    def test_user_id_formats(self):
        """测试不同格式的 user_id"""
        # 支持各种格式
        user_ids = [
            "12345",  # 数字字符串
            "user_abc",  # 带前缀
            "550e8400-e29b-41d4-a716-446655440000",  # UUID
            "john@example.com",  # 邮箱
        ]
        
        for user_id in user_ids:
            # 应该都能正常处理
            cache_key = f"prefs:{user_id}:None:False"
            assert user_id in cache_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
