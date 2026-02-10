"""
Redis Cache Unit Tests
测试 Redis 分布式缓存功能

Author: AGI Demo Project
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from dki.cache.redis_client import (
    DKIRedisClient,
    RedisConfig,
    REDIS_AVAILABLE,
)
from dki.cache.preference_cache import (
    PreferenceCacheManager,
    CacheConfig,
    CacheTier,
)


class TestRedisConfig:
    """测试 Redis 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RedisConfig()
        
        assert config.enabled == False
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.max_connections == 50
        assert config.enable_compression == True
    
    def test_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "enabled": True,
            "host": "redis.example.com",
            "port": 6380,
            "password": "secret",
            "db": 1,
            "max_connections": 100,
        }
        
        config = RedisConfig.from_dict(data)
        
        assert config.enabled == True
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.db == 1
        assert config.max_connections == 100


class TestDKIRedisClient:
    """测试 Redis 客户端"""
    
    def test_not_available_when_disabled(self):
        """测试禁用时不可用"""
        config = RedisConfig(enabled=False)
        client = DKIRedisClient(config)
        
        assert client.is_available == False
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis library not installed")
    @pytest.mark.asyncio
    async def test_connect_disabled(self):
        """测试禁用时连接返回 False"""
        config = RedisConfig(enabled=False)
        client = DKIRedisClient(config)
        
        result = await client.connect()
        
        assert result == False
        assert client.is_available == False
    
    def test_make_key(self):
        """测试键生成"""
        config = RedisConfig(key_prefix="test")
        client = DKIRedisClient(config)
        
        key = client._make_key("user:123")
        
        assert key == "test:user:123"
    
    def test_compress_small_data(self):
        """测试小数据不压缩"""
        config = RedisConfig(
            enable_compression=True,
            compression_threshold=1024,
        )
        client = DKIRedisClient(config)
        
        small_data = b"small data"
        result = client._compress(small_data)
        
        # 小数据应该不压缩，只添加标记
        assert result[0:1] == b'\x00'
        assert result[1:] == small_data
    
    def test_compress_large_data(self):
        """测试大数据压缩"""
        config = RedisConfig(
            enable_compression=True,
            compression_threshold=100,
        )
        client = DKIRedisClient(config)
        
        large_data = b"x" * 200
        result = client._compress(large_data)
        
        # 大数据应该压缩
        assert result[0:1] == b'\x01'
        assert len(result) < len(large_data)
    
    def test_decompress(self):
        """测试解压缩"""
        config = RedisConfig(enable_compression=True)
        client = DKIRedisClient(config)
        
        original = b"test data"
        compressed = client._compress(original)
        decompressed = client._decompress(compressed)
        
        assert decompressed == original
    
    def test_get_stats(self):
        """测试获取统计"""
        config = RedisConfig(enabled=True, host="test.com", port=6380)
        client = DKIRedisClient(config)
        
        stats = client.get_stats()
        
        assert stats["enabled"] == True
        assert stats["connected"] == False
        assert stats["host"] == "test.com:6380"


class TestCacheConfig:
    """测试缓存配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = CacheConfig()
        
        assert config.l1_max_size == 1000
        assert config.l2_enabled == False
        assert config.enable_compression == True
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "l1_max_size": 500,
            "l2_enabled": True,
            "l2_ttl_seconds": 7200,
        }
        
        config = CacheConfig.from_dict(data)
        
        assert config.l1_max_size == 500
        assert config.l2_enabled == True
        assert config.l2_ttl_seconds == 7200


class TestPreferenceCacheManager:
    """测试偏好缓存管理器"""
    
    def test_init_without_redis(self):
        """测试无 Redis 初始化"""
        config = CacheConfig(l2_enabled=False)
        manager = PreferenceCacheManager(config=config)
        
        assert manager._is_l2_available() == False
    
    def test_init_with_redis_not_connected(self):
        """测试 Redis 未连接时"""
        config = CacheConfig(l2_enabled=True)
        redis_client = Mock()
        redis_client.is_available = False
        
        manager = PreferenceCacheManager(
            redis_client=redis_client,
            config=config,
        )
        
        assert manager._is_l2_available() == False
    
    def test_init_with_redis_connected(self):
        """测试 Redis 已连接时"""
        config = CacheConfig(l2_enabled=True)
        redis_client = Mock()
        redis_client.is_available = True
        
        manager = PreferenceCacheManager(
            redis_client=redis_client,
            config=config,
        )
        
        assert manager._is_l2_available() == True
    
    def test_compute_preference_hash(self):
        """测试偏好哈希计算"""
        manager = PreferenceCacheManager()
        
        hash1 = manager._compute_preference_hash("preference text")
        hash2 = manager._compute_preference_hash("preference text")
        hash3 = manager._compute_preference_hash("different text")
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_make_cache_key(self):
        """测试缓存键生成"""
        manager = PreferenceCacheManager()
        
        key = manager._make_cache_key("user_123", "abc123")
        
        assert key == "user_123:abc123"
    
    def test_make_redis_key(self):
        """测试 Redis 键生成"""
        config = CacheConfig(l2_key_prefix="test:pref")
        manager = PreferenceCacheManager(config=config)
        
        key = manager._make_redis_key("user_123:abc")
        
        assert key == "test:pref:user_123:abc"
    
    def test_get_stats(self):
        """测试获取统计"""
        manager = PreferenceCacheManager()
        
        stats = manager.get_stats()
        
        assert "total_requests" in stats
        assert "l1_hits" in stats
        assert "l2_hits" in stats
        assert "overall_hit_rate" in stats
        assert "l2_enabled" in stats
        assert "l2_available" in stats


class TestPreferenceCacheManagerAsync:
    """测试偏好缓存管理器异步方法"""
    
    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        """测试 L1 缓存命中"""
        manager = PreferenceCacheManager()
        
        # 模拟模型
        mock_model = Mock()
        mock_model.compute_kv = Mock(return_value=([], None))
        
        # 第一次请求 (L3 compute)
        _, tier_info1 = await manager.get_preference_kv(
            user_id="user_1",
            preference_text="test preference",
            model=mock_model,
        )
        
        assert tier_info1.tier == CacheTier.L3_COMPUTE
        assert tier_info1.hit == False
        
        # 第二次请求 (L1 hit)
        _, tier_info2 = await manager.get_preference_kv(
            user_id="user_1",
            preference_text="test preference",
            model=mock_model,
        )
        
        assert tier_info2.tier == CacheTier.L1_MEMORY
        assert tier_info2.hit == True
    
    @pytest.mark.asyncio
    async def test_invalidate(self):
        """测试缓存失效"""
        manager = PreferenceCacheManager()
        
        # 模拟模型
        mock_model = Mock()
        mock_model.compute_kv = Mock(return_value=([], None))
        
        # 添加缓存
        await manager.get_preference_kv(
            user_id="user_1",
            preference_text="test preference",
            model=mock_model,
        )
        
        # 失效
        count = await manager.invalidate("user_1")
        
        assert count >= 1
        
        # 再次请求应该是 L3
        _, tier_info = await manager.get_preference_kv(
            user_id="user_1",
            preference_text="test preference",
            model=mock_model,
        )
        
        assert tier_info.tier == CacheTier.L3_COMPUTE
    
    @pytest.mark.asyncio
    async def test_clear_all(self):
        """测试清除所有缓存"""
        manager = PreferenceCacheManager()
        
        # 模拟模型
        mock_model = Mock()
        mock_model.compute_kv = Mock(return_value=([], None))
        
        # 添加多个用户的缓存
        for i in range(5):
            await manager.get_preference_kv(
                user_id=f"user_{i}",
                preference_text=f"preference {i}",
                model=mock_model,
            )
        
        # 清除所有
        await manager.clear_all()
        
        # 统计应该重置
        stats = manager.get_stats()
        assert stats["total_bytes_cached"] == 0


# 集成测试 (需要真实 Redis)
@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis library not installed")
class TestRedisIntegration:
    """Redis 集成测试 (需要运行中的 Redis)"""
    
    @pytest.mark.asyncio
    async def test_full_flow_with_redis(self):
        """测试完整流程 (需要 Redis)"""
        # 这个测试需要真实的 Redis 服务器
        # 在 CI 环境中可能需要跳过
        
        config = RedisConfig(
            enabled=True,
            host="localhost",
            port=6379,
        )
        
        client = DKIRedisClient(config)
        connected = await client.connect()
        
        if not connected:
            pytest.skip("Redis not available")
        
        try:
            # 测试 set/get
            await client.set("test:key", {"value": 123})
            result = await client.get("test:key")
            
            assert result == {"value": 123}
            
            # 测试 delete
            await client.delete("test:key")
            result = await client.get("test:key")
            
            assert result is None
            
        finally:
            await client.close()
