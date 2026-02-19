"""
单元测试: 2026-02-18 Cache 模块审查修复验证

审查发现的问题:
1. preference_cache / user_isolation — _serialize_kv 对 bfloat16 崩溃
2. preference_cache / user_isolation — _compute_kv 缺少 GPU 显存清理
3. user_isolation — signed_redis_key 计算后未使用 (已删除死代码)
4. user_isolation — UserScopedCacheStore get/put 无并发保护 (已加锁)
5. preference_cache — LRUCache 文档声称 Thread-safe 但实际仅 async-safe

附加测试:
- CacheKeySigner HMAC 签名验证
- UserIsolationContext 创建和验证
- InferenceContextGuard 推理隔离
- PreferenceCacheManager L1 缓存流程
- IsolatedPreferenceCacheManager 用户隔离流程
- NonVectorizedDataHandler 搜索策略选择
- DKIRedisClient 压缩/解压协议
"""

import os
import sys
import time
import inspect
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch


# ============================================================================
# BUG 1: _serialize_kv 对 bfloat16 的处理
# ============================================================================

class TestSerializeKVBfloat16:
    """验证 _serialize_kv 对 bfloat16 张量的正确处理"""

    def test_preference_cache_serialize_handles_bfloat16(self):
        """PreferenceCacheManager._serialize_kv 应能处理 bfloat16"""
        from dki.cache.preference_cache import PreferenceCacheManager
        source = inspect.getsource(PreferenceCacheManager._serialize_kv)
        assert "bfloat16" in source, (
            "_serialize_kv should handle bfloat16 explicitly"
        )
        assert ".float()" in source, (
            "_serialize_kv should convert bfloat16 to float32"
        )
        assert "original_dtype" in source, (
            "_serialize_kv should record original_dtype for restoration"
        )

    def test_preference_cache_deserialize_restores_bfloat16(self):
        """PreferenceCacheManager._deserialize_kv 应恢复 bfloat16"""
        from dki.cache.preference_cache import PreferenceCacheManager
        source = inspect.getsource(PreferenceCacheManager._deserialize_kv)
        assert "original_dtype" in source, (
            "_deserialize_kv should check original_dtype"
        )
        assert "bfloat16" in source, (
            "_deserialize_kv should handle bfloat16 restoration"
        )

    def test_isolated_cache_serialize_handles_bfloat16(self):
        """IsolatedPreferenceCacheManager._serialize_kv 应能处理 bfloat16"""
        from dki.cache.user_isolation import IsolatedPreferenceCacheManager
        source = inspect.getsource(IsolatedPreferenceCacheManager._serialize_kv)
        assert "bfloat16" in source
        assert ".float()" in source
        assert "original_dtype" in source

    def test_isolated_cache_deserialize_restores_bfloat16(self):
        """IsolatedPreferenceCacheManager._deserialize_kv 应恢复 bfloat16"""
        from dki.cache.user_isolation import IsolatedPreferenceCacheManager
        source = inspect.getsource(IsolatedPreferenceCacheManager._deserialize_kv)
        assert "original_dtype" in source

    def test_preference_cache_serialize_roundtrip_bfloat16(self):
        """bfloat16 序列化/反序列化往返"""
        from dki.cache.preference_cache import PreferenceCacheManager
        from dki.models.base import KVCacheEntry

        cache = PreferenceCacheManager()
        entries = [
            KVCacheEntry(
                key=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
                value=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
                layer_idx=0,
            )
        ]

        serialized = cache._serialize_kv(entries)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        restored = cache._deserialize_kv(serialized)
        assert len(restored) == 1
        assert restored[0].key.dtype == torch.bfloat16
        assert restored[0].value.dtype == torch.bfloat16
        assert restored[0].key.shape == (1, 2, 3, 4)
        assert restored[0].layer_idx == 0

    def test_preference_cache_serialize_roundtrip_float16(self):
        """float16 序列化/反序列化往返"""
        from dki.cache.preference_cache import PreferenceCacheManager
        from dki.models.base import KVCacheEntry

        cache = PreferenceCacheManager()
        entries = [
            KVCacheEntry(
                key=torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float16),
                value=torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float16),
                layer_idx=5,
            )
        ]

        serialized = cache._serialize_kv(entries)
        restored = cache._deserialize_kv(serialized)
        assert len(restored) == 1
        assert torch.allclose(entries[0].key, restored[0].key)
        assert torch.allclose(entries[0].value, restored[0].value)
        assert restored[0].layer_idx == 5

    def test_preference_cache_serialize_roundtrip_float32(self):
        """float32 序列化/反序列化往返"""
        from dki.cache.preference_cache import PreferenceCacheManager
        from dki.models.base import KVCacheEntry

        cache = PreferenceCacheManager()
        entries = [
            KVCacheEntry(
                key=torch.tensor([[[[1.5, 2.5]]]], dtype=torch.float32),
                value=torch.tensor([[[[3.5, 4.5]]]], dtype=torch.float32),
                layer_idx=2,
            )
        ]

        serialized = cache._serialize_kv(entries)
        restored = cache._deserialize_kv(serialized)
        assert torch.allclose(entries[0].key, restored[0].key)
        assert torch.allclose(entries[0].value, restored[0].value)


# ============================================================================
# BUG 2: _compute_kv GPU 显存清理
# ============================================================================

class TestComputeKVGPUCleanup:
    """验证 _compute_kv 有 GPU 显存清理"""

    def test_preference_cache_compute_kv_has_cleanup(self):
        """PreferenceCacheManager._compute_kv 应有 empty_cache"""
        from dki.cache.preference_cache import PreferenceCacheManager
        source = inspect.getsource(PreferenceCacheManager._compute_kv)
        assert "empty_cache()" in source, (
            "_compute_kv should call torch.cuda.empty_cache()"
        )

    def test_isolated_cache_compute_kv_has_cleanup(self):
        """IsolatedPreferenceCacheManager._compute_kv 应有 empty_cache"""
        from dki.cache.user_isolation import IsolatedPreferenceCacheManager
        source = inspect.getsource(IsolatedPreferenceCacheManager._compute_kv)
        assert "empty_cache()" in source

    def test_preference_cache_compute_kv_error_path_cleanup(self):
        """_compute_kv 异常路径也应清理"""
        from dki.cache.preference_cache import PreferenceCacheManager
        source = inspect.getsource(PreferenceCacheManager._compute_kv)
        # Should have empty_cache in except block too
        lines = source.split('\n')
        except_found = False
        cleanup_after_except = False
        for line in lines:
            if 'except' in line:
                except_found = True
            if except_found and 'empty_cache' in line:
                cleanup_after_except = True
                break
        assert cleanup_after_except, (
            "_compute_kv should also clean up GPU memory on error path"
        )


# ============================================================================
# BUG 4: signed_redis_key 死代码已删除
# ============================================================================

class TestDeadCodeRemoved:
    """验证死代码已删除"""

    def test_signed_redis_key_removed(self):
        """IsolatedPreferenceCacheManager.get_preference_kv 不应有 signed_redis_key"""
        from dki.cache.user_isolation import IsolatedPreferenceCacheManager
        source = inspect.getsource(IsolatedPreferenceCacheManager.get_preference_kv)
        assert "signed_redis_key" not in source, (
            "Dead code signed_redis_key should be removed"
        )


# ============================================================================
# BUG 5: LRUCache 文档修正
# ============================================================================

class TestLRUCacheDocstring:
    """验证 LRUCache 文档修正"""

    def test_lru_cache_not_thread_safe(self):
        """LRUCache 文档不应声称 Thread-safe"""
        from dki.cache.preference_cache import LRUCache
        docstring = LRUCache.__doc__
        assert "Thread-safe" not in docstring, (
            "LRUCache should not claim Thread-safe (uses asyncio.Lock)"
        )
        assert "Async-safe" in docstring or "async" in docstring.lower(), (
            "LRUCache should mention async safety"
        )


# ============================================================================
# BUG 6: UserScopedCacheStore 并发保护
# ============================================================================

class TestUserScopedCacheStoreLocking:
    """验证 UserScopedCacheStore 的 get/put 使用了锁"""

    def test_get_uses_global_lock(self):
        """get 方法应使用 _global_lock"""
        from dki.cache.user_isolation import UserScopedCacheStore
        source = inspect.getsource(UserScopedCacheStore.get)
        assert "_global_lock" in source, (
            "get() should use _global_lock for concurrency safety"
        )

    def test_put_uses_global_lock(self):
        """put 方法应使用 _global_lock"""
        from dki.cache.user_isolation import UserScopedCacheStore
        source = inspect.getsource(UserScopedCacheStore.put)
        assert "_global_lock" in source, (
            "put() should use _global_lock for concurrency safety"
        )


# ============================================================================
# CacheKeySigner HMAC 签名
# ============================================================================

class TestCacheKeySigner:
    """验证 HMAC 签名的正确性"""

    def test_sign_and_verify_success(self):
        """签名后验证应成功"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="test_secret_123")
        
        signed = signer.sign_key("user_001", "abc123def456")
        assert signer.verify_key(signed, "user_001") is True

    def test_verify_wrong_user_fails(self):
        """验证错误用户应失败"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="test_secret_123")
        
        signed = signer.sign_key("user_001", "abc123def456")
        assert signer.verify_key(signed, "user_002") is False

    def test_verify_tampered_signature_fails(self):
        """篡改签名应失败"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="test_secret_123")
        
        signed = signer.sign_key("user_001", "abc123")
        # Tamper with the signature
        tampered = signed[:-1] + ("a" if signed[-1] != "a" else "b")
        assert signer.verify_key(tampered, "user_001") is False

    def test_different_secrets_different_signatures(self):
        """不同密钥应产生不同签名"""
        from dki.cache.user_isolation import CacheKeySigner
        signer1 = CacheKeySigner(secret="secret_A")
        signer2 = CacheKeySigner(secret="secret_B")
        
        signed1 = signer1.sign_key("user_001", "hash123")
        signed2 = signer2.sign_key("user_001", "hash123")
        
        assert signed1 != signed2

    def test_extract_user_id(self):
        """应能从签名键中提取 user_id"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="test")
        
        signed = signer.sign_key("user_abc", "hash123")
        extracted = signer.extract_user_id(signed)
        assert extracted == "user_abc"

    def test_auto_generated_secret(self):
        """空密钥应自动生成"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="")
        
        signed = signer.sign_key("user_001", "hash123")
        assert signer.verify_key(signed, "user_001") is True

    def test_signed_key_format(self):
        """签名键格式应为 user_id:content_hash:signature"""
        from dki.cache.user_isolation import CacheKeySigner
        signer = CacheKeySigner(secret="test")
        
        signed = signer.sign_key("user_001", "hash123")
        parts = signed.split(":")
        assert len(parts) == 3
        assert parts[0] == "user_001"
        assert parts[1] == "hash123"
        assert len(parts[2]) == 32  # 128-bit truncated hex


# ============================================================================
# UserIsolationContext
# ============================================================================

class TestUserIsolationContext:
    """验证用户隔离上下文"""

    def test_create_valid_context(self):
        """应能创建有效上下文"""
        from dki.cache.user_isolation import UserIsolationContext
        ctx = UserIsolationContext.create(user_id="user_123", session_id="sess_456")
        
        assert ctx.user_id == "user_123"
        assert ctx.session_id == "sess_456"
        assert len(ctx.request_id) == 16  # token_hex(8) = 16 chars
        assert ctx.created_at > 0

    def test_create_strips_whitespace(self):
        """应去除 user_id 前后空格"""
        from dki.cache.user_isolation import UserIsolationContext
        ctx = UserIsolationContext.create(user_id="  user_123  ")
        assert ctx.user_id == "user_123"

    def test_create_empty_user_raises(self):
        """空 user_id 应抛出异常"""
        from dki.cache.user_isolation import UserIsolationContext
        with pytest.raises(ValueError):
            UserIsolationContext.create(user_id="")
        
        with pytest.raises(ValueError):
            UserIsolationContext.create(user_id="   ")

    def test_sign_and_verify_with_signer(self):
        """带签名器的上下文应能签名和验证"""
        from dki.cache.user_isolation import UserIsolationContext, CacheKeySigner
        signer = CacheKeySigner(secret="test")
        ctx = UserIsolationContext.create(user_id="user_001", signer=signer)
        
        signed = ctx.sign_cache_key("hash123")
        assert ctx.verify_cache_key(signed) is True

    def test_sign_without_signer_fallback(self):
        """无签名器时应退化为简单键"""
        from dki.cache.user_isolation import UserIsolationContext
        ctx = UserIsolationContext.create(user_id="user_001")
        
        signed = ctx.sign_cache_key("hash123")
        assert signed == "user_001:hash123"
        assert ctx.verify_cache_key(signed) is True

    def test_make_redis_namespace(self):
        """应生成正确的 Redis 命名空间"""
        from dki.cache.user_isolation import UserIsolationContext
        ctx = UserIsolationContext.create(user_id="user_001")
        
        ns = ctx.make_redis_namespace()
        assert ns == "dki:pref_kv:user:user_001"
        
        ns_custom = ctx.make_redis_namespace(prefix="custom")
        assert ns_custom == "custom:user:user_001"


# ============================================================================
# InferenceContextGuard
# ============================================================================

class TestInferenceContextGuard:
    """验证推理上下文隔离守卫"""

    def test_scoped_inference_basic(self):
        """基本推理作用域"""
        from dki.cache.user_isolation import InferenceContextGuard
        guard = InferenceContextGuard()
        
        with guard.scoped_inference(user_id="user_001") as scope:
            assert scope.user_id == "user_001"
            assert guard._active_user == "user_001"
        
        # 退出后应清理
        assert guard._active_user is None
        assert guard._stats["inferences"] == 1
        assert guard._stats["cleanups"] == 1

    def test_scoped_inference_clears_kv_references(self):
        """退出作用域应清理 K/V 引用"""
        from dki.cache.user_isolation import InferenceContextGuard
        from dki.models.base import KVCacheEntry
        guard = InferenceContextGuard()
        
        entries = [
            KVCacheEntry(
                key=torch.randn(1, 2, 3),
                value=torch.randn(1, 2, 3),
                layer_idx=0,
            )
        ]
        
        with guard.scoped_inference(user_id="user_001", kv_entries=entries):
            assert entries[0].key is not None
        
        # 退出后 K/V 引用应被清除
        assert entries[0].key is None
        assert entries[0].value is None

    def test_concurrent_violation_detected(self):
        """应检测并发冲突"""
        from dki.cache.user_isolation import InferenceContextGuard
        guard = InferenceContextGuard()
        
        # 模拟: 手动设置 active_user
        guard._active_user = "user_001"
        
        with guard.scoped_inference(user_id="user_002"):
            pass
        
        assert guard._stats["concurrent_violations"] == 1

    def test_cleanup_callback_called(self):
        """应调用自定义清理回调"""
        from dki.cache.user_isolation import InferenceContextGuard
        guard = InferenceContextGuard()
        
        callback_called = [False]
        def cleanup():
            callback_called[0] = True
        
        with guard.scoped_inference(user_id="user_001", cleanup_callback=cleanup):
            pass
        
        assert callback_called[0] is True

    def test_exception_still_cleans_up(self):
        """异常时也应清理"""
        from dki.cache.user_isolation import InferenceContextGuard
        guard = InferenceContextGuard()
        
        with pytest.raises(ValueError):
            with guard.scoped_inference(user_id="user_001"):
                raise ValueError("test error")
        
        assert guard._active_user is None
        assert guard._stats["cleanups"] == 1


# ============================================================================
# CacheAuditLog
# ============================================================================

class TestCacheAuditLog:
    """验证审计日志"""

    def test_record_success(self):
        """应记录成功访问"""
        from dki.cache.user_isolation import CacheAuditLog
        log = CacheAuditLog(max_entries=100)
        
        log.record(
            user_id="user_001",
            action="get",
            cache_key="key_123",
            cache_tier="L1_memory",
            success=True,
        )
        
        assert log._total_count == 1
        assert log._denial_count == 0
        records = log.get_records()
        assert len(records) == 1
        assert records[0].user_id == "user_001"

    def test_record_denial(self):
        """应记录拒绝访问"""
        from dki.cache.user_isolation import CacheAuditLog
        log = CacheAuditLog(max_entries=100)
        
        log.record(
            user_id="user_001",
            action="get",
            cache_key="key_123",
            cache_tier="L1_memory",
            success=False,
            denial_reason="HMAC failed",
        )
        
        assert log._denial_count == 1
        stats = log.get_denial_stats()
        assert stats["total_denials"] == 1
        assert stats["denial_rate"] == 1.0

    def test_ring_buffer_eviction(self):
        """超过上限应丢弃最旧记录"""
        from dki.cache.user_isolation import CacheAuditLog
        log = CacheAuditLog(max_entries=5)
        
        for i in range(10):
            log.record(
                user_id=f"user_{i}",
                action="get",
                cache_key=f"key_{i}",
                cache_tier="L1_memory",
                success=True,
            )
        
        assert len(log._records) == 5
        # 最旧的应是 user_5 (0-4 被丢弃)
        assert log._records[0].user_id == "user_5"

    def test_filter_by_user(self):
        """应能按用户过滤"""
        from dki.cache.user_isolation import CacheAuditLog
        log = CacheAuditLog()
        
        log.record("user_a", "get", "k1", "L1", True)
        log.record("user_b", "get", "k2", "L1", True)
        log.record("user_a", "put", "k3", "L1", True)
        
        records = log.get_records(user_id="user_a")
        assert len(records) == 2


# ============================================================================
# PreferenceCacheManager L1 缓存流程
# ============================================================================

class TestPreferenceCacheManagerL1:
    """验证 PreferenceCacheManager 的 L1 缓存流程"""

    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        """L1 缓存命中"""
        from dki.cache.preference_cache import PreferenceCacheManager
        
        cache = PreferenceCacheManager()
        
        # 模拟 model
        mock_model = MagicMock()
        mock_model.compute_kv = MagicMock(return_value=([], None))
        
        # 第一次: L3 计算
        kv1, tier1 = await cache.get_preference_kv(
            user_id="user_001",
            preference_text="test preference",
            model=mock_model,
        )
        assert tier1.tier.value == "L3_COMPUTE"
        assert tier1.hit is False
        
        # 第二次: L1 命中
        kv2, tier2 = await cache.get_preference_kv(
            user_id="user_001",
            preference_text="test preference",
            model=mock_model,
        )
        assert tier2.tier.value == "L1_MEMORY"
        assert tier2.hit is True

    @pytest.mark.asyncio
    async def test_invalidate_clears_l1(self):
        """invalidate 应清除 L1 缓存"""
        from dki.cache.preference_cache import PreferenceCacheManager
        
        cache = PreferenceCacheManager()
        mock_model = MagicMock()
        mock_model.compute_kv = MagicMock(return_value=([], None))
        
        # 填充缓存
        await cache.get_preference_kv("user_001", "pref", mock_model)
        
        # 失效
        count = await cache.invalidate("user_001")
        assert count >= 1
        
        # 再次查询应 miss
        _, tier = await cache.get_preference_kv("user_001", "pref", mock_model)
        assert tier.tier.value == "L3_COMPUTE"

    @pytest.mark.asyncio
    async def test_empty_user_id_skips_cache(self):
        """空 user_id 应跳过缓存"""
        from dki.cache.preference_cache import PreferenceCacheManager
        
        cache = PreferenceCacheManager()
        mock_model = MagicMock()
        mock_model.compute_kv = MagicMock(return_value=([], None))
        
        _, tier = await cache.get_preference_kv("", "pref", mock_model)
        assert tier.user_id == "<anonymous>"

    @pytest.mark.asyncio
    async def test_force_recompute(self):
        """force_recompute 应强制重新计算"""
        from dki.cache.preference_cache import PreferenceCacheManager
        
        cache = PreferenceCacheManager()
        mock_model = MagicMock()
        mock_model.compute_kv = MagicMock(return_value=([], None))
        
        # 填充缓存
        await cache.get_preference_kv("user_001", "pref", mock_model)
        
        # 强制重新计算
        _, tier = await cache.get_preference_kv(
            "user_001", "pref", mock_model, force_recompute=True
        )
        assert tier.tier.value == "L3_COMPUTE"
        assert tier.hit is False

    def test_stats(self):
        """统计应正确"""
        from dki.cache.preference_cache import PreferenceCacheManager
        cache = PreferenceCacheManager()
        stats = cache.get_stats()
        
        assert "total_requests" in stats
        assert "l1_hits" in stats
        assert "l2_hits" in stats
        assert "overall_hit_rate" in stats
        assert "l1_cache" in stats


# ============================================================================
# UserScopedCacheStore 功能测试
# ============================================================================

class TestUserScopedCacheStore:
    """验证用户级隔离缓存存储"""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """存取应正常"""
        from dki.cache.user_isolation import (
            UserScopedCacheStore, UserIsolationContext, CacheKeySigner
        )
        signer = CacheKeySigner(secret="test")
        store = UserScopedCacheStore(signer=signer)
        ctx = UserIsolationContext.create(user_id="user_001", signer=signer)
        
        await store.put(ctx, "hash_abc", {"data": "value"})
        result = await store.get(ctx, "hash_abc")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_cross_user_isolation(self):
        """不同用户应隔离"""
        from dki.cache.user_isolation import (
            UserScopedCacheStore, UserIsolationContext, CacheKeySigner
        )
        signer = CacheKeySigner(secret="test")
        store = UserScopedCacheStore(signer=signer)
        
        ctx1 = UserIsolationContext.create(user_id="user_001", signer=signer)
        ctx2 = UserIsolationContext.create(user_id="user_002", signer=signer)
        
        await store.put(ctx1, "hash_abc", {"user": "001"})
        await store.put(ctx2, "hash_abc", {"user": "002"})
        
        result1 = await store.get(ctx1, "hash_abc")
        result2 = await store.get(ctx2, "hash_abc")
        
        assert result1 == {"user": "001"}
        assert result2 == {"user": "002"}

    @pytest.mark.asyncio
    async def test_per_user_eviction(self):
        """单用户超过上限应驱逐最旧条目"""
        from dki.cache.user_isolation import (
            UserScopedCacheStore, UserIsolationContext,
            CacheKeySigner, UserIsolationConfig
        )
        config = UserIsolationConfig(per_user_max_entries=2)
        signer = CacheKeySigner(secret="test")
        store = UserScopedCacheStore(config=config, signer=signer)
        ctx = UserIsolationContext.create(user_id="user_001", signer=signer)
        
        await store.put(ctx, "hash_1", "data_1")
        await store.put(ctx, "hash_2", "data_2")
        await store.put(ctx, "hash_3", "data_3")  # 应驱逐 hash_1
        
        assert await store.get(ctx, "hash_1") is None
        assert await store.get(ctx, "hash_2") == "data_2"
        assert await store.get(ctx, "hash_3") == "data_3"

    @pytest.mark.asyncio
    async def test_invalidate_user(self):
        """应能清除指定用户的所有缓存"""
        from dki.cache.user_isolation import (
            UserScopedCacheStore, UserIsolationContext, CacheKeySigner
        )
        signer = CacheKeySigner(secret="test")
        store = UserScopedCacheStore(signer=signer)
        ctx = UserIsolationContext.create(user_id="user_001", signer=signer)
        
        await store.put(ctx, "h1", "d1")
        await store.put(ctx, "h2", "d2")
        
        count = await store.invalidate_user("user_001")
        assert count == 2
        
        assert await store.get(ctx, "h1") is None
        assert await store.get(ctx, "h2") is None


# ============================================================================
# NonVectorizedDataHandler 策略选择
# ============================================================================

class TestNonVectorizedDataHandler:
    """验证非向量化数据处理器"""

    def test_strategy_selection_lazy(self):
        """小数据集应选择 LAZY"""
        from dki.cache.non_vectorized_handler import (
            NonVectorizedDataHandler, SearchStrategy
        )
        mock_embedding = MagicMock()
        handler = NonVectorizedDataHandler(mock_embedding)
        
        assert handler._select_strategy(50) == SearchStrategy.LAZY
        assert handler._select_strategy(100) == SearchStrategy.LAZY

    def test_strategy_selection_hybrid(self):
        """中等数据集应选择 HYBRID"""
        from dki.cache.non_vectorized_handler import (
            NonVectorizedDataHandler, SearchStrategy
        )
        mock_embedding = MagicMock()
        handler = NonVectorizedDataHandler(mock_embedding)
        
        assert handler._select_strategy(500) == SearchStrategy.HYBRID

    def test_strategy_selection_batch(self):
        """大数据集应选择 BATCH"""
        from dki.cache.non_vectorized_handler import (
            NonVectorizedDataHandler, SearchStrategy
        )
        mock_embedding = MagicMock()
        handler = NonVectorizedDataHandler(mock_embedding)
        
        assert handler._select_strategy(1000) == SearchStrategy.BATCH
        assert handler._select_strategy(10000) == SearchStrategy.BATCH

    def test_cosine_similarity(self):
        """余弦相似度计算应正确"""
        from dki.cache.non_vectorized_handler import NonVectorizedDataHandler
        mock_embedding = MagicMock()
        handler = NonVectorizedDataHandler(mock_embedding)
        
        # 相同向量 → 1.0
        assert abs(handler._cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6
        
        # 正交向量 → 0.0
        assert abs(handler._cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-6
        
        # 零向量 → 0.0
        assert handler._cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_tokenize(self):
        """分词应正确"""
        from dki.cache.non_vectorized_handler import NonVectorizedDataHandler
        mock_embedding = MagicMock()
        handler = NonVectorizedDataHandler(mock_embedding)
        
        tokens = handler._tokenize("Hello, World! Test 123")
        assert tokens == ["hello", "world", "test", "123"]


# ============================================================================
# DKIRedisClient 压缩协议
# ============================================================================

class TestRedisClientCompression:
    """验证 Redis 客户端的压缩/解压协议"""

    def test_compress_small_data_no_compression(self):
        """小数据不应压缩"""
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        config = RedisConfig(compression_threshold=1024)
        client = DKIRedisClient(config)
        
        small_data = b"hello"
        result = client._compress(small_data)
        
        assert result[0:1] == b'\x00'  # 未压缩标记
        assert result[1:] == small_data

    def test_compress_large_data_compressed(self):
        """大数据应压缩"""
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        config = RedisConfig(compression_threshold=10)
        client = DKIRedisClient(config)
        
        large_data = b"x" * 100
        result = client._compress(large_data)
        
        assert result[0:1] == b'\x01'  # 已压缩标记
        assert len(result) < len(large_data)  # 应该更小

    def test_compress_decompress_roundtrip(self):
        """压缩/解压往返应保持数据一致"""
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        config = RedisConfig(compression_threshold=10)
        client = DKIRedisClient(config)
        
        original = b"test data " * 100
        compressed = client._compress(original)
        decompressed = client._decompress(compressed)
        
        assert decompressed == original

    def test_decompress_uncompressed_data(self):
        """未压缩数据也应正确解压"""
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        client = DKIRedisClient(RedisConfig())
        
        data = b'\x00' + b"raw data"
        result = client._decompress(data)
        assert result == b"raw data"

    def test_decompress_empty_data(self):
        """空数据应返回空"""
        from dki.cache.redis_client import DKIRedisClient, RedisConfig
        client = DKIRedisClient(RedisConfig())
        
        assert client._decompress(b"") == b""


# ============================================================================
# LRUCache 功能测试
# ============================================================================

class TestLRUCache:
    """验证 LRU 缓存的基本功能"""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """存取应正常"""
        from dki.cache.preference_cache import LRUCache, CacheEntry
        cache = LRUCache(maxsize=10)
        
        entry = CacheEntry(
            kv_data="test",
            preference_hash="hash",
            created_at=time.time(),
            last_accessed=time.time(),
        )
        
        await cache.put("key1", entry)
        result = await cache.get("key1")
        assert result is not None
        assert result.kv_data == "test"

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """超过容量应驱逐最旧条目"""
        from dki.cache.preference_cache import LRUCache, CacheEntry
        cache = LRUCache(maxsize=2)
        
        for i in range(3):
            entry = CacheEntry(
                kv_data=f"data_{i}",
                preference_hash=f"hash_{i}",
                created_at=time.time(),
                last_accessed=time.time(),
            )
            await cache.put(f"key_{i}", entry)
        
        # key_0 应被驱逐
        assert await cache.get("key_0") is None
        assert (await cache.get("key_1")).kv_data == "data_1"
        assert (await cache.get("key_2")).kv_data == "data_2"

    @pytest.mark.asyncio
    async def test_delete(self):
        """删除应正常"""
        from dki.cache.preference_cache import LRUCache, CacheEntry
        cache = LRUCache(maxsize=10)
        
        entry = CacheEntry(
            kv_data="test",
            preference_hash="hash",
            created_at=time.time(),
            last_accessed=time.time(),
        )
        await cache.put("key1", entry)
        
        result = await cache.delete("key1")
        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_prefix(self):
        """前缀删除应正常"""
        from dki.cache.preference_cache import LRUCache, CacheEntry
        cache = LRUCache(maxsize=10)
        
        for key in ["user1:a", "user1:b", "user2:a"]:
            entry = CacheEntry(
                kv_data=key,
                preference_hash="h",
                created_at=time.time(),
                last_accessed=time.time(),
            )
            await cache.put(key, entry)
        
        count = await cache.delete_prefix("user1:")
        assert count == 2
        assert await cache.get("user1:a") is None
        assert (await cache.get("user2:a")).kv_data == "user2:a"

    def test_stats(self):
        """统计应正确"""
        from dki.cache.preference_cache import LRUCache
        cache = LRUCache(maxsize=10)
        stats = cache.get_stats()
        
        assert stats["size"] == 0
        assert stats["maxsize"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0


# ============================================================================
# EmbeddingCache 功能测试
# ============================================================================

class TestEmbeddingCache:
    """验证嵌入缓存"""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """存取应正常"""
        from dki.cache.non_vectorized_handler import EmbeddingCache
        cache = EmbeddingCache(max_size=10)
        
        await cache.put("msg_1", [0.1, 0.2, 0.3])
        result = await cache.get("msg_1")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """超过容量应驱逐"""
        from dki.cache.non_vectorized_handler import EmbeddingCache
        cache = EmbeddingCache(max_size=2)
        
        await cache.put("a", [1.0])
        await cache.put("b", [2.0])
        await cache.put("c", [3.0])  # 驱逐 a
        
        assert await cache.get("a") is None
        assert await cache.get("b") == [2.0]
        assert await cache.get("c") == [3.0]

    @pytest.mark.asyncio
    async def test_clear(self):
        """清除应正常"""
        from dki.cache.non_vectorized_handler import EmbeddingCache
        cache = EmbeddingCache(max_size=10)
        
        await cache.put("a", [1.0])
        await cache.clear()
        assert len(cache) == 0
