"""
User Isolation 单元测试

测试用户级缓存隔离的核心组件:
- UserIsolationConfig: 配置
- CacheKeySigner: HMAC 签名
- UserIsolationContext: 隔离上下文
- UserScopedCacheStore: 用户级分区缓存
- InferenceContextGuard: 推理上下文守卫
- CacheAuditLog: 审计日志
- IsolatedPreferenceCacheManager: 完整隔离缓存管理器

Author: AGI Demo Project
"""

import asyncio
import time
from collections import OrderedDict
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from dki.cache.user_isolation import (
    UserIsolationConfig,
    CacheAccessRecord,
    CacheAuditLog,
    CacheKeySigner,
    UserIsolationContext,
    UserScopedCacheStore,
    InferenceContextGuard,
    IsolatedPreferenceCacheManager,
)


# ============================================================
# UserIsolationConfig 测试
# ============================================================

class TestUserIsolationConfig:
    """UserIsolationConfig 配置测试"""

    def test_default_values(self):
        config = UserIsolationConfig()
        assert config.security_level == "strict"
        assert config.hmac_secret == ""
        assert config.signature_length == 16
        assert config.enable_audit_log is True
        assert config.enable_inference_guard is True
        assert config.max_user_partitions == 10000
        assert config.per_user_max_entries == 100
        assert config.allow_cross_user_migration is False

    def test_from_dict(self):
        data = {
            "security_level": "relaxed",
            "enable_audit_log": False,
            "per_user_max_entries": 50,
        }
        config = UserIsolationConfig.from_dict(data)
        assert config.security_level == "relaxed"
        assert config.enable_audit_log is False
        assert config.per_user_max_entries == 50
        # 未指定字段保持默认
        assert config.enable_inference_guard is True

    def test_from_dict_ignores_unknown_keys(self):
        data = {"security_level": "standard", "unknown_key": "value"}
        config = UserIsolationConfig.from_dict(data)
        assert config.security_level == "standard"

    def test_production_preset(self):
        config = UserIsolationConfig.production()
        assert config.security_level == "strict"
        assert config.enable_audit_log is True
        assert config.enable_inference_guard is True

    def test_development_preset(self):
        config = UserIsolationConfig.development()
        assert config.security_level == "relaxed"
        assert config.enable_audit_log is False
        assert config.enable_inference_guard is True


# ============================================================
# CacheKeySigner 测试
# ============================================================

class TestCacheKeySigner:
    """CacheKeySigner HMAC 签名测试"""

    def test_sign_key_format(self):
        signer = CacheKeySigner(secret="test_secret")
        signed = signer.sign_key("user_123", "abc123")
        parts = signed.split(":")
        assert len(parts) == 3
        assert parts[0] == "user_123"
        assert parts[1] == "abc123"
        assert len(parts[2]) == 32  # 128-bit hex

    def test_verify_key_valid(self):
        signer = CacheKeySigner(secret="test_secret")
        signed = signer.sign_key("user_123", "abc123")
        assert signer.verify_key(signed, "user_123") is True

    def test_verify_key_wrong_user(self):
        """不同用户的签名应验证失败"""
        signer = CacheKeySigner(secret="test_secret")
        signed = signer.sign_key("user_123", "abc123")
        assert signer.verify_key(signed, "user_456") is False

    def test_verify_key_tampered_signature(self):
        """篡改签名应验证失败"""
        signer = CacheKeySigner(secret="test_secret")
        signed = signer.sign_key("user_123", "abc123")
        tampered = signed[:-1] + ("0" if signed[-1] != "0" else "1")
        assert signer.verify_key(tampered, "user_123") is False

    def test_verify_key_malformed(self):
        """格式错误的键应验证失败"""
        signer = CacheKeySigner(secret="test_secret")
        assert signer.verify_key("no_colons", "user_123") is False
        assert signer.verify_key("one:colon", "user_123") is False

    def test_different_secrets_produce_different_signatures(self):
        signer_a = CacheKeySigner(secret="secret_a")
        signer_b = CacheKeySigner(secret="secret_b")
        signed_a = signer_a.sign_key("user_123", "abc")
        signed_b = signer_b.sign_key("user_123", "abc")
        assert signed_a != signed_b

    def test_same_secret_produces_consistent_signatures(self):
        signer_a = CacheKeySigner(secret="same_secret")
        signer_b = CacheKeySigner(secret="same_secret")
        assert signer_a.sign_key("u1", "h1") == signer_b.sign_key("u1", "h1")

    def test_auto_generated_secret(self):
        """空密钥时自动生成随机密钥"""
        signer = CacheKeySigner(secret="")
        signed = signer.sign_key("user_123", "abc")
        assert signer.verify_key(signed, "user_123") is True

    def test_extract_user_id(self):
        signer = CacheKeySigner(secret="test")
        signed = signer.sign_key("user_123", "hash")
        assert signer.extract_user_id(signed) == "user_123"

    def test_cross_user_forgery_impossible(self):
        """
        核心安全测试: 即使知道 user_id 和 content_hash,
        没有 secret 也无法构造有效签名
        """
        signer = CacheKeySigner(secret="real_secret")
        attacker_signer = CacheKeySigner(secret="guessed_secret")

        # 攻击者尝试伪造 user_123 的缓存键
        forged = attacker_signer.sign_key("user_123", "target_hash")
        assert signer.verify_key(forged, "user_123") is False


# ============================================================
# UserIsolationContext 测试
# ============================================================

class TestUserIsolationContext:
    """UserIsolationContext 隔离上下文测试"""

    def test_create_basic(self):
        ctx = UserIsolationContext.create(user_id="user_123")
        assert ctx.user_id == "user_123"
        assert ctx.request_id != ""
        assert ctx.created_at > 0

    def test_create_with_session(self):
        ctx = UserIsolationContext.create(
            user_id="user_123",
            session_id="sess_456",
        )
        assert ctx.session_id == "sess_456"

    def test_create_strips_whitespace(self):
        ctx = UserIsolationContext.create(user_id="  user_123  ")
        assert ctx.user_id == "user_123"

    def test_create_empty_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            UserIsolationContext.create(user_id="")

    def test_create_whitespace_only_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            UserIsolationContext.create(user_id="   ")

    def test_sign_cache_key_with_signer(self):
        signer = CacheKeySigner(secret="test")
        ctx = UserIsolationContext.create(
            user_id="user_123",
            signer=signer,
        )
        signed = ctx.sign_cache_key("content_hash")
        assert "user_123" in signed
        assert signer.verify_key(signed, "user_123") is True

    def test_sign_cache_key_without_signer(self):
        """无签名器时退化为简单键"""
        ctx = UserIsolationContext.create(user_id="user_123")
        signed = ctx.sign_cache_key("content_hash")
        assert signed == "user_123:content_hash"

    def test_verify_cache_key_with_signer(self):
        signer = CacheKeySigner(secret="test")
        ctx = UserIsolationContext.create(user_id="user_123", signer=signer)
        signed = ctx.sign_cache_key("hash")
        assert ctx.verify_cache_key(signed) is True

    def test_verify_cache_key_without_signer(self):
        ctx = UserIsolationContext.create(user_id="user_123")
        assert ctx.verify_cache_key("user_123:hash") is True
        assert ctx.verify_cache_key("user_456:hash") is False

    def test_make_redis_namespace(self):
        ctx = UserIsolationContext.create(user_id="user_123")
        ns = ctx.make_redis_namespace()
        assert ns == "dki:pref_kv:user:user_123"

    def test_make_redis_namespace_custom_prefix(self):
        ctx = UserIsolationContext.create(user_id="user_123")
        ns = ctx.make_redis_namespace(prefix="custom:prefix")
        assert ns == "custom:prefix:user:user_123"

    def test_unique_request_ids(self):
        ctx_a = UserIsolationContext.create(user_id="user_123")
        ctx_b = UserIsolationContext.create(user_id="user_123")
        assert ctx_a.request_id != ctx_b.request_id


# ============================================================
# CacheAuditLog 测试
# ============================================================

class TestCacheAuditLog:
    """CacheAuditLog 审计日志测试"""

    def test_record_and_retrieve(self):
        log = CacheAuditLog(max_entries=100)
        log.record(
            user_id="user_123",
            action="get",
            cache_key="key_1",
            cache_tier="L1_memory",
            success=True,
        )
        records = log.get_records()
        assert len(records) == 1
        assert records[0].user_id == "user_123"
        assert records[0].action == "get"
        assert records[0].success is True

    def test_denial_counting(self):
        log = CacheAuditLog()
        log.record("u1", "get", "k1", "L1_memory", success=True)
        log.record("u2", "get", "k2", "L1_memory", success=False, denial_reason="HMAC fail")
        log.record("u1", "put", "k3", "L1_memory", success=True)

        stats = log.get_denial_stats()
        assert stats["total_accesses"] == 3
        assert stats["total_denials"] == 1
        assert 0 < stats["denial_rate"] < 1

    def test_ring_buffer(self):
        """超过上限时应丢弃最旧记录"""
        log = CacheAuditLog(max_entries=5)
        for i in range(10):
            log.record(f"user_{i}", "get", f"key_{i}", "L1_memory", success=True)
        records = log.get_records()
        assert len(records) == 5
        # 最旧的 5 条 (user_0 ~ user_4) 应被丢弃
        assert records[0].user_id == "user_5"

    def test_filter_by_user(self):
        log = CacheAuditLog()
        log.record("u1", "get", "k1", "L1_memory", success=True)
        log.record("u2", "get", "k2", "L1_memory", success=True)
        log.record("u1", "put", "k3", "L1_memory", success=True)

        u1_records = log.get_records(user_id="u1")
        assert len(u1_records) == 2
        assert all(r.user_id == "u1" for r in u1_records)

    def test_filter_by_action(self):
        log = CacheAuditLog()
        log.record("u1", "get", "k1", "L1_memory", success=True)
        log.record("u1", "put", "k2", "L1_memory", success=True)

        get_records = log.get_records(action="get")
        assert len(get_records) == 1

    def test_filter_by_success(self):
        log = CacheAuditLog()
        log.record("u1", "get", "k1", "L1_memory", success=True)
        log.record("u1", "get", "k2", "L1_memory", success=False, denial_reason="test")

        denied = log.get_records(success=False)
        assert len(denied) == 1
        assert denied[0].denial_reason == "test"

    def test_get_stats(self):
        log = CacheAuditLog(max_entries=100)
        log.record("u1", "get", "k1", "L1_memory", success=True)
        stats = log.get_stats()
        assert stats["total_records"] == 1
        assert stats["max_entries"] == 100


# ============================================================
# UserScopedCacheStore 测试
# ============================================================

class TestUserScopedCacheStore:
    """UserScopedCacheStore 用户级分区缓存测试"""

    @pytest.fixture
    def config(self):
        return UserIsolationConfig(
            security_level="strict",
            hmac_secret="test_secret",
            per_user_max_entries=5,
            max_user_partitions=3,
            enable_audit_log=True,
        )

    @pytest.fixture
    def store(self, config):
        return UserScopedCacheStore(config=config)

    @pytest.fixture
    def ctx_user1(self, store):
        return UserIsolationContext.create(
            user_id="user_1",
            signer=store.signer,
        )

    @pytest.fixture
    def ctx_user2(self, store):
        return UserIsolationContext.create(
            user_id="user_2",
            signer=store.signer,
        )

    @pytest.mark.asyncio
    async def test_put_and_get(self, store, ctx_user1):
        """基本存取"""
        await store.put(ctx_user1, "hash_a", {"data": "value_a"})
        result = await store.get(ctx_user1, "hash_a")
        assert result == {"data": "value_a"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store, ctx_user1):
        """获取不存在的键返回 None"""
        result = await store.get(ctx_user1, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_isolation(self, store, ctx_user1, ctx_user2):
        """核心隔离测试: 用户 A 的数据对用户 B 不可见"""
        await store.put(ctx_user1, "shared_hash", {"data": "user1_data"})
        
        # 用户 2 使用相同的 content_hash 获取不到用户 1 的数据
        result = await store.get(ctx_user2, "shared_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_independent_partitions(self, store, ctx_user1, ctx_user2):
        """不同用户有独立分区"""
        await store.put(ctx_user1, "hash_a", "data_1")
        await store.put(ctx_user2, "hash_a", "data_2")

        assert await store.get(ctx_user1, "hash_a") == "data_1"
        assert await store.get(ctx_user2, "hash_a") == "data_2"

    @pytest.mark.asyncio
    async def test_per_user_capacity_limit(self, store, ctx_user1):
        """单用户容量上限 (per_user_max_entries=5)"""
        for i in range(7):
            await store.put(ctx_user1, f"hash_{i}", f"data_{i}")

        # 最旧的 2 条应被驱逐
        assert store.get_user_entry_count("user_1") == 5
        assert await store.get(ctx_user1, "hash_0") is None
        assert await store.get(ctx_user1, "hash_1") is None
        assert await store.get(ctx_user1, "hash_6") == "data_6"

    @pytest.mark.asyncio
    async def test_max_partitions_limit(self, store):
        """全局分区上限 (max_user_partitions=3)"""
        for i in range(5):
            ctx = UserIsolationContext.create(
                user_id=f"user_{i}",
                signer=store.signer,
            )
            await store.put(ctx, "hash", f"data_{i}")

        stats = store.get_stats()
        assert stats["user_partitions"] <= 3

    @pytest.mark.asyncio
    async def test_delete(self, store, ctx_user1):
        await store.put(ctx_user1, "hash_a", "data")
        result = await store.delete(ctx_user1, "hash_a")
        assert result is True
        assert await store.get(ctx_user1, "hash_a") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store, ctx_user1):
        result = await store.delete(ctx_user1, "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_user(self, store, ctx_user1, ctx_user2):
        """清除用户所有缓存"""
        await store.put(ctx_user1, "h1", "d1")
        await store.put(ctx_user1, "h2", "d2")
        await store.put(ctx_user2, "h1", "d3")

        count = await store.invalidate_user("user_1")
        assert count == 2
        assert store.get_user_entry_count("user_1") == 0
        # 用户 2 不受影响
        assert await store.get(ctx_user2, "h1") == "d3"

    @pytest.mark.asyncio
    async def test_clear_all(self, store, ctx_user1, ctx_user2):
        await store.put(ctx_user1, "h1", "d1")
        await store.put(ctx_user2, "h1", "d2")
        await store.clear_all()
        assert store.get_stats()["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_stats(self, store, ctx_user1):
        await store.put(ctx_user1, "h1", "d1")
        await store.get(ctx_user1, "h1")  # hit
        await store.get(ctx_user1, "h2")  # miss

        stats = store.get_stats()
        assert stats["total_puts"] == 1
        assert stats["total_gets"] == 2
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_audit_log_recorded(self, store, ctx_user1):
        """操作应记录到审计日志"""
        await store.put(ctx_user1, "h1", "d1")
        await store.get(ctx_user1, "h1")

        records = store.audit_log.get_records(user_id="user_1")
        assert len(records) == 2  # put + get
        assert records[0].action == "put"
        assert records[1].action == "get"

    @pytest.mark.asyncio
    async def test_lru_ordering(self, store, ctx_user1):
        """访问应更新 LRU 顺序"""
        await store.put(ctx_user1, "h1", "d1")
        await store.put(ctx_user1, "h2", "d2")
        await store.put(ctx_user1, "h3", "d3")

        # 访问 h1 使其变为最新
        await store.get(ctx_user1, "h1")

        # 填满至驱逐
        await store.put(ctx_user1, "h4", "d4")
        await store.put(ctx_user1, "h5", "d5")
        await store.put(ctx_user1, "h6", "d6")

        # h2 应该被驱逐 (最旧), h1 因为被访问过应该保留
        assert await store.get(ctx_user1, "h2") is None
        assert await store.get(ctx_user1, "h1") is not None


# ============================================================
# InferenceContextGuard 测试
# ============================================================

class TestInferenceContextGuard:
    """InferenceContextGuard 推理上下文守卫测试"""

    def test_scoped_inference_basic(self):
        guard = InferenceContextGuard()
        with guard.scoped_inference(user_id="user_123") as scope:
            assert scope.user_id == "user_123"
            assert guard._active_user == "user_123"

        # 退出后 active_user 应为 None
        assert guard._active_user is None

    def test_scoped_inference_stats(self):
        guard = InferenceContextGuard()
        with guard.scoped_inference(user_id="user_123"):
            pass

        stats = guard.get_stats()
        assert stats["inferences"] == 1
        assert stats["cleanups"] == 1
        assert stats["active_user"] is None

    def test_concurrent_violation_detection(self):
        guard = InferenceContextGuard()
        # 模拟并发: 设置 active_user
        guard._active_user = "user_A"

        with guard.scoped_inference(user_id="user_B"):
            pass

        assert guard._stats["concurrent_violations"] == 1

    def test_cleanup_on_exception(self):
        guard = InferenceContextGuard()
        with pytest.raises(RuntimeError):
            with guard.scoped_inference(user_id="user_123"):
                raise RuntimeError("Inference failed")

        # 即使异常, active_user 也应被清理
        assert guard._active_user is None
        assert guard._stats["cleanups"] == 1

    def test_cleanup_callback(self):
        guard = InferenceContextGuard()
        callback_called = False

        def cleanup():
            nonlocal callback_called
            callback_called = True

        with guard.scoped_inference(user_id="user_123", cleanup_callback=cleanup):
            pass

        assert callback_called is True

    def test_kv_reference_clearing(self):
        """K/V 张量引用应在退出时清理"""
        guard = InferenceContextGuard()

        try:
            import torch
            from dki.models.base import KVCacheEntry

            entry = KVCacheEntry(
                layer_idx=0,
                key=torch.randn(1, 32, 10, 128),
                value=torch.randn(1, 32, 10, 128),
            )

            with guard.scoped_inference(user_id="user_123", kv_entries=[entry]):
                assert entry.key is not None

            # 退出后 K/V 引用应被清理
            assert entry.key is None
            assert entry.value is None
        except ImportError:
            pytest.skip("torch not available")

    def test_verify_no_residual_no_method(self):
        """模型无 has_injected_kv 方法时假设无残留"""
        guard = InferenceContextGuard()
        model = MagicMock(spec=[])  # 无 has_injected_kv
        assert guard.verify_no_residual(model) is True

    def test_verify_no_residual_clean(self):
        guard = InferenceContextGuard()
        model = MagicMock()
        model.has_injected_kv.return_value = False
        assert guard.verify_no_residual(model) is True

    def test_verify_no_residual_dirty(self):
        guard = InferenceContextGuard()
        model = MagicMock()
        model.has_injected_kv.return_value = True
        assert guard.verify_no_residual(model) is False
        assert guard._stats["residual_detections"] == 1


# ============================================================
# IsolatedPreferenceCacheManager 测试
# ============================================================

class TestIsolatedPreferenceCacheManager:
    """IsolatedPreferenceCacheManager 完整隔离缓存管理器测试"""

    @pytest.fixture
    def cache_manager(self):
        config = UserIsolationConfig(
            hmac_secret="test_secret",
            per_user_max_entries=10,
        )
        return IsolatedPreferenceCacheManager(isolation_config=config)

    def test_create_context(self, cache_manager):
        ctx = cache_manager.create_context(user_id="user_123", session_id="sess_1")
        assert ctx.user_id == "user_123"
        assert ctx.session_id == "sess_1"
        # 上下文应使用管理器的签名器
        signed = ctx.sign_cache_key("test_hash")
        assert cache_manager.signer.verify_key(signed, "user_123") is True

    @pytest.mark.asyncio
    async def test_get_preference_kv_compute(self, cache_manager):
        """首次获取应触发计算"""
        ctx = cache_manager.create_context("user_123")

        mock_model = MagicMock()
        try:
            import torch
            from dki.models.base import KVCacheEntry
            mock_kv = [KVCacheEntry(layer_idx=0, key=torch.randn(1, 32, 5, 128), value=torch.randn(1, 32, 5, 128))]
            mock_model.compute_kv.return_value = (mock_kv, 5)
        except ImportError:
            mock_model.compute_kv.return_value = ([], 0)

        kv, tier_info = await cache_manager.get_preference_kv(
            ctx=ctx,
            preference_text="素食主义者",
            model=mock_model,
        )
        assert tier_info["tier"] == "compute"
        assert tier_info["hit"] is False
        assert tier_info["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_get_preference_kv_l1_hit(self, cache_manager):
        """第二次获取应命中 L1 缓存"""
        ctx = cache_manager.create_context("user_123")

        mock_model = MagicMock()
        try:
            import torch
            from dki.models.base import KVCacheEntry
            mock_kv = [KVCacheEntry(layer_idx=0, key=torch.randn(1, 32, 5, 128), value=torch.randn(1, 32, 5, 128))]
            mock_model.compute_kv.return_value = (mock_kv, 5)
        except ImportError:
            mock_model.compute_kv.return_value = ([], 0)

        # 第一次: 计算
        await cache_manager.get_preference_kv(ctx, "素食主义者", mock_model)

        # 第二次: L1 命中
        kv, tier_info = await cache_manager.get_preference_kv(ctx, "素食主义者", mock_model)
        assert tier_info["tier"] == "L1_memory"
        assert tier_info["hit"] is True

    @pytest.mark.asyncio
    async def test_cross_user_isolation(self, cache_manager):
        """不同用户使用相同偏好文本不应命中缓存"""
        ctx1 = cache_manager.create_context("user_1")
        ctx2 = cache_manager.create_context("user_2")

        mock_model = MagicMock()
        try:
            import torch
            from dki.models.base import KVCacheEntry
            mock_kv = [KVCacheEntry(layer_idx=0, key=torch.randn(1, 32, 5, 128), value=torch.randn(1, 32, 5, 128))]
            mock_model.compute_kv.return_value = (mock_kv, 5)
        except ImportError:
            mock_model.compute_kv.return_value = ([], 0)

        # 用户 1 写入
        await cache_manager.get_preference_kv(ctx1, "素食主义者", mock_model)

        # 用户 2 使用相同文本, 应该 miss (不同分区)
        _, tier_info = await cache_manager.get_preference_kv(ctx2, "素食主义者", mock_model)
        assert tier_info["tier"] == "compute"
        assert tier_info["hit"] is False

    @pytest.mark.asyncio
    async def test_force_recompute(self, cache_manager):
        ctx = cache_manager.create_context("user_123")

        mock_model = MagicMock()
        try:
            import torch
            from dki.models.base import KVCacheEntry
            mock_kv = [KVCacheEntry(layer_idx=0, key=torch.randn(1, 32, 5, 128), value=torch.randn(1, 32, 5, 128))]
            mock_model.compute_kv.return_value = (mock_kv, 5)
        except ImportError:
            mock_model.compute_kv.return_value = ([], 0)

        # 第一次: 计算
        await cache_manager.get_preference_kv(ctx, "素食主义者", mock_model)

        # 强制重新计算
        _, tier_info = await cache_manager.get_preference_kv(
            ctx, "素食主义者", mock_model, force_recompute=True,
        )
        assert tier_info["tier"] == "compute"

    @pytest.mark.asyncio
    async def test_invalidate(self, cache_manager):
        ctx = cache_manager.create_context("user_123")

        mock_model = MagicMock()
        mock_model.compute_kv.return_value = ([], 0)

        await cache_manager.get_preference_kv(ctx, "素食主义者", mock_model)
        count = await cache_manager.invalidate("user_123")
        assert count >= 0

        # 清除后应 miss
        _, tier_info = await cache_manager.get_preference_kv(ctx, "素食主义者", mock_model)
        assert tier_info["hit"] is False

    def test_get_stats(self, cache_manager):
        stats = cache_manager.get_stats()
        assert "total_requests" in stats
        assert "l1_hits" in stats
        assert "l1_store" in stats
        assert "inference_guard" in stats
        assert "audit" in stats

    def test_properties(self, cache_manager):
        assert cache_manager.signer is not None
        assert cache_manager.audit_log is not None
        assert cache_manager.inference_guard is not None


# ============================================================
# CacheAccessRecord 测试
# ============================================================

class TestCacheAccessRecord:

    def test_basic_record(self):
        record = CacheAccessRecord(
            timestamp=time.time(),
            user_id="user_123",
            action="get",
            cache_key="key_1",
            cache_tier="L1_memory",
            success=True,
        )
        assert record.user_id == "user_123"
        assert record.denial_reason == ""

    def test_denial_record(self):
        record = CacheAccessRecord(
            timestamp=time.time(),
            user_id="attacker",
            action="get",
            cache_key="forged_key",
            cache_tier="L1_memory",
            success=False,
            denial_reason="HMAC verification failed",
        )
        assert record.success is False
        assert "HMAC" in record.denial_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
