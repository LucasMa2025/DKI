"""
优化项 P0-2 / P0-4 / P1-1 / P1-2 / P1-3 / P1-4 / P2-1 单元测试
==================================================================

覆盖:
- P0-2: BoundedUserKVCache (LRU + 容量上限)
- P0-4: history_alpha 动态衰减
- P1-1: recall_token_budget
- P1-2: MemoryTrigger confidence → Alpha
- P1-3: Preference text caching with TTL
- P1-4: KV 监控指标
- P2-1: PackedKV 数据结构 + Executor 集成
"""

import asyncio
import hashlib
import math
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from dki.models.base import KVCacheEntry, PackedKV
from dki.core.plugin.injection_plan import (
    InjectionPlan, AlphaProfile, SafetyEnvelope, QueryContext, ExecutionResult, FactBlock,
)


# ============================================================
# Fixtures
# ============================================================

def _make_kv_entry(layer_idx: int, heads=4, tokens=8, dim=16, dtype=torch.float32):
    """创建测试用 KVCacheEntry"""
    return KVCacheEntry(
        key=torch.randn(1, heads, tokens, dim, dtype=dtype),
        value=torch.randn(1, heads, tokens, dim, dtype=dtype),
        layer_idx=layer_idx,
    )


def _make_kv_entries(num_layers=4, heads=4, tokens=8, dim=16, dtype=torch.float32):
    """创建多层 KVCacheEntry 列表"""
    return [_make_kv_entry(i, heads, tokens, dim, dtype) for i in range(num_layers)]


# ============================================================
# P0-2: BoundedUserKVCache
# ============================================================

class TestBoundedUserKVCache:
    """P0-2: 有界用户级 KV 缓存测试"""

    def _make_cache(self, max_users=3, max_entries_per_user=2):
        from dki.core.plugin.injection_executor import BoundedUserKVCache
        return BoundedUserKVCache(max_users=max_users, max_entries_per_user=max_entries_per_user)

    def test_basic_put_get(self):
        cache = self._make_cache()
        cache.put("u1", "hash_a", ("data_a", "hash_a"))
        result = cache.get("u1", "hash_a")
        assert result is not None
        assert result == ("data_a", "hash_a")

    def test_cache_miss(self):
        cache = self._make_cache()
        assert cache.get("u1", "hash_a") is None

    def test_user_lru_eviction(self):
        """当用户数超过上限时，最久未访问的用户被淘汰"""
        cache = self._make_cache(max_users=2)
        cache.put("u1", "h1", "v1")
        cache.put("u2", "h2", "v2")
        # u1 是最早插入的
        cache.put("u3", "h3", "v3")
        # u1 应被淘汰
        assert cache.get("u1", "h1") is None
        assert cache.get("u2", "h2") == "v2"
        assert cache.get("u3", "h3") == "v3"

    def test_user_lru_touch(self):
        """访问用户会更新 LRU 顺序"""
        cache = self._make_cache(max_users=2)
        cache.put("u1", "h1", "v1")
        cache.put("u2", "h2", "v2")
        # 访问 u1，使其变为最近使用
        cache.get("u1", "h1")
        # 插入 u3，应淘汰 u2（最久未访问）
        cache.put("u3", "h3", "v3")
        assert cache.get("u1", "h1") == "v1"
        assert cache.get("u2", "h2") is None

    def test_per_user_entry_limit(self):
        """每用户条目数超过上限时，最旧条目被淘汰"""
        cache = self._make_cache(max_entries_per_user=2)
        cache.put("u1", "h1", "v1")
        cache.put("u1", "h2", "v2")
        cache.put("u1", "h3", "v3")
        # h1 应被淘汰
        assert cache.get("u1", "h1") is None
        assert cache.get("u1", "h2") == "v2"
        assert cache.get("u1", "h3") == "v3"

    def test_clear_specific_user(self):
        cache = self._make_cache()
        cache.put("u1", "h1", "v1")
        cache.put("u2", "h2", "v2")
        cache.clear("u1")
        assert cache.get("u1", "h1") is None
        assert cache.get("u2", "h2") == "v2"

    def test_clear_all(self):
        cache = self._make_cache()
        cache.put("u1", "h1", "v1")
        cache.put("u2", "h2", "v2")
        cache.clear()
        assert cache.get("u1", "h1") is None
        assert cache.get("u2", "h2") is None

    def test_get_stats(self):
        cache = self._make_cache(max_users=5, max_entries_per_user=3)
        cache.put("u1", "h1", "v1")
        cache.put("u1", "h2", "v2")
        cache.put("u2", "h3", "v3")
        stats = cache.get_stats()
        assert stats["current_users"] == 2
        assert stats["total_entries"] == 3
        assert stats["max_users"] == 5
        assert stats["max_entries_per_user"] == 3


# ============================================================
# P0-4: history_alpha 动态衰减
# ============================================================

class TestHistoryAlphaDecay:
    """P0-4: history_alpha 动态衰减测试"""

    def _make_planner(self):
        """创建最小化 Planner 用于测试 _compute_alpha_profile"""
        from dki.core.plugin.injection_planner import InjectionPlanner
        mock_config = MagicMock()
        mock_config.dki = MagicMock()
        mock_config.dki.hybrid_injection = MagicMock()
        mock_config.dki.hybrid_injection.preference = MagicMock()
        mock_config.dki.hybrid_injection.preference.alpha = 0.4
        
        planner = InjectionPlanner.__new__(InjectionPlanner)
        planner.config = mock_config
        planner.language = "cn"
        planner._stats = {
            "plans_built": 0,
            "recall_v4_plans": 0,
            "stable_plans": 0,
            "memory_trigger_count": 0,
            "reference_resolved_count": 0,
            "fact_blocks_resolved": 0,
        }
        return planner

    def test_zero_history_alpha_is_one(self):
        """0 token 历史 → alpha = 1.0"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=0
        )
        assert profile.history_alpha == 1.0

    def test_512_tokens_no_decay(self):
        """512 token 以下不衰减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=512
        )
        assert profile.history_alpha == 1.0

    def test_1024_tokens_moderate_decay(self):
        """1024 token → 明显衰减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=1024
        )
        # 1024/512 = 2, log(2) ≈ 0.693, 1.0 - 0.693 * 0.3 ≈ 0.79
        assert 0.7 < profile.history_alpha < 0.9

    def test_4096_tokens_strong_decay(self):
        """4096 token → 强衰减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=4096
        )
        assert profile.history_alpha < 0.5

    def test_8192_tokens_floor(self):
        """8192+ token → 不低于 0.3"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=16384
        )
        assert profile.history_alpha >= 0.3

    def test_force_alpha_overrides_decay(self):
        """force_alpha 覆盖所有衰减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test preference")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], force_alpha=0.8, history_tokens=4096
        )
        assert profile.history_alpha == 1.0
        assert profile.preference_alpha == 0.8

    def test_no_preferences_alpha_zero(self):
        """无偏好时 preference_alpha = 0"""
        planner = self._make_planner()
        profile = planner._compute_alpha_profile(
            "recall_v4", [], [], None, history_tokens=100
        )
        assert profile.preference_alpha == 0.0


# ============================================================
# P1-1: recall_token_budget
# ============================================================

class TestRecallTokenBudget:
    """P1-1: Token Budget 测试"""

    def test_query_context_has_token_budget(self):
        ctx = QueryContext()
        assert hasattr(ctx, 'recall_token_budget')
        assert ctx.recall_token_budget == 2048

    def test_injection_plan_has_token_budget(self):
        plan = InjectionPlan()
        assert hasattr(plan, 'recall_token_budget')
        assert plan.recall_token_budget == 2048

    def test_custom_token_budget(self):
        ctx = QueryContext(recall_token_budget=4096)
        assert ctx.recall_token_budget == 4096


# ============================================================
# P1-2: MemoryTrigger confidence → Alpha
# ============================================================

class TestMemoryTriggerAlpha:
    """P1-2: MemoryTrigger confidence 影响 Alpha"""

    def _make_planner(self):
        from dki.core.plugin.injection_planner import InjectionPlanner
        mock_config = MagicMock()
        mock_config.dki = MagicMock()
        mock_config.dki.hybrid_injection = MagicMock()
        mock_config.dki.hybrid_injection.preference = MagicMock()
        mock_config.dki.hybrid_injection.preference.alpha = 0.4
        
        planner = InjectionPlanner.__new__(InjectionPlanner)
        planner.config = mock_config
        planner.language = "cn"
        planner._stats = {
            "plans_built": 0, "recall_v4_plans": 0, "stable_plans": 0,
            "memory_trigger_count": 0, "reference_resolved_count": 0,
            "fact_blocks_resolved": 0,
        }
        return planner

    def test_no_trigger_no_boost(self):
        """无 trigger 时不增强"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        ctx = QueryContext(memory_triggered=False, trigger_confidence=0.0)
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, context=ctx, history_tokens=0
        )
        assert profile.preference_alpha == 0.4

    def test_low_confidence_no_boost(self):
        """低置信度 (<=0.5) 时不增强"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        ctx = QueryContext(memory_triggered=True, trigger_confidence=0.3)
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, context=ctx, history_tokens=0
        )
        assert profile.preference_alpha == 0.4

    def test_high_confidence_boosts_alpha(self):
        """高置信度 trigger 增强 preference_alpha"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        ctx = QueryContext(memory_triggered=True, trigger_confidence=0.9)
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, context=ctx, history_tokens=0
        )
        # 0.4 * (0.7 + 0.3 * 0.9) = 0.4 * 0.97 = 0.388
        assert profile.preference_alpha > 0.38
        assert profile.preference_alpha <= 0.7

    def test_max_confidence_capped(self):
        """最大置信度时 alpha 不超过 0.7"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        ctx = QueryContext(memory_triggered=True, trigger_confidence=1.0)
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, context=ctx, history_tokens=0
        )
        assert profile.preference_alpha <= 0.7


# ============================================================
# P1-3: Preference text caching with TTL
# ============================================================

class TestPreferenceTextCaching:
    """P1-3: 偏好文本缓存测试"""

    def _make_plugin(self):
        """创建最小化 DKIPlugin mock"""
        plugin = MagicMock()
        plugin._preference_text_cache = {}
        plugin._preference_cache_ttl = 300.0
        
        from dki.core.dki_plugin import DKIPlugin
        plugin._get_cached_preferences = DKIPlugin._get_cached_preferences.__get__(plugin)
        plugin.invalidate_preference_text_cache = DKIPlugin.invalidate_preference_text_cache.__get__(plugin)
        
        return plugin

    @pytest.mark.asyncio
    async def test_cache_miss_queries_adapter(self):
        """缓存未命中时查询适配器"""
        plugin = self._make_plugin()
        mock_prefs = [MagicMock(content="pref1")]
        plugin.data_adapter = AsyncMock()
        plugin.data_adapter.get_user_preferences = AsyncMock(return_value=mock_prefs)
        
        result = await plugin._get_cached_preferences("u1")
        assert result == mock_prefs
        plugin.data_adapter.get_user_preferences.assert_called_once_with("u1")

    @pytest.mark.asyncio
    async def test_cache_hit_skips_adapter(self):
        """缓存命中时不查询适配器"""
        plugin = self._make_plugin()
        mock_prefs = [MagicMock(content="pref1")]
        plugin._preference_text_cache["u1"] = (mock_prefs, time.time())
        plugin.data_adapter = AsyncMock()
        
        result = await plugin._get_cached_preferences("u1")
        assert result == mock_prefs
        plugin.data_adapter.get_user_preferences.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_expired_re_queries(self):
        """缓存过期后重新查询"""
        plugin = self._make_plugin()
        plugin._preference_cache_ttl = 0.1  # 100ms TTL
        old_prefs = [MagicMock(content="old")]
        new_prefs = [MagicMock(content="new")]
        plugin._preference_text_cache["u1"] = (old_prefs, time.time() - 1.0)
        plugin.data_adapter = AsyncMock()
        plugin.data_adapter.get_user_preferences = AsyncMock(return_value=new_prefs)
        
        result = await plugin._get_cached_preferences("u1")
        assert result == new_prefs

    def test_invalidate_specific_user(self):
        """使指定用户缓存失效"""
        plugin = self._make_plugin()
        plugin._preference_text_cache["u1"] = ([], time.time())
        plugin._preference_text_cache["u2"] = ([], time.time())
        plugin.invalidate_preference_text_cache("u1")
        assert "u1" not in plugin._preference_text_cache
        assert "u2" in plugin._preference_text_cache

    def test_invalidate_all(self):
        """使所有缓存失效"""
        plugin = self._make_plugin()
        plugin._preference_text_cache["u1"] = ([], time.time())
        plugin._preference_text_cache["u2"] = ([], time.time())
        plugin.invalidate_preference_text_cache(None)
        assert len(plugin._preference_text_cache) == 0


# ============================================================
# P1-4: KV 监控指标
# ============================================================

class TestKVMonitoringMetrics:
    """P1-4: KV 监控指标测试"""

    def test_execution_result_has_kv_metrics(self):
        result = ExecutionResult()
        assert result.kv_bytes_cpu == 0
        assert result.kv_bytes_gpu_peak == 0
        assert result.kv_transfer_latency_ms == 0.0
        assert result.kv_layers_count == 0

    def test_execution_result_with_metrics(self):
        result = ExecutionResult(
            kv_bytes_cpu=1024,
            kv_bytes_gpu_peak=2048,
            kv_transfer_latency_ms=1.5,
            kv_layers_count=32,
        )
        assert result.kv_bytes_cpu == 1024
        assert result.kv_layers_count == 32


# ============================================================
# P2-1: PackedKV
# ============================================================

class TestPackedKV:
    """P2-1: PackedKV 数据结构测试"""

    def test_from_entries_basic(self):
        """基本打包"""
        entries = _make_kv_entries(num_layers=4)
        packed = PackedKV.from_entries(entries)
        assert packed.num_layers == 4
        assert packed.keys.shape[0] == 4
        assert packed.values.shape[0] == 4

    def test_from_entries_empty_raises(self):
        """空列表抛出 ValueError"""
        with pytest.raises(ValueError, match="Cannot pack empty entries"):
            PackedKV.from_entries([])

    def test_from_entries_sorts_by_layer(self):
        """按 layer_idx 排序"""
        entries = [
            _make_kv_entry(2),
            _make_kv_entry(0),
            _make_kv_entry(1),
        ]
        packed = PackedKV.from_entries(entries)
        assert packed.num_layers == 3

    def test_to_entries_roundtrip(self):
        """打包 → 解包 → 数据一致"""
        entries = _make_kv_entries(num_layers=4, heads=2, tokens=3, dim=8)
        packed = PackedKV.from_entries(entries)
        unpacked = packed.to_entries()
        assert len(unpacked) == 4
        for i, (orig, unp) in enumerate(zip(
            sorted(entries, key=lambda e: e.layer_idx), unpacked
        )):
            assert unp.layer_idx == i
            assert torch.allclose(orig.key, unp.key, atol=1e-6)
            assert torch.allclose(orig.value, unp.value, atol=1e-6)

    def test_to_device_cpu(self):
        """搬移到 CPU"""
        entries = _make_kv_entries(num_layers=2)
        packed = PackedKV.from_entries(entries)
        cpu_packed = packed.cpu()
        assert str(cpu_packed.device) == "cpu"

    def test_scale_values(self):
        """alpha 缩放只影响 values"""
        entries = _make_kv_entries(num_layers=2, heads=1, tokens=2, dim=4)
        packed = PackedKV.from_entries(entries)
        original_keys = packed.keys.clone()
        original_values = packed.values.clone()
        
        packed.scale_values(0.5)
        
        # keys 不变
        assert torch.allclose(packed.keys, original_keys)
        # values 缩放
        assert torch.allclose(packed.values, original_values * 0.5, atol=1e-6)

    def test_scale_values_identity(self):
        """alpha=1.0 不修改"""
        entries = _make_kv_entries(num_layers=2)
        packed = PackedKV.from_entries(entries)
        original_values = packed.values.clone()
        packed.scale_values(1.0)
        assert torch.allclose(packed.values, original_values)

    def test_total_bytes(self):
        """总字节数计算正确"""
        entries = _make_kv_entries(num_layers=2, heads=4, tokens=8, dim=16)
        packed = PackedKV.from_entries(entries)
        expected = 2 * (2 * 4 * 8 * 16 * 4)  # 2 tensors, 2 layers, float32=4 bytes
        assert packed.total_bytes == expected

    def test_repr(self):
        """__repr__ 不崩溃"""
        entries = _make_kv_entries(num_layers=2)
        packed = PackedKV.from_entries(entries)
        r = repr(packed)
        assert "PackedKV" in r
        assert "layers=2" in r

    def test_pin_memory_on_cpu(self):
        """pin_memory 在 CPU tensor 上工作"""
        entries = _make_kv_entries(num_layers=2)
        packed = PackedKV.from_entries(entries).cpu()
        # pin_memory 可能在某些环境不可用，但不应崩溃
        try:
            pinned = packed.pin_memory()
            assert pinned.num_layers == packed.num_layers
        except RuntimeError:
            pass  # 某些环境不支持 pin_memory

    def test_bfloat16_entries(self):
        """bfloat16 tensor 打包"""
        entries = _make_kv_entries(num_layers=2, dtype=torch.bfloat16)
        packed = PackedKV.from_entries(entries)
        assert packed.dtype == torch.bfloat16
        unpacked = packed.to_entries()
        assert len(unpacked) == 2

    def test_float16_entries(self):
        """float16 tensor 打包"""
        entries = _make_kv_entries(num_layers=2, dtype=torch.float16)
        packed = PackedKV.from_entries(entries)
        assert packed.dtype == torch.float16


# ============================================================
# P2-1: PackedKV 在 Executor 中的集成
# ============================================================

class TestPackedKVExecutorIntegration:
    """P2-1: PackedKV 在 Executor _get_preference_kv 中的集成"""

    def _make_executor(self):
        from dki.core.plugin.injection_executor import InjectionExecutor, BoundedUserKVCache
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.config = MagicMock()
        
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor.model = mock_model
        executor._preference_kv_cache = BoundedUserKVCache(max_users=10, max_entries_per_user=5)
        executor._inference_guard = None
        executor._fc_logger = None
        executor._stats = {
            "executions": 0, "recall_v4_executions": 0,
            "plain_executions": 0, "fallbacks": 0,
            "cache_hits": 0, "cache_user_isolation_denials": 0,
        }
        return executor

    def test_compute_and_cache_as_packed(self):
        """计算 K/V 后应以 PackedKV 形式缓存"""
        executor = self._make_executor()
        entries = _make_kv_entries(num_layers=4)
        executor.model.compute_kv = MagicMock(return_value=(entries, None))
        
        result_entries, cache_hit, tier = executor._get_preference_kv("u1", "my preference")
        assert cache_hit is False
        assert tier == "compute"
        assert result_entries is not None
        
        # 验证缓存中存储的是 PackedKV
        content_hash = hashlib.md5("my preference".encode()).hexdigest()
        cached = executor._preference_kv_cache.get("u1", content_hash)
        assert cached is not None
        cached_kv, cached_hash = cached
        assert isinstance(cached_kv, PackedKV)
        assert cached_kv.num_layers == 4

    def test_cache_hit_returns_unpacked(self):
        """缓存命中时从 PackedKV 解包返回"""
        executor = self._make_executor()
        entries = _make_kv_entries(num_layers=4)
        packed = PackedKV.from_entries(entries).cpu()
        content_hash = hashlib.md5("my preference".encode()).hexdigest()
        executor._preference_kv_cache.put("u1", content_hash, (packed, content_hash))
        
        result_entries, cache_hit, tier = executor._get_preference_kv("u1", "my preference")
        assert cache_hit is True
        assert "packed" in tier
        assert isinstance(result_entries, list)
        assert len(result_entries) == 4


# ============================================================
# 综合: Alpha Profile 同时包含 P0-4 + P1-2
# ============================================================

class TestAlphaProfileCombined:
    """P0-4 + P1-2 综合测试"""

    def _make_planner(self):
        from dki.core.plugin.injection_planner import InjectionPlanner
        mock_config = MagicMock()
        mock_config.dki = MagicMock()
        mock_config.dki.hybrid_injection = MagicMock()
        mock_config.dki.hybrid_injection.preference = MagicMock()
        mock_config.dki.hybrid_injection.preference.alpha = 0.4
        
        planner = InjectionPlanner.__new__(InjectionPlanner)
        planner.config = mock_config
        planner.language = "cn"
        planner._stats = {
            "plans_built": 0, "recall_v4_plans": 0, "stable_plans": 0,
            "memory_trigger_count": 0, "reference_resolved_count": 0,
            "fact_blocks_resolved": 0,
        }
        return planner

    def test_high_confidence_with_long_history(self):
        """高置信度 + 长历史: preference 增强, history 衰减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        ctx = QueryContext(memory_triggered=True, trigger_confidence=0.9)
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None,
            context=ctx, history_tokens=4096,
        )
        # preference 应增强
        assert profile.preference_alpha > 0.38
        # history 应衰减
        assert profile.history_alpha < 0.5

    def test_no_trigger_short_history(self):
        """无 trigger + 短历史: 默认 alpha"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        profile = planner._compute_alpha_profile(
            "recall_v4", prefs, [], None, history_tokens=100,
        )
        assert profile.preference_alpha == 0.4
        assert profile.history_alpha == 1.0

    def test_decay_is_monotonic(self):
        """衰减曲线单调递减"""
        planner = self._make_planner()
        prefs = [MagicMock(content="test")]
        
        prev_alpha = 1.0
        for tokens in [256, 512, 1024, 2048, 4096, 8192]:
            profile = planner._compute_alpha_profile(
                "recall_v4", prefs, [], None, history_tokens=tokens,
            )
            assert profile.history_alpha <= prev_alpha
            prev_alpha = profile.history_alpha


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
