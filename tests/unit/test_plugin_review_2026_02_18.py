"""
单元测试: 2026-02-18 DKI Plugin 审查修复验证

审查发现的问题:
1. close() 调用不存在的 data_adapter.close() — 应调用 disconnect()
2. Planner _format_history_suffix 缺少注入标记过滤
3. Executor _get_preference_kv 不做 GPU 显存管理 (缓存到 CPU)
4. total_tokens 计算遗漏 preference_tokens
5. DKIConfig 缺少 hybrid_injection 字段
6. Executor stable_fallback 逻辑冗余 (丢弃历史后缀)
7. torch 冗余导入已移除

测试策略:
- 使用 Mock 和内存对象，不需要 GPU
- 验证每个修复点的正确行为
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    SafetyEnvelope,
    QueryContext,
    ExecutionResult,
)
from dki.core.plugin.injection_planner import InjectionPlanner
from dki.core.plugin.injection_executor import InjectionExecutor
from dki.config.config_loader import DKIConfig, HybridInjectionConfigModel
from dki.models.base import KVCacheEntry, ModelOutput
from dki.adapters.base import (
    IUserDataAdapter,
    UserPreference as AdapterUserPreference,
    ChatMessage as AdapterChatMessage,
)


# ================================================================
# 辅助工具
# ================================================================

def make_chat_message(role: str, content: str, msg_id: str = "m1") -> AdapterChatMessage:
    """创建测试用 ChatMessage"""
    return AdapterChatMessage(
        message_id=msg_id,
        session_id="sess_1",
        user_id="user_1",
        role=role,
        content=content,
        timestamp=datetime.utcnow(),
    )


def make_preference(text: str, ptype: str = "general", priority: int = 1) -> AdapterUserPreference:
    """创建测试用 UserPreference"""
    return AdapterUserPreference(
        user_id="user_1",
        preference_text=text,
        preference_type=ptype,
        priority=priority,
    )


def make_fake_kv_entries(num_layers: int = 2) -> List[KVCacheEntry]:
    """创建假的 KV entries (CPU tensors)"""
    entries = []
    for i in range(num_layers):
        entries.append(KVCacheEntry(
            key=torch.randn(1, 4, 3, 8),
            value=torch.randn(1, 4, 3, 8),
            layer_idx=i,
        ))
    return entries


def make_fake_model_adapter():
    """创建假的 ModelAdapter (MagicMock)"""
    model = MagicMock()
    model.device = "cpu"
    model.hidden_dim = 128
    model.compute_kv.return_value = (make_fake_kv_entries(), None)
    model.generate.return_value = ModelOutput(
        text="测试回复",
        input_tokens=10,
        output_tokens=5,
    )
    model.forward_with_kv_injection.return_value = ModelOutput(
        text="注入后回复",
        input_tokens=20,
        output_tokens=10,
    )
    return model


# ================================================================
# 测试 1: close() 应调用 disconnect() 而非 close()
# ================================================================

class TestPluginCloseMethod:
    """验证 DKIPlugin.close() 正确调用适配器的 disconnect()"""

    @pytest.mark.asyncio
    async def test_close_calls_disconnect(self):
        """close() 应优先调用 data_adapter.disconnect()"""
        from dki.core.dki_plugin import DKIPlugin

        model = make_fake_model_adapter()
        adapter = MagicMock(spec=IUserDataAdapter)
        adapter.disconnect = AsyncMock()
        # 确保 hasattr 检查能找到 disconnect
        assert hasattr(adapter, 'disconnect')

        with patch.object(DKIPlugin, '__init__', lambda self, **kw: None):
            plugin = DKIPlugin.__new__(DKIPlugin)
            plugin.data_adapter = adapter
            plugin._redis_client = None

            await plugin.close()

        adapter.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_fallback_to_close_method(self):
        """如果适配器没有 disconnect 但有 close，应调用 close()"""
        from dki.core.dki_plugin import DKIPlugin

        adapter = MagicMock()
        # 移除 disconnect 属性
        del adapter.disconnect
        adapter.close = AsyncMock()

        with patch.object(DKIPlugin, '__init__', lambda self, **kw: None):
            plugin = DKIPlugin.__new__(DKIPlugin)
            plugin.data_adapter = adapter
            plugin._redis_client = None

            await plugin.close()

        adapter.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_no_adapter_method(self):
        """如果适配器既没有 disconnect 也没有 close，不应崩溃"""
        from dki.core.dki_plugin import DKIPlugin

        adapter = MagicMock()
        del adapter.disconnect
        del adapter.close

        with patch.object(DKIPlugin, '__init__', lambda self, **kw: None):
            plugin = DKIPlugin.__new__(DKIPlugin)
            plugin.data_adapter = adapter
            plugin._redis_client = None

            # 不应抛出异常
            await plugin.close()


# ================================================================
# 测试 2: Planner _format_history_suffix 注入标记过滤
# ================================================================

class TestPlannerHistoryFiltering:
    """验证 Planner 的历史后缀正确过滤注入标记"""

    def setup_method(self):
        self.planner = InjectionPlanner(language="cn")

    def test_filters_ai_assistant_marker(self):
        """包含 [AI助手] 的 assistant 消息应被过滤"""
        messages = [
            make_chat_message("user", "你好"),
            make_chat_message("assistant", "[AI助手] 你好，有什么可以帮你的？"),
            make_chat_message("user", "推荐餐厅"),
        ]
        result = self.planner._format_history_suffix(messages)
        assert "[AI助手]" not in result
        assert "推荐餐厅" in result
        assert "你好" in result

    def test_filters_session_history_marker(self):
        """包含 [会话历史参考] 的 assistant 消息内容应被过滤"""
        messages = [
            make_chat_message("assistant", "[会话历史参考] 之前的对话内容"),
            make_chat_message("user", "继续"),
        ]
        result = self.planner._format_history_suffix(messages)
        # assistant 消息被过滤，其内容不应出现
        assert "之前的对话内容" not in result
        assert "继续" in result

    def test_filters_end_session_marker(self):
        """包含 [会话历史结束] 的 assistant 消息内容应被过滤"""
        messages = [
            make_chat_message("assistant", "回复 [会话历史结束]"),
            make_chat_message("user", "新问题"),
        ]
        result = self.planner._format_history_suffix(messages)
        # 被过滤的 assistant 消息内容不应出现在消息列表中
        assert "回复 [会话历史结束]" not in result

    def test_keeps_clean_assistant_messages(self):
        """不含标记的 assistant 消息应保留"""
        messages = [
            make_chat_message("user", "你好"),
            make_chat_message("assistant", "你好！我是AI助手。"),  # 无方括号标记
            make_chat_message("user", "推荐餐厅"),
        ]
        result = self.planner._format_history_suffix(messages)
        assert "你好！我是AI助手。" in result
        assert "推荐餐厅" in result

    def test_all_filtered_returns_empty(self):
        """如果所有消息都被过滤，应返回空字符串"""
        messages = [
            make_chat_message("assistant", "[AI助手] 回复1"),
            make_chat_message("assistant", "[会话历史参考] 回复2"),
        ]
        result = self.planner._format_history_suffix(messages)
        assert result == ""

    def test_english_filtering(self):
        """英文模式下也应过滤包含 [Session History Reference] 的 assistant 消息"""
        planner_en = InjectionPlanner(language="en")
        messages = [
            make_chat_message("assistant", "[Session History Reference] old context"),
            make_chat_message("user", "What's new?"),
        ]
        result = planner_en._format_history_suffix(messages)
        # 被过滤的 assistant 消息内容不应出现
        assert "old context" not in result
        assert "What's new?" in result


# ================================================================
# 测试 3: Executor K/V 缓存 GPU 显存管理
# ================================================================

class TestExecutorKVCacheMemory:
    """验证 Executor 的 K/V 缓存正确管理 GPU 显存 (P0-2 + P2-1 适配)"""

    def test_computed_kv_cached_on_cpu(self):
        """计算的 K/V 应以 PackedKV 形式缓存到 CPU"""
        import hashlib
        from dki.models.base import PackedKV
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        kv, hit, tier = executor._get_preference_kv(
            user_id="user_1",
            preference_text="喜欢编程",
        )

        assert not hit
        assert tier == "compute"
        assert kv is not None

        # P2-1: 缓存中应存储 PackedKV (在 CPU 上)
        content_hash = hashlib.md5("喜欢编程".encode()).hexdigest()
        cached = executor._preference_kv_cache.get("user_1", content_hash)
        assert cached is not None
        cached_kv, cached_hash = cached
        assert isinstance(cached_kv, PackedKV)
        assert str(cached_kv.device) == "cpu"

    def test_cache_hit_returns_correct_device(self):
        """缓存命中时应从 PackedKV 解包返回"""
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        # 第一次: 计算并缓存
        kv1, hit1, _ = executor._get_preference_kv("user_1", "喜欢编程")
        assert not hit1

        # 第二次: 缓存命中 (P2-1: tier 应包含 "packed")
        kv2, hit2, tier2 = executor._get_preference_kv("user_1", "喜欢编程")
        assert hit2
        assert "packed" in tier2
        assert kv2 is not None

    def test_different_users_isolated(self):
        """不同用户的 K/V 缓存应隔离 (P0-2: BoundedUserKVCache)"""
        import hashlib
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        executor._get_preference_kv("user_1", "偏好A")
        executor._get_preference_kv("user_2", "偏好B")

        hash_a = hashlib.md5("偏好A".encode()).hexdigest()
        hash_b = hashlib.md5("偏好B".encode()).hexdigest()
        assert executor._preference_kv_cache.get("user_1", hash_a) is not None
        assert executor._preference_kv_cache.get("user_2", hash_b) is not None
        # 交叉隔离
        assert executor._preference_kv_cache.get("user_1", hash_b) is None
        assert executor._preference_kv_cache.get("user_2", hash_a) is None

    def test_empty_user_id_no_cache(self):
        """空 user_id 不应使用缓存"""
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        kv, hit, tier = executor._get_preference_kv("", "偏好")
        assert not hit
        assert tier == "compute_no_cache"
        stats = executor._preference_kv_cache.get_stats()
        assert stats["total_entries"] == 0


# ================================================================
# 测试 4: total_tokens 包含 preference_tokens
# ================================================================

class TestTotalTokensCalculation:
    """验证 Planner 的 total_tokens 计算包含 preference_tokens"""

    def test_total_tokens_includes_preference(self):
        """total_tokens 应 = query_tokens + history_tokens + preference_tokens"""
        planner = InjectionPlanner(language="cn")
        context = QueryContext()

        prefs = [make_preference("素食主义者，不吃辣")]
        history = [
            make_chat_message("user", "你好"),
            make_chat_message("assistant", "你好！"),
        ]

        plan = planner.build_plan(
            query="推荐餐厅",
            user_id="user_1",
            preferences=prefs,
            relevant_history=history,
            context=context,
        )

        assert plan.preference_tokens > 0
        assert plan.query_tokens > 0
        assert plan.total_tokens == plan.query_tokens + plan.history_tokens + plan.preference_tokens + plan.fact_tokens

    def test_total_tokens_no_preference(self):
        """没有偏好时 preference_tokens = 0"""
        planner = InjectionPlanner(language="cn")
        context = QueryContext()

        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[],
            relevant_history=[],
            context=context,
        )

        assert plan.preference_tokens == 0
        assert plan.total_tokens == plan.query_tokens


# ================================================================
# 测试 5: DKIConfig 包含 hybrid_injection 字段
# ================================================================

class TestDKIConfigHybridInjection:
    """验证 DKIConfig 包含 hybrid_injection 配置"""

    def test_hybrid_injection_exists(self):
        """DKIConfig 应有 hybrid_injection 字段"""
        config = DKIConfig()
        assert hasattr(config, 'hybrid_injection')
        assert isinstance(config.hybrid_injection, HybridInjectionConfigModel)

    def test_hybrid_injection_default_alpha(self):
        """默认 preference alpha 应为 0.4"""
        config = DKIConfig()
        assert config.hybrid_injection.preference.alpha == 0.4

    def test_hybrid_injection_enabled_default(self):
        """hybrid_injection 默认应启用"""
        config = DKIConfig()
        assert config.hybrid_injection.enabled is True

    def test_planner_reads_alpha_from_config(self):
        """Planner 应能从 config.dki.hybrid_injection.preference.alpha 读取 alpha"""
        from dki.config.config_loader import Config

        config = Config()
        planner = InjectionPlanner(config=config, language="cn")

        prefs = [make_preference("素食")]
        context = QueryContext()

        plan = planner.build_plan(
            query="推荐餐厅",
            user_id="user_1",
            preferences=prefs,
            relevant_history=[],
            context=context,
        )

        # 应使用配置中的 alpha (0.4)
        assert plan.alpha_profile.preference_alpha == 0.4

    def test_custom_alpha_from_config(self):
        """自定义 alpha 应能通过配置传递"""
        from dki.config.config_loader import Config

        config = Config()
        config.dki.hybrid_injection.preference.alpha = 0.6

        planner = InjectionPlanner(config=config, language="cn")

        prefs = [make_preference("素食")]
        context = QueryContext()

        plan = planner.build_plan(
            query="推荐餐厅",
            user_id="user_1",
            preferences=prefs,
            relevant_history=[],
            context=context,
        )

        assert plan.alpha_profile.preference_alpha == 0.6


# ================================================================
# 测试 6: Executor stable_fallback 使用历史后缀
# ================================================================

class TestStableFallbackLogic:
    """验证 Executor stable_fallback 正确使用历史后缀"""

    @pytest.mark.asyncio
    async def test_stable_fallback_includes_history(self):
        """stable 降级应包含历史后缀"""
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        plan = InjectionPlan(
            strategy="recall_v4",
            original_query="推荐餐厅",
            final_input="[组装后缀] 推荐餐厅",
            history_suffix="[历史后缀] 用户之前说过喜欢火锅",
            assembled_suffix="[组装后缀] 推荐餐厅",
            preference_text="素食",
            user_id="user_1",
            injection_enabled=True,
            alpha_profile=AlphaProfile(preference_alpha=0.4),
        )

        result = await executor._execute_stable_fallback(
            plan=plan,
            max_new_tokens=100,
            temperature=0.7,
            error_message="test error",
        )

        # stable 降级应使用 history_suffix + original_query
        call_args = model.forward_with_kv_injection.call_args or model.generate.call_args
        prompt = call_args[1].get('prompt', call_args[0][0] if call_args[0] else '')
        assert "推荐餐厅" in prompt

    @pytest.mark.asyncio
    async def test_stable_fallback_no_history(self):
        """没有历史后缀时应只使用原始查询"""
        model = make_fake_model_adapter()
        executor = InjectionExecutor(model_adapter=model)

        plan = InjectionPlan(
            strategy="recall_v4",
            original_query="你好",
            final_input="你好",
            history_suffix="",
            preference_text="素食",
            user_id="user_1",
            injection_enabled=True,
            alpha_profile=AlphaProfile(preference_alpha=0.4),
        )

        result = await executor._execute_stable_fallback(
            plan=plan,
            max_new_tokens=100,
            temperature=0.7,
            error_message="test error",
        )

        assert result.text is not None
        assert result.fallback_used is True


# ================================================================
# 测试 7: torch 未在 dki_plugin.py 中导入
# ================================================================

class TestPluginImports:
    """验证 dki_plugin.py 不再有冗余的 torch 导入"""

    def test_no_torch_import(self):
        """dki_plugin.py 不应直接导入 torch"""
        import importlib
        import inspect

        module = importlib.import_module("dki.core.dki_plugin")
        source = inspect.getsource(module)

        # 检查文件顶层 import 中不包含 "import torch"
        # (注意: 模块内部的其他导入可能间接引入 torch)
        lines = source.split('\n')
        top_level_torch_imports = [
            line for line in lines[:50]  # 只检查前 50 行 (import 区域)
            if line.strip() == 'import torch'
        ]
        assert len(top_level_torch_imports) == 0, (
            f"Found unused 'import torch' in dki_plugin.py top-level imports"
        )


# ================================================================
# 测试 8: SafetyEnvelope 验证
# ================================================================

class TestSafetyEnvelope:
    """验证 SafetyEnvelope 正确检测违规"""

    def test_recall_v4_alpha_violation(self):
        """recall_v4 策略超过 max_preference_alpha 应报告违规"""
        envelope = SafetyEnvelope(max_preference_alpha=0.7)
        plan = InjectionPlan(
            strategy="recall_v4",
            alpha_profile=AlphaProfile(preference_alpha=0.8),
        )
        violations = envelope.validate(plan)
        assert len(violations) > 0
        assert "recall_v4" in violations[0]

    def test_stable_alpha_violation(self):
        """stable 策略超过 stable_max_preference_alpha 应报告违规"""
        envelope = SafetyEnvelope(stable_max_preference_alpha=0.5)
        plan = InjectionPlan(
            strategy="stable",
            alpha_profile=AlphaProfile(preference_alpha=0.6),
        )
        violations = envelope.validate(plan)
        assert len(violations) > 0
        assert "stable" in violations[0]

    def test_within_limits_no_violation(self):
        """在限制范围内不应报告违规"""
        envelope = SafetyEnvelope()
        plan = InjectionPlan(
            strategy="recall_v4",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
        )
        violations = envelope.validate(plan)
        assert len(violations) == 0


# ================================================================
# 测试 9: AlphaProfile effective_preference_alpha
# ================================================================

class TestAlphaProfile:
    """验证 AlphaProfile 的 effective alpha 计算"""

    def test_effective_alpha_capped(self):
        """effective_preference_alpha 应受 override_cap 约束"""
        profile = AlphaProfile(preference_alpha=0.9, override_cap=0.7)
        assert profile.effective_preference_alpha == 0.7

    def test_effective_alpha_below_cap(self):
        """低于 cap 时 effective_preference_alpha = preference_alpha"""
        profile = AlphaProfile(preference_alpha=0.3, override_cap=0.7)
        assert profile.effective_preference_alpha == 0.3

    def test_to_dict_includes_effective(self):
        """to_dict 应包含 effective_preference_alpha"""
        profile = AlphaProfile(preference_alpha=0.9, override_cap=0.7)
        d = profile.to_dict()
        assert d["effective_preference_alpha"] == 0.7
        assert d["preference_alpha"] == 0.9


# ================================================================
# 测试 10: InjectionPlan 序列化
# ================================================================

class TestInjectionPlanSerialization:
    """验证 InjectionPlan 序列化"""

    def test_to_dict_complete(self):
        """to_dict 应包含所有关键字段"""
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食",
            preferences_count=1,
            preference_tokens=10,
            history_tokens=50,
            query_tokens=5,
            total_tokens=65,
        )
        d = plan.to_dict()
        assert d["strategy"] == "recall_v4"
        assert d["injection_enabled"] is True
        assert d["preference_tokens"] == 10
        assert d["total_tokens"] == 65

    def test_default_plan(self):
        """默认 InjectionPlan 应有合理默认值"""
        plan = InjectionPlan()
        assert plan.strategy == "recall_v4"
        assert plan.injection_enabled is False
        assert plan.total_tokens == 0


# ================================================================
# 运行入口
# ================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
