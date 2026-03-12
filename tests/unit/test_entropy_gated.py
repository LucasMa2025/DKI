# -*- coding: utf-8 -*-
"""
单元测试: Entropy-Gated Metacognitive Fact Retrieval (v8.0)

测试范围:
- EntropyMonitor 熵计算与突增检测
- EGDMIController 两阶段编排
- EGDMIPromptBuilder 提示词构造
- InjectionExecutor._execute_entropy_gated 路由
- _get_fact_retrieve_method 对 entropy_gated / auto 的路由
- RecallConfig entropy_gated 新字段解析
- DKIPlugin.chat / chat_stream entropy_gated 路由
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# Mock 类
# ============================================================

@dataclass
class MockModelOutput:
    """模拟 ModelOutput"""
    text: str = ""
    input_tokens: int = 100
    output_tokens: int = 50
    latency_ms: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    logprobs: Optional[List[List[float]]] = None


class MockRecallConfig:
    """模拟 RecallConfig"""
    def __init__(self, method="post_hoc", **extra):
        self.fact_call = MockFactCallConfig(method=method, **extra)


class MockFactCallConfig:
    """模拟 RecallFactCallConfig"""
    def __init__(
        self,
        method="post_hoc",
        max_rounds=3,
        max_param_tokens=64,
        max_fact_tokens=800,
        entropy_probe_tokens=64,
        entropy_logprobs_k=5,
        entropy_spike_threshold=1.5,
        entropy_absolute_threshold=3.0,
        entropy_window_size=32,
        entropy_max_retrievals=2,
    ):
        self.enabled = True
        self.max_rounds = max_rounds
        self.max_fact_tokens = max_fact_tokens
        self.batch_size = 5
        self.fact_retrieve_method = method
        self.inline_intercept_max_param_tokens = max_param_tokens
        self.inline_intercept_max_rounds = max_rounds
        self.entropy_probe_tokens = entropy_probe_tokens
        self.entropy_logprobs_k = entropy_logprobs_k
        self.entropy_spike_threshold = entropy_spike_threshold
        self.entropy_absolute_threshold = entropy_absolute_threshold
        self.entropy_window_size = entropy_window_size
        self.entropy_max_retrievals = entropy_max_retrievals


class MockModelAdapter:
    """模拟 ModelAdapter"""
    def __init__(self, is_closed_source=False):
        self.is_closed_source = is_closed_source
        self.model_name = "test-model"
        self.max_model_len = 4096
        self._generate_calls = []

    async def async_generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        self._generate_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs,
        })
        return MockModelOutput(text="Test response")

    def get_model_info(self):
        return {"model_name": self.model_name}


# ============================================================
# 测试: EntropyMonitor
# ============================================================

class TestEntropyMonitor:
    """测试 EntropyMonitor 熵计算与突增检测"""

    def test_entropy_from_uniform_logprobs(self):
        """均匀分布 → 高熵"""
        from dki.core.entropy_gated_injection import EntropyMonitor
        # 5 个等概率 → H = log(5) ≈ 1.609
        logprobs = [math.log(0.2)] * 5
        entropy = EntropyMonitor._compute_entropy_from_logprobs(logprobs)
        assert abs(entropy - math.log(5)) < 0.01

    def test_entropy_from_peaked_logprobs(self):
        """尖峰分布 → 低熵"""
        from dki.core.entropy_gated_injection import EntropyMonitor
        logprobs = [math.log(0.99), math.log(0.005), math.log(0.005)]
        entropy = EntropyMonitor._compute_entropy_from_logprobs(logprobs)
        assert entropy < 0.1

    def test_entropy_from_empty_logprobs(self):
        """空 logprobs → 0"""
        from dki.core.entropy_gated_injection import EntropyMonitor
        assert EntropyMonitor._compute_entropy_from_logprobs([]) == 0.0

    def test_should_check_interval(self):
        """按 check_interval 间隔检查"""
        from dki.core.entropy_gated_injection import EntropyMonitor, EGDMIConfig
        config = EGDMIConfig(entropy_check_interval=4, max_retrieval_per_generation=10)
        monitor = EntropyMonitor(config)

        # total_tokens 初始为 0, 0 % 4 == 0 → True
        assert monitor.should_check() is True

        # 手动推进 tokens
        for _ in range(3):
            monitor.update([math.log(0.5), math.log(0.5)])
        # 第 3 次 update 后 total_tokens=3, 3 % 4 ≠ 0
        assert monitor.should_check() is False

        # 再推进 1 次, total_tokens=4
        monitor.update([math.log(0.5), math.log(0.5)])
        assert monitor.should_check() is True

    def test_max_retrieval_limit(self):
        """超出 max_retrieval 后不再检查"""
        from dki.core.entropy_gated_injection import EntropyMonitor, EGDMIConfig
        config = EGDMIConfig(
            entropy_check_interval=1,
            max_retrieval_per_generation=1,
            entropy_absolute_threshold=0.5,  # 很低, 容易触发
        )
        monitor = EntropyMonitor(config)

        # 给均匀 logprobs 触发高熵
        uniform_lp = [math.log(0.2)] * 5
        state1 = monitor.update(uniform_lp)
        assert state1.is_high_entropy is True
        # retrieval_count = 1 → 已达到上限
        assert monitor.should_check() is False

    def test_reset_clears_state(self):
        """reset 后状态清零"""
        from dki.core.entropy_gated_injection import EntropyMonitor, EGDMIConfig
        config = EGDMIConfig(entropy_check_interval=1)
        monitor = EntropyMonitor(config)

        monitor.update([math.log(0.5), math.log(0.5)])
        assert monitor._total_tokens == 1

        monitor.reset()
        assert monitor._total_tokens == 0
        assert monitor._retrieval_count == 0
        assert len(monitor._entropy_window) == 0

    def test_spike_detection_relative(self):
        """相对突增检测 (对比窗口内统计)"""
        from dki.core.entropy_gated_injection import EntropyMonitor, EGDMIConfig
        config = EGDMIConfig(
            entropy_check_interval=1,
            entropy_window_size=8,
            entropy_spike_threshold=1.5,
            entropy_absolute_threshold=100.0,  # 关闭绝对阈值
            max_retrieval_per_generation=5,
        )
        monitor = EntropyMonitor(config)

        # 先填充窗口: 低熵 token
        low_entropy_lp = [math.log(0.99), math.log(0.01)]
        for _ in range(8):
            state = monitor.update(low_entropy_lp)
            assert state.is_high_entropy is False

        # 突然一个高熵 token
        high_entropy_lp = [math.log(0.2)] * 5
        state_high = monitor.update(high_entropy_lp)
        assert state_high.is_high_entropy is True
        assert state_high.relative_entropy_spike > 1.5


# ============================================================
# 测试: EGDMIPromptBuilder
# ============================================================

class TestEGDMIPromptBuilder:
    """测试 EGDMI 提示词构造"""

    def test_basic_prompt_build(self):
        """基本提示词构造"""
        from dki.core.entropy_gated_injection import EGDMIPromptBuilder, EGDMIConfig
        config = EGDMIConfig(context_window=4096, preference_max_tokens=200)
        builder = EGDMIPromptBuilder(config)

        prompt = builder.build(
            query="你好吗",
            preferences=["喜欢户外运动", "住在上海"],
            history_summary="之前聊了天气",
            summary_trace_ids=["msg_001"],
            recent_messages=[
                {"role": "user", "content": "昨天的天气如何?"},
                {"role": "assistant", "content": "昨天晴天。"},
            ],
        )

        assert prompt.full_prompt is not None
        assert "喜欢户外运动" in prompt.full_prompt
        assert "你好吗" in prompt.full_prompt
        assert prompt.total_tokens > 0

    def test_empty_preferences(self):
        """无偏好时仍能构建"""
        from dki.core.entropy_gated_injection import EGDMIPromptBuilder, EGDMIConfig
        builder = EGDMIPromptBuilder(EGDMIConfig())
        prompt = builder.build(
            query="test", preferences=[], history_summary="",
            summary_trace_ids=[], recent_messages=[],
        )
        assert "test" in prompt.full_prompt
        assert prompt.preference_tokens == 0

    def test_token_estimation(self):
        """token 估算: 中英文混合"""
        from dki.core.entropy_gated_injection import EGDMIPromptBuilder
        tokens = EGDMIPromptBuilder._estimate_tokens("Hello 你好世界")
        assert tokens > 0
        assert EGDMIPromptBuilder._estimate_tokens("") == 0


# ============================================================
# 测试: EGDMIController
# ============================================================

class TestEGDMIController:
    """测试 EGDMI 主控制器"""

    def test_init_defaults(self):
        """默认初始化"""
        from dki.core.entropy_gated_injection import EGDMIController
        ctrl = EGDMIController()
        assert ctrl.prompt_builder is not None
        assert ctrl.entropy_monitor is not None
        assert ctrl.retriever is None

    def test_set_retriever(self):
        """设置记忆检索器"""
        from dki.core.entropy_gated_injection import EGDMIController
        ctrl = EGDMIController()
        mock_recall = MagicMock()
        ctrl.set_retriever(mock_recall)
        assert ctrl.retriever is not None

    def test_inject_grounding_into_prompt(self):
        """grounding 注入到 prompt"""
        from dki.core.entropy_gated_injection import EGDMIController, GroundingContext
        ctrl = EGDMIController()

        prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        grounding = GroundingContext(
            facts=["用户喜欢登山", "用户住在北京"],
            source_trace_ids=["t1", "t2"],
        )

        result = ctrl.inject_grounding_into_prompt(prompt, grounding)
        assert "用户喜欢登山" in result
        assert "用户住在北京" in result
        assert "<|im_start|>assistant\n" in result

    def test_inject_grounding_no_marker(self):
        """prompt 无 assistant 标记时追加到末尾"""
        from dki.core.entropy_gated_injection import EGDMIController, GroundingContext
        ctrl = EGDMIController()

        prompt = "Simple prompt"
        grounding = GroundingContext(facts=["Fact1"])
        result = ctrl.inject_grounding_into_prompt(prompt, grounding)
        assert "Fact1" in result

    def test_create_entropy_callback(self):
        """entropy callback 创建与基本调用"""
        from dki.core.entropy_gated_injection import EGDMIController
        ctrl = EGDMIController()

        callback = ctrl.create_entropy_callback(
            original_query="test",
            session_id="s1",
        )
        assert callable(callback)

        # 低熵 token → 无触发
        result = callback([math.log(0.99), math.log(0.01)], "hello")
        assert result is None

    def test_create_entropy_callback_no_logprobs(self):
        """无 logprobs 时回调返回 None"""
        from dki.core.entropy_gated_injection import EGDMIController
        ctrl = EGDMIController()
        callback = ctrl.create_entropy_callback(
            original_query="test", session_id="s1",
        )
        assert callback(None, "token") is None


# ============================================================
# 测试: _get_fact_retrieve_method — entropy_gated 路由
# ============================================================

class TestGetFactRetrieveMethodEntropyGated:
    """测试 entropy_gated 在 _get_fact_retrieve_method 中的路由"""

    def test_entropy_gated_from_config(self):
        """直接配置 entropy_gated"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="entropy_gated")
        result = executor._get_fact_retrieve_method()
        assert result == "entropy_gated"

    def test_auto_open_source_routes_to_entropy_gated(self):
        """auto + 开源模型 → entropy_gated (v8.0 更新)"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="auto")
        executor.model = MockModelAdapter(is_closed_source=False)
        result = executor._get_fact_retrieve_method()
        assert result == "entropy_gated"

    def test_auto_closed_source_routes_to_native(self):
        """auto + 闭源模型 → native_tool_calls"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="auto")
        executor.model = MockModelAdapter(is_closed_source=True)
        result = executor._get_fact_retrieve_method()
        assert result == "native_tool_calls"


# ============================================================
# 测试: RecallConfig entropy_gated 新字段
# ============================================================

class TestRecallConfigEntropyFields:
    """测试 RecallConfig entropy_gated 相关字段"""

    def test_entropy_gated_method_value(self):
        """entropy_gated 作为有效方法"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig(fact_retrieve_method="entropy_gated")
        assert config.fact_retrieve_method == "entropy_gated"

    def test_entropy_fields_default(self):
        """entropy 相关字段默认值"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig()
        assert config.entropy_probe_tokens == 64
        assert config.entropy_logprobs_k == 5
        assert config.entropy_spike_threshold == 1.5
        assert config.entropy_absolute_threshold == 3.0
        assert config.entropy_window_size == 32
        assert config.entropy_max_retrievals == 2

    def test_from_dict_entropy_gated(self):
        """from_dict 解析 entropy_gated 字段"""
        from dki.core.recall.recall_config import RecallConfig
        config_dict = {
            "fact_call": {
                "enabled": True,
                "fact_retrieve_method": "entropy_gated",
                "entropy_probe_tokens": 128,
                "entropy_spike_threshold": 2.0,
                "entropy_max_retrievals": 3,
            }
        }
        config = RecallConfig.from_dict(config_dict)
        assert config.fact_call.fact_retrieve_method == "entropy_gated"
        assert config.fact_call.entropy_probe_tokens == 128
        assert config.fact_call.entropy_spike_threshold == 2.0
        assert config.fact_call.entropy_max_retrievals == 3


# ============================================================
# 测试: _execute_entropy_gated (异步)
# ============================================================

class TestExecuteEntropyGated:
    """测试 entropy_gated 执行流程"""

    @pytest.fixture
    def setup_executor(self):
        """创建配置好的 executor"""
        from dki.core.plugin.injection_executor import InjectionExecutor

        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(
            method="entropy_gated",
            entropy_probe_tokens=64,
            entropy_logprobs_k=5,
            entropy_spike_threshold=1.5,
            entropy_absolute_threshold=3.0,
        )
        executor._fact_retriever = None
        executor._prompt_formatter = None
        executor._fc_logger = None
        executor._stats = {}

        return executor

    @pytest.mark.asyncio
    async def test_no_entropy_spike_continuation(self, setup_executor):
        """无高熵 → probe + continuation"""
        executor = setup_executor
        call_count = [0]

        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Probe: 低熵 logprobs
                return MockModelOutput(
                    text="这是一个",
                    output_tokens=4,
                    logprobs=[
                        [math.log(0.95), math.log(0.05)]
                        for _ in range(4)
                    ],
                )
            else:
                # Continuation
                return MockModelOutput(
                    text="正常的回答。",
                    output_tokens=5,
                )

        executor._model_generate = mock_generate

        plan = MagicMock()
        plan.session_id = "session_1"
        plan.user_id = "user_1"
        plan.original_query = "你好"
        plan.has_fact_call_instruction = False

        result = await executor._execute_entropy_gated(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )

        assert "这是一个" in result.text
        assert "正常的回答" in result.text
        assert call_count[0] == 2  # probe + continuation
        assert executor._stats["entropy_gated_executions"] == 1
        assert executor._stats.get("entropy_gated_triggers", 0) == 0

    @pytest.mark.asyncio
    async def test_probe_generates_enough_tokens(self, setup_executor):
        """probe 已生成足够 tokens → 不需要 continuation"""
        executor = setup_executor

        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="完整的回答内容",
                output_tokens=max_new_tokens,  # 达到限制
                logprobs=[
                    [math.log(0.9), math.log(0.1)]
                    for _ in range(max_new_tokens)
                ],
            )

        executor._model_generate = mock_generate

        plan = MagicMock()
        plan.session_id = "s1"
        plan.user_id = "u1"
        plan.original_query = "test"

        result = await executor._execute_entropy_gated(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=10,
            temperature=0.7,
        )

        assert result.text == "完整的回答内容"

    @pytest.mark.asyncio
    async def test_no_logprobs_in_probe(self, setup_executor):
        """probe 没有返回 logprobs → 跳过熵分析, 正常 continuation"""
        executor = setup_executor
        call_count = [0]

        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="探测",
                    output_tokens=2,
                    logprobs=None,  # 无 logprobs
                )
            else:
                return MockModelOutput(
                    text="继续生成",
                    output_tokens=4,
                )

        executor._model_generate = mock_generate

        plan = MagicMock()
        plan.session_id = "s1"
        plan.user_id = "u1"
        plan.original_query = "test"

        result = await executor._execute_entropy_gated(
            plan=plan,
            prompt="prompt",
            max_new_tokens=100,
            temperature=0.7,
        )

        assert "探测" in result.text
        assert "继续生成" in result.text
        assert executor._stats.get("entropy_gated_triggers", 0) == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, setup_executor):
        """统计信息被正确更新"""
        executor = setup_executor

        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="回答",
                output_tokens=max_new_tokens,
                logprobs=[[math.log(0.9), math.log(0.1)]],
            )

        executor._model_generate = mock_generate

        plan = MagicMock()
        plan.session_id = "s1"
        plan.user_id = "u1"
        plan.original_query = "test"

        await executor._execute_entropy_gated(
            plan=plan, prompt="p", max_new_tokens=5, temperature=0.7,
        )

        assert executor._stats["entropy_gated_executions"] == 1

    @pytest.mark.asyncio
    async def test_f14_strip_residual_calls(self, setup_executor):
        """F1-4: 剥离残留的 retrieve_fact 调用"""
        executor = setup_executor

        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text='答案是42 retrieve_fact(trace_id="msg_001") 结束',
                output_tokens=max_new_tokens,
                logprobs=[[math.log(0.9), math.log(0.1)]],
            )

        executor._model_generate = mock_generate

        plan = MagicMock()
        plan.session_id = "s1"
        plan.user_id = "u1"
        plan.original_query = "test"

        result = await executor._execute_entropy_gated(
            plan=plan, prompt="p", max_new_tokens=5, temperature=0.7,
        )

        assert "retrieve_fact" not in result.text
        assert executor._stats.get("fact_calls_stripped", 0) > 0


# ============================================================
# 测试: EntropyTriggeredRetriever
# ============================================================

class TestEntropyTriggeredRetriever:
    """测试熵触发的记忆检索器"""

    def test_no_recall_system(self):
        """无 recall system 返回空 grounding"""
        from dki.core.entropy_gated_injection import (
            EntropyTriggeredRetriever, EGDMIConfig, GroundingContext,
        )
        retriever = EntropyTriggeredRetriever(EGDMIConfig(), memory_recall=None)
        result = retriever.retrieve_grounding(
            original_query="test",
            generated_so_far="some text",
            session_id="s1",
        )
        assert isinstance(result, GroundingContext)
        assert result.facts == []

    def test_retrieval_exception(self):
        """检索异常时返回空 grounding"""
        from dki.core.entropy_gated_injection import (
            EntropyTriggeredRetriever, EGDMIConfig,
        )
        mock_recall = MagicMock()
        mock_recall.recall.side_effect = Exception("DB error")
        retriever = EntropyTriggeredRetriever(EGDMIConfig(), memory_recall=mock_recall)

        result = retriever.retrieve_grounding(
            original_query="test",
            generated_so_far="text",
            session_id="s1",
        )
        assert result.facts == []
        assert "retrieval_error" in result.injection_reason

    def test_filter_already_used_ids(self):
        """过滤已使用的记忆 ID"""
        from dki.core.entropy_gated_injection import (
            EntropyTriggeredRetriever, EGDMIConfig,
        )

        # Mock recall result
        mock_item_1 = MagicMock()
        mock_item_1.message_id = "msg_001"
        mock_item_1.content = "Fact 1"
        mock_item_1.final_score = 0.9

        mock_item_2 = MagicMock()
        mock_item_2.message_id = "msg_002"
        mock_item_2.content = "Fact 2"
        mock_item_2.final_score = 0.8

        mock_recall_result = MagicMock()
        mock_recall_result.items = [mock_item_1, mock_item_2]

        mock_recall = MagicMock()
        mock_recall.recall.return_value = mock_recall_result

        retriever = EntropyTriggeredRetriever(EGDMIConfig(), memory_recall=mock_recall)

        result = retriever.retrieve_grounding(
            original_query="test",
            generated_so_far="text",
            session_id="s1",
            already_used_ids={"msg_001"},  # msg_001 已使用
        )

        assert "msg_002" in result.source_trace_ids
        assert "msg_001" not in result.source_trace_ids

    def test_filter_low_score(self):
        """过滤低分结果"""
        from dki.core.entropy_gated_injection import (
            EntropyTriggeredRetriever, EGDMIConfig,
        )

        mock_item = MagicMock()
        mock_item.message_id = "msg_low"
        mock_item.content = "Low score fact"
        mock_item.final_score = 0.1  # 低于 retrieval_min_score (0.3)

        mock_recall_result = MagicMock()
        mock_recall_result.items = [mock_item]

        mock_recall = MagicMock()
        mock_recall.recall.return_value = mock_recall_result

        retriever = EntropyTriggeredRetriever(EGDMIConfig(), memory_recall=mock_recall)

        result = retriever.retrieve_grounding(
            original_query="test",
            generated_so_far="text",
            session_id="s1",
        )

        assert result.facts == []


# ============================================================
# 测试: TwoStageGenerator
# ============================================================

class TestTwoStageGenerator:
    """测试两阶段生成器"""

    def test_init(self):
        """基本初始化"""
        from dki.core.entropy_gated_injection import (
            TwoStageGenerator, EGDMIController,
        )
        ctrl = EGDMIController()
        gen = TwoStageGenerator(controller=ctrl)
        assert gen.controller is ctrl
        assert gen.model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
