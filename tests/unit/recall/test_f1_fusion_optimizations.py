"""
F1 融合权重 & 缓存优化 — 单元测试

测试范围:
- F1-1: Epistemic Mode Selection (认知态模式选择)
- F1-2: Signal Confidence Gating (信号置信度门控)
- F1-3: Score Normalization 统一 (min-max 归一化)
- F1-4: Executor 防御性拦截 (retrieve_fact 工具过滤)

Author: AGI Demo Project
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from dki.core.recall.recall_config import (
    RecallConfig,
    RecallResult,
    RecallScoreWeights,
    EpistemicModeConfig,
    SignalGatingConfig,
)
from dki.core.recall.multi_signal_recall import (
    MultiSignalRecall,
    SignalConfidence,
    select_epistemic_mode,
    get_weights_for_mode,
)


# ============================================================
# Mock 数据
# ============================================================

@dataclass
class FakeMessage:
    id: str
    content: str
    role: str = "user"


@dataclass
class FakeSearchResult:
    memory_id: str
    score: float
    content: str = ""


class FakeConversationRepo:
    def __init__(self, messages: List[FakeMessage] = None):
        self._messages = messages or []

    def get_by_session(self, session_id: str, **kwargs) -> List[FakeMessage]:
        return self._messages

    def get_recent(self, session_id: str, limit: int = 10, **kwargs) -> List[FakeMessage]:
        return self._messages[-limit:]


class FakeMemoryRouter:
    def __init__(self, results: List[FakeSearchResult] = None):
        self._results = results or []

    def search(self, query: str, top_k: int = 10) -> List[FakeSearchResult]:
        return self._results[:top_k]


@dataclass
class FakeQueryContext:
    """模拟 QueryContext (属性名与 select_epistemic_mode 中 getattr 一致)"""
    # select_epistemic_mode 检查的属性:
    reference_resolved: bool = False      # 规则 1: 指代解析成功
    memory_triggered: bool = False        # 规则 2: 记忆触发
    trigger_confidence: float = 0.0       # 规则 2: 触发置信度
    trigger_type: Optional[str] = None    # 规则 3: 触发类型 (含 "correct" → correction)
    # 额外属性 (兼容)
    memory_trigger_confidence: float = 0.0
    has_reference: bool = False
    reference_type: Optional[str] = None
    reference_scope: Optional[str] = None
    recall_token_budget: int = 2048


# ============================================================
# F1-1: Epistemic Mode Selection 测试
# ============================================================

class TestEpistemicModeSelection:
    """F1-1: 认知态模式选择"""

    def test_disabled_returns_fallback(self):
        """禁用时返回 fallback_mode"""
        config = EpistemicModeConfig(enabled=False, fallback_mode="direct_lookup")
        mode = select_epistemic_mode(context=None, config=config)
        assert mode == "direct_lookup"

    def test_no_context_returns_fallback(self):
        """无 context 返回 fallback"""
        config = EpistemicModeConfig(enabled=True, fallback_mode="direct_lookup")
        mode = select_epistemic_mode(context=None, config=config)
        assert mode == "direct_lookup"

    def test_clarification_mode(self):
        """reference_resolved=True → clarification"""
        config = EpistemicModeConfig(enabled=True)
        ctx = FakeQueryContext(reference_resolved=True)
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "clarification"

    def test_semantic_search_mode(self):
        """memory_triggered + high confidence → semantic_search"""
        config = EpistemicModeConfig(enabled=True)
        ctx = FakeQueryContext(memory_triggered=True, trigger_confidence=0.8)
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "semantic_search"

    def test_correction_mode(self):
        """trigger_type 含 'correct' → correction"""
        config = EpistemicModeConfig(enabled=True)
        ctx = FakeQueryContext(trigger_type="correction")
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "correction"

    def test_high_confidence_semantic_search(self):
        """高 trigger_confidence + memory_triggered → semantic_search"""
        config = EpistemicModeConfig(enabled=True)
        ctx = FakeQueryContext(memory_triggered=True, trigger_confidence=0.9)
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "semantic_search"

    def test_priority_clarification_over_semantic_search(self):
        """clarification (规则1) 优先于 semantic_search (规则2)"""
        config = EpistemicModeConfig(enabled=True)
        ctx = FakeQueryContext(
            reference_resolved=True,
            memory_triggered=True,
            trigger_confidence=0.9,
        )
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "clarification"

    def test_fallback_mode_when_no_signal(self):
        """所有信号为 False 时返回 fallback"""
        config = EpistemicModeConfig(enabled=True, fallback_mode="direct_lookup")
        ctx = FakeQueryContext()
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "direct_lookup"
    
    def test_low_confidence_not_semantic_search(self):
        """memory_triggered=True 但 confidence 低 → fallback"""
        config = EpistemicModeConfig(enabled=True, fallback_mode="direct_lookup")
        ctx = FakeQueryContext(memory_triggered=True, trigger_confidence=0.3)
        mode = select_epistemic_mode(context=ctx, config=config)
        assert mode == "direct_lookup"


class TestGetWeightsForMode:
    """F1-1: 获取模式权重"""

    def test_known_mode_returns_profile(self):
        """已知模式返回对应权重"""
        config = EpistemicModeConfig(enabled=True)
        default = RecallScoreWeights()
        
        weights = get_weights_for_mode("clarification", config, default)
        assert weights.keyword_weight == 0.50
        assert weights.vector_weight == 0.10
        assert weights.recency_weight == 0.40

    def test_unknown_mode_returns_default(self):
        """未知模式返回默认权重"""
        config = EpistemicModeConfig(enabled=True)
        default = RecallScoreWeights(keyword_weight=0.4, vector_weight=0.35, recency_weight=0.25)
        
        weights = get_weights_for_mode("nonexistent_mode", config, default)
        assert weights.keyword_weight == default.keyword_weight
        assert weights.vector_weight == default.vector_weight
        assert weights.recency_weight == default.recency_weight

    def test_continuation_mode_weights(self):
        """continuation 模式: 高 recency, 低 keyword"""
        config = EpistemicModeConfig(enabled=True)
        default = RecallScoreWeights()
        
        weights = get_weights_for_mode("continuation", config, default)
        assert weights.recency_weight > weights.keyword_weight
        assert weights.recency_weight == 0.50

    def test_semantic_search_mode_weights(self):
        """semantic_search 模式: 高 vector"""
        config = EpistemicModeConfig(enabled=True)
        default = RecallScoreWeights()
        
        weights = get_weights_for_mode("semantic_search", config, default)
        assert weights.vector_weight > weights.keyword_weight
        assert weights.vector_weight > weights.recency_weight

    def test_custom_profiles(self):
        """自定义 profiles"""
        config = EpistemicModeConfig(
            enabled=True,
            profiles={
                "custom_mode": {
                    "keyword_weight": 0.1,
                    "vector_weight": 0.8,
                    "recency_weight": 0.1,
                },
            },
        )
        default = RecallScoreWeights()
        
        weights = get_weights_for_mode("custom_mode", config, default)
        assert weights.vector_weight == 0.8


# ============================================================
# F1-2: Signal Confidence Gating 测试
# ============================================================

class TestSignalConfidenceGating:
    """F1-2: 信号置信度门控"""

    def test_keyword_confidence_no_hits(self):
        """无命中 → 置信度 0"""
        conf = MultiSignalRecall._compute_keyword_confidence({}, 5, 0)
        assert conf == 0.0

    def test_keyword_confidence_full_coverage(self):
        """全覆盖 → 高置信度"""
        scored = {"a": 1.0, "b": 0.8, "c": 0.5}
        conf = MultiSignalRecall._compute_keyword_confidence(scored, 3, 3)
        # coverage=1.0, density=min(1.0, 3/3)=1.0 → conf=1.0
        assert conf == 1.0

    def test_keyword_confidence_partial_coverage(self):
        """部分覆盖 → 中等置信度"""
        scored = {"a": 1.0}
        conf = MultiSignalRecall._compute_keyword_confidence(scored, 5, 2)
        # coverage=2/5=0.4, density=min(1.0, 1/3)≈0.333 → conf≈0.133
        assert 0.0 < conf < 0.5

    def test_keyword_confidence_zero_query_terms(self):
        """query_terms=0 → 置信度 0"""
        conf = MultiSignalRecall._compute_keyword_confidence({"a": 1.0}, 0, 0)
        assert conf == 0.0

    def test_vector_confidence_no_results(self):
        """无结果 → 置信度 0"""
        conf = MultiSignalRecall._compute_vector_confidence({})
        assert conf == 0.0

    def test_vector_confidence_high_scores(self):
        """高分 → 高置信度"""
        scored = {"a": 0.95, "b": 0.90, "c": 0.85, "d": 0.60}
        conf = MultiSignalRecall._compute_vector_confidence(scored)
        # top-3: 0.95, 0.90, 0.85 → mean=0.9
        assert abs(conf - 0.9) < 1e-6

    def test_vector_confidence_single_result(self):
        """单结果 → 置信度 = 该分数"""
        conf = MultiSignalRecall._compute_vector_confidence({"a": 0.7})
        assert abs(conf - 0.7) < 1e-6

    def test_vector_confidence_low_scores(self):
        """低分 → 低置信度"""
        scored = {"a": 0.2, "b": 0.1}
        conf = MultiSignalRecall._compute_vector_confidence(scored)
        assert conf < 0.2

    def test_signal_gating_config_defaults(self):
        """默认配置: 启用, 阈值 0.15"""
        config = SignalGatingConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.15


class TestSignalGatingInRecall:
    """F1-2: 信号门控在 recall() 中的集成"""

    @patch("dki.core.recall.multi_signal_recall.JIEBA_AVAILABLE", True)
    @patch("dki.core.recall.multi_signal_recall.jieba", create=True)
    def test_low_keyword_confidence_gated_out(self, mock_jieba):
        """关键词置信度低于阈值时被门控排除"""
        mock_jieba.analyse.extract_tags.return_value = [
            ("不存在的词", 0.3),
            ("另一个不存在的词", 0.2),
        ]

        messages = [
            FakeMessage(id="msg-001", content="完全不相关的内容"),
        ]
        vector_results = [
            FakeSearchResult(memory_id="msg-001", score=0.9),
        ]

        config = RecallConfig.from_dict({
            "signal_gating": {"enabled": True, "confidence_threshold": 0.5},
            "budget": {"min_recent_turns": 0},
        })

        recall = MultiSignalRecall(
            config=config,
            memory_router=FakeMemoryRouter(vector_results),
            conversation_repo=FakeConversationRepo(messages),
        )
        result = recall.recall(query="不存在的词", session_id="s1")

        # 关键词全部不命中 → 置信度=0 → 被门控
        # 向量有命中 → 应有结果
        assert result.vector_hits > 0

    def test_gating_disabled_all_signals_active(self):
        """门控禁用时所有信号都参与"""
        config = RecallConfig.from_dict({
            "signal_gating": {"enabled": False},
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 0},
        })
        recall = MultiSignalRecall(config=config)
        result = recall.recall(query="test", session_id="s1")
        # 无信号源但不应报错
        assert isinstance(result, RecallResult)


# ============================================================
# F1-3: Score Normalization 测试
# ============================================================

class TestMinMaxNormalization:
    """F1-3: 统一 min-max 归一化"""

    def test_empty_input(self):
        assert MultiSignalRecall._min_max_normalize({}) == {}

    def test_single_value(self):
        """单值 → 1.0"""
        result = MultiSignalRecall._min_max_normalize({"a": 5.0})
        assert result["a"] == 1.0

    def test_identical_values(self):
        """所有值相同 → 全部 1.0"""
        result = MultiSignalRecall._min_max_normalize({"a": 3.0, "b": 3.0, "c": 3.0})
        assert all(v == 1.0 for v in result.values())

    def test_min_maps_to_zero(self):
        """最小值 → 0.0"""
        result = MultiSignalRecall._min_max_normalize({"a": 1.0, "b": 5.0})
        assert result["a"] == 0.0

    def test_max_maps_to_one(self):
        """最大值 → 1.0"""
        result = MultiSignalRecall._min_max_normalize({"a": 1.0, "b": 5.0})
        assert result["b"] == 1.0

    def test_intermediate_values(self):
        """中间值正确归一化"""
        result = MultiSignalRecall._min_max_normalize(
            {"a": 0.0, "b": 5.0, "c": 10.0}
        )
        assert result["a"] == 0.0
        assert abs(result["b"] - 0.5) < 1e-6
        assert result["c"] == 1.0

    def test_all_values_in_range(self):
        """所有值 ∈ [0, 1]"""
        raw = {"a": -10, "b": 0, "c": 100, "d": 50}
        result = MultiSignalRecall._min_max_normalize(raw)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_preserves_ordering(self):
        """归一化保持原始排序"""
        raw = {"a": 1.0, "b": 3.0, "c": 2.0}
        result = MultiSignalRecall._min_max_normalize(raw)
        assert result["a"] < result["c"] < result["b"]

    def test_negative_values(self):
        """负值正确处理"""
        result = MultiSignalRecall._min_max_normalize({"a": -5.0, "b": 5.0})
        assert result["a"] == 0.0
        assert result["b"] == 1.0


# ============================================================
# F1-4: Executor 防御性拦截 测试
# ============================================================

class TestRetrieveFactStripping:
    """F1-4: retrieve_fact 工具调用过滤"""

    def test_import(self):
        """确保 _strip_retrieve_fact_calls 可导入"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        assert callable(_strip_retrieve_fact_calls)

    def test_no_fact_calls(self):
        """无 retrieve_fact 调用 → 不变"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = "这是一个正常的回答，没有任何工具调用。"
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert cleaned == text
        assert count == 0

    def test_generic_fact_call_stripped(self):
        """Generic 格式 retrieve_fact 被剥离"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = '好的，让我查一下。retrieve_fact(trace_id="msg-005", offset=0, limit=5) 这是结果。'
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "retrieve_fact" not in cleaned
        assert count == 1

    def test_deepseek_fact_call_stripped(self):
        """DeepSeek 格式 retrieve_fact 被剥离"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = (
            '好的。\n'
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>retrieve_fact\n'
            '{"trace_id": "msg-005", "offset": 0, "limit": 5}\n'
            '<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n'
            '这是结果。'
        )
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "<｜tool▁calls▁begin｜>" not in cleaned
        assert count >= 1

    def test_glm_fact_call_stripped(self):
        """GLM 格式 retrieve_fact 被剥离"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = (
            '好的。\n'
            '<|tool_call|>retrieve_fact\n'
            '{"trace_id": "msg-005", "offset": 0, "limit": 5}\n'
            '这是结果。'
        )
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "<|tool_call|>" not in cleaned
        assert count >= 1

    def test_multiple_fact_calls_stripped(self):
        """多个 retrieve_fact 调用全部剥离"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = (
            'retrieve_fact(trace_id="msg-001") '
            '中间文本 '
            'retrieve_fact(trace_id="msg-002", offset=5, limit=10)'
        )
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "retrieve_fact" not in cleaned
        assert count == 2

    def test_excessive_newlines_cleaned(self):
        """剥离后多余空行被清理"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = '第一段\n\n\nretrieve_fact(trace_id="msg-001")\n\n\n第二段'
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "\n\n\n" not in cleaned
        assert count == 1

    def test_case_insensitive_matching(self):
        """大小写不敏感"""
        from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
        text = 'RETRIEVE_FACT(trace_id="msg-001")'
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert "RETRIEVE_FACT" not in cleaned
        assert count == 1


# ============================================================
# 集成: recall() 中 F1-1 + F1-2 + F1-3 协同测试
# ============================================================

class TestRecallWithF1Optimizations:
    """F1 优化在 recall() 中的端到端集成"""

    @pytest.fixture
    def messages(self):
        return [
            FakeMessage(id="msg-001", content="我喜欢吃四川火锅", role="user"),
            FakeMessage(id="msg-002", content="好的记住了", role="assistant"),
            FakeMessage(id="msg-003", content="明天有会议", role="user"),
            FakeMessage(id="msg-004", content="已记录", role="assistant"),
        ]

    def test_recall_with_query_context(self, messages):
        """传入 query_context 不报错"""
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 1},
            "epistemic_modes": {"enabled": True},
        })
        repo = FakeConversationRepo(messages)
        recall = MultiSignalRecall(config=config, conversation_repo=repo)

        ctx = FakeQueryContext(memory_triggered=True, trigger_confidence=0.5)
        result = recall.recall(
            query="继续", session_id="s1", query_context=ctx,
        )
        assert isinstance(result, RecallResult)

    def test_recall_without_query_context(self, messages):
        """不传 query_context 也不报错 (向后兼容)"""
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 1},
        })
        repo = FakeConversationRepo(messages)
        recall = MultiSignalRecall(config=config, conversation_repo=repo)

        result = recall.recall(query="你好", session_id="s1")
        assert isinstance(result, RecallResult)

    def test_stats_include_epistemic_mode_counts(self, messages):
        """统计包含认知态模式计数"""
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 0},
            "epistemic_modes": {"enabled": True},
        })
        recall = MultiSignalRecall(
            config=config,
            conversation_repo=FakeConversationRepo(messages),
        )

        ctx = FakeQueryContext(reference_resolved=True)
        recall.recall(query="之前说的", session_id="s1", query_context=ctx)
        recall.recall(query="之前说的2", session_id="s1", query_context=ctx)

        stats = recall.get_stats()
        assert "epistemic_mode_counts" in stats
        assert stats["epistemic_mode_counts"].get("clarification", 0) == 2

    def test_stats_include_signal_gating_dropped(self, messages):
        """统计包含信号门控丢弃数"""
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 0},
            "signal_gating": {"enabled": True, "confidence_threshold": 0.5},
        })
        recall = MultiSignalRecall(
            config=config,
            conversation_repo=FakeConversationRepo(messages),
        )

        recall.recall(query="test", session_id="s1")
        stats = recall.get_stats()
        assert "signal_gating_dropped_total" in stats


class TestSignalConfidenceDataclass:
    """SignalConfidence 数据类测试"""

    def test_default_values(self):
        sc = SignalConfidence()
        assert sc.score == 0.0
        assert sc.confidence == 0.0
        assert sc.coverage == 0.0
        assert sc.active is True

    def test_custom_values(self):
        sc = SignalConfidence(score=0.8, confidence=0.9, coverage=0.7, active=False)
        assert sc.score == 0.8
        assert sc.confidence == 0.9
        assert sc.coverage == 0.7
        assert sc.active is False
