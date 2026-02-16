"""
RecallConfig 单元测试

测试 Recall v4 配置数据结构的创建、序列化和边界条件
"""

import pytest
from dki.core.recall.recall_config import (
    RecallConfig,
    RecallSignalsConfig,
    RecallBudgetConfig,
    RecallSummaryConfig,
    RecallFactCallConfig,
    RecallScoreWeights,
    HistoryItem,
    RecallResult,
    AssembledSuffix,
    FactRequest,
    FactResponse,
)


class TestRecallSignalsConfig:
    """RecallSignalsConfig 测试"""

    def test_default_values(self):
        cfg = RecallSignalsConfig()
        assert cfg.keyword_enabled is True
        assert cfg.keyword_topk == 5
        assert cfg.keyword_method == "tfidf"
        assert cfg.vector_enabled is True
        assert cfg.vector_top_k == 10
        assert cfg.vector_threshold == 0.5
        assert cfg.reference_enabled is True

    def test_custom_values(self):
        cfg = RecallSignalsConfig(
            keyword_enabled=False,
            keyword_topk=10,
            keyword_method="textrank",
            vector_top_k=20,
            vector_threshold=0.8,
        )
        assert cfg.keyword_enabled is False
        assert cfg.keyword_topk == 10
        assert cfg.keyword_method == "textrank"
        assert cfg.vector_top_k == 20
        assert cfg.vector_threshold == 0.8


class TestRecallBudgetConfig:
    """RecallBudgetConfig 测试"""

    def test_default_values(self):
        cfg = RecallBudgetConfig()
        assert cfg.generation_reserve == 512
        assert cfg.instruction_reserve == 150
        assert cfg.min_recent_turns == 2
        assert cfg.max_recent_turns == 5

    def test_zero_reserve(self):
        """零预留值应正常工作"""
        cfg = RecallBudgetConfig(generation_reserve=0, instruction_reserve=0)
        assert cfg.generation_reserve == 0
        assert cfg.instruction_reserve == 0


class TestRecallSummaryConfig:
    """RecallSummaryConfig 测试"""

    def test_default_values(self):
        cfg = RecallSummaryConfig()
        assert cfg.per_message_threshold == 200
        assert cfg.max_tokens_per_summary == 150
        assert cfg.strategy == "extractive"

    def test_llm_strategy(self):
        cfg = RecallSummaryConfig(strategy="llm")
        assert cfg.strategy == "llm"


class TestRecallFactCallConfig:
    """RecallFactCallConfig 测试"""

    def test_default_values(self):
        cfg = RecallFactCallConfig()
        assert cfg.enabled is True
        assert cfg.max_rounds == 3
        assert cfg.max_fact_tokens == 800
        assert cfg.batch_size == 5

    def test_disabled(self):
        cfg = RecallFactCallConfig(enabled=False)
        assert cfg.enabled is False


class TestRecallScoreWeights:
    """RecallScoreWeights 测试"""

    def test_default_weights_sum_to_one(self):
        w = RecallScoreWeights()
        total = w.keyword_weight + w.vector_weight + w.recency_weight
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"

    def test_keyword_ge_vector(self):
        """补充建议: w1 >= w2 (事实准确优先)"""
        w = RecallScoreWeights()
        assert w.keyword_weight >= w.vector_weight


class TestRecallConfig:
    """RecallConfig 测试"""

    def test_default_config(self):
        cfg = RecallConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "summary_with_fact_call"
        assert isinstance(cfg.signals, RecallSignalsConfig)
        assert isinstance(cfg.budget, RecallBudgetConfig)
        assert isinstance(cfg.summary, RecallSummaryConfig)
        assert isinstance(cfg.fact_call, RecallFactCallConfig)
        assert isinstance(cfg.score_weights, RecallScoreWeights)
        assert cfg.prompt_formatter == "auto"

    def test_from_dict_empty(self):
        """空字典应返回默认配置"""
        cfg = RecallConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.strategy == "summary_with_fact_call"

    def test_from_dict_none(self):
        """None 应返回默认配置"""
        cfg = RecallConfig.from_dict(None)
        assert cfg.enabled is True

    def test_from_dict_full(self):
        """完整字典应正确解析"""
        d = {
            "enabled": True,
            "strategy": "summary_with_fact_call",
            "signals": {
                "keyword_enabled": True,
                "keyword_topk": 8,
                "keyword_method": "textrank",
                "vector_enabled": False,
                "vector_top_k": 15,
                "vector_threshold": 0.6,
            },
            "budget": {
                "generation_reserve": 1024,
                "instruction_reserve": 200,
                "min_recent_turns": 3,
                "max_recent_turns": 8,
            },
            "summary": {
                "per_message_threshold": 300,
                "max_tokens_per_summary": 200,
                "strategy": "llm",
            },
            "fact_call": {
                "enabled": True,
                "max_rounds": 5,
                "max_fact_tokens": 1000,
                "batch_size": 10,
            },
            "score_weights": {
                "keyword_weight": 0.5,
                "vector_weight": 0.3,
                "recency_weight": 0.2,
            },
            "prompt_formatter": "deepseek",
        }
        cfg = RecallConfig.from_dict(d)

        assert cfg.signals.keyword_topk == 8
        assert cfg.signals.keyword_method == "textrank"
        assert cfg.signals.vector_enabled is False
        assert cfg.signals.vector_top_k == 15
        assert cfg.budget.generation_reserve == 1024
        assert cfg.budget.min_recent_turns == 3
        assert cfg.summary.per_message_threshold == 300
        assert cfg.summary.strategy == "llm"
        assert cfg.fact_call.max_rounds == 5
        assert cfg.fact_call.batch_size == 10
        assert cfg.score_weights.keyword_weight == 0.5
        assert cfg.prompt_formatter == "deepseek"

    def test_from_dict_partial(self):
        """部分字典应正确填充默认值"""
        d = {
            "enabled": False,
            "signals": {"keyword_topk": 3},
        }
        cfg = RecallConfig.from_dict(d)
        assert cfg.enabled is False
        assert cfg.signals.keyword_topk == 3
        assert cfg.signals.keyword_method == "tfidf"  # 默认值
        assert cfg.budget.generation_reserve == 512  # 默认值
        assert cfg.fact_call.max_rounds == 3  # 默认值

    def test_from_dict_flat_history_strategy(self):
        """flat_history 策略"""
        d = {"strategy": "flat_history"}
        cfg = RecallConfig.from_dict(d)
        assert cfg.strategy == "flat_history"


class TestHistoryItem:
    """HistoryItem 测试"""

    def test_summary_item(self):
        item = HistoryItem(
            type="summary",
            content="用户提到了一家餐厅",
            trace_id="msg-001",
            role="user",
            token_count=15,
            confidence="medium",
            facts_covered=["餐厅名称"],
            facts_missing=["价格", "营业时间"],
        )
        assert item.type == "summary"
        assert item.trace_id == "msg-001"
        assert item.confidence == "medium"
        assert "餐厅名称" in item.facts_covered
        assert "价格" in item.facts_missing

    def test_message_item(self):
        item = HistoryItem(
            type="message",
            content="你好，请推荐一家餐厅",
            trace_id="msg-002",
            role="user",
            token_count=10,
        )
        assert item.type == "message"
        assert item.confidence == "high"  # 默认值
        assert item.facts_covered == []
        assert item.facts_missing == []

    def test_default_fields(self):
        item = HistoryItem(type="message", content="test", trace_id="t1")
        assert item.role is None
        assert item.token_count == 0
        assert item.confidence == "high"


class TestRecallResult:
    """RecallResult 测试"""

    def test_empty_result(self):
        result = RecallResult()
        assert result.messages == []
        assert result.keyword_hits == 0
        assert result.vector_hits == 0
        assert result.reference_scope is None
        assert result.recent_turns_added == 0
        assert result.scores == {}


class TestAssembledSuffix:
    """AssembledSuffix 测试"""

    def test_empty_suffix(self):
        suffix = AssembledSuffix()
        assert suffix.text == ""
        assert suffix.items == []
        assert suffix.total_tokens == 0
        assert suffix.message_count == 0
        assert suffix.summary_count == 0
        assert suffix.has_fact_call_instruction is False
        assert suffix.trace_ids == []


class TestFactRequest:
    """FactRequest 测试"""

    def test_basic_request(self):
        req = FactRequest(trace_id="msg-005")
        assert req.trace_id == "msg-005"
        assert req.offset == 0
        assert req.limit == 5

    def test_paginated_request(self):
        req = FactRequest(trace_id="msg-010", offset=5, limit=10)
        assert req.offset == 5
        assert req.limit == 10


class TestFactResponse:
    """FactResponse 测试"""

    def test_empty_response(self):
        resp = FactResponse()
        assert resp.messages == []
        assert resp.trace_id == ""
        assert resp.total_count == 0
        assert resp.has_more is False

    def test_response_with_messages(self):
        resp = FactResponse(
            messages=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
            ],
            trace_id="msg-001",
            total_count=5,
            offset=0,
            has_more=True,
        )
        assert len(resp.messages) == 2
        assert resp.has_more is True
        assert resp.total_count == 5
