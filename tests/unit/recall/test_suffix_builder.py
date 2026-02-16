"""
SuffixBuilder 单元测试

测试后缀组装器:
- 逐消息阈值判断 (summary vs 原文)
- Context 预算管理
- 认知标记提取
- 抽取式摘要 (jieba TextRank)
- 粗估 token 计数
- 格式化完整后缀
"""

import re
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from dki.core.recall.recall_config import (
    RecallConfig,
    HistoryItem,
    AssembledSuffix,
)
from dki.core.recall.prompt_formatter import GenericFormatter
from dki.core.recall.suffix_builder import SuffixBuilder


# ============================================================
# Mock 数据
# ============================================================

@dataclass
class FakeMessage:
    """模拟消息对象"""
    id: str
    content: str
    role: str = "user"
    message_id: Optional[str] = None


# ============================================================
# 测试类
# ============================================================

class TestSuffixBuilder:
    """SuffixBuilder 测试"""

    @pytest.fixture
    def config(self):
        return RecallConfig.from_dict({
            "summary": {
                "per_message_threshold": 50,  # 低阈值便于测试
                "max_tokens_per_summary": 30,
                "strategy": "extractive",
            },
            "budget": {
                "generation_reserve": 100,
                "instruction_reserve": 50,
            },
        })

    @pytest.fixture
    def formatter(self):
        return GenericFormatter(language="cn")

    @pytest.fixture
    def builder(self, config, formatter):
        return SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )

    @pytest.fixture
    def short_messages(self):
        """短消息 (低于阈值)"""
        return [
            FakeMessage(id="msg-001", content="你好", role="user"),
            FakeMessage(id="msg-002", content="你好！有什么可以帮您？", role="assistant"),
            FakeMessage(id="msg-003", content="推荐一家餐厅", role="user"),
        ]

    @pytest.fixture
    def long_message(self):
        """长消息 (超过阈值)"""
        # 构造一个超过 50 token 的消息
        long_text = "这是一条非常长的消息。" * 20  # 约 200 字
        return FakeMessage(id="msg-long", content=long_text, role="user")

    # ============ 基础测试 ============

    def test_init_default(self, config, formatter):
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        assert builder.config is config
        assert builder.formatter is formatter

    def test_build_empty_messages(self, builder):
        """空消息列表应返回仅包含 query 的后缀"""
        result = builder.build(
            query="你好",
            recalled_messages=[],
            context_window=4096,
        )
        assert isinstance(result, AssembledSuffix)
        assert result.text == "你好"
        assert result.items == []
        assert result.total_tokens == 0

    # ============ 短消息测试 (全部保留原文) ============

    def test_build_short_messages(self, builder, short_messages):
        """短消息应全部保留为原文"""
        result = builder.build(
            query="你好",
            recalled_messages=short_messages,
            context_window=4096,
        )
        assert result.message_count == 3
        assert result.summary_count == 0
        assert result.has_fact_call_instruction is False
        assert "msg-001" in result.trace_ids

    # ============ 长消息测试 (生成 summary) ============

    def test_build_long_message_generates_summary(self, builder, long_message):
        """超阈值消息应生成 summary"""
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=4096,
        )
        assert result.summary_count >= 1
        assert result.has_fact_call_instruction is True
        assert "msg-long" in result.trace_ids

    def test_summary_item_has_epistemic_markers(self, builder, long_message):
        """summary 条目应包含认知标记"""
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=4096,
        )
        summary_items = [i for i in result.items if i.type == "summary"]
        if summary_items:
            item = summary_items[0]
            assert item.confidence == "medium"
            assert item.trace_id == "msg-long"

    # ============ 混合消息测试 ============

    def test_build_mixed_messages(self, builder, short_messages, long_message):
        """混合消息: 短消息保留原文, 长消息生成 summary"""
        messages = short_messages + [long_message]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=4096,
        )
        assert result.message_count >= 1
        assert result.summary_count >= 1

    # ============ 预算限制测试 ============

    def test_build_budget_exhausted(self, config, formatter):
        """预算耗尽时应停止添加消息"""
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        # 很小的 context_window
        messages = [
            FakeMessage(id=f"msg-{i}", content="这是一条测试消息" * 5, role="user")
            for i in range(100)
        ]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=300,  # 很小的窗口
        )
        # 不应包含所有 100 条消息
        assert len(result.items) < 100

    def test_build_zero_budget(self, config, formatter):
        """零预算应返回仅 query"""
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        messages = [FakeMessage(id="msg-1", content="test", role="user")]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=100,  # 小于 reserve 总和
            preference_tokens=50,
        )
        # 预算耗尽, 应返回 query
        assert "你好" in result.text

    # ============ 格式化测试 ============

    def test_suffix_contains_history_header(self, builder, short_messages):
        """后缀应包含 [会话历史参考] 头"""
        result = builder.build(
            query="你好",
            recalled_messages=short_messages,
            context_window=4096,
        )
        assert "[会话历史参考]" in result.text

    def test_suffix_contains_query(self, builder, short_messages):
        """后缀应包含用户查询"""
        result = builder.build(
            query="推荐一家餐厅",
            recalled_messages=short_messages,
            context_window=4096,
        )
        assert "推荐一家餐厅" in result.text

    def test_suffix_contains_constraint_when_summary(self, builder, long_message):
        """有 summary 时后缀应包含限定提示"""
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=4096,
        )
        if result.summary_count > 0:
            assert "可信" in result.text or "SUMMARY" in result.text

    # ============ Token 计数测试 ============

    def test_rough_token_count_chinese(self):
        """中文粗估 token 计数"""
        count = SuffixBuilder._rough_token_count("你好世界")
        assert count > 0
        # 4 个中文字 * 1.5 = 6
        assert count == 6

    def test_rough_token_count_english(self):
        """英文粗估 token 计数"""
        count = SuffixBuilder._rough_token_count("hello world")
        assert count > 0

    def test_rough_token_count_empty(self):
        assert SuffixBuilder._rough_token_count("") == 0

    def test_rough_token_count_mixed(self):
        """中英混合"""
        count = SuffixBuilder._rough_token_count("你好 hello 世界 world")
        assert count > 0

    # ============ 自定义 token_counter 测试 ============

    def test_custom_token_counter(self, config, formatter):
        """自定义 token 计数器"""
        counter = lambda text: len(text)  # 简单按字符计数
        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
            token_counter=counter,
        )
        messages = [FakeMessage(id="msg-1", content="abc", role="user")]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=4096,
        )
        assert result.total_tokens > 0

    # ============ 认知标记提取测试 ============

    def test_extract_epistemic_markers_date(self, builder):
        """应检测日期/时间"""
        original = "2025年3月15日下午3点在会议室开会"
        summary = "提到了开会"
        facts_covered, facts_missing = builder._extract_epistemic_markers(
            original, summary
        )
        assert "日期/时间" in facts_missing

    def test_extract_epistemic_markers_price(self, builder):
        """应检测价格/数字"""
        original = "这个产品售价299元"
        summary = "提到了一个产品的价格是299元"
        facts_covered, facts_missing = builder._extract_epistemic_markers(
            original, summary
        )
        assert "价格/数字" in facts_covered

    def test_extract_epistemic_markers_long_vs_short(self, builder):
        """原文远长于 summary 时应标注详细内容遗漏"""
        original = "x" * 300
        summary = "x" * 50
        facts_covered, facts_missing = builder._extract_epistemic_markers(
            original, summary
        )
        assert "详细内容" in facts_missing

    # ============ 截断测试 ============

    def test_truncate_to_tokens(self, builder):
        """截断应保持在 token 限制内"""
        long_text = "你好世界" * 100
        truncated = builder._truncate_to_tokens(long_text, 20)
        tokens = builder._count_tokens(truncated)
        assert tokens <= 20 + 5  # 容许小误差 (因为 "..." 后缀)

    def test_truncate_short_text(self, builder):
        """短文本无需截断"""
        text = "你好"
        result = builder._truncate_to_tokens(text, 100)
        assert result == text


class TestSuffixBuilderExtractSummarize:
    """SuffixBuilder 抽取式摘要测试"""

    @pytest.fixture
    def builder(self):
        config = RecallConfig.from_dict({
            "summary": {
                "per_message_threshold": 50,
                "max_tokens_per_summary": 100,
                "strategy": "extractive",
            },
        })
        formatter = GenericFormatter(language="cn")
        return SuffixBuilder(config=config, prompt_formatter=formatter)

    @patch("dki.core.recall.suffix_builder.JIEBA_AVAILABLE", False)
    def test_summarize_without_jieba(self, builder):
        """无 jieba 时应使用截断"""
        text = "这是一段很长的文本。" * 30
        summary = builder._summarize(text)
        assert len(summary) > 0
        assert len(summary) <= len(text)

    def test_summarize_returns_nonempty(self, builder):
        """摘要不应为空"""
        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。"
        summary = builder._summarize(text)
        assert len(summary) > 0
