"""
SuffixBuilder 单元测试

测试后缀组装器 (v6.0: 两阶段全局预算分配):
- Phase 1: 完整收集 (不压缩)
- Phase 2: 全局预算分配 — 短消息优先保留, 长消息只在预算不足时压缩
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
        """短消息应全部保留为原文, trace_ids 不包含原文消息"""
        result = builder.build(
            query="你好",
            recalled_messages=short_messages,
            context_window=4096,
        )
        assert result.message_count == 3
        assert result.summary_count == 0
        assert result.has_fact_call_instruction is False
        # v6.1: trace_ids 只收集 summary 类型, 原文消息不需要 retrieve_fact
        assert result.trace_ids == []

    # ============ 长消息测试 (v6.0: 全局预算分配) ============

    def test_build_long_message_fits_budget_keeps_full(self, builder, long_message):
        """v6.0: 长消息在预算充足时应保留全文, 不压缩"""
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=4096,
        )
        # 预算充足: 4096 - 1228(gen) - 50(tag) - 9(query) = 2809 >> 310
        # 应保留原文, 不压缩
        assert result.message_count == 1
        assert result.summary_count == 0
        assert result.has_fact_call_instruction is False
        # v6.1: 原文消息不出现在 trace_ids 中
        assert result.trace_ids == []

    def test_build_long_message_generates_summary(self, builder, long_message):
        """v6.0: 长消息在预算不足时应生成 summary"""
        # 使用很小的 context_window, 迫使长消息被压缩
        # long_message 约 310 tokens, 设 context_window=600
        # budget = 600 - 180(gen) - 50(tag) - 9(query) = 361
        # 310 <= 361 → 仍然放得下
        # 需要更小: context_window=400
        # budget = 400 - 120(gen) - 50(tag) - 9(query) = 221 < 310 → 需要压缩
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=400,
        )
        assert result.summary_count >= 1
        assert result.has_fact_call_instruction is True
        # v6.1: trace_ids 只包含 summary 类型的 trace_id
        assert "msg-long" in result.trace_ids

    def test_summary_item_has_epistemic_markers(self, builder, long_message):
        """summary 条目应包含认知标记"""
        result = builder.build(
            query="你好",
            recalled_messages=[long_message],
            context_window=400,  # 小窗口迫使压缩
        )
        summary_items = [i for i in result.items if i.type == "summary"]
        if summary_items:
            item = summary_items[0]
            assert item.confidence == "medium"
            assert item.trace_id == "msg-long"

    # ============ 混合消息测试 ============

    def test_build_mixed_messages_budget_sufficient(self, builder, short_messages, long_message):
        """v6.0: 预算充足时, 混合消息全部保留原文"""
        messages = short_messages + [long_message]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=4096,
        )
        # 所有消息都应保留为原文
        assert result.message_count == 4
        assert result.summary_count == 0

    def test_build_mixed_messages_budget_tight(self, builder, short_messages, long_message):
        """v6.0: 预算紧张时, 短消息保留原文, 长消息压缩"""
        messages = short_messages + [long_message]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=400,  # 小窗口迫使长消息压缩
        )
        assert result.message_count >= 1  # 至少短消息保留
        assert result.summary_count >= 1  # 长消息被压缩

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
            context_window=400,  # 小窗口迫使压缩
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


# ============================================================
# v6.1: trace_ids 只收集 summary + LLM summary 验证
# ============================================================

class TestTraceIdsOnlySummary:
    """v6.1: 验证 trace_ids 只收集 summary 类型条目的 trace_id"""

    @pytest.fixture
    def config(self):
        return RecallConfig.from_dict({
            "summary": {
                "per_message_threshold": 50,
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
        return SuffixBuilder(config=config, prompt_formatter=formatter)

    def test_all_messages_no_trace_ids(self, builder):
        """全部为原文消息时, trace_ids 应为空列表"""
        messages = [
            FakeMessage(id=f"msg-{i}", content=f"短消息{i}", role="user")
            for i in range(5)
        ]
        result = builder.build(
            query="你好",
            recalled_messages=messages,
            context_window=4096,
        )
        assert result.summary_count == 0
        assert result.trace_ids == []
        assert result.has_fact_call_instruction is False

    def test_mixed_only_summary_trace_ids(self, builder):
        """混合消息时, trace_ids 只包含 summary 的 trace_id"""
        short_msg = FakeMessage(id="msg-short", content="短消息", role="user")
        long_msg = FakeMessage(
            id="msg-long-001",
            content="这是一条非常长的消息。" * 20,
            role="assistant",
        )
        result = builder.build(
            query="你好",
            recalled_messages=[short_msg, long_msg],
            context_window=400,  # 小窗口迫使长消息压缩
        )
        # 短消息保留原文, 长消息被压缩为 summary
        if result.summary_count > 0:
            # trace_ids 只包含 summary 的 ID, 不包含原文的 ID
            assert "msg-short" not in result.trace_ids
            assert "msg-long-001" in result.trace_ids

    def test_multiple_summaries_multiple_trace_ids(self, builder):
        """多条 summary 时, 每条 summary 对应一个 trace_id"""
        long_msgs = [
            FakeMessage(
                id=f"msg-long-{i}",
                content=f"这是第{i}条非常长的消息。" * 20,
                role="user" if i % 2 == 0 else "assistant",
            )
            for i in range(3)
        ]
        result = builder.build(
            query="你好",
            recalled_messages=long_msgs,
            context_window=400,  # 小窗口
        )
        # 所有 trace_ids 都应来自 summary 类型的 items
        summary_ids = {i.trace_id for i in result.items if i.type == "summary"}
        assert set(result.trace_ids) == summary_ids

    def test_constraint_instruction_only_has_summary_trace_ids(self, builder):
        """[可信+推理限定] 块只包含 summary 的 trace_id, 不包含原文消息的"""
        short_msg = FakeMessage(id="msg-short", content="你好", role="user")
        long_msg = FakeMessage(
            id="msg-long-for-constraint",
            content="这是一条非常长的消息。" * 20,
            role="assistant",
        )
        result = builder.build(
            query="你好",
            recalled_messages=[short_msg, long_msg],
            context_window=400,
        )
        if result.summary_count > 0:
            # 后缀文本中的 [可信+推理限定] 块不应包含 msg-short
            assert "msg-short" not in result.text or "msg-short" in result.text.split("[可信+推理限定]")[0]
            # 但应包含 summary 的 trace_id
            assert "msg-long-for-constraint" in result.text


class TestLLMSummarizeValidation:
    """v6.1: 验证 _llm_summarize 输出验证逻辑"""

    def test_invalid_summary_empty(self):
        """空输出应被判定为无效"""
        assert SuffixBuilder._is_summary_invalid("", "原文") is True

    def test_invalid_summary_too_short(self):
        """太短的输出应被判定为无效"""
        assert SuffixBuilder._is_summary_invalid("abc", "原文") is True

    def test_invalid_summary_only_role_label(self):
        """只包含角色标签的输出应被判定为无效"""
        assert SuffixBuilder._is_summary_invalid("assistant", "原文") is True
        assert SuffixBuilder._is_summary_invalid("assistant.", "原文") is True
        assert SuffixBuilder._is_summary_invalid("assistant。", "原文") is True
        assert SuffixBuilder._is_summary_invalid("user", "原文") is True
        assert SuffixBuilder._is_summary_invalid("助手", "原文") is True
        assert SuffixBuilder._is_summary_invalid("用户", "原文") is True

    def test_invalid_summary_with_template_markers(self):
        """包含 chat template 标记的输出应被判定为无效"""
        assert SuffixBuilder._is_summary_invalid(
            "<|im_start|>assistant\n摘要内容", "原文"
        ) is True
        assert SuffixBuilder._is_summary_invalid(
            "<|begin_of_text|>摘要内容", "原文"
        ) is True
        assert SuffixBuilder._is_summary_invalid(
            "<|start_header_id|>assistant<|end_header_id|>摘要内容", "原文"
        ) is True

    def test_valid_summary(self):
        """正常的摘要应被判定为有效"""
        assert SuffixBuilder._is_summary_invalid(
            "用户提到了ERP系统的选择，讨论了金蝶和SAP的产品对比", "原文"
        ) is False

    def test_valid_summary_short_but_meaningful(self):
        """短但有意义的摘要应有效"""
        assert SuffixBuilder._is_summary_invalid(
            "讨论了ERP产品的价格和功能对比", "原文"
        ) is False

    def test_llm_summarize_with_chat_template(self):
        """_llm_summarize 应使用 chat template 格式"""
        config = RecallConfig.from_dict({
            "summary": {"strategy": "llm", "max_tokens_per_summary": 100},
        })
        formatter = GenericFormatter(language="cn")
        
        # Mock model adapter with tokenizer
        mock_adapter = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter.tokenizer = mock_tokenizer
        
        # 模拟 apply_chat_template 成功
        mock_tokenizer.apply_chat_template.return_value = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "你是一个摘要助手...\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "请将以下对话内容压缩为摘要...\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        # 模拟 generate 返回有效摘要
        mock_output = MagicMock()
        mock_output.text = "用户讨论了ERP系统的选择方案"
        mock_adapter.generate.return_value = mock_output
        
        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
            model_adapter=mock_adapter,
        )
        
        result = builder._llm_summarize("很长的对话内容...", max_tokens=100)
        
        # 验证使用了 apply_chat_template
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1].get('conversation', call_args[1].get('messages'))
        # 验证 messages 格式: system + user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        assert result == "用户讨论了ERP系统的选择方案"

    def test_llm_summarize_invalid_output_fallback(self):
        """LLM 输出无效时应回退到抽取式摘要"""
        config = RecallConfig.from_dict({
            "summary": {"strategy": "llm", "max_tokens_per_summary": 100},
        })
        formatter = GenericFormatter(language="cn")
        
        mock_adapter = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter.tokenizer = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        # 模拟 encode 返回合理的 token 列表 (用于 _count_tokens)
        mock_tokenizer.encode.side_effect = lambda text: list(range(len(text)))
        
        # 模拟 generate 返回无效输出 (llama3.1-8b 的典型问题)
        mock_output = MagicMock()
        mock_output.text = "assistant。assistant。assistant。assistant。assistant"
        mock_adapter.generate.return_value = mock_output
        
        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
            model_adapter=mock_adapter,
        )
        
        long_text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。"
        result = builder._llm_summarize(long_text, max_tokens=100)
        
        # 应回退到抽取式摘要, 而不是返回 "assistant..."
        assert "assistant" not in result.lower()
        assert len(result) > 0
