"""
PromptFormatter 单元测试

测试模型特定提示格式化器:
- GenericFormatter: 通用格式
- DeepSeekFormatter: DeepSeek 特殊 token
- GLMFormatter: ChatGLM observation 格式
- create_formatter 工厂函数
- Function Call 检测
- 结构化 epistemic marker
- 强约束句式
"""

import pytest

from dki.core.recall.recall_config import (
    HistoryItem,
    FactRequest,
    FactResponse,
)
from dki.core.recall.prompt_formatter import (
    PromptFormatter,
    GenericFormatter,
    DeepSeekFormatter,
    GLMFormatter,
    create_formatter,
)


# ============================================================
# GenericFormatter 测试
# ============================================================

class TestGenericFormatter:
    """GenericFormatter 测试"""

    @pytest.fixture
    def formatter_cn(self):
        return GenericFormatter(language="cn")

    @pytest.fixture
    def formatter_en(self):
        return GenericFormatter(language="en")

    @pytest.fixture
    def summary_item(self):
        return HistoryItem(
            type="summary",
            content="用户提到了一家位于北京的餐厅",
            trace_id="msg-005",
            role="user",
            token_count=15,
            confidence="medium",
            facts_covered=["餐厅名称", "大致位置"],
            facts_missing=["营业时间", "价格", "预约方式"],
        )

    @pytest.fixture
    def message_item(self):
        return HistoryItem(
            type="message",
            content="你好，请推荐一家日料店",
            trace_id="msg-010",
            role="user",
            token_count=10,
        )

    # ============ format_summary_item 测试 ============

    def test_summary_item_contains_trace_id(self, formatter_cn, summary_item):
        result = formatter_cn.format_summary_item(summary_item)
        assert 'trace_id="msg-005"' in result

    def test_summary_item_contains_confidence(self, formatter_cn, summary_item):
        result = formatter_cn.format_summary_item(summary_item)
        assert 'confidence="medium"' in result

    def test_summary_item_structured_markers(self, formatter_cn, summary_item):
        """结构化认知标记 (补充建议: 机器可读)"""
        result = formatter_cn.format_summary_item(summary_item)
        assert "[SUMMARY" in result
        assert "[/SUMMARY]" in result
        assert "facts_covered:" in result
        assert "facts_missing:" in result
        assert '"餐厅名称"' in result
        assert '"营业时间"' in result

    def test_summary_item_content(self, formatter_cn, summary_item):
        result = formatter_cn.format_summary_item(summary_item)
        assert "用户提到了一家位于北京的餐厅" in result

    def test_summary_item_no_facts(self, formatter_cn):
        """无 facts 时不应包含 facts 行"""
        item = HistoryItem(
            type="summary",
            content="简单摘要",
            trace_id="msg-001",
            confidence="medium",
        )
        result = formatter_cn.format_summary_item(item)
        assert "facts_covered:" not in result
        assert "facts_missing:" not in result

    # ============ format_message_item 测试 ============

    def test_message_item_cn(self, formatter_cn, message_item):
        result = formatter_cn.format_message_item(message_item)
        assert "用户:" in result or "用户: " in result
        assert "你好，请推荐一家日料店" in result

    def test_message_item_en(self, formatter_en, message_item):
        result = formatter_en.format_message_item(message_item)
        assert "User:" in result or "User: " in result

    def test_message_item_assistant_role(self, formatter_cn):
        item = HistoryItem(
            type="message",
            content="推荐松子日本料理",
            trace_id="msg-011",
            role="assistant",
        )
        result = formatter_cn.format_message_item(item)
        assert "助手:" in result or "助手: " in result

    # ============ format_constraint_instruction 测试 ============

    def test_constraint_instruction_cn(self, formatter_cn):
        result = formatter_cn.format_constraint_instruction(["msg-005", "msg-010"])
        assert "可信" in result
        assert "retrieve_fact" in result
        assert '"msg-005"' in result
        assert '"msg-010"' in result

    def test_constraint_instruction_en(self, formatter_en):
        result = formatter_en.format_constraint_instruction(["msg-005"])
        assert "Trustworthy" in result
        assert "retrieve_fact" in result

    def test_constraint_instruction_strong_constraint_cn(self, formatter_cn):
        """补充建议: 强约束句式"""
        result = formatter_cn.format_constraint_instruction(["msg-001"])
        assert "无效" in result or "INVALID" in result
        assert "强制约束" in result or "MANDATORY" in result

    def test_constraint_instruction_strong_constraint_en(self, formatter_en):
        result = formatter_en.format_constraint_instruction(["msg-001"])
        assert "INVALID" in result
        assert "MANDATORY" in result

    # ============ format_fact_segment 测试 ============

    def test_format_fact_segment(self, formatter_cn):
        response = FactResponse(
            messages=[
                {"role": "user", "content": "我想预约明天下午三点"},
                {"role": "assistant", "content": "好的，已为您预约"},
            ],
            trace_id="msg-005",
            total_count=2,
            offset=0,
            has_more=False,
        )
        result = formatter_cn.format_fact_segment(response)
        assert "[FACT_SEGMENT" in result
        assert 'trace_id="msg-005"' in result
        assert "我想预约明天下午三点" in result
        assert "[/FACT_SEGMENT]" in result

    def test_format_fact_segment_has_more(self, formatter_cn):
        """has_more=True 时应提示继续调用"""
        response = FactResponse(
            messages=[{"role": "user", "content": "内容片段1"}],
            trace_id="msg-005",
            total_count=5,
            offset=0,
            has_more=True,
        )
        result = formatter_cn.format_fact_segment(response)
        assert "retrieve_fact" in result
        assert "offset=1" in result  # offset + len(messages) = 0 + 1

    # ============ detect_fact_request 测试 ============

    def test_detect_fact_request_basic(self, formatter_cn):
        """基本 function call 检测"""
        output = '我需要查看原文。retrieve_fact(trace_id="msg-005", offset=0, limit=5)'
        req = formatter_cn.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-005"
        assert req.offset == 0
        assert req.limit == 5

    def test_detect_fact_request_minimal(self, formatter_cn):
        """最小 function call (仅 trace_id)"""
        output = 'retrieve_fact(trace_id="msg-010")'
        req = formatter_cn.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-010"
        assert req.offset == 0  # 默认值
        assert req.limit == 5  # 默认值

    def test_detect_fact_request_with_pagination(self, formatter_cn):
        """带分页参数"""
        output = 'retrieve_fact(trace_id="msg-005", offset=5, limit=10)'
        req = formatter_cn.detect_fact_request(output)
        assert req is not None
        assert req.offset == 5
        assert req.limit == 10

    def test_detect_fact_request_none(self, formatter_cn):
        """无 function call 时应返回 None"""
        output = "这是一个普通的回答，没有调用任何函数。"
        req = formatter_cn.detect_fact_request(output)
        assert req is None

    def test_detect_fact_request_case_insensitive(self, formatter_cn):
        """大小写不敏感"""
        output = 'RETRIEVE_FACT(trace_id="msg-001")'
        req = formatter_cn.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-001"

    def test_detect_fact_request_in_long_text(self, formatter_cn):
        """在长文本中检测"""
        output = (
            "根据摘要，用户之前提到了一家餐厅，但我需要确认具体信息。\n"
            '让我查看原始记录：retrieve_fact(trace_id="msg-005", offset=0, limit=5)\n'
            "等待结果..."
        )
        req = formatter_cn.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-005"

    # ============ format_history_list 测试 ============

    def test_format_history_list(self, formatter_cn, summary_item, message_item):
        result = formatter_cn.format_history_list([summary_item, message_item])
        assert "[SUMMARY" in result
        assert "用户:" in result or "用户: " in result
        assert "你好，请推荐一家日料店" in result

    def test_format_history_list_empty(self, formatter_cn):
        result = formatter_cn.format_history_list([])
        assert result == ""

    # ============ format_full_suffix 测试 ============

    def test_format_full_suffix_cn(self, formatter_cn, summary_item, message_item):
        result = formatter_cn.format_full_suffix(
            items=[summary_item, message_item],
            trace_ids=["msg-005", "msg-010"],
            query="推荐一家日料店",
        )
        assert "[会话历史参考]" in result
        assert "推荐一家日料店" in result
        assert "retrieve_fact" in result  # 有 summary → 有限定提示

    def test_format_full_suffix_en(self, formatter_en, message_item):
        result = formatter_en.format_full_suffix(
            items=[message_item],
            trace_ids=["msg-010"],
            query="Recommend a restaurant",
        )
        assert "[Session History Reference]" in result
        assert "Recommend a restaurant" in result

    def test_format_full_suffix_no_summary(self, formatter_cn, message_item):
        """无 summary 时不应包含限定提示"""
        result = formatter_cn.format_full_suffix(
            items=[message_item],
            trace_ids=["msg-010"],
            query="你好",
        )
        assert "可信" not in result or "retrieve_fact" not in result


# ============================================================
# DeepSeekFormatter 测试
# ============================================================

class TestDeepSeekFormatter:
    """DeepSeekFormatter 测试"""

    @pytest.fixture
    def formatter(self):
        return DeepSeekFormatter(language="cn")

    def test_is_subclass(self):
        assert issubclass(DeepSeekFormatter, GenericFormatter)

    def test_constraint_includes_tool_definition(self, formatter):
        """DeepSeek 版本应包含 TOOL_DEFINITION"""
        result = formatter.format_constraint_instruction(["msg-001"])
        assert "[TOOL_DEFINITION]" in result
        assert "retrieve_fact" in result
        assert "trace_id" in result

    def test_detect_deepseek_tool_call(self, formatter):
        """检测 DeepSeek 特殊 token 格式"""
        output = (
            '<｜tool▁call▁begin｜>retrieve_fact\n'
            '{"trace_id": "msg-005", "offset": 0, "limit": 5}\n'
            '<｜tool▁call▁end｜>'
        )
        req = formatter.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-005"

    def test_detect_fallback_to_generic(self, formatter):
        """无 DeepSeek 特殊 token 时回退到通用正则"""
        output = 'retrieve_fact(trace_id="msg-010")'
        req = formatter.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-010"

    def test_detect_no_call(self, formatter):
        output = "这是普通回答"
        req = formatter.detect_fact_request(output)
        assert req is None


# ============================================================
# GLMFormatter 测试
# ============================================================

class TestGLMFormatter:
    """GLMFormatter 测试"""

    @pytest.fixture
    def formatter(self):
        return GLMFormatter(language="cn")

    def test_is_subclass(self):
        assert issubclass(GLMFormatter, GenericFormatter)

    def test_format_fact_segment_observation(self, formatter):
        """GLM 格式应包含 <|observation|> 标记"""
        response = FactResponse(
            messages=[{"role": "user", "content": "测试内容"}],
            trace_id="msg-001",
            total_count=1,
            offset=0,
            has_more=False,
        )
        result = formatter.format_fact_segment(response)
        assert "<|observation|>" in result
        assert "<|/observation|>" in result
        assert "[FACT_SEGMENT" in result

    def test_detect_glm_tool_call(self, formatter):
        """检测 GLM 格式的 tool call"""
        output = (
            '<|tool_call|>retrieve_fact\n'
            '{"trace_id": "msg-003"}\n'
        )
        req = formatter.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-003"

    def test_detect_fallback_to_generic(self, formatter):
        output = 'retrieve_fact(trace_id="msg-010")'
        req = formatter.detect_fact_request(output)
        assert req is not None
        assert req.trace_id == "msg-010"


# ============================================================
# create_formatter 工厂函数测试
# ============================================================

class TestCreateFormatter:
    """create_formatter 工厂函数测试"""

    def test_auto_generic(self):
        f = create_formatter(model_name="llama-3-8b", formatter_type="auto")
        assert isinstance(f, GenericFormatter)

    def test_auto_deepseek(self):
        f = create_formatter(model_name="deepseek-v2-chat", formatter_type="auto")
        assert isinstance(f, DeepSeekFormatter)

    def test_auto_glm(self):
        f = create_formatter(model_name="chatglm3-6b", formatter_type="auto")
        assert isinstance(f, GLMFormatter)

    def test_auto_glm_variant(self):
        f = create_formatter(model_name="glm-4-9b-chat", formatter_type="auto")
        assert isinstance(f, GLMFormatter)

    def test_explicit_deepseek(self):
        f = create_formatter(model_name="some-model", formatter_type="deepseek")
        assert isinstance(f, DeepSeekFormatter)

    def test_explicit_glm(self):
        f = create_formatter(model_name="some-model", formatter_type="glm")
        assert isinstance(f, GLMFormatter)

    def test_explicit_generic(self):
        f = create_formatter(model_name="deepseek-v2", formatter_type="generic")
        assert isinstance(f, GenericFormatter)

    def test_unknown_type_defaults_to_generic(self):
        f = create_formatter(model_name="", formatter_type="unknown")
        assert isinstance(f, GenericFormatter)

    def test_language_propagation(self):
        f = create_formatter(model_name="", language="en")
        assert f.language == "en"

    def test_empty_model_name(self):
        f = create_formatter(model_name="", formatter_type="auto")
        assert isinstance(f, GenericFormatter)
