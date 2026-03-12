"""
Unit Tests for StreamThinkDetector (v5.9)

测试流式思考内容检测器的核心功能:
1. 正常文本透传 (type="token")
2. <think>...</think> 检测 (type="thinking")
3. <|think|>...<|/think|> 检测
4. "Thinking Process:" 纯文本检测
5. 中文 "思考过程:" 检测
6. 混合内容: 思考 + 正常输出
7. get_clean_text() 最终过滤
8. show_thinking 配置控制
9. StreamThinkFilter 向后兼容

Author: AGI Demo Project
"""

import pytest
from typing import List, Tuple

from dki.core.text_utils import (
    StreamThinkDetector,
    StreamThinkFilter,
    ThinkContentFilter,
    init_think_filter,
    get_show_thinking,
    create_stream_detector,
    create_stream_filter,
    strip_think_content,
)


# ============================================================
# Helper: 模拟逐 token 输入
# ============================================================

def simulate_stream(
    detector: StreamThinkDetector,
    text: str,
    chunk_size: int = 1,
) -> List[Tuple[str, str]]:
    """模拟流式输入, 收集所有事件"""
    all_events: List[Tuple[str, str]] = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        events = detector.feed(chunk)
        all_events.extend(events)
    # flush
    all_events.extend(detector.flush())
    return all_events


def collect_by_type(
    events: List[Tuple[str, str]],
    event_type: str,
) -> str:
    """收集指定类型事件的文本"""
    return "".join(text for t, text in events if t == event_type)


# ============================================================
# 测试: 正常文本透传
# ============================================================

class TestNormalPassthrough:
    """正常文本应全部作为 token 事件输出"""

    def test_simple_text(self):
        detector = StreamThinkDetector()
        events = simulate_stream(detector, "你好，我是AI助手。")
        tokens = collect_by_type(events, "token")
        thinking = collect_by_type(events, "thinking")
        assert tokens == "你好，我是AI助手。"
        assert thinking == ""

    def test_multiline_text(self):
        text = "第一行内容\n第二行内容\n第三行内容"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text)
        tokens = collect_by_type(events, "token")
        assert tokens == text

    def test_empty_input(self):
        detector = StreamThinkDetector()
        events = detector.feed("")
        assert events == []

    def test_large_chunk(self):
        detector = StreamThinkDetector()
        text = "Hello world, this is a normal response."
        events = simulate_stream(detector, text, chunk_size=10)
        tokens = collect_by_type(events, "token")
        assert tokens == text


# ============================================================
# 测试: DeepSeek-R1 <think>...</think>
# ============================================================

class TestDeepSeekThink:
    """DeepSeek-R1 格式: <think>推理内容</think>最终回复"""

    def test_complete_think_block(self):
        text = "<think>让我分析一下这个问题...</think>你好！这是我的回复。"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "让我分析" in thinking
        assert "你好" in tokens
        assert "<think>" not in tokens

    def test_think_block_char_by_char(self):
        """逐字符输入, 测试缓冲机制"""
        text = "<think>分析中</think>回复内容"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=1)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "分析中" in thinking
        assert "回复内容" in tokens

    def test_think_with_spaces(self):
        text = "< think >推理过程< / think >正式回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "推理过程" in thinking
        assert "正式回复" in tokens

    def test_get_clean_text(self):
        """get_clean_text() 应返回过滤后的干净文本"""
        text = "<think>推理</think>干净回复"
        detector = StreamThinkDetector()
        simulate_stream(detector, text)
        clean = detector.get_clean_text()
        assert "<think>" not in clean
        assert "推理" not in clean
        assert "干净回复" in clean


# ============================================================
# 测试: Qwen3 <|think|>...<|/think|>
# ============================================================

class TestQwen3Think:
    """Qwen3/3.5 格式: <|think|>推理内容<|/think|>"""

    def test_qwen3_think_block(self):
        text = "<|think|>分析请求...<|/think|>这是回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=3)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "分析请求" in thinking
        assert "这是回复" in tokens


# ============================================================
# 测试: "Thinking Process:" 纯文本格式
# ============================================================

class TestThinkingProcess:
    """Qwen 系列: Thinking Process:\\n分析内容...\\n正式回复"""

    def test_thinking_process_block(self):
        text = "Thinking Process:\n1. Analyze the request\n2. Draft response\nHello! Here is my answer."
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=5)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        # 注意: 纯文本格式没有明确的结束标记,
        # 检测器可能将全部内容标记为 thinking
        # 但 get_clean_text() 会用全量正则过滤
        assert "Thinking Process:" in thinking or "Thinking Process:" not in tokens

    def test_get_clean_text_removes_thinking_process(self):
        """全量过滤应能移除 Thinking Process 块"""
        # 初始化过滤器 (含 Qwen 规则)
        init_think_filter({
            "enabled": True,
            "show_thinking": True,
            "rules": [
                {
                    "name": "qwen_thinking_process",
                    "pattern": r"(?:^|\n)Thinking Process:\s*\n[\s\S]*?(?=\n(?:Hello|Hi|OK|Sure|Let me|I |Based on|Here|The |In |For )|\Z)",
                    "flags": "DOTALL|MULTILINE",
                    "action": "remove",
                    "scope": "all",
                },
            ],
        })
        text = "Thinking Process:\n1. Analyze\n2. Draft\nHello! Here is my answer."
        detector = StreamThinkDetector()
        simulate_stream(detector, text, chunk_size=5)
        clean = detector.get_clean_text()
        assert "Thinking Process:" not in clean
        assert "Hello" in clean


# ============================================================
# 测试: 中文思考过程
# ============================================================

class TestChineseThinking:
    """中文格式: 思考过程:\\n分析...\\n回复"""

    def test_chinese_thinking_detected(self):
        text = "思考过程：\n分析用户的问题\n制定回复策略\n你好！这是回复。"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=3)
        thinking = collect_by_type(events, "thinking")
        assert "思考过程" in thinking


# ============================================================
# 测试: 混合内容
# ============================================================

class TestMixedContent:
    """思考块 + 正常内容混合"""

    def test_think_then_normal(self):
        text = "<think>step1\nstep2</think>正式回复内容"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=2)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "step1" in thinking
        assert "正式回复" in tokens

    def test_normal_then_think_then_normal(self):
        """正常 → 思考 → 正常"""
        text = "开头内容<think>推理</think>结尾内容"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=2)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "推理" in thinking
        assert "开头内容" in tokens
        assert "结尾内容" in tokens


# ============================================================
# 测试: StreamThinkFilter 向后兼容
# ============================================================

class TestStreamThinkFilterCompat:
    """StreamThinkFilter 向后兼容行为"""

    def test_feed_returns_only_tokens(self):
        """feed() 应只返回正常内容, 过滤掉思考"""
        f = StreamThinkFilter(
            think_filter=ThinkContentFilter(),
            buffer_max_chars=200,
        )
        text = "<think>推理</think>回复"
        result = ""
        for ch in text:
            out = f.feed(ch)
            result += out
        result += f.flush()
        assert "推理" not in result
        assert "回复" in result

    def test_get_clean_full_text(self):
        f = StreamThinkFilter(
            think_filter=ThinkContentFilter(),
            buffer_max_chars=200,
        )
        text = "<think>推理</think>干净回复"
        for ch in text:
            f.feed(ch)
        f.flush()
        clean = f.get_clean_full_text()
        assert "推理" not in clean
        assert "干净回复" in clean


# ============================================================
# 测试: show_thinking 配置
# ============================================================

class TestShowThinkingConfig:
    """show_thinking 配置控制"""

    def test_show_thinking_true(self):
        init_think_filter({
            "enabled": True,
            "show_thinking": True,
            "rules": [],
        })
        assert get_show_thinking() is True

    def test_show_thinking_false(self):
        init_think_filter({
            "enabled": True,
            "show_thinking": False,
            "rules": [],
        })
        assert get_show_thinking() is False

    def test_show_thinking_string_true(self):
        init_think_filter({
            "enabled": True,
            "show_thinking": "true",
            "rules": [],
        })
        assert get_show_thinking() is True

    def test_show_thinking_string_false(self):
        init_think_filter({
            "enabled": True,
            "show_thinking": "false",
            "rules": [],
        })
        assert get_show_thinking() is False

    def test_show_thinking_default(self):
        """未配置时默认为 True"""
        init_think_filter({
            "enabled": True,
            "rules": [],
        })
        assert get_show_thinking() is True


# ============================================================
# 测试: create_stream_detector 工厂函数
# ============================================================

class TestFactoryFunctions:
    """工厂函数"""

    def test_create_stream_detector(self):
        detector = create_stream_detector(buffer_max_chars=100)
        assert isinstance(detector, StreamThinkDetector)
        assert detector.state == "NORMAL"

    def test_create_stream_filter_compat(self):
        f = create_stream_filter(buffer_max_chars=100)
        assert isinstance(f, StreamThinkFilter)


# ============================================================
# 测试: 边界情况
# ============================================================

class TestEdgeCases:
    """边界情况"""

    def test_incomplete_think_tag(self):
        """未闭合的 <think> 标签"""
        text = "<think>推理内容没有结束标签"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=3)
        # flush 时应释放缓冲区中的思考内容
        thinking = collect_by_type(events, "thinking")
        assert "推理内容" in thinking

    def test_only_closing_tag(self):
        """仅有 </think> (DeepSeek-R1 截断情况)"""
        text = "推理过程的结尾</think>正式回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=2)
        # get_clean_text 应能处理这种情况
        clean = detector.get_clean_text()
        assert "正式回复" in clean

    def test_multiple_think_blocks(self):
        """多个 think 块"""
        text = "<think>第一次推理</think>中间内容<think>第二次推理</think>最终回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=3)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert "第一次推理" in thinking
        assert "第二次推理" in thinking
        assert "中间内容" in tokens
        assert "最终回复" in tokens

    def test_empty_think_block(self):
        """空的 think 块"""
        text = "<think></think>回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=1)
        tokens = collect_by_type(events, "token")
        assert "回复" in tokens

    def test_very_long_thinking(self):
        """超长思考内容 (测试缓冲区释放)"""
        long_think = "x" * 5000
        text = f"<think>{long_think}</think>回复"
        detector = StreamThinkDetector()
        events = simulate_stream(detector, text, chunk_size=100)
        thinking = collect_by_type(events, "thinking")
        tokens = collect_by_type(events, "token")
        assert len(thinking) > 2000
        assert "回复" in tokens


# ============================================================
# 测试: strip_think_content 兼容旧接口
# ============================================================

class TestStripThinkContentCompat:
    """strip_think_content 兼容旧接口"""

    def test_basic_strip(self):
        text = "<think>推理</think>干净内容"
        clean, stripped = strip_think_content(text)
        assert stripped is True
        assert "推理" not in clean
        assert "干净内容" in clean

    def test_no_think(self):
        text = "普通文本无思考"
        clean, stripped = strip_think_content(text)
        assert stripped is False
        assert clean == text
