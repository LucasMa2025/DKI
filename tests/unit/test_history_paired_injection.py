"""
v7.3 历史消息成对注入 + 时间戳 + 排序 单元测试

测试目标:
1. HistoryItem 新增 timestamp / pair_id 字段
2. SuffixBuilder._collect_messages 收集时间戳、按时间排序、成对处理
3. SuffixBuilder._ensure_paired_messages 配对逻辑
4. GenericFormatter.format_message_item 时间戳显示
5. InjectionPlanner._format_history_suffix 时间戳 + 成对 + 排序
6. DKIPlugin._merge_recent_and_recalled 合并后按时间排序

Author: AGI Demo Project
Date: 2026-03-12
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from unittest.mock import MagicMock

from dki.core.recall.recall_config import HistoryItem, RecallConfig
from dki.core.recall.suffix_builder import SuffixBuilder
from dki.core.recall.prompt_formatter import GenericFormatter


# ============================================================
# Mock ChatMessage (模拟 AdapterChatMessage)
# ============================================================

@dataclass
class MockChatMessage:
    """模拟 ChatMessage 数据结构"""
    message_id: str
    session_id: str = "session_1"
    user_id: str = "user_1"
    role: str = "user"
    content: str = ""
    timestamp: Optional[datetime] = None
    parent_id: Optional[str] = None
    embedding: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None


# ============================================================
# Test: HistoryItem 新字段
# ============================================================

class TestHistoryItemFields:
    """测试 HistoryItem v7.3 新增字段"""

    def test_timestamp_field_default_none(self):
        """timestamp 默认为 None"""
        item = HistoryItem(type="message", content="hello", trace_id="1")
        assert item.timestamp is None

    def test_timestamp_field_set(self):
        """timestamp 可以设置"""
        item = HistoryItem(
            type="message", content="hello", trace_id="1",
            timestamp="2025-03-12 22:00",
        )
        assert item.timestamp == "2025-03-12 22:00"

    def test_pair_id_field_default_none(self):
        """pair_id 默认为 None"""
        item = HistoryItem(type="message", content="hello", trace_id="1")
        assert item.pair_id is None

    def test_pair_id_field_set(self):
        """pair_id 可以设置"""
        item = HistoryItem(
            type="message", content="hello", trace_id="1",
            pair_id="pair_1",
        )
        assert item.pair_id == "pair_1"


# ============================================================
# Test: GenericFormatter.format_message_item 时间戳
# ============================================================

class TestFormatterTimestamp:
    """测试 GenericFormatter 时间戳显示"""

    def test_format_with_timestamp_cn(self):
        """中文格式化带时间戳"""
        formatter = GenericFormatter(language="cn")
        item = HistoryItem(
            type="message", content="你好", trace_id="1",
            role="user", timestamp="2025-03-12 22:00",
        )
        result = formatter.format_message_item(item)
        assert "[2025-03-12 22:00]" in result
        assert "用户:" in result
        assert "你好" in result

    def test_format_with_timestamp_en(self):
        """英文格式化带时间戳"""
        formatter = GenericFormatter(language="en")
        item = HistoryItem(
            type="message", content="Hello", trace_id="1",
            role="user", timestamp="2025-03-12 22:00",
        )
        result = formatter.format_message_item(item)
        assert "[2025-03-12 22:00]" in result
        assert "User:" in result

    def test_format_without_timestamp(self):
        """无时间戳时不显示方括号"""
        formatter = GenericFormatter(language="cn")
        item = HistoryItem(
            type="message", content="你好", trace_id="1",
            role="user",
        )
        result = formatter.format_message_item(item)
        assert "[" not in result
        assert "用户: 你好" == result

    def test_format_assistant_with_timestamp(self):
        """assistant 消息带时间戳"""
        formatter = GenericFormatter(language="cn")
        item = HistoryItem(
            type="message", content="你好呀", trace_id="2",
            role="assistant", timestamp="2025-03-12 22:01",
        )
        result = formatter.format_message_item(item)
        assert "[2025-03-12 22:01]" in result
        assert "助手:" in result


# ============================================================
# Test: SuffixBuilder._collect_messages 时间戳收集 + 排序
# ============================================================

class TestCollectMessagesTimestamp:
    """测试 SuffixBuilder._collect_messages v7.3 改进"""

    @pytest.fixture
    def builder(self):
        """创建最小化 SuffixBuilder"""
        config = MagicMock()
        config.budget = MagicMock()
        config.budget.instruction_reserve = 120
        config.summary = MagicMock()
        config.summary.per_message_threshold = 200
        config.summary.strategy = "extractive"
        config.summary.max_tokens_per_summary = 100
        formatter = GenericFormatter(language="cn")
        return SuffixBuilder(config=config, prompt_formatter=formatter)

    def test_collect_preserves_timestamp(self, builder):
        """收集消息时保留时间戳"""
        now = datetime.now()
        msgs = [
            MockChatMessage(
                message_id="m1", role="user", content="问题1",
                timestamp=now,
            ),
        ]
        collected = builder._collect_messages(msgs)
        assert len(collected) == 1
        assert collected[0]['timestamp'] != ""
        assert ":" in collected[0]['timestamp']  # 包含时间格式

    def test_collect_sorts_by_timestamp(self, builder):
        """收集后按时间戳排序"""
        t1 = datetime(2025, 3, 12, 22, 0)
        t2 = datetime(2025, 3, 12, 21, 0)  # 更早
        t3 = datetime(2025, 3, 12, 23, 0)  # 更晚
        msgs = [
            MockChatMessage(message_id="m3", role="user", content="C", timestamp=t3),
            MockChatMessage(message_id="m1", role="user", content="A", timestamp=t1),
            MockChatMessage(message_id="m2", role="user", content="B", timestamp=t2),
        ]
        collected = builder._collect_messages(msgs)
        # 应该按时间排序: B(21:00) → A(22:00) → C(23:00)
        assert collected[0]['content'] == "B"
        assert collected[1]['content'] == "A"
        assert collected[2]['content'] == "C"

    def test_collect_no_timestamp_still_works(self, builder):
        """无时间戳的消息仍然正常收集"""
        msgs = [
            MockChatMessage(message_id="m1", role="user", content="问题1"),
        ]
        collected = builder._collect_messages(msgs)
        assert len(collected) == 1
        assert collected[0]['timestamp'] == ""


# ============================================================
# Test: SuffixBuilder._ensure_paired_messages 成对逻辑
# ============================================================

class TestEnsurePairedMessages:
    """测试消息成对处理"""

    @pytest.fixture
    def builder(self):
        config = MagicMock()
        config.budget = MagicMock()
        config.budget.instruction_reserve = 120
        config.summary = MagicMock()
        config.summary.per_message_threshold = 200
        config.summary.strategy = "extractive"
        config.summary.max_tokens_per_summary = 100
        formatter = GenericFormatter(language="cn")
        return SuffixBuilder(config=config, prompt_formatter=formatter)

    def test_adjacent_user_assistant_paired(self, builder):
        """相邻的 user→assistant 自动配对"""
        messages = [
            {'msg_id': 'm1', 'role': 'user', 'content': 'Q1',
             'tokens': 10, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:00',
             'parent_id': None},
            {'msg_id': 'm2', 'role': 'assistant', 'content': 'A1',
             'tokens': 20, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:01',
             'parent_id': None},
        ]
        result = builder._ensure_paired_messages(messages)
        assert result[0].get('pair_id') is not None
        assert result[0]['pair_id'] == result[1]['pair_id']

    def test_parent_id_based_pairing(self, builder):
        """通过 parent_id 配对"""
        messages = [
            {'msg_id': 'm1', 'role': 'user', 'content': 'Q1',
             'tokens': 10, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:00',
             'parent_id': None},
            {'msg_id': 'm2', 'role': 'assistant', 'content': 'A1',
             'tokens': 20, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:01',
             'parent_id': 'm1'},
        ]
        result = builder._ensure_paired_messages(messages)
        assert result[0]['pair_id'] == result[1]['pair_id']

    def test_orphan_assistant_no_crash(self, builder):
        """孤立的 assistant 消息不崩溃"""
        messages = [
            {'msg_id': 'm2', 'role': 'assistant', 'content': 'A1',
             'tokens': 20, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:01',
             'parent_id': None},
        ]
        result = builder._ensure_paired_messages(messages)
        assert len(result) == 1
        assert result[0].get('pair_id') is None

    def test_multiple_pairs(self, builder):
        """多组对话成对"""
        messages = [
            {'msg_id': 'm1', 'role': 'user', 'content': 'Q1',
             'tokens': 10, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:00',
             'parent_id': None},
            {'msg_id': 'm2', 'role': 'assistant', 'content': 'A1',
             'tokens': 20, 'timestamp': '', 'ts_sort_key': '2025-01-01T01:01',
             'parent_id': None},
            {'msg_id': 'm3', 'role': 'user', 'content': 'Q2',
             'tokens': 10, 'timestamp': '', 'ts_sort_key': '2025-01-01T02:00',
             'parent_id': None},
            {'msg_id': 'm4', 'role': 'assistant', 'content': 'A2',
             'tokens': 20, 'timestamp': '', 'ts_sort_key': '2025-01-01T02:01',
             'parent_id': None},
        ]
        result = builder._ensure_paired_messages(messages)
        # 两组对话, 各自成对
        assert result[0]['pair_id'] == result[1]['pair_id']
        assert result[2]['pair_id'] == result[3]['pair_id']
        assert result[0]['pair_id'] != result[2]['pair_id']

    def test_empty_messages(self, builder):
        """空消息列表"""
        result = builder._ensure_paired_messages([])
        assert result == []


# ============================================================
# Test: InjectionPlanner._format_history_suffix 时间戳+排序
# ============================================================

class TestPlannerFormatHistorySuffix:
    """测试 InjectionPlanner._format_history_suffix v7.3"""

    @pytest.fixture
    def planner(self):
        from dki.core.plugin.injection_planner import InjectionPlanner
        return InjectionPlanner(language="cn")

    def test_format_with_timestamps(self, planner):
        """格式化带时间戳"""
        t1 = datetime(2025, 3, 12, 22, 0)
        t2 = datetime(2025, 3, 12, 22, 1)
        msgs = [
            MockChatMessage(message_id="m1", role="user", content="你好", timestamp=t1),
            MockChatMessage(message_id="m2", role="assistant", content="你好呀", timestamp=t2),
        ]
        result = planner._format_history_suffix(msgs)
        assert "[2025-03-12 22:00]" in result
        assert "[2025-03-12 22:01]" in result
        assert "用户:" in result
        assert "助手:" in result

    def test_format_sorted_by_time(self, planner):
        """消息按时间排序"""
        t_early = datetime(2025, 3, 12, 20, 0)
        t_late = datetime(2025, 3, 12, 22, 0)
        msgs = [
            MockChatMessage(message_id="m2", role="user", content="晚问", timestamp=t_late),
            MockChatMessage(message_id="m1", role="user", content="早问", timestamp=t_early),
        ]
        result = planner._format_history_suffix(msgs)
        # 早问应该在晚问前面
        idx_early = result.index("早问")
        idx_late = result.index("晚问")
        assert idx_early < idx_late

    def test_format_paired_user_assistant(self, planner):
        """user+assistant 成对出现"""
        t1 = datetime(2025, 3, 12, 22, 0)
        t2 = datetime(2025, 3, 12, 22, 1)
        msgs = [
            MockChatMessage(message_id="m1", role="user", content="问题", timestamp=t1),
            MockChatMessage(message_id="m2", role="assistant", content="回答", timestamp=t2),
        ]
        result = planner._format_history_suffix(msgs)
        # 问题和回答应该紧邻
        idx_q = result.index("问题")
        idx_a = result.index("回答")
        assert idx_q < idx_a

    def test_format_filters_injection_markers(self, planner):
        """过滤包含注入标记的 assistant 消息"""
        t1 = datetime(2025, 3, 12, 22, 0)
        msgs = [
            MockChatMessage(
                message_id="m1", role="assistant",
                content="[会话历史参考] 一些旧注入内容", timestamp=t1,
            ),
        ]
        result = planner._format_history_suffix(msgs)
        assert "一些旧注入内容" not in result

    def test_format_empty_messages(self, planner):
        """空消息列表返回空字符串"""
        result = planner._format_history_suffix([])
        assert result == ""


# ============================================================
# Test: DKIPlugin._merge_recent_and_recalled 时间排序
# ============================================================

class TestMergeRecentAndRecalled:
    """测试 _merge_recent_and_recalled v7.3 时间排序"""

    def test_merged_sorted_by_timestamp(self):
        """合并后按时间排序"""
        from dki.core.dki_plugin import DKIPlugin

        t1 = datetime(2025, 3, 12, 20, 0)  # 最早
        t2 = datetime(2025, 3, 12, 21, 0)
        t3 = datetime(2025, 3, 12, 22, 0)
        t4 = datetime(2025, 3, 12, 19, 0)  # BM25 召回的更早消息

        recent = [
            MockChatMessage(message_id="r1", content="recent1", timestamp=t1),
            MockChatMessage(message_id="r2", content="recent2", timestamp=t3),
        ]
        recalled = [
            MockChatMessage(message_id="b1", content="bm25_1", timestamp=t2),
            MockChatMessage(message_id="b2", content="bm25_2", timestamp=t4),
        ]

        # 创建一个最小化的 plugin 实例来调用方法
        plugin = object.__new__(DKIPlugin)
        merged = plugin._merge_recent_and_recalled(recent, recalled)

        assert len(merged) == 4
        # 按时间排序: t4(19:00) → t1(20:00) → t2(21:00) → t3(22:00)
        assert merged[0].content == "bm25_2"
        assert merged[1].content == "recent1"
        assert merged[2].content == "bm25_1"
        assert merged[3].content == "recent2"

    def test_deduplication(self):
        """去重: 相同 message_id 只保留一次"""
        from dki.core.dki_plugin import DKIPlugin

        t1 = datetime(2025, 3, 12, 22, 0)
        msg = MockChatMessage(message_id="dup1", content="same", timestamp=t1)

        plugin = object.__new__(DKIPlugin)
        merged = plugin._merge_recent_and_recalled([msg], [msg])

        assert len(merged) == 1
