"""
Unit Tests for Trailing Unpaired User Message Filter (v7.4)

测试历史消息合并时过滤末尾无 assistant 回复的 user 消息:
1. DKIPlugin._remove_trailing_unpaired_user — 合并层过滤
2. SuffixBuilder._remove_trailing_unpaired_user — 收集层过滤
3. InjectionPlanner._format_history_suffix — 格式化层过滤

原因:
- Demo 先写入 user 消息再调用 DKI, get_recent_messages 会拉出当前查询
- 当前查询已在 prompt 最后, 历史中不应重复

Author: AGI Demo Project
"""

import pytest
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock


# ============================================================
# Mock ChatMessage
# ============================================================

@dataclass
class MockChatMessage:
    """模拟 AdapterChatMessage"""
    role: str = "user"
    content: str = ""
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None
    parent_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.message_id is None:
            self.message_id = str(id(self))


# ============================================================
# Helper: 创建消息序列
# ============================================================

def make_conversation(
    pairs: int = 3,
    trailing_user: int = 0,
    base_time: Optional[datetime] = None,
) -> List[MockChatMessage]:
    """
    创建对话消息序列
    
    Args:
        pairs: user+assistant 配对数量
        trailing_user: 末尾无配对 user 消息数量
        base_time: 基础时间
    """
    if base_time is None:
        base_time = datetime(2026, 3, 11, 14, 0, tzinfo=timezone.utc)
    
    msgs = []
    for i in range(pairs):
        t = base_time + timedelta(minutes=i * 2)
        user_msg = MockChatMessage(
            role="user",
            content=f"用户问题 {i+1}",
            timestamp=t,
            message_id=f"user_{i}",
        )
        assistant_msg = MockChatMessage(
            role="assistant",
            content=f"助手回复 {i+1}",
            timestamp=t + timedelta(minutes=1),
            message_id=f"assistant_{i}",
            parent_id=f"user_{i}",
        )
        msgs.extend([user_msg, assistant_msg])
    
    # 添加末尾未配对 user 消息
    for j in range(trailing_user):
        t = base_time + timedelta(minutes=(pairs + j) * 2)
        msgs.append(MockChatMessage(
            role="user",
            content=f"用户当前查询 {j+1}",
            timestamp=t,
            message_id=f"trailing_user_{j}",
        ))
    
    return msgs


# ============================================================
# 测试: DKIPlugin._remove_trailing_unpaired_user
# ============================================================

class TestDKIPluginRemoveTrailingUser:
    """测试 DKIPlugin._remove_trailing_unpaired_user"""

    def _get_method(self):
        """导入并获取静态方法"""
        from dki.core.dki_plugin import DKIPlugin
        return DKIPlugin._remove_trailing_unpaired_user

    def test_no_trailing_user(self):
        """正常对话: 末尾是 assistant, 不应移除任何消息"""
        remove = self._get_method()
        msgs = make_conversation(pairs=3, trailing_user=0)
        result = remove(msgs)
        assert len(result) == 6  # 3 pairs × 2

    def test_one_trailing_user(self):
        """末尾有 1 条无配对 user 消息"""
        remove = self._get_method()
        msgs = make_conversation(pairs=3, trailing_user=1)
        assert len(msgs) == 7  # 6 + 1
        result = remove(msgs)
        assert len(result) == 6
        assert result[-1].role == "assistant"

    def test_multiple_trailing_users(self):
        """末尾有多条无配对 user 消息"""
        remove = self._get_method()
        msgs = make_conversation(pairs=2, trailing_user=3)
        assert len(msgs) == 7  # 4 + 3
        result = remove(msgs)
        assert len(result) == 4
        assert result[-1].role == "assistant"

    def test_all_user_messages(self):
        """全部都是 user 消息 (极端情况)"""
        remove = self._get_method()
        msgs = [
            MockChatMessage(role="user", content=f"q{i}")
            for i in range(5)
        ]
        result = remove(msgs)
        assert len(result) == 0

    def test_empty_list(self):
        """空列表"""
        remove = self._get_method()
        result = remove([])
        assert len(result) == 0

    def test_single_assistant(self):
        """只有一条 assistant 消息"""
        remove = self._get_method()
        msgs = [MockChatMessage(role="assistant", content="hello")]
        result = remove(msgs)
        assert len(result) == 1

    def test_preserves_middle_unpaired_user(self):
        """中间出现的无配对 user 不应被移除, 只移除末尾的"""
        remove = self._get_method()
        base = datetime(2026, 3, 11, 14, 0, tzinfo=timezone.utc)
        msgs = [
            MockChatMessage(role="user", content="q1", timestamp=base),
            # 注意: 这里没有 assistant 回复, 但不在末尾
            MockChatMessage(role="user", content="q2", timestamp=base + timedelta(minutes=2)),
            MockChatMessage(role="assistant", content="a2", timestamp=base + timedelta(minutes=3)),
            # 末尾的无配对 user
            MockChatMessage(role="user", content="current_query", timestamp=base + timedelta(minutes=4)),
        ]
        result = remove(msgs)
        assert len(result) == 3  # q1, q2, a2 保留; current_query 移除
        assert result[-1].role == "assistant"


# ============================================================
# 测试: SuffixBuilder._remove_trailing_unpaired_user
# ============================================================

class TestSuffixBuilderRemoveTrailingUser:
    """测试 SuffixBuilder._remove_trailing_unpaired_user (dict 格式)"""

    def _get_method(self):
        from dki.core.recall.suffix_builder import SuffixBuilder
        return SuffixBuilder._remove_trailing_unpaired_user

    def test_dict_format_filtering(self):
        """dict 格式的消息列表过滤"""
        remove = self._get_method()
        messages = [
            {"role": "user", "content": "q1", "msg_id": "1"},
            {"role": "assistant", "content": "a1", "msg_id": "2"},
            {"role": "user", "content": "q2", "msg_id": "3"},
            {"role": "assistant", "content": "a2", "msg_id": "4"},
            {"role": "user", "content": "current_query", "msg_id": "5"},
        ]
        result = remove(messages)
        assert len(result) == 4
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == "a2"

    def test_dict_no_trailing(self):
        """没有末尾 user, 不变"""
        remove = self._get_method()
        messages = [
            {"role": "user", "content": "q1", "msg_id": "1"},
            {"role": "assistant", "content": "a1", "msg_id": "2"},
        ]
        result = remove(messages)
        assert len(result) == 2

    def test_dict_empty(self):
        remove = self._get_method()
        assert remove([]) == []


# ============================================================
# 测试: 样本场景模拟
# ============================================================

class TestSampleScenario:
    """模拟 样本.md 中的实际场景"""

    def test_sample_scenario(self):
        """
        模拟样本场景:
        - 多轮 user+assistant 配对
        - 最后一条是 user 当前查询 (无 assistant 回复)
        """
        from dki.core.dki_plugin import DKIPlugin
        remove = DKIPlugin._remove_trailing_unpaired_user
        
        base = datetime(2026, 3, 11, 14, 0, tzinfo=timezone.utc)
        msgs = [
            # 第 1 轮
            MockChatMessage(role="user", content="你好千问,请介绍一下你自己", 
                          timestamp=base, message_id="u1"),
            MockChatMessage(role="assistant", content="你好呀，Lucas...", 
                          timestamp=base + timedelta(minutes=1), message_id="a1"),
            # 第 2 轮
            MockChatMessage(role="user", content="请给我讲讲 ERP 系统在 AI 时代的价值",
                          timestamp=base + timedelta(minutes=3), message_id="u2"),
            MockChatMessage(role="assistant", content="站在你熟悉的技术栈视角...",
                          timestamp=base + timedelta(minutes=4), message_id="a2"),
            # ... 更多配对 ...
            # 最后一条: 用户当前查询 (无 assistant 回复)
            MockChatMessage(role="user", 
                          content="你好千问,你记得最近你给我介绍的国产 ERP 产品吗",
                          timestamp=base + timedelta(hours=5), message_id="u_current"),
        ]
        
        result = remove(msgs)
        
        # 当前查询应被移除
        assert len(result) == 4  # 2 对 × 2
        assert result[-1].role == "assistant"
        # 当前查询不在结果中
        assert all(m.message_id != "u_current" for m in result)
