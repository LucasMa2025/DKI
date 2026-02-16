"""
FactRetriever 单元测试

测试事实检索器:
- 通过 trace_id 检索消息
- 短消息直接返回
- 长消息分块返回
- 分页 (offset + limit)
- 边界条件 (找不到消息、无 repo)
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest

from dki.core.recall.recall_config import (
    RecallConfig,
    FactRequest,
    FactResponse,
)
from dki.core.recall.fact_retriever import FactRetriever


# ============================================================
# Mock 数据
# ============================================================

@dataclass
class FakeMessage:
    """模拟消息对象"""
    id: str
    content: str
    role: str = "user"
    created_at: str = "2025-01-01 12:00:00"
    message_id: Optional[str] = None


class FakeConversationRepo:
    """模拟 ConversationRepository"""

    def __init__(self, messages: List[FakeMessage] = None):
        self._messages = messages or []

    def get_by_session(self, session_id: str, **kwargs) -> List[FakeMessage]:
        return self._messages

    def get_by_id(self, message_id: str) -> Optional[FakeMessage]:
        for msg in self._messages:
            if str(msg.id) == message_id:
                return msg
        return None


# ============================================================
# 测试类
# ============================================================

class TestFactRetriever:
    """FactRetriever 测试"""

    @pytest.fixture
    def config(self):
        return RecallConfig.from_dict({
            "fact_call": {
                "enabled": True,
                "max_rounds": 3,
                "max_fact_tokens": 800,
                "batch_size": 5,
            },
        })

    @pytest.fixture
    def short_message(self):
        return FakeMessage(
            id="msg-001",
            content="你好，我想预约明天下午三点",
            role="user",
        )

    @pytest.fixture
    def long_message(self):
        # 构造一个超过 500 字符的消息 (每行内容唯一, 避免分块后内容重复)
        content = "\n".join(f"第{i}段非常详细的对话内容，包含编号{i}的信息。" for i in range(100))
        return FakeMessage(
            id="msg-long",
            content=content,
            role="user",
        )

    @pytest.fixture
    def repo_with_short(self, short_message):
        return FakeConversationRepo([short_message])

    @pytest.fixture
    def repo_with_long(self, long_message):
        return FakeConversationRepo([long_message])

    # ============ 初始化测试 ============

    def test_init(self, config):
        retriever = FactRetriever(config=config)
        assert retriever.config is config
        assert retriever._stats["retrievals"] == 0

    # ============ 无 repo 测试 ============

    def test_retrieve_no_repo(self, config):
        """无 conversation_repo 时应返回空响应"""
        retriever = FactRetriever(config=config, conversation_repo=None)
        result = retriever.retrieve(trace_id="msg-001", session_id="sess-001")

        assert isinstance(result, FactResponse)
        assert result.messages == []
        assert result.total_count == 0
        assert result.has_more is False

    # ============ 消息未找到测试 ============

    def test_retrieve_not_found(self, config, repo_with_short):
        """找不到消息时应返回空响应"""
        retriever = FactRetriever(config=config, conversation_repo=repo_with_short)
        result = retriever.retrieve(trace_id="nonexistent", session_id="sess-001")

        assert result.messages == []
        assert result.total_count == 0
        assert retriever._stats["not_found"] == 1

    # ============ 短消息检索测试 ============

    def test_retrieve_short_message(self, config, repo_with_short):
        """短消息应直接返回完整内容"""
        retriever = FactRetriever(config=config, conversation_repo=repo_with_short)
        result = retriever.retrieve(trace_id="msg-001", session_id="sess-001")

        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "你好，我想预约明天下午三点"
        assert result.messages[0]["role"] == "user"
        assert result.total_count == 1
        assert result.has_more is False

    # ============ 长消息分块测试 ============

    def test_retrieve_long_message_chunked(self, config, repo_with_long):
        """长消息应分块返回"""
        retriever = FactRetriever(config=config, conversation_repo=repo_with_long)
        result = retriever.retrieve(
            trace_id="msg-long",
            session_id="sess-001",
            offset=0,
            limit=5,
        )

        assert len(result.messages) > 0
        assert result.total_count > 1  # 应该有多个 chunk
        assert result.trace_id == "msg-long"

    def test_retrieve_long_message_pagination(self, config, repo_with_long):
        """分页应正确工作"""
        retriever = FactRetriever(config=config, conversation_repo=repo_with_long)

        # 第一页
        page1 = retriever.retrieve(
            trace_id="msg-long",
            session_id="sess-001",
            offset=0,
            limit=2,
        )

        if page1.total_count > 2:
            assert page1.has_more is True

            # 第二页
            page2 = retriever.retrieve(
                trace_id="msg-long",
                session_id="sess-001",
                offset=2,
                limit=2,
            )
            assert len(page2.messages) > 0

            # 内容不应重叠
            content1 = set(m["content"] for m in page1.messages)
            content2 = set(m["content"] for m in page2.messages)
            assert content1.isdisjoint(content2), "Pages should not overlap"

    # ============ limit 限制测试 ============

    def test_limit_capped_by_batch_size(self, config, repo_with_long):
        """limit 应被 batch_size 限制"""
        retriever = FactRetriever(config=config, conversation_repo=repo_with_long)
        result = retriever.retrieve(
            trace_id="msg-long",
            session_id="sess-001",
            offset=0,
            limit=100,  # 远大于 batch_size=5
        )
        # 返回的消息数不应超过 batch_size
        assert len(result.messages) <= config.fact_call.batch_size

    # ============ 统计测试 ============

    def test_stats_tracking(self, config, repo_with_short):
        retriever = FactRetriever(config=config, conversation_repo=repo_with_short)
        retriever.retrieve(trace_id="msg-001", session_id="sess-001")
        retriever.retrieve(trace_id="nonexistent", session_id="sess-001")

        stats = retriever.get_stats()
        assert stats["retrievals"] == 2
        assert stats["not_found"] == 1

    # ============ get_by_id 回退测试 ============

    def test_retrieve_via_get_by_id(self, config, short_message):
        """优先使用 get_by_id 方法"""
        repo = FakeConversationRepo([short_message])
        retriever = FactRetriever(config=config, conversation_repo=repo)
        result = retriever.retrieve(trace_id="msg-001", session_id="sess-001")

        assert len(result.messages) == 1
        assert result.messages[0]["content"] == short_message.content

    def test_retrieve_fallback_to_session_scan(self, config, short_message):
        """get_by_id 不存在时回退到会话扫描"""

        class RepoWithoutGetById:
            def __init__(self, messages):
                self._messages = messages

            def get_by_session(self, session_id, **kwargs):
                return self._messages

        repo = RepoWithoutGetById([short_message])
        retriever = FactRetriever(config=config, conversation_repo=repo)
        result = retriever.retrieve(trace_id="msg-001", session_id="sess-001")

        assert len(result.messages) == 1


class TestFactRetrieverChunking:
    """FactRetriever 内容分块测试"""

    def test_chunk_content_short(self):
        """短内容不分块"""
        chunks = FactRetriever._chunk_content("短内容", 500)
        assert len(chunks) == 1
        assert chunks[0] == "短内容"

    def test_chunk_content_paragraphs(self):
        """按段落分块"""
        content = "第一段内容\n" * 10 + "第二段内容\n" * 10
        chunks = FactRetriever._chunk_content(content, 50)
        assert len(chunks) > 1
        # 重组后应包含所有内容
        rejoined = "\n".join(chunks)
        assert "第一段内容" in rejoined
        assert "第二段内容" in rejoined

    def test_chunk_content_single_long_paragraph(self):
        """单个超长段落应按句子分块"""
        content = "这是一句话。" * 100  # 约 600 字符
        chunks = FactRetriever._chunk_content(content, 100)
        assert len(chunks) > 1

    def test_chunk_by_sentence(self):
        """按句子分块"""
        text = "第一句话。第二句话！第三句话？第四句话。第五句话。"
        chunks = FactRetriever._chunk_by_sentence(text, 30)
        assert len(chunks) >= 1
        # 所有句子应被保留
        rejoined = "".join(chunks)
        assert "第一句话" in rejoined
        assert "第五句话" in rejoined

    def test_chunk_empty_content(self):
        """空内容应返回原内容"""
        chunks = FactRetriever._chunk_content("", 500)
        assert len(chunks) == 1
