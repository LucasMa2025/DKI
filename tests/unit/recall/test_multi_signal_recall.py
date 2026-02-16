"""
MultiSignalRecall 单元测试

测试多信号融合召回器:
- 关键词+权重检索
- 向量相似度检索
- 分数归一化与融合
- 近期轮次补充
- 去重逻辑
"""

import math
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from dki.core.recall.recall_config import RecallConfig, RecallResult
from dki.core.recall.multi_signal_recall import MultiSignalRecall


# ============================================================
# Mock 数据
# ============================================================

@dataclass
class FakeMessage:
    """模拟消息对象"""
    id: str
    content: str
    role: str = "user"
    created_at: Optional[str] = None


@dataclass
class FakeSearchResult:
    """模拟向量搜索结果"""
    memory_id: str
    score: float
    content: str = ""


class FakeConversationRepo:
    """模拟 ConversationRepository 包装器"""

    def __init__(self, messages: List[FakeMessage] = None):
        self._messages = messages or []

    def get_by_session(self, session_id: str, **kwargs) -> List[FakeMessage]:
        return self._messages

    def get_recent(self, session_id: str, limit: int = 10, **kwargs) -> List[FakeMessage]:
        return self._messages[-limit:]


class FakeMemoryRouter:
    """模拟 MemoryRouter"""

    def __init__(self, results: List[FakeSearchResult] = None):
        self._results = results or []

    def search(self, query: str, top_k: int = 10) -> List[FakeSearchResult]:
        return self._results[:top_k]


# ============================================================
# 测试类
# ============================================================

class TestMultiSignalRecall:
    """MultiSignalRecall 测试"""

    @pytest.fixture
    def sample_messages(self):
        return [
            FakeMessage(id="msg-001", content="我喜欢吃四川火锅，特别是海底捞", role="user"),
            FakeMessage(id="msg-002", content="好的，我记住了您喜欢四川火锅", role="assistant"),
            FakeMessage(id="msg-003", content="明天下午三点我有个会议", role="user"),
            FakeMessage(id="msg-004", content="已经记录了您明天下午三点的会议", role="assistant"),
            FakeMessage(id="msg-005", content="推荐一家北京的日料店", role="user"),
            FakeMessage(id="msg-006", content="推荐您试试松子日本料理", role="assistant"),
        ]

    @pytest.fixture
    def config(self):
        return RecallConfig()

    @pytest.fixture
    def config_no_keyword(self):
        return RecallConfig.from_dict({
            "signals": {"keyword_enabled": False}
        })

    @pytest.fixture
    def config_no_vector(self):
        return RecallConfig.from_dict({
            "signals": {"vector_enabled": False}
        })

    # ============ 初始化测试 ============

    def test_init_default(self, config):
        recall = MultiSignalRecall(config=config)
        assert recall.config is config
        assert recall._stats["recalls"] == 0

    def test_init_with_components(self, config, sample_messages):
        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter()
        recall = MultiSignalRecall(
            config=config,
            reference_resolver=None,
            memory_router=router,
            conversation_repo=repo,
        )
        assert recall._memory_router is router
        assert recall._conversation_repo is repo

    # ============ 空输入测试 ============

    def test_recall_empty_query(self, config):
        recall = MultiSignalRecall(config=config)
        result = recall.recall(query="", session_id="sess-001")
        assert isinstance(result, RecallResult)
        assert result.messages == []

    def test_recall_no_repo(self, config):
        """无 conversation_repo 时应返回空结果"""
        recall = MultiSignalRecall(config=config)
        result = recall.recall(query="你好", session_id="sess-001")
        assert result.keyword_hits == 0
        assert result.vector_hits == 0
        assert result.messages == []

    # ============ 仅向量检索测试 ============

    def test_recall_vector_only(self, config_no_keyword, sample_messages):
        vector_results = [
            FakeSearchResult(memory_id="msg-001", score=0.9),
            FakeSearchResult(memory_id="msg-005", score=0.7),
        ]
        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter(vector_results)

        recall = MultiSignalRecall(
            config=config_no_keyword,
            memory_router=router,
            conversation_repo=repo,
        )
        result = recall.recall(query="推荐火锅", session_id="sess-001")

        assert result.vector_hits == 2
        assert result.keyword_hits == 0
        assert len(result.scores) > 0

    # ============ 仅关键词检索测试 ============

    @patch("dki.core.recall.multi_signal_recall.JIEBA_AVAILABLE", True)
    @patch("dki.core.recall.multi_signal_recall.jieba", create=True)
    def test_recall_keyword_only(self, mock_jieba, config_no_vector, sample_messages):
        """关键词检索 (mock jieba)"""
        mock_jieba.analyse.extract_tags.return_value = [
            ("火锅", 0.8),
            ("四川", 0.6),
        ]

        repo = FakeConversationRepo(sample_messages)
        recall = MultiSignalRecall(
            config=config_no_vector,
            conversation_repo=repo,
        )
        result = recall.recall(query="四川火锅", session_id="sess-001")

        assert result.keyword_hits > 0
        assert result.vector_hits == 0

    # ============ 融合测试 ============

    @patch("dki.core.recall.multi_signal_recall.JIEBA_AVAILABLE", True)
    @patch("dki.core.recall.multi_signal_recall.jieba", create=True)
    def test_recall_fusion(self, mock_jieba, config, sample_messages):
        """关键词+向量融合"""
        mock_jieba.analyse.extract_tags.return_value = [
            ("火锅", 0.8),
        ]

        vector_results = [
            FakeSearchResult(memory_id="msg-005", score=0.85),
        ]

        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter(vector_results)

        recall = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        result = recall.recall(query="火锅", session_id="sess-001")

        assert result.keyword_hits > 0
        assert result.vector_hits > 0
        # 融合后应有结果
        assert len(result.scores) > 0

    # ============ 近期轮次补充测试 ============

    def test_recent_turns_added(self, sample_messages):
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": False},
            "budget": {"min_recent_turns": 2},
        })
        repo = FakeConversationRepo(sample_messages)

        recall = MultiSignalRecall(
            config=config,
            conversation_repo=repo,
        )
        result = recall.recall(query="你好", session_id="sess-001")

        assert result.recent_turns_added > 0
        assert len(result.messages) > 0

    # ============ 去重测试 ============

    def test_deduplication(self, sample_messages):
        """近期消息和召回消息重叠时应去重"""
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": True},
            "budget": {"min_recent_turns": 3},
        })

        # 向量返回最后两条消息 (与近期轮次重叠)
        vector_results = [
            FakeSearchResult(memory_id="msg-005", score=0.9),
            FakeSearchResult(memory_id="msg-006", score=0.8),
        ]

        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter(vector_results)

        recall = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        result = recall.recall(query="日料", session_id="sess-001")

        # 检查无重复 message_id
        msg_ids = [
            str(getattr(m, "id", None) or getattr(m, "message_id", id(m)))
            for m in result.messages
        ]
        assert len(msg_ids) == len(set(msg_ids)), "Messages should be deduplicated"

    # ============ 分数归一化测试 ============

    def test_normalize_scores_sigmoid(self):
        """sigmoid 归一化应将分数映射到 [0, 1]"""
        raw = {"a": 0.1, "b": 0.5, "c": 1.0, "d": 2.0}
        normalized = MultiSignalRecall._normalize_scores(raw)

        for v in normalized.values():
            assert 0.0 <= v <= 1.0, f"Normalized score {v} out of [0, 1]"

    def test_normalize_scores_empty(self):
        assert MultiSignalRecall._normalize_scores({}) == {}

    def test_normalize_scores_single(self):
        """单个值应归一化为 0.5 (sigmoid 中位数)"""
        raw = {"a": 1.0}
        normalized = MultiSignalRecall._normalize_scores(raw)
        assert abs(normalized["a"] - 0.5) < 1e-6

    def test_clip_scores(self):
        raw = {"a": -0.5, "b": 0.5, "c": 1.5}
        clipped = MultiSignalRecall._clip_scores(raw)
        assert clipped["a"] == 0.0
        assert clipped["b"] == 0.5
        assert clipped["c"] == 1.0

    # ============ 统计测试 ============

    def test_stats_increment(self, config):
        recall = MultiSignalRecall(config=config)
        recall.recall(query="test1", session_id="s1")
        recall.recall(query="test2", session_id="s1")
        stats = recall.get_stats()
        assert stats["recalls"] == 2

    # ============ 向量阈值过滤测试 ============

    def test_vector_threshold_filtering(self, config, sample_messages):
        """低于阈值的向量结果应被过滤"""
        vector_results = [
            FakeSearchResult(memory_id="msg-001", score=0.9),
            FakeSearchResult(memory_id="msg-002", score=0.3),  # 低于默认阈值 0.5
        ]
        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter(vector_results)

        recall = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        result = recall.recall(query="test", session_id="sess-001")

        # 只有 msg-001 应通过阈值
        assert result.vector_hits == 1

    # ============ max_results 限制测试 ============

    def test_max_results_limit(self, sample_messages):
        config = RecallConfig.from_dict({
            "signals": {"keyword_enabled": False, "vector_enabled": True, "vector_threshold": 0.0},
            "budget": {"min_recent_turns": 0},  # 禁用近期轮次, 只测试 max_results 限制
        })
        vector_results = [
            FakeSearchResult(memory_id=f"msg-{i:03d}", score=0.9 - i * 0.1)
            for i in range(1, 7)
        ]
        repo = FakeConversationRepo(sample_messages)
        router = FakeMemoryRouter(vector_results)

        recall = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        result = recall.recall(query="test", session_id="sess-001", max_results=2)

        # 无近期轮次时, scores 应该最多 2 个 (由 max_results 限制)
        assert len(result.scores) <= 2
        assert len(result.messages) <= 2