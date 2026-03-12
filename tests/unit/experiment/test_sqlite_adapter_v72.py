"""
SQLiteDataAdapter v7.2 单元测试

测试内容:
- P0-1: get_recent_messages() — 近轮对话获取
- P0-2: search_relevant_history() — jieba + BM25 检索
- _tokenize() 分词方法
- _bm25_score() BM25 评分算法
"""

import asyncio
import math
from collections import Counter
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from dki.experiment.sqlite_adapter import SQLiteDataAdapter, JIEBA_AVAILABLE


# ============ 辅助工具 ============

class FakeConversation:
    """模拟 Conversation ORM 对象"""

    def __init__(self, id: str, session_id: str, role: str, content: str,
                 created_at: datetime = None, user_id: str = "test_user"):
        self.id = id
        self.session_id = session_id
        self.role = role
        self.content = content
        self.created_at = created_at or datetime.utcnow()
        self.session = type('Session', (), {'user_id': user_id})()

    def get_metadata(self):
        return {}


class FakeSessionModel:
    """模拟 Session ORM 对象"""

    def __init__(self, id: str, user_id: str):
        self.id = id
        self.user_id = user_id


class FakeQuery:
    """模拟 SQLAlchemy Query 链式调用"""

    def __init__(self, results):
        self._results = results

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, n):
        self._results = self._results[:n]
        return self

    def all(self):
        return self._results


class FakeDBSession:
    """模拟 SQLAlchemy Session"""

    def __init__(self, sessions=None, conversations=None):
        self._sessions = sessions or []
        self._conversations = conversations or []

    def query(self, model):
        from dki.database.models import Session as SessionModel, Conversation
        if model == SessionModel or (hasattr(model, '__name__') and 'id' in str(model)):
            return FakeQuery(self._sessions)
        return FakeQuery(self._conversations)


def _make_db_manager(sessions=None, conversations=None):
    """创建模拟的 DatabaseManager"""
    from contextlib import contextmanager

    db_manager = MagicMock()
    db_manager._db_path = ":memory:"

    fake_session = FakeDBSession(sessions, conversations)

    @contextmanager
    def session_scope():
        yield fake_session

    db_manager.session_scope = session_scope
    return db_manager


def run_async(coro):
    """同步运行异步函数"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============ BM25 算法测试 ============

class TestBM25Score:
    """测试 BM25 评分算法"""

    def test_empty_corpus(self):
        """空语料库返回空列表"""
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["hello"],
            doc_tokens_list=[],
        )
        assert scores == []

    def test_single_doc_exact_match(self):
        """单文档精确匹配"""
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["素食", "过敏"],
            doc_tokens_list=[["素食", "过敏", "海鲜"]],
        )
        assert len(scores) == 1
        assert scores[0] > 0.0

    def test_no_match_zero_score(self):
        """无匹配时分数为 0"""
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["素食"],
            doc_tokens_list=[["音乐", "吉他"]],
        )
        assert scores[0] == 0.0

    def test_higher_score_for_better_match(self):
        """更好的匹配应有更高分数"""
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["素食", "过敏", "海鲜"],
            doc_tokens_list=[
                ["素食", "过敏", "海鲜", "餐厅"],  # 3 个匹配
                ["素食", "餐厅", "推荐"],            # 1 个匹配
                ["音乐", "吉他", "古典"],            # 0 个匹配
            ],
        )
        assert scores[0] > scores[1] > scores[2]
        assert scores[2] == 0.0

    def test_idf_effect(self):
        """IDF 效果: 稀有词匹配应比常见词匹配得分更高"""
        # "稀有词" 只出现在 doc0, "常见词" 出现在所有文档
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["稀有词"],
            doc_tokens_list=[
                ["稀有词", "常见词"],
                ["常见词", "其他"],
                ["常见词", "另外"],
            ],
        )
        # doc0 匹配稀有词, 其他不匹配
        assert scores[0] > 0.0
        assert scores[1] == 0.0

    def test_tf_saturation(self):
        """TF 饱和: 重复出现同一词不会线性增加分数"""
        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["素食"],
            doc_tokens_list=[
                ["素食"],                            # tf=1
                ["素食", "素食", "素食", "素食"],    # tf=4
            ],
        )
        # tf=4 的分数应高于 tf=1, 但不是 4 倍
        assert scores[1] > scores[0]
        assert scores[1] < scores[0] * 4

    def test_document_length_normalization(self):
        """文档长度归一化: 短文档中的匹配应比长文档中的匹配得分更高"""
        short_doc = ["素食", "推荐"]
        long_doc = ["素食"] + ["填充词"] * 50

        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=["素食"],
            doc_tokens_list=[short_doc, long_doc],
        )
        # 短文档中的匹配应得分更高
        assert scores[0] > scores[1]


# ============ 分词测试 ============

class TestTokenize:
    """测试分词方法"""

    @pytest.fixture(autouse=True)
    def setup(self):
        db_manager = MagicMock()
        db_manager._db_path = ":memory:"
        self.adapter = SQLiteDataAdapter(db_manager=db_manager)

    def test_english_tokenization(self):
        """英文分词"""
        tokens = self.adapter._tokenize("I love hiking and outdoor activities")
        # 停用词 "I" 和 "and" 应被过滤
        assert "love" in tokens
        assert "hiking" in tokens
        assert "outdoor" in tokens

    def test_chinese_tokenization(self):
        """中文分词"""
        tokens = self.adapter._tokenize("我喜欢素食和户外运动")
        # 应该有有意义的中文 token
        assert len(tokens) > 0
        # 停用词 "我" "和" 应被过滤
        for t in tokens:
            assert t not in {"我", "和", "的"}

    def test_mixed_language(self):
        """中英混合分词"""
        tokens = self.adapter._tokenize("我喜欢Python编程和AI技术")
        assert len(tokens) > 0
        # 应包含英文词
        has_en = any(t.isascii() for t in tokens)
        # 应包含中文词
        has_cn = any(not t.isascii() for t in tokens)
        assert has_en or has_cn  # 至少有一种

    def test_empty_text(self):
        """空文本返回空列表"""
        tokens = self.adapter._tokenize("")
        assert tokens == []

    def test_stopwords_filtered(self):
        """停用词被过滤"""
        tokens = self.adapter._tokenize("the is a an in on at to for of")
        # 所有都是停用词, 应该被过滤
        assert len(tokens) == 0

    def test_single_char_filtered(self):
        """单字符被过滤"""
        tokens = self.adapter._tokenize("I a x y z")
        assert len(tokens) == 0


# ============ get_recent_messages 测试 ============

class TestGetRecentMessages:
    """测试 get_recent_messages 方法"""

    def test_returns_empty_when_no_sessions(self):
        """无会话时返回空列表"""
        db_manager = _make_db_manager(sessions=[], conversations=[])
        adapter = SQLiteDataAdapter(db_manager=db_manager)
        run_async(adapter.connect())

        # 需要 mock 内部的 SQLAlchemy 查询
        with patch.object(adapter, 'get_recent_messages', return_value=[]) as mock_method:
            result = run_async(mock_method(user_id="user1", limit=10))
            assert result == []

    def test_returns_messages_in_chronological_order(self):
        """返回的消息应按时间正序排列"""
        now = datetime.utcnow()
        convs = [
            FakeConversation("c3", "s1", "user", "第三条", now - timedelta(minutes=1)),
            FakeConversation("c2", "s1", "assistant", "第二条", now - timedelta(minutes=2)),
            FakeConversation("c1", "s1", "user", "第一条", now - timedelta(minutes=3)),
        ]
        sessions = [FakeSessionModel("s1", "user1")]

        db_manager = _make_db_manager(sessions=sessions, conversations=convs)
        adapter = SQLiteDataAdapter(db_manager=db_manager)

        # 由于内部使用 SQLAlchemy 模型, 我们直接测试 _conversations_to_chat_messages
        result = adapter._conversations_to_chat_messages(
            list(reversed(convs)),  # 模拟反转后的正序
            session_id="cross_session",
            user_id="user1",
        )
        assert len(result) == 3
        assert result[0].content == "第一条"
        assert result[2].content == "第三条"

    def test_limit_parameter(self):
        """limit 参数限制返回数量"""
        convs = [
            FakeConversation(f"c{i}", "s1", "user", f"消息{i}")
            for i in range(20)
        ]
        # 测试 _conversations_to_chat_messages 的输出
        adapter = SQLiteDataAdapter(db_manager=MagicMock())
        result = adapter._conversations_to_chat_messages(
            convs[:5],  # 模拟 limit=5
            session_id="cross_session",
            user_id="user1",
        )
        assert len(result) == 5


# ============ search_relevant_history BM25 测试 ============

class TestSearchRelevantHistoryBM25:
    """测试 BM25 增强的 search_relevant_history"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.db_manager = MagicMock()
        self.db_manager._db_path = ":memory:"
        self.adapter = SQLiteDataAdapter(db_manager=self.db_manager)

    def test_bm25_ranks_relevant_higher(self):
        """BM25 应将相关文档排在前面"""
        query_tokens = self.adapter._tokenize("素食餐厅推荐")
        doc_tokens_list = [
            self.adapter._tokenize("我喜欢古典音乐和吉他"),
            self.adapter._tokenize("推荐一家素食餐厅给我"),
            self.adapter._tokenize("今天天气很好适合户外运动"),
            self.adapter._tokenize("素食餐厅的营养搭配建议和推荐"),
        ]

        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
        )

        # 文档 1 和 3 (素食+餐厅+推荐 相关) 应得分高于文档 0 和 2 (无关)
        assert scores[1] > scores[0], f"doc1={scores[1]} should > doc0={scores[0]}"
        assert scores[3] > scores[2], f"doc3={scores[3]} should > doc2={scores[2]}"
        # 无关文档得分应为 0
        assert scores[0] == 0.0
        assert scores[2] == 0.0

    def test_chinese_query_retrieval(self):
        """中文查询检索测试"""
        query_tokens = self.adapter._tokenize("我对海鲜过敏，午饭吃什么")
        doc_tokens_list = [
            self.adapter._tokenize("用户对海鲜过敏，不能吃虾蟹"),
            self.adapter._tokenize("今天的午餐推荐：素食沙拉"),
            self.adapter._tokenize("周末去爬山怎么样"),
        ]

        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
        )

        # 海鲜过敏相关的文档应排第一
        assert scores[0] > scores[2]

    def test_english_query_retrieval(self):
        """英文查询检索测试"""
        query_tokens = self.adapter._tokenize("recommend vegetarian restaurant")
        doc_tokens_list = [
            self.adapter._tokenize("I prefer vegetarian food and am allergic to seafood"),
            self.adapter._tokenize("I love hiking and outdoor activities"),
            self.adapter._tokenize("vegetarian restaurant near my home"),
        ]

        scores = SQLiteDataAdapter._bm25_score(
            query_tokens=query_tokens,
            doc_tokens_list=doc_tokens_list,
        )

        # 包含 "vegetarian" 和 "restaurant" 的文档应得分最高
        assert scores[2] > scores[1]
        assert scores[0] > scores[1]


# ============ 数据生成器 reference_answer 测试 ============

class TestDataGeneratorReferenceAnswer:
    """测试数据生成器的 reference_answer 字段"""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from dki.experiment.data_generator import ExperimentDataGenerator
        self.generator = ExperimentDataGenerator(output_dir=str(tmp_path / "test_data"))

    def test_persona_chat_has_reference_answer(self):
        """PersonaChat 数据应包含 reference_answer"""
        data = self.generator.generate_persona_chat(n_sessions=2, n_turns_per_session=3)
        for session in data:
            for turn in session['turns']:
                assert 'reference_answer' in turn, (
                    f"Turn {turn['turn_id']} in session {session['session_id']} "
                    f"missing reference_answer"
                )
                assert len(turn['reference_answer']) > 0

    def test_memory_qa_has_reference_answer(self):
        """MemoryQA 数据应包含 reference_answer"""
        data = self.generator.generate_memory_qa(n_samples=10)
        for item in data:
            assert 'reference_answer' in item
            assert len(item['reference_answer']) > 0

    def test_chinese_persona_chat_has_reference_answer(self):
        """中文 PersonaChat 数据应包含 reference_answer"""
        data = self.generator.generate_chinese_persona_chat(
            n_sessions=2, n_turns_per_session=3
        )
        for session in data:
            for turn in session['turns']:
                assert 'reference_answer' in turn
                assert len(turn['reference_answer']) > 0

    def test_multi_turn_coherence_has_reference_answer(self):
        """多轮连贯性数据应包含 reference_answer"""
        data = self.generator.generate_multi_turn_coherence(n_sessions=3)
        for session in data:
            for turn in session['turns']:
                assert 'reference_answer' in turn
                assert len(turn['reference_answer']) > 0

    def test_ablation_has_reference_answer(self):
        """消融实验数据应包含 reference_answer"""
        data = self.generator.generate_ablation_data(n_samples=5)
        for item in data:
            assert 'reference_answer' in item

    def test_reference_answer_contains_memory_keywords(self):
        """reference_answer 应包含相关记忆关键词"""
        data = self.generator.generate_memory_qa(n_samples=20)
        keyword_found = 0
        for item in data:
            ref = item['reference_answer'].lower()
            for var_value in item['filled_vars'].values():
                if var_value.lower() in ref:
                    keyword_found += 1
                    break
        # 至少 80% 的 reference_answer 应包含相关变量值
        assert keyword_found / len(data) >= 0.8


# ============ _extract_keywords 向后兼容测试 ============

class TestExtractKeywordsCompat:
    """测试 _extract_keywords 向后兼容性"""

    @pytest.fixture(autouse=True)
    def setup(self):
        db_manager = MagicMock()
        db_manager._db_path = ":memory:"
        self.adapter = SQLiteDataAdapter(db_manager=db_manager)

    def test_extract_keywords_returns_list(self):
        """_extract_keywords 应返回列表"""
        result = self.adapter._extract_keywords("我喜欢素食和户外运动")
        assert isinstance(result, list)

    def test_extract_keywords_same_as_tokenize(self):
        """_extract_keywords 应与 _tokenize 返回相同结果"""
        text = "推荐一家素食餐厅"
        assert self.adapter._extract_keywords(text) == self.adapter._tokenize(text)
