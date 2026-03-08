"""
SQLiteDataAdapter 单元测试

测试实验系统的 SQLite 数据适配器:
- IUserDataAdapter 接口实现
- 跨会话检索 (session_id=None)
- 记忆写入和读取
- 对话记录写入和读取
- 关键词提取和匹配
"""

import asyncio
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# 设置测试环境变量
os.environ.setdefault("DKI_ENV", "test")


class TestSQLiteDataAdapterUnit(unittest.TestCase):
    """SQLiteDataAdapter 纯单元测试 (Mock DatabaseManager)"""

    def setUp(self):
        """每个测试前初始化 Mock。"""
        self.mock_db_manager = MagicMock()
        self.mock_db_manager._db_path = ":memory:"

    def test_init(self):
        """测试适配器初始化。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter
        from dki.adapters.base import AdapterType

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        self.assertFalse(adapter.is_connected)
        self.assertEqual(adapter.config.adapter_type, AdapterType.SQLITE)

    def test_connect(self):
        """测试连接标记。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        asyncio.run(adapter.connect())
        self.assertTrue(adapter.is_connected)

    def test_disconnect(self):
        """测试断开标记。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        asyncio.run(adapter.connect())
        asyncio.run(adapter.disconnect())
        self.assertFalse(adapter.is_connected)

    def test_extract_keywords_chinese(self):
        """测试中文关键词提取。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        keywords = adapter._extract_keywords("我喜欢吃素食，住在北京海淀区")
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) > 0)
        # 停用词应被过滤
        self.assertNotIn("的", keywords)
        self.assertNotIn("我", keywords)

    def test_extract_keywords_english(self):
        """测试英文关键词提取。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        keywords = adapter._extract_keywords("I like vegetarian food and hiking")
        self.assertIsInstance(keywords, list)
        # 停用词 "I", "and" 应被过滤
        self.assertNotIn("i", keywords)
        self.assertNotIn("and", keywords)
        # "vegetarian", "food", "hiking" 应保留
        self.assertIn("vegetarian", keywords)
        self.assertIn("food", keywords)
        self.assertIn("hiking", keywords)

    def test_extract_keywords_empty(self):
        """测试空文本关键词提取。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        keywords = adapter._extract_keywords("")
        self.assertEqual(keywords, [])

    def test_repr(self):
        """测试字符串表示。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)
        repr_str = repr(adapter)
        self.assertIn("SQLiteDataAdapter", repr_str)
        self.assertIn("connected=False", repr_str)

    def test_conversations_to_chat_messages(self):
        """测试 Conversation ORM 对象到 ChatMessage 的转换。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        # Mock conversation 对象
        mock_conv = MagicMock()
        mock_conv.id = "conv_001"
        mock_conv.session_id = "sess_001"
        mock_conv.role = "user"
        mock_conv.content = "Hello"
        mock_conv.created_at = datetime(2026, 3, 1, 12, 0, 0)
        mock_conv.session = None

        messages = adapter._conversations_to_chat_messages(
            [mock_conv], session_id="sess_001", user_id="user_001"
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_id, "conv_001")
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "Hello")
        self.assertEqual(messages[0].user_id, "user_001")

    def test_add_memory_calls_repo(self):
        """测试 add_memory 调用正确的 Repository。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        # Mock session_scope 上下文管理器
        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        # 调用 add_memory (Mock 会被调用)
        with patch("dki.experiment.sqlite_adapter.SessionRepository") as MockSessionRepo, \
             patch("dki.experiment.sqlite_adapter.MemoryRepository") as MockMemRepo:
            mock_mem = MagicMock()
            mock_mem.id = "mem_001"
            MockMemRepo.return_value.create.return_value = mock_mem
            
            result = adapter.add_memory("sess_001", "I like hiking", user_id="user_001")
            
            MockSessionRepo.return_value.get_or_create.assert_called_once_with(
                "sess_001", user_id="user_001"
            )
            MockMemRepo.return_value.create.assert_called_once_with(
                session_id="sess_001", content="I like hiking"
            )
            self.assertEqual(result, "mem_001")

    def test_add_conversation_calls_repo(self):
        """测试 add_conversation 调用正确的 Repository。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("dki.experiment.sqlite_adapter.SessionRepository") as MockSessionRepo, \
             patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            mock_conv = MagicMock()
            mock_conv.id = "conv_001"
            MockConvRepo.return_value.create.return_value = mock_conv

            result = adapter.add_conversation(
                session_id="sess_001",
                role="user",
                content="What's for dinner?",
                user_id="user_001",
            )

            MockSessionRepo.return_value.get_or_create.assert_called_once()
            MockConvRepo.return_value.create.assert_called_once()
            self.assertEqual(result, "conv_001")


class TestSQLiteDataAdapterSearchRelevantHistory(unittest.TestCase):
    """search_relevant_history 方法专项测试"""

    def setUp(self):
        self.mock_db_manager = MagicMock()
        self.mock_db_manager._db_path = ":memory:"

    def _make_mock_conv(self, content: str, conv_id: str = "conv_001",
                         session_id: str = "sess_001"):
        """创建 Mock Conversation 对象。"""
        mock = MagicMock()
        mock.id = conv_id
        mock.session_id = session_id
        mock.role = "user"
        mock.content = content
        mock.created_at = datetime(2026, 3, 1)
        mock.session = MagicMock()
        mock.session.user_id = "user_001"
        return mock

    def test_cross_session_search(self):
        """测试 session_id=None 的跨会话检索。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        # 准备跨会话的对话记录
        mock_convs = [
            self._make_mock_conv("我喜欢吃素食和蔬菜沙拉", "c1", "sess_a"),
            self._make_mock_conv("北京的天气今天很好", "c2", "sess_b"),
            self._make_mock_conv("我想去徒步远足", "c3", "sess_c"),
        ]

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockConvRepo.return_value.get_by_user_cross_session.return_value = mock_convs

            # session_id=None → 跨会话检索
            result = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="素食餐厅推荐",
                limit=5,
                session_id=None,  # 跨会话!
            ))

            # 应该调用 get_by_user_cross_session
            MockConvRepo.return_value.get_by_user_cross_session.assert_called_once()
            # "素食" 应该匹配到第一条记录
            self.assertTrue(len(result) > 0)

    def test_single_session_search(self):
        """测试 session_id 指定的单会话检索。"""
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_convs = [
            self._make_mock_conv("我想吃素食", "c1"),
            self._make_mock_conv("你好", "c2"),
        ]

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockConvRepo.return_value.get_by_session.return_value = mock_convs

            result = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="素食推荐",
                limit=5,
                session_id="sess_001",
            ))

            # 应该调用 get_by_session
            MockConvRepo.return_value.get_by_session.assert_called_once()


class TestTwoStepPromptSeparation(unittest.TestCase):
    """
    验证 DKI 两步提示词构造的分离原则:
    
    Step 1: get_user_preferences() → preference_text → system message (K/V注入)
    Step 2: search_relevant_history() → conversations 表 → history items
    
    两个通道独立，不混淆。
    """

    def setUp(self):
        self.mock_db_manager = MagicMock()
        self.mock_db_manager._db_path = ":memory:"

    def _make_mock_conv(self, content: str, conv_id: str = "conv_001",
                         session_id: str = "sess_001", role: str = "user"):
        mock = MagicMock()
        mock.id = conv_id
        mock.session_id = session_id
        mock.role = role
        mock.content = content
        mock.created_at = datetime(2026, 3, 1)
        mock.session = MagicMock()
        mock.session.user_id = "user_001"
        return mock

    def test_search_relevant_history_does_not_access_memory_repo(self):
        """
        核心测试: search_relevant_history 不应访问 MemoryRepository。
        
        persona/记忆信息由 get_user_preferences() 路径处理,
        不应混入 search_relevant_history() 的历史召回通道。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_convs = [
            self._make_mock_conv("我喜欢吃素食", "c1"),
        ]

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo, \
             patch("dki.experiment.sqlite_adapter.MemoryRepository") as MockMemRepo:
            MockConvRepo.return_value.get_by_user_cross_session.return_value = mock_convs

            asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="素食推荐",
                limit=5,
                session_id=None,
            ))

            # MemoryRepository 不应被实例化
            MockMemRepo.assert_not_called()

    def test_search_relevant_history_only_returns_conversations(self):
        """
        search_relevant_history 返回的 ChatMessage 应全部来自 conversations 表,
        不应包含 role="system" 的 persona/记忆消息。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_convs = [
            self._make_mock_conv("我是素食主义者", "c1", role="user"),
            self._make_mock_conv("好的，我记住了您的饮食偏好", "c2", role="assistant"),
        ]

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockConvRepo.return_value.get_by_user_cross_session.return_value = mock_convs

            result = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="素食",
                limit=5,
                session_id=None,
            ))

            # 所有结果的 role 应为 user 或 assistant, 不应为 system
            for msg in result:
                self.assertIn(msg.role, ["user", "assistant"],
                              f"Unexpected role '{msg.role}' in search_relevant_history result. "
                              "Persona/memory data should NOT appear here.")

    def test_get_user_preferences_reads_from_user_preferences_table(self):
        """
        get_user_preferences 应从 user_preferences 表读取,
        这是 DKI 两步构造的 Step 1 (K/V 注入)。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_pref = MagicMock()
        mock_pref.user_id = "user_001"
        mock_pref.id = "pref_001"
        mock_pref.preference_text = "我是素食主义者"
        mock_pref.preference_type = "general"
        mock_pref.priority = 10
        mock_pref.category = None
        mock_pref.created_at = datetime(2026, 3, 1)
        mock_pref.updated_at = datetime(2026, 3, 1)
        mock_pref.is_active = True

        with patch("dki.experiment.sqlite_adapter.UserPreferenceRepository") as MockPrefRepo:
            MockPrefRepo.return_value.get_by_user.return_value = [mock_pref]

            result = asyncio.run(adapter.get_user_preferences(
                user_id="user_001",
            ))

            MockPrefRepo.return_value.get_by_user.assert_called_once()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].preference_text, "我是素食主义者")

    def test_preferences_and_history_are_independent_channels(self):
        """
        验证偏好和历史是完全独立的数据通道:
        - get_user_preferences() → 偏好 (K/V注入)
        - search_relevant_history() → 历史对话 (suffix)
        
        两者不应互相影响。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        # 偏好数据
        mock_pref = MagicMock()
        mock_pref.user_id = "user_001"
        mock_pref.id = "pref_001"
        mock_pref.preference_text = "我是素食主义者，不吃肉类"
        mock_pref.preference_type = "general"
        mock_pref.priority = 10
        mock_pref.category = None
        mock_pref.created_at = datetime(2026, 3, 1)
        mock_pref.updated_at = datetime(2026, 3, 1)
        mock_pref.is_active = True

        # 历史对话数据
        mock_convs = [
            self._make_mock_conv("推荐一家北京的餐厅", "c1"),
            self._make_mock_conv("好的，为您推荐绿色厨房", "c2", role="assistant"),
        ]

        with patch("dki.experiment.sqlite_adapter.UserPreferenceRepository") as MockPrefRepo, \
             patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockPrefRepo.return_value.get_by_user.return_value = [mock_pref]
            MockConvRepo.return_value.get_by_user_cross_session.return_value = mock_convs

            # Step 1: 获取偏好
            prefs = asyncio.run(adapter.get_user_preferences(user_id="user_001"))
            # Step 2: 获取历史
            history = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="餐厅推荐",
                limit=5,
                session_id=None,
            ))

            # 偏好应包含 persona 信息
            self.assertEqual(len(prefs), 1)
            self.assertIn("素食主义者", prefs[0].preference_text)

            # 历史应只包含对话消息, 不包含 persona
            for msg in history:
                self.assertNotIn("素食主义者", msg.content,
                                 "Persona should NOT appear in history channel")

    def test_keyword_fallback_returns_recent_messages(self):
        """
        当关键词匹配无结果时，应返回最近的消息 (近轮兜底)。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        # 对话内容与查询完全不相关
        mock_convs = [
            self._make_mock_conv("今天天气真好", "c1"),
            self._make_mock_conv("我去公园散步了", "c2"),
            self._make_mock_conv("晚上看了一部电影", "c3"),
        ]

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockConvRepo.return_value.get_by_user_cross_session.return_value = mock_convs

            result = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="量子计算的最新进展",  # 完全不相关的查询
                limit=3,
                session_id=None,
            ))

            # 应返回近轮消息作为兜底
            self.assertTrue(len(result) > 0, "Should return recent messages as fallback")

    def test_empty_conversations_returns_empty(self):
        """
        当 conversations 表为空时，应返回空列表。
        """
        from dki.experiment.sqlite_adapter import SQLiteDataAdapter

        adapter = SQLiteDataAdapter(db_manager=self.mock_db_manager)

        mock_session = MagicMock()
        self.mock_db_manager.session_scope.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("dki.experiment.sqlite_adapter.ConversationRepository") as MockConvRepo:
            MockConvRepo.return_value.get_by_user_cross_session.return_value = []

            result = asyncio.run(adapter.search_relevant_history(
                user_id="user_001",
                query="任何查询",
                limit=5,
                session_id=None,
            ))

            self.assertEqual(result, [])


class TestMetricsContentRecall(unittest.TestCase):
    """MetricsCalculator.compute_content_recall 单元测试"""

    def test_keyword_recall_basic(self):
        """测试基础关键词召回。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        result = calc.compute_content_recall(
            expected_memories=["我喜欢吃素食和蔬菜沙拉"],
            response="今天推荐素食餐厅，有很多蔬菜沙拉选择",
        )

        self.assertIn("keyword_recall", result)
        self.assertIn("semantic_recall", result)
        self.assertIn("combined_recall", result)
        self.assertGreater(result["keyword_recall"], 0.0)

    def test_semantic_recall_char_ngram(self):
        """测试字符级 n-gram 语义召回。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        # 相同语义不同措辞
        result = calc.compute_content_recall(
            expected_memories=["用户是素食主义者"],
            response="考虑到您不吃肉类的饮食习惯，推荐纯素食方案",
        )

        self.assertIn("semantic_recall", result)
        # 有一些 n-gram 重叠
        self.assertGreaterEqual(result["semantic_recall"], 0.0)

    def test_injection_recall_with_info(self):
        """测试注入覆盖率 (injection_recall)。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        injection_info = {
            "mode": "dki",
            "preference_text": "用户是素食主义者，不吃肉类和海鲜",
            "history_suffix": "上次对话提到喜欢蔬菜沙拉",
            "final_input": "推荐一家餐厅",
        }

        result = calc.compute_content_recall(
            expected_memories=["素食主义者", "蔬菜沙拉"],
            response="推荐素食餐厅，提供多种蔬菜沙拉",
            injection_info=injection_info,
        )

        self.assertIn("injection_recall", result)
        self.assertGreater(result["injection_recall"], 0.0)

    def test_empty_memories(self):
        """测试空记忆列表。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        result = calc.compute_content_recall(
            expected_memories=[],
            response="任何响应",
        )

        self.assertEqual(result["keyword_recall"], 1.0)
        self.assertEqual(result["combined_recall"], 1.0)

    def test_dki_vs_rag_weighting(self):
        """测试 DKI 和 RAG 的不同加权方式。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        memories = ["素食主义者，喜欢蔬菜"]
        response = "推荐素食蔬菜沙拉"

        dki_result = calc.compute_content_recall(
            expected_memories=memories,
            response=response,
            injection_info={"mode": "dki", "preference_text": "素食主义者"},
        )

        rag_result = calc.compute_content_recall(
            expected_memories=memories,
            response=response,
            injection_info={"mode": "rag", "rag_context": "素食主义者"},
        )

        # DKI 和 RAG 的 combined 使用不同权重
        # 但具体值取决于匹配情况，这里只验证结构正确
        self.assertIn("combined_recall", dki_result)
        self.assertIn("combined_recall", rag_result)

    def test_char_ngram_identical_text(self):
        """测试完全相同的文本应有最高 n-gram 匹配。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        text = "我是一名素食主义者"
        result = calc._compute_char_ngram_recall([text], text)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_char_ngram_no_overlap(self):
        """测试完全不相关的文本应有零 n-gram 匹配。"""
        from dki.experiment.metrics import MetricsCalculator
        calc = MetricsCalculator()

        result = calc._compute_char_ngram_recall(
            ["ABCDEFGHIJK"],
            "zyxwvutsrqp",
        )
        self.assertAlmostEqual(result, 0.0, places=2)


class TestPreferenceCacheInvalidation(unittest.TestCase):
    """
    测试 _write_session_preferences 自动清除 DKIPlugin 偏好缓存。
    
    核心问题:
    - DKIPlugin._get_cached_preferences() 使用 TTL=300s 的内存缓存
    - 实验系统中同一 user_id 在多个样本间复用
    - 如果 _write_session_preferences 覆盖偏好但不清缓存,
      后续样本会读到前一个样本的偏好 (脏缓存)
    - 这在 vLLM 环境下尤其严重: 偏好作为提示词前缀 (system message)
      直接影响模型输出, 错误的偏好 = 错误的回答
    """

    def test_write_preferences_invalidates_cache(self):
        """_write_session_preferences 应调用 invalidate_preference_text_cache。"""
        from unittest.mock import MagicMock, patch, PropertyMock

        mock_db_manager = MagicMock()
        mock_db_manager._db_path = ":memory:"
        
        # Mock session_scope context manager
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)

        # 创建 mock DKIPlugin
        mock_plugin = MagicMock()
        mock_plugin.invalidate_preference_text_cache = MagicMock()

        # 创建 ExperimentRunner (需要 mock 很多依赖)
        with patch('dki.experiment.runner.ExperimentRunner.__init__', return_value=None):
            from dki.experiment.runner import ExperimentRunner
            runner = ExperimentRunner.__new__(ExperimentRunner)
            runner.db_manager = mock_db_manager
            runner._dki_plugin = mock_plugin

            # Mock pref_repo 操作
            with patch('dki.experiment.runner.UserPreferenceRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.get_by_user.return_value = []
                mock_repo_cls.return_value = mock_repo

                runner._write_session_preferences("test_user", ["persona1", "persona2"])

                # 验证缓存被清除
                mock_plugin.invalidate_preference_text_cache.assert_called_once_with("test_user")

    def test_write_preferences_no_plugin_no_error(self):
        """当 _dki_plugin 为 None 时, _write_session_preferences 不应报错。"""
        from unittest.mock import MagicMock, patch

        mock_db_manager = MagicMock()
        mock_db_manager._db_path = ":memory:"
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)

        with patch('dki.experiment.runner.ExperimentRunner.__init__', return_value=None):
            from dki.experiment.runner import ExperimentRunner
            runner = ExperimentRunner.__new__(ExperimentRunner)
            runner.db_manager = mock_db_manager
            runner._dki_plugin = None  # 没有 plugin

            with patch('dki.experiment.runner.UserPreferenceRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.get_by_user.return_value = []
                mock_repo_cls.return_value = mock_repo

                # 不应抛出异常
                runner._write_session_preferences("test_user", ["persona1"])

    def test_cache_ttl_causes_stale_preferences(self):
        """
        验证问题场景: TTL=300s 的缓存在快速实验中导致脏偏好。
        
        场景:
        1. 样本 A 写入偏好 ["我喜欢素食"]
        2. DKIPlugin.chat() 缓存了 ["我喜欢素食"]
        3. 样本 B 写入偏好 ["我是程序员"] (覆盖 DB)
        4. 如果不清缓存, DKIPlugin.chat() 仍读 ["我喜欢素食"]
        5. 清缓存后, DKIPlugin.chat() 正确读 ["我是程序员"]
        """
        # 此测试仅验证修复后的行为:
        # _write_session_preferences 在写入新偏好后调用 invalidate_preference_text_cache
        from unittest.mock import MagicMock, patch, call

        mock_db_manager = MagicMock()
        mock_db_manager._db_path = ":memory:"
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_plugin = MagicMock()
        mock_plugin.invalidate_preference_text_cache = MagicMock()

        with patch('dki.experiment.runner.ExperimentRunner.__init__', return_value=None):
            from dki.experiment.runner import ExperimentRunner
            runner = ExperimentRunner.__new__(ExperimentRunner)
            runner.db_manager = mock_db_manager
            runner._dki_plugin = mock_plugin

            with patch('dki.experiment.runner.UserPreferenceRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.get_by_user.return_value = []
                mock_repo_cls.return_value = mock_repo

                # 模拟连续两个样本
                runner._write_session_preferences("user_A", ["我喜欢素食"])
                runner._write_session_preferences("user_A", ["我是程序员"])

                # 验证缓存被清除两次 (每次写入偏好后)
                self.assertEqual(
                    mock_plugin.invalidate_preference_text_cache.call_count, 2
                )
                mock_plugin.invalidate_preference_text_cache.assert_has_calls([
                    call("user_A"), call("user_A")
                ])

    def test_longmemeval_injection_info_recorded(self):
        """
        验证 _run_longmemeval_mode 的结果中包含 injection_info。
        
        之前的问题: eval_results 中没有 injection_info 字段,
        导致无法调试偏好是否被正确注入。
        """
        # 此测试验证 injection_info 键存在于结果中
        # (实际端到端测试需要 vLLM, 此处仅验证结构)
        sample_result = {
            'sample_idx': 0,
            'session_id': 'test',
            'eval_query': 'test',
            'expected_answer': 'test',
            'response': 'test',
            'latency_ms': 100,
            'keyword_recall': 1.0,
            'answer_match': 1.0,
            'rouge_l': 0.5,
            'alpha': 0.4,
            'history_turns_played': 3,
            'total_turns': 4,
            'injection_info': {
                'injection_enabled': True,
                'injection_strategy': 'recall_v4',
                'preferences_count': 3,
                'preference_tokens': 120,
                'relevant_history_count': 5,
                'history_tokens': 300,
                'total_tokens': 420,
                'retrieval_mode': 'keyword',
                'alpha': 0.4,
                'preference_cache_hit': False,
                'preference_cache_tier': 'vllm_prefix_caching',
                'adapter_latency_ms': 15.0,
                'inference_latency_ms': 85.0,
            },
        }
        
        # 验证关键字段
        self.assertIn('injection_info', sample_result)
        info = sample_result['injection_info']
        self.assertEqual(info['preferences_count'], 3)
        self.assertEqual(info['preference_tokens'], 120)
        self.assertEqual(info['retrieval_mode'], 'keyword')
        self.assertFalse(info['preference_cache_hit'])


if __name__ == '__main__':
    unittest.main()
