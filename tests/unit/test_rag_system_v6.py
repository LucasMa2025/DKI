"""
Unit tests for RAG System v6.0 enhancements:
- async_chat()
- chat_stream()
- Preference caching (TTL + SingleFlight)
- Structured exception handling (RAGError hierarchy)
- invalidate_preference_cache()
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Dict, Any, List, Optional


# ============================================================
# Mock 依赖
# ============================================================

class FakeModelOutput:
    """模拟 ModelOutput"""
    def __init__(self, text: str, input_tokens: int = 10, output_tokens: int = 20):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeModelAdapter:
    """模拟 BaseModelAdapter"""
    def __init__(self, response_text: str = "This is a test response."):
        self.response_text = response_text
        self.model_name = "test-model"
        self.is_loaded = True
        self.max_model_len = 4096
        self.tokenizer = None
        self._generate_count = 0

    def generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        self._generate_count += 1
        return FakeModelOutput(self.response_text)

    async def async_generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        self._generate_count += 1
        return FakeModelOutput(self.response_text)

    async def async_stream_generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        """模拟流式生成"""
        tokens = self.response_text.split()
        for token in tokens:
            yield token + " "
            await asyncio.sleep(0)  # yield control

    def get_model_info(self):
        return {"model_name": self.model_name, "is_loaded": self.is_loaded}


class FakeMemoryRouter:
    """模拟 MemoryRouter"""
    def __init__(self, memories=None):
        self._memories = memories or []

    def search(self, query, embedding=None, top_k=5, **kwargs):
        return self._memories[:top_k]

    def add_memory(self, **kwargs):
        pass

    def get_stats(self):
        return {"total_memories": len(self._memories)}


class FakeEmbeddingService:
    """模拟 EmbeddingService"""
    def embed(self, text):
        return [0.1] * 128


class FakePreference:
    """模拟用户偏好"""
    def __init__(self, text: str):
        self.preference_text = text


class FakeDBSession:
    """模拟数据库 session"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class FakeDBManager:
    """模拟 DatabaseManager"""
    def __init__(self, preferences=None, conversations=None):
        self._preferences = preferences or []
        self._conversations = conversations or []

    def session_scope(self):
        return FakeDBSession()


# ============================================================
# 测试用的 RAGSystem 子类 (绕过真实初始化)
# ============================================================

def create_test_rag_system(
    model_response: str = "Test response.",
    preferences: Optional[List[str]] = None,
    memories=None,
    preference_cache_ttl: float = 300.0,
):
    """
    创建一个用于测试的 RAGSystem 实例, 绕过真实的 DB/Model 初始化。
    """
    # 延迟导入以避免模块级别的初始化问题
    with patch('dki.core.rag_system.ConfigLoader') as mock_config_loader, \
         patch('dki.core.rag_system.DatabaseManager') as mock_db_manager, \
         patch('dki.core.rag_system.ModelFactory'):

        # 模拟配置
        mock_config = MagicMock()
        mock_config.database.path = ":memory:"
        mock_config.database.echo = False
        mock_config.dki.recall.budget.max_recent_turns = 5
        mock_config.rag = MagicMock()
        mock_config.rag.preference_cache_ttl_seconds = preference_cache_ttl

        mock_config_loader_instance = MagicMock()
        mock_config_loader_instance.config = mock_config
        mock_config_loader.return_value = mock_config_loader_instance

        mock_db_manager.return_value = FakeDBManager(
            preferences=[FakePreference(p) for p in (preferences or [])]
        )

        from dki.core.rag_system import RAGSystem

        rag = RAGSystem(
            model_adapter=FakeModelAdapter(response_text=model_response),
            memory_router=FakeMemoryRouter(memories or []),
            embedding_service=FakeEmbeddingService(),
            preference_cache_ttl=preference_cache_ttl,
        )

        return rag


# ============================================================
# 测试: 偏好缓存
# ============================================================

class TestPreferenceCache:
    """测试偏好缓存 (TTL + SingleFlight)"""

    def test_cache_hit(self):
        """缓存命中时应直接返回, 不查询 DB"""
        rag = create_test_rag_system()

        # 手动填充缓存
        rag._preference_cache["user1"] = ("喜欢辣的食物", time.time())

        result = rag._load_user_preferences("user1")
        assert result == "喜欢辣的食物"
        assert rag._stats["preference_cache_hits"] == 1

    def test_cache_miss_expired(self):
        """缓存过期时应重新查询 DB (DB 不可用时抛出 RAGPreferenceError)"""
        from dki.core.rag_system import RAGPreferenceError

        rag = create_test_rag_system(preference_cache_ttl=0.001)

        # 填充已过期的缓存
        rag._preference_cache["user1"] = ("旧偏好", time.time() - 10)

        # FakeDBSession 没有 query 方法, 会触发 RAGPreferenceError
        with pytest.raises(RAGPreferenceError):
            rag._load_user_preferences("user1")
        assert rag._stats["preference_cache_misses"] >= 1

    def test_cache_empty_user_id(self):
        """空 user_id 应直接返回 None"""
        rag = create_test_rag_system()
        result = rag._load_user_preferences("")
        assert result is None
        result = rag._load_user_preferences(None)
        assert result is None

    def test_invalidate_single_user(self):
        """使单个用户缓存失效"""
        rag = create_test_rag_system()
        rag._preference_cache["user1"] = ("偏好1", time.time())
        rag._preference_cache["user2"] = ("偏好2", time.time())

        rag.invalidate_preference_cache("user1")

        assert "user1" not in rag._preference_cache
        assert "user2" in rag._preference_cache

    def test_invalidate_all(self):
        """清除所有缓存"""
        rag = create_test_rag_system()
        rag._preference_cache["user1"] = ("偏好1", time.time())
        rag._preference_cache["user2"] = ("偏好2", time.time())

        rag.invalidate_preference_cache()

        assert len(rag._preference_cache) == 0


# ============================================================
# 测试: 异步偏好加载 (SingleFlight)
# ============================================================

class TestAsyncPreferenceLoading:
    """测试异步偏好加载 (SingleFlight 防惊群)"""

    @pytest.mark.asyncio
    async def test_async_cache_hit(self):
        """异步加载时缓存命中"""
        rag = create_test_rag_system()
        rag._preference_cache["user1"] = ("异步偏好", time.time())

        result = await rag._load_user_preferences_async("user1")
        assert result == "异步偏好"
        assert rag._stats["preference_cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_async_empty_user_id(self):
        """异步加载空 user_id"""
        rag = create_test_rag_system()
        result = await rag._load_user_preferences_async("")
        assert result is None

    @pytest.mark.asyncio
    async def test_single_flight_prevents_duplicate_calls(self):
        """SingleFlight 应防止并发请求重复查询 DB"""
        rag = create_test_rag_system()

        call_count = 0
        original_sync = rag._load_user_preferences_sync

        def counting_sync(user_id):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)  # 模拟 DB 延迟
            return "偏好数据"

        rag._load_user_preferences_sync = counting_sync

        # 并发发起 5 个请求
        results = await asyncio.gather(
            rag._load_user_preferences_async("user1"),
            rag._load_user_preferences_async("user1"),
            rag._load_user_preferences_async("user1"),
            rag._load_user_preferences_async("user1"),
            rag._load_user_preferences_async("user1"),
        )

        # 所有结果应相同
        for r in results:
            assert r == "偏好数据"

        # DB 只应被调用 1 次 (SingleFlight)
        assert call_count == 1


# ============================================================
# 测试: 结构化异常
# ============================================================

class TestRAGExceptions:
    """测试 RAG 结构化异常层次"""

    def test_rag_error_hierarchy(self):
        """RAGError 应继承自 DKIError"""
        from dki.core.rag_system import RAGError
        from dki.core.exceptions import DKIError

        err = RAGError("test error")
        assert isinstance(err, DKIError)
        assert err.error_code == "RAG_ERROR"
        assert err.retryable is False

    def test_rag_memory_search_error(self):
        """RAGMemorySearchError 应继承自 RAGError"""
        from dki.core.rag_system import RAGMemorySearchError, RAGError

        err = RAGMemorySearchError("search failed")
        assert isinstance(err, RAGError)
        assert err.error_code == "RAG_MEMORY_SEARCH"
        assert err.retryable is True

    def test_rag_preference_error(self):
        """RAGPreferenceError 应继承自 RAGError"""
        from dki.core.rag_system import RAGPreferenceError, RAGError

        err = RAGPreferenceError("pref failed")
        assert isinstance(err, RAGError)
        assert err.error_code == "RAG_PREFERENCE"
        assert err.retryable is False

    def test_rag_prompt_build_error(self):
        """RAGPromptBuildError 应继承自 RAGError"""
        from dki.core.rag_system import RAGPromptBuildError, RAGError

        err = RAGPromptBuildError("prompt build failed")
        assert isinstance(err, RAGError)
        assert err.error_code == "RAG_PROMPT_BUILD"

    def test_rag_error_with_cause(self):
        """RAGError 应支持 cause 属性"""
        from dki.core.rag_system import RAGError

        original = ValueError("original error")
        err = RAGError("wrapped error", cause=original)
        # DKIError 使用 self.cause 而非 __cause__
        assert err.cause == original


# ============================================================
# 测试: async_chat
# ============================================================

class TestAsyncChat:
    """测试异步 chat"""

    @pytest.mark.asyncio
    async def test_async_chat_basic(self):
        """基本异步 chat 应返回 RAGResponse"""
        rag = create_test_rag_system(model_response="Async response here.")

        # Mock async_generate
        rag.model.async_generate = AsyncMock(
            return_value=FakeModelOutput("Async response here.")
        )

        # Mock _build_prompt to return a simple prompt
        rag._build_prompt = MagicMock(return_value=(
            "test prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))

        # Mock DB operations
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)

        # Mock DB session for logging
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        from dki.core.rag_system import RAGResponse
        response = await rag.async_chat(
            query="Hello",
            session_id="test-session",
            user_id="user1",
        )

        assert isinstance(response, RAGResponse)
        assert "Async response" in response.text or response.text != ""

    @pytest.mark.asyncio
    async def test_async_chat_increments_stats(self):
        """async_chat 应增加统计计数"""
        rag = create_test_rag_system()

        rag.model.async_generate = AsyncMock(
            return_value=FakeModelOutput("Response")
        )
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        initial_total = rag._stats["total_requests"]
        initial_async = rag._stats["async_requests"]

        await rag.async_chat(query="Test", session_id="s1")

        assert rag._stats["total_requests"] == initial_total + 1
        assert rag._stats["async_requests"] == initial_async + 1


# ============================================================
# 测试: chat_stream
# ============================================================

class TestChatStream:
    """测试流式 chat"""

    @pytest.mark.asyncio
    async def test_stream_yields_metadata_first(self):
        """流式 chat 应先 yield metadata"""
        rag = create_test_rag_system(model_response="Hello world stream")

        # 设置 async_stream_generate
        async def fake_stream(**kwargs):
            for token in ["Hello ", "world ", "stream"]:
                yield token

        rag.model.async_stream_generate = fake_stream
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        chunks = []
        async for chunk in rag.chat_stream(query="Test", session_id="s1"):
            chunks.append(chunk)

        # 第一个 chunk 应是 metadata
        assert chunks[0]["type"] == "metadata"
        # metadata 字段直接在 chunk 中 (memories_count, preference_injected, history_turns)
        assert "memories_count" in chunks[0] or "preference_injected" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        """流式 chat 应 yield token chunks"""
        rag = create_test_rag_system()

        async def fake_stream(**kwargs):
            for token in ["A ", "B ", "C"]:
                yield token

        rag.model.async_stream_generate = fake_stream
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        token_chunks = []
        async for chunk in rag.chat_stream(query="Test", session_id="s1"):
            if chunk.get("type") == "token":
                token_chunks.append(chunk["content"])

        assert len(token_chunks) == 3
        assert "A " in token_chunks
        assert "B " in token_chunks

    @pytest.mark.asyncio
    async def test_stream_yields_done(self):
        """流式 chat 最后应 yield done"""
        rag = create_test_rag_system()

        async def fake_stream(**kwargs):
            yield "Hello"

        rag.model.async_stream_generate = fake_stream
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        chunks = []
        async for chunk in rag.chat_stream(query="Test", session_id="s1"):
            chunks.append(chunk)

        # 最后一个 chunk 应是 done
        done_chunks = [c for c in chunks if c.get("type") == "done"]
        assert len(done_chunks) == 1
        assert "text" in done_chunks[0]
        assert "latency_ms" in done_chunks[0]

    @pytest.mark.asyncio
    async def test_stream_increments_stats(self):
        """流式 chat 应增加 stream_requests 统计"""
        rag = create_test_rag_system()

        async def fake_stream(**kwargs):
            yield "token"

        rag.model.async_stream_generate = fake_stream
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        initial = rag._stats["stream_requests"]

        async for _ in rag.chat_stream(query="Test", session_id="s1"):
            pass

        assert rag._stats["stream_requests"] == initial + 1

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_chunk(self):
        """流式 chat 出错时应 yield error chunk"""
        rag = create_test_rag_system()

        async def failing_stream(**kwargs):
            raise RuntimeError("Stream generation failed")
            yield  # make it a generator

        rag.model.async_stream_generate = failing_stream
        rag._build_prompt = MagicMock(return_value=(
            "prompt",
            MagicMock(retrieved_context="", history_messages=[], truncated_history=0)
        ))
        rag._get_session_history = MagicMock(return_value=[])
        rag._load_user_preferences_async = AsyncMock(return_value=None)
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        chunks = []
        async for chunk in rag.chat_stream(query="Test", session_id="s1"):
            chunks.append(chunk)

        error_chunks = [c for c in chunks if c.get("type") == "error"]
        assert len(error_chunks) >= 1
        assert "error" in error_chunks[0]


# ============================================================
# 测试: get_stats
# ============================================================

class TestGetStats:
    """测试统计信息"""

    def test_stats_include_preference_cache(self):
        """get_stats 应包含偏好缓存统计"""
        rag = create_test_rag_system()

        # 填充一些缓存
        rag._preference_cache["user1"] = ("偏好1", time.time())
        rag._stats["preference_cache_hits"] = 5
        rag._stats["preference_cache_misses"] = 2

        stats = rag.get_stats()

        assert "preference_cache" in stats
        assert stats["preference_cache"]["size"] == 1
        assert stats["preference_cache"]["hits"] == 5
        assert stats["preference_cache"]["misses"] == 2

    def test_stats_include_request_counts(self):
        """get_stats 应包含请求计数 (嵌套在 requests 下)"""
        rag = create_test_rag_system()
        rag._stats["total_requests"] = 10
        rag._stats["async_requests"] = 3
        rag._stats["stream_requests"] = 2

        stats = rag.get_stats()

        assert "requests" in stats
        assert stats["requests"]["total"] == 10
        assert stats["requests"]["async"] == 3
        assert stats["requests"]["stream"] == 2


# ============================================================
# 测试: 同步 chat (确保不被破坏)
# ============================================================

class TestSyncChat:
    """确保同步 chat 在 v6.0 修改后仍正常工作"""

    def test_sync_chat_basic(self):
        """同步 chat 应正常返回"""
        rag = create_test_rag_system(model_response="Sync response.")

        # Mock DB operations
        rag.db_manager = MagicMock()
        rag.db_manager.session_scope = MagicMock(return_value=FakeDBSession())

        from dki.core.rag_system import RAGResponse
        response = rag.chat(
            query="Hello sync",
            session_id="test-session",
        )

        assert isinstance(response, RAGResponse)
        assert response.text is not None
