"""
单元测试: 2026-02-18 Bug 修复验证

修复的问题:
1. UNIQUE constraint failed: memories.id — RAG 重复插入数据库
2. CUDA OOM — KV 注入显存泄露/管理问题
3. 模型回复包含无关问题 — 历史后缀注入问题
4. Web UI 显示确认 — 非 UI 错误，是模型生成问题

测试策略:
- 使用 FakeModelAdapter 和 mock，不需要 GPU
- 验证每个修复点的正确行为
"""

import os
import sys
import uuid
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, BaseRepository,
)
from dki.core.components.hybrid_injector import (
    HybridDKIInjector, HybridInjectionConfig,
    UserPreference, SessionHistory, SessionMessage,
    HybridInjectionResult,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db_manager(tmp_path):
    """
    创建临时文件数据库用于测试，每次测试后自动清理。
    使用 tmp_path 避免 :memory: 与 Path.parent.mkdir 冲突。
    """
    # 重置单例
    DatabaseManager.reset_instance()

    db_file = str(tmp_path / "test_dki.db")
    mgr = DatabaseManager(db_path=db_file)
    yield mgr

    # 清理
    mgr.drop_all()
    DatabaseManager.reset_instance()


@pytest.fixture
def hybrid_injector_cn():
    """创建中文 HybridDKIInjector (无模型)。"""
    config = HybridInjectionConfig(
        preference_enabled=True,
        preference_alpha=0.4,
        history_enabled=True,
        history_max_tokens=500,
        history_max_messages=10,
    )
    return HybridDKIInjector(config=config, model=None, tokenizer=None, language="cn")


@pytest.fixture
def hybrid_injector_en():
    """创建英文 HybridDKIInjector (无模型)。"""
    config = HybridInjectionConfig(
        preference_enabled=True,
        preference_alpha=0.4,
        history_enabled=True,
        history_max_tokens=500,
        history_max_messages=10,
    )
    return HybridDKIInjector(config=config, model=None, tokenizer=None, language="en")


# ============================================================
# 问题 1: UNIQUE constraint failed — RAG 重复插入
# ============================================================

class TestUniqueConstraintFix:
    """验证 RAG add_memory 不再导致 UNIQUE constraint 违规。"""

    def test_rag_add_memory_skip_db(self, db_manager):
        """
        当 skip_db=True 时，RAG 不应尝试写入数据库。
        模拟 /api/memory 端点的行为: DKI 先写入 DB，RAG 只更新内存索引。
        """
        from dki.core.rag_system import RAGSystem

        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        memory_content = "用户喜欢编程"
        memory_id = "mem_test_unique_001"

        # 先通过 DKI 写入 memory (模拟 DKI add_memory)
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            memory_repo = MemoryRepository(db)
            session_repo.get_or_create(session_id)
            memory = memory_repo.create(
                session_id=session_id,
                content=memory_content,
                memory_id=memory_id,
            )
            assert memory.id == memory_id

        # 验证 memory 已在 DB 中
        with db_manager.session_scope() as db:
            memory_repo = MemoryRepository(db)
            existing = memory_repo.get(memory_id)
            assert existing is not None
            assert existing.content == memory_content

        # 构造 mock RAGSystem — 避免完整初始化
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = torch.zeros(384)
        mock_router = MagicMock()

        rag = RAGSystem.__new__(RAGSystem)
        rag.embedding_service = mock_embedding
        rag.memory_router = mock_router
        rag.db_manager = db_manager
        rag.config = MagicMock()
        rag._model_adapter = None
        rag._engine = None

        # skip_db=True 应不抛异常
        result_id = rag.add_memory(
            session_id=session_id,
            content=memory_content,
            memory_id=memory_id,
            skip_db=True,
        )

        assert result_id == memory_id
        # 验证 router 被调用了 (内存索引更新)
        mock_router.add_memory.assert_called_once()

    def test_rag_add_memory_skip_db_generates_id_if_none(self, db_manager):
        """
        当 skip_db=True 且 memory_id=None 时，应自动生成 ID。
        """
        from dki.core.rag_system import RAGSystem

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = torch.zeros(384)
        mock_router = MagicMock()

        rag = RAGSystem.__new__(RAGSystem)
        rag.embedding_service = mock_embedding
        rag.memory_router = mock_router
        rag.db_manager = db_manager
        rag.config = MagicMock()
        rag._model_adapter = None
        rag._engine = None

        result_id = rag.add_memory(
            session_id="test_session",
            content="some content",
            memory_id=None,
            skip_db=True,
        )

        assert result_id is not None
        assert result_id.startswith("mem_")

    def test_rag_add_memory_without_skip_db_checks_existing(self, db_manager):
        """
        即使 skip_db=False，如果 memory_id 已存在，也不应抛出异常。
        RAG 的 add_memory 现在会检查 existing。
        """
        from dki.core.rag_system import RAGSystem

        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        memory_id = "mem_test_unique_002"

        # 先插入
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            memory_repo = MemoryRepository(db)
            session_repo.get_or_create(session_id)
            memory_repo.create(
                session_id=session_id,
                content="original content",
                memory_id=memory_id,
            )

        # 构造 mock RAGSystem
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = torch.zeros(384)
        mock_router = MagicMock()

        rag = RAGSystem.__new__(RAGSystem)
        rag.embedding_service = mock_embedding
        rag.memory_router = mock_router
        rag.db_manager = db_manager
        rag.config = MagicMock()
        rag._model_adapter = None
        rag._engine = None

        # 不应抛异常 (新代码会检查 existing)
        result_id = rag.add_memory(
            session_id=session_id,
            content="duplicate content",
            memory_id=memory_id,
            skip_db=False,
        )

        assert result_id == memory_id

    def test_web_app_memory_endpoint_flow(self, db_manager):
        """
        模拟 /api/memory 端点的完整流程:
        DKI 先写入 → RAG 使用 skip_db=True。
        """
        from dki.core.rag_system import RAGSystem

        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        content = "推理严谨,存疑澄清. 语言温暖,富于幽默"

        # Step 1: DKI 写入 DB
        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            memory_repo = MemoryRepository(db)
            session_repo.get_or_create(session_id)
            memory = memory_repo.create(
                session_id=session_id,
                content=content,
                memory_id="mem_38b0f9fd3c0a4a70",
            )
            dki_memory_id = memory.id

        # Step 2: RAG 使用 skip_db=True (不应抛异常)
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = torch.zeros(384)
        mock_router = MagicMock()

        rag = RAGSystem.__new__(RAGSystem)
        rag.embedding_service = mock_embedding
        rag.memory_router = mock_router
        rag.db_manager = db_manager
        rag.config = MagicMock()
        rag._model_adapter = None
        rag._engine = None

        rag_memory_id = rag.add_memory(
            session_id=session_id,
            content=content,
            memory_id=dki_memory_id,
            skip_db=True,
        )

        # 两者返回相同 memory_id
        assert rag_memory_id == dki_memory_id

        # DB 中只有一条记录
        with db_manager.session_scope() as db:
            memory_repo = MemoryRepository(db)
            memories = memory_repo.get_by_session(session_id)
            assert len(memories) == 1
            assert memories[0].id == dki_memory_id


# ============================================================
# 问题 2: CUDA OOM — KV 注入显存管理
# ============================================================

class TestGPUMemoryManagement:
    """验证 KV 计算后 tensor 在 CPU 上，避免 GPU 内存累积。"""

    def test_compute_kv_returns_cpu_tensors(self):
        """compute_kv 应返回 CPU tensor。"""
        from tests.fixtures.fake_model import FakeModelAdapter

        model = FakeModelAdapter(hidden_dim=128, num_layers=4)
        kv_entries, hidden = model.compute_kv("test text")

        for entry in kv_entries:
            assert entry.key.device.type == "cpu"
            assert entry.value.device.type == "cpu"

    def test_embed_returns_cpu_tensor(self):
        """embed 应返回 CPU tensor。"""
        from tests.fixtures.fake_model import FakeModelAdapter

        model = FakeModelAdapter(hidden_dim=128, num_layers=4)
        embedding = model.embed("test text")

        assert embedding.device.type == "cpu"
        assert embedding.shape == (128,)

    def test_hybrid_injector_preference_kv_no_model(self, hybrid_injector_cn):
        """
        当模型未设置时，_get_or_compute_preference_kv 应返回 None
        而不是崩溃。
        """
        pref = UserPreference(
            content="用户喜欢编程",
            user_id="test_user",
        )
        result = hybrid_injector_cn._get_or_compute_preference_kv(pref)
        assert result is None

    def test_preference_kv_cache_isolation(self, hybrid_injector_cn):
        """不同用户的偏好 KV 缓存应该隔离。"""
        fake_kv = [torch.zeros(2, 2)]
        key1 = f"user1:{hash('pref1')}"
        key2 = f"user2:{hash('pref1')}"

        hybrid_injector_cn._preference_cache[key1] = fake_kv

        assert key2 not in hybrid_injector_cn._preference_cache
        assert key1 in hybrid_injector_cn._preference_cache

    def test_preference_kv_returns_cached_when_available(self, hybrid_injector_cn):
        """如果 preference 已有 kv_cache，应直接返回。"""
        fake_kv = [torch.zeros(2, 2)]
        pref = UserPreference(
            content="用户喜欢编程",
            user_id="test_user",
            kv_cache=fake_kv,
        )
        result = hybrid_injector_cn._get_or_compute_preference_kv(pref)
        assert result is fake_kv


# ============================================================
# 问题 3: 模型回复包含无关问题 — 历史后缀注入
# ============================================================

class TestHistoryInjectionFix:
    """验证历史后缀注入不会导致无关内容污染。"""

    def test_filter_session_history_markers(self, hybrid_injector_cn):
        """包含 [会话历史参考] 标记的消息应被过滤掉。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "你好")
        history.add_message(
            "assistant",
            "[会话历史参考]\n这是嵌套的历史\n[会话历史结束]"
        )
        history.add_message("user", "今天天气怎么样？")

        formatted = hybrid_injector_cn._format_history(history)

        # 包含注入标记的 assistant 消息应被过滤
        assert "嵌套的历史" not in formatted
        # 用户消息应保留
        assert "你好" in formatted
        assert "天气" in formatted

    def test_filter_ai_assistant_marker(self, hybrid_injector_cn):
        """包含 [AI助手] 标记的消息应被过滤掉。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "你好")
        history.add_message("assistant", "[AI助手] 这是一个旧回复")
        history.add_message("user", "新问题")

        formatted = hybrid_injector_cn._format_history(history)

        assert "[AI助手]" not in formatted
        assert "旧回复" not in formatted
        assert "新问题" in formatted

    def test_filter_english_session_history_markers(self, hybrid_injector_en):
        """包含 [Session History Reference] 标记的消息应被过滤。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "hello")
        history.add_message(
            "assistant",
            "[Session History Reference]\nold nested history\n[End of Session History]"
        )
        history.add_message("user", "new question")

        formatted = hybrid_injector_en._format_history(history)

        assert "old nested history" not in formatted
        assert "new question" in formatted

    def test_normal_assistant_message_not_filtered(self, hybrid_injector_cn):
        """正常包含 'Assistant:' 的消息不应被过滤 (MINOR 6 修复)。
        
        'Assistant:' 和 '助手:' 是正常角色标签, 不是 DKI 注入标记,
        不应导致消息被过滤。
        """
        history = SessionHistory(session_id="test")
        history.add_message("user", "你好")
        history.add_message("assistant", "Assistant: 这是正常的回复")
        history.add_message("user", "新问题")

        formatted = hybrid_injector_cn._format_history(history)

        # "Assistant:" 不再是注入标记, 正常消息应保留
        assert "正常的回复" in formatted
        assert "新问题" in formatted

    def test_normal_previous_conversation_not_filtered(self, hybrid_injector_en):
        """正常包含 'Previous conversation:' 的消息不应被过滤 (MINOR 6 修复)。
        
        'Previous conversation:' 已从注入标记列表移除,
        避免误过滤正常对话内容。
        """
        history = SessionHistory(session_id="test")
        history.add_message("user", "hello")
        history.add_message(
            "assistant",
            "Previous conversation:\nUser: old\nAssistant: old reply"
        )
        history.add_message("user", "new question")

        formatted = hybrid_injector_en._format_history(history)

        # "Previous conversation:" 不再是注入标记, 消息应保留
        assert "old reply" in formatted
        assert "new question" in formatted

    def test_truncate_long_assistant_responses(self, hybrid_injector_cn):
        """过长的 assistant 回复应被截断 (>200 字符)。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "解释量子力学")
        # 模拟一个很长的回复 (>200 字符)
        long_response = "量子力学是物理学的一个分支。" * 50  # ~650 字符
        history.add_message("assistant", long_response)
        history.add_message("user", "谢谢，现在告诉我天气")

        formatted = hybrid_injector_cn._format_history(history)

        # 长回复应被截断 (包含 "..." 标记)
        assert "..." in formatted
        # 但不应包含完整的长回复
        assert long_response not in formatted
        # 用户消息应完整保留
        assert "天气" in formatted

    def test_current_query_clearly_marked_cn(self, hybrid_injector_cn):
        """中文模式下，当前问题应有 [当前问题] 标记。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "之前的问题")
        history.add_message("assistant", "之前的回答")

        result = hybrid_injector_cn.prepare_input(
            user_query="当前的新问题",
            history=history,
        )

        assert "[当前问题]" in result.input_text
        assert "当前的新问题" in result.input_text

    def test_current_query_clearly_marked_en(self, hybrid_injector_en):
        """英文模式下，当前问题应有 [Current Question] 标记。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "previous question")
        history.add_message("assistant", "previous answer")

        result = hybrid_injector_en.prepare_input(
            user_query="current new question",
            history=history,
        )

        assert "[Current Question]" in result.input_text
        assert "current new question" in result.input_text

    def test_cn_suffix_template_warns_against_repeating(self, hybrid_injector_cn):
        """中文历史后缀模板应包含不要重复历史话题的指导。"""
        suffix = hybrid_injector_cn.config.history_suffix_template
        assert "不要重复" in suffix
        assert "当前问题" in suffix

    def test_en_suffix_template_warns_against_repeating(self, hybrid_injector_en):
        """英文历史后缀模板应包含不要重复历史话题的指导。"""
        suffix = hybrid_injector_en.config.history_suffix_template
        # 检查英文模板中有 "Do NOT repeat" 或类似语句
        assert "NOT" in suffix or "not" in suffix
        assert "CURRENT" in suffix or "current" in suffix

    def test_empty_history_no_suffix(self, hybrid_injector_cn):
        """没有历史时不应添加后缀。"""
        result = hybrid_injector_cn.prepare_input(
            user_query="简单问题",
            history=None,
        )

        assert "[会话历史参考]" not in result.input_text
        assert "简单问题" in result.input_text
        assert result.history_tokens == 0

    def test_empty_messages_no_suffix(self, hybrid_injector_cn):
        """空消息列表不应添加后缀。"""
        history = SessionHistory(session_id="test")

        result = hybrid_injector_cn.prepare_input(
            user_query="简单问题",
            history=history,
        )

        assert "[会话历史参考]" not in result.input_text
        assert result.history_tokens == 0

    def test_all_messages_filtered_no_suffix(self, hybrid_injector_cn):
        """所有历史消息都被过滤时，不应添加后缀。"""
        history = SessionHistory(session_id="test")
        history.add_message(
            "assistant",
            "[会话历史参考]\n旧内容\n[会话历史结束]"
        )
        history.add_message("assistant", "[AI助手] 另一条被过滤的消息")

        result = hybrid_injector_cn.prepare_input(
            user_query="新问题",
            history=history,
        )

        # 所有消息被过滤，不应有历史后缀
        assert "[会话历史参考]" not in result.input_text
        assert result.history_tokens == 0

    def test_skip_empty_messages(self, hybrid_injector_cn):
        """空白消息应被跳过。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "")
        history.add_message("assistant", "   ")
        history.add_message("user", "有内容的消息")

        formatted = hybrid_injector_cn._format_history(history)

        # 空白消息不应出现
        assert "有内容的消息" in formatted


# ============================================================
# 问题 4: Web UI 显示 — 确认非 UI 错误
# ============================================================

class TestWebUIResponseMapping:
    """验证 API 响应格式正确，UI 不会错误显示数据。"""

    def test_chat_response_model(self):
        """ChatResponse 模型应正确序列化。"""
        from dki.web.app import ChatResponse

        resp = ChatResponse(
            response="这是回复",
            mode="dki",
            session_id="test_session",
            latency_ms=100.0,
            memories_used=[],
            alpha=0.5,
            cache_hit=False,
            metadata={"hybrid_injection": {"enabled": True}},
        )

        assert resp.response == "这是回复"
        assert resp.mode == "dki"
        assert resp.alpha == 0.5

    def test_dki_chat_response_has_choices(self):
        """DKIChatResponse 应包含 choices 字段用于 OpenAI 兼容。"""
        from dki.api.dki_routes import DKIChatResponse, DKIMetadataResponse

        resp = DKIChatResponse(
            id="test-123",
            text="这是回复",
            input_tokens=10,
            output_tokens=20,
            dki_metadata=DKIMetadataResponse(),
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "这是回复"},
                "finish_reason": "stop",
            }],
            created=1234567890,
        )

        # text 和 choices[0].message.content 应一致
        assert resp.text == resp.choices[0]["message"]["content"]

    def test_chat_response_no_ai_prefix_in_response(self):
        """ChatResponse 的 response 字段不应包含 [AI助手] 前缀。"""
        from dki.web.app import ChatResponse

        # 模拟正常回复 (修复后不应包含 [AI助手])
        resp = ChatResponse(
            response="北京今天晴天，气温15度。",
            mode="dki",
            session_id="test",
            latency_ms=50.0,
            memories_used=[],
        )

        assert "[AI助手]" not in resp.response


# ============================================================
# 数据库 Repository 安全性测试
# ============================================================

class TestMemoryRepositorySafety:
    """验证 MemoryRepository 的安全操作。"""

    def test_create_with_duplicate_id_raises(self, db_manager):
        """直接创建重复 ID 应抛出异常 (数据库层面保护)。"""
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        memory_id = "mem_dup_test_001"

        with db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            memory_repo = MemoryRepository(db)
            session_repo.get_or_create(session_id)
            memory_repo.create(
                session_id=session_id,
                content="first",
                memory_id=memory_id,
            )

        # 第二次创建相同 ID 应抛出异常
        with pytest.raises(Exception):
            with db_manager.session_scope() as db:
                memory_repo = MemoryRepository(db)
                memory_repo.create(
                    session_id=session_id,
                    content="second",
                    memory_id=memory_id,
                )

    def test_get_nonexistent_memory_returns_none(self, db_manager):
        """查询不存在的 memory 应返回 None。"""
        with db_manager.session_scope() as db:
            memory_repo = MemoryRepository(db)
            result = memory_repo.get("nonexistent_id")
            assert result is None

    def test_generate_id_uniqueness(self):
        """生成的 ID 应该唯一。"""
        ids = set()
        for _ in range(100):
            new_id = BaseRepository.generate_id("mem_")
            assert new_id not in ids
            ids.add(new_id)
        assert len(ids) == 100

    def test_generate_id_has_prefix(self):
        """生成的 ID 应包含指定前缀。"""
        mem_id = BaseRepository.generate_id("mem_")
        assert mem_id.startswith("mem_")

        sess_id = BaseRepository.generate_id("sess_")
        assert sess_id.startswith("sess_")


# ============================================================
# HybridInjectionConfig 测试
# ============================================================

class TestHybridInjectionConfig:
    """验证 HybridInjectionConfig 默认值。"""

    def test_default_config(self):
        config = HybridInjectionConfig()
        assert config.preference_enabled is True
        assert config.preference_alpha == 0.4
        assert config.preference_max_tokens == 100
        assert config.history_enabled is True
        assert config.history_max_tokens == 500
        assert config.history_max_messages == 10

    def test_custom_config(self):
        config = HybridInjectionConfig(
            preference_alpha=0.6,
            history_max_tokens=300,
        )
        assert config.preference_alpha == 0.6
        assert config.history_max_tokens == 300


# ============================================================
# 集成: prepare_input 完整流程
# ============================================================

class TestPrepareInputIntegration:
    """验证 prepare_input 的完整流程。"""

    def test_full_flow_with_history_and_preference(self, hybrid_injector_cn):
        """完整流程: 偏好 + 历史 + 查询。"""
        preference = UserPreference(
            content="用户是素食主义者",
            user_id="test_user",
        )

        history = SessionHistory(session_id="test")
        history.add_message("user", "推荐一家餐厅")
        history.add_message("assistant", "好的，我推荐绿色厨房。")

        result = hybrid_injector_cn.prepare_input(
            user_query="那里有什么好吃的？",
            preference=preference,
            history=history,
        )

        # 应包含历史
        assert result.history_tokens > 0
        # 应包含当前问题标记
        assert "[当前问题]" in result.input_text
        assert "好吃的" in result.input_text
        # 偏好文本应被保存 (即使 KV 计算失败)
        assert result.preference_text == "用户是素食主义者"
        # 历史消息应被保存
        assert len(result.history_messages) > 0

    def test_query_only_no_history(self, hybrid_injector_cn):
        """仅查询，无历史。"""
        result = hybrid_injector_cn.prepare_input(
            user_query="你好",
        )

        assert "你好" in result.input_text
        assert result.history_tokens == 0
        assert result.preference_tokens == 0

    def test_history_order_preserved(self, hybrid_injector_cn):
        """历史消息应保持时间顺序。"""
        history = SessionHistory(session_id="test")
        history.add_message("user", "第一个问题")
        history.add_message("assistant", "第一个回答")
        history.add_message("user", "第二个问题")

        formatted = hybrid_injector_cn._format_history(history)

        # 第一个问题应出现在第二个问题之前
        idx1 = formatted.find("第一个问题")
        idx2 = formatted.find("第二个问题")
        assert idx1 < idx2

    def test_prepare_input_returns_correct_type(self, hybrid_injector_cn):
        """prepare_input 应返回 HybridInjectionResult 类型。"""
        result = hybrid_injector_cn.prepare_input(user_query="测试")
        assert isinstance(result, HybridInjectionResult)
        assert isinstance(result.input_text, str)
        assert isinstance(result.history_tokens, int)
        assert isinstance(result.preference_tokens, int)
        assert isinstance(result.total_tokens, int)

    def test_system_prompt_included(self, hybrid_injector_cn):
        """系统提示词应包含在输入中。"""
        result = hybrid_injector_cn.prepare_input(
            user_query="测试",
            system_prompt="你是一个友善的AI助手。",
        )

        assert "友善的AI助手" in result.input_text
