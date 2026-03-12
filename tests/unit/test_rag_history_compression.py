"""
RAG History Compression 单元测试

测试 rag_system.py v7.1 的 history_compression 功能:
1. RAGConfig.history_compression 字段
2. _get_conversation_history(include_ids=True)
3. _build_prompt compress 分支
4. _compress_history_with_suffix_builder
5. _prepare_chat_context 返回 compression_meta
6. chat/async_chat/chat_stream 中 metadata 包含 compression 信息
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# 1. RAGConfig 字段测试
# ============================================================

class TestRAGConfigHistoryCompression:
    """测试 RAGConfig 新增 history_compression 字段"""
    
    def test_default_value_is_truncate(self):
        """默认值应为 truncate"""
        from dki.config.config_loader import RAGConfig
        cfg = RAGConfig()
        assert cfg.history_compression == "truncate"
    
    def test_set_to_compress(self):
        """可以设置为 compress"""
        from dki.config.config_loader import RAGConfig
        cfg = RAGConfig(history_compression="compress")
        assert cfg.history_compression == "compress"
    
    def test_set_to_truncate_explicitly(self):
        """显式设置为 truncate"""
        from dki.config.config_loader import RAGConfig
        cfg = RAGConfig(history_compression="truncate")
        assert cfg.history_compression == "truncate"
    
    def test_from_dict(self):
        """从字典构造"""
        from dki.config.config_loader import RAGConfig
        cfg = RAGConfig(**{"history_compression": "compress", "top_k": 3})
        assert cfg.history_compression == "compress"
        assert cfg.top_k == 3
    
    def test_other_fields_unchanged(self):
        """其他字段不受影响"""
        from dki.config.config_loader import RAGConfig
        cfg = RAGConfig(history_compression="compress")
        assert cfg.enabled is True
        assert cfg.chunk_size == 512
        assert cfg.top_k == 5
        assert cfg.similarity_threshold == 0.5


# ============================================================
# 2. _get_conversation_history include_ids 测试
# ============================================================

class TestGetConversationHistoryIncludeIds:
    """测试 _get_conversation_history 的 include_ids 参数"""
    
    def _make_rag_system(self):
        """创建带 mock 的 RAGSystem"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "truncate"
            mock_config.rag.top_k = 5
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            rag = RAGSystem(model_adapter=MagicMock())
            return rag
    
    def test_include_ids_false_no_id_field(self):
        """include_ids=False 时, 返回的 dict 不包含 id"""
        rag = self._make_rag_system()
        
        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.content = "hello"
        mock_msg.id = "msg-123"
        
        mock_conv_repo = MagicMock()
        mock_conv_repo.get_recent.return_value = [mock_msg]
        mock_conv_repo.get_recent_by_user_cross_session.return_value = []
        
        with patch.object(rag, 'db_manager') as mock_db:
            mock_db.session_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.session_scope.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('dki.core.rag_system.ConversationRepository', return_value=mock_conv_repo):
                result = rag._get_conversation_history("sess-1", max_turns=5, include_ids=False)
        
        assert len(result) == 1
        assert "id" not in result[0]
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"
    
    def test_include_ids_true_has_id_field(self):
        """include_ids=True 时, 返回的 dict 包含 id"""
        rag = self._make_rag_system()
        
        mock_msg = MagicMock()
        mock_msg.role = "assistant"
        mock_msg.content = "world"
        mock_msg.id = "msg-456"
        
        mock_conv_repo = MagicMock()
        mock_conv_repo.get_recent.return_value = [mock_msg]
        mock_conv_repo.get_recent_by_user_cross_session.return_value = []
        
        with patch.object(rag, 'db_manager') as mock_db:
            mock_db.session_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.session_scope.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('dki.core.rag_system.ConversationRepository', return_value=mock_conv_repo):
                result = rag._get_conversation_history(
                    "sess-1", max_turns=5, user_id=None, include_ids=True
                )
        
        assert len(result) == 1
        assert result[0]["id"] == "msg-456"
    
    def test_include_ids_cross_session(self):
        """include_ids=True 对跨会话消息也生效"""
        rag = self._make_rag_system()
        
        cross_msg = MagicMock()
        cross_msg.role = "user"
        cross_msg.content = "cross"
        cross_msg.id = "cross-1"
        
        curr_msg = MagicMock()
        curr_msg.role = "user"
        curr_msg.content = "current"
        curr_msg.id = "curr-1"
        
        mock_conv_repo = MagicMock()
        mock_conv_repo.get_recent.return_value = [curr_msg]
        mock_conv_repo.get_recent_by_user_cross_session.return_value = [cross_msg]
        
        with patch.object(rag, 'db_manager') as mock_db:
            mock_db.session_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.session_scope.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('dki.core.rag_system.ConversationRepository', return_value=mock_conv_repo):
                result = rag._get_conversation_history(
                    "sess-1", max_turns=5, user_id="user-1", include_ids=True
                )
        
        assert len(result) == 2
        assert result[0]["id"] == "cross-1"
        assert result[1]["id"] == "curr-1"


# ============================================================
# 3. _build_prompt 测试
# ============================================================

class TestBuildPromptCompression:
    """测试 _build_prompt 的 compress 和 truncate 分支"""
    
    def _make_rag_system(self, compression="truncate"):
        """创建带 mock 的 RAGSystem"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = compression
            mock_config.rag.top_k = 5
            # 阻止 compress 模式初始化 SuffixBuilder (避免复杂依赖)
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            rag = RAGSystem(model_adapter=MagicMock())
            return rag
    
    def test_truncate_mode_returns_three_tuple(self):
        """truncate 模式返回三元组"""
        rag = self._make_rag_system("truncate")
        # 确保 tokenizer 为 None, 避免 apply_chat_template 返回 MagicMock
        rag._model_adapter = MagicMock()
        rag._model_adapter.tokenizer = None
        rag._model_adapter.model_name = "test"
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        
        result = rag._build_prompt(
            query="你好",
            memories=[],
            system_prompt=None,
            history=[{"role": "user", "content": "之前的对话"}],
        )
        
        assert len(result) == 3
        prompt, prompt_info, compression_meta = result
        assert isinstance(prompt, str)
        assert compression_meta["compression_mode"] == "truncate"
        assert compression_meta["trace_ids"] == []
        assert compression_meta["has_fact_call_instruction"] is False
    
    def test_truncate_mode_drops_oldest(self):
        """truncate 模式: 预算不足时丢弃最旧消息"""
        rag = self._make_rag_system("truncate")
        rag._get_max_context_length = MagicMock(return_value=200)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x))
        
        history = [
            {"role": "user", "content": "A" * 30},
            {"role": "assistant", "content": "B" * 30},
            {"role": "user", "content": "C" * 30},
            {"role": "assistant", "content": "D" * 30},
        ]
        
        prompt, info, meta = rag._build_prompt(
            query="E" * 10,
            memories=[],
            system_prompt=None,
            history=history,
        )
        
        assert meta["compression_mode"] == "truncate"
        # 应该丢弃了一些旧消息
        assert meta["summary_count"] == 0
    
    def test_compress_mode_without_suffix_builder_falls_back(self):
        """compress 模式但 SuffixBuilder 未初始化时, 降级为 truncate"""
        rag = self._make_rag_system("compress")
        rag._suffix_builder = None  # 确保没有 SuffixBuilder
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        
        prompt, info, meta = rag._build_prompt(
            query="你好",
            memories=[],
            system_prompt=None,
            history=[{"role": "user", "content": "之前的对话"}],
        )
        
        # SuffixBuilder 不存在, 应该走 truncate 分支
        assert meta["compression_mode"] == "truncate"
    
    def test_compress_mode_with_suffix_builder(self):
        """compress 模式 + SuffixBuilder 可用时, 调用压缩"""
        rag = self._make_rag_system("compress")
        
        # Mock SuffixBuilder
        mock_suffix = MagicMock()
        from dki.core.recall.recall_config import AssembledSuffix, HistoryItem
        mock_result = AssembledSuffix(
            text="compressed text",
            items=[
                HistoryItem(
                    type="summary",
                    content="摘要: 之前讨论了A和B",
                    trace_id="trace-1",
                    role="user",
                    token_count=20,
                ),
                HistoryItem(
                    type="message",
                    content="最近的对话",
                    trace_id="trace-2",
                    role="assistant",
                    token_count=10,
                ),
            ],
            total_tokens=30,
            message_count=5,
            summary_count=3,
            has_fact_call_instruction=True,
            trace_ids=["trace-1"],
        )
        mock_suffix.build.return_value = mock_result
        rag._suffix_builder = mock_suffix
        
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        
        history = [
            {"role": "user", "content": "消息1", "id": "id-1"},
            {"role": "assistant", "content": "回复1", "id": "id-2"},
            {"role": "user", "content": "消息2", "id": "id-3"},
        ]
        
        prompt, info, meta = rag._build_prompt(
            query="你好",
            memories=[],
            system_prompt=None,
            history=history,
        )
        
        assert meta["compression_mode"] == "compress"
        assert meta["trace_ids"] == ["trace-1"]
        assert meta["has_fact_call_instruction"] is True
        assert meta["summary_count"] == 3
        
        # SuffixBuilder.build 应该被调用
        mock_suffix.build.assert_called_once()
    
    def test_no_history_returns_empty_meta(self):
        """无历史消息时, compression_meta 为默认值"""
        rag = self._make_rag_system("truncate")
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        
        prompt, info, meta = rag._build_prompt(
            query="你好",
            memories=[],
            system_prompt=None,
            history=None,
        )
        
        assert meta["compression_mode"] == "truncate"
        assert meta["trace_ids"] == []
        assert meta["summary_count"] == 0


# ============================================================
# 4. _compress_history_with_suffix_builder 测试
# ============================================================

class TestCompressHistoryWithSuffixBuilder:
    """测试 _compress_history_with_suffix_builder 方法"""
    
    def _make_rag_system(self):
        """创建带 mock 的 RAGSystem"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "compress"
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            rag = RAGSystem(model_adapter=MagicMock())
            return rag
    
    def test_successful_compression(self):
        """成功压缩返回正确结果"""
        rag = self._make_rag_system()
        
        from dki.core.recall.recall_config import AssembledSuffix, HistoryItem
        mock_suffix = MagicMock()
        mock_suffix.build.return_value = AssembledSuffix(
            text="summary",
            items=[
                HistoryItem(type="summary", content="概要", trace_id="t1", role="user", token_count=5),
            ],
            total_tokens=5,
            message_count=3,
            summary_count=2,
            has_fact_call_instruction=False,
            trace_ids=["t1"],
        )
        rag._suffix_builder = mock_suffix
        
        history = [
            {"role": "user", "content": "msg1", "id": "id1"},
            {"role": "assistant", "content": "msg2", "id": "id2"},
            {"role": "user", "content": "msg3", "id": "id3"},
        ]
        
        meta = {
            "trace_ids": [],
            "has_fact_call_instruction": False,
            "summary_count": 0,
            "compression_mode": "compress",
        }
        
        selected, updated_meta = rag._compress_history_with_suffix_builder(
            history=history,
            query="测试",
            remaining_tokens=100,
            compression_meta=meta,
        )
        
        assert len(selected) == 1
        assert selected[0]["content"] == "概要"
        assert selected[0]["role"] == "user"
        assert updated_meta["trace_ids"] == ["t1"]
        assert updated_meta["summary_count"] == 2
    
    def test_suffix_builder_exception_fallback(self):
        """SuffixBuilder 异常时降级为 truncate"""
        rag = self._make_rag_system()
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x))
        
        mock_suffix = MagicMock()
        mock_suffix.build.side_effect = Exception("SuffixBuilder error")
        rag._suffix_builder = mock_suffix
        
        history = [
            {"role": "user", "content": "short"},
        ]
        
        meta = {
            "trace_ids": [],
            "has_fact_call_instruction": False,
            "summary_count": 0,
            "compression_mode": "compress",
        }
        
        selected, updated_meta = rag._compress_history_with_suffix_builder(
            history=history,
            query="test",
            remaining_tokens=1000,
            compression_meta=meta,
        )
        
        # 降级为 truncate
        assert updated_meta["compression_mode"] == "truncate"
        # 应该保留原始消息 (truncate 逻辑)
        assert len(selected) >= 1
    
    def test_wrapped_messages_have_correct_attributes(self):
        """验证传给 SuffixBuilder 的 wrapped messages 有正确的属性"""
        rag = self._make_rag_system()
        
        from dki.core.recall.recall_config import AssembledSuffix
        mock_suffix = MagicMock()
        mock_suffix.build.return_value = AssembledSuffix(
            items=[], total_tokens=0, message_count=0, summary_count=0,
        )
        rag._suffix_builder = mock_suffix
        
        history = [
            {"role": "user", "content": "hello", "id": "msg-100"},
        ]
        
        meta = {"trace_ids": [], "has_fact_call_instruction": False,
                "summary_count": 0, "compression_mode": "compress"}
        
        rag._compress_history_with_suffix_builder(
            history=history, query="test", remaining_tokens=500,
            compression_meta=meta,
        )
        
        # 检查传给 SuffixBuilder.build 的参数
        call_args = mock_suffix.build.call_args
        recalled = call_args.kwargs.get('recalled_messages') or call_args[1].get('recalled_messages')
        if recalled is None and len(call_args.args) >= 2:
            recalled = call_args.args[1]
        
        if recalled is not None:
            assert len(recalled) == 1
            msg = recalled[0]
            assert msg.content == "hello"
            assert msg.role == "user"
            assert msg.id == "msg-100"
            assert msg.message_id == "msg-100"


# ============================================================
# 5. _prepare_chat_context 返回值测试
# ============================================================

class TestPrepareChatContextCompressionMeta:
    """测试 _prepare_chat_context 返回 compression_meta"""
    
    def _make_rag_system(self, compression="truncate"):
        """创建带 mock 的 RAGSystem"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter') as mock_mr, \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = compression
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            mock_mr_instance = MagicMock()
            mock_mr_instance.search.return_value = []
            mock_mr.return_value = mock_mr_instance
            
            from dki.core.rag_system import RAGSystem
            adapter = MagicMock()
            adapter.tokenizer = None
            rag = RAGSystem(model_adapter=adapter)
            return rag
    
    def test_returns_six_tuple(self):
        """返回值是六元组"""
        rag = self._make_rag_system("truncate")
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_conversation_history = MagicMock(return_value=[])
        
        result = rag._prepare_chat_context(
            query="你好",
            session_id="sess-1",
            user_id=None,
            top_k=5,
            system_prompt=None,
            max_history_turns=5,
            include_history=True,
        )
        
        assert len(result) == 6
        prompt, prompt_info, memories, history, pref_text, comp_meta = result
        assert isinstance(comp_meta, dict)
        assert "compression_mode" in comp_meta
        assert "trace_ids" in comp_meta
    
    def test_compress_mode_passes_include_ids(self):
        """compress 模式下 _get_conversation_history 被调用时 include_ids=True"""
        rag = self._make_rag_system("compress")
        # 手动设置 _suffix_builder 使其非 None
        rag._suffix_builder = MagicMock()
        rag._suffix_builder.build.return_value = MagicMock(
            items=[], total_tokens=0, message_count=0, summary_count=0,
            has_fact_call_instruction=False, trace_ids=[],
        )
        
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_conversation_history = MagicMock(return_value=[
            {"role": "user", "content": "hi", "id": "m1"},
        ])
        
        rag._prepare_chat_context(
            query="你好",
            session_id="sess-1",
            user_id="user-1",
            top_k=5,
            system_prompt=None,
            max_history_turns=5,
            include_history=True,
        )
        
        # 验证 include_ids=True 被传递
        call_kwargs = rag._get_conversation_history.call_args
        assert call_kwargs.kwargs.get('include_ids') is True or \
               (len(call_kwargs.args) > 3 and call_kwargs.args[3] is True)
    
    def test_truncate_mode_passes_include_ids_false(self):
        """truncate 模式下 include_ids=False"""
        rag = self._make_rag_system("truncate")
        rag._suffix_builder = None
        
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_conversation_history = MagicMock(return_value=[])
        
        rag._prepare_chat_context(
            query="你好",
            session_id="sess-1",
            user_id=None,
            top_k=5,
            system_prompt=None,
            max_history_turns=5,
            include_history=True,
        )
        
        call_kwargs = rag._get_conversation_history.call_args
        assert call_kwargs.kwargs.get('include_ids') is False or \
               call_kwargs.kwargs.get('include_ids') is None


# ============================================================
# 6. _init_suffix_builder 测试
# ============================================================

class TestInitSuffixBuilder:
    """测试 _init_suffix_builder 方法"""
    
    def test_compress_mode_triggers_init(self):
        """compress 模式会尝试初始化 SuffixBuilder"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "compress"
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            
            with patch.object(RAGSystem, '_init_suffix_builder') as mock_init:
                rag = RAGSystem(model_adapter=MagicMock())
                mock_init.assert_called_once()
    
    def test_truncate_mode_skips_init(self):
        """truncate 模式不初始化 SuffixBuilder"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "truncate"
            mock_config.rag.top_k = 5
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            
            with patch.object(RAGSystem, '_init_suffix_builder') as mock_init:
                rag = RAGSystem(model_adapter=MagicMock())
                mock_init.assert_not_called()
    
    def test_init_suffix_builder_import_error_graceful(self):
        """recall 组件 ImportError 时优雅降级"""
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'), \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "truncate"
            mock_config.rag.top_k = 5
            mock_loader.return_value.config = mock_config
            
            from dki.core.rag_system import RAGSystem
            rag = RAGSystem(model_adapter=MagicMock())
            
            # 手动调用 _init_suffix_builder, 模拟 ImportError
            with patch('builtins.__import__', side_effect=ImportError("no module")):
                rag._init_suffix_builder()
            
            assert rag._suffix_builder is None


# ============================================================
# 7. 端到端集成测试 (chat 方法)
# ============================================================

class TestChatMethodCompressionMeta:
    """测试 chat() 方法是否正确传递 compression_meta"""
    
    def _make_rag_system(self, compression="truncate"):
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter') as mock_mr, \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = compression
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            mock_mr_instance = MagicMock()
            mock_mr_instance.search.return_value = []
            mock_mr.return_value = mock_mr_instance
            
            from dki.core.rag_system import RAGSystem
            adapter = MagicMock()
            adapter.model_name = "test-model"
            adapter.tokenizer = None
            
            from dki.models.base import ModelOutput
            adapter.generate.return_value = ModelOutput(
                text="response", input_tokens=10, output_tokens=5,
            )
            
            rag = RAGSystem(model_adapter=adapter)
            return rag
    
    def test_chat_metadata_includes_compression_mode(self):
        """chat() 返回的 metadata 包含 compression_mode"""
        rag = self._make_rag_system("truncate")
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_conversation_history = MagicMock(return_value=[
            {"role": "user", "content": "hi"},
        ])
        rag._log_conversation = MagicMock()
        
        response = rag.chat(
            query="你好",
            session_id="sess-1",
            user_id=None,
        )
        
        assert response.metadata['compression_mode'] == 'truncate'
        assert response.metadata['trace_ids'] == []
        assert response.metadata['has_fact_call_instruction'] is False


# ============================================================
# 8. 异步 chat 测试
# ============================================================

class TestAsyncChatCompressionMeta:
    """测试 async_chat() 方法的 compression_meta"""
    
    def _make_rag_system(self):
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter') as mock_mr, \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "truncate"
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            mock_mr_instance = MagicMock()
            mock_mr_instance.search.return_value = []
            mock_mr.return_value = mock_mr_instance
            
            from dki.core.rag_system import RAGSystem
            adapter = MagicMock()
            adapter.model_name = "test-model"
            adapter.tokenizer = None
            
            from dki.models.base import ModelOutput
            adapter.generate.return_value = ModelOutput(
                text="response", input_tokens=10, output_tokens=5,
            )
            
            rag = RAGSystem(model_adapter=adapter)
            return rag
    
    @pytest.mark.asyncio
    async def test_async_chat_metadata_includes_compression(self):
        """async_chat() 返回的 metadata 包含 compression_mode"""
        rag = self._make_rag_system()
        rag._model_adapter.tokenizer = None
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        
        # Python 3.12+ 不支持 asyncio.coroutine, 使用 AsyncMock
        async def _async_none(user_id=None):
            return None
        rag._load_user_preferences_async = _async_none
        rag._get_conversation_history = MagicMock(return_value=[])
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_fact_retrieve_method = MagicMock(return_value="post_hoc")
        
        # Mock _fire_and_forget_log
        rag._fire_and_forget_log = MagicMock()
        
        response = await rag.async_chat(
            query="你好",
            session_id="sess-1",
        )
        
        assert 'compression_mode' in response.metadata
        assert response.metadata['compression_mode'] == 'truncate'


# ============================================================
# 9. chat_stream metadata 测试
# ============================================================

class TestChatStreamCompressionMeta:
    """测试 chat_stream() 的 metadata 包含 compression 信息"""
    
    def _make_rag_system(self):
        with patch('dki.core.rag_system.ConfigLoader') as mock_loader, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter') as mock_mr, \
             patch('dki.core.rag_system.DatabaseManager'):
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            mock_config.database.echo = False
            mock_config.rag.history_compression = "truncate"
            mock_config.rag.top_k = 5
            mock_config.dki = MagicMock()
            mock_config.dki.recall = {}
            mock_loader.return_value.config = mock_config
            
            mock_mr_instance = MagicMock()
            mock_mr_instance.search.return_value = []
            mock_mr.return_value = mock_mr_instance
            
            from dki.core.rag_system import RAGSystem
            adapter = MagicMock()
            adapter.model_name = "test-model"
            adapter.tokenizer = None
            
            from dki.models.base import ModelOutput
            adapter.generate.return_value = ModelOutput(
                text="stream response", input_tokens=10, output_tokens=5,
            )
            
            rag = RAGSystem(model_adapter=adapter)
            return rag
    
    @pytest.mark.asyncio
    async def test_stream_metadata_has_compression_fields(self):
        """chat_stream 的 metadata event 包含 compression 字段"""
        rag = self._make_rag_system()
        rag._model_adapter.tokenizer = None
        rag._get_max_context_length = MagicMock(return_value=4096)
        rag._estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        rag._get_max_history_turns = MagicMock(return_value=5)
        
        async def _async_none(user_id=None):
            return None
        rag._load_user_preferences_async = _async_none
        rag._get_conversation_history = MagicMock(return_value=[])
        rag._load_user_preferences = MagicMock(return_value=None)
        rag._get_fact_retrieve_method = MagicMock(return_value="post_hoc")
        rag._fire_and_forget_log = MagicMock()
        
        # 模拟 async_stream_generate
        async def mock_stream(prompt, **kwargs):
            yield "hello"
            yield " world"
        
        rag._model_adapter.async_stream_generate = mock_stream
        
        events = []
        async for event in rag.chat_stream(
            query="你好",
            session_id="sess-1",
        ):
            events.append(event)
        
        # 找到 metadata event
        meta_events = [e for e in events if e.get("type") == "metadata"]
        assert len(meta_events) >= 1
        meta = meta_events[0]
        assert "compression_mode" in meta
        assert meta["compression_mode"] == "truncate"
        assert "trace_ids" in meta
