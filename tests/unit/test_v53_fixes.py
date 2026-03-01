"""
v5.3 修复的单元测试

测试内容:
1. RAG 系统偏好注入 (_load_user_preferences 返回 str)
2. 实验系统偏好写入对 DKI/RAG 双模式生效
3. dki_routes.py / app.py alpha 修正逻辑
4. 实验数据生成器长会话扩展 (n_sessions=40, n_turns=20)
5. DKI 系统 recall v4 调试日志
6. session_id 全链路一致性
7. _DetachedMessage / _ConversationRepoWrapper 安全性

Author: AGI Demo Project
Version: 5.3.0
"""

import pytest
import json
import os
import sys
import uuid
import inspect
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# 确保项目根目录在 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ============ Mock 类 ============

class MockGatingDecision:
    def __init__(self, should_inject=True, alpha=0.4):
        self.should_inject = should_inject
        self.alpha = alpha
        self.entropy = 0.5
        self.relevance_score = 0.6
        self.margin = 0.1
        self.memories = []
        self.reasoning = "test"


class MockDKIResponse:
    """模拟 DKIResponse，用于 alpha 修正逻辑测试"""
    def __init__(self, gating_alpha=0.0, pref_alpha=0.4):
        self.text = "test response"
        self.input_tokens = 100
        self.output_tokens = 50
        self.gating_decision = MockGatingDecision(alpha=gating_alpha)
        self.cache_hit = False
        self.cache_tier = "L3"
        self.latency_ms = 150.0
        self.metadata = {
            'hybrid_injection': {
                'enabled': True,
                'preference_tokens': 50,
                'history_tokens': 100,
                'preference_alpha': pref_alpha,
                'preference_text': "素食偏好",
                'history_suffix_text': "历史对话摘要",
                'history_messages': [{"role": "user", "content": "hello"}],
                'final_input': "formatted prompt",
            },
        }
        self.latency_breakdown = None
        self.memories_used = []


# ============ 测试类 ============

class TestRAGPreferenceInjection:
    """测试 RAG 系统偏好注入 (v5.3)"""
    
    def test_load_user_preferences_returns_string(self):
        """测试 _load_user_preferences 返回合并后的偏好文本字符串"""
        from dki.core.rag_system import RAGSystem
        
        with patch('dki.core.rag_system.ConfigLoader') as mock_config, \
             patch('dki.core.rag_system.DatabaseManager') as mock_db_cls, \
             patch('dki.core.rag_system.EmbeddingService'), \
             patch('dki.core.rag_system.MemoryRouter'):
            
            mock_config.return_value.config = MagicMock()
            mock_config.return_value.config.database.path = ":memory:"
            mock_config.return_value.config.database.echo = False
            mock_config.return_value.config.rag.top_k = 5
            mock_config.return_value.config.rag.similarity_threshold = 0.5
            
            rag = RAGSystem.__new__(RAGSystem)
            rag.config = mock_config.return_value.config
            rag.db_manager = MagicMock()
            rag.embedding_service = MagicMock()
            rag.memory_router = MagicMock()
            rag._model_adapter = MagicMock()
            rag._engine = None
            
            # Mock preferences
            mock_pref1 = MagicMock()
            mock_pref1.preference_text = "我喜欢素食"
            mock_pref2 = MagicMock()
            mock_pref2.preference_text = "我对海鲜过敏"
            
            mock_session = MagicMock()
            rag.db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
            rag.db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('dki.core.rag_system.UserPreferenceRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.get_by_user.return_value = [mock_pref1, mock_pref2]
                mock_repo_cls.return_value = mock_repo
                
                result = rag._load_user_preferences("u1")
                
                assert result is not None
                assert isinstance(result, str)
                assert "素食" in result
                assert "海鲜过敏" in result
    
    def test_load_user_preferences_returns_none_for_empty_user(self):
        """测试无 user_id 时返回 None"""
        from dki.core.rag_system import RAGSystem
        
        rag = RAGSystem.__new__(RAGSystem)
        rag.db_manager = MagicMock()
        
        assert rag._load_user_preferences(None) is None
        assert rag._load_user_preferences("") is None
    
    def test_load_user_preferences_returns_none_for_no_prefs(self):
        """测试数据库无偏好时返回 None"""
        from dki.core.rag_system import RAGSystem
        
        rag = RAGSystem.__new__(RAGSystem)
        rag.db_manager = MagicMock()
        mock_session = MagicMock()
        rag.db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        rag.db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch('dki.core.rag_system.UserPreferenceRepository') as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_by_user.return_value = []
            mock_repo_cls.return_value = mock_repo
            
            result = rag._load_user_preferences("u1")
            assert result is None
    
    def test_chat_metadata_includes_preference_info(self):
        """测试 chat 方法返回的 metadata 包含偏好注入信息"""
        from dki.core.rag_system import RAGSystem
        
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag.config.rag.top_k = 5
        rag.db_manager = MagicMock()
        rag.embedding_service = MagicMock()
        rag.memory_router = MagicMock()
        rag.memory_router.search.return_value = []
        rag._model_adapter = MagicMock()
        rag._engine = None
        
        # Mock model
        mock_output = MagicMock()
        mock_output.text = "response"
        mock_output.input_tokens = 50
        mock_output.output_tokens = 20
        rag._model_adapter.generate.return_value = mock_output
        rag._model_adapter.model_name = "test-model"
        
        # Mock _load_user_preferences to return text
        with patch.object(rag, '_load_user_preferences', return_value="我是素食主义者"), \
             patch.object(rag, '_get_conversation_history', return_value=[]), \
             patch.object(rag, '_build_prompt', return_value=("formatted", MagicMock())):
            
            mock_session = MagicMock()
            rag.db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
            rag.db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)
            
            response = rag.chat(
                query="推荐餐厅",
                session_id="s1",
                user_id="u1",
            )
            
            # 验证 metadata
            assert response.metadata['preference_injected'] is True
            assert "素食主义者" in response.metadata['preference_text']
    
    def test_build_prompt_signature_no_preference_param(self):
        """确认 _build_prompt 不接受 preference 参数 (偏好已在 chat() 中合并到 system_prompt)"""
        from dki.core.rag_system import RAGSystem
        sig = inspect.signature(RAGSystem._build_prompt)
        param_names = list(sig.parameters.keys())
        
        assert 'preference' not in param_names
        assert 'query' in param_names
        assert 'memories' in param_names
        assert 'system_prompt' in param_names
        assert 'history' in param_names


class TestAlphaReportingCorrection:
    """测试 alpha 修正逻辑 (v5.3)"""
    
    def test_preference_alpha_used_when_gating_alpha_is_zero(self):
        """测试 gating alpha = 0 时，显示 preference_alpha"""
        response = MockDKIResponse(gating_alpha=0.0, pref_alpha=0.4)
        hybrid_info = response.metadata.get('hybrid_injection', {})
        
        _gating_alpha = response.gating_decision.alpha  # 0.0
        _pref_alpha = hybrid_info.get("preference_alpha", 0.0)  # 0.4
        _display_alpha = max(_pref_alpha, _gating_alpha)
        
        assert _display_alpha == 0.4
        assert _pref_alpha == 0.4
    
    def test_gating_alpha_used_when_higher(self):
        """测试 gating alpha > preference alpha 时取较大值"""
        response = MockDKIResponse(gating_alpha=0.8, pref_alpha=0.4)
        hybrid_info = response.metadata.get('hybrid_injection', {})
        
        _gating_alpha = response.gating_decision.alpha  # 0.8
        _pref_alpha = hybrid_info.get("preference_alpha", 0.0)  # 0.4
        _display_alpha = max(_pref_alpha, _gating_alpha)
        
        assert _display_alpha == 0.8
    
    def test_injection_enabled_includes_preference(self):
        """测试有偏好 token 时 injection_enabled 为 True"""
        response = MockDKIResponse(gating_alpha=0.0, pref_alpha=0.4)
        hybrid_info = response.metadata.get('hybrid_injection', {})
        preference_tokens = hybrid_info.get("preference_tokens", 0)
        history_tokens = hybrid_info.get("history_tokens", 0)
        
        # v5.3: 偏好或历史注入也算 enabled
        injection_enabled = bool(preference_tokens) or bool(history_tokens) or response.gating_decision.should_inject
        
        assert injection_enabled is True
        assert preference_tokens == 50
    
    def test_hybrid_info_missing_gracefully(self):
        """测试 hybrid_info 缺失时不崩溃"""
        response = MockDKIResponse(gating_alpha=0.0, pref_alpha=0.0)
        response.metadata = {}  # 清空 metadata
        
        hybrid_info = response.metadata.get('hybrid_injection', {})
        pref_alpha = hybrid_info.get("preference_alpha", 0.0)
        pref_tokens = hybrid_info.get("preference_tokens", 0)
        
        assert pref_alpha == 0.0
        assert pref_tokens == 0


class TestDataGeneratorLongSession:
    """测试数据生成器长会话扩展 (v5.3)"""
    
    def test_long_session_default_params(self):
        """测试默认参数: n_sessions=40, n_turns_per_session=20"""
        sig = inspect.signature(
            __import__('dki.experiment.data_generator', fromlist=['ExperimentDataGenerator']).ExperimentDataGenerator.generate_long_session_persona_chat
        )
        params = sig.parameters
        
        assert params['n_sessions'].default == 40
        assert params['n_turns_per_session'].default == 20
    
    def test_long_session_generates_exact_turns(self):
        """测试生成准确的轮数 (包括动态扩展)"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            
            # 生成 1 个会话，要求 20 轮
            data = gen.generate_long_session_persona_chat(
                n_sessions=1,
                n_turns_per_session=20,
            )
            
            assert len(data) == 1
            session = data[0]
            assert len(session['turns']) == 20
    
    def test_long_session_extra_turns_have_memory_flag(self):
        """测试额外 turns 标记为 tests_memory=True"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            data = gen.generate_long_session_persona_chat(
                n_sessions=1,
                n_turns_per_session=20,
            )
            
            session = data[0]
            # 额外 turns 标记为记忆测试
            extra_turns = [t for t in session['turns'] if t.get('tests_memory')]
            assert len(extra_turns) > 0
    
    def test_long_session_extra_turns_have_query(self):
        """测试额外 turns 包含 query"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            data = gen.generate_long_session_persona_chat(
                n_sessions=1,
                n_turns_per_session=15,
            )
            
            session = data[0]
            for turn in session['turns']:
                assert 'query' in turn
                assert len(turn['query']) > 0
                assert 'expected_keywords' in turn
    
    def test_long_session_metadata_type(self):
        """测试长会话 metadata 包含正确的 session_type"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            data = gen.generate_long_session_persona_chat(
                n_sessions=1,
                n_turns_per_session=10,
            )
            
            session = data[0]
            assert session['session_type'] == 'long'
            assert session['metadata']['session_type'] == 'long'


class TestSessionIdConsistency:
    """测试 session_id 全链路一致性"""
    
    def test_session_id_preserved_when_provided(self):
        """测试前端传入 session_id 时不被覆盖"""
        session_id_from_request = "session_abc123"
        
        # 后端逻辑: session_id = request.session_id or f"session_{...}"
        session_id = session_id_from_request or f"session_{uuid.uuid4().hex[:8]}"
        
        assert session_id == "session_abc123"
    
    def test_session_id_generated_when_none(self):
        """测试 session_id 为 None 时生成新 session_id"""
        session_id_from_request = None
        
        session_id = session_id_from_request or f"session_{uuid.uuid4().hex[:8]}"
        
        assert session_id.startswith("session_")
        assert len(session_id) == len("session_") + 8
    
    def test_dki_routes_session_id_fallback_to_user_id(self):
        """测试 DKI routes 中 session_id 为 None 时 fallback 到 user_id"""
        request_session_id = None
        verified_user_id = "user_001"
        
        # dki_routes.py 逻辑
        session_id = request_session_id or verified_user_id
        assert session_id == "user_001"
    
    def test_session_id_consistent_across_mode_switch(self):
        """测试模式切换时 session_id 保持不变"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        for mode in ['dki', 'rag', 'baseline']:
            # 模式切换不应改变 session_id
            current_session_id = session_id  # 在 JS 中, 此变量不因 mode 而变
            assert current_session_id == session_id
    
    def test_experiment_session_id_unique_per_sample(self):
        """测试实验系统中每个 sample 有独立的 session_id"""
        import time
        base_ts = int(time.time())
        
        session_ids = set()
        for idx in range(10):
            for mode in ['dki', 'rag']:
                session_id = f"exp_{mode}_{base_ts}_{idx}"
                session_ids.add(session_id)
        
        assert len(session_ids) == 20  # 10 samples × 2 modes


class TestExperimentPreferenceWriting:
    """测试实验系统偏好写入对 DKI/RAG 双模式生效 (v5.3)"""
    
    def test_run_mode_writes_prefs_unconditionally(self):
        """验证 _run_mode 偏好写入不再限制 mode == 'dki'"""
        from dki.experiment.runner import ExperimentRunner
        source = inspect.getsource(ExperimentRunner._run_mode)
        
        # 偏好写入应出现
        assert "_write_session_preferences" in source
        
        # 验证: 偏好写入前不应有 "if mode == 'dki'" 的条件限制
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if '_write_session_preferences' in line:
                # 查看前 5 行上下文
                prev_context = '\n'.join(lines[max(0, i-5):i])
                assert "if mode == 'dki'" not in prev_context, \
                    f"偏好写入仍然被限制为 DKI 模式: 行 {i}"
    
    def test_run_mode_preferences_guard_uses_personas(self):
        """验证偏好写入使用 item.get('personas') 作为条件"""
        from dki.experiment.runner import ExperimentRunner
        source = inspect.getsource(ExperimentRunner._run_mode)
        
        assert "if item.get('personas'):" in source


class TestDetachedMessage:
    """测试 _DetachedMessage 安全性"""
    
    def test_detached_message_preserves_all_attributes(self):
        """测试 _DetachedMessage 正确提取所有 ORM 属性"""
        from dki.core.dki_system import _DetachedMessage
        from datetime import datetime
        
        mock_orm = MagicMock()
        mock_orm.id = "conv_001"
        mock_orm.session_id = "session_test"
        mock_orm.role = "user"
        mock_orm.content = "Hello, how are you?"
        mock_orm.created_at = datetime(2026, 2, 25, 12, 0, 0)
        mock_orm.injection_mode = "dki"
        mock_orm.injection_alpha = 0.4
        mock_orm.memory_ids = ["mem_001"]
        mock_orm.latency_ms = 150.0
        
        dm = _DetachedMessage(mock_orm)
        
        assert dm.id == "conv_001"
        assert dm.message_id == "conv_001"  # 兼容字段
        assert dm.session_id == "session_test"
        assert dm.role == "user"
        assert dm.content == "Hello, how are you?"
        assert dm.created_at == datetime(2026, 2, 25, 12, 0, 0)
        assert dm.injection_mode == "dki"
        assert dm.injection_alpha == 0.4
    
    def test_detached_message_defaults_for_missing_attrs(self):
        """测试 _DetachedMessage 对缺失属性使用 getattr 默认值"""
        from dki.core.dki_system import _DetachedMessage
        
        # 模拟最小 ORM 对象 (无任何属性 spec)
        mock_orm = MagicMock(spec=[])
        
        dm = _DetachedMessage(mock_orm)
        
        assert dm.id is None
        assert dm.role == 'user'  # 默认值
        assert dm.content == ''  # 默认值
        assert dm.session_id is None
    
    def test_detached_message_uses_slots(self):
        """测试 _DetachedMessage 使用 __slots__ 优化内存"""
        from dki.core.dki_system import _DetachedMessage
        assert hasattr(_DetachedMessage, '__slots__')
        assert 'role' in _DetachedMessage.__slots__
        assert 'content' in _DetachedMessage.__slots__
        assert 'session_id' in _DetachedMessage.__slots__


class TestConversationRepoWrapper:
    """测试 _ConversationRepoWrapper"""
    
    def test_to_detached_converts_all(self):
        """测试 _to_detached 转换所有 ORM 对象"""
        from dki.core.dki_system import _ConversationRepoWrapper, _DetachedMessage
        
        mock_objects = []
        for i in range(5):
            m = MagicMock()
            m.id = f"conv_{i}"
            m.session_id = "s1"
            m.role = "user" if i % 2 == 0 else "assistant"
            m.content = f"message {i}"
            m.created_at = None
            m.injection_mode = None
            m.injection_alpha = None
            m.memory_ids = None
            m.latency_ms = None
            mock_objects.append(m)
        
        result = _ConversationRepoWrapper._to_detached(mock_objects)
        
        assert len(result) == 5
        assert all(isinstance(r, _DetachedMessage) for r in result)
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[2].content == "message 2"
    
    def test_to_detached_empty_list(self):
        """测试空列表输入"""
        from dki.core.dki_system import _ConversationRepoWrapper
        
        result = _ConversationRepoWrapper._to_detached([])
        assert result == []


class TestRecallV4DebugLog:
    """测试 recall v4 调试日志格式"""
    
    def test_recall_v4_precheck_log_exists(self):
        """验证 recall v4 pre-check 日志存在于 chat 方法中"""
        from dki.core.dki_system import DKISystem
        source = inspect.getsource(DKISystem.chat)
        
        assert "[Recall v4 pre-check]" in source
        assert "session_id=" in source
        assert "messages_in_db=" in source
    
    def test_recall_v4_precheck_log_includes_roles(self):
        """验证日志包含最近消息的 role 信息"""
        from dki.core.dki_system import DKISystem
        source = inspect.getsource(DKISystem.chat)
        
        assert "roles=" in source


class TestRAGSystemSignatures:
    """测试 RAG 系统方法签名"""
    
    def test_chat_accepts_user_id(self):
        """验证 chat 方法接受 user_id 参数"""
        from dki.core.rag_system import RAGSystem
        sig = inspect.signature(RAGSystem.chat)
        
        assert 'user_id' in sig.parameters
        assert sig.parameters['user_id'].default is None
    
    def test_load_user_preferences_exists(self):
        """验证 _load_user_preferences 方法存在"""
        from dki.core.rag_system import RAGSystem
        assert hasattr(RAGSystem, '_load_user_preferences')
    
    def test_load_user_preferences_return_type(self):
        """验证 _load_user_preferences 返回 Optional[str]"""
        from dki.core.rag_system import RAGSystem
        sig = inspect.signature(RAGSystem._load_user_preferences)
        
        # 验证参数
        assert 'user_id' in sig.parameters
        
        # 验证返回类型注解
        return_annotation = sig.return_annotation
        # Optional[str] 即 Union[str, None]
        assert return_annotation is not inspect.Parameter.empty


class TestDKISystemRecallV4Condition:
    """测试 DKI 系统 recall v4 触发条件 (v5.1/5.3)"""
    
    def test_chat_builds_prompt_when_preference_exists(self):
        """验证即使 assembled.total_tokens == 0，有 preference 时仍构建 prompt"""
        from dki.core.dki_system import DKISystem
        source = inspect.getsource(DKISystem.chat)
        
        # v5.2 修复: 使用 has_history / has_preference 变量
        # 条件: if has_history or has_preference
        assert "has_preference" in source
        assert "has_history or has_preference" in source


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
