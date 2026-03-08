"""
Unit Tests for max_new_tokens 配置传递

测试 max_new_tokens 配置项在各层级的正确传递:
1. config.yaml → ModelConfig.max_new_tokens
2. DKIPlugin.chat() 从配置读取默认值
3. InjectionExecutor.execute() 默认值更新
4. Demo API chat.py 请求级覆盖

Author: AGI Demo Project
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List


# ============ Test 1: ModelConfig 默认值 ============

class TestModelConfigMaxNewTokens:
    """测试 ModelConfig 中 max_new_tokens 字段"""
    
    def test_model_config_default_value(self):
        """max_new_tokens 默认值应为 2048"""
        from dki.config.config_loader import ModelConfig
        config = ModelConfig()
        assert config.max_new_tokens == 2048
    
    def test_model_config_custom_value(self):
        """max_new_tokens 可自定义"""
        from dki.config.config_loader import ModelConfig
        config = ModelConfig(max_new_tokens=4096)
        assert config.max_new_tokens == 4096
    
    def test_model_config_from_dict(self):
        """从字典创建 ModelConfig 时 max_new_tokens 正确解析"""
        from dki.config.config_loader import ModelConfig
        data = {"default_engine": "llama", "max_new_tokens": 1024}
        config = ModelConfig(**data)
        assert config.max_new_tokens == 1024
        assert config.default_engine == "llama"
    
    def test_model_config_in_full_config(self):
        """Config 中的 model.max_new_tokens 正确嵌套"""
        from dki.config.config_loader import Config
        config = Config()
        assert hasattr(config.model, 'max_new_tokens')
        assert config.model.max_new_tokens == 2048


# ============ Test 2: DKIPlugin.chat() 默认值 ============

class TestDKIPluginMaxNewTokens:
    """测试 DKIPlugin.chat() 中 max_new_tokens 的默认值行为"""
    
    def _create_mock_plugin(self, config_max_tokens: int = 2048):
        """创建用于测试的 DKIPlugin (mock 依赖)"""
        from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
        from dki.adapters.base import IUserDataAdapter, UserPreference, ChatMessage
        
        # Mock model adapter
        model = MagicMock(spec=BaseModelAdapter)
        model.hidden_dim = 4096
        model.generate.return_value = ModelOutput(
            text="test response", input_tokens=10, output_tokens=5
        )
        
        # Mock user data adapter
        adapter = AsyncMock(spec=IUserDataAdapter)
        adapter.get_user_preferences = AsyncMock(return_value=[])
        adapter.get_chat_messages = AsyncMock(return_value=[])
        adapter.get_user_profile = AsyncMock(return_value=None)
        
        # Mock config
        config = MagicMock()
        config.model.max_new_tokens = config_max_tokens
        config.dki.recall = MagicMock()
        config.dki.recall.strategy = "recall_v4"
        
        return model, adapter, config
    
    def test_chat_max_new_tokens_default_none(self):
        """当 max_new_tokens 未指定 (None) 时, 应从 config 读取"""
        from dki.core.dki_plugin import DKIPlugin
        import inspect
        
        sig = inspect.signature(DKIPlugin.chat)
        param = sig.parameters.get('max_new_tokens')
        assert param is not None
        assert param.default is None, "max_new_tokens 默认值应为 None (由 config 决定)"
    
    @pytest.mark.asyncio
    async def test_chat_uses_config_max_new_tokens(self):
        """chat() 应使用 config.model.max_new_tokens 作为默认值"""
        from dki.core.dki_plugin import DKIPlugin
        from dki.models.base import ModelOutput
        
        model, adapter, config = self._create_mock_plugin(config_max_tokens=1024)
        
        # 使用 patch 避免实际初始化
        with patch.object(DKIPlugin, '__init__', return_value=None):
            plugin = DKIPlugin.__new__(DKIPlugin)
            plugin.model = model
            plugin.data_adapter = adapter
            plugin.config = config
            plugin.language = "cn"
            plugin._mis = None
            plugin._gating = None
            plugin._planner = MagicMock()
            plugin._executor = AsyncMock()
            plugin._executor.execute = AsyncMock(return_value=MagicMock(
                text="response", input_tokens=10, output_tokens=5,
                injection_applied=False, preference_cache_hit=False,
                preference_cache_tier=None, inference_latency_ms=100.0,
            ))
            plugin._preference_text_cache = {}
            plugin._preference_text_cache_ttl = {}
            plugin._preference_kv_cache = MagicMock()
            plugin._stats = {
                "total_chats": 0, "injections": 0, "injection_errors": 0,
                "cache_hits": 0, "fallbacks": 0,
            }
            
            # Mock planner
            mock_context = MagicMock()
            mock_context.memory_triggered = False
            mock_context.trigger_type = None
            mock_context.reference_resolved = False
            mock_context.reference_type = None
            mock_context.reference_scope = None
            plugin._planner.analyze_query.return_value = mock_context
            
            mock_plan = MagicMock()
            mock_plan.injection_enabled = False
            mock_plan.preference_text = ""
            mock_plan.final_input = "test"
            mock_plan.original_query = "test"
            mock_plan.strategy = "recall_v4"
            mock_plan.history_items = []
            mock_plan.assembled_suffix = None
            plugin._planner.build_plan.return_value = mock_plan


# ============ Test 3: InjectionExecutor 默认值 ============

class TestExecutorMaxNewTokens:
    """测试 InjectionExecutor.execute() 的 max_new_tokens 默认值"""
    
    def test_executor_default_max_new_tokens(self):
        """InjectionExecutor.execute() 的 max_new_tokens 默认值应为 2048"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        import inspect
        
        sig = inspect.signature(InjectionExecutor.execute)
        param = sig.parameters.get('max_new_tokens')
        assert param is not None
        assert param.default == 2048, (
            f"InjectionExecutor.execute() max_new_tokens 默认值应为 2048, "
            f"实际: {param.default}"
        )


# ============ Test 4: Demo API 请求模型 ============

class TestDemoAPIChatRequest:
    """测试 Demo API 的 ChatSendRequest 模型"""
    
    def test_chat_request_default_max_tokens(self):
        """ChatSendRequest 默认 max_tokens 应为 2048"""
        from demo.api.chat import ChatSendRequest
        
        request = ChatSendRequest(query="test", user_id="user01")
        assert request.max_tokens == 2048, (
            f"ChatSendRequest.max_tokens 默认值应为 2048, 实际: {request.max_tokens}"
        )
    
    def test_chat_request_custom_max_tokens(self):
        """ChatSendRequest 可自定义 max_tokens"""
        from demo.api.chat import ChatSendRequest
        
        request = ChatSendRequest(query="test", user_id="user01", max_tokens=4096)
        assert request.max_tokens == 4096
    
    def test_chat_request_max_tokens_range(self):
        """ChatSendRequest max_tokens 应在 1-8192 范围内"""
        from demo.api.chat import ChatSendRequest
        from pydantic import ValidationError
        
        # 最小值
        request = ChatSendRequest(query="test", user_id="user01", max_tokens=1)
        assert request.max_tokens == 1
        
        # 最大值
        request = ChatSendRequest(query="test", user_id="user01", max_tokens=8192)
        assert request.max_tokens == 8192
        
        # 超出范围
        with pytest.raises(ValidationError):
            ChatSendRequest(query="test", user_id="user01", max_tokens=0)
        
        with pytest.raises(ValidationError):
            ChatSendRequest(query="test", user_id="user01", max_tokens=8193)


# ============ Test 5: DKISystem 默认值 ============

class TestDKISystemMaxNewTokens:
    """测试 DKISystem.chat() 的 max_new_tokens 默认值"""
    
    def test_dki_system_default_max_new_tokens(self):
        """DKISystem.chat() 的 max_new_tokens 默认值应为 2048"""
        from dki.core.dki_system import DKISystem
        import inspect
        
        sig = inspect.signature(DKISystem.chat)
        param = sig.parameters.get('max_new_tokens')
        assert param is not None
        assert param.default == 2048, (
            f"DKISystem.chat() max_new_tokens 默认值应为 2048, "
            f"实际: {param.default}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
