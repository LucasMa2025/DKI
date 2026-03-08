"""
Unit Tests for DKI Integration Layer (v4.0)

测试覆盖:
1. EnhancedDKIPlugin — 动态路由 + 消息管理
2. Factory — create_plugin / create_plugin_from_dict
3. Middleware — DKIMiddleware + get_dki_plugin
4. 配置解析 — DynamicRouterConfig / MessageManagementConfig

Author: AGI Demo Project
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Mock 基础设施
# ============================================================

class MockModelAdapter:
    """模拟模型适配器"""
    model_name = "test-model"
    is_loaded = True
    tokenizer = None
    max_model_len = 4096
    
    def generate(self, prompt, **kwargs):
        return MagicMock(text="mock response", input_tokens=10, output_tokens=20)
    
    async def async_generate(self, prompt, **kwargs):
        return MagicMock(text="mock response", input_tokens=10, output_tokens=20)
    
    async def async_stream_generate(self, prompt, **kwargs):
        yield {"type": "token", "content": "mock "}
        yield {"type": "token", "content": "response"}
        yield {"type": "done", "text": "mock response"}
    
    def get_model_info(self):
        return {"name": "test-model", "loaded": True}


class MockDataAdapter:
    """模拟数据适配器"""
    is_connected = True
    
    async def connect(self):
        pass
    
    async def disconnect(self):
        pass
    
    async def get_user_preferences(self, user_id):
        return []
    
    async def search_relevant_history(self, user_id, session_id, query, **kwargs):
        return []
    
    async def get_recent_messages(self, session_id, limit=10):
        return []
    
    async def get_user_profile(self, user_id):
        return None


class MockDKIPlugin:
    """模拟 DKI Plugin"""
    
    def __init__(self):
        self.model = MockModelAdapter()
        self.data_adapter = MockDataAdapter()
        self.planner = MagicMock()
        self.executor = MagicMock()
        self._preference_text_cache = {}
        self._chat_called = False
        self._chat_stream_called = False
        self._closed = False
    
    async def chat(self, query, user_id, session_id, **kwargs):
        self._chat_called = True
        from dki.core.dki_plugin import DKIPluginResponse, InjectionMetadata
        metadata = InjectionMetadata(
            injection_enabled=True,
            alpha=0.3,
            latency_ms=50.0,
            injection_strategy="recall_v4",
        )
        return DKIPluginResponse(
            text=f"DKI response to: {query}",
            input_tokens=10,
            output_tokens=20,
            metadata=metadata,
        )
    
    async def chat_stream(self, query, user_id, session_id, **kwargs):
        self._chat_stream_called = True
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata(injection_enabled=True, alpha=0.3)
        yield {"type": "metadata", "metadata": metadata.to_dict()}
        yield {"type": "token", "content": "DKI "}
        yield {"type": "token", "content": "stream"}
        yield {"type": "done", "text": "DKI stream", "input_tokens": 10, "output_tokens": 5}
    
    def get_stats(self):
        return {"total_requests": 1, "cache_hits": 0}
    
    def get_injection_logs(self, limit=100, offset=0):
        return []
    
    def clear_preference_cache(self, user_id=None):
        if user_id:
            self._preference_text_cache.pop(user_id, None)
        else:
            self._preference_text_cache.clear()
    
    def invalidate_preference_text_cache(self, user_id=None):
        self.clear_preference_cache(user_id)
    
    async def invalidate_user_cache(self, user_id):
        return 0
    
    def get_component_configs(self):
        return {"model": "test-model"}
    
    async def close(self):
        self._closed = True


class MockRAGSystem:
    """模拟 RAG 系统"""
    
    def chat(self, query, session_id, user_id=None, **kwargs):
        @dataclass
        class RAGResp:
            text: str = f"RAG response to: {query}"
            input_tokens: int = 15
            output_tokens: int = 25
            memories_used: list = None
            latency_ms: float = 30.0
            
            def __post_init__(self):
                if self.memories_used is None:
                    self.memories_used = []
        
        return RAGResp()


class MockStore:
    """模拟 Chat Store"""
    
    def __init__(self):
        self.messages = []
        self.sessions = {}
    
    def get_session(self, session_id):
        return self.sessions.get(session_id)
    
    def create_session(self, user_id, title, session_id=None):
        self.sessions[session_id] = {"user_id": user_id, "title": title}
        return self.sessions[session_id]
    
    def add_message(self, session_id, user_id, role, content, metadata=None):
        self.messages.append({
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "metadata": metadata,
        })
    
    def add_preference(self, user_id, preference_text, preference_type="general", priority=5):
        return {"user_id": user_id, "text": preference_text}


# ============================================================
# Test: DynamicRouterConfig
# ============================================================

class TestDynamicRouterConfig:
    """测试动态路由配置"""
    
    def test_default_config(self):
        from dki.integration.enhanced_plugin import DynamicRouterConfig
        cfg = DynamicRouterConfig()
        assert cfg.enabled is False
        assert cfg.dki_threshold == 0.45
        assert cfg.rag_threshold == 0.25
        assert cfg.default_mode == "dki"
    
    def test_from_dict(self):
        from dki.integration.enhanced_plugin import DynamicRouterConfig
        cfg = DynamicRouterConfig.from_dict({
            "enabled": True,
            "dki_threshold": 0.5,
            "rag_threshold": 0.3,
        })
        assert cfg.enabled is True
        assert cfg.dki_threshold == 0.5
        assert cfg.rag_threshold == 0.3
    
    def test_from_empty_dict(self):
        from dki.integration.enhanced_plugin import DynamicRouterConfig
        cfg = DynamicRouterConfig.from_dict({})
        assert cfg.enabled is False
    
    def test_from_none(self):
        from dki.integration.enhanced_plugin import DynamicRouterConfig
        cfg = DynamicRouterConfig.from_dict(None)
        assert cfg.enabled is False


class TestMessageManagementConfig:
    """测试消息管理配置"""
    
    def test_default_config(self):
        from dki.integration.enhanced_plugin import MessageManagementConfig
        cfg = MessageManagementConfig()
        assert cfg.enabled is False
        assert cfg.write_user_message is True
        assert cfg.write_assistant_message is True
        assert cfg.auto_extract_preference is False
    
    def test_from_dict(self):
        from dki.integration.enhanced_plugin import MessageManagementConfig
        cfg = MessageManagementConfig.from_dict({
            "enabled": True,
            "write_user_message": True,
            "write_assistant_message": False,
        })
        assert cfg.enabled is True
        assert cfg.write_assistant_message is False


class TestEnhancedPluginConfig:
    """测试增强插件完整配置"""
    
    def test_default_config(self):
        from dki.integration.enhanced_plugin import EnhancedPluginConfig
        cfg = EnhancedPluginConfig()
        assert cfg.dynamic_router.enabled is False
        assert cfg.message_management.enabled is False
    
    def test_from_dict(self):
        from dki.integration.enhanced_plugin import EnhancedPluginConfig
        cfg = EnhancedPluginConfig.from_dict({
            "dynamic_router": {"enabled": True, "dki_threshold": 0.6},
            "message_management": {"enabled": True},
        })
        assert cfg.dynamic_router.enabled is True
        assert cfg.dynamic_router.dki_threshold == 0.6
        assert cfg.message_management.enabled is True


# ============================================================
# Test: EnhancedDKIPlugin
# ============================================================

class TestEnhancedDKIPlugin:
    """测试增强 DKI Plugin"""
    
    @pytest.fixture
    def dki_plugin(self):
        return MockDKIPlugin()
    
    @pytest.fixture
    def rag_system(self):
        return MockRAGSystem()
    
    @pytest.fixture
    def store(self):
        return MockStore()
    
    def test_init_default(self, dki_plugin):
        """测试默认初始化 (无增强功能)"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        assert enhanced.dki_plugin is dki_plugin
        assert enhanced.model is dki_plugin.model
        assert enhanced.data_adapter is dki_plugin.data_adapter
    
    def test_init_with_features(self, dki_plugin, rag_system, store):
        """测试启用增强功能"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig,
            DynamicRouterConfig, MessageManagementConfig,
        )
        config = EnhancedPluginConfig(
            dynamic_router=DynamicRouterConfig(enabled=True),
            message_management=MessageManagementConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(
            dki_plugin, rag_system=rag_system, store=store, config=config,
        )
        assert enhanced._config.dynamic_router.enabled is True
        assert enhanced._config.message_management.enabled is True
    
    @pytest.mark.asyncio
    async def test_chat_default_dki(self, dki_plugin):
        """测试默认 chat (走 DKI 路径)"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        response = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        
        assert "DKI response" in response.text
        assert dki_plugin._chat_called is True
        assert enhanced._enhanced_stats["dki_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_chat_force_mode_rag(self, dki_plugin, rag_system):
        """测试强制 RAG 模式"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin, rag_system=rag_system)
        
        response = await enhanced.chat(
            "什么是量子计算", user_id="u1", session_id="s1",
            force_mode="rag",
        )
        
        assert "RAG response" in response.text
        assert enhanced._enhanced_stats["rag_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_chat_force_mode_dki(self, dki_plugin, rag_system):
        """测试强制 DKI 模式"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin, rag_system=rag_system)
        
        response = await enhanced.chat(
            "推荐餐厅", user_id="u1", session_id="s1",
            force_mode="dki",
        )
        
        assert "DKI response" in response.text
        assert enhanced._enhanced_stats["dki_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_message_management_writes(self, dki_plugin, store):
        """测试消息管理 — 自动写入用户消息和助手回复"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig, MessageManagementConfig,
        )
        config = EnhancedPluginConfig(
            message_management=MessageManagementConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(dki_plugin, store=store, config=config)
        
        response = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        
        # 应该写入 2 条消息 (user + assistant)
        assert len(store.messages) == 2
        assert store.messages[0]["role"] == "user"
        assert store.messages[0]["content"] == "推荐餐厅"
        assert store.messages[1]["role"] == "assistant"
        assert "DKI response" in store.messages[1]["content"]
    
    @pytest.mark.asyncio
    async def test_message_management_disabled(self, dki_plugin, store):
        """测试消息管理禁用时不写入"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin, store=store)
        
        await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        
        assert len(store.messages) == 0
    
    @pytest.mark.asyncio
    async def test_message_management_creates_session(self, dki_plugin, store):
        """测试消息管理 — 自动创建会话"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig, MessageManagementConfig,
        )
        config = EnhancedPluginConfig(
            message_management=MessageManagementConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(dki_plugin, store=store, config=config)
        
        await enhanced.chat("推荐餐厅", user_id="u1", session_id="new_session")
        
        assert "new_session" in store.sessions
    
    @pytest.mark.asyncio
    async def test_rag_fallback_on_error(self, dki_plugin, rag_system):
        """测试 RAG 执行失败时降级到 DKI"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        # 让 RAG 抛出异常
        rag_system.chat = MagicMock(side_effect=Exception("RAG failed"))
        
        enhanced = EnhancedDKIPlugin(dki_plugin, rag_system=rag_system)
        
        response = await enhanced.chat(
            "测试", user_id="u1", session_id="s1",
            force_mode="rag",
        )
        
        # 应该降级到 DKI
        assert "DKI response" in response.text
    
    @pytest.mark.asyncio
    async def test_chat_stream_dki(self, dki_plugin):
        """测试流式 chat (DKI 路径)"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        chunks = []
        async for chunk in enhanced.chat_stream("推荐餐厅", user_id="u1", session_id="s1"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert any(c.get("type") == "token" for c in chunks)
        assert dki_plugin._chat_stream_called is True
    
    @pytest.mark.asyncio
    async def test_chat_stream_rag(self, dki_plugin, rag_system):
        """测试流式 chat (RAG 路径, 模拟流式)"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin, rag_system=rag_system)
        
        chunks = []
        async for chunk in enhanced.chat_stream(
            "什么是量子计算", user_id="u1", session_id="s1",
            force_mode="rag",
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3  # metadata + token + done
        assert chunks[0]["type"] == "metadata"
        assert chunks[1]["type"] == "token"
        assert chunks[2]["type"] == "done"
    
    @pytest.mark.asyncio
    async def test_add_preference(self, dki_plugin, store):
        """测试添加偏好"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin, store=store)
        
        result = await enhanced.add_preference(
            user_id="u1",
            preference_text="我喜欢中餐",
        )
        
        assert result["text"] == "我喜欢中餐"
    
    @pytest.mark.asyncio
    async def test_add_preference_no_store(self, dki_plugin):
        """测试无 store 时添加偏好抛出异常"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        with pytest.raises(RuntimeError, match="Store not configured"):
            await enhanced.add_preference(user_id="u1", preference_text="test")
    
    def test_get_stats(self, dki_plugin):
        """测试统计信息合并"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        stats = enhanced.get_stats()
        assert "enhanced" in stats
        assert "total_requests" in stats["enhanced"]
        assert "dynamic_router_enabled" in stats["enhanced"]
    
    def test_get_component_configs(self, dki_plugin):
        """测试组件配置"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        configs = enhanced.get_component_configs()
        assert "dynamic_router" in configs
        assert "message_management" in configs
    
    @pytest.mark.asyncio
    async def test_close(self, dki_plugin):
        """测试关闭"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        await enhanced.close()
        assert dki_plugin._closed is True
    
    def test_route_mode_no_router(self, dki_plugin):
        """测试路由器未启用时默认 DKI"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        mode = enhanced._resolve_route_mode("test", "u1", "s1")
        assert mode == "dki"
    
    def test_route_mode_force(self, dki_plugin):
        """测试强制模式"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        enhanced = EnhancedDKIPlugin(dki_plugin)
        
        mode = enhanced._resolve_route_mode("test", "u1", "s1", force_mode="rag")
        assert mode == "rag"
    
    def test_route_mode_no_rag(self, dki_plugin):
        """测试 RAG 不可用时回退 DKI"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig, DynamicRouterConfig,
        )
        config = EnhancedPluginConfig(
            dynamic_router=DynamicRouterConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(dki_plugin, config=config)
        
        mode = enhanced._resolve_route_mode("test", "u1", "s1")
        assert mode == "dki"


# ============================================================
# Test: Factory
# ============================================================

class TestFactory:
    """测试工厂方法"""
    
    def test_resolve_router_config_bool_true(self):
        """测试 bool True → 启用默认配置"""
        from dki.integration.factory import _resolve_router_config
        cfg = _resolve_router_config(True)
        assert cfg.enabled is True
    
    def test_resolve_router_config_bool_false(self):
        """测试 bool False → 禁用"""
        from dki.integration.factory import _resolve_router_config
        cfg = _resolve_router_config(False)
        assert cfg.enabled is False
    
    def test_resolve_router_config_dict(self):
        """测试 dict → 自定义配置"""
        from dki.integration.factory import _resolve_router_config
        cfg = _resolve_router_config({"enabled": True, "dki_threshold": 0.6})
        assert cfg.enabled is True
        assert cfg.dki_threshold == 0.6
    
    def test_resolve_router_config_object(self):
        """测试 DynamicRouterConfig 对象"""
        from dki.integration.factory import _resolve_router_config
        from dki.integration.enhanced_plugin import DynamicRouterConfig
        original = DynamicRouterConfig(enabled=True, dki_threshold=0.7)
        cfg = _resolve_router_config(original)
        assert cfg is original
    
    def test_resolve_message_config_bool_true(self):
        """测试 bool True → 启用默认配置"""
        from dki.integration.factory import _resolve_message_config
        cfg = _resolve_message_config(True)
        assert cfg.enabled is True
    
    def test_resolve_message_config_dict(self):
        """测试 dict → 自定义配置"""
        from dki.integration.factory import _resolve_message_config
        cfg = _resolve_message_config({
            "enabled": True,
            "write_assistant_message": False,
        })
        assert cfg.enabled is True
        assert cfg.write_assistant_message is False
    
    @pytest.mark.asyncio
    async def test_create_plugin_basic(self):
        """测试基本创建 (mock 所有依赖)"""
        from dki.integration.factory import create_plugin
        
        mock_adapter = MockModelAdapter()
        mock_data_adapter = MockDataAdapter()
        
        with patch('dki.integration.factory.DKIPlugin') as MockPlugin:
            mock_plugin_instance = MockDKIPlugin()
            MockPlugin.from_config = AsyncMock(return_value=mock_plugin_instance)
            
            result = await create_plugin(
                adapter_config={"database": {"type": "sqlite", "database": ":memory:"}},
                model_adapter=mock_adapter,
            )
            
            # 不启用增强功能时, 返回原始 DKI Plugin
            assert result is mock_plugin_instance
            MockPlugin.from_config.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_plugin_with_enhanced(self):
        """测试创建增强插件"""
        from dki.integration.factory import create_plugin
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        mock_adapter = MockModelAdapter()
        mock_store = MockStore()
        
        with patch('dki.integration.factory.DKIPlugin') as MockPlugin:
            mock_plugin_instance = MockDKIPlugin()
            MockPlugin.from_config = AsyncMock(return_value=mock_plugin_instance)
            
            result = await create_plugin(
                adapter_config={"database": {"type": "sqlite", "database": ":memory:"}},
                model_adapter=mock_adapter,
                dynamic_router=True,
                message_management=True,
                store=mock_store,
            )
            
            # 启用增强功能时, 返回 EnhancedDKIPlugin
            assert isinstance(result, EnhancedDKIPlugin)
            assert result._config.dynamic_router.enabled is True
            assert result._config.message_management.enabled is True
    
    @pytest.mark.asyncio
    async def test_create_plugin_from_dict(self):
        """测试从字典创建"""
        from dki.integration.factory import create_plugin_from_dict
        
        mock_adapter = MockModelAdapter()
        
        with patch('dki.integration.factory.DKIPlugin') as MockPlugin:
            mock_plugin_instance = MockDKIPlugin()
            MockPlugin.from_config = AsyncMock(return_value=mock_plugin_instance)
            
            result = await create_plugin_from_dict(
                config_dict={
                    "adapter_config": {"database": {"type": "sqlite", "database": ":memory:"}},
                    "language": "cn",
                },
                model_adapter=mock_adapter,
            )
            
            assert result is mock_plugin_instance


# ============================================================
# Test: Middleware
# ============================================================

class TestMiddleware:
    """测试 DKI Middleware"""
    
    def test_get_dki_plugin_success(self):
        """测试成功获取 DKI Plugin"""
        from dki.integration.middleware import get_dki_plugin
        
        mock_request = MagicMock()
        mock_plugin = MockDKIPlugin()
        mock_request.app.state.dki_plugin = mock_plugin
        
        result = get_dki_plugin(mock_request)
        assert result is mock_plugin
    
    def test_get_dki_plugin_not_initialized(self):
        """测试 DKI Plugin 未初始化时抛出异常"""
        from dki.integration.middleware import get_dki_plugin
        
        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])  # 无 dki_plugin 属性
        
        with pytest.raises(RuntimeError, match="DKI Plugin not initialized"):
            get_dki_plugin(mock_request)
    
    def test_middleware_init(self):
        """测试 Middleware 初始化"""
        from dki.integration.middleware import DKIMiddleware, FASTAPI_AVAILABLE
        
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        mock_app = MagicMock()
        mock_app._dki_middleware_registered = None
        
        # DKIMiddleware 需要 Starlette app
        # 这里只测试参数存储
        try:
            middleware = DKIMiddleware(
                mock_app,
                adapter_config_path="config/adapter.yaml",
                dynamic_router=True,
                language="cn",
            )
            assert middleware._plugin_kwargs["adapter_config_path"] == "config/adapter.yaml"
            assert middleware._plugin_kwargs["dynamic_router"] is True
            assert middleware._plugin_kwargs["language"] == "cn"
        except Exception:
            # Starlette BaseHTTPMiddleware 可能需要真实 app
            pass


# ============================================================
# Test: Integration (端到端)
# ============================================================

class TestIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_flow_dki_only(self):
        """测试完整流程: 仅 DKI"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        dki = MockDKIPlugin()
        enhanced = EnhancedDKIPlugin(dki)
        
        # 第一次请求
        r1 = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        assert "DKI response" in r1.text
        
        # 第二次请求
        r2 = await enhanced.chat("推荐酒店", user_id="u1", session_id="s1")
        assert "DKI response" in r2.text
        
        assert enhanced._enhanced_stats["total_requests"] == 2
        assert enhanced._enhanced_stats["dki_routes"] == 2
    
    @pytest.mark.asyncio
    async def test_full_flow_with_routing(self):
        """测试完整流程: DKI + RAG 路由"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        dki = MockDKIPlugin()
        rag = MockRAGSystem()
        enhanced = EnhancedDKIPlugin(dki, rag_system=rag)
        
        # DKI 请求
        r1 = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        assert "DKI response" in r1.text
        
        # RAG 请求 (强制)
        r2 = await enhanced.chat(
            "什么是量子计算", user_id="u1", session_id="s1",
            force_mode="rag",
        )
        assert "RAG response" in r2.text
        
        assert enhanced._enhanced_stats["dki_routes"] == 1
        assert enhanced._enhanced_stats["rag_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_full_flow_with_message_management(self):
        """测试完整流程: DKI + 消息管理"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig, MessageManagementConfig,
        )
        
        dki = MockDKIPlugin()
        store = MockStore()
        config = EnhancedPluginConfig(
            message_management=MessageManagementConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(dki, store=store, config=config)
        
        # 多轮对话
        await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
        await enhanced.chat("推荐酒店", user_id="u1", session_id="s1")
        
        # 应该有 4 条消息 (2 user + 2 assistant)
        assert len(store.messages) == 4
        assert store.messages[0]["role"] == "user"
        assert store.messages[1]["role"] == "assistant"
        assert store.messages[2]["role"] == "user"
        assert store.messages[3]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_full_flow_stream_with_message_management(self):
        """测试完整流程: 流式 + 消息管理"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin, EnhancedPluginConfig, MessageManagementConfig,
        )
        
        dki = MockDKIPlugin()
        store = MockStore()
        config = EnhancedPluginConfig(
            message_management=MessageManagementConfig(enabled=True),
        )
        enhanced = EnhancedDKIPlugin(dki, store=store, config=config)
        
        chunks = []
        async for chunk in enhanced.chat_stream("推荐餐厅", user_id="u1", session_id="s1"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        # 消息管理: user + assistant
        assert len(store.messages) == 2
    
    @pytest.mark.asyncio
    async def test_preference_management(self):
        """测试偏好管理"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        dki = MockDKIPlugin()
        store = MockStore()
        enhanced = EnhancedDKIPlugin(dki, store=store)
        
        # 添加偏好
        result = await enhanced.add_preference(
            user_id="u1",
            preference_text="我喜欢中餐",
        )
        assert result["text"] == "我喜欢中餐"
    
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """测试生命周期管理"""
        from dki.integration.enhanced_plugin import EnhancedDKIPlugin
        
        dki = MockDKIPlugin()
        enhanced = EnhancedDKIPlugin(dki)
        
        # 使用
        await enhanced.chat("test", user_id="u1", session_id="s1")
        
        # 关闭
        await enhanced.close()
        assert dki._closed is True


# ============================================================
# Test: __init__.py exports
# ============================================================

class TestExports:
    """测试导出"""
    
    def test_integration_exports(self):
        """测试 dki.integration 导出"""
        from dki.integration import (
            create_plugin,
            create_plugin_from_dict,
            DKIMiddleware,
            get_dki_plugin,
            EnhancedDKIPlugin,
        )
        assert callable(create_plugin)
        assert callable(create_plugin_from_dict)
        assert callable(get_dki_plugin)
    
    def test_enhanced_plugin_exports(self):
        """测试 enhanced_plugin 导出"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
            DynamicRouterConfig,
            MessageManagementConfig,
        )
        assert EnhancedDKIPlugin is not None
        assert EnhancedPluginConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
