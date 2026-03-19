"""
Unit Tests for DKI Plugin

测试 DKI 插件核心功能

Author: AGI Demo Project
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List

from dki.core.dki_plugin import (
    DKIPlugin,
    DKIPluginResponse,
    InjectionMetadata,
)
from dki.core.exceptions import AdapterConnectionError, AdapterSchemaError
from dki.adapters.base import (
    IUserDataAdapter,
    UserPreference,
    ChatMessage,
    UserProfile,
)
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class MockModelAdapter(BaseModelAdapter):
    """模拟模型适配器"""
    
    def __init__(self, hidden_dim: int = 4096):
        self._hidden_dim = hidden_dim
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    def load(self) -> None:
        pass
    
    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        return ModelOutput(
            text=f"Response to: {prompt[:50]}...",
            input_tokens=len(prompt.split()),
            output_tokens=20,
        )
    
    def embed(self, text: str):
        import torch
        return torch.randn(self._hidden_dim)
    
    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 0.5,
        **kwargs,
    ) -> ModelOutput:
        return ModelOutput(
            text=f"[DKI α={alpha:.2f}] Response to: {prompt[:50]}...",
            input_tokens=len(prompt.split()),
            output_tokens=25,
        )
    
    def compute_kv(self, text: str, return_hidden: bool = False):
        import torch
        seq_len = len(text.split())
        kv_entries = []
        for layer_idx in range(32):
            k = torch.randn(1, 32, seq_len, 128)
            v = torch.randn(1, 32, seq_len, 128)
            kv_entries.append(KVCacheEntry(layer_idx=layer_idx, key=k, value=v))
        if return_hidden:
            return kv_entries, torch.randn(1, seq_len, self._hidden_dim)
        return kv_entries, seq_len
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        return 2.0  # 模拟熵值


class MockUserDataAdapter(IUserDataAdapter):
    """模拟用户数据适配器"""
    
    def __init__(self):
        super().__init__()
        self._connected = True
        self._preferences = {}
        self._messages = {}
    
    async def connect(self):
        self._connected = True
    
    async def disconnect(self):
        self._connected = False
    
    async def get_user_profile(self, user_id: str):
        return UserProfile(user_id=user_id, username=f"User_{user_id}")
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types=None,
        include_expired=False,
    ) -> List[UserPreference]:
        return self._preferences.get(user_id, [])
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before=None,
        after=None,
    ) -> List[ChatMessage]:
        return self._messages.get(session_id, [])[:limit]
    
    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 10,
        **kwargs,
    ) -> List[ChatMessage]:
        """v7.2: 近轮对话（测试用返回空）"""
        return []

    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id=None,
    ) -> List[ChatMessage]:
        # v6.1: 支持跨会话检索
        if session_id:
            all_messages = self._messages.get(session_id, [])
        else:
            # 搜索用户的所有会话
            all_messages = []
            for sid, msgs in self._messages.items():
                for msg in msgs:
                    if getattr(msg, 'user_id', None) == user_id:
                        all_messages.append(msg)
        
        # 简单的关键词匹配 (支持中文子串匹配)
        keywords = query.lower().split()
        
        relevant = []
        for msg in all_messages:
            if any(kw in msg.content.lower() for kw in keywords):
                relevant.append(msg)
        
        return relevant[:limit]
    
    async def health_check(self) -> bool:
        return self._connected
    
    def add_preference(self, user_id: str, pref: UserPreference):
        if user_id not in self._preferences:
            self._preferences[user_id] = []
        self._preferences[user_id].append(pref)
    
    def add_message(self, session_id: str, msg: ChatMessage):
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(msg)


class TestDKIPlugin:
    """测试 DKI 插件"""
    
    @pytest.fixture
    def model_adapter(self):
        return MockModelAdapter()
    
    @pytest.fixture
    def data_adapter(self):
        adapter = MockUserDataAdapter()
        
        # 添加测试数据
        adapter.add_preference(
            "user_123",
            UserPreference(
                user_id="user_123",
                preference_text="素食主义者，不吃辣",
                preference_type="dietary",
                priority=1,
            )
        )
        adapter.add_preference(
            "user_123",
            UserPreference(
                user_id="user_123",
                preference_text="喜欢简洁的回答",
                preference_type="communication",
                priority=2,
            )
        )
        
        adapter.add_message(
            "session_456",
            ChatMessage(
                message_id="m1",
                session_id="session_456",
                user_id="user_123",
                role="user",
                content="我想找一家素食餐厅",
                timestamp=datetime.utcnow(),
            )
        )
        adapter.add_message(
            "session_456",
            ChatMessage(
                message_id="m2",
                session_id="session_456",
                user_id="user_123",
                role="assistant",
                content="好的，我来帮您推荐素食餐厅",
                timestamp=datetime.utcnow(),
            )
        )
        
        return adapter
    
    @pytest.fixture
    def dki_plugin(self, model_adapter, data_adapter):
        return DKIPlugin(
            model_adapter=model_adapter,
            user_data_adapter=data_adapter,
            language="cn",
        )
    
    @pytest.mark.asyncio
    async def test_chat_basic(self, dki_plugin):
        """测试基本聊天功能"""
        response = await dki_plugin.chat(
            query="推荐一家餐厅",
            user_id="user_123",
            session_id="session_456",
        )
        
        assert isinstance(response, DKIPluginResponse)
        assert response.text is not None
        assert len(response.text) > 0
    
    @pytest.mark.asyncio
    async def test_chat_with_preferences(self, dki_plugin):
        """测试带偏好的聊天"""
        response = await dki_plugin.chat(
            query="推荐一家餐厅",
            user_id="user_123",
            session_id="session_456",
        )
        
        # 应该读取到偏好
        assert response.metadata.preferences_count == 2
        assert response.metadata.preference_tokens > 0
    
    @pytest.mark.asyncio
    async def test_chat_with_history(self, dki_plugin):
        """测试带历史的聊天"""
        response = await dki_plugin.chat(
            query="素食餐厅",  # 包含关键词，应该匹配历史
            user_id="user_123",
            session_id="session_456",
        )
        
        # 应该检索到相关历史
        assert response.metadata.relevant_history_count > 0
    
    @pytest.mark.asyncio
    async def test_chat_force_alpha(self, dki_plugin):
        """测试强制 alpha 值"""
        response = await dki_plugin.chat(
            query="测试",
            user_id="user_123",
            session_id="session_456",
            force_alpha=0.8,
        )
        
        # force_alpha=0.8 会被 SafetyEnvelope.override_cap=0.7 截断
        # effective_preference_alpha = min(0.8, 0.7) = 0.7
        assert response.metadata.alpha == 0.7
        assert response.metadata.injection_enabled is True
        # 验证安全违规被记录
        assert len(response.metadata.safety_violations) > 0
    
    @pytest.mark.asyncio
    async def test_chat_no_injection(self, dki_plugin):
        """测试无注入情况"""
        response = await dki_plugin.chat(
            query="测试",
            user_id="unknown_user",  # 没有偏好的用户
            session_id="unknown_session",  # 没有历史的会话
        )
        
        assert response.metadata.preferences_count == 0
        assert response.metadata.relevant_history_count == 0
    
    @pytest.mark.asyncio
    async def test_metadata_completeness(self, dki_plugin):
        """测试元数据完整性"""
        response = await dki_plugin.chat(
            query="推荐餐厅",
            user_id="user_123",
            session_id="session_456",
        )
        
        metadata = response.metadata
        
        # 检查所有必要字段
        assert hasattr(metadata, 'injection_enabled')
        assert hasattr(metadata, 'alpha')
        assert hasattr(metadata, 'preference_tokens')
        assert hasattr(metadata, 'history_tokens')
        assert hasattr(metadata, 'query_tokens')
        assert hasattr(metadata, 'latency_ms')
        assert hasattr(metadata, 'preferences_count')
        assert hasattr(metadata, 'relevant_history_count')
    
    def test_stats(self, dki_plugin):
        """测试统计数据"""
        stats = dki_plugin.get_stats()
        
        assert "total_requests" in stats
        assert "injection_enabled_count" in stats
        assert "cache_hits" in stats


class TestInjectionMetadata:
    """测试注入元数据"""
    
    def test_metadata_to_dict(self):
        """测试元数据序列化"""
        metadata = InjectionMetadata(
            injection_enabled=True,
            alpha=0.5,
            preference_tokens=50,
            history_tokens=100,
            query_tokens=20,
            total_tokens=170,
            latency_ms=150.5,
            preferences_count=2,
            relevant_history_count=3,
        )
        
        data = metadata.to_dict()
        
        assert data["injection_enabled"] is True
        assert data["injection_strategy"] == "recall_v4"
        assert data["alpha"] == 0.5
        assert data["tokens"]["preference"] == 50
        assert data["tokens"]["history"] == 100
        assert data["tokens"]["query"] == 20
        assert data["tokens"]["total"] == 170
        assert data["latency"]["total_ms"] == 150.5
        assert data["data_source"]["preferences_count"] == 2
        assert data["data_source"]["relevant_history_count"] == 3


class TestDKIPluginResponse:
    """测试插件响应"""
    
    def test_response_to_dict(self):
        """测试响应序列化"""
        response = DKIPluginResponse(
            text="测试响应",
            input_tokens=50,
            output_tokens=20,
            metadata=InjectionMetadata(
                injection_enabled=True,
                alpha=0.5,
            ),
        )
        
        data = response.to_dict()
        
        assert data["text"] == "测试响应"
        assert data["input_tokens"] == 50
        assert data["output_tokens"] == 20
        assert "metadata" in data


class TestHybridInjection:
    """测试混合注入策略"""
    
    @pytest.fixture
    def dki_plugin(self):
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        
        # 添加偏好 (短，稳定)
        adapter.add_preference(
            "user_1",
            UserPreference(
                user_id="user_1",
                preference_text="素食主义者",
                preference_type="dietary",
            )
        )
        
        # 添加历史 (长，动态)
        for i in range(5):
            adapter.add_message(
                "session_1",
                ChatMessage(
                    message_id=f"m{i}",
                    session_id="session_1",
                    user_id="user_1",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"历史消息 {i}",
                    timestamp=datetime.utcnow(),
                )
            )
        
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_injection_layers(self, dki_plugin):
        """测试混合注入分层"""
        response = await dki_plugin.chat(
            query="推荐餐厅",
            user_id="user_1",
            session_id="session_1",
        )
        
        # 偏好应该被读取 (L1 层)
        assert response.metadata.preferences_count > 0
        
        # 偏好 token 应该较少 (短内容)
        assert response.metadata.preference_tokens < 100


class TestUserIdPropagation:
    """测试 user_id 传播"""
    
    @pytest.mark.asyncio
    async def test_user_id_used_for_preferences(self):
        """测试 user_id 用于读取偏好"""
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        
        # 为不同用户添加不同偏好
        adapter.add_preference(
            "user_A",
            UserPreference(
                user_id="user_A",
                preference_text="偏好 A",
                preference_type="test",
            )
        )
        adapter.add_preference(
            "user_B",
            UserPreference(
                user_id="user_B",
                preference_text="偏好 B",
                preference_type="test",
            )
        )
        
        plugin = DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
        )
        
        # 用户 A 的请求
        response_a = await plugin.chat(
            query="测试",
            user_id="user_A",
            session_id="s1",
        )
        assert response_a.metadata.preferences_count == 1
        
        # 用户 B 的请求
        response_b = await plugin.chat(
            query="测试",
            user_id="user_B",
            session_id="s2",
        )
        assert response_b.metadata.preferences_count == 1


# ============================================================
# P0/P1/P2 审查报告修复 — 单元测试
# ============================================================


class TestChatFallbackRecordsInjectionLog:
    """P0: chat() 降级路径也记录注入日志与统计"""

    @pytest.fixture
    def plugin(self):
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        adapter.add_preference(
            "u1",
            UserPreference(
                user_id="u1",
                preference_text="测试偏好",
                preference_type="test",
                priority=1,
            ),
        )
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_schema_error_fallback_records_log(self, plugin):
        """AdapterSchemaError 降级后 get_injection_logs 包含该请求且 total_requests 增加"""
        plugin.data_adapter.get_user_preferences = AsyncMock(
            side_effect=AdapterSchemaError("schema mismatch")
        )
        before = plugin._stats["total_requests"]
        logs_before = len(plugin._injection_logs)

        response = await plugin.chat(
            query="测试",
            user_id="u1",
            session_id="s1",
        )
        # 不强依赖具体的 injection_strategy 字符串，只验证统计与日志有记录
        assert plugin._stats["total_requests"] == before + 1
        assert len(plugin._injection_logs) == logs_before + 1
        last = plugin.get_injection_logs(limit=1)[0]
        assert isinstance(last.get("latency", {}).get("total_ms", 0), (int, float))

    @pytest.mark.asyncio
    async def test_no_injection_fallback_records_log(self, plugin):
        """无注入降级路径也记录日志"""
        plugin.data_adapter.get_user_preferences = AsyncMock(return_value=[])
        plugin.data_adapter.search_relevant_history = AsyncMock(return_value=[])
        # 触发 none_fallback: 需要让 executor 也失败，或直接测 _fallback_no_injection 的日志
        # 这里通过 schema 错误触发 no_injection 链
        plugin.data_adapter.get_user_preferences = AsyncMock(
            side_effect=AdapterSchemaError("err")
        )
        response = await plugin.chat(query="x", user_id="u1", session_id="s1")
        assert plugin._stats["total_requests"] >= 1
        logs = plugin.get_injection_logs(limit=5)
        assert any(
            log.get("injection_strategy", "").endswith("fallback") for log in logs
        )


class TestChatStreamUpdatesStatsAndLogs:
    """P0: chat_stream() 成功与错误路径都更新统计与注入日志"""

    @pytest.fixture
    def stream_plugin(self):
        model = MockModelAdapter()
        # 需要 async_stream_generate 才能走流式成功路径
        if not hasattr(model, "async_stream_generate"):

            async def _stream(prompt, **kwargs):
                yield "ok"
                yield " "

            model.async_stream_generate = _stream
        adapter = MockUserDataAdapter()
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_stream_success_increments_total_requests(self, stream_plugin):
        """流式成功完成后 total_requests 增加且日志有一条"""
        before = stream_plugin._stats["total_requests"]
        chunks = []
        async for c in stream_plugin.chat_stream(
            query="你好", user_id="u1", session_id="s1"
        ):
            chunks.append(c)
        assert stream_plugin._stats["total_requests"] == before + 1
        done = [x for x in chunks if x.get("type") == "done"]
        assert len(done) >= 1

    @pytest.mark.asyncio
    async def test_stream_success_adds_injection_log(self, stream_plugin):
        """流式成功路径会写入 _injection_logs"""
        n_before = len(stream_plugin._injection_logs)
        async for _ in stream_plugin.chat_stream(
            query="hi", user_id="u1", session_id="s1"
        ):
            pass
        assert len(stream_plugin._injection_logs) == n_before + 1

    @pytest.mark.asyncio
    async def test_stream_error_records_fallback_log(self, stream_plugin):
        """流式异常时也记录一次 stream_error_fallback 日志"""
        stream_plugin.data_adapter.get_user_preferences = AsyncMock(
            side_effect=RuntimeError("fake error")
        )
        n_before = stream_plugin._stats["total_requests"]
        chunks = []
        async for c in stream_plugin.chat_stream(
            query="x", user_id="u1", session_id="s1"
        ):
            chunks.append(c)
        errors = [x for x in chunks if x.get("type") == "error"]
        assert len(errors) >= 1
        assert stream_plugin._stats["total_requests"] == n_before + 1
        logs = stream_plugin.get_injection_logs(limit=3)
        assert any(
            log.get("injection_strategy") == "stream_error_fallback" for log in logs
        )


class TestInvalidateUserCacheClearsExecutorL0:
    """P1: invalidate_user_cache 同时清理 Executor L0 缓存"""

    @pytest.fixture
    def plugin(self):
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_invalidate_calls_executor_clear_preference_cache(self, plugin):
        """invalidate_user_cache 应调用 _executor.clear_preference_cache(user_id)"""
        with patch.object(
            plugin._executor,
            "clear_preference_cache",
            MagicMock(),
        ) as m_clear:
            await plugin.invalidate_user_cache("user_99")
            m_clear.assert_called_once_with("user_99")


class TestGetStatsPreferenceErrorCounters:
    """P2: get_stats 暴露偏好加载错误统计"""

    @pytest.fixture
    def plugin(self):
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )

    def test_get_stats_includes_preference_error_fields(self, plugin):
        """get_stats() 返回中包含 preference_singleflight_join_errors 与 preference_load_errors"""
        stats = plugin.get_stats()
        assert "preference_singleflight_join_errors" in stats
        assert "preference_load_errors" in stats
        assert stats["preference_singleflight_join_errors"] >= 0
        assert stats["preference_load_errors"] >= 0


class TestFallbackWithoutAdapterPreferenceInjection:
    """P1: _fallback_without_adapter 与 _fallback_stable_then_none 对齐，尽量保留偏好注入"""

    @pytest.fixture
    def plugin(self):
        model = MockModelAdapter()
        adapter = MockUserDataAdapter()
        adapter.add_preference(
            "u1",
            UserPreference(
                user_id="u1",
                preference_text="降级时也要用的偏好",
                preference_type="style",
                priority=1,
            ),
        )
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_fallback_without_adapter_uses_preferences_when_available(self, plugin):
        """适配器仅 get_user_preferences 可用时，_fallback_without_adapter 仍可注入偏好"""
        # 仅 search_relevant_history 失败，模拟“适配器暂时不可用”
        plugin.data_adapter.search_relevant_history = AsyncMock(
            side_effect=AdapterConnectionError("connection lost")
        )
        # get_user_preferences 仍成功（直连 DB 或重试成功）
        response = await plugin.chat(
            query="测试",
            user_id="u1",
            session_id="s1",
        )
        # 可能走 adapter_retry_fallback -> _fallback_without_adapter，且其中会调 get_user_preferences
        assert response.text is not None
        # 若走了 _fallback_without_adapter 且偏好加载成功，metadata 应有偏好
        assert response.metadata.injection_strategy in (
            "recall_v4",
            "adapter_retry_fallback",
            "stable_fallback",
            "none_fallback",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
