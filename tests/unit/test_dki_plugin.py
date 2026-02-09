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
    
    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        return ModelOutput(
            text=f"Response to: {prompt[:50]}...",
            input_tokens=len(prompt.split()),
            output_tokens=20,
        )
    
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
    
    def compute_kv(self, text: str):
        import torch
        seq_len = len(text.split())
        kv_entries = []
        for layer_idx in range(32):
            k = torch.randn(1, 32, seq_len, 128)
            v = torch.randn(1, 32, seq_len, 128)
            kv_entries.append(KVCacheEntry(layer_idx=layer_idx, key=k, value=v))
        return kv_entries, seq_len


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
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id=None,
    ) -> List[ChatMessage]:
        # 简单的关键词匹配
        all_messages = self._messages.get(session_id or user_id, [])
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
        
        assert response.metadata.alpha == 0.8
        assert response.metadata.injection_enabled is True
    
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
        
        assert data["injection"]["enabled"] is True
        assert data["injection"]["alpha"] == 0.5
        assert data["tokens"]["preference"] == 50
        assert data["tokens"]["history"] == 100
        assert data["performance"]["latency_ms"] == 150.5


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
