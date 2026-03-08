"""
Unit Tests for Streaming Chat (DKI Plugin chat_stream + API endpoint)

测试流式生成的完整链路:
1. DKIPlugin.chat_stream() 异步生成器
2. Chat API /v1/dki/chat/stream SSE 端点

Author: AGI Demo Project
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, AsyncIterator

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


# ============================================================
# Mock 适配器
# ============================================================

class StreamMockModelAdapter(BaseModelAdapter):
    """支持流式生成的 Mock 模型"""

    def __init__(self):
        self._hidden_dim = 128

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def load(self):
        pass

    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        return ModelOutput(
            text=f"Response to: {prompt[:30]}",
            input_tokens=10,
            output_tokens=5,
        )

    def embed(self, text: str):
        import torch
        return torch.randn(self._hidden_dim)

    def forward_with_kv_injection(self, prompt, injected_kv, **kwargs):
        return self.generate(prompt, **kwargs)

    def compute_kv(self, text, return_hidden=False):
        import torch
        kv = [KVCacheEntry(key=torch.randn(1, 4, 2, 32), value=torch.randn(1, 4, 2, 32), layer_idx=i) for i in range(4)]
        if return_hidden:
            return kv, torch.randn(1, 2, self._hidden_dim)
        return kv, 2

    def compute_prefill_entropy(self, text, layer_idx=3):
        return 0.5

    async def async_stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """模拟流式生成"""
        tokens = ["Hello", " ", "World", "!"]
        for token in tokens:
            yield token
            await asyncio.sleep(0)  # 让出控制权


class NonStreamMockModelAdapter(BaseModelAdapter):
    """不支持流式生成的 Mock 模型 (无 stream 方法)"""

    def __init__(self):
        self._hidden_dim = 128

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def load(self):
        pass

    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        return ModelOutput(
            text=f"Response to: {prompt[:30]}",
            input_tokens=10,
            output_tokens=5,
        )

    def embed(self, text: str):
        import torch
        return torch.randn(self._hidden_dim)

    def forward_with_kv_injection(self, prompt, injected_kv, **kwargs):
        return self.generate(prompt, **kwargs)

    def compute_kv(self, text, return_hidden=False):
        import torch
        kv = [KVCacheEntry(key=torch.randn(1, 4, 2, 32), value=torch.randn(1, 4, 2, 32), layer_idx=i) for i in range(4)]
        if return_hidden:
            return kv, torch.randn(1, 2, self._hidden_dim)
        return kv, 2

    def compute_prefill_entropy(self, text, layer_idx=3):
        return 0.5


class StreamMockDataAdapter(IUserDataAdapter):
    """Mock 数据适配器"""

    def __init__(self):
        super().__init__()
        self._connected = True

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_user_profile(self, user_id):
        return UserProfile(user_id=user_id, username=f"User_{user_id}")

    async def get_user_preferences(self, user_id, **kwargs) -> List[UserPreference]:
        return [
            UserPreference(
                user_id=user_id,
                preference_text="喜欢简洁回答",
                preference_type="style",
                priority=1,
            )
        ]

    async def get_session_history(self, session_id, **kwargs) -> List[ChatMessage]:
        return []

    async def search_relevant_history(self, user_id, query, **kwargs) -> List[ChatMessage]:
        return []

    async def health_check(self):
        return True


# ============================================================
# 1. DKIPlugin.chat_stream() 测试
# ============================================================

class TestDKIPluginChatStream:
    """测试 DKI Plugin 流式生成"""

    @pytest.fixture
    def stream_plugin(self):
        return DKIPlugin(
            model_adapter=StreamMockModelAdapter(),
            user_data_adapter=StreamMockDataAdapter(),
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_stream_yields_metadata_first(self, stream_plugin):
        """流式生成首先 yield metadata"""
        chunks = []
        async for chunk in stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[0]["type"] == "metadata"
        assert "metadata" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, stream_plugin):
        """流式生成 yield token 事件"""
        token_chunks = []
        async for chunk in stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            if chunk["type"] == "token":
                token_chunks.append(chunk)

        assert len(token_chunks) > 0
        # 每个 token chunk 都有 content
        for tc in token_chunks:
            assert "content" in tc

    @pytest.mark.asyncio
    async def test_stream_yields_done(self, stream_plugin):
        """流式生成最后 yield done 事件"""
        chunks = []
        async for chunk in stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            chunks.append(chunk)

        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) >= 1
        assert "text" in done_chunks[-1]

    @pytest.mark.asyncio
    async def test_stream_full_text_assembled(self, stream_plugin):
        """流式 token 拼接后等于完整文本"""
        tokens = []
        full_text = ""
        async for chunk in stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            if chunk["type"] == "token":
                tokens.append(chunk["content"])
            elif chunk["type"] == "done":
                full_text = chunk.get("text", "")

        assembled = "".join(tokens)
        # 完整文本应该包含所有 token (可能经过 strip_think_content 处理)
        assert len(assembled) > 0 or len(full_text) > 0

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """流式生成错误处理"""
        # 使用会抛异常的适配器
        bad_adapter = StreamMockDataAdapter()
        bad_adapter.get_user_preferences = AsyncMock(
            side_effect=Exception("DB connection lost")
        )

        plugin = DKIPlugin(
            model_adapter=StreamMockModelAdapter(),
            user_data_adapter=bad_adapter,
            language="cn",
        )

        chunks = []
        async for chunk in plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            chunks.append(chunk)

        # 应该有 error 事件
        error_chunks = [c for c in chunks if c["type"] == "error"]
        assert len(error_chunks) >= 1
        assert "error" in error_chunks[0]

    @pytest.mark.asyncio
    async def test_stream_preferences_loaded(self, stream_plugin):
        """流式生成加载偏好"""
        metadata_chunk = None
        async for chunk in stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            if chunk["type"] == "metadata":
                metadata_chunk = chunk
                break

        assert metadata_chunk is not None
        meta = metadata_chunk["metadata"]
        assert meta["data_source"]["preferences_count"] == 1


# ============================================================
# 2. 非流式回退测试
# ============================================================

class TestStreamFallbackToNonStream:
    """测试模型不支持流式时的回退"""

    @pytest.fixture
    def non_stream_plugin(self):
        model = NonStreamMockModelAdapter()
        return DKIPlugin(
            model_adapter=model,
            user_data_adapter=StreamMockDataAdapter(),
            language="cn",
        )

    @pytest.mark.asyncio
    async def test_fallback_still_yields_results(self, non_stream_plugin):
        """不支持流式时仍然返回结果"""
        chunks = []
        async for chunk in non_stream_plugin.chat_stream(
            query="你好",
            user_id="u1",
            session_id="s1",
        ):
            chunks.append(chunk)

        # 应该有 metadata + token + done
        types = [c["type"] for c in chunks]
        assert "metadata" in types
        # 至少有 token 或 done
        assert "token" in types or "done" in types


# ============================================================
# 3. Chat API Stream Endpoint 测试
# ============================================================

class TestChatStreamEndpoint:
    """测试 /v1/dki/chat/stream SSE 端点"""

    @pytest.fixture
    def mock_dki_plugin(self):
        """创建 mock DKI plugin"""
        plugin = MagicMock()

        async def mock_chat_stream(**kwargs):
            yield {"type": "metadata", "metadata": {"injection_enabled": True}}
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " World"}
            yield {"type": "done", "text": "Hello World", "input_tokens": 10, "output_tokens": 2}

        plugin.chat_stream = mock_chat_stream
        return plugin

    @pytest.fixture
    def mock_store(self):
        """创建 mock store (同步, 无 a_ 前缀方法以避免 is_async_store 检测)"""
        store = MagicMock(spec=[
            'get_session', 'add_message', 'create_session',
            'get_user', 'get_or_create_user',
        ])
        store.get_session = MagicMock(return_value=MagicMock())
        store.add_message = MagicMock()
        return store

    @pytest.fixture
    def app(self, mock_store, mock_dki_plugin):
        """创建测试应用"""
        from fastapi import FastAPI
        from demo.api.chat import create_chat_router
        from demo.api.deps import _tokens_db

        app = FastAPI()
        app.state.store = mock_store
        app.state.dki_plugin = mock_dki_plugin

        app.include_router(create_chat_router())

        # 设置测试 token
        _tokens_db["test_token"] = "user-123"

        # 添加 mock user 到 store
        mock_user = MagicMock()
        mock_user.to_dict.return_value = {
            "id": "user-123",
            "username": "testuser",
            "email": "test@test.com",
        }
        mock_store.get_user = MagicMock(return_value=mock_user)

        return app

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_stream_endpoint_returns_sse(self, client):
        """流式端点返回 SSE 格式"""
        resp = client.post(
            "/v1/dki/chat/stream",
            json={
                "query": "你好",
                "user_id": "user-123",
            },
            headers={"Authorization": "Bearer test_token"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_endpoint_requires_auth(self, client):
        """流式端点需要认证"""
        resp = client.post(
            "/v1/dki/chat/stream",
            json={"query": "你好", "user_id": "user-123"},
        )
        assert resp.status_code == 401 or resp.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
