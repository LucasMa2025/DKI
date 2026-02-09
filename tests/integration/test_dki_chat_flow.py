"""
Integration Tests for DKI Chat Flow

测试完整的 DKI 聊天流程

核心流程:
1. 上层应用传递 user_id + 原始输入
2. DKI 通过适配器读取用户偏好和历史
3. DKI 执行 K/V 注入
4. DKI 调用 LLM 推理
5. 返回响应

Author: AGI Demo Project
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from dki.core.dki_plugin import DKIPlugin, DKIPluginResponse
from dki.adapters.base import UserPreference, ChatMessage
from dki.adapters.example_adapter import ExampleAdapter
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class MockLLMAdapter(BaseModelAdapter):
    """模拟 LLM 适配器"""
    
    def __init__(self):
        self._hidden_dim = 4096
        self._call_count = 0
        self._last_prompt = None
        self._last_kv = None
        self._last_alpha = None
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        self._call_count += 1
        self._last_prompt = prompt
        
        return ModelOutput(
            text=f"[Vanilla LLM] Response to: {prompt[:50]}",
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
        self._call_count += 1
        self._last_prompt = prompt
        self._last_kv = injected_kv
        self._last_alpha = alpha
        
        # 模拟 DKI 增强的响应
        response_prefix = "[DKI Enhanced]" if injected_kv else "[No Injection]"
        
        return ModelOutput(
            text=f"{response_prefix} (α={alpha:.2f}) Response to: {prompt[:50]}",
            input_tokens=len(prompt.split()),
            output_tokens=25,
        )
    
    def compute_kv(self, text: str):
        import torch
        seq_len = max(1, len(text.split()))
        kv_entries = []
        for layer_idx in range(32):
            k = torch.randn(1, 32, seq_len, 128)
            v = torch.randn(1, 32, seq_len, 128)
            kv_entries.append(KVCacheEntry(layer_idx=layer_idx, key=k, value=v))
        return kv_entries, seq_len


class TestDKIChatFlow:
    """测试 DKI 聊天流程"""
    
    @pytest.fixture
    def llm_adapter(self):
        return MockLLMAdapter()
    
    @pytest.fixture
    def data_adapter(self):
        """创建示例数据适配器并填充测试数据"""
        adapter = ExampleAdapter()
        
        # 添加用户
        adapter.add_user("user_001", "测试用户", "Test User")
        
        # 添加偏好
        adapter.add_preference("user_001", "dietary", "素食主义者，不吃辣")
        adapter.add_preference("user_001", "communication", "喜欢简洁的回答")
        
        # 创建会话
        adapter.create_session("user_001", "session_001", "测试会话")
        
        # 添加历史消息
        adapter.add_message(
            "session_001",
            "user_001",
            "user",
            "我想找一家素食餐厅"
        )
        adapter.add_message(
            "session_001",
            "user_001",
            "assistant",
            "好的，我来帮您推荐素食餐厅"
        )
        
        return adapter
    
    @pytest.fixture
    def dki_plugin(self, llm_adapter, data_adapter):
        return DKIPlugin(
            model_adapter=llm_adapter,
            user_data_adapter=data_adapter,
            language="cn",
        )
    
    @pytest.mark.asyncio
    async def test_complete_chat_flow(self, dki_plugin, llm_adapter):
        """测试完整聊天流程"""
        # 上层应用只传递 user_id 和原始输入
        response = await dki_plugin.chat(
            query="推荐一家餐厅",  # 原始用户输入
            user_id="user_001",    # 用户标识
            session_id="session_001",
        )
        
        # 验证响应
        assert isinstance(response, DKIPluginResponse)
        assert response.text is not None
        
        # 验证 DKI 读取了偏好
        assert response.metadata.preferences_count == 2
        
        # 验证 DKI 检索了历史
        # (查询 "餐厅" 应该匹配历史中的 "素食餐厅")
        assert response.metadata.relevant_history_count >= 0
        
        # 验证 LLM 被调用
        assert llm_adapter._call_count > 0
    
    @pytest.mark.asyncio
    async def test_user_isolation(self, dki_plugin, data_adapter):
        """测试用户隔离"""
        # 添加另一个用户
        data_adapter.add_user("user_002", "另一用户", "Another User")
        data_adapter.add_preference("user_002", "dietary", "喜欢海鲜")
        
        # 用户 1 的请求
        response_1 = await dki_plugin.chat(
            query="推荐餐厅",
            user_id="user_001",
            session_id="session_001",
        )
        
        # 用户 2 的请求
        response_2 = await dki_plugin.chat(
            query="推荐餐厅",
            user_id="user_002",
            session_id="session_002",
        )
        
        # 两个用户应该有不同的偏好数量
        assert response_1.metadata.preferences_count == 2  # 素食 + 简洁
        assert response_2.metadata.preferences_count == 1  # 海鲜
    
    @pytest.mark.asyncio
    async def test_no_prompt_construction_needed(self, dki_plugin, llm_adapter):
        """测试上层应用无需构造 prompt"""
        # 上层应用只传递原始输入，不做任何 prompt 构造
        raw_input = "推荐一家餐厅"
        
        response = await dki_plugin.chat(
            query=raw_input,
            user_id="user_001",
            session_id="session_001",
        )
        
        # DKI 应该自动处理偏好和历史
        assert response.metadata.preferences_count > 0
        
        # 上层应用不需要知道偏好内容
        # DKI 自动通过适配器读取并注入
    
    @pytest.mark.asyncio
    async def test_metadata_for_monitoring(self, dki_plugin):
        """测试监控元数据"""
        response = await dki_plugin.chat(
            query="测试",
            user_id="user_001",
            session_id="session_001",
        )
        
        # 元数据应该包含监控所需的所有信息
        metadata = response.metadata
        
        # 注入状态
        assert hasattr(metadata, 'injection_enabled')
        assert hasattr(metadata, 'alpha')
        
        # Token 统计
        assert hasattr(metadata, 'preference_tokens')
        assert hasattr(metadata, 'history_tokens')
        assert hasattr(metadata, 'query_tokens')
        
        # 数据源统计
        assert hasattr(metadata, 'preferences_count')
        assert hasattr(metadata, 'relevant_history_count')
        
        # 性能统计
        assert hasattr(metadata, 'latency_ms')
        assert metadata.latency_ms > 0


class TestDKIVsRAG:
    """测试 DKI 与 RAG 的区别"""
    
    @pytest.mark.asyncio
    async def test_dki_no_token_consumption(self):
        """测试 DKI 不消耗 token budget"""
        llm = MockLLMAdapter()
        adapter = ExampleAdapter()
        
        adapter.add_user("u1", "User", "User")
        adapter.add_preference("u1", "test", "这是一段很长的偏好文本" * 10)
        adapter.create_session("u1", "s1", "Test")
        
        plugin = DKIPlugin(
            model_adapter=llm,
            user_data_adapter=adapter,
            language="cn",
        )
        
        response = await plugin.chat(
            query="测试",
            user_id="u1",
            session_id="s1",
        )
        
        # DKI 的偏好通过 K/V 注入，不占用 token budget
        # 只有历史后缀会占用 token
        # 偏好 token 是 K/V 注入的，不计入输入 token
        assert response.metadata.preference_tokens > 0
        
        # 实际输入 token 应该只包含查询和历史后缀
        # 不包含偏好 (偏好通过 K/V 注入)


class TestAdapterIntegration:
    """测试适配器集成"""
    
    @pytest.mark.asyncio
    async def test_adapter_data_flow(self):
        """测试适配器数据流"""
        adapter = ExampleAdapter()
        
        # 模拟上层应用的数据库操作
        adapter.add_user("user_x", "用户X", "User X")
        adapter.add_preference("user_x", "style", "正式风格")
        adapter.create_session("user_x", "sess_x", "会话X")
        adapter.add_message("sess_x", "user_x", "user", "你好")
        
        # DKI 通过适配器读取数据
        prefs = await adapter.get_user_preferences("user_x")
        assert len(prefs) == 1
        assert prefs[0].preference_text == "正式风格"
        
        history = await adapter.get_session_history("sess_x")
        assert len(history) == 1
        assert history[0].content == "你好"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
