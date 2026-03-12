# -*- coding: utf-8 -*-
"""
单元测试: Native Tool Calls Fact Retrieval (闭源模型)

测试范围:
- RAGSystem._get_fact_retrieve_method
- RAGSystem._get_fact_call_config
- RAGSystem._prompt_to_messages
- RAGSystem._call_with_tools / _call_without_tools
- RAGSystem._retrieve_fact_for_rag
- RAGSystem._format_fact_result
- RAGSystem._generate_with_tool_calls
- RAGSystem.async_chat native_tool_calls 路由
- RAGSystem.chat_stream native_tool_calls 路由
- RETRIEVE_FACT_TOOL_DEF 常量
- ClosedSourceAdapter.async_generate_with_tools
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ============================================================
# Mock 类
# ============================================================

@dataclass
class MockModelOutput:
    """模拟 ModelOutput"""
    text: str = ""
    input_tokens: int = 100
    output_tokens: int = 50
    latency_ms: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockFactResponse:
    """模拟 FactResponse"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    trace_id: str = ""
    total_count: int = 0
    offset: int = 0
    has_more: bool = False


# ============================================================
# 测试: RETRIEVE_FACT_TOOL_DEF
# ============================================================

class TestRetrieveFactToolDef:
    """测试工具定义常量"""
    
    def test_tool_def_structure(self):
        """验证工具定义结构"""
        from dki.core.rag_system import RETRIEVE_FACT_TOOL_DEF
        
        assert RETRIEVE_FACT_TOOL_DEF["type"] == "function"
        assert "function" in RETRIEVE_FACT_TOOL_DEF
        
        func = RETRIEVE_FACT_TOOL_DEF["function"]
        assert func["name"] == "retrieve_fact"
        assert "description" in func
        assert "parameters" in func
    
    def test_tool_def_parameters(self):
        """验证参数定义"""
        from dki.core.rag_system import RETRIEVE_FACT_TOOL_DEF
        
        params = RETRIEVE_FACT_TOOL_DEF["function"]["parameters"]
        assert params["type"] == "object"
        assert "trace_id" in params["properties"]
        assert "offset" in params["properties"]
        assert "limit" in params["properties"]
        assert "trace_id" in params["required"]
    
    def test_tool_def_serializable(self):
        """工具定义可以 JSON 序列化"""
        from dki.core.rag_system import RETRIEVE_FACT_TOOL_DEF
        
        serialized = json.dumps(RETRIEVE_FACT_TOOL_DEF)
        deserialized = json.loads(serialized)
        assert deserialized == RETRIEVE_FACT_TOOL_DEF


# ============================================================
# 测试: _prompt_to_messages
# ============================================================

class TestPromptToMessages:
    """测试 prompt → messages 转换"""
    
    @staticmethod
    def _create_rag_system():
        """创建一个最小化的 RAGSystem 实例"""
        from dki.core.rag_system import RAGSystem
        
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag._model_adapter = MagicMock()
        rag._model_adapter.model_name = "test-model"
        rag._model_adapter.is_closed_source = True
        return rag
    
    def test_chatml_format(self):
        """ChatML 格式解析"""
        rag = self._create_rag_system()
        
        prompt = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        messages = rag._prompt_to_messages(prompt)
        
        assert len(messages) == 2  # system + user (assistant 没有 im_end 不匹配)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    def test_chatml_with_history(self):
        """ChatML 格式含历史消息"""
        rag = self._create_rag_system()
        
        prompt = (
            "<|im_start|>system\nSystem prompt<|im_end|>\n"
            "<|im_start|>user\nPrevious question<|im_end|>\n"
            "<|im_start|>assistant\nPrevious answer<|im_end|>\n"
            "<|im_start|>user\nCurrent question<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        messages = rag._prompt_to_messages(prompt)
        
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
    
    def test_plain_text_fallback(self):
        """纯文本回退为 user message"""
        rag = self._create_rag_system()
        
        prompt = "What is the meaning of life?"
        messages = rag._prompt_to_messages(prompt)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the meaning of life?"


# ============================================================
# 测试: _format_fact_result
# ============================================================

class TestFormatFactResult:
    """测试事实格式化"""
    
    @staticmethod
    def _create_rag_system():
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag._model_adapter = MagicMock()
        return rag
    
    def test_format_with_messages(self):
        """格式化有消息的 FactResponse"""
        rag = self._create_rag_system()
        
        fact_resp = MockFactResponse(
            messages=[
                {"role": "user", "content": "我昨天买了3本书"},
                {"role": "assistant", "content": "好的，记下了"},
            ],
            trace_id="msg_001",
            total_count=2,
        )
        
        result = rag._format_fact_result(fact_resp)
        assert "我昨天买了3本书" in result
        assert "好的，记下了" in result
    
    def test_format_empty_response(self):
        """格式化空 FactResponse"""
        rag = self._create_rag_system()
        
        result = rag._format_fact_result(None)
        assert "未找到" in result
    
    def test_format_empty_messages(self):
        """格式化无消息的 FactResponse"""
        rag = self._create_rag_system()
        
        fact_resp = MockFactResponse(messages=[], trace_id="msg_002")
        result = rag._format_fact_result(fact_resp)
        assert "未找到" in result
    
    def test_format_with_has_more(self):
        """格式化含 has_more 的 FactResponse"""
        rag = self._create_rag_system()
        
        fact_resp = MockFactResponse(
            messages=[{"role": "user", "content": "消息1"}],
            trace_id="msg_003",
            total_count=10,
            offset=0,
            has_more=True,
        )
        
        result = rag._format_fact_result(fact_resp)
        assert "还有更多" in result
        assert "msg_003" in result


# ============================================================
# 测试: _get_fact_retrieve_method (RAGSystem)
# ============================================================

class TestRAGGetFactRetrieveMethod:
    """测试 RAGSystem 的 fact_retrieve_method 获取"""
    
    @staticmethod
    def _create_rag_system(method="post_hoc", is_closed_source=False):
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        
        # Mock config
        config = MagicMock()
        fact_call = MagicMock()
        fact_call.fact_retrieve_method = method
        config.dki.recall.fact_call = fact_call
        rag.config = config
        
        # Mock model
        model = MagicMock()
        model.is_closed_source = is_closed_source
        model.model_name = "test-model"
        rag._model_adapter = model
        
        return rag
    
    def test_default_post_hoc(self):
        """默认返回 post_hoc"""
        rag = self._create_rag_system(method="post_hoc")
        assert rag._get_fact_retrieve_method() == "post_hoc"
    
    def test_native_tool_calls(self):
        """返回 native_tool_calls"""
        rag = self._create_rag_system(method="native_tool_calls")
        assert rag._get_fact_retrieve_method() == "native_tool_calls"
    
    def test_auto_closed_source(self):
        """auto + 闭源模型 → native_tool_calls"""
        rag = self._create_rag_system(method="auto", is_closed_source=True)
        assert rag._get_fact_retrieve_method() == "native_tool_calls"
    
    def test_auto_open_source(self):
        """auto + 开源模型 → inline_intercept"""
        rag = self._create_rag_system(method="auto", is_closed_source=False)
        assert rag._get_fact_retrieve_method() == "inline_intercept"
    
    def test_config_error_fallback(self):
        """配置错误时回退到 post_hoc"""
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag.config.dki = None  # 触发 AttributeError
        rag._model_adapter = MagicMock()
        
        assert rag._get_fact_retrieve_method() == "post_hoc"


# ============================================================
# 测试: _generate_with_tool_calls
# ============================================================

class TestGenerateWithToolCalls:
    """测试 native tool_calls 生成流程"""
    
    @staticmethod
    def _create_rag_system():
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        
        model = MagicMock()
        model.is_closed_source = True
        model.model_name = "gpt-4"
        rag._model_adapter = model
        
        # Mock db_manager
        rag.db_manager = MagicMock()
        
        return rag
    
    @pytest.mark.asyncio
    async def test_no_tool_calls_normal_response(self):
        """正常响应, 无 tool_calls"""
        rag = self._create_rag_system()
        
        # Mock _call_with_tools
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            return MockModelOutput(
                text="这是一个正常回答。",
                metadata={
                    "finish_reason": "stop",
                    "raw_message": {"content": "这是一个正常回答。"},
                },
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        clean_text, think_stripped, output = await rag._generate_with_tool_calls(
            prompt="测试问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "正常回答" in clean_text
    
    @pytest.mark.asyncio
    async def test_tool_call_triggered(self):
        """触发 tool_call → 检索事实 → 再次调用"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一轮: 返回 tool_calls
                return MockModelOutput(
                    text="",
                    metadata={
                        "finish_reason": "tool_calls",
                        "raw_message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_001",
                                "function": {
                                    "name": "retrieve_fact",
                                    "arguments": json.dumps({"trace_id": "msg_001"}),
                                },
                            }],
                        },
                    },
                )
            else:
                # 第二轮: 正常回答
                return MockModelOutput(
                    text="根据事实, 你昨天买了3本书。",
                    metadata={
                        "finish_reason": "stop",
                        "raw_message": {"content": "根据事实, 你昨天买了3本书。"},
                    },
                )
        
        async def mock_call_without_tools(messages, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="最终回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "最终回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._call_without_tools = mock_call_without_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        rag._retrieve_fact_for_rag = lambda tid, sid: MockFactResponse(
            messages=[{"role": "user", "content": "我昨天买了3本书"}],
            trace_id="msg_001",
            total_count=1,
        )
        rag._format_fact_result = lambda fr: "user: 我昨天买了3本书"
        
        clean_text, think_stripped, output = await rag._generate_with_tool_calls(
            prompt="我昨天买了什么?",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert call_count[0] == 2
        assert "3本书" in clean_text
    
    @pytest.mark.asyncio
    async def test_max_rounds_reached(self):
        """达到最大轮次"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            # 每轮都返回 tool_calls
            return MockModelOutput(
                text="",
                metadata={
                    "finish_reason": "tool_calls",
                    "raw_message": {
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{call_count[0]:03d}",
                            "function": {
                                "name": "retrieve_fact",
                                "arguments": json.dumps({"trace_id": f"msg_{call_count[0]:03d}"}),
                            },
                        }],
                    },
                },
            )
        
        async def mock_call_without_tools(messages, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="强制最终回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "强制最终回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._call_without_tools = mock_call_without_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        rag._retrieve_fact_for_rag = lambda tid, sid: MockFactResponse(
            messages=[{"role": "user", "content": "事实"}],
            trace_id=tid,
        )
        rag._format_fact_result = lambda fr: "事实内容"
        
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
            max_rounds=2,
        )
        
        # 2 轮 tool_calls + 1 轮 final
        assert call_count[0] == 2
        assert "强制最终回答" in clean_text
    
    @pytest.mark.asyncio
    async def test_fact_token_budget_exhausted(self):
        """事实 token 预算耗尽 → 最后一轮不带 tools"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="",
                    metadata={
                        "finish_reason": "tool_calls",
                        "raw_message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_001",
                                "function": {
                                    "name": "retrieve_fact",
                                    "arguments": json.dumps({"trace_id": "msg_001"}),
                                },
                            }],
                        },
                    },
                )
            return MockModelOutput(
                text="回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "回答"}},
            )
        
        without_tools_called = [False]
        
        async def mock_call_without_tools(messages, max_new_tokens, temperature, **kwargs):
            without_tools_called[0] = True
            return MockModelOutput(
                text="budget exhausted answer",
                metadata={"finish_reason": "stop", "raw_message": {"content": "budget exhausted answer"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._call_without_tools = mock_call_without_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        # 返回大量文本的事实 (每个字符约 0.5 token, 5000 字符 ≈ 2500 tokens)
        large_fact = " ".join(["word"] * 5000)  # 用空格分隔确保 token 估算足够大
        rag._retrieve_fact_for_rag = lambda tid, sid: MockFactResponse(
            messages=[{"role": "user", "content": large_fact}],
            trace_id=tid,
        )
        rag._format_fact_result = lambda fr: large_fact
        
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
            max_fact_tokens=10,  # 很小的预算 (10 tokens)
        )
        
        # 验证: 预算耗尽后调用了 _call_without_tools
        assert without_tools_called[0], "Should have called _call_without_tools after budget exhaustion"
        assert "budget exhausted answer" in clean_text
    
    @pytest.mark.asyncio
    async def test_api_error_fallback(self):
        """API 错误时降级到无 tools 调用"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            raise Exception("API error: tools not supported")
        
        async def mock_call_without_tools(messages, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="降级回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "降级回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._call_without_tools = mock_call_without_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "降级回答" in clean_text
    
    @pytest.mark.asyncio
    async def test_unknown_tool_name(self):
        """未知 tool name 被忽略"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="",
                    metadata={
                        "finish_reason": "tool_calls",
                        "raw_message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_001",
                                "function": {
                                    "name": "unknown_tool",
                                    "arguments": "{}",
                                },
                            }],
                        },
                    },
                )
            return MockModelOutput(
                text="正常回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "正常回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "正常回答" in clean_text
    
    @pytest.mark.asyncio
    async def test_empty_trace_id(self):
        """空 trace_id 被跳过"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="",
                    metadata={
                        "finish_reason": "tool_calls",
                        "raw_message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_001",
                                "function": {
                                    "name": "retrieve_fact",
                                    "arguments": json.dumps({"trace_id": ""}),
                                },
                            }],
                        },
                    },
                )
            return MockModelOutput(
                text="回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "回答" in clean_text
    
    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self):
        """无效 JSON 参数被处理"""
        rag = self._create_rag_system()
        
        call_count = [0]
        
        async def mock_call_with_tools(messages, max_new_tokens, temperature, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="",
                    metadata={
                        "finish_reason": "tool_calls",
                        "raw_message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_001",
                                "function": {
                                    "name": "retrieve_fact",
                                    "arguments": "invalid json{{{",
                                },
                            }],
                        },
                    },
                )
            return MockModelOutput(
                text="回答",
                metadata={"finish_reason": "stop", "raw_message": {"content": "回答"}},
            )
        
        rag._call_with_tools = mock_call_with_tools
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        # Should not raise, should handle gracefully
        clean_text, _, _ = await rag._generate_with_tool_calls(
            prompt="问题",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "回答" in clean_text


# ============================================================
# 测试: ClosedSourceAdapter.async_generate_with_tools
# ============================================================

class TestClosedSourceAdapterWithTools:
    """测试闭源模型适配器的 tools 支持"""
    
    def test_adapter_has_method(self):
        """适配器有 async_generate_with_tools 方法"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        assert hasattr(ClosedSourceAdapter, "async_generate_with_tools")
    
    @pytest.mark.asyncio
    async def test_generate_with_tools_normal(self):
        """带 tools 的正常生成"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        
        adapter = ClosedSourceAdapter.__new__(ClosedSourceAdapter)
        adapter.model_name = "gpt-4"
        adapter.api_key = "test-key"
        adapter.api_base = "http://test.api"
        adapter.timeout = 60
        adapter.max_retries = 1
        adapter._loaded = True
        
        # Mock _async_api_call
        async def mock_api_call(request_body):
            assert "tools" in request_body
            assert request_body["tools"][0]["function"]["name"] == "retrieve_fact"
            return {
                "choices": [{
                    "message": {
                        "content": "正常回答",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
                "model": "gpt-4",
            }
        
        adapter._async_api_call = mock_api_call
        adapter._make_output = lambda **kwargs: MockModelOutput(**kwargs)
        
        from dki.core.rag_system import RETRIEVE_FACT_TOOL_DEF
        
        output = await adapter.async_generate_with_tools(
            messages=[{"role": "user", "content": "测试"}],
            tools=[RETRIEVE_FACT_TOOL_DEF],
        )
        
        assert output.text == "正常回答"
    
    @pytest.mark.asyncio
    async def test_generate_with_tools_returns_tool_calls(self):
        """返回 tool_calls 的生成"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        
        adapter = ClosedSourceAdapter.__new__(ClosedSourceAdapter)
        adapter.model_name = "gpt-4"
        adapter._loaded = True
        
        async def mock_api_call(request_body):
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "retrieve_fact",
                                "arguments": '{"trace_id": "msg_001"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "model": "gpt-4",
            }
        
        adapter._async_api_call = mock_api_call
        adapter._make_output = lambda **kwargs: MockModelOutput(**kwargs)
        
        output = await adapter.async_generate_with_tools(
            messages=[{"role": "user", "content": "测试"}],
            tools=[{"type": "function", "function": {"name": "retrieve_fact"}}],
        )
        
        assert output.metadata["finish_reason"] == "tool_calls"
        assert "tool_calls" in output.metadata["raw_message"]


# ============================================================
# 测试: _call_with_tools / _call_without_tools
# ============================================================

class TestCallWithTools:
    """测试 RAGSystem 的 API 调用封装"""
    
    @staticmethod
    def _create_rag_system():
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        return rag
    
    @pytest.mark.asyncio
    async def test_call_with_tools_uses_adapter(self):
        """使用适配器的 async_generate_with_tools"""
        rag = self._create_rag_system()
        
        mock_adapter = MagicMock()
        mock_adapter.is_closed_source = True
        
        async def mock_gen(**kwargs):
            return MockModelOutput(text="OK")
        
        mock_adapter.async_generate_with_tools = mock_gen
        rag._model_adapter = mock_adapter
        
        result = await rag._call_with_tools(
            messages=[{"role": "user", "content": "test"}],
            max_new_tokens=2048,
            temperature=0.7,
            tools=[{"type": "function"}],
        )
        
        assert result.text == "OK"
    
    @pytest.mark.asyncio
    async def test_call_without_tools(self):
        """不带 tools 调用"""
        rag = self._create_rag_system()
        
        mock_adapter = MagicMock()
        
        async def mock_gen(**kwargs):
            assert kwargs.get("tools") is None
            return MockModelOutput(text="OK")
        
        mock_adapter.async_generate_with_tools = mock_gen
        rag._model_adapter = mock_adapter
        
        result = await rag._call_without_tools(
            messages=[{"role": "user", "content": "test"}],
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        assert result.text == "OK"


# ============================================================
# 测试: _get_fact_call_config (RAGSystem)
# ============================================================

class TestRAGGetFactCallConfig:
    """测试 RAGSystem 的 fact_call 配置获取"""
    
    def test_returns_config_from_dataclass(self):
        """从 dataclass 配置返回"""
        from dki.core.rag_system import RAGSystem
        from dki.core.recall.recall_config import RecallFactCallConfig
        
        rag = RAGSystem.__new__(RAGSystem)
        config = MagicMock()
        config.dki.recall.fact_call = RecallFactCallConfig(
            fact_retrieve_method="native_tool_calls",
            max_fact_tokens=1000,
        )
        rag.config = config
        
        result = rag._get_fact_call_config()
        assert isinstance(result, RecallFactCallConfig)
        assert result.fact_retrieve_method == "native_tool_calls"
        assert result.max_fact_tokens == 1000
    
    def test_returns_config_from_dict(self):
        """从 dict 配置返回"""
        from dki.core.rag_system import RAGSystem
        from dki.core.recall.recall_config import RecallFactCallConfig
        
        rag = RAGSystem.__new__(RAGSystem)
        config = MagicMock()
        config.dki.recall.fact_call = {
            "fact_retrieve_method": "native_tool_calls",
            "max_fact_tokens": 500,
        }
        rag.config = config
        
        result = rag._get_fact_call_config()
        assert isinstance(result, RecallFactCallConfig)
        assert result.fact_retrieve_method == "native_tool_calls"
    
    def test_returns_default_on_error(self):
        """配置错误时返回默认值"""
        from dki.core.rag_system import RAGSystem
        from dki.core.recall.recall_config import RecallFactCallConfig
        
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag.config.dki = None
        
        result = rag._get_fact_call_config()
        assert isinstance(result, RecallFactCallConfig)
        assert result.fact_retrieve_method == "post_hoc"


# ============================================================
# 测试: think content stripping
# ============================================================

class TestThinkContentStripping:
    """测试 _generate_with_tool_calls 中的 think 内容剥离"""
    
    @staticmethod
    def _create_rag_system():
        from dki.core.rag_system import RAGSystem
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = MagicMock()
        rag._model_adapter = MagicMock()
        rag._model_adapter.model_name = "test"
        rag._model_adapter.is_closed_source = True
        rag.db_manager = MagicMock()
        return rag
    
    @pytest.mark.asyncio
    async def test_think_content_stripped(self):
        """<think> 内容被正确剥离"""
        rag = self._create_rag_system()
        
        async def mock_call(messages, max_new_tokens, temperature, tools, **kwargs):
            return MockModelOutput(
                text="<think>内部推理</think>最终回答",
                metadata={
                    "finish_reason": "stop",
                    "raw_message": {"content": "<think>内部推理</think>最终回答"},
                },
            )
        
        rag._call_with_tools = mock_call
        rag._prompt_to_messages = lambda p: [{"role": "user", "content": p}]
        
        clean_text, think_stripped, _ = await rag._generate_with_tool_calls(
            prompt="测试",
            max_new_tokens=2048,
            temperature=0.7,
            session_id="session_1",
        )
        
        assert "内部推理" not in clean_text
        assert "最终回答" in clean_text
        assert think_stripped is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
