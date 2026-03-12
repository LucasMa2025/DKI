# -*- coding: utf-8 -*-
"""
单元测试: Inline Intercept Fact Retrieval (v7.1)

测试范围:
- InjectionExecutor._execute_inline_intercept
- InjectionExecutor._parse_trace_id
- InjectionExecutor._retrieve_fact
- InjectionExecutor._format_fact_for_continuation
- InjectionExecutor._get_fact_retrieve_method
- InjectionExecutor._get_fact_call_config
- DKIPlugin.chat_stream inline_intercept 路由
- RecallConfig 新字段解析
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

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


class MockRecallConfig:
    """模拟 RecallConfig"""
    def __init__(self, method="post_hoc", max_rounds=3, max_param_tokens=64, max_fact_tokens=800):
        self.fact_call = MockFactCallConfig(
            method=method,
            max_rounds=max_rounds,
            max_param_tokens=max_param_tokens,
            max_fact_tokens=max_fact_tokens,
        )


class MockFactCallConfig:
    """模拟 RecallFactCallConfig"""
    def __init__(self, method="post_hoc", max_rounds=3, max_param_tokens=64, max_fact_tokens=800):
        self.enabled = True
        self.max_rounds = max_rounds
        self.max_fact_tokens = max_fact_tokens
        self.batch_size = 5
        self.fact_retrieve_method = method
        self.inline_intercept_max_param_tokens = max_param_tokens
        self.inline_intercept_max_rounds = max_rounds


class MockFactRetriever:
    """模拟 FactRetriever"""
    def __init__(self, responses=None):
        self._responses = responses or {}
        self.call_count = 0
    
    def retrieve(self, trace_id, session_id, **kwargs):
        self.call_count += 1
        return self._responses.get(trace_id)


class MockPromptFormatter:
    """模拟 PromptFormatter"""
    def format_fact_segment(self, response):
        parts = [f"[FACT trace_id={response.trace_id}]"]
        for msg in response.messages:
            parts.append(f"  {msg.get('role', '')}: {msg.get('content', '')}")
        parts.append("[/FACT]")
        return "\n".join(parts)


class MockModelAdapter:
    """模拟 ModelAdapter"""
    def __init__(self, is_closed_source=False):
        self.is_closed_source = is_closed_source
        self.model_name = "test-model"
        self.max_model_len = 4096
        self._generate_calls = []
    
    async def async_generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        self._generate_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs,
        })
        return MockModelOutput(text="Test response")
    
    def generate(self, prompt, max_new_tokens=2048, temperature=0.7, **kwargs):
        self._generate_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs,
        })
        return MockModelOutput(text="Test response")
    
    def get_model_info(self):
        return {"model_name": self.model_name}


# ============================================================
# 测试: _parse_trace_id
# ============================================================

class TestParseTraceId:
    """测试 trace_id 解析的容错能力"""
    
    @staticmethod
    def _import_executor():
        """延迟导入, 避免模块级导入失败"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        return InjectionExecutor
    
    def test_key_value_double_quote(self):
        """trace_id="abc123" 格式"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id('trace_id="abc123"')
        assert result == "abc123"
    
    def test_key_value_single_quote(self):
        """trace_id='abc123' 格式"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id("trace_id='abc123'")
        assert result == "abc123"
    
    def test_key_value_no_quote(self):
        """trace_id=abc123 格式"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id("trace_id=abc123")
        assert result == "abc123"
    
    def test_key_value_with_spaces(self):
        """trace_id = "abc123" 格式 (有空格)"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id('trace_id = "abc123"')
        assert result == "abc123"
    
    def test_pure_string_double_quote(self):
        """纯字符串 "abc123" """
        Executor = self._import_executor()
        result = Executor._parse_trace_id('"abc123"')
        assert result == "abc123"
    
    def test_pure_string_single_quote(self):
        """纯字符串 'abc123' """
        Executor = self._import_executor()
        result = Executor._parse_trace_id("'abc123'")
        assert result == "abc123"
    
    def test_pure_id(self):
        """纯 ID abc123"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id("abc123")
        assert result == "abc123"
    
    def test_id_with_underscores(self):
        """带下划线的 ID msg_12345"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id('trace_id="msg_12345"')
        assert result == "msg_12345"
    
    def test_id_with_hyphens(self):
        """带连字符的 ID msg-12345"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id('trace_id="msg-12345"')
        assert result == "msg-12345"
    
    def test_empty_string(self):
        """空字符串"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id("")
        assert result is None
    
    def test_none(self):
        """None"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id(None)
        assert result is None
    
    def test_short_id_rejected(self):
        """过短的 ID (< 4 字符) 被拒绝"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id("ab")
        assert result is None
    
    def test_complex_format(self):
        """复杂格式: 带额外参数"""
        Executor = self._import_executor()
        result = Executor._parse_trace_id('trace_id="msg_001", offset=0, limit=5')
        assert result == "msg_001"


# ============================================================
# 测试: _get_fact_retrieve_method
# ============================================================

class TestGetFactRetrieveMethod:
    """测试 fact_retrieve_method 获取逻辑"""
    
    def test_default_post_hoc(self):
        """默认返回 post_hoc"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = None
        result = executor._get_fact_retrieve_method()
        assert result == "post_hoc"
    
    def test_inline_intercept_from_config(self):
        """从配置读取 inline_intercept"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="inline_intercept")
        result = executor._get_fact_retrieve_method()
        assert result == "inline_intercept"
    
    def test_auto_open_source(self):
        """auto + 开源模型 → entropy_gated (v8.0 更新)"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="auto")
        executor.model = MockModelAdapter(is_closed_source=False)
        result = executor._get_fact_retrieve_method()
        assert result == "entropy_gated"
    
    def test_auto_closed_source(self):
        """auto + 闭源模型 → native_tool_calls"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="auto")
        executor.model = MockModelAdapter(is_closed_source=True)
        result = executor._get_fact_retrieve_method()
        assert result == "native_tool_calls"


# ============================================================
# 测试: _get_fact_call_config
# ============================================================

class TestGetFactCallConfig:
    """测试 fact_call 配置获取"""
    
    def test_returns_config_when_available(self):
        """有配置时返回配置"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(method="inline_intercept", max_rounds=5)
        config = executor._get_fact_call_config()
        assert config.inline_intercept_max_rounds == 5
    
    def test_returns_default_when_no_config(self):
        """无配置时返回默认值"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        from dki.core.recall.recall_config import RecallFactCallConfig
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = None
        config = executor._get_fact_call_config()
        assert isinstance(config, RecallFactCallConfig)
        assert config.fact_retrieve_method == "post_hoc"


# ============================================================
# 测试: _retrieve_fact
# ============================================================

class TestRetrieveFact:
    """测试事实检索"""
    
    def test_retrieve_success(self):
        """成功检索事实"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        
        fact_resp = MockFactResponse(
            messages=[{"role": "user", "content": "Hello"}],
            trace_id="msg_001",
        )
        executor._fact_retriever = MockFactRetriever(responses={"msg_001": fact_resp})
        
        result = executor._retrieve_fact("msg_001", "session_1")
        assert result is not None
        assert result.trace_id == "msg_001"
    
    def test_retrieve_not_found(self):
        """未找到事实"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._fact_retriever = MockFactRetriever(responses={})
        
        result = executor._retrieve_fact("nonexistent", "session_1")
        assert result is None
    
    def test_retrieve_no_retriever(self):
        """未配置 fact_retriever"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._fact_retriever = None
        
        result = executor._retrieve_fact("msg_001", "session_1")
        assert result is None
    
    def test_retrieve_exception(self):
        """检索异常时返回 None"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("DB error")
        executor._fact_retriever = mock_retriever
        
        result = executor._retrieve_fact("msg_001", "session_1")
        assert result is None


# ============================================================
# 测试: _format_fact_for_continuation
# ============================================================

class TestFormatFactForContinuation:
    """测试事实格式化"""
    
    def test_with_prompt_formatter(self):
        """使用 PromptFormatter 格式化"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._prompt_formatter = MockPromptFormatter()
        
        fact_resp = MockFactResponse(
            messages=[{"role": "user", "content": "Hello world"}],
            trace_id="msg_001",
        )
        
        result = executor._format_fact_for_continuation(fact_resp)
        assert "[FACT trace_id=msg_001]" in result
        assert "Hello world" in result
    
    def test_fallback_format(self):
        """无 PromptFormatter 时使用回退格式"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._prompt_formatter = None
        
        fact_resp = MockFactResponse(
            messages=[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            trace_id="msg_002",
        )
        
        result = executor._format_fact_for_continuation(fact_resp)
        assert "[事实内容]" in result
        assert "Question" in result
        assert "Answer" in result
        assert "[/事实]" in result
    
    def test_formatter_exception_fallback(self):
        """PromptFormatter 异常时回退"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        executor = InjectionExecutor.__new__(InjectionExecutor)
        
        mock_formatter = MagicMock()
        mock_formatter.format_fact_segment.side_effect = Exception("Format error")
        executor._prompt_formatter = mock_formatter
        
        fact_resp = MockFactResponse(
            messages=[{"role": "user", "content": "Test"}],
            trace_id="msg_003",
        )
        
        result = executor._format_fact_for_continuation(fact_resp)
        assert "[事实内容]" in result
        assert "Test" in result


# ============================================================
# 测试: RecallConfig 新字段解析
# ============================================================

class TestRecallConfigNewFields:
    """测试 RecallConfig 新字段"""
    
    def test_fact_retrieve_method_default(self):
        """默认值为 post_hoc"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig()
        assert config.fact_retrieve_method == "post_hoc"
    
    def test_fact_retrieve_method_inline_intercept(self):
        """设置 inline_intercept"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig(fact_retrieve_method="inline_intercept")
        assert config.fact_retrieve_method == "inline_intercept"
    
    def test_fact_retrieve_method_native_tool_calls(self):
        """设置 native_tool_calls"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig(fact_retrieve_method="native_tool_calls")
        assert config.fact_retrieve_method == "native_tool_calls"
    
    def test_inline_intercept_max_param_tokens_default(self):
        """默认 max_param_tokens = 64"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig()
        assert config.inline_intercept_max_param_tokens == 64
    
    def test_inline_intercept_max_rounds_default(self):
        """默认 max_rounds = 3"""
        from dki.core.recall.recall_config import RecallFactCallConfig
        config = RecallFactCallConfig()
        assert config.inline_intercept_max_rounds == 3
    
    def test_from_dict_with_new_fields(self):
        """from_dict 解析新字段"""
        from dki.core.recall.recall_config import RecallConfig
        
        config_dict = {
            "fact_call": {
                "enabled": True,
                "max_rounds": 5,
                "max_fact_tokens": 1000,
                "fact_retrieve_method": "inline_intercept",
                "inline_intercept_max_param_tokens": 128,
                "inline_intercept_max_rounds": 5,
            }
        }
        
        config = RecallConfig.from_dict(config_dict)
        assert config.fact_call.fact_retrieve_method == "inline_intercept"
        assert config.fact_call.inline_intercept_max_param_tokens == 128
        assert config.fact_call.inline_intercept_max_rounds == 5
    
    def test_from_dict_without_new_fields(self):
        """from_dict 缺少新字段时使用默认值"""
        from dki.core.recall.recall_config import RecallConfig
        
        config_dict = {
            "fact_call": {
                "enabled": True,
                "max_rounds": 3,
            }
        }
        
        config = RecallConfig.from_dict(config_dict)
        assert config.fact_call.fact_retrieve_method == "post_hoc"
        assert config.fact_call.inline_intercept_max_param_tokens == 64


# ============================================================
# 测试: _execute_inline_intercept (异步)
# ============================================================

class TestExecuteInlineIntercept:
    """测试 inline_intercept 执行流程"""
    
    @pytest.fixture
    def setup_executor(self):
        """创建配置好的 executor"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor._recall_config = MockRecallConfig(
            method="inline_intercept",
            max_rounds=3,
            max_param_tokens=64,
            max_fact_tokens=800,
        )
        executor._fact_retriever = MockFactRetriever(responses={
            "msg_001": MockFactResponse(
                messages=[{"role": "user", "content": "原始消息内容"}],
                trace_id="msg_001",
                total_count=1,
            ),
        })
        executor._prompt_formatter = MockPromptFormatter()
        executor._fc_logger = None
        executor._stats = {}
        
        return executor
    
    @pytest.mark.asyncio
    async def test_no_fact_call_normal_end(self, setup_executor):
        """模型正常结束, 不触发 fact call"""
        executor = setup_executor
        
        # Mock _model_generate: 返回正常文本, finish_reason=eos
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="这是一个正常的回答。",
                metadata={"finish_reason": "eos"},
            )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        from dki.core.plugin.injection_executor import ExecutionResult
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        assert "正常的回答" in result.text
        assert executor._fact_retriever.call_count == 0
    
    @pytest.mark.asyncio
    async def test_fact_call_triggered(self, setup_executor):
        """模型触发 fact call → 检索事实 → 继续生成"""
        executor = setup_executor
        
        call_count = [0]
        
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            stop = kwargs.get("stop", [])
            
            if call_count[0] == 1:
                # Round A: 触发 stop string
                return MockModelOutput(
                    text="我需要查询一下事实: ",
                    metadata={"finish_reason": "stop"},
                )
            elif call_count[0] == 2:
                # Round B: 参数提取
                return MockModelOutput(
                    text='trace_id="msg_001"',
                    metadata={"finish_reason": "stop"},
                )
            else:
                # Round C: 继续生成
                return MockModelOutput(
                    text="根据事实, 答案是42。",
                    metadata={"finish_reason": "eos"},
                )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        # 验证事实被检索
        assert executor._fact_retriever.call_count == 1
        # 验证文本累积 (Round A + Round C)
        assert "我需要查询一下事实" in result.text
        assert "根据事实" in result.text
    
    @pytest.mark.asyncio
    async def test_max_rounds_limit(self, setup_executor):
        """达到 max_rounds 限制"""
        executor = setup_executor
        executor._recall_config = MockRecallConfig(
            method="inline_intercept",
            max_rounds=1,  # 只允许 1 轮
        )
        
        call_count = [0]
        
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return MockModelOutput(
                    text='trace_id="msg_001"' if call_count[0] == 2 else "查询: ",
                    metadata={"finish_reason": "stop"},
                )
            return MockModelOutput(
                text="最终回答",
                metadata={"finish_reason": "eos"},
            )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        # 1 轮后应该停止
        assert executor._fact_retriever.call_count <= 1
    
    @pytest.mark.asyncio
    async def test_trace_id_parse_failure(self, setup_executor):
        """trace_id 解析失败时中断"""
        executor = setup_executor
        
        call_count = [0]
        
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="需要查询: ",
                    metadata={"finish_reason": "stop"},
                )
            elif call_count[0] == 2:
                # 返回无法解析的参数
                return MockModelOutput(
                    text="",
                    metadata={"finish_reason": "stop"},
                )
            return MockModelOutput(
                text="fallback",
                metadata={"finish_reason": "eos"},
            )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        # 未触发检索
        assert executor._fact_retriever.call_count == 0
    
    @pytest.mark.asyncio
    async def test_fact_token_budget_exhausted(self, setup_executor):
        """事实 token 预算耗尽"""
        executor = setup_executor
        executor._recall_config = MockRecallConfig(
            method="inline_intercept",
            max_rounds=5,
            max_fact_tokens=10,  # 很小的预算
        )
        
        # 添加一个大事实
        executor._fact_retriever = MockFactRetriever(responses={
            "msg_001": MockFactResponse(
                messages=[{"role": "user", "content": "A" * 1000}],  # 大量文本
                trace_id="msg_001",
                total_count=1,
            ),
        })
        
        call_count = [0]
        
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockModelOutput(
                    text="查询: ",
                    metadata={"finish_reason": "stop"},
                )
            elif call_count[0] == 2:
                return MockModelOutput(
                    text='trace_id="msg_001"',
                    metadata={"finish_reason": "stop"},
                )
            return MockModelOutput(
                text="最终回答",
                metadata={"finish_reason": "eos"},
            )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        # 检索了 1 次但超预算后停止
        assert executor._fact_retriever.call_count == 1
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, setup_executor):
        """统计信息被正确更新"""
        executor = setup_executor
        
        async def mock_generate(prompt, max_new_tokens, temperature, **kwargs):
            return MockModelOutput(
                text="正常回答",
                metadata={"finish_reason": "eos"},
            )
        
        executor._model_generate = mock_generate
        
        plan = MagicMock()
        plan.session_id = "session_1"
        plan.has_fact_call_instruction = True
        
        result = await executor._execute_inline_intercept(
            plan=plan,
            prompt="test prompt",
            max_new_tokens=2048,
            temperature=0.7,
        )
        
        assert executor._stats.get("inline_intercept_executions", 0) == 1


# ============================================================
# 测试: _strip_retrieve_fact_calls 防御性拦截
# ============================================================

class TestStripRetrieveFactCalls:
    """测试残留 fact call 剥离"""
    
    def test_strip_residual_calls(self):
        """剥离残留的 retrieve_fact 调用"""
        try:
            from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
            
            text = '这是回答 retrieve_fact(trace_id="msg_001") 结束。'
            clean, count = _strip_retrieve_fact_calls(text)
            assert count >= 1
            assert "retrieve_fact" not in clean
        except ImportError:
            pytest.skip("_strip_retrieve_fact_calls not available")
    
    def test_no_residual_calls(self):
        """无残留调用时不变"""
        try:
            from dki.core.plugin.injection_executor import _strip_retrieve_fact_calls
            
            text = "这是一个正常的回答。"
            clean, count = _strip_retrieve_fact_calls(text)
            assert count == 0
            assert clean == text
        except ImportError:
            pytest.skip("_strip_retrieve_fact_calls not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
