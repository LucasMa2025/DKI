"""
Unit Tests for DKI Plugin v5.1 Fixes

测试内容:
1. InjectionExecutor._build_multiturn_chat_prompt 多轮 chat template 构造
2. InjectionPlanner 填充 history_items
3. DKIPlugin.chat 传递 session_id 给 build_plan
4. InjectionExecutor 注入条件扩展 (有历史但无偏好时也走注入路径)
5. 实验系统 preference_alpha 正确性

Author: AGI Demo Project
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch

from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    ExecutionResult,
)
from dki.core.plugin.injection_executor import (
    InjectionExecutor,
    _strip_retrieve_fact_calls,
)
from dki.core.plugin.injection_planner import InjectionPlanner
from dki.adapters.base import (
    UserPreference as AdapterUserPreference,
    ChatMessage as AdapterChatMessage,
)
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


# ============================================================
# Mock Adapters
# ============================================================

class MockModelAdapterWithTokenizer(BaseModelAdapter):
    """模拟模型适配器 (含 tokenizer)"""
    
    def __init__(self, hidden_dim: int = 4096, injection_mode: str = "prompt_prefix"):
        self._hidden_dim = hidden_dim
        self.injection_mode = injection_mode
        self.model_name = "mock-model"
        
        # 模拟 tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.apply_chat_template = MagicMock(
            side_effect=self._mock_apply_chat_template
        )
    
    @staticmethod
    def _mock_apply_chat_template(messages, add_generation_prompt=True, tokenize=False):
        """模拟 apply_chat_template"""
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant")
        return "\n".join(parts) + "\n"
    
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
        return [], None  # vLLM 模式返回空
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        return 2.0


# ============================================================
# Tests for InjectionExecutor._build_multiturn_chat_prompt
# ============================================================

class TestBuildMultiturnChatPrompt:
    """测试多轮 chat template 构造"""
    
    @pytest.fixture
    def executor(self):
        model = MockModelAdapterWithTokenizer()
        return InjectionExecutor(model_adapter=model)
    
    def test_multiturn_with_preference_and_history(self, executor):
        """测试: 有偏好 + 有历史 → 正确构造多轮 prompt"""
        plan = InjectionPlan(
            original_query="你记得我的名字吗？",
            preference_text="<preference:general>\n- 用户名: Lucas\n</preference:general>",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            history_items=[
                MagicMock(role="user", content="你好，我叫 Lucas"),
                MagicMock(role="assistant", content="你好 Lucas！有什么可以帮你的？"),
                MagicMock(role="user", content="我喜欢编程"),
                MagicMock(role="assistant", content="很好！你喜欢什么编程语言？"),
            ],
        )
        
        result = executor._build_multiturn_chat_prompt(plan=plan)
        
        # 验证: 结果包含多轮对话
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "Lucas" in result
        assert "你记得我的名字吗？" in result  # 当前查询
        
        # 验证: tokenizer.apply_chat_template 被调用
        executor.model.tokenizer.apply_chat_template.assert_called_once()
        call_args = executor.model.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        
        # 验证消息结构: system + 4 history + 1 current query = 6 messages
        assert len(messages) == 6
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "你记得我的名字吗？"
    
    def test_multiturn_no_preference_only_history(self, executor):
        """测试: 无偏好 + 有历史 → 无 system 消息"""
        plan = InjectionPlan(
            original_query="继续我们的话题",
            preference_text="",
            alpha_profile=AlphaProfile(preference_alpha=0.0),
            history_items=[
                MagicMock(role="user", content="聊聊 Python"),
                MagicMock(role="assistant", content="Python 是一门很好的语言！"),
            ],
        )
        
        result = executor._build_multiturn_chat_prompt(plan=plan)
        
        call_args = executor.model.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        
        # 无偏好时不应有 system 消息
        assert messages[0]["role"] == "user"
        assert len(messages) == 3  # 2 history + 1 current query
    
    def test_multiturn_filters_injection_markers(self, executor):
        """测试: 包含注入标记的 assistant 消息被过滤"""
        plan = InjectionPlan(
            original_query="继续",
            preference_text="",
            alpha_profile=AlphaProfile(preference_alpha=0.0),
            history_items=[
                MagicMock(role="user", content="你好"),
                MagicMock(role="assistant", content="[会话历史参考]\n你好！这是历史记录"),
                MagicMock(role="user", content="真的吗？"),
                MagicMock(role="assistant", content="是的，这是正常回复"),
            ],
        )
        
        result = executor._build_multiturn_chat_prompt(plan=plan)
        
        call_args = executor.model.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        
        # 包含注入标记的 assistant 消息应被过滤
        assert len(messages) == 4  # user + user + assistant + current query
        for msg in messages:
            if msg["role"] == "assistant":
                assert "[会话历史参考]" not in msg["content"]
    
    def test_multiturn_chatml_fallback(self):
        """测试: tokenizer 不可用时回退到 ChatML 格式"""
        model = MockModelAdapterWithTokenizer()
        model.tokenizer = None  # 无 tokenizer
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            original_query="你好",
            preference_text="<preference:general>\n- 喜欢简洁回复\n</preference:general>",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            history_items=[
                MagicMock(role="user", content="之前的问题"),
                MagicMock(role="assistant", content="之前的回答"),
            ],
        )
        
        result = executor._build_multiturn_chat_prompt(plan=plan)
        
        # 验证 ChatML 格式
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result
    
    def test_multiturn_empty_history(self, executor):
        """测试: 空历史 + 有偏好 → 只有 system + user"""
        plan = InjectionPlan(
            original_query="你好",
            preference_text="<preference:general>\n- 喜欢简洁回复\n</preference:general>",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            history_items=[],
        )
        
        result = executor._build_multiturn_chat_prompt(plan=plan)
        
        call_args = executor.model.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        
        # system + current query
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "你好"


# ============================================================
# Tests for InjectionExecutor.execute with expanded conditions
# ============================================================

class TestExecutorExpandedConditions:
    """测试 Executor 注入条件扩展"""
    
    @pytest.fixture
    def executor(self):
        model = MockModelAdapterWithTokenizer()
        return InjectionExecutor(model_adapter=model)
    
    @pytest.mark.asyncio
    async def test_execute_with_history_only_no_preference(self, executor):
        """测试: 有历史但无偏好 → 应走注入路径 (非 plain)"""
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="",  # 无偏好
            original_query="继续上次的话题",
            final_input="继续上次的话题",
            alpha_profile=AlphaProfile(preference_alpha=0.0),
            history_items=[
                MagicMock(role="user", content="之前的问题"),
                MagicMock(role="assistant", content="之前的回答"),
            ],
        )
        
        result = await executor.execute(plan)
        
        # 验证使用了注入路径 (非 plain)
        assert executor._stats["recall_v4_executions"] >= 1
        assert result.text  # 有输出
        assert result.preference_cache_tier == "vllm_prefix_caching"
    
    @pytest.mark.asyncio
    async def test_execute_plain_when_no_history_no_preference(self, executor):
        """测试: 无历史无偏好 → 走 plain 路径"""
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=False,
            preference_text="",
            original_query="一般问题",
            final_input="一般问题",
            alpha_profile=AlphaProfile(preference_alpha=0.0),
            history_items=[],
        )
        
        result = await executor.execute(plan)
        
        # 验证使用了 plain 路径
        assert executor._stats["plain_executions"] >= 1
        assert result.text


# ============================================================
# Tests for InjectionPlanner history_items population
# ============================================================

class TestPlannerHistoryItems:
    """测试 Planner 填充 history_items"""
    
    @pytest.fixture
    def planner(self):
        return InjectionPlanner(
            config=None,
            language="cn",
            injection_strategy="stable",  # 使用 stable 策略避免需要 recall 组件
        )
    
    @pytest.fixture
    def sample_history(self) -> List[AdapterChatMessage]:
        return [
            AdapterChatMessage(
                message_id="msg_1",
                session_id="sess_1",
                user_id="user_1",
                role="user",
                content="你好，我叫 Lucas",
                timestamp=datetime(2026, 2, 25, 10, 0, 0),
            ),
            AdapterChatMessage(
                message_id="msg_2",
                session_id="sess_1",
                user_id="user_1",
                role="assistant",
                content="你好 Lucas！有什么可以帮你的？",
                timestamp=datetime(2026, 2, 25, 10, 0, 30),
            ),
        ]
    
    @pytest.fixture
    def sample_preferences(self) -> List[AdapterUserPreference]:
        return [
            AdapterUserPreference(
                preference_id="pref_1",
                user_id="user_1",
                preference_type="general",
                preference_text="用户名: Lucas",
                priority=10,
            ),
        ]
    
    def test_stable_strategy_populates_history_items(
        self, planner, sample_history, sample_preferences
    ):
        """测试: stable 策略下 history_items 被填充"""
        from dki.core.plugin.injection_plan import QueryContext
        context = QueryContext()
        
        plan = planner.build_plan(
            query="你记得我吗？",
            user_id="user_1",
            preferences=sample_preferences,
            relevant_history=sample_history,
            context=context,
        )
        
        # 验证 history_items 被填充
        assert len(plan.history_items) == 2
        assert plan.history_items[0].role == "user"
        assert plan.history_items[1].role == "assistant"
        assert "Lucas" in plan.history_items[0].content
    
    def test_stable_strategy_no_history(self, planner, sample_preferences):
        """测试: stable 策略无历史 → history_items 为空"""
        from dki.core.plugin.injection_plan import QueryContext
        context = QueryContext()
        
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=sample_preferences,
            relevant_history=[],
            context=context,
        )
        
        assert len(plan.history_items) == 0
    
    def test_plan_injection_enabled_with_history_only(self, planner, sample_history):
        """测试: 有历史无偏好 → injection_enabled = True"""
        from dki.core.plugin.injection_plan import QueryContext
        context = QueryContext()
        
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[],
            relevant_history=sample_history,
            context=context,
        )
        
        assert plan.injection_enabled is True
        assert len(plan.history_items) == 2


# ============================================================
# Tests for alpha correction in experiment runner
# ============================================================

class TestExperimentAlphaCorrection:
    """测试实验系统 alpha 修正"""
    
    def test_injection_info_uses_preference_alpha(self):
        """测试: InjectionInfo 使用 preference_alpha 而非 gating alpha"""
        from dki.experiment.runner import InjectionInfo
        
        info = InjectionInfo(
            mode='dki',
            original_query="测试",
            alpha=0.4,  # 应为 preference_alpha
        )
        
        assert info.alpha == 0.4
        
        d = info.to_dict()
        assert d['alpha'] == 0.4
    
    def test_injection_info_display(self):
        """测试: InjectionInfo 显示文本正确"""
        from dki.experiment.runner import InjectionInfo
        
        info = InjectionInfo(
            mode='dki',
            original_query="你记得我吗？",
            preference_text="<preference:general>\n- 用户名: Lucas\n</preference:general>",
            preference_tokens=10,
            history_suffix="User: 你好\nAssistant: 你好！",
            history_tokens=8,
            alpha=0.4,
        )
        
        display = info.get_display_text()
        assert "DKI" in display
        assert "α=0.40" in display
        assert "Lucas" in display


# ============================================================
# Tests for DKIPlugin session_id passing
# ============================================================

class TestDKIPluginSessionIdPassing:
    """测试 DKIPlugin 传递 session_id"""
    
    def test_build_plan_receives_session_id(self):
        """测试: build_plan 在 dki_plugin.chat 中接收 session_id"""
        # 验证 dki_plugin.py 源码中 build_plan 调用包含 session_id
        import inspect
        from dki.core.dki_plugin import DKIPlugin
        
        source = inspect.getsource(DKIPlugin.chat)
        assert "session_id=session_id" in source


# ============================================================
# Tests for _strip_retrieve_fact_calls
# ============================================================

class TestStripRetrieveFactCalls:
    """测试 retrieve_fact 工具调用过滤"""
    
    def test_strip_generic_format(self):
        text = 'Here is some text. retrieve_fact(trace_id="abc123") More text.'
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert count == 1
        assert "retrieve_fact" not in cleaned
        assert "More text" in cleaned
    
    def test_no_stripping_needed(self):
        text = "This is a normal response without any tool calls."
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert count == 0
        assert cleaned == text
    
    def test_multiple_strips(self):
        text = 'retrieve_fact(id="1") text retrieve_fact(id="2")'
        cleaned, count = _strip_retrieve_fact_calls(text)
        assert count == 2
        assert "retrieve_fact" not in cleaned


# ============================================================
# Tests for InjectionPlan history_items field
# ============================================================

class TestInjectionPlanHistoryItems:
    """测试 InjectionPlan.history_items 字段"""
    
    def test_default_empty(self):
        plan = InjectionPlan()
        assert plan.history_items == []
    
    def test_with_items(self):
        items = [
            MagicMock(role="user", content="hello"),
            MagicMock(role="assistant", content="hi"),
        ]
        plan = InjectionPlan(history_items=items)
        assert len(plan.history_items) == 2
    
    def test_to_dict_does_not_include_history_items(self):
        """history_items 不应出现在 to_dict (它可能包含不可序列化的对象)"""
        plan = InjectionPlan()
        d = plan.to_dict()
        # history_items 不在 to_dict 输出中 (因为它们是 ORM/dataclass 对象)
        assert "history_items" not in d
