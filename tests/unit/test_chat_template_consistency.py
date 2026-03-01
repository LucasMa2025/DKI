"""
单元测试: Chat Template 一致性验证

验证所有模型适配器和系统组件的 chat template 构造
与官方标准 (DKI/docs/标准chat模板.md) 一致:

1. DeepSeek/Qwen (ChatML):
   <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n

2. Llama 3.x:
   <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}\n
   <|start_header_id|>user<|end_header_id|>\n{user}\n
   <|start_header_id|>assistant<|end_header_id|>\n

3. Llama 2:
   [INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]

测试范围:
- _format_prompt: 各 adapter 的 prompt 格式化 (含 apply_chat_template 回退)
- _has_chat_template_tokens: 各 adapter 的已有标记检测
- generate / forward_with_kv_injection: 双重包装防护
- RAG system: _build_prompt 使用 apply_chat_template
- InjectionExecutor: _build_chat_template_prompt / _build_preference_prefix

Author: AGI Demo Project
"""

import os
import sys
import inspect
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional, List, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# Mock Tokenizer 辅助工厂
# ============================================================================

def make_chatml_tokenizer():
    """创建 ChatML (DeepSeek/Qwen) 风格的 mock tokenizer"""
    tok = MagicMock()

    def _apply(messages, add_generation_prompt=False, tokenize=True):
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        result = "\n".join(parts)
        if add_generation_prompt:
            result += "\n<|im_start|>assistant\n"
        return result

    tok.apply_chat_template = _apply
    tok.pad_token_id = 0
    return tok


def make_llama3_tokenizer():
    """创建 Llama 3.x 风格的 mock tokenizer"""
    tok = MagicMock()

    def _apply(messages, add_generation_prompt=False, tokenize=True):
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            parts.append(
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}"
            )
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "\n".join(parts)

    tok.apply_chat_template = _apply
    tok.pad_token_id = 0
    return tok


def make_no_template_tokenizer():
    """创建不支持 apply_chat_template 的 mock tokenizer"""
    tok = MagicMock()
    tok.pad_token_id = 0
    # 确保没有 apply_chat_template 属性
    del tok.apply_chat_template
    return tok


# ============================================================================
# DeepSeekAdapter 模板测试
# ============================================================================

class TestDeepSeekAdapterChatTemplate:
    """验证 DeepSeekAdapter 的 chat template 构造"""

    @pytest.fixture
    def adapter(self):
        """构造不加载实际模型的 DeepSeekAdapter"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        with patch.object(DeepSeekAdapter, '__init__', lambda self, **kw: None):
            a = DeepSeekAdapter.__new__(DeepSeekAdapter)
            a.model_name = "deepseek-chat"
            a.device = "cpu"
            a.tokenizer = make_chatml_tokenizer()
            a._is_loaded = True
            return a

    def test_format_prompt_user_only(self, adapter):
        """_format_prompt: 仅 user 消息"""
        result = adapter._format_prompt("Hello")
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result
        assert "<|im_start|>system" not in result

    def test_format_prompt_with_system(self, adapter):
        """_format_prompt: 带 system prompt"""
        result = adapter._format_prompt("Hello", system_prompt="Be helpful.")
        assert "<|im_start|>system\nBe helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result

    def test_format_prompt_fallback_without_apply_chat_template(self, adapter):
        """_format_prompt: tokenizer 不支持 apply_chat_template 时回退到手动 ChatML"""
        adapter.tokenizer = make_no_template_tokenizer()
        result = adapter._format_prompt("Hello", system_prompt="Sys")
        assert "<|im_start|>system\nSys<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant" in result

    def test_has_chat_template_tokens_chatml(self, adapter):
        """_has_chat_template_tokens: 检测 ChatML 标记"""
        assert adapter._has_chat_template_tokens("<|im_start|>user\nHello<|im_end|>")
        assert not adapter._has_chat_template_tokens("Hello, how are you?")

    def test_has_chat_template_tokens_llama3(self, adapter):
        """_has_chat_template_tokens: 检测 Llama 3 标记"""
        assert adapter._has_chat_template_tokens("<|begin_of_text|>Hello")
        assert adapter._has_chat_template_tokens("<|start_header_id|>user<|end_header_id|>")

    def test_has_chat_template_tokens_llama2(self, adapter):
        """_has_chat_template_tokens: 检测 Llama 2 标记"""
        assert adapter._has_chat_template_tokens("[INST] Hello [/INST]")

    def test_double_wrapping_prevention_source(self):
        """验证 generate() 源码包含双重包装防护逻辑"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        source = inspect.getsource(DeepSeekAdapter.generate)
        assert "_has_chat_template_tokens" in source, (
            "DeepSeekAdapter.generate should check _has_chat_template_tokens "
            "to prevent double-wrapping"
        )

    def test_double_wrapping_prevention_forward(self):
        """验证 forward_with_kv_injection() 源码包含双重包装防护逻辑"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        source = inspect.getsource(DeepSeekAdapter.forward_with_kv_injection)
        assert "_has_chat_template_tokens" in source, (
            "DeepSeekAdapter.forward_with_kv_injection should check "
            "_has_chat_template_tokens to prevent double-wrapping"
        )


# ============================================================================
# LlamaAdapter 模板测试
# ============================================================================

class TestLlamaAdapterChatTemplate:
    """验证 LlamaAdapter 的 chat template 构造"""

    @pytest.fixture
    def llama3_adapter(self):
        """构造 Llama 3 Instruct adapter"""
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            a.device = "cpu"
            a.tokenizer = make_llama3_tokenizer()
            a._is_loaded = True
            return a

    @pytest.fixture
    def llama2_adapter(self):
        """构造 Llama 2 Chat adapter (tokenizer 无 apply_chat_template)"""
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Llama-2-7b-chat-hf"
            a.device = "cpu"
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            return a

    def test_is_chat_model_instruct(self, llama3_adapter):
        """_is_chat_model: Instruct 模型返回 True"""
        assert llama3_adapter._is_chat_model()

    def test_is_chat_model_chat(self, llama2_adapter):
        """_is_chat_model: Chat 模型返回 True"""
        assert llama2_adapter._is_chat_model()

    def test_is_llama3(self, llama3_adapter):
        """_is_llama3: Llama 3 模型返回 True"""
        assert llama3_adapter._is_llama3()

    def test_is_not_llama3(self, llama2_adapter):
        """_is_llama3: Llama 2 模型返回 False"""
        assert not llama2_adapter._is_llama3()

    def test_format_prompt_llama3_user_only(self, llama3_adapter):
        """_format_prompt: Llama 3 仅 user"""
        result = llama3_adapter._format_prompt("Hello")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_format_prompt_llama3_with_system(self, llama3_adapter):
        """_format_prompt: Llama 3 带 system"""
        result = llama3_adapter._format_prompt("Hello", system_prompt="Be helpful.")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "Be helpful." in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_format_prompt_llama2_fallback_user_only(self, llama2_adapter):
        """_format_prompt: Llama 2 回退模板 (仅 user)"""
        result = llama2_adapter._format_prompt("Hello")
        assert "[INST] Hello [/INST]" in result

    def test_format_prompt_llama2_fallback_with_system(self, llama2_adapter):
        """_format_prompt: Llama 2 回退模板 (带 system)"""
        result = llama2_adapter._format_prompt("Hello", system_prompt="Be safe.")
        assert "[INST]" in result
        assert "<<SYS>>" in result
        assert "Be safe." in result
        assert "<</SYS>>" in result
        assert "Hello" in result
        assert "[/INST]" in result

    def test_has_chat_template_tokens_llama3(self, llama3_adapter):
        """_has_chat_template_tokens: 检测 Llama 3 标记"""
        assert llama3_adapter._has_chat_template_tokens(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHi"
        )
        assert not llama3_adapter._has_chat_template_tokens("Hello")

    def test_has_chat_template_tokens_chatml(self, llama3_adapter):
        """_has_chat_template_tokens: 也能检测 ChatML 标记"""
        assert llama3_adapter._has_chat_template_tokens("<|im_start|>user\nHi<|im_end|>")

    def test_double_wrapping_prevention_source(self):
        """验证 generate() 源码包含双重包装防护逻辑"""
        from dki.models.llama_adapter import LlamaAdapter
        source = inspect.getsource(LlamaAdapter.generate)
        assert "_has_chat_template_tokens" in source

    def test_double_wrapping_prevention_forward(self):
        """验证 forward_with_kv_injection() 源码包含双重包装防护逻辑"""
        from dki.models.llama_adapter import LlamaAdapter
        source = inspect.getsource(LlamaAdapter.forward_with_kv_injection)
        assert "_has_chat_template_tokens" in source


# ============================================================================
# VLLMAdapter 模板测试
# ============================================================================

class TestVLLMAdapterChatTemplate:
    """验证 VLLMAdapter 的 chat template 构造"""

    @pytest.fixture
    def adapter(self):
        """构造不加载实际 vLLM 引擎的 VLLMAdapter"""
        from dki.models.vllm_adapter import VLLMAdapter
        with patch.object(VLLMAdapter, '__init__', lambda self, **kw: None):
            a = VLLMAdapter.__new__(VLLMAdapter)
            a.model_name = "Qwen/Qwen2.5-7B-Instruct"
            a.device = "cpu"
            a.tokenizer = make_chatml_tokenizer()
            a._is_loaded = True
            a.llm = MagicMock()
            return a

    def test_format_prompt_user_only(self, adapter):
        """_format_prompt: 仅 user"""
        result = adapter._format_prompt("Hello")
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result

    def test_format_prompt_with_system(self, adapter):
        """_format_prompt: 带 system"""
        result = adapter._format_prompt("Hello", system_prompt="Be safe.")
        assert "<|im_start|>system\nBe safe.<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result

    def test_format_prompt_fallback_chatml(self, adapter):
        """_format_prompt: apply_chat_template 失败时回退到手动 ChatML"""
        adapter.tokenizer = make_no_template_tokenizer()
        adapter.tokenizer.apply_chat_template = MagicMock(side_effect=Exception("fail"))
        # 由于 hasattr 检测会通过 (我们重新添加了 mock), 但调用抛异常
        result = adapter._format_prompt("Hello", system_prompt="Sys")
        assert "<|im_start|>system\nSys<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant" in result

    def test_has_chat_template_tokens(self, adapter):
        """_has_chat_template_tokens: 检测各种标记"""
        assert adapter._has_chat_template_tokens("<|im_start|>user\nHello")
        assert adapter._has_chat_template_tokens("<|begin_of_text|>Hello")
        assert adapter._has_chat_template_tokens("<|start_header_id|>user")
        assert adapter._has_chat_template_tokens("[INST] Hello [/INST]")
        assert not adapter._has_chat_template_tokens("Hello world")

    def test_is_chat_model_instruct(self, adapter):
        """_is_chat_model: Instruct 模型返回 True"""
        assert adapter._is_chat_model()

    def test_is_chat_model_base(self, adapter):
        """_is_chat_model: base 模型返回 False"""
        adapter.model_name = "Qwen/Qwen2.5-7B"
        assert not adapter._is_chat_model()

    def test_generate_uses_apply_chat_template_source(self):
        """验证 generate() 源码包含 chat template 检测"""
        from dki.models.vllm_adapter import VLLMAdapter
        source = inspect.getsource(VLLMAdapter.generate)
        assert "_has_chat_template_tokens" in source
        assert "_format_prompt" in source

    def test_forward_uses_apply_chat_template_source(self):
        """验证 forward_with_kv_injection() 源码包含 chat template 检测"""
        from dki.models.vllm_adapter import VLLMAdapter
        source = inspect.getsource(VLLMAdapter.forward_with_kv_injection)
        assert "_has_chat_template_tokens" in source
        assert "_format_prompt" in source


# ============================================================================
# GLMAdapter 模板测试 (已有正确实现, 验证不被破坏)
# ============================================================================

class TestGLMAdapterChatTemplate:
    """验证 GLMAdapter 的 chat template 构造 (已有正确实现)"""

    def test_build_chat_input_uses_apply_chat_template(self):
        """验证 _build_chat_input 源码使用 apply_chat_template"""
        from dki.models.glm_adapter import GLMAdapter
        source = inspect.getsource(GLMAdapter._build_chat_input)
        assert "apply_chat_template" in source, (
            "GLMAdapter._build_chat_input should use tokenizer.apply_chat_template"
        )


# ============================================================================
# RAG System 模板测试
# ============================================================================

class TestRAGSystemChatTemplate:
    """验证 RAGSystem._build_prompt 使用 apply_chat_template"""

    def test_build_prompt_uses_apply_chat_template_source(self):
        """验证 _build_prompt 源码使用 apply_chat_template"""
        from dki.core.rag_system import RAGSystem
        source = inspect.getsource(RAGSystem._build_prompt)
        assert "apply_chat_template" in source, (
            "RAGSystem._build_prompt should use tokenizer.apply_chat_template"
        )

    def test_build_prompt_constructs_messages_list(self):
        """验证 _build_prompt 源码构造 messages 列表"""
        from dki.core.rag_system import RAGSystem
        source = inspect.getsource(RAGSystem._build_prompt)
        assert '"role"' in source and '"system"' in source, (
            "RAGSystem._build_prompt should construct message dicts with role keys"
        )
        assert '"role": "user"' in source or "'role': 'user'" in source or (
            '"user"' in source and 'messages.append' in source
        ), (
            "RAGSystem._build_prompt should add user messages to messages list"
        )

    def test_build_prompt_has_fallback(self):
        """验证 _build_prompt 有回退逻辑"""
        from dki.core.rag_system import RAGSystem
        source = inspect.getsource(RAGSystem._build_prompt)
        assert "fallback" in source.lower() or "System:" in source, (
            "RAGSystem._build_prompt should have fallback format when "
            "apply_chat_template is not available"
        )

    @pytest.fixture
    def mock_rag_system(self):
        """构造 mock RAGSystem"""
        from dki.core.rag_system import RAGSystem
        with patch.object(RAGSystem, '__init__', lambda self, **kw: None):
            rag = RAGSystem.__new__(RAGSystem)
            mock_model = MagicMock()
            mock_model.tokenizer = make_chatml_tokenizer()
            # model 是 property, 需要设置底层 _model_adapter
            rag._model_adapter = mock_model
            rag._engine = None
            # 需要设置 _get_max_context_length 和 _estimate_tokens
            rag._get_max_context_length = lambda: 4096
            rag._estimate_tokens = lambda text: len(text.split())
            return rag

    def test_build_prompt_with_tokenizer(self, mock_rag_system):
        """_build_prompt: 有 tokenizer 时使用 apply_chat_template"""
        from dki.core.memory_router import MemorySearchResult
        memories = [
            MagicMock(content="Memory 1"),
            MagicMock(content="Memory 2"),
        ]

        prompt, info = mock_rag_system._build_prompt(
            query="What is DKI?",
            memories=memories,
            system_prompt="You are helpful.",
        )

        # 验证输出包含 ChatML 标记
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt
        # 验证内容
        assert "What is DKI?" in prompt
        assert "You are helpful." in prompt

    def test_build_prompt_fallback_without_tokenizer(self, mock_rag_system):
        """_build_prompt: tokenizer 无 apply_chat_template 时回退"""
        mock_rag_system._model_adapter.tokenizer = make_no_template_tokenizer()

        prompt, info = mock_rag_system._build_prompt(
            query="What is DKI?",
            memories=[],
            system_prompt="You are helpful.",
        )

        # v5.5: 回退格式已改为 ChatML (而非旧的 System:/User: 纯文本)
        assert "<|im_start|>" in prompt or "System:" in prompt or "User:" in prompt
        assert "What is DKI?" in prompt

    def test_build_prompt_with_history(self, mock_rag_system):
        """_build_prompt: 带对话历史"""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        prompt, info = mock_rag_system._build_prompt(
            query="How are you?",
            memories=[],
            history=history,
        )

        # 验证历史和当前查询都在 prompt 中
        assert "Hi" in prompt
        assert "Hello!" in prompt
        assert "How are you?" in prompt
        assert "<|im_start|>" in prompt  # 使用了 chat template


# ============================================================================
# InjectionExecutor 模板测试
# ============================================================================

class TestInjectionExecutorChatTemplate:
    """验证 InjectionExecutor 的 chat template 构造"""

    def test_build_preference_prefix_high_alpha(self):
        """_build_preference_prefix: alpha >= 0.7 → 严格遵循"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        result = InjectionExecutor._build_preference_prefix("user likes cats", 0.8)
        assert "严格遵循" in result
        assert "user likes cats" in result

    def test_build_preference_prefix_medium_alpha(self):
        """_build_preference_prefix: alpha >= 0.4 → 适当参考"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        result = InjectionExecutor._build_preference_prefix("user likes cats", 0.5)
        assert "适当参考" in result

    def test_build_preference_prefix_low_alpha(self):
        """_build_preference_prefix: alpha < 0.4 → 轻微参考"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        result = InjectionExecutor._build_preference_prefix("user likes cats", 0.2)
        assert "轻微参考" in result

    def test_build_preference_prefix_empty_text(self):
        """_build_preference_prefix: 空文本返回空字符串"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        assert InjectionExecutor._build_preference_prefix("", 0.8) == ""
        assert InjectionExecutor._build_preference_prefix("  ", 0.8) == ""

    @pytest.fixture
    def executor(self):
        """构造 mock InjectionExecutor"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        model = MagicMock()
        model.tokenizer = make_chatml_tokenizer()
        # Bypass full init
        with patch.object(InjectionExecutor, '__init__', lambda self, **kw: None):
            exe = InjectionExecutor.__new__(InjectionExecutor)
            exe.model = model
            return exe

    def test_build_chat_template_prompt_with_system(self, executor):
        """_build_chat_template_prompt: system + user → 完整 chat template"""
        result = executor._build_chat_template_prompt(
            system_content="请严格遵循以下用户偏好:\nuser likes cats",
            user_content="What pet should I get?",
        )
        assert "<|im_start|>system" in result
        assert "user likes cats" in result
        assert "<|im_start|>user" in result
        assert "What pet should I get?" in result
        assert "<|im_start|>assistant\n" in result

    def test_build_chat_template_prompt_no_system(self, executor):
        """_build_chat_template_prompt: 无 system → 仅 user message"""
        result = executor._build_chat_template_prompt(
            system_content="",
            user_content="Hello",
        )
        assert "<|im_start|>system" not in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\n" in result

    def test_build_chat_template_prompt_fallback(self, executor):
        """_build_chat_template_prompt: tokenizer 无 apply_chat_template 时回退"""
        executor.model.tokenizer = make_no_template_tokenizer()
        result = executor._build_chat_template_prompt(
            system_content="Be helpful.",
            user_content="Hello",
        )
        # 回退格式: system + "\n\n" + user
        assert "Be helpful." in result
        assert "Hello" in result

    def test_build_chat_template_prompt_llama3(self):
        """_build_chat_template_prompt: Llama 3 tokenizer"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        model = MagicMock()
        model.tokenizer = make_llama3_tokenizer()
        with patch.object(InjectionExecutor, '__init__', lambda self, **kw: None):
            exe = InjectionExecutor.__new__(InjectionExecutor)
            exe.model = model

        result = exe._build_chat_template_prompt(
            system_content="Follow preferences.",
            user_content="Hi",
        )
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "Follow preferences." in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hi" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_executor_vllm_path_uses_chat_template_source(self):
        """验证 vLLM 路径使用 _build_chat_template_prompt"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        source = inspect.getsource(InjectionExecutor._execute_with_kv_injection)
        assert "_build_chat_template_prompt" in source, (
            "InjectionExecutor._execute_with_kv_injection should use "
            "_build_chat_template_prompt for vLLM path"
        )

    def test_executor_stable_fallback_uses_chat_template_source(self):
        """验证 stable fallback 也使用 _build_chat_template_prompt"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        source = inspect.getsource(InjectionExecutor._execute_stable_fallback)
        assert "_build_chat_template_prompt" in source, (
            "InjectionExecutor._execute_stable_fallback should use "
            "_build_chat_template_prompt for vLLM path"
        )


# ============================================================================
# Cross-adapter 一致性测试
# ============================================================================

class TestCrossAdapterConsistency:
    """验证所有 adapter 的 chat template 处理一致性"""

    ADAPTERS_WITH_DOUBLE_WRAP_PREVENTION = [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.vllm_adapter", "VLLMAdapter"),
    ]

    @pytest.mark.parametrize("module_path,class_name", ADAPTERS_WITH_DOUBLE_WRAP_PREVENTION)
    def test_generate_has_double_wrap_prevention(self, module_path, class_name):
        """所有 adapter 的 generate 应包含双重包装防护"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        source = inspect.getsource(cls.generate)
        assert "_has_chat_template_tokens" in source or "chat_template" in source.lower(), (
            f"{class_name}.generate should check for existing chat template tokens"
        )

    @pytest.mark.parametrize("module_path,class_name", ADAPTERS_WITH_DOUBLE_WRAP_PREVENTION)
    def test_forward_has_double_wrap_prevention(self, module_path, class_name):
        """所有 adapter 的 forward_with_kv_injection 应包含双重包装防护"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        source = inspect.getsource(cls.forward_with_kv_injection)
        assert "_has_chat_template_tokens" in source or "chat_template" in source.lower(), (
            f"{class_name}.forward_with_kv_injection should check for "
            "existing chat template tokens"
        )

    ADAPTERS_WITH_FORMAT_PROMPT = [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter", "_format_prompt"),
        ("dki.models.llama_adapter", "LlamaAdapter", "_format_prompt"),
        ("dki.models.vllm_adapter", "VLLMAdapter", "_format_prompt"),
        ("dki.models.glm_adapter", "GLMAdapter", "_build_chat_input"),
    ]

    @pytest.mark.parametrize("module_path,class_name,method_name", ADAPTERS_WITH_FORMAT_PROMPT)
    def test_format_method_uses_apply_chat_template(self, module_path, class_name, method_name):
        """所有 adapter 的格式化方法应优先使用 apply_chat_template"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        method = getattr(cls, method_name)
        source = inspect.getsource(method)
        assert "apply_chat_template" in source, (
            f"{class_name}.{method_name} should use tokenizer.apply_chat_template"
        )

    @pytest.mark.parametrize("module_path,class_name,method_name", ADAPTERS_WITH_FORMAT_PROMPT)
    def test_format_method_has_fallback(self, module_path, class_name, method_name):
        """所有 adapter 的格式化方法应有回退逻辑"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        method = getattr(cls, method_name)
        source = inspect.getsource(method)
        # 应该有 except 或 else 分支作为回退
        has_fallback = (
            "except" in source
            or "else:" in source
            or "fallback" in source.lower()
        )
        assert has_fallback, (
            f"{class_name}.{method_name} should have a fallback when "
            "apply_chat_template is not available"
        )

    @pytest.mark.parametrize("module_path,class_name", ADAPTERS_WITH_DOUBLE_WRAP_PREVENTION)
    def test_has_chat_template_tokens_exists(self, module_path, class_name):
        """所有 adapter 应有 _has_chat_template_tokens 方法"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        assert hasattr(cls, '_has_chat_template_tokens'), (
            f"{class_name} should have _has_chat_template_tokens method"
        )


# ============================================================================
# 官方模板格式验证 (与 标准chat模板.md 对照)
# ============================================================================

class TestOfficialTemplateFormat:
    """验证手动回退模板与官方标准一致"""

    def test_deepseek_manual_template_format(self):
        """DeepSeek 手动模板应符合 ChatML 标准"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        with patch.object(DeepSeekAdapter, '__init__', lambda self, **kw: None):
            a = DeepSeekAdapter.__new__(DeepSeekAdapter)
            a.model_name = "deepseek-chat"
            a.tokenizer = make_no_template_tokenizer()

        result = a._format_prompt("你好", system_prompt="你是助手")

        # 标准格式: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
        expected_parts = [
            "<|im_start|>system\n你是助手<|im_end|>",
            "<|im_start|>user\n你好<|im_end|>",
            "<|im_start|>assistant",
        ]
        for part in expected_parts:
            assert part in result, f"Missing expected part: {part}"

    def test_llama3_manual_template_format(self):
        """Llama 3 手动模板应符合官方标准"""
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            a.tokenizer = make_no_template_tokenizer()

        result = a._format_prompt("你好", system_prompt="你是助手")

        # 标准格式:
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}\n
        # <|start_header_id|>user<|end_header_id|>\n{user}\n
        # <|start_header_id|>assistant<|end_header_id|>\n
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "你是助手" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "你好" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_llama2_manual_template_format(self):
        """Llama 2 手动模板应符合官方标准"""
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Llama-2-7b-chat-hf"
            a.tokenizer = make_no_template_tokenizer()

        result = a._format_prompt("你好", system_prompt="你是助手")

        # 标准格式: [INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]
        assert "[INST]" in result
        assert "<<SYS>>" in result
        assert "你是助手" in result
        assert "<</SYS>>" in result
        assert "你好" in result
        assert "[/INST]" in result

    def test_vllm_manual_template_fallback(self):
        """VLLMAdapter 手动回退模板应使用 ChatML 格式"""
        from dki.models.vllm_adapter import VLLMAdapter
        with patch.object(VLLMAdapter, '__init__', lambda self, **kw: None):
            a = VLLMAdapter.__new__(VLLMAdapter)
            a.model_name = "Qwen/Qwen2.5-7B-Instruct"
            tok = MagicMock()
            tok.apply_chat_template = MagicMock(side_effect=Exception("fail"))
            a.tokenizer = tok

        result = a._format_prompt("你好", system_prompt="你是助手")

        # ChatML 回退
        assert "<|im_start|>system\n你是助手<|im_end|>" in result
        assert "<|im_start|>user\n你好<|im_end|>" in result
        assert "<|im_start|>assistant" in result
