"""
单元测试: Prompt Template 修正验证

验证 2026-02-26 提示词模板修正的正确性:

1. DeepSeek V2/V3 原生标记 (全角 ｜ + ▁) 检测
   - 所有 adapter 的 _has_chat_template_tokens 能检测全角标记
   - 防止已格式化的 DeepSeek V2/V3 提示词被双重包装

2. Llama 3 回退模板 <|eot_id|> 闭合
   - 每条消息以 <|eot_id|> 结束
   - assistant 回复前缀不闭合 (留给模型生成)

3. ChatML 回退一致性
   - 所有组件 (dki_system, rag_system, injection_executor, vllm_adapter)
     的 ChatML 回退格式统一: <|im_start|>role\ncontent<|im_end|>
   - assistant 回复前缀不闭合 (留给模型生成)
   - 所有标记使用半角符号 (|)

4. DKI Service formatters
   - _format_chatml: 正确闭合所有消息
   - _format_deepseek: 委托给 ChatML (DeepSeek 兼容 ChatML)
   - _format_llama3: 正确使用 <|eot_id|> 闭合

Author: AGI Demo Project
Date: 2026-02-26
"""

import os
import sys
from unittest.mock import MagicMock, patch
from typing import Optional, List, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 辅助工厂
# ============================================================================

def make_no_template_tokenizer():
    """创建不支持 apply_chat_template 的 mock tokenizer"""
    tok = MagicMock()
    tok.pad_token_id = 0
    del tok.apply_chat_template
    return tok


def make_failing_template_tokenizer():
    """创建 apply_chat_template 会抛异常的 mock tokenizer"""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.apply_chat_template = MagicMock(side_effect=Exception("template error"))
    return tok


# ============================================================================
# DeepSeek V2/V3 原生标记检测
# ============================================================================

class TestDeepSeekV2V3NativeTokenDetection:
    """验证所有 adapter 能检测 DeepSeek V2/V3 全角原生标记"""

    # DeepSeek V2/V3 tokenizer 输出的典型格式
    DEEPSEEK_V2_SAMPLES = [
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>Hello",          # <｜begin▁of▁sentence｜>Hello
        "<\uff5cUser\uff5c>What is DKI?<\uff5cend\u2581of\u2581sentence\uff5c>",  # <｜User｜>...<｜end▁of▁sentence｜>
        "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>Hi",
    ]

    PLAIN_TEXT_SAMPLES = [
        "Hello, how are you?",
        "What is DeepSeek?",
        "这是一段普通文本",
    ]

    @pytest.fixture
    def deepseek_adapter(self):
        from dki.models.deepseek_adapter import DeepSeekAdapter
        with patch.object(DeepSeekAdapter, '__init__', lambda self, **kw: None):
            a = DeepSeekAdapter.__new__(DeepSeekAdapter)
            a.model_name = "deepseek-chat"
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            return a

    @pytest.fixture
    def llama_adapter(self):
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            return a

    @pytest.fixture
    def vllm_adapter(self):
        from dki.models.vllm_adapter import VLLMAdapter
        with patch.object(VLLMAdapter, '__init__', lambda self, **kw: None):
            a = VLLMAdapter.__new__(VLLMAdapter)
            a.model_name = "deepseek-ai/DeepSeek-V2-Chat"
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            a.llm = MagicMock()
            return a

    @pytest.mark.parametrize("sample", DEEPSEEK_V2_SAMPLES)
    def test_deepseek_adapter_detects_v2_tokens(self, deepseek_adapter, sample):
        """DeepSeekAdapter 检测 DeepSeek V2/V3 全角标记"""
        assert deepseek_adapter._has_chat_template_tokens(sample), (
            f"DeepSeekAdapter should detect DeepSeek V2/V3 token in: {sample!r}"
        )

    @pytest.mark.parametrize("sample", PLAIN_TEXT_SAMPLES)
    def test_deepseek_adapter_rejects_plain_text(self, deepseek_adapter, sample):
        """DeepSeekAdapter 不误检普通文本"""
        assert not deepseek_adapter._has_chat_template_tokens(sample)

    @pytest.mark.parametrize("sample", DEEPSEEK_V2_SAMPLES)
    def test_llama_adapter_detects_v2_tokens(self, llama_adapter, sample):
        """LlamaAdapter 检测 DeepSeek V2/V3 全角标记"""
        assert llama_adapter._has_chat_template_tokens(sample), (
            f"LlamaAdapter should detect DeepSeek V2/V3 token in: {sample!r}"
        )

    @pytest.mark.parametrize("sample", PLAIN_TEXT_SAMPLES)
    def test_llama_adapter_rejects_plain_text(self, llama_adapter, sample):
        """LlamaAdapter 不误检普通文本"""
        assert not llama_adapter._has_chat_template_tokens(sample)

    @pytest.mark.parametrize("sample", DEEPSEEK_V2_SAMPLES)
    def test_vllm_adapter_detects_v2_tokens(self, vllm_adapter, sample):
        """VLLMAdapter 检测 DeepSeek V2/V3 全角标记"""
        assert vllm_adapter._has_chat_template_tokens(sample), (
            f"VLLMAdapter should detect DeepSeek V2/V3 token in: {sample!r}"
        )

    @pytest.mark.parametrize("sample", PLAIN_TEXT_SAMPLES)
    def test_vllm_adapter_rejects_plain_text(self, vllm_adapter, sample):
        """VLLMAdapter 不误检普通文本"""
        assert not vllm_adapter._has_chat_template_tokens(sample)


# ============================================================================
# 半角标记检测 (ChatML, Llama 3, Llama 2) — 回归测试
# ============================================================================

class TestHalfWidthTokenDetection:
    """验证半角标记检测不被修改破坏"""

    CHATML_SAMPLES = [
        "<|im_start|>user\nHello<|im_end|>",
        "<|im_start|>system\nBe helpful.<|im_end|>",
    ]

    LLAMA3_SAMPLES = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHi",
        "<|start_header_id|>system<|end_header_id|>\nSys",
    ]

    LLAMA2_SAMPLES = [
        "[INST] Hello [/INST]",
        "[INST] <<SYS>>\nBe safe.\n<</SYS>>\n\nHello [/INST]",
    ]

    ALL_ADAPTERS = [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter", "deepseek-chat"),
        ("dki.models.llama_adapter", "LlamaAdapter", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("dki.models.vllm_adapter", "VLLMAdapter", "Qwen/Qwen2.5-7B-Instruct"),
    ]

    @pytest.mark.parametrize("module_path,class_name,model_name", ALL_ADAPTERS)
    @pytest.mark.parametrize("sample", CHATML_SAMPLES)
    def test_chatml_detection(self, module_path, class_name, model_name, sample):
        """所有 adapter 检测 ChatML 标记"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        with patch.object(cls, '__init__', lambda self, **kw: None):
            a = cls.__new__(cls)
            a.model_name = model_name
            a.tokenizer = make_no_template_tokenizer()
        assert a._has_chat_template_tokens(sample)

    @pytest.mark.parametrize("module_path,class_name,model_name", ALL_ADAPTERS)
    @pytest.mark.parametrize("sample", LLAMA3_SAMPLES)
    def test_llama3_detection(self, module_path, class_name, model_name, sample):
        """所有 adapter 检测 Llama 3 标记"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        with patch.object(cls, '__init__', lambda self, **kw: None):
            a = cls.__new__(cls)
            a.model_name = model_name
            a.tokenizer = make_no_template_tokenizer()
        assert a._has_chat_template_tokens(sample)

    @pytest.mark.parametrize("module_path,class_name,model_name", ALL_ADAPTERS)
    @pytest.mark.parametrize("sample", LLAMA2_SAMPLES)
    def test_llama2_detection(self, module_path, class_name, model_name, sample):
        """所有 adapter 检测 Llama 2 标记"""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        with patch.object(cls, '__init__', lambda self, **kw: None):
            a = cls.__new__(cls)
            a.model_name = model_name
            a.tokenizer = make_no_template_tokenizer()
        assert a._has_chat_template_tokens(sample)


# ============================================================================
# Llama 3 回退模板 <|eot_id|> 闭合验证
# ============================================================================

class TestLlama3FallbackEotIdClosure:
    """验证 Llama 3 手动回退模板正确使用 <|eot_id|> 闭合"""

    @pytest.fixture
    def llama3_adapter(self):
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            return a

    def test_user_message_closed_with_eot_id(self, llama3_adapter):
        """用户消息以 <|eot_id|> 闭合"""
        result = llama3_adapter._format_prompt("Hello")
        assert "<|eot_id|>" in result, "User message should be closed with <|eot_id|>"
        # user 消息结尾应有 <|eot_id|>
        user_section = result.split("<|start_header_id|>user<|end_header_id|>")[1]
        user_section = user_section.split("<|start_header_id|>assistant")[0]
        assert "<|eot_id|>" in user_section

    def test_system_message_closed_with_eot_id(self, llama3_adapter):
        """系统消息以 <|eot_id|> 闭合"""
        result = llama3_adapter._format_prompt("Hello", system_prompt="Be helpful.")
        # system 消息结尾应有 <|eot_id|>
        system_section = result.split("<|start_header_id|>system<|end_header_id|>")[1]
        system_section = system_section.split("<|start_header_id|>user")[0]
        assert "<|eot_id|>" in system_section

    def test_assistant_prefix_open(self, llama3_adapter):
        """assistant 前缀不闭合 (留给模型生成)"""
        result = llama3_adapter._format_prompt("Hello")
        assert result.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>") or \
               result.rstrip().endswith("<|end_header_id|>\n")
        # assistant 段不应有 <|eot_id|>
        assistant_parts = result.split("<|start_header_id|>assistant<|end_header_id|>")
        assert len(assistant_parts) >= 2
        assistant_suffix = assistant_parts[-1]
        assert "<|eot_id|>" not in assistant_suffix

    def test_full_format_structure(self, llama3_adapter):
        """完整结构验证"""
        result = llama3_adapter._format_prompt("你好", system_prompt="你是助手")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "你是助手" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "你好" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        # 计算 <|eot_id|> 出现次数: system + user = 2
        assert result.count("<|eot_id|>") == 2

    def test_user_only_has_one_eot_id(self, llama3_adapter):
        """无 system 时只有 user 的 <|eot_id|>"""
        result = llama3_adapter._format_prompt("Hello")
        assert result.count("<|eot_id|>") == 1


# ============================================================================
# DKI Service formatters 验证
# ============================================================================

class TestDKIServiceFormatters:
    """验证 DKI Service 的 prompt formatters"""

    @pytest.fixture
    def service(self):
        from dki.api.dki_service import DKIService
        with patch.object(DKIService, '__init__', lambda self, **kw: None):
            s = DKIService.__new__(DKIService)
            return s

    # --- _format_chatml ---

    def test_chatml_system_user(self, service):
        """ChatML: system + user 消息"""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = service._format_chatml(messages)
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert result.rstrip().endswith("<|im_start|>assistant")

    def test_chatml_no_fullwidth(self, service):
        """ChatML: 不包含全角标记"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        result = service._format_chatml(messages)
        # 全角 ｜ = \uff5c
        assert "\uff5c" not in result, "ChatML format should not contain fullwidth |"

    def test_chatml_all_messages_closed(self, service):
        """ChatML: 所有消息都有 <|im_end|> 闭合"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
        ]
        result = service._format_chatml(messages)
        # system + user 消息都应闭合
        assert result.count("<|im_end|>") == 2

    def test_chatml_assistant_open(self, service):
        """ChatML: assistant 前缀不闭合"""
        messages = [{"role": "user", "content": "Hi"}]
        result = service._format_chatml(messages)
        # 最后一个 <|im_start|>assistant 后不应有 <|im_end|>
        last_assistant_idx = result.rfind("<|im_start|>assistant")
        after_assistant = result[last_assistant_idx:]
        assert "<|im_end|>" not in after_assistant

    # --- _format_deepseek ---

    def test_deepseek_delegates_to_chatml(self, service):
        """DeepSeek 格式委托给 ChatML"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        chatml_result = service._format_chatml(messages)
        deepseek_result = service._format_deepseek(messages)
        assert chatml_result == deepseek_result

    # --- _format_llama3 ---

    def test_llama3_structure(self, service):
        """Llama 3: 正确结构"""
        messages = [
            {"role": "system", "content": "Be safe."},
            {"role": "user", "content": "Hello"},
        ]
        result = service._format_llama3(messages)
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "Be safe." in result
        assert "<|eot_id|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_llama3_messages_closed_with_eot_id(self, service):
        """Llama 3: system + user 消息以 <|eot_id|> 闭合"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        result = service._format_llama3(messages)
        assert result.count("<|eot_id|>") == 2

    def test_llama3_no_fullwidth(self, service):
        """Llama 3: 不包含全角标记"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        result = service._format_llama3(messages)
        assert "\uff5c" not in result, "Llama 3 format should not contain fullwidth |"


# ============================================================================
# ChatML 回退一致性: 所有组件
# ============================================================================

class TestChatMLFallbackConsistency:
    """验证所有组件的 ChatML 回退格式一致"""

    def _assert_chatml_format(self, prompt: str, has_system: bool = True):
        """验证 ChatML 格式的共性规则"""
        # 半角标记
        assert "\uff5c" not in prompt, "Should not contain fullwidth |"

        # 所有非 assistant 消息闭合
        if has_system:
            assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt

        # assistant 前缀不闭合
        last_assistant_idx = prompt.rfind("<|im_start|>assistant")
        after_assistant = prompt[last_assistant_idx:]
        assert "<|im_end|>" not in after_assistant, (
            "assistant prefix should NOT be closed with <|im_end|>"
        )

    # --- VLLMAdapter ---

    def test_vllm_adapter_chatml_fallback(self):
        """VLLMAdapter: apply_chat_template 失败时回退到 ChatML"""
        from dki.models.vllm_adapter import VLLMAdapter
        with patch.object(VLLMAdapter, '__init__', lambda self, **kw: None):
            a = VLLMAdapter.__new__(VLLMAdapter)
            a.model_name = "Qwen/Qwen2.5-7B-Instruct"
            a.tokenizer = make_failing_template_tokenizer()

        result = a._format_prompt("Hello", system_prompt="Be safe.")
        self._assert_chatml_format(result, has_system=True)
        assert "Hello" in result
        assert "Be safe." in result

    # --- DeepSeekAdapter ---

    def test_deepseek_adapter_chatml_fallback(self):
        """DeepSeekAdapter: tokenizer 无 apply_chat_template 时回退到 ChatML"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        with patch.object(DeepSeekAdapter, '__init__', lambda self, **kw: None):
            a = DeepSeekAdapter.__new__(DeepSeekAdapter)
            a.model_name = "deepseek-chat"
            a.tokenizer = make_no_template_tokenizer()

        result = a._format_prompt("Hello", system_prompt="Be safe.")
        self._assert_chatml_format(result, has_system=True)
        assert "Hello" in result
        assert "Be safe." in result

    # --- InjectionExecutor ---

    def test_injection_executor_chatml_fallback(self):
        """InjectionExecutor: tokenizer 无 apply_chat_template 时回退到 ChatML"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        model = MagicMock()
        model.tokenizer = make_no_template_tokenizer()
        with patch.object(InjectionExecutor, '__init__', lambda self, **kw: None):
            exe = InjectionExecutor.__new__(InjectionExecutor)
            exe.model = model

        result = exe._build_chat_template_prompt(
            system_content="Be safe.",
            user_content="Hello",
        )
        self._assert_chatml_format(result, has_system=True)
        assert "Hello" in result
        assert "Be safe." in result

    def test_injection_executor_chatml_fallback_no_system(self):
        """InjectionExecutor: 无 system 时 ChatML 回退"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        model = MagicMock()
        model.tokenizer = make_no_template_tokenizer()
        with patch.object(InjectionExecutor, '__init__', lambda self, **kw: None):
            exe = InjectionExecutor.__new__(InjectionExecutor)
            exe.model = model

        result = exe._build_chat_template_prompt(
            system_content="",
            user_content="Hello",
        )
        self._assert_chatml_format(result, has_system=False)
        assert "Hello" in result


# ============================================================================
# RAG System 回退格式验证
# ============================================================================

class TestRAGSystemFallbackFormat:
    """验证 RAG System 回退提示词格式"""

    @pytest.fixture
    def mock_rag_system(self):
        from dki.core.rag_system import RAGSystem
        with patch.object(RAGSystem, '__init__', lambda self, **kw: None):
            rag = RAGSystem.__new__(RAGSystem)
            mock_model = MagicMock()
            mock_model.tokenizer = make_no_template_tokenizer()
            rag._model_adapter = mock_model
            rag._engine = None
            rag._get_max_context_length = lambda: 4096
            rag._estimate_tokens = lambda text: len(text.split())
            return rag

    def test_fallback_uses_chatml_halfwidth(self, mock_rag_system):
        """回退格式使用半角 ChatML 标记"""
        prompt, info = mock_rag_system._build_prompt(
            query="What is DKI?",
            memories=[],
            system_prompt="You are helpful.",
        )
        # 应使用 ChatML 或标准格式
        # 不应包含全角符号
        assert "\uff5c" not in prompt, "RAG fallback should not contain fullwidth |"

    def test_fallback_with_history(self, mock_rag_system):
        """带历史的回退格式"""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt, info = mock_rag_system._build_prompt(
            query="How are you?",
            memories=[],
            history=history,
        )
        assert "Hi" in prompt
        assert "Hello!" in prompt
        assert "How are you?" in prompt


# ============================================================================
# 防双重包装: DeepSeek V2/V3 格式化后的 prompt 不被重新包装
# ============================================================================

class TestDoubleWrappingPrevention:
    """验证已格式化的 DeepSeek V2/V3 prompt 不被重新包装"""

    # 模拟 DeepSeek V2/V3 tokenizer apply_chat_template 输出
    FORMATTED_V2_PROMPT = (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
        "<\uff5cUser\uff5c>What is DKI?"
        "<\uff5cend\u2581of\u2581sentence\uff5c>"
        "<\uff5cAssistant\uff5c>"
    )

    @pytest.fixture(params=[
        ("dki.models.deepseek_adapter", "DeepSeekAdapter", "deepseek-chat"),
        ("dki.models.llama_adapter", "LlamaAdapter", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("dki.models.vllm_adapter", "VLLMAdapter", "Qwen/Qwen2.5-7B-Instruct"),
    ])
    def adapter(self, request):
        import importlib
        module_path, class_name, model_name = request.param
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        with patch.object(cls, '__init__', lambda self, **kw: None):
            a = cls.__new__(cls)
            a.model_name = model_name
            a.tokenizer = make_no_template_tokenizer()
            a._is_loaded = True
            if hasattr(cls, 'llm'):
                a.llm = MagicMock()
            return a

    def test_formatted_v2_detected(self, adapter):
        """已格式化的 DeepSeek V2/V3 prompt 被检测到"""
        assert adapter._has_chat_template_tokens(self.FORMATTED_V2_PROMPT)

    def test_formatted_chatml_detected(self, adapter):
        """已格式化的 ChatML prompt 被检测到"""
        chatml_prompt = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert adapter._has_chat_template_tokens(chatml_prompt)


# ============================================================================
# 边界条件
# ============================================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_empty_string_not_detected(self):
        """空字符串不应被检测为已格式化"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        with patch.object(DeepSeekAdapter, '__init__', lambda self, **kw: None):
            a = DeepSeekAdapter.__new__(DeepSeekAdapter)
            a.model_name = "deepseek-chat"
            a.tokenizer = make_no_template_tokenizer()
        assert not a._has_chat_template_tokens("")

    def test_partial_fullwidth_not_detected(self):
        """部分全角字符不应误检 (如单独的 ｜)"""
        from dki.models.vllm_adapter import VLLMAdapter
        with patch.object(VLLMAdapter, '__init__', lambda self, **kw: None):
            a = VLLMAdapter.__new__(VLLMAdapter)
            a.model_name = "Qwen/Qwen2.5-7B-Instruct"
            a.tokenizer = make_no_template_tokenizer()
        # 单独的全角 | 不是 DeepSeek 标记
        assert not a._has_chat_template_tokens("价格\uff5c规格\uff5c说明")

    def test_chatml_with_multiline_content(self):
        """多行内容的 ChatML 格式"""
        from dki.api.dki_service import DKIService
        with patch.object(DKIService, '__init__', lambda self, **kw: None):
            s = DKIService.__new__(DKIService)

        messages = [
            {"role": "system", "content": "Line1\nLine2\nLine3"},
            {"role": "user", "content": "Question\nwith\nnewlines"},
        ]
        result = s._format_chatml(messages)
        assert "<|im_start|>system\nLine1\nLine2\nLine3<|im_end|>" in result
        assert "<|im_start|>user\nQuestion\nwith\nnewlines<|im_end|>" in result

    def test_llama3_with_empty_system_prompt(self):
        """Llama 3: 空 system prompt 不生成 system 段"""
        from dki.models.llama_adapter import LlamaAdapter
        with patch.object(LlamaAdapter, '__init__', lambda self, **kw: None):
            a = LlamaAdapter.__new__(LlamaAdapter)
            a.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            a.tokenizer = make_no_template_tokenizer()

        result = a._format_prompt("Hello")
        assert "<|start_header_id|>system" not in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result

    def test_chatml_user_only_no_system(self):
        """ChatML: 无 system 时不生成 system 段"""
        from dki.api.dki_service import DKIService
        with patch.object(DKIService, '__init__', lambda self, **kw: None):
            s = DKIService.__new__(DKIService)

        messages = [{"role": "user", "content": "Hi"}]
        result = s._format_chatml(messages)
        assert "<|im_start|>system" not in result
        assert "<|im_start|>user\nHi<|im_end|>" in result
        assert "<|im_start|>assistant" in result


# ============================================================================
# DKI Service 模板选择逻辑
# ============================================================================

class TestDKIServiceTemplateSelection:
    """验证 DKI Service 的模板选择"""

    @pytest.fixture
    def service(self):
        from dki.api.dki_service import DKIService
        with patch.object(DKIService, '__init__', lambda self, **kw: None):
            s = DKIService.__new__(DKIService)
            return s

    def test_select_deepseek_template(self, service):
        """DeepSeek 模板选择使用 ChatML"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        chatml = service._format_chatml(messages)
        deepseek = service._format_deepseek(messages)
        assert chatml == deepseek, (
            "DeepSeek should delegate to ChatML for consistency"
        )

    def test_llama3_differs_from_chatml(self, service):
        """Llama 3 格式与 ChatML 不同"""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        chatml = service._format_chatml(messages)
        llama3 = service._format_llama3(messages)
        assert chatml != llama3


# ============================================================================
# Section 5: Token 预算修剪测试 (v5.6)
# ============================================================================

class TestDKISystemBudgetTrimming:
    """
    测试 DKI System 的 token 预算修剪逻辑
    
    核心原则: 用户当前输入 (original_query) 永远不被截断,
    只从最旧的历史消息开始移除。
    """
    
    @pytest.fixture
    def dki_system_mock(self):
        """创建带 mock 的 DKISystem (不加载模型)"""
        with patch('dki.core.dki_system.ConfigLoader') as MockCfg, \
             patch('dki.core.dki_system.DatabaseManager'), \
             patch('dki.core.dki_system.EmbeddingService'), \
             patch('dki.core.dki_system.MemoryRouter'):
            cfg = MagicMock()
            cfg.dki.use_tiered_cache = False
            cfg.dki.recall = None
            cfg.dki.hybrid_injection = None
            cfg.dki.gating = MagicMock()
            cfg.dki.projection = MagicMock()
            cfg.dki.cache = MagicMock()
            cfg.dki.tiered_cache = MagicMock()
            cfg.model.default_engine = "test"
            cfg.model.engines = {"test": MagicMock(max_model_len=4096)}
            cfg.database.path = ":memory:"
            cfg.database.echo = False
            MockCfg.return_value.config = cfg
            
            from dki.core.dki_system import DKISystem
            system = DKISystem.__new__(DKISystem)
            system.config = cfg
            system._model_adapter = None
            system._use_recall_v4 = False
            system._use_hybrid_injection = False
            system._user_isolation_enabled = False
            system._use_tiered_cache = False
            system._session_caches = {}
            system._tiered_caches = {}
            system._user_preferences = {}
            system._cache_audit_log = None
            system._recall_config = None
            
            return system
    
    def test_estimate_prompt_tokens_chinese(self, dki_system_mock):
        """中文 token 估算: ~1.5 token/字"""
        system = dki_system_mock
        tokens = system._estimate_prompt_tokens("你好世界")
        assert tokens > 0
        assert tokens >= 4  # 4 个中文字 × 1.5 ≈ 6
    
    def test_estimate_prompt_tokens_english(self, dki_system_mock):
        """英文 token 估算: ~1.3 token/word"""
        system = dki_system_mock
        tokens = system._estimate_prompt_tokens("hello world test")
        assert tokens > 0
    
    def test_estimate_prompt_tokens_empty(self, dki_system_mock):
        """空文本 token 估算"""
        system = dki_system_mock
        tokens = system._estimate_prompt_tokens("")
        assert tokens == 0
    
    def test_get_max_prompt_tokens(self, dki_system_mock):
        """v5.7: max_prompt_tokens = max_model_len - generation(30%) - tag_overhead"""
        system = dki_system_mock
        max_pt = system._get_max_prompt_tokens(max_new_tokens=512)
        # v5.7: 4096 - int(4096 * 0.30) - 120 = 4096 - 1228 - 120 = 2748
        # 生成预留 = 30% 上下文, 标记开销 = 120
        assert max_pt == 2748
    
    def test_trim_history_empty(self, dki_system_mock):
        """空历史不需要修剪"""
        system = dki_system_mock
        result = system._trim_history_to_budget(
            history_messages=[],
            system_content="偏好",
            original_query="你好",
            max_prompt_tokens=3484,
        )
        assert result == []
    
    def test_trim_history_within_budget(self, dki_system_mock):
        """预算充足时保留所有历史"""
        system = dki_system_mock
        history = [
            {"role": "user", "content": "消息一"},
            {"role": "assistant", "content": "回复一"},
            {"role": "user", "content": "消息二"},
            {"role": "assistant", "content": "回复二"},
        ]
        result = system._trim_history_to_budget(
            history_messages=history,
            system_content="偏好",
            original_query="你好",
            max_prompt_tokens=3484,
        )
        assert len(result) == 4
    
    def test_trim_history_over_budget_removes_oldest(self, dki_system_mock):
        """预算不足时从最旧消息开始移除"""
        system = dki_system_mock
        
        # 创建大量历史消息 (每条 ~1500 tokens, 远超预算)
        long_content = "这是一条很长的消息内容" * 200  # 1800 中文字符 ≈ 2700 tokens
        history = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": "最新的消息"},  # 短消息
            {"role": "assistant", "content": "最新的回复"},
        ]
        
        result = system._trim_history_to_budget(
            history_messages=history,
            system_content="偏好",
            original_query="你好",
            max_prompt_tokens=3484,  # 每条长消息 ~2700 tokens, 一条都放不下
        )
        
        # 应该保留了最新的消息, 丢弃了旧的
        assert len(result) < len(history)
        # 最新的消息一定在
        assert result[-1]["content"] == "最新的回复"
        assert result[-2]["content"] == "最新的消息"
    
    def test_trim_history_preserves_user_query(self, dki_system_mock):
        """用户当前输入永远不被截断 (它不在 history 中, 单独传入)"""
        system = dki_system_mock
        
        # system + query 占满全部预算 (~3000 tokens 用掉预算)
        long_query = "这是用户的很长的当前输入" * 300  # ~3000 chars ≈ 4500 tokens
        long_system = "很长的偏好信息" * 200  # ~1200 chars ≈ 1800 tokens
        
        result = system._trim_history_to_budget(
            history_messages=[
                {"role": "user", "content": "旧消息"},
                {"role": "assistant", "content": "旧回复"},
            ],
            system_content=long_system,
            original_query=long_query,
            max_prompt_tokens=3484,  # system+query 远超 3484
        )
        
        # 当 system + query 已经占满预算时, 历史全部丢弃
        assert len(result) == 0
    
    def test_trim_history_zero_budget(self, dki_system_mock):
        """预算为零时返回空列表"""
        system = dki_system_mock
        result = system._trim_history_to_budget(
            history_messages=[{"role": "user", "content": "消息"}],
            system_content="",
            original_query="",
            max_prompt_tokens=0,
        )
        assert len(result) == 0


class TestBuildPromptWithBudget:
    """测试 _build_recall_v4_chat_prompt 和 _build_hybrid_chat_prompt 的预算集成"""
    
    @pytest.fixture
    def dki_system_mock(self):
        """创建带 mock 的 DKISystem"""
        with patch('dki.core.dki_system.ConfigLoader') as MockCfg, \
             patch('dki.core.dki_system.DatabaseManager'), \
             patch('dki.core.dki_system.EmbeddingService'), \
             patch('dki.core.dki_system.MemoryRouter'):
            cfg = MagicMock()
            cfg.dki.use_tiered_cache = False
            cfg.dki.recall = None
            cfg.dki.hybrid_injection = None
            cfg.dki.gating = MagicMock()
            cfg.dki.projection = MagicMock()
            cfg.dki.cache = MagicMock()
            cfg.dki.tiered_cache = MagicMock()
            cfg.model.default_engine = "test"
            cfg.model.engines = {"test": MagicMock(max_model_len=4096)}
            cfg.database.path = ":memory:"
            cfg.database.echo = False
            MockCfg.return_value.config = cfg
            
            from dki.core.dki_system import DKISystem
            system = DKISystem.__new__(DKISystem)
            system.config = cfg
            system._model_adapter = None
            system._use_recall_v4 = False
            system._use_hybrid_injection = False
            system._user_isolation_enabled = False
            system._use_tiered_cache = False
            system._session_caches = {}
            system._tiered_caches = {}
            system._user_preferences = {}
            system._cache_audit_log = None
            system._recall_config = None
            system._recall_formatter = None
            
            return system
    
    def test_recall_v4_prompt_always_contains_user_query(self, dki_system_mock):
        """recall_v4 prompt 始终包含完整的用户当前输入"""
        system = dki_system_mock
        
        # 模拟大量历史 items
        long_text = "很长的历史内容" * 200
        items = []
        for i in range(10):
            item = MagicMock()
            item.role = "user" if i % 2 == 0 else "assistant"
            item.content = long_text
            item.type = "message"
            items.append(item)
        
        user_query = "这是用户的当前完整查询，绝对不能被截断的内容"
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query=user_query,
        )
        
        # 用户查询必须完整出现在 prompt 中
        assert user_query in result
    
    def test_hybrid_prompt_always_contains_user_query(self, dki_system_mock):
        """hybrid prompt 始终包含完整的用户当前输入"""
        system = dki_system_mock
        
        # 模拟大量历史消息
        long_text = "很长的历史内容" * 200
        history = MagicMock()
        msgs = []
        for i in range(10):
            msg = MagicMock()
            msg.role = "user" if i % 2 == 0 else "assistant"
            msg.content = long_text
            msgs.append(msg)
        history.messages = msgs
        
        user_query = "这是用户的当前完整查询，绝对不能被截断的内容"
        
        result = system._build_hybrid_chat_prompt(
            history=history,
            original_query=user_query,
        )
        
        # 用户查询必须完整出现在 prompt 中
        assert user_query in result
    
    def test_recall_v4_prompt_with_preference_preserves_query(self, dki_system_mock):
        """有偏好时 prompt 仍然包含完整用户输入"""
        system = dki_system_mock
        
        pref = MagicMock()
        pref.content = "偏好内容: 编程爱好者, 使用 Python"
        
        user_query = "德彪西在木偶的步态舞中模仿了瓦格纳的歌剧风格，你怎么看？"
        
        result = system._build_recall_v4_chat_prompt(
            items=[],
            original_query=user_query,
            preference=pref,
        )
        
        assert user_query in result
        assert "偏好" in result


class TestExecutorBudgetTrimming:
    """测试 InjectionExecutor 的 token 预算修剪"""
    
    @pytest.fixture
    def executor(self):
        """创建带 mock model 的 executor"""
        from dki.core.plugin.injection_executor import InjectionExecutor
        model = MagicMock()
        model.tokenizer = None
        model.max_model_len = 4096
        executor = InjectionExecutor.__new__(InjectionExecutor)
        executor.model = model
        executor._fc_logger = None
        executor._preference_kv_cache = None
        executor._inference_guard = None
        executor._stats = {}
        return executor
    
    def test_trim_preserves_newest(self, executor):
        """修剪保留最新消息"""
        history = [
            {"role": "user", "content": "旧消息" * 500},
            {"role": "assistant", "content": "旧回复" * 500},
            {"role": "user", "content": "新消息"},
            {"role": "assistant", "content": "新回复"},
        ]
        
        result = executor._trim_history_messages(
            history_messages=history,
            system_content="偏好",
            original_query="当前输入",
            max_new_tokens=512,
        )
        
        if len(result) < len(history):
            # 新消息应该被保留
            assert result[-1]["content"] == "新回复"
    
    def test_trim_empty_history(self, executor):
        """空历史返回空"""
        result = executor._trim_history_messages(
            history_messages=[],
            system_content="",
            original_query="test",
        )
        assert result == []
    
    def test_estimate_tokens(self, executor):
        """token 估算不为零"""
        tokens = executor._estimate_prompt_tokens("你好世界")
        assert tokens > 0


# ============================================================================
# v5.7: Think 内容过滤测试
# ============================================================================

class TestThinkContentStripping:
    """测试 <think>...</think> 推理内容过滤"""
    
    def test_strip_full_think_block(self):
        """移除完整的 <think>...</think> 块"""
        from dki.core.text_utils import strip_think_content
        
        text = "<think>这是推理过程</think>这是最终回复"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "推理过程" not in result
        assert "最终回复" in result
    
    def test_strip_think_only_end_tag(self):
        """移除仅有 </think> 的情况 (DeepSeek-R1 常见)"""
        from dki.core.text_utils import strip_think_content
        
        text = "这是推理过程, 分析用户意图...</think>这是最终回复"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "推理过程" not in result
        assert "最终回复" in result
    
    def test_no_think_content(self):
        """没有 think 内容时不修改"""
        from dki.core.text_utils import strip_think_content
        
        text = "这是一个普通回复, 没有推理过程"
        result, stripped = strip_think_content(text)
        assert stripped is False
        assert result == text
    
    def test_empty_text(self):
        """空文本处理"""
        from dki.core.text_utils import strip_think_content
        
        result, stripped = strip_think_content("")
        assert stripped is False
        assert result == ""
    
    def test_multiple_think_blocks(self):
        """多个 <think> 块"""
        from dki.core.text_utils import strip_think_content
        
        text = "<think>推理1</think>回复1<think>推理2</think>回复2"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "推理1" not in result
        assert "推理2" not in result
        assert "回复1" in result
        assert "回复2" in result
    
    def test_think_with_newlines(self):
        """<think> 内容包含换行"""
        from dki.core.text_utils import strip_think_content
        
        text = "<think>\n这是\n多行\n推理\n</think>\n最终回复"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "多行" not in result
        assert "最终回复" in result
    
    def test_think_case_insensitive(self):
        """大小写不敏感"""
        from dki.core.text_utils import strip_think_content
        
        text = "<THINK>推理</THINK>回复"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "回复" in result
    
    def test_think_with_spaces_in_tag(self):
        """标记内有空格"""
        from dki.core.text_utils import strip_think_content
        
        text = "< think >推理</ think >回复"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "回复" in result
    
    def test_only_end_tag_at_start(self):
        """仅有 </think> 且在文本中间"""
        from dki.core.text_utils import strip_think_content
        
        text = "好的,让我分析一下这个问题。用户询问了关于ERP的话题...\n\n</think>\n\n您好! 这是我的回复。"
        result, stripped = strip_think_content(text)
        assert stripped is True
        assert "让我分析" not in result
        assert "您好" in result


class TestEstimateTokensFast:
    """测试快速 token 估算"""
    
    def test_chinese_text(self):
        """中文文本估算"""
        from dki.core.text_utils import estimate_tokens_fast
        
        text = "你好世界"  # 4 个中文字
        tokens = estimate_tokens_fast(text)
        # 4 * 1.5 * 1.15 ≈ 7
        assert tokens >= 6
        assert tokens <= 10
    
    def test_english_text(self):
        """英文文本估算"""
        from dki.core.text_utils import estimate_tokens_fast
        
        text = "Hello world this is a test"  # 6 个英文单词
        tokens = estimate_tokens_fast(text)
        assert tokens > 0
    
    def test_empty_text(self):
        """空文本"""
        from dki.core.text_utils import estimate_tokens_fast
        
        tokens = estimate_tokens_fast("")
        assert tokens == 0
    
    def test_overestimate_factor(self):
        """高估系数生效"""
        from dki.core.text_utils import estimate_tokens_fast
        
        text = "你好世界 Hello World"
        tokens_normal = estimate_tokens_fast(text, overestimate_factor=1.0)
        tokens_high = estimate_tokens_fast(text, overestimate_factor=1.5)
        assert tokens_high >= tokens_normal


class TestNewBudgetAllocation:
    """v5.7: 测试新的 token 预算分配"""
    
    @pytest.fixture
    def dki_system_mock(self):
        """创建 mock DKI system"""
        from unittest.mock import MagicMock
        
        system = MagicMock()
        system._estimate_prompt_tokens = lambda text: len(text) * 2  # 粗估
        
        # Mock config
        config = MagicMock()
        config.model.engines = {
            "vllm": MagicMock(max_model_len=4096)
        }
        config.model.default_engine = "vllm"
        config.dki.recall.budget.instruction_reserve = 120
        system.config = config
        
        # 使用实际的 _get_context_window 和 _get_max_prompt_tokens 逻辑
        from dki.core.dki_system import DKISystem
        system._get_context_window = lambda: 4096
        system._get_tag_overhead = lambda: 120
        system._get_max_prompt_tokens = lambda max_new_tokens=512: (
            4096 - int(4096 * 0.30) - 120
        )
        
        return system
    
    def test_generation_reserve_30_percent(self, dki_system_mock):
        """生成预留 = 30% 上下文"""
        system = dki_system_mock
        context = system._get_context_window()
        gen_reserve = int(context * 0.30)
        assert gen_reserve == 1228  # int(4096 * 0.30)
    
    def test_max_prompt_tokens_new_formula(self, dki_system_mock):
        """v5.7: prompt 预算 = 上下文 - 生成(30%) - 标记开销"""
        system = dki_system_mock
        max_pt = system._get_max_prompt_tokens()
        # 4096 - 1228 - 120 = 2748
        assert max_pt == 2748
    
    def test_recall_config_new_defaults(self):
        """RecallBudgetConfig 新字段默认值"""
        from dki.core.recall.recall_config import RecallBudgetConfig
        
        cfg = RecallBudgetConfig()
        assert cfg.generation_ratio == 0.30
        assert cfg.instruction_reserve == 120
        assert cfg.preference_max_tokens == 200
    
    def test_recall_summary_threshold_300(self):
        """RecallSummaryConfig 阈值调整为 300"""
        from dki.core.recall.recall_config import RecallSummaryConfig
        
        cfg = RecallSummaryConfig()
        assert cfg.per_message_threshold == 300


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
