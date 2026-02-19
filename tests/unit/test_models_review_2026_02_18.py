"""
单元测试: 2026-02-18 模型适配器审查修复验证

审查发现的问题:
1. DeepSeek/GLM/LLaMA — compute_kv 不 detach 也不移到 CPU (显存泄露)
2. DeepSeek/GLM/LLaMA — embed 不 detach 也不移到 CPU
3. DeepSeek/GLM/LLaMA — forward_with_kv_injection 无显存清理
4. DeepSeek/GLM/LLaMA — compute_kv 不清理 outputs 中间张量
5. KVCacheEntry.to_bytes — bfloat16 序列化问题
6. KVCacheEntry.from_bytes — bfloat16 反序列化映射到 float32

测试策略:
- 使用 inspect 源码分析验证代码模式
- 使用 Mock 对象验证行为
- 不需要 GPU 或真实模型
"""

import os
import sys
import inspect
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


# ============================================================================
# BUG 1: compute_kv 应 detach + cpu
# ============================================================================

class TestComputeKVDetachCPU:
    """验证所有 HuggingFace 适配器的 compute_kv 使用 detach().cpu()"""

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_compute_kv_has_detach_cpu(self, adapter_module, adapter_class):
        """compute_kv 源码应包含 .detach().cpu()"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.compute_kv)
        
        assert "detach().cpu()" in source, (
            f"{adapter_class}.compute_kv should use .detach().cpu() "
            f"to prevent GPU memory leak"
        )

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_compute_kv_deletes_outputs(self, adapter_module, adapter_class):
        """compute_kv 源码应包含 del outputs"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.compute_kv)
        
        assert "del outputs" in source, (
            f"{adapter_class}.compute_kv should delete outputs to free GPU memory"
        )

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_compute_kv_calls_empty_cache(self, adapter_module, adapter_class):
        """compute_kv 源码应包含 torch.cuda.empty_cache()"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.compute_kv)
        
        assert "empty_cache()" in source, (
            f"{adapter_class}.compute_kv should call torch.cuda.empty_cache()"
        )

    def test_vllm_adapter_already_has_detach_cpu(self):
        """vllm_adapter 的 compute_kv 应已经有 detach().cpu()"""
        from dki.models.vllm_adapter import VLLMAdapter
        source = inspect.getsource(VLLMAdapter.compute_kv)
        assert "detach().cpu()" in source


# ============================================================================
# BUG 2: embed 应 detach + cpu
# ============================================================================

class TestEmbedDetachCPU:
    """验证所有 HuggingFace 适配器的 embed 使用 detach().cpu()"""

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_embed_has_detach_cpu(self, adapter_module, adapter_class):
        """embed 源码应包含 .detach().cpu()"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.embed)
        
        assert "detach().cpu()" in source, (
            f"{adapter_class}.embed should use .detach().cpu() "
            f"to return CPU tensor"
        )

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_embed_cleans_up_gpu(self, adapter_module, adapter_class):
        """embed 源码应包含显存清理"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.embed)
        
        assert "del outputs" in source, (
            f"{adapter_class}.embed should delete outputs to free GPU memory"
        )
        assert "empty_cache()" in source, (
            f"{adapter_class}.embed should call torch.cuda.empty_cache()"
        )


# ============================================================================
# BUG 3: forward_with_kv_injection 应清理显存
# ============================================================================

class TestForwardKVInjectionCleanup:
    """验证所有适配器的 forward_with_kv_injection 有显存清理"""

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
        ("dki.models.vllm_adapter", "VLLMAdapter"),
    ])
    def test_forward_kv_injection_cleans_up(self, adapter_module, adapter_class):
        """forward_with_kv_injection 应有显存清理"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.forward_with_kv_injection)
        
        assert "empty_cache()" in source, (
            f"{adapter_class}.forward_with_kv_injection should call "
            f"torch.cuda.empty_cache()"
        )

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
    ])
    def test_forward_kv_injection_deletes_tensors(self, adapter_module, adapter_class):
        """forward_with_kv_injection 应删除中间张量"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        source = inspect.getsource(cls.forward_with_kv_injection)
        
        assert "del " in source, (
            f"{adapter_class}.forward_with_kv_injection should delete "
            f"intermediate tensors"
        )


# ============================================================================
# BUG 5: KVCacheEntry bfloat16 序列化/反序列化
# ============================================================================

class TestKVCacheEntryBfloat16:
    """验证 KVCacheEntry 对 bfloat16 的序列化/反序列化正确性"""

    def test_to_bytes_bfloat16_no_error(self):
        """bfloat16 张量的 to_bytes 不应抛出异常"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
            value=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
            layer_idx=0,
        )
        
        # 不应抛出 RuntimeError (bfloat16 无法直接 numpy)
        key_bytes, value_bytes = entry.to_bytes()
        
        assert isinstance(key_bytes, bytes)
        assert isinstance(value_bytes, bytes)
        # bfloat16 → float32 → numpy, 每个元素 4 bytes
        expected_size = 1 * 2 * 3 * 4 * 4  # float32 = 4 bytes
        assert len(key_bytes) == expected_size
        assert len(value_bytes) == expected_size

    def test_to_bytes_float16_correct(self):
        """float16 张量的 to_bytes 应正确"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4, dtype=torch.float16),
            value=torch.randn(1, 2, 3, 4, dtype=torch.float16),
            layer_idx=0,
        )
        
        key_bytes, value_bytes = entry.to_bytes()
        # float16 = 2 bytes per element
        expected_size = 1 * 2 * 3 * 4 * 2
        assert len(key_bytes) == expected_size

    def test_to_bytes_float32_correct(self):
        """float32 张量的 to_bytes 应正确"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4, dtype=torch.float32),
            value=torch.randn(1, 2, 3, 4, dtype=torch.float32),
            layer_idx=0,
        )
        
        key_bytes, value_bytes = entry.to_bytes()
        # float32 = 4 bytes per element
        expected_size = 1 * 2 * 3 * 4 * 4
        assert len(key_bytes) == expected_size

    def test_from_bytes_bfloat16_maps_to_float32(self):
        """from_bytes 对 bfloat16 应使用 float32 解析"""
        # 创建 bfloat16 entry, 序列化
        original = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
            value=torch.randn(1, 2, 3, 4, dtype=torch.bfloat16),
            layer_idx=5,
        )
        key_bytes, value_bytes = original.to_bytes()
        
        # 反序列化
        restored = KVCacheEntry.from_bytes(
            key_bytes=key_bytes,
            value_bytes=value_bytes,
            shape=(1, 2, 3, 4),
            layer_idx=5,
            dtype=torch.bfloat16,
        )
        
        assert restored.key.dtype == torch.bfloat16
        assert restored.value.dtype == torch.bfloat16
        assert restored.key.shape == (1, 2, 3, 4)
        assert restored.layer_idx == 5

    def test_roundtrip_float16(self):
        """float16 序列化/反序列化往返应保持数值一致"""
        original = KVCacheEntry(
            key=torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float16),
            value=torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]], dtype=torch.float16),
            layer_idx=3,
        )
        
        key_bytes, value_bytes = original.to_bytes()
        restored = KVCacheEntry.from_bytes(
            key_bytes=key_bytes,
            value_bytes=value_bytes,
            shape=(1, 1, 2, 2),
            layer_idx=3,
            dtype=torch.float16,
        )
        
        assert torch.allclose(original.key, restored.key)
        assert torch.allclose(original.value, restored.value)
        assert restored.layer_idx == 3

    def test_roundtrip_float32(self):
        """float32 序列化/反序列化往返应保持数值一致"""
        original = KVCacheEntry(
            key=torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32),
            value=torch.tensor([[[[5.5, 6.5], [7.5, 8.5]]]], dtype=torch.float32),
            layer_idx=7,
        )
        
        key_bytes, value_bytes = original.to_bytes()
        restored = KVCacheEntry.from_bytes(
            key_bytes=key_bytes,
            value_bytes=value_bytes,
            shape=(1, 1, 2, 2),
            layer_idx=7,
            dtype=torch.float32,
        )
        
        assert torch.allclose(original.key, restored.key)
        assert torch.allclose(original.value, restored.value)

    def test_to_bytes_source_converts_bfloat16_to_float(self):
        """验证 to_bytes 源码中对 bfloat16 的处理"""
        source = inspect.getsource(KVCacheEntry.to_bytes)
        assert "bfloat16" in source, "to_bytes should handle bfloat16 explicitly"
        assert ".float()" in source, "to_bytes should convert bfloat16 to float32"

    def test_from_bytes_bfloat16_uses_float32(self):
        """验证 from_bytes 映射表中 bfloat16 对应 float32"""
        source = inspect.getsource(KVCacheEntry.from_bytes)
        assert "np.float32" in source, "from_bytes should map bfloat16 to np.float32"


# ============================================================================
# KVCacheEntry.to_device
# ============================================================================

class TestKVCacheEntryToDevice:
    """验证 KVCacheEntry.to_device 的正确性"""

    def test_to_device_cpu(self):
        """to_device('cpu') 应返回 CPU 张量"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4),
            value=torch.randn(1, 2, 3, 4),
            layer_idx=0,
        )
        
        result = entry.to_device("cpu")
        assert result.key.device.type == "cpu"
        assert result.value.device.type == "cpu"
        assert result.layer_idx == 0

    def test_to_device_preserves_dtype(self):
        """to_device 应保持 dtype 不变"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4, dtype=torch.float16),
            value=torch.randn(1, 2, 3, 4, dtype=torch.float16),
            layer_idx=1,
        )
        
        result = entry.to_device("cpu")
        assert result.key.dtype == torch.float16
        assert result.value.dtype == torch.float16

    def test_to_device_returns_new_entry(self):
        """to_device 应返回新的 KVCacheEntry"""
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 3, 4),
            value=torch.randn(1, 2, 3, 4),
            layer_idx=2,
        )
        
        result = entry.to_device("cpu")
        assert result is not entry  # 新对象


# ============================================================================
# ModelOutput 数据类
# ============================================================================

class TestModelOutput:
    """验证 ModelOutput 数据类的完整性"""

    def test_default_values(self):
        """默认值应正确"""
        output = ModelOutput(text="hello")
        assert output.text == "hello"
        assert output.tokens is None
        assert output.logits is None
        assert output.latency_ms == 0.0
        assert output.input_tokens == 0
        assert output.output_tokens == 0
        assert output.metadata == {}

    def test_with_all_fields(self):
        """所有字段应可设置"""
        output = ModelOutput(
            text="response",
            tokens=[1, 2, 3],
            latency_ms=100.5,
            input_tokens=10,
            output_tokens=3,
            metadata={"alpha": 0.5},
        )
        assert output.tokens == [1, 2, 3]
        assert output.latency_ms == 100.5
        assert output.metadata["alpha"] == 0.5


# ============================================================================
# BaseModelAdapter 接口
# ============================================================================

class TestBaseModelAdapterInterface:
    """验证 BaseModelAdapter 的接口完整性"""

    def test_abstract_methods_defined(self):
        """应定义所有必需的抽象方法"""
        abstract_methods = BaseModelAdapter.__abstractmethods__
        
        expected = {
            "load",
            "generate",
            "embed",
            "compute_kv",
            "forward_with_kv_injection",
            "compute_prefill_entropy",
        }
        
        assert expected == abstract_methods, (
            f"Missing abstract methods: {expected - abstract_methods}"
        )

    def test_concrete_methods_available(self):
        """应有 tokenize, decode, unload, get_model_info 等具体方法"""
        assert hasattr(BaseModelAdapter, "tokenize")
        assert hasattr(BaseModelAdapter, "decode")
        assert hasattr(BaseModelAdapter, "unload")
        assert hasattr(BaseModelAdapter, "get_model_info")
        assert hasattr(BaseModelAdapter, "enable_flash_attention")
        assert hasattr(BaseModelAdapter, "disable_flash_attention")

    def test_all_adapters_implement_interface(self):
        """所有适配器应实现 BaseModelAdapter 接口"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        from dki.models.llama_adapter import LlamaAdapter
        from dki.models.glm_adapter import GLMAdapter
        from dki.models.vllm_adapter import VLLMAdapter
        
        for cls in [DeepSeekAdapter, LlamaAdapter, GLMAdapter, VLLMAdapter]:
            assert issubclass(cls, BaseModelAdapter), (
                f"{cls.__name__} should be subclass of BaseModelAdapter"
            )
            # 确认没有未实现的抽象方法 (即类可以实例化, 除了 __init__ 的参数)
            assert not getattr(cls, '__abstractmethods__', set()), (
                f"{cls.__name__} has unimplemented abstract methods: "
                f"{cls.__abstractmethods__}"
            )


# ============================================================================
# ModelFactory 注册和查找
# ============================================================================

class TestModelFactory:
    """验证 ModelFactory 的注册和查找"""

    def test_all_adapters_registered(self):
        """所有适配器应在工厂中注册"""
        from dki.models.factory import ModelFactory
        
        assert "vllm" in ModelFactory._adapters
        assert "llama" in ModelFactory._adapters
        assert "deepseek" in ModelFactory._adapters
        assert "glm" in ModelFactory._adapters

    def test_register_custom_adapter(self):
        """应能注册自定义适配器"""
        from dki.models.factory import ModelFactory
        
        # 创建一个假适配器
        mock_adapter = MagicMock(spec=BaseModelAdapter)
        
        ModelFactory.register_adapter("test_engine", type(mock_adapter))
        assert "test_engine" in ModelFactory._adapters
        
        # 清理
        del ModelFactory._adapters["test_engine"]

    def test_factory_adapter_classes_match(self):
        """工厂注册的类应与导入的类一致"""
        from dki.models.factory import ModelFactory
        from dki.models.vllm_adapter import VLLMAdapter
        from dki.models.llama_adapter import LlamaAdapter
        from dki.models.deepseek_adapter import DeepSeekAdapter
        from dki.models.glm_adapter import GLMAdapter
        
        assert ModelFactory._adapters["vllm"] is VLLMAdapter
        assert ModelFactory._adapters["llama"] is LlamaAdapter
        assert ModelFactory._adapters["deepseek"] is DeepSeekAdapter
        assert ModelFactory._adapters["glm"] is GLMAdapter


# ============================================================================
# 适配器一致性: 所有适配器的方法签名应与基类一致
# ============================================================================

class TestAdapterSignatureConsistency:
    """验证所有适配器的方法签名与基类一致"""

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
        ("dki.models.vllm_adapter", "VLLMAdapter"),
    ])
    def test_generate_signature(self, adapter_module, adapter_class):
        """generate 方法签名应包含 prompt, max_new_tokens, temperature, top_p"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        
        sig = inspect.signature(cls.generate)
        params = list(sig.parameters.keys())
        
        assert "prompt" in params
        assert "max_new_tokens" in params
        assert "temperature" in params
        assert "top_p" in params

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
        ("dki.models.vllm_adapter", "VLLMAdapter"),
    ])
    def test_forward_with_kv_injection_signature(self, adapter_module, adapter_class):
        """forward_with_kv_injection 签名应包含 prompt, injected_kv, alpha"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        
        sig = inspect.signature(cls.forward_with_kv_injection)
        params = list(sig.parameters.keys())
        
        assert "prompt" in params
        assert "injected_kv" in params
        assert "alpha" in params
        assert "max_new_tokens" in params

    @pytest.mark.parametrize("adapter_module,adapter_class", [
        ("dki.models.deepseek_adapter", "DeepSeekAdapter"),
        ("dki.models.llama_adapter", "LlamaAdapter"),
        ("dki.models.glm_adapter", "GLMAdapter"),
        ("dki.models.vllm_adapter", "VLLMAdapter"),
    ])
    def test_compute_kv_signature(self, adapter_module, adapter_class):
        """compute_kv 签名应包含 text, return_hidden"""
        import importlib
        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        
        sig = inspect.signature(cls.compute_kv)
        params = list(sig.parameters.keys())
        
        assert "text" in params
        assert "return_hidden" in params


# ============================================================================
# GLM 适配器特殊处理
# ============================================================================

class TestGLMAdapterSpecifics:
    """验证 GLM 适配器的特殊处理"""

    def test_glm_handles_non_tuple_kv(self):
        """GLM compute_kv 应处理非标准 K/V 格式"""
        from dki.models.glm_adapter import GLMAdapter
        
        source = inspect.getsource(GLMAdapter.compute_kv)
        assert "isinstance(layer_kv, tuple)" in source, (
            "GLM compute_kv should check for tuple K/V format"
        )

    def test_glm_build_chat_input_with_template(self):
        """GLM 应支持 apply_chat_template"""
        from dki.models.glm_adapter import GLMAdapter
        
        source = inspect.getsource(GLMAdapter._build_chat_input)
        assert "apply_chat_template" in source

    def test_glm_fallback_format(self):
        """GLM 应有旧版 ChatGLM 的回退格式"""
        from dki.models.glm_adapter import GLMAdapter
        
        source = inspect.getsource(GLMAdapter._build_chat_input)
        assert "[Round 1]" in source, (
            "GLM should have fallback format for older ChatGLM"
        )


# ============================================================================
# DeepSeek 适配器特殊处理
# ============================================================================

class TestDeepSeekAdapterSpecifics:
    """验证 DeepSeek 适配器的特殊处理"""

    def test_deepseek_format_prompt(self):
        """DeepSeek 应有正确的聊天格式"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        
        source = inspect.getsource(DeepSeekAdapter._format_prompt)
        assert "User:" in source
        assert "Assistant:" in source

    def test_deepseek_generate_checks_chat_model(self):
        """generate 应检查是否为 chat 模型"""
        from dki.models.deepseek_adapter import DeepSeekAdapter
        
        source = inspect.getsource(DeepSeekAdapter.generate)
        assert "'chat'" in source, (
            "DeepSeek generate should check if model name contains 'chat'"
        )


# ============================================================================
# vLLM 适配器特殊处理
# ============================================================================

class TestVLLMAdapterSpecifics:
    """验证 vLLM 适配器的特殊处理"""

    def test_vllm_has_hf_model_lazy_loading(self):
        """vLLM 应有 HF 模型的懒加载"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        assert hasattr(VLLMAdapter, "_load_hf_model")

    def test_vllm_load_hf_checks_gpu_memory(self):
        """_load_hf_model 应检查 GPU 显存"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        source = inspect.getsource(VLLMAdapter._load_hf_model)
        assert "mem_get_info" in source, (
            "_load_hf_model should check GPU memory before loading"
        )
        assert "use_cpu" in source, (
            "_load_hf_model should have CPU fallback"
        )

    def test_vllm_prefill_entropy_pre_check(self):
        """compute_prefill_entropy 应有 GPU 显存预检查"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        source = inspect.getsource(VLLMAdapter.compute_prefill_entropy)
        assert "mem_get_info" in source, (
            "compute_prefill_entropy should pre-check GPU memory"
        )
        assert "256" in source, (
            "Should skip if less than 256MB free"
        )

    def test_vllm_unload_cleans_both_models(self):
        """vLLM unload 应清理 llm 和 hf_model"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        source = inspect.getsource(VLLMAdapter.unload)
        assert "self.llm" in source
        assert "self.hf_model" in source

    def test_vllm_flash_attention_path(self):
        """vLLM forward_with_kv_injection 应有 FlashAttention 路径"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        source = inspect.getsource(VLLMAdapter.forward_with_kv_injection)
        assert "flash_attn_enabled" in source
        assert "_forward_with_flash_attention" in source
