"""
单元测试: SGLang 适配器

验证 SGLang 适配器的核心逻辑:
1. SGLangAdapter 初始化和参数传递
2. 量化配置映射 (GPTQ, AWQ, 4bit, 8bit)
3. 注入模式兼容性
4. Chat Template 处理
5. Stop strings 生成
6. generate / forward_with_kv_injection 推理接口
7. 安全降级方法 (embed, compute_kv, compute_prefill_entropy)
8. get_model_info 诊断信息
9. ModelFactory 集成
10. 引擎生命周期 (load / unload)
11. 事件循环冲突处理 (_call_engine_generate)

不依赖真实模型加载, 使用 Mock 模拟所有 SGLang/HuggingFace 组件。

Author: AGI Demo Project
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.models.base import BaseModelAdapter, KVCacheEntry, ModelOutput
from dki.models.sglang_adapter import SGLangAdapter


# ============================================================================
# 1. SGLangAdapter 初始化测试
# ============================================================================

class TestSGLangAdapterInit:
    """测试 SGLangAdapter 初始化"""

    def test_default_initialization(self):
        """默认初始化"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        assert adapter.model_name == "test-model"
        assert adapter.device == "cpu"
        assert adapter.quantization == "none"
        assert adapter.injection_mode == "prompt_prefix"
        assert adapter.tensor_parallel_size == 1
        assert adapter.max_model_len == 8192
        assert adapter.gpu_memory_utilization == 0.9
        assert adapter.trust_remote_code is True
        assert adapter.mem_fraction_static == 0.80
        assert adapter.schedule_policy == "lpm"
        assert adapter.chunked_prefill_size == 8192
        assert adapter.engine is None
        assert adapter.sampling_params is None
        assert not adapter.is_loaded

    def test_custom_initialization(self):
        """自定义参数初始化"""
        adapter = SGLangAdapter(
            model_name="Qwen/Qwen3.5-27B-Instruct",
            device="cuda",
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            mem_fraction_static=0.80,
            schedule_policy="random",
            chunked_prefill_size=4096,
        )
        assert adapter.model_name == "Qwen/Qwen3.5-27B-Instruct"
        assert adapter.tensor_parallel_size == 2
        assert adapter.max_model_len == 4096
        assert adapter.gpu_memory_utilization == 0.85
        assert adapter.mem_fraction_static == 0.80
        assert adapter.schedule_policy == "random"
        assert adapter.chunked_prefill_size == 4096

    def test_quantization_initialization(self):
        """量化参数初始化"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq",
            quantization_config={"bits": 4},
        )
        assert adapter.quantization == "gptq"
        assert adapter.is_quantized
        assert adapter.quantization_config == {"bits": 4}

    def test_4bit_quantization(self):
        """4bit 量化"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit",
        )
        assert adapter.quantization == "4bit"
        assert adapter.is_quantized
        assert adapter.is_4bit

    def test_8bit_quantization(self):
        """8bit 量化"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            quantization="8bit",
        )
        assert adapter.quantization == "8bit"
        assert adapter.is_quantized
        assert adapter.is_8bit

    def test_awq_quantization(self):
        """AWQ 量化"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            quantization="awq",
        )
        assert adapter.quantization == "awq"
        assert adapter.is_quantized

    def test_none_quantization(self):
        """无量化"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            quantization="none",
        )
        assert adapter.quantization == "none"
        assert not adapter.is_quantized


# ============================================================================
# 2. 注入模式兼容性测试
# ============================================================================

class TestSGLangInjectionMode:
    """测试注入模式兼容性"""

    def test_auto_mode(self):
        """auto 模式"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            injection_mode="auto",
        )
        assert adapter.injection_mode == "prompt_prefix"
        assert adapter.effective_injection_mode == "prompt_prefix"

    def test_prompt_prefix_mode(self):
        """prompt_prefix 模式"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            injection_mode="prompt_prefix",
        )
        assert adapter.injection_mode == "prompt_prefix"

    def test_hf_kv_mode_deprecated(self):
        """hf_kv 模式 (已废弃, 接受但内部走 SGLang)"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            injection_mode="hf_kv",
        )
        assert adapter.injection_mode == "prompt_prefix"

    def test_vllm_kv_mode(self):
        """vllm_kv 模式 (接受)"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            injection_mode="vllm_kv",
        )
        assert adapter.injection_mode == "prompt_prefix"

    def test_unknown_mode_fallback(self):
        """未知模式回退到 auto"""
        adapter = SGLangAdapter(
            model_name="test-model", device="cpu",
            injection_mode="unknown_mode",
        )
        assert adapter.injection_mode == "prompt_prefix"


# ============================================================================
# 3. Chat Template 处理测试
# ============================================================================

class TestSGLangChatTemplate:
    """测试 Chat Template 处理"""

    def setup_method(self):
        self.adapter = SGLangAdapter(model_name="test-model", device="cpu")

    def test_detect_chatml_tokens(self):
        """检测 ChatML 标记"""
        assert self.adapter._has_chat_template_tokens("<|im_start|>user\nhello<|im_end|>")

    def test_detect_llama3_tokens(self):
        """检测 Llama 3 标记"""
        assert self.adapter._has_chat_template_tokens("<|begin_of_text|>hello")
        assert self.adapter._has_chat_template_tokens("<|start_header_id|>user<|end_header_id|>")

    def test_detect_llama2_tokens(self):
        """检测 Llama 2 标记"""
        assert self.adapter._has_chat_template_tokens("[INST] hello [/INST]")

    def test_no_template_tokens(self):
        """无模板标记"""
        assert not self.adapter._has_chat_template_tokens("hello world")
        assert not self.adapter._has_chat_template_tokens("this is a normal prompt")

    def test_is_chat_model(self):
        """判断 Chat 模型"""
        adapter = SGLangAdapter(model_name="Qwen3.5-27B-Instruct", device="cpu")
        assert adapter._is_chat_model()

        adapter2 = SGLangAdapter(model_name="Qwen3.5-27B-Chat", device="cpu")
        assert adapter2._is_chat_model()

    def test_is_not_chat_model(self):
        """判断非 Chat 模型"""
        adapter = SGLangAdapter(model_name="Qwen3.5-27B", device="cpu")
        assert not adapter._is_chat_model()

    def test_format_prompt_fallback(self):
        """ChatML 回退格式"""
        self.adapter.tokenizer = None
        result = self.adapter._format_prompt("hello", system_prompt="system msg")
        assert "<|im_start|>system" in result
        assert "system msg" in result
        assert "<|im_start|>user" in result
        assert "hello" in result
        assert "<|im_start|>assistant" in result

    def test_format_prompt_no_system(self):
        """无 system prompt"""
        self.adapter.tokenizer = None
        result = self.adapter._format_prompt("hello")
        assert "<|im_start|>system" not in result
        assert "<|im_start|>user" in result
        assert "hello" in result

    def test_format_prompt_with_tokenizer(self):
        """使用 tokenizer 的 chat template"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>hello</formatted>"
        self.adapter.tokenizer = mock_tokenizer
        result = self.adapter._format_prompt("hello")
        assert result == "<formatted>hello</formatted>"
        mock_tokenizer.apply_chat_template.assert_called_once()


# ============================================================================
# 4. Stop Strings 测试
# ============================================================================

class TestSGLangStopStrings:
    """测试 stop strings 生成"""

    def test_qwen_stop_strings(self):
        """Qwen 模型 stop strings"""
        adapter = SGLangAdapter(model_name="Qwen3.5-27B-Instruct", device="cpu")
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.eos_token = "<|endoftext|>"
        stops = adapter._get_stop_strings()
        assert "<|im_end|>" in stops
        assert "<|endoftext|>" in stops

    def test_deepseek_stop_strings(self):
        """DeepSeek 模型 stop strings"""
        adapter = SGLangAdapter(model_name="deepseek-v3-chat", device="cpu")
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.eos_token = "</s>"
        stops = adapter._get_stop_strings()
        assert "<|im_end|>" in stops

    def test_llama3_stop_strings(self):
        """Llama 3 模型 stop strings"""
        adapter = SGLangAdapter(model_name="llama-3.1-8b-instruct", device="cpu")
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.eos_token = "</s>"
        stops = adapter._get_stop_strings()
        assert "<|eot_id|>" in stops

    def test_default_stop_strings(self):
        """默认 stop strings (无 tokenizer)"""
        adapter = SGLangAdapter(model_name="unknown-model", device="cpu")
        adapter.tokenizer = None
        stops = adapter._get_stop_strings()
        assert "<|im_end|>" in stops


# ============================================================================
# 5. Load 方法测试
# ============================================================================

class TestSGLangLoad:
    """测试 SGLang 引擎加载"""

    def _setup_load_mocks(self, mock_compat, mock_sgl_module, mock_tokenizer_cls, mock_config_cls,
                           hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                           pad_token="<pad>", eos_token="<|endoftext|>"):
        """统一设置 load() 所需的 mock 对象"""
        mock_engine = MagicMock()
        mock_sgl_module.Engine.return_value = mock_engine

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = pad_token
        mock_tokenizer.eos_token = eos_token
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_config = MagicMock()
        mock_config.hidden_size = hidden_size
        mock_config.num_hidden_layers = num_hidden_layers
        mock_config.num_attention_heads = num_attention_heads
        mock_config_cls.from_pretrained.return_value = mock_config

        return mock_engine, mock_tokenizer, mock_config

    def _load_with_mocks(self, adapter, hidden_size=4096, num_layers=32, num_heads=32,
                          pad_token="<pad>"):
        """辅助方法: 用 mock 执行 load()
        
        由于 sglang 和 transformers 都是在 load() 内部局部导入的,
        需要通过 sys.modules 和 patch transformers 模块属性来 mock。
        """
        mock_sgl = MagicMock()
        mock_engine = MagicMock()
        mock_sgl.Engine.return_value = mock_engine

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = pad_token
        mock_tokenizer.eos_token = "<|endoftext|>"

        mock_config = MagicMock()
        mock_config.hidden_size = hidden_size
        mock_config.num_hidden_layers = num_layers
        mock_config.num_attention_heads = num_heads

        # 创建 mock transformers 模块
        import transformers as real_transformers
        original_auto_tokenizer = real_transformers.AutoTokenizer
        original_auto_config = real_transformers.AutoConfig

        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch.dict('sys.modules', {'sglang': mock_sgl}):
                with patch.object(real_transformers, 'AutoTokenizer') as mock_tok_cls:
                    with patch.object(real_transformers, 'AutoConfig') as mock_cfg_cls:
                        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
                        mock_cfg_cls.from_pretrained.return_value = mock_config
                        adapter.load()

        return mock_sgl, mock_engine

    def test_load_basic(self):
        """基本加载测试"""
        adapter = SGLangAdapter(model_name="Qwen/Qwen3.5-7B-Instruct", device="cuda")
        mock_sgl, mock_engine = self._load_with_mocks(
            adapter, hidden_size=3584, num_layers=28, num_heads=28, pad_token=None
        )

        assert adapter._is_loaded
        assert adapter.engine is mock_engine
        assert adapter.hidden_dim == 3584
        assert adapter.num_layers == 28
        assert adapter.num_heads == 28

    def test_load_with_gptq(self):
        """GPTQ 量化加载 — 应映射为 gptq_marlin, 并设置 dtype/mamba 兼容参数"""
        adapter = SGLangAdapter(
            model_name="Qwen/Qwen3.5-27B-GPTQ-Int4",
            device="cuda",
            quantization="gptq",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'gptq_marlin', (
            "GPTQ should be mapped to gptq_marlin for better compatibility "
            "with hybrid architectures (Mamba + Transformer) like Qwen3.5"
        )
        assert call_kwargs.get('dtype') == 'float16', (
            "GPTQ should force dtype=float16"
        )
        assert call_kwargs.get('mamba_ssm_dtype') == 'float16', (
            "GPTQ should force mamba_ssm_dtype=float16"
        )

    def test_load_gptq_forces_float16_dtype(self):
        """GPTQ 量化应自动强制 dtype=float16 及 mamba_ssm_dtype=float16"""
        adapter = SGLangAdapter(
            model_name="Qwen/Qwen3.5-27B-GPTQ-Int4",
            device="cuda",
            quantization="gptq",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'gptq_marlin'
        assert call_kwargs['dtype'] == 'float16', (
            "GPTQ quantization must force dtype=float16 "
            "(bfloat16 is not supported by GPTQ)"
        )
        assert call_kwargs['mamba_ssm_dtype'] == 'float16', (
            "GPTQ must force mamba_ssm_dtype=float16 for Mamba layer compatibility"
        )

    def test_load_awq_forces_bfloat16_dtype(self):
        """AWQ 量化应映射为 awq_marlin 并强制 dtype=bfloat16 + mamba_ssm_dtype=bfloat16"""
        adapter = SGLangAdapter(
            model_name="test-model-awq",
            device="cuda",
            quantization="awq",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'awq_marlin', (
            "AWQ should be mapped to awq_marlin for better performance"
        )
        assert call_kwargs['dtype'] == 'bfloat16', (
            "AWQ quantization must force dtype=bfloat16 "
            "(bfloat16 range ±3.4e38 prevents Mamba SSM NaN overflow)"
        )
        assert call_kwargs['mamba_ssm_dtype'] == 'bfloat16', (
            "AWQ must force mamba_ssm_dtype=bfloat16 for Mamba layer compatibility"
        )

    def test_load_4bit_forces_bfloat16_dtype(self):
        """4bit bitsandbytes 量化应强制 dtype=bfloat16 + mamba_ssm_dtype=bfloat16"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            quantization="4bit",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'bitsandbytes'
        assert call_kwargs['dtype'] == 'bfloat16', (
            "bitsandbytes quantization must force dtype=bfloat16 "
            "(bfloat16 range ±3.4e38 prevents Mamba SSM NaN overflow)"
        )
        assert call_kwargs['mamba_ssm_dtype'] == 'bfloat16', (
            "bitsandbytes must force mamba_ssm_dtype=bfloat16"
        )

    def test_load_no_quant_no_dtype_override(self):
        """无量化不应设置 dtype"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert 'quantization' not in call_kwargs
        assert 'dtype' not in call_kwargs

    def test_load_with_awq(self):
        """AWQ 量化加载 — 应映射为 awq_marlin"""
        adapter = SGLangAdapter(
            model_name="test-model-awq",
            device="cuda",
            quantization="awq",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'awq_marlin'

    def test_load_with_4bit(self):
        """4bit 量化加载 → bitsandbytes, dtype/mamba_ssm_dtype 应被设置为 bfloat16"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            quantization="4bit",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'bitsandbytes'
        assert call_kwargs['dtype'] == 'bfloat16'
        assert call_kwargs['mamba_ssm_dtype'] == 'bfloat16'

    def test_load_with_8bit(self):
        """8bit 量化加载 → bitsandbytes, dtype/mamba_ssm_dtype 应被设置为 bfloat16"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            quantization="8bit",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['quantization'] == 'bitsandbytes'
        assert call_kwargs['dtype'] == 'bfloat16'
        assert call_kwargs['mamba_ssm_dtype'] == 'bfloat16'

    def test_load_no_quantization(self):
        """无量化加载 — 不传 quantization 参数"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert 'quantization' not in call_kwargs

    def test_load_all_quant_types_force_dtype_and_mamba(self):
        """GPTQ 用 float16, AWQ/bitsandbytes 用 bfloat16 (防 NaN)"""
        # (user_quant, expected_sglang_quant, expected_dtype)
        quant_cases = [
            ("gptq", "gptq_marlin", "float16"),       # GPTQ Marlin 限制 float16
            ("awq", "awq_marlin", "bfloat16"),         # AWQ Marlin 支持 bf16, 防 NaN
            ("4bit", "bitsandbytes", "bfloat16"),      # bitsandbytes 支持 bf16
            ("8bit", "bitsandbytes", "bfloat16"),      # bitsandbytes 支持 bf16
        ]
        for user_quant, expected_sglang_quant, expected_dtype in quant_cases:
            adapter = SGLangAdapter(
                model_name="test-model",
                device="cuda",
                quantization=user_quant,
            )
            mock_sgl, _ = self._load_with_mocks(adapter)
            call_kwargs = mock_sgl.Engine.call_args[1]
            assert call_kwargs['quantization'] == expected_sglang_quant, (
                f"quantization={user_quant} should map to {expected_sglang_quant}"
            )
            assert call_kwargs['dtype'] == expected_dtype, (
                f"quantization={user_quant} should force dtype={expected_dtype}"
            )
            assert call_kwargs['mamba_ssm_dtype'] == expected_dtype, (
                f"quantization={user_quant} should force mamba_ssm_dtype={expected_dtype}"
            )
            # mamba_backend 不应被设置 (不受支持)
            assert 'mamba_backend' not in call_kwargs, (
                f"quantization={user_quant} should not set mamba_backend"
            )

    def test_load_sglang_specific_params(self):
        """SGLang 特有参数传递"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            mem_fraction_static=0.80,
            schedule_policy="random",
            chunked_prefill_size=4096,
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['mem_fraction_static'] == 0.80
        assert call_kwargs['schedule_policy'] == 'random'
        assert call_kwargs['chunked_prefill_size'] == 4096

    def test_load_tensor_parallel(self):
        """张量并行参数传递"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            tensor_parallel_size=2,
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert call_kwargs['tp_size'] == 2

    def test_load_idempotent(self):
        """重复加载不报错 (幂等)"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter._is_loaded = True
        adapter.load()  # 不应抛异常
        assert adapter._is_loaded

    def test_load_sglang_not_installed(self):
        """SGLang 未安装时报错"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")

        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch.dict('sys.modules', {'sglang': None}):
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    adapter.load()

    def test_load_pad_token_auto_set(self):
        """pad_token 为 None 时自动设置为 eos_token"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        self._load_with_mocks(adapter, pad_token=None)
        assert adapter.tokenizer.pad_token == adapter.tokenizer.eos_token

    def test_load_auto_processor_preimport(self):
        """AutoProcessor 在 sgl.Engine() 之前被预加载"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        
        import_order = []
        
        mock_sgl = MagicMock()
        mock_engine = MagicMock()
        mock_sgl.Engine.return_value = mock_engine
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<|endoftext|>"
        
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        
        import transformers as real_transformers
        
        # 记录 AutoProcessor 导入时机
        original_auto_processor = getattr(real_transformers, 'AutoProcessor', None)
        
        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch.dict('sys.modules', {'sglang': mock_sgl}):
                with patch.object(real_transformers, 'AutoTokenizer') as mock_tok_cls:
                    with patch.object(real_transformers, 'AutoConfig') as mock_cfg_cls:
                        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
                        mock_cfg_cls.from_pretrained.return_value = mock_config
                        adapter.load()
        
        # 验证加载成功 (AutoProcessor 预导入没有阻碍加载)
        assert adapter._is_loaded
        assert adapter.engine is mock_engine

    def test_load_auto_processor_import_failure_non_blocking(self):
        """AutoProcessor 导入失败(非 hf_hub 问题)不应阻止纯文本模型加载"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        
        mock_sgl = MagicMock()
        mock_engine = MagicMock()
        mock_sgl.Engine.return_value = mock_engine
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<|endoftext|>"
        
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        
        import transformers as real_transformers
        
        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch('dki.models.hf_compat._get_hf_hub_version', return_value="1.5.0"):
                with patch.dict('sys.modules', {'sglang': mock_sgl}):
                    with patch.object(real_transformers, 'AutoTokenizer') as mock_tok_cls:
                        with patch.object(real_transformers, 'AutoConfig') as mock_cfg_cls:
                            # 模拟 AutoProcessor 导入失败 (非 hf_hub 版本问题)
                            with patch.object(
                                real_transformers, 'AutoProcessor',
                                side_effect=ImportError("No module named 'PIL'"),
                                create=True,
                            ):
                                mock_tok_cls.from_pretrained.return_value = mock_tokenizer
                                mock_cfg_cls.from_pretrained.return_value = mock_config
                                # hf_hub 版本正常时, AutoProcessor 失败只是 warning, 加载继续
                                adapter.load()
        
        assert adapter._is_loaded
        assert adapter.engine is mock_engine

    def test_load_auto_processor_failure_due_to_hf_hub_version(self):
        """AutoProcessor 导入失败 + hf_hub 版本过低 → 应抛出明确错误"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        
        mock_sgl = MagicMock()
        mock_sgl.Engine.return_value = MagicMock()
        
        import transformers as real_transformers
        import builtins
        original_import = builtins.__import__
        
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            # 拦截 from transformers import AutoProcessor
            if name == 'transformers' and fromlist and 'AutoProcessor' in fromlist:
                raise ImportError("cannot import name 'AutoProcessor'")
            return original_import(name, globals, locals, fromlist, level)
        
        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch('dki.models.hf_compat._get_hf_hub_version', return_value="0.36.2"):
                with patch.dict('sys.modules', {'sglang': mock_sgl}):
                    with patch.object(real_transformers, 'AutoTokenizer') as mock_tok:
                        with patch.object(real_transformers, 'AutoConfig') as mock_cfg:
                            mock_tok.from_pretrained.return_value = MagicMock()
                            mock_cfg.from_pretrained.return_value = MagicMock()
                            with patch.object(builtins, '__import__', side_effect=fake_import):
                                with pytest.raises(ImportError, match="huggingface-hub"):
                                    adapter.load()

    def test_load_sglang_internal_autoprocessor_error(self):
        """SGLang 内部 AutoProcessor 错误应有清晰的错误信息"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        
        mock_sgl = MagicMock()
        # 模拟 SGLang Engine 内部抛出 AutoProcessor ImportError
        mock_sgl.Engine.side_effect = ImportError(
            "Could not import module 'AutoProcessor'. "
            "Are this object's requirements defined correctly?"
        )
        
        import transformers as real_transformers
        
        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch.dict('sys.modules', {'sglang': mock_sgl}):
                with patch.object(real_transformers, 'AutoTokenizer'):
                    with patch.object(real_transformers, 'AutoConfig'):
                        with pytest.raises(ImportError, match="AutoProcessor"):
                            adapter.load()

    def test_get_sglang_version(self):
        """测试 _get_sglang_version 辅助方法"""
        # 已安装时返回版本号
        mock_sgl = MagicMock()
        mock_sgl.__version__ = "0.5.9"
        with patch.dict('sys.modules', {'sglang': mock_sgl}):
            version = SGLangAdapter._get_sglang_version()
            assert version == "0.5.9"

    def test_get_sglang_version_not_installed(self):
        """SGLang 未安装时返回 unknown"""
        with patch.dict('sys.modules', {'sglang': None}):
            version = SGLangAdapter._get_sglang_version()
            assert version == "unknown"


# ============================================================================
# 6. Generate 方法测试
# ============================================================================

class TestSGLangGenerate:
    """测试 SGLang generate 方法"""

    def _make_loaded_adapter(self):
        """创建一个已加载的 adapter"""
        adapter = SGLangAdapter(model_name="Qwen3.5-27B-Instruct", device="cpu")
        adapter._is_loaded = True
        adapter.engine = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        adapter.tokenizer.eos_token = "<|endoftext|>"
        return adapter

    def test_generate_basic(self):
        """基本生成"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {
            "text": "你好！有什么可以帮你的？",
            "meta_info": {
                "output_ids": [1, 2, 3],
                "prompt_tokens": 10,
                "completion_tokens": 8,
            }
        }

        output = adapter.generate("hello", max_new_tokens=256)

        assert isinstance(output, ModelOutput)
        assert output.text == "你好！有什么可以帮你的？"
        assert output.input_tokens == 10
        assert output.output_tokens == 8
        assert output.latency_ms > 0

    def test_generate_with_chat_template_tokens(self):
        """已有 chat template 标记的 prompt 不二次包装"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {
            "text": "response",
            "meta_info": {}
        }

        prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        adapter.generate(prompt)

        # engine.generate 应该收到原始 prompt (不二次包装)
        call_args = adapter.engine.generate.call_args
        assert call_args[0][0] == prompt

    def test_generate_auto_load(self):
        """未加载时自动加载"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter._is_loaded = False

        with patch.object(adapter, 'load') as mock_load:
            adapter._is_loaded = False

            def side_effect():
                adapter._is_loaded = True
                adapter.engine = MagicMock()
                adapter.engine.generate.return_value = {"text": "ok", "meta_info": {}}
                adapter.tokenizer = None

            mock_load.side_effect = side_effect
            output = adapter.generate("hello")
            mock_load.assert_called_once()

    def test_generate_string_output(self):
        """SGLang 返回字符串时的处理"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = "plain text response"

        output = adapter.generate("hello")
        assert output.text == "plain text response"


# ============================================================================
# 7. forward_with_kv_injection 方法测试
# ============================================================================

class TestSGLangForwardWithKVInjection:
    """测试 forward_with_kv_injection 方法"""

    def _make_loaded_adapter(self):
        adapter = SGLangAdapter(model_name="Qwen3.5-27B-Instruct", device="cpu")
        adapter._is_loaded = True
        adapter.engine = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.apply_chat_template.return_value = "<formatted>"
        adapter.tokenizer.eos_token = "<|endoftext|>"
        return adapter

    def test_forward_basic(self):
        """基本 KV 注入推理"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {
            "text": "注入后的回复",
            "meta_info": {"prompt_tokens": 50, "completion_tokens": 10}
        }

        output = adapter.forward_with_kv_injection(
            prompt="偏好前缀\n\nhello",
            injected_kv=[],
            alpha=0.7,
            max_new_tokens=256,
        )

        assert isinstance(output, ModelOutput)
        assert output.text == "注入后的回复"
        assert output.metadata['alpha'] == 0.7
        assert output.metadata['injection_mode'] == 'sglang_native_radix_attention'

    def test_forward_ignores_injected_kv(self):
        """injected_kv 参数被忽略 (签名兼容)"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "ok", "meta_info": {}}

        fake_kv = [MagicMock(spec=KVCacheEntry)]
        output = adapter.forward_with_kv_injection(
            prompt="hello",
            injected_kv=fake_kv,
        )
        assert output.text == "ok"

    def test_forward_with_string_kv(self):
        """injected_kv 为字符串时不报错"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "ok", "meta_info": {}}

        output = adapter.forward_with_kv_injection(
            prompt="hello",
            injected_kv="some_kv_string",
        )
        assert output.text == "ok"


# ============================================================================
# 8. 安全降级方法测试
# ============================================================================

class TestSGLangSafetyDegradation:
    """测试安全降级方法"""

    def test_embed_raises(self):
        """embed() 应该抛出 RuntimeError"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        with pytest.raises(RuntimeError, match="embed.*not available"):
            adapter.embed("test")

    def test_compute_kv_returns_empty(self):
        """compute_kv() 返回空列表"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        kv_list, hidden = adapter.compute_kv("test")
        assert kv_list == []
        assert hidden is None

    def test_compute_prefill_entropy_returns_default(self):
        """compute_prefill_entropy() 返回 0.5"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        entropy = adapter.compute_prefill_entropy("test")
        assert entropy == 0.5


# ============================================================================
# 9. get_model_info 测试
# ============================================================================

class TestSGLangModelInfo:
    """测试 get_model_info"""

    def test_model_info_not_loaded(self):
        """未加载时的 model info"""
        adapter = SGLangAdapter(
            model_name="Qwen/Qwen3.5-27B-Instruct",
            device="cpu",
            quantization="gptq",
        )
        info = adapter.get_model_info()

        assert info['model_name'] == "Qwen/Qwen3.5-27B-Instruct"
        assert info['engine'] == 'sglang'
        assert info['sglang_native_kv'] is True
        assert info['radix_attention_enabled'] is True
        assert info['hf_model_loaded'] is False
        assert info['sglang_engine_loaded'] is False
        assert info['quantization'] == 'gptq'
        assert info['schedule_policy'] == 'lpm'
        assert info['injection_mode'] == 'prompt_prefix'
        assert info['effective_injection_mode'] == 'prompt_prefix'

    def test_model_info_loaded(self):
        """已加载时的 model info"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter.engine = MagicMock()
        info = adapter.get_model_info()
        assert info['sglang_engine_loaded'] is True


# ============================================================================
# 10. Unload 测试
# ============================================================================

class TestSGLangUnload:
    """测试 unload"""

    def test_unload_with_engine(self):
        """卸载已加载的引擎"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter.engine = MagicMock()
        adapter._is_loaded = True

        adapter.unload()

        assert adapter.engine is None
        assert not adapter._is_loaded

    def test_unload_without_engine(self):
        """卸载未加载的引擎 (不报错)"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter.unload()  # 不应抛异常

    def test_unload_shutdown_called(self):
        """unload 调用 engine.shutdown()"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        mock_engine = MagicMock()
        adapter.engine = mock_engine
        adapter._is_loaded = True

        adapter.unload()

        mock_engine.shutdown.assert_called_once()

    def test_unload_shutdown_error_handled(self):
        """engine.shutdown() 报错时不抛异常"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        mock_engine = MagicMock()
        mock_engine.shutdown.side_effect = RuntimeError("shutdown failed")
        adapter.engine = mock_engine
        adapter._is_loaded = True

        adapter.unload()  # 不应抛异常
        assert adapter.engine is None


# ============================================================================
# 11. Tokenize / Decode 测试
# ============================================================================

class TestSGLangTokenization:
    """测试 tokenize 和 decode"""

    def test_tokenize_without_tokenizer(self):
        """无 tokenizer 时报错"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter.tokenizer = None
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            adapter.tokenize("hello")

    def test_decode_without_tokenizer(self):
        """无 tokenizer 时报错"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        adapter.tokenizer = None
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            adapter.decode([1, 2, 3])

    def test_tokenize_with_tokenizer(self):
        """有 tokenizer 时正常工作"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        adapter.tokenizer = mock_tokenizer
        adapter.max_model_len = 8192

        adapter.tokenize("hello")
        mock_tokenizer.assert_called_once()

    def test_decode_with_tokenizer(self):
        """有 tokenizer 时正常 decode"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello world"
        adapter.tokenizer = mock_tokenizer

        result = adapter.decode([1, 2, 3])
        assert result == "hello world"

    def test_decode_tensor_input(self):
        """decode 接受 tensor 输入"""
        adapter = SGLangAdapter(model_name="test-model", device="cpu")
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello"
        adapter.tokenizer = mock_tokenizer

        result = adapter.decode(torch.tensor([1, 2, 3]))
        assert result == "hello"


# ============================================================================
# 12. ModelFactory 集成测试
# ============================================================================

class TestSGLangFactoryIntegration:
    """测试 ModelFactory 对 SGLang 的支持"""

    def test_sglang_registered(self):
        """SGLang 已注册到 ModelFactory"""
        from dki.models.factory import ModelFactory
        assert 'sglang' in ModelFactory._adapters
        assert ModelFactory._adapters['sglang'] is SGLangAdapter

    def test_factory_create_sglang(self):
        """通过 ModelFactory 创建 SGLang adapter"""
        from dki.models.factory import ModelFactory
        from dki.config.config_loader import ConfigLoader, EngineConfig, Config, ModelConfig

        # 重置 ConfigLoader 单例
        ConfigLoader.reset()

        # 构建真实的 EngineConfig
        sglang_engine_config = EngineConfig(
            model_name="Qwen/Qwen3.5-27B-Instruct",
            enabled=True,
            quantization="gptq",
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            injection_mode="prompt_prefix",
            mem_fraction_static=0.80,
            schedule_policy="lpm",
            chunked_prefill_size=4096,
        )

        # 构建完整 Config 对象
        model_config = ModelConfig(
            default_engine="sglang",
            engines={"sglang": sglang_engine_config},
        )
        full_config = Config(model=model_config)

        with patch.object(ConfigLoader, '__init__', lambda self, *a, **kw: None):
            with patch.object(ConfigLoader, 'config', new_callable=PropertyMock, return_value=full_config):
                adapter = ModelFactory.create(engine="sglang")

                assert isinstance(adapter, SGLangAdapter)
                assert adapter.model_name == "Qwen/Qwen3.5-27B-Instruct"
                assert adapter.quantization == "gptq"
                assert adapter.mem_fraction_static == 0.80
                assert adapter.schedule_policy == "lpm"
                assert adapter.chunked_prefill_size == 4096

        # 清理
        ConfigLoader.reset()


# ============================================================================
# 13. EngineConfig SGLang 字段测试
# ============================================================================

class TestEngineConfigSGLangFields:
    """测试 EngineConfig 的 SGLang 特有字段"""

    def test_default_sglang_fields(self):
        """SGLang 字段默认值"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(model_name="test-model")
        assert config.mem_fraction_static == 0.80
        assert config.schedule_policy == "lpm"
        assert config.chunked_prefill_size == 8192

    def test_custom_sglang_fields(self):
        """SGLang 字段自定义值"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            mem_fraction_static=0.75,
            schedule_policy="random",
            chunked_prefill_size=4096,
        )
        assert config.mem_fraction_static == 0.75
        assert config.schedule_policy == "random"
        assert config.chunked_prefill_size == 4096

    def test_sglang_fields_from_yaml_dict(self):
        """从 YAML dict 解析 SGLang 字段"""
        from dki.config.config_loader import EngineConfig
        yaml_dict = {
            "model_name": "Qwen/Qwen3.5-27B-Instruct",
            "enabled": True,
            "quantization": "gptq",
            "mem_fraction_static": 0.80,
            "schedule_policy": "lpm",
            "chunked_prefill_size": 8192,
        }
        config = EngineConfig(**yaml_dict)
        assert config.model_name == "Qwen/Qwen3.5-27B-Instruct"
        assert config.quantization == "gptq"
        assert config.mem_fraction_static == 0.80
        assert config.schedule_policy == "lpm"


# ============================================================================
# 14. _filter_engine_kwargs 参数过滤测试
# ============================================================================

class TestSGLangFilterEngineKwargs:
    """测试 _filter_engine_kwargs 参数过滤逻辑"""

    def test_filter_with_real_server_args(self):
        """ServerArgs 有明确参数签名时, 正确过滤"""
        import inspect

        mock_sgl = MagicMock()

        # 模拟 ServerArgs 有明确的参数签名
        class FakeServerArgs:
            def __init__(self, model_path="", tp_size=1, mem_fraction_static=0.9,
                         schedule_policy="lpm", trust_remote_code=False,
                         context_length=8192, quantization=None):
                pass

        mock_server_args_mod = MagicMock()
        mock_server_args_mod.ServerArgs = FakeServerArgs

        engine_kwargs = {
            'model_path': '/path/to/model',
            'tp_size': 2,
            'mem_fraction_static': 0.8,
            'schedule_policy': 'lpm',
            'unsupported_param': 'should_be_dropped',
        }

        with patch.dict('sys.modules', {'sglang.srt.server_args': mock_server_args_mod}):
            result = SGLangAdapter._filter_engine_kwargs(mock_sgl, engine_kwargs)

        assert 'model_path' in result
        assert 'tp_size' in result
        assert 'mem_fraction_static' in result
        assert 'schedule_policy' in result
        assert 'unsupported_param' not in result

    def test_filter_wrapper_class_detected(self):
        """检测到包装类/工厂类时, 跳过过滤, 保留所有参数"""
        import inspect

        mock_sgl = MagicMock()

        # 模拟 SGLang 0.5.9 的 Engine 是工厂类 (只有 class_name, module_name)
        class FakeWrapperEngine:
            def __init__(self, class_name=None, module_name=None):
                pass

        mock_sgl.Engine = FakeWrapperEngine

        # ServerArgs 导入失败 (模拟无法找到)
        engine_kwargs = {
            'model_path': '/path/to/model',
            'tp_size': 2,
            'mem_fraction_static': 0.8,
            'quantization': 'gptq',
            'schedule_policy': 'lpm',
        }

        with patch.dict('sys.modules', {
            'sglang.srt.server_args': None,
            'sglang.srt.utils.server_args': None,
        }):
            result = SGLangAdapter._filter_engine_kwargs(mock_sgl, engine_kwargs)

        # 包装类检测: 所有参数应被保留
        assert result == engine_kwargs

    def test_filter_engine_accepts_kwargs(self):
        """Engine.__init__ 接受 **kwargs 时, 跳过过滤"""
        mock_sgl = MagicMock()

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

        mock_sgl.Engine = FakeEngine

        engine_kwargs = {
            'model_path': '/path/to/model',
            'tp_size': 2,
            'any_param': 'should_be_kept',
        }

        with patch.dict('sys.modules', {
            'sglang.srt.server_args': None,
            'sglang.srt.utils.server_args': None,
        }):
            result = SGLangAdapter._filter_engine_kwargs(mock_sgl, engine_kwargs)

        assert result == engine_kwargs

    def test_filter_server_args_with_kwargs(self):
        """ServerArgs 接受 **kwargs 时, 跳过过滤"""
        mock_sgl = MagicMock()

        class FakeServerArgs:
            def __init__(self, model_path="", **kwargs):
                pass

        mock_server_args_mod = MagicMock()
        mock_server_args_mod.ServerArgs = FakeServerArgs

        engine_kwargs = {
            'model_path': '/path/to/model',
            'any_param': 'should_be_kept',
        }

        with patch.dict('sys.modules', {'sglang.srt.server_args': mock_server_args_mod}):
            result = SGLangAdapter._filter_engine_kwargs(mock_sgl, engine_kwargs)

        assert result == engine_kwargs

    def test_filter_no_detection_possible(self):
        """无法检测参数时, 保留所有参数"""
        mock_sgl = MagicMock()
        # MagicMock 的 __init__ signature 检测会抛 ValueError
        mock_sgl.Engine = MagicMock()

        engine_kwargs = {
            'model_path': '/path/to/model',
            'tp_size': 2,
        }

        with patch.dict('sys.modules', {
            'sglang.srt.server_args': None,
            'sglang.srt.utils.server_args': None,
        }):
            result = SGLangAdapter._filter_engine_kwargs(mock_sgl, engine_kwargs)

        # 无法检测时, 应保留所有参数
        assert result == engine_kwargs

    def test_filter_core_params_validation(self):
        """_SGLANG_CORE_PARAMS 包含预期的核心参数"""
        assert 'model_path' in SGLangAdapter._SGLANG_CORE_PARAMS
        assert 'tp_size' in SGLangAdapter._SGLANG_CORE_PARAMS
        assert 'mem_fraction_static' in SGLangAdapter._SGLANG_CORE_PARAMS


# ============================================================================
# 15. 错误处理测试
# ============================================================================

class TestSGLangErrorHandling:
    """测试错误处理和诊断信息"""

    def test_load_huggingface_hub_error(self):
        """huggingface-hub 版本错误应有清晰提示"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        
        import transformers as real_transformers
        import builtins
        original_import = builtins.__import__
        
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            # 拦截 from transformers import AutoTokenizer — 模拟 huggingface-hub 问题
            if name == 'transformers' and fromlist and 'AutoTokenizer' in fromlist:
                raise ImportError("huggingface-hub>=1.3.0 is required")
            return original_import(name, globals, locals, fromlist, level)

        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch('dki.models.hf_compat._get_hf_hub_version', return_value="1.5.0"):
                with patch.dict('sys.modules', {'sglang': MagicMock()}):
                    with patch.object(builtins, '__import__', side_effect=fake_import):
                        with pytest.raises(ImportError, match="huggingface"):
                            adapter.load()

    def test_load_autoprocessor_internal_error_diagnostics(self):
        """SGLang 内部 AutoProcessor 错误应提供修复建议"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")

        mock_sgl = MagicMock()
        mock_sgl.Engine.side_effect = ImportError(
            "Could not import module 'AutoProcessor'"
        )

        import transformers as real_transformers

        with patch('dki.models.hf_compat.ensure_hf_compat'):
            with patch('dki.models.hf_compat._get_hf_hub_version', return_value="1.5.0"):
                with patch.dict('sys.modules', {'sglang': mock_sgl}):
                    with patch.object(real_transformers, 'AutoTokenizer'):
                        with patch.object(real_transformers, 'AutoConfig'):
                            with pytest.raises(ImportError, match="AutoProcessor"):
                                adapter.load()


# ============================================================================
# 16. 事件循环冲突处理测试 (_call_engine_generate)
# ============================================================================

class TestSGLangEventLoopHandling:
    """
    测试 _call_engine_generate 的事件循环冲突处理.
    
    核心场景:
    - 非 async 上下文 (CLI/测试): 直接调用 engine.generate()
    - async 上下文 (FastAPI/uvicorn): 使用 async_generate 或线程池
    """

    def _make_loaded_adapter(self):
        """创建已加载的 adapter"""
        adapter = SGLangAdapter(model_name="Qwen3.5-27B-Instruct", device="cpu")
        adapter._is_loaded = True
        adapter.engine = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.apply_chat_template.return_value = "<formatted>"
        adapter.tokenizer.eos_token = "<|endoftext|>"
        return adapter

    def test_call_engine_generate_no_event_loop(self):
        """非 async 上下文: 直接调用同步 engine.generate()"""
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "ok", "meta_info": {}}

        result = adapter._call_engine_generate("hello", {"max_new_tokens": 10})

        adapter.engine.generate.assert_called_once_with("hello", {"max_new_tokens": 10})
        assert result == {"text": "ok", "meta_info": {}}

    def test_call_engine_generate_async_with_async_generate(self):
        """
        _call_engine_generate_async: engine 有 async_generate 时
        应直接 await engine.async_generate() (不走 run_until_complete)
        """
        adapter = self._make_loaded_adapter()

        # 模拟 async_generate
        async def fake_async_generate(prompt, params):
            return {"text": "async result", "meta_info": {}}
        adapter.engine.async_generate = fake_async_generate

        async def run_test():
            result = await adapter._call_engine_generate_async("hello", {"max_new_tokens": 10})
            assert result == {"text": "async result", "meta_info": {}}

        asyncio.run(run_test())

    def test_call_engine_generate_in_running_loop_no_async_generate(self):
        """
        async 上下文 + engine 没有 async_generate:
        应回退到线程池执行同步 generate
        """
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "thread pool result", "meta_info": {}}
        # 确保没有 async_generate 方法
        if hasattr(adapter.engine, 'async_generate'):
            del adapter.engine.async_generate

        async def run_test():
            result = adapter._call_engine_generate("hello", {"max_new_tokens": 10})
            assert result == {"text": "thread pool result", "meta_info": {}}

        asyncio.run(run_test())

    def test_call_engine_generate_in_running_loop_same_loop(self):
        """
        async 上下文 + engine 有 async_generate 但循环相同:
        应回退到线程池执行
        """
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "same loop result", "meta_info": {}}

        # 模拟 async_generate 存在
        async def fake_async_generate(prompt, params):
            return {"text": "should not reach", "meta_info": {}}
        adapter.engine.async_generate = fake_async_generate

        async def run_test():
            # engine.loop 与当前循环相同
            adapter.engine.loop = asyncio.get_running_loop()
            result = adapter._call_engine_generate("hello", {"max_new_tokens": 10})
            assert result == {"text": "same loop result", "meta_info": {}}

        asyncio.run(run_test())

    def test_call_engine_generate_in_running_loop_no_engine_loop(self):
        """
        async 上下文 + engine 有 async_generate 但无 engine.loop:
        应回退到线程池执行
        """
        adapter = self._make_loaded_adapter()
        adapter.engine.generate.return_value = {"text": "no loop result", "meta_info": {}}

        # 模拟 async_generate 存在但没有 engine.loop
        async def fake_async_generate(prompt, params):
            return {"text": "should not reach", "meta_info": {}}
        adapter.engine.async_generate = fake_async_generate
        adapter.engine.loop = None

        async def run_test():
            result = adapter._call_engine_generate("hello", {"max_new_tokens": 10})
            assert result == {"text": "no loop result", "meta_info": {}}

        asyncio.run(run_test())

    def test_generate_uses_call_engine_generate(self):
        """generate() 内部应调用 _call_engine_generate"""
        adapter = self._make_loaded_adapter()

        with patch.object(adapter, '_call_engine_generate') as mock_call:
            mock_call.return_value = {"text": "ok", "meta_info": {}}
            output = adapter.generate("hello")

            mock_call.assert_called_once()
            assert output.text == "ok"

    def test_forward_with_kv_injection_uses_call_engine_generate(self):
        """forward_with_kv_injection() 内部应调用 _call_engine_generate"""
        adapter = self._make_loaded_adapter()

        with patch.object(adapter, '_call_engine_generate') as mock_call:
            mock_call.return_value = {
                "text": "injected result",
                "meta_info": {"prompt_tokens": 50, "completion_tokens": 10}
            }
            output = adapter.forward_with_kv_injection(
                prompt="prefix\nhello",
                injected_kv=[],
                alpha=0.7,
            )

            mock_call.assert_called_once()
            assert output.text == "injected result"
            assert output.metadata['alpha'] == 0.7

    def test_parse_engine_output_dict(self):
        """_parse_engine_output 解析 dict 输出"""
        adapter = self._make_loaded_adapter()
        text, meta = adapter._parse_engine_output({
            "text": "hello",
            "meta_info": {"prompt_tokens": 5}
        })
        assert text == "hello"
        assert meta["prompt_tokens"] == 5

    def test_parse_engine_output_string(self):
        """_parse_engine_output 解析字符串输出"""
        adapter = self._make_loaded_adapter()
        text, meta = adapter._parse_engine_output("plain text")
        assert text == "plain text"
        assert meta == {}

    def test_parse_engine_output_empty_dict(self):
        """_parse_engine_output 解析空 dict"""
        adapter = self._make_loaded_adapter()
        text, meta = adapter._parse_engine_output({})
        assert text == ""
        assert meta == {}

    def test_parse_engine_output_nan_detected(self):
        """_parse_engine_output 检测到 NaN happened 时应返回空文本并记录错误"""
        adapter = self._make_loaded_adapter()
        nan_output = {
            "text": "",
            "meta_info": {
                "finish_reason": {
                    "type": "stop",
                    "matched": "NaN happened",
                },
                "prompt_tokens": 4352,
                "completion_tokens": 0,
            }
        }
        text, meta = adapter._parse_engine_output(nan_output)
        assert text == ""
        assert meta["finish_reason"]["matched"] == "NaN happened"

    def test_parse_engine_output_nan_with_partial_text(self):
        """NaN 发生前可能已生成部分文本, 应保留"""
        adapter = self._make_loaded_adapter()
        nan_output = {
            "text": "部分生成的文本",
            "meta_info": {
                "finish_reason": {
                    "type": "stop",
                    "matched": "NaN happened",
                },
            }
        }
        text, meta = adapter._parse_engine_output(nan_output)
        assert text == "部分生成的文本"

    def test_parse_engine_output_normal_finish_reason(self):
        """正常 finish_reason 不触发 NaN 检测"""
        adapter = self._make_loaded_adapter()
        normal_output = {
            "text": "正常输出",
            "meta_info": {
                "finish_reason": {
                    "type": "stop",
                    "matched": "<|im_end|>",
                },
            }
        }
        text, meta = adapter._parse_engine_output(normal_output)
        assert text == "正常输出"

    def test_parse_engine_output_list(self):
        """_parse_engine_output 解析 list 输出 (批量推理)"""
        adapter = self._make_loaded_adapter()
        list_output = [
            {"text": "first", "meta_info": {"prompt_tokens": 10}},
            {"text": "second", "meta_info": {"prompt_tokens": 10}},
        ]
        text, meta = adapter._parse_engine_output(list_output)
        assert text == "first"
        assert meta["prompt_tokens"] == 10

    def test_parse_engine_output_empty_list(self):
        """_parse_engine_output 解析空 list"""
        adapter = self._make_loaded_adapter()
        text, meta = adapter._parse_engine_output([])
        assert text == ""
        assert meta == {}

    def test_parse_engine_output_none(self):
        """_parse_engine_output 解析 None"""
        adapter = self._make_loaded_adapter()
        text, meta = adapter._parse_engine_output(None)
        assert text == ""
        assert meta == {}

    def test_parse_engine_output_object_with_text(self):
        """_parse_engine_output 解析带 .text 属性的对象"""
        adapter = self._make_loaded_adapter()
        obj = MagicMock()
        obj.text = "object text"
        obj.meta_info = {"key": "value"}
        text, meta = adapter._parse_engine_output(obj)
        assert text == "object text"
        assert meta == {"key": "value"}

    def test_parse_engine_output_object_with_outputs(self):
        """_parse_engine_output 解析带 .outputs 列表的对象"""
        adapter = self._make_loaded_adapter()
        inner = MagicMock()
        inner.text = "inner text"
        obj = MagicMock(spec=[])  # no .text attribute
        obj.outputs = [inner]
        # Ensure obj doesn't have .text
        assert not hasattr(obj, 'text')
        text, meta = adapter._parse_engine_output(obj)
        assert text == "inner text"


# ============================================================================
# 12. NaN 防护和内存管理测试
# ============================================================================

class TestSGLangNaNPrevention(TestSGLangLoad):
    """测试 NaN 防护和内存管理 (继承 TestSGLangLoad 的 mock 辅助方法)"""

    def test_quant_does_not_enable_fp32_lm_head(self):
        """量化模型不应启用 enable_fp32_lm_head (会导致 OOM)
        
        enable_fp32_lm_head=True 会在 _compute_lm_head 中将 hidden_states 和
        lm_head.weight 都转换为 float32, 需要额外 ~4.7 GiB 显存.
        对于 L20 (46GB) + 27B AWQ-int4 模型, 这会直接导致 OOM.
        """
        for quant in ["gptq", "awq", "4bit", "8bit"]:
            adapter = SGLangAdapter(
                model_name="test-model",
                device="cuda",
                quantization=quant,
            )
            mock_sgl, _ = self._load_with_mocks(adapter)
            call_kwargs = mock_sgl.Engine.call_args[1]
            assert 'enable_fp32_lm_head' not in call_kwargs, (
                f"Quantization '{quant}' should NOT set enable_fp32_lm_head "
                f"(causes ~4.7 GiB extra VRAM, OOM on L20 + 27B model)"
            )

    def test_no_quant_no_fp32_lm_head(self):
        """无量化时也不应设置 enable_fp32_lm_head"""
        adapter = SGLangAdapter(
            model_name="test-model",
            device="cuda",
            quantization="none",
        )
        mock_sgl, _ = self._load_with_mocks(adapter)
        call_kwargs = mock_sgl.Engine.call_args[1]
        assert 'enable_fp32_lm_head' not in call_kwargs, (
            "Non-quantized models should not set enable_fp32_lm_head"
        )

    def test_quant_sets_cuda_graph_max_bs(self):
        """量化模型应设置 cuda_graph_max_bs=4 减少 CUDA Graph 内存"""
        for quant in ["gptq", "awq", "4bit", "8bit"]:
            adapter = SGLangAdapter(
                model_name="test-model",
                device="cuda",
                quantization=quant,
            )
            mock_sgl, _ = self._load_with_mocks(adapter)
            call_kwargs = mock_sgl.Engine.call_args[1]
            assert call_kwargs.get('cuda_graph_max_bs') == 4, (
                f"Quantization '{quant}' should set cuda_graph_max_bs=4 "
                f"to reduce CUDA Graph memory usage"
            )

    def test_default_mem_fraction_static_is_0_80(self):
        """默认 mem_fraction_static 应为 0.80"""
        adapter = SGLangAdapter(model_name="test-model", device="cuda")
        assert adapter.mem_fraction_static == 0.80

    def test_generate_with_nan_returns_empty_text(self):
        """当 SGLang 返回 NaN happened 时, generate 应返回空文本"""
        adapter = SGLangAdapter(model_name="test-model-instruct", device="cpu", quantization="gptq")
        adapter._is_loaded = True
        adapter.engine = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.pad_token = "<pad>"
        adapter.tokenizer.eos_token = "<|im_end|>"
        adapter.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")

        # 模拟 NaN 输出
        adapter.engine.generate.return_value = {
            "text": "",
            "meta_info": {
                "finish_reason": {"type": "stop", "matched": "NaN happened"},
                "prompt_tokens": 4352,
                "completion_tokens": 0,
            }
        }

        output = adapter.generate("hello", max_new_tokens=512)
        assert output.text == ""
        assert output.output_tokens == 0


# ============================================================================
# 13. 异步推理接口测试
# ============================================================================

class TestSGLangAsyncGenerate:
    """测试异步推理接口"""

    def _make_loaded_adapter(self):
        adapter = SGLangAdapter(model_name="test-model-instruct", device="cpu")
        adapter._is_loaded = True
        adapter.engine = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.pad_token = "<pad>"
        adapter.tokenizer.eos_token = "<|im_end|>"
        adapter.tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        return adapter

    def test_async_generate_basic(self):
        """async_generate 基本功能"""
        adapter = self._make_loaded_adapter()

        async def fake_async_gen(prompt, params):
            return {"text": "async output", "meta_info": {"prompt_tokens": 5, "completion_tokens": 3}}

        adapter.engine.async_generate = fake_async_gen

        async def run():
            output = await adapter.async_generate("hello", max_new_tokens=100)
            assert output.text == "async output"
            assert output.input_tokens == 5
            assert output.output_tokens == 3

        asyncio.run(run())

    def test_async_forward_with_kv_injection_basic(self):
        """async_forward_with_kv_injection 基本功能"""
        adapter = self._make_loaded_adapter()

        async def fake_async_gen(prompt, params):
            return {
                "text": "async injected output",
                "meta_info": {"prompt_tokens": 10, "completion_tokens": 5}
            }

        adapter.engine.async_generate = fake_async_gen

        async def run():
            output = await adapter.async_forward_with_kv_injection(
                prompt="prefix\nhello",
                injected_kv=[],
                alpha=0.5,
                max_new_tokens=200,
            )
            assert output.text == "async injected output"
            assert output.metadata['alpha'] == 0.5
            assert output.metadata['injection_mode'] == 'sglang_native_radix_attention'

        asyncio.run(run())

    def test_async_generate_nan_output(self):
        """async_generate 遇到 NaN 时应返回空文本"""
        adapter = self._make_loaded_adapter()

        async def fake_async_gen(prompt, params):
            return {
                "text": "",
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": "NaN happened"},
                    "prompt_tokens": 4352,
                    "completion_tokens": 0,
                }
            }

        adapter.engine.async_generate = fake_async_gen

        async def run():
            output = await adapter.async_generate("hello", max_new_tokens=512)
            assert output.text == ""
            assert output.output_tokens == 0

        asyncio.run(run())

    def test_async_generate_empty_output_logs_warning(self):
        """async_generate 空输出时应记录警告 (包含 raw output)"""
        adapter = self._make_loaded_adapter()

        async def fake_async_gen(prompt, params):
            return {"text": "", "meta_info": {}}

        adapter.engine.async_generate = fake_async_gen

        async def run():
            output = await adapter.async_generate("hello")
            assert output.text == ""

        asyncio.run(run())
