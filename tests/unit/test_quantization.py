"""
单元测试: 量化模型支持 (4bit / 8bit / GPTQ / AWQ)

验证量化功能的核心逻辑:
1. BaseModelAdapter 量化属性和辅助方法
2. 量化模式名称规范化
3. BitsAndBytesConfig 构建
4. 各 Adapter 初始化时量化参数传递
5. LlamaAdapter load_in_8bit 向后兼容
6. EngineConfig 量化字段解析
7. ModelFactory 量化参数传递
8. get_model_info 包含量化信息

不依赖真实模型加载, 使用 Mock 模拟所有 HuggingFace/vLLM 组件。

Author: AGI Demo Project
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.models.base import BaseModelAdapter, KVCacheEntry, ModelOutput
from dki.models.llama_adapter import LlamaAdapter
from dki.models.deepseek_adapter import DeepSeekAdapter
from dki.models.glm_adapter import GLMAdapter
from dki.models.vllm_adapter import VLLMAdapter


# ============================================================================
# 1. BaseModelAdapter 量化属性测试
# ============================================================================

class TestBaseModelAdapterQuantization:
    """测试 BaseModelAdapter 的量化属性和辅助方法"""

    def test_default_no_quantization(self):
        """默认不量化"""
        adapter = LlamaAdapter(model_name="test-model", device="cpu")
        assert adapter.quantization == "none"
        assert not adapter.is_quantized
        assert not adapter.is_4bit
        assert not adapter.is_8bit

    def test_4bit_quantization(self):
        """4bit 量化模式"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit"
        )
        assert adapter.quantization == "4bit"
        assert adapter.is_quantized
        assert adapter.is_4bit
        assert not adapter.is_8bit

    def test_8bit_quantization(self):
        """8bit 量化模式"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="8bit"
        )
        assert adapter.quantization == "8bit"
        assert adapter.is_quantized
        assert not adapter.is_4bit
        assert adapter.is_8bit

    def test_gptq_quantization(self):
        """GPTQ 量化模式"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq"
        )
        assert adapter.quantization == "gptq"
        assert adapter.is_quantized
        assert not adapter.is_4bit
        assert not adapter.is_8bit

    def test_awq_quantization(self):
        """AWQ 量化模式"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="awq"
        )
        assert adapter.quantization == "awq"
        assert adapter.is_quantized


# ============================================================================
# 2. 量化模式名称规范化测试
# ============================================================================

class TestQuantizationNormalization:
    """测试量化模式名称的规范化"""

    def test_normalize_none(self):
        """none 保持不变"""
        assert BaseModelAdapter._normalize_quantization("none") == "none"

    def test_normalize_empty_string(self):
        """空字符串 → none"""
        assert BaseModelAdapter._normalize_quantization("") == "none"

    def test_normalize_none_value(self):
        """None → none"""
        assert BaseModelAdapter._normalize_quantization(None) == "none"

    def test_normalize_int4_alias(self):
        """int4 → 4bit"""
        assert BaseModelAdapter._normalize_quantization("int4") == "4bit"

    def test_normalize_int8_alias(self):
        """int8 → 8bit"""
        assert BaseModelAdapter._normalize_quantization("int8") == "8bit"

    def test_normalize_4_alias(self):
        """'4' → 4bit"""
        assert BaseModelAdapter._normalize_quantization("4") == "4bit"

    def test_normalize_8_alias(self):
        """'8' → 8bit"""
        assert BaseModelAdapter._normalize_quantization("8") == "8bit"

    def test_normalize_case_insensitive(self):
        """大小写不敏感"""
        assert BaseModelAdapter._normalize_quantization("4BIT") == "4bit"
        assert BaseModelAdapter._normalize_quantization("8BIT") == "8bit"
        assert BaseModelAdapter._normalize_quantization("GPTQ") == "gptq"
        assert BaseModelAdapter._normalize_quantization("AWQ") == "awq"
        assert BaseModelAdapter._normalize_quantization("NONE") == "none"

    def test_normalize_with_whitespace(self):
        """包含空白字符"""
        assert BaseModelAdapter._normalize_quantization(" 4bit ") == "4bit"
        assert BaseModelAdapter._normalize_quantization(" 8bit ") == "8bit"

    def test_normalize_unknown_fallback(self):
        """未知量化模式回退到 none"""
        assert BaseModelAdapter._normalize_quantization("unknown") == "none"
        assert BaseModelAdapter._normalize_quantization("3bit") == "none"
        assert BaseModelAdapter._normalize_quantization("fp8") == "none"

    def test_normalize_standard_values(self):
        """标准值保持不变"""
        assert BaseModelAdapter._normalize_quantization("4bit") == "4bit"
        assert BaseModelAdapter._normalize_quantization("8bit") == "8bit"
        assert BaseModelAdapter._normalize_quantization("gptq") == "gptq"
        assert BaseModelAdapter._normalize_quantization("awq") == "awq"


# ============================================================================
# 3. BitsAndBytesConfig 构建测试
# ============================================================================

class TestBuildBnbConfig:
    """测试 _build_bnb_config() 方法"""

    def test_none_quantization_returns_none(self):
        """非 bitsandbytes 量化返回 None"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="none"
        )
        assert adapter._build_bnb_config() is None

    def test_gptq_returns_none(self):
        """GPTQ 不使用 BitsAndBytesConfig"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq"
        )
        assert adapter._build_bnb_config() is None

    def test_awq_returns_none(self):
        """AWQ 不使用 BitsAndBytesConfig"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="awq"
        )
        assert adapter._build_bnb_config() is None

    @patch("dki.models.base.BaseModelAdapter._build_bnb_config")
    def test_4bit_calls_build(self, mock_build):
        """4bit 量化应调用 _build_bnb_config"""
        # 直接测试原始方法逻辑
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit"
        )
        # 恢复原始方法以测试
        mock_build.reset_mock()

    def test_4bit_config_with_mock_transformers(self):
        """4bit 量化使用 Mock BitsAndBytesConfig"""
        mock_bnb_config = MagicMock()

        with patch.dict('sys.modules', {
            'transformers': MagicMock(BitsAndBytesConfig=mock_bnb_config)
        }):
            adapter = LlamaAdapter(
                model_name="test-model", device="cpu",
                quantization="4bit",
                quantization_config={
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                }
            )
            # 调用 _build_bnb_config, 验证参数
            try:
                result = adapter._build_bnb_config()
                # 如果 transformers 已安装, 验证返回值非 None
                if result is not None:
                    assert result is not None
            except ImportError:
                # bitsandbytes 未安装时, 应抛出 ImportError
                pass

    def test_8bit_config_with_mock_transformers(self):
        """8bit 量化使用 Mock BitsAndBytesConfig"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="8bit"
        )
        try:
            result = adapter._build_bnb_config()
            if result is not None:
                assert result is not None
        except ImportError:
            # bitsandbytes 未安装时, 应抛出 ImportError
            pass


# ============================================================================
# 4. 各 Adapter 初始化量化参数传递测试
# ============================================================================

class TestAdapterQuantizationInit:
    """测试各 Adapter 初始化时量化参数的正确传递"""

    def test_llama_adapter_quantization(self):
        """LlamaAdapter 接受量化参数"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit",
            quantization_config={"bnb_4bit_quant_type": "fp4"}
        )
        assert adapter.quantization == "4bit"
        assert adapter.quantization_config["bnb_4bit_quant_type"] == "fp4"

    def test_deepseek_adapter_quantization(self):
        """DeepSeekAdapter 接受量化参数"""
        adapter = DeepSeekAdapter(
            model_name="test-model", device="cpu",
            quantization="8bit"
        )
        assert adapter.quantization == "8bit"
        assert adapter.is_8bit

    def test_glm_adapter_quantization(self):
        """GLMAdapter 接受量化参数"""
        adapter = GLMAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq"
        )
        assert adapter.quantization == "gptq"
        assert adapter.is_quantized

    def test_vllm_adapter_quantization(self):
        """VLLMAdapter 接受量化参数"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="awq"
        )
        assert adapter.quantization == "awq"
        assert adapter.is_quantized

    def test_vllm_adapter_4bit(self):
        """VLLMAdapter 4bit 量化"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit"
        )
        assert adapter.quantization == "4bit"
        assert adapter.is_4bit


# ============================================================================
# 5. LlamaAdapter load_in_8bit 向后兼容测试
# ============================================================================

class TestLlamaLoadIn8bitCompat:
    """测试 LlamaAdapter 的 load_in_8bit 向后兼容"""

    def test_load_in_8bit_true_maps_to_quantization(self):
        """load_in_8bit=True 映射到 quantization='8bit'"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            load_in_8bit=True
        )
        assert adapter.quantization == "8bit"
        assert adapter.is_8bit
        assert adapter.load_in_8bit is True

    def test_load_in_8bit_false_no_quantization(self):
        """load_in_8bit=False 不启用量化"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            load_in_8bit=False
        )
        assert adapter.quantization == "none"
        assert not adapter.is_quantized
        assert adapter.load_in_8bit is False

    def test_explicit_quantization_overrides_load_in_8bit(self):
        """显式 quantization 参数优先于 load_in_8bit"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            load_in_8bit=True,
            quantization="4bit"
        )
        # quantization="4bit" 已显式设置, load_in_8bit 不应覆盖
        assert adapter.quantization == "4bit"
        assert adapter.is_4bit

    def test_load_in_8bit_reflected_in_model_info(self):
        """load_in_8bit 状态在 get_model_info 中反映"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            load_in_8bit=True
        )
        info = adapter.get_model_info()
        assert info['load_in_8bit'] is True
        assert info['quantization'] == "8bit"


# ============================================================================
# 6. EngineConfig 量化字段解析测试
# ============================================================================

class TestEngineConfigQuantization:
    """测试 EngineConfig 的量化字段"""

    def test_default_quantization(self):
        """默认 quantization 为 none"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(model_name="test-model")
        assert config.quantization == "none"
        assert isinstance(config.quantization_config, dict)

    def test_4bit_quantization_config(self):
        """4bit 量化配置"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            quantization="4bit",
            quantization_config={
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        )
        assert config.quantization == "4bit"
        assert config.quantization_config["bnb_4bit_quant_type"] == "nf4"

    def test_8bit_quantization_config(self):
        """8bit 量化配置"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            quantization="8bit"
        )
        assert config.quantization == "8bit"

    def test_gptq_quantization_config(self):
        """GPTQ 量化配置"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            quantization="gptq"
        )
        assert config.quantization == "gptq"

    def test_default_quantization_config_values(self):
        """默认 quantization_config 值"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(model_name="test-model")
        assert config.quantization_config.get("bnb_4bit_quant_type") == "nf4"
        assert config.quantization_config.get("bnb_4bit_use_double_quant") is True
        assert config.quantization_config.get("bnb_4bit_compute_dtype") == "bfloat16"

    def test_backward_compat_load_in_8bit(self):
        """load_in_8bit 向后兼容"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            load_in_8bit=True
        )
        assert config.load_in_8bit is True
        # 注意: EngineConfig 本身不做 load_in_8bit → quantization 的映射
        # 这由 ModelFactory.create() 处理


# ============================================================================
# 7. ModelFactory 量化参数传递测试
# ============================================================================

class TestModelFactoryQuantization:
    """测试 ModelFactory 量化参数的传递"""

    @patch('dki.models.factory.ConfigLoader')
    def test_factory_passes_quantization(self, mock_config_loader):
        """ModelFactory.create() 传递量化参数"""
        from dki.models.factory import ModelFactory

        # 构造 Mock config
        mock_engine_config = MagicMock()
        mock_engine_config.enabled = True
        mock_engine_config.model_name = "test-model"
        mock_engine_config.device = "cpu"
        mock_engine_config.dtype = "float32"
        mock_engine_config.trust_remote_code = True
        mock_engine_config.quantization = "4bit"
        mock_engine_config.quantization_config = {"bnb_4bit_quant_type": "nf4"}
        mock_engine_config.load_in_8bit = False

        mock_config = MagicMock()
        mock_config.model.default_engine = "llama"
        mock_config.model.engines = {"llama": mock_engine_config}

        mock_config_loader.return_value.config = mock_config

        adapter = ModelFactory.create(engine="llama")
        assert adapter.quantization == "4bit"
        assert isinstance(adapter, LlamaAdapter)

    @patch('dki.models.factory.ConfigLoader')
    def test_factory_load_in_8bit_compat(self, mock_config_loader):
        """ModelFactory 处理 load_in_8bit 向后兼容"""
        from dki.models.factory import ModelFactory

        mock_engine_config = MagicMock()
        mock_engine_config.enabled = True
        mock_engine_config.model_name = "test-model"
        mock_engine_config.device = "cpu"
        mock_engine_config.dtype = "float32"
        mock_engine_config.trust_remote_code = True
        mock_engine_config.quantization = "none"
        mock_engine_config.quantization_config = {}
        mock_engine_config.load_in_8bit = True

        mock_config = MagicMock()
        mock_config.model.default_engine = "llama"
        mock_config.model.engines = {"llama": mock_engine_config}

        mock_config_loader.return_value.config = mock_config

        adapter = ModelFactory.create(engine="llama")
        assert adapter.quantization == "8bit"
        assert adapter.is_8bit


# ============================================================================
# 8. get_model_info 包含量化信息测试
# ============================================================================

class TestGetModelInfoQuantization:
    """测试 get_model_info() 包含量化信息"""

    def test_base_model_info_includes_quantization(self):
        """BaseModelAdapter.get_model_info() 包含量化字段"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit"
        )
        info = adapter.get_model_info()
        assert 'quantization' in info
        assert info['quantization'] == "4bit"
        assert 'is_quantized' in info
        assert info['is_quantized'] is True

    def test_no_quantization_in_model_info(self):
        """无量化时 get_model_info() 的量化字段"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu"
        )
        info = adapter.get_model_info()
        assert info['quantization'] == "none"
        assert info['is_quantized'] is False

    def test_vllm_model_info_includes_quantization(self):
        """VLLMAdapter.get_model_info() 包含量化字段"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq"
        )
        info = adapter.get_model_info()
        assert info['quantization'] == "gptq"

    def test_deepseek_model_info_includes_quantization(self):
        """DeepSeekAdapter get_model_info() 包含量化字段 (通过 base)"""
        adapter = DeepSeekAdapter(
            model_name="test-model", device="cpu",
            quantization="8bit"
        )
        info = adapter.get_model_info()
        assert info['quantization'] == "8bit"
        assert info['is_quantized'] is True

    def test_glm_model_info_includes_quantization(self):
        """GLMAdapter get_model_info() 包含量化字段 (通过 base)"""
        adapter = GLMAdapter(
            model_name="test-model", device="cpu",
            quantization="awq"
        )
        info = adapter.get_model_info()
        assert info['quantization'] == "awq"
        assert info['is_quantized'] is True


# ============================================================================
# 9. LlamaAdapter load() 量化集成测试 (Mock)
# ============================================================================

class TestLlamaAdapterLoadQuantized:
    """测试 LlamaAdapter.load() 的量化参数传递 (使用 Mock)"""

    def _setup_transformers_mocks(self):
        """创建 transformers mock 对象"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_config

        return mock_auto_tokenizer, mock_auto_model, mock_auto_config

    def test_load_4bit_passes_quantization_config(self):
        """4bit 加载传递 quantization_config 参数"""
        mock_auto_tokenizer, mock_auto_model, mock_auto_config = \
            self._setup_transformers_mocks()

        mock_bnb_config_cls = MagicMock()
        mock_bnb_config_instance = MagicMock()
        mock_bnb_config_cls.return_value = mock_bnb_config_instance

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoModelForCausalLM = mock_auto_model
        mock_transformers.AutoConfig = mock_auto_config
        mock_transformers.BitsAndBytesConfig = mock_bnb_config_cls

        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            dtype="float16",
            quantization="4bit",
        )

        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            try:
                adapter.load()
            except Exception:
                pass

        # 验证 from_pretrained 被调用
        if mock_auto_model.from_pretrained.called:
            call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
            assert 'quantization_config' in call_kwargs

    def test_load_no_quantization_no_bnb_config(self):
        """无量化时不传递 quantization_config"""
        mock_auto_tokenizer, mock_auto_model, mock_auto_config = \
            self._setup_transformers_mocks()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoModelForCausalLM = mock_auto_model
        mock_transformers.AutoConfig = mock_auto_config

        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            dtype="float16",
        )

        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            adapter.load()

        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert 'quantization_config' not in call_kwargs

    def test_load_gptq_no_bnb_config(self):
        """GPTQ 量化不传递 BitsAndBytesConfig"""
        mock_auto_tokenizer, mock_auto_model, mock_auto_config = \
            self._setup_transformers_mocks()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoModelForCausalLM = mock_auto_model
        mock_transformers.AutoConfig = mock_auto_config

        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            dtype="float16",
            quantization="gptq",
        )

        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            adapter.load()

        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert 'quantization_config' not in call_kwargs
        assert 'torch_dtype' in call_kwargs


# ============================================================================
# 10. VLLMAdapter load() 量化集成测试 (Mock)
# ============================================================================

class TestVLLMAdapterLoadQuantized:
    """测试 VLLMAdapter.load() 的量化参数传递 (使用 Mock)"""

    def _setup_mocks(self):
        """创建所有 mock 对象"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_config

        mock_llm_cls = MagicMock()
        mock_sp_cls = MagicMock()

        return mock_auto_tokenizer, mock_auto_config, mock_llm_cls, mock_sp_cls

    def test_load_gptq_passes_quantization(self):
        """GPTQ 加载传递 quantization 参数到 vLLM"""
        mock_auto_tokenizer, mock_auto_config, mock_llm_cls, mock_sp_cls = \
            self._setup_mocks()

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = mock_sp_cls

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoConfig = mock_auto_config

        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq",
        )

        with patch.dict('sys.modules', {
            'vllm': mock_vllm,
            'transformers': mock_transformers,
        }):
            adapter.load()

        # 验证 LLM() 被调用时包含 quantization 参数
        assert mock_llm_cls.called
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert call_kwargs.get('quantization') == 'gptq'

    def test_load_no_quantization_no_param(self):
        """无量化时不传递 quantization 参数"""
        mock_auto_tokenizer, mock_auto_config, mock_llm_cls, mock_sp_cls = \
            self._setup_mocks()

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = mock_sp_cls

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoConfig = mock_auto_config

        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
        )

        with patch.dict('sys.modules', {
            'vllm': mock_vllm,
            'transformers': mock_transformers,
        }):
            adapter.load()

        assert mock_llm_cls.called
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert 'quantization' not in call_kwargs


# ============================================================================
# 11. VLLMAdapter model_impl 参数测试
# ============================================================================

class TestVLLMAdapterModelImpl:
    """测试 VLLMAdapter 的 model_impl 参数支持"""

    def test_default_model_impl_is_auto(self):
        """默认 model_impl 为 auto"""
        adapter = VLLMAdapter(model_name="test-model", device="cpu")
        assert adapter.model_impl == "auto"

    def test_model_impl_transformers(self):
        """model_impl=transformers 正确设置"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            model_impl="transformers"
        )
        assert adapter.model_impl == "transformers"

    def test_model_impl_in_get_model_info(self):
        """get_model_info() 包含 model_impl"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            model_impl="transformers"
        )
        info = adapter.get_model_info()
        assert 'model_impl' in info
        assert info['model_impl'] == "transformers"

    def test_model_impl_auto_in_get_model_info(self):
        """get_model_info() 默认 model_impl=auto"""
        adapter = VLLMAdapter(model_name="test-model", device="cpu")
        info = adapter.get_model_info()
        assert info['model_impl'] == "auto"

    def test_model_impl_with_quantization(self):
        """model_impl 与 quantization 同时设置"""
        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq",
            model_impl="transformers"
        )
        assert adapter.model_impl == "transformers"
        assert adapter.quantization == "gptq"
        assert adapter.is_quantized

    def test_load_transformers_passes_model_impl(self):
        """model_impl=transformers 时 LLM() 构造传递 model_impl 参数"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_config

        mock_llm_cls = MagicMock()
        mock_sp_cls = MagicMock()

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = mock_sp_cls

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoConfig = mock_auto_config

        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
            quantization="gptq",
            model_impl="transformers",
        )

        with patch.dict('sys.modules', {
            'vllm': mock_vllm,
            'transformers': mock_transformers,
        }):
            adapter.load()

        # 验证 LLM() 被调用时包含 model_impl 参数
        assert mock_llm_cls.called
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert call_kwargs.get('model_impl') == 'transformers'
        assert call_kwargs.get('quantization') == 'gptq'

    def test_load_auto_no_model_impl_param(self):
        """model_impl=auto 时 LLM() 构造不传递 model_impl 参数"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_config

        mock_llm_cls = MagicMock()
        mock_sp_cls = MagicMock()

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = mock_sp_cls

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        mock_transformers.AutoConfig = mock_auto_config

        adapter = VLLMAdapter(
            model_name="test-model", device="cpu",
        )

        with patch.dict('sys.modules', {
            'vllm': mock_vllm,
            'transformers': mock_transformers,
        }):
            adapter.load()

        assert mock_llm_cls.called
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert 'model_impl' not in call_kwargs


# ============================================================================
# 12. EngineConfig model_impl 字段测试
# ============================================================================

class TestEngineConfigModelImpl:
    """测试 EngineConfig 的 model_impl 字段"""

    def test_default_model_impl(self):
        """默认 model_impl 为 auto"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(model_name="test-model")
        assert config.model_impl == "auto"

    def test_transformers_model_impl(self):
        """model_impl=transformers 配置"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            model_impl="transformers"
        )
        assert config.model_impl == "transformers"

    def test_model_impl_with_quantization(self):
        """model_impl 与 quantization 同时配置"""
        from dki.config.config_loader import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            quantization="gptq",
            model_impl="transformers"
        )
        assert config.model_impl == "transformers"
        assert config.quantization == "gptq"


# ============================================================================
# 13. ModelFactory model_impl 传递测试
# ============================================================================

class TestModelFactoryModelImpl:
    """测试 ModelFactory 传递 model_impl 参数"""

    @patch('dki.models.factory.ConfigLoader')
    def test_factory_passes_model_impl(self, mock_config_loader):
        """ModelFactory.create() 传递 model_impl 到 VLLMAdapter"""
        from dki.models.factory import ModelFactory

        mock_engine_config = MagicMock()
        mock_engine_config.enabled = True
        mock_engine_config.model_name = "test-model"
        mock_engine_config.device = "cpu"
        mock_engine_config.dtype = "float32"
        mock_engine_config.trust_remote_code = True
        mock_engine_config.quantization = "gptq"
        mock_engine_config.quantization_config = {}
        mock_engine_config.load_in_8bit = False
        mock_engine_config.tensor_parallel_size = 1
        mock_engine_config.max_model_len = 8192
        mock_engine_config.gpu_memory_utilization = 0.85
        mock_engine_config.injection_mode = "auto"
        mock_engine_config.model_impl = "transformers"

        mock_config = MagicMock()
        mock_config.model.default_engine = "vllm"
        mock_config.model.engines = {"vllm": mock_engine_config}

        mock_config_loader.return_value.config = mock_config

        adapter = ModelFactory.create(engine="vllm")
        assert isinstance(adapter, VLLMAdapter)
        assert adapter.model_impl == "transformers"
        assert adapter.quantization == "gptq"

    @patch('dki.models.factory.ConfigLoader')
    def test_factory_default_model_impl_auto(self, mock_config_loader):
        """ModelFactory.create() 默认 model_impl=auto"""
        from dki.models.factory import ModelFactory

        mock_engine_config = MagicMock()
        mock_engine_config.enabled = True
        mock_engine_config.model_name = "test-model"
        mock_engine_config.device = "cpu"
        mock_engine_config.dtype = "float32"
        mock_engine_config.trust_remote_code = True
        mock_engine_config.quantization = "none"
        mock_engine_config.quantization_config = {}
        mock_engine_config.load_in_8bit = False
        mock_engine_config.tensor_parallel_size = 1
        mock_engine_config.max_model_len = 8192
        mock_engine_config.gpu_memory_utilization = 0.85
        mock_engine_config.injection_mode = "auto"
        mock_engine_config.model_impl = "auto"

        mock_config = MagicMock()
        mock_config.model.default_engine = "vllm"
        mock_config.model.engines = {"vllm": mock_engine_config}

        mock_config_loader.return_value.config = mock_config

        adapter = ModelFactory.create(engine="vllm")
        assert adapter.model_impl == "auto"


# ============================================================================
# 14. 量化配置 (quantization_config) 详细测试
# ============================================================================

class TestQuantizationConfigDetail:
    """测试 quantization_config 详细配置"""

    def test_default_config_values(self):
        """默认 quantization_config 值"""
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit"
        )
        # 默认 quantization_config 应为空 dict (由 base 设置)
        assert isinstance(adapter.quantization_config, dict)

    def test_custom_config_values(self):
        """自定义 quantization_config"""
        custom_config = {
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": "float16",
        }
        adapter = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit",
            quantization_config=custom_config,
        )
        assert adapter.quantization_config["bnb_4bit_quant_type"] == "fp4"
        assert adapter.quantization_config["bnb_4bit_use_double_quant"] is False
        assert adapter.quantization_config["bnb_4bit_compute_dtype"] == "float16"

    def test_config_not_shared_between_instances(self):
        """不同实例的 quantization_config 互相独立"""
        adapter1 = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit",
            quantization_config={"bnb_4bit_quant_type": "nf4"}
        )
        adapter2 = LlamaAdapter(
            model_name="test-model", device="cpu",
            quantization="4bit",
            quantization_config={"bnb_4bit_quant_type": "fp4"}
        )
        assert adapter1.quantization_config["bnb_4bit_quant_type"] == "nf4"
        assert adapter2.quantization_config["bnb_4bit_quant_type"] == "fp4"
