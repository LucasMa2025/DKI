"""
单元测试: LlamaAdapter 真正的 K/V 注入实现

验证 LlamaAdapter 的核心功能:
1. 初始化与配置
2. Chat Template 处理 (Llama 2/3 检测)
3. compute_kv() — 偏好 KV 计算 (含截断保护)
4. forward_with_kv_injection() — K/V 注入推理
5. _build_attention_bias() — Attention Bias 构建 (论文 §4.2)
6. _prepare_kv_for_injection() — KV 准备 (Value 缩放, Key 不缩放)
7. compute_prefill_entropy() — 熵计算
8. Executor 兼容性 (无 injection_mode 属性)
9. 安全保证 (alpha 截断, 降级回退)

不依赖真实模型加载, 使用 Mock 模拟所有 HuggingFace 组件。

Author: AGI Demo Project
"""

import math
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call

import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.models.llama_adapter import LlamaAdapter
from dki.models.base import KVCacheEntry, ModelOutput


# ============================================================================
# Helper: 模拟 tokenizer 输出 (支持 .to() + dict 访问)
# ============================================================================

class _TokenizerOutput(dict):
    """模拟 tokenizer 输出 (支持 .to() 方法和属性访问)"""
    def to(self, device):
        return _TokenizerOutput({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.items()
        })

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_tokenizer_output(seq_len: int = 5):
    """创建标准 tokenizer 输出"""
    return _TokenizerOutput({
        'input_ids': torch.tensor([[1, 2, 3, 4, 5][:seq_len]]),
        'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
    })


def _make_long_tokenizer_output(seq_len: int = 300):
    """创建超长 tokenizer 输出 (用于截断测试)"""
    return _TokenizerOutput({
        'input_ids': torch.ones(1, seq_len, dtype=torch.long),
        'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
    })


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def adapter():
    """创建未加载的 LlamaAdapter 实例"""
    return LlamaAdapter(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        device="cpu",
        dtype="float32",
    )


@pytest.fixture
def adapter_llama2():
    """创建 Llama 2 Chat 适配器"""
    return LlamaAdapter(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device="cpu",
        dtype="float32",
    )


@pytest.fixture
def mock_kv_entries():
    """创建模拟的 KVCacheEntry 列表 (4 层, 2 头, 10 token, 64 dim)"""
    entries = []
    for layer_idx in range(4):
        entry = KVCacheEntry(
            key=torch.randn(1, 2, 10, 64),
            value=torch.randn(1, 2, 10, 64),
            layer_idx=layer_idx,
        )
        entries.append(entry)
    return entries


def _build_mock_tokenizer():
    """构建完整的 mock tokenizer (支持 __call__ → _TokenizerOutput)"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.unk_token_id = 3
    mock_tokenizer.convert_tokens_to_ids.return_value = 128009  # <|eot_id|>
    mock_tokenizer.decode.return_value = "test output"
    # apply_chat_template 返回格式化后的字符串
    mock_tokenizer.apply_chat_template.return_value = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    # __call__ 返回 _TokenizerOutput (支持 .to())
    mock_tokenizer.return_value = _make_tokenizer_output(5)
    return mock_tokenizer


@pytest.fixture
def loaded_adapter(adapter):
    """创建已加载的 Mock LlamaAdapter"""
    adapter._is_loaded = True
    adapter.num_layers = 4
    adapter.num_heads = 2
    adapter.hidden_dim = 128
    adapter.head_dim = 64
    adapter.device = "cpu"

    adapter.tokenizer = _build_mock_tokenizer()
    adapter.model = MagicMock()

    return adapter


# ============================================================================
# 初始化与配置
# ============================================================================

class TestLlamaAdapterInit:
    """验证 LlamaAdapter 初始化"""

    def test_default_init(self):
        """默认初始化参数"""
        adapter = LlamaAdapter()
        assert adapter.model_name == "meta-llama/Llama-3.2-3B-Instruct"
        assert adapter.device == "cuda"
        assert adapter.load_in_8bit is False
        assert adapter.trust_remote_code is True

    def test_custom_init(self):
        """自定义初始化参数"""
        adapter = LlamaAdapter(
            model_name="custom/model",
            device="cpu",
            dtype="float32",
            load_in_8bit=True,
        )
        assert adapter.model_name == "custom/model"
        assert adapter.device == "cpu"
        assert adapter.load_in_8bit is True

    def test_safety_constants(self, adapter):
        """安全常量验证"""
        assert adapter.ALPHA_OVERRIDE_CAP == 0.7
        assert adapter.MAX_PREF_TOKENS == 200
        assert adapter.DEFAULT_ENTROPY == 0.5

    def test_no_injection_mode_attribute(self, adapter):
        """LlamaAdapter 不应有 injection_mode 属性 (Executor 兼容性)"""
        assert not hasattr(adapter, 'injection_mode')

    def test_executor_prompt_prefix_mode_returns_false(self, adapter):
        """
        Executor._is_prompt_prefix_mode() 应返回 False
        (因为 LlamaAdapter 无 injection_mode 属性)
        """
        injection_mode = getattr(adapter, 'injection_mode', None)
        assert injection_mode is None


# ============================================================================
# Chat Template 检测
# ============================================================================

class TestChatTemplateDetection:
    """验证 Chat Template 检测逻辑"""

    def test_is_chat_model_instruct(self, adapter):
        """Instruct 模型应被识别为 chat model"""
        assert adapter._is_chat_model() is True

    def test_is_chat_model_base(self):
        """Base 模型不应被识别为 chat model"""
        adapter = LlamaAdapter(model_name="meta-llama/Llama-3.2-3B")
        assert adapter._is_chat_model() is False

    def test_is_llama3(self, adapter):
        """Llama 3 模型应被正确识别"""
        assert adapter._is_llama3() is True

    def test_is_not_llama3(self, adapter_llama2):
        """Llama 2 模型不应被识别为 Llama 3"""
        assert adapter_llama2._is_llama3() is False

    def test_has_chat_template_tokens_llama3(self, adapter):
        """检测 Llama 3 chat template 标记"""
        text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello"
        assert adapter._has_chat_template_tokens(text) is True

    def test_has_chat_template_tokens_llama2(self, adapter):
        """检测 Llama 2 chat template 标记"""
        text = "[INST] Hello [/INST]"
        assert adapter._has_chat_template_tokens(text) is True

    def test_has_chat_template_tokens_chatml(self, adapter):
        """检测 ChatML 标记"""
        text = "<|im_start|>user\nHello<|im_end|>"
        assert adapter._has_chat_template_tokens(text) is True

    def test_no_chat_template_tokens(self, adapter):
        """普通文本不应被检测为含 chat template"""
        text = "Hello, how are you?"
        assert adapter._has_chat_template_tokens(text) is False

    def test_format_prompt_safe_skips_already_formatted(self, loaded_adapter):
        """已含 template 标记的 prompt 不应被二次包装"""
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello"
        result = loaded_adapter._format_prompt_safe(prompt)
        assert result == prompt

    def test_format_prompt_llama2_fallback(self, adapter_llama2):
        """Llama 2 手动模板回退"""
        adapter_llama2.tokenizer = MagicMock()
        adapter_llama2.tokenizer.apply_chat_template.side_effect = Exception("no template")

        result = adapter_llama2._format_prompt("Hello")
        assert "[INST]" in result
        assert "Hello" in result
        assert "[/INST]" in result

    def test_format_prompt_llama2_with_system(self, adapter_llama2):
        """Llama 2 手动模板 (含 system prompt)"""
        adapter_llama2.tokenizer = MagicMock()
        adapter_llama2.tokenizer.apply_chat_template.side_effect = Exception("no template")

        result = adapter_llama2._format_prompt("Hello", system_prompt="You are helpful")
        assert "<<SYS>>" in result
        assert "You are helpful" in result
        assert "<</SYS>>" in result


# ============================================================================
# Attention Bias 构建 (论文 §4.2)
# ============================================================================

class TestAttentionBias:
    """验证 _build_attention_bias 的正确性"""

    def test_bias_shape(self, adapter):
        """Attention bias 形状应为 [1, 1, n_query, n_pref + n_query]"""
        bias = adapter._build_attention_bias(
            n_pref=10, n_query=5, alpha=0.5,
            device=torch.device('cpu'), dtype=torch.float32,
        )
        assert bias.shape == (1, 1, 5, 15)

    def test_bias_pref_positions_with_alpha(self, adapter):
        """alpha < 1 时偏好位置应有 log(alpha) bias"""
        alpha = 0.5
        bias = adapter._build_attention_bias(
            n_pref=10, n_query=5, alpha=alpha,
            device=torch.device('cpu'), dtype=torch.float32,
        )
        expected_bias = math.log(alpha)
        # 偏好位置 (前 10 列) 应有 log(0.5) bias
        assert torch.allclose(
            bias[0, 0, :, :10],
            torch.full((5, 10), expected_bias),
            atol=1e-6,
        )

    def test_bias_alpha_one_no_bias(self, adapter):
        """alpha=1.0 时偏好位置 bias 应为 0"""
        bias = adapter._build_attention_bias(
            n_pref=10, n_query=5, alpha=1.0,
            device=torch.device('cpu'), dtype=torch.float32,
        )
        # 偏好位置 bias 应为 0
        assert torch.allclose(
            bias[0, 0, :, :10],
            torch.zeros(5, 10),
        )

    def test_bias_alpha_zero_blocks_pref(self, adapter):
        """alpha=0 时偏好位置应被完全屏蔽 (-inf)"""
        bias = adapter._build_attention_bias(
            n_pref=10, n_query=5, alpha=0.0,
            device=torch.device('cpu'), dtype=torch.float32,
        )
        # 偏好位置应为 -inf
        assert torch.all(bias[0, 0, :, :10] == float('-inf'))

    def test_bias_causal_mask(self, adapter):
        """查询部分应有 causal mask (下三角)"""
        bias = adapter._build_attention_bias(
            n_pref=3, n_query=4, alpha=1.0,
            device=torch.device('cpu'), dtype=torch.float32,
        )
        # query token 0 不能 attend 到 query token 1, 2, 3
        assert bias[0, 0, 0, 4] == float('-inf')  # n_pref + 1
        assert bias[0, 0, 0, 5] == float('-inf')  # n_pref + 2
        assert bias[0, 0, 0, 6] == float('-inf')  # n_pref + 3

        # query token 0 可以 attend 到自身
        assert bias[0, 0, 0, 3] == 0.0  # n_pref + 0

        # query token 3 (最后一个) 可以 attend 到所有
        assert bias[0, 0, 3, 3] == 0.0
        assert bias[0, 0, 3, 4] == 0.0
        assert bias[0, 0, 3, 5] == 0.0
        assert bias[0, 0, 3, 6] == 0.0


# ============================================================================
# KV 准备 (Value 缩放)
# ============================================================================

class TestKVPreparation:
    """验证 _prepare_kv_for_injection 的 Value 缩放逻辑"""

    def test_value_scaled_key_unchanged(self, adapter, mock_kv_entries):
        """Value 应被 alpha 缩放, Key 不应被缩放"""
        alpha = 0.5
        original_keys = [e.key.clone() for e in mock_kv_entries]
        original_values = [e.value.clone() for e in mock_kv_entries]

        past_kv, mem_len = adapter._prepare_kv_for_injection(
            mock_kv_entries, alpha, torch.device('cpu')
        )

        assert mem_len == 10

        for i, (key, value) in enumerate(past_kv):
            # Key 不缩放
            assert torch.allclose(key, original_keys[i], atol=1e-6), \
                f"Layer {i}: Key should NOT be scaled"
            # Value 缩放
            assert torch.allclose(value, original_values[i] * alpha, atol=1e-6), \
                f"Layer {i}: Value should be scaled by alpha={alpha}"

    def test_alpha_one_no_scaling(self, adapter, mock_kv_entries):
        """alpha=1.0 时 Value 不应被缩放"""
        original_values = [e.value.clone() for e in mock_kv_entries]

        # alpha=1.0 但会被 ALPHA_OVERRIDE_CAP=0.7 截断
        past_kv, _ = adapter._prepare_kv_for_injection(
            mock_kv_entries, 1.0, torch.device('cpu')
        )

        for i, (key, value) in enumerate(past_kv):
            # alpha=1.0 被截断为 0.7, 所以 Value 应被缩放
            assert torch.allclose(
                value, original_values[i] * adapter.ALPHA_OVERRIDE_CAP, atol=1e-6
            )

    def test_alpha_clamped_by_override_cap(self, adapter, mock_kv_entries):
        """alpha 应被 ALPHA_OVERRIDE_CAP 截断"""
        original_values = [e.value.clone() for e in mock_kv_entries]

        past_kv, _ = adapter._prepare_kv_for_injection(
            mock_kv_entries, 0.9, torch.device('cpu')
        )

        # 0.9 > 0.7, 应被截断为 0.7
        for i, (key, value) in enumerate(past_kv):
            assert torch.allclose(
                value, original_values[i] * 0.7, atol=1e-6
            )

    def test_mem_len_correct(self, adapter, mock_kv_entries):
        """mem_len 应等于偏好 token 数"""
        _, mem_len = adapter._prepare_kv_for_injection(
            mock_kv_entries, 0.5, torch.device('cpu')
        )
        assert mem_len == 10  # mock_kv_entries 的 seq_len=10


# ============================================================================
# compute_kv (偏好 KV 计算)
# ============================================================================

class TestComputeKV:
    """验证 compute_kv 的偏好编码逻辑"""

    def test_compute_kv_returns_entries(self, loaded_adapter):
        """compute_kv 应返回 KVCacheEntry 列表"""
        # Mock model forward
        mock_past_kv = tuple(
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
            for _ in range(4)
        )
        mock_output = MagicMock()
        mock_output.past_key_values = mock_past_kv
        mock_output.hidden_states = None
        loaded_adapter.model.return_value = mock_output

        kv_entries, hidden = loaded_adapter.compute_kv("素食主义者")

        assert len(kv_entries) == 4
        assert hidden is None
        for i, entry in enumerate(kv_entries):
            assert isinstance(entry, KVCacheEntry)
            assert entry.layer_idx == i
            assert entry.key.device == torch.device('cpu')
            assert entry.value.device == torch.device('cpu')

    def test_compute_kv_truncates_long_text(self, loaded_adapter):
        """超过 MAX_PREF_TOKENS 的偏好应被截断"""
        # Mock tokenizer 返回超长 input_ids
        loaded_adapter.tokenizer.return_value = _make_long_tokenizer_output(300)

        # Mock model forward — 截断后应为 200 token
        mock_past_kv = tuple(
            (torch.randn(1, 2, 200, 64), torch.randn(1, 2, 200, 64))
            for _ in range(4)
        )
        mock_output = MagicMock()
        mock_output.past_key_values = mock_past_kv
        mock_output.hidden_states = None
        loaded_adapter.model.return_value = mock_output

        kv_entries, _ = loaded_adapter.compute_kv("very long preference text...")

        # 验证 model 被调用时 input_ids 被截断到 MAX_PREF_TOKENS
        call_kwargs = loaded_adapter.model.call_args[1]
        actual_input_ids = call_kwargs['input_ids']
        assert actual_input_ids.shape[1] == loaded_adapter.MAX_PREF_TOKENS
        assert len(kv_entries) == 4

    def test_compute_kv_with_hidden_states(self, loaded_adapter):
        """return_hidden=True 时应返回 hidden states"""
        mock_past_kv = tuple(
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
            for _ in range(4)
        )
        mock_hidden = (torch.randn(1, 5, 128),) * 5  # 5 层 hidden states
        mock_output = MagicMock()
        mock_output.past_key_values = mock_past_kv
        mock_output.hidden_states = mock_hidden
        loaded_adapter.model.return_value = mock_output

        kv_entries, hidden = loaded_adapter.compute_kv("test", return_hidden=True)

        assert len(kv_entries) == 4
        assert hidden is not None
        assert hidden.device == torch.device('cpu')


# ============================================================================
# forward_with_kv_injection (K/V 注入推理)
# ============================================================================

class TestForwardWithKVInjection:
    """验证 forward_with_kv_injection 的注入逻辑"""

    def test_no_kv_falls_back_to_generate(self, loaded_adapter):
        """无 KV 时应降级为标准 generate"""
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=[],
            alpha=0.5,
        )

        assert isinstance(result, ModelOutput)

    def test_low_alpha_falls_back(self, loaded_adapter):
        """alpha 太低时应降级为标准 generate"""
        mock_kv = [KVCacheEntry(
            key=torch.randn(1, 2, 5, 64),
            value=torch.randn(1, 2, 5, 64),
            layer_idx=0,
        )]

        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv,
            alpha=0.005,  # 太低
        )

        assert isinstance(result, ModelOutput)

    def test_alpha_clamped(self, loaded_adapter, mock_kv_entries):
        """alpha 应被 ALPHA_OVERRIDE_CAP 截断"""
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv_entries,
            alpha=0.9,  # > 0.7
        )

        assert isinstance(result, ModelOutput)
        assert result.metadata.get('alpha_clamped', 1.0) <= loaded_adapter.ALPHA_OVERRIDE_CAP

    def test_injection_extends_attention_mask(self, loaded_adapter, mock_kv_entries):
        """注入时 attention_mask 应被扩展"""
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv_entries,
            alpha=0.5,
        )

        # 验证 generate 被调用
        assert loaded_adapter.model.generate.called

        # 验证 attention_mask 被扩展 (原 5 token + 10 偏好 token = 15)
        call_kwargs = loaded_adapter.model.generate.call_args[1]
        extended_mask = call_kwargs['attention_mask']
        assert extended_mask.shape[1] == 15  # 10 (pref) + 5 (query)

    def test_injection_passes_past_key_values(self, loaded_adapter, mock_kv_entries):
        """注入时应传递 past_key_values"""
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv_entries,
            alpha=0.5,
        )

        call_kwargs = loaded_adapter.model.generate.call_args[1]
        past_kv = call_kwargs['past_key_values']
        assert past_kv is not None
        assert len(past_kv) == 4  # 4 layers

    def test_exception_falls_back_to_generate(self, loaded_adapter, mock_kv_entries):
        """注入失败时应降级为标准 generate"""
        # 第一次 generate 调用 (注入路径) 抛出异常
        # 第二次 generate 调用 (降级路径) 返回正常结果
        loaded_adapter.model.generate.side_effect = [
            RuntimeError("injection failed"),
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),  # 降级调用
        ]

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv_entries,
            alpha=0.5,
        )

        assert isinstance(result, ModelOutput)
        # generate 应被调用两次 (注入失败 + 降级)
        assert loaded_adapter.model.generate.call_count == 2

    def test_metadata_contains_injection_info(self, loaded_adapter, mock_kv_entries):
        """返回的 metadata 应包含注入信息"""
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        loaded_adapter.model.generate.return_value = mock_outputs

        result = loaded_adapter.forward_with_kv_injection(
            prompt="Hello",
            injected_kv=mock_kv_entries,
            alpha=0.5,
        )

        assert 'alpha' in result.metadata
        assert 'mem_len' in result.metadata
        assert 'injection_mode' in result.metadata
        assert result.metadata['injection_mode'] == 'hf_kv_negative_position'
        assert result.metadata['mem_len'] == 10


# ============================================================================
# compute_prefill_entropy
# ============================================================================

class TestPrefillEntropy:
    """验证 prefill 熵计算"""

    def test_returns_default_on_no_attentions(self, loaded_adapter):
        """无 attention 输出时返回默认值"""
        mock_output = MagicMock()
        mock_output.attentions = None
        loaded_adapter.model.return_value = mock_output

        entropy = loaded_adapter.compute_prefill_entropy("test")
        assert entropy == loaded_adapter.DEFAULT_ENTROPY

    def test_returns_valid_entropy(self, loaded_adapter):
        """正常情况返回有效熵值"""
        # 创建 mock attention weights (uniform distribution → high entropy)
        seq_len = 5
        attn = torch.ones(1, 2, seq_len, seq_len) / seq_len  # uniform
        mock_output = MagicMock()
        mock_output.attentions = [attn] * 4  # 4 layers
        loaded_adapter.model.return_value = mock_output

        entropy = loaded_adapter.compute_prefill_entropy("test", layer_idx=0)
        assert isinstance(entropy, float)
        assert entropy > 0  # uniform distribution has positive entropy

    def test_returns_default_on_exception(self, loaded_adapter):
        """异常时返回默认值"""
        loaded_adapter.model.side_effect = RuntimeError("model error")

        entropy = loaded_adapter.compute_prefill_entropy("test")
        assert entropy == loaded_adapter.DEFAULT_ENTROPY


# ============================================================================
# get_model_info
# ============================================================================

class TestModelInfo:
    """验证模型信息"""

    def test_model_info_keys(self, adapter):
        """get_model_info 应包含必要字段"""
        info = adapter.get_model_info()

        assert 'adapter_type' in info
        assert info['adapter_type'] == 'llama_hf_kv_injection'
        assert 'kv_injection_type' in info
        assert info['kv_injection_type'] == 'negative_position'
        assert 'alpha_override_cap' in info
        assert info['alpha_override_cap'] == 0.7
        assert 'max_pref_tokens' in info
        assert info['max_pref_tokens'] == 200
        assert 'attention_bias_available' in info
        assert info['attention_bias_available'] is True


# ============================================================================
# Stop Token IDs
# ============================================================================

class TestStopTokenIds:
    """验证 stop token 获取"""

    def test_llama3_has_eot_id(self, loaded_adapter):
        """Llama 3 模型应包含 <|eot_id|> stop token"""
        stop_ids = loaded_adapter._get_stop_token_ids()
        assert 128009 in stop_ids  # <|eot_id|>

    def test_always_has_eos(self, loaded_adapter):
        """应始终包含 eos_token_id"""
        stop_ids = loaded_adapter._get_stop_token_ids()
        assert 2 in stop_ids  # eos_token_id

    def test_no_tokenizer_returns_empty(self, adapter):
        """无 tokenizer 时返回空列表"""
        adapter.tokenizer = None
        stop_ids = adapter._get_stop_token_ids()
        assert stop_ids == []


# ============================================================================
# 与 Executor 的兼容性
# ============================================================================

class TestExecutorCompatibility:
    """验证与 InjectionExecutor 的兼容性"""

    def test_no_injection_mode_property(self, adapter):
        """
        LlamaAdapter 不应有 injection_mode 实例属性。
        这确保 Executor._is_prompt_prefix_mode() 返回 False,
        从而走 HF KV 注入路径 (非 vLLM prompt_prefix 路径)。
        """
        # getattr 应返回 None (无此属性)
        assert getattr(adapter, 'injection_mode', None) is None

    def test_compute_kv_signature_compatible(self, adapter):
        """compute_kv 签名应与 BaseModelAdapter 兼容"""
        import inspect
        sig = inspect.signature(adapter.compute_kv)
        params = list(sig.parameters.keys())
        assert 'text' in params
        assert 'return_hidden' in params

    def test_forward_with_kv_injection_signature_compatible(self, adapter):
        """forward_with_kv_injection 签名应与 BaseModelAdapter 兼容"""
        import inspect
        sig = inspect.signature(adapter.forward_with_kv_injection)
        params = list(sig.parameters.keys())
        assert 'prompt' in params
        assert 'injected_kv' in params
        assert 'alpha' in params
        assert 'max_new_tokens' in params
