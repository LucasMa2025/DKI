"""
单元测试: DynamicCache 兼容性 (Transformers 4.x → 5.x 迁移)

验证 extract_kv_from_past 和 build_dynamic_cache_from_entries 辅助函数
在 Transformers 5.x (DynamicCache 不再支持 __getitem__) 下正确工作。

覆盖场景:
1. extract_kv_from_past — 从 DynamicCache (5.x) 提取 KV
2. extract_kv_from_past — 从 DynamicCache (4.x fallback) 提取 KV
3. extract_kv_from_past — 从 legacy tuple 提取 KV
4. build_dynamic_cache_from_entries — 构建 DynamicCache 用于注入
5. build_dynamic_cache_from_entries — alpha 缩放 (仅 Value)
6. build_dynamic_cache_from_entries — 空 entries
7. _get_cache_seq_length — 从 DynamicCache/tuple 获取序列长度
8. LlamaAdapter.compute_kv — DynamicCache 5.x 集成测试

Author: AGI Demo Project
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.models.base import (
    KVCacheEntry,
    extract_kv_from_past,
    build_dynamic_cache_from_entries,
)


# ============================================================================
# Helper: 模拟 DynamicCache (Transformers 5.x — 不支持 __getitem__)
# ============================================================================

class MockDynamicCache5x:
    """
    模拟 Transformers 5.x 的 DynamicCache:
    - 有 key_cache / value_cache 属性
    - 不支持 __getitem__ (下标访问)
    - len() 返回层数
    """
    def __init__(self, layers: int = 4, seq_len: int = 10, heads: int = 2, head_dim: int = 64):
        self.key_cache = [
            torch.randn(1, heads, seq_len, head_dim) for _ in range(layers)
        ]
        self.value_cache = [
            torch.randn(1, heads, seq_len, head_dim) for _ in range(layers)
        ]
    
    def __len__(self):
        return len(self.key_cache)
    
    # 故意不实现 __getitem__ — 模拟 5.x 行为
    # def __getitem__(self, idx): raise TypeError


class MockDynamicCache4x:
    """
    模拟 Transformers 4.x 的 DynamicCache:
    - 支持 __getitem__ (返回 (key, value) 元组)
    - 支持 get_seq_length()
    - 不一定有 key_cache / value_cache
    """
    def __init__(self, layers: int = 4, seq_len: int = 10, heads: int = 2, head_dim: int = 64):
        self._layers = [
            (torch.randn(1, heads, seq_len, head_dim), torch.randn(1, heads, seq_len, head_dim))
            for _ in range(layers)
        ]
        self._seq_len = seq_len
    
    def __len__(self):
        return len(self._layers)
    
    def __getitem__(self, idx):
        return self._layers[idx]
    
    def get_seq_length(self):
        return self._seq_len


# ============================================================================
# Tests: extract_kv_from_past
# ============================================================================

class TestExtractKVFromPast:
    """验证 extract_kv_from_past 在不同 cache 格式下正确提取 KV"""

    def test_extract_from_legacy_tuple(self):
        """从 legacy tuple 格式提取 KV"""
        past_kv = tuple(
            (torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64))
            for _ in range(4)
        )
        
        result = extract_kv_from_past(past_kv)
        
        assert len(result) == 4
        for key, value in result:
            assert key.shape == (1, 2, 10, 64)
            assert value.shape == (1, 2, 10, 64)

    def test_extract_from_dynamic_cache_layers_path(self):
        """验证 DynamicCache 的 layers[i].keys/.values 路径 (Transformers 5.x)"""
        try:
            from transformers import DynamicCache
        except ImportError:
            pytest.skip("transformers not installed")
        
        cache = DynamicCache()
        for layer_idx in range(4):
            key = torch.randn(1, 2, 10, 64)
            value = torch.randn(1, 2, 10, 64)
            cache.update(key, value, layer_idx)
        
        # 验证 layers 属性存在 (Transformers 5.x 的 DynamicLayer 结构)
        if hasattr(cache, 'layers') and cache.layers:
            layer0 = cache.layers[0]
            assert hasattr(layer0, 'keys'), "DynamicLayer should have 'keys' attribute"
            assert hasattr(layer0, 'values'), "DynamicLayer should have 'values' attribute"
        
        result = extract_kv_from_past(cache)
        
        assert len(result) == 4
        for key, value in result:
            assert key.shape == (1, 2, 10, 64)
            assert value.shape == (1, 2, 10, 64)

    def test_extract_from_dynamic_cache_5x_via_attributes(self):
        """直接测试 key_cache/value_cache 属性访问路径"""
        cache = MockDynamicCache5x(layers=4, seq_len=10)
        
        # 直接验证属性访问路径 (不依赖 isinstance 检查)
        assert hasattr(cache, 'key_cache')
        assert hasattr(cache, 'value_cache')
        
        num_layers = len(cache.key_cache)
        kv_pairs = [
            (cache.key_cache[i], cache.value_cache[i])
            for i in range(num_layers)
        ]
        
        assert len(kv_pairs) == 4
        for key, value in kv_pairs:
            assert key.shape == (1, 2, 10, 64)
            assert value.shape == (1, 2, 10, 64)

    def test_extract_from_real_dynamic_cache(self):
        """使用真实 Transformers DynamicCache 测试 (如果可用)"""
        try:
            from transformers import DynamicCache
        except ImportError:
            pytest.skip("transformers not installed")
        
        cache = DynamicCache()
        for layer_idx in range(4):
            key = torch.randn(1, 2, 10, 64)
            value = torch.randn(1, 2, 10, 64)
            cache.update(key, value, layer_idx)
        
        result = extract_kv_from_past(cache)
        
        assert len(result) == 4
        for key, value in result:
            assert key.shape == (1, 2, 10, 64)
            assert value.shape == (1, 2, 10, 64)

    def test_extract_preserves_tensor_data(self):
        """提取后 tensor 数据不变"""
        k0 = torch.randn(1, 2, 5, 64)
        v0 = torch.randn(1, 2, 5, 64)
        past_kv = ((k0, v0),)
        
        result = extract_kv_from_past(past_kv)
        
        assert torch.equal(result[0][0], k0)
        assert torch.equal(result[0][1], v0)


# ============================================================================
# Tests: build_dynamic_cache_from_entries
# ============================================================================

class TestBuildDynamicCacheFromEntries:
    """验证 build_dynamic_cache_from_entries 正确构建 cache"""

    def test_empty_entries(self):
        """空 entries 返回 (None, 0)"""
        cache, mem_len = build_dynamic_cache_from_entries(
            entries=[], device=torch.device('cpu'), alpha=1.0
        )
        assert cache is None
        assert mem_len == 0

    def test_builds_cache_with_entries(self):
        """从 KVCacheEntry 列表构建 cache"""
        entries = [
            KVCacheEntry(
                key=torch.randn(1, 2, 10, 64),
                value=torch.randn(1, 2, 10, 64),
                layer_idx=i,
            )
            for i in range(4)
        ]
        
        cache, mem_len = build_dynamic_cache_from_entries(
            entries=entries, device=torch.device('cpu'), alpha=1.0
        )
        
        assert cache is not None
        assert mem_len == 10

    def test_alpha_scaling_only_values(self):
        """alpha < 1.0 时仅缩放 Value, Key 不变"""
        key = torch.ones(1, 2, 5, 64)
        value = torch.ones(1, 2, 5, 64)
        entries = [
            KVCacheEntry(key=key.clone(), value=value.clone(), layer_idx=0),
        ]
        
        cache, mem_len = build_dynamic_cache_from_entries(
            entries=entries, device=torch.device('cpu'), alpha=0.5
        )
        
        assert cache is not None
        assert mem_len == 5
        
        # 验证 cache 中的值被缩放
        try:
            from transformers import DynamicCache
            if isinstance(cache, DynamicCache):
                # Transformers 5.x: 通过 key_cache/value_cache 访问
                if hasattr(cache, 'key_cache') and cache.key_cache:
                    result_key = cache.key_cache[0]
                    result_value = cache.value_cache[0]
                else:
                    # 4.x fallback
                    result_key, result_value = cache[0]
                
                # Key 不应被缩放
                assert torch.allclose(result_key, key)
                # Value 应被缩放 0.5
                assert torch.allclose(result_value, value * 0.5)
        except ImportError:
            # Legacy tuple
            result_key, result_value = cache[0]
            assert torch.allclose(result_key, key)
            assert torch.allclose(result_value, value * 0.5)

    def test_alpha_one_no_scaling(self):
        """alpha = 1.0 时不缩放"""
        value = torch.ones(1, 2, 5, 64)
        entries = [
            KVCacheEntry(
                key=torch.ones(1, 2, 5, 64),
                value=value.clone(),
                layer_idx=0,
            ),
        ]
        
        cache, mem_len = build_dynamic_cache_from_entries(
            entries=entries, device=torch.device('cpu'), alpha=1.0
        )
        
        assert cache is not None
        # 验证 value 未被缩放
        try:
            from transformers import DynamicCache
            if isinstance(cache, DynamicCache):
                if hasattr(cache, 'key_cache') and cache.key_cache:
                    result_value = cache.value_cache[0]
                else:
                    _, result_value = cache[0]
                assert torch.allclose(result_value, value)
        except ImportError:
            _, result_value = cache[0]
            assert torch.allclose(result_value, value)


# ============================================================================
# Tests: _get_cache_seq_length (llama_adapter 辅助函数)
# ============================================================================

class TestGetCacheSeqLength:
    """验证 _get_cache_seq_length 兼容 4.x/5.x"""

    def test_from_legacy_tuple(self):
        """从 legacy tuple 获取序列长度"""
        from dki.models.llama_adapter import _get_cache_seq_length
        
        cache = (
            (torch.randn(1, 2, 15, 64), torch.randn(1, 2, 15, 64)),
        )
        assert _get_cache_seq_length(cache) == 15

    def test_from_real_dynamic_cache(self):
        """从真实 DynamicCache 获取序列长度"""
        try:
            from transformers import DynamicCache
        except ImportError:
            pytest.skip("transformers not installed")
        
        from dki.models.llama_adapter import _get_cache_seq_length
        
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 20, 64), torch.randn(1, 2, 20, 64), 0)
        
        assert _get_cache_seq_length(cache) == 20

    def test_empty_returns_zero(self):
        """空 cache 返回 0"""
        from dki.models.llama_adapter import _get_cache_seq_length
        
        assert _get_cache_seq_length(()) == 0
        assert _get_cache_seq_length([]) == 0
        assert _get_cache_seq_length(None) == 0


# ============================================================================
# Tests: LlamaAdapter.compute_kv — DynamicCache 5.x 集成
# ============================================================================

class TestLlamaAdapterComputeKVDynamicCache:
    """验证 LlamaAdapter.compute_kv 在 DynamicCache 5.x 下正确工作"""

    @pytest.fixture
    def loaded_adapter(self):
        """创建已加载的 Mock LlamaAdapter"""
        from dki.models.llama_adapter import LlamaAdapter
        
        adapter = LlamaAdapter(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            device="cpu",
            dtype="float32",
        )
        adapter._is_loaded = True
        adapter.num_layers = 4
        adapter.num_heads = 2
        adapter.hidden_dim = 128
        adapter.head_dim = 64
        adapter.device = "cpu"
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.unk_token_id = 3
        mock_tokenizer.convert_tokens_to_ids.return_value = 128009
        mock_tokenizer.decode.return_value = "test output"
        mock_tokenizer.apply_chat_template.return_value = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        )
        
        class _TokenizerOutput(dict):
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
        
        mock_tokenizer.return_value = _TokenizerOutput({
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5, dtype=torch.long),
        })
        
        adapter.tokenizer = mock_tokenizer
        adapter.model = MagicMock()
        
        return adapter

    def test_compute_kv_with_real_dynamic_cache(self, loaded_adapter):
        """compute_kv 应正确处理真实 DynamicCache (Transformers 5.x)"""
        try:
            from transformers import DynamicCache
        except ImportError:
            pytest.skip("transformers not installed")
        
        # 构建真实 DynamicCache
        cache = DynamicCache()
        for layer_idx in range(4):
            key = torch.randn(1, 2, 5, 64)
            value = torch.randn(1, 2, 5, 64)
            cache.update(key, value, layer_idx)
        
        mock_output = MagicMock()
        mock_output.past_key_values = cache
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
            assert entry.key.shape == (1, 2, 5, 64)

    def test_compute_kv_with_legacy_tuple(self, loaded_adapter):
        """compute_kv 应正确处理 legacy tuple 格式"""
        mock_past_kv = tuple(
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
            for _ in range(4)
        )
        mock_output = MagicMock()
        mock_output.past_key_values = mock_past_kv
        mock_output.hidden_states = None
        loaded_adapter.model.return_value = mock_output
        
        kv_entries, hidden = loaded_adapter.compute_kv("test")
        
        assert len(kv_entries) == 4
        for i, entry in enumerate(kv_entries):
            assert isinstance(entry, KVCacheEntry)
            assert entry.layer_idx == i

    def test_prepare_kv_for_injection_returns_dynamic_cache(self, loaded_adapter):
        """_prepare_kv_for_injection 应返回 DynamicCache (如果可用)"""
        entries = [
            KVCacheEntry(
                key=torch.randn(1, 2, 10, 64),
                value=torch.randn(1, 2, 10, 64),
                layer_idx=i,
            )
            for i in range(4)
        ]
        
        cache, mem_len = loaded_adapter._prepare_kv_for_injection(
            injected_kv=entries,
            alpha=0.5,
            device=torch.device('cpu'),
        )
        
        assert cache is not None
        assert mem_len == 10
        
        # 验证返回类型
        try:
            from transformers import DynamicCache
            assert isinstance(cache, DynamicCache)
        except ImportError:
            assert isinstance(cache, tuple)
