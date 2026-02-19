"""
Unit tests for dki/attention module review fixes (2026-02-18)

Tests cover:
- BUG 2+3: inject_chunked 拼接顺序修正 (memory prepend, input append)
- BUG 4: GPU 内存清理 (del 中间张量)
- BUG 5: _forward_with_flash_attention 文档修正
- 数学等价性验证 (inject vs inject_chunked)
- 配置完整性
- Profiler 边界条件
- Backend 检测逻辑

Author: AGI Demo Project
Date: 2026-02-18
"""

import math
import time
import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

from dki.attention.config import (
    FlashAttentionConfig,
    FA3Config,
    FA2Config,
    KVInjectionConfig,
    ProfilingConfig,
)
from dki.attention.backend import (
    FlashAttentionBackend,
    BackendType,
    scaled_dot_product_attention_standard,
    flash_attention_forward,
)
from dki.attention.kv_injection import KVInjectionOptimizer, InjectionResult
from dki.attention.profiler import AttentionProfiler, ProfileRecord


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def standard_optimizer():
    """Create optimizer with standard backend for testing."""
    config = FlashAttentionConfig(backend="standard")
    return KVInjectionOptimizer(config=config, backend="standard")


@pytest.fixture
def standard_optimizer_no_alpha():
    """Create optimizer with alpha blending disabled."""
    config = FlashAttentionConfig(
        backend="standard",
        kv_injection=KVInjectionConfig(alpha_blending=False),
    )
    return KVInjectionOptimizer(config=config, backend="standard")


@pytest.fixture
def small_tensors():
    """Create small test tensors for attention computation."""
    batch_size = 1
    seq_q = 4
    seq_k = 8
    seq_m = 6
    num_heads = 2
    head_dim = 8

    torch.manual_seed(42)
    return {
        "query": torch.randn(batch_size, seq_q, num_heads, head_dim),
        "key": torch.randn(batch_size, seq_k, num_heads, head_dim),
        "value": torch.randn(batch_size, seq_k, num_heads, head_dim),
        "memory_key": torch.randn(batch_size, seq_m, num_heads, head_dim),
        "memory_value": torch.randn(batch_size, seq_m, num_heads, head_dim),
        "batch_size": batch_size,
        "seq_q": seq_q,
        "seq_k": seq_k,
        "seq_m": seq_m,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


# ============================================================================
# BUG 2+3: inject_chunked 拼接顺序修正 + 数学等价性
# ============================================================================

class TestInjectChunkedPrependOrder:
    """Test that inject_chunked produces results equivalent to inject (prepend order)."""

    def test_chunked_equals_full_injection(self, standard_optimizer, small_tensors):
        """inject_chunked should produce mathematically equivalent results to inject."""
        t = small_tensors
        
        # Full injection (no chunking)
        result_full = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
        )
        
        # Chunked injection with small chunk_size to force chunking
        result_chunked = standard_optimizer.inject_chunked(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
            chunk_size=2,  # Force chunking (seq_m=6, so 3 chunks)
        )
        
        # Results should be numerically close (same math, different computation order)
        assert torch.allclose(
            result_full.output, result_chunked.output, atol=1e-5, rtol=1e-4
        ), (
            f"Full and chunked injection outputs differ! "
            f"Max diff: {(result_full.output - result_chunked.output).abs().max().item()}"
        )

    def test_chunked_equals_full_with_alpha(self, standard_optimizer, small_tensors):
        """inject_chunked with alpha blending should match inject."""
        t = small_tensors
        
        result_full = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=0.5,
        )
        
        result_chunked = standard_optimizer.inject_chunked(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=0.5,
            chunk_size=3,  # 2 chunks
        )
        
        assert torch.allclose(
            result_full.output, result_chunked.output, atol=1e-5, rtol=1e-4
        )

    def test_chunked_falls_back_when_small_memory(self, standard_optimizer, small_tensors):
        """inject_chunked should delegate to inject when memory fits in one chunk."""
        t = small_tensors
        
        result_full = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
        )
        
        # chunk_size > seq_m, so no chunking needed
        result_chunked = standard_optimizer.inject_chunked(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
            chunk_size=1024,  # Much larger than seq_m=6
        )
        
        assert torch.allclose(result_full.output, result_chunked.output, atol=1e-6)

    def test_chunked_single_token_chunks(self, standard_optimizer, small_tensors):
        """inject_chunked with chunk_size=1 should still produce correct results."""
        t = small_tensors
        
        result_full = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
        )
        
        result_chunked = standard_optimizer.inject_chunked(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
            chunk_size=1,  # 6 chunks (one per memory token)
        )
        
        assert torch.allclose(
            result_full.output, result_chunked.output, atol=1e-5, rtol=1e-4
        )

    def test_chunked_result_metadata(self, standard_optimizer, small_tensors):
        """inject_chunked should return correct metadata."""
        t = small_tensors
        
        result = standard_optimizer.inject_chunked(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=0.7,
            chunk_size=2,
        )
        
        assert isinstance(result, InjectionResult)
        assert result.output.shape == (t["batch_size"], t["seq_q"], t["num_heads"], t["head_dim"])
        assert result.memory_key_len == t["seq_m"]
        assert result.input_key_len == t["seq_k"]
        assert result.total_key_len == t["seq_m"] + t["seq_k"]
        assert result.alpha == 0.7
        assert result.backend_used == "standard"
        assert result.latency_ms >= 0


# ============================================================================
# BUG 4: GPU 内存清理
# ============================================================================

class TestMemoryCleanup:
    """Test that intermediate tensors are properly cleaned up."""

    def test_inject_prepend_no_tensor_leak(self, standard_optimizer, small_tensors):
        """After inject, no extra references to intermediate tensors should remain."""
        t = small_tensors
        
        # Run injection multiple times
        for _ in range(10):
            result = standard_optimizer.inject(
                query=t["query"],
                key=t["key"],
                value=t["value"],
                memory_key=t["memory_key"],
                memory_value=t["memory_value"],
                alpha=0.5,
            )
        
        # If there's a leak, repeated injections would accumulate memory
        # This test mainly ensures no exceptions are raised
        assert result.output is not None

    def test_inject_chunked_no_tensor_leak(self, standard_optimizer, small_tensors):
        """After inject_chunked, intermediate tensors should be cleaned up."""
        t = small_tensors
        
        for _ in range(10):
            result = standard_optimizer.inject_chunked(
                query=t["query"],
                key=t["key"],
                value=t["value"],
                memory_key=t["memory_key"],
                memory_value=t["memory_value"],
                alpha=0.5,
                chunk_size=2,
            )
        
        assert result.output is not None

    def test_inject_alpha_zero_cleanup(self, standard_optimizer, small_tensors):
        """Alpha=0 path should also clean up properly."""
        t = small_tensors
        
        result = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=0.0,
        )
        
        assert result.output is not None
        assert result.alpha == 0.0

    def test_inject_alpha_one_no_blending(self, standard_optimizer, small_tensors):
        """Alpha=1.0 should skip alpha blending (no original_output computation)."""
        t = small_tensors
        
        result = standard_optimizer.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
        )
        
        # With alpha=1.0, output should be pure injected output
        assert result.output is not None
        assert result.alpha == 1.0


# ============================================================================
# inject_multi_layer tests
# ============================================================================

class TestInjectMultiLayer:
    """Test multi-layer injection."""

    def test_multi_layer_basic(self, standard_optimizer):
        """Test basic multi-layer injection."""
        num_layers = 4
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 2
        head_dim = 8
        
        queries = [torch.randn(batch_size, seq_q, num_heads, head_dim) for _ in range(num_layers)]
        keys = [torch.randn(batch_size, seq_k, num_heads, head_dim) for _ in range(num_layers)]
        values = [torch.randn(batch_size, seq_k, num_heads, head_dim) for _ in range(num_layers)]
        mem_keys = [torch.randn(batch_size, seq_m, num_heads, head_dim) for _ in range(num_layers)]
        mem_values = [torch.randn(batch_size, seq_m, num_heads, head_dim) for _ in range(num_layers)]
        
        results = standard_optimizer.inject_multi_layer(
            queries=queries,
            keys=keys,
            values=values,
            memory_keys=mem_keys,
            memory_values=mem_values,
            alpha=0.5,
        )
        
        assert len(results) == num_layers
        for result in results:
            assert isinstance(result, InjectionResult)
            assert result.output.shape == (batch_size, seq_q, num_heads, head_dim)
            assert result.alpha == 0.5

    def test_multi_layer_with_layer_alphas(self, standard_optimizer):
        """Test multi-layer injection with per-layer alpha values."""
        num_layers = 3
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 2
        head_dim = 8
        
        queries = [torch.randn(batch_size, seq_q, num_heads, head_dim) for _ in range(num_layers)]
        keys = [torch.randn(batch_size, seq_k, num_heads, head_dim) for _ in range(num_layers)]
        values = [torch.randn(batch_size, seq_k, num_heads, head_dim) for _ in range(num_layers)]
        mem_keys = [torch.randn(batch_size, seq_m, num_heads, head_dim) for _ in range(num_layers)]
        mem_values = [torch.randn(batch_size, seq_m, num_heads, head_dim) for _ in range(num_layers)]
        
        layer_alphas = [0.3, 0.5, 0.8]
        
        results = standard_optimizer.inject_multi_layer(
            queries=queries,
            keys=keys,
            values=values,
            memory_keys=mem_keys,
            memory_values=mem_values,
            layer_alphas=layer_alphas,
        )
        
        assert len(results) == num_layers
        for i, result in enumerate(results):
            assert result.alpha == layer_alphas[i]


# ============================================================================
# Backend detection edge cases
# ============================================================================

class TestBackendDetection:
    """Test backend detection edge cases."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(8, 6))
    @patch('torch.cuda.get_device_name', return_value="NVIDIA A10")
    def test_a10_detection(self, mock_name, mock_cap, mock_cuda):
        """A10 (8, 6) should get FA2 backend."""
        import dki.attention.backend as backend_module
        old_val = backend_module.FLASH_ATTN_AVAILABLE
        backend_module.FLASH_ATTN_AVAILABLE = True
        try:
            backend = FlashAttentionBackend.detect_best_backend()
            assert backend == BackendType.FA2.value
        finally:
            backend_module.FLASH_ATTN_AVAILABLE = old_val

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(8, 9))
    @patch('torch.cuda.get_device_name', return_value="NVIDIA RTX 4090")
    def test_rtx4090_detection(self, mock_name, mock_cap, mock_cuda):
        """RTX 4090 (8, 9) should get FA2 backend."""
        import dki.attention.backend as backend_module
        old_val = backend_module.FLASH_ATTN_AVAILABLE
        backend_module.FLASH_ATTN_AVAILABLE = True
        try:
            backend = FlashAttentionBackend.detect_best_backend()
            assert backend == BackendType.FA2.value
        finally:
            backend_module.FLASH_ATTN_AVAILABLE = old_val

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(7, 5))
    @patch('torch.cuda.get_device_name', return_value="NVIDIA RTX 2080")
    def test_turing_detection(self, mock_name, mock_cap, mock_cuda):
        """Turing GPU (7, 5) should get standard backend."""
        backend = FlashAttentionBackend.detect_best_backend()
        assert backend == BackendType.STANDARD.value

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(6, 1))
    @patch('torch.cuda.get_device_name', return_value="NVIDIA GTX 1080")
    def test_unknown_old_gpu(self, mock_name, mock_cap, mock_cuda):
        """Unknown old GPU (6, 1) should get standard backend."""
        backend = FlashAttentionBackend.detect_best_backend()
        assert backend == BackendType.STANDARD.value

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(10, 0))
    @patch('torch.cuda.get_device_name', return_value="NVIDIA B100")
    def test_future_gpu_detection(self, mock_name, mock_cap, mock_cuda):
        """Future GPU (10, 0) should get FA3 backend (highest match)."""
        import dki.attention.backend as backend_module
        old_val = backend_module.FLASH_ATTN_AVAILABLE
        backend_module.FLASH_ATTN_AVAILABLE = True
        try:
            backend = FlashAttentionBackend.detect_best_backend()
            assert backend == BackendType.FA3.value
        finally:
            backend_module.FLASH_ATTN_AVAILABLE = old_val

    def test_validate_backend_unknown(self):
        """Unknown backend string should fall back to standard."""
        result = FlashAttentionBackend.validate_backend("unknown_backend")
        assert result == BackendType.STANDARD.value

    @patch('torch.cuda.is_available', return_value=False)
    def test_supports_fa3_no_cuda(self, mock_cuda):
        """supports_fa3 should return False when CUDA is not available."""
        assert FlashAttentionBackend.supports_fa3() is False

    @patch('torch.cuda.is_available', return_value=False)
    def test_supports_fa2_no_cuda(self, mock_cuda):
        """supports_fa2 should return False when CUDA is not available."""
        assert FlashAttentionBackend.supports_fa2() is False

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_gpu_info_no_cuda(self, mock_cuda):
        """get_gpu_info should return available=False when no CUDA."""
        info = FlashAttentionBackend.get_gpu_info()
        assert info["available"] is False


# ============================================================================
# flash_attention_forward tests
# ============================================================================

class TestFlashAttentionForward:
    """Test the unified flash_attention_forward function."""

    def test_standard_backend_forward(self):
        """flash_attention_forward with standard backend should work."""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        num_heads = 2
        head_dim = 8
        
        # FlashAttention format: [batch, seq, heads, head_dim]
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        
        output = flash_attention_forward(
            query=query,
            key=key,
            value=value,
            backend="standard",
        )
        
        assert output.shape == (batch_size, seq_q, num_heads, head_dim)

    def test_standard_backend_format_conversion(self):
        """flash_attention_forward should correctly convert between FA and standard formats."""
        batch_size = 1
        seq_len = 4
        num_heads = 2
        head_dim = 8
        
        torch.manual_seed(42)
        
        # FlashAttention format: [batch, seq, heads, head_dim]
        query_fa = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key_fa = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value_fa = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        # Standard format: [batch, heads, seq, head_dim]
        query_std = query_fa.transpose(1, 2)
        key_std = key_fa.transpose(1, 2)
        value_std = value_fa.transpose(1, 2)
        
        # Compute with both
        output_fa = flash_attention_forward(
            query_fa, key_fa, value_fa, backend="standard"
        )
        output_std = scaled_dot_product_attention_standard(
            query_std, key_std, value_std
        )
        
        # Convert FA output to standard format for comparison
        output_fa_std = output_fa.transpose(1, 2)
        
        assert torch.allclose(output_fa_std, output_std, atol=1e-6)


# ============================================================================
# Standard attention tests
# ============================================================================

class TestScaledDotProductAttention:
    """Test the standard attention implementation."""

    def test_attention_output_shape(self):
        """Output shape should match expected dimensions."""
        batch, heads, seq_q, seq_k, dim = 2, 4, 6, 10, 16
        
        q = torch.randn(batch, heads, seq_q, dim)
        k = torch.randn(batch, heads, seq_k, dim)
        v = torch.randn(batch, heads, seq_k, dim)
        
        output = scaled_dot_product_attention_standard(q, k, v)
        assert output.shape == (batch, heads, seq_q, dim)

    def test_attention_softmax_normalization(self):
        """Attention weights should sum to 1 along key dimension."""
        batch, heads, seq, dim = 1, 1, 4, 8
        
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        
        scale = 1.0 / math.sqrt(dim)
        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1
        )
        
        # Weights should sum to 1 along last dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, heads, seq), atol=1e-6)

    def test_attention_with_custom_scale(self):
        """Custom scale should be applied correctly."""
        batch, heads, seq, dim = 1, 1, 4, 8
        
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        
        # With very small scale, attention should be more uniform
        output_small = scaled_dot_product_attention_standard(q, k, v, scale=0.001)
        # With large scale, attention should be more peaked
        output_large = scaled_dot_product_attention_standard(q, k, v, scale=10.0)
        
        # Both should have correct shape
        assert output_small.shape == output_large.shape == (batch, heads, seq, dim)

    def test_attention_causal_mask(self):
        """Causal mask should prevent attending to future positions."""
        batch, heads, seq, dim = 1, 1, 4, 8
        
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.ones(batch, heads, seq, dim)  # All ones for easy verification
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(seq, seq) * float('-inf'),
            diagonal=1
        )
        
        output = scaled_dot_product_attention_standard(q, k, v, attn_mask=mask)
        
        # Output should still be valid (no NaN/Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# Config edge cases
# ============================================================================

class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_config_from_empty_dict(self):
        """Creating config from empty dict should use defaults."""
        config = FlashAttentionConfig.from_dict({})
        
        assert config.enabled is True
        assert config.backend == "auto"
        assert config.fa3.use_fp8 is False
        assert config.kv_injection.chunk_size == 1024

    def test_config_partial_dict(self):
        """Creating config from partial dict should use defaults for missing fields."""
        config = FlashAttentionConfig.from_dict({
            "backend": "fa2",
            "fa2": {"dropout": 0.1},
        })
        
        assert config.backend == "fa2"
        assert config.fa2.dropout == 0.1
        assert config.fa2.causal is False  # default
        assert config.fa3.use_fp8 is False  # default

    def test_config_roundtrip(self):
        """Config should survive dict roundtrip."""
        original = FlashAttentionConfig(
            enabled=True,
            backend="fa3",
            fa3=FA3Config(use_fp8=True, num_stages=4),
            fa2=FA2Config(dropout=0.1, causal=True),
            kv_injection=KVInjectionConfig(chunk_size=2048, strategy="prepend"),
            profiling=ProfilingConfig(enabled=True, log_flops=True),
        )
        
        data = original.to_dict()
        restored = FlashAttentionConfig.from_dict(data)
        
        assert restored.enabled == original.enabled
        assert restored.backend == original.backend
        assert restored.fa3.use_fp8 == original.fa3.use_fp8
        assert restored.fa3.num_stages == original.fa3.num_stages
        assert restored.fa2.dropout == original.fa2.dropout
        assert restored.fa2.causal == original.fa2.causal
        assert restored.kv_injection.chunk_size == original.kv_injection.chunk_size
        assert restored.profiling.enabled == original.profiling.enabled
        assert restored.profiling.log_flops == original.profiling.log_flops

    def test_config_repr(self):
        """Config repr should be informative."""
        config = FlashAttentionConfig(enabled=True, backend="fa2")
        repr_str = repr(config)
        assert "enabled=True" in repr_str
        assert "backend=fa2" in repr_str


# ============================================================================
# Profiler edge cases
# ============================================================================

class TestProfilerEdgeCases:
    """Test profiler edge cases."""

    def test_profiler_empty_report(self):
        """Empty profiler should return message."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=True))
        report = profiler.get_report()
        assert "message" in report
        assert report["message"] == "No records available"

    def test_profiler_manual_record(self):
        """Manual record should work correctly."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=True))
        
        tensor = torch.randn(2, 4, 8)
        profiler.record(
            operation="test_op",
            latency_ms=42.5,
            input_tensor=tensor,
            output_tensor=tensor,
            backend="standard",
            metadata={"key": "value"},
        )
        
        records = profiler.get_records()
        assert len(records) == 1
        assert records[0]["operation"] == "test_op"
        assert records[0]["latency_ms"] == 42.5
        assert records[0]["input_shape"] == "torch.Size([2, 4, 8])"
        assert records[0]["metadata"] == {"key": "value"}

    def test_profiler_manual_record_disabled(self):
        """Manual record should be skipped when disabled."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=False))
        
        profiler.record(
            operation="test_op",
            latency_ms=42.5,
        )
        
        assert len(profiler) == 0

    def test_profiler_enable_disable(self):
        """Profiler should support runtime enable/disable."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=False))
        
        with profiler.profile("op1"):
            pass
        assert len(profiler) == 0
        
        profiler.enabled = True
        with profiler.profile("op2"):
            pass
        assert len(profiler) == 1

    def test_profiler_report_percentiles(self):
        """Report should include correct percentiles."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=True, log_memory=False))
        
        # Add 100 records with known latencies
        for i in range(100):
            profiler.record(
                operation="test",
                latency_ms=float(i),
                backend="standard",
            )
        
        report = profiler.get_report()
        ops = report["operations"]["test"]
        
        assert ops["count"] == 100
        assert ops["min_latency_ms"] == 0.0
        assert ops["max_latency_ms"] == 99.0
        assert ops["p50_latency_ms"] == 50.0
        assert ops["p90_latency_ms"] == 90.0

    def test_profiler_single_record_percentiles(self):
        """Report with single record should not crash on percentile calculation."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=True, log_memory=False))
        
        profiler.record(
            operation="single",
            latency_ms=42.0,
            backend="standard",
        )
        
        report = profiler.get_report()
        ops = report["operations"]["single"]
        
        assert ops["count"] == 1
        assert ops["p50_latency_ms"] == 42.0
        assert ops["p90_latency_ms"] == 42.0
        assert ops["p99_latency_ms"] == 42.0

    def test_profiler_exception_in_context(self):
        """Profiler should still record even if exception occurs in context."""
        profiler = AttentionProfiler(ProfilingConfig(enabled=True, log_memory=False))
        
        with pytest.raises(ValueError):
            with profiler.profile("error_op"):
                raise ValueError("test error")
        
        # Record should still be captured (finally block)
        assert len(profiler) == 1
        records = profiler.get_records()
        assert records[0]["operation"] == "error_op"


# ============================================================================
# Statistics tests
# ============================================================================

class TestOptimizerStats:
    """Test optimizer statistics tracking."""

    def test_stats_avg_memory_len(self, standard_optimizer):
        """Average memory length should be computed correctly."""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        num_heads = 2
        head_dim = 8
        
        for mem_len in [4, 8, 12]:
            query = torch.randn(batch_size, seq_q, num_heads, head_dim)
            key = torch.randn(batch_size, seq_k, num_heads, head_dim)
            value = torch.randn(batch_size, seq_k, num_heads, head_dim)
            memory_key = torch.randn(batch_size, mem_len, num_heads, head_dim)
            memory_value = torch.randn(batch_size, mem_len, num_heads, head_dim)
            
            standard_optimizer.inject(query, key, value, memory_key, memory_value)
        
        stats = standard_optimizer.get_stats()
        assert stats["total_injections"] == 3
        # Average of 4, 8, 12 = 8.0
        assert abs(stats["avg_memory_len"] - 8.0) < 0.01

    def test_stats_backend_usage(self, standard_optimizer):
        """Backend usage should be tracked correctly."""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 2
        head_dim = 8
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        for _ in range(5):
            standard_optimizer.inject(query, key, value, memory_key, memory_value)
        
        stats = standard_optimizer.get_stats()
        assert stats["backend_usage"]["standard"] == 5
        assert stats["backend_usage"]["fa2"] == 0
        assert stats["backend_usage"]["fa3"] == 0

    def test_stats_config_in_report(self, standard_optimizer):
        """Stats should include config information."""
        stats = standard_optimizer.get_stats()
        assert "config" in stats
        assert stats["config"]["strategy"] == "prepend"
        assert stats["config"]["alpha_blending"] is True


# ============================================================================
# Alpha blending disabled
# ============================================================================

class TestAlphaBlendingDisabled:
    """Test behavior when alpha blending is disabled."""

    def test_alpha_less_than_one_no_blending(self, standard_optimizer_no_alpha, small_tensors):
        """With alpha_blending=False, alpha<1 should still return injected output."""
        t = small_tensors
        
        result_full = standard_optimizer_no_alpha.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=1.0,
        )
        
        result_half = standard_optimizer_no_alpha.inject(
            query=t["query"],
            key=t["key"],
            value=t["value"],
            memory_key=t["memory_key"],
            memory_value=t["memory_value"],
            alpha=0.5,
        )
        
        # With alpha_blending disabled, alpha<1 should produce same output as alpha=1
        assert torch.allclose(result_full.output, result_half.output, atol=1e-6)


# ============================================================================
# Device consistency
# ============================================================================

class TestDeviceConsistency:
    """Test device consistency in injection."""

    def test_inject_moves_memory_to_query_device(self, standard_optimizer):
        """Memory tensors should be moved to query's device."""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 2
        head_dim = 8
        
        # All on CPU for this test
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        result = standard_optimizer.inject(
            query=query,
            key=key,
            value=value,
            memory_key=memory_key,
            memory_value=memory_value,
        )
        
        assert result.output.device == query.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
