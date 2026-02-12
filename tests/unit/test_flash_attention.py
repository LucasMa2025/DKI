"""
Unit tests for FlashAttention integration

Tests:
- Backend detection
- Configuration loading
- K/V injection optimizer
- Profiler functionality
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

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
)
from dki.attention.kv_injection import KVInjectionOptimizer, InjectionResult
from dki.attention.profiler import AttentionProfiler


class TestFlashAttentionConfig:
    """Test FlashAttention configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FlashAttentionConfig()
        
        assert config.enabled is True
        assert config.backend == "auto"
        assert config.fa3.use_fp8 is False
        assert config.fa3.enable_async is True
        assert config.fa2.causal is False
        assert config.fa2.dropout == 0.0
        assert config.kv_injection.enabled is True
        assert config.kv_injection.strategy == "prepend"
        assert config.profiling.enabled is False
    
    def test_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            "enabled": True,
            "backend": "fa3",
            "fa3": {
                "use_fp8": True,
                "enable_async": False,
            },
            "kv_injection": {
                "chunk_size": 2048,
            },
        }
        
        config = FlashAttentionConfig.from_dict(data)
        
        assert config.backend == "fa3"
        assert config.fa3.use_fp8 is True
        assert config.fa3.enable_async is False
        assert config.kv_injection.chunk_size == 2048
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = FlashAttentionConfig(
            enabled=True,
            backend="fa2",
        )
        
        data = config.to_dict()
        
        assert data["enabled"] is True
        assert data["backend"] == "fa2"
        assert "fa3" in data
        assert "fa2" in data
        assert "kv_injection" in data


class TestFlashAttentionBackend:
    """Test backend detection and selection"""
    
    @patch('torch.cuda.is_available')
    def test_no_cuda(self, mock_cuda):
        """Test fallback when CUDA is not available"""
        mock_cuda.return_value = False
        
        backend = FlashAttentionBackend.detect_best_backend()
        
        assert backend == BackendType.STANDARD.value
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_name')
    def test_h100_detection(self, mock_name, mock_cap, mock_cuda):
        """Test H100 GPU detection (FA3)"""
        mock_cuda.return_value = True
        mock_cap.return_value = (9, 0)
        mock_name.return_value = "NVIDIA H100"
        
        # Mock flash_attn import
        with patch.dict('sys.modules', {'flash_attn': MagicMock()}):
            import dki.attention.backend as backend_module
            backend_module.FLASH_ATTN_AVAILABLE = True
            
            backend = FlashAttentionBackend.detect_best_backend()
            
            assert backend == BackendType.FA3.value
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_name')
    def test_a100_detection(self, mock_name, mock_cap, mock_cuda):
        """Test A100 GPU detection (FA2)"""
        mock_cuda.return_value = True
        mock_cap.return_value = (8, 0)
        mock_name.return_value = "NVIDIA A100"
        
        with patch.dict('sys.modules', {'flash_attn': MagicMock()}):
            import dki.attention.backend as backend_module
            backend_module.FLASH_ATTN_AVAILABLE = True
            
            backend = FlashAttentionBackend.detect_best_backend()
            
            assert backend == BackendType.FA2.value
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_name')
    def test_v100_detection(self, mock_name, mock_cap, mock_cuda):
        """Test V100 GPU detection (standard)"""
        mock_cuda.return_value = True
        mock_cap.return_value = (7, 0)
        mock_name.return_value = "NVIDIA V100"
        
        backend = FlashAttentionBackend.detect_best_backend()
        
        assert backend == BackendType.STANDARD.value
    
    def test_validate_backend_auto(self):
        """Test backend validation with auto"""
        with patch.object(FlashAttentionBackend, 'detect_best_backend') as mock:
            mock.return_value = "fa2"
            
            result = FlashAttentionBackend.validate_backend("auto")
            
            assert result == "fa2"
    
    def test_validate_backend_standard(self):
        """Test backend validation with standard"""
        result = FlashAttentionBackend.validate_backend("standard")
        
        assert result == BackendType.STANDARD.value


class TestStandardAttention:
    """Test standard attention implementation"""
    
    def test_basic_attention(self):
        """Test basic scaled dot-product attention"""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = scaled_dot_product_attention_standard(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_attention_with_mask(self):
        """Test attention with mask"""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        
        output = scaled_dot_product_attention_standard(
            query, key, value, attn_mask=mask
        )
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestKVInjectionOptimizer:
    """Test K/V injection optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer with standard backend"""
        config = FlashAttentionConfig(backend="standard")
        return KVInjectionOptimizer(config=config, backend="standard")
    
    def test_basic_injection(self, optimizer):
        """Test basic K/V injection"""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4  # memory
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        result = optimizer.inject(
            query=query,
            key=key,
            value=value,
            memory_key=memory_key,
            memory_value=memory_value,
            alpha=1.0,
        )
        
        assert isinstance(result, InjectionResult)
        assert result.output.shape == (batch_size, seq_q, num_heads, head_dim)
        assert result.memory_key_len == seq_m
        assert result.input_key_len == seq_k
        assert result.total_key_len == seq_m + seq_k
        assert result.alpha == 1.0
    
    def test_alpha_blending(self, optimizer):
        """Test alpha blending"""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        # Full injection
        result_full = optimizer.inject(
            query, key, value, memory_key, memory_value, alpha=1.0
        )
        
        # Partial injection
        result_partial = optimizer.inject(
            query, key, value, memory_key, memory_value, alpha=0.5
        )
        
        # No injection
        result_none = optimizer.inject(
            query, key, value, memory_key, memory_value, alpha=0.0
        )
        
        # Results should be different
        assert not torch.allclose(result_full.output, result_partial.output)
        assert not torch.allclose(result_partial.output, result_none.output)
    
    def test_stats(self, optimizer):
        """Test statistics tracking"""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        # Perform multiple injections
        for _ in range(5):
            optimizer.inject(query, key, value, memory_key, memory_value)
        
        stats = optimizer.get_stats()
        
        assert stats["total_injections"] == 5
        assert stats["avg_memory_len"] == seq_m
        assert stats["backend"] == "standard"
    
    def test_reset_stats(self, optimizer):
        """Test statistics reset"""
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        optimizer.inject(query, key, value, memory_key, memory_value)
        optimizer.reset_stats()
        
        stats = optimizer.get_stats()
        assert stats["total_injections"] == 0


class TestAttentionProfiler:
    """Test attention profiler"""
    
    def test_disabled_profiler(self):
        """Test profiler when disabled"""
        config = ProfilingConfig(enabled=False)
        profiler = AttentionProfiler(config)
        
        with profiler.profile("test_op"):
            pass
        
        assert len(profiler) == 0
    
    def test_enabled_profiler(self):
        """Test profiler when enabled"""
        config = ProfilingConfig(enabled=True)
        profiler = AttentionProfiler(config)
        
        with profiler.profile("test_op", backend="standard"):
            import time
            time.sleep(0.01)  # 10ms
        
        assert len(profiler) == 1
        
        records = profiler.get_records()
        assert records[0]["operation"] == "test_op"
        assert records[0]["backend"] == "standard"
        assert records[0]["latency_ms"] >= 10
    
    def test_report_generation(self):
        """Test report generation"""
        config = ProfilingConfig(enabled=True)
        profiler = AttentionProfiler(config)
        
        # Add multiple records
        for i in range(10):
            with profiler.profile(f"op_{i % 2}", backend="standard"):
                pass
        
        report = profiler.get_report()
        
        assert report["total_records"] == 10
        assert "operations" in report
        assert "op_0" in report["operations"]
        assert "op_1" in report["operations"]
        assert report["operations"]["op_0"]["count"] == 5
        assert report["operations"]["op_1"]["count"] == 5
    
    def test_clear(self):
        """Test clearing records"""
        config = ProfilingConfig(enabled=True)
        profiler = AttentionProfiler(config)
        
        with profiler.profile("test"):
            pass
        
        assert len(profiler) == 1
        
        profiler.clear()
        
        assert len(profiler) == 0


class TestIntegration:
    """Integration tests"""
    
    def test_config_to_optimizer(self):
        """Test creating optimizer from config"""
        config = FlashAttentionConfig(
            enabled=True,
            backend="standard",
            kv_injection=KVInjectionConfig(
                chunk_size=512,
                alpha_blending=True,
            ),
        )
        
        optimizer = KVInjectionOptimizer(config=config)
        
        assert optimizer.backend == "standard"
        assert optimizer.config.kv_injection.chunk_size == 512
    
    def test_profiler_with_optimizer(self):
        """Test using profiler with optimizer"""
        config = FlashAttentionConfig(
            backend="standard",
            profiling=ProfilingConfig(enabled=True),
        )
        
        optimizer = KVInjectionOptimizer(config=config)
        profiler = AttentionProfiler(config.profiling)
        
        batch_size = 1
        seq_q = 4
        seq_k = 8
        seq_m = 4
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_q, num_heads, head_dim)
        key = torch.randn(batch_size, seq_k, num_heads, head_dim)
        value = torch.randn(batch_size, seq_k, num_heads, head_dim)
        memory_key = torch.randn(batch_size, seq_m, num_heads, head_dim)
        memory_value = torch.randn(batch_size, seq_m, num_heads, head_dim)
        
        with profiler.profile("kv_injection", backend=optimizer.backend):
            result = optimizer.inject(
                query, key, value, memory_key, memory_value
            )
        
        assert len(profiler) == 1
        assert result.output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
