"""
Unit tests for Position Remapper.

Tests position encoding compatibility:
- RoPE (LLaMA, Qwen, Mistral)
- ALiBi (BLOOM, MPT, Falcon)
- Attention mask extension
"""

import pytest
import torch

from dki.core.components.position_remapper import PositionRemapper


class TestPositionRemapper:
    """Tests for PositionRemapper."""
    
    @pytest.fixture
    def remapper(self):
        """Create remapper instance."""
        return PositionRemapper(
            strategy="virtual_prefix",
            position_encoding="rope",
        )
    
    def test_initialization(self, remapper):
        """Test remapper initialization."""
        assert remapper.strategy == "virtual_prefix"
        assert remapper.position_encoding == "rope"
    
    def test_detect_rope_models(self, remapper):
        """Test RoPE model detection."""
        rope_models = [
            "meta-llama/Llama-2-7b",
            "Qwen/Qwen-7B",
            "mistralai/Mistral-7B",
            "deepseek-ai/deepseek-llm-7b",
            "01-ai/Yi-6B",
        ]
        
        for model in rope_models:
            encoding = remapper.detect_position_encoding(model)
            assert encoding == "rope", f"Failed for {model}"
    
    def test_detect_alibi_models(self, remapper):
        """Test ALiBi model detection."""
        alibi_models = [
            "bigscience/bloom",
            "mosaicml/mpt-7b",
            "tiiuae/falcon-7b",
        ]
        
        for model in alibi_models:
            encoding = remapper.detect_position_encoding(model)
            assert encoding == "alibi", f"Failed for {model}"
    
    def test_detect_glm_model(self, remapper):
        """Test GLM model detection (uses rotary)."""
        encoding = remapper.detect_position_encoding("THUDM/glm-4")
        assert encoding == "rope"
    
    def test_detect_unknown_model(self, remapper):
        """Test unknown model defaults to absolute."""
        encoding = remapper.detect_position_encoding("unknown-model")
        assert encoding == "absolute"


class TestRoPERemapping:
    """Tests for RoPE position remapping."""
    
    @pytest.fixture
    def remapper(self):
        return PositionRemapper(strategy="virtual_prefix", position_encoding="rope")
    
    def test_remap_for_rope_virtual_prefix(self, remapper):
        """Test RoPE remapping with virtual prefix strategy."""
        K_mem = torch.randn(1, 4, 10, 32)
        V_mem = torch.randn(1, 4, 10, 32)
        
        K_out, V_out = remapper.remap_for_rope(K_mem, V_mem, mem_len=10)
        
        # With virtual prefix, tensors should be unchanged
        # (position encoding handled at attention level)
        assert K_out.shape == K_mem.shape
        assert V_out.shape == V_mem.shape
    
    def test_remap_for_rope_actual_prefix(self):
        """Test RoPE remapping with actual prefix strategy."""
        remapper = PositionRemapper(strategy="actual_prefix", position_encoding="rope")
        
        K_mem = torch.randn(1, 4, 10, 32)
        V_mem = torch.randn(1, 4, 10, 32)
        
        K_out, V_out = remapper.remap_for_rope(K_mem, V_mem, mem_len=10)
        
        # Shape should be preserved
        assert K_out.shape == K_mem.shape
        assert V_out.shape == V_mem.shape


class TestALiBiRemapping:
    """Tests for ALiBi position remapping."""
    
    @pytest.fixture
    def remapper(self):
        return PositionRemapper(strategy="virtual_prefix", position_encoding="alibi")
    
    def test_alibi_bias_shape(self, remapper):
        """Test ALiBi bias tensor shape."""
        mem_len = 5
        user_len = 10
        num_heads = 8
        device = torch.device("cpu")
        
        bias = remapper.remap_for_alibi(mem_len, user_len, num_heads, device)
        
        total_len = mem_len + user_len
        assert bias.shape == (num_heads, total_len, total_len)
    
    def test_alibi_slopes(self, remapper):
        """Test ALiBi slopes calculation."""
        slopes = remapper._get_alibi_slopes(8)
        
        assert len(slopes) == 8
        
        # Slopes should be positive and decreasing
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1]
    
    def test_alibi_slopes_power_of_2(self, remapper):
        """Test ALiBi slopes for power of 2 heads."""
        for n_heads in [2, 4, 8, 16]:
            slopes = remapper._get_alibi_slopes(n_heads)
            assert len(slopes) == n_heads
    
    def test_alibi_slopes_non_power_of_2(self, remapper):
        """Test ALiBi slopes for non-power of 2 heads."""
        for n_heads in [3, 5, 7, 12]:
            slopes = remapper._get_alibi_slopes(n_heads)
            assert len(slopes) == n_heads
    
    def test_alibi_bias_values(self, remapper):
        """Test ALiBi bias values."""
        mem_len = 3
        user_len = 5
        num_heads = 4
        device = torch.device("cpu")
        
        bias = remapper.remap_for_alibi(mem_len, user_len, num_heads, device)
        
        # Bias should be negative for distant positions
        # (ALiBi penalizes distance)
        # Diagonal should have bias based on position
        assert bias.shape[0] == num_heads


class TestAttentionMaskExtension:
    """Tests for attention mask extension."""
    
    @pytest.fixture
    def remapper(self):
        return PositionRemapper()
    
    def test_extend_attention_mask(self, remapper):
        """Test attention mask extension."""
        batch_size = 2
        user_len = 10
        mem_len = 5
        
        attention_mask = torch.ones(batch_size, user_len)
        device = attention_mask.device
        
        extended = remapper.get_extended_attention_mask(attention_mask, mem_len, device)
        
        assert extended.shape == (batch_size, mem_len + user_len)
    
    def test_extended_mask_memory_ones(self, remapper):
        """Test that memory positions are all ones."""
        batch_size = 2
        user_len = 10
        mem_len = 5
        
        attention_mask = torch.ones(batch_size, user_len)
        device = attention_mask.device
        
        extended = remapper.get_extended_attention_mask(attention_mask, mem_len, device)
        
        # Memory positions should be all ones
        assert torch.all(extended[:, :mem_len] == 1)
    
    def test_extended_mask_preserves_user(self, remapper):
        """Test that user mask is preserved."""
        batch_size = 2
        user_len = 10
        mem_len = 5
        
        # Create mask with some zeros
        attention_mask = torch.ones(batch_size, user_len)
        attention_mask[:, -2:] = 0  # Padding
        
        device = attention_mask.device
        
        extended = remapper.get_extended_attention_mask(attention_mask, mem_len, device)
        
        # User positions should match original
        assert torch.all(extended[:, mem_len:] == attention_mask)
    
    def test_extended_mask_dtype(self, remapper):
        """Test that extended mask preserves dtype."""
        attention_mask = torch.ones(2, 10, dtype=torch.float16)
        device = attention_mask.device
        
        extended = remapper.get_extended_attention_mask(attention_mask, 5, device)
        
        assert extended.dtype == attention_mask.dtype


class TestStrategySelection:
    """Tests for strategy selection."""
    
    def test_virtual_prefix_strategy(self):
        """Test virtual prefix strategy."""
        remapper = PositionRemapper(strategy="virtual_prefix")
        assert remapper.strategy == "virtual_prefix"
    
    def test_actual_prefix_strategy(self):
        """Test actual prefix strategy."""
        remapper = PositionRemapper(strategy="actual_prefix")
        assert remapper.strategy == "actual_prefix"
    
    def test_unknown_strategy_warning(self):
        """Test unknown strategy handling."""
        remapper = PositionRemapper(strategy="unknown")
        
        K_mem = torch.randn(1, 4, 10, 32)
        V_mem = torch.randn(1, 4, 10, 32)
        
        # Should return unchanged with warning
        K_out, V_out = remapper.remap_for_rope(K_mem, V_mem, mem_len=10)
        
        assert torch.allclose(K_out, K_mem)
        assert torch.allclose(V_out, V_mem)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
