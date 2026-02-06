"""
Unit tests for Query-Conditioned Memory Projection.

Tests the FiLM-style modulation:
- Memory-centric projection (query modulates, not re-encodes)
- Residual connection preservation
- Parameter efficiency
"""

import pytest
import torch
import torch.nn as nn

from dki.core.components.query_conditioned_projection import (
    QueryConditionedProjection,
    IdentityProjection,
)


class TestQueryConditionedProjection:
    """Tests for QueryConditionedProjection."""
    
    @pytest.fixture
    def projection(self):
        """Create projection instance."""
        return QueryConditionedProjection(
            hidden_dim=128,
            rank=16,
            dropout=0.0,  # Disable dropout for deterministic tests
        )
    
    def test_initialization(self, projection):
        """Test projection initialization."""
        assert projection.hidden_dim == 128
        assert projection.rank == 16
        assert projection.W_mem.shape == (128, 16)
    
    def test_forward_2d_input(self, projection):
        """Test forward pass with 2D input."""
        X_mem = torch.randn(10, 128)  # [mem_len, hidden_dim]
        X_user = torch.randn(5, 128)  # [user_len, hidden_dim]
        
        output = projection(X_mem, X_user)
        
        assert output.shape == X_mem.shape
    
    def test_forward_3d_input(self, projection):
        """Test forward pass with 3D input (batched)."""
        X_mem = torch.randn(2, 10, 128)  # [batch, mem_len, hidden_dim]
        X_user = torch.randn(2, 5, 128)  # [batch, user_len, hidden_dim]
        
        output = projection(X_mem, X_user)
        
        assert output.shape == X_mem.shape
    
    def test_residual_connection(self, projection):
        """Test that residual connection is preserved."""
        X_mem = torch.randn(10, 128)
        X_user = torch.randn(5, 128)
        
        output = projection(X_mem, X_user)
        
        # Output should be similar to input (residual dominant)
        # Due to initialization, projection should be near-identity initially
        correlation = torch.corrcoef(
            torch.stack([X_mem.flatten(), output.flatten()])
        )[0, 1]
        
        # Should have positive correlation (residual preserved)
        assert correlation > 0
    
    def test_return_modulation(self, projection):
        """Test returning modulation parameters."""
        X_mem = torch.randn(10, 128)
        X_user = torch.randn(5, 128)
        
        output, gamma, beta = projection(X_mem, X_user, return_modulation=True)
        
        assert output.shape == X_mem.shape
        # gamma and beta shape is (1, rank) due to batch dimension
        assert gamma.shape[-1] == projection.rank
        assert beta.shape[-1] == projection.rank
    
    def test_gamma_initialization(self, projection):
        """Test gamma initialized to ones."""
        # Check gamma_net bias initialized to ones
        assert torch.allclose(
            projection.gamma_net.bias,
            torch.ones_like(projection.gamma_net.bias),
        )
    
    def test_beta_initialization(self, projection):
        """Test beta initialized to zeros."""
        # Check beta_net bias initialized to zeros
        assert torch.allclose(
            projection.beta_net.bias,
            torch.zeros_like(projection.beta_net.bias),
        )
    
    def test_parameter_count(self, projection):
        """Test parameter count."""
        param_count = projection.get_parameter_count()
        
        # Should have reasonable number of parameters
        assert param_count > 0
        
        # For hidden_dim=128, rank=16:
        # W_mem: 128 * 16 = 2048
        # gamma_net: 128 * 16 + 16 = 2064
        # beta_net: 128 * 16 + 16 = 2064
        # proj_out: 16 * 128 + 128 = 2176
        # layer_norm: 128 * 2 = 256
        # Total: ~8608
        assert param_count < 100000  # Should be efficient
    
    def test_memory_overhead(self, projection):
        """Test memory overhead reporting."""
        overhead = projection.get_memory_overhead()
        
        assert isinstance(overhead, str)
        assert 'KB' in overhead or 'B' in overhead or 'MB' in overhead
    
    def test_different_sequence_lengths(self, projection):
        """Test with different sequence lengths."""
        # Short memory
        X_mem_short = torch.randn(5, 128)
        X_user = torch.randn(10, 128)
        output_short = projection(X_mem_short, X_user)
        assert output_short.shape == X_mem_short.shape
        
        # Long memory
        X_mem_long = torch.randn(100, 128)
        output_long = projection(X_mem_long, X_user)
        assert output_long.shape == X_mem_long.shape
    
    def test_gradient_flow(self, projection):
        """Test gradient flow through projection."""
        X_mem = torch.randn(10, 128, requires_grad=True)
        X_user = torch.randn(5, 128, requires_grad=True)
        
        output = projection(X_mem, X_user)
        loss = output.sum()
        loss.backward()
        
        # Gradients should flow
        assert X_mem.grad is not None
        assert X_user.grad is not None


class TestFiLMModulation:
    """Tests for FiLM-style modulation behavior."""
    
    @pytest.fixture
    def projection(self):
        return QueryConditionedProjection(hidden_dim=128, rank=16, dropout=0.0)
    
    def test_different_queries_different_modulation(self, projection):
        """Test that different queries produce different outputs."""
        torch.manual_seed(42)
        X_mem = torch.randn(10, 128)
        X_user_1 = torch.randn(5, 128) * 10  # Larger values
        torch.manual_seed(123)
        X_user_2 = torch.randn(5, 128) * -10  # Different query with opposite sign
        
        output_1, _, _ = projection(X_mem, X_user_1, return_modulation=True)
        output_2, _, _ = projection(X_mem, X_user_2, return_modulation=True)
        
        # Different queries should produce different outputs
        assert not torch.allclose(output_1, output_2)
    
    def test_same_query_same_modulation(self, projection):
        """Test that same query produces same modulation."""
        X_mem = torch.randn(10, 128)
        X_user = torch.randn(5, 128)
        
        _, gamma_1, beta_1 = projection(X_mem, X_user, return_modulation=True)
        _, gamma_2, beta_2 = projection(X_mem, X_user, return_modulation=True)
        
        # Same query should produce same modulation
        assert torch.allclose(gamma_1, gamma_2)
        assert torch.allclose(beta_1, beta_2)
    
    def test_modulation_bounded(self, projection):
        """Test that modulation values are bounded."""
        X_mem = torch.randn(10, 128)
        X_user = torch.randn(5, 128)
        
        _, gamma, beta = projection(X_mem, X_user, return_modulation=True)
        
        # Gamma should be around 1 (initialized to 1)
        # Beta should be around 0 (initialized to 0)
        # Values shouldn't explode
        assert gamma.abs().max() < 100
        assert beta.abs().max() < 100


class TestIdentityProjection:
    """Tests for IdentityProjection (ablation baseline)."""
    
    @pytest.fixture
    def identity(self):
        return IdentityProjection()
    
    def test_identity_forward(self, identity):
        """Test identity projection returns input unchanged."""
        X_mem = torch.randn(10, 128)
        X_user = torch.randn(5, 128)
        
        output = identity(X_mem, X_user)
        
        assert torch.allclose(output, X_mem)
    
    def test_identity_with_batch(self, identity):
        """Test identity projection with batched input."""
        X_mem = torch.randn(2, 10, 128)
        X_user = torch.randn(2, 5, 128)
        
        output = identity(X_mem, X_user)
        
        assert torch.allclose(output, X_mem)


class TestProjectionEfficiency:
    """Tests for projection efficiency."""
    
    def test_low_rank_efficiency(self):
        """Test that low rank reduces parameters."""
        proj_high_rank = QueryConditionedProjection(hidden_dim=128, rank=64)
        proj_low_rank = QueryConditionedProjection(hidden_dim=128, rank=16)
        
        high_params = proj_high_rank.get_parameter_count()
        low_params = proj_low_rank.get_parameter_count()
        
        assert low_params < high_params
    
    def test_hidden_dim_scaling(self):
        """Test parameter scaling with hidden dimension."""
        proj_small = QueryConditionedProjection(hidden_dim=128, rank=16)
        proj_large = QueryConditionedProjection(hidden_dim=512, rank=16)
        
        small_params = proj_small.get_parameter_count()
        large_params = proj_large.get_parameter_count()
        
        # Parameters should scale with hidden_dim
        assert large_params > small_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
