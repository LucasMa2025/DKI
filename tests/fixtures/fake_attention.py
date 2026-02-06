"""
Fake Attention mechanisms for CPU-only testing.
Provides deterministic attention computation without GPU.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FakeAttentionOutput:
    """Output from fake attention computation."""
    output: torch.Tensor
    attention_weights: torch.Tensor
    kv_len: int
    query_len: int


class FakeAttention:
    """
    Fake attention mechanism for testing.
    
    Implements standard scaled dot-product attention
    without requiring GPU or optimized kernels.
    """
    
    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.0,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        mem_len: int = 0,
    ) -> FakeAttentionOutput:
        """
        Compute attention with optional memory injection scaling.
        
        Args:
            query: [batch, heads, q_len, head_dim]
            key: [batch, heads, kv_len, head_dim]
            value: [batch, heads, kv_len, head_dim]
            attention_mask: Optional mask
            alpha: Scaling factor for memory positions
            mem_len: Number of memory positions (for alpha scaling)
            
        Returns:
            FakeAttentionOutput with output and attention weights
        """
        batch_size, num_heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply alpha scaling to memory positions
        if alpha < 1.0 and mem_len > 0:
            # Apply log(alpha) bias to memory positions
            logit_bias = torch.log(torch.tensor(alpha + 1e-9))
            scores[..., :mem_len] = scores[..., :mem_len] + logit_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout (disabled for testing)
        if self.dropout > 0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.dropout)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return FakeAttentionOutput(
            output=output,
            attention_weights=attention_weights,
            kv_len=kv_len,
            query_len=q_len,
        )
    
    def compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution.
        
        Higher entropy = more uniform attention = higher uncertainty.
        """
        # Avoid log(0)
        eps = 1e-9
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + eps),
            dim=-1
        )
        return entropy.mean().item()


def create_fake_attention_output(
    batch_size: int = 1,
    num_heads: int = 4,
    q_len: int = 10,
    kv_len: int = 15,
    head_dim: int = 32,
    device: str = "cpu",
) -> FakeAttentionOutput:
    """
    Create fake attention output for testing.
    
    Returns deterministic output with proper shapes.
    """
    torch.manual_seed(42)
    
    output = torch.randn(batch_size, num_heads, q_len, head_dim, device=device)
    
    # Create normalized attention weights
    attention_weights = torch.softmax(
        torch.randn(batch_size, num_heads, q_len, kv_len, device=device),
        dim=-1
    )
    
    return FakeAttentionOutput(
        output=output,
        attention_weights=attention_weights,
        kv_len=kv_len,
        query_len=q_len,
    )


class FakeMultiHeadAttention:
    """
    Fake multi-head attention for testing DKI injection.
    
    Supports K/V injection at the attention level.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention = FakeAttention(num_heads, self.head_dim)
        
        # Fake projection weights
        torch.manual_seed(42)
        self.W_q = torch.randn(hidden_dim, hidden_dim) * 0.02
        self.W_k = torch.randn(hidden_dim, hidden_dim) * 0.02
        self.W_v = torch.randn(hidden_dim, hidden_dim) * 0.02
        self.W_o = torch.randn(hidden_dim, hidden_dim) * 0.02
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        injected_key: Optional[torch.Tensor] = None,
        injected_value: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional K/V injection.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            injected_key: Optional injected K [batch, heads, mem_len, head_dim]
            injected_value: Optional injected V [batch, heads, mem_len, head_dim]
            alpha: Injection strength
            
        Returns:
            output: [batch, seq, hidden_dim]
            attention_weights: [batch, heads, seq, total_kv_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = torch.matmul(hidden_states, self.W_q)
        K = torch.matmul(hidden_states, self.W_k)
        V = torch.matmul(hidden_states, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Inject K/V if provided
        mem_len = 0
        if injected_key is not None and injected_value is not None:
            mem_len = injected_key.shape[2]
            K = torch.cat([injected_key, K], dim=2)
            V = torch.cat([injected_value, V], dim=2)
        
        # Compute attention
        attn_output = self.attention.forward(Q, K, V, alpha=alpha, mem_len=mem_len)
        
        # Reshape and project output
        output = attn_output.output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        output = torch.matmul(output, self.W_o)
        
        return output, attn_output.attention_weights
