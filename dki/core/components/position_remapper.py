"""
Position Remapper for DKI System
Handles position encoding compatibility for K/V injection
"""

import torch
from typing import Optional, Tuple
from loguru import logger

from dki.config.config_loader import ConfigLoader


class PositionRemapper:
    """
    Position encoding compatibility layer for K/V injection.
    
    Supported schemes:
    - RoPE (LLaMA, Qwen, Mistral): Remap to negative/prefix positions
    - ALiBi (BLOOM, MPT, Falcon): Adjust attention bias
    
    Strategy options:
    - Strategy 1 (Virtual prefix): Map memory to negative positions
      + Simple implementation
      - May be out of training distribution
    
    - Strategy 2 (Actual prefix): Insert memory at position 0
      + Within training distribution
      - Requires recomputing user input RoPE
    
    Current implementation uses Strategy 1; A/B testing recommended
    for production deployment.
    """
    
    def __init__(
        self,
        strategy: str = "virtual_prefix",
        position_encoding: str = "rope",
    ):
        config = ConfigLoader().config
        
        self.strategy = strategy or config.dki.position.strategy
        self.position_encoding = position_encoding
    
    def remap_for_rope(
        self,
        K_mem: torch.Tensor,
        V_mem: torch.Tensor,
        mem_len: int,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remap K/V for RoPE-based models.
        
        Maps memory to positions before user input.
        
        Args:
            K_mem: Memory key tensor [batch, heads, seq, head_dim]
            V_mem: Memory value tensor [batch, heads, seq, head_dim]
            mem_len: Length of memory sequence
            rope_cache: Optional pre-computed RoPE cache
            
        Returns:
            Remapped (K_mem, V_mem) tensors
        """
        if self.strategy == "virtual_prefix":
            # Map memory to negative positions: [-mem_len, -1]
            # For RoPE, this means computing cos/sin for negative positions
            # Most implementations will just work with the relative distances
            
            # If RoPE is already applied to K_mem, we may need to re-apply
            # with new positions. For simplicity, we assume K_mem already
            # has appropriate position encoding.
            
            return K_mem, V_mem
            
        elif self.strategy == "actual_prefix":
            # Insert at position 0, shift user positions
            # This requires recomputing RoPE for user input
            # Implementation depends on model architecture
            
            # For now, return unchanged (simplified)
            return K_mem, V_mem
        
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using identity")
            return K_mem, V_mem
    
    def remap_for_alibi(
        self,
        mem_len: int,
        user_len: int,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute ALiBi attention bias for memory + user sequence.
        
        Args:
            mem_len: Memory sequence length
            user_len: User input sequence length
            num_heads: Number of attention heads
            device: Target device
            
        Returns:
            ALiBi bias tensor [heads, total_len, total_len]
        """
        total_len = mem_len + user_len
        
        # Compute ALiBi slopes for each head
        # Slopes decrease exponentially: 2^(-8/n * i) for head i
        slopes = self._get_alibi_slopes(num_heads)
        slopes = slopes.to(device)
        
        # Create position matrix
        # Positions: [-mem_len, ..., -1, 0, 1, ..., user_len-1]
        positions = list(range(-mem_len, 0)) + list(range(user_len))
        positions = torch.tensor(positions, device=device)
        
        # Compute relative distances
        # distance[i, j] = positions[i] - positions[j]
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # ALiBi bias = slopes * distances
        # Shape: [heads, total_len, total_len]
        alibi_bias = slopes.view(-1, 1, 1) * distances.unsqueeze(0)
        
        return alibi_bias
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Get ALiBi slopes for attention heads.
        
        Standard ALiBi uses slopes that decrease exponentially.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)) - 3).ceil().item()))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if num_heads == 0:
            return torch.tensor([])
        
        # Check if num_heads is power of 2
        if (num_heads & (num_heads - 1)) == 0:
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Find closest power of 2
            closest_power = 2 ** ((num_heads - 1).bit_length() - 1)
            slopes = get_slopes_power_of_2(closest_power)
            
            # Add extra slopes for remaining heads
            extra_slopes = get_slopes_power_of_2(2 * closest_power)
            extra_slopes = extra_slopes[0::2][:num_heads - closest_power]
            slopes = slopes + extra_slopes
        
        return torch.tensor(slopes)
    
    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        mem_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extend attention mask to include memory positions.
        
        Args:
            attention_mask: Original attention mask [batch, user_len]
            mem_len: Memory sequence length
            device: Target device
            
        Returns:
            Extended mask [batch, mem_len + user_len]
        """
        batch_size = attention_mask.shape[0]
        
        # Create memory mask (all ones - always attend to memory)
        mem_mask = torch.ones(batch_size, mem_len, device=device, dtype=attention_mask.dtype)
        
        # Concatenate memory and user masks
        extended_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        return extended_mask
    
    def detect_position_encoding(self, model_name: str) -> str:
        """
        Detect position encoding type from model name.
        
        Args:
            model_name: Model name/path
            
        Returns:
            Position encoding type: 'rope', 'alibi', 'absolute', 'unknown'
        """
        model_lower = model_name.lower()
        
        # RoPE models
        rope_models = ['llama', 'qwen', 'mistral', 'yi', 'deepseek', 'baichuan']
        if any(m in model_lower for m in rope_models):
            return 'rope'
        
        # ALiBi models
        alibi_models = ['bloom', 'mpt', 'falcon']
        if any(m in model_lower for m in alibi_models):
            return 'alibi'
        
        # GLM uses rotary
        if 'glm' in model_lower:
            return 'rope'
        
        # Default to absolute (GPT-style)
        return 'absolute'
