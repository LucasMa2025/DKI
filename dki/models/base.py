"""
Base Model Adapter for DKI System
Abstract base class for all model adapters

Supports FlashAttention-3/2 integration for optimized K/V injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from dki.attention import FlashAttentionConfig, KVInjectionOptimizer


@dataclass
class ModelOutput:
    """Standard output format for model generation."""
    
    text: str
    tokens: Optional[List[int]] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    
    # Metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KVCacheEntry:
    """Key-Value cache entry for a single layer."""
    
    key: torch.Tensor  # [batch, num_heads, seq_len, head_dim]
    value: torch.Tensor  # [batch, num_heads, seq_len, head_dim]
    layer_idx: int
    
    def to_device(self, device: str) -> 'KVCacheEntry':
        """Move tensors to specified device."""
        return KVCacheEntry(
            key=self.key.to(device),
            value=self.value.to(device),
            layer_idx=self.layer_idx,
        )
    
    def to_bytes(self) -> Tuple[bytes, bytes]:
        """Serialize to bytes for storage."""
        return (
            self.key.cpu().numpy().tobytes(),
            self.value.cpu().numpy().tobytes(),
        )
    
    @classmethod
    def from_bytes(
        cls,
        key_bytes: bytes,
        value_bytes: bytes,
        shape: Tuple[int, ...],
        layer_idx: int,
        dtype: torch.dtype = torch.float16,
    ) -> 'KVCacheEntry':
        """Deserialize from bytes."""
        key = torch.from_numpy(
            np.frombuffer(key_bytes, dtype=np.float16).reshape(shape)
        ).to(dtype)
        value = torch.from_numpy(
            np.frombuffer(value_bytes, dtype=np.float16).reshape(shape)
        ).to(dtype)
        return cls(key=key, value=value, layer_idx=layer_idx)


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model engines (vLLM, LLaMA, DeepSeek, GLM) must implement this interface.
    
    Supports FlashAttention-3/2 integration for optimized K/V injection.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Model architecture info
        self.hidden_dim: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        
        # FlashAttention support
        self._flash_attn_config: Optional["FlashAttentionConfig"] = None
        self._flash_attn_backend: Optional[str] = None
        self._kv_injection_optimizer: Optional["KVInjectionOptimizer"] = None
        
        logger.info(f"Initializing {self.__class__.__name__} with {model_name}")
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        pass
    
    @abstractmethod
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute Key-Value representations for text.
        
        Args:
            text: Input text
            return_hidden: Whether to return hidden states
            
        Returns:
            List of KVCacheEntry for each layer, optional hidden states
        """
        pass
    
    @abstractmethod
    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> ModelOutput:
        """
        Generate with injected K/V cache.
        
        Args:
            prompt: User input prompt
            injected_kv: Pre-computed K/V cache to inject
            alpha: Injection strength (0-1)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            ModelOutput with generated text
        """
        pass
    
    @abstractmethod
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute prefill-stage entropy for gating.
        
        Args:
            text: Input text
            layer_idx: Which layer's attention to use
            
        Returns:
            Entropy value
        """
        pass
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
    
    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Decode tokens to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache()
        self._is_loaded = False
        logger.info(f"Unloaded {self.model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dtype': str(self.torch_dtype),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'is_loaded': self._is_loaded,
            'flash_attn_enabled': self._flash_attn_backend is not None,
            'flash_attn_backend': self._flash_attn_backend,
        }
    
    # ============ FlashAttention Support ============
    
    def enable_flash_attention(
        self,
        config: Optional["FlashAttentionConfig"] = None,
    ) -> str:
        """
        Enable FlashAttention optimization.
        
        Args:
            config: FlashAttention configuration (optional)
            
        Returns:
            Selected backend name ("fa3" | "fa2" | "standard")
        """
        from dki.attention import (
            FlashAttentionConfig,
            FlashAttentionBackend,
            KVInjectionOptimizer,
        )
        
        self._flash_attn_config = config or FlashAttentionConfig()
        
        # Detect best backend
        if self._flash_attn_config.backend == "auto":
            self._flash_attn_backend = FlashAttentionBackend.detect_best_backend()
        else:
            self._flash_attn_backend = FlashAttentionBackend.validate_backend(
                self._flash_attn_config.backend
            )
        
        # Create K/V injection optimizer
        self._kv_injection_optimizer = KVInjectionOptimizer(
            config=self._flash_attn_config,
            backend=self._flash_attn_backend,
        )
        
        logger.info(
            f"FlashAttention enabled: backend={self._flash_attn_backend}, "
            f"model={self.model_name}"
        )
        
        return self._flash_attn_backend
    
    def disable_flash_attention(self):
        """Disable FlashAttention optimization."""
        self._flash_attn_config = None
        self._flash_attn_backend = None
        self._kv_injection_optimizer = None
        logger.info(f"FlashAttention disabled for {self.model_name}")
    
    @property
    def flash_attn_enabled(self) -> bool:
        """Check if FlashAttention is enabled."""
        return self._flash_attn_backend is not None
    
    @property
    def flash_attn_backend(self) -> Optional[str]:
        """Get current FlashAttention backend."""
        return self._flash_attn_backend
    
    def get_flash_attn_stats(self) -> Dict[str, Any]:
        """Get FlashAttention statistics."""
        if self._kv_injection_optimizer is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "backend": self._flash_attn_backend,
            **self._kv_injection_optimizer.get_stats(),
        }
    
    def __repr__(self) -> str:
        fa_info = f", flash_attn={self._flash_attn_backend}" if self._flash_attn_backend else ""
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device}{fa_info})"
