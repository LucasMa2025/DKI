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
        """Serialize to bytes for storage.
        
        Note: bfloat16 tensors cannot be directly converted to numpy.
        We convert to float32 first, then serialize. The from_bytes()
        method handles the reverse conversion using the dtype parameter.
        """
        key_cpu = self.key.cpu()
        value_cpu = self.value.cpu()
        # bfloat16 has no numpy equivalent; convert to float32 for serialization
        if key_cpu.dtype == torch.bfloat16:
            key_cpu = key_cpu.float()
        if value_cpu.dtype == torch.bfloat16:
            value_cpu = value_cpu.float()
        return (
            key_cpu.numpy().tobytes(),
            value_cpu.numpy().tobytes(),
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
        """Deserialize from bytes.
        
        Args:
            key_bytes: Serialized key tensor bytes
            value_bytes: Serialized value tensor bytes
            shape: Tensor shape
            layer_idx: Layer index
            dtype: Target torch dtype (also used to determine numpy dtype for parsing)
        """
        # Map torch dtype to numpy dtype for correct byte interpretation
        # Previously hardcoded np.float16, causing data corruption for float32/bfloat16 models
        _torch_to_numpy = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bfloat16: np.float32,  # bfloat16 serialized as float32 in to_bytes()
        }
        np_dtype = _torch_to_numpy.get(dtype, np.float16)
        
        key = torch.from_numpy(
            np.frombuffer(key_bytes, dtype=np_dtype).copy().reshape(shape)
        ).to(dtype)
        value = torch.from_numpy(
            np.frombuffer(value_bytes, dtype=np_dtype).copy().reshape(shape)
        ).to(dtype)
        return cls(key=key, value=value, layer_idx=layer_idx)


@dataclass
class PackedKV:
    """
    打包的 KV 缓存 (P2-1 优化)
    
    将所有层的 KV 合并为单一 tensor，大幅减少 CPU⇄GPU 传输次数。
    
    Shape:
    - keys:   [L, H, T, D]  (L=layers, H=heads, T=tokens, D=head_dim)
    - values: [L, H, T, D]
    
    优势:
    - CPU→GPU 拷贝: 64 次 → 2 次 (32 层模型)
    - Redis 序列化: 64 次 metadata → 1 次
    - Alpha scaling: per-layer loop → 一次 vectorized mul_
    - allocator 碎片显著降低
    
    兼容性:
    - from_entries(): 从 List[KVCacheEntry] 打包
    - to_entries(): 解包为 List[KVCacheEntry] (兼容旧接口)
    - to(): 整体搬移到目标设备 (单次传输)
    """
    keys: torch.Tensor       # [L, H, T, D]
    values: torch.Tensor     # [L, H, T, D]
    num_layers: int
    dtype: torch.dtype
    
    @classmethod
    def from_entries(cls, entries: List[KVCacheEntry]) -> "PackedKV":
        """
        从 List[KVCacheEntry] 打包为 PackedKV
        
        Args:
            entries: KV 缓存条目列表 (每层一个)
            
        Returns:
            PackedKV 实例
            
        Raises:
            ValueError: 如果 entries 为空
        """
        if not entries:
            raise ValueError("Cannot pack empty entries")
        
        # 按 layer_idx 排序确保顺序正确
        sorted_entries = sorted(entries, key=lambda e: e.layer_idx)
        
        # stack: [L, H, T, D] (squeeze batch dim if present)
        keys = torch.stack([e.key.squeeze(0) for e in sorted_entries])
        values = torch.stack([e.value.squeeze(0) for e in sorted_entries])
        
        return cls(
            keys=keys,
            values=values,
            num_layers=len(sorted_entries),
            dtype=keys.dtype,
        )
    
    def to_entries(self) -> List[KVCacheEntry]:
        """
        解包为 List[KVCacheEntry] (兼容旧接口)
        
        Returns:
            KV 缓存条目列表
        """
        return [
            KVCacheEntry(
                key=self.keys[i].unsqueeze(0),    # 恢复 batch dim
                value=self.values[i].unsqueeze(0),
                layer_idx=i,
            )
            for i in range(self.num_layers)
        ]
    
    def to(self, device, non_blocking: bool = False) -> "PackedKV":
        """
        整体搬移到目标设备 (单次传输，替代 64 次逐层传输)
        
        Args:
            device: 目标设备 (如 "cuda:0", "cpu")
            non_blocking: 是否使用非阻塞传输 (需配合 pinned memory)
            
        Returns:
            新的 PackedKV 实例 (在目标设备上)
        """
        return PackedKV(
            keys=self.keys.to(device, non_blocking=non_blocking),
            values=self.values.to(device, non_blocking=non_blocking),
            num_layers=self.num_layers,
            dtype=self.dtype,
        )
    
    def cpu(self) -> "PackedKV":
        """搬移到 CPU"""
        return self.to("cpu")
    
    def pin_memory(self) -> "PackedKV":
        """
        将 CPU tensor 转为 pinned memory (P2-2 预留)
        
        Pinned memory 可被 GPU 直接 DMA 访问，
        配合 non_blocking=True 可实现传输与计算重叠。
        """
        if self.keys.is_cuda:
            logger.warning("pin_memory() called on CUDA tensor, ignoring")
            return self
        return PackedKV(
            keys=self.keys.pin_memory(),
            values=self.values.pin_memory(),
            num_layers=self.num_layers,
            dtype=self.dtype,
        )
    
    def scale_values(self, alpha: float) -> "PackedKV":
        """
        Vectorized alpha scaling (inplace)
        
        一次 mul_ 替代 per-layer loop。
        注意: Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
        
        Args:
            alpha: 缩放因子
            
        Returns:
            self (inplace 修改)
        """
        if alpha != 1.0:
            self.values.mul_(alpha)
        return self
    
    @property
    def total_bytes(self) -> int:
        """总字节数 (keys + values)"""
        return (
            self.keys.nelement() * self.keys.element_size()
            + self.values.nelement() * self.values.element_size()
        )
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        return self.keys.device
    
    def __repr__(self) -> str:
        return (
            f"PackedKV(layers={self.num_layers}, "
            f"shape={list(self.keys.shape)}, "
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"bytes={self.total_bytes})"
        )


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
