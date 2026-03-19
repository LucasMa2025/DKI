"""
Base Model Adapter for DKI System
Abstract base class for all model adapters

Supports FlashAttention-3/2 integration for optimized K/V injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import asyncio
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
    
    # Logprobs (for entropy-gated retrieval)
    # List of per-token top-k log probabilities: [[logp1, logp2, ...], ...]
    logprobs: Optional[List[List[float]]] = None
    
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
            dtype: Target torch dtype (also used to determine numpy dtype for parsing).
                   建议显式传入模型的实际 dtype，避免 bfloat16 模型使用默认
                   float16 导致精度静默损失。
        """
        # 默认 float16 警告：bfloat16 模型若未显式传入 dtype，
        # 反序列化后的 KV 精度会低于模型实际精度
        if dtype == torch.float16:
            import warnings
            warnings.warn(
                "KVCacheEntry.from_bytes: dtype defaults to float16. "
                "If your model uses bfloat16, pass dtype=torch.bfloat16 explicitly "
                "to avoid silent precision loss.",
                UserWarning,
                stacklevel=2,
            )
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
        Vectorized alpha scaling (返回新对象，不修改原始 tensor)
        
        一次 mul 替代 per-layer loop。
        注意: Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
        
        设计说明: 使用非 inplace 操作 (mul 而非 mul_)，避免缓存复用时
        alpha 累乘问题——同一 PackedKV 被多次请求以不同 alpha 复用时，
        inplace 修改会导致实际缩放因子为历次 alpha 的乘积。
        
        Args:
            alpha: 缩放因子
            
        Returns:
            新的 PackedKV 实例 (values 已缩放，keys 不变)
        """
        if alpha == 1.0:
            return self
        return PackedKV(
            keys=self.keys,
            values=self.values.mul(alpha),
            num_layers=self.num_layers,
            dtype=self.dtype,
        )
    
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


def extract_kv_from_past(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    从 model.forward() 返回的 past_key_values 中提取 (key, value) 对列表。
    
    兼容 Transformers 4.x 和 5.x:
    - Transformers < 4.45: past_key_values 是 tuple of (key, value) per layer
    - Transformers 4.45-4.x: DynamicCache 支持 __getitem__ 返回 (key, value) 元组
    - Transformers 5.x: DynamicCache 使用 DynamicLayer 结构:
      - 5.x 某些版本仍支持 __getitem__ / 迭代 (返回 tuple)
      - 5.x 某些版本移除了 __getitem__, 需通过 layers[i].keys/.values 访问
      - 也可能有 key_cache / value_cache 列表属性
    
    Args:
        past_key_values: model forward 返回的 past_key_values
        
    Returns:
        List[(key_tensor, value_tensor)] — 每层一个元组
    """
    # 尝试导入 DynamicCache (可能不可用)
    try:
        from transformers import DynamicCache
        _is_dynamic_cache = isinstance(past_key_values, DynamicCache)
    except ImportError:
        _is_dynamic_cache = False
    
    if _is_dynamic_cache:
        # 策略 1: key_cache / value_cache 列表属性 (某些 5.x 版本)
        # 注意: 5.x 的 DynamicCache.key_cache 可能存在但为空列表 (模型刚初始化时),
        # 必须同时检查第一个元素非 None, 否则会错误降级到策略2/3/4
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            kc = past_key_values.key_cache
            vc = past_key_values.value_cache
            if isinstance(kc, (list, tuple)) and len(kc) > 0 and kc[0] is not None:
                return [(kc[i], vc[i]) for i in range(len(kc))]
        
        # 策略 2: layers[i].keys / .values (Transformers 5.x DynamicLayer)
        if hasattr(past_key_values, 'layers'):
            layers = past_key_values.layers
            if layers and hasattr(layers[0], 'keys') and hasattr(layers[0], 'values'):
                return [(layer.keys, layer.values) for layer in layers]
        
        # 策略 3: __getitem__ (Transformers 4.45+ 和部分 5.x)
        try:
            num_layers = len(past_key_values)
            result = past_key_values[0]
            if isinstance(result, tuple) and len(result) >= 2:
                return [past_key_values[i] for i in range(num_layers)]
        except (TypeError, IndexError):
            pass
        
        # 策略 4: 迭代
        try:
            return list(past_key_values)
        except TypeError:
            pass
        
        # 策略 5: to_legacy_cache() 方法
        if hasattr(past_key_values, 'to_legacy_cache'):
            legacy = past_key_values.to_legacy_cache()
            return list(legacy)
        
        raise TypeError(
            f"Cannot extract KV from DynamicCache: no known access method works. "
            f"Attributes: {[a for a in dir(past_key_values) if not a.startswith('_')]}"
        )
    
    # Legacy tuple format (Transformers < 4.45)
    return list(past_key_values)


def build_dynamic_cache_from_entries(
    entries: List[KVCacheEntry],
    device: torch.device,
    alpha: float = 1.0,
) -> Tuple[Any, int]:
    """
    从 KVCacheEntry 列表构建 DynamicCache (或 legacy tuple) 用于注入。
    
    兼容 Transformers 4.x 和 5.x:
    - Transformers >= 4.45: 返回 DynamicCache 实例
    - Transformers < 4.45: 返回 tuple of (key, value) per layer
    
    Value 会按 alpha 缩放 (Key 不缩放, 保护 attention addressing)。
    
    Args:
        entries: KVCacheEntry 列表 (每层一个, 通常在 CPU 上)
        device: 目标设备 (通常是 GPU)
        alpha: Value 缩放因子 [0, 1]
        
    Returns:
        (past_kv, mem_len):
        - past_kv: DynamicCache 或 tuple, 可直接传给 model.generate()
        - mem_len: 偏好 token 数
    """
    if not entries:
        return None, 0
    
    mem_len = entries[0].key.shape[2]
    
    try:
        from transformers import DynamicCache
        cache = DynamicCache()
        for entry in entries:
            key = entry.key.to(device)
            value = entry.value.to(device)
            if alpha < 1.0:
                value = value * alpha
            cache.update(key, value, entry.layer_idx)
        return cache, mem_len
    except ImportError:
        # Legacy tuple format
        scaled_kv = []
        for entry in entries:
            key = entry.key.to(device)
            value = entry.value.to(device)
            if alpha < 1.0:
                value = value * alpha
            scaled_kv.append((key, value))
        return tuple(scaled_kv), mem_len


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model engines (vLLM, LLaMA, DeepSeek, GLM) must implement this interface.
    
    Supports FlashAttention-3/2 integration for optimized K/V injection.
    """
    
    # 支持的量化模式
    SUPPORTED_QUANTIZATIONS = ("none", "4bit", "int4", "8bit", "int8", "gptq", "awq", "fp8")
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        quantization: str = "none",
        quantization_config: Optional[Dict[str, Any]] = None,
        # 向后兼容: 接受旧参数名 torch_dtype
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        # 向后兼容: torch_dtype 优先 (旧代码可能传入), 否则使用 dtype
        _dtype_str = torch_dtype if torch_dtype is not None else dtype
        self.dtype = getattr(torch, _dtype_str, torch.float16)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # ============ 量化配置 ============
        self.quantization = self._normalize_quantization(quantization)
        self.quantization_config = quantization_config or {}
        
        # Model architecture info
        self.hidden_dim: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        
        # FlashAttention support
        self._flash_attn_config: Optional["FlashAttentionConfig"] = None
        self._flash_attn_backend: Optional[str] = None
        self._kv_injection_optimizer: Optional["KVInjectionOptimizer"] = None
        
        quant_info = f", quantization={self.quantization}" if self.quantization != "none" else ""
        logger.info(f"Initializing {self.__class__.__name__} with {model_name}{quant_info}")
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
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
        max_new_tokens: int = 2048,
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
    
    # ================================================================
    # 异步与流式生成 (默认实现, 子类可覆盖)
    # ================================================================
    
    async def async_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """
        Async version of generate().
        
        Default implementation: runs synchronous generate() in a thread pool
        to avoid blocking the event loop. Subclasses (e.g. SGLangAdapter)
        can override with native async implementations.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            ),
        )
    
    async def async_stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async streaming generation — yields text chunks as they are produced.
        
        Default implementation: falls back to async_generate() and yields
        the full text as a single chunk. Subclasses should override this
        with true token-by-token streaming when the engine supports it.
        
        Yields:
            str: text chunks (tokens or groups of tokens)
        """
        output = await self.async_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        yield output.text
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Synchronous streaming generation — yields text chunks.
        
        Default implementation: falls back to generate() and yields
        the full text as a single chunk. Subclasses should override this
        with true token-by-token streaming when the engine supports it.
        
        Yields:
            str: text chunks
        """
        output = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        yield output.text
    
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
    
    # ============ 量化支持 ============
    
    @classmethod
    def _normalize_quantization(cls, quantization: str) -> str:
        """
        规范化量化模式名称.
        
        将各种别名统一为标准名称:
        - "int4" → "4bit"
        - "int8" → "8bit"
        - "none" / "" / None → "none"
        
        Args:
            quantization: 量化模式字符串
            
        Returns:
            规范化后的量化模式
        """
        if not quantization or quantization.lower() in ("none", ""):
            return "none"
        
        normalized = quantization.lower().strip()
        
        # 别名映射
        alias_map = {
            "int4": "4bit",
            "int8": "8bit",
            "4": "4bit",
            "8": "8bit",
            "float8": "fp8",
            "e4m3": "fp8",
            "fp8_e4m3fn": "fp8",
        }
        normalized = alias_map.get(normalized, normalized)
        
        if normalized not in cls.SUPPORTED_QUANTIZATIONS:
            logger.warning(
                f"Unknown quantization mode '{quantization}', "
                f"supported: {cls.SUPPORTED_QUANTIZATIONS}. Falling back to 'none'."
            )
            return "none"
        
        return normalized
    
    @property
    def is_quantized(self) -> bool:
        """是否启用了量化."""
        return self.quantization != "none"
    
    @property
    def is_4bit(self) -> bool:
        """是否使用 4-bit 量化."""
        return self.quantization == "4bit"
    
    @property
    def is_8bit(self) -> bool:
        """是否使用 8-bit 量化."""
        return self.quantization == "8bit"
    
    @property
    def is_fp8(self) -> bool:
        """是否使用 FP8 量化."""
        return self.quantization == "fp8"
    
    def _build_bnb_config(self) -> Optional[Any]:
        """
        构建 BitsAndBytesConfig (4bit/8bit 量化).
        
        仅在 quantization 为 "4bit" 或 "8bit" 时有效。
        GPTQ/AWQ 不使用 BitsAndBytesConfig, 由 AutoModelForCausalLM 自动检测。
        
        Returns:
            BitsAndBytesConfig 实例, 或 None (非 bitsandbytes 量化)
            
        Raises:
            ImportError: bitsandbytes 未安装
        """
        if self.quantization not in ("4bit", "8bit"):
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytesConfig requires 'bitsandbytes' package. "
                "Install with: pip install bitsandbytes>=0.41.0"
            )
        
        if self.quantization == "4bit":
            # 4-bit NF4 量化 (QLoRA 论文推荐)
            compute_dtype_str = self.quantization_config.get(
                "bnb_4bit_compute_dtype", "bfloat16"
            )
            compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quantization_config.get(
                    "bnb_4bit_quant_type", "nf4"
                ),
                bnb_4bit_use_double_quant=self.quantization_config.get(
                    "bnb_4bit_use_double_quant", True
                ),
                bnb_4bit_compute_dtype=compute_dtype,
            )
            logger.info(
                f"4-bit quantization config: "
                f"quant_type={bnb_config.bnb_4bit_quant_type}, "
                f"double_quant={bnb_config.bnb_4bit_use_double_quant}, "
                f"compute_dtype={compute_dtype}"
            )
            return bnb_config
        
        elif self.quantization == "8bit":
            # 8-bit LLM.int8() 量化
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("8-bit quantization config: LLM.int8()")
            return bnb_config
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dtype': str(self.dtype),
            'quantization': self.quantization,
            'is_quantized': self.is_quantized,
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
