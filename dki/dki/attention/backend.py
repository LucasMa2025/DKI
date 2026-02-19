"""
FlashAttention Backend Detection and Selection
后端检测与选择模块

Supports:
- FlashAttention-3 (H100/H200)
- FlashAttention-2 (A100/RTX 4090)
- Standard PyTorch (fallback)

Author: AGI Demo Project
Version: 1.0.0
"""

import math
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

# 检测 FlashAttention 是否可用
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = getattr(flash_attn, "__version__", "unknown")
    logger.info(f"FlashAttention available: version {FLASH_ATTN_VERSION}")
except ImportError:
    logger.warning(
        "FlashAttention not installed. "
        "Install with: pip install flash-attn --no-build-isolation"
    )


class BackendType(str, Enum):
    """后端类型"""
    FA3 = "fa3"  # FlashAttention-3 (H100+)
    FA2 = "fa2"  # FlashAttention-2 (A100/4090)
    STANDARD = "standard"  # PyTorch 标准实现
    AUTO = "auto"  # 自动检测


class FlashAttentionBackend:
    """
    FlashAttention 后端管理
    
    自动检测 GPU 能力并选择最佳后端
    
    Example:
        # 自动检测
        backend = FlashAttentionBackend.detect_best_backend()
        
        # 检查是否支持 FA3
        if FlashAttentionBackend.supports_fa3():
            print("H100 detected, using FA3")
        
        # 获取 GPU 信息
        info = FlashAttentionBackend.get_gpu_info()
    """
    
    # GPU 能力映射
    _GPU_CAPABILITIES = {
        # Hopper (H100/H200) - 完整 FA3 支持
        (9, 0): BackendType.FA3,
        # Ada Lovelace (RTX 4090) - FA2 支持
        (8, 9): BackendType.FA2,
        # Ampere (A100) - FA2 支持
        (8, 0): BackendType.FA2,
        # Ampere (A10/A30) - FA2 支持
        (8, 6): BackendType.FA2,
        # Turing (RTX 20xx) - 标准
        (7, 5): BackendType.STANDARD,
        # Volta (V100) - 标准
        (7, 0): BackendType.STANDARD,
    }
    
    @classmethod
    def detect_best_backend(cls) -> str:
        """
        自动检测最佳后端
        
        Returns:
            后端类型字符串: "fa3" | "fa2" | "standard"
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using standard backend")
            return BackendType.STANDARD.value
        
        if not FLASH_ATTN_AVAILABLE:
            logger.info("FlashAttention not installed, using standard backend")
            return BackendType.STANDARD.value
        
        # 获取 GPU 计算能力
        compute_capability = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        
        logger.info(f"Detected GPU: {gpu_name} (compute capability {compute_capability})")
        
        # 根据计算能力选择后端
        for cap, backend in cls._GPU_CAPABILITIES.items():
            if compute_capability >= cap:
                logger.info(f"Selected backend: {backend.value}")
                return backend.value
        
        # 默认使用标准实现
        logger.info("Using standard backend (GPU not optimized for FlashAttention)")
        return BackendType.STANDARD.value
    
    @classmethod
    def supports_fa3(cls) -> bool:
        """检查是否支持 FlashAttention-3"""
        if not torch.cuda.is_available() or not FLASH_ATTN_AVAILABLE:
            return False
        
        compute_capability = torch.cuda.get_device_capability(0)
        return compute_capability >= (9, 0)
    
    @classmethod
    def supports_fa2(cls) -> bool:
        """检查是否支持 FlashAttention-2"""
        if not torch.cuda.is_available() or not FLASH_ATTN_AVAILABLE:
            return False
        
        compute_capability = torch.cuda.get_device_capability(0)
        return compute_capability >= (8, 0)
    
    @classmethod
    def get_gpu_info(cls) -> Dict[str, Any]:
        """获取 GPU 信息"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        return {
            "available": True,
            "device_id": device,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "multi_processor_count": props.multi_processor_count,
            "flash_attn_available": FLASH_ATTN_AVAILABLE,
            "flash_attn_version": FLASH_ATTN_VERSION,
            "supports_fa3": cls.supports_fa3(),
            "supports_fa2": cls.supports_fa2(),
            "recommended_backend": cls.detect_best_backend(),
        }
    
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """
        验证并返回有效的后端
        
        Args:
            backend: 请求的后端 ("auto" | "fa3" | "fa2" | "standard")
            
        Returns:
            有效的后端类型
        """
        if backend == BackendType.AUTO.value:
            return cls.detect_best_backend()
        
        if backend == BackendType.FA3.value:
            if not cls.supports_fa3():
                logger.warning("FA3 requested but not supported, falling back to FA2/standard")
                return cls.detect_best_backend()
            return backend
        
        if backend == BackendType.FA2.value:
            if not cls.supports_fa2():
                logger.warning("FA2 requested but not supported, falling back to standard")
                return BackendType.STANDARD.value
            return backend
        
        return BackendType.STANDARD.value


def scaled_dot_product_attention_standard(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    标准 Scaled Dot-Product Attention
    
    用于不支持 FlashAttention 的 GPU
    
    Args:
        query: [batch, heads, seq_q, head_dim]
        key: [batch, heads, seq_k, head_dim]
        value: [batch, heads, seq_k, head_dim]
        attn_mask: 可选的注意力掩码
        dropout_p: Dropout 概率
        scale: Softmax scale (默认 1/sqrt(head_dim))
        
    Returns:
        output: [batch, heads, seq_q, head_dim]
    """
    head_dim = query.shape[-1]
    scale = scale or (1.0 / math.sqrt(head_dim))
    
    # 计算注意力分数
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # 应用掩码
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # Softmax
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # Dropout
    if dropout_p > 0.0 and query.requires_grad:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    
    # 计算输出
    output = torch.matmul(attn_weights, value)
    
    return output


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: str = "auto",
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    统一的 FlashAttention 前向接口
    
    根据后端自动选择实现
    
    Args:
        query: [batch, seq_q, heads, head_dim] (FlashAttention 格式)
        key: [batch, seq_k, heads, head_dim]
        value: [batch, seq_k, heads, head_dim]
        backend: 后端类型
        causal: 是否使用 causal masking
        dropout_p: Dropout 概率
        softmax_scale: Softmax scale
        
    Returns:
        output: [batch, seq_q, heads, head_dim]
    """
    # 验证后端
    actual_backend = FlashAttentionBackend.validate_backend(backend)
    
    if actual_backend in (BackendType.FA3.value, BackendType.FA2.value):
        # 使用 FlashAttention
        try:
            output = flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            return output
        except Exception as e:
            logger.warning(f"FlashAttention failed: {e}, falling back to standard")
    
    # 标准实现 (需要转换格式)
    # FlashAttention 格式: [batch, seq, heads, head_dim]
    # 标准格式: [batch, heads, seq, head_dim]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    
    output = scaled_dot_product_attention_standard(
        q, k, v,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    
    # 转换回 FlashAttention 格式
    return output.transpose(1, 2)
