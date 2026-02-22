"""
FlashAttention Configuration
配置管理模块

Author: AGI Demo Project
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml
from loguru import logger


@dataclass
class FA3Config:
    """FlashAttention-3 特定配置 (H100+)"""
    # 是否启用 FP8 精度 (进一步降低内存，但可能影响精度)
    use_fp8: bool = False
    # 是否启用异步执行 (利用 TMA)
    enable_async: bool = True
    # 是否启用 Warp 特化
    enable_warp_specialization: bool = True
    # 软件流水线阶段数
    num_stages: int = 2


@dataclass
class FA2Config:
    """FlashAttention-2 配置 (A100/4090)"""
    # 是否启用 causal masking
    causal: bool = False
    # Dropout 率 (训练时使用，推理时设为 0)
    dropout: float = 0.0
    # Softmax scale (None = 1/sqrt(head_dim))
    softmax_scale: Optional[float] = None
    # 是否返回 softmax 统计 (用于调试)
    return_softmax: bool = False


@dataclass
class KVInjectionConfig:
    """K/V 注入优化配置"""
    # 是否启用优化的 K/V 注入
    enabled: bool = True
    # 注入位置策略: prepend (前置) | interleave (交错)
    strategy: str = "prepend"
    # 是否使用分块注入 (大 K/V 时)
    chunked: bool = True
    # 分块大小
    chunk_size: int = 1024
    # 是否启用 alpha 混合 (软注入)
    alpha_blending: bool = True


@dataclass
class ProfilingConfig:
    """性能监控配置"""
    # 是否启用性能监控
    enabled: bool = False
    # 是否记录内存使用
    log_memory: bool = True
    # 是否记录延迟
    log_latency: bool = True
    # 是否记录 FLOPS
    log_flops: bool = False
    # 日志输出路径
    log_path: Optional[str] = None


@dataclass
class FlashAttentionConfig:
    """
    FlashAttention 完整配置
    
    支持从 YAML 文件或字典加载
    
    Example:
        # 从配置文件加载
        config = FlashAttentionConfig.from_yaml("config/config.yaml")
        
        # 从字典加载
        config = FlashAttentionConfig.from_dict({
            "enabled": True,
            "backend": "auto",
        })
        
        # 直接创建
        config = FlashAttentionConfig(
            enabled=True,
            backend="fa3",
            fa3=FA3Config(use_fp8=True),
        )
    """
    # 是否启用 FlashAttention 优化
    enabled: bool = True
    
    # 后端选择: auto | fa3 | fa2 | standard
    # auto: 自动检测 GPU 并选择最佳后端
    backend: str = "auto"
    
    # FlashAttention-3 配置
    fa3: FA3Config = field(default_factory=FA3Config)
    
    # FlashAttention-2 配置
    fa2: FA2Config = field(default_factory=FA2Config)
    
    # K/V 注入配置
    kv_injection: KVInjectionConfig = field(default_factory=KVInjectionConfig)
    
    # 性能监控配置
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlashAttentionConfig":
        """从字典创建配置"""
        fa3_data = data.get("fa3", {})
        fa2_data = data.get("fa2", {})
        kv_injection_data = data.get("kv_injection", {})
        profiling_data = data.get("profiling", {})
        
        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "auto"),
            fa3=FA3Config(
                use_fp8=fa3_data.get("use_fp8", False),
                enable_async=fa3_data.get("enable_async", True),
                enable_warp_specialization=fa3_data.get("enable_warp_specialization", True),
                num_stages=fa3_data.get("num_stages", 2),
            ),
            fa2=FA2Config(
                causal=fa2_data.get("causal", False),
                dropout=fa2_data.get("dropout", 0.0),
                softmax_scale=fa2_data.get("softmax_scale"),
                return_softmax=fa2_data.get("return_softmax", False),
            ),
            kv_injection=KVInjectionConfig(
                enabled=kv_injection_data.get("enabled", True),
                strategy=kv_injection_data.get("strategy", "prepend"),
                chunked=kv_injection_data.get("chunked", True),
                chunk_size=kv_injection_data.get("chunk_size", 1024),
                alpha_blending=kv_injection_data.get("alpha_blending", True),
            ),
            profiling=ProfilingConfig(
                enabled=profiling_data.get("enabled", False),
                log_memory=profiling_data.get("log_memory", True),
                log_latency=profiling_data.get("log_latency", True),
                log_flops=profiling_data.get("log_flops", False),
                log_path=profiling_data.get("log_path"),
            ),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FlashAttentionConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        flash_attn_data = data.get("flash_attention", {})
        return cls.from_dict(flash_attn_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "fa3": {
                "use_fp8": self.fa3.use_fp8,
                "enable_async": self.fa3.enable_async,
                "enable_warp_specialization": self.fa3.enable_warp_specialization,
                "num_stages": self.fa3.num_stages,
            },
            "fa2": {
                "causal": self.fa2.causal,
                "dropout": self.fa2.dropout,
                "softmax_scale": self.fa2.softmax_scale,
                "return_softmax": self.fa2.return_softmax,
            },
            "kv_injection": {
                "enabled": self.kv_injection.enabled,
                "strategy": self.kv_injection.strategy,
                "chunked": self.kv_injection.chunked,
                "chunk_size": self.kv_injection.chunk_size,
                "alpha_blending": self.kv_injection.alpha_blending,
            },
            "profiling": {
                "enabled": self.profiling.enabled,
                "log_memory": self.profiling.log_memory,
                "log_latency": self.profiling.log_latency,
                "log_flops": self.profiling.log_flops,
                "log_path": self.profiling.log_path,
            },
        }
    
    def __repr__(self) -> str:
        return f"FlashAttentionConfig(enabled={self.enabled}, backend={self.backend})"
