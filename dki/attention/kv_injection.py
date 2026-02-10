"""
K/V Injection Optimizer
优化的 K/V 注入模块

使用 FlashAttention 的分块计算特性，
高效地将预计算的 K/V 注入到注意力计算中

Author: AGI Demo Project
Version: 1.0.0
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.attention.config import FlashAttentionConfig, KVInjectionConfig
from dki.attention.backend import (
    FlashAttentionBackend,
    flash_attention_forward,
    scaled_dot_product_attention_standard,
    FLASH_ATTN_AVAILABLE,
)


@dataclass
class InjectionResult:
    """K/V 注入结果"""
    output: torch.Tensor
    latency_ms: float
    memory_key_len: int
    input_key_len: int
    total_key_len: int
    backend_used: str
    alpha: float


class KVInjectionOptimizer:
    """
    优化的 K/V 注入器
    
    核心功能:
    1. 将预计算的 memory K/V 与 input K/V 高效拼接
    2. 使用 FlashAttention 进行优化的注意力计算
    3. 支持 alpha 混合 (软注入)
    4. 支持分块处理 (大 K/V)
    
    DKI 注入公式:
        Attention(Q, [K_mem; K], [V_mem; V]) = softmax(Q[K_mem; K]^T / √d) [V_mem; V]
    
    Alpha 混合:
        output = α * injected_output + (1-α) * original_output
    
    Example:
        optimizer = KVInjectionOptimizer()
        
        result = optimizer.inject(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            memory_key=memory_k,
            memory_value=memory_v,
            alpha=0.5,
        )
    """
    
    def __init__(
        self,
        config: Optional[FlashAttentionConfig] = None,
        backend: Optional[str] = None,
    ):
        """
        初始化 K/V 注入优化器
        
        Args:
            config: FlashAttention 配置
            backend: 后端类型 (覆盖配置)
        """
        self.config = config or FlashAttentionConfig()
        
        # 确定后端
        if backend:
            self._backend = FlashAttentionBackend.validate_backend(backend)
        elif self.config.backend == "auto":
            self._backend = FlashAttentionBackend.detect_best_backend()
        else:
            self._backend = FlashAttentionBackend.validate_backend(self.config.backend)
        
        # 统计
        self._stats = {
            "total_injections": 0,
            "total_latency_ms": 0.0,
            "avg_memory_len": 0.0,
            "backend_usage": {
                "fa3": 0,
                "fa2": 0,
                "standard": 0,
            },
        }
        
        logger.info(f"KVInjectionOptimizer initialized (backend={self._backend})")
    
    @property
    def backend(self) -> str:
        """获取当前后端"""
        return self._backend
    
    def inject(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        alpha: float = 1.0,
        causal: bool = False,
    ) -> InjectionResult:
        """
        执行 K/V 注入
        
        Args:
            query: 查询张量 [batch, seq_q, heads, head_dim]
            key: 输入键张量 [batch, seq_k, heads, head_dim]
            value: 输入值张量 [batch, seq_k, heads, head_dim]
            memory_key: 记忆键张量 [batch, seq_m, heads, head_dim]
            memory_value: 记忆值张量 [batch, seq_m, heads, head_dim]
            alpha: 注入强度 (0-1)
            causal: 是否使用 causal masking
            
        Returns:
            InjectionResult 包含输出和元数据
        """
        start_time = time.perf_counter()
        
        # 获取维度信息
        batch_size, seq_q, num_heads, head_dim = query.shape
        seq_k = key.shape[1]
        seq_m = memory_key.shape[1]
        
        # 确保设备一致
        device = query.device
        memory_key = memory_key.to(device)
        memory_value = memory_value.to(device)
        
        # 根据策略执行注入
        if self.config.kv_injection.strategy == "prepend":
            output = self._inject_prepend(
                query, key, value,
                memory_key, memory_value,
                alpha, causal,
            )
        else:
            # interleave 策略 (未来扩展)
            output = self._inject_prepend(
                query, key, value,
                memory_key, memory_value,
                alpha, causal,
            )
        
        # 计算延迟
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 更新统计
        self._update_stats(seq_m, latency_ms)
        
        return InjectionResult(
            output=output,
            latency_ms=latency_ms,
            memory_key_len=seq_m,
            input_key_len=seq_k,
            total_key_len=seq_m + seq_k,
            backend_used=self._backend,
            alpha=alpha,
        )
    
    def _inject_prepend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        alpha: float,
        causal: bool,
    ) -> torch.Tensor:
        """
        前置注入策略
        
        将 memory K/V 拼接到 input K/V 之前
        """
        # 拼接 K/V
        full_key = torch.cat([memory_key, key], dim=1)
        full_value = torch.cat([memory_value, value], dim=1)
        
        # 使用 FlashAttention 计算
        injected_output = flash_attention_forward(
            query=query,
            key=full_key,
            value=full_value,
            backend=self._backend,
            causal=causal,
            dropout_p=self.config.fa2.dropout,
            softmax_scale=self.config.fa2.softmax_scale,
        )
        
        # Alpha 混合
        if alpha < 1.0 and self.config.kv_injection.alpha_blending:
            # 计算原始输出 (无注入)
            original_output = flash_attention_forward(
                query=query,
                key=key,
                value=value,
                backend=self._backend,
                causal=causal,
                dropout_p=self.config.fa2.dropout,
                softmax_scale=self.config.fa2.softmax_scale,
            )
            
            # 混合
            output = alpha * injected_output + (1 - alpha) * original_output
        else:
            output = injected_output
        
        return output
    
    def inject_chunked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        alpha: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> InjectionResult:
        """
        分块 K/V 注入
        
        用于处理非常大的 memory K/V
        
        Args:
            query: 查询张量
            key: 输入键张量
            value: 输入值张量
            memory_key: 记忆键张量 (可能很大)
            memory_value: 记忆值张量
            alpha: 注入强度
            chunk_size: 分块大小 (默认从配置读取)
            
        Returns:
            InjectionResult
        """
        start_time = time.perf_counter()
        
        chunk_size = chunk_size or self.config.kv_injection.chunk_size
        seq_m = memory_key.shape[1]
        
        # 如果 memory 不大，直接使用普通注入
        if seq_m <= chunk_size:
            return self.inject(
                query, key, value,
                memory_key, memory_value,
                alpha,
            )
        
        # 分块处理
        num_chunks = (seq_m + chunk_size - 1) // chunk_size
        outputs = []
        weights = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_m)
            
            chunk_key = memory_key[:, start_idx:end_idx, :, :]
            chunk_value = memory_value[:, start_idx:end_idx, :, :]
            
            # 计算这个 chunk 的注意力
            chunk_output = self._inject_prepend(
                query, key, value,
                chunk_key, chunk_value,
                alpha=1.0,  # 先不混合
                causal=False,
            )
            
            outputs.append(chunk_output)
            weights.append(end_idx - start_idx)
        
        # 加权平均
        total_weight = sum(weights)
        output = sum(
            w / total_weight * o
            for w, o in zip(weights, outputs)
        )
        
        # Alpha 混合
        if alpha < 1.0 and self.config.kv_injection.alpha_blending:
            original_output = flash_attention_forward(
                query=query,
                key=key,
                value=value,
                backend=self._backend,
            )
            output = alpha * output + (1 - alpha) * original_output
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(seq_m, latency_ms)
        
        return InjectionResult(
            output=output,
            latency_ms=latency_ms,
            memory_key_len=seq_m,
            input_key_len=key.shape[1],
            total_key_len=seq_m + key.shape[1],
            backend_used=self._backend,
            alpha=alpha,
        )
    
    def inject_multi_layer(
        self,
        queries: List[torch.Tensor],
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        memory_keys: List[torch.Tensor],
        memory_values: List[torch.Tensor],
        alpha: float = 1.0,
        layer_alphas: Optional[List[float]] = None,
    ) -> List[InjectionResult]:
        """
        多层 K/V 注入
        
        用于 Transformer 的多层注入
        
        Args:
            queries: 每层的查询张量列表
            keys: 每层的键张量列表
            values: 每层的值张量列表
            memory_keys: 每层的记忆键张量列表
            memory_values: 每层的记忆值张量列表
            alpha: 全局注入强度
            layer_alphas: 每层的注入强度 (可选，覆盖全局)
            
        Returns:
            每层的 InjectionResult 列表
        """
        num_layers = len(queries)
        
        if layer_alphas is None:
            layer_alphas = [alpha] * num_layers
        
        results = []
        for i in range(num_layers):
            result = self.inject(
                query=queries[i],
                key=keys[i],
                value=values[i],
                memory_key=memory_keys[i],
                memory_value=memory_values[i],
                alpha=layer_alphas[i],
            )
            results.append(result)
        
        return results
    
    def _update_stats(self, memory_len: int, latency_ms: float):
        """更新统计信息"""
        self._stats["total_injections"] += 1
        self._stats["total_latency_ms"] += latency_ms
        
        # 更新平均 memory 长度
        n = self._stats["total_injections"]
        old_avg = self._stats["avg_memory_len"]
        self._stats["avg_memory_len"] = old_avg + (memory_len - old_avg) / n
        
        # 更新后端使用统计
        self._stats["backend_usage"][self._backend] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["total_injections"]
        
        return {
            "total_injections": total,
            "total_latency_ms": self._stats["total_latency_ms"],
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / total if total > 0 else 0
            ),
            "avg_memory_len": self._stats["avg_memory_len"],
            "backend": self._backend,
            "backend_usage": self._stats["backend_usage"],
            "config": {
                "strategy": self.config.kv_injection.strategy,
                "chunked": self.config.kv_injection.chunked,
                "chunk_size": self.config.kv_injection.chunk_size,
                "alpha_blending": self.config.kv_injection.alpha_blending,
            },
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            "total_injections": 0,
            "total_latency_ms": 0.0,
            "avg_memory_len": 0.0,
            "backend_usage": {
                "fa3": 0,
                "fa2": 0,
                "standard": 0,
            },
        }
