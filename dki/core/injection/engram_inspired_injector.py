"""
Engram-Inspired Full Attention Injector - Enhanced Plan C

借鉴 Engram 论文 (Conditional Memory via Scalable Lookup, arXiv:2601.07372)
的记忆处理机制，增强 DKI Full-Attention 注入策略。

核心借鉴点 (从记忆处理视角):
1. Context-Aware Per-Token Gating (Engram Eq. 4)
   - 不再使用全局统一 α，而是每个记忆 token 根据与当前上下文的对齐程度动态门控
   - α_t = σ(RMSNorm(h_query)^T · RMSNorm(k_mem_t) / √d)
   
2. Value-Only Gating (Engram 设计原则)
   - Key 不缩放 (作为注意力寻址地址)
   - 仅对 Value 施加门控 (调制输出贡献)
   
3. Depthwise Causal Convolution Refinement (Engram Eq. 5)
   - 对门控后的 Value 进行轻量卷积平滑
   - Y = SiLU(Conv1D(RMSNorm(V̄))) + V̄
   - 扩展记忆 token 的感受野，捕获相邻 token 的模式
   
4. Layer-Selective Injection Policy (Engram Section 2.5 + 6.2)
   - 不在所有层注入，而是选择性注入
   - 早期层: 偏好注入 (事实性基底)
   - 中期层: 历史注入 (推理上下文)

理论基础:
- Engram 证明了: 将静态模式存储从动态计算中结构性分离，可以有效增加模型的
  "有效深度" (effective depth)，释放注意力容量用于全局推理
- DKI 的用户级记忆虽非模型级通识，但从记忆处理角度看，偏好 (稳定、重复)
  类比于 Engram 的 N-gram 静态模式，历史 (动态、上下文相关) 类比于需要
  注意力处理的全局上下文
- 因此，Engram 的 "静态记忆用 lookup/gating 卸载，释放注意力容量"
  策略可以映射为 "用户偏好用 gated K/V 注入，释放 context 窗口给推理"

Author: AGI Demo Project
Version: 2.0.0
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

class GatingMode(Enum):
    """门控模式"""
    UNIFORM = "uniform"              # 原始 DKI: 全局统一 α
    CONTEXT_AWARE = "context_aware"  # Engram-inspired: 逐 token 上下文门控
    HYBRID = "hybrid"                # 混合: 全局 α × 逐 token 门控


class RefinementMode(Enum):
    """Value 精化模式"""
    NONE = "none"                    # 不精化
    CONV1D = "conv1d"                # Engram-inspired: 因果卷积 + SiLU
    LINEAR = "linear"               # 轻量线性映射


@dataclass
class LayerInjectionPolicy:
    """
    层级注入策略 (借鉴 Engram Section 2.5 + 6.2)
    
    Engram 的关键发现:
    - 早期层从记忆注入中获益最多 (卸载静态模式重建)
    - 并非所有层都需要注入 (选择性注入更高效)
    - 注入位置需平衡建模性能和系统延迟
    
    DKI 映射:
    - 偏好 (类 Engram 静态模式): 注入早期层
    - 历史 (类全局上下文): 注入中期层
    """
    # 注入层索引 (相对于模型总层数的比例)
    preference_layer_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2]  # 前 20% 的层
    )
    history_layer_ratios: List[float] = field(
        default_factory=lambda: [0.3, 0.5]  # 30%-50% 的层
    )
    
    # 是否为每层使用独立的门控参数
    per_layer_gating: bool = True
    
    def get_preference_layers(self, num_layers: int) -> List[int]:
        """获取偏好注入层"""
        return sorted(set(
            max(0, min(num_layers - 1, int(r * num_layers)))
            for r in self.preference_layer_ratios
        ))
    
    def get_history_layers(self, num_layers: int) -> List[int]:
        """获取历史注入层"""
        return sorted(set(
            max(0, min(num_layers - 1, int(r * num_layers)))
            for r in self.history_layer_ratios
        ))
    
    def get_all_injection_layers(self, num_layers: int) -> List[int]:
        """获取所有注入层 (用于预分配资源)"""
        return sorted(set(
            self.get_preference_layers(num_layers) +
            self.get_history_layers(num_layers)
        ))


@dataclass
class EngramInspiredConfig:
    """Engram-Inspired 注入器配置"""
    enabled: bool = True
    
    # === 门控配置 (借鉴 Engram Eq. 4) ===
    gating_mode: GatingMode = GatingMode.CONTEXT_AWARE
    gating_temperature: float = 1.0  # 门控温度 (√d 的缩放因子)
    gating_bias: float = 0.0         # 门控偏置 (正值倾向开放门控)
    use_rmsnorm: bool = True         # 是否使用 RMSNorm (Engram 推荐)
    
    # === Value 精化配置 (借鉴 Engram Eq. 5) ===
    refinement_mode: RefinementMode = RefinementMode.CONV1D
    conv_kernel_size: int = 4        # 卷积核大小 (Engram 默认 4)
    conv_dilation: int = 1           # 膨胀率
    
    # === 层级注入策略 (借鉴 Engram Section 2.5) ===
    layer_policy: LayerInjectionPolicy = field(
        default_factory=LayerInjectionPolicy
    )
    
    # === 偏好 K/V 配置 ===
    preference_position_start: int = -100
    preference_base_alpha: float = 0.4   # 基础 α (context-aware 模式下作为上限)
    preference_max_tokens: int = 100
    
    # === 历史 K/V 配置 ===
    history_position_start: int = -500
    history_base_alpha: float = 0.3
    history_max_tokens: int = 400
    history_max_messages: int = 10
    
    # === 全局指示 ===
    global_indication_enabled: bool = True
    global_indication_cn: str = "[记忆上下文可用]"
    global_indication_en: str = "[Memory Context Available]"
    
    # === 安全设置 ===
    max_total_kv_tokens: int = 600
    fallback_to_stable: bool = True
    log_attention_patterns: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EngramInspiredConfig":
        """从字典创建配置"""
        gating_mode = config.get("gating_mode", "context_aware")
        if isinstance(gating_mode, str):
            gating_mode = GatingMode(gating_mode)
        
        refinement_mode = config.get("refinement_mode", "conv1d")
        if isinstance(refinement_mode, str):
            refinement_mode = RefinementMode(refinement_mode)
        
        layer_policy_dict = config.get("layer_policy", {})
        layer_policy = LayerInjectionPolicy(
            preference_layer_ratios=layer_policy_dict.get(
                "preference_layer_ratios", [0.0, 0.1, 0.2]
            ),
            history_layer_ratios=layer_policy_dict.get(
                "history_layer_ratios", [0.3, 0.5]
            ),
            per_layer_gating=layer_policy_dict.get("per_layer_gating", True),
        )
        
        return cls(
            enabled=config.get("enabled", True),
            gating_mode=gating_mode,
            gating_temperature=config.get("gating_temperature", 1.0),
            gating_bias=config.get("gating_bias", 0.0),
            use_rmsnorm=config.get("use_rmsnorm", True),
            refinement_mode=refinement_mode,
            conv_kernel_size=config.get("conv_kernel_size", 4),
            conv_dilation=config.get("conv_dilation", 1),
            layer_policy=layer_policy,
            preference_position_start=config.get(
                "preference", {}
            ).get("position_start", -100),
            preference_base_alpha=config.get(
                "preference", {}
            ).get("base_alpha", 0.4),
            preference_max_tokens=config.get(
                "preference", {}
            ).get("max_tokens", 100),
            history_position_start=config.get(
                "history", {}
            ).get("position_start", -500),
            history_base_alpha=config.get(
                "history", {}
            ).get("base_alpha", 0.3),
            history_max_tokens=config.get(
                "history", {}
            ).get("max_tokens", 400),
            history_max_messages=config.get(
                "history", {}
            ).get("max_messages", 10),
            max_total_kv_tokens=config.get(
                "safety", {}
            ).get("max_total_kv_tokens", 600),
            fallback_to_stable=config.get(
                "safety", {}
            ).get("fallback_to_stable", True),
        )


# =============================================================================
# Core Modules (Engram-Inspired)
# =============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    
    Engram 在门控计算中使用 RMSNorm 而非 LayerNorm:
    - 计算更轻量 (无需减均值)
    - 梯度更稳定
    - 与 DeepSeek 等模型架构一致
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ContextAwareGating(nn.Module):
    """
    Context-Aware Per-Token Gating (借鉴 Engram Eq. 4)
    
    Engram 原文:
        α_t = σ(RMSNorm(h_t)^T · RMSNorm(k_t) / √d)
    
    DKI 适配:
        对于每个记忆 token m_i，计算其与当前查询上下文的对齐门控:
        α_i = σ(RMSNorm(h_query) · RMSNorm(K_mem[i]) / (√d_head × temperature) + bias)
    
    设计考量:
    - h_query: 查询隐藏状态的池化表示 (mean pooling over query tokens)
    - K_mem[i]: 第 i 个记忆 token 的 Key 向量
    - 输出: 每个记忆 token 独立的门控标量 α_i ∈ (0, 1)
    
    为什么这比 DKI 原始的统一 α 更好:
    1. 偏好中 "用户喜欢辣食" 的 "辣" 在讨论食物时 α 高，讨论天气时 α 低
    2. 历史中最近一轮的 token 通常比更早的 token 获得更高的 α
    3. 无关 token (如格式符号) 自动被抑制
    """
    
    def __init__(
        self,
        head_dim: int,
        temperature: float = 1.0,
        bias: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.temperature = temperature
        self.bias = bias
        self.use_rmsnorm = use_rmsnorm
        
        if use_rmsnorm:
            self.query_norm = RMSNorm(head_dim)
            self.key_norm = RMSNorm(head_dim)
        
        self.scale = 1.0 / (math.sqrt(head_dim) * temperature)
    
    def forward(
        self,
        query_hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        base_alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        计算逐 token 门控值
        
        Args:
            query_hidden: 查询隐藏状态 [batch, num_heads, head_dim]
                          (query tokens 的池化表示)
            memory_keys: 记忆 Key 张量 [batch, num_heads, mem_len, head_dim]
            base_alpha: 基础 α (作为门控上限)
            
        Returns:
            gates: 门控值 [batch, num_heads, mem_len, 1]
                   每个记忆 token 一个标量门控
        """
        # Normalize
        if self.use_rmsnorm:
            q = self.query_norm(query_hidden)  # [batch, heads, head_dim]
            k = self.key_norm(memory_keys)     # [batch, heads, mem_len, head_dim]
        else:
            q = query_hidden
            k = memory_keys
        
        # Compute alignment scores
        # q: [batch, heads, 1, head_dim] (unsqueeze for broadcasting)
        # k: [batch, heads, mem_len, head_dim]
        q = q.unsqueeze(2)  # [batch, heads, 1, head_dim]
        
        # Dot product: [batch, heads, 1, head_dim] × [batch, heads, mem_len, head_dim]^T
        # → [batch, heads, 1, mem_len] → squeeze → [batch, heads, mem_len]
        scores = torch.sum(q * k, dim=-1)  # [batch, heads, mem_len]
        scores = scores * self.scale + self.bias
        
        # Sigmoid gate (Engram uses σ)
        gates = torch.sigmoid(scores)  # [batch, heads, mem_len]
        
        # Scale by base_alpha (DKI's global control)
        gates = gates * base_alpha
        
        # Add trailing dimension for broadcasting with Values
        return gates.unsqueeze(-1)  # [batch, heads, mem_len, 1]


class ConvolutionRefinement(nn.Module):
    """
    Depthwise Causal Convolution Refinement (借鉴 Engram Eq. 5)
    
    Engram 原文:
        Y = SiLU(Conv1D(RMSNorm(V̄))) + V̄
    
    设计考量:
    - Causal Conv1D: 确保每个位置只看到之前的记忆 token (维护因果性)
    - Depthwise: 每个头独立卷积，不跨头混合
    - SiLU 激活: 平滑非线性，优于 ReLU
    - 残差连接: 初始时卷积权重为 0，确保 Y ≈ V̄ (恒等映射)
    
    为什么需要:
    - 相邻的记忆 token 之间存在局部依赖 (e.g., "用户" "喜欢" "辣食")
    - 单独的 K/V token 无法捕获这些 n-gram 模式
    - 轻量卷积以最小开销扩展感受野
    """
    
    def __init__(
        self,
        head_dim: int,
        kernel_size: int = 4,
        dilation: int = 1,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # RMSNorm before convolution
        self.norm = RMSNorm(head_dim)
        
        # Depthwise causal convolution
        # padding = (kernel_size - 1) * dilation for causal (left padding)
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=head_dim,  # Depthwise
            bias=True,
        )
        
        # Initialize to zero for identity mapping at start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, v_gated: torch.Tensor) -> torch.Tensor:
        """
        精化门控后的 Value
        
        Args:
            v_gated: 门控后的 Value [batch, num_heads, mem_len, head_dim]
            
        Returns:
            v_refined: 精化后的 Value [batch, num_heads, mem_len, head_dim]
        """
        batch, heads, mem_len, dim = v_gated.shape
        
        if mem_len <= 1:
            return v_gated
        
        # Apply RMSNorm
        v_normed = self.norm(v_gated)  # [batch, heads, mem_len, dim]
        
        # Reshape for Conv1D: [batch * heads, dim, mem_len]
        v_conv = v_normed.reshape(batch * heads, mem_len, dim)
        v_conv = v_conv.transpose(1, 2)  # [batch * heads, dim, mem_len]
        
        # Causal left-padding
        v_conv = F.pad(v_conv, (self.causal_padding, 0))
        
        # Apply convolution
        v_conv = self.conv(v_conv)  # [batch * heads, dim, mem_len]
        
        # SiLU activation
        v_conv = F.silu(v_conv)
        
        # Reshape back
        v_conv = v_conv.transpose(1, 2)  # [batch * heads, mem_len, dim]
        v_conv = v_conv.reshape(batch, heads, mem_len, dim)
        
        # Residual connection (Engram Eq. 5)
        return v_conv + v_gated


# =============================================================================
# Injection Result
# =============================================================================

@dataclass
class EngramInjectionResult:
    """注入结果 (增强版)"""
    success: bool = False
    
    # K/V 数据 (per-layer)
    # Dict[layer_idx, (K_mem, V_gated_refined)]
    layer_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    
    # 门控统计
    per_token_gates: Optional[Dict[str, torch.Tensor]] = None  # 记录门控值用于可视化
    avg_preference_gate: float = 0.0
    avg_history_gate: float = 0.0
    
    # Token 统计
    preference_tokens: int = 0
    history_tokens: int = 0
    total_kv_tokens: int = 0
    
    # 层级策略
    preference_layers: List[int] = field(default_factory=list)
    history_layers: List[int] = field(default_factory=list)
    
    # 全局指示
    global_indication: str = ""
    
    # 元数据
    gating_mode: str = ""
    refinement_mode: str = ""
    fallback_triggered: bool = False
    error_message: str = ""
    
    # 性能
    compute_time_ms: float = 0.0
    gating_time_ms: float = 0.0
    refinement_time_ms: float = 0.0


# =============================================================================
# Main Injector
# =============================================================================

class EngramInspiredFullAttentionInjector:
    """
    Engram-Inspired Full Attention 注入器
    
    在 DKI Full-Attention 方案 C 的基础上，融入 Engram 论文的记忆处理智慧:
    
    ┌───────────────────────────────────────────────────────────────────┐
    │  Engram-Inspired DKI Full-Attention 注入流程                      │
    ├───────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │  1. 记忆收集                                                      │
    │     ├─ 用户偏好 → tokenize → model.compute_kv() → K_pref, V_pref │
    │     └─ 会话历史 → tokenize → model.compute_kv() → K_hist, V_hist │
    │                                                                   │
    │  2. 上下文感知门控 (Engram Eq. 4)                                  │
    │     ├─ 查询表示: h_q = mean_pool(model.embed(query))              │
    │     ├─ 偏好门控: α_i^pref = σ(norm(h_q)·norm(K_pref[i])/√d)     │
    │     ├─ 历史门控: α_j^hist = σ(norm(h_q)·norm(K_hist[j])/√d)     │
    │     └─ Value 门控: V_pref'[i] = α_i^pref · V_pref[i]            │
    │                    V_hist'[j] = α_j^hist · V_hist[j]             │
    │                                                                   │
    │  3. 卷积精化 (Engram Eq. 5)                                       │
    │     ├─ V_pref'' = SiLU(Conv1D(RMSNorm(V_pref'))) + V_pref'      │
    │     └─ V_hist'' = SiLU(Conv1D(RMSNorm(V_hist'))) + V_hist'      │
    │                                                                   │
    │  4. 层级选择性注入 (Engram Section 2.5)                            │
    │     ├─ 早期层 (e.g., L0, L3, L6): 注入偏好 K/V                   │
    │     │   [K_pref; K_input] × [V_pref''; V_input]                  │
    │     ├─ 中期层 (e.g., L9, L15): 注入历史 K/V                      │
    │     │   [K_hist; K_input] × [V_hist''; V_input]                  │
    │     └─ 其余层: 原始注意力 (无注入)                                 │
    │                                                                   │
    │  5. 注意力计算                                                     │
    │     ├─ Attn_l = softmax(Q_l · [K_mem; K_input]^T / √d)           │
    │     │           × [V_gated_refined; V_input]                      │
    │     └─ 注意: Key 不缩放 (保持寻址准确性)                          │
    │           Value 已通过门控 + 精化调制                               │
    │                                                                   │
    │  6. 残差整合                                                       │
    │     H^(l) ← H^(l) + Attn_l (标准残差连接)                        │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
    
    与原始 Full-Attention 的关键区别:
    ┌────────────────────┬──────────────────┬──────────────────────────┐
    │  维度              │  原始 DKI         │  Engram-Inspired DKI     │
    ├────────────────────┼──────────────────┼──────────────────────────┤
    │  α 缩放粒度        │  全局统一         │  逐 token 上下文门控      │
    │  α 作用对象        │  K 和 V 都缩放    │  仅缩放 V (Key 不变)     │
    │  Value 处理        │  原始直出         │  卷积精化 + 残差          │
    │  注入层策略        │  所有层           │  层级选择性 (早期+中期)   │
    │  门控归一化        │  无               │  RMSNorm (梯度稳定)      │
    │  感受野            │  单 token          │  扩展 (Conv kernel=4)    │
    └────────────────────┴──────────────────┴──────────────────────────┘
    """
    
    def __init__(
        self,
        config: Optional[EngramInspiredConfig] = None,
        language: str = "en",
        head_dim: int = 128,
        num_heads: int = 32,
        num_layers: int = 30,
    ):
        """
        初始化 Engram-Inspired Full Attention 注入器
        
        Args:
            config: 配置
            language: 语言 ("en" | "cn")
            head_dim: 注意力头维度 (通常 hidden_dim / num_heads)
            num_heads: 注意力头数
            num_layers: 模型总层数
        """
        self.config = config or EngramInspiredConfig()
        self.language = language
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # === 初始化 Engram-Inspired 模块 ===
        
        # Context-Aware Gating (Engram Eq. 4)
        self.gating: Optional[ContextAwareGating] = None
        if self.config.gating_mode in (GatingMode.CONTEXT_AWARE, GatingMode.HYBRID):
            self.gating = ContextAwareGating(
                head_dim=head_dim,
                temperature=self.config.gating_temperature,
                bias=self.config.gating_bias,
                use_rmsnorm=self.config.use_rmsnorm,
            )
            logger.info(
                f"Context-Aware Gating initialized "
                f"(head_dim={head_dim}, temperature={self.config.gating_temperature})"
            )
        
        # Convolution Refinement (Engram Eq. 5)
        self.refinement: Optional[ConvolutionRefinement] = None
        if self.config.refinement_mode == RefinementMode.CONV1D:
            self.refinement = ConvolutionRefinement(
                head_dim=head_dim,
                kernel_size=self.config.conv_kernel_size,
                dilation=self.config.conv_dilation,
            )
            logger.info(
                f"Conv1D Refinement initialized "
                f"(kernel={self.config.conv_kernel_size}, "
                f"dilation={self.config.conv_dilation})"
            )
        
        # Layer injection policy
        self._pref_layers = self.config.layer_policy.get_preference_layers(num_layers)
        self._hist_layers = self.config.layer_policy.get_history_layers(num_layers)
        self._all_layers = self.config.layer_policy.get_all_injection_layers(num_layers)
        
        # 统计数据
        self._stats = {
            "total_injections": 0,
            "successful_injections": 0,
            "fallback_count": 0,
            "avg_preference_gate": 0.0,
            "avg_history_gate": 0.0,
            "avg_compute_time_ms": 0.0,
        }
        
        # Attention pattern 日志
        self._attention_logs: List[Dict[str, Any]] = []
        self._max_attention_logs = 100
        
        logger.info(
            f"EngramInspiredFullAttentionInjector initialized "
            f"(gating={self.config.gating_mode.value}, "
            f"refinement={self.config.refinement_mode.value}, "
            f"pref_layers={self._pref_layers}, "
            f"hist_layers={self._hist_layers})"
        )
    
    def inject(
        self,
        model_adapter: Any,
        preference_text: str,
        history_messages: List[Dict[str, str]],
        query: str,
        query_hidden: Optional[torch.Tensor] = None,
    ) -> EngramInjectionResult:
        """
        执行 Engram-Inspired Full Attention 注入
        
        完整注入流程:
        
        Phase 1: 记忆收集
          - 用户偏好文本 → compute_kv → K_pref, V_pref
          - 历史消息文本 → compute_kv → K_hist, V_hist
        
        Phase 2: 上下文感知门控 (Engram-inspired)
          - 使用查询隐藏状态作为上下文信号
          - 逐 token 计算对齐门控 α_i
          - 仅对 Value 施加门控
        
        Phase 3: 卷积精化 (Engram-inspired)
          - 对门控后的 Value 进行因果卷积
          - 残差连接保持信息完整性
        
        Phase 4: 层级组装
          - 按照注入策略分配到指定层
          - 返回 per-layer K/V 数据
        
        Args:
            model_adapter: 模型适配器
            preference_text: 用户偏好文本
            history_messages: 历史消息列表
            query: 用户查询
            query_hidden: 查询隐藏状态 (可选，用于 context-aware gating)
            
        Returns:
            EngramInjectionResult
        """
        start_time = time.time()
        result = EngramInjectionResult()
        result.gating_mode = self.config.gating_mode.value
        result.refinement_mode = self.config.refinement_mode.value
        result.preference_layers = list(self._pref_layers)
        result.history_layers = list(self._hist_layers)
        
        self._stats["total_injections"] += 1
        
        try:
            # ============ Phase 1: 记忆收集 ============
            preference_tokens = self._estimate_tokens(preference_text)
            history_text = self._format_history(history_messages)
            history_tokens = self._estimate_tokens(history_text)
            total_tokens = preference_tokens + history_tokens
            
            result.preference_tokens = preference_tokens
            result.history_tokens = history_tokens
            result.total_kv_tokens = total_tokens
            
            # 安全检查
            if total_tokens > self.config.max_total_kv_tokens:
                if self.config.fallback_to_stable:
                    result.fallback_triggered = True
                    result.error_message = (
                        f"Total K/V tokens ({total_tokens}) exceeds limit "
                        f"({self.config.max_total_kv_tokens}), fallback"
                    )
                    self._stats["fallback_count"] += 1
                    logger.warning(result.error_message)
                    return result
                else:
                    history_messages = self._truncate_history(
                        history_messages,
                        self.config.max_total_kv_tokens - preference_tokens,
                    )
                    history_text = self._format_history(history_messages)
                    history_tokens = self._estimate_tokens(history_text)
            
            # 计算 K/V
            preference_kv = None
            if preference_text and preference_tokens > 0:
                preference_kv, _ = model_adapter.compute_kv(preference_text)
            
            history_kv = None
            if history_text and history_tokens > 0:
                history_kv, _ = model_adapter.compute_kv(history_text)
            
            if not preference_kv and not history_kv:
                result.success = True
                result.compute_time_ms = (time.time() - start_time) * 1000
                return result
            
            # ============ Phase 2: 上下文感知门控 ============
            gating_start = time.time()
            
            # 获取查询隐藏状态 (用于 context-aware gating)
            if query_hidden is None and self.config.gating_mode != GatingMode.UNIFORM:
                query_hidden = self._compute_query_representation(
                    model_adapter, query
                )
            
            # 对偏好 K/V 应用门控
            gated_pref_kv = None
            if preference_kv:
                gated_pref_kv = self._apply_gating(
                    kv_entries=preference_kv,
                    query_hidden=query_hidden,
                    base_alpha=self.config.preference_base_alpha,
                    memory_type="preference",
                    result=result,
                )
            
            # 对历史 K/V 应用门控
            gated_hist_kv = None
            if history_kv:
                gated_hist_kv = self._apply_gating(
                    kv_entries=history_kv,
                    query_hidden=query_hidden,
                    base_alpha=self.config.history_base_alpha,
                    memory_type="history",
                    result=result,
                )
            
            result.gating_time_ms = (time.time() - gating_start) * 1000
            
            # ============ Phase 3: 卷积精化 ============
            refinement_start = time.time()
            
            if gated_pref_kv:
                gated_pref_kv = self._apply_refinement(gated_pref_kv)
            
            if gated_hist_kv:
                gated_hist_kv = self._apply_refinement(gated_hist_kv)
            
            result.refinement_time_ms = (time.time() - refinement_start) * 1000
            
            # ============ Phase 4: 层级组装 ============
            layer_kv = self._assemble_layer_kv(
                preference_kv=gated_pref_kv,
                history_kv=gated_hist_kv,
            )
            
            result.layer_kv = layer_kv
            
            # ============ 全局指示 ============
            if self.config.global_indication_enabled:
                result.global_indication = (
                    self.config.global_indication_cn if self.language == "cn"
                    else self.config.global_indication_en
                )
            
            # ============ 完成 ============
            result.success = True
            result.compute_time_ms = (time.time() - start_time) * 1000
            
            self._stats["successful_injections"] += 1
            self._update_stats(result)
            
            if self.config.log_attention_patterns:
                self._log_injection(result, query)
            
            logger.debug(
                f"Engram-inspired injection: "
                f"pref={preference_tokens}tok (gate={result.avg_preference_gate:.3f}), "
                f"hist={history_tokens}tok (gate={result.avg_history_gate:.3f}), "
                f"layers={len(layer_kv)}, "
                f"time={result.compute_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.compute_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Engram-inspired injection failed: {e}")
            return result
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _compute_query_representation(
        self,
        model_adapter: Any,
        query: str,
    ) -> Optional[torch.Tensor]:
        """
        计算查询的隐藏状态表示 (用于 context-aware gating)
        
        使用模型的 embedding 层获取查询的向量表示，
        然后 mean-pool 为单个向量。
        
        Returns:
            query_hidden: [1, num_heads, head_dim] (池化后的查询表示)
        """
        try:
            if hasattr(model_adapter, 'embed_text'):
                # 使用适配器的 embed 方法
                embedding = model_adapter.embed_text(query)  # [1, seq_len, hidden_dim]
                # Mean pool over sequence
                pooled = embedding.mean(dim=1)  # [1, hidden_dim]
                # Reshape to [1, num_heads, head_dim]
                return pooled.view(1, self.num_heads, self.head_dim)
            elif hasattr(model_adapter, 'compute_kv'):
                # Fallback: 使用 compute_kv 获取 Key 然后 pool
                kv_entries, _ = model_adapter.compute_kv(query)
                if kv_entries:
                    # 取第一层的 Key 并 pool
                    first_key = kv_entries[0].key  # [1, heads, seq, head_dim]
                    return first_key.mean(dim=2)   # [1, heads, head_dim]
            
            return None
        except Exception as e:
            logger.debug(f"Failed to compute query representation: {e}")
            return None
    
    def _apply_gating(
        self,
        kv_entries: List,
        query_hidden: Optional[torch.Tensor],
        base_alpha: float,
        memory_type: str,
        result: EngramInjectionResult,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        应用上下文感知门控
        
        Args:
            kv_entries: K/V 条目列表 (per-layer)
            query_hidden: 查询隐藏表示
            base_alpha: 基础 α
            memory_type: "preference" or "history"
            result: 结果对象 (用于记录统计)
            
        Returns:
            gated_kv: List of (K_unchanged, V_gated) per layer
        """
        gated_kv = []
        total_gate_sum = 0.0
        total_gate_count = 0
        
        for layer_idx, entry in enumerate(kv_entries):
            # 获取 K 和 V
            if hasattr(entry, 'key') and hasattr(entry, 'value'):
                k = entry.key   # [batch, heads, seq, head_dim]
                v = entry.value  # [batch, heads, seq, head_dim]
            elif isinstance(entry, tuple) and len(entry) == 2:
                k, v = entry
            else:
                gated_kv.append(entry)
                continue
            
            # === 核心: 上下文感知门控 ===
            if (self.config.gating_mode == GatingMode.CONTEXT_AWARE
                    and self.gating is not None
                    and query_hidden is not None):
                # Engram-style per-token gating
                gates = self.gating(
                    query_hidden=query_hidden,
                    memory_keys=k,
                    base_alpha=base_alpha,
                )  # [batch, heads, mem_len, 1]
                
                # 只门控 Value, Key 保持不变 (Engram 设计原则)
                v_gated = v * gates
                
                # 记录统计
                avg_gate = gates.mean().item()
                total_gate_sum += avg_gate
                total_gate_count += 1
                
            elif self.config.gating_mode == GatingMode.HYBRID:
                # 混合模式: 全局 α × 逐 token 门控
                if self.gating is not None and query_hidden is not None:
                    gates = self.gating(
                        query_hidden=query_hidden,
                        memory_keys=k,
                        base_alpha=1.0,  # 不在这里应用 base_alpha
                    )
                    # 全局 α 作为整体缩放
                    v_gated = v * gates * base_alpha
                    
                    avg_gate = (gates.mean().item() * base_alpha)
                    total_gate_sum += avg_gate
                    total_gate_count += 1
                else:
                    v_gated = v * base_alpha
                    total_gate_sum += base_alpha
                    total_gate_count += 1
            else:
                # UNIFORM 模式: 原始 DKI 行为
                # 注意: 这里修正了原始实现中对 K 也缩放的问题
                v_gated = v * base_alpha
                total_gate_sum += base_alpha
                total_gate_count += 1
            
            gated_kv.append((k, v_gated))
        
        # 更新结果统计
        avg_gate = total_gate_sum / max(total_gate_count, 1)
        if memory_type == "preference":
            result.avg_preference_gate = avg_gate
        else:
            result.avg_history_gate = avg_gate
        
        return gated_kv
    
    def _apply_refinement(
        self,
        gated_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        应用卷积精化 (Engram Eq. 5)
        
        仅精化 Value, Key 保持不变。
        """
        if self.refinement is None or self.config.refinement_mode == RefinementMode.NONE:
            return gated_kv
        
        refined_kv = []
        for k, v_gated in gated_kv:
            # 将 [batch, heads, seq, dim] 转为 refinement 期望的格式
            v_refined = self.refinement(v_gated)
            refined_kv.append((k, v_refined))
        
        return refined_kv
    
    def _assemble_layer_kv(
        self,
        preference_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        history_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        按照层级策略组装 per-layer K/V
        
        遵循 Engram 的层级选择性原则:
        - 偏好 → 早期层 (卸载"静态模式重建")
        - 历史 → 中期层 (提供"推理上下文")
        """
        layer_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        if preference_kv:
            num_pref_layers = len(preference_kv)
            for target_layer in self._pref_layers:
                # 映射到可用的 K/V 层
                source_layer = min(target_layer, num_pref_layers - 1)
                k, v = preference_kv[source_layer]
                
                if target_layer in layer_kv:
                    # 该层已有数据 (历史)，需要拼接
                    existing_k, existing_v = layer_kv[target_layer]
                    layer_kv[target_layer] = (
                        torch.cat([k, existing_k], dim=2),
                        torch.cat([v, existing_v], dim=2),
                    )
                else:
                    layer_kv[target_layer] = (k, v)
        
        if history_kv:
            num_hist_layers = len(history_kv)
            for target_layer in self._hist_layers:
                source_layer = min(target_layer, num_hist_layers - 1)
                k, v = history_kv[source_layer]
                
                if target_layer in layer_kv:
                    existing_k, existing_v = layer_kv[target_layer]
                    layer_kv[target_layer] = (
                        torch.cat([existing_k, k], dim=2),
                        torch.cat([existing_v, v], dim=2),
                    )
                else:
                    layer_kv[target_layer] = (k, v)
        
        return layer_kv
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        """格式化历史消息"""
        if not messages:
            return ""
        
        lines = []
        for msg in messages[-self.config.history_max_messages:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if self.language == "cn":
                role_label = "用户" if role == "user" else "助手"
            else:
                role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")
        
        return "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数"""
        if not text:
            return 0
        if self.language == "cn":
            return int(len(text) * 0.7)
        return int(len(text.split()) * 1.3)
    
    def _truncate_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """截断历史消息"""
        if not messages:
            return []
        
        result = []
        total = 0
        for msg in reversed(messages):
            tokens = self._estimate_tokens(msg.get("content", ""))
            if total + tokens > max_tokens:
                break
            result.insert(0, msg)
            total += tokens
        
        return result
    
    def _update_stats(self, result: EngramInjectionResult):
        """更新统计"""
        n = self._stats["successful_injections"]
        self._stats["avg_preference_gate"] = (
            (self._stats["avg_preference_gate"] * (n - 1) + result.avg_preference_gate) / n
        )
        self._stats["avg_history_gate"] = (
            (self._stats["avg_history_gate"] * (n - 1) + result.avg_history_gate) / n
        )
        self._stats["avg_compute_time_ms"] = (
            (self._stats["avg_compute_time_ms"] * (n - 1) + result.compute_time_ms) / n
        )
    
    def _log_injection(self, result: EngramInjectionResult, query: str):
        """记录注入日志"""
        log_entry = {
            "timestamp": time.time(),
            "query": query[:100],
            "gating_mode": result.gating_mode,
            "refinement_mode": result.refinement_mode,
            "preference_tokens": result.preference_tokens,
            "history_tokens": result.history_tokens,
            "avg_preference_gate": result.avg_preference_gate,
            "avg_history_gate": result.avg_history_gate,
            "preference_layers": result.preference_layers,
            "history_layers": result.history_layers,
            "gating_time_ms": result.gating_time_ms,
            "refinement_time_ms": result.refinement_time_ms,
            "compute_time_ms": result.compute_time_ms,
        }
        
        self._attention_logs.append(log_entry)
        if len(self._attention_logs) > self._max_attention_logs:
            self._attention_logs = self._attention_logs[-self._max_attention_logs:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "config": {
                "gating_mode": self.config.gating_mode.value,
                "refinement_mode": self.config.refinement_mode.value,
                "preference_base_alpha": self.config.preference_base_alpha,
                "history_base_alpha": self.config.history_base_alpha,
                "preference_layers": self._pref_layers,
                "history_layers": self._hist_layers,
                "max_total_kv_tokens": self.config.max_total_kv_tokens,
                "conv_kernel_size": self.config.conv_kernel_size,
                "gating_temperature": self.config.gating_temperature,
                "use_rmsnorm": self.config.use_rmsnorm,
            },
        }
    
    def get_attention_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取注入日志"""
        return self._attention_logs[-limit:]
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息 (参数量、内存开销等)"""
        total_params = 0
        info = {}
        
        if self.gating is not None:
            gating_params = sum(
                p.numel() for p in self.gating.parameters()
            )
            total_params += gating_params
            info["gating_params"] = gating_params
            info["gating_memory_kb"] = gating_params * 4 / 1024
        
        if self.refinement is not None:
            refinement_params = sum(
                p.numel() for p in self.refinement.parameters()
            )
            total_params += refinement_params
            info["refinement_params"] = refinement_params
            info["refinement_memory_kb"] = refinement_params * 4 / 1024
        
        info["total_params"] = total_params
        info["total_memory_kb"] = total_params * 4 / 1024
        
        return info
