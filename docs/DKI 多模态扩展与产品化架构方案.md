# DKI 多模态扩展与产品化架构方案

## 第一部分：多模态 K/V 注入位置策略分析

### 1.1 三种方案的稳定性分析

#### 方案 A: Prefix KV 注入 (Position 0~N) + 小 α

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Prefix KV 注入方案分析                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  位置布局:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [Vision KV]  │  [Text Memory KV]  │  [User Input KV]           │   │
│  │  pos: 0~N     │  pos: N+1~M        │  pos: M+1~L                │   │
│  │  α: 0.1~0.3   │  α: 0.3~0.7        │  α: 1.0                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  优点:                                                                  │
│  ✅ 完全在训练分布内 (正位置)                                           │
│  ✅ RoPE 相对位置计算正常                                               │
│  ✅ 2025 年 MLLM 框架标配，工程验证充分                                 │
│  ✅ 小 α 模拟"背景知识"，不会过度干扰                                   │
│                                                                         │
│  缺点:                                                                  │
│  ❌ 消耗 Token Budget (与 RAG 相同问题)                                 │
│  ❌ 视觉 token 数量大 (256~576)，占用显著                               │
│  ❌ 需要 recompute 后续位置的 RoPE                                      │
│                                                                         │
│  稳定性评估: ⭐⭐⭐⭐⭐ (最稳定)                                          │
│  Token 效率: ⭐⭐☆☆☆ (较差)                                             │
│  研究价值: ⭐⭐☆☆☆ (工程方案，非创新)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**数学分析**：

对于 RoPE，相对位置 $r = i - j$ 决定注意力：

-   Vision at pos 0, User at pos 300: $r = 300$ ✅ 在训练分布内
-   但 Vision 会"挤占" User 的位置空间

#### 方案 B: 虚拟负位置 + Decoupled RoPE / Global Bias

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    虚拟负位置方案分析                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  位置布局:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [Vision KV]     │  [Text Memory KV]  │  [User Input KV]        │   │
│  │  pos: -N~-1      │  pos: -M~-N-1      │  pos: 0~L               │   │
│  │  + global bias   │  α: 0.3~0.7        │  α: 1.0                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  实现方式 1: Decoupled RoPE                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  标准 RoPE: Q_rot = Q * e^{i·pos·θ}                              │   │
│  │  Decoupled: 对 Vision KV 使用独立的 θ_vision                     │   │
│  │             或完全不应用 RoPE (NoPE)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  实现方式 2: Global Bias                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Attention_logits[..., :vision_len] += global_bias              │   │
│  │  global_bias > 0: 拉近视觉 KV (增加注意力)                       │   │
│  │  等效于让视觉 KV "看起来更近"                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  优点:                                                                  │
│  ✅ 不消耗 Token Budget                                                 │
│  ✅ 保持 DKI 核心优势                                                   │
│  ✅ Global bias 可学习/可调                                             │
│                                                                         │
│  缺点:                                                                  │
│  ⚠️ 负位置可能 OOD (需要验证)                                          │
│  ⚠️ Decoupled RoPE 需要修改模型代码                                    │
│  ⚠️ 不同模型的 RoPE 实现不同                                           │
│                                                                         │
│  稳定性评估: ⭐⭐⭐☆☆ (需要实验验证)                                     │
│  Token 效率: ⭐⭐⭐⭐⭐ (最优)                                            │
│  研究价值: ⭐⭐⭐⭐☆ (有创新空间)                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**OOD 风险分析**：

```python
# RoPE 的相对位置计算
# 对于 Vision at pos -256, User at pos 100:
relative_pos = 100 - (-256) = 356

# 如果训练时最大 context = 4096，则 356 << 4096，理论上安全
# 但负位置本身的 cos/sin 值可能未被训练过

# 缓解方案：使用 NTK-aware RoPE 或 YaRN 扩展
```

#### 方案 C: 常量位置 / 视觉锚点投影 (研究级)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    激进实验方案分析                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  方案 C1: 常量位置 (Position = -10000 或 Phase = 0)                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  所有 Vision KV 使用相同位置编码                                 │   │
│  │  效果：视觉信息作为"全局上下文"，无位置依赖                      │   │
│  │                                                                  │   │
│  │  RoPE with pos = -10000:                                         │   │
│  │  cos(-10000 * θ), sin(-10000 * θ) → 可能数值不稳定               │   │
│  │                                                                  │   │
│  │  Phase = 0 (NoPE for Vision):                                    │   │
│  │  Vision KV 不应用任何位置编码 → 位置无关                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  方案 C2: 每层独立视觉锚点投影                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer l: K_vision_l = W_anchor_l @ Vision_features              │   │
│  │                                                                  │   │
│  │  每层学习一个投影矩阵，将视觉特征投影到该层的 KV 空间            │   │
│  │  注入到 KV cache[0] 位置                                         │   │
│  │                                                                  │   │
│  │  参数量: L × d × d_vision (可用低秩分解减少)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  方案 C3: Cache Steering (Prefill 后 Overwrite)                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. 正常 Prefill 用户输入                                        │   │
│  │  2. 计算视觉 KV                                                  │   │
│  │  3. Overwrite KV cache 开头位置                                  │   │
│  │                                                                  │   │
│  │  KV_cache[:, :, :vision_len, :] = Vision_KV                      │   │
│  │                                                                  │   │
│  │  问题：破坏了原有的 KV，可能导致不一致                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  稳定性评估: ⭐⭐☆☆☆ (高风险，需要大量实验)                             │
│  Token 效率: ⭐⭐⭐⭐⭐ (最优)                                            │
│  研究价值: ⭐⭐⭐⭐⭐ (高创新，可发论文)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 研究阶段推荐路径

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    研究阶段实验路径                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 基线建立 (1-2 周)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  实验 1.1: Prefix KV (Position 0~N) + α=0.2                      │   │
│  │  - 使用 LLaVA-1.5 或 Qwen-VL 作为基座                            │   │
│  │  - 验证视觉 KV 注入的基本可行性                                  │   │
│  │  - 建立 baseline metrics                                         │   │
│  │                                                                  │   │
│  │  实验 1.2: 不同 α 的影响                                         │   │
│  │  - α ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}                           │   │
│  │  - 观察输出质量 vs 视觉信息利用率                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 2: 负位置探索 (2-3 周)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  实验 2.1: 负位置 + 标准 RoPE                                    │   │
│  │  - Vision at pos -256~-1                                         │   │
│  │  - 观察 attention pattern 是否正常                               │   │
│  │  - 对比 Phase 1 的输出质量                                       │   │
│  │                                                                  │   │
│  │  实验 2.2: NoPE for Vision                                       │   │
│  │  - Vision KV 不应用 RoPE                                         │   │
│  │  - Text/User KV 正常应用 RoPE                                    │   │
│  │  - 验证"位置无关视觉"假设                                        │   │
│  │                                                                  │   │
│  │  实验 2.3: Global Bias                                           │   │
│  │  - 在 attention logits 上加 learnable bias                       │   │
│  │  - bias ∈ {0.5, 1.0, 2.0, 5.0}                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 3: 激进实验 (3-4 周)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  实验 3.1: 常量位置                                              │   │
│  │  - pos = -10000 (极端负)                                         │   │
│  │  - pos = 0 (所有视觉 token 同位置)                               │   │
│  │                                                                  │   │
│  │  实验 3.2: 视觉锚点投影                                          │   │
│  │  - 每层学习 W_anchor                                             │   │
│  │  - 低秩分解: W_anchor = A @ B, rank=64                          │   │
│  │                                                                  │   │
│  │  实验 3.3: Cache Steering                                        │   │
│  │  - Prefill → Overwrite → Generate                                │   │
│  │  - 对比直接注入 vs Overwrite                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 4: 最优方案确定 (1-2 周)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 综合 Phase 1-3 结果                                           │   │
│  │  - 确定最优位置策略                                              │   │
│  │  - 确定最优 α 范围                                               │   │
│  │  - 撰写技术报告/论文                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 实验代码框架

```python
# dki/research/multimodal_injection.py

import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List

class PositionStrategy(Enum):
    """位置编码策略"""
    PREFIX_POSITIVE = "prefix_positive"      # 方案 A: 正位置前缀
    NEGATIVE_ROPE = "negative_rope"          # 方案 B1: 负位置 + RoPE
    NEGATIVE_NOPE = "negative_nope"          # 方案 B2: 负位置 + NoPE
    GLOBAL_BIAS = "global_bias"              # 方案 B3: Global Bias
    CONSTANT_POSITION = "constant_position"  # 方案 C1: 常量位置
    ANCHOR_PROJECTION = "anchor_projection"  # 方案 C2: 锚点投影
    CACHE_STEERING = "cache_steering"        # 方案 C3: Cache Steering


@dataclass
class MultimodalInjectionConfig:
    """多模态注入配置"""
    position_strategy: PositionStrategy
    vision_alpha: float = 0.2
    text_memory_alpha: float = 0.5

    # 方案 B3: Global Bias
    global_bias_value: float = 1.0

    # 方案 C1: 常量位置
    constant_position: int = -10000

    # 方案 C2: 锚点投影
    anchor_rank: int = 64

    # 实验追踪
    experiment_name: str = ""
    log_attention_patterns: bool = True


class MultimodalKVInjector:
    """
    多模态 K/V 注入器 - 研究版本

    支持多种位置编码策略的实验对比
    """

    def __init__(
        self,
        config: MultimodalInjectionConfig,
        model_hidden_dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
    ):
        self.config = config
        self.hidden_dim = model_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 方案 C2: 初始化锚点投影
        if config.position_strategy == PositionStrategy.ANCHOR_PROJECTION:
            self._init_anchor_projections()

        # 实验日志
        self.attention_logs: List[torch.Tensor] = []

    def _init_anchor_projections(self):
        """初始化每层的视觉锚点投影"""
        self.anchor_projections = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.config.anchor_rank),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.anchor_rank, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])

    def inject_vision_kv(
        self,
        vision_kv: List[Tuple[torch.Tensor, torch.Tensor]],  # [(K, V)] per layer
        text_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        user_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        注入视觉 K/V

        Returns:
            合并后的 K/V 列表，每层一个 (K, V) tuple
        """
        strategy = self.config.position_strategy

        if strategy == PositionStrategy.PREFIX_POSITIVE:
            return self._inject_prefix_positive(vision_kv, text_kv, user_kv)

        elif strategy == PositionStrategy.NEGATIVE_ROPE:
            return self._inject_negative_rope(vision_kv, text_kv, user_kv, rope_cache)

        elif strategy == PositionStrategy.NEGATIVE_NOPE:
            return self._inject_negative_nope(vision_kv, text_kv, user_kv, rope_cache)

        elif strategy == PositionStrategy.GLOBAL_BIAS:
            return self._inject_with_global_bias(vision_kv, text_kv, user_kv)

        elif strategy == PositionStrategy.CONSTANT_POSITION:
            return self._inject_constant_position(vision_kv, text_kv, user_kv, rope_cache)

        elif strategy == PositionStrategy.ANCHOR_PROJECTION:
            return self._inject_anchor_projection(vision_kv, text_kv, user_kv)

        elif strategy == PositionStrategy.CACHE_STEERING:
            return self._inject_cache_steering(vision_kv, text_kv, user_kv)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _inject_prefix_positive(
        self,
        vision_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        text_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        user_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        方案 A: 正位置前缀注入

        位置: [Vision: 0~N] [Text: N+1~M] [User: M+1~L]
        """
        merged_kv = []

        for layer_idx in range(self.num_layers):
            v_k, v_v = vision_kv[layer_idx]
            t_k, t_v = text_kv[layer_idx] if text_kv else (None, None)
            u_k, u_v = user_kv[layer_idx]

            # 应用 α 缩放
            v_k = v_k * self.config.vision_alpha
            v_v = v_v * self.config.vision_alpha

            if t_k is not None:
                t_k = t_k * self.config.text_memory_alpha
                t_v = t_v * self.config.text_memory_alpha

            # 拼接: [Vision, Text, User]
            if t_k is not None:
                merged_k = torch.cat([v_k, t_k, u_k], dim=2)
                merged_v = torch.cat([v_v, t_v, u_v], dim=2)
            else:
                merged_k = torch.cat([v_k, u_k], dim=2)
                merged_v = torch.cat([v_v, u_v], dim=2)

            merged_kv.append((merged_k, merged_v))

        return merged_kv

    def _inject_negative_nope(
        self,
        vision_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        text_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        user_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        方案 B2: 负位置 + NoPE (视觉不应用位置编码)

        Vision KV 不应用 RoPE，作为"位置无关"的全局上下文
        """
        merged_kv = []

        for layer_idx in range(self.num_layers):
            v_k, v_v = vision_kv[layer_idx]
            t_k, t_v = text_kv[layer_idx] if text_kv else (None, None)
            u_k, u_v = user_kv[layer_idx]

            # Vision KV 不应用 RoPE (已经是原始 KV)
            # Text 和 User KV 假设已经应用了 RoPE

            # 应用 α 缩放
            v_k = v_k * self.config.vision_alpha
            v_v = v_v * self.config.vision_alpha

            if t_k is not None:
                t_k = t_k * self.config.text_memory_alpha
                t_v = t_v * self.config.text_memory_alpha

            # 拼接
            if t_k is not None:
                merged_k = torch.cat([v_k, t_k, u_k], dim=2)
                merged_v = torch.cat([v_v, t_v, u_v], dim=2)
            else:
                merged_k = torch.cat([v_k, u_k], dim=2)
                merged_v = torch.cat([v_v, u_v], dim=2)

            merged_kv.append((merged_k, merged_v))

        return merged_kv

    def _inject_with_global_bias(
        self,
        vision_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        text_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        user_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        方案 B3: Global Bias

        在 attention logits 上为视觉 KV 添加正偏置
        """
        # 注意：Global Bias 需要在 attention 计算时应用
        # 这里只做 KV 拼接，bias 在 forward 时应用

        merged_kv = []
        vision_len = vision_kv[0][0].shape[2] if vision_kv else 0

        for layer_idx in range(self.num_layers):
            v_k, v_v = vision_kv[layer_idx]
            t_k, t_v = text_kv[layer_idx] if text_kv else (None, None)
            u_k, u_v = user_kv[layer_idx]

            # 拼接
            if t_k is not None:
                merged_k = torch.cat([v_k, t_k, u_k], dim=2)
                merged_v = torch.cat([v_v, t_v, u_v], dim=2)
            else:
                merged_k = torch.cat([v_k, u_k], dim=2)
                merged_v = torch.cat([v_v, u_v], dim=2)

            merged_kv.append((merged_k, merged_v))

        # 存储 vision_len 用于后续 bias 应用
        self._vision_len_for_bias = vision_len

        return merged_kv

    def apply_global_bias(
        self,
        attention_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用 Global Bias 到 attention logits

        Args:
            attention_logits: [batch, heads, seq, total_len]

        Returns:
            带 bias 的 attention logits
        """
        if not hasattr(self, '_vision_len_for_bias'):
            return attention_logits

        vision_len = self._vision_len_for_bias
        if vision_len == 0:
            return attention_logits

        # 为视觉位置添加正偏置
        bias = torch.zeros_like(attention_logits)
        bias[..., :vision_len] = self.config.global_bias_value

        return attention_logits + bias

    def _inject_anchor_projection(
        self,
        vision_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        text_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        user_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        方案 C2: 每层独立视觉锚点投影

        每层学习一个投影，将视觉特征投影到该层的 KV 空间
        """
        merged_kv = []

        for layer_idx in range(self.num_layers):
            v_k, v_v = vision_kv[layer_idx]
            t_k, t_v = text_kv[layer_idx] if text_kv else (None, None)
            u_k, u_v = user_kv[layer_idx]

            # 应用锚点投影
            # v_k shape: [batch, heads, seq, head_dim]
            # 需要 reshape 来应用投影
            batch, heads, seq, head_dim = v_k.shape

            # 合并 heads 和 head_dim
            v_k_flat = v_k.permute(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)
            v_v_flat = v_v.permute(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)

            # 应用投影
            v_k_proj = self.anchor_projections[layer_idx](v_k_flat)
            v_v_proj = self.anchor_projections[layer_idx](v_v_flat)

            # Reshape 回原形状
            v_k_proj = v_k_proj.reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3)
            v_v_proj = v_v_proj.reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3)

            # 应用 α 缩放
            v_k_proj = v_k_proj * self.config.vision_alpha
            v_v_proj = v_v_proj * self.config.vision_alpha

            # 拼接
            if t_k is not None:
                merged_k = torch.cat([v_k_proj, t_k, u_k], dim=2)
                merged_v = torch.cat([v_v_proj, t_v, u_v], dim=2)
            else:
                merged_k = torch.cat([v_k_proj, u_k], dim=2)
                merged_v = torch.cat([v_v_proj, u_v], dim=2)

            merged_kv.append((merged_k, merged_v))

        return merged_kv

    def log_attention_pattern(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int,
    ):
        """记录 attention pattern 用于分析"""
        if self.config.log_attention_patterns:
            self.attention_logs.append({
                'layer': layer_idx,
                'weights': attention_weights.detach().cpu(),
                'strategy': self.config.position_strategy.value,
            })

    def get_attention_analysis(self) -> dict:
        """分析 attention patterns"""
        if not self.attention_logs:
            return {}

        # 计算视觉区域的平均注意力
        vision_attention_by_layer = []

        for log in self.attention_logs:
            weights = log['weights']
            # 假设视觉在前 256 个位置
            vision_attn = weights[..., :256].mean().item()
            vision_attention_by_layer.append({
                'layer': log['layer'],
                'vision_attention': vision_attn,
            })

        return {
            'strategy': self.config.position_strategy.value,
            'vision_attention_by_layer': vision_attention_by_layer,
            'avg_vision_attention': sum(
                x['vision_attention'] for x in vision_attention_by_layer
            ) / len(vision_attention_by_layer),
        }
```

---

## 第二部分：DKI 产品化架构方案

### 2.1 产品化挑战分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DKI 产品化挑战                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  规模挑战:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. 用户规模: 百万级用户，每用户数十条记忆                       │   │
│  │  2. 并发请求: 数千 QPS                                           │   │
│  │  3. 延迟要求: P99 < 500ms                                        │   │
│  │  4. 可用性: 99.9% SLA                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  技术挑战:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. KV Cache 内存: 每记忆 ~100MB (7B 模型)                       │   │
│  │  2. 多租户隔离: 用户记忆不能混淆                                 │   │
│  │  3. 模型升级: 新模型需要重算 KV                                  │   │
│  │  4. 冷启动: 新用户/新会话的首次延迟                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  运维挑战:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. 监控: 注入效果、α 分布、缓存命中率                           │   │
│  │  2. 调试: 为什么某次注入效果不好？                               │   │
│  │  3. 回滚: 新版本有问题时快速回滚                                 │   │
│  │  4. 合规: 审计日志、数据隐私                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 产品化架构设计

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DKI Production Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              API Gateway Layer                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │   Rate      │  │   Auth      │  │   Load      │  │   Circuit   │            │   │
│  │  │   Limiter   │  │   Service   │  │   Balancer  │  │   Breaker   │            │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                             │
│                                           ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              DKI Service Layer                                   │   │
│  │                                                                                  │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐     │   │
│  │  │   DKI Coordinator   │  │   DKI Coordinator   │  │   DKI Coordinator   │     │   │
│  │  │   (Stateless)       │  │   (Stateless)       │  │   (Stateless)       │     │   │
│  │  │                     │  │                     │  │                     │     │   │
│  │  │  - Gating Decision  │  │  - Gating Decision  │  │  - Gating Decision  │     │   │
│  │  │  - α Computation    │  │  - α Computation    │  │  - α Computation    │     │   │
│  │  │  - Request Routing  │  │  - Request Routing  │  │  - Request Routing  │     │   │
│  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                             │
│                    ┌──────────────────────┼──────────────────────┐                      │
│                    ↓                      ↓                      ↓                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐             │
│  │   Memory Service    │  │   KV Cache Service  │  │   LLM Inference     │             │
│  │                     │  │                     │  │   Service           │             │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │             │
│  │  │ Memory Router │  │  │  │ L1: GPU HBM   │  │  │  │ vLLM Cluster  │  │             │
│  │  │ (FAISS/Milvus)│  │  │  │ (Hot Cache)   │  │  │  │               │  │             │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  │ - Model Shard │  │             │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  │ - KV Injection│  │             │
│  │  │ Embedding     │  │  │  │ L2: Redis     │  │  │  │ - Generation  │  │             │
│  │  │ Service       │  │  │  │ (Warm Cache)  │  │  │  └───────────────┘  │             │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  ┌───────────────┐  │             │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  │ TensorRT-LLM  │  │             │
│  │  │ Memory Store  │  │  │  │ L3: Object    │  │  │  │ Cluster       │  │             │
│  │  │ (PostgreSQL)  │  │  │  │ Storage (S3)  │  │  │  └───────────────┘  │             │
│  │  └───────────────┘  │  │  └───────────────┘  │  └─────────────────────┘             │
│  └─────────────────────┘  └─────────────────────┘                                       │
│                                           │                                             │
│                                           ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Data & Storage Layer                                │   │
│  │                                                                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │   PostgreSQL    │  │   Milvus/       │  │   Redis         │                  │   │
│  │  │   (Metadata)    │  │   Qdrant        │  │   (Session)     │                  │   │
│  │  │                 │  │   (Vectors)     │  │                 │                  │   │
│  │  │  - Users        │  │                 │  │  - Session KV   │                  │   │
│  │  │  - Memories     │  │  - Embeddings   │  │  - User Prefs   │                  │   │
│  │  │  - Sessions     │  │  - ANN Index    │  │  - Rate Limits  │                  │   │
│  │  │  - Audit Logs   │  │                 │  │                 │                  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                             │
│                                           ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Observability Layer                                 │   │
│  │                                                                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │   Prometheus    │  │   Grafana       │  │   Jaeger        │                  │   │
│  │  │   (Metrics)     │  │   (Dashboards)  │  │   (Tracing)     │                  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │   ELK Stack     │  │   PagerDuty     │  │   DataDog       │                  │   │
│  │  │   (Logs)        │  │   (Alerts)      │  │   (APM)         │                  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 核心服务详细设计

#### 2.3.1 DKI Coordinator Service

```python
# dki/production/coordinator.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNTER = Counter('dki_requests_total', 'Total DKI requests', ['status', 'injection_mode'])
LATENCY_HISTOGRAM = Histogram('dki_latency_seconds', 'DKI request latency', ['stage'])
ALPHA_GAUGE = Gauge('dki_alpha_distribution', 'Alpha value distribution', ['bucket'])
CACHE_HIT_COUNTER = Counter('dki_cache_hits_total', 'KV cache hits', ['tier'])


class InjectionMode(Enum):
    FULL_DKI = "full_dki"           # K/V 注入
    HYBRID_ONLY = "hybrid_only"     # 仅 Suffix Prompt
    FALLBACK = "fallback"           # 降级到 Vanilla


@dataclass
class CoordinatorConfig:
    """Coordinator 配置"""
    # 服务发现
    memory_service_url: str
    kv_cache_service_url: str
    llm_service_url: str

    # 超时配置
    memory_timeout_ms: int = 50
    kv_cache_timeout_ms: int = 100
    llm_timeout_ms: int = 30000

    # 降级配置
    fallback_on_memory_timeout: bool = True
    fallback_on_cache_miss: bool = False
    max_retries: int = 2

    # 特性开关
    enable_multimodal: bool = False
    enable_hybrid_injection: bool = True

    # A/B 测试
    ab_test_enabled: bool = False
    ab_test_dki_percentage: int = 50


class DKICoordinator:
    """
    DKI 协调器 - 无状态服务

    职责:
    1. 接收请求，决定注入模式
    2. 协调 Memory Service、KV Cache Service、LLM Service
    3. 处理超时和降级
    4. 记录 metrics 和 traces
    """

    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """启动服务"""
        self.session = aiohttp.ClientSession()

    async def stop(self):
        """停止服务"""
        if self.session:
            await self.session.close()

    async def process_request(
        self,
        user_id: str,
        session_id: str,
        query: str,
        preferences: Optional[str] = None,
        allow_injection: bool = True,
        force_mode: Optional[InjectionMode] = None,
    ) -> Dict[str, Any]:
        """
        处理 DKI 请求

        流程:
        1. 决定注入模式 (A/B 测试、特性开关)
        2. 检索相关记忆
        3. 获取/计算 KV Cache
        4. 调用 LLM 生成
        5. 记录审计日志
        """
        request_id = self._generate_request_id()

        with LATENCY_HISTOGRAM.labels(stage='total').time():
            try:
                # Step 1: 决定注入模式
                mode = force_mode or self._decide_injection_mode(user_id, allow_injection)

                if mode == InjectionMode.FALLBACK:
                    return await self._fallback_generation(query, request_id)

                # Step 2: 检索记忆
                with LATENCY_HISTOGRAM.labels(stage='memory_retrieval').time():
                    memories = await self._retrieve_memories(user_id, query)

                if not memories and mode == InjectionMode.FULL_DKI:
                    # 无相关记忆，降级到 Hybrid
                    mode = InjectionMode.HYBRID_ONLY

                # Step 3: Gating 决策
                gating_decision = await self._make_gating_decision(query, memories)

                if not gating_decision['should_inject']:
                    return await self._fallback_generation(query, request_id)

                # Step 4: 获取 KV Cache (仅 FULL_DKI 模式)
                kv_cache = None
                if mode == InjectionMode.FULL_DKI:
                    with LATENCY_HISTOGRAM.labels(stage='kv_cache').time():
                        kv_cache = await self._get_or_compute_kv(
                            user_id, session_id, memories
                        )

                # Step 5: 调用 LLM
                with LATENCY_HISTOGRAM.labels(stage='llm_generation').time():
                    response = await self._generate_with_injection(
                        query=query,
                        memories=memories,
                        kv_cache=kv_cache,
                        alpha=gating_decision['alpha'],
                        mode=mode,
                        preferences=preferences,
                    )

                # 记录 metrics
                REQUEST_COUNTER.labels(status='success', injection_mode=mode.value).inc()
                self._record_alpha(gating_decision['alpha'])

                return {
                    'request_id': request_id,
                    'response': response,
                    'mode': mode.value,
                    'alpha': gating_decision['alpha'],
                    'memories_used': len(memories),
                    'cache_hit': kv_cache.get('cache_hit', False) if kv_cache else False,
                }

            except asyncio.TimeoutError:
                REQUEST_COUNTER.labels(status='timeout', injection_mode='fallback').inc()
                return await self._fallback_generation(query, request_id)

            except Exception as e:
                REQUEST_COUNTER.labels(status='error', injection_mode='fallback').inc()
                # Log error and fallback
                return await self._fallback_generation(query, request_id)

    def _decide_injection_mode(
        self,
        user_id: str,
        allow_injection: bool,
    ) -> InjectionMode:
        """决定注入模式"""
        if not allow_injection:
            return InjectionMode.FALLBACK

        # A/B 测试
        if self.config.ab_test_enabled:
            user_bucket = hash(user_id) % 100
            if user_bucket >= self.config.ab_test_dki_percentage:
                return InjectionMode.FALLBACK

        # 特性开关
        if not self.config.enable_hybrid_injection:
            return InjectionMode.FULL_DKI

        return InjectionMode.FULL_DKI

    async def _retrieve_memories(
        self,
        user_id: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """从 Memory Service 检索记忆"""
        try:
            async with self.session.post(
                f"{self.config.memory_service_url}/search",
                json={"user_id": user_id, "query": query, "top_k": 5},
                timeout=aiohttp.ClientTimeout(
                    total=self.config.memory_timeout_ms / 1000
                ),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json())['results']
                return []
        except asyncio.TimeoutError:
            if self.config.fallback_on_memory_timeout:
                return []
            raise

    async def _get_or_compute_kv(
        self,
        user_id: str,
        session_id: str,
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """获取或计算 KV Cache"""
        memory_ids = [m['id'] for m in memories]

        async with self.session.post(
            f"{self.config.kv_cache_service_url}/get_or_compute",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "memory_ids": memory_ids,
                "memory_contents": [m['content'] for m in memories],
            },
            timeout=aiohttp.ClientTimeout(
                total=self.config.kv_cache_timeout_ms / 1000
            ),
        ) as resp:
            result = await resp.json()

            # 记录缓存命中
            if result.get('cache_hit'):
                CACHE_HIT_COUNTER.labels(tier=result.get('cache_tier', 'unknown')).inc()

            return result

    async def _generate_with_injection(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        kv_cache: Optional[Dict[str, Any]],
        alpha: float,
        mode: InjectionMode,
        preferences: Optional[str],
    ) -> str:
        """调用 LLM 生成"""
        payload = {
            "query": query,
            "mode": mode.value,
            "alpha": alpha,
        }

        if mode == InjectionMode.FULL_DKI and kv_cache:
            payload["kv_cache_ref"] = kv_cache.get('cache_ref')

        if mode == InjectionMode.HYBRID_ONLY or preferences:
            # 构建 Hybrid prompt
            payload["memories_text"] = self._format_memories_for_prompt(memories)
            payload["preferences"] = preferences

        async with self.session.post(
            f"{self.config.llm_service_url}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(
                total=self.config.llm_timeout_ms / 1000
            ),
        ) as resp:
            result = await resp.json()
            return result['text']

    async def _fallback_generation(
        self,
        query: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """降级生成"""
        async with self.session.post(
            f"{self.config.llm_service_url}/generate",
            json={"query": query, "mode": "vanilla"},
            timeout=aiohttp.ClientTimeout(
                total=self.config.llm_timeout_ms / 1000
            ),
        ) as resp:
            result = await resp.json()
            return {
                'request_id': request_id,
                'response': result['text'],
                'mode': 'fallback',
                'alpha': 0.0,
                'memories_used': 0,
                'cache_hit': False,
            }

    def _record_alpha(self, alpha: float):
        """记录 α 分布"""
        if alpha < 0.2:
            ALPHA_GAUGE.labels(bucket='0.0-0.2').inc()
        elif alpha < 0.4:
            ALPHA_GAUGE.labels(bucket='0.2-0.4').inc()
        elif alpha < 0.6:
            ALPHA_GAUGE.labels(bucket='0.4-0.6').inc()
        elif alpha < 0.8:
            ALPHA_GAUGE.labels(bucket='0.6-0.8').inc()
        else:
            ALPHA_GAUGE.labels(bucket='0.8-1.0').inc()
```

#### 2.3.2 KV Cache Service

```python
# dki/production/kv_cache_service.py

import asyncio
import hashlib
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import redis.asyncio as redis
import aioboto3
from prometheus_client import Counter, Histogram, Gauge

# Metrics
KV_COMPUTE_COUNTER = Counter('dki_kv_compute_total', 'KV computations')
KV_CACHE_SIZE_GAUGE = Gauge('dki_kv_cache_size_bytes', 'KV cache size', ['tier'])


class CacheTier(Enum):
    L1_GPU = "l1_gpu"      # GPU HBM, 本地
    L2_REDIS = "l2_redis"  # Redis, 分布式
    L3_S3 = "l3_s3"        # S3, 持久化
    COMPUTE = "compute"    # 重新计算


@dataclass
class KVCacheServiceConfig:
    """KV Cache Service 配置"""
    # Redis 配置
    redis_url: str
    redis_ttl_seconds: int = 3600

    # S3 配置
    s3_bucket: str
    s3_prefix: str = "kv_cache/"

    # LLM Service (用于计算 KV)
    llm_service_url: str

    # 缓存策略
    l1_max_entries: int = 100  # 每 GPU
    l2_max_size_gb: float = 10.0
    enable_l3: bool = True

    # 压缩
    enable_compression: bool = True
    compression_level: int = 6


class KVCacheService:
    """
    分布式 KV Cache 服务

    三层缓存架构:
    - L1: GPU HBM (本地，最快)
    - L2: Redis Cluster (分布式，中等)
    - L3: S3 (持久化，最慢)
    """

    def __init__(self, config: KVCacheServiceConfig):
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.s3_session: Optional[aioboto3.Session] = None

        # L1 本地缓存 (LRU)
        self._l1_cache: Dict[str, Any] = {}
        self._l1_order: List[str] = []

    async def start(self):
        """启动服务"""
        self.redis = redis.from_url(self.config.redis_url)
        self.s3_session = aioboto3.Session()

    async def stop(self):
        """停止服务"""
        if self.redis:
            await self.redis.close()

    def _make_cache_key(
        self,
        user_id: str,
        memory_ids: List[str],
        model_version: str = "v1",
    ) -> str:
        """生成缓存 key"""
        content = f"{user_id}:{':'.join(sorted(memory_ids))}:{model_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get_or_compute(
        self,
        user_id: str,
        session_id: str,
        memory_ids: List[str],
        memory_contents: List[str],
    ) -> Dict[str, Any]:
        """
        获取或计算 KV Cache

        查找顺序: L1 → L2 → L3 → Compute
        """
        cache_key = self._make_cache_key(user_id, memory_ids)

        # L1: 本地 GPU 缓存
        if cache_key in self._l1_cache:
            self._touch_l1(cache_key)
            return {
                'cache_ref': cache_key,
                'cache_hit': True,
                'cache_tier': CacheTier.L1_GPU.value,
                'kv_data': self._l1_cache[cache_key],
            }

        # L2: Redis
        kv_data = await self._get_from_redis(cache_key)
        if kv_data is not None:
            # 提升到 L1
            self._put_l1(cache_key, kv_data)
            return {
                'cache_ref': cache_key,
                'cache_hit': True,
                'cache_tier': CacheTier.L2_REDIS.value,
                'kv_data': kv_data,
            }

        # L3: S3
        if self.config.enable_l3:
            kv_data = await self._get_from_s3(cache_key)
            if kv_data is not None:
                # 提升到 L2 和 L1
                await self._put_redis(cache_key, kv_data)
                self._put_l1(cache_key, kv_data)
                return {
                    'cache_ref': cache_key,
                    'cache_hit': True,
                    'cache_tier': CacheTier.L3_S3.value,
                    'kv_data': kv_data,
                }

        # Compute: 计算新的 KV
        kv_data = await self._compute_kv(memory_contents)
        KV_COMPUTE_COUNTER.inc()

        # 存储到所有层
        self._put_l1(cache_key, kv_data)
        await self._put_redis(cache_key, kv_data)
        if self.config.enable_l3:
            await self._put_s3(cache_key, kv_data)

        return {
            'cache_ref': cache_key,
            'cache_hit': False,
            'cache_tier': CacheTier.COMPUTE.value,
            'kv_data': kv_data,
        }

    def _put_l1(self, key: str, data: Any):
        """存入 L1 缓存"""
        if key in self._l1_cache:
            self._touch_l1(key)
            return

        # 驱逐
        while len(self._l1_cache) >= self.config.l1_max_entries:
            evict_key = self._l1_order.pop(0)
            del self._l1_cache[evict_key]

        self._l1_cache[key] = data
        self._l1_order.append(key)

    def _touch_l1(self, key: str):
        """更新 L1 访问顺序"""
        if key in self._l1_order:
            self._l1_order.remove(key)
            self._l1_order.append(key)

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """从 Redis 获取"""
        try:
            data = await self.redis.get(f"kv:{key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception:
            return None

    async def _put_redis(self, key: str, data: Any):
        """存入 Redis"""
        try:
            serialized = pickle.dumps(data)
            await self.redis.setex(
                f"kv:{key}",
                self.config.redis_ttl_seconds,
                serialized,
            )
        except Exception:
            pass  # 非关键路径，忽略错误

    async def _get_from_s3(self, key: str) -> Optional[Any]:
        """从 S3 获取"""
        try:
            async with self.s3_session.client('s3') as s3:
                response = await s3.get_object(
                    Bucket=self.config.s3_bucket,
                    Key=f"{self.config.s3_prefix}{key}",
                )
                data = await response['Body'].read()
                return pickle.loads(data)
        except Exception:
            return None

    async def _put_s3(self, key: str, data: Any):
        """存入 S3"""
        try:
            async with self.s3_session.client('s3') as s3:
                serialized = pickle.dumps(data)
                await s3.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=f"{self.config.s3_prefix}{key}",
                    Body=serialized,
                )
        except Exception:
            pass  # 非关键路径

    async def _compute_kv(self, memory_contents: List[str]) -> Any:
        """调用 LLM Service 计算 KV"""
        # 这里调用 LLM Service 的 compute_kv 接口
        # 实际实现需要与 LLM Service 集成
        pass

    async def invalidate(self, user_id: str, memory_ids: List[str]):
        """使缓存失效"""
        cache_key = self._make_cache_key(user_id, memory_ids)

        # L1
        if cache_key in self._l1_cache:
            del self._l1_cache[cache_key]
            self._l1_order.remove(cache_key)

        # L2
        await self.redis.delete(f"kv:{cache_key}")

        # L3
        if self.config.enable_l3:
            try:
                async with self.s3_session.client('s3') as s3:
                    await s3.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=f"{self.config.s3_prefix}{cache_key}",
                    )
            except Exception:
                pass
```

### 2.4 部署架构

```yaml
# kubernetes/dki-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
    name: dki-coordinator
    labels:
        app: dki
        component: coordinator
spec:
    replicas: 3
    selector:
        matchLabels:
            app: dki
            component: coordinator
    template:
        metadata:
            labels:
                app: dki
                component: coordinator
            annotations:
                prometheus.io/scrape: "true"
                prometheus.io/port: "8080"
        spec:
            containers:
                - name: coordinator
                  image: dki/coordinator:latest
                  ports:
                      - containerPort: 8080
                  resources:
                      requests:
                          cpu: "1"
                          memory: "2Gi"
                      limits:
                          cpu: "2"
                          memory: "4Gi"
                  env:
                      - name: MEMORY_SERVICE_URL
                        value: "http://memory-service:8080"
                      - name: KV_CACHE_SERVICE_URL
                        value: "http://kv-cache-service:8080"
                      - name: LLM_SERVICE_URL
                        value: "http://llm-service:8080"
                  livenessProbe:
                      httpGet:
                          path: /health
                          port: 8080
                      initialDelaySeconds: 10
                      periodSeconds: 10
                  readinessProbe:
                      httpGet:
                          path: /ready
                          port: 8080
                      initialDelaySeconds: 5
                      periodSeconds: 5

---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: kv-cache-service
    labels:
        app: dki
        component: kv-cache
spec:
    replicas: 2
    selector:
        matchLabels:
            app: dki
            component: kv-cache
    template:
        metadata:
            labels:
                app: dki
                component: kv-cache
        spec:
            containers:
                - name: kv-cache
                  image: dki/kv-cache-service:latest
                  ports:
                      - containerPort: 8080
                  resources:
                      requests:
                          cpu: "2"
                          memory: "8Gi"
                          nvidia.com/gpu: "1"
                      limits:
                          cpu: "4"
                          memory: "16Gi"
                          nvidia.com/gpu: "1"
                  env:
                      - name: REDIS_URL
                        valueFrom:
                            secretKeyRef:
                                name: dki-secrets
                                key: redis-url
                      - name: S3_BUCKET
                        value: "dki-kv-cache"

---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: llm-service
    labels:
        app: dki
        component: llm
spec:
    replicas: 4
    selector:
        matchLabels:
            app: dki
            component: llm
    template:
        metadata:
            labels:
                app: dki
                component: llm
        spec:
            containers:
                - name: vllm
                  image: vllm/vllm-openai:latest
                  ports:
                      - containerPort: 8000
                  resources:
                      requests:
                          nvidia.com/gpu: "1"
                      limits:
                          nvidia.com/gpu: "1"
                  args:
                      - "--model"
                      - "Qwen/Qwen2-7B-Instruct"
                      - "--tensor-parallel-size"
                      - "1"
                      - "--enable-dki" # 自定义 vLLM 支持 DKI
                      - "--dki-config"
                      - "/config/dki.yaml"
                  volumeMounts:
                      - name: dki-config
                        mountPath: /config
            volumes:
                - name: dki-config
                  configMap:
                      name: dki-llm-config

---
apiVersion: v1
kind: Service
metadata:
    name: dki-api
spec:
    type: LoadBalancer
    ports:
        - port: 80
          targetPort: 8080
    selector:
        app: dki
        component: coordinator
```

### 2.5 监控与告警

```yaml
# prometheus/dki-alerts.yaml

groups:
    - name: dki-alerts
      rules:
          # 延迟告警
          - alert: DKIHighLatency
            expr: histogram_quantile(0.99, rate(dki_latency_seconds_bucket[5m])) > 0.5
            for: 5m
            labels:
                severity: warning
            annotations:
                summary: "DKI P99 latency is high"
                description: "P99 latency is {{ $value }}s, threshold is 500ms"

          # 错误率告警
          - alert: DKIHighErrorRate
            expr: |
                sum(rate(dki_requests_total{status="error"}[5m])) /
                sum(rate(dki_requests_total[5m])) > 0.01
            for: 5m
            labels:
                severity: critical
            annotations:
                summary: "DKI error rate is high"
                description: "Error rate is {{ $value | humanizePercentage }}"

          # 缓存命中率告警
          - alert: DKILowCacheHitRate
            expr: |
                sum(rate(dki_cache_hits_total[5m])) /
                sum(rate(dki_requests_total{injection_mode="full_dki"}[5m])) < 0.5
            for: 15m
            labels:
                severity: warning
            annotations:
                summary: "DKI cache hit rate is low"
                description: "Cache hit rate is {{ $value | humanizePercentage }}"

          # α 分布异常
          - alert: DKIAbnormalAlphaDistribution
            expr: |
                sum(dki_alpha_distribution{bucket="0.8-1.0"}) /
                sum(dki_alpha_distribution) > 0.3
            for: 30m
            labels:
                severity: warning
            annotations:
                summary: "Too many high-alpha injections"
                description: "{{ $value | humanizePercentage }} of requests have α > 0.8"
```

### 2.6 Grafana Dashboard

```json
{
    "dashboard": {
        "title": "DKI Production Dashboard",
        "panels": [
            {
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sum(rate(dki_requests_total[1m])) by (injection_mode)",
                        "legendFormat": "{{injection_mode}}"
                    }
                ]
            },
            {
                "title": "Latency Distribution",
                "type": "heatmap",
                "targets": [
                    {
                        "expr": "sum(rate(dki_latency_seconds_bucket[5m])) by (le, stage)"
                    }
                ]
            },
            {
                "title": "Cache Hit Rate by Tier",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sum(rate(dki_cache_hits_total[5m])) by (tier)",
                        "legendFormat": "{{tier}}"
                    }
                ]
            },
            {
                "title": "Alpha Distribution",
                "type": "piechart",
                "targets": [
                    {
                        "expr": "sum(dki_alpha_distribution) by (bucket)"
                    }
                ]
            },
            {
                "title": "KV Cache Memory Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sum(dki_kv_cache_size_bytes) by (tier)",
                        "legendFormat": "{{tier}}"
                    }
                ]
            },
            {
                "title": "Error Rate",
                "type": "singlestat",
                "targets": [
                    {
                        "expr": "sum(rate(dki_requests_total{status='error'}[5m])) / sum(rate(dki_requests_total[5m]))"
                    }
                ],
                "thresholds": "0.001,0.01",
                "colors": ["green", "yellow", "red"]
            }
        ]
    }
}
```

---

## 第三部分：总结与路线图

### 3.1 研究阶段路线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    研究阶段路线图 (3-6 个月)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Month 1-2: 多模态位置编码实验                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Week 1-2: 基线建立 (Prefix KV + 小 α)                          │   │
│  │  Week 3-4: 负位置 + NoPE 实验                                    │   │
│  │  Week 5-6: Global Bias 实验                                      │   │
│  │  Week 7-8: 激进方案 (常量位置、锚点投影)                         │   │
│  │  输出: 技术报告，确定最优位置策略                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Month 3-4: 多模态 DKI 系统实现                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Week 1-2: 统一 Embedding 空间 (CLIP/SigLIP)                    │   │
│  │  Week 3-4: 跨模态 Memory Router                                  │   │
│  │  Week 5-6: VLM K/V 注入实现                                      │   │
│  │  Week 7-8: 端到端测试与优化                                      │   │
│  │  输出: 多模态 DKI 原型系统                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Month 5-6: 论文撰写与开源准备                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Week 1-4: 论文撰写 (实验、分析、结论)                           │   │
│  │  Week 5-6: 代码整理、文档完善                                    │   │
│  │  Week 7-8: 开源发布、社区推广                                    │   │
│  │  输出: 论文投稿、GitHub 开源                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 产品化路线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    产品化路线图 (6-12 个月)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: MVP (Month 1-3)                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 单机部署版本                                                  │   │
│  │  - 基本 API (add_memory, chat)                                   │   │
│  │  - 简单 KV Cache (内存)                                          │   │
│  │  - 基础监控                                                      │   │
│  │  目标: 100 QPS, 10K 用户                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 2: 分布式 (Month 4-6)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Kubernetes 部署                                               │   │
│  │  - 分布式 KV Cache (Redis + S3)                                  │   │
│  │  - 多模型支持                                                    │   │
│  │  - A/B 测试框架                                                  │   │
│  │  目标: 1K QPS, 100K 用户                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 3: 规模化 (Month 7-9)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 多区域部署                                                    │   │
│  │  - 自动扩缩容                                                    │   │
│  │  - 高级监控与告警                                                │   │
│  │  - SLA 保障 (99.9%)                                              │   │
│  │  目标: 10K QPS, 1M 用户                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 4: 企业级 (Month 10-12)                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 多租户隔离                                                    │   │
│  │  - 合规与审计                                                    │   │
│  │  - 私有化部署支持                                                │   │
│  │  - SDK (Python, TypeScript, Java)                                │   │
│  │  目标: 企业客户上线                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 关键决策点

| 决策点             | 选项                         | 推荐                   | 理由             |
| ------------------ | ---------------------------- | ---------------------- | ---------------- |
| **多模态位置策略** | Prefix / 负位置 / NoPE       | 先 Prefix，再探索 NoPE | 稳定性优先       |
| **KV Cache 存储**  | Redis / Memcached / 自研     | Redis Cluster          | 生态成熟         |
| **向量数据库**     | FAISS / Milvus / Qdrant      | Milvus                 | 分布式支持好     |
| **LLM 推理框架**   | vLLM / TensorRT-LLM / SGLang | vLLM                   | 社区活跃，易扩展 |
| **部署平台**       | K8s / ECS / 自建             | Kubernetes             | 标准化，可移植   |

---

这份文档涵盖了：

1. **多模态位置策略的详细分析**（稳定性、Token 效率、研究价值）
2. **研究阶段的实验路径**（4 个 Phase，代码框架）
3. **产品化架构设计**（服务拆分、缓存层级、部署方案）
4. **监控与运维**（Prometheus、Grafana、告警规则）
5. **路线图**（研究 3-6 个月，产品化 6-12 个月）
