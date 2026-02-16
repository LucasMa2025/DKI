# EngramInspiredFullAttentionInjector 模块说明书

> 源文件: `DKI/dki/core/injection/engram_inspired_injector.py`  
> 模块路径: `dki.core.injection.engram_inspired_injector`  
> 文件行数: 1107 行

---

## 1. 模块概述

`EngramInspiredFullAttentionInjector` 是 DKI Full-Attention 方案 C 的**增强版**，借鉴了 Engram 论文 (arXiv:2601.07372) 的记忆处理机制。相比原始 Full Attention 注入器，引入了三项核心改进:

1. **Context-Aware Per-Token Gating** (Engram Eq. 4): 逐 token 上下文门控
2. **Depthwise Causal Convolution Refinement** (Engram Eq. 5): 卷积精化
3. **Layer-Selective Injection Policy** (Engram Section 2.5): 层级选择性注入

---

## 2. 理论基础

### 2.1 Engram 论文核心思想

Engram 证明了: 将**静态模式存储**从**动态计算**中结构性分离，可以有效增加模型的"有效深度"(effective depth)，释放注意力容量用于全局推理。

### 2.2 DKI 映射

| Engram 概念 | DKI 映射 | 说明 |
|-------------|---------|------|
| N-gram 静态模式 | 用户偏好 | 稳定、重复、可缓存 |
| 全局上下文 | 会话历史 | 动态、上下文相关 |
| Lookup/Gating 卸载 | Gated K/V 注入 | 释放 context 窗口给推理 |
| 早期层注入 | 偏好注入层 | 事实性基底 |
| 中期层注入 | 历史注入层 | 推理上下文 |

### 2.3 与原始 DKI Full-Attention 对比

| 维度 | 原始 DKI | Engram-Inspired DKI |
|------|---------|---------------------|
| α 缩放粒度 | 全局统一 | 逐 token 上下文门控 |
| α 作用对象 | K 和 V 都缩放 | 仅缩放 V (Key 不变) |
| Value 处理 | 原始直出 | 卷积精化 + 残差 |
| 注入层策略 | 所有层 | 层级选择性 (早期+中期) |
| 门控归一化 | 无 | RMSNorm (梯度稳定) |
| 感受野 | 单 token | 扩展 (Conv kernel=4) |

---

## 3. 数据结构

### 3.1 GatingMode (门控模式)

| 模式 | 值 | 说明 |
|------|-----|------|
| `UNIFORM` | `"uniform"` | 原始 DKI: 全局统一 α |
| `CONTEXT_AWARE` | `"context_aware"` | Engram-inspired: 逐 token 上下文门控 |
| `HYBRID` | `"hybrid"` | 混合: 全局 α × 逐 token 门控 |

### 3.2 RefinementMode (Value 精化模式)

| 模式 | 值 | 说明 |
|------|-----|------|
| `NONE` | `"none"` | 不精化 |
| `CONV1D` | `"conv1d"` | Engram-inspired: 因果卷积 + SiLU |
| `LINEAR` | `"linear"` | 轻量线性映射 |

### 3.3 LayerInjectionPolicy (层级注入策略)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `preference_layer_ratios` | `[0.0, 0.1, 0.2]` | 偏好注入层 (前 20%) |
| `history_layer_ratios` | `[0.3, 0.5]` | 历史注入层 (30%-50%) |
| `per_layer_gating` | `True` | 每层独立门控参数 |

**层级映射示例 (30 层模型):**
```
偏好注入层: [0, 3, 6]   (层 0, 3, 6 — 早期层)
历史注入层: [9, 15]      (层 9, 15 — 中期层)
其余层:     无注入        (层 1,2,4,5,7,8,10-14,16-29)
```

### 3.4 EngramInspiredConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用 |
| `gating_mode` | `CONTEXT_AWARE` | 门控模式 |
| `gating_temperature` | `1.0` | 门控温度 |
| `gating_bias` | `0.0` | 门控偏置 |
| `use_rmsnorm` | `True` | 使用 RMSNorm |
| `refinement_mode` | `CONV1D` | 精化模式 |
| `conv_kernel_size` | `4` | 卷积核大小 |
| `conv_dilation` | `1` | 膨胀率 |
| `layer_policy` | `LayerInjectionPolicy()` | 层级策略 |
| `preference_position_start` | `-100` | 偏好起始位置 |
| `preference_base_alpha` | `0.4` | 偏好基础 α |
| `preference_max_tokens` | `100` | 偏好最大 token |
| `history_position_start` | `-500` | 历史起始位置 |
| `history_base_alpha` | `0.3` | 历史基础 α |
| `history_max_tokens` | `400` | 历史最大 token |
| `history_max_messages` | `10` | 最大消息数 |
| `max_total_kv_tokens` | `600` | K/V 总 token 上限 |
| `fallback_to_stable` | `True` | 超限时降级 |

### 3.5 EngramInjectionResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | 是否成功 |
| `layer_kv` | `Dict[int, (Tensor, Tensor)]` | 每层的 K/V 数据 |
| `per_token_gates` | `Optional[Dict]` | 逐 token 门控值 |
| `avg_preference_gate` | `float` | 偏好平均门控值 |
| `avg_history_gate` | `float` | 历史平均门控值 |
| `preference_tokens` | `int` | 偏好 token 数 |
| `history_tokens` | `int` | 历史 token 数 |
| `preference_layers` | `List[int]` | 偏好注入层 |
| `history_layers` | `List[int]` | 历史注入层 |
| `gating_mode` | `str` | 门控模式 |
| `refinement_mode` | `str` | 精化模式 |
| `compute_time_ms` | `float` | 总计算耗时 |
| `gating_time_ms` | `float` | 门控计算耗时 |
| `refinement_time_ms` | `float` | 精化计算耗时 |

---

## 4. 核心模块详解

### 4.1 RMSNorm (Root Mean Square Layer Normalization)

```python
class RMSNorm(nn.Module):
    """
    Engram 在门控计算中使用 RMSNorm 而非 LayerNorm:
    - 计算更轻量 (无需减均值)
    - 梯度更稳定
    - 与 DeepSeek 等模型架构一致
    """
    def forward(self, x):
        rms = sqrt(mean(x^2, dim=-1) + eps)
        return x / rms * weight
```

**数学公式:**
```
RMSNorm(x) = x / sqrt(mean(x_i^2) + eps) * gamma
```

### 4.2 ContextAwareGating (上下文感知门控) — Engram Eq. 4

**核心公式:**
```
alpha_i = sigma(RMSNorm(h_query) . RMSNorm(K_mem[i]) / (sqrt(d_head) * temperature) + bias)
```

其中:
- `h_query`: 查询隐藏状态的池化表示 (mean pooling over query tokens)
- `K_mem[i]`: 第 i 个记忆 token 的 Key 向量
- `sigma`: Sigmoid 函数
- `d_head`: 注意力头维度
- `temperature`: 温度参数 (控制门控锐度)
- `bias`: 偏置 (正值倾向开放门控)

**计算流程:**

```
ContextAwareGating.forward(query_hidden, memory_keys, base_alpha)
  │
  ├─ 1. 归一化
  │   ├─ q = RMSNorm(query_hidden)   # [batch, heads, head_dim]
  │   └─ k = RMSNorm(memory_keys)    # [batch, heads, mem_len, head_dim]
  │
  ├─ 2. 计算对齐分数
  │   ├─ q = q.unsqueeze(2)          # [batch, heads, 1, head_dim]
  │   ├─ scores = sum(q * k, dim=-1) # [batch, heads, mem_len]
  │   └─ scores = scores * scale + bias
  │
  ├─ 3. Sigmoid 门控
  │   └─ gates = sigmoid(scores)     # [batch, heads, mem_len]
  │
  ├─ 4. 基础 alpha 缩放
  │   └─ gates = gates * base_alpha
  │
  └─ 5. 返回
      └─ gates.unsqueeze(-1)         # [batch, heads, mem_len, 1]
```

**为什么逐 token 门控优于全局 alpha?**

示例: 用户偏好 = "素食主义者，住在北京，喜欢辣食"

| 查询 | "推荐一家餐厅" | "明天天气怎样" |
|------|---------------|---------------|
| "素食" token 门控 | α ≈ 0.8 (高相关) | α ≈ 0.1 (低相关) |
| "北京" token 门控 | α ≈ 0.6 (中相关) | α ≈ 0.7 (高相关) |
| "辣食" token 门控 | α ≈ 0.7 (高相关) | α ≈ 0.05 (极低) |

全局统一 α 无法实现这种细粒度控制。

### 4.3 ConvolutionRefinement (卷积精化) — Engram Eq. 5

**核心公式:**
```
Y = SiLU(Conv1D(RMSNorm(V_gated))) + V_gated
```

**设计要点:**

| 特性 | 说明 |
|------|------|
| Causal Conv1D | 确保每个位置只看到之前的记忆 token (维护因果性) |
| Depthwise | 每个头独立卷积，不跨头混合 (groups=head_dim) |
| SiLU 激活 | 平滑非线性，优于 ReLU |
| 残差连接 | 初始时卷积权重为 0，确保 Y 约等于 V_gated (恒等映射) |
| Kernel=4 | 扩展感受野到 4 个相邻 token |

**计算流程:**

```
ConvolutionRefinement.forward(v_gated)
  │  v_gated: [batch, heads, mem_len, head_dim]
  │
  ├─ 1. RMSNorm
  │   └─ v_normed = RMSNorm(v_gated)
  │
  ├─ 2. Reshape for Conv1D
  │   └─ v_conv: [batch*heads, head_dim, mem_len]
  │
  ├─ 3. Causal Left-Padding
  │   └─ pad(v_conv, (causal_padding, 0))
  │       causal_padding = (kernel_size - 1) * dilation = 3
  │
  ├─ 4. Depthwise Conv1D
  │   └─ v_conv = Conv1D(v_conv)  # groups=head_dim
  │
  ├─ 5. SiLU Activation
  │   └─ v_conv = SiLU(v_conv) = v_conv * sigmoid(v_conv)
  │
  ├─ 6. Reshape Back
  │   └─ v_conv: [batch, heads, mem_len, head_dim]
  │
  └─ 7. Residual Connection
      └─ return v_conv + v_gated
```

**为什么需要卷积精化?**

相邻的记忆 token 之间存在局部依赖:
```
"用户" "喜欢" "辣食"  ← 三个 token 构成一个完整语义
```

单独的 K/V token 无法捕获这些 n-gram 模式。轻量卷积以最小开销扩展感受野。

**初始化策略:**
```python
# 卷积权重初始化为 0
nn.init.zeros_(self.conv.weight)
nn.init.zeros_(self.conv.bias)
# 确保初始时 Y = 0 + V_gated = V_gated (恒等映射)
# 训练过程中逐渐学习有用的局部模式
```

---

## 5. inject() 主流程

```
EngramInspiredFullAttentionInjector.inject(model_adapter, preference_text, history_messages, query)
  │
  ├─ Phase 1: 记忆收集
  │   ├─ 估算 token 数量
  │   ├─ 安全检查 (total > max_total_kv_tokens?)
  │   │   ├─ fallback_to_stable=True → 触发降级
  │   │   └─ fallback_to_stable=False → 截断历史
  │   ├─ model_adapter.compute_kv(preference_text) → preference_kv
  │   └─ model_adapter.compute_kv(history_text) → history_kv
  │
  ├─ Phase 2: 上下文感知门控 (Engram Eq. 4)
  │   ├─ 计算查询表示
  │   │   └─ _compute_query_representation(model_adapter, query)
  │   │       ├─ model.embed_text(query) → mean_pool → [1, heads, head_dim]
  │   │       └─ 或 model.compute_kv(query) → 取第一层 Key → mean_pool
  │   │
  │   ├─ 偏好门控: _apply_gating(preference_kv, query_hidden, base_alpha=0.4)
  │   │   ├─ CONTEXT_AWARE: gates = sigma(norm(h_q) . norm(K_pref) / sqrt(d))
  │   │   │                 V_gated = V * gates * base_alpha
  │   │   ├─ HYBRID: gates = sigma(...) * base_alpha
  │   │   └─ UNIFORM: V_gated = V * base_alpha
  │   │
  │   └─ 历史门控: _apply_gating(history_kv, query_hidden, base_alpha=0.3)
  │       └─ (同上)
  │
  ├─ Phase 3: 卷积精化 (Engram Eq. 5)
  │   ├─ _apply_refinement(gated_pref_kv)
  │   │   └─ V_refined = SiLU(Conv1D(RMSNorm(V_gated))) + V_gated
  │   └─ _apply_refinement(gated_hist_kv)
  │       └─ (同上)
  │
  ├─ Phase 4: 层级组装
  │   └─ _assemble_layer_kv(preference_kv, history_kv)
  │       ├─ 偏好 → 早期层 (e.g., [0, 3, 6])
  │       ├─ 历史 → 中期层 (e.g., [9, 15])
  │       └─ 同层有偏好+历史 → torch.cat 拼接
  │
  └─ 返回 EngramInjectionResult
      ├─ layer_kv: {0: (K, V), 3: (K, V), 6: (K, V), 9: (K, V), 15: (K, V)}
      ├─ avg_preference_gate: 0.45
      ├─ avg_history_gate: 0.32
      └─ compute_time_ms: 25.3
```

---

## 6. 层级组装算法

```python
def _assemble_layer_kv(preference_kv, history_kv):
    """
    按照层级策略组装 per-layer K/V
    
    遵循 Engram 的层级选择性原则:
    - 偏好 → 早期层 (卸载"静态模式重建")
    - 历史 → 中期层 (提供"推理上下文")
    
    示例 (30 层模型):
      偏好注入层: [0, 3, 6]
      历史注入层: [9, 15]
      
      layer_kv = {
          0:  (K_pref, V_pref_gated_refined),   # 早期层: 偏好
          3:  (K_pref, V_pref_gated_refined),   # 早期层: 偏好
          6:  (K_pref, V_pref_gated_refined),   # 早期层: 偏好
          9:  (K_hist, V_hist_gated_refined),   # 中期层: 历史
          15: (K_hist, V_hist_gated_refined),   # 中期层: 历史
      }
      
      其余 25 层: 无注入 (原始注意力)
    """
```

**层级映射逻辑:**
```
target_layer = int(ratio * num_layers)
source_layer = min(target_layer, num_available_layers - 1)
```

如果同一层同时有偏好和历史 (层级重叠):
```python
layer_kv[target_layer] = (
    torch.cat([K_pref, K_hist], dim=2),
    torch.cat([V_pref, V_hist], dim=2),
)
```

---

## 7. 查询表示计算

```
_compute_query_representation(model_adapter, query)
  │
  ├─ 方式 1: model_adapter.embed_text(query)
  │   ├─ embedding: [1, seq_len, hidden_dim]
  │   ├─ mean_pool: [1, hidden_dim]
  │   └─ reshape: [1, num_heads, head_dim]
  │
  ├─ 方式 2 (Fallback): model_adapter.compute_kv(query)
  │   ├─ 取第一层 Key: [1, heads, seq, head_dim]
  │   └─ mean_pool(dim=2): [1, heads, head_dim]
  │
  └─ 失败 → 返回 None (降级为 UNIFORM 模式)
```

---

## 8. 三种门控模式对比

### 8.1 UNIFORM (原始 DKI)

```
V_gated = V * base_alpha
```
- 所有记忆 token 使用相同的 alpha
- 最简单，无额外参数

### 8.2 CONTEXT_AWARE (Engram-inspired)

```
gates = sigma(RMSNorm(h_query) . RMSNorm(K_mem) / sqrt(d) + bias)
V_gated = V * gates * base_alpha
```
- 每个记忆 token 独立的门控值
- base_alpha 作为上限
- 需要 RMSNorm 参数

### 8.3 HYBRID (混合)

```
gates = sigma(RMSNorm(h_query) . RMSNorm(K_mem) / sqrt(d) + bias)
V_gated = V * gates * base_alpha
```
- 逐 token 门控 × 全局 alpha
- 两级控制: 全局强度 + 局部相关性

---

## 9. 模块参数量分析

```
get_module_info() 返回:

ContextAwareGating 参数:
  - query_norm.weight: head_dim (128)
  - key_norm.weight: head_dim (128)
  总计: 256 参数 ≈ 1 KB

ConvolutionRefinement 参数:
  - norm.weight: head_dim (128)
  - conv.weight: head_dim × 1 × kernel_size (128 × 1 × 4 = 512)
  - conv.bias: head_dim (128)
  总计: 768 参数 ≈ 3 KB

总计: ~1024 参数 ≈ 4 KB (极轻量)
```

---

## 10. 注入流程可视化

```
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
│     ├─ 偏好门控: alpha_i = sigma(norm(h_q).norm(K_pref[i])/sqrt(d))│
│     ├─ 历史门控: alpha_j = sigma(norm(h_q).norm(K_hist[j])/sqrt(d))│
│     └─ Value 门控: V_pref'[i] = alpha_i * V_pref[i]              │
│                    V_hist'[j] = alpha_j * V_hist[j]               │
│                                                                   │
│  3. 卷积精化 (Engram Eq. 5)                                       │
│     ├─ V_pref'' = SiLU(Conv1D(RMSNorm(V_pref'))) + V_pref'      │
│     └─ V_hist'' = SiLU(Conv1D(RMSNorm(V_hist'))) + V_hist'      │
│                                                                   │
│  4. 层级选择性注入 (Engram Section 2.5)                            │
│     ├─ 早期层 (L0, L3, L6): 注入偏好 K/V                         │
│     │   [K_pref; K_input] x [V_pref''; V_input]                  │
│     ├─ 中期层 (L9, L15): 注入历史 K/V                             │
│     │   [K_hist; K_input] x [V_hist''; V_input]                  │
│     └─ 其余层: 原始注意力 (无注入)                                 │
│                                                                   │
│  5. 注意力计算                                                     │
│     ├─ Attn_l = softmax(Q_l . [K_mem; K_input]^T / sqrt(d))      │
│     │           x [V_gated_refined; V_input]                      │
│     └─ 注意: Key 不缩放 (保持寻址准确性)                          │
│           Value 已通过门控 + 精化调制                               │
│                                                                   │
│  6. 残差整合                                                       │
│     H^(l) <- H^(l) + Attn_l (标准残差连接)                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 11. YAML 配置示例

```yaml
dki:
  injection_strategy: engram_inspired
  
  engram_inspired:
    enabled: true
    
    # 门控配置
    gating_mode: context_aware
    gating_temperature: 1.0
    gating_bias: 0.0
    use_rmsnorm: true
    
    # Value 精化配置
    refinement_mode: conv1d
    conv_kernel_size: 4
    conv_dilation: 1
    
    # 层级注入策略
    layer_policy:
      preference_layer_ratios: [0.0, 0.1, 0.2]
      history_layer_ratios: [0.3, 0.5]
      per_layer_gating: true
    
    # 偏好配置
    preference:
      position_start: -100
      base_alpha: 0.4
      max_tokens: 100
    
    # 历史配置
    history:
      position_start: -500
      base_alpha: 0.3
      max_tokens: 400
      max_messages: 10
    
    # 安全设置
    safety:
      max_total_kv_tokens: 600
      fallback_to_stable: true
      log_attention_patterns: true
```

---

## 12. 研究数据收集

### 12.1 注入日志

每次成功注入后记录:

```json
{
  "timestamp": 1708000000.0,
  "query": "推荐一家餐厅...",
  "gating_mode": "context_aware",
  "refinement_mode": "conv1d",
  "preference_tokens": 50,
  "history_tokens": 200,
  "avg_preference_gate": 0.45,
  "avg_history_gate": 0.32,
  "preference_layers": [0, 3, 6],
  "history_layers": [9, 15],
  "gating_time_ms": 5.2,
  "refinement_time_ms": 3.1,
  "compute_time_ms": 25.3
}
```

### 12.2 统计指标

| 指标 | 说明 |
|------|------|
| `total_injections` | 总注入次数 |
| `successful_injections` | 成功次数 |
| `fallback_count` | 降级次数 |
| `avg_preference_gate` | 偏好平均门控值 |
| `avg_history_gate` | 历史平均门控值 |
| `avg_compute_time_ms` | 平均计算耗时 |

---

## 13. 安全机制

与 Full Attention 注入器相同:
- Token 上限检查 (max_total_kv_tokens)
- 自动降级到 Stable 策略
- 异常捕获和错误报告
- 卷积权重零初始化 (初始恒等映射)
