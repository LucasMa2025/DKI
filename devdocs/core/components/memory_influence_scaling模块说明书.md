# Memory Influence Scaling (MIS) 模块说明书

> 文件路径: `DKI/dki/core/components/memory_influence_scaling.py`

## 1. 模块概述

本模块实现了 DKI 论文 Section 3.3 中提出的**记忆影响力缩放 (Memory Influence Scaling, MIS)** 机制。MIS 提供连续、可微的注入强度控制信号 α ∈ [0, 1]，是 DKI 区别于 RAG 二元决策的关键特性。

### 核心数学公式

$$\text{Attn}_\alpha(Q, K_{\text{aug}}, V_{\text{aug}}) = \text{softmax}\left(\frac{Q K_{\text{aug}}^T}{\sqrt{d}} + \alpha \cdot \text{bias}_{\text{mem}}\right) V_{\text{aug}}$$

其中 $\text{bias}_{\text{mem}} = [\log(\alpha), 0, 0, \ldots]$ 是记忆位置的 logit 偏置。

### 关键特性

| 特性 | 说明 |
|------|------|
| **优雅降级** | α → 0 平滑恢复为原始 LLM 行为 |
| **可控性** | 防止"注入失控"(记忆主导输出) |
| **可测试性** | α → output 映射支持系统化调试 |
| **可微性** | 支持基于梯度的 α 优化 |

### 与 RAG 的对比

- **RAG**: 二元决策 — 记忆要么完全包含在 prompt 中，要么完全不包含
- **DKI+MIS**: 连续控制 — 精确调节模型对记忆的关注程度

## 2. 核心类: MemoryInfluenceScaling (nn.Module)

### 2.1 初始化

```python
mis = MemoryInfluenceScaling(
    hidden_dim=4096,        # 模型隐藏维度
    use_learned_alpha=True, # 是否使用学习的 α 预测网络
    alpha_min=0.0,          # α 最小值
    alpha_max=1.0,          # α 最大值
    alpha_default=0.5,      # 默认 α 值
)
```

**α 预测网络结构 (当 `use_learned_alpha=True`):**

```
输入: [query_embedding (hidden_dim), memory_relevance (1), entropy (1)]
  │
  ├── Linear(hidden_dim + 2, 64)
  ├── ReLU()
  ├── Dropout(0.1)
  ├── Linear(64, 1)
  └── Sigmoid() → α_raw ∈ (0, 1)
```

**权重初始化:** Xavier 均匀初始化 + 零偏置，使初始预测趋向中等 α 值。

### 2.2 compute_alpha() — 计算注入强度

**两种模式:**

#### 模式 A: 启发式 α (use_learned_alpha=False)

```python
alpha = 0.5 * memory_relevance + 0.3 * min(entropy, 1.0) + 0.2
alpha = clamp(alpha, alpha_min, alpha_max)
```

简单线性组合，无需训练。

#### 模式 B: 学习的 α (use_learned_alpha=True)

**流程:**

```
输入: query_embedding, memory_relevance, entropy
  │
  ├── 1. 维度处理
  │     ├── 3D [batch, seq, hidden] → mean(dim=(0,1)) → [hidden]
  │     └── 2D [seq, hidden] → mean(dim=0) → [hidden]
  │
  ├── 2. 构建特征向量
  │     features = concat([query_embedding, relevance, entropy])
  │     shape: [hidden_dim + 2]
  │
  ├── 3. 预测 α (无梯度)
  │     alpha_raw = alpha_predictor(features.unsqueeze(0)).item()
  │
  └── 4. 缩放到 [alpha_min, alpha_max]
        alpha = alpha_min + alpha_raw × (alpha_max - alpha_min)
```

### 2.3 apply_scaling() — Logit Bias 缩放 (推荐方式)

**关键算法 — Pre-Softmax Logit Bias:**

```
输入: attention_logits [batch, heads, seq, total_len], mem_len, alpha
  │
  ├── alpha >= 1.0 或 mem_len == 0 → 直接返回 (无操作)
  │
  ├── alpha <= 0.0 → 记忆位置设为 -inf (完全屏蔽)
  │
  └── 0 < alpha < 1 → 应用 logit bias
        logit_bias = log(α + 1e-9)
        logits[:, :, :, :mem_len] += logit_bias
        logits[:, :, :, mem_len:] 不变
```

**数学原理:**

在 softmax 之前对记忆位置加 $\log(\alpha)$ 偏置，等效于在 softmax 之后将记忆的注意力权重乘以 α：

$$\text{softmax}(z_i + \log\alpha) = \frac{\alpha \cdot e^{z_i}}{\alpha \cdot \sum_{j \in \text{mem}} e^{z_j} + \sum_{k \notin \text{mem}} e^{z_k}}$$

这比直接在 softmax 后缩放更数学正确，因为后者会破坏概率归一化。

### 2.4 scale_kv_values() — Value-Only 缩放 (替代方式)

```
输入: key, value, alpha
  │
  ├── alpha >= 1.0 → 返回原始 (key, value)
  ├── alpha <= 0.0 → 返回 (key, value×0)    ← Key 不变!
  └── 0 < alpha < 1 → 返回 (key, value×α)   ← Key 不变!
```

**Value-Only Scaling 原则 (论文 Section 3.3):**

| 张量 | 是否缩放 | 原因 |
|------|---------|------|
| **Key** | ❌ 永不缩放 | Key 决定注意力寻址 (哪些位置被关注)，缩放 Key 会扭曲注意力分布，导致模型"忘记"记忆 token 的位置，产生不可预测行为 |
| **Value** | ✅ 按 α 缩放 | Value 决定输出内容 (关注后获取什么信息)，缩放 Value 仅调节记忆对输出的贡献强度，不影响注意力路由 |

这与 Engram 论文的 "Value-Only Gating" 设计一致：α 仅门控 V，K 保持不变以保留注意力寻址精度。

**关键区别:** 即使 `alpha=0.0`，Key 也保持原值返回 (仅 Value 被置零)。这确保注意力机制仍能"看到"记忆位置，只是不从中提取任何信息。

### 2.5 forward() — 前向传播

直接调用 `compute_alpha()`，用于 PyTorch 训练流程。

## 3. 三种缩放实现对比

| 方式 | 方法 | Key 处理 | Value 处理 | 优点 | 缺点 |
|------|------|---------|-----------|------|------|
| **Logit Bias** | `apply_scaling()` | 不修改 | 通过 logit 间接缩放 | 数学正确，保持概率归一化 | 需要访问 attention logits |
| **Value-Only Scaling** | `scale_kv_values()` | **不修改** | `value × α` | 实现简单，不需要修改注意力计算 | 近似方法，非精确 |
| **Attention Masking** | (未实现) | — | — | 最简单 | 二元决策，失去连续性 |

**共同原则:** 所有方式均遵循 **Key 不可变** 原则，仅通过调节 Value 或 logit 来控制记忆影响力。

**推荐:** 优先使用 Logit Bias 方式。

## 4. 配置依赖

| 配置路径 | 说明 |
|---------|------|
| `config.dki.mis.alpha_min` | α 最小值 |
| `config.dki.mis.alpha_max` | α 最大值 |
| `config.dki.mis.alpha_default` | 默认 α 值 |

## 5. 辅助类: IdentityProjection

消融实验用的恒等投影，直接返回输入的记忆嵌入，不做任何变换。

```python
class IdentityProjection(nn.Module):
    def forward(self, X_mem, X_user, **kwargs):
        return X_mem
```

## 6. 数据库交互

本模块不涉及数据库交互。

## 7. 注意事项

1. **alpha_min/alpha_max 的初始化**: 构造函数使用 `alpha_min if alpha_min is not None else config...` 判断，正确处理 `alpha_min=0.0` 的情况 (已修复)
2. **Value-Only Scaling 原则**: `scale_kv_values()` 永远不修改 Key 张量，即使 `alpha=0.0` 时 Key 也保持原值。这与 Engram 论文的设计一致，确保注意力寻址精度不受影响
3. **学习的 α 使用 `torch.no_grad()`**: `compute_alpha()` 中预测网络在 `no_grad` 下运行，意味着当前实现不支持端到端训练 α 预测器
4. **logit_bias 的数值稳定性**: 使用 `log(alpha + 1e-9)` 避免 `log(0)` 的数值问题
