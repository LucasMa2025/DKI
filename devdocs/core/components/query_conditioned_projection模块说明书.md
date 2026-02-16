# Query-Conditioned Projection 模块说明书

> 文件路径: `DKI/dki/core/components/query_conditioned_projection.py`

## 1. 模块概述

本模块实现了 DKI 系统的**查询条件化记忆投影 (Query-Conditioned Memory Projection)**，采用 FiLM (Feature-wise Linear Modulation) 风格的调制机制，根据当前用户查询自适应地调整记忆表示。

### 核心设计原则: 记忆中心投影

投影是**记忆中心 (Memory-Centric)** 的，而非查询中心的：
- 查询仅**调制 (Modulate)** 记忆，不重新编码记忆语义
- 与 Cross-Attention 不同，后者由查询主动重新加权记忆
- 投影只能调整已有信息的**强调程度**，不能"幻想"新的记忆内容

### 结构约束 (隐式正则化)

| 约束 | 说明 |
|------|------|
| γ 有界 | 通过初始化和训练动态约束 ‖γ‖₂ |
| β 零均值 | 初始化为零，偏移应保持较小 |
| 残差主导 | X_proj ≈ X_mem + ε·f(X_mem, X_user) |

### 与替代方案的对比

| 方案 | 参数量 | 计算开销 | 适用性 |
|------|--------|---------|--------|
| Full Cross-Attention | ~数百万 | 高 | 过于昂贵 |
| Adapter Explosion | ~数十万/层 | 中 | 参数过多 |
| **FiLM Projection (本方案)** | **~128KB** | **+5-10ms** | **低秩、高效** |

## 2. 核心类: QueryConditionedProjection (nn.Module)

### 2.1 网络结构

```
输入:
  X_mem: [batch, mem_len, hidden_dim]    (记忆嵌入)
  X_user: [batch, user_len, hidden_dim]  (用户输入嵌入)

                    X_user
                      │
                 mean pooling
                      │
                   q_ctx [batch, hidden_dim]
                   /         \
            gamma_net       beta_net
            Linear          Linear
           (hidden→rank)   (hidden→rank)
                |               |
                γ               β
                [batch, rank]   [batch, rank]

  X_mem ──→ W_mem ──→ X_mem_low [batch, mem_len, rank]
                          │
                    FiLM: γ * X_mem_low + β
                          │
                       Dropout
                          │
                       proj_out (rank → hidden_dim)
                          │
                    X_mem + proj_out  (残差连接)
                          │
                      LayerNorm
                          │
                       X_mem_proj [batch, mem_len, hidden_dim]
```

### 2.2 初始化参数

```python
qcp = QueryConditionedProjection(
    hidden_dim=4096,  # 模型隐藏维度
    rank=64,          # 低秩投影维度
    dropout=0.1,      # Dropout 率
)
```

**网络组件:**

| 组件 | 形状 | 说明 |
|------|------|------|
| `W_mem` | `[hidden_dim, rank]` | 低秩投影矩阵 (可学习参数) |
| `gamma_net` | `Linear(hidden_dim, rank)` | FiLM 缩放参数生成器 |
| `beta_net` | `Linear(hidden_dim, rank)` | FiLM 偏移参数生成器 |
| `proj_out` | `Linear(rank, hidden_dim)` | 输出投影 (恢复维度) |
| `dropout` | `Dropout(0.1)` | 正则化 |
| `layer_norm` | `LayerNorm(hidden_dim)` | 层归一化 |

**权重初始化策略:**

| 组件 | 初始化 | 目的 |
|------|--------|------|
| `gamma_net.weight` | 零 | 初始 γ ≈ 1 (恒等缩放) |
| `gamma_net.bias` | 全一 | 初始 γ = 1 |
| `beta_net.weight` | 零 | 初始 β = 0 (无偏移) |
| `beta_net.bias` | 零 | 初始 β = 0 |
| `proj_out.weight` | Xavier 均匀 | 标准初始化 |
| `proj_out.bias` | 零 | 标准初始化 |

> 初始化使得网络初始行为接近恒等变换：X_proj ≈ X_mem

### 2.3 forward() — 前向传播

**详细流程:**

```
输入: X_mem [batch?, mem_len, hidden_dim], X_user [batch?, user_len, hidden_dim]
  │
  ├── 1. 批次维度处理
  │     如果输入是 2D (无 batch)，自动添加 batch=1
  │
  ├── 2. 查询上下文摘要
  │     q_ctx = X_user.mean(dim=1)  → [batch, hidden_dim]
  │     (对用户 token 做平均池化)
  │
  ├── 3. 生成 FiLM 调制参数
  │     γ = gamma_net(q_ctx)  → [batch, rank]
  │     β = beta_net(q_ctx)   → [batch, rank]
  │
  ├── 4. 低秩投影
  │     X_mem_low = X_mem @ W_mem  → [batch, mem_len, rank]
  │
  ├── 5. FiLM 调制
  │     γ = γ.unsqueeze(1)  → [batch, 1, rank]  (广播)
  │     β = β.unsqueeze(1)  → [batch, 1, rank]
  │     X_modulated = γ * X_mem_low + β  → [batch, mem_len, rank]
  │
  ├── 6. Dropout
  │     X_modulated = dropout(X_modulated)
  │
  ├── 7. 输出投影
  │     X_proj = proj_out(X_modulated)  → [batch, mem_len, hidden_dim]
  │
  ├── 8. 残差连接 + LayerNorm
  │     X_proj = LayerNorm(X_mem + X_proj)
  │
  └── 9. 返回 X_proj (如果输入无 batch 则去除 batch 维)
```

**关键算法 — FiLM 调制:**

$$X_{\text{proj}} = \text{LayerNorm}\left(X_{\text{mem}} + W_{\text{out}} \cdot \text{Dropout}\left(\gamma(q) \odot (X_{\text{mem}} W_{\text{mem}}) + \beta(q)\right)\right)$$

其中:
- $\gamma(q) = W_\gamma \cdot \text{mean}(X_{\text{user}}) + b_\gamma$ — 查询条件化缩放
- $\beta(q) = W_\beta \cdot \text{mean}(X_{\text{user}}) + b_\beta$ — 查询条件化偏移
- $\odot$ 表示逐元素乘法

**计算复杂度分析:**

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| Mean pooling | O(user_len × hidden_dim) | 查询摘要 |
| gamma_net/beta_net | O(hidden_dim × rank) | FiLM 参数 |
| W_mem 投影 | O(mem_len × hidden_dim × rank) | 低秩投影 |
| FiLM 调制 | O(mem_len × rank) | 逐元素操作 |
| proj_out | O(mem_len × rank × hidden_dim) | 恢复维度 |
| **总计** | **O(mem_len × hidden_dim × rank)** | 主导项 |

### 2.4 辅助方法

| 方法 | 说明 |
|------|------|
| `get_parameter_count()` | 返回可训练参数总数 |
| `get_memory_overhead()` | 返回人类可读的内存开销 (如 "128.5 KB") |

**参数量估算 (hidden_dim=4096, rank=64):**

| 组件 | 参数量 |
|------|--------|
| W_mem | 4096 × 64 = 262,144 |
| gamma_net | 4096 × 64 + 64 = 262,208 |
| beta_net | 4096 × 64 + 64 = 262,208 |
| proj_out | 64 × 4096 + 4096 = 266,240 |
| layer_norm | 4096 × 2 = 8,192 |
| **总计** | **~1,060,992 (~4.05 MB)** |

## 3. 辅助类: IdentityProjection

消融实验用的恒等投影，直接返回原始记忆嵌入。

```python
class IdentityProjection(nn.Module):
    def forward(self, X_mem, X_user, **kwargs):
        return X_mem
```

用于对比实验：有/无查询条件化投影对 DKI 效果的影响。

## 4. 配置依赖

| 配置路径 | 说明 |
|---------|------|
| `config.dki.projection.rank` | 低秩投影维度 |
| `config.dki.projection.dropout` | Dropout 率 |

## 5. 数据库交互

本模块不涉及数据库交互。

## 6. 注意事项

1. **rank 和 dropout 的 or 逻辑**: 与 MIS 模块相同，`self.rank = rank or config...` 在 `rank=0` 时会回退到配置值
2. **Mean Pooling 的局限性**: 查询上下文通过简单平均池化获取，可能丢失位置信息。后续可考虑使用 [CLS] token 或加权池化
3. **残差连接的重要性**: 确保初始行为接近恒等变换，训练过程中逐步学习有意义的调制
4. **return_modulation 参数**: 设为 True 时返回 (X_proj, gamma, beta)，用于调试和可视化调制参数
