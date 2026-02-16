# Dual-Factor Gating 模块说明书

> 文件路径: `DKI/dki/core/components/dual_factor_gating.py`

## 1. 模块概述

本模块实现了 DKI 系统的**双因子门控机制 (Dual-Factor Gating)**，通过同时考量模型不确定性 (Entropy) 和记忆相关性 (Relevance) 两个因子来决定是否注入记忆以及注入强度。

### 核心设计理念

单因子门控（仅基于模型不确定性）不足以做出正确的注入决策：
- 高不确定性可能是开放性问题的固有属性，而非需要记忆补充
- 低不确定性时仍可能需要记忆来丰富上下文

因此采用双因子联合决策：

| 不确定性 | 相关性 | 决策 | 说明 |
|---------|--------|------|------|
| 高 | 高 | ✅ 注入 (高 α) | 模型不确定且有相关记忆 |
| 高 | 低 | ❌ 不注入 | 开放性问题，无相关记忆 |
| 低 | 高 | ✅ 注入 (中 α) | 模型自信但可用记忆丰富 |
| 低 | 低 | ❌ 不注入 | 模型自信且无需记忆 |

### 重要限制说明

注意力熵作为不确定性的**启发式代理 (Heuristic Proxy)**，而非严格的不确定性估计器：
- **事实问答**: 高熵可能表示认知不确定性 (epistemic uncertainty)
- **头脑风暴/创意**: 高熵是任务固有属性
- **摘要生成**: 分散注意力是正常的，不代表不确定

## 2. 数据结构

### 2.1 GatingDecision

门控决策结果数据类。

| 字段 | 类型 | 说明 |
|------|------|------|
| `should_inject` | `bool` | 是否应该注入记忆 |
| `alpha` | `float` | 注入强度 (0.0-1.0) |
| `entropy` | `float` | 模型不确定性 (归一化后 0-1) |
| `relevance_score` | `float` | Top-1 记忆相关性分数 |
| `margin` | `float` | Top-1 与 Top-2 的分数差距 |
| `memories` | `List[MemorySearchResult]` | 检索到的记忆列表 |
| `reasoning` | `str` | 决策推理说明 |

## 3. 核心类: DualFactorGating

### 3.1 初始化

```python
gating = DualFactorGating(
    entropy_threshold=None,     # 从配置加载
    relevance_threshold=None,   # 从配置加载
    use_margin=True,            # 是否使用 margin 调节
    margin_weight=0.3,          # margin 权重
)
```

**依赖:**
- `ConfigLoader` — 加载 `config.dki.gating` 配置
- `BaseModelAdapter` — 模型适配器 (计算熵)
- `MemoryRouter` — 记忆路由器 (检索相关记忆)

### 3.2 compute_prefill_entropy() — 计算预填充熵

**流程:**

```
输入: model, query, layer_idx=3
  │
  ├── 1. 调用 model.compute_prefill_entropy(query, layer_idx)
  │      获取指定层的注意力熵 (原始值)
  │
  ├── 2. Sigmoid 归一化到 [0, 1]
  │      normalized = 1 / (1 + exp(-entropy / 10))
  │
  └── 3. 返回归一化熵值
```

**关键算法 — Sigmoid 归一化:**

$$H_{\text{norm}} = \sigma\left(\frac{H_{\text{raw}}}{10}\right) = \frac{1}{1 + e^{-H_{\text{raw}}/10}}$$

- 除以 10 是温度参数，控制归一化的灵敏度
- 输出范围 (0, 1)，中心点在 entropy=0 处为 0.5
- 不同模型的原始熵值范围不同，此归一化提供统一尺度

### 3.3 should_inject() — 核心门控决策

**完整流程:**

```
输入: model, query, router, top_k=5
  │
  ├── 1. 计算不确定性因子
  │     entropy = compute_prefill_entropy(model, query)
  │
  ├── 2. 计算相关性因子
  │     top_results = router.search(query, top_k)
  │     ├── 无结果 → 返回 (不注入, α=0)
  │     ├── sim_top1 = top_results[0].score
  │     └── margin = top1.score - top2.score (如有 top2)
  │
  ├── 3. 双因子决策
  │     entropy_condition = entropy > entropy_threshold
  │     relevance_condition = sim_top1 > relevance_threshold
  │     │
  │     ├── relevance=True + entropy=True  → 注入 (强)
  │     ├── relevance=True + entropy=False → 注入 (中)
  │     ├── relevance=False + entropy=True → 不注入
  │     └── relevance=False + entropy=False → 不注入
  │
  ├── 4. 计算连续 α (仅在注入时)
  │     alpha_input = 2.0 × (entropy - threshold_e)
  │                 + 1.5 × (sim_top1 - threshold_r)
  │                 + 0.3 × margin  (可选)
  │     alpha = sigmoid(alpha_input)
  │     alpha = max(0.1, alpha)  # 最低 0.1
  │
  └── 5. 返回 GatingDecision
```

**关键算法 — α 计算公式:**

$$\alpha = \max\left(0.1,\; \sigma\left(w_e \cdot (H - \tau_e) + w_r \cdot (s_1 - \tau_r) + w_m \cdot m\right)\right)$$

其中:
- $w_e = 2.0$ (不确定性权重)
- $w_r = 1.5$ (相关性权重)
- $w_m = 0.3$ (margin 权重)
- $\tau_e$ = entropy_threshold
- $\tau_r$ = relevance_threshold
- $H$ = 归一化熵
- $s_1$ = Top-1 相似度
- $m$ = Top-1 与 Top-2 的分数差距

**设计要点:**
- 不确定性权重 (2.0) > 相关性权重 (1.5)：因为不确定性是注入的主要驱动力
- margin 用于衡量检索置信度：margin 大说明 Top-1 显著优于其他结果
- 最低 α=0.1：一旦决定注入，至少保证最小影响力

### 3.4 force_inject() — 强制注入

绕过门控检查，直接以指定 α 值注入。用于调试和特殊场景。

```python
decision = gating.force_inject(
    router=router,
    query="用户查询",
    alpha=1.0,        # 固定注入强度
    top_k=5,
    threshold=-1.0,   # 极低阈值，确保返回结果
)
```

### 3.5 update_thresholds() — 动态更新阈值

支持运行时调整阈值，用于 A/B 测试和在线调优。

## 4. 配置依赖

| 配置路径 | 说明 |
|---------|------|
| `config.dki.gating.entropy_threshold` | 熵阈值 |
| `config.dki.gating.relevance_threshold` | 相关性阈值 |
| `config.dki.gating.margin_weight` | margin 权重 |

## 5. 数据库交互

本模块不直接与数据库交互。记忆检索通过 `MemoryRouter` 完成，后者使用 FAISS 内存索引。

## 6. 注意事项

1. **熵计算依赖模型**: `compute_prefill_entropy` 调用 `model.compute_prefill_entropy()`，需要模型适配器实现此方法
2. **Sigmoid 归一化的温度参数**: 硬编码为 10，不同模型可能需要不同的温度值
3. **权重硬编码**: `entropy_weight=2.0` 和 `relevance_weight=1.5` 在构造函数中硬编码，建议后续移至配置
4. **margin 的意义**: 当 Top-1 和 Top-2 分数接近时 (margin 小)，说明检索结果不够确定，应降低 α
