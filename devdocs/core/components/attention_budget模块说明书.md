# Attention Budget Analysis 模块说明书

> 文件路径: `DKI/dki/core/components/attention_budget.py`

## 1. 模块概述

本模块实现了 DKI 论文 Section 3.2 中提出的**注意力预算重分配假说 (Attention Budget Reallocation Hypothesis)**。核心思想是：DKI 通过将记忆从 Token Budget（硬约束）转移到 Attention Budget（软约束），在不损失信息的前提下释放上下文窗口空间。

### 核心假说

$$\frac{\partial \text{TaskSuccess}}{\partial B_t^{\text{free}}} > \frac{\partial \text{Latency}}{\partial B_a}$$

在推理密集型任务中，释放 Token Budget 的边际收益超过增加 Attention Budget 的边际成本。

## 2. 数据结构

### 2.1 BudgetAnalysis

预算分析结果数据类，对比 RAG 与 DKI 的资源利用情况。

| 字段 | 类型 | 说明 |
|------|------|------|
| `context_window` | `int` | 上下文窗口大小 |
| `user_tokens` | `int` | 用户输入 token 数 (n_u) |
| `memory_tokens` | `int` | 记忆 token 数 (n_m) |
| `rag_token_budget_used` | `int` | RAG 使用的 token 预算 = n_m + n_u |
| `rag_token_budget_free` | `int` | RAG 剩余 token 预算 |
| `rag_attention_budget` | `int` | RAG 注意力预算 = (n_m + n_u)² |
| `dki_token_budget_used` | `int` | DKI 使用的 token 预算 = n_u |
| `dki_token_budget_free` | `int` | DKI 剩余 token 预算 |
| `dki_attention_budget` | `int` | DKI 注意力预算 = n_u × (n_m + n_u) |
| `token_budget_saved` | `int` | DKI 节省的 token 数 |
| `attention_budget_increase` | `int` | DKI 增加的注意力计算量 |
| `token_efficiency_gain` | `float` | token 效率提升百分比 |
| `attention_overhead_ratio` | `float` | 注意力开销比率 |

### 2.2 LatencyBreakdown

DKI 延迟分解数据类，记录各阶段耗时。

| 字段 | 类型 | 说明 |
|------|------|------|
| `router_ms` | `float` | 记忆路由耗时 |
| `gating_ms` | `float` | 门控决策耗时 |
| `kv_compute_ms` | `float` | K/V 计算耗时 |
| `kv_load_ms` | `float` | K/V 加载耗时 |
| `projection_ms` | `float` | 投影计算耗时 |
| `prefill_ms` | `float` | 预填充耗时 |
| `decode_ms` | `float` | 解码耗时 |
| `total_ms` | `float` | 总耗时 |
| `cache_hit` | `bool` | 是否命中缓存 |
| `cache_tier` | `str` | 命中的缓存层级 |

## 3. 核心类: AttentionBudgetAnalyzer

### 3.1 初始化

```python
analyzer = AttentionBudgetAnalyzer(context_window=4096)
```

- `context_window`: 模型上下文窗口大小，默认 4096
- 内部维护 `_history` (分析历史) 和 `_latency_history` (延迟历史)

### 3.2 analyze() — 预算分析

**流程说明:**

```
输入: user_tokens (n_u), memory_tokens (n_m)
  │
  ├── 1. 计算 RAG 预算
  │     ├── token_used = n_m + n_u
  │     ├── token_free = context_window - (n_m + n_u)
  │     └── attention = (n_m + n_u)²
  │
  ├── 2. 计算 DKI 预算
  │     ├── token_used = n_u  (记忆不占 token)
  │     ├── token_free = context_window - n_u
  │     └── attention = n_u × (n_m + n_u)
  │
  ├── 3. 计算对比指标
  │     ├── token_saved = n_m
  │     ├── attention_increase = dki_attention - n_u²
  │     ├── token_efficiency = n_m / context_window
  │     └── attention_overhead = attention_increase / n_u²
  │
  └── 4. 返回 BudgetAnalysis 并记录历史
```

**关键算法 — RAG vs DKI 预算公式:**

| 指标 | RAG | DKI |
|------|-----|-----|
| Token Budget 使用 | B_t = n_m + n_u | B_t = n_u |
| Attention Budget | B_a = (n_m + n_u)² | B_a = n_u × (n_m + n_u) |
| Token 效率 | — | 节省 n_m / context_window |
| Attention 开销 | — | 增加 n_u × n_m / n_u² = n_m / n_u |

**数值示例:**

假设 `context_window=4096, user_tokens=500, memory_tokens=1000`:

| 指标 | RAG | DKI | 差异 |
|------|-----|-----|------|
| Token 使用 | 1500 | 500 | 节省 1000 (24.4%) |
| Token 剩余 | 2596 | 3596 | 多 1000 |
| Attention 计算 | 2,250,000 | 750,000 | DKI 更少 |
| Attention 开销比 | — | 3.0 | 相对 vanilla |

### 3.3 should_prefer_dki() — 策略推荐

**决策因子 (5 项):**

| 因子 | 条件 | 说明 |
|------|------|------|
| `memory_tokens_significant` | n_m > 200 | 记忆量足够大 |
| `context_constrained` | RAG 剩余 < 1000 | 上下文空间紧张 |
| `token_efficiency_high` | 效率提升 > 10% | DKI 节省显著 |
| `attention_overhead_acceptable` | 开销比 < 2.0 | 注意力增加可接受 |
| `reasoning_task` | 任务类型为推理/数学/规划/编码 | 推理密集型任务 |

**决策规则:** 满足 ≥ 3 项因子时推荐使用 DKI。

**流程:**

```
输入: user_tokens, memory_tokens, task_type
  │
  ├── 1. 调用 analyze() 获取预算分析
  ├── 2. 评估 5 个决策因子
  ├── 3. 计算得分 (满足因子数)
  ├── 4. score >= 3 → 推荐 DKI
  │      score < 3  → 推荐 RAG
  └── 5. 返回推荐结果 + 推理说明
```

### 3.4 get_average_latency() — 平均延迟统计

遍历 `_latency_history` 列表，计算各阶段的平均耗时和缓存命中率。

### 3.5 get_stats() — 综合统计

返回分析次数、平均 token 效率提升、平均注意力开销、平均记忆 token 数及延迟统计。

## 4. 辅助类: LatencyTimer

上下文管理器，用于精确计时 DKI 各阶段操作。

### 使用方式

```python
with LatencyTimer() as timer:
    timer.start_stage("router")
    # ... 路由操作 ...
    timer.start_stage("gating")
    # ... 门控操作 ...
    timer.start_stage("kv_compute")
    # ... K/V 计算 ...

# timer.breakdown 包含各阶段耗时
analyzer.record_latency(timer.breakdown)
```

### 工作原理

- `__enter__`: 记录总开始时间
- `start_stage(stage)`: 如果有正在计时的阶段则先结束，然后开始新阶段
- `end_stage()`: 计算当前阶段耗时，写入 `breakdown` 对应属性
- `__exit__`: 计算总耗时
- `set_cache_info(hit, tier)`: 设置缓存命中信息

**阶段名称映射:** 阶段名通过 `f"{stage}_ms"` 映射到 `LatencyBreakdown` 的属性名。例如 `start_stage("router")` 对应 `breakdown.router_ms`。

## 5. 数据库交互

本模块不涉及数据库交互。所有数据均为内存中的计算结果。

## 6. 注意事项

1. `attention_increase` 的计算基准是 vanilla LLM (无记忆)，即 `dki_attention - n_u²`，而非与 RAG 对比
2. `_history` 和 `_latency_history` 会持续增长，长时间运行需调用 `clear()` 释放
3. `should_prefer_dki()` 的决策因子权重和阈值为启发式设定，生产环境建议通过 A/B 测试校准
