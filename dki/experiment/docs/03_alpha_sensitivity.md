# 实验 3: Alpha 敏感性分析

> **入口方法**: `ExperimentRunner.run_alpha_sensitivity()`  
> **数据文件**: `data/alpha_sensitivity.json`  
> **输出文件**: `experiment_results/alpha_sensitivity_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

Alpha 敏感性实验研究**注入强度参数 α** 对 DKI 系统性能的影响。α 控制偏好 KV 注入在注意力机制中的混合权重:

- `α = 0.0`: 完全不注入偏好 KV (等同于无偏好注入)
- `α = 0.4`: 论文推荐的默认值
- `α = 1.0`: 完全使用偏好 KV (可能过度注入)

核心验证目标:
- **最优 α 值**: 找到 recall 和 hallucination 的最佳平衡点
- **α 对 recall 的影响**: 更大的 α 是否带来更好的记忆召回
- **α 对 hallucination 的影响**: 过大的 α 是否增加幻觉率
- **α 对延迟的影响**: 不同 α 值的推理速度变化

### 1.2 论文对应

对应论文中的 **α Sensitivity Analysis** (Figure 4), 展示 α 值与各指标的关系曲线。

---

## 2. 样本数据实例

### 2.1 数据格式 (`alpha_sensitivity.json`)

```json
{
  "id": "alpha_0000_00",
  "memory": "User prefers vegetarian food and is allergic to seafood.",
  "query": "Suggest a career development path.",
  "alpha": 0.0,
  "metadata": {
    "dataset": "alpha_sensitivity",
    "generated_at": "2026-03-12T13:07:46.474862"
  }
}
```

### 2.2 数据特征

- 每组数据由**同一个 memory + query** 在不同 α 值下重复
- 数据文件中的 `alpha` 字段为数据级别标记, 实际实验使用 `force_alpha` 参数覆盖
- 查询涵盖多个领域: 职业发展、饮食推荐、活动建议等

### 2.3 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一 ID (`alpha_{样本号}_{α×10}`) |
| `memory` | string | 用户记忆/偏好文本 |
| `query` | string | 用户查询 |
| `alpha` | float | 建议的 α 值 (实际由 `force_alpha` 控制) |
| `relevant_memories` | string[] | 可选, 与查询相关的记忆 |
| `reference_answer` | string | 可选, 参考答案 |

---

## 3. 结果预期

### 3.1 α-指标关系预期

| α 值 | Memory Recall | Fabricated Halluc | BLEU-4 | ROUGE-L | 延迟 |
|------|--------------|-------------------|--------|---------|------|
| 0.0 | 最低 | 最低 | 基线 | 基线 | 基线 |
| 0.3 | 中等 | 低 | 提升 | 提升 | ≈基线 |
| **0.4** | **较高** | **低** | **最佳** | **最佳** | **≈基线** |
| 0.5 | 高 | 中等 | 高 | 高 | ≈基线 |
| 0.7 | 最高 | 中高 | 下降 | 下降 | ≈基线 |
| 1.0 | 最高 | 最高 | 明显下降 | 明显下降 | ≈基线 |

### 3.2 期望结论

- **最优 α ≈ 0.4**: recall 和 hallucination 的最佳平衡点
- **α > 0.5**: recall 饱和, hallucination 显著上升
- **α = 0.0**: 等同于 baseline, 无记忆增益
- **延迟**: 对 α 值不敏感 (KV 注入不改变计算量)

### 3.3 结果文件结构

```json
{
  "alpha_values": [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
  "results_by_alpha": {
    "0.4": {
      "samples": [
        {
          "query": "Suggest a career development path.",
          "response": "...",
          "latency_ms": 135.2,
          "actual_alpha": 0.4,
          "memory_recall": 0.80,
          "fabricated_halluc": 0.05,
          "bleu4": 0.35,
          "rouge_l": 0.42
        }
      ],
      "latency_stats": { "mean": 140.0, "p50": 135.0, "p95": 200.0 },
      "bleu4_mean": 0.30,
      "rouge_l_mean": 0.38,
      "memory_recall_mean": 0.75,
      "fabricated_halluc_mean": 0.08
    }
  },
  "summary_table": [
    { "alpha": 0.0, "bleu4": 0.15, "rouge_l": 0.20, "memory_recall": 0.10, "fabricated_halluc": 0.02, "latency_p50": 130 },
    { "alpha": 0.4, "bleu4": 0.30, "rouge_l": 0.38, "memory_recall": 0.75, "fabricated_halluc": 0.08, "latency_p50": 135 }
  ]
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_alpha_sensitivity(data_path, alpha_values, setup_users)
│
├── _ensure_systems()
│   ├── 创建 SQLiteChatStore (dki.db)
│   ├── 创建 DKIPlugin
│   └── 创建 RAGSystem
│
├── setup_experiment_users() (如果需要)
│
├── 加载数据 → data/alpha_sensitivity.json
│
├── user_id = _get_first_experiment_user_id()
│
└── for alpha in alpha_values: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    │
    ├── 创建 session: "alpha_exp_{ts}_{idx}"
    ├── 确保用户和 session 存在
    │
    ├── 写入 memories 到 demo_messages (供 BM25 检索)
    │
    └── for item in data[:50]:
        │
        ├── _run_plugin_chat(query, force_alpha=alpha)
        │   └── DKIPlugin.chat()
        │       ├── ConfigDrivenAdapter 读取偏好/历史
        │       ├── InjectionPlanner 使用 force_alpha=alpha
        │       └── KV 注入 (强度由 alpha 控制)
        │
        ├── 计算 memory_recall:
        │   └── metrics.compute_memory_recall(relevant_memories, response)
        │
        ├── 计算 hallucination:
        │   └── metrics.compute_hallucination_decomposed(response, grounding, query)
        │
        ├── 计算 BLEU-4 / ROUGE-L (如有 reference_answer):
        │   ├── metrics.compute_bleu(reference, response)
        │   └── metrics.compute_rouge(reference, response)
        │
        └── 记录 actual_alpha (= response.metadata.alpha)

    汇总每个 alpha 的统计:
    ├── latency_stats (mean, p50, p95)
    ├── bleu4_mean, rouge_l_mean
    ├── memory_recall_mean
    └── fabricated_halluc_mean

生成 summary_table → 保存 alpha_sensitivity_{timestamp}.json
```

### 4.2 关键实现细节

#### force_alpha 参数

`_run_plugin_chat(force_alpha=alpha)` 强制 DKIPlugin 使用指定的 α 值, 覆盖其自动计算。
这确保了实验的可控性 — 每个 α 值下的注入强度是确定的。

#### 同一 session 复用

每个 α 值使用一个独立的 session, 所有 50 个样本在同一 session 下运行。
memories 在 session 开始时一次性写入 `demo_messages`, 供后续查询的 BM25 检索使用。

#### 评估指标

- **memory_recall**: 使用 `MetricsCalculator.compute_memory_recall()`, 将 relevant_memories 与 response 文本对比
- **hallucination**: 使用 `compute_hallucination_decomposed()`, 分为 fabricated (凭空编造) 和 irrelevant (偏题) 两类
- **BLEU-4 / ROUGE-L**: 仅在有 reference_answer 时计算

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **force_alpha 传递**: `_run_plugin_chat` 正确传递 `force_alpha` 参数给 DKIPlugin.chat()
- ✅ **actual_alpha 记录**: 使用 `response.metadata.alpha` 记录实际使用的 α 值
- ✅ **独立 session**: 每个 α 值使用独立的 session_id, 避免交叉污染
- ✅ **指标计算完整**: recall, hallucination, BLEU, ROUGE 全部计算并记录
- ✅ **summary_table**: 提供便于绘图的汇总表格

### 5.2 注意事项

- 默认取数据前 50 个样本, 每个 α 值跑一遍
- memories 写入与偏好写入是分开的: memories → `demo_messages`, personas → `demo_preferences`
- α = 0.0 时 DKI 仍会执行检索流程, 但 KV 注入混合权重为 0
- 如果数据中 `relevant_memories` 为空, recall 默认 0.0

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_alpha_sensitivity(
    data_path="./data/alpha_sensitivity.json",
    alpha_values=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
    setup_users=True,
)
```
