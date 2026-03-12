# 实验 4: 消融实验 (Ablation Study)

> **入口方法**: `ExperimentRunner.run_ablation_study()`  
> **数据文件**: `data/ablation.json`  
> **输出文件**: `experiment_results/ablation_study_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

消融实验通过**逐步移除 DKI 系统的各个组件**, 量化每个组件对最终性能的贡献。这是验证系统架构合理性的关键实验。

核心验证目标:
- **每个组件的独立贡献**: KV 注入、多信号检索、Entropy-Gated 元认知检索各贡献多少
- **组件交互效应**: 组合移除是否产生非线性的性能下降
- **与 RAG/Baseline 的差距**: 完整 DKI 相比基线的提升幅度

### 1.2 消融变体 (7 种)

| 变体 | 标识 | 系统 | α | 记忆 | 事实检索 | KV注入 | 检索模式 |
|------|------|------|---|------|---------|--------|---------|
| **完整 DKI** | `full_dki` | dki | 0.4 | ✅ | entropy_gated | ✅ | 默认 |
| **去掉事实调用** | `wo_fact_call` | dki | 0.4 | ✅ | post_hoc | ✅ | 默认 |
| **去掉多信号检索** | `wo_multi_signal` | dki | 0.4 | ✅ | entropy_gated | ✅ | vector_only |
| **去掉 KV 注入** | `wo_kv_injection` | dki | 0.0 | ✅ | entropy_gated | ❌ | 默认 |
| **仅 Stable 回退** | `stable_fallback_only` | dki | 0.4 | ✅ | post_hoc | ✅ | stable |
| **RAG 基线** | `rag_baseline` | rag | N/A | ✅ | N/A | ❌ | N/A |
| **原始 LLM** | `vanilla_llm` | baseline | N/A | ❌ | N/A | ❌ | N/A |

### 1.3 论文对应

对应论文中的 **Table 3: Ablation Study**, 量化各组件对 Memory Recall、Hallucination Rate、BLEU-4 等指标的影响。

---

## 2. 样本数据实例

### 2.1 数据格式 (`ablation.json`)

```json
{
  "id": "ablation_0000",
  "memory": "用户养了两只猫，叫花花和豆豆。",
  "all_memories": [
    "用户是素食主义者，不吃肉类和海鲜。",
    "用户住在北京海淀区，经常在中关村附近活动。",
    "用户喜欢户外运动，特别是徒步和骑行。",
    "用户是一名数据科学家，擅长机器学习。",
    "用户养了两只猫，叫花花和豆豆。"
  ],
  "query": "附近有什么好的运动场所",
  "relevant_memories": [
    "用户住在北京海淀区，经常在中关村附近活动。",
    "用户喜欢户外运动，特别是徒步和骑行。"
  ],
  "reference_answer": "你住在海淀区中关村附近，喜欢徒步和骑行，附近有圆明园、颐和园适合骑行，百望山适合徒步。",
  "ablation_modes": [
    "full_dki", "wo_fact_call", "wo_multi_signal",
    "wo_kv_injection", "stable_fallback_only",
    "rag_baseline", "vanilla_llm"
  ],
  "metadata": {
    "dataset": "ablation",
    "language": "zh"
  }
}
```

### 2.2 数据特征

- **中文数据**: 主要使用中文查询和记忆, 测试中文场景
- **多记忆上下文**: `all_memories` 包含 5 条记忆, `relevant_memories` 标注相关子集
- **参考答案**: 每个样本提供 `reference_answer` 用于 BLEU/ROUGE 计算
- **明确的相关性标注**: `relevant_memories` 精确标注哪些记忆与查询相关

### 2.3 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本 ID |
| `memory` | string | 主记忆 (写入 demo_messages) |
| `all_memories` | string[] | 所有可用记忆 |
| `query` | string | 用户查询 |
| `relevant_memories` | string[] | 与查询相关的记忆子集 |
| `reference_answer` | string | 参考答案 |
| `ablation_modes` | string[] | 该样本应运行的消融变体 |

---

## 3. 结果预期

### 3.1 各变体预期指标

| 变体 | Memory Recall ↑ | Fab. Halluc ↓ | BLEU-4 ↑ | ROUGE-L ↑ | 延迟 |
|------|-----------------|---------------|----------|-----------|------|
| `full_dki` | **最高** | 低 | **最高** | **最高** | 中等 |
| `wo_fact_call` | 高 | 略高 | 高 | 高 | 较快 |
| `wo_multi_signal` | 中等 | 低 | 中等 | 中等 | 中等 |
| `wo_kv_injection` | 低 | 最低 | 低 | 低 | 快 |
| `stable_fallback_only` | 中低 | 中等 | 中低 | 中低 | 较快 |
| `rag_baseline` | 中等 | 中等 | 中等 | 中等 | 慢 |
| `vanilla_llm` | **最低** | 最低 | 最低 | 最低 | 最快 |

### 3.2 期望结论

- **KV 注入贡献最大**: `full_dki` vs `wo_kv_injection` 差距最大
- **多信号检索有明显作用**: `full_dki` vs `wo_multi_signal` 有显著差异
- **Entropy-Gated 有增益**: `full_dki` vs `wo_fact_call` 有一定提升
- **DKI > RAG**: `full_dki` 在 recall 和延迟上优于 `rag_baseline`
- **记忆系统有价值**: `rag_baseline` 和 `full_dki` 均远超 `vanilla_llm`

### 3.3 结果文件结构

```json
{
  "full_dki": {
    "samples": [
      {
        "query": "附近有什么好的运动场所",
        "response": "你住在海淀区...",
        "latency_ms": 145.0,
        "memory_recall": 0.85,
        "fabricated_halluc": 0.05,
        "irrelevant_halluc": 0.03,
        "total_halluc": 0.08,
        "bleu4": 0.40,
        "rouge_l": 0.48
      }
    ],
    "latencies": [145.0, ...]
  },
  "summary": {
    "full_dki": {
      "sample_count": 30,
      "memory_recall": 0.75,
      "fabricated_halluc_rate": 0.08,
      "irrelevant_halluc_rate": 0.05,
      "total_halluc_rate": 0.13,
      "bleu4_mean": 0.35,
      "rouge_l_mean": 0.42,
      "mean_latency_ms": 150.0,
      "p95_latency_ms": 250.0
    },
    "vanilla_llm": {
      "memory_recall": 0.05,
      "fabricated_halluc_rate": 0.02,
      "total_halluc_rate": 0.05,
      "bleu4_mean": 0.10,
      "rouge_l_mean": 0.12
    }
  }
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_ablation_study(data_path, setup_users)
│
├── _ensure_systems()
│
├── setup_experiment_users() (如果需要)
│
├── 加载数据 → data/ablation.json
│
├── user_id = _get_first_experiment_user_id()
│
└── for ablation_mode, config in ablation_configs.items():
    │   (full_dki, wo_fact_call, wo_multi_signal, wo_kv_injection,
    │    stable_fallback_only, rag_baseline, vanilla_llm)
    │
    ├── 创建 session: "ablation_{mode}_{ts}"
    ├── _store_ensure_session()
    │
    ├── 写入 memories (如果 use_memory=True):
    │   ├── DKI: _store_add_message(memory) → demo_messages
    │   └── RAG: rag_system.add_memory(memory)
    │
    └── for item in data[:30]:
        │
        ├── 根据 config 选择系统和参数:
        │   ├── system='dki':
        │   │   ├── 构建 dki_kwargs
        │   │   ├── 设置 fact_retrieve_method (entropy_gated / post_hoc)
        │   │   ├── 如果 use_kv_injection=False → force_alpha=0.0
        │   │   └── _run_plugin_chat(**dki_kwargs)
        │   │
        │   ├── system='rag':
        │   │   └── rag_system.chat(query)
        │   │
        │   └── system='baseline':
        │       └── model.generate(query) (无记忆)
        │
        ├── 计算指标:
        │   ├── memory_recall (relevant_memories vs response)
        │   ├── hallucination_decomposed:
        │   │   ├── fabricated_rate (凭空编造)
        │   │   ├── irrelevant_rate (偏题)
        │   │   └── total_rate
        │   ├── BLEU-4 (vs reference_answer)
        │   └── ROUGE-L (vs reference_answer)
        │
        └── 记录到 results[ablation_mode]

汇总 summary → 每个变体的均值统计
保存 → ablation_study_{timestamp}.json
```

### 4.2 关键实现细节

#### 消融控制方式

不同消融变体通过以下参数组合控制:

```python
ablation_configs = {
    'full_dki': {
        'system': 'dki',
        'force_alpha': 0.4,
        'fact_retrieve_method': 'entropy_gated',
        'use_kv_injection': True,
    },
    'wo_kv_injection': {
        'system': 'dki',
        'force_alpha': 0.4,  # → 被覆盖为 0.0
        'fact_retrieve_method': 'entropy_gated',
        'use_kv_injection': False,  # → force_alpha=0.0
    },
    # ...
}
```

当 `use_kv_injection=False` 时, `force_alpha` 被强制设为 0.0, 使 KV 注入无效。

#### Hallucination 分解

`compute_hallucination_decomposed()` 将幻觉分为:
- **Fabricated**: 模型回复中包含的事实不在 grounding_texts 中 (凭空编造)
- **Irrelevant**: 模型回复中包含的内容与 query 无关 (偏题)

grounding_texts 使用 `relevant_memories` 和 `memory` 的并集。

#### 数据量

默认取每种变体前 30 个样本, 7 种变体共计 210 次推理调用。

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **7 种变体完整覆盖**: 配置映射正确, 每种变体独立运行
- ✅ **force_alpha 覆盖**: `wo_kv_injection` 正确将 alpha 设为 0.0
- ✅ **fact_retrieve_method 传递**: 正确传递 `entropy_gated` / `post_hoc` 给 DKIPlugin
- ✅ **基线公平性**: `vanilla_llm` 不使用任何记忆 (use_memory=False)
- ✅ **指标完整**: 计算 recall, fabricated/irrelevant hallucination, BLEU, ROUGE
- ✅ **数据隔离**: 每个变体使用独立的 session_id

### 5.2 注意事项

- `recall_mode` 参数 (`vector_only`, `stable`) 需要 DKIPlugin 支持; 如不支持, 实际行为可能与预期不同
- `fact_retrieve_method` 需要底层模型支持 entropy 计算; 如不支持, 会退化为 `post_hoc`
- `vanilla_llm` 模式下 `use_memory=False`, 不会写入任何记忆到 demo_messages
- summary 中排除了 key='summary' 自身, 避免递归

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_ablation_study(
    data_path="./data/ablation.json",
    setup_users=True,
)
```
