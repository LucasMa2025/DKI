# 实验 5: 上下文约束实验 (Context-Constrained)

> **入口方法**: `ExperimentRunner.run_context_constrained()`  
> **数据文件**: `data/context_constrained.json`  
> **输出文件**: `experiment_results/context_constrained_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

上下文约束实验验证 DKI 在**固定上下文窗口**下, 随着用户记忆量增长, 相比 RAG 的优势。

核心验证目标:
- **记忆长度 vs 性能**: 当记忆量从 500 增长到 3500 tokens 时, DKI 和 RAG 的表现变化
- **DKI 的上下文效率**: DKI 通过 KV 注入不占用 prompt 上下文, 在长记忆场景下优势更明显
- **RAG 的上下文瓶颈**: RAG 需要将检索结果拼入 prompt, 当记忆过长时被迫截断
- **任务成功率对比**: 在不同记忆长度下, DKI vs RAG 的任务完成率差异

### 1.2 论文对应

对应论文中的 **Table 2: Context-Constrained Evaluation**, 展示记忆长度与任务成功率的关系。

---

## 2. 样本数据实例

### 2.1 数据格式 (`context_constrained.json`)

```json
{
  "id": "ctx_1000_0",
  "memory_length_tokens": 1000,
  "context_budget": 4096,
  "memory_text": "用户住在上海浦东新区，每天通勤约50分钟。\n用户对坚果不过敏，但对贝类严重过敏...\n...",
  "memory_fragments": [
    "用户住在上海浦东新区，每天通勤约50分钟。",
    "用户对坚果不过敏，但对贝类严重过敏，曾因误食虾仁住院。",
    "用户曾在2023年获得全国素食烹饪大赛的银奖。",
    "用户有一个5岁的孩子，对食品安全非常关注。",
    "用户是中国农业大学的营养学研究员，主要研究植物性蛋白质。",
    "..."
  ],
  "query": "根据我的个人情况，帮我安排下周的出差行程。",
  "expected_keywords": ["出差", "安排"],
  "experiment_user": "exp_user_tech"
}
```

### 2.2 数据特征

- **分级记忆长度**: 数据按 `memory_length_tokens` 分组 (500, 1000, 1500, 2000, 2500, 3000, 3500)
- **丰富的用户画像**: `memory_fragments` 包含 10-25+ 条详细的用户信息
- **中文数据**: 全部使用中文, 覆盖饮食、工作、兴趣、健康等多维度信息
- **上下文预算**: `context_budget=4096` 模拟有限上下文窗口

### 2.3 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本 ID (`ctx_{长度}_{序号}`) |
| `memory_length_tokens` | int | 记忆总 token 长度 |
| `context_budget` | int | 上下文窗口预算 |
| `memory_text` | string | 完整记忆文本 (换行分隔) |
| `memory_fragments` | string[] | 记忆片段列表 |
| `query` | string | 用户查询 |
| `expected_keywords` | string[] | 期望关键词 |
| `experiment_user` | string | 实验用户标识 |

---

## 3. 结果预期

### 3.1 记忆长度 vs 任务成功率

| 记忆长度 (tokens) | DKI Task Success | RAG Task Success | Δ (DKI - RAG) |
|:-:|:-:|:-:|:-:|
| 500 | ~0.85 | ~0.80 | +0.05 |
| 1000 | ~0.82 | ~0.70 | +0.12 |
| 1500 | ~0.80 | ~0.60 | +0.20 |
| 2000 | ~0.78 | ~0.50 | **+0.28** |
| 2500 | ~0.75 | ~0.40 | **+0.35** |
| 3000 | ~0.72 | ~0.30 | **+0.42** |
| 3500 | ~0.70 | ~0.25 | **+0.45** |

### 3.2 期望结论

- **DKI 优势随记忆增长而增大**: Δ 从 +0.05 增长到 +0.45
- **DKI 性能缓慢下降**: KV 注入不占用上下文, 主要受 BM25 检索质量影响
- **RAG 性能急剧下降**: 当记忆超过上下文预算时, 被迫截断, 性能断崖式下降
- **转折点 ~1500 tokens**: DKI 优势从此开始显著

### 3.3 结果文件结构

```json
{
  "context_budget": 4096,
  "memory_lengths": [500, 1000, 1500, 2000, 2500, 3000, 3500],
  "results_by_length": {
    "1000": {
      "samples": {
        "dki": [
          {
            "sample_id": "ctx_1000_0",
            "query": "...",
            "response": "...",
            "task_success": 0.80,
            "memory_recall": 0.65,
            "fabricated_halluc": 0.05,
            "irrelevant_halluc": 0.03
          }
        ],
        "rag": [...]
      },
      "summary": {
        "dki": {
          "sample_count": 30,
          "task_success": 0.82,
          "memory_recall": 0.70,
          "fabricated_halluc": 0.06,
          "mean_latency_ms": 150.0
        },
        "rag": { "task_success": 0.70, ... }
      }
    }
  },
  "comparison_table": [
    { "memory_length": 1000, "rag_success": 0.70, "dki_success": 0.82, "delta": 0.12 }
  ]
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_context_constrained(data_path, memory_lengths, context_budget, setup_users)
│
├── _ensure_systems()
│
├── setup_experiment_users() (如果需要)
│
├── 加载数据 → data/context_constrained.json
│
├── user_id = _get_first_experiment_user_id()
│
└── for mem_length in memory_lengths: [500, 1000, ..., 3500]
    │
    ├── 筛选该长度的样本:
    │   length_samples = [d for d in data if d['memory_length_tokens'] == mem_length]
    │
    └── for mode in ['dki', 'rag']:
        │
        └── for sample in length_samples[:30]:
            │
            ├── 创建 session: "ctx_{mode}_{length}_{id}_{ts}"
            ├── _store_ensure_session()
            │
            ├── 写入 memory_fragments:
            │   ├── DKI: _store_add_message(frag) → demo_messages
            │   └── RAG: rag_system.add_memory(frag)
            │
            ├── 写入偏好 (前 5 个 fragments):
            │   └── _write_session_preferences(user_id, frags[:5])
            │
            ├── 执行查询:
            │   ├── DKI: _run_plugin_chat(query, force_alpha=0.5, max_new_tokens=1024)
            │   └── RAG: rag_system.chat(query, max_new_tokens=1024)
            │
            └── 计算指标:
                ├── task_success = 关键词命中 / 总关键词
                ├── memory_recall (前 5 个 fragments vs response)
                └── hallucination_decomposed (所有 fragments vs response)

汇总每个 memory_length 的 dki/rag 均值统计
生成 comparison_table → 保存 context_constrained_{timestamp}.json
```

### 4.2 关键实现细节

#### 记忆写入策略

每个样本的 `memory_fragments` (10-25 条) 全部写入:
- **DKI**: 通过 `_store_add_message()` 写入 `demo_messages`, DKIPlugin 通过 BM25 检索
- **RAG**: 通过 `rag_system.add_memory()` 写入 RAG 的内存向量库

同时, 前 5 条 fragment 作为偏好写入 `demo_preferences`, DKIPlugin 通过 ConfigDrivenAdapter 读取。

#### force_alpha = 0.5

上下文约束实验使用 `force_alpha=0.5` (略高于默认 0.4), 增强偏好注入强度以测试极端场景。

#### 任务成功率计算

```python
task_success = kw_hits / len(expected_keywords)
```

使用 `expected_keywords` 作为任务成功的标准, 衡量回复是否包含关键信息。

#### 对比表格

`comparison_table` 是实验的核心输出, 直接对应论文 Table 2:

```python
table_rows.append({
    'memory_length': mem_length,
    'rag_success': rag_task_success,
    'dki_success': dki_task_success,
    'delta': dki_task_success - rag_task_success,
})
```

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **记忆长度分组正确**: 按 `memory_length_tokens` 筛选对应样本
- ✅ **DKI/RAG 公平对比**: 使用相同的 memory_fragments, 仅写入方式不同
- ✅ **偏好 + 消息双写入**: fragments[:5] → demo_preferences, 全部 fragments → demo_messages
- ✅ **缓存失效**: `_write_session_preferences()` 调用 `invalidate_preference_text_cache()`
- ✅ **指标完整**: task_success, memory_recall, fabricated/irrelevant hallucination
- ✅ **comparison_table 生成**: 便于直接对比绘图

### 5.2 注意事项

- 每个 memory_length 取最多 30 个样本, 不同长度的样本数可能不同
- `context_budget` 参数目前仅作为元数据记录, 不直接限制 DKIPlugin 的上下文
- DKI 模式使用 `max_new_tokens=1024`, 限制生成长度
- 如果某个 memory_length 没有样本, 该长度会被跳过并记录警告
- `force_alpha=0.5` 是此实验特有设置, 与其他实验的 0.4 不同

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_context_constrained(
    data_path="./data/context_constrained.json",
    memory_lengths=[500, 1000, 1500, 2000, 2500, 3000, 3500],
    context_budget=4096,
    setup_users=True,
)
```
