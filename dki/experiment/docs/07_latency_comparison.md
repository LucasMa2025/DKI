# 实验 7: 延迟对比实验 (Latency Comparison)

> **入口方法**: `ExperimentRunner.run_latency_comparison()`  
> **数据文件**: 无外部数据文件 (使用内置查询列表)  
> **输出文件**: `experiment_results/latency_comparison_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

延迟对比实验测量 DKI 和 RAG 在**多轮对话**中的推理延迟, 特别关注:

核心验证目标:
- **首轮延迟**: DKI 首轮需要构建 KV cache (冷启动), 是否显著慢于 RAG
- **后续轮延迟**: DKI 的偏好 KV cache 复用后, 是否比 RAG 快
- **cache 命中效果**: DKI 的 `preference_cache_hit` 对延迟的影响
- **加速比**: DKI 后续轮平均延迟 vs RAG 平均延迟的比值

### 1.2 实验设计

- **固定查询**: 使用 10 个预设查询, 模拟自然对话场景
- **DKI 先跑**: 记录每轮延迟和 cache 命中状态
- **RAG 后跑**: 使用相同查询, 记录延迟
- **对比分析**: 计算首轮延迟、后续轮均值、加速比

### 1.3 论文对应

对应论文中的 **Latency Analysis** (Figure 5), 展示 DKI 的 cache 复用带来的延迟优势。

---

## 2. 样本数据实例

### 2.1 内置查询列表

实验使用内置的查询列表, 不依赖外部数据文件:

```python
queries = [
    "What should I eat for dinner?",
    "Recommend a weekend activity.",
    "What's the weather like?",
    "Suggest a restaurant.",
    "What hobbies should I try?",
] * 2  # 重复一次, 共 10 个查询
```

### 2.2 内置记忆

```python
memories = [
    "User prefers vegetarian food.",
    "User lives in Beijing.",
    "User enjoys hiking.",
]
```

### 2.3 数据特征

- **轻量化设计**: 无需外部数据文件, 开箱即用
- **重复查询**: 同一查询出现两次, 测试 cache 复用效果
- **简单记忆**: 仅 3 条短记忆, 消除记忆复杂度对延迟的干扰

---

## 3. 结果预期

### 3.1 延迟对比预期

| 指标 | DKI | RAG |
|------|-----|-----|
| 首轮延迟 | 较高 (KV cache 冷构建) | 中等 |
| 后续轮均值 | **较低** (cache 复用) | 中等 (每次检索) |
| 加速比 | — | 基准 |

### 3.2 期望结论

- **DKI 首轮 > RAG**: 首轮需构建偏好 KV cache, 延迟略高
- **DKI 后续轮 < RAG**: cache 复用后, 无需重复构建偏好 KV
- **加速比 > 1.0**: DKI 后续轮比 RAG 平均更快
- **cache_hit 变化**: 第 1 轮 cache_hit=False, 第 2 轮起 cache_hit=True

### 3.3 结果文件结构

```json
{
  "dki_latencies": [
    { "turn": 1, "latency_ms": 250.0, "cache_hit": false },
    { "turn": 2, "latency_ms": 120.0, "cache_hit": true },
    { "turn": 3, "latency_ms": 115.0, "cache_hit": true },
    { "turn": 4, "latency_ms": 118.0, "cache_hit": true },
    { "turn": 5, "latency_ms": 122.0, "cache_hit": true }
  ],
  "rag_latencies": [
    { "turn": 1, "latency_ms": 180.0 },
    { "turn": 2, "latency_ms": 175.0 },
    { "turn": 3, "latency_ms": 185.0 },
    { "turn": 4, "latency_ms": 170.0 },
    { "turn": 5, "latency_ms": 182.0 }
  ],
  "summary": {
    "dki_first_turn": 250.0,
    "dki_subsequent_mean": 118.75,
    "rag_mean": 178.4,
    "speedup_subsequent": 1.50
  }
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_latency_comparison(n_turns=10, setup_users=True)
│
├── _ensure_systems()
│
├── setup_experiment_users() (如果需要)
│
├── session_id = "latency_exp_{ts}"
├── user_id = _get_first_experiment_user_id()
├── _store_ensure_session()
│
├── 写入 memories → demo_messages:
│   ├── "User prefers vegetarian food."
│   ├── "User lives in Beijing."
│   └── "User enjoys hiking."
│
├── ═══ DKI Phase ═══
│   └── for i in range(n_turns):  # 默认 10 轮
│       ├── _run_plugin_chat(query, session_id, user_id)
│       │   └── DKIPlugin.chat()
│       │       ├── Turn 1: 构建偏好 KV cache (冷启动)
│       │       └── Turn 2+: 复用偏好 KV cache
│       ├── 记录: { turn, latency_ms, cache_hit }
│       ├── _store_add_message(user) → demo_messages
│       └── _store_add_message(assistant) → demo_messages
│
├── ═══ RAG Phase ═══
│   ├── rag_session_id = "rag_latency_exp_{ts}"
│   ├── 写入 memories → rag_system.add_memory()
│   └── for i in range(n_turns):
│       ├── rag_system.chat(query)
│       └── 记录: { turn, latency_ms }
│
└── ═══ Summary ═══
    ├── dki_first_turn = dki_latencies[0]
    ├── dki_subsequent_mean = mean(dki_latencies[1:])
    ├── rag_mean = mean(rag_latencies)
    └── speedup_subsequent = rag_mean / dki_subsequent_mean

保存 → latency_comparison_{timestamp}.json
```

### 4.2 关键实现细节

#### DKI cache 机制

DKI 的延迟优势来自偏好 KV cache 复用:

```
Turn 1: 读取偏好 → 编码偏好 KV → 注入 → 推理 (慢)
Turn 2: cache hit → 直接复用 KV → 注入 → 推理 (快)
Turn 3+: cache hit → 直接复用 KV → 注入 → 推理 (快)
```

`response.metadata.preference_cache_hit` 记录每轮是否命中 cache。

#### 加速比计算

```python
speedup = rag_mean / dki_subsequent_mean
```

- `speedup > 1.0`: DKI 后续轮比 RAG 快
- `speedup < 1.0`: DKI 后续轮比 RAG 慢 (不期望)
- `speedup ≈ 1.0`: 两者延迟接近

#### DKI/RAG 独立 session

DKI 和 RAG 使用不同的 session_id:
- DKI: `latency_exp_{ts}` + `demo_messages` 存储
- RAG: `rag_latency_exp_{ts}` + RAG 内部存储

确保两者的历史消息不互相影响。

#### DKI 对话存储

DKI 模式每轮对话后存储 user + assistant 消息到 `demo_messages`。
这模拟真实使用场景, 同时为后续轮的 BM25 历史检索提供数据。

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **DKI cache_hit 记录**: 通过 `response.metadata.preference_cache_hit` 正确记录
- ✅ **DKI 对话持久化**: 每轮 user + assistant 消息写入 `demo_messages`
- ✅ **RAG 独立 session**: 使用独立的 `rag_session_id`, 不与 DKI 共享
- ✅ **加速比计算正确**: 避免除零 (当 `dki_subsequent_mean > 0` 时才计算)
- ✅ **summary 结构完整**: dki_first_turn, dki_subsequent_mean, rag_mean, speedup

### 5.2 注意事项

- 默认 `n_turns=10`, 实际查询列表长度为 10 (5 个查询 × 2)
- DKI 和 RAG 使用不同的 session, 但相同的查询序列
- memories 仅 3 条, 测试延迟而非记忆质量
- 此实验不计算 recall 或 hallucination 指标, 仅关注延迟
- 如果模型不支持 KV cache, DKI 的 cache_hit 可能始终为 False

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_latency_comparison(
    n_turns=10,
    setup_users=True,
)

# 查看结果
s = results['summary']
print(f"DKI first turn: {s['dki_first_turn']:.1f}ms")
print(f"DKI subsequent mean: {s['dki_subsequent_mean']:.1f}ms")
print(f"RAG mean: {s['rag_mean']:.1f}ms")
print(f"Speedup (DKI subsequent vs RAG): {s['speedup_subsequent']:.2f}x")
```
