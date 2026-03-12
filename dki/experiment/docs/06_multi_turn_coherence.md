# 实验 6: 多轮连贯性实验 (Multi-Turn Coherence)

> **入口方法**: `ExperimentRunner.run_multi_turn_coherence()`  
> **数据文件**: `data/multi_turn_coherence.json`  
> **输出文件**: `experiment_results/multi_turn_coherence_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

多轮连贯性实验验证 DKI 系统在**多轮对话**中维持用户信息一致性的能力。不同于 PersonaChat 关注个性化, 此实验聚焦于:

核心验证目标:
- **跨轮记忆保持**: 系统能否在第 N 轮回忆起第 1 轮的信息
- **记忆衰减曲线**: 随着对话轮次增加, 记忆召回率的变化趋势
- **DKI vs RAG 连贯性**: 两种方式在维持多轮连贯性上的差异
- **测试轮标记**: 精确控制哪些轮次测试记忆, 哪些轮次仅为自然对话

### 1.2 实验设计

- 每个 session 有 5 轮对话, 部分轮次标记为 `tests_memory=True`
- 测试轮需要回忆之前的偏好 (personas), 通过 `expected_recall` 关键词验证
- 非测试轮为自然对话, 不计入召回统计

### 1.3 论文对应

对应论文中的 **Multi-Turn Coherence Analysis**, 分析 DKI 的记忆保持能力。

---

## 2. 样本数据实例

### 2.1 数据格式 (`multi_turn_coherence.json`)

```json
{
  "session_id": "coherence_session_0000",
  "personas": [
    "用户是素食主义者",
    "用户住在上海浦东",
    "用户有一只叫小花的猫"
  ],
  "turns": [
    {
      "query": "你好，我想找个好吃的餐厅",
      "tests_memory": false,
      "reference_answer": "好的，请问你有什么饮食偏好或限制吗？..."
    },
    {
      "query": "我之前说过我的饮食习惯，帮我推荐吧",
      "tests_memory": true,
      "expected_recall": ["素食"],
      "reference_answer": "你是素食主义者，我推荐你去一家纯素餐厅..."
    },
    {
      "query": "离我家近一点的",
      "tests_memory": true,
      "expected_recall": ["上海", "浦东"],
      "reference_answer": "你住在上海浦东，我帮你找浦东附近的素食餐厅..."
    },
    {
      "query": "对了，我想给我的宠物买个玩具",
      "tests_memory": true,
      "expected_recall": ["猫", "小花"],
      "reference_answer": "给你的猫小花买个逗猫棒或者猫抓板吧..."
    },
    {
      "query": "总结一下今天我们聊了什么",
      "tests_memory": true,
      "expected_recall": ["素食", "餐厅", "宠物"],
      "reference_answer": "今天我们聊了：1）推荐素食餐厅；2）给猫小花挑选玩具。"
    }
  ],
  "metadata": {
    "dataset": "multi_turn_coherence",
    "language": "zh",
    "scenario_idx": 0
  }
}
```

### 2.2 数据特征

- **中文对话**: 自然的中文多轮对话场景
- **渐进式记忆测试**: 从简单 (单个关键词) 到复杂 (多关键词+跨轮信息)
- **明确的测试标记**: `tests_memory` 标记哪些轮次需要召回偏好信息
- **递进式难度**: 最后一轮 "总结" 需要回忆整个对话中涉及的所有偏好

### 2.3 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `personas` | string[] | 用户偏好 (写入 demo_preferences + demo_messages) |
| `turns[].query` | string | 用户查询 |
| `turns[].tests_memory` | bool | 是否为记忆测试轮 |
| `turns[].expected_recall` | string[] | 期望回复中包含的关键词 |
| `turns[].reference_answer` | string | 参考答案 |

---

## 3. 结果预期

### 3.1 逐轮召回率预期

| 轮次 | 测试内容 | DKI Recall | RAG Recall |
|------|---------|------------|------------|
| Turn 0 | 非测试轮 | — | — |
| Turn 1 | 回忆 "素食" | ~0.90 | ~0.80 |
| Turn 2 | 回忆 "上海浦东" | ~0.85 | ~0.70 |
| Turn 3 | 回忆 "猫小花" | ~0.80 | ~0.65 |
| Turn 4 | 综合回忆 | ~0.70 | ~0.55 |

### 3.2 期望结论

- **DKI overall_recall > RAG overall_recall**: DKI 的 KV 注入更稳定地保持记忆
- **DKI 衰减更缓慢**: 随轮次增加, DKI 的 recall 下降幅度小于 RAG
- **综合回忆轮差距最大**: "总结" 类查询需要同时回忆多个偏好, DKI 优势最明显
- **DKI 偏好注入持久性**: 通过 KV 注入的偏好在整个 session 中保持有效

### 3.3 结果文件结构

```json
{
  "dki": {
    "sessions": [
      {
        "session_id": "coherence_session_0000",
        "turns": [
          {
            "turn_idx": 0,
            "query": "你好，我想找个好吃的餐厅",
            "response": "...",
            "tests_memory": false,
            "expected_recall": [],
            "recall_score": 0.0
          },
          {
            "turn_idx": 1,
            "query": "我之前说过我的饮食习惯...",
            "tests_memory": true,
            "expected_recall": ["素食"],
            "recall_score": 1.0
          }
        ]
      }
    ],
    "per_turn_recall": {
      "turn_1": [1.0, 0.8, 1.0, ...],
      "turn_2": [0.5, 1.0, 0.5, ...]
    },
    "per_turn_summary": {
      "turn_1": { "mean_recall": 0.90, "count": 15 },
      "turn_2": { "mean_recall": 0.85, "count": 12 }
    },
    "overall_recall": 0.82
  },
  "rag": {
    "overall_recall": 0.65,
    ...
  }
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_multi_turn_coherence(data_path, setup_users)
│
├── _ensure_systems()
│
├── setup_experiment_users() (如果需要)
│
├── 加载数据 → data/multi_turn_coherence.json
│
└── for mode in ['dki', 'rag']:
    │
    └── for session_data in data[:20]:
        │
        ├── 生成 session_id: "coherence_{mode}_{session_id}"
        ├── 获取 user_id
        │
        ├── _write_session_preferences(user_id, personas)
        │   ├── 清除旧偏好 → demo_preferences
        │   ├── 写入新偏好 → demo_preferences
        │   └── 清除 DKIPlugin 缓存
        │
        ├── _store_ensure_session()
        │
        ├── 写入 personas 作为消息:
        │   ├── DKI: _store_add_message(mem) → demo_messages
        │   └── RAG: rag_system.add_memory(mem)
        │
        └── for turn_idx, turn in enumerate(turns):
            │
            ├── DKI 模式:
            │   ├── _run_plugin_chat(query, session_id, user_id)
            │   │   └── DKIPlugin.chat()
            │   │       ├── ConfigDrivenAdapter 读取偏好 (demo_preferences)
            │   │       ├── ConfigDrivenAdapter BM25 搜索历史 (demo_messages)
            │   │       └── KV 注入 + 推理
            │   ├── _store_add_message(user, query) → demo_messages
            │   └── _store_add_message(assistant, response) → demo_messages
            │
            └── RAG 模式:
                └── rag_system.chat(query, session_id, user_id)

            计算 recall_score (仅 tests_memory=True 时):
            ├── hits = count(kw in response for kw in expected_recall)
            └── recall_score = hits / len(expected_recall)

            按 turn_idx 分组统计 → per_turn_recall

汇总:
├── per_turn_summary: 每轮的平均 recall 和样本数
└── overall_recall: 所有测试轮的平均 recall

保存 → multi_turn_coherence_{timestamp}.json
```

### 4.2 关键实现细节

#### 测试轮筛选

仅 `tests_memory=True` 的轮次纳入 `per_turn_recall` 统计:

```python
if turn.get('tests_memory'):
    results[mode]['per_turn_recall'][turn_key].append(recall_score)
```

非测试轮仍然执行对话和存储, 但不影响召回统计。

#### 对话持久化 (DKI 模式)

DKI 模式下, 每轮对话结束后:
1. 用户消息写入 `demo_messages` → 后续轮次可通过 BM25 检索到
2. 助手回复写入 `demo_messages` → 为后续的"总结"查询提供上下文

这保证了多轮连贯性 — 每轮都能检索到之前的完整对话历史。

#### 偏好双写策略

personas 同时写入:
- `demo_preferences` (通过 `_write_session_preferences`): 供 DKIPlugin KV 注入
- `demo_messages` (通过 `_store_add_message`): 供 BM25 历史检索

这是实验系统的标准做法, 确保 DKI 的偏好注入和历史召回都能正常工作。

#### overall_recall 计算

```python
all_recalls = [s for scores in per_turn.values() for s in scores]
overall_recall = np.mean(all_recalls)
```

将所有 session 的所有测试轮的 recall 汇总为一个数字, 便于 DKI vs RAG 直接对比。

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **偏好注入路径**: personas → `demo_preferences` → ConfigDrivenAdapter → DKIPlugin KV 注入
- ✅ **历史注入路径**: 每轮对话 → `demo_messages` → BM25 搜索 → 历史后缀注入
- ✅ **测试轮筛选**: 仅 `tests_memory=True` 的轮次计入 recall 统计
- ✅ **DKI 对话存储**: 每轮结束后 user + assistant 消息都写入 `demo_messages`
- ✅ **缓存失效**: 每个 session 开始时清除偏好缓存
- ✅ **per_turn_recall 分组**: 按 turn_idx 正确分组, 便于分析记忆衰减曲线
- ✅ **overall_recall 汇总**: 正确合并所有 session 的测试轮 recall

### 5.2 注意事项

- 默认取前 20 个 session, 每个 session 约 5 轮 → 每个模式约 100 轮
- 每个 session 独立创建, 不同 session 之间的偏好和历史不互相影响
- `expected_recall` 关键词匹配为大小写不敏感
- RAG 模式不写入 `demo_messages`, 依赖 RAG 系统内部的历史管理
- 如果数据文件不存在, 会调用 `ExperimentDataGenerator.generate_multi_turn_coherence()` 自动生成

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_multi_turn_coherence(
    data_path="./data/multi_turn_coherence.json",
    setup_users=True,
)

# 查看结果
print(f"DKI overall recall: {results['dki']['overall_recall']:.3f}")
print(f"RAG overall recall: {results['rag']['overall_recall']:.3f}")
```
