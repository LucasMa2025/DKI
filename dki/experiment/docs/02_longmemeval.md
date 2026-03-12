# 实验 2: LongMemEval — 长期记忆评估基准

> **入口方法**: `ExperimentRunner.run_longmemeval()`  
> **数据文件**: `data/longmemeval_multi_turn.json`, `data/longmemeval_needle.json`  
> **输出文件**: `experiment_results/longmemeval_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

LongMemEval 是一个标准化的**长期记忆评估基准**, 用于测试系统在长距离对话历史中检索和利用关键信息的能力。

核心验证目标:
- **长距离记忆召回**: 当目标信息被大量无关对话"淹没"时, 系统能否准确检索
- **Needle-in-a-Haystack**: 在长历史中精确定位特定事实
- **多轮对话记忆**: 跨多个 session 的信息持久化和检索
- **三模式对比**: DKI vs RAG vs Baseline 在长记忆任务上的差异

### 1.2 实验子模式

| 子模式 | 数据文件 | 说明 |
|--------|---------|------|
| `multi_turn` | `longmemeval_multi_turn.json` | 多轮对话, 评估跨轮记忆 |
| `needle` | `longmemeval_needle.json` | 大海捞针, 在长历史中检索特定事实 |

### 1.3 运行模式

| 模式 | 描述 |
|------|------|
| **dki** | DKIPlugin KV 注入 (偏好 + BM25 历史检索) |
| **rag** | RAGSystem 向量检索 + 提示词拼接 |
| **baseline** | 原始 LLM 生成 (无记忆增强) |

### 1.4 论文对应

对应论文中的 **Long-term Memory Evaluation**, 对齐 LongMemEval benchmark (ShareGPT 子集)。

---

## 2. 样本数据实例

### 2.1 多轮模式数据 (`longmemeval_multi_turn.json`)

```json
{
  "session_id": "longmem_mt_e47becba",
  "experiment_user": "exp_user_longmem",
  "personas": [
    "I'm trying to organize my life a bit better, can you recommend some task management apps...",
    "I've been using a planner, but I think I need something more digital.",
    "I think I'll try out Todoist and Trello.",
    "By the way, do you have any tips on creating a routine for my new job?",
    "I graduated with a degree in Business Administration..."
  ],
  "turns": [
    {
      "turn_id": 0,
      "query": "ok fine. he's not drunk. just very pirate'y",
      "expected_keywords": [],
      "source_session": "sharegpt_QZMeA7V_17",
      "session_date": "2023/05/29 (Mon) 19:30",
      "expected_response": "Aye, I understand. Here's a version of the cover letter..."
    },
    {
      "turn_id": 4,
      "query": "I'm looking for some gift ideas for a baby boy...",
      "expected_keywords": [],
      "source_session": "e348269f_1",
      "session_date": "2023/05/30 (Tue) 02:51",
      "expected_response": "What a lovely gesture! Congratulations to Rachel..."
    }
  ]
}
```

### 2.2 数据特征

- **personas**: 用户长期记忆信息 (跨 session 的个人信息)
- **turns**: 来自不同 `source_session` 和 `session_date` 的对话, 模拟真实用户的长期使用
- **is_eval_query**: 标记评估查询 (仅对评估查询计算指标)
- **expected_answer**: 期望的完整回复文本
- **expected_keywords**: 期望回复中包含的关键词

### 2.3 数据字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | LongMemEval 会话 ID |
| `personas` | string[] | 用户长期记忆/人设 |
| `turns[].query` | string | 用户查询 |
| `turns[].source_session` | string | 原始来源 session (ShareGPT) |
| `turns[].session_date` | string | 原始对话日期 |
| `turns[].is_eval_query` | bool | 是否为评估查询 |
| `turns[].expected_answer` | string | 期望回答 |
| `turns[].expected_keywords` | string[] | 期望关键词 |

---

## 3. 结果预期

### 3.1 指标说明

| 指标 | 计算方法 | 期望 |
|------|---------|------|
| `keyword_recall` | 命中关键词数 / 总关键词数 | DKI > RAG > Baseline |
| `answer_match` | 期望答案中的词命中比例 | DKI > Baseline |
| `rouge_l` | ROUGE-L F1 score | DKI ≥ RAG |
| `latency_ms` | 推理延迟 (毫秒) | DKI < RAG |

### 3.2 DKI vs RAG vs Baseline 预期

| 指标 | DKI | RAG | Baseline |
|------|-----|-----|----------|
| keyword_recall | 最高 | 中等 | 最低 |
| answer_match | 最高 | 中等 | 最低 |
| latency | 最低 | 中等 | 最低 (无检索) |

### 3.3 结果文件结构

```json
{
  "benchmark": "longmemeval",
  "config": {
    "modes": ["dki", "rag", "baseline"],
    "longmemeval_modes": ["multi_turn", "needle"],
    "max_samples": 50,
    "force_alpha": 0.4
  },
  "results_by_dataset": {
    "longmemeval_multi_turn": {
      "dki": {
        "mode": "dki",
        "samples": [
          {
            "sample_idx": 0,
            "eval_query": "...",
            "expected_answer": "...",
            "response": "...",
            "keyword_recall": 0.75,
            "answer_match": 0.60,
            "rouge_l": 0.45,
            "injection_info": {
              "injection_enabled": true,
              "preference_tokens": 128,
              "history_tokens": 256,
              "preference_text": "...",
              "history_suffix_text": "..."
            }
          }
        ],
        "metrics": {
          "total_samples": 50,
          "keyword_recall_mean": 0.65,
          "answer_match_mean": 0.55,
          "rouge_l_mean": 0.40
        }
      }
    }
  },
  "summary": {
    "longmemeval_multi_turn": {
      "dki": { "keyword_recall": 0.65, "answer_match": 0.55, "rouge_l": 0.40 },
      "rag": { "keyword_recall": 0.50, "answer_match": 0.40, "rouge_l": 0.30 },
      "baseline": { "keyword_recall": 0.10, "answer_match": 0.15, "rouge_l": 0.10 }
    }
  }
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_longmemeval(modes, longmemeval_modes, max_samples, force_alpha)
│
├── _ensure_systems()
│   ├── 创建 SQLiteChatStore (dki.db)
│   ├── 创建 DKIPlugin (via create_plugin + ConfigDrivenAdapter)
│   └── 创建 RAGSystem
│
├── setup_users=True 时:
│   ├── 创建默认实验用户 (vegetarian, outdoor, tech, music)
│   └── 额外创建 exp_user_longmem (LongMemEval 专用用户)
│
├── for lm_mode in longmemeval_modes: (multi_turn, needle)
│   │
│   ├── 加载数据 data/{longmemeval_{lm_mode}}.json
│   │   └── 如果文件不存在且 auto_generate=True → 调用 DataGenerator 生成
│   │
│   └── for mode in modes: (dki, rag, baseline)
│       └── _run_longmemeval_mode(mode, samples, force_alpha)
│
└── 汇总 & 保存 → longmemeval_{timestamp}.json


_run_longmemeval_mode(mode, samples, force_alpha)
│
└── for idx, item in enumerate(samples):
    │
    ├── 生成 session_id: "longmem_{mode}_{ts}_{idx}"
    ├── 获取 user_id (从 item 或匹配实验用户)
    ├── _store_ensure_session()
    │
    ├── 写入 personas 偏好
    │   └── _write_session_preferences(user_id, personas)
    │       ├── 清除旧偏好 → demo_preferences
    │       ├── 写入新偏好 → demo_preferences
    │       └── 清除 DKIPlugin 缓存
    │
    ├── 写入历史记忆 (仅 DKI: → demo_messages, RAG: → rag memory)
    │
    ├── 分离 turns → history_turns + eval_turn
    │   └── eval_turn: is_eval_query=True 的那一轮
    │
    ├── 播放 history_turns (非评估轮):
    │   ├── DKI: _store_add_message() → 存入 demo_messages
    │   └── RAG: rag_system.chat() → 更新 RAG 内部历史
    │
    ├── 执行 eval_turn (评估查询):
    │   ├── DKI: _run_plugin_chat(eval_query, force_alpha)
    │   │   └── DKIPlugin.chat() → 注入偏好 + 检索历史 → 推理
    │   │       提取 injection_info (preference_tokens, history_tokens, ...)
    │   │       _store_add_message() × 2 (user + assistant)
    │   │
    │   ├── RAG: rag_system.chat(eval_query)
    │   └── Baseline: model.generate(eval_query)
    │
    └── 计算评估指标:
        ├── keyword_recall = 关键词命中 / 总关键词
        ├── answer_match = 答案词命中比例
        └── rouge_l = ROUGE-L F1
```

### 4.2 关键实现细节

#### 评估轮分离

LongMemEval 数据中, 不是所有 turn 都需要评估。`is_eval_query=True` 标记的轮次才是评估目标,
其余轮次只用于构建对话历史:

```python
for t in turns:
    if t.get('is_eval_query'):
        eval_turn = t        # 只对这个轮次计算指标
    else:
        history_turns.append(t)  # 先播放历史
```

#### 历史注入机制

DKI 模式下, `history_turns` 的 query 和 expected_response 都写入 `demo_messages`,
这样当执行 eval_query 时, ConfigDrivenAdapter 能通过 BM25 搜索到相关的历史对话。

#### 三模式公平对比

- **DKI**: personas → `demo_preferences`, history → `demo_messages` → BM25 检索
- **RAG**: personas → `rag_system.add_memory()`, history → `rag_system.chat()` (内部维护)
- **Baseline**: 仅使用 eval_query, 无任何记忆增强

### 4.3 按问题类型分析

结果按 `question_type` 分组统计 (e.g., factual, temporal, referential),
帮助分析 DKI 在不同类型问题上的表现差异:

```json
"by_question_type": {
  "factual": { "count": 15, "keyword_recall": 0.70 },
  "temporal": { "count": 10, "keyword_recall": 0.50 },
  "referential": { "count": 8, "keyword_recall": 0.55 }
}
```

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **偏好写入路径**: personas → `_write_session_preferences()` → `demo_preferences` → ConfigDrivenAdapter
- ✅ **历史写入路径**: history_turns → `_store_add_message()` → `demo_messages` → BM25 检索
- ✅ **评估轮分离**: 正确区分 `is_eval_query` 和非评估轮次
- ✅ **三模式隔离**: DKI/RAG/Baseline 使用不同的 session_id, 不互相干扰
- ✅ **缓存失效**: 每个 sample 写入偏好后调用 `invalidate_preference_text_cache()`
- ✅ **注入信息记录**: DKI 模式下完整记录 injection_info (含 preference_text, history_suffix_text 等)

### 5.2 注意事项

- 如果 `longmemeval_multi_turn.json` 不存在且 `auto_generate=True`, 需要 `longmem/longmemeval_s_cleaned.json` 源文件
- `max_samples` 默认 50, 可根据计算资源调整
- `force_alpha` 默认 0.4, 建议与论文设置一致
- Baseline 模式无记忆注入, keyword_recall 预期较低

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_longmemeval(
    modes=["dki", "rag", "baseline"],
    longmemeval_modes=["multi_turn", "needle"],
    max_samples=50,
    force_alpha=0.4,
    setup_users=True,
    auto_generate=True,
)
```
