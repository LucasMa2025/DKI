# 实验 1: PersonaChat — 个性化对话实验

> **入口方法**: `ExperimentRunner.run_persona_chat_experiment()`  
> **数据文件**: `data/persona_chat.json`, `data/long_session_persona_chat.json`  
> **输出文件**: `experiment_results/persona_chat_experiment_{timestamp}.json`

---

## 1. 实验说明

### 1.1 目的

PersonaChat 实验验证 DKI (Dynamic KV Injection) 系统在**个性化对话**场景中的表现。

核心验证目标:
- **偏好注入有效性**: 用户偏好 (personas) 通过 KV 注入后, 模型能否在回复中体现这些偏好
- **历史记忆召回**: 多轮对话中, 模型能否利用注入的历史消息回答与之前对话相关的问题
- **DKI vs RAG 对比**: 在相同偏好/历史条件下, DKI 注入模式与 RAG 提示词拼接模式的效果差异
- **短/长会话对比**: 验证 DKI 在短会话 (3-5 轮) 和长会话 (10-20+ 轮) 场景下的表现差异

### 1.2 实验模式

| 模式 | 描述 |
|------|------|
| **dki** | 使用 DKIPlugin 进行 KV 注入 (偏好 + 历史后缀) |
| **rag** | 使用 RAGSystem 进行提示词拼接 (检索 + 拼接上下文) |

### 1.3 论文对应

对应论文中的 **Personalization Quality** 评估, 衡量系统将用户特征融入对话的能力。

---

## 2. 样本数据实例

### 2.1 短会话数据 (`persona_chat.json`)

```json
{
  "session_id": "persona_session_0000",
  "experiment_user": "exp_user_vegetarian",
  "personas": [
    "I usually wake up early around 6 AM.",
    "I prefer coffee over tea.",
    "I prefer vegetarian food and am allergic to seafood."
  ],
  "turns": [
    {
      "turn_id": 0,
      "query": "What skills should I learn?",
      "expected_keywords": ["guitar", "AI", "artificial intelligence"],
      "relevant_memories": [],
      "reference_answer": "Given your interest in AI and that you're learning guitar, you could deepen your guitar skills or explore AI/machine learning courses online."
    },
    {
      "turn_id": 1,
      "query": "What music should I listen to while working?",
      "expected_keywords": ["classical music", "guitar"],
      "relevant_memories": [],
      "reference_answer": "As a fan of classical music who is learning guitar, you might enjoy listening to classical guitar pieces..."
    }
  ],
  "metadata": {
    "dataset": "persona_chat",
    "experiment_user": "exp_user_vegetarian",
    "generated_at": "2026-03-12T13:07:46.424852"
  }
}
```

### 2.2 长会话数据 (`long_session_persona_chat.json`)

```json
{
  "session_id": "long_session_0000",
  "session_type": "long",
  "experiment_user": "exp_user_vegetarian",
  "personas": [
    "我是严格的素食主义者，已经坚持素食15年了。我对所有动物制品都非常敏感...",
    "我有严重的海鲜过敏史，曾经因为误食含有虾仁的菜品而住院过...",
    "我是一名营养学研究者，在中国农业大学工作...",
    "我每周末会去有机农场采购新鲜蔬果...",
    "我正在写一本关于中国传统素食文化的书..."
  ],
  "turns": [
    {
      "turn_id": 0,
      "query": "你好，我最近在研究北京地区的素食餐厅分布情况...",
      "expected_keywords": ["素食", "海淀", "餐厅"],
      "expected_length_range": [512, 2048],
      "relevant_memories": ["我是严格的素食主义者...", "我是一名营养学研究者...", "我正在写一本书..."]
    }
  ]
}
```

### 2.3 数据字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话唯一标识 |
| `experiment_user` | string | 映射到实验用户 (如 `exp_user_vegetarian`) |
| `personas` | string[] | 用户偏好/人设描述列表 |
| `turns[].query` | string | 用户查询 |
| `turns[].expected_keywords` | string[] | 预期回复中应包含的关键词 |
| `turns[].relevant_memories` | string[] | 与该查询相关的偏好记忆 |
| `turns[].reference_answer` | string | 参考回答 (用于评估) |
| `turns[].expected_length_range` | int[] | 预期回复长度范围 (长会话) |

---

## 3. 结果预期

### 3.1 DKI 模式预期

| 指标 | 期望 | 说明 |
|------|------|------|
| **关键词召回率 (recall)** | ≥ 0.4 | 回复中应包含偏好相关关键词 |
| **延迟 (latency)** | < RAG | DKI 的 KV 注入不增加输入长度, 推理更快 |
| **偏好注入 (preference_tokens)** | > 0 | 偏好应成功注入到 KV cache |
| **历史注入 (history_tokens)** | > 0 (多轮) | 历史消息应通过后缀方式注入 |

### 3.2 DKI vs RAG 比较

| 对比项 | DKI 优势 | RAG 优势 |
|--------|---------|---------|
| 延迟 | ✅ KV 注入不增加 prompt 长度 | - |
| 个性化 | ✅ 偏好通过 attention 级别注入 | 偏好作为文本拼入 prompt |
| 长会话 | ✅ 上下文窗口利用更高效 | 上下文窗口被检索内容占用 |
| 可解释性 | - | ✅ 注入内容在 prompt 中可直接查看 |

### 3.3 结果文件结构

```json
{
  "short_sessions": {
    "dki": [
      {
        "session_id": "exp_dki_short_...",
        "turns": [
          {
            "turn_idx": 0,
            "query": "...",
            "response": "...",
            "latency_ms": 120.5,
            "recall_score": 0.67,
            "injection_info": {
              "mode": "dki",
              "preference_text": "用户偏好文本...",
              "preference_tokens": 128,
              "history_suffix": "...",
              "history_tokens": 64,
              "final_input": "..."
            }
          }
        ]
      }
    ],
    "rag": [...]
  },
  "long_sessions": { ... },
  "summary": {
    "short_sessions_dki": {
      "session_count": 20,
      "total_turns": 100,
      "mean_latency_ms": 150.0,
      "p95_latency_ms": 280.0,
      "mean_recall": 0.55
    }
  }
}
```

---

## 4. 程序流程

### 4.1 流程图

```
run_persona_chat_experiment()
│
├── _ensure_systems()  ─────────────────────────────────────────────┐
│   ├── 创建 SQLiteChatStore (连接 dki.db)                          │
│   ├── 通过 build_experiment_adapter_config() 生成适配器配置         │
│   ├── 通过 create_plugin(adapter_config=...) 创建 DKIPlugin       │
│   └── 创建 RAGSystem                                             │
│                                                                   │
├── setup_experiment_users()  ──────────────────────────────────────┤
│   ├── 遍历默认实验用户 (vegetarian, outdoor, tech, music)           │
│   ├── 调用 store.get_or_create_user() → demo_users 表              │
│   ├── 清除旧偏好 → store.delete_preference()                       │
│   └── 写入新偏好 → store.add_preference() → demo_preferences 表    │
│                                                                   │
├── 加载数据                                                         │
│   ├── persona_chat.json → short_data                               │
│   └── long_session_persona_chat.json → long_data                   │
│                                                                   │
├── 短会话实验 (short_data[:20])                                     │
│   └── for mode in ['dki', 'rag']:                                  │
│       └── for session_data in short_data:                          │
│           └── _run_session(mode, session_data, 'short')  ─────────┤
│                                                                   │
├── 长会话实验 (long_data[:10])                                      │
│   └── for mode in ['dki', 'rag']:                                  │
│       └── for session_data in long_data:                           │
│           └── _run_session(mode, session_data, 'long')             │
│                                                                   │
├── 汇总指标 (mean_latency, p95_latency, mean_recall)                │
└── 保存结果 → persona_chat_experiment_{timestamp}.json               │
                                                                     │
                                                                     │
_run_session(mode, session_data, session_type)                       │
│                                                                   │
├── 生成 session_id: "exp_{mode}_{type}_{session_id}"                │
├── _store_ensure_session() → 确保 session 存在于 demo_sessions       │
├── _write_session_preferences() → 写入 personas 到 demo_preferences │
│   └── 清除 DKIPlugin 偏好缓存 (invalidate_preference_text_cache)   │
├── 写入 personas 到 demo_messages (DKI) 或 RAG memory               │
│                                                                   │
└── for turn in session_data['turns']:                               │
    ├── DKI 模式:                                                    │
    │   ├── _run_plugin_chat(query, session_id, user_id)             │
    │   │   └── DKIPlugin.chat()                                     │
    │   │       ├── ConfigDrivenAdapter 从 demo_* 表读取偏好/历史      │
    │   │       ├── InjectionPlanner.build_plan() 生成注入计划         │
    │   │       └── 执行 KV 注入 + 模型推理                            │
    │   ├── 提取 InjectionInfo (从 metadata 正式字段)                  │
    │   ├── _store_add_message(user) → demo_messages                  │
    │   └── _store_add_message(assistant) → demo_messages             │
    │                                                                │
    └── RAG 模式:                                                    │
        ├── rag_system.chat(query, session_id, user_id)              │
        └── 提取 InjectionInfo (从 prompt_info)                       │
```

### 4.2 关键实现细节

#### 数据写入与读取的对称性

```
写入 (ExperimentRunner)                    读取 (DKIPlugin via ConfigDrivenAdapter)
─────────────────────────                  ─────────────────────────────────────────
store.add_preference()                     adapter.get_user_preferences(user_id)
  → demo_preferences 表                      → 从 demo_preferences 表读取

store.add_message()                        adapter.search_relevant_history(user_id, query)
  → demo_messages 表                          → 从 demo_messages 表 BM25 搜索
```

#### 偏好缓存管理

每个 session 开始前, `_write_session_preferences()` 会:
1. 软删除该用户的旧偏好 (`delete_preference`)
2. 写入新偏好 (`add_preference`)  
3. 清除 DKIPlugin 的偏好文本缓存 (`invalidate_preference_text_cache`)

这确保 DKIPlugin 在每个 session 中读到正确的偏好数据。

#### 对话持久化

DKI 模式下, 每轮对话后:
1. 用户消息写入 `demo_messages` (store.add_message, role='user')
2. 助手回复写入 `demo_messages` (store.add_message, role='assistant')

这保证后续轮次的 `search_relevant_history()` 能找到之前的对话。

### 4.3 评估指标计算

| 指标 | 计算方法 |
|------|---------|
| `recall_score` | 关键词命中数 / 预期关键词总数 |
| `mean_latency_ms` | 所有轮次延迟的平均值 |
| `p95_latency_ms` | 所有轮次延迟的 P95 |
| `mean_recall` | 所有有 expected_keywords 轮次的平均 recall_score |

---

## 5. 正确性审查

### 5.1 已验证的正确性

- ✅ **偏好注入路径**: personas → `demo_preferences` → ConfigDrivenAdapter → DKIPlugin.chat() → KV 注入
- ✅ **历史注入路径**: 对话 → `demo_messages` → ConfigDrivenAdapter → BM25 搜索 → 历史后缀注入
- ✅ **缓存失效**: 每次 `_write_session_preferences()` 后调用 `invalidate_preference_text_cache()`
- ✅ **独立数据库**: 使用 `data/dki.db` (默认), 不与 `demo.db` 冲突
- ✅ **注入信息记录**: 通过 `InjectionMetadata` 正式字段获取, 无需 hack

### 5.2 注意事项

- 短会话默认取前 20 个 session, 长会话取前 10 个
- 每个 session 的 personas 同时写入 `demo_preferences` (偏好) 和 `demo_messages` (历史)
- RAG 模式不写入 `demo_messages`, 使用 RAGSystem 自有的内存存储
- `expected_keywords` 匹配为大小写不敏感

---

## 6. 运行方式

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(
    output_dir="./experiment_results",
    db_path="./data/dki.db",
)

results = runner.run_persona_chat_experiment(
    data_path="./data/persona_chat.json",     # 可选, 默认自动加载
    include_long_sessions=True,                # 是否包含长会话
    setup_users=True,                          # 是否自动创建实验用户
)
```
