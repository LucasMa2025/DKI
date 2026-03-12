# DKI 实验系统概览

> **版本**: v9.1 (重构后)  
> **架构**: 复用 demo store, 使用独立 `dki.db`  
> **入口**: `dki.experiment.runner.ExperimentRunner`

---

## 1. 系统架构

### 1.1 整体架构

```
ExperimentRunner (上层应用, 自动化实验)
│
├── SQLiteChatStore (dki.experiment.store, 独立 dki.db)
│   ├── demo_users       — 实验用户
│   ├── demo_sessions     — 实验会话
│   ├── demo_messages     — 对话消息 (供 BM25 检索)
│   └── demo_preferences  — 用户偏好 (供 KV 注入)
│
├── DKIPlugin (通过 create_plugin + ConfigDrivenAdapter 标准集成)
│   ├── ConfigDrivenAdapter (只读 demo_* 表, 从 dki.db)
│   │   ├── get_user_preferences() → demo_preferences
│   │   └── search_relevant_history() → demo_messages (BM25)
│   └── InjectionPlanner → 完整注入计划 + KV 注入
│
├── RAGSystem (独立的 RAG 对比基线)
│   └── 向量检索 + prompt 拼接
│
└── ModelAdapter (LLM 推理)
    └── 原始模型生成 (baseline)
```

### 1.2 数据流

```
写入 (ExperimentRunner → SQLiteChatStore)     读取 (DKIPlugin ← ConfigDrivenAdapter)
────────────────────────────────────────      ──────────────────────────────────────
store.add_preference(user_id, text, ...)  →   adapter.get_user_preferences(user_id)
store.add_message(session_id, role, ...)  →   adapter.search_relevant_history(user_id, query)

双方操作同一个 dki.db 文件, 表结构完全一致。
```

### 1.3 与 demo 的关系

| 对比项 | demo 应用 | 实验系统 |
|--------|----------|---------|
| 持久化层 | `demo.store.SQLiteChatStore` | `dki.experiment.store.SQLiteChatStore` |
| 数据库文件 | `data/demo.db` | `data/dki.db` (独立) |
| ORM 模型 | `DemoBase` (demo_* 表) | `DemoBase` (demo_* 表, 独立 declarative_base) |
| DKI 集成 | `create_plugin(adapter_config)` | `create_plugin(adapter_config)` ← 完全相同 |
| 适配器配置 | `demo.dki_bridge` | `dki.experiment.dki_bridge` |
| 用途 | 交互式 Web 对话 | 自动化批量实验 |

---

## 2. 实验列表

| # | 实验名称 | 入口方法 | 数据文件 | 文档 |
|---|---------|---------|---------|------|
| 1 | [PersonaChat](01_persona_chat.md) | `run_persona_chat_experiment()` | `persona_chat.json`, `long_session_persona_chat.json` | 个性化对话 |
| 2 | [LongMemEval](02_longmemeval.md) | `run_longmemeval()` | `longmemeval_multi_turn.json`, `longmemeval_needle.json` | 长期记忆评估 |
| 3 | [Alpha 敏感性](03_alpha_sensitivity.md) | `run_alpha_sensitivity()` | `alpha_sensitivity.json` | 注入强度分析 |
| 4 | [消融实验](04_ablation_study.md) | `run_ablation_study()` | `ablation.json` | 组件贡献分析 |
| 5 | [上下文约束](05_context_constrained.md) | `run_context_constrained()` | `context_constrained.json` | 长记忆场景 |
| 6 | [多轮连贯性](06_multi_turn_coherence.md) | `run_multi_turn_coherence()` | `multi_turn_coherence.json` | 记忆保持 |
| 7 | [延迟对比](07_latency_comparison.md) | `run_latency_comparison()` | 内置查询 | cache 加速 |

此外, `run_experiment()` 是通用实验入口, 支持自定义数据集和多模式对比。

---

## 3. 通用实验入口 (`run_experiment`)

### 3.1 说明

`run_experiment()` 是最基础的实验方法, 支持:
- 自定义 `ExperimentConfig` (模式、数据集、α 值等)
- 加载 `persona_chat.json` / `memory_qa.json` 等数据
- 在 dki / rag / baseline 三种模式下运行
- 自动记录到 `experiments` 和 `experiment_results` 表

### 3.2 数据格式

支持两种通用数据格式:

**PersonaChat 格式** (`persona_chat.json`):
```json
{
  "session_id": "...",
  "personas": ["偏好1", "偏好2"],
  "turns": [{ "query": "...", "expected_keywords": [...] }]
}
```

**Memory QA 格式** (`memory_qa.json`):
```json
{
  "id": "memqa_0000",
  "memory": "I prefer sporty style.",
  "query": "Help me pick an outfit.",
  "expected_memory_use": true,
  "reference_answer": "..."
}
```

### 3.3 运行方式

```python
from dki.experiment import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner(db_path="./data/dki.db")

config = ExperimentConfig(
    name="my_experiment",
    description="Custom experiment",
    modes=["dki", "rag", "baseline"],
    datasets=["persona_chat", "memory_qa"],
    max_samples=50,
    force_alpha=0.4,
)

results = runner.run_experiment(config, setup_users=True)
```

---

## 4. 实验用户

### 4.1 默认实验用户

系统预配置 4 个实验用户, 覆盖不同的偏好场景:

| 用户 | 用户名 | 偏好关键词 |
|------|-------|-----------|
| 素食用户 | `exp_user_vegetarian` | 素食、海鲜过敏、北京海淀 |
| 户外用户 | `exp_user_outdoor` | 户外运动、上海浦东、金毛犬 |
| 技术用户 | `exp_user_tech` | 数据科学、Python、科幻小说 |
| 音乐用户 | `exp_user_music` | 古典音乐、吉他、辣椒过敏 |

### 4.2 用户匹配机制

对于 LongMemEval 等外部数据, `_match_user_by_personas()` 会根据 `personas` 关键词自动匹配最相似的实验用户:

```python
# personas: ["I prefer vegetarian food"]
# → 匹配 exp_user_vegetarian (包含 "素食" 关键词)
```

---

## 5. 评估指标

### 5.1 指标一览

| 指标 | 说明 | 使用实验 |
|------|------|---------|
| `memory_recall` | 记忆召回率 | 1, 3, 4, 5, 6 |
| `keyword_recall` | 关键词召回率 | 1, 2, 5, 6 |
| `answer_match` | 答案匹配率 | 2 |
| `rouge_l` | ROUGE-L F1 | 2, 3, 4 |
| `bleu4` | BLEU-4 | 3, 4 |
| `fabricated_halluc` | 编造幻觉率 | 3, 4, 5 |
| `irrelevant_halluc` | 偏题幻觉率 | 4, 5 |
| `task_success` | 任务成功率 | 5 |
| `latency_ms` | 推理延迟 | 全部 |
| `cache_hit` | 缓存命中 | 1, 7 |

### 5.2 指标计算

所有指标由 `dki.experiment.metrics.MetricsCalculator` 统一计算:
- `compute_memory_recall()`: 记忆文本 vs 回复文本的覆盖度
- `compute_hallucination_decomposed()`: 分解为 fabricated + irrelevant
- `compute_bleu()`: 标准 BLEU-4 计算
- `compute_rouge()`: ROUGE-L F1 计算
- `compute_latency_stats()`: mean, p50, p95, min, max

---

## 6. 注入信息可视化

### 6.1 InjectionInfo

每个 DKI 模式的实验结果都包含 `InjectionInfo`, 记录完整的注入过程:

```python
InjectionInfo(
    mode='dki',
    original_query="推荐一个餐厅",
    preference_text="用户是素食主义者...",     # 偏好 KV 注入内容
    preference_tokens=128,                    # 偏好 token 数
    history_suffix="[历史] 用户: 你好...",     # 历史后缀内容
    history_tokens=64,                        # 历史 token 数
    history_messages=[...],                   # 检索到的历史消息
    final_input="推荐一个餐厅 [历史后缀]",     # 最终发送给模型的输入
    alpha=0.4,                                # 实际使用的 α 值
)
```

### 6.2 InjectionInfoViewer

提供格式化显示和对比工具:

```python
from dki.experiment import InjectionInfoViewer

viewer = InjectionInfoViewer()
viewer.display(injection_info)  # 格式化显示
viewer.compare(dki_info, rag_info)  # DKI vs RAG 对比
viewer.save_to_file(injection_info)  # 保存到文件
```

---

## 7. 快速开始

### 7.1 运行单个实验

```python
from dki.experiment import ExperimentRunner

runner = ExperimentRunner(db_path="./data/dki.db")

# 选择一个实验运行
results = runner.run_persona_chat_experiment()
# 或: runner.run_longmemeval()
# 或: runner.run_alpha_sensitivity()
# 或: runner.run_ablation_study()
# 或: runner.run_context_constrained()
# 或: runner.run_multi_turn_coherence()
# 或: runner.run_latency_comparison()
```

### 7.2 运行全部实验

```python
runner = ExperimentRunner(db_path="./data/dki.db")

# 按顺序运行所有实验
runner.run_persona_chat_experiment()
runner.run_longmemeval()
runner.run_alpha_sensitivity()
runner.run_ablation_study()
runner.run_context_constrained()
runner.run_multi_turn_coherence()
runner.run_latency_comparison()

# 结果保存在 experiment_results/ 目录
```

### 7.3 自定义数据库路径

```python
runner = ExperimentRunner(
    db_path="./my_experiment.db",  # 自定义数据库
    output_dir="./my_results",     # 自定义输出目录
)
```

---

## 8. 文件结构

```
dki/experiment/
├── __init__.py              # 包导出
├── runner.py                # ExperimentRunner (核心)
├── metrics.py               # MetricsCalculator
├── data_generator.py        # ExperimentDataGenerator
├── dki_bridge.py            # ConfigDrivenAdapter 配置生成
├── store/                   # 独立持久化层 (从 demo/store 复制)
│   ├── __init__.py
│   ├── base.py              # IChatStore 接口
│   ├── base_impl.py         # BaseChatStore 实现
│   ├── bm25_mixin.py        # BM25 搜索混入
│   ├── connection.py        # ExperimentDBConfig + ExperimentDBManager
│   ├── factory.py           # create_experiment_store()
│   ├── models.py            # ORM 模型 (DemoBase, DemoUser, ...)
│   └── sqlite_store.py      # SQLiteChatStore
├── docs/                    # 实验说明文档
│   ├── 00_overview.md       # 本文件
│   ├── 01_persona_chat.md
│   ├── 02_longmemeval.md
│   ├── 03_alpha_sensitivity.md
│   ├── 04_ablation_study.md
│   ├── 05_context_constrained.md
│   ├── 06_multi_turn_coherence.md
│   └── 07_latency_comparison.md
├── tests/                   # 测试文件
│   ├── test_runner_refactored.py
│   └── test_sqlite_adapter.py
├── REFACTORING_PROPOSAL.md  # 重构方案文档
└── README.md                # 旧 README
```
