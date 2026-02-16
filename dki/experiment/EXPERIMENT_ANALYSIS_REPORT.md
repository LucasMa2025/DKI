# DKI Experiment 模块分析报告

**分析日期**: 2026-02-13  
**分析范围**: `dki/experiment/` 目录全部文件  
**分析版本**: 2.0.0

---

## 1. 目录结构与职责

| 文件 | 职责 | 代码质量 |
|------|------|----------|
| `__init__.py` | 统一导出接口 | ✅ 优良 |
| `runner.py` | 实验运行器 (核心) | ⚠️ 已修正 2 处错误 + 补全 |
| `data_generator.py` | 合成测试数据生成 | ⚠️ 已修正 1 处错误 + 补全 |
| `metrics.py` | 评估指标计算 | ✅ 优良 |

---

## 2. 各文件详细分析

### 2.1 `runner.py` — 实验运行器

**修正 2 处错误 + 补全 2 个实验:**

#### ❌ 错误 1: `run_alpha_sensitivity` 和 `run_latency_comparison` 缺少 `user_id` 参数

**严重度**: 🟡 中

**问题**: 之前修正了 `_run_single_query` 中的 `user_id` 传递，但 `run_alpha_sensitivity` (line 614) 和 `run_latency_comparison` (line 681, 694) 中的 `dki_system.chat()` 和 `rag_system.chat()` 调用仍未传递 `user_id`。这可能导致:
- 数据库记录缺少用户标识
- 偏好缓存无法正确工作

**修正**: 在所有 chat 调用中传递 `user_id="experiment_user"`。

#### ❌ 错误 2: `_compute_mode_metrics` 未计算核心评估指标

**严重度**: 🟡 中

**问题**: `MetricsCalculator` 提供了 `compute_memory_recall`, `compute_hallucination_rate`, `compute_bleu`, `compute_rouge` 等方法，但 `_compute_mode_metrics` 仅计算了延迟统计和 alpha 分布，完全没有调用质量评估指标。这导致实验结果缺乏关键的对比数据。

**修正**: 
- 新增错误计数统计 (`error_count`, `valid_count`)
- 新增 memory recall 计算
- 新增 hallucination rate 启发式估计
- 新增 response length 统计

---

### 2.2 `data_generator.py` — 测试数据生成

**修正 1 处错误 + 补全 3 个数据集:**

#### ❌ 错误: `generate_hotpot_qa` 中 format 调用可能 KeyError

**严重度**: 🟡 中

**问题**: 
```python
question = q_template.format(**{k: entity_values.get(k, '') for k in q_entities if k in entity_values})
```
当 `q_entities` 中有实体未出现在 `entity_values` 中时 (因为随机选择了 2 个 fact 模板，可能不覆盖 question 模板需要的所有实体)，`if k in entity_values` 过滤掉了这些键，导致 `.format()` 因缺少占位符变量而抛出 `KeyError`。

**修正**: 
1. 优先选择所有必需实体都可用的 question 模板
2. 对不可用实体提供默认值 `'Unknown'`

#### 补全: 3 个新数据集

1. **`generate_chinese_persona_chat`**: 中文版 PersonaChat 对话数据
   - 15 条中文偏好记忆
   - 10 种中文查询 + 期望关键词
   - 支持中文场景的记忆召回测试

2. **`generate_multi_turn_coherence`**: 多轮连贯性测试数据
   - 3 个精心设计的对话场景
   - 每个场景 5 轮对话，递进式引用记忆
   - 每轮有明确的期望记忆召回关键词
   - 测试早期、中期、后期的记忆保持

3. **`generate_ablation_data`**: 消融实验数据
   - 5 条中文偏好记忆
   - 5 种查询，标注了相关记忆索引
   - 6 种消融模式定义

---

### 2.3 `metrics.py` — 评估指标

**评估**: ✅ 无错误

**正确性分析**:
- `compute_bleu`: 使用 NLTK 的 sentence_bleu + smoothing，标准实现
- `compute_rouge`: 使用 rouge_score 库，支持 ROUGE-1/2/L
- `compute_memory_recall`: 基于关键词匹配，可配置阈值
- `compute_hallucination_rate`: 启发式方法，基于事实指示词 + 关键词匹配
- `compute_latency_stats`: 完整的统计量 (p50/p95/p99/mean/std/min/max)

**设计建议**:
- `compute_hallucination_rate` 是启发式方法，生产环境建议集成专用的 NLI 模型 (如 MiniLM for NLI)
- `_extract_keywords` 支持中英文 (`\u4e00-\u9fff`)，满足双语需求

---

## 3. 实验完整性分析

### 论文要求 vs 实现覆盖

| 论文实验 | 实现状态 | 说明 |
|----------|----------|------|
| DKI vs RAG 对比 | ✅ `run_experiment` | 支持 dki, rag, baseline 三模式 |
| α 敏感性分析 | ✅ `run_alpha_sensitivity` | 测试 α ∈ [0, 1] |
| 延迟对比 (首轮 vs 后续) | ✅ `run_latency_comparison` | 测试 Session KV Cache 效果 |
| 多轮连贯性测试 | ✅ 新增 `run_multi_turn_coherence` | 精确衡量记忆保持率 |
| 消融实验 | ✅ 新增 `run_ablation_study` | 测试各组件独立贡献 |
| 幻觉率评估 | ✅ `_compute_mode_metrics` 新增 | 启发式幻觉检测 |
| Memory Recall 评估 | ✅ `_compute_mode_metrics` 新增 | 关键词匹配召回率 |
| 中文场景测试 | ✅ 新增中文数据集 | `cn_persona_chat` + 中文消融数据 |
| BLEU/ROUGE 文本质量 | ⚡ 已实现未集成 | `MetricsCalculator` 可用，需参考文本 |

### 实验运行流程

```
ExperimentRunner
├── run_experiment()              # 主实验: DKI vs RAG vs Baseline
│   ├── _run_mode()               # 按模式运行
│   │   ├── _run_single_query()   # 单次查询 + 注入信息收集
│   │   └── _compute_mode_metrics() # 计算评估指标
│   ├── _aggregate_metrics()      # 跨模式聚合
│   └── _save_results()           # 保存 JSON 结果
│
├── run_alpha_sensitivity()       # α 敏感性分析
├── run_latency_comparison()      # 延迟对比实验
├── run_multi_turn_coherence()    # [新增] 多轮连贯性实验
└── run_ablation_study()          # [新增] 消融实验
```

### 数据生成流程

```
ExperimentDataGenerator
├── generate_persona_chat()         # 英文 PersonaChat (100 sessions)
├── generate_chinese_persona_chat() # [新增] 中文 PersonaChat (100 sessions)
├── generate_hotpot_qa()            # HotpotQA 多跳推理 (100 samples)
├── generate_memory_qa()            # 记忆召回测试 (100 samples)
├── generate_multi_turn_coherence() # [新增] 多轮连贯性 (50 sessions)
├── generate_ablation_data()        # [新增] 消融实验 (50 samples)
├── generate_alpha_sensitivity_data() # α 敏感性 (50 × 6 = 300 samples)
└── generate_all()                  # 一键生成所有数据集
```

---

## 4. 修正汇总

| # | 文件 | 问题 | 严重度 | 状态 |
|---|------|------|--------|------|
| 1 | `runner.py` | 缺少 user_id 参数 | 🟡 中 | ✅ 已修正 |
| 2 | `runner.py` | `_compute_mode_metrics` 未计算核心指标 | 🟡 中 | ✅ 已修正 |
| 3 | `data_generator.py` | HotpotQA format KeyError | 🟡 中 | ✅ 已修正 |

## 5. 补全汇总

| # | 文件 | 补全内容 | 说明 |
|---|------|----------|------|
| 1 | `data_generator.py` | `generate_chinese_persona_chat` | 中文偏好+对话数据 |
| 2 | `data_generator.py` | `generate_multi_turn_coherence` | 多轮连贯性测试数据 |
| 3 | `data_generator.py` | `generate_ablation_data` | 消融实验数据 |
| 4 | `runner.py` | `run_multi_turn_coherence` | 多轮连贯性实验 |
| 5 | `runner.py` | `run_ablation_study` | 消融实验 |
| 6 | `runner.py` | `_compute_mode_metrics` 增强 | 记忆召回+幻觉率+响应长度 |

---

## 6. 实验运行指南

### 生成测试数据
```bash
cd DKI
python -m dki.experiment.data_generator
```

### 运行完整实验
```python
from dki.experiment import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()

# 1. DKI vs RAG 对比实验
config = ExperimentConfig(
    name="dki_vs_rag",
    modes=["dki", "rag", "baseline"],
    datasets=["persona_chat", "memory_qa"],
)
results = runner.run_experiment(config)

# 2. α 敏感性分析
alpha_results = runner.run_alpha_sensitivity()

# 3. 延迟对比
latency_results = runner.run_latency_comparison(n_turns=10)

# 4. 多轮连贯性实验
coherence_results = runner.run_multi_turn_coherence()

# 5. 消融实验
ablation_results = runner.run_ablation_study()
```
