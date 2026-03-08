# DKI 实验系统

## 概述

DKI 实验系统是一个独立的**数据验证模块**，用于系统地对比评估 DKI (Dynamic KV Injection) 与 RAG (Retrieval-Augmented Generation) 在多种场景下的表现。

**设计原则**: 实验系统与 chat demo、dki_plugin 等功能模块相互独立，数据依赖在实验目录下单独实现 (`SQLiteDataAdapter`)，不影响生产模块。

## 架构

```
experiment/
├── runner.py            # 实验运行器 (ExperimentRunner)
├── metrics.py           # 评估指标计算器 (MetricsCalculator)  
├── data_generator.py    # 实验数据生成器 (ExperimentDataGenerator)
├── sqlite_adapter.py    # 实验专用数据适配器 (SQLiteDataAdapter)
├── __init__.py          # 包导出
├── tests/               # 单元测试
│   ├── test_sqlite_adapter.py      # SQLiteDataAdapter 测试
│   └── test_runner_refactored.py   # Runner 重构验证测试
└── README.md            # 本文档
```

## v7.0 重构说明

- **移除 DKISystem 依赖**: 实验 DKI 模式通过 `DKIPlugin + SQLiteDataAdapter` 运行
- **数据隔离**: 新增 `SQLiteDataAdapter` (实现 `IUserDataAdapter` 接口)，与 `dki/database` 模块解耦
- **跨会话检索**: `SQLiteDataAdapter.search_relevant_history(session_id=None)` 支持跨会话关键词匹配
- **召回率增强**: `MetricsCalculator.compute_content_recall()` 支持多维度评估 (关键词 + 注入覆盖 + 语义 n-gram)

## 完整实验清单

### 1. 核心对比实验

#### 1.1 Persona Chat (英文人设对话)
- **数据文件**: `data/persona_chat.json`
- **生成器**: `ExperimentDataGenerator.generate_persona_chat()`
- **运行器**: `ExperimentRunner.run_persona_chat_experiment()`
- **评估目标**: 多轮对话中人设一致性保持能力
- **对比模式**: DKI vs RAG vs Baseline
- **核心指标**: Memory Recall, BLEU, ROUGE-L, Hallucination Rate

#### 1.2 Chinese Persona Chat (中文人设对话)
- **数据文件**: `data/cn_persona_chat.json`
- **生成器**: `ExperimentDataGenerator.generate_chinese_persona_chat()`
- **运行器**: `ExperimentRunner.run_persona_chat_experiment()`
- **评估目标**: 中文场景下的偏好召回和个性化响应质量
- **特色**: 包含中文饮食偏好、城市生活、职业技能等人设

#### 1.3 Memory QA (记忆问答)
- **数据文件**: `data/memory_qa.json`
- **生成器**: `ExperimentDataGenerator.generate_memory_qa()`
- **运行器**: `ExperimentRunner.run_experiment()`
- **评估目标**: 对先前对话中提到的事实进行精确召回

#### 1.4 HotpotQA (多跳推理)
- **数据文件**: `data/hotpot_qa.json`
- **生成器**: `ExperimentDataGenerator.generate_hotpot_qa()`
- **运行器**: `ExperimentRunner.run_experiment()`
- **评估目标**: 跨文档多跳推理能力 (需要组合多个记忆片段)

### 2. 长会话与跨会话实验

#### 2.1 Long Session Persona Chat (长会话人设)
- **数据文件**: `data/long_session_persona_chat.json`
- **生成器**: `ExperimentDataGenerator.generate_long_session_persona_chat()`
- **运行器**: `ExperimentRunner.run_persona_chat_experiment()`
- **评估目标**: 长对话 (50+ 轮) 中早期记忆的保持能力
- **特色**: 模拟真实用户多日交互场景

#### 2.2 LongMemEval 基准 (长期记忆评估)
- **数据文件**: `data/longmemeval_multi_turn.json`, `data/longmemeval_needle.json`, `data/longmemeval_oracle.json`
- **生成器**: `ExperimentDataGenerator.generate_longmemeval()`
- **运行器**: `ExperimentRunner.run_longmemeval()`
- **评估目标**: 标准化长期记忆基准测试
- **子任务**:
  - **Multi-turn**: 多轮对话中的记忆检索
  - **Needle**: "大海捞针" — 在大量对话中精确定位特定信息
  - **Oracle**: 理想条件下的记忆利用上限

### 3. 敏感性与消融实验

#### 3.1 Alpha Sensitivity (注入强度敏感性)
- **数据文件**: `data/alpha_sensitivity.json`
- **生成器**: `ExperimentDataGenerator.generate_alpha_sensitivity_data()`
- **运行器**: `ExperimentRunner.run_alpha_sensitivity()`
- **评估目标**: 分析注入强度 α (0.0~1.0) 对响应质量的影响
- **Alpha 取值**: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

#### 3.2 Ablation Study (消融实验)
- **数据文件**: `data/ablation.json`
- **生成器**: `ExperimentDataGenerator.generate_ablation_data()`
- **运行器**: `ExperimentRunner.run_ablation_study()`
- **评估目标**: 分析各组件贡献 (偏好注入 / 历史注入 / 门控机制)
- **消融条件**: full, no_preference, no_history, no_gating, baseline

### 4. 质量评估实验

#### 4.1 Multi-turn Coherence (多轮连贯性)
- **数据文件**: `data/multi_turn_coherence.json`
- **生成器**: `ExperimentDataGenerator.generate_multi_turn_coherence()`
- **运行器**: `ExperimentRunner.run_multi_turn_coherence()`
- **评估目标**: 连续多轮对话中的上下文一致性和流畅度

#### 4.2 Context Constrained (受限上下文)
- **数据文件**: `data/context_constrained.json`
- **生成器**: `ExperimentDataGenerator.generate_context_constrained_data()`
- **运行器**: `ExperimentRunner.run_context_constrained()`
- **评估目标**: 在有限上下文窗口下的记忆利用效率

### 5. 性能评估

#### 5.1 Latency Comparison (延迟对比)
- **运行器**: `ExperimentRunner.run_latency_comparison()`
- **评估目标**: DKI vs RAG vs Baseline 的推理延迟对比
- **指标**: 首轮延迟、后续轮延迟、归一化延迟 (ms/char)

## 评估指标体系

### 文本质量
| 指标 | 方法 | 说明 |
|---|---|---|
| BLEU | `compute_bleu()` | n-gram 精确率 (1~4 gram) |
| ROUGE-L | `compute_rouge()` | 最长公共子序列 F1 |

### 记忆召回
| 指标 | 方法 | 说明 |
|---|---|---|
| Memory Recall | `compute_memory_recall()` | 期望记忆在响应中的命中率 |
| Keyword Recall | `compute_content_recall()` | 关键词级别召回 |
| Injection Recall | `compute_content_recall()` | 实际注入/检索内容在响应中的覆盖率 |
| Semantic Recall | `compute_content_recall()` | 字符 n-gram 语义重叠度 |
| Combined Recall | `compute_content_recall()` | 加权综合召回 (DKI/RAG 使用不同权重) |

### 可靠性
| 指标 | 方法 | 说明 |
|---|---|---|
| Hallucination Rate | `compute_hallucination_rate()` | 整体幻觉率 |
| Hallucination Decomposed | `compute_hallucination_decomposed()` | 分解式幻觉评估 (新增事实/矛盾/不支持) |

### 性能
| 指标 | 方法 | 说明 |
|---|---|---|
| Latency Stats | `compute_latency_stats()` | P50/P95/P99 延迟统计 |

## 数据文件清单

| 文件名 | 类型 | 说明 |
|---|---|---|
| `persona_chat.json` | 人设对话 | 英文 persona (姓名/职业/爱好) |
| `cn_persona_chat.json` | 人设对话 | 中文 persona (饮食/城市/技能) |
| `memory_qa.json` | 记忆问答 | 事实性记忆精确召回 |
| `hotpot_qa.json` | 多跳推理 | 跨文档组合推理 |
| `long_session_persona_chat.json` | 长会话 | 50+ 轮长对话记忆保持 |
| `longmemeval_multi_turn.json` | LongMemEval | 多轮记忆基准 |
| `longmemeval_needle.json` | LongMemEval | 大海捞针记忆基准 |
| `longmemeval_oracle.json` | LongMemEval | 理想条件记忆基准 |
| `alpha_sensitivity.json` | 敏感性 | α 参数敏感性分析 |
| `ablation.json` | 消融 | 组件贡献分析 |
| `multi_turn_coherence.json` | 连贯性 | 多轮上下文一致性 |
| `context_constrained.json` | 受限上下文 | 有限窗口记忆效率 |

## 快速使用

### 生成实验数据

```python
from dki.experiment import ExperimentDataGenerator

gen = ExperimentDataGenerator()
gen.generate_all(output_dir="data/")
```

### 运行实验

```python
from dki.experiment import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    name="dki_vs_rag",
    modes=["dki", "rag", "baseline"],
    datasets=["persona_chat", "memory_qa"],
    max_samples=50,
    force_alpha=0.4,
)
runner = ExperimentRunner(config=config)
results = runner.run_experiment()
```

### 运行 LongMemEval

```python
runner = ExperimentRunner(config=config)
results = runner.run_longmemeval(data_path="data/longmemeval_multi_turn.json")
```

### 运行单元测试

```bash
cd DKI
python -m pytest dki/experiment/tests/ -v
```

## 实验结果

实验结果输出到 `experiment_results/` 目录，格式为 JSON，包含:
- 实验配置
- 每个样本的详细结果 (响应文本、延迟、注入信息)
- 聚合指标 (平均 BLEU/ROUGE/Recall/Latency)
- 实验报告 (Markdown) 由 `scripts/generate_experiment_report.py` 生成
