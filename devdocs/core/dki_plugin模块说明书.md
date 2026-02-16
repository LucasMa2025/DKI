# DKIPlugin 插件核心模块说明书 (v3.1 — Recall v4 集成版)

> 源文件: `DKI/dki/core/dki_plugin.py` (1021 行, 瘦 Facade)  
> 子组件包: `DKI/dki/core/plugin/` (4 文件)  
> 记忆召回包: `DKI/dki/core/recall/` (6 文件, Recall v4)  
> 模块路径: `dki.core.dki_plugin` + `dki.core.plugin` + `dki.core.recall`  
> 版本: 3.1.0

---

## 1. 模块概述

### 1.1 重构背景

v2.x 版本的 `DKIPlugin` 是一个 "God Plugin"，单文件承担了 7 项职责:

| 职责 | 说明 |
|------|------|
| 编排 | 串联 Memory Trigger → 数据加载 → 注入 → 推理 |
| 数据访问 | 通过适配器读取偏好和历史 |
| 语义理解 | Memory Trigger + Reference Resolver |
| 注入决策 | Alpha 计算、门控、策略选择 |
| 注入执行 | K/V 计算、缓存、模型调用 |
| 运维 | 统计、日志、可视化 |
| 配置管理 | 运行时策略切换、参数更新 |

v3.0 将其拆分为 **"决策-执行"分离架构**:

```
上层应用
   ↓ (稳定通信协议: chat(query, user_id, session_id))
DKIPlugin (瘦 Facade, 对外接口不变)
   ↓
InjectionPlanner (纯决策, 不碰模型)
   ↓ InjectionPlan (中间产物, 纯数据)
InjectionExecutor (纯执行, 不做决策)
   ↓
ModelAdapter (LLM 推理)
```

v3.1 新增 **Recall v4 记忆召回策略**，集成到 Planner/Executor 流程中:

```
InjectionPlanner
   ├─ MultiSignalRecall (多信号召回: 关键词 + 向量 + 指代)
   ├─ SuffixBuilder (逐消息 Summary + 后缀组装)
   └─ PromptFormatter (模型适配提示格式化)

InjectionExecutor
   ├─ FactRetriever (事实补充: trace_id → 原文)
   └─ PromptFormatter (Fact Call 检测 + 格式化)
```

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **对外接口不变** | `chat()`, `from_config()`, `get_stats()` 等签名完全兼容 v2.x |
| **决策与执行分离** | Planner 不持有模型引用, Executor 不做决策 |
| **中间产物可序列化** | `InjectionPlan` 是纯数据结构, 可序列化/缓存/重放/测试 |
| **Key 不缩放** | Key tensor 永远不被 alpha 缩放 (由 Executor 保证) |
| **安全边界** | `SafetyEnvelope` 验证参数合法性, 防止实验参数泄漏到生产 |
| **自动降级** | 任何异常自动降级到无注入推理 |
| **Recall v4 可选** | 通过配置启用, 未启用时回退到原有 flat_history 策略 |

### 1.3 文件结构

```
dki/core/
├── dki_plugin.py                    # 瘦 Facade (1021 行)
│   ├── InjectionMetadata            # 注入元数据 (监控用)
│   ├── DKIPluginResponse            # 插件响应
│   └── DKIPlugin                    # 主类 (Facade)
│
├── plugin/                          # 子组件包
│   ├── __init__.py                  # 包导出
│   ├── injection_plan.py            # 数据结构
│   │   ├── AlphaProfile             # 分层 Alpha 控制
│   │   ├── SafetyEnvelope           # 安全边界
│   │   ├── QueryContext             # 查询分析上下文
│   │   ├── InjectionPlan            # 注入计划 (核心中间产物, 含 Recall v4 字段)
│   │   └── ExecutionResult          # 执行结果 (含 fact_rounds_used, fact_tokens_total)
│   ├── injection_planner.py         # 决策层 (含 Recall v4 plan 构建)
│   │   └── InjectionPlanner         # 注入计划生成器
│   └── injection_executor.py        # 执行层 (含 Fact Call 循环)
│       └── InjectionExecutor        # 注入计划执行器
│
└── recall/                          # 记忆召回策略 v4 (新增)
    ├── __init__.py                  # 包导出
    ├── recall_config.py             # 配置数据结构 (RecallConfig, HistoryItem, ...)
    ├── multi_signal_recall.py       # 多信号召回 (关键词 + 向量 + 指代)
    ├── suffix_builder.py            # 后缀组装 (逐消息 Summary + 原文)
    ├── fact_retriever.py            # 事实检索 (trace_id → 原文, 支持分块)
    └── prompt_formatter.py          # 模型适配格式化 (Generic/DeepSeek/GLM)
```

### 1.4 策略概览

| 策略 | 状态 | 偏好注入 | 历史注入 | Context 占用 | 适用场景 |
|------|------|---------|---------|-------------|---------|
| **Stable** | ✅ 生产推荐 | K/V (负位置) | Suffix Prompt | 中 | 生产环境 |
| **Recall v4** | ✅ 新增 (v3.1) | K/V (负位置) | 多信号召回 + Summary + Fact Call | 动态 | 长历史场景 |
| **Full Attention** | ⚠️ 已弃用 | K/V (负位置) | K/V (负位置) | 极低 | 仅研究 |

---

## 2. 数据结构

### 2.1 AlphaProfile — 分层 Alpha 控制

> 源文件: `dki/core/plugin/injection_plan.py` 第 32-65 行

将单一 `alpha: float` 升级为结构化的分层控制，匹配 DKI 的三层语义模型:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `preference_alpha` | `float` | `0.4` | 偏好 K/V 注入强度 |
| `history_alpha` | `float` | `1.0` | 历史注入强度 (Stable=1.0, FA<1.0) |
| `override_cap` | `float` | `0.7` | 偏好 alpha 安全上限 |

**关键属性:**

```python
@property
def effective_preference_alpha(self) -> float:
    """确保偏好 alpha 不超过安全上限"""
    return min(self.preference_alpha, self.override_cap)
```

**三层语义模型:**

| 层 | 语义 | 稳定性要求 | Alpha 范围 |
|---|------|-----------|-----------|
| Preference | Personality / Bias | 极高 | 0.0 ~ override_cap |
| History | Episodic Memory | 中 | Stable=1.0, FA=0.1~0.5 |
| Query | Current Intent | 最高优先级 | 始终 1.0 |

**安全不变量:**
- `preference_alpha` 永远不超过 `override_cap`
- Key tensor 永远不被 alpha 缩放 (由 Executor 保证)

### 2.2 SafetyEnvelope — 安全边界

> 源文件: `dki/core/plugin/injection_plan.py` 第 72-126 行

不同策略下的参数安全约束。违规情况会被记录到 `InjectionPlan.safety_violations`，但不会阻止执行 (仅告警)。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stable_max_preference_alpha` | `float` | `0.5` | Stable 策略偏好 alpha 上限 |
| `stable_history_alpha_must_be` | `float` | `1.0` | Stable 策略历史 alpha 必须为 1.0 |
| `full_attention_max_preference_alpha` | `float` | `0.7` | FA 策略偏好 alpha 上限 |
| `full_attention_max_history_alpha` | `float` | `0.5` | FA 策略历史 alpha 上限 |
| `max_total_kv_tokens` | `int` | `600` | K/V token 总量上限 |

**验证逻辑:**

```python
def validate(self, plan: InjectionPlan) -> List[str]:
    violations = []
    if plan.strategy == "stable":
        if plan.alpha_profile.preference_alpha > self.stable_max_preference_alpha:
            violations.append(f"Stable: preference_alpha > max")
    elif plan.strategy == "full_attention":
        if plan.alpha_profile.preference_alpha > self.full_attention_max_preference_alpha:
            violations.append(f"FullAttention: preference_alpha > max")
        if plan.alpha_profile.history_alpha > self.full_attention_max_history_alpha:
            violations.append(f"FullAttention: history_alpha > max")
    return violations
```

### 2.3 QueryContext — 查询分析上下文

> 源文件: `dki/core/plugin/injection_plan.py` 第 132-157 行

由 `Planner.analyze_query()` 生成，在数据加载之前确定召回范围。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `recall_limit` | `int` | `10` | 历史召回轮数 |
| `memory_triggered` | `bool` | `False` | 是否触发记忆存储 |
| `trigger_type` | `Optional[str]` | `None` | 触发类型 |
| `trigger_confidence` | `float` | `0.0` | 触发置信度 |
| `reference_resolved` | `bool` | `False` | 是否解析了指代 |
| `reference_type` | `Optional[str]` | `None` | 指代类型 |
| `reference_scope` | `Optional[str]` | `None` | 指代范围 |

### 2.4 InjectionPlan — 注入计划 (核心中间产物)

> 源文件: `dki/core/plugin/injection_plan.py` 第 163-274 行

Planner 生成、Executor 消费的中间产物。纯数据结构，不包含任何逻辑。

| 字段分组 | 字段 | 类型 | 说明 |
|----------|------|------|------|
| **策略** | `strategy` | `str` | `"stable"` / `"full_attention"` / `"none"` |
| **偏好** | `preference_text` | `str` | 格式化后的偏好文本 |
| | `preferences_count` | `int` | 偏好数量 |
| | `preference_tokens` | `int` | 偏好 token 数 |
| **历史** | `history_suffix` | `str` | 格式化后的 suffix prompt (始终准备, 用于 fallback) |
| | `history_messages` | `List[Dict]` | 原始消息列表 (仅 FA 策略) |
| | `history_tokens` | `int` | 历史 token 数 |
| | `relevant_history_count` | `int` | 相关历史数量 |
| **查询** | `user_id` | `str` | 用户标识 (用于缓存键) |
| | `original_query` | `str` | 原始用户输入 |
| | `final_input` | `str` | 最终发给模型的输入 |
| | `query_tokens` | `int` | 查询 token 数 |
| | `total_tokens` | `int` | 总 token 数 |
| **Alpha** | `alpha_profile` | `AlphaProfile` | 分层 Alpha 控制 |
| **决策** | `injection_enabled` | `bool` | 是否启用注入 |
| | `gating_decision` | `Dict` | 门控决策详情 |
| **Trigger** | `memory_triggered` | `bool` | 是否触发记忆 |
| | `trigger_type` | `Optional[str]` | 触发类型 |
| **Reference** | `reference_resolved` | `bool` | 是否解析指代 |
| | `reference_type` | `Optional[str]` | 指代类型 |
| | `reference_scope` | `Optional[str]` | 指代范围 |
| | `recall_limit` | `int` | 召回轮数 |
| **FA 特有** | `global_indication` | `str` | 全局指示文本 |
| | `full_attention_fallback` | `bool` | 是否降级 |
| | `history_kv_tokens` | `int` | 历史 K/V token 数 |
| **Recall v4** | `assembled_suffix` | `str` | 组装后的后缀文本 (替代 history_suffix) |
| | `recall_strategy` | `str` | 召回策略 (`"summary_with_fact_call"` / `"flat_history"`) |
| | `summary_count` | `int` | suffix 中的 summary 条目数 |
| | `message_count` | `int` | suffix 中的原文消息数 |
| | `trace_ids` | `List[str]` | 所有 trace_id (用于 Fact Call) |
| | `has_fact_call_instruction` | `bool` | 是否包含 fact call 指导 |
| | `fact_rounds_used` | `int` | 实际使用的 fact call 轮次 |
| | `session_id` | `str` | 会话 ID (用于 fact retriever) |
| **安全** | `safety_violations` | `List[str]` | 安全违规列表 |

**可独立测试:**

```python
plan = InjectionPlan(
    strategy="stable",
    injection_enabled=True,
    preference_text="素食主义者",
    alpha_profile=AlphaProfile(preference_alpha=0.3),
)
assert plan.alpha_profile.effective_preference_alpha == 0.3
assert plan.to_dict()["strategy"] == "stable"
```

### 2.5 ExecutionResult — 执行结果

> 源文件: `dki/core/plugin/injection_plan.py` 第 281-313 行

Executor 的输出，包含模型输出和执行性能数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 生成的回复文本 |
| `input_tokens` | `int` | 输入 token 数 |
| `output_tokens` | `int` | 输出 token 数 |
| `raw_output` | `Optional[Any]` | 原始 ModelOutput |
| `inference_latency_ms` | `float` | 推理耗时 (ms) |
| `preference_cache_hit` | `bool` | 偏好缓存是否命中 |
| `preference_cache_tier` | `str` | 缓存层级 (`"memory"` / `"compute"` / `"error"`) |
| `full_attention_fallback` | `bool` | FA 是否降级 |
| `full_attention_position_mode` | `str` | FA 位置模式 |
| `full_attention_preference_tokens` | `int` | FA 偏好 token 数 |
| `full_attention_history_tokens` | `int` | FA 历史 token 数 |
| `fallback_used` | `bool` | 是否使用了降级 |
| `error_message` | `Optional[str]` | 错误信息 |
| `fact_rounds_used` | `int` | Recall v4 fact call 实际轮次 |
| `fact_tokens_total` | `int` | Recall v4 fact call 总 token 数 |

### 2.6 InjectionMetadata — 注入元数据 (监控用)

> 源文件: `dki/core/dki_plugin.py` 第 89-190 行

由 `DKIPlugin.chat()` 在 Planner 和 Executor 执行后组装，用于监控 API。

| 字段 | 类型 | 说明 |
|------|------|------|
| `injection_enabled` | `bool` | 注入是否启用 |
| `alpha` | `float` | 有效注入强度 (effective_preference_alpha) |
| `injection_strategy` | `str` | 注入策略 |
| `preference_tokens` | `int` | 偏好 token 数 |
| `history_tokens` | `int` | 历史 token 数 (Suffix Prompt) |
| `history_kv_tokens` | `int` | 历史 K/V token 数 (FA 特有) |
| `query_tokens` | `int` | 查询 token 数 |
| `total_tokens` | `int` | 总 token 数 |
| `preference_cache_hit` | `bool` | 偏好缓存命中 |
| `preference_cache_tier` | `str` | 缓存层级 |
| `latency_ms` | `float` | 总延迟 |
| `adapter_latency_ms` | `float` | 适配器延迟 |
| `injection_latency_ms` | `float` | 注入延迟 |
| `inference_latency_ms` | `float` | 推理延迟 |
| `gating_decision` | `Optional[Dict]` | 门控决策 |
| `preferences_count` | `int` | 偏好数量 |
| `history_messages_count` | `int` | 历史消息数量 |
| `relevant_history_count` | `int` | 相关历史数量 |
| `memory_triggered` | `bool` | 是否触发记忆 |
| `trigger_type` | `Optional[str]` | 触发类型 |
| `reference_resolved` | `bool` | 是否解析指代 |
| `reference_type` | `Optional[str]` | 指代类型 |
| `reference_scope` | `Optional[str]` | 指代范围 |
| `full_attention_fallback` | `bool` | FA 是否降级 |
| `alpha_profile` | `Optional[Dict]` | 分层 Alpha 详情 (v3.0 新增) |
| `safety_violations` | `Optional[List[str]]` | 安全违规列表 (v3.0 新增) |
| `request_id` | `str` | 请求唯一标识 (8位 UUID) |
| `timestamp` | `datetime` | 时间戳 |

### 2.7 DKIPluginResponse — 插件响应

> 源文件: `dki/core/dki_plugin.py` 第 193-215 行

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 生成的回复文本 |
| `input_tokens` | `int` | 输入 token 数 |
| `output_tokens` | `int` | 输出 token 数 |
| `metadata` | `InjectionMetadata` | 注入元数据 |
| `raw_output` | `Optional[ModelOutput]` | 原始模型输出 |

---

## 3. DKIPlugin — 瘦 Facade

> 源文件: `dki/core/dki_plugin.py` 第 218-1021 行

### 3.1 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_adapter` | `BaseModelAdapter` | LLM 模型适配器 |
| `user_data_adapter` | `IUserDataAdapter` | 外部数据适配器 |
| `config` | `Optional[Any]` | 配置 (默认从 config.yaml 加载) |
| `language` | `str` | 语言 (`"en"` / `"cn"`) |
| `memory_trigger_config` | `Optional[MemoryTriggerConfig]` | Memory Trigger 配置 |
| `reference_resolver_config` | `Optional[ReferenceResolverConfig]` | Reference Resolver 配置 |
| `redis_client` | `Optional[DKIRedisClient]` | Redis 客户端 |
| `cache_config` | `Optional[CacheConfig]` | 缓存配置 |

### 3.2 初始化流程

```
DKIPlugin.__init__()
  ├─ 存储核心依赖 (model_adapter, data_adapter, config)
  ├─ DKI 组件 (延迟初始化)
  │   ├─ _mis: MemoryInfluenceScaling (property 延迟创建)
  │   └─ _gating: DualFactorGating (property 延迟创建)
  ├─ 确定注入策略 (_get_injection_strategy)
  │   ├─ config.dki.injection_strategy
  │   └─ 默认 "stable"
  ├─ 创建 InjectionPlanner (纯决策)
  │   ├─ 传入 config, language, injection_strategy
  │   ├─ 内部创建 MemoryTrigger
  │   ├─ 内部创建 ReferenceResolver
  │   ├─ 内部创建 SafetyEnvelope
  │   └─ (v3.1) 接受 recall_config, multi_signal_recall, suffix_builder
  ├─ 创建 FullAttentionInjector (仅 full_attention 策略)
  ├─ 创建 InjectionExecutor (纯执行)
  │   ├─ 传入 model_adapter
  │   ├─ 传入 full_attention_injector (可选)
  │   └─ (v3.1) 接受 fact_retriever, prompt_formatter, recall_config
  ├─ 创建 PreferenceCacheManager (支持 L1 + L2 Redis)
  ├─ 工作日志 (_injection_logs, 最多 1000 条)
  └─ 统计数据 (_stats)
```

### 3.3 from_config() — 推荐创建方式

```
DKIPlugin.from_config(model_adapter, adapter_config_path, ...)
  ├─ 加载全局配置 (ConfigLoader)
  ├─ 加载原始 YAML 配置字典
  ├─ 创建配置驱动的适配器
  │   ├─ adapter_config_path → ConfigDrivenAdapter.from_yaml(path)
  │   ├─ adapter_config (dict) → ConfigDrivenAdapter.from_dict(config)
  │   └─ 默认路径搜索 → config/adapter_config.yaml 等
  ├─ 连接数据库: await user_adapter.connect()
  ├─ 处理 Memory Trigger 配置
  │   ├─ 参数传入 → MemoryTriggerConfig(**config)
  │   └─ 配置文件 → MemoryTriggerConfig.from_dict(raw_config['memory_trigger'])
  ├─ 处理 Reference Resolver 配置
  │   ├─ 参数传入 → ReferenceResolverConfig(**config)
  │   └─ 配置文件 → ReferenceResolverConfig.from_dict(raw_config['reference_resolver'])
  ├─ 处理 Redis 配置
  │   ├─ enable_redis 参数 或 配置文件 redis.enabled
  │   ├─ 创建 DKIRedisClient → await redis_client.connect()
  │   └─ 连接失败 → 降级为 L1 only
  └─ 创建 DKIPlugin 实例
```

### 3.4 组件访问器 (高级用法 / 测试)

```python
# 直接访问 Planner (用于测试/调试)
context = plugin.planner.analyze_query("推荐餐厅")
plan = plugin.planner.build_plan(query, user_id, prefs, history, context)
print(plan.to_dict())  # 检查决策

# 直接访问 Executor (用于测试/调试)
result = await plugin.executor.execute(plan)

# 延迟初始化的 DKI 组件
mis = plugin.mis       # MemoryInfluenceScaling
gating = plugin.gating # DualFactorGating
```

---

## 4. chat() 主流程 — 完整调用链

> 源文件: `dki/core/dki_plugin.py` 第 571-720 行

```
DKIPlugin.chat(query, user_id, session_id, ...)
  │
  ├─ Step 1: 分析查询 (Planner Phase 1)
  │   └─ planner.analyze_query(query) → QueryContext
  │       ├─ MemoryTrigger.detect(query)
  │       │   └─ 检测用户是否在告知偏好/记忆 (如 "我对花生过敏")
  │       │       返回 TriggerResult(triggered, trigger_type, confidence)
  │       │
  │       └─ ReferenceResolver.resolve(query)
  │           └─ 解析指代表达，确定历史召回范围
  │               "刚刚说的" → recall_limit=5
  │               "最近聊的" → recall_limit=20
  │               "那件事"   → recall_limit=15
  │               无指代     → recall_limit=10 (默认)
  │
  ├─ Step 2: 通过适配器读取外部数据
  │   ├─ data_adapter.get_user_preferences(user_id)
  │   │   └─ 从上层应用数据库读取用户偏好
  │   │       返回 List[UserPreference]
  │   │
  │   └─ data_adapter.search_relevant_history(
  │       user_id, query, session_id, limit=context.recall_limit)
  │       └─ 从上层应用数据库检索相关历史
  │           返回 List[ChatMessage]
  │
  ├─ Step 3: 构建注入计划 (Planner Phase 2)
  │   └─ planner.build_plan(query, user_id, prefs, history, context)
  │       → InjectionPlan (纯数据, 可序列化)
  │       ├─ 格式化偏好文本
  │       ├─ 策略分支:
  │       │   ├─ Recall v4 (如果配置启用且组件可用):
  │       │   │   ├─ MultiSignalRecall.recall() → 多信号召回
  │       │   │   ├─ SuffixBuilder.build() → 逐消息 Summary + 后缀组装
  │       │   │   └─ 填充 assembled_suffix, trace_ids, summary_count 等
  │       │   │
  │       │   ├─ Stable: 格式化历史后缀 (始终准备, 用于 fallback)
  │       │   └─ Full Attention: 额外准备原始消息列表
  │       │
  │       ├─ 计算分层 Alpha (AlphaProfile)
  │       ├─ 注入决策 (是否注入)
  │       ├─ 构造最终输入 (final_input)
  │       ├─ 门控决策记录
  │       └─ 安全验证 (SafetyEnvelope)
  │
  ├─ Step 4: 执行注入计划 (Executor)
  │   └─ executor.execute(plan, max_new_tokens, temperature)
  │       → ExecutionResult
  │       ├─ strategy == "full_attention" → _execute_full_attention()
  │       ├─ injection_enabled + 有偏好 → _execute_stable()
  │       │   └─ (v3.1) 如果 plan.has_fact_call_instruction:
  │       │       └─ _execute_fact_call_loop() → 事实补充循环
  │       └─ 其他 → _execute_plain()
  │
  ├─ Step 5: 记录工作数据
  │   ├─ 组装 InjectionMetadata (从 plan + result)
  │   ├─ _record_injection_log(metadata)
  │   └─ record_visualization() → 可视化 API 数据
  │
  └─ 返回 DKIPluginResponse
  
  异常处理:
  └─ 降级: 直接调用 model.generate(query) (无注入)
```

### 4.1 数据流图

```
                    ┌──────────────────────────────────────────────────┐
                    │              DKIPlugin.chat()                     │
                    │                (瘦 Facade)                       │
                    └──────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
            │  Planner     │  │  Adapter      │  │  Executor        │
            │  Phase 1     │  │  (数据加载)   │  │  (模型调用)      │
            │  analyze_    │  │  get_prefs()  │  │  execute()       │
            │  query()     │  │  search_      │  │                  │
            │              │  │  history()    │  │                  │
            └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
                   │                 │                    │
                   ▼                 ▼                    ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
            │ QueryContext │  │ preferences  │  │ ExecutionResult  │
            │ (recall_     │  │ + history    │  │ (text, latency,  │
            │  limit)      │  │              │  │  cache_hit,      │
            └──────┬───────┘  └──────┬───────┘  │  fact_rounds)    │
                   │                 │           └────────┬─────────┘
                   └────────┬────────┘                    │
                            ▼                             │
                    ┌──────────────┐                      │
                    │  Planner     │                      │
                    │  Phase 2     │                      │
                    │  build_plan()│                      │
                    │  ┌─────────┐ │                      │
                    │  │Recall v4│ │                      │
                    │  │ (可选)  │ │                      │
                    │  └─────────┘ │                      │
                    └──────┬───────┘                      │
                           │                              │
                           ▼                              │
                    ┌──────────────┐                      │
                    │ InjectionPlan│──────────────────────┘
                    │ (纯数据)     │
                    │ + trace_ids  │
                    │ + assembled_ │
                    │   suffix     │
                    └──────────────┘
```

---

## 5. InjectionPlanner — 决策层

> 源文件: `dki/core/plugin/injection_planner.py` (616 行)

### 5.1 设计约束

| 约束 | 说明 |
|------|------|
| 不持有模型引用 | Planner 不依赖 `BaseModelAdapter` |
| 不执行推理 | 不调用 `model.generate()` 或 `model.compute_kv()` |
| 输出可序列化 | `InjectionPlan` 是纯数据, 可 `to_dict()` |
| 可独立测试 | 不需要 GPU 或模型即可测试所有决策逻辑 |

### 5.2 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `Optional[Any]` | DKI 配置对象 |
| `language` | `str` | 语言 (`"en"` / `"cn"`) |
| `injection_strategy` | `str` | 默认注入策略 (`"stable"` / `"full_attention"` / `"recall_v4"`) |
| `memory_trigger` | `Optional[MemoryTrigger]` | Memory Trigger 实例 (可注入) |
| `memory_trigger_config` | `Optional[MemoryTriggerConfig]` | Memory Trigger 配置 |
| `reference_resolver` | `Optional[ReferenceResolver]` | Reference Resolver 实例 (可注入) |
| `reference_resolver_config` | `Optional[ReferenceResolverConfig]` | Reference Resolver 配置 |
| `safety_envelope` | `Optional[SafetyEnvelope]` | 安全边界 (可注入) |
| `recall_config` | `Optional[RecallConfig]` | Recall v4 配置 (v3.1 新增) |
| `multi_signal_recall` | `Optional[MultiSignalRecall]` | 多信号召回器 (v3.1 新增) |
| `suffix_builder` | `Optional[SuffixBuilder]` | 后缀组装器 (v3.1 新增) |

### 5.3 Phase 1: analyze_query() — 查询分析

> 源文件: `dki/core/plugin/injection_planner.py` 第 181-228 行

在数据加载之前调用，为适配器提供 `recall_limit`。

```
analyze_query(query)
  ├─ MemoryTrigger.detect(query)
  │   ├─ 匹配正则模式 (元认知/状态变化/长期价值/召回请求/观点查询)
  │   └─ 返回 TriggerResult(triggered, trigger_type, confidence)
  │
  └─ ReferenceResolver.resolve(query)
      ├─ 匹配指代关键词 ("刚刚"/"最近"/"那件事"/"你之前说的")
      └─ 返回 ResolvedReference(reference_type, scope, recall_turns)
```

**指代类型与召回轮数:**

| 指代类型 | 触发词示例 | 默认召回轮数 | 说明 |
|----------|-----------|-------------|------|
| `NONE` | (无指代) | 10 | 默认召回 |
| `JUST_NOW` | "刚刚"、"刚才" | 5 | 最近几轮 |
| `RECENTLY` | "最近"、"前几天" | 20 | 较大范围 |
| `LAST_TOPIC` | "那件事"、"上次聊的" | 15 | 话题相关 |
| `ASSISTANT_STANCE` | "你之前说的"、"你建议的" | 10 | 助手立场 |

### 5.4 Phase 2: build_plan() — 构建注入计划

> 源文件: `dki/core/plugin/injection_planner.py` 第 234-394 行

```
build_plan(query, user_id, preferences, relevant_history, context, force_alpha,
           strategy_override, session_id, context_window, db_session)
  │
  ├─ Step 1: 格式化偏好
  │   └─ _format_preferences(preferences)
  │       ├─ 按 priority 降序排序
  │       ├─ 过滤过期偏好 (is_expired())
  │       └─ 格式: "- {type}: {text}"
  │
  ├─ Step 2: 格式化历史 (策略分支)
  │   ├─ 判断 use_recall_v4:
  │   │   ├─ strategy == "recall_v4"
  │   │   └─ strategy == "stable" + recall_config.enabled + strategy == "summary_with_fact_call"
  │   │
  │   ├─ Recall v4 分支:
  │   │   └─ _build_recall_v4_plan(plan, query, session_id, ...)
  │   │       ├─ MultiSignalRecall.recall(query, session_id, user_id)
  │   │       ├─ SuffixBuilder.build(query, recalled_messages, context_window, ...)
  │   │       └─ 填充 plan.assembled_suffix, trace_ids, summary_count 等
  │   │       └─ 异常时回退到 flat_history_fallback
  │   │
  │   ├─ Stable 分支: _format_history_suffix(relevant_history) → 始终准备
  │   └─ Full Attention 分支: 额外准备原始消息列表
  │
  ├─ Step 3: 计算分层 Alpha
  │   └─ _compute_alpha_profile(strategy, preferences, history, force_alpha)
  │
  ├─ Step 4: 注入决策
  │   ├─ force_alpha > 0.1 → 启用
  │   └─ 有偏好 or 有历史 or recall_v4 有结果 → 启用
  │
  ├─ Step 5: 构造最终输入
  │   ├─ Recall v4 + assembled_suffix → final_input = assembled_suffix
  │   ├─ Full Attention → final_input = query (全局指示由 Executor 添加)
  │   └─ Stable → final_input = history_suffix + "\n\n" + query
  │
  ├─ Step 6: 门控决策记录
  │   └─ gating_decision = {should_inject, alpha, strategy, recall_strategy, ...}
  │
  └─ Step 7: 安全验证
      └─ SafetyEnvelope.validate(plan) → violations
```

### 5.5 Recall v4 计划构建 (_build_recall_v4_plan)

> 源文件: `dki/core/plugin/injection_planner.py` 第 400-451 行

```
_build_recall_v4_plan(plan, query, session_id, user_id, context_window, db_session)
  │
  ├─ Phase 1: 多信号召回
  │   └─ MultiSignalRecall.recall(query, session_id, user_id, db_session)
  │       ├─ 关键词检索 (TF-IDF/TextRank + jieba 分词)
  │       ├─ 向量相似度检索 (MemoryRouter/EmbeddingService)
  │       ├─ 指代解析 (ReferenceResolver)
  │       ├─ 分数归一化 + 加权融合
  │       │   └─ final_score = w1 * norm_keyword + w2 * norm_vector + w3 * recency
  │       └─ 返回 RecallResult(messages, scores)
  │
  ├─ Phase 2: 后缀组装
  │   └─ SuffixBuilder.build(query, recalled_messages, context_window, preference_tokens)
  │       ├─ 遍历消息, 逐条判断:
  │       │   ├─ token数 > per_message_threshold → Summary + trace_id
  │       │   │   └─ 结构化认知标记: [SUMMARY] facts_covered/missing, confidence, trace_id [/SUMMARY]
  │       │   └─ token数 <= threshold → 保留原文
  │       ├─ 动态 token 预算管理 (context_window - generation_reserve - instruction_reserve - ...)
  │       ├─ 格式化: PromptFormatter.format_summary_item() / format_message_item()
  │       ├─ 添加约束指令: PromptFormatter.get_constraint_instruction()
  │       │   └─ 包含强约束: "若未调用 retrieve_fact 直接基于 summary 给出具体数值/时间/引用原话, 该回答视为无效"
  │       └─ 返回 AssembledSuffix(text, total_tokens, trace_ids, summary_count, ...)
  │
  ├─ 填充 plan 字段
  │   ├─ plan.assembled_suffix = assembled.text
  │   ├─ plan.history_suffix = assembled.text  # 兼容 fallback
  │   ├─ plan.trace_ids = assembled.trace_ids
  │   └─ plan.has_fact_call_instruction = assembled.has_fact_call_instruction
  │
  └─ 异常处理
      └─ 回退到 flat_history_fallback
```

### 5.6 偏好格式化示例

```
输入偏好:
  - UserPreference(text="素食主义者，不吃肉", type="dietary", priority=10)
  - UserPreference(text="花生过敏", type="allergy", priority=9)
  - UserPreference(text="喜欢简洁的回复风格", type="style", priority=5)

格式化输出:
  - dietary: 素食主义者，不吃肉
  - allergy: 花生过敏
  - style: 喜欢简洁的回复风格
```

### 5.7 历史后缀格式化示例 (Stable 策略, 中文)

```
[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
---
用户: 推荐一家北京的餐厅
助手: 推荐海底捞，他们可以根据过敏情况定制菜单。
用户: 那家店在哪里?
助手: 海底捞在朝阳区有多家分店，最近的在望京。
---
[会话历史结束]
请基于以上历史和用户当前问题给出回复。
```

### 5.8 Alpha 计算规则

```
_compute_alpha_profile(strategy, preferences, history, force_alpha)
  │
  ├─ force_alpha 指定:
  │   └─ AlphaProfile(
  │        preference_alpha=force_alpha,
  │        history_alpha=1.0 if stable else force_alpha)
  │
  ├─ Stable 策略 (含 Recall v4):
  │   ├─ preference_alpha = config.dki.hybrid_injection.preference.alpha (默认 0.4)
  │   └─ history_alpha = 1.0 (suffix prompt 全量使用)
  │
  ├─ Full Attention 策略:
  │   ├─ preference_alpha = config.dki.full_attention.preference_alpha
  │   └─ history_alpha = config.dki.full_attention.history_alpha
  │
  └─ 无偏好:
      └─ preference_alpha = 0.0
```

### 5.9 运行时配置更新

```python
# 更新 Reference Resolver
planner.update_reference_resolver_config(
    just_now_turns=3,
    recently_turns=30,
)

# 更新 Memory Trigger
planner.update_memory_trigger_config(
    enabled=True,
    custom_patterns=[...],
)

# 切换注入策略
planner.injection_strategy = "recall_v4"  # v3.1 新增
```

### 5.10 统计数据

```python
planner.get_stats()
# {
#     "plans_created": 150,
#     "safety_violations": 2,
#     "memory_trigger_count": 15,
#     "reference_resolved_count": 8,
#     "recall_v4_plans": 42,          # v3.1 新增
#     "strategy": "stable",
#     "memory_trigger": {...},
#     "reference_resolver": {...},
# }
```

---

## 6. InjectionExecutor — 执行层

> 源文件: `dki/core/plugin/injection_executor.py` (614 行)

### 6.1 设计约束

| 约束 | 说明 |
|------|------|
| 不做决策 | 不判断"是否注入"，只按 Plan 执行 |
| Key 不缩放 | Key tensor 永远不被 alpha 缩放 |
| Alpha 受限 | 使用 `effective_preference_alpha` (受 override_cap 约束) |
| 自动降级 | 任何异常自动降级到无注入推理 |

### 6.2 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_adapter` | `BaseModelAdapter` | LLM 模型适配器 |
| `full_attention_injector` | `Optional[FullAttentionInjector]` | FA 注入器 (可选) |
| `fact_retriever` | `Optional[FactRetriever]` | 事实检索器 (Recall v4, 可选) |
| `prompt_formatter` | `Optional[PromptFormatter]` | 提示格式化器 (Recall v4, 可选) |
| `recall_config` | `Optional[RecallConfig]` | 召回配置 (Recall v4, 可选) |

### 6.3 execute() — 主执行入口

> 源文件: `dki/core/plugin/injection_executor.py` 第 114-176 行

```
execute(plan, max_new_tokens, temperature)
  │
  ├─ 路由判断:
  │   ├─ strategy=="full_attention" + 注入器存在 + injection_enabled
  │   │   → _execute_full_attention()
  │   ├─ injection_enabled + 有偏好文本
  │   │   → _execute_stable()
  │   │   └─ (v3.1) 如果 plan.has_fact_call_instruction + recall 组件可用:
  │   │       └─ _execute_fact_call_loop() → 事实补充循环
  │   └─ 其他
  │       → _execute_plain()
  │
  └─ 异常处理:
      └─ _execute_fallback() (无注入推理)
```

### 6.4 Stable 策略执行

> 源文件: `dki/core/plugin/injection_executor.py` 第 182-236 行

```
_execute_stable(plan)
  │
  ├─ 获取偏好 K/V (含内存缓存)
  │   └─ _get_preference_kv(user_id, preference_text)
  │       ├─ cache_key = f"{user_id}:{md5(preference_text)}"
  │       ├─ 检查内存缓存 → 命中返回 (kv, True, "memory")
  │       ├─ 计算 K/V: model.compute_kv(preference_text)
  │       ├─ 存入缓存
  │       └─ 返回 (kv, False, "compute")
  │
  ├─ 使用 effective alpha (受 override_cap 约束)
  │   └─ alpha = plan.alpha_profile.effective_preference_alpha
  │
  └─ LLM 推理
      ├─ 有 K/V + alpha > 0.1
      │   → model.forward_with_kv_injection(final_input, preference_kv, alpha)
      └─ 否则
          → model.generate(final_input)
```

### 6.5 Fact Call 循环 (Recall v4)

> 源文件: `dki/core/plugin/injection_executor.py` 第 433-537 行

```
_execute_fact_call_loop(plan, initial_result, max_new_tokens, temperature)
  │
  ├─ 配置读取
  │   ├─ max_rounds = recall_config.fact_call.max_rounds (默认 3)
  │   └─ max_fact_tokens = recall_config.fact_call.max_fact_tokens (默认 800)
  │
  ├─ 循环 (最多 max_rounds 轮):
  │   ├─ 检测 Fact Call
  │   │   └─ prompt_formatter.detect_fact_request(result.text)
  │   │       ├─ 解析 retrieve_fact(trace_id="msg-005", offset=0, limit=5)
  │   │       └─ 返回 FactRequest 或 None
  │   │
  │   ├─ 如果无 Fact Call → break ✅
  │   │
  │   ├─ 检索事实
  │   │   └─ fact_retriever.retrieve(trace_id, session_id, offset, limit)
  │   │       └─ 返回 FactResponse(messages, total_count, has_more)
  │   │
  │   ├─ 格式化事实段落
  │   │   └─ prompt_formatter.format_fact_segment(fact_response)
  │   │
  │   ├─ Token 预算检查
  │   │   └─ total_fact_tokens > max_fact_tokens → break
  │   │
  │   ├─ 追加事实到 prompt
  │   │   └─ prompt += "\n\n" + fact_text + "\n\n" + continuation
  │   │
  │   └─ 重新推理 (带偏好 K/V 注入)
  │
  └─ 更新 result.fact_rounds_used, result.fact_tokens_total
```

**Fact Call 循环示意:**

```
Round 0: 初始推理 → 模型输出包含 retrieve_fact(trace_id="msg-005")
  ↓
Round 1: 检索 msg-005 原文 → 追加到 prompt → 重新推理
  ↓ 模型输出包含 retrieve_fact(trace_id="msg-012", offset=500, limit=500)
Round 2: 检索 msg-012 后半段 → 追加到 prompt → 重新推理
  ↓ 模型输出不含 fact call
结束: 返回最终结果 (fact_rounds_used=2)
```

### 6.6 Full Attention 策略执行 (⚠️ 已弃用)

> 源文件: `dki/core/plugin/injection_executor.py` 第 242-325 行

```
_execute_full_attention(plan)
  │
  ├─ 调用 Full Attention 注入器
  │   └─ fa_injector.inject(model, preference_text, history_messages, query)
  │       └─ 返回 InjectionResult
  │
  ├─ 降级检查
  │   └─ 失败或 fallback → _execute_stable_fallback()
  │
  ├─ 构造最终输入 (仅包含全局指示 + 查询)
  │   └─ final_input = global_indication + "\n" + query
  │
  └─ LLM 推理
      └─ model.forward_with_kv_injection(final_input, merged_kv, alpha)
```

> ⚠️ **弃用说明**: Full Attention 策略因 K/V 容量有限、不可引用、OOD 风险等原因已弃用。
> 推荐使用 Recall v4 策略处理长历史场景。代码保留用于向后兼容和研究目的。

### 6.7 降级机制

```
降级层级:
  ├─ Full Attention 失败
  │   └─ _execute_stable_fallback()
  │       ├─ 使用 plan.history_suffix 重构 stable 输入
  │       └─ 调用 _execute_stable()
  │
  ├─ Recall v4 失败
  │   └─ Planner 回退到 flat_history_fallback
  │       └─ 使用原有 stable 格式化
  │
  └─ 任何执行异常
      └─ _execute_fallback()
          └─ model.generate(plan.original_query) (仅原始查询, 无注入)
```

### 6.8 K/V 缓存

> 源文件: `dki/core/plugin/injection_executor.py` 第 543-571 行

```
_get_preference_kv(user_id, preference_text)
  ├─ 生成缓存键: cache_key = f"{user_id}:{md5(preference_text)}"
  ├─ 检查内存缓存: _preference_kv_cache[cache_key]
  │   └─ 命中 + hash 匹配 → 返回 (kv_entries, True, "memory")
  ├─ 计算 K/V: model.compute_kv(preference_text)
  │   └─ 返回 List[KVCacheEntry]
  │       每个 entry: key=[B, H, S, D], value=[B, H, S, D]
  ├─ 存入缓存: _preference_kv_cache[cache_key] = (kv_entries, content_hash)
  └─ 返回 (kv_entries, False, "compute")
  
  异常:
  └─ 返回 (None, False, "error")
```

### 6.9 统计数据

```python
executor.get_stats()
# {
#     "executions": 150,
#     "stable_executions": 120,
#     "full_attention_executions": 5,
#     "plain_executions": 10,
#     "recall_v4_executions": 42,    # v3.1 新增
#     "fallbacks": 2,
#     "cache_hits": 85,
#     "fact_call_rounds": 15,        # v3.1 新增
#     "kv_cache_size": 15,
#     "full_attention_available": True,
# }
```

---

## 7. 注入策略对比

### 7.1 Stable 策略 (默认, 生产推荐)

| 维度 | 说明 |
|------|------|
| **偏好注入** | K/V 注入 (负位置), alpha=0.3~0.5 |
| **历史注入** | Suffix Prompt (正位置), alpha=1.0 |
| **Context 占用** | 历史后缀 + 查询 (中等) |
| **稳定性** | 高 (历史可引用, 可追溯) |
| **适用场景** | 生产环境, 需要可解释性, 短历史 |

### 7.2 Recall v4 策略 (v3.1 新增, 长历史推荐)

| 维度 | 说明 |
|------|------|
| **偏好注入** | K/V 注入 (负位置), alpha=0.3~0.5 |
| **历史注入** | 多信号召回 + 逐消息 Summary + Fact Call |
| **Context 占用** | 动态 (根据 context_window 自适应) |
| **稳定性** | 高 (summary 含 trace_id, 可追溯补充事实) |
| **适用场景** | 长历史场景, 需要精确事实的场景 |
| **依赖** | jieba (中文分词), 可选 LLM Summary |

### 7.3 Full Attention 策略 (⚠️ 已弃用)

| 维度 | 说明 |
|------|------|
| **偏好注入** | K/V 注入 (负位置 -100~-1) |
| **历史注入** | K/V 注入 (负位置 -500~-101) |
| **Context 占用** | 仅全局指示 + 查询 (约 3-5 tokens) |
| **稳定性** | 低 (K/V 容量有限, OOD 风险, 不可引用) |
| **弃用原因** | 长历史场景不可用, 事实准确性不足 |

### 7.4 策略选择决策树

```
                    ┌─────────────────┐
                    │ injection_enabled│
                    │     == False?    │
                    └────────┬────────┘
                         Yes │ No
                             │
                    ┌────────▼────────┐
                    │  _execute_plain  │
                    │  (无注入推理)    │
                    └─────────────────┘
                             │ No
                    ┌────────▼────────┐
                    │ strategy ==      │
                    │ "full_attention"?│
                    └────────┬────────┘
                         Yes │ No
                             │
              ┌──────────────▼──────────────┐
              │ _execute_full_attention      │
              │ (⚠️ 已弃用, 仅研究)         │
              │ 失败? → _execute_stable_     │
              │         fallback             │
              └──────────────────────────────┘
                             │ No
              ┌──────────────▼──────────────┐
              │ preference_text 非空?        │
              └──────────────┬──────────────┘
                         Yes │ No
                             │
              ┌──────────────▼──────────────┐
              │ _execute_stable              │
              │ (偏好 K/V + 历史 Suffix)    │
              │                              │
              │ has_fact_call_instruction?    │
              │   → _execute_fact_call_loop  │
              │     (Recall v4 事实补充)     │
              └──────────────────────────────┘
                             │ No
              ┌──────────────▼──────────────┐
              │ _execute_plain               │
              │ (无注入推理)                 │
              └──────────────────────────────┘
```

---

## 8. 缓存架构

### 8.1 偏好 K/V 缓存层级

```
┌─────────────────────────────────────────────────────────────┐
│  层级          │  存储              │  命中率影响            │
├────────────────┼────────────────────┼────────────────────────┤
│  L0: Executor  │  _preference_kv_   │  最快, 单实例有效      │
│  内存字典      │  cache (Dict)      │                        │
│  L1: 内存 LRU  │  PreferenceCache   │  单实例 ~70%          │
│                │  Manager           │                        │
│  L2: Redis     │  DKIRedisClient    │  多实例 ~70% (恒定)   │
│  L3: 重算      │  model.compute_kv  │  始终可用             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 缓存键设计

```
cache_key = f"{user_id}:{md5(preference_text)}"

示例:
  user_123:a1b2c3d4e5f6...
```

当用户偏好更新时，`md5(preference_text)` 变化，旧缓存自动失效。

### 8.3 多实例部署

```
没有 Redis: 命中率 = 70% / N (N = 实例数)
  - 2 实例: 35%
  - 4 实例: 17.5%
  
有 Redis: 命中率 = 70% (恒定)
  - 所有实例共享 L2 缓存
```

---

## 9. 数据库交互

### 9.1 通过适配器访问的上层应用表

| 表名 | 操作 | 说明 |
|------|------|------|
| `user_preferences` (上层应用) | 读 | 用户偏好 |
| `chat_messages` (上层应用) | 读 | 历史消息 |

### 9.2 DKI 内部存储

| 存储 | 操作 | 说明 |
|------|------|------|
| 可视化数据 (内存) | 写 | `record_visualization()` |
| 注入日志 (内存) | 写 | `_injection_logs` (最多 1000 条) |
| 偏好 K/V 缓存 (内存) | 读/写 | `_preference_kv_cache` |

### 9.3 Recall v4 数据库交互

| 组件 | 表 | 操作 | 说明 |
|------|-----|------|------|
| `MultiSignalRecall` | `conversations` | 读 | 检索历史消息 (关键词 + 向量) |
| `FactRetriever` | `conversations` | 读 | 按 trace_id 检索原文 (支持分块) |

---

## 10. 监控 API

### 10.1 get_stats() — 统计数据

> 源文件: `dki/core/dki_plugin.py` 第 793-848 行

返回聚合统计，包含 Facade 层、Planner 层、Executor 层三级数据:

```python
plugin.get_stats()
# {
#     # Facade 层统计
#     "total_requests": 150,
#     "injection_enabled_count": 120,
#     "injection_rate": 0.8,
#     "cache_hits": 85,
#     "cache_hit_rate": 0.567,
#     "avg_latency_ms": 45.3,
#     "avg_alpha": 0.35,
#
#     # Planner 层统计
#     "planner": {
#         "plans_created": 150,
#         "safety_violations": 2,
#         "memory_trigger_count": 15,
#         "reference_resolved_count": 8,
#         "recall_v4_plans": 42,          # v3.1 新增
#         "strategy": "stable",
#         "memory_trigger": {...},
#         "reference_resolver": {...},
#     },
#
#     # Executor 层统计
#     "executor": {
#         "executions": 150,
#         "stable_executions": 120,
#         "full_attention_executions": 5,
#         "plain_executions": 10,
#         "recall_v4_executions": 42,     # v3.1 新增
#         "fallbacks": 2,
#         "cache_hits": 85,
#         "fact_call_rounds": 15,         # v3.1 新增
#         "kv_cache_size": 15,
#         "full_attention_available": True,
#     },
#
#     # 缓存统计 (含 Redis)
#     "cache": {...},
#
#     # 向后兼容: 扁平化旧字段
#     "memory_trigger_count": 15,
#     "memory_trigger_rate": 0.1,
#     "reference_resolved_count": 8,
#     "reference_resolved_rate": 0.053,
# }
```

### 10.2 get_injection_logs() — 注入日志

```python
logs = plugin.get_injection_logs(limit=100, offset=0)
# [
#     {
#         "request_id": "a1b2c3d4",
#         "timestamp": "2026-02-16T10:30:00",
#         "injection_enabled": True,
#         "injection_strategy": "stable",
#         "alpha": 0.35,
#         "alpha_profile": {
#             "preference_alpha": 0.4,
#             "history_alpha": 1.0,
#             "override_cap": 0.7,
#             "effective_preference_alpha": 0.4,
#         },
#         "tokens": {"preference": 25, "history": 150, ...},
#         "cache": {"preference_hit": True, "preference_tier": "memory"},
#         "latency": {"total_ms": 42.5, ...},
#         "safety_violations": [],
#         ...
#     },
#     ...
# ]
```

---

## 11. 运行时配置更新

### 11.1 切换注入策略

```python
# 运行时切换到 Recall v4 策略 (v3.1 新增)
plugin.switch_injection_strategy("recall_v4")

# 切换到 Stable 策略
plugin.switch_injection_strategy("stable")

# 切换到 Full Attention 策略 (⚠️ 已弃用)
plugin.switch_injection_strategy("full_attention")
# 内部: 如果 executor 没有 FA 注入器, 自动初始化
```

### 11.2 更新 Reference Resolver 配置

```python
plugin.update_reference_resolver_config(
    just_now_turns=3,      # "刚刚" 召回 3 轮
    recently_turns=30,     # "最近" 召回 30 轮
    last_topic_turns=20,   # "那件事" 召回 20 轮
)
# 委托给: planner.update_reference_resolver_config(...)
```

### 11.3 更新 Memory Trigger 配置

```python
plugin.update_memory_trigger_config(
    enabled=True,
    custom_patterns=[{"pattern": r"我的名字是", "type": "identity"}],
)
# 委托给: planner.update_memory_trigger_config(...)
```

### 11.4 更新 Full Attention 配置 (⚠️ 已弃用)

```python
plugin.update_full_attention_config(
    preference_alpha=0.3,
    history_alpha=0.5,
    max_total_kv_tokens=800,
)
# 委托给: executor.full_attention_injector.update_config(...)
```

### 11.5 获取组件配置

```python
configs = plugin.get_component_configs()
# {
#     "memory_trigger": {...},
#     "reference_resolver": {...},
#     "injection_strategy": "stable",
#     "full_attention": {...},  # 仅当 FA 注入器存在时
# }
```

---

## 12. 错误处理与降级

```
异常处理策略 (四级降级):
  │
  ├─ Level 0: Recall v4 多信号召回失败
  │   └─ Planner 回退到 flat_history_fallback
  │       └─ 使用原有 stable 历史格式化
  │
  ├─ Level 1: Full Attention 注入失败
  │   └─ _execute_stable_fallback()
  │       ├─ 使用 plan.history_suffix 重构 stable 输入
  │       └─ 调用 _execute_stable() (偏好 K/V + 历史 Suffix)
  │
  ├─ Level 2: Executor 执行异常
  │   └─ _execute_fallback()
  │       └─ model.generate(plan.original_query) (仅原始查询, 无注入)
  │
  └─ Level 3: chat() 主流程异常
      └─ model.generate(query) (仅原始查询, 无注入)
      
  缓存相关:
  ├─ 偏好 K/V 计算失败
  │   └─ 返回 (None, False, "error"), 跳过偏好注入
  │
  └─ Redis 连接失败
      └─ 降级为 L1 only (内存缓存)
```

---

## 13. 使用示例

### 13.1 从配置文件创建 (推荐)

```python
# 上层应用只需提供配置文件
dki = await DKIPlugin.from_config(
    model_adapter=vllm_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 调用时只需传递 user_id 和原始输入
response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_123",
    session_id="session_456",
)

print(response.text)
print(response.metadata.alpha)
print(response.metadata.alpha_profile)
```

### 13.2 启用 Redis 分布式缓存

```python
dki = await DKIPlugin.from_config(
    model_adapter=vllm_adapter,
    adapter_config_path="config/adapter_config.yaml",
    enable_redis=True,
    redis_config={"host": "redis.example.com", "port": 6379},
)
```

### 13.3 高级用法: 直接访问 Planner / Executor

```python
# 生成计划但不执行 (用于测试/调试)
context = dki.planner.analyze_query("推荐餐厅")
plan = dki.planner.build_plan(
    query="推荐餐厅",
    user_id="user_123",
    preferences=prefs,
    relevant_history=history,
    context=context,
    session_id="session_456",       # v3.1: Recall v4 需要
    context_window=8192,            # v3.1: Recall v4 需要
)
print(plan.to_dict())  # 检查决策
print(plan.trace_ids)  # v3.1: 查看 trace_ids

# 手动执行
result = await dki.executor.execute(plan)
print(result.text)
print(result.fact_rounds_used)  # v3.1: 查看 fact call 轮次
```

### 13.4 强制 Alpha (实验用)

```python
# 强制 alpha=0.8 (跳过门控)
response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_123",
    session_id="session_456",
    force_alpha=0.8,
)
```

### 13.5 关闭插件

```python
await dki.close()  # 关闭 Redis + 数据库连接
```

---

## 14. v2.x → v3.0 → v3.1 迁移指南

### 14.1 对外接口 (无变化)

| 方法 | v2.x | v3.0 | v3.1 | 变化 |
|------|------|------|------|------|
| `chat()` | ✅ | ✅ | ✅ | 签名不变, 返回类型不变 |
| `from_config()` | ✅ | ✅ | ✅ | 签名不变 |
| `get_stats()` | ✅ | ✅ | ✅ | v3.1 新增 recall_v4/fact_call 统计 |
| `get_injection_logs()` | ✅ | ✅ | ✅ | 不变 |
| `switch_injection_strategy()` | ✅ | ✅ | ✅ | v3.1 支持 "recall_v4" |
| `update_reference_resolver_config()` | ✅ | ✅ | ✅ | 不变 |
| `update_memory_trigger_config()` | ✅ | ✅ | ✅ | 不变 |
| `update_full_attention_config()` | ✅ | ✅ | ✅ | 不变 (⚠️ FA 已弃用) |
| `close()` | ✅ | ✅ | ✅ | 不变 |

### 14.2 InjectionPlan (v3.1 新增字段)

| 字段 | v3.0 | v3.1 | 说明 |
|------|------|------|------|
| `assembled_suffix` | ❌ | ✅ | Recall v4 组装后缀 |
| `recall_strategy` | ❌ | ✅ | 召回策略标识 |
| `summary_count` | ❌ | ✅ | Summary 条目数 |
| `message_count` | ❌ | ✅ | 原文消息数 |
| `trace_ids` | ❌ | ✅ | trace_id 列表 |
| `has_fact_call_instruction` | ❌ | ✅ | 是否含 fact call 指导 |
| `fact_rounds_used` | ❌ | ✅ | fact call 实际轮次 |
| `session_id` | ❌ | ✅ | 会话 ID |

### 14.3 ExecutionResult (v3.1 新增字段)

| 字段 | v3.0 | v3.1 | 说明 |
|------|------|------|------|
| `fact_rounds_used` | ❌ | ✅ | fact call 实际轮次 |
| `fact_tokens_total` | ❌ | ✅ | fact call 总 token 数 |

### 14.4 内部架构变化

| 组件 | v2.x | v3.0 | v3.1 |
|------|------|------|------|
| 决策逻辑 | `DKIPlugin.chat()` | `InjectionPlanner` | `InjectionPlanner` + Recall v4 |
| 执行逻辑 | `DKIPlugin._inject_*()` | `InjectionExecutor` | `InjectionExecutor` + Fact Call |
| 记忆召回 | 无 | 无 | `MultiSignalRecall` (关键词 + 向量 + 指代) |
| 后缀组装 | 无 | 无 | `SuffixBuilder` (逐消息 Summary) |
| 事实补充 | 无 | 无 | `FactRetriever` (trace_id → 原文) |
| 提示格式化 | 无 | 无 | `PromptFormatter` (Generic/DeepSeek/GLM) |

---

## 15. 依赖关系图

```
dki.core.dki_plugin (Facade)
  ├─ dki.core.plugin.injection_planner (决策)
  │   ├─ dki.core.plugin.injection_plan (数据结构)
  │   ├─ dki.core.components.memory_trigger
  │   ├─ dki.core.components.reference_resolver
  │   └─ dki.core.recall (v3.1, 可选)
  │       ├─ MultiSignalRecall
  │       ├─ SuffixBuilder
  │       └─ PromptFormatter
  │
  ├─ dki.core.plugin.injection_executor (执行)
  │   ├─ dki.core.plugin.injection_plan (数据结构)
  │   ├─ dki.models.base (BaseModelAdapter)
  │   ├─ dki.core.injection.full_attention_injector (⚠️ 已弃用)
  │   └─ dki.core.recall (v3.1, 可选)
  │       ├─ FactRetriever
  │       └─ PromptFormatter
  │
  ├─ dki.adapters.base (IUserDataAdapter)
  ├─ dki.cache (PreferenceCacheManager, DKIRedisClient)
  ├─ dki.config.config_loader (ConfigLoader)
  ├─ dki.core.components.memory_influence_scaling (延迟初始化)
  ├─ dki.core.components.dual_factor_gating (延迟初始化)
  └─ dki.api.visualization_routes (record_visualization)
```
