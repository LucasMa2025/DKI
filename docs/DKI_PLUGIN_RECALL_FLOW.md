# DKI Plugin 记忆召回流程说明

## 概述

DKI Plugin 的记忆召回系统是一个多信号融合的跨会话记忆检索框架。从上层 `chat()` 调用开始，经过 **查询分析 → 数据加载 → 计划构建 → 执行注入 → 后处理** 五个阶段，最终将用户偏好和历史记忆注入到 LLM 推理过程中。

核心设计原则：**决策与执行分离**

-   `InjectionPlanner` 只做决策，不碰模型
-   `InjectionExecutor` 只做执行，不做决策
-   `InjectionPlan` 是两者之间的纯数据中间产物 (IR)，可序列化、可缓存、可重放

---

## 1. 召回方式配置

### 1.1 配置文件位置

```yaml
# config/config.yaml → dki 段
dki:
    injection_strategy: "recall_v4" # recall_v4 | stable
    recall:
        enabled: true
        strategy: "summary_with_fact_call"
```

### 1.2 核心配置项

| 配置项         | 路径                                   | 默认值                   | 说明                          |
| -------------- | -------------------------------------- | ------------------------ | ----------------------------- |
| **策略**       | `dki.injection_strategy`               | `recall_v4`              | 注入策略 (recall_v4 / stable) |
| **召回策略**   | `dki.recall.strategy`                  | `summary_with_fact_call` | 召回内部策略                  |
| **关键词检索** | `dki.recall.signals.keyword_enabled`   | `true`                   | jieba TF-IDF 关键词检索       |
| **BM25 检索**  | `dki.recall.signals.bm25_enabled`      | `true`                   | BM25 全文检索                 |
| **向量检索**   | `dki.recall.signals.vector_enabled`    | `true`                   | FAISS 向量语义检索            |
| **指代解析**   | `dki.recall.signals.reference_enabled` | `true`                   | 时间/指代表达解析             |
| **BM25 Top-K** | `dki.recall.signals.bm25_top_k`        | `15`                     | BM25 返回候选数               |
| **向量 Top-K** | `dki.recall.signals.vector_top_k`      | `10`                     | 向量检索返回数                |
| **向量阈值**   | `dki.recall.signals.vector_threshold`  | `0.5`                    | 向量相似度最低阈值            |

### 1.3 分数融合权重

```yaml
dki:
    recall:
        score_weights:
            keyword_weight: 0.3 # jieba TF-IDF 关键词权重
            bm25_weight: 0.2 # BM25 全文检索权重
            vector_weight: 0.3 # 向量相似度权重
            recency_weight: 0.2 # 时间近度权重
```

### 1.4 Token 预算配置

```yaml
dki:
    recall:
        budget:
            generation_ratio: 0.30 # 生成预留 = 30% 上下文窗口
            instruction_reserve: 120 # chat template 标记开销
            preference_max_tokens: 200 # 偏好最大 token 数
            min_recent_turns: 2 # 至少保留的近期完整轮次
            max_recent_turns: 5 # 最多保留的近期完整轮次
            cross_session_limit: 10 # 跨会话最大召回消息数
```

### 1.5 认知态模式 (Epistemic Modes)

```yaml
dki:
    recall:
        epistemic_modes:
            enabled: false
            fallback_mode: "direct_lookup"
            profiles:
                clarification: # 用户回忆/澄清 → 近期权重高
                    keyword_weight: 0.35
                    bm25_weight: 0.20
                    vector_weight: 0.10
                    recency_weight: 0.35
                semantic_search: # 语义检索 → 向量权重高
                    keyword_weight: 0.10
                    bm25_weight: 0.15
                    vector_weight: 0.50
                    recency_weight: 0.25
                correction: # 修正信息 → 近期权重最高
                    keyword_weight: 0.10
                    bm25_weight: 0.10
                    vector_weight: 0.10
                    recency_weight: 0.70
                direct_lookup: # 默认 → 关键词权重高
                    keyword_weight: 0.40
                    bm25_weight: 0.25
                    vector_weight: 0.20
                    recency_weight: 0.15
```

### 1.6 信号门控 (Signal Gating)

```yaml
dki:
    recall:
        signal_gating:
            enabled: true
            confidence_threshold: 0.15 # 低于此值的信号不参与融合
```

---

## 2. 完整召回流程 (从 `chat()` 开始)

### 2.1 流程总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DKIPlugin.chat()                                │
│                                                                     │
│  Step 1: 查询分析 (Planner.analyze_query)                          │
│    ├── MemoryTrigger: 检测记忆触发信号                               │
│    └── ReferenceResolver: 解析时间/指代表达 → 确定 recall_limit      │
│                                                                     │
│  Step 2: 数据加载 (Plugin 通过 DataAdapter)                         │
│    ├── 加载用户偏好 (带 TTL 缓存)                                    │
│    ├── 跨会话历史检索 (adapter.search_relevant_history)              │
│    ├── 获取近轮对话 (adapter.get_recent_messages)                    │
│    └── 合并: 近轮优先 + 召回去重                                     │
│                                                                     │
│  Step 3: 构建注入计划 (Planner.build_plan)                          │
│    ├── 偏好文本格式化 (Block 结构)                                   │
│    ├── 历史召回 (两条路径, 见下文)                                   │
│    ├── Alpha 计算 (分层: preference / history)                      │
│    ├── 构造 final_input (assembled_suffix / history_suffix + query) │
│    ├── 记忆元数据提示 (Memory Metadata Block)                       │
│    └── SafetyEnvelope 验证                                          │
│                                                                     │
│  Step 3.5: 模糊指代澄清 (v6.5)                                     │
│    └── 历史不足时注入澄清指令                                       │
│                                                                     │
│  Step 4: 执行注入 (Executor.execute)                                │
│    ├── 偏好 K/V 计算 (BoundedUserKVCache, 按 user_id 分区)         │
│    ├── Prompt 组装 (chat template / prompt_prefix 模式)             │
│    └── 模型推理 (vLLM/SGLang/LLaMA)                                │
│                                                                     │
│  Step 5: 后处理                                                     │
│    ├── 移除 <think> 推理内容                                        │
│    └── 返回 DKIPluginResponse                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Step 1: 查询分析 — `Planner.analyze_query(query)`

**入口**: `DKIPlugin.chat()` → `self._planner.analyze_query(query)`

**职责**: 在数据加载之前，分析用户查询的意图，确定召回范围。

```python
# dki/core/plugin/injection_planner.py
context = self._planner.analyze_query(query)
# 输出: QueryContext
#   - recall_limit: 召回消息数上限 (由指代解析决定, 默认 10)
#   - recall_token_budget: token 软预算 (默认 2048, 可动态调整)
#   - memory_triggered: 是否触发记忆信号
#   - trigger_type: 触发类型 (meta_cognitive / correction / ...)
#   - trigger_confidence: 触发置信度
#   - reference_resolved: 是否解析到指代
#   - reference_type: 指代类型
#   - reference_scope: 指代范围 (last_1_3_turns / recent / session)
```

**子组件**:

1. **MemoryTrigger** (`dki/core/components/memory_trigger.py`)

    - 检测用户输入中的记忆触发信号
    - 模式: "你还记得吗"、"之前说过"、"上次聊的" 等
    - 输出: `TriggerResult(triggered, trigger_type, confidence)`

2. **ReferenceResolver** (`dki/core/components/reference_resolver.py`)
    - 解析时间/指代表达，确定 `recall_limit`
    - "刚刚" → `last_1_3_turns` (recall_limit=5)
    - "最近" → `recent` (recall_limit=10)
    - "之前" → `session` (recall_limit=50)

**Token Budget 动态调整**:

-   明确引用时: `base_budget × 1.5`
-   高置信度 trigger (>0.8): `base_budget × 1.2`

### 2.3 Step 2: 数据加载 — Plugin 通过 DataAdapter

**入口**: `DKIPlugin.chat()` 中直接调用 `self.data_adapter`

这一步由 **Plugin (Facade)** 负责，不在 Planner 内部。

```python
# dki/core/dki_plugin.py — chat() 方法

# 1. 加载用户偏好 (带 TTL 缓存)
preferences = await self._get_cached_preferences(user_id)

# 2. 检测适配器的向量检索能力
retrieval_mode = self._detect_retrieval_mode()
# → "bm25_embedding" | "bm25_only" | "keyword" | "unknown"

# 3. 跨会话历史检索 (session_id=None → 搜索所有会话)
relevant_history = await self.data_adapter.search_relevant_history(
    user_id=user_id,
    query=query,
    session_id=None,  # 跨会话检索
    limit=context.recall_limit,
)

# 4. 获取近轮对话 (按时间近度, 确保多轮连贯性)
recent_messages = await self.data_adapter.get_recent_messages(
    user_id=user_id,
    limit=max_recent * 2,  # 每轮 user+assistant
)

# 5. 合并: 近轮优先 + 召回去重
relevant_history = self._merge_recent_and_recalled(
    recent_messages, relevant_history
)
```

**为什么需要近轮对话?**
BM25 只做语义相关性召回，可能遗漏最近的对话内容。近轮对话确保模型能看到最近的上下文，维持多轮连贯性。

### 2.4 Step 3: 构建注入计划 — `Planner.build_plan()`

**入口**: `DKIPlugin.chat()` → `self._planner.build_plan(...)`

```python
plan = self._planner.build_plan(
    query=query,
    user_id=user_id,
    preferences=preferences,
    relevant_history=relevant_history,
    context=context,
    force_alpha=force_alpha,
    session_id=session_id,
    context_window=self._context_window,
)
# 输出: InjectionPlan (纯数据, 可序列化)
```

**Planner 内部完整流程**:

```
┌──────────────────────────────────────────────────────────────┐
│              InjectionPlanner.build_plan()                    │
│                                                              │
│  Step 1: 格式化偏好                                          │
│    preferences → 按 type 分组 → Block 结构化文本              │
│    <preference:dietary>                                      │
│    - 素食主义者                                               │
│    </preference:dietary>                                     │
│                                                              │
│  Step 2: 历史召回 (三条路径, 按条件选择)                      │
│    ├─ Path A: recall_v4 完整路径                             │
│    │   (有 MultiSignalRecall + SuffixBuilder)                │
│    │   → _build_recall_v4_plan()                             │
│    │                                                         │
│    ├─ Path B: suffix-only 路径 (v6.2)                        │
│    │   (有 SuffixBuilder, 无 MultiSignalRecall)              │
│    │   → _build_suffix_only_plan()                           │
│    │                                                         │
│    └─ Path C: stable 策略 / 无 SuffixBuilder                 │
│        → 平铺历史后缀 (无压缩)                               │
│                                                              │
│  Step 3: Alpha 计算 (分层)                                   │
│    ├─ preference_alpha: 从配置读取, 默认 0.4                  │
│    │   + P1-2: MemoryTrigger confidence 增强 (最多 +30%)     │
│    └─ history_alpha: 对数衰减 (512 token 以下不衰减)          │
│                                                              │
│  Step 4: 注入决策                                            │
│    injection_enabled = 有偏好 OR 有历史                       │
│                                                              │
│  Step 5: 构造 final_input                                    │
│    ├─ 有 assembled_suffix → 直接使用                          │
│    ├─ 有 history_suffix → history_suffix + query             │
│    └─ 都没有 → 仅 query                                     │
│                                                              │
│  Step 5.5: 记忆元数据提示 (P0-1)                             │
│    [DMI 记忆状态]                                             │
│    偏好: 3 条活跃 (食物, 沟通风格)                            │
│    召回: 5 条相关历史 (含 2 条摘要)                           │
│    [DMI 记忆状态结束]                                         │
│    → 插入到 final_input 前面                                  │
│                                                              │
│  Step 6: 门控决策记录                                        │
│                                                              │
│  Step 7: SafetyEnvelope 验证                                 │
│    - preference_alpha ≤ 0.7 (recall_v4) / 0.5 (stable)      │
│                                                              │
│  输出: InjectionPlan                                         │
└──────────────────────────────────────────────────────────────┘
```

#### Path A: recall_v4 完整路径 — `_build_recall_v4_plan()`

当 Planner 同时持有 `MultiSignalRecall` 和 `SuffixBuilder` 时走此路径。

```
┌──────────────────────────────────────────────────────────────┐
│           _build_recall_v4_plan()                             │
│                                                              │
│  Phase 1: MultiSignalRecall.recall()                         │
│    → 四路信号融合 (详见 §3)                                   │
│    → 输出: RecallResult (messages + scores)                  │
│                                                              │
│  Phase 2: SuffixBuilder.build()                              │
│    → 输入: query + recall_result.messages + context_window   │
│    → 两阶段全局预算分配 (详见 §4)                             │
│    → 输出: AssembledSuffix                                   │
│                                                              │
│  Phase 3: Planner 填充 InjectionPlan                         │
│    plan.assembled_suffix = assembled.text                    │
│    plan.history_suffix = assembled.text  (兼容 fallback)     │
│    plan.history_tokens = assembled.total_tokens              │
│    plan.summary_count = assembled.summary_count              │
│    plan.message_count = assembled.message_count              │
│    plan.trace_ids = assembled.trace_ids                      │
│    plan.history_items = assembled.items  (HistoryItem 列表)  │
│                                                              │
│  异常处理:                                                   │
│    任何异常 → recall_strategy = "flat_history_fallback"       │
│    → 回退到 stable 策略                                      │
└──────────────────────────────────────────────────────────────┘
```

#### Path B: suffix-only 路径 — `_build_suffix_only_plan()` (v6.2)

当 Plugin 的 adapter 已完成召回 (有 `relevant_history`)，但 Planner 没有 `MultiSignalRecall` 时走此路径。

```
┌──────────────────────────────────────────────────────────────┐
│           _build_suffix_only_plan()                           │
│                                                              │
│  跳过 MultiSignalRecall (adapter 已完成召回)                  │
│                                                              │
│  直接调用 SuffixBuilder.build()                              │
│    → 输入: query + relevant_history + context_window         │
│    → 两阶段全局预算分配                                       │
│    → 输出: AssembledSuffix                                   │
│                                                              │
│  Planner 填充 InjectionPlan (同 Path A)                      │
│                                                              │
│  异常处理:                                                   │
│    → 回退到 stable 策略 (平铺历史后缀)                        │
└──────────────────────────────────────────────────────────────┘
```

**关键区别**: Path A 由 `MultiSignalRecall` 在 Planner 内部完成召回；Path B 由 Plugin 的 `DataAdapter` 在外部完成召回，Planner 只负责 token 预算分配和格式化。

---

## 3. MultiSignalRecall: 四路信号融合

```
┌──────────────────────────────────────────────────────────────┐
│              MultiSignalRecall.recall()                       │
│                                                              │
│  1. 指代解析 → 确定回溯范围 (recall_turns, 默认 10)           │
│                                                              │
│  2. 关键词+权重检索 (jieba TF-IDF)                           │
│     query → jieba.analyse.extract_tags(topK=5)               │
│     → 遍历会话历史 (限 recall_turns×2 条)                     │
│     → 关键词命中计分 (score += weight)                        │
│     → 同时追踪命中关键词数 (用于置信度计算)                    │
│                                                              │
│  3. BM25 全文检索 (rank_bm25)                                │
│     会话历史 → jieba 精确模式分词 → 过滤单字符                 │
│     → BM25Okapi 构建请求级临时索引                            │
│     query → jieba 分词 → bm25.get_scores()                   │
│     → top_k=15 候选                                          │
│                                                              │
│  4. 向量检索 (FAISS)                                         │
│     query → MemoryRouter.search(top_k=10, threshold=0.5)     │
│     → cosine similarity ≥ 0.5 的结果                         │
│                                                              │
│  5. 认知态模式选择 (F1-1)                                     │
│     QueryContext → 规则匹配 → 选择权重预设                     │
│     ├─ reference_resolved → clarification                    │
│     ├─ memory_triggered + high conf → semantic_search        │
│     ├─ trigger_type=correction → correction                  │
│     └─ fallback → direct_lookup                              │
│                                                              │
│  6. 统一 min-max 归一化 (F1-3)                                │
│     各路原始分数 → [0, 1] 归一化                              │
│     单元素 → 1.0; 多元素 → (x-min)/(max-min)                │
│                                                              │
│  7. 信号置信度门控 (F1-2)                                     │
│     ├─ keyword: coverage × density                           │
│     │   coverage = hit_terms / query_terms                   │
│     │   density = min(1.0, len(scored) / 3)                  │
│     ├─ BM25: 2×(sigmoid(mean_top3/5) - 0.5)                 │
│     ├─ vector: mean(top-3 scores)                            │
│     └─ recency: 始终 1.0                                     │
│     低置信度 (< 0.15) → 剔除, 不参与融合                      │
│     剩余信号权重 → 动态归一化到 1.0                            │
│                                                              │
│  8. 加权融合排序                                              │
│     final_score = Σ(norm_weight_i × norm_score_i)            │
│     → 按 final_score 降序 → top max_results                  │
│                                                              │
│  9. 获取完整消息对象                                          │
│     sorted_ids → fetch_messages_by_ids()                     │
│                                                              │
│  10. 补充近期轮次 (min_recent_turns=2)                        │
│      + 时间近度 bonus (recency 权重)                          │
│                                                              │
│  11. 跨会话历史召回 (cross_session_limit=10)                  │
│                                                              │
│  12. 合并去重 (近期优先)                                      │
│                                                              │
│  输出: RecallResult                                           │
│    - messages: 排序后的消息列表                                │
│    - scores: {msg_id → final_score}                          │
│    - keyword_hits / bm25_hits / vector_hits                  │
│    - recent_turns_added / reference_scope                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. SuffixBuilder: 两阶段全局预算分配

**职责**: 接收召回的消息列表，进行 token 预算分配和格式化，返回 `AssembledSuffix`。

**注意**: SuffixBuilder **只返回完整的 AssembledSuffix 数据**，最终的 `InjectionPlan` 填充由 **Planner** 完成。

```
┌──────────────────────────────────────────────────────────────┐
│              SuffixBuilder.build()                            │
│                                                              │
│  输入: query, recalled_messages, context_window,             │
│        preference_tokens                                     │
│                                                              │
│  ============ 计算可用预算 ============                       │
│  context_budget = context_window                             │
│    - int(context_window × 0.30)    # 生成预留 30%            │
│    - instruction_reserve (120)     # chat template 标记      │
│    - preference_tokens             # 偏好占用                │
│    - query_tokens                  # 当前输入                │
│                                                              │
│  ============ Phase 1: 完整收集 ============                  │
│  遍历所有消息:                                                │
│    - 提取 content, role, msg_id                              │
│    - 移除 assistant 消息中的 <think> 推理内容                 │
│    - 计算每条消息的 token 数                                  │
│    → collected: [{msg_id, content, role, tokens}, ...]       │
│                                                              │
│  ============ Phase 2: 全局预算分配 ============              │
│  计算 total_tokens = Σ(msg.tokens)                           │
│                                                              │
│  Case A: total_tokens ≤ budget                               │
│    → 全部保留原文 (最优, 零信息损失)                          │
│                                                              │
│  Case B: total_tokens > budget                               │
│    1. 分类: 短消息 (≤ 300 tokens) vs 长消息 (> 300 tokens)   │
│    2. 短消息全部保留原文 (优先级最高)                          │
│    3. 剩余预算分配给长消息:                                   │
│       ├─ 预算够 → 保留原文                                   │
│       ├─ 预算不够原文但够 summary → 压缩                      │
│       │   summary 策略: extractive (jieba TextRank)           │
│       │                 或 llm (调用系统 LLM)                 │
│       │   + 提取认知标记 (facts_covered / facts_missing)      │
│       └─ 连 summary 都放不下 → 跳过                          │
│    4. 按原始顺序合并 (短消息 + 长消息)                        │
│                                                              │
│  ============ Phase 3: 格式化 ============                    │
│  调用 PromptFormatter.format_full_suffix():                  │
│    [会话历史参考]                                             │
│    + format_history_list(items)                              │
│      ├─ message → format_message_item()                      │
│      └─ summary → format_summary_item() (含 trace_id)       │
│    + format_constraint_instruction() (仅当有 summary 时)     │
│    + "用户当前问题: {query}"                                  │
│                                                              │
│  输出: AssembledSuffix                                       │
│    - text: 最终格式化文本                                    │
│    - items: List[HistoryItem] (结构化条目, 含角色信息)       │
│    - total_tokens / message_count / summary_count            │
│    - trace_ids: summary 条目的溯源 ID                        │
│    - has_fact_call_instruction: 是否包含 fact call 指导      │
└──────────────────────────────────────────────────────────────┘
```

**HistoryItem 数据结构**:

```python
@dataclass
class HistoryItem:
    type: str              # "summary" | "message"
    content: str           # 文本内容
    trace_id: str          # 溯源 ID (message_id)
    role: Optional[str]    # "user" | "assistant"
    token_count: int       # token 数
    confidence: str        # "high" (原文) | "medium" (summary)
    facts_covered: List[str]   # 已覆盖的事实类别
    facts_missing: List[str]   # 可能遗漏的事实类别
```

---

## 5. Executor: 执行注入

**入口**: `DKIPlugin.chat()` → `self._executor.execute(plan, ...)`

```
┌──────────────────────────────────────────────────────────────┐
│              InjectionExecutor.execute()                      │
│                                                              │
│  判断: has_injectable_content?                                │
│    = plan.injection_enabled                                  │
│      AND (plan.preference_text OR plan.history_items)        │
│                                                              │
│  ┌─ YES → _execute_with_kv_injection()                      │
│  │                                                           │
│  │   1. 偏好 K/V 计算                                       │
│  │      BoundedUserKVCache (按 user_id 物理分区)             │
│  │      ├─ 缓存命中: 直接复用                                │
│  │      └─ 缓存未命中: model.compute_kv(preference_text)     │
│  │                                                           │
│  │   2. 判断注入模式                                         │
│  │      ├─ prompt_prefix 模式 (vLLM/SGLang)                  │
│  │      │   偏好 → 系统指令前缀 (alpha 映射为自然语言强度)    │
│  │      │   + chat template 构造 prompt                      │
│  │      │   → model.generate(prompt)                         │
│  │      │   利用 vLLM prefix_caching / SGLang RadixAttention │
│  │      │                                                    │
│  │      └─ HF KV 注入模式 (LLaMA)                           │
│  │          → model.forward_with_kv_injection(               │
│  │              prompt, kv_entries, alpha)                    │
│  │                                                           │
│  │   3. 过滤 retrieve_fact() 残留调用 (F1-4)                │
│  │      支持 Generic / DeepSeek / GLM 三种格式               │
│  │                                                           │
│  └─ NO → _execute_plain()                                   │
│      直接调用 model.generate(query)                          │
│                                                              │
│  异常处理 (两级降级):                                        │
│    Level 1: recall_v4 失败 → stable 策略                     │
│    Level 2: stable 失败 → 无注入推理                         │
│                                                              │
│  安全不变量:                                                 │
│    - Key tensor 永远不被 alpha 缩放                          │
│    - 偏好 K/V 缓存按 user_id 物理分区                       │
│    - 推理上下文隔离 (InferenceContextGuard)                  │
│                                                              │
│  输出: ExecutionResult                                       │
│    - text: 生成文本                                          │
│    - input_tokens / output_tokens                            │
│    - inference_latency_ms                                    │
│    - preference_cache_hit / preference_cache_tier            │
│    - fallback_used / error_message                           │
└──────────────────────────────────────────────────────────────┘
```

### 2.6 Step 5: 后处理与降级

```python
# dki/core/dki_plugin.py — chat() 方法

# 移除 <think> 推理内容 (DeepSeek-R1 等 thinking 模型)
clean_text, _ = strip_think_content(result.text)

# 返回结构化响应
return DKIPluginResponse(
    text=clean_text,
    input_tokens=result.input_tokens,
    output_tokens=result.output_tokens,
    metadata=metadata,  # 包含完整的注入元数据
)
```

**三级降级策略** (在 `DKIPlugin.chat()` 的 except 块中):

```
Level 1: recall_v4 失败
  → 构建 stable plan (偏好 K/V + 原始查询)
  → Executor.execute(stable_plan)

Level 2: stable 失败
  → 直接调用 model.generate(query) (无注入)

Level 3: LLM 推理失败
  → raise 异常
```

---

## 6. 架构关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    DKIPlugin (瘦 Facade)                         │
│                     chat() 入口                                  │
│                                                                  │
│  职责: 数据加载 + 编排 Planner/Executor + 降级 + 日志            │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│  ┌───────────────────┐   │   ┌─────────────────────────────┐   │
│  │ InjectionPlanner  │   │   │   InjectionExecutor         │   │
│  │ (纯决策层)        │   │   │   (纯执行层)                │   │
│  │                   │   │   │                             │   │
│  │ analyze_query()   │   │   │ ├─ BoundedUserKVCache      │   │
│  │   ├─ MemoryTrigger│   │   │ │  (用户级 LRU, 物理分区)  │   │
│  │   └─ RefResolver  │   │   │ │                           │   │
│  │                   │   │   │ ├─ prompt_prefix 模式       │   │
│  │ build_plan()      │   │   │ │  (vLLM/SGLang)           │   │
│  │   ├─ 偏好格式化   │   │   │ │                           │   │
│  │   ├─ 历史召回     │──→│──→│ ├─ HF KV 注入模式          │   │
│  │   │  (3条路径)    │   │   │ │  (LLaMA)                 │   │
│  │   ├─ Alpha 计算   │   │   │ │                           │   │
│  │   ├─ final_input  │   │   │ └─ InferenceContextGuard   │   │
│  │   ├─ 记忆元数据   │   │   │                             │   │
│  │   └─ Safety 验证  │   │   └─────────────────────────────┘   │
│  └───────────────────┘   │                                      │
│                          │                                      │
│  InjectionPlan ──────────┘                                      │
│  (纯数据 IR, 可序列化)                                          │
├─────────────────────────────────────────────────────────────────┤
│                  Recall v4 组件 (Planner 内部)                   │
│  ├─ MultiSignalRecall (四路信号融合)                             │
│  │   ├─ jieba TF-IDF 关键词检索                                 │
│  │   ├─ BM25Okapi 全文检索 (rank_bm25)                          │
│  │   ├─ FAISS 向量检索 (MemoryRouter)                           │
│  │   └─ 认知态模式 + 信号门控 + 加权融合                        │
│  ├─ SuffixBuilder (两阶段全局预算分配)                           │
│  │   ├─ Phase 1: 完整收集 (不压缩)                              │
│  │   ├─ Phase 2: 全局预算分配 (短消息优先, 长消息智能压缩)      │
│  │   └─ Phase 3: PromptFormatter 格式化                         │
│  └─ FactRetriever (fact_call 回调, 由 Executor 按需触发)        │
├─────────────────────────────────────────────────────────────────┤
│                  DataAdapter (外部数据, Plugin 调用)              │
│  ├─ get_user_preferences()                                      │
│  ├─ search_relevant_history()  (BM25 + Vector / BM25-only)      │
│  └─ get_recent_messages()                                       │
├─────────────────────────────────────────────────────────────────┤
│                  ModelAdapter (推理引擎)                          │
│  ├─ VLLMAdapter   (vLLM + prefix_caching, prompt_prefix 模式)   │
│  ├─ SGLangAdapter (SGLang + RadixAttention, prompt_prefix 模式) │
│  └─ LlamaAdapter  (HuggingFace Transformers, HF KV 注入模式)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键数据结构

### QueryContext (查询分析结果)

```python
@dataclass
class QueryContext:
    recall_limit: int = 10           # 召回消息数上限
    recall_token_budget: int = 2048  # token 软预算 (P1-1)
    memory_triggered: bool = False   # 是否触发记忆信号
    trigger_type: Optional[str]      # 触发类型
    trigger_confidence: float = 0.0  # 触发置信度
    reference_resolved: bool = False # 是否解析到指代
    reference_type: Optional[str]    # 指代类型
    reference_scope: Optional[str]   # 指代范围
```

### InjectionPlan (注入计划 — 核心 IR)

```python
@dataclass
class InjectionPlan:
    # 策略
    strategy: str = "recall_v4"       # recall_v4 | stable | none

    # 偏好数据 (K/V 注入)
    preference_text: str = ""
    preferences_count: int = 0
    preference_tokens: int = 0

    # 历史数据
    history_suffix: str = ""          # 格式化后的 suffix prompt
    assembled_suffix: str = ""        # recall_v4 组装的后缀 (替代 history_suffix)
    history_tokens: int = 0
    relevant_history_count: int = 0
    history_items: List[HistoryItem]  # 结构化历史条目 (含角色信息)

    # 查询
    original_query: str = ""
    final_input: str = ""             # 最终发给模型的输入
    user_id: str = ""

    # Alpha 控制 (分层)
    alpha_profile: AlphaProfile       # preference_alpha + history_alpha

    # 注入决策
    injection_enabled: bool = False

    # Recall v4 特有
    recall_strategy: str = ""         # summary_with_fact_call | suffix_only | flat_history
    summary_count: int = 0
    message_count: int = 0
    trace_ids: List[str]              # summary 的溯源 ID
    has_fact_call_instruction: bool = False

    # 记忆元数据
    memory_metadata: str = ""

    # 安全
    safety_violations: List[str]
```

### AlphaProfile (分层 Alpha 控制)

```python
@dataclass
class AlphaProfile:
    preference_alpha: float = 0.4    # 偏好注入强度 (K/V)
    history_alpha: float = 1.0       # 历史注入强度
    override_cap: float = 0.7        # 偏好 alpha 安全上限

    @property
    def effective_preference_alpha(self) -> float:
        """确保偏好 alpha 不超过安全上限"""
        return min(self.preference_alpha, self.override_cap)
```

### RecallResult (多信号召回结果)

```python
@dataclass
class RecallResult:
    messages: List[Any]              # 召回的消息列表 (已排序)
    keyword_hits: int = 0
    bm25_hits: int = 0
    vector_hits: int = 0
    reference_scope: Optional[str]
    recent_turns_added: int = 0
    scores: Dict[str, float]         # msg_id → final_score
```

### AssembledSuffix (SuffixBuilder 输出)

```python
@dataclass
class AssembledSuffix:
    text: str = ""                   # 最终格式化文本
    items: List[HistoryItem]         # 结构化条目列表
    total_tokens: int = 0
    message_count: int = 0           # 原文消息数量
    summary_count: int = 0           # summary 数量
    has_fact_call_instruction: bool   # 是否包含 fact call 指导
    trace_ids: List[str]             # summary 的溯源 ID
```

---

## 8. 配置示例

### 8.1 标准配置 (推荐)

```yaml
dki:
    injection_strategy: "recall_v4"
    recall:
        enabled: true
        strategy: "summary_with_fact_call"
        signals:
            keyword_enabled: true
            bm25_enabled: true
            vector_enabled: true
            reference_enabled: true
        score_weights:
            keyword_weight: 0.3
            bm25_weight: 0.2
            vector_weight: 0.3
            recency_weight: 0.2
        signal_gating:
            enabled: true
            confidence_threshold: 0.15
```

### 8.2 仅 BM25 (无向量索引)

```yaml
dki:
    recall:
        signals:
            keyword_enabled: true
            bm25_enabled: true
            vector_enabled: false
        score_weights:
            keyword_weight: 0.4
            bm25_weight: 0.4
            vector_weight: 0.0
            recency_weight: 0.2
```

### 8.3 高精度语义检索

```yaml
dki:
    recall:
        signals:
            keyword_enabled: false
            bm25_enabled: true
            vector_enabled: true
            vector_top_k: 20
            vector_threshold: 0.3
        score_weights:
            keyword_weight: 0.0
            bm25_weight: 0.2
            vector_weight: 0.6
            recency_weight: 0.2
```

---

## 9. 分析说明

### 9.1 关于 chatgpt.md 中提出的三个改进建议

**建议 1: 去掉单字符过滤 → 改为 stopword filtering**

chatgpt.md 指出中文单字符 ("云"、"核"、"端") 可能是关键实体 token，硬删会损失信息。

**实际情况**: 代码中的单字符过滤是基于 jieba 分词结果的，jieba 精确模式会将 "云计算" 作为一个整词输出，不会拆成 "云" + "计算"。过滤的单字符主要是标点、助词 ("的"、"了"、"是") 等高频无语义字符。这与 chatgpt.md 假设的"硬删单字"不同，实际上更接近 stopword filtering。

不过，仍有改进空间：可以维护一个显式的中文停用词表，替代 `len(t.strip()) > 1` 的简单规则，避免误删有语义的单字词 (如专业术语中的单字)。

**建议 2: 替换 min-max → sigmoid scaling**

chatgpt.md 指出 min-max 在小样本中不稳定，建议使用 sigmoid scaling。

**实际情况**: 代码中已有两层保护机制来缓解 min-max 的不稳定性：

1. **信号置信度门控 (F1-2)**: 低置信度信号 (< 0.15) 直接剔除，不参与融合。这意味着当某路信号的 top-3 分数均值很低时，该信号不会"拖累"融合结果。
2. **认知态模式 (F1-1)**: 根据查询上下文动态调整各信号权重，减少对单一信号归一化质量的依赖。

但 chatgpt.md 的观察是正确的：当 BM25 原始分数分布很窄 (如 [0.9, 0.8, 0.7, 0.6]) 时，min-max 会人为拉大差距。对于 BM25 信号，代码中的 `_compute_bm25_confidence` 已经使用了 sigmoid 映射 (`sigmoid(mean_top3/5)`)，但这只用于置信度门控，归一化本身仍是 min-max。

**可考虑的改进**: 对 BM25 和 keyword 信号使用 sigmoid scaling 替代 min-max，向量信号保持 clip 到 [0,1] (因为 cosine similarity 本身就有固定范围)。

**建议 3: BM25 增加 recency boost**

chatgpt.md 建议 `score = bm25 × (1 + 0.1 × recency_rank)`。

**实际情况**: 代码中 recency 是作为独立信号参与融合的 (权重 0.2)，而不是作为 BM25 的 boost 因子。这种设计更灵活：

-   recency 权重可以通过认知态模式动态调整 (correction 模式下 recency=0.70)
-   recency 可以独立被门控 (虽然当前实现中 recency 置信度始终为 1.0)

两种方式各有优劣。独立信号更灵活、更可控；boost 因子更简单、计算更快。当前的独立信号设计在对话场景下已经足够。

### 9.2 SuffixBuilder 的职责边界

SuffixBuilder 是一个**纯数据处理组件**，职责是：

1. 收集消息 (Phase 1)
2. 全局预算分配 — 决定每条消息保留原文还是压缩 (Phase 2)
3. 格式化输出 (Phase 3)

它**不负责**：

-   填充 `InjectionPlan` 的字段 (由 Planner 完成)
-   构造 `final_input` (由 Planner 完成)
-   Alpha 计算 (由 Planner 完成)
-   安全验证 (由 Planner 完成)
-   记忆元数据提示 (由 Planner 完成)

这种设计保证了 SuffixBuilder 可以独立测试，不依赖 Planner 的其他逻辑。

### 9.3 两条召回路径的设计意图

-   **Path A (recall_v4 完整路径)**: Planner 内部持有 `MultiSignalRecall`，自己完成四路信号融合。适用于 DKI 独立部署场景。
-   **Path B (suffix-only 路径)**: Plugin 的 `DataAdapter` 已通过外部系统完成召回 (如 PostgreSQL BM25 + pgvector)，Planner 只需做 token 预算分配和格式化。适用于 DKI 作为插件集成到已有系统的场景。

两条路径最终都通过 SuffixBuilder 进行统一的预算分配和格式化，保证输出质量一致。
