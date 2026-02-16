# DKISystem 核心系统模块说明书 (v3.1 — Recall v4 集成版)

> 源文件: `DKI/dki/core/dki_system.py` (1440 行)  
> 辅助类: `_ConversationRepoWrapper` (同文件)  
> 记忆召回包: `DKI/dki/core/recall/` (6 文件, Recall v4)  
> 模块路径: `dki.core.dki_system`  
> 版本: 3.1.0

---

## 1. 模块概述

`DKISystem` 是 DKI 论文的**核心实现**，实现了注意力层级别的记忆增强生成系统。与 RAG 将记忆拼接到 Prompt 不同，DKI 将记忆以 K/V 张量形式注入 Transformer 的自注意力层，不占用 Token Budget。

**核心能力:**

- 记忆检索 + K/V 注入生成
- 杂化注入 (Hybrid Injection): 偏好 K/V + 历史 Suffix Prompt
- **Recall v4 记忆召回策略 (v3.1 新增)**: 多信号召回 + 逐消息 Summary + Fact Call
- 双因子门控 (Dual-Factor Gating): 相关性驱动 + 熵调制
- 分层 K/V 缓存: L1(GPU) → L2(CPU) → L3(SSD) → L4(重算)
- 位置重映射: RoPE/ALiBi 兼容
- 注意力预算分析: Token vs Attention Budget

### 1.1 v3.1 新增特性

| 特性 | 说明 |
|------|------|
| **Recall v4 记忆召回** | 多信号召回 (关键词 + 向量 + 指代) + 逐消息 Summary + Fact Call |
| **动态 History 预算** | 根据 context_window 自适应调整历史大小 |
| **结构化认知标记** | `[SUMMARY]...[/SUMMARY]` 机器可读格式 |
| **事实补充循环** | 模型可通过 `retrieve_fact()` 请求原文补充 |
| **模型适配格式化** | Generic/DeepSeek/GLM 三种格式化器 |
| **Full Attention 弃用** | 因长历史场景不可用, 推荐使用 Recall v4 |

---

## 2. 数据结构

### 2.1 DKIResponse

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 生成的回复文本 |
| `memories_used` | `List[MemorySearchResult]` | 使用的记忆列表 |
| `gating_decision` | `GatingDecision` | 门控决策详情 |
| `latency_ms` | `float` | 总延迟 (毫秒) |
| `input_tokens` | `int` | 输入 token 数 |
| `output_tokens` | `int` | 输出 token 数 |
| `cache_hit` | `bool` | 是否命中缓存 |
| `cache_tier` | `str` | 命中的缓存层级 |
| `budget_analysis` | `Optional[BudgetAnalysis]` | 预算分析 |
| `latency_breakdown` | `Optional[LatencyBreakdown]` | 延迟分解 |
| `metadata` | `Dict[str, Any]` | 附加元数据 (含 recall_v4 信息) |

**metadata 结构 (v3.1 新增 recall_v4 字段):**

```python
metadata = {
    "model": "deepseek-7b",
    "session_cache_stats": {...},
    "task_type": "reasoning",
    "hybrid_injection": {
        "enabled": True,
        "preference_tokens": 25,
        "history_tokens": 150,
        "preference_alpha": 0.4,
        "preference_text": "- [dietary] 素食主义者",
        "history_suffix_text": "...",
        "history_messages": [...],
        "final_input": "...",
    },
    # v3.1 新增
    "recall_v4": {
        "enabled": True,
        "strategy": "summary_with_fact_call",
        "trace_ids": ["msg-001", "msg-005", "msg-012"],
        "fact_rounds_used": 1,
    },
}
```

---

## 3. DKISystem 初始化

### 3.1 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_adapter` | `Optional[BaseModelAdapter]` | 模型适配器 (延迟加载) |
| `memory_router` | `Optional[MemoryRouter]` | 记忆路由器 |
| `embedding_service` | `Optional[EmbeddingService]` | 嵌入服务 |
| `engine` | `Optional[str]` | 推理引擎名称 |

### 3.2 初始化流程

```
DKISystem.__init__()
  ├─ 加载配置 (ConfigLoader)
  ├─ 初始化核心服务
  │   ├─ EmbeddingService (单例)
  │   └─ MemoryRouter (FAISS 索引)
  ├─ 模型适配器 (延迟加载)
  ├─ DKI 组件 (延迟初始化)
  │   ├─ MemoryInfluenceScaling (MIS)
  │   ├─ QueryConditionedProjection
  │   ├─ DualFactorGating
  │   └─ PositionRemapper
  ├─ 缓存系统
  │   ├─ _use_tiered_cache (配置决定)
  │   ├─ _session_caches: Dict[str, SessionKVCache]
  │   └─ _tiered_caches: Dict[str, TieredKVCache]
  ├─ AttentionBudgetAnalyzer
  ├─ HybridDKIInjector (延迟初始化)
  ├─ _user_preferences: Dict[str, UserPreference]
  ├─ DatabaseManager (SQLite/PostgreSQL)
  │
  └─ (v3.1) Recall v4 组件初始化
      ├─ 检测 RECALL_V4_AVAILABLE (recall 包是否可导入)
      ├─ 读取 config.dki.recall 配置
      │   ├─ 如果是 dict → 直接使用
      │   └─ 如果是对象 → _obj_to_dict() 递归转换
      ├─ RecallConfig.from_dict(recall_dict)
      ├─ _use_recall_v4 = config.enabled AND strategy == "summary_with_fact_call"
      └─ 惰性组件 (首次 chat() 时初始化):
          ├─ _multi_signal_recall: Optional[MultiSignalRecall]
          ├─ _suffix_builder: Optional[SuffixBuilder]
          ├─ _fact_retriever: Optional[FactRetriever]
          └─ _recall_formatter: Optional[PromptFormatter]
```

### 3.3 延迟加载属性 (Property)

所有核心组件均采用**延迟加载**模式，首次访问时创建:

| 属性 | 组件 | 初始化依赖 |
|------|------|-----------|
| `model` | `BaseModelAdapter` | `ModelFactory.get_or_create(engine)` |
| `mis` | `MemoryInfluenceScaling` | `model.hidden_dim` |
| `projection` | `QueryConditionedProjection` | `model.hidden_dim` |
| `gating` | `DualFactorGating` | 无 |
| `position_remapper` | `PositionRemapper` | `model.model_name` (检测位置编码类型) |
| `hybrid_injector` | `HybridDKIInjector` | 配置 + model_adapter + tokenizer |

### 3.4 Recall v4 组件惰性初始化

> 源文件: `dki/core/dki_system.py` 第 405-468 行

Recall v4 组件在首次 `chat()` 调用时通过 `_init_recall_v4_components()` 初始化:

```
_init_recall_v4_components()
  ├─ 前置条件: _use_recall_v4 == True AND RECALL_V4_AVAILABLE
  ├─ 幂等性: 已初始化则直接返回
  │
  ├─ 获取模型名称 (用于自动选择格式化器)
  │   ├─ model_adapter.model_name
  │   └─ 或 config.model.engines[default_engine].model_name
  │
  ├─ 创建 PromptFormatter (模型适配)
  │   └─ create_formatter(model_name, formatter_type, language)
  │       ├─ model_name 含 "deepseek" → DeepSeekFormatter
  │       ├─ model_name 含 "glm"/"chatglm" → GLMFormatter
  │       └─ 其他 → GenericFormatter
  │
  ├─ 创建 _ConversationRepoWrapper (数据库桥接)
  │   └─ 为 recall 组件提供统一的数据库访问接口
  │
  ├─ 创建 MultiSignalRecall (多信号召回器)
  │   ├─ config: RecallConfig
  │   ├─ memory_router: MemoryRouter (向量检索)
  │   └─ conversation_repo: _ConversationRepoWrapper
  │
  ├─ 创建 SuffixBuilder (后缀组装器)
  │   ├─ config: RecallConfig
  │   ├─ prompt_formatter: PromptFormatter
  │   ├─ token_counter: 使用 model tokenizer (如果可用)
  │   └─ model_adapter: 用于 LLM 摘要 (可选)
  │
  └─ 创建 FactRetriever (事实检索器)
      ├─ config: RecallConfig
      └─ conversation_repo: _ConversationRepoWrapper
```

---

## 4. 杂化注入策略 (Hybrid Injection) — 论文 Section 3.9

### 4.1 设计理念

杂化注入是 DKI 系统的核心创新之一，模拟人类认知的分层信息处理:

```
┌──────────────────────────────────────────────────────────────────┐
│  层级  │  内容       │  注入方式           │  影响方式   │  类比    │
├────────┼─────────────┼─────────────────────┼─────────────┼──────────┤
│  L1    │  用户偏好   │  K/V (负位置)       │  隐式影响   │  人格    │
│  L2    │  会话历史   │  Suffix Prompt      │  显式参考   │  记忆    │
│  L3    │  当前查询   │  Input (正位置)     │  主要焦点   │  当前    │
└──────────────────────────────────────────────────────────────────┘
```

**为什么偏好用 K/V 注入?**

- 偏好内容短 (通常 < 100 tokens)，OOD 风险低
- 偏好稳定，K/V 可跨会话缓存复用
- 负位置注入实现"背景影响"，不干扰主要生成

**为什么历史用 Suffix Prompt?**

- 历史内容长 (100-2000+ tokens)，K/V 注入 OOD 风险高
- 历史动态变化，每轮对话都不同
- Prompt 形式支持模型引用和溯源
- 信任建立提示词减少幻觉

### 4.2 HybridDKIInjector 数据结构

**UserPreference:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `str` | 偏好文本 |
| `user_id` | `str` | 用户标识 |
| `metadata` | `Dict` | 元数据 |
| `kv_cache` | `Optional[Any]` | 缓存的 K/V 张量 |
| `token_count` | `int` | Token 数量 |

**SessionHistory:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[SessionMessage]` | 消息列表 |
| `session_id` | `str` | 会话标识 |
| `max_tokens` | `int` | 最大 token 数 |

**HybridInjectionConfig:**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `preference_enabled` | `True` | 启用偏好注入 |
| `preference_alpha` | `0.4` | 偏好注入强度 |
| `preference_max_tokens` | `100` | 偏好最大 token |
| `preference_position_strategy` | `"negative"` | 位置策略 |
| `history_enabled` | `True` | 启用历史注入 |
| `history_max_tokens` | `500` | 历史最大 token |
| `history_max_messages` | `10` | 最大消息数 |
| `history_method` | `"suffix_prompt"` | 注入方式 |

### 4.3 杂化注入完整流程

```
HybridDKIInjector.prepare_input(user_query, preference, history, system_prompt)
  │
  ├─ 1. 构建文本输入 (正确顺序: System → History → Query)
  │   ├─ text_parts = []
  │   ├─ 添加 system_prompt (如果有)
  │   ├─ 格式化历史 → _format_history(history)
  │   │   ├─ 获取最近 N 条消息
  │   │   ├─ 安全过滤: 跳过包含注入模板标记的消息 (防递归嵌套)
  │   │   │   └─ 检查 "[会话历史参考]", "[Session History Reference]" 等标记
  │   │   ├─ 格式化: "用户: xxx" / "助手: xxx" (根据语言)
  │   │   ├─ 截断: 超过 max_tokens 时从最早消息开始删除
  │   │   └─ 包装: prefix_template + content + suffix_template
  │   ├─ 添加 "User: {query}"
  │   └─ result.input_text = "\n\n".join(text_parts)
  │
  ├─ 2. 安全检查: 总 prompt 长度
  │   ├─ 获取 max_model_len (从 model 或 tokenizer)
  │   ├─ max_prompt_tokens = max_model_len - 512 (预留生成空间)
  │   └─ 超长时截断历史，只保留 system_prompt + query
  │
  ├─ 3. 偏好 K/V 计算 (如果启用)
  │   └─ _get_or_compute_preference_kv(preference)
  │       ├─ 检查 preference.kv_cache (对象级缓存)
  │       ├─ 检查 _preference_cache (实例级缓存, key = "user_id:hash(content)")
  │       └─ 计算 K/V
  │           ├─ model.compute_kv(content) → kv_entries (DKI 适配器)
  │           └─ 或 model.model(input_ids, use_cache=True) → past_key_values (HuggingFace)
  │
  └─ 4. 返回 HybridInjectionResult
      ├─ input_text: 组装后的完整文本
      ├─ preference_kv: 偏好 K/V 张量
      ├─ preference_alpha: 偏好注入强度
      ├─ preference_tokens / history_tokens / total_tokens
      ├─ preference_text: 偏好原文 (用于前端显示)
      ├─ history_suffix_text: 历史后缀原文 (用于前端显示)
      └─ history_messages: 历史消息列表 (用于前端显示)
```

### 4.4 历史消息格式化示例

**中文模板 (language="cn"):**

```
[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
请在理解历史上下文后，给出连贯的整体回复。
重要：请使用中文回复用户。
---
用户: 我对花生过敏
助手: 好的，我记住了。我会在推荐食物时避免含花生的选项。
用户: 推荐一个北京的餐厅
助手: 推荐海底捞，他们可以根据过敏情况定制菜单。
---
[会话历史结束]
请基于以上历史和用户当前问题，使用中文给出回复。
注意：历史信息仅供参考，请综合回答。
```

**英文模板 (language="en"):**

```
[Session History Reference]
Before responding, please refer to the following session history.
These are real conversation records between you and the user, and are trustworthy.
Please provide a coherent response after understanding the historical context.
---
User: I'm allergic to peanuts
Assistant: Got it. I'll avoid recommending foods with peanuts.
User: Recommend a restaurant in Beijing
Assistant: I recommend Haidilao, they can customize menus for allergies.
---
[End of Session History]
Please respond based on the above history and the user's current question.
Note: Historical information is for reference; please answer comprehensively.
```

### 4.5 Token 估算算法

当 tokenizer 不可用时，使用启发式估算:

```python
# 中文字符: ~1.5 tokens/字 (子词分词)
# 英文单词: ~1.3 tokens/词
# 标点/空格: ~1 token

chinese_chars = count(CJK字符)
english_words = count(非CJK文本的空格分词)
estimated = int(chinese_chars * 1.5 + english_words * 1.3)
```

---

## 5. Recall v4 记忆召回策略 (v3.1 新增)

### 5.1 设计背景

原有杂化注入 (Hybrid Injection) 使用 `get_session_history()` 获取最近 N 条消息作为 Suffix Prompt。此方案在**短历史**场景下工作良好，但在**长历史**场景下存在局限:

| 问题 | 说明 |
|------|------|
| 固定窗口 | 仅取最近 N 条，可能遗漏重要历史 |
| 无语义召回 | 不根据查询内容检索相关历史 |
| 无摘要机制 | 长消息全量放入，浪费 context 预算 |
| 无事实补充 | 摘要后无法追溯原文细节 |

Recall v4 解决了这些问题，通过多信号召回 + 逐消息 Summary + Fact Call 实现了稳定可靠的长历史处理。

### 5.2 Recall v4 架构

```
用户输入 (query)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  MultiSignalRecall (多信号召回)                    │
│  ├─ 关键词检索 (jieba + TF-IDF/TextRank)         │
│  ├─ 向量相似度检索 (MemoryRouter/EmbeddingService)│
│  ├─ 指代解析 (ReferenceResolver)                  │
│  ├─ 近期轮次补充 (min_recent_turns ~ max_recent)  │
│  └─ 分数归一化 + 加权融合                          │
│      final_score = w1*keyword + w2*vector + w3*rec │
└──────────────────┬───────────────────────────────┘
                   │ RecallResult(messages, scores)
                   ▼
┌──────────────────────────────────────────────────┐
│  SuffixBuilder (后缀组装)                          │
│  ├─ 动态 token 预算: context_window - reserves     │
│  ├─ 遍历消息:                                      │
│  │   ├─ token数 > threshold → Summary + trace_id   │
│  │   │   └─ [SUMMARY] facts_covered/missing,       │
│  │   │       confidence, trace_id [/SUMMARY]        │
│  │   └─ token数 <= threshold → 保留原文             │
│  ├─ 添加约束指令 (PromptFormatter)                 │
│  └─ 返回 AssembledSuffix                           │
└──────────────────┬───────────────────────────────┘
                   │ assembled_suffix (text + trace_ids)
                   ▼
┌──────────────────────────────────────────────────┐
│  DKISystem.chat() (主流程)                         │
│  ├─ 偏好 K/V 注入 (不变, HybridDKIInjector)       │
│  ├─ 历史后缀: 使用 assembled_suffix 替代           │
│  ├─ 门控 + 推理                                    │
│  └─ Fact Call 循环 (_execute_fact_call_loop)       │
│      ├─ 检测 retrieve_fact() 调用                  │
│      ├─ FactRetriever.retrieve(trace_id, offset)   │
│      ├─ 追加事实到 prompt                          │
│      └─ 重新推理 (最多 max_rounds 轮)              │
└──────────────────────────────────────────────────┘
```

### 5.3 多信号召回 (MultiSignalRecall)

```
MultiSignalRecall.recall(query, session_id, user_id, db_session, max_results)
  │
  ├─ 1. 关键词检索 (如果 keyword_enabled)
  │   ├─ jieba 分词 + 停用词过滤
  │   ├─ TF-IDF 或 TextRank 提取关键词 (keyword_topk 个)
  │   ├─ 遍历会话消息, 计算关键词命中分数
  │   └─ 归一化: sigmoid(score) → [0, 1]
  │
  ├─ 2. 向量检索 (如果 vector_enabled)
  │   ├─ memory_router.search(query, top_k=vector_top_k)
  │   ├─ 过滤: score >= vector_threshold
  │   └─ 归一化: clip(score, 0, 1) → [0, 1]
  │
  ├─ 3. 指代解析 (如果 reference_enabled)
  │   └─ reference_resolver.resolve(query)
  │       → 调整 recall_limit
  │
  ├─ 4. 分数融合
  │   └─ final_score = w1 * norm_keyword + w2 * norm_vector + w3 * recency_bonus
  │       w1 >= w2 (关键词优先, 保证事实准确性)
  │
  ├─ 5. 排序 + 截断 (max_results)
  │
  ├─ 6. 补充近期轮次 (min_recent_turns ~ max_recent_turns)
  │   └─ 确保最近的对话不会被遗漏
  │
  └─ 返回 RecallResult(messages, scores)
```

### 5.4 后缀组装 (SuffixBuilder)

```
SuffixBuilder.build(query, recalled_messages, context_window, preference_tokens)
  │
  ├─ 1. 计算 token 预算
  │   └─ budget = context_window - generation_reserve - instruction_reserve
  │            - preference_tokens - query_tokens
  │
  ├─ 2. 遍历消息 (按时间/相关性排序)
  │   ├─ 估算消息 token 数
  │   ├─ token数 > per_message_threshold:
  │   │   ├─ 生成摘要 (extractive: jieba TextRank / llm: 模型调用)
  │   │   ├─ 添加结构化认知标记:
  │   │   │   [SUMMARY]
  │   │   │   facts_covered: ["餐厅名称", "大致位置"]
  │   │   │   facts_missing: ["营业时间", "价格", "预约方式"]
  │   │   │   confidence: medium
  │   │   │   trace_id: msg-005
  │   │   │   [/SUMMARY]
  │   │   └─ 格式化: prompt_formatter.format_summary_item(item)
  │   │
  │   └─ token数 <= per_message_threshold:
  │       └─ 格式化: prompt_formatter.format_message_item(item)
  │
  ├─ 3. 预算检查
  │   └─ 累积 token 超过 budget → 停止添加
  │
  ├─ 4. 添加约束指令
  │   └─ prompt_formatter.get_constraint_instruction()
  │       ├─ "以下信息来自用户历史对话, 内容可信"
  │       ├─ "标记为 [SUMMARY] 的是摘要, 可能缺失细节"
  │       ├─ "如需精确信息, 请调用 retrieve_fact(trace_id=...)"
  │       └─ 强约束: "若未调用 retrieve_fact 直接基于 summary 给出具体数值/时间, 该回答视为无效"
  │
  └─ 返回 AssembledSuffix
      ├─ text: 完整后缀文本
      ├─ total_tokens: 总 token 数
      ├─ summary_count: 摘要条目数
      ├─ message_count: 原文消息数
      ├─ trace_ids: 所有 trace_id 列表
      └─ has_fact_call_instruction: 是否包含 fact call 指导
```

### 5.5 事实补充循环 (_execute_fact_call_loop)

> 源文件: `dki/core/dki_system.py` 第 1129-1208 行

```
_execute_fact_call_loop(output, prompt, session_id, max_new_tokens, temperature,
                        preference_kv, preference_alpha)
  │
  ├─ 配置:
  │   ├─ max_rounds = recall_config.fact_call.max_rounds (默认 3)
  │   └─ max_fact_tokens = recall_config.fact_call.max_fact_tokens (默认 800)
  │
  ├─ 循环 (最多 max_rounds 轮):
  │   │
  │   ├─ 检测 Fact Call
  │   │   └─ _recall_formatter.detect_fact_request(output.text)
  │   │       ├─ 解析 retrieve_fact(trace_id="msg-005", offset=0, limit=5)
  │   │       └─ 返回 FactRequest 或 None
  │   │
  │   ├─ 如果无 Fact Call → return (output, round_idx) ✅
  │   │
  │   ├─ 检索事实
  │   │   └─ _fact_retriever.retrieve(trace_id, session_id, offset, limit)
  │   │       ├─ 从 ConversationRepository 获取原文
  │   │       ├─ 支持分块: offset + limit (用于长文本)
  │   │       └─ 返回 FactResponse(messages, total_count, has_more)
  │   │
  │   ├─ 如果无消息 → return (output, round_idx)
  │   │
  │   ├─ 格式化事实段落
  │   │   └─ _recall_formatter.format_fact_segment(fact_response)
  │   │
  │   ├─ Token 预算检查
  │   │   └─ total_fact_tokens > max_fact_tokens → return
  │   │
  │   ├─ 追加事实到 prompt
  │   │   └─ prompt += "\n\n" + fact_text + "\n\n" + "请基于以上补充事实回答用户问题。"
  │   │
  │   └─ 重新推理
  │       ├─ 有偏好 K/V → model.forward_with_kv_injection(prompt, preference_kv, alpha)
  │       └─ 无偏好 K/V → model.generate(prompt)
  │
  └─ 返回 (output, max_rounds)
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

### 5.6 配置 (config.yaml)

```yaml
dki:
    recall:
        enabled: true
        strategy: "summary_with_fact_call"  # summary_with_fact_call | flat_history

        signals:
            keyword_enabled: true
            keyword_topk: 5
            keyword_method: "tfidf"         # tfidf | textrank
            vector_enabled: true
            vector_top_k: 10
            vector_threshold: 0.5
            reference_enabled: true

        budget:
            generation_reserve: 512
            instruction_reserve: 150
            min_recent_turns: 2
            max_recent_turns: 5

        summary:
            per_message_threshold: 200
            max_tokens_per_summary: 150
            strategy: "extractive"          # extractive | llm

        fact_call:
            enabled: true
            max_rounds: 3
            max_fact_tokens: 800
            batch_size: 5

        prompt_formatter: "auto"            # auto | generic | deepseek | glm
```

### 5.7 Recall v4 与原有 Hybrid 的关系

```
                    ┌──────────────────────────────────┐
                    │ DKISystem.chat() 路由判断         │
                    └──────────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ _use_recall_v4  │    │ use_hybrid_mode │    │ 其他             │
    │ == True          │    │ == True          │    │                  │
    │ + hybrid + allow │    │ + allow          │    │                  │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Recall v4 路径   │    │ 原有 Hybrid 路径 │    │ 标准 DKI 路径   │
    │                  │    │                  │    │                  │
    │ 多信号召回       │    │ get_session_     │    │ 直接门控 +      │
    │ + 后缀组装       │    │ history()        │    │ K/V 注入        │
    │ + 偏好 K/V       │    │ + prepare_input()│    │                  │
    │ + Fact Call       │    │                  │    │                  │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

**关键点:**
- Recall v4 **替代**了 Hybrid 的历史部分 (`get_session_history`)
- Recall v4 **保留**了 Hybrid 的偏好 K/V 注入部分
- 当 Recall v4 启用时，`hybrid_result` 仅用于偏好 K/V，`history=None`

---

## 6. 偏好注入详细说明

### 6.1 偏好数据来源

```
用户偏好来源链路:
  前端 → POST /api/preferences → SQLite user_preferences 表
                                         │
  DKI System ← _load_user_preferences_from_db() ← SELECT ... FROM user_preferences
                                         │
                                    _user_preferences 内存缓存
                                         │
                                    HybridDKIInjector.prepare_input()
```

### 6.2 偏好加载 SQL

```sql
SELECT preference_text, preference_type, priority, category
FROM user_preferences
WHERE user_id = :user_id AND is_active = 1
ORDER BY priority DESC
```

### 6.3 偏好合并格式

多条偏好记录合并为单一文本:

```
- [dietary] 素食主义者，不吃肉
- [allergy] 花生过敏
- [style] 喜欢简洁的回复风格
- [location] 住在北京朝阳区
```

### 6.4 偏好 K/V 计算与缓存

```
偏好 K/V 缓存层级:
  1. preference.kv_cache (对象级) — 最快，随对象生命周期
  2. _preference_cache[user_id:hash(content)] (实例级) — 跨请求复用
  3. 重新计算 — model.compute_kv(preference_text)
```

### 6.5 偏好 K/V 注入位置

偏好 K/V 在注意力计算中被**前置 (prepend)** 到记忆 K/V 之前:

```
最终 K/V 序列:
  [偏好_K/V (α_pref 缩放)] + [记忆_K/V (α_gate 缩放)] + [用户输入_K/V]
       负位置区域                  记忆区域                   正位置区域
```

数学表示:

```
K_aug = [K_pref ; α_gate · K_mem ; K_user]
V_aug = [α_pref · V_pref ; α_gate · V_mem ; V_user]
Attn = softmax(Q · K_aug^T / sqrt(d)) · V_aug
```

> **注意**: Key tensor 不被 alpha 缩放 (v3.1 设计不变量)，仅 Value tensor 被缩放。

---

## 7. 历史消息检索召回方案

### 7.1 原有历史检索流程 (flat_history)

```
DKISystem.get_session_history(session_id, max_messages=10)
  ├─ 创建 SessionHistory 对象
  ├─ 数据库查询
  │   └─ ConversationRepository.get_recent(session_id, limit=max_messages)
  │       └─ SELECT * FROM conversations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
  ├─ 遍历消息，添加到 SessionHistory
  │   └─ history.add_message(role, content, timestamp)
  └─ 返回 SessionHistory
```

> **注意**: 当 Recall v4 启用时，此方法不再用于历史注入。历史由 `MultiSignalRecall` + `SuffixBuilder` 处理。

### 7.2 历史消息安全过滤

在格式化历史时，系统会过滤掉包含注入模板标记的消息，防止递归嵌套:

```python
injection_markers = [
    "[会话历史参考]",
    "[Session History Reference]",
    "[会话历史结束]",
    "[End of Session History]",
]
# 如果消息内容包含这些标记，跳过该消息
```

**为什么需要过滤?**

- 上一轮对话中，助手的回复可能包含了注入的历史后缀
- 如果不过滤，下一轮会将上一轮的历史后缀再次嵌套进去
- 导致 prompt 无限膨胀

### 7.3 历史截断策略

当历史消息超过 `history_max_tokens` 时:

```
截断算法:
  while estimate_tokens(history_content) > max_tokens AND len(lines) > 1:
      lines.pop(0)  # 删除最早的消息
      history_content = "\n".join(lines)
```

**策略:** 保留最近的消息，删除最早的消息 (时间优先)。

### 7.4 数据库记录策略

对话记录使用**原始用户输入** (不含历史后缀):

```python
# 保存原始查询 (避免历史递归嵌套)
original_query = query

# ... 杂化注入/Recall v4 后 query 可能包含历史后缀 ...

# 记录到数据库时使用原始查询
self._log_conversation(
    query=original_query,  # 不是修改后的 query
    ...
)
```

---

## 8. chat() 主流程 — 完整调用链 (v3.1)

```
DKISystem.chat(query, session_id, user_id, ...)
  │
  ├─ 0. 确定注入模式 (hybrid / standard)
  │   └─ use_hybrid_mode = use_hybrid or config.dki.hybrid_injection.enabled
  │
  ├─ 0.1 保存原始查询 (original_query = query)
  │
  ├─ 0.2 加载用户偏好 (如果 hybrid 模式且未缓存)
  │   └─ _load_user_preferences_from_db(user_id)
  │       └─ SELECT FROM user_preferences WHERE user_id = ? AND is_active = 1
  │
  ├─ 0.5 历史准备 (三选一):
  │   │
  │   ├─ (A) Recall v4 路径 [_use_recall_v4 + hybrid + allow_injection]
  │   │   ├─ timer.start_stage("recall_v4")
  │   │   ├─ _init_recall_v4_components() (惰性初始化)
  │   │   ├─ 获取偏好: get_user_preference(user_id)
  │   │   ├─ 多信号召回: _multi_signal_recall.recall(query, session_id, user_id)
  │   │   ├─ 计算 context_window (从 config.model.engines)
  │   │   ├─ 估算 preference_tokens
  │   │   ├─ 后缀组装: _suffix_builder.build(query, messages, context_window, pref_tokens)
  │   │   ├─ recall_v4_suffix = assembled.text
  │   │   ├─ recall_v4_trace_ids = assembled.trace_ids
  │   │   ├─ query = assembled.text (替换为组装后的后缀)
  │   │   ├─ 偏好 K/V 准备: hybrid_injector.prepare_input(query, preference, history=None)
  │   │   │   └─ 仅用于偏好 K/V, 不处理历史
  │   │   └─ timer.end_stage()
  │   │
  │   ├─ (B) 原有 Hybrid 路径 [use_hybrid_mode + allow_injection]
  │   │   ├─ timer.start_stage("hybrid_prep")
  │   │   ├─ 获取偏好: get_user_preference(user_id)
  │   │   ├─ 获取历史: get_session_history(session_id)
  │   │   ├─ hybrid_injector.prepare_input(query, preference, history)
  │   │   ├─ query = hybrid_result.input_text (包含历史后缀)
  │   │   └─ timer.end_stage()
  │   │
  │   └─ (C) 无历史准备 (标准 DKI)
  │
  ├─ 1. 门控决策 (Gating)
  │   ├─ allow_injection=False → 跳过注入
  │   ├─ force_alpha 指定 → 强制注入
  │   └─ 否则 → gating.should_inject(model, query, router)
  │       ├─ 计算相关性分数 (语义检索)
  │       ├─ 计算模型熵 (不确定性)
  │       └─ 返回 GatingDecision(should_inject, alpha, memories)
  │
  ├─ 1.5 预算分析 (Budget Analysis)
  │   └─ _budget_analyzer.analyze(user_tokens, memory_tokens)
  │
  ├─ 2. 生成
  │   ├─ 不注入 (alpha < 0.1 或 should_inject=False):
  │   │   └─ model.generate(prompt=query)  // query 可能包含 recall_v4 后缀
  │   │
  │   └─ DKI 注入:
  │       └─ _generate_with_injection(query, gating_decision, ...)
  │           ├─ 收集记忆 K/V (缓存 → 计算)
  │           ├─ 合并多记忆 K/V
  │           ├─ 合并偏好 K/V (杂化注入)
  │           ├─ MIS alpha 缩放 (仅 Value)
  │           └─ model.forward_with_kv_injection(prompt, injected_kv, alpha)
  │
  ├─ 2.5 (v3.1) Fact Call 循环 [如果 recall_v4 启用]
  │   └─ _execute_fact_call_loop(output, prompt, session_id, ...)
  │       ├─ 检测 retrieve_fact() 调用
  │       ├─ 检索事实 (FactRetriever)
  │       ├─ 追加事实到 prompt
  │       └─ 重新推理 (最多 max_rounds 轮)
  │
  ├─ 3. 记录延迟
  │   └─ _budget_analyzer.record_latency(timer.breakdown)
  │
  ├─ 4. 构建元数据
  │   ├─ hybrid_injection 信息 (偏好/历史 tokens, alpha, 明文)
  │   └─ (v3.1) recall_v4 信息 (enabled, strategy, trace_ids, fact_rounds_used)
  │
  ├─ 5. 记录对话到数据库 (使用 original_query)
  │   ├─ conversations 表: user 消息 + assistant 消息
  │   └─ audit_logs 表: 审计日志
  │
  └─ 6. 返回 DKIResponse
```

---

## 9. K/V 注入生成详细流程

### 9.1 _generate_with_injection()

> 源文件: `dki/core/dki_system.py` 第 967-1127 行

```
_generate_with_injection(query, gating_decision, session_cache, ...)
  │
  ├─ 遍历 gating_decision.memories:
  │   ├─ 尝试缓存加载
  │   │   ├─ TieredKVCache: session_cache.get(memory_id, query, model)
  │   │   │   └─ 返回 (kv_entries, tier) — tier 可能是 L1/L2/L3/L4
  │   │   └─ SessionKVCache: session_cache.get(memory_id, query)
  │   │
  │   ├─ 缓存未命中 → 计算 K/V
  │   │   └─ model.compute_kv(memory.content) → (kv_entries, token_count)
  │   │       kv_entries = [KVCacheEntry(key, value, layer_idx) for each layer]
  │   │       key shape: [batch, num_heads, seq_len, head_dim]
  │   │       value shape: [batch, num_heads, seq_len, head_dim]
  │   │
  │   └─ 存入缓存
  │       ├─ TieredKVCache: put(memory_id, kv_entries, query, alpha, text_content)
  │       └─ SessionKVCache: put(memory_id, entries, query, alpha)
  │
  ├─ 合并多记忆 K/V (_merge_kv_entries)
  │   └─ 按层合并: torch.cat(keys, dim=2) — 沿序列维度拼接
  │       输入: [[Layer0_mem1, Layer1_mem1], [Layer0_mem2, Layer1_mem2]]
  │       输出: [Layer0_merged, Layer1_merged]
  │       Layer0_merged.key shape: [batch, heads, seq_mem1+seq_mem2, head_dim]
  │
  ├─ 合并偏好 K/V (如果有)
  │   ├─ mis.scale_kv_values(pref_key, pref_value, pref_alpha)
  │   │   └─ Key 不缩放, Value 缩放: scaled_value = value * sqrt(alpha)
  │   └─ 前置: merged_kv = [scaled_pref_kv] + [merged_memory_kv]
  │
  ├─ MIS alpha 缩放 (如果 alpha < 1.0)
  │   └─ 对每层: Key 不缩放, Value *= sqrt(alpha)
  │
  └─ 注入生成
      └─ model.forward_with_kv_injection(prompt, injected_kv, alpha, ...)
          内部实现:
          ├─ 正常 tokenize prompt
          ├─ 计算 prompt 的 K/V
          ├─ 拼接: K_aug = [K_injected ; K_prompt]
          │         V_aug = [V_injected ; V_prompt]
          ├─ 位置重映射 (如果需要)
          └─ 自回归生成
```

### 9.2 K/V 合并算法

```python
def _merge_kv_entries(kv_list):
    """
    合并多个记忆的 K/V 条目。

    输入: kv_list = [memory1_kv, memory2_kv, ...]
          memory_kv = [KVCacheEntry(layer_0), KVCacheEntry(layer_1), ...]

    算法: 按层拼接序列维度
    """
    num_layers = len(kv_list[0])
    merged = []

    for layer_idx in range(num_layers):
        keys = [kv[layer_idx].key for kv in kv_list]    # 每个 shape: [B, H, S_i, D]
        values = [kv[layer_idx].value for kv in kv_list]

        merged_key = torch.cat(keys, dim=2)    # [B, H, S_1+S_2+..., D]
        merged_value = torch.cat(values, dim=2)

        merged.append(KVCacheEntry(key=merged_key, value=merged_value, layer_idx=layer_idx))

    return merged
```

---

## 10. 记忆管理

### 10.1 add_memory() — 添加记忆

```
add_memory(session_id, content, memory_id, metadata)
  ├─ 计算嵌入: embedding_service.embed(content)
  ├─ 存入数据库
  │   ├─ session_repo.get_or_create(session_id)
  │   └─ memory_repo.create(session_id, content, embedding, memory_id, metadata)
  ├─ 添加到路由器: memory_router.add_memory(memory_id, content, embedding, metadata)
  └─ 返回 memory_id
```

### 10.2 load_memories_from_db() — 从数据库加载

```
load_memories_from_db(session_id)
  ├─ memory_repo.get_by_session(session_id)
  ├─ 遍历每条记忆:
  │   ├─ 获取嵌入: memory_repo.get_embedding(mem.id)
  │   └─ 添加到路由器: memory_router.add_memory(...)
  └─ 返回加载数量
```

---

## 11. 缓存系统

### 11.1 会话缓存选择

```
_get_session_cache(session_id)
  ├─ _use_tiered_cache = True?
  │   └─ TieredKVCache(l1_max_size, l2_max_size, l3_path, ...)
  └─ _use_tiered_cache = False?
      └─ SessionKVCache()
```

### 11.2 分层缓存架构

```
┌─────────────────────────────────────────────────────────┐
│  L1: GPU HBM (Hot)     - 未压缩 FP16, 最快访问        │
│  L2: CPU RAM (Warm)    - 压缩 2-4x, 较快访问          │
│  L3: NVMe SSD (Cold)   - 量化 INT8 8x, 持久化存储     │
│  L4: Text Only         - 仅存文本, 按需重算 K/V        │
└─────────────────────────────────────────────────────────┘
```

---

## 12. 数据库交互

### 12.1 涉及的数据库表

| 表名 | 操作 | 说明 |
|------|------|------|
| `sessions` | 读/写 | 会话管理 (get_or_create) |
| `memories` | 读/写 | 记忆存储 (create, get_by_session) |
| `memory_embeddings` | 读 | 记忆嵌入向量 |
| `conversations` | 读/写 | 对话记录 (create, get_recent, get_by_session, get_by_id) |
| `user_preferences` | 读 | 用户偏好 (SELECT ... WHERE is_active=1) |
| `audit_logs` | 写 | 审计日志 |

### 12.2 conversations 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `INTEGER` | 主键 |
| `session_id` | `TEXT` | 会话标识 |
| `role` | `TEXT` | 角色 (user/assistant) |
| `content` | `TEXT` | 消息内容 |
| `injection_mode` | `TEXT` | 注入模式 (dki/none) |
| `injection_alpha` | `REAL` | 注入强度 |
| `memory_ids` | `TEXT` | 使用的记忆 ID (JSON) |
| `latency_ms` | `REAL` | 延迟 (毫秒) |
| `created_at` | `DATETIME` | 创建时间 |

### 12.3 user_preferences 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | `TEXT` | 用户标识 |
| `preference_text` | `TEXT` | 偏好文本 |
| `preference_type` | `TEXT` | 偏好类型 (dietary/allergy/style/...) |
| `priority` | `INTEGER` | 优先级 |
| `category` | `TEXT` | 分类 |
| `is_active` | `INTEGER` | 是否激活 (0/1) |

### 12.4 audit_logs 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | `TEXT` | 操作类型 (dki_generate) |
| `session_id` | `TEXT` | 会话标识 |
| `memory_ids` | `TEXT` | 记忆 ID 列表 (JSON) |
| `alpha` | `REAL` | 注入强度 |
| `mode` | `TEXT` | 模式 (dki) |
| `metadata` | `TEXT` | 元数据 (JSON: entropy, relevance, reasoning) |

### 12.5 Recall v4 数据库交互 (v3.1 新增)

| 组件 | 表 | 操作 | 说明 |
|------|-----|------|------|
| `MultiSignalRecall` | `conversations` | 读 | 关键词 + 向量检索历史消息 |
| `FactRetriever` | `conversations` | 读 | 按 trace_id (message_id) 检索原文 |
| `_ConversationRepoWrapper` | `conversations` | 读 | 统一数据库访问接口 |

**_ConversationRepoWrapper:**

> 源文件: `dki/core/dki_system.py` 第 1402-1440 行

为 Recall v4 组件提供简化的 `ConversationRepository` 接口，自动管理数据库 session:

```python
class _ConversationRepoWrapper:
    def __init__(self, db_manager: DatabaseManager): ...
    
    def get_by_session(self, session_id, db_session=None, **kwargs) -> List[Any]:
        """获取会话的所有消息 (自动管理 db session)"""
    
    def get_recent(self, session_id, limit=10, **kwargs) -> List[Any]:
        """获取最近的消息"""
    
    def get_by_id(self, message_id) -> Optional[Any]:
        """根据 ID 获取消息 (用于 FactRetriever)"""
```

---

## 13. 辅助方法

### 13.1 _obj_to_dict() — 配置对象转字典 (v3.1 新增)

> 源文件: `dki/core/dki_system.py` 第 387-403 行

递归将配置对象 (如 `config.dki.recall`) 转换为字典，用于 `RecallConfig.from_dict()`:

```python
@staticmethod
def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    """将配置对象递归转为字典"""
    if isinstance(obj, dict):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith('_'):
            continue
        val = getattr(obj, key)
        if callable(val):
            continue
        if hasattr(val, '__dict__') and not isinstance(val, (str, int, float, bool, list)):
            result[key] = DKISystem._obj_to_dict(val)
        else:
            result[key] = val
    return result
```

### 13.2 should_use_dki() — DKI 使用建议

基于预算分析推荐是否使用 DKI (论文 Section 7.3):

```
should_use_dki(query, memory_count, task_type)
  ├─ 估算 user_tokens = len(query.split()) * 1.3
  ├─ 估算 memory_tokens = memory_count * 200 (假设平均 200 tokens/记忆)
  └─ _budget_analyzer.should_prefer_dki(user_tokens, memory_tokens, task_type)
```

### 13.3 get_stats() — 系统统计

返回完整的系统状态:

- 路由器统计 (记忆数量、索引大小)
- 模型信息
- 会话缓存统计
- 分层缓存统计 + 内存占用
- 预算分析统计
- 配置信息

### 13.4 clear_session_cache() — 清除会话缓存

```
clear_session_cache(session_id)
  ├─ 清除 SessionKVCache (如果存在)
  └─ 清除 TieredKVCache (如果存在)
```

### 13.5 search_memories() — 记忆搜索

```
search_memories(query, top_k)
  └─ memory_router.search(query, top_k=top_k)
```

---

## 14. 设计不变量 (论文 Section 5)

1. **存储模型无关**: 仅存储文本 + 路由向量，不存储 K/V
2. **注入模型一致**: K/V 由目标模型计算，确保语义空间一致
3. **会话缓存可丢弃**: 仅推理时使用，不影响持久化数据
4. **优雅降级**: alpha → 0 恢复原始行为
5. **审计日志**: 所有注入操作记录到 audit_logs 表
6. **Key 不缩放**: Key tensor 永远不被 alpha 缩放 (v3.1 新增)
7. **原始查询记录**: 数据库记录使用 original_query，避免历史递归嵌套

---

## 15. 依赖关系图

```
dki.core.dki_system (DKISystem)
  ├─ dki.core.memory_router (MemoryRouter, MemorySearchResult)
  ├─ dki.core.embedding_service (EmbeddingService)
  ├─ dki.core.components
  │   ├─ memory_influence_scaling (MemoryInfluenceScaling)
  │   ├─ query_conditioned_projection (QueryConditionedProjection)
  │   ├─ dual_factor_gating (DualFactorGating, GatingDecision)
  │   ├─ session_kv_cache (SessionKVCache)
  │   ├─ tiered_kv_cache (TieredKVCache, CacheTier)
  │   ├─ position_remapper (PositionRemapper)
  │   ├─ attention_budget (AttentionBudgetAnalyzer, BudgetAnalysis, LatencyTimer)
  │   └─ hybrid_injector (HybridDKIInjector, HybridInjectionConfig, UserPreference, SessionHistory)
  ├─ dki.models.factory (ModelFactory)
  ├─ dki.models.base (BaseModelAdapter, ModelOutput, KVCacheEntry)
  ├─ dki.database.connection (DatabaseManager)
  ├─ dki.database.repository (SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository)
  ├─ dki.config.config_loader (ConfigLoader)
  └─ (v3.1, 可选) dki.core.recall
      ├─ RecallConfig, FactRequest
      ├─ MultiSignalRecall
      ├─ SuffixBuilder
      ├─ FactRetriever
      └─ create_formatter (→ PromptFormatter)
```

---

## 16. DKISystem vs DKIPlugin 对比

| 维度 | DKISystem | DKIPlugin |
|------|-----------|-----------|
| **定位** | 完整的演示系统 | 可嵌入的插件 |
| **数据访问** | 直接访问数据库 | 通过 IUserDataAdapter 适配器 |
| **偏好来源** | SQLite user_preferences 表 | 上层应用通过适配器提供 |
| **历史来源** | SQLite conversations 表 | 上层应用通过适配器提供 |
| **门控** | DualFactorGating (内置) | DualFactorGating (延迟初始化) |
| **缓存** | SessionKVCache / TieredKVCache | Executor 内存缓存 + PreferenceCacheManager |
| **策略** | Hybrid + Recall v4 | Stable + Recall v4 + Full Attention (弃用) |
| **Recall v4** | ✅ (直接集成) | ✅ (通过 Planner/Executor) |
| **Fact Call** | ✅ (_execute_fact_call_loop) | ✅ (Executor._execute_fact_call_loop) |
| **架构** | 单类编排 | 决策-执行分离 (Planner + Executor) |
| **适用场景** | Demo / 独立部署 | 集成到现有应用 |
