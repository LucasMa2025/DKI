### DKIPlugin 插件核心模块说明书 (v3.3+ — Recall v4 + Stable 回退 + 用户隔离 + 熵门控/事实补充)

> **源文件**: `DKI/dki/core/dki_plugin.py`  
> **子组件包**: `DKI/dki/core/plugin/` (Planner / Executor / Plan)  
> **记忆召回包**: `DKI/dki/core/recall/` (Recall v4)  
> **缓存与用户隔离**: `DKI/dki/cache/`  
> **版本说明**: 本文档在 v3.2 版《dki_plugin 模块说明书》的基础上，按当前代码 (含 entropy_gated / inline_intercept / 偏好文本缓存 / 流式接口 等) 重新校对和修订，保持原有架构与章节结构的一致性。

---

## 1. 模块概述

### 1.1 重构背景与当前版本

-   v2.x: `DKIPlugin` 是单文件 "God Plugin"，集成编排、数据访问、注入决策与执行、监控等全部逻辑。
-   v3.0: 引入 **Planner/Executor 分层架构**，`DKIPlugin` 退化为瘦 Facade。
-   v3.1: 加入 **Recall v4 多信号召回 + SuffixBuilder + FactRetriever**。
-   v3.2: 引入 **Stable 回退策略 + 用户级 K/V 缓存隔离**，移除 Full Attention。
-   v3.3+ (当前实现的核心增量)：
    -   增加 **偏好文本 TTL + LRU 缓存 (P1‑3)**，减少每次 chat 的 DB 压力。
    -   使用 **Async SingleFlight (P0‑4)** 合并并发偏好加载，避免 thundering herd。
    -   引入 **entropy_gated / inline_intercept** 两种事实检索模式，与 Recall v4 配合。
    -   增强 **跨会话检索** 与 **最近对话 (recent_messages) 合并逻辑**，兼顾长历史与局部连贯性。
    -   扩展 **流式接口 `chat_stream()`**，对接思考内容过滤与 pseudo‑streaming。

### 1.2 当前架构 (保持与旧版一致的高层结构)

```text
上层应用
   ↓ (稳定接口: chat(query, user_id, session_id), chat_stream(...))
DKIPlugin (瘦 Facade, 只负责编排与降级策略)
   ↓
InjectionPlanner (纯决策, 不碰模型, 不做 I/O)
   ↓ InjectionPlan (中间产物, 纯数据结构)
InjectionExecutor (纯执行, 不做决策)
   ↓
ModelAdapter (LLM 推理: VLLMAdapter / SGLangAdapter / LlamaAdapter / ...)
```

**新增要点 (与 v3.2 相比)**：

-   Facade 层增加：
    -   偏好文本缓存 `_preference_text_cache` + Async SingleFlight `_preference_single_flight`。
    -   `_detect_retrieval_mode()`、`_get_max_recent_turns()` 与 `_merge_recent_and_recalled()`，实现 BM25 + 近轮历史的融合。
    -   `chat_stream()` 对应非阻塞流式输出。
-   Planner/Executor 与 3.2 版文档基本一致，但在执行路径中新增：
    -   `entropy_gated` / `inline_intercept` 两种 fact 检索模式，由 Executor 内部 `_get_fact_retrieve_method(plan)` 决定。
    -   对这些模式下的回退逻辑进行扩展（见 §4.3）。

### 1.3 模块职责总结

| 层级                | 职责                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------- |
| `DKIPlugin`         | 外部接口 (`chat`/`chat_stream`/`get_stats`)、异常降级、缓存装配、日志与可视化           |
| `InjectionPlanner`  | 查询分析 (MemoryTrigger + ReferenceResolver)；构建 `InjectionPlan` (Recall v4 / Stable) |
| `InjectionExecutor` | 按 `InjectionPlan` 执行注入：K/V 计算与缓存命中、LLM 推理、Fact Call 循环、降级         |
| `ModelAdapter`      | 抽象具体模型/引擎（vLLM/SGLang/LLaMA 等）的生成/流式/KV 注入接口                        |

---

## 2. 核心数据结构 (与旧版保持一致并按当前代码校正)

### 2.1 InjectionMetadata — 注入元数据 (监控用)

> 源代码: `dki/core/dki_plugin.py` `InjectionMetadata` dataclass

`InjectionMetadata` 由 `DKIPlugin.chat()` / `chat_stream()` 在 Planner+Executor 流程结束后组装，用于监控 API、实验系统和可视化。

关键字段分组如下：

-   **注入状态与策略**
    -   `injection_enabled: bool` — 是否实际启用注入。
    -   `alpha: float` — 有效注入强度 (通常为 `AlphaProfile.effective_preference_alpha`)。
    -   `injection_strategy: str` — `"recall_v4"` / `"stable"` / 降级策略名。
    -   `alpha_profile: Optional[Dict]` — 分层 Alpha 信息 (由 Planner 提供)。
-   **Token 统计**
    -   `preference_tokens` / `history_tokens` / `query_tokens` / `total_tokens`。
-   **缓存状态**
    -   `preference_cache_hit: bool` — 偏好 K/V 缓存是否命中（由 Executor 填充）。
    -   `preference_cache_tier: str` — `"memory"|"compute"|"error"|...`。
-   **延迟拆分**
    -   `latency_ms` — 整体请求耗时；
    -   `adapter_latency_ms` — 适配器数据访问时间；
    -   `injection_latency_ms` — Planner+Executor 注入逻辑耗时（不含模型推理）；
    -   `inference_latency_ms` — 模型推理耗时。
-   **门控 / Trigger / 指代解析**
    -   `gating_decision: Optional[Dict]` — DualFactorGating 决策详情。
    -   `memory_triggered: bool` + `trigger_type: Optional[str]`。
    -   `reference_resolved: bool` + `reference_type` + `reference_scope`。
-   **数据规模统计**
    -   `preferences_count`, `history_messages_count`, `relevant_history_count`。
-   **检索模式**
    -   `retrieval_mode: str` — `"bm25_only"|"bm25_embedding"|"keyword"|"pgvector"|"unknown"`。
    -   来自 `_detect_retrieval_mode()` 对 ConfigDrivenAdapter 配置的检查。
-   **安全与时间信息**
    -   `safety_violations: Optional[List[str]]` — SafetyEnvelope 报告的违规项。
    -   `timestamp: datetime` / `request_id: str` (8 位短 UUID)。
-   **v9.0+: 注入明文详情 (供 Web UI / 实验系统直接读取)**
    -   `preference_text: Optional[str]` — Planner 生成的偏好文本；
    -   `history_suffix_text: Optional[str]` — 历史 suffix（Recall v4 assembled_suffix 或 stable 平铺）；
    -   `history_messages: Optional[List[Dict[str,str]]]` — 被注入的历史消息列表（role+content）；
    -   `final_input: Optional[str]` — 最终发送给模型的 prompt。

`to_dict()` 会将上述信息序列化为嵌套字典，供 API 和实验结果写入 JSON 使用。

### 2.2 DKIPluginResponse — 插件响应

> 源代码: `dki/core/dki_plugin.py` `DKIPluginResponse`

对外统一返回类型：

-   `text: str` — 最终回复文本（**已通过 `strip_think_content` 去除 `<think>` 块**）。
-   `input_tokens: int` / `output_tokens: int` — Token 统计。
-   `metadata: InjectionMetadata` — 注入元数据。
-   `raw_output: Optional[ModelOutput]` — 适配器原始输出（可选）。

`chat()` 返回 `DKIPluginResponse`；`chat_stream()` 则将 `metadata.to_dict()` 和最终 text 嵌入 streaming 事件中。

### 2.3 InjectionPlan / ExecutionResult / AlphaProfile / SafetyEnvelope

> 实现在 `dki/core/plugin/injection_plan.py`，结构与旧版文档保持一致，这里仅补充与当前 `DKIPlugin` 交互相关的关键点。

-   `InjectionPlan`：
    -   `strategy: "recall_v4"|"stable"|"none"`；
    -   `assembled_suffix` / `history_suffix` / `trace_ids` / `has_fact_call_instruction` 等 Recall v4 字段；
    -   `alpha_profile: AlphaProfile`（含 `effective_preference_alpha`）；
    -   `final_input`: 最终传给模型的 prompt (由 Planner 产生)；
    -   `safety_violations: List[str]`：由 `SafetyEnvelope.validate(plan)` 填充。
-   `ExecutionResult`：
    -   `text`, `input_tokens`, `output_tokens`, `raw_output`, `inference_latency_ms`；
    -   `preference_cache_hit`, `preference_cache_tier`；
    -   `fact_rounds_used`, `fact_tokens_total`（Recall v4 Fact Call 统计）。
-   `AlphaProfile`：
    -   与旧版一致：`preference_alpha`、`history_alpha`、`override_cap` 等；
    -   当前版本中 `DKIPlugin` 读取 `plan.alpha_profile.effective_preference_alpha` 作为 `metadata.alpha`。
-   `SafetyEnvelope`：
    -   对各策略下的 α 上限和 K/V 总 token 限制进行检查；
    -   违规项写入 `InjectionPlan.safety_violations`，随后复制到 `InjectionMetadata.safety_violations`。

---

## 3. DKIPlugin — 瘦 Facade (当前代码行为)

### 3.1 构造函数参数与初始化流程

> 源代码: `DKIPlugin.__init__`

构造函数参数与旧版保持一致：

-   `model_adapter: BaseModelAdapter` — 具体引擎适配器（vLLM/SGLang/LLaMA 等）。
-   `user_data_adapter: IUserDataAdapter` — 上层应用数据适配器（通常为 `ConfigDrivenAdapter`）。
-   `config: Optional[Any]` — DKI 全局配置（默认为 `ConfigLoader().config`）。
-   `language: str` — `"en"` 或 `"cn"`。
-   `memory_trigger_config: Optional[MemoryTriggerConfig]`。
-   `reference_resolver_config: Optional[ReferenceResolverConfig]`。
-   `redis_client: Optional[DKIRedisClient]` + `cache_config: Optional[CacheConfig]`。

初始化关键步骤（对比旧版补充）：

1. **思考内容过滤器** (`think_filter`) 初始化：
    - 读取 `config.dki.think_filter`（可为 dict / pydantic 模型 / 任意对象），调用 `init_think_filter` 注册；
    - 用于后续 `strip_think_content` 与流式检测。
2. **Recall v4 组件初始化**：
    - 从配置中提取 `recall` 字段，构造 `RecallConfig`；
    - 创建 `PromptFormatter`（根据 model_name 与语言选择合适模板）；
    - 创建 `SuffixBuilder`（v6.2 修复历史压缩缺失问题）；
    - 创建 `FactRetriever`（仅依赖 config 与 conversation_repo，后者在 demo 或服务中注入）。
3. **上下文窗口解析** `_resolve_context_window()`：
    - 优先从 `config.model.engines.{engine}.max_model_len` 获取；
    - 回退到 `model_adapter.tokenizer.model_max_length`；
    - 再回退到默认 `4096`。
4. **Planner 与 Executor 创建**：
    - Planner: 注入 `config`、`language`、`injection_strategy="recall_v4"`，并传入 `recall_config`/`SuffixBuilder`/`FactRetriever`/`PromptFormatter`。
    - Executor: 注入 `model_adapter`、`function_call_logger`（如可用）、`FactRetriever`、`PromptFormatter`、`RecallConfig`。
5. **偏好 K/V 缓存 (L1/L2)**：
    - `_preference_cache = PreferenceCacheManager(redis_client, cache_config)`；
    - 可选通过 Redis 实现 L2 分布式缓存。
6. **偏好文本缓存 (v3.3 新增)**：
    - `_preference_text_cache: OrderedDict[str, Tuple[List[UserPreference], float]]`：
        - LRU + TTL (默认 5 分钟)，缓存「偏好列表」而非 KV。
    - `_preference_single_flight: Dict[str, asyncio.Future]`：
        - 合并同一 `user_id` 的并发偏好加载。
7. **统计与日志**：
    - `_injection_logs: List[InjectionMetadata]`，最多 1000 条；
    - `_stats` 记录总请求数、注入启用次数、缓存命中数、总延迟、平均 alpha 等；
    - 通过 `record_visualization()` 将可视化数据写入 Web UI/实验系统使用的可视化存储。

### 3.2 from_config() — 推荐创建方式 (当前行为)

> 源代码: `DKIPlugin.from_config`

流程与旧文档一致，这里仅强调当前代码中实际执行的关键点：

1. 使用 `ConfigLoader()` 读取 `config.yaml`，并尝试加载原始 YAML 为 `_raw_config`；
2. 创建 `ConfigDrivenAdapter`：
    - 优先使用 `adapter_config_path`；
    - 其次使用 `adapter_config`（可为 dict 或 YAML 路径字符串）；
    - 否则从默认路径列表中搜索 YAML。
3. 调用 `await user_adapter.connect()` 建立数据库连接。
4. 根据参数或 `_raw_config` 构造 `MemoryTriggerConfig` / `ReferenceResolverConfig`。
5. 处理 Redis：
    - 依据 `enable_redis` 参数或 `_raw_config['redis'].enabled` 决定是否尝试；
    - 若安装了 redis 库且连接成功，则启用 L2 缓存；
    - 否则降级为 L1-only，并给出日志提示。
6. 最终构建 `DKIPlugin` 实例，并返回。

### 3.3 组件访问器与配置更新

-   `planner` / `executor` / `mis` / `gating` 属性：与旧版一致，用于测试和高级用法。
-   `get_stats()` / `get_injection_logs()` / `get_cache_stats()`：聚合 Facade+Planner+Executor+Cache 多层统计，与旧文档中示例保持兼容，但增加了：
    -   `retrieval_mode` 相关统计；
    -   用户隔离与缓存命中率等字段。
-   `update_reference_resolver_config()` / `update_memory_trigger_config()` / `get_component_configs()`：
    -   委托给 Planner 内部，接口语义维持不变。

---

## 4. chat() 主流程 — 新版调用链

> 源代码: `DKIPlugin.chat`

### 4.1 整体流程 (与旧版保持结构一致，补充当前逻辑)

```text
DKIPlugin.chat(query, user_id, session_id, force_alpha=None, ...)
  │
  ├─ Step 1: Planner.analyze_query(query) → QueryContext
  │   ├─ MemoryTrigger.detect(query) → TriggerResult
  │   └─ ReferenceResolver.resolve(query) → ResolvedReference
  │
  ├─ Step 2: 通过适配器与缓存加载数据
  │   ├─ _get_cached_preferences(user_id)
  │   │   ├─ LRU+TTL 文本缓存 (P1‑3)
  │   │   └─ Async SingleFlight 合并并发请求 (P0‑4)
  │   ├─ _detect_retrieval_mode() → retrieval_mode (bm25_only/bm25_embedding/...)
  │   ├─ data_adapter.search_relevant_history(user_id, query, session_id=None, limit=...)
  │   └─ data_adapter.get_recent_messages(user_id, limit=2*max_recent_turns)
  │       └─ _merge_recent_and_recalled(recent_messages, recalled_messages)
  │
  ├─ Step 3: Planner.build_plan(...)
  │   └─ 生成 InjectionPlan (recall_v4/ stable + assembled_suffix/trace_ids/alpha_profile/...)
  │
  ├─ Step 3.5: 模糊指代澄清 (detect_vague_reference + build_clarification_instruction)
  │   └─ 在历史不足时向 plan 注入 "澄清提问" 指令 (可选)
  │
  ├─ Step 4: Executor 执行注入计划
  │   ├─ fact_method = executor._get_fact_retrieve_method(plan)
  │   ├─ if fact_method == "entropy_gated":
  │   │     → _execute_entropy_gated(plan, prompt=plan.final_input, ...)
  │   ├─ elif fact_method == "inline_intercept":
  │   │     → _execute_inline_intercept(plan, prompt=plan.final_input, ...)
  │   └─ else:
  │         → executor.execute(plan, max_new_tokens, temperature, ...)
  │
  ├─ Step 5: 记录注入日志 (_record_injection_log)
  │   ├─ 将 plan/ExecutionResult 填充到 InjectionMetadata
  │   └─ record_visualization(...) 记录可视化数据
  │
  └─ Step 6: strip_think_content(result.text) → 返回 DKIPluginResponse
```

### 4.2 数据加载与历史合并 (相对旧版的重要变化)

1. **偏好加载 `_get_cached_preferences(user_id)`**：
    - 首先从 `_preference_text_cache` (LRU+TTL) 读取 `(preferences, cached_at)`。
    - TTL 过期则删除缓存并继续 DB 加载。
    - 使用 `AsyncSingleFlight` 合并同一用户的并发请求，避免同时大量命中 DB。
2. **历史加载**：
    - `search_relevant_history(user_id, query, session_id=None)`：
        - 允许跨会话召回相关历史，依赖 ConfigDrivenAdapter 的 BM25/向量/关键词能力。
    - `get_recent_messages(user_id, limit)`：
        - 召回最近若干轮（user+assistant 成对）对话，用于保持多轮连贯性。
    - `_merge_recent_and_recalled(recent, recalled)`：
        - 近轮优先 + 去重；
        - 按时间戳排序，确保历史按时间线注入；
        - `_remove_trailing_unpaired_user` 从尾部删除没有 assistant 回复的 user 消息，避免“当前查询”在历史中重复。

### 4.3 Executor 路由与事实检索模式

`chat()` 在调用 Executor 之前，会根据 `plan` 和 `self._fact_retriever` 获取事实检索模式：

-   `fact_method = self._executor._get_fact_retrieve_method(plan)`：
    -   `"entropy_gated"`：2 阶段熵门控检索模式。
    -   `"inline_intercept"`：停用符号拦截，在线追加事实。
    -   `"post_hoc"` / 其他：常规（非熵门控）路径。

路由规则：

-   `entropy_gated` 且 `plan.has_fact_call_instruction` 且存在 `_fact_retriever`：
    -   调用 `_execute_entropy_gated(plan, prompt=plan.final_input, ...)`；
    -   阶段 1：探测阶段，短生成 + logprobs 计算熵，决定是否检索；
    -   阶段 2：检索事实并增强 prompt，再执行完整生成。
-   `inline_intercept` 条件类似：
    -   调用 `_execute_inline_intercept(...)`；
    -   通过 stop token 截断与 Fact Call 拦截，在单次生成内部完成检索与补充。
-   否则：
    -   调用 `executor.execute(plan, ...)`，使用 Recall v4 默认 Fact Call 循环或 Stable 路径。

**注意**：`chat_stream()` 对 `entropy_gated` / `inline_intercept` 目前采用「回退到非流式，再模拟流式输出」策略，以保证逻辑正确性优先。

### 4.4 异常与降级策略 (Facade 级别)

`chat()` 对核心路径中的结构化异常进行分类处理：

-   `AdapterConnectionError`：
    -   记录为可重试错误；
    -   尝试 `_fallback_without_adapter()`（不依赖适配器数据，仅用偏好 K/V + 原始查询）；失败时再退到 `_fallback_no_injection()`。
-   `AdapterSchemaError`：
    -   被视作永久性错误，直接退到 `_fallback_no_injection()`。
-   `KVOOMError` / `ModelOOMError`：
    -   清理 GPU cache（如可用），退到 `_fallback_no_injection()`。
-   一般 `DKIError`：
    -   调用 `_fallback_stable_then_none()`：
        -   先尝试构建 stable_plan 并执行；
        -   失败则退到 `_fallback_no_injection()`。
-   其他未捕获异常：
    -   也走 `_fallback_stable_then_none()`。

降级路径中的核心行为与旧盗版文档中的 "Stable 回退" 描述保持一致，只是加入了更多日志信息与错误分类。

---

## 5. chat_stream() — 流式生成流程

> 源代码: `DKIPlugin.chat_stream`

### 5.1 与 chat() 的共享阶段

-   Step 1: `analyze_query()`；
-   Step 2: `_get_cached_preferences()` + `search_relevant_history()` + `get_recent_messages()` + `_merge_recent_and_recalled()`；
-   Step 3: `build_plan()`；
-   然后立即 `yield {"type": "metadata", "metadata": metadata.to_dict()}` 把注入元数据先行推给客户端。

### 5.2 流式执行路径

1. 事实检索模式检查：
    - 若 `fact_method in ("entropy_gated", "inline_intercept")` 且有 Fact Call 指令与 `FactRetriever`：
        - 退回到非流式执行 (`_execute_entropy_gated` 或 `_execute_inline_intercept`)；
        - 使用 `yield {"type": "token", ...}` + `yield {"type": "done", ...}` 模拟流式。
2. 检查模型是否支持流式：
    - 支持：
        - 优先选择 `async_stream_generate`（如适配器提供），否则使用同步 `stream_generate`；
        - 通过 `StreamThinkDetector` 将 chunk 划分为：
            - type="token" 正常内容；
            - type="thinking" 思考内容（仅在 `show_thinking` 配置为真时下发）。
        - 结束后 `detector.flush()` 输出缓冲区残留；
        - 使用 `detector.get_clean_text()` 得到已清理 think 内容的最终文本；
        - 利用 `estimate_tokens_fast` 对 input/output tokens 做快速估算。
    - 不支持：
        - 退回 `executor.execute(plan, ...)` 非流式执行；
        - 模拟流式输出全部文本。

### 5.3 错误事件

-   若流式过程出现异常，将 `yield {"type": "error", "error": str(e), "error_code": ...}`；
-   不会抛出异常打断上层 event loop，方便前端处理。

---

## 6. 缓存与用户隔离 (当前实现视角)

### 6.1 偏好文本缓存 `_get_cached_preferences()`

与旧文档中只描述偏好 K/V 缓存不同，当前实现增加了「偏好文本缓存」这一层：

-   **结构**：`_preference_text_cache: OrderedDict[user_id → (preferences, cached_at)]`；
-   **策略**：
    -   LRU + TTL：
        -   `maxsize=1000`，超过时弹出最旧条目；
        -   TTL 默认 300 秒；
    -   SingleFlight：
        -   `_preference_single_flight: Dict[flight_key → Future]`；
        -   同一 `user_id` 并发请求共享一次 DB 访问结果。

这种缓存位于 **Facade 层**，为 Planner 和 Executor 提供稳定且低延迟的偏好列表输入。

### 6.2 偏好 K/V 缓存与用户隔离

偏好 K/V 缓存在 `InjectionExecutor` 中实现，当前代码仍遵循 v3.2 文档的设计：

-   按 `user_id` 做物理分区：`_preference_kv_cache[user_id][content_hash]`；
-   `content_hash = md5(preference_text)`，偏好更新后自然 invalid；
-   通过 `InferenceContextGuard` 保证每次执行结束后 KV 缓存不会泄漏到其他请求；
-   Redis/L1 等多层缓存策略无变更，详见旧版文档的缓存章节。

### 6.3 invalidate_user_cache()

`DKIPlugin.invalidate_user_cache(user_id)`：

-   首先清除偏好文本缓存 (`invalidate_preference_text_cache(user_id)`)；
-   再调用 `_preference_cache.invalidate(user_id)` 清理偏好 K/V 缓存；
-   适用于用户偏好更新后的立即生效场景。

---

## 7. 与旧版说明书的差异小结

为便于迁移和对比，这里列出新版文档相对 `dki_plugin模块说明书.md (v3.2)` 的**主要补充点**：

-   **新增逻辑**（旧文档未覆盖或仅简略提及）：
    -   偏好文本缓存 + Async SingleFlight (`_get_cached_preferences`)；
    -   检测检索模式 `_detect_retrieval_mode` 与 `retrieval_mode` 写入 `InjectionMetadata`；
    -   近轮对话获取 `get_recent_messages` + 合并 `_merge_recent_and_recalled` + 末尾 user 过滤；
    -   事实检索模式路由：`entropy_gated` / `inline_intercept`；
    -   流式接口 `chat_stream()` 的详细行为 (metadata 首包、think 过滤、伪流式 fallback)；
    -   `InjectionMetadata` 中的 `injection_detail` 明文注入信息字段。
-   **保持不变的高层设计**：
    -   Facade / Planner / Executor 三层架构；
    -   Recall v4 与 Stable 策略的基本语义与数据结构；
    -   Full Attention 已移除，不再在代码或文档中保留任何执行路径。

---

## 8. 使用建议与注意事项

-   若你是在 **实验系统** 中使用 DKI：
    -   建议开启 Recall v4（默认），并在结果分析中同时记录 `InjectionMetadata.retrieval_mode` 与 `injection_detail` 字段，以便分析注入 vs. 召回行为。
    -   对需要严格控制实验条件的场景，可以通过 `force_alpha` 和 Planner 配置将策略固定到 `"recall_v4"` 或 `"stable"`。
-   若你是在 **生产系统** 中使用 DKI：
    -   推荐开启偏好文本缓存与 Redis L2 缓存，减小高并发场景下的 DB 压力。
    -   `chat_stream()` 在出现复杂 fact 模式时会自动回退为「非流式 + 模拟流式」，需要在前端留意「首个 metadata 事件 + 若干 token 事件 + done 事件」这一协议。

本说明书可与旧版 `dki_plugin模块说明书.md` 并存：  
旧文档适合理解 v3.2 级别的高层设计，新文档对当前代码中的新特性做了**补丁式对齐**，两者在整体架构与术语上保持一致。

---

## 9. 缓存架构（与 v3.2 保持一致的整体设计）

### 9.1 偏好 K/V 缓存层级

> 这一部分的实现主要在 `InjectionExecutor` 内部，当前 DKIPlugin 版本仍沿用 v3.2 设计。

```text
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

-   **L0**: Executor 本地 `_preference_kv_cache[user_id][content_hash]`，用户级隔离，命中时不需要任何 I/O。
-   **L1**: `PreferenceCacheManager` 内存 LRU，供同一进程内多个 Executor/Plugin 共享。
-   **L2**: Redis 分布式缓存（如启用），在多实例部署下保持整体命中率稳定。
-   **L3**: 调用 `model.compute_kv(preference_text)` 重新计算 KV，作为最终兜底路径。

DKIPlugin 在构造时通过 `CacheConfig` 和 `DKIRedisClient` 决定是否启用 L2 缓存，并通过 `get_cache_stats()` 提供多层命中率观测。

### 9.2 K/V 缓存键与用户隔离

> 详细逻辑实现在 `dki/core/plugin/injection_executor.py` 中，这里只总结设计要点。

```text
缓存结构: {user_id: {content_hash: (kv_entries, content_hash)}}

cache_key = md5(preference_text)
缓存分区 = _preference_kv_cache[user_id]

示例:
  _preference_kv_cache["user_123"]["a1b2..."] = (kv_entries, "a1b2...")
  _preference_kv_cache["user_456"]["b2c3..."] = (kv_entries, "b2c3...")
```

安全保证：

-   按 `user_id` 物理分区，任何用户的 KV 不会出现在其他用户的分区中。
-   当用户偏好发生变化时，拼接出的 `preference_text` 发生变化，`md5` 自然改变，旧缓存失效。
-   配合 `InferenceContextGuard`，确保每次推理结束后 KV 不会残留在当前上下文之外。

### 9.3 多实例部署下的命中率

-   没有 Redis（仅 L0+L1）：
    -   单实例命中率约 ~70%；
    -   多实例时命中率按实例数折减（2 实例约 35%，4 实例约 17.5%）。
-   有 Redis（启用 L2）：
    -   所有实例共享 L2 缓存，整体命中率可保持在单实例水平（~70%）。

---

## 10. 数据库交互

### 10.1 通过适配器访问的上层应用表

> 与 v3.2 一致，DKI 不直接操作上层应用的表结构，只通过 `IUserDataAdapter`（通常是 `ConfigDrivenAdapter`）按接口访问。

典型上层应用表（以 demo 集成为例）：

| 表名                          | 操作 | 说明     |
| ----------------------------- | ---- | -------- |
| `user_preferences` (上层应用) | 读   | 用户偏好 |
| `chat_messages` (上层应用)    | 读   | 历史消息 |

具体表名与字段映射通过 adapter 配置 (`adapter_config.yaml`) 决定。

### 10.2 DKI 内部存储

DKIPlugin 及其组件主要使用以下内部存储：

| 存储                 | 操作  | 说明                                       |
| -------------------- | ----- | ------------------------------------------ |
| 可视化数据 (内存)    | 写    | `record_visualization()`，用于 Web UI 图表 |
| 注入日志 (内存)      | 写    | `_injection_logs`，最多 1000 条            |
| 偏好 K/V 缓存 (内存) | 读/写 | Executor 内部 `_preference_kv_cache`       |
| 偏好文本缓存 (内存)  | 读/写 | DKIPlugin `_preference_text_cache`         |

### 10.3 Recall v4 相关数据库交互

Recall v4 组件（`MultiSignalRecall` / `FactRetriever`）通常与上层会话表交互：

| 组件                | 表              | 操作 | 说明                            |
| ------------------- | --------------- | ---- | ------------------------------- |
| `MultiSignalRecall` | `conversations` | 读   | 检索历史消息 (关键词 + 向量)    |
| `FactRetriever`     | `conversations` | 读   | 按 trace_id 检索原文 (支持分块) |

在实验系统中，这些表通常映射到 `demo_messages` / `demo_sessions` 等实验专用表，由 `ConfigDrivenAdapter` 的配置负责字段映射。

---

## 11. 监控 API

### 11.1 get_stats() — 统计数据

> 源代码: `dki/core/dki_plugin.py` `get_stats`

`get_stats()` 聚合 Facade、Planner、Executor、Cache 多层统计数据，用于监控页面和实验分析。

返回示例（结构与旧版一致，字段可能有所扩展）：

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
#         "recall_v4_plans": 42,
#         "stable_fallback_plans": 3,
#         "strategy": "recall_v4",
#         "memory_trigger": {...},
#         "reference_resolver": {...},
#     },
#
#     # Executor 层统计
#     "executor": {
#         "executions": 150,
#         "recall_v4_executions": 42,
#         "plain_executions": 10,
#         "fallbacks": 2,
#         "stable_fallbacks": 3,
#         "cache_hits": 85,
#         "cache_user_isolation_denials": 0,
#         "fact_call_rounds": 15,
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

### 11.2 get_injection_logs() — 注入日志

> 源代码: `dki/core/dki_plugin.py` `get_injection_logs`

返回最近的注入日志列表，每条日志对应一个 `InjectionMetadata`：

```python
logs = plugin.get_injection_logs(limit=100, offset=0)
# [
#   {
#     "request_id": "a1b2c3d4",
#     "timestamp": "2026-02-16T10:30:00",
#     "injection_enabled": True,
#     "injection_strategy": "stable",
#     "alpha": 0.35,
#     "alpha_profile": {...},
#     "tokens": {"preference": 25, "history": 150, "query": 20, "total": 195},
#     "cache": {"preference_hit": True, "preference_tier": "memory"},
#     "latency": {"total_ms": 42.5, ...},
#     "data_source": {...},
#     "memory_trigger": {...},
#     "reference_resolver": {...},
#     "retrieval_mode": "bm25_embedding",
#     "injection_detail": {
#         "preference_text": "...",
#         "history_suffix_text": "...",
#         "history_messages": [...],
#         "final_input": "...",
#     },
#     ...
#   },
#   ...
# ]
```

这些日志既用于线上监控，也可用于离线实验分析（对比不同注入策略、不同 alpha 配置的效果）。

---

## 12. 运行时配置更新

### 12.1 切换注入策略（推荐通过 Planner 控制）

虽然当前代码中注入策略主要由 Planner 内部配置与 Recall v4/Stable 回退逻辑决定，但依然可以通过修改 Planner 配置来“偏向”某种策略：

```python
# 获取组件配置
configs = plugin.get_component_configs()
# {
#     "memory_trigger": {...},
#     "reference_resolver": {...},
#     "injection_strategy": "recall_v4",
# }
```

如需临时禁用 Recall v4，可在配置层面将 `recall` 相关参数关闭，Planner 会自然退化为 Stable 路径；  
不再提供显式的 Full Attention 开关（该策略已在实现中彻底移除）。

### 12.2 更新 Reference Resolver 配置

> 源代码: `DKIPlugin.update_reference_resolver_config`

```python
plugin.update_reference_resolver_config(
    just_now_turns=3,      # "刚刚" 召回 3 轮
    recently_turns=30,     # "最近" 召回 30 轮
    last_topic_turns=20,   # "那件事" 召回 20 轮
)
```

内部委托给 `InjectionPlanner.update_reference_resolver_config`，实时影响后续 `analyze_query` 的召回策略。

### 12.3 更新 Memory Trigger 配置

```python
plugin.update_memory_trigger_config(
    enabled=True,
    custom_patterns=[{"pattern": r"我的名字是", "type": "identity"}],
)
```

支持运行时启停 Memory Trigger，或添加新的触发模式（例如身份、自我介绍等）。

### 12.4 获取组件配置快照

```python
configs = plugin.get_component_configs()
# {
#     "memory_trigger": {...},
#     "reference_resolver": {...},
#     "injection_strategy": "recall_v4",
# }
```

可用于在管理端展示当前运行时策略或导出配置。

---

## 13. 错误处理与降级 (v3.3 视角)

整体降级策略仍延续 v3.2 的三级思路，只是在 `DKIPlugin.chat()` 顶层增加了更细的异常分类：

```text
异常处理层级:

  Level 0: Planner 阶段 Recall v4 失败
    └─ Planner 回退到 flat_history_fallback
       ├─ plan.recall_strategy → "flat_history_fallback"
       ├─ plan.strategy → "stable"
       └─ 使用平铺历史后缀

  Level 1: Executor 执行异常
    └─ _execute_stable_fallback()
       ├─ 使用 plan.history_suffix 重构 stable 输入
       ├─ 获取偏好 K/V (如果可用)
       └─ model.forward_with_kv_injection() 或 model.generate()
       └─ 失败 → _execute_fallback()
                └─ model.generate(plan.original_query) (无注入)

  Level 2: DKIPlugin.chat() 顶层异常
    ├─ AdapterConnectionError:
    │   └─ _fallback_without_adapter() (偏好 K/V + 原始查询)
    │       └─ 失败 → _fallback_no_injection()
    ├─ AdapterSchemaError:
    │   └─ _fallback_no_injection()
    ├─ KVOOMError / ModelOOMError:
    │   └─ 清理 GPU cache → _fallback_no_injection()
    ├─ 其他 DKIError:
    │   └─ _fallback_stable_then_none()
    └─ 未分类异常:
        └─ _fallback_stable_then_none()
```

在所有降级路径中，都会继续记录 `InjectionMetadata`，仅改变 `injection_strategy` 字段值，例如 `"adapter_retry_fallback"`、`"schema_error_fallback"`、`"oom_fallback"`、`"none_fallback"` 等，方便后续统计和报警。

---

## 14. 使用示例（按当前代码校正）

### 14.1 从配置文件创建（推荐）

```python
from dki.core.dki_plugin import DKIPlugin
from dki.models import VLLMAdapter

# 1. 创建模型适配器
model_adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")

# 2. 从 adapter_config.yaml 创建 DKI 插件
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
)

# 3. 发起一次带注入的对话
response = await dki.chat(
    query="推荐一家适合素食者的北京餐厅",
    user_id="user_123",
    session_id="session_456",
)

print(response.text)
print(response.metadata.alpha)
print(response.metadata.alpha_profile)
print(response.metadata.retrieval_mode)
```

### 14.2 启用 Redis 分布式缓存

```python
dki = await DKIPlugin.from_config(
    model_adapter=model_adapter,
    adapter_config_path="config/adapter_config.yaml",
    enable_redis=True,
    redis_config={"host": "redis.example.com", "port": 6379},
)
```

### 14.3 高级用法：直接访问 Planner / Executor

```python
# 生成计划但不执行 (测试/调试)
context = dki.planner.analyze_query("推荐餐厅")
plan = dki.planner.build_plan(
    query="推荐餐厅",
    user_id="user_123",
    preferences=prefs,
    relevant_history=history,
    context=context,
    session_id="session_456",
    context_window=8192,
)

print(plan.to_dict())
print(plan.trace_ids)  # Recall v4 中的 trace_ids

# 手动执行计划
result = await dki.executor.execute(plan)
print(result.text)
print(result.fact_rounds_used)
```

### 14.4 强制 Alpha（实验用途）

```python
response = await dki.chat(
    query="推荐一家餐厅",
    user_id="user_123",
    session_id="session_456",
    force_alpha=0.8,  # 跳过门控，强制注入强度
)
```

### 14.5 关闭插件

```python
await dki.close()  # 关闭 Redis + 数据库连接
```

---

## 15. 依赖关系图（更新版）

```text
dki.core.dki_plugin (Facade)
  ├─ dki.core.plugin.injection_planner (决策)
  │   ├─ dki.core.plugin.injection_plan (数据结构)
  │   ├─ dki.core.components.memory_trigger
  │   ├─ dki.core.components.reference_resolver
  │   └─ dki.core.recall (Recall v4)
  │       ├─ MultiSignalRecall
  │       ├─ SuffixBuilder
  │       └─ PromptFormatter
  │
  ├─ dki.core.plugin.injection_executor (执行)
  │   ├─ dki.core.plugin.injection_plan (数据结构)
  │   ├─ dki.models.base (BaseModelAdapter)
  │   ├─ dki.cache.user_isolation (InferenceContextGuard)
  │   └─ dki.core.recall
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

至此，`dki_plugin模块说明书_v3_3.md` 覆盖了旧版说明书中的全部关键章节（架构、数据结构、流程、缓存、数据库交互、监控、配置更新、降级策略、使用示例与依赖关系），并对照当前 `dki_plugin.py` 代码补充了偏好文本缓存、熵门控/inline_intercept、流式接口以及检索模式等新增能力，实现**完整、面向当前实现的高质量程序说明**。
