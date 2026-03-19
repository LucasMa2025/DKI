### DKIPlugin 代码审查报告（`dki/core/dki_plugin.py`）v20260312

> 审查范围：`DKI/dki/core/dki_plugin.py` 为主，结合 `dki/core/plugin/*` 与 `dki/core/recall/*` 的文档和调用关系做交叉校验。  
> 审查目标：梳理程序流程与组件职责，找出现有实现中的明确 bug 与高风险逻辑问题，并给出修改建议。

---

## 1. 架构与流程概览（基于当前源码）

### 1.1 顶层职责

`DKIPlugin` 作为瘦 Facade，负责：

- 对外提供稳定接口：
  - `chat()`：单次请求的非流式增强对话；
  - `chat_stream()`：流式增强对话；
  - 工厂方法 `from_config()`；
  - 统计与监控接口 `get_stats()` / `get_injection_logs()`；
  - 缓存与组件配置接口。
- 内部编排：
  - 调用 `InjectionPlanner` 完成查询分析与注入计划构建；
  - 调用 `InjectionExecutor` 执行注入计划、模型推理与 Fact Call；
  - 管理偏好文本缓存与偏好 K/V 缓存；
  - 在成功路径下更新统计、记录可视化数据。

### 1.2 `chat()` 主流程（源码级梳理）

按当前实现，`chat()` 的主路径为：

1. **参数准备**：
   - 若 `max_new_tokens` 未显式传入，则从 `config.model.max_new_tokens` 读取，缺省为 2048。
   - 创建新的 `InjectionMetadata` 实例。
2. **Step 1：Planner Phase 1 — 查询分析**：
   - `context = self._planner.analyze_query(query)`：
     - 内部触发 MemoryTrigger / ReferenceResolver；
     - 将 `memory_triggered` / `trigger_type` / `reference_resolved` / `reference_type` / `reference_scope` 写入 `context`；
   - 将上述字段复制到 `metadata`。
3. **Step 2：适配器与缓存加载数据**：
   - 偏好：
     - `preferences = await self._get_cached_preferences(user_id)`；
     - 该方法内部实现 TTL+LRU + AsyncSingleFlight（见 §3.1）。
     - `metadata.preferences_count = len(preferences)`。
   - 检索模式：
     - `retrieval_mode = self._detect_retrieval_mode()`；
     - 根据 ConfigDrivenAdapter 的 `adapter_config.vector_search` 推断 `"bm25_only" | "bm25_embedding" | "pgvector" | "keyword" | "unknown"`；
     - 结果写入 `metadata.retrieval_mode`。
   - 历史：
     - 主召回：`relevant_history = await self.data_adapter.search_relevant_history(user_id, query, session_id=None, limit=context.recall_limit)`；
       - 支持跨会话召回；
     - 近轮对话：`recent_messages = await self.data_adapter.get_recent_messages(user_id, limit=max_recent * 2)`；
     - 合并：若存在 `recent_messages`，则调用 `_merge_recent_and_recalled(recent_messages, recalled_messages)`：
       - 近轮优先插入、按 message_id 去重；
       - 按时间戳排序；
       - 末尾 user 无 assistant 配对时，通过 `_remove_trailing_unpaired_user` 删除，避免当前 query 重复出现在历史中。
     - `metadata.relevant_history_count = len(relevant_history)`。
   - 记录 `metadata.adapter_latency_ms`。
4. **Step 3：Planner Phase 2 — 构建注入计划**：
   - `plan = self._planner.build_plan(...)`，传入：
     - `query / user_id / preferences / relevant_history / context / force_alpha / session_id / context_window`；
   - 从 `plan` 填充：
     - `metadata.injection_strategy`（策略名）；
     - `metadata.injection_enabled`；
     - `metadata.alpha = plan.alpha_profile.effective_preference_alpha`；
     - `metadata.alpha_profile = plan.alpha_profile.to_dict()`；
     - `metadata.preference_tokens` / `history_tokens` / `query_tokens` / `total_tokens`；
     - `metadata.gating_decision` / `metadata.safety_violations`。
5. **Step 3.5：模糊指代澄清**：
   - `_vague_ref = detect_vague_reference(query)`；
   - 若判定为模糊指代且历史太少 (`len(relevant_history) <= 2`)，则为 `plan` 增加 `clarification_instruction`。
6. **Step 4：Executor 执行计划**：
   - 通过 `self._executor._get_fact_retrieve_method(plan)` 决定 fact 路由：
     - `"entropy_gated"` 且 `plan.has_fact_call_instruction` 且有 `_fact_retriever`：
       - 调用 `_execute_entropy_gated(plan, prompt=plan.final_input, ...)` 执行两阶段熵门控检索；
     - `"inline_intercept"` 类似，用 `_execute_inline_intercept(...)` 执行 stop-token 拦截式补充；
     - 其他情况默认走 `self._executor.execute(plan, ...)` 常规路径。
   - 从 `ExecutionResult` 填充：
     - `metadata.inference_latency_ms`；
     - `metadata.preference_cache_hit` / `metadata.preference_cache_tier`；
     - 用总执行时间减去 `inference_latency_ms` 得到 `injection_latency_ms`。
7. **Step 5：记录日志与可视化**：
   - 成功路径中调用 `_record_injection_log(metadata, query, user_id, session_id, final_input=plan.final_input, plan=plan)`：
     - 将 `metadata` 存入 `_injection_logs`；
     - 更新 `_stats` 中各项聚合统计；
     - 从 `plan` 构造注入明文信息与 Recall v4 trace 元数据；
     - 调用 `record_visualization(...)` 写入可视化数据。
8. **Step 6：strip_think 并返回**：
   - 通过 `strip_think_content(result.text)` 移除 `<think>` 内容；
   - 构造 `DKIPluginResponse` 返回。

错误路径下，采用多级降级辅助方法 `_fallback_without_adapter` / `_fallback_no_injection` / `_fallback_stable_then_none`，但目前存在监控与日志上的一致性问题（见 §2.1）。

### 1.3 `chat_stream()` 主流程

`chat_stream()` 在 Planner + Adapter 阶段几乎与 `chat()` 完全一致，差别在于：

- 在构建完 `InjectionPlan` 并填充部分 `metadata` 后：
  - 先 `yield {"type": "metadata", "metadata": metadata.to_dict()}`；
  - 然后进入流式执行分支：
    - 若 fact 模式为 `entropy_gated` / `inline_intercept`，则退回非流式执行（`_execute_entropy_gated` / `_execute_inline_intercept`），再模拟流式输出；
    - 若模型支持 `async_stream_generate` / `stream_generate`，则直接走模型流式接口，并通过 `StreamThinkDetector` 拆分 thinking / token；
    - 否则退回 `executor.execute(plan, ...)` 非流式执行，再按整段文本模拟流式。

需要注意的是，当前实现中 `chat_stream()` **不会调用 `_record_injection_log` 和 `_stats` 更新逻辑**，这是一个和 `chat()` 不对称的点（见 §2.2）。

---

## 2. 明确 bug 与高风险逻辑问题

本节只列出可以在当前源码基础上“定位到具体代码片段”的问题，并按严重程度给出建议。

### 2.1 问题一：异常与降级路径不记录注入日志 / 统计（监控缺失）

**现象：**

- 在 `chat()` 成功路径中，会调用 `_record_injection_log(...)` 更新 `_stats` 和 `_injection_logs`，并写可视化数据；
- 但在所有 `except` 分支里（`AdapterConnectionError` / `AdapterSchemaError` / `KVOOMError` / `ModelOOMError` / `DKIError` / 其他异常），在调用 `_fallback_*` 并返回前**都没有调用 `_record_injection_log`**。

**影响：**

- 所有触发降级路径的请求（包括适配器失败、OOM、DKI 内部错误等）：
  - 不会出现在 `get_injection_logs()` 返回的日志中；
  - 不会计入 `get_stats()` 的 `total_requests` / `cache_hits` / `avg_latency_ms` 等统计；
  - 不会写入 `record_visualization` 可视化数据。
- 监控层面会出现明显偏差：只看到“正常请求”，看不到“异常/降级请求”，难以分析故障。

**建议修复方案：**

- 将 `_record_injection_log(...)` 从“成功路径专用”提升到“**统一的 finally 阶段**”，例如：
  - 在 `chat()` 中使用 try/except/`finally`，在 finally 内判断是否已构造 `plan`/`result` 并填充必要字段，然后统一调用 `_record_injection_log`；
  - 或在各 `_fallback_*` 方法内部调用 `_record_injection_log`，并保证这些方法在返回前**至少调用一次**。
- 同时，确保降级路径下 `InjectionMetadata` 至少包含：
  - `injection_strategy`（`adapter_retry_fallback` / `schema_error_fallback` / `oom_fallback` / `none_fallback` 等）；
  - 合理的 `latency_ms` 与 `total_tokens`（即使为 0 也要显示出来）。

### 2.2 问题二：`chat_stream()` 不更新统计与注入日志（与 `chat()` 不一致）

**现象：**

- `chat_stream()` 只在开始阶段构建 `metadata` 并 `yield` 一次 `"metadata"` 事件；
- 整个函数内部**没有任何地方调用 `_record_injection_log` 或更新 `_stats`**；
- 与之对比，`chat()` 在成功路径会在返回前调用 `_record_injection_log`。

**影响：**

- 通过 `chat_stream()` 发起的所有请求：
  - 不计入 `get_stats()` 中的任何统计；
  - 不会添加到 `_injection_logs` 中；
  - 不会被 `record_visualization` 记录。
- 大部分前端/服务端如果统一使用流式接口，将导致监控“几乎为空”，只看得到少量非流式请求。

**建议修复方案：**

- 在 `chat_stream()` 内，模仿 `chat()` 结束路径，在每个成功返回前（即 `yield "done"` 事件前）调用 `_record_injection_log(...)`，例如：
  - 在 entropy_gated/inline_intercept 回退分支中，在 `yield "done"` 之前调用；
  - 在 has_stream 分支中，在 `yield "done"` 之前调用；
  - 在非流式 fallback 分支中，同样处理。
- 需要注意：
  - 由于 `chat_stream()` 是一个 async generator，不能简单用 finally 包裹整个函数；可在每个正常结束路径显式调用；
  - `metadata.latency_ms` 需要在计算完后写入。

### 2.3 问题三：`_fallback_without_adapter` 的实现与文档不一致

**现象：**

- 文档注释：  
  > “降级: 不使用适配器数据, 仅偏好 K/V + 原始查询”
- 实现代码：
  - 构造 `stable_plan` 时设置 `injection_enabled=False`，且没有填充任何 `preference_text`；
  - 直接调用 `self._executor.execute(stable_plan, ...)`。

结合 Executor 的典型实现（按 v3.2 文档与现有代码）：

- 当 `injection_enabled=False` 或不存在 `preference_text` 时，Executor 会走纯 `generate()` 路径，而不会做 KV 注入；
- 因此 `_fallback_without_adapter` 实际行为更接近“**无注入的稳定回退**”，而非“偏好 K/V + 原始查询”。

**影响：**

- 注释与实现不一致可能误导后来维护者；
- 对某些场景，可能期望“适配器挂了还能用偏好 K/V 注入”，但实际行为是完全无注入。

**建议修复方案（两种取其一）：**

- 若设计目标确实是“无注入稳定回退”：
  - 将 docstring 更新为“降级: 不依赖适配器, 直接调用 Executor 执行 stable 计划（不保证有偏好注入）”，并在调用者侧也相应修改注释；
- 若设计目标是“尽可能保留偏好注入”：
  - 在 `_fallback_without_adapter` 中模仿 `_fallback_stable_then_none` 的逻辑：
    - 尝试 `get_user_preferences(user_id)`；
    - 构造 `preference_text` 与 `alpha_profile`；
    - 设置 `injection_enabled=True`；
    - 然后交给 Executor 做 K/V 注入推理。

目前从 `_fallback_stable_then_none` 的写法看，更符合“**fallback 时尽量保留偏好注入**”的原则，建议两者逻辑保持一致。

### 2.4 问题四：`invalidate_user_cache` 未清理 Executor 本地 L0 K/V 缓存

**现象：**

- `invalidate_user_cache` 文档：
  > “当用户偏好更新时调用此方法:  
  >  - 使偏好文本缓存失效 (P1-3)  
  >  - 使偏好 KV 缓存失效”
- 实现：

```python
self.invalidate_preference_text_cache(user_id)
return await self._preference_cache.invalidate(user_id)
```

仅清理了：

- Facade 层的 `_preference_text_cache`（偏好文本缓存）；
- `PreferenceCacheManager` 中的 L1/L2 K/V 缓存。

没有清理：

- Executor 内部的 L0 `_preference_kv_cache[user_id][content_hash]`。

**影响：**

- 在长生命周期进程中，如果 Executor 的 L0 KV 缓存已经存在某用户的 KV：
  - 调用 `invalidate_user_cache(user_id)` 只会清理文本缓存和 L1/L2；
  - 由于 L0 命中优先，后续请求仍可能使用过期 KV；
  - 对“偏好更新后立即生效”的语义不满足。

**建议修复方案：**

- 在 `invalidate_user_cache` 中增加对 Executor 的 L0 缓存清理，例如：

```python
async def invalidate_user_cache(self, user_id: str) -> int:
    self.invalidate_preference_text_cache(user_id)
    # 清理 Executor L0 KV 缓存
    self._executor.clear_preference_cache(user_id)
    # 清理 PreferenceCacheManager (L1/L2)
    return await self._preference_cache.invalidate(user_id)
```

- 或者将 `clear_preference_cache` 的职责统一封装到 `PreferenceCacheManager`，由 Executor 委托给管理器，但总之需要保证**三层缓存同时失效**。

### 2.5 问题五：`_get_cached_preferences` 的错误传播语义可能掩盖系统性故障

**现象：**

- SingleFlight 等待分支：

```python
if flight_key in self._preference_single_flight:
    try:
        return await asyncio.shield(self._preference_single_flight[flight_key])
    except Exception:
        # 降级: 主请求异常时, 等待方返回空列表而不是向上传播
        return []
```

- 发起方分支：
  - 异常时将异常设置到 future，并重新抛出，让调用方感知错误。

**影响：**

- 对于同一时刻并发的多请求：
  - 首个（发起方）会在 DB/Adapter 抛错时收到异常；
  - 后续 join 的所有请求只会收到 `[]` 偏好列表，而不会知道错误发生。
- 这会导致：
  - 某类系统性错误（例如 DB 权限错误、字段缺失）在高并发场景下**只暴露给少数请求**；
  - 大部分请求“静默降级”为“无偏好”，难以通过错误率明显观察到问题。

**建议修复方案：**

- 视项目容忍度调整策略：
  - 若希望“容错优先”：当前做法可以接受，但建议至少打一个 warning，把这种 silent fallback 计入某个统计字段（例如 metadata 或 stats 中增加 `preference_load_errors` 计数）；
  - 若希望“错误显性化”：可以选择在 join 方也重新抛出异常（即不捕获），由上层统一处理。

---

## 3. 设计与实现上的正面评价（简要）

在审查过程中，也有一些设计点值得保留和强调：

- **Facade/Planner/Executor 分层清晰**：
  - DKIPlugin 不直接操作模型与 KV，仅负责控制流和监控；
  - Planner/Executor 可以在不依赖 DKIPlugin 的情况下独立测试。
- **Recall v4 与 Stable 策略实现回退闭环**：
  - Planner 在 Recall v4 失败时自动降级到 Stable（flat history）；
  - Executor 遇到执行异常时再进入 Stable / 无注入多级降级；
  - DKIPlugin 在顶层又增加一层稳定回退（尤其面对 Adapter 错误和 OOM）。
- **缓存设计整体合理**：
  - 区分偏好文本缓存与 K/V 缓存；
  - 使用 TTL + LRU + SingleFlight 减轻高并发 DB 压力；
  - L0/L1/L2 多层缓存架构在大规模部署中更易调优。
- **流式接口与思考内容过滤**：
  - `chat_stream()` 在模型支持 streaming 时优先走真实流式；
  - 通过 `StreamThinkDetector` 和 `strip_think_content` 保证持久化文本干净；
  - 对不支持 streaming 或复杂 fact 模式的情况有合理的回退策略。

---

## 4. 建议的后续工作与优先级

结合上述问题，推荐的修复优先级如下：

- **P0（尽快修复）**：
  1. 为 `chat()` 的所有降级路径补充 `_record_injection_log` 调用，保证监控完整；
  2. 为 `chat_stream()` 补充 `_record_injection_log` / `_stats` 更新逻辑。
- **P1（中期修复）**：
  3. 修正 `invalidate_user_cache` 未清理 Executor L0 KV 缓存的问题；
  4. 明确 `_fallback_without_adapter` 的预期行为，统一其实现与注释（建议与 `_fallback_stable_then_none` 对齐，使其尽量保留偏好注入）。
- **P2（可选优化）**：
  5. 根据产品需求，决定 `_get_cached_preferences` 对 Secondary callers 是否应传播异常，或至少增加错误统计；
  6. 补充一两组 **集成测试**：
     - 针对 `chat_stream()` 的统计与日志；
     - 针对 `invalidate_user_cache()` 对三层缓存（文本/L0/L1/L2）的行为。

若你愿意，我可以在下一步直接基于本审查报告，在代码中实现上述 P0/P1 修复，并同步更新 `dki_plugin模块说明书_v3_3.md` 中相关章节，保证“实现与文档”再次保持严格一致。 

