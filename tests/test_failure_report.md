# 当前测试失败/错误清单（基于最近一次 `pytest` 全量运行）

> 统计来源：`python -m pytest -q`，参见日志文件  
> `agent-tools/d33641d9-4d2b-4f3c-bb86-5890486ae290.txt`

本文按“测试文件 → 失败/错误用例 → 关联被测模块/类”汇总，方便后续分阶段修复或下线。

---

## 1. 顶层基础测试

- **tests/test_basic.py**
  - 失败用例：
    - `test_embedding_service` — ImportError: huggingface / huggingface-hub 相关依赖缺失
    - `test_memory_router`
  - 关联模块/类：
    - `dki.core.rag_system.EmbeddingService`
    - `dki.core.memory_router.MemoryRouter`（或类似命名的路由组件）

---

## 2. 实验系统（experiment）相关

- **tests/unit/experiment/test_experiment_updates.py**
  - 失败：
    - `TestDecomposedHallucination::test_fabricated_detail_with_dates`
  - 错误：
    - `TestRunContextConstrained::test_runner_has_run_context_constrained_method`
    - `TestAblationVariants::test_runner_has_ablation_method`
    - `TestAblationVariants::test_runner_has_alpha_sensitivity_method`
  - 关联模块/类：
    - `dki.experiment.runner.ExperimentRunner`
    - 以及与 *context_constrained* / *ablation* / *alpha_sensitivity* 相关的 Runner 方法

- **tests/unit/experiment/test_runner.py**
  - 失败：
    - `TestInjectionInfoViewer::test_save_comparison`
    - `TestInjectionInfoViewer::test_get_copyable_text`
    - `TestExperimentRunnerUtils::test_cache_key_signer_initialized`
  - 错误（大量）：
    - `TestExperimentRunnerUtils::test_extract_queries_from_query`
    - `...::test_extract_queries_from_question`
    - `...::test_extract_queries_from_turns`
    - `...::test_extract_queries_empty`
    - `...::test_extract_queries_priority`
    - `...::test_get_experiment_user_id_explicit`
    - `...::test_get_experiment_user_id_from_map`
    - `...::test_get_experiment_user_id_default`
    - `...::test_get_experiment_user_id_custom_default`
    - `...::test_get_default_experiment_users`
    - `...::test_get_default_experiment_users_from_config`
    - `...::test_match_user_by_personas_no_map`
    - `...::test_match_user_by_personas_with_map`
    - `...::test_aggregate_metrics`
    - `...::test_aggregate_metrics_empty`
    - `...::test_save_results`
    - `...::test_compute_mode_metrics_basic`
    - `...::test_compute_mode_metrics_with_errors`
    - `...::test_compute_mode_metrics_empty`
    - `...::test_ensure_systems_already_set`
    - `...::test_get_first_experiment_user_id_with_map`
    - `...::test_get_first_experiment_user_id_no_map`
    - `...::test_get_first_experiment_user_id_empty_map`
    - `...::test_write_session_preferences_calls_clear_cache`
    - `TestRunSession::test_run_session_short_dki`
    - `TestRunSession::test_run_session_long_rag`
    - `TestRunSession::test_run_session_injection_info`
    - `TestRunSession::test_run_session_recall_score`
    - `TestRunSession::test_run_session_error_handling`
    - `TestRunSession::test_run_session_empty_turns`
    - `TestRunSession::test_run_session_writes_preferences_for_dki`
    - `TestRunPersonaChatExperiment::test_persona_chat_loads_short_data`
    - `TestRunPersonaChatExperiment::test_persona_chat_result_structure`
  - 关联模块/类：
    - `dki.experiment.runner.ExperimentRunner`
    - `ExperimentRunnerUtils` 辅助方法（query 抽取、用户映射、metrics 聚合等）
    - 结果持久化与 session 运行逻辑

- **tests/unit/test_experiment_review_2026_02_18.py**
  - 失败：
    - `TestExperimentRunEndpoint::test_persona_chat_endpoint_also_passes_shared_systems`
    - `TestExperimentUseDKISystem::test_runner_stores_dki_system`
    - `...::test_runner_lazy_creates_systems`
    - `...::test_runner_calls_dki_chat`
    - `...::test_runner_calls_rag_chat`
  - 关联模块/类：
    - `dki.experiment.runner.ExperimentRunner`
    - 与 DKI / RAG 系统集成的实验入口（可能是 FastAPI 路由或 CLI）

- **tests/unit/test_v8_session_viz_fixes.py**
  - 失败：
    - `TestExperimentRunnerInit::test_init_with_dki_plugin`
  - 关联模块/类：
    - `ExperimentRunner` 与 `DKIPlugin` 的集成初始化逻辑

---

## 3. Recall / BM25 / 后缀构造相关

- **tests/unit/recall/test_suffix_builder.py**
  - 失败：
    - `TestSuffixBuilder::test_build_short_messages`
    - `...::test_build_long_message_fits_budget_keeps_full`
    - `...::test_build_long_message_generates_summary`
    - `...::test_build_mixed_messages_budget_sufficient`
    - `...::test_build_mixed_messages_budget_tight`
    - `...::test_custom_token_counter`
    - `...::test_extract_epistemic_markers_date`
    - `...::test_extract_epistemic_markers_price`
    - `...::test_extract_epistemic_markers_long_vs_short`
  - 关联模块/类：
    - `dki.core.recall.suffix_builder.SuffixBuilder`

- **tests/unit/test_bm25_chinese_recall.py**
  - 失败：
    - `TestBM25Tokenizer::test_bm25_score_returns_positive_for_relevant_messages`
    - `...::test_bm25_specific_keywords_match`
  - 关联模块/类：
    - 中文 BM25 分词与召回逻辑（通常在 `dki.core.recall` / `dki.adapters` 中）

- **tests/unit/test_history_paired_injection.py**
  - 失败：
    - `TestCollectMessagesTimestamp::test_collect_preserves_timestamp`
    - `...::test_collect_sorts_by_timestamp`
    - `...::test_collect_no_timestamp_still_works`
    - `TestPlannerFormatHistorySuffix::test_format_sorted_by_time`
    - `TestMergeRecentAndRecalled::test_merged_sorted_by_timestamp`
    - `...::test_deduplication`
  - 关联模块/类：
    - `dki.core.plugin.injection_planner` 中的历史消息收集 / 排序逻辑
    - `DKIPlugin._merge_recent_and_recalled`（合并近轮与 BM25 历史）

- **tests/unit/test_recent_messages_merge.py**
  - 失败：
    - `TestMergeRecentAndRecalled::test_recalled_only`
    - `...::test_dedup_by_message_id`
    - `...::test_recent_first_then_bm25`
    - `...::test_large_overlap`
  - 关联模块/类：
    - `dki.core.dki_plugin.DKIPlugin._merge_recent_and_recalled`
    - `DKIPlugin._remove_trailing_unpaired_user`

---

## 4. RAG 系统相关（v5.3 / v6）

- **tests/unit/test_v53_fixes.py**
  - 失败：
    - `TestRAGPreferenceInjection::test_load_user_preferences_returns_string`
    - `...::test_load_user_preferences_returns_none_for_no_prefs`
    - `...::test_chat_metadata_includes_preference_info`
  - 关联模块/类：
    - `dki.core.rag_system.RAGSystem._load_user_preferences`
    - `RAGSystem.chat` 的 metadata / preference 注入行为

- **tests/unit/test_rag_system_v6.py**
  - 失败：
    - `TestPreferenceCache::test_cache_hit`
    - `...::test_cache_miss_expired`
    - `...::test_invalidate_single_user`
    - `...::test_invalidate_all`
    - `TestAsyncPreferenceLoading::test_async_cache_hit`
    - `TestAsyncChat::test_async_chat_basic`
    - `...::test_async_chat_increments_stats`
    - `TestChatStream::test_stream_yields_metadata_first`
    - `...::test_stream_yields_tokens`
    - `...::test_stream_yields_done`
    - `TestGetStats::test_stats_include_preference_cache`
    - `TestSyncChat::test_sync_chat_basic`
  - 关联模块/类：
    - `dki.core.rag_system.RAGSystem`（偏好缓存、异步/同步 chat、stream、get_stats）

---

## 5. RAG / Prompt 模板一致性与修复

- **tests/unit/test_chat_template_consistency.py**
  - 失败：
    - `TestLlamaAdapterChatTemplate::test_double_wrapping_prevention_source`
    - `...::test_double_wrapping_prevention_forward`
    - `TestRAGSystemChatTemplate::test_build_prompt_with_tokenizer`
    - `...::test_build_prompt_fallback_without_tokenizer`
    - `...::test_build_prompt_with_history`
    - `TestCrossAdapterConsistency::test_generate_has_double_wrap_prevention[dki.models.llama_adapter-LlamaAdapter]`
    - `...::test_forward_has_double_wrap_prevention[dki.models.llama_adapter-LlamaAdapter]`
  - 关联模块/类：
    - `dki.models.llama_adapter.LlamaAdapter`
    - `dki.core.rag_system.RAGSystem._build_prompt` 及相关模板逻辑

- **tests/unit/test_prompt_template_fixes.py**
  - 失败：
    - `TestRAGSystemFallbackFormat::test_fallback_uses_chatml_halfwidth`
    - `...::test_fallback_with_history`
  - 关联模块/类：
    - RAG fallback prompt 模板实现（ChatML / 半角引号处理）

---

## 6. 模型适配器相关（GLM / VLLM / SGLang / Closed-source）

- **tests/unit/test_closed_source_adapter.py**
  - 失败：
    - `TestGenerate::test_sync_generate`
  - 关联模块/类：
    - 闭源模型适配器，实现文件位于 `dki.models.*`（具体需查看该测试文件 import）

- **tests/unit/test_models_review_2026_02_18.py**
  - 失败：
    - `TestGLMAdapterSpecifics::test_glm_handles_non_tuple_kv`
    - `TestVLLMAdapterSpecifics::test_vllm_prefix_caching_enabled`
  - 关联模块/类：
    - `dki.models.glm_adapter.GLMAdapter`
    - `dki.models.vllm_adapter.VLLMAdapter`

- **tests/unit/test_sglang_adapter.py**
  - 失败（大量）：
    - `TestSGLangAdapterInit::test_default_initialization`
    - `TestSGLangLoad::test_load_basic`
    - `...::test_load_with_gptq`
    - `...::test_load_gptq_forces_float16_dtype`
    - `...::test_load_awq_forces_bfloat16_dtype`
    - `...::test_load_4bit_forces_bfloat16_dtype`
    - `...::test_load_no_quant_no_dtype_override`
    - `...::test_load_with_awq`
    - `...::test_load_with_4bit`
    - `...::test_load_with_8bit`
    - `...::test_load_no_quantization`
    - `...::test_load_all_quant_types_force_dtype_and_mamba`
    - `...::test_load_sglang_specific_params`
    - `...::test_load_tensor_parallel`
    - `...::test_load_pad_token_auto_set`
    - `...::test_load_auto_processor_preimport`
    - `...::test_load_auto_processor_import_failure_non_blocking`
    - `...::test_load_auto_processor_failure_due_to_hf_hub_version`
    - `...::test_load_sglang_internal_autoprocessor_error`
    - `TestSGLangErrorHandling::test_load_huggingface_hub_error`
    - `...::test_load_autoprocessor_internal_error_diagnostics`
    - `TestSGLangNaNPrevention::test_load_basic`
    - `...::test_load_with_gptq`
    - `...::test_load_gptq_forces_float16_dtype`
    - `...::test_load_awq_forces_bfloat16_dtype`
    - `...::test_load_4bit_forces_bfloat16_dtype`
    - `...::test_load_no_quant_no_dtype_override`
    - `...::test_load_with_awq`
    - `...::test_load_with_4bit`
    - `...::test_load_with_8bit`
    - `...::test_load_no_quantization`
    - `...::test_load_all_quant_types_force_dtype_and_mamba`
    - `...::test_load_sglang_specific_params`
    - `...::test_load_tensor_parallel`
    - `...::test_load_pad_token_auto_set`
    - `...::test_load_auto_processor_preimport`
    - `...::test_load_auto_processor_import_failure_non_blocking`
    - `...::test_load_auto_processor_failure_due_to_hf_hub_version`
    - `...::test_load_sglang_internal_autoprocessor_error`
    - `...::test_quant_does_not_enable_fp32_lm_head`
    - `...::test_no_quant_no_fp32_lm_head`
    - `...::test_quant_sets_cuda_graph_max_bs`
  - 关联模块/类：
    - `dki.models.sglang_adapter.SGLangAdapter` 及其加载/量化/容错逻辑

- **tests/unit/test_quantization.py**
  - 失败：
    - `TestQuantizationNormalization::test_normalize_unknown_fallback`
  - 关联模块/类：
    - 量化配置规范化工具（通常在 `dki.models.*` 或单独 util 模块）

---

## 7. DKI Plugin / 历史注入补充（旧审查用例）

- **tests/unit/test_plugin_review_2026_02_18.py**
  - 失败：
    - `TestPlannerHistoryFiltering::test_filters_ai_assistant_marker`
    - `...::test_filters_session_history_marker`
    - `...::test_keeps_clean_assistant_messages`
    - `...::test_english_filtering`
    - `TestStableFallbackLogic::test_stable_fallback_includes_history`
    - `...::test_stable_fallback_no_history`
  - 关联模块/类：
    - 历史过滤与 stable fallback 逻辑，涉及：
      - `dki.core.plugin.injection_planner`
      - `dki.core.dki_plugin.DKIPlugin._fallback_stable_then_none`

---

## 8. 其它配置 / Demo / API 相关

- **tests/unit/test_llama_engine_config.py**
  - 失败：
    - `TestConfigYamlHasLlamaEngine::test_config_env_yaml_llama_uses_env_vars`
    - `TestStartScriptLlamaAlignment::test_start_script_has_cuda_for_llama`
  - 关联模块/文件：
    - `config/config.yaml` 中 engine 配置
    - 启动脚本（例如 `start_llama_alignment.sh` 或 Python 入口）

- **tests/unit/test_max_new_tokens_config.py**
  - 失败：
    - `TestDemoAPIChatRequest::test_chat_request_default_max_tokens`
    - `...::test_chat_request_max_tokens_range`
  - 关联模块/类：
    - demo 系统的 API 请求模型（通常在 `demo/api/chat.py` 中的 `ChatSendRequest`）

---

## 9. 总结与后续建议

- 当前仍有 **127 个失败 + 36 个错误**，主要集中在：
  - 实验与评测系统（ExperimentRunner + 各类实验用例）
  - RAG 系统（RAGSystem v5.3 / v6、BM25 召回、后缀构造）
  - 各类模型适配器（SGLang / GLM / VLLM / 闭源适配器）、量化工具
  - 部分早期审查用例和 demo API 配置
- 建议按子系统分阶段处理（例如先修 RAGSystem v6，再修 ExperimentRunner，再看 SGLangAdapter），避免一次性改动过大导致新回归。

