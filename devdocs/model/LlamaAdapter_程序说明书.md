### LlamaAdapter 程序说明书（`dki/models/llama_adapter.py`）

#### 1. 模块定位与设计目标

- **模块角色**：`LlamaAdapter` 是 DKI 系统中基于 HuggingFace Transformers 的 LLaMA 系列模型适配器，提供“真正的 K/V 注入”能力，是 DKI 论文级实验的核心实现。
- **设计目标**：
  - 完整实现 DKI 论文 §4.2 中的偏好 K/V 注入公式，包括 Value 缩放与 attention bias。
  - 兼容 Transformers 4.x/5.x 的 `DynamicCache`，在 past_key_values 结构变化下仍能稳定抽取和重建 KV。
  - 提供基础版 KV 注入（生成 API 友好）和带 attention bias 的完整公式版（研究用途）。

#### 2. 主要类与结构

- **类名**：`LlamaAdapter`
- **继承关系**：`LlamaAdapter(BaseModelAdapter)`
- **关键常量**：
  - `ALPHA_OVERRIDE_CAP = 0.7`：注入强度 α 的安全上限，防止 attention 极端偏置。
  - `MAX_PREF_TOKENS = 200`：偏好文本最大 token 数，超出将截断以控制 OOD 风险。
  - `DEFAULT_ENTROPY = 0.5`：prefill 熵计算失败时的默认熵值。

- **核心属性**：
  - `model_name`、`device`、`dtype`：HF 模型与设备配置。
  - `quantization` / `quantization_config` / `load_in_8bit`：量化方式与向后兼容参数。
  - `tokenizer`: `AutoTokenizer`，负责 Chat 模板与 token 处理。
  - `model`: `AutoModelForCausalLM`，支持输出 attentions、hidden states、past_key_values。
  - 结构信息：`hidden_dim`、`num_layers`、`num_heads`、`head_dim`。

- **DynamicCache 相关**：
  - 使用 `extract_kv_from_past` 与 `build_dynamic_cache_from_entries`（来自 `dki.models.base`）统一处理：
    - 4.x 时代 tuple 形式的 past_key_values；
    - 5.x 时代的 `DynamicCache` 结构。

#### 3. 模型加载与量化策略

- **`load()` 方法**：
  1. 通过 `AutoTokenizer.from_pretrained(model_name)` 加载 tokenizer，并设置 `pad_token = eos_token`。
  2. 构造 `model_kwargs`：
     - `trust_remote_code=True`，支持部分厂商自定义实现；
     - `device_map="auto"`，按 GPU 自动分布模型；
     - `attn_implementation="eager"`，确保支持：
       - `output_attentions=True`（prefill 熵计算）；
       - 4D attention mask（浮点 bias）注入。
  3. 根据 `quantization` 执行不同策略：
     - `"4bit"/"8bit"`：通过 BitsAndBytesConfig 配置 bitsandbytes 量化，4bit 模式下使用 `dtype` 作为非量化层精度；
     - `"gptq"/"awq"`：加载预量化模型，通常使用 `float16`；
     - `"fp8"`：配置 `torch_dtype` 为 `bfloat16` 或其他 FP8 相关 dtype（实验性特性，生产建议使用 vLLM/SGLang FP8）；
     - `"none"`：使用 `dtype` 作为全模型精度。
  4. 通过 `AutoModelForCausalLM.from_pretrained` 加载模型并 `.eval()`；
  5. 读取 `AutoConfig` 获取结构信息。

#### 4. Chat 模板与 stop token 处理

- **`_is_chat_model()`**：
  - 检测模型名称中是否包含 `"chat"` 或 `"instruct"`。

- **`_is_llama3()`**：
  - 通过模型名称中是否包含 `llama-3`、`llama3` 等标记识别 Llama 3.x 系列。

- **`_format_prompt(prompt, system_prompt=None)`**：
  - 构造 `messages` 列表（system + user），优先使用：
    - `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`；
  - 若模板调用失败，按官方规范手动构造：
    - Llama 3.x：
      - 使用 `<|begin_of_text|>`、`<|start_header_id|>`、`<|eot_id|>` 等标记；
    - Llama 2 Chat：
      - 使用 `[INST]` / `<<SYS>>` / `<</SYS>>` 模板。

- **`_has_chat_template_tokens(text)`**：
  - 检测文本中是否包含：
    - Llama 3 标记：`<|begin_of_text|>`、`<|start_header_id|>`；
    - Llama 2 标记：`[INST]`；
    - DeepSeek/Qwen 的 ChatML 标记：`<|im_start|>` / 全角竖线形式。

- **`_format_prompt_safe(prompt)`**：
  - 如果 prompt 已包含 chat 模板标记，直接返回；
  - 若模型为 Chat 型，则调用 `_format_prompt`；
  - 否则原样返回。

- **`_get_stop_token_ids()`**：
  - 为 Llama3 添加 `<|eot_id|>` 对应的 token id；
  - 始终包含 `eos_token_id`。

#### 5. 标准推理与 Streaming

- **`generate(prompt, ...) → ModelOutput`**：
  - 使用 `_format_prompt_safe` 格式化 prompt；
  - 调用 `model.generate(input_ids, attention_mask, max_new_tokens, temperature, top_p, pad_token_id)`；
  - 从生成结果中截取新生成 tokens（跳过输入长度），解码为文本；
  - 返回 `ModelOutput(text, tokens, latency_ms, input_tokens, output_tokens)`。

- **`async_stream_generate(...)` / `stream_generate(...)`**：
  - 使用 `transformers.TextIteratorStreamer` 实现真正的 token 级流式：
    - 后台线程中运行 `model.generate(..., streamer=TextIteratorStreamer(...))`；
    - 前台（async 或 sync）逐 token/文本片段从 streamer 中读取并 yield；
  - 支持在 FastAPI/uvicorn 下通过协程流式返回生成内容。

#### 6. 偏好 KV 计算：`compute_kv(text, return_hidden=False)`

- **目标**：将偏好文本编码为每层一组 K/V，用于后续 KV 注入。

- **流程**：
  1. 调用 `tokenize(text)` 获取 `input_ids` / `attention_mask`。
  2. 若 `n_tokens > MAX_PREF_TOKENS`：
     - 记录警告日志；
     - 截断到 `MAX_PREF_TOKENS`，并同步截断 `attention_mask`。
  3. `model(input_ids, attention_mask, output_hidden_states=return_hidden, use_cache=True, return_dict=True)`：
     - 得到 `outputs.past_key_values` 和可选的 `hidden_states`。
  4. 使用 `extract_kv_from_past(outputs.past_key_values)`：
     - 跨 Transformers 4.x/5.x，将过去值统一抽象为 `(key_tensor, value_tensor)` 的列表；
  5. 对于每一层：
     - 构造 `KVCacheEntry(key=key.detach().cpu(), value=value.detach().cpu(), layer_idx=idx)`；
     - 移到 CPU，避免 GPU 内存泄露。
  6. 若 `return_hidden=True` 且 `outputs.hidden_states` 不为 None：
     - 返回最后一层 hidden states 的 CPU 拷贝。
  7. 显式删除 `outputs`，并在 CUDA 可用时调用 `torch.cuda.empty_cache()`。

- **返回值**：
  - `kv_entries: List[KVCacheEntry]`；
  - `hidden_states: Optional[torch.Tensor]`。

#### 7. K/V 注入推理（基础版）：`forward_with_kv_injection(...)`

- **设计语义**：
  - 实现 DKI 论文中的“负位置 KV 注入”：
    - K_aug = [K_p; K_u]；
    - V_aug = [α·V_p; V_u]；
  - 使用 `model.generate`，在 HF 官方支持的 KV 机制下工作，便于与普通推理 API 对齐。

- **公开接口**：
  - `prompt`: 已包含 recall_v4 组装的历史后缀与用户问题；
  - `injected_kv`: 来自 `compute_kv` 的偏好 KV 列表；
  - `alpha`: 注入强度 [0,1]；
  - `max_new_tokens`: 最大生成长度。

- **逻辑步骤**：
  1. 在入口处裁剪 α：
     - `clamped_alpha = min(max(alpha, 0.0), ALPHA_OVERRIDE_CAP)`。
  2. 若 `injected_kv` 为空或 `clamped_alpha <= 0.01`：
     - 记录 debug 日志；
     - 退化为标准 `generate()` 调用。
  3. 否则调用 `_forward_with_kv_injection_impl(...)`：
     - 使用 `_format_prompt_safe` 格式化 prompt；
     - `tokenize` 得到 `input_ids / attention_mask`；
     - 使用 `_prepare_kv_for_injection(injected_kv, clamped_alpha, device)`：
       - 将 KVCacheEntry 移至 GPU，Value 按 α 缩放；
       - 调用 `build_dynamic_cache_from_entries` 构建适配当前 Transformers 版本的 `past_key_values`；
       - 返回 KV cache 与 `mem_len`（偏好 token 数）。
     - 构造扩展 `attention_mask`：
       - 在原有 mask 前拼接 `mem_len` 长度的全 1 区域，使偏好 KV 对所有查询可见；
     - 调用：
       - `model.generate(input_ids, attention_mask=extended_mask, past_key_values=past_kv, ...)`；
       - 模型内部会将输入 tokens 的 position 从 `mem_len` 开始，使偏好 KV 占据“负位置”。
     - 从完整序列中截取新生成 tokens（跳过输入长度），解码文本。
  4. 返回 `ModelOutput`，并在 `metadata` 中记录：
     - `alpha`、`alpha_clamped`；
     - `mem_len`；
     - `injection_mode="hf_kv_negative_position"`。

#### 8. K/V 注入推理（带 Attention Bias）：`forward_with_kv_injection_and_bias(...)`

- **目标**：实现 DKI 论文 §4.2 的完整公式，引入显式的 attention bias `B_alpha`：
  - `Attn_DKI = Softmax(Q[K_p;K_u]^T / √d + B_alpha) [α·V_p; V_u]`。

- **关键组件**：
  - `_build_attention_bias(n_pref, n_query, alpha, device, dtype)`：
    - 创建 `[1,1,n_query,n_pref+n_query]` 的 4D bias tensor：
      - 对偏好位置施加 `log(alpha)` 或 `-inf`（α=0 时完全屏蔽偏好）；
      - 对查询 token 施加因果遮罩（只能看到之前的 token）。
  - `_get_cache_seq_length(cache)`：
    - 统一从 `DynamicCache` 或 tuple 结构中获取 KV cache 长度，用于 decode 阶段 mask 构造。

- **实现流程（`_forward_with_bias_impl`）**：
  1. 格式化 prompt 并 tokenize；
  2. 使用 `_prepare_kv_for_injection` 得到 `past_kv` 与 `mem_len`；
  3. 构造 attention bias：
     - 调用 `_build_attention_bias(n_pref=mem_len, n_query=input_len, alpha, device, dtype)`；
  4. Prefill 阶段：
     - `model(input_ids, past_key_values=past_kv, attention_mask=attention_bias, use_cache=True)`；
     - 得到包含偏好 + 查询的完整 KV cache `full_kv` 以及最后一 token 的 logits；
  5. Decode 阶段（逐 token 自回归）：
     - 使用 top-p 采样策略，从 `next_logits` 中采样下一 token；
     - 对每一个新 token：
       - 构造 decode 阶段 4D mask（所有 past tokens 可见，偏好部分继续施加 `log(alpha)` bias）；
       - 调用 `model(next_input, past_key_values=full_kv, attention_mask=decode_mask, use_cache=True)` 更新 KV cache；
       - 直到遇到 stop token（`eos_token` 或 `_get_stop_token_ids()` 中的任一标记）。
  6. 解码生成 tokens，释放 KV cache 和中间张量（必要时清理 CUDA cache）。
  7. 返回 `ModelOutput`，并在 `metadata` 中标记：
     - `injection_mode="hf_kv_attention_bias"`；
     - `attention_bias_applied=True`。

- **适用场景**：
  - 需要严谨复现论文公式、精细控制偏好影响强度的研究实验；
  - 相对于基础版 KV 注入，性能略差但行为与数学抽象更接近。

#### 9. Prefill 熵计算：`compute_prefill_entropy(text, layer_idx=3)`

- **用途**：计算预填阶段某一层 attention 的平均熵，作为 alpha 门控（熵门控）的信号。

- **实现步骤**：
  1. `tokenize(text)`；
  2. 调用 `model(..., output_attentions=True, return_dict=True)`；
  3. 选择第 `layer_idx` 层的 attention（若索引过大则自动裁剪到最后一层）；
  4. 对每一行 attention 分布计算熵：`-Σ p log p`；
  5. 在 batch/head/seq 维度上平均，得到标量熵值；
  6. 若任一步失败或 `outputs.attentions` 为空，则返回 `DEFAULT_ENTROPY` 并记录 warning。

#### 10. Embedding 与模型信息

- **`embed(text)`**：
  - 使用 `model(..., output_hidden_states=True)` 获取最后一层 hidden states；
  - 在序列维度上取平均，得到文本 embedding 向量；
  - 将结果迁移到 CPU，并清理中间张量和 CUDA cache。

- **`get_model_info()`**：
  - 在基类信息基础上增加：
    - `adapter_type="llama_hf_kv_injection"`；
    - `injection_mode="hf_kv"`（Executor 因此走 HF KV 注入路径，而非 prompt_prefix 模式）；
    - `kv_injection_type="negative_position"`；
    - `alpha_override_cap`、`max_pref_tokens`；
    - `attention_bias_available=True`；
    - 量化信息与 `load_in_8bit` 状态。

#### 11. 与其他适配器的差异概览

- 与 `VLLMAdapter` / `SGLangAdapter` 相比：
  - `LlamaAdapter` 持有完整 HF 模型与显式 K/V，提供最强的**可解释性与可控性**；
  - 可以访问 attention / hidden states，支持熵门控、精细 bias 注入、embedding 等研究功能；
  - 推理性能与显存占用不如 vLLM/SGLang 适合大规模在线服务，更适合作为 DKI 论文复现与实验系统的标准后端。

