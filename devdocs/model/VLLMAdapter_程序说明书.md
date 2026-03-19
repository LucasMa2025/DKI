### VLLMAdapter 程序说明书（`dki/models/vllm_adapter.py`）

#### 1. 模块定位与设计目标

-   **模块角色**：`VLLMAdapter` 是 DKI 系统中基于 vLLM 引擎的模型适配器，实现与 `BaseModelAdapter` 统一的接口，用于高性能推理和偏好注入实验。
-   **设计目标**：
    -   只使用 vLLM 作为推理引擎，不再加载 HuggingFace 模型，避免“双份模型”导致的显存浪费。
    -   通过「偏好/历史文本作为 prompt 前缀 + vLLM `enable_prefix_caching`」实现**原生 KV 注入与复用**，在语义上对齐 DKI 论文中的 KV 注入。
    -   保留旧版接口签名（如 `forward_with_kv_injection`、`injection_mode`），为历史代码提供安全降级。

#### 2. 主要类与结构

-   **类名**：`VLLMAdapter`
-   **继承关系**：`VLLMAdapter(BaseModelAdapter)`
-   **核心属性**：
    -   `model_name`: 模型名称或路径，传给 vLLM。
    -   `tensor_parallel_size`: 张量并行大小。
    -   `max_model_len`: 最大上下文长度。
    -   `gpu_memory_utilization`: GPU 显存利用率。
    -   `quantization` / `quantization_config`: 量化模式及其配置（gptq/awq/fp8/bitsandbytes 等）。
    -   `model_impl`: vLLM 的后端选择（`"auto"` 或 `"transformers"`）。
    -   `injection_mode`: 对外暴露为 `"prompt_prefix"`，用于让 Executor 判断这是“前缀注入模式”。
    -   `llm`: vLLM 的 `LLM` 实例。
    -   `sampling_params`: 默认采样配置（温度、top_p、max_tokens、stop 等）。
    -   `tokenizer`: `AutoTokenizer`，负责 chat 模板与 stop 字符串。

#### 3. 初始化与加载流程

-   **构造函数 `__init__`**：
    -   接收模型、并行、上下文、量化、后端等参数。
    -   对 `injection_mode` 做参数校验，兼容 `"auto"|"prompt_prefix"|"hf_kv"|"vllm_kv"`，但统一内部存储为 `"prompt_prefix"`，以兼容 Executor 的 `_is_prompt_prefix_mode()` 检查。
-   **`load()` 方法**：
    -   在导入 vLLM / transformers 之前调用 `ensure_hf_compat()`，修复 `huggingface_hub` ≥0.25 移除 `is_offline_mode` 导致的兼容性问题。
    -   构造 vLLM `LLM`：
        -   关键参数：
            -   `model`: `self.model_name`
            -   `tensor_parallel_size`: `self.tensor_parallel_size`
            -   `max_model_len`: `self.max_model_len`
            -   `gpu_memory_utilization`: `self.gpu_memory_utilization`
            -   `trust_remote_code`: `self.trust_remote_code`
            -   `enable_prefix_caching=True`（**核心**：自动复用相同前缀的 KV Cache）
            -   可选 `model_impl`（当需要 Transformers backend 时）
        -   根据 `quantization` 映射到 vLLM 支持的量化模式，例如：
            -   `"gptq"` → `"gptq"`，并强制 `dtype="float16"` 避免 bfloat16 不兼容。
            -   `"awq"` → `"awq"`。
            -   `"4bit"/"8bit"` → `"bitsandbytes"`。
            -   `"fp8"` → `"fp8"` 并结合 `quantization_config` 调整 `dtype`、`kv_cache_dtype` 等。
    -   加载 `AutoTokenizer`：
        -   设置 `pad_token = eos_token`（避免 padding 相关错误）。
    -   构造默认 `SamplingParams`：
        -   `temperature=0.7`、`top_p=0.9`、`max_tokens=512`。
        -   `stop` 由 `_get_stop_strings()` 计算。
    -   通过 `AutoConfig` 读取并缓存模型结构信息（hidden_size、num_layers、num_heads 等）。

#### 4. Prompt 处理与 stop 规则

-   **`_has_chat_template_tokens(text: str)`**：
    -   判断 prompt 是否已包括 Chat 模板标记，避免二次包装。
    -   支持识别：
        -   ChatML 标记：`<|im_start|>`；
        -   DeepSeek V2/V3 标记：全角竖线形式；
        -   Llama 3 标记：`<|begin_of_text|>`、`<|start_header_id|>`；
        -   Llama 2 标记：`[INST]`。
-   **`_format_prompt(prompt, system_prompt=None)`**：
    -   构造 messages 列表：`[{"role": "system", ...}, {"role": "user", ...}]`。
    -   优先调用 `tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)`。
    -   失败时回退到 ChatML 模板：
        -   `system` / `user` / `assistant` 段使用 `<|im_start|>role ... <|im_end|>`。
-   **`_is_chat_model()`**：
    -   根据模型名称中是否包含 `"chat"` 或 `"instruct"` 判断。
-   **`_get_stop_strings()`**：
    -   针对不同模型设置合理的 stop：
        -   DeepSeek/Qwen：`<|im_end|>`;
        -   Llama 3：`<|eot_id|>`;
        -   通用：tokenizer 的 `eos_token`；
    -   至少保证一个 stop 为 `<|im_end|>`，避免模型在多轮 ChatML 模板下无限生成。

#### 5. 核心推理接口

##### 5.1 `generate(...) → ModelOutput`

-   **用途**：标准生成接口，被 Executor、DKIPlugin 在无 KV 显式注入或熵门控探测阶段广泛调用。
-   **主要步骤**：
    1. 若尚未加载模型，调用 `load()`。
    2. 根据 `_has_chat_template_tokens` 与 `_is_chat_model` 决定是否需要通过 `_format_prompt` 包装成 Chat 模板。
    3. 提取 `kwargs.get("logprobs")`（如有），并在构造 `SamplingParams` 时传入 `logprobs` 字段，启用 vLLM 的 logprobs 输出能力，以支持熵门控。
    4. 调用 `self.llm.generate([formatted_prompt], sampling_params)`。
    5. 解析首个输出：
        - 文本：`output.outputs[0].text`；
        - token IDs：`output.outputs[0].token_ids`；
        - 通过 `_parse_vllm_logprobs` 将 `output.outputs[0].logprobs` 转为 `List[List[float]]`。
    6. 计算延迟与 token 统计，封装为 `ModelOutput`：
        - `text`：生成文本；
        - `tokens`：生成 token 序列；
        - `logprobs`：可选的 log 概率矩阵；
        - `latency_ms`：推理耗时；
        - `input_tokens` / `output_tokens`。

##### 5.2 `forward_with_kv_injection(...) → ModelOutput`

-   **接口语义**：
    -   对外保持与 HF KV 注入适配器一致的签名，以方便旧代码复用；
    -   在 vLLM 模式下，**真正的 KV 注入是通过 prompt 前缀 + prefix caching 实现的**，不再使用显式 `KVCacheEntry`。
-   **行为特点**：
    -   `prompt` 已由上游 Executor 拼接完“偏好前缀 + 历史后缀 + 用户问题”，包含所有注入语义。
    -   `injected_kv` 参数仅为兼容保留，不参与计算。
    -   内部逻辑与 `generate()` 高度一致：使用 chat 模板包装 + vLLM generate。
    -   `alpha` 对注入强度的影响在 Executor 的前缀构造阶段体现（例如偏好文本长度、显式权重提示等），vLLM 端只负责把前缀转为 KV Cache 并复用。

#### 6. 异步与流式接口

-   **`async_generate(...)`**：
    -   由于 vLLM 的 `LLM.generate()` 不依赖 asyncio 事件循环，而是采用多进程 + 阻塞通信，适配器通过：
        -   `loop.run_in_executor(None, lambda: self.generate(...))`
    -   在 FastAPI/uvicorn 等 async 框架中使用时不会引起 event loop 冲突。
-   **`async_stream_generate(...)` / `stream_generate(...)`**：
    -   vLLM 离线模式下不提供原生 token 级流式，因此采用“**完整生成 + 人为分段**”策略：
        -   先调用 `self.llm.generate` 获得完整文本；
        -   再按固定字符块（例如 4 字符）迭代 yield，模拟流式输出；
    -   适合作为 SSE/前端逐步展示的简易方案，但不是严格的 per-token streaming。

#### 7. 安全降级与受限能力

-   **`embed()`**：
    -   vLLM 原生不提供 embedding 接口，调用时直接抛出 `RuntimeError`，提示使用独立 embedding 服务（如 sentence-transformers）。
-   **`compute_kv()`**：
    -   在 vLLM-native 模式下不需要显式 `compute_kv`，偏好 KV 由 prefill 自然生成并缓存；
    -   函数返回 `([], None)`，并记录 debug 日志，作为防御性降级。
-   **`compute_prefill_entropy()`**：
    -   vLLM 不暴露 attention 权重，无法直接计算熵；
    -   返回固定中等值 `0.5` 作为门控的保底值。

#### 8. 模型信息与诊断

-   **`get_model_info()`**：
    -   在 `BaseModelAdapter.get_model_info()` 的基础上增加：
        -   `injection_mode = "prompt_prefix"`；
        -   `effective_injection_mode = "prompt_prefix"`；
        -   `vllm_native_kv = True`；
        -   `prefix_caching_enabled = True`；
        -   `hf_model_loaded = False`（v5.0 起不再加载 HF 模型）；
        -   `vllm_engine_loaded`、`quantization`、`model_impl` 等。
    -   用于在日志或调试接口中快速确认当前适配器的运行模式与引擎配置。

#### 9. 与其他适配器的对比（摘要）

-   与 `LlamaAdapter`（HF KV 注入）相比：
    -   `VLLMAdapter` 不持有 HF 模型与显式 KVCacheEntry，不支持内部 attention/熵等细粒度研究；
    -   优势在于 **高吞吐、高并发、显存利用更优**，适合在线服务和实验系统的大规模运行。
-   与 `SGLangAdapter` 相比：
    -   二者接口设计高度对称，差异主要在底层引擎（PagedAttention vs RadixAttention）、量化实现和事件循环处理逻辑；
    -   在 DKI 配置中可以通过切换适配器完成「vLLM ↔ SGLang」的对比试验。
