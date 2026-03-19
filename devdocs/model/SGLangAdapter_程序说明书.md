### SGLangAdapter 程序说明书（`dki/models/sglang_adapter.py`）

#### 1. 模块定位与设计目标

- **模块角色**：`SGLangAdapter` 是 DKI 系统中基于 SGLang 引擎的模型适配器，与 `VLLMAdapter` 对称，负责将 DKI 的统一调用接口映射到 SGLang 的 Engine。
- **设计目标**：
  - 利用 SGLang 的 **RadixAttention** 实现高效的前缀 KV 复用，特别适配 Qwen3.5 等新架构。
  - 在多版本 SGLang / transformers / huggingface-hub 环境下提供**稳健的兼容性与错误诊断**。
  - 与 `BaseModelAdapter`、`VLLMAdapter` 保持接口一致，方便在配置层面切换引擎进行对比实验。

#### 2. 主要类与结构

- **类名**：`SGLangAdapter`
- **继承关系**：`SGLangAdapter(BaseModelAdapter)`
- **核心属性**：
  - 通用模型参数：
    - `model_name`: SGLang Engine 所使用的模型路径；
    - `tensor_parallel_size`: 张量并行大小；
    - `max_model_len`: 上下文长度（SGLang 的 `context_length`）；
    - `gpu_memory_utilization`: 高层语义，底层映射到 `mem_fraction_static`。
  - SGLang 特有参数：
    - `mem_fraction_static`: 静态内存比例，用于 CUDA graph 与工作内存预留；
    - `schedule_policy`: 调度策略（`"lpm"|"random"|"fcfs"`，推荐 `"lpm"` = Longest Prefix Match）；
    - `chunked_prefill_size`: 分块 prefill 尺寸，缓解首 token 延迟。
  - 量化配置：
    - `quantization`: `"none"|"gptq"|"awq"|"4bit"|"8bit"|"fp8"`；
    - `quantization_config`: 量化细节（如 dtype、Mamba SSM dtype 等）。
  - 引擎组件：
    - `engine`: SGLang `Engine` 实例；
    - `sampling_params`: 字典形式的采样参数；
  - 其他：
    - `injection_mode`: 统一设置为 `"prompt_prefix"`，用于 Executor 判断注入模式；
    - `_SGLANG_CORE_PARAMS`: 用于识别 Engine 核心参数的集合（`{"model_path","tp_size","mem_fraction_static"}`）。

#### 3. 初始化与加载流程

- **构造函数 `__init__`**：
  - 接收模型、并行、上下文、量化、调度等参数；
  - 对 `injection_mode` 做兼容性检查，接受 `"auto"|"prompt_prefix"|"hf_kv"|"vllm_kv"`，统一内部存储为 `"prompt_prefix"`；
  - 将 `gpu_memory_utilization` 映射给 SGLang 的 `mem_fraction_static`，保证显存控制语义一致。

- **`load()` 方法**：
  1. 调用 `ensure_hf_compat()` 修复 huggingface-hub 与 transformers 的版本兼容问题。
  2. 预加载 `AutoProcessor`：
     - 解决部分 SGLang 版本在 transformers ≥4.50 + 低版本 huggingface-hub 时，lazy import AutoProcessor 导致的错误；
     - 如检测到 huggingface-hub 版本过低，会给出明确的安装升级建议。
  3. 构造 `engine_kwargs`：
     - 基础参数：
       - `model_path`: 模型路径；
       - `tp_size`: 张量并行大小；
       - `context_length`: 最大上下文长度；
       - `mem_fraction_static`: 静态显存比例；
       - `trust_remote_code`: 是否信任远程代码；
       - `chunked_prefill_size`: 分块 prefill；
       - `schedule_policy`: 调度策略；
       - `cuda_graph_max_bs`: 限制 CUDA graph 的最大 batch size，减小额外显存占用。
     - 量化映射：
       - 使用 `sglang_quant_map` 将 `"gptq"|"awq"|"4bit"|"8bit"|"fp8"` 映射到：
         - `"gptq_marlin"/"awq_marlin"`：兼容 Qwen3.5 + Mamba 的高性能内核；
         - `"bitsandbytes"`：4bit/8bit 量化；
         - `"fp8"`：FP8 模式。
       - 不同量化模式下，针对 dtype / `mamba_ssm_dtype` / `kv_cache_dtype` / 后端选择进行细致配置，防止数值溢出与 NaN。
       - 典型策略：
         - GPTQ：强制 `dtype='float16'` / `mamba_ssm_dtype='float16'`；
         - AWQ/bitsandbytes：统一使用 `bfloat16`，避免 Mamba SSM 在 float16 下溢出产生 NaN；
         - FP8：启用支持 FP8 的 attention/mamba 后端，并优化显存利用与调度。
  4. 通过 `_filter_engine_kwargs()` 过滤不被当前 SGLang 版本支持的参数：
     - 优先从 `ServerArgs` 或 `Engine.__init__` 的签名中获取可接受参数集合；
     - 若检测到仅为“包装类”（不含核心参数），则跳过过滤，保留所有参数交由 SGLang 自行处理；
     - 在过滤过程中记录被丢弃参数及升级建议。
  5. 创建 SGLang `Engine` 实例：
     - 注意：Engine 必须在主线程中创建，因为内部会注册 signal handler；
     - 适配器内部假定这是在主线程环境中运行。
  6. 加载 `AutoTokenizer` 与 `AutoConfig`：
     - 设置 `pad_token = eos_token`；
     - 从 Config 中提取结构信息（hidden_size/num_layers/num_heads）。

#### 4. Prompt 处理与 stop 规则

- **`_has_chat_template_tokens(text)`**：
  - 判断文本是否已经包含 Chat 模板标记，避免再次包装：
    - ChatML：`<|im_start|>`；
    - Llama3：`<|begin_of_text|>`、`<|start_header_id|>`；
    - Llama2：`[INST]`；
    - DeepSeek/Qwen 特定标记。

- **`_format_prompt(prompt, system_prompt=None)`**：
  - 与 `VLLMAdapter` 类似，优先使用 `tokenizer.apply_chat_template`；
  - 失败时回退到 ChatML 模板。

- **`_is_chat_model()`**：
  - 通过模型名称包含 `"chat"` 或 `"instruct"` 判定。

- **`_get_stop_strings()`**：
  - 根据模型名称与 tokenizer 确定 stop 字符串列表：
    - DeepSeek/Qwen：`<|im_end|>`；
    - Llama3：`<|eot_id|>`；
    - 其余补充 `eos_token`；
  - 至少存在一个 `<|im_end|>` 作为兜底。

#### 5. Logprobs 解析与熵门控支持

- **`_parse_sglang_logprobs(meta_info: dict)`**：
  - 从 SGLang 输出的 `meta_info` 中提取 logprobs：
    - 优先使用 `"output_top_logprobs"`：`List[Dict[int, float]]`；
    - 若不存在则退回到 `"output_token_logprobs"`：`List[float]`；
  - 统一转换为 `List[List[float]]`（每个 token 的 top-k logprob 向量），以适配 DKI 的 `EntropyMonitor`。

- 通过在 `generate()` / `async_generate()` 中传入：
  - `return_logprob=True`；
  - `top_logprobs_num=k`；
  - 实现对熵门控路径 `_execute_entropy_gated` 的支持。

#### 6. 引擎调用与事件循环安全

- **`_call_engine_generate(prompt, sampling_params)`**（同步）：
  - 直接调用 `self.engine.generate(prompt, sampling_params)`。
  - 若内部因 `self.loop.run_until_complete()` 与外部 event loop 冲突抛出 `"event loop is already running"` 等错误：
    - 尝试通过 `asyncio.run_coroutine_threadsafe(engine.async_generate(...), engine.loop)` 把请求提交到 Engine 的事件循环；
    - 否则创建新的 event loop 来运行 `engine.async_generate`。

- **`_call_engine_generate_async(prompt, sampling_params)`**（异步）：
  - 首选路径：如果 Engine 提供 `async_generate`，直接 `await engine.async_generate(...)`，完全避免 `run_until_complete` 嵌套问题；
  - 否则回退为在线程池中调用同步 `engine.generate`，并给出升级 SGLang 版本的提示。

#### 7. 输出解析与 NaN 诊断

- **`_parse_engine_output(output)`**：
  - 兼容多种返回格式：
    - `dict`：`{"text": "...", "meta_info": {...}}`；
    - `list[dict]`：批量输出，取首元素；
    - 对象（`GenerateOutput/RequestOutput` 等）：尝试 `.text/.output_text/.outputs[0].text` 等路径；
    - `str`：直接返回文本；
    - 其他：尝试 `output["text"]`，最终回退到 `str(output)`。
  - 对 `meta_info.finish_reason` 中 `"NaN happened"` 等提示进行专门检测与日志输出，给出：
    - NaN 发生原因（量化 + float16 + Mamba）；
    - 建议解决方案（使用 bfloat16、部署 dtype 对齐补丁等）。

- **`_extract_from_object(obj, label)`**：
  - 从结构化对象中提取 `text` 和 `meta_info`，用于兼容不同版本的输出类型。

#### 8. 核心推理与 KV 注入接口

- **`generate(prompt, ...) → ModelOutput`**：
  - 确保模型已加载；
  - 使用 chat 模板格式化 prompt；
  - 根据 `kwargs["logprobs"]` 决定是否在 `sampling_params` 中启用 logprobs 返回；
  - 调用 `_call_engine_generate` 获得原始输出与 `meta_info`；
  - 使用 `_parse_sglang_logprobs` 提取 logprobs；
  - 构造 `ModelOutput`：
    - `text`、`tokens`（`output_ids`）、`logprobs`、`latency_ms`、`input_tokens`、`output_tokens`。

- **`forward_with_kv_injection(prompt, injected_kv, alpha, ...) → ModelOutput`**：
  - 语义上与 `generate` 相同，但在 `metadata` 中增加：
    - `alpha`；
    - `injection_mode="sglang_native_radix_attention"`。
  - `injected_kv` 仅为保留签名，真正的注入由“偏好前缀 + RadixAttention 前缀复用”实现。

#### 9. 异步与流式推理接口

- **`async_generate(...)` / `async_forward_with_kv_injection(...)`**：
  - 使用 `_call_engine_generate_async` 调用 Engine；
  - 记录详细日志（延迟、输出长度、raw 输出类型），方便在线服务排障；
  - 返回 `ModelOutput`，结构与同步接口一致。

- **`async_stream_generate(...)`**：
  - 若 Engine 支持 streaming：
    - 调用 `engine.async_generate(formatted_prompt, sampling_params, stream=True)`；
    - 逐 chunk 解析输出并 yield；
  - 若不支持 stream 参数：
    - 完整生成后按小块模拟流式（与 vLLM 模拟流式类似）。

- **`stream_generate(...)`**：
  - 同步环境下调用 `generate()`，再按字符块切分文本 yield。

#### 10. 安全降级与模型信息

- **`embed()` / `compute_kv()` / `compute_prefill_entropy()`**：
  - 与 `VLLMAdapter` 一致，SGLang 原生不提供 embedding 与显式 KV/attention 访问：
    - `embed()` 抛出 `RuntimeError`；
    - `compute_kv()` 返回空列表与 `None`；
    - `compute_prefill_entropy()` 返回固定默认值 `0.5`。

- **`get_model_info()`**：
  - 在基类信息的基础上增加：
    - `engine="sglang"`；
    - `injection_mode="prompt_prefix"`；
    - `effective_injection_mode="prompt_prefix"`；
    - `sglang_native_kv=True`；
    - `radix_attention_enabled=True`；
    - `sglang_engine_loaded`、`quantization`、`schedule_policy`、`mem_fraction_static`、`chunked_prefill_size` 等。

#### 11. 与其他适配器的差异概览

- 相比 `VLLMAdapter`：
  - 引擎从 vLLM 换成 SGLang，KV 复用机制从 PagedAttention + prefix caching 换成 RadixAttention；
  - 模型支持上更偏向 Qwen3.5 等新架构，structured generation 支持更丰富；
  - 对事件循环与量化 NaN 问题做了大量额外处理，工程复杂度更高。

- 相比 `LlamaAdapter`：
  - `SGLangAdapter` 不持有 HF 模型，也不实现显式 K/V 注入，而是依赖引擎内部的 KV 管理；
  - 适合高并发在线服务和大规模实验，而非用于研究级的 attention/KV 内部可视化与精细控制。

