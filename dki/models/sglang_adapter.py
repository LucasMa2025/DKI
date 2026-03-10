"""
SGLang Model Adapter for DKI System

High-performance inference with SGLang engine.
SGLang 原生支持 Qwen3.5 等新架构, 无需 model_impl="transformers" 回退。

核心设计:
- 与 VLLMAdapter 完全对称: 100% SGLang-only, 无 HF 模型
- 偏好/历史文本作为 prompt 前缀, 由 SGLang prefill 阶段自然计算 KV
- SGLang RadixAttention 自动复用相同前缀的 KV Cache (零额外代码)
- 这就是 DKI 论文的 KV 注入: 偏好信息的 KV 表示通过 attention 机制影响后续推理

SGLang vs vLLM:
- SGLang RadixAttention: 基于基数树的前缀复用, 更细粒度的 KV Cache 管理
- 原生支持 Qwen3.5 等新架构 (无需 transformers backend 回退)
- 更好的 structured generation 支持 (JSON, regex 等)
- 与 vLLM 类似的 PagedAttention 内存管理

接口兼容性:
- 保留 BaseModelAdapter 所有抽象方法签名
- compute_kv / embed / compute_prefill_entropy 返回安全降级值
- forward_with_kv_injection 保留签名, 内部统一走 SGLang generate
- injection_mode 参数保留, 但所有值最终都路由到 SGLang 原生推理

安全不变量:
- 默认且唯一的推理引擎是 SGLang
- 偏好注入通过 prompt 前缀 + SGLang RadixAttention 实现
- 所有公开接口签名与 BaseModelAdapter 完全兼容
- 异常时自动降级到无注入推理 (fail-open)

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class SGLangAdapter(BaseModelAdapter):
    """
    SGLang-based model adapter with native KV injection via RadixAttention.
    
    核心: 100% SGLang-only, 无 HF 模型, 偏好通过 prompt 前缀注入,
    SGLang RadixAttention 自动复用相同前缀的 KV Cache。
    
    与 DKI 论文一致:
    - 偏好 KV 注入 = 偏好文本作为 prefix → SGLang prefill → attention 层注入
    - KV Cache 复用 = SGLang RadixAttention (相同偏好前缀自动复用)
    - 无需 HF 模型计算 KV, 无 ~14GB VRAM 浪费
    
    SGLang 优势 (相比 vLLM):
    - RadixAttention: 基数树 KV Cache 管理, 更高效的前缀复用
    - 原生支持 Qwen3.5 等新架构, 无需 model_impl="transformers" 回退
    - 更好的 structured generation 支持
    
    Usage:
        # 标准用法 (推荐)
        adapter = SGLangAdapter(model_name="Qwen/Qwen3.5-27B-Instruct")
        adapter.load()
        output = adapter.generate("你好")
        
        # 带偏好 KV 注入
        output = adapter.forward_with_kv_injection(
            prompt="用户偏好前缀\\n\\n你好",
            injected_kv=[],   # 不再使用, 保留签名兼容
            alpha=0.7,
        )
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        device: str = "cuda",
        injection_mode: str = "auto",  # 保留参数兼容, 但不影响行为
        quantization: str = "none",
        quantization_config: dict = None,
        # SGLang 特有参数
        mem_fraction_static: float = 0.80,
        schedule_policy: str = "lpm",  # lpm = Longest Prefix Match
        chunked_prefill_size: int = 4096,
        **kwargs
    ):
        """
        初始化 SGLang 适配器
        
        Args:
            model_name: 模型名称或路径
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度 (SGLang 中为 context_length)
            gpu_memory_utilization: GPU 显存利用率 (映射到 mem_fraction_static)
            trust_remote_code: 是否信任远程代码
            device: 设备
            injection_mode: 偏好注入模式 (保留兼容, 所有值最终走 SGLang 原生推理)
            quantization: 量化模式
                - "none" (默认): 不量化
                - "gptq": GPTQ 量化
                - "awq": AWQ 量化
                - "fp8": FP8 量化 (E4M3 格式, 需要 Ada Lovelace+ GPU)
                - "4bit" / "8bit": bitsandbytes 量化
            quantization_config: 量化详细配置
            mem_fraction_static: SGLang 静态内存比例 (默认 0.80, 为 CUDA graph 和临时计算留足空间)
            schedule_policy: 调度策略
                - "lpm": Longest Prefix Match (推荐, 最大化前缀复用)
                - "random": 随机调度
                - "fcfs": 先来先服务
            chunked_prefill_size: 分块预填充大小 (减少首 token 延迟)
        """
        super().__init__(
            model_name, device,
            quantization=quantization,
            quantization_config=quantization_config,
            **kwargs
        )
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.mem_fraction_static = mem_fraction_static
        self.schedule_policy = schedule_policy
        self.chunked_prefill_size = chunked_prefill_size
        
        # 注入模式: 保留参数兼容, 但所有值最终走 SGLang 原生推理
        valid_modes = ("auto", "prompt_prefix", "hf_kv", "vllm_kv")
        if injection_mode not in valid_modes:
            logger.warning(
                f"Unknown injection_mode '{injection_mode}', "
                f"defaulting to 'auto' (SGLang native KV injection)"
            )
            injection_mode = "auto"
        
        if injection_mode == "hf_kv":
            logger.warning(
                "injection_mode='hf_kv' is deprecated. "
                "HF model loading has been removed. "
                "All inference now uses SGLang native KV injection "
                "(equivalent effect, no VRAM waste). "
                "This parameter is accepted for backward compatibility."
            )
        
        # 统一存储为 "prompt_prefix" 以兼容 Executor 的 _is_prompt_prefix_mode() 检查
        self.injection_mode = "prompt_prefix"
        
        # SGLang 核心组件
        self.engine = None
        self.sampling_params = None
    
    # SGLang Engine 核心参数 — 这些参数在所有版本中都应被接受
    # 如果参数检测结果不包含这些参数, 说明检测到了包装类而非真正的 Engine
    _SGLANG_CORE_PARAMS = {'model_path', 'tp_size', 'mem_fraction_static'}
    
    @staticmethod
    def _filter_engine_kwargs(sgl_module, engine_kwargs: dict) -> dict:
        """
        过滤 SGLang Engine 不支持的参数, 确保跨版本兼容性.
        
        不同版本的 SGLang 的 ServerArgs 支持不同的参数:
        - schedule_policy: 较新版本 (替代旧版 schedule_heuristic)
        - chunked_prefill_size: 部分版本可能不支持
        - 其他参数: 随版本变化
        
        策略: 
        1. 优先从 ServerArgs 获取参数列表
        2. 回退到 Engine.__init__ 检测
        3. 安全检查: 如果检测结果不包含核心参数 (model_path 等),
           说明检测到了包装类/工厂类, 跳过过滤
        4. 如果检测失败, 保留所有参数
        """
        import inspect
        
        accepted_params = None
        source = None
        
        # 尝试从 ServerArgs 获取参数列表 (SGLang Engine 内部使用 ServerArgs)
        for import_path in [
            'sglang.srt.server_args',      # SGLang >= 0.4.x
            'sglang.srt.utils.server_args', # 某些版本的备选路径
        ]:
            try:
                import importlib
                mod = importlib.import_module(import_path)
                ServerArgs = getattr(mod, 'ServerArgs')
                sig = inspect.signature(ServerArgs.__init__)
                params = set(sig.parameters.keys()) - {'self'}
                # 检查是否有 **kwargs
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if has_var_keyword:
                    # ServerArgs 接受 **kwargs, 无法可靠过滤
                    logger.debug(f"ServerArgs ({import_path}) 接受 **kwargs, 跳过过滤")
                    return engine_kwargs
                accepted_params = params
                source = f"ServerArgs ({import_path})"
                break
            except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
                continue
        
        # 回退: 尝试从 Engine.__init__ 获取
        if accepted_params is None:
            try:
                sig = inspect.signature(sgl_module.Engine.__init__)
                params = set(sig.parameters.keys()) - {'self'}
                # 如果 Engine.__init__ 接受 **kwargs, 则无法过滤
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if has_var_keyword:
                    logger.debug("Engine.__init__ 接受 **kwargs, 跳过过滤")
                    return engine_kwargs
                accepted_params = params
                source = "Engine.__init__"
            except (AttributeError, ValueError):
                pass
        
        if accepted_params is None:
            # 无法检测, 保留所有参数 (让 SGLang 自己报错)
            logger.debug("无法检测 SGLang Engine 接受的参数, 保留所有参数")
            return engine_kwargs
        
        # ============ 安全检查: 检测是否为包装类/工厂类 ============
        # SGLang 某些版本的 Engine 是工厂类 (只接受 class_name, module_name),
        # 实际参数通过内部机制传递给 ServerArgs。
        # 如果检测结果不包含核心参数, 说明检测到了包装类, 跳过过滤。
        if not SGLangAdapter._SGLANG_CORE_PARAMS.intersection(accepted_params):
            logger.info(
                f"检测到 {source} 可能是包装类/工厂类 "
                f"(参数: {sorted(accepted_params)}), "
                f"跳过参数过滤, 保留所有参数直接传递给 SGLang Engine"
            )
            return engine_kwargs
        
        filtered = {}
        dropped = []
        for key, value in engine_kwargs.items():
            if key in accepted_params:
                filtered[key] = value
            else:
                dropped.append(key)
        
        if dropped:
            logger.warning(
                f"SGLang Engine 不支持以下参数 (已自动过滤): {dropped}。"
                f"如需使用这些参数, 请升级 SGLang 版本。"
                f"接受的参数 ({source}): {sorted(accepted_params)}"
            )
        else:
            logger.debug(f"所有参数均被 {source} 接受")
        
        return filtered
    
    @staticmethod
    def _get_sglang_version() -> str:
        """获取 SGLang 版本号, 用于错误诊断"""
        try:
            import sglang
            return getattr(sglang, '__version__', 'unknown')
        except Exception:
            return 'unknown'
    
    @property
    def effective_injection_mode(self) -> str:
        """
        实际注入模式 (始终返回 "prompt_prefix")
        
        所有模式统一为 SGLang 原生 KV 注入,
        通过 prompt 前缀 + RadixAttention 实现。
        返回 "prompt_prefix" 以兼容 Executor 的模式检查。
        """
        return "prompt_prefix"
    
    def load(self) -> None:
        """
        Load SGLang engine.
        
        SGLang 原生支持 Qwen3.5 等新架构, 无需 model_impl="transformers" 回退。
        RadixAttention 自动复用相同前缀的 KV Cache。
        
        量化支持:
        - "none": 原始精度
        - "gptq": GPTQ 量化 (需要预量化模型)
        - "awq": AWQ 量化 (需要预量化模型)
        - "4bit": bitsandbytes 4-bit 量化
        - "8bit": bitsandbytes 8-bit 量化
        - "fp8": FP8 量化 (需要预量化模型)
        """
        if self._is_loaded:
            return
        
        try:
            # ============ HuggingFace Hub 兼容性补丁 ============
            from dki.models.hf_compat import ensure_hf_compat
            ensure_hf_compat()
            
            import sglang as sgl
            from transformers import AutoTokenizer, AutoConfig
            
            # ============ 预加载 AutoProcessor (SGLang 兼容性) ============
            # SGLang 0.5.x 内部使用自定义 lazy import 机制加载 AutoProcessor,
            # 在 transformers >= 4.50 的 _LazyModule 系统下可能失败, 
            # 报错: "Could not import module 'AutoProcessor'"
            # 
            # 常见原因:
            #   1. huggingface-hub 版本过低 (如 0.36.2), 导致 transformers 的
            #      _LazyModule 系统无法正常工作, 所有 lazy import 都会失败
            #      → 修复: pip install huggingface-hub>=1.3.0 --upgrade
            #   2. SGLang 自定义 import 机制与 transformers _LazyModule 冲突
            #      → 修复: pip install sglang --upgrade
            #
            # 解决方案: 在 sgl.Engine() 初始化之前, 先通过标准 import 将
            # AutoProcessor 加载到 sys.modules 中, 这样 SGLang 内部就能找到它
            try:
                from transformers import AutoProcessor  # noqa: F401
                logger.debug("AutoProcessor 预加载成功 (SGLang 兼容性)")
            except ImportError as _ap_err:
                # 检查是否因 huggingface-hub 版本过低导致
                from dki.models.hf_compat import _get_hf_hub_version
                _hf_ver = _get_hf_hub_version()
                _is_hf_hub_issue = False
                try:
                    from packaging.version import Version
                    if _hf_ver != "unknown" and Version(_hf_ver) < Version("1.3.0"):
                        _is_hf_hub_issue = True
                except Exception:
                    pass
                
                if _is_hf_hub_issue:
                    logger.error(
                        f"AutoProcessor 导入失败, 根因: huggingface-hub 版本过低 "
                        f"({_hf_ver}, 需要 >=1.3.0)。\n"
                        f"  transformers 4.50+ 的 lazy import 系统依赖新版 "
                        f"huggingface-hub, 版本过低会导致所有组件导入失败。\n"
                        f"  修复: pip install huggingface-hub>=1.3.0 --upgrade\n"
                        f"  或一次性升级: pip install huggingface-hub transformers "
                        f"sglang --upgrade"
                    )
                    raise ImportError(
                        f"huggingface-hub 版本过低 ({_hf_ver}), 导致 transformers "
                        f"组件 (AutoProcessor) 无法导入。"
                        f"修复: pip install huggingface-hub>=1.3.0 --upgrade"
                    ) from _ap_err
                else:
                    logger.warning(
                        "AutoProcessor 导入失败, SGLang 可能在初始化时报错。"
                        "对于纯文本模型, 这通常不影响功能。"
                        "如需多模态支持, 请安装: pip install transformers[vision]"
                    )
            
            quant_desc = f" (quantization={self.quantization})" if self.is_quantized else ""
            logger.info(
                f"Loading SGLang engine (RadixAttention KV injection): "
                f"{self.model_name}{quant_desc}"
            )
            
            # ============ SGLang 引擎参数 ============
            engine_kwargs = {
                'model_path': self.model_name,
                'tp_size': self.tensor_parallel_size,
                'context_length': self.max_model_len,
                'mem_fraction_static': self.mem_fraction_static,
                'trust_remote_code': self.trust_remote_code,
                'chunked_prefill_size': self.chunked_prefill_size,
                'schedule_policy': self.schedule_policy,
                # ============ CUDA Graph 内存控制 ============
                # CUDA Graph capture 需要额外显存来预分配计算图.
                # 默认 cuda_graph_max_bs=32 对大模型 (27B) 可能导致 OOM,
                # 尤其是在单卡 (L20 46GB) 上显存已被模型权重占满的情况下.
                # 设为 4 可以显著减少 CUDA Graph 占用 (~4-5 GiB → ~1 GiB),
                # 同时保留 CUDA Graph 的性能优势 (不建议完全禁用).
                'cuda_graph_max_bs': 2,
            }
            
            # ============ 量化配置 ============
            if self.quantization != "none":
                sglang_quant_map = {
                    # GPTQ / AWQ: 优先使用 Marlin kernel 变体
                    # Marlin kernel 对 Qwen3.5 等 hybrid 架构 (Mamba + Transformer)
                    # 的兼容性和性能更好:
                    #   - gptq_marlin: 避免 causal_conv1d dtype 不匹配
                    #   - awq_marlin: SGLang 自动检测到可用时也会建议使用
                    #     (日志: "Detected that the model can run with awq_marlin...
                    #      Use quantization=awq_marlin for faster inference")
                    "gptq": "gptq_marlin",
                    "awq": "awq_marlin",
                    "4bit": "bitsandbytes",
                    "8bit": "bitsandbytes",
                    "fp8": "fp8",
                }
                sglang_quant = sglang_quant_map.get(self.quantization)
                if sglang_quant:
                    engine_kwargs['quantization'] = sglang_quant
                    logger.info(f"SGLang quantization: {sglang_quant}")
                    
                    # ============ dtype + Mamba 兼容性 (量化类型分策略) ============
                    # Qwen3.5 等 hybrid 架构包含 Mamba (linear attention) 层,
                    # 其 causal_conv1d 操作要求 conv_states 和 input 的 dtype 一致.
                    # (causal_conv1d.py 已添加 _ensure_conv_states_dtype 对齐保护)
                    #
                    # dtype 选择策略:
                    #
                    # 1) GPTQ: 反量化内核严格限制 float16 输出.
                    #    → dtype='float16', mamba_ssm_dtype='float16'
                    #    → causal_conv1d.py 中 conv_states 自动对齐为 float16
                    #    → float16 数值范围 ±65504, 对 Mamba SSM 可能偏小,
                    #      但 GPTQ 不支持 bfloat16, 这是唯一选择.
                    #
                    # 2) AWQ (awq_marlin): Marlin kernel 支持 bfloat16 反量化.
                    #    → dtype='bfloat16', mamba_ssm_dtype='bfloat16'
                    #    → bfloat16 数值范围 ±3.4×10³⁸ (与 float32 相同),
                    #      Mamba SSM 指数运算和状态累乘不会溢出 → 不产生 NaN.
                    #    → L20 (Ada Lovelace, CC 8.9) 原生支持 bfloat16.
                    #    → Qwen3.5 原始训练 dtype 就是 bfloat16, 最佳兼容.
                    #
                    # 3) bitsandbytes (4bit/8bit): 支持 bfloat16 计算.
                    #    → 同 AWQ, 使用 bfloat16.
                    #
                    # NaN 根因 (已解决):
                    #   之前强制 dtype='float16' → Mamba SSM 中间值超过 65504
                    #   → 溢出 → NaN → completion_tokens=1 就终止.
                    #   改用 bfloat16 (AWQ/bitsandbytes) 后, 数值范围扩大到 ±3.4×10³⁸,
                    #   彻底解决 NaN.
                    #
                    # 注意: 不要启用 enable_fp32_lm_head=True!
                    # 原因: LM head 的 fp32 投影需要额外 ~4.7 GiB 显存,
                    # 对 NVIDIA L20 (46 GiB) + 27B AWQ-int4 会直接 OOM.
                    
                    if self.quantization == "gptq":
                        # GPTQ: 必须 float16 (Marlin kernel 限制)
                        engine_kwargs['dtype'] = 'float16'
                        engine_kwargs['mamba_ssm_dtype'] = 'float16'
                        logger.info(
                            f"Quantization ({sglang_quant}) detected — "
                            f"forcing dtype=float16, mamba_ssm_dtype=float16 "
                            f"(GPTQ Marlin kernel only supports float16)"
                        )
                    elif self.quantization == "fp8":
                        # 2. FP8 专属配置 (核心修改)
                        # FP8 模型基于 bfloat16 训练，强制使用 bfloat16 避免数值溢出
                        engine_kwargs['dtype'] = 'bfloat16'
                        engine_kwargs['kv_cache_dtype'] = 'bfloat16'
                        engine_kwargs['mamba_ssm_dtype'] = 'bfloat16'
                        # FP8 无需禁用 radix cache (减少性能损耗)，仅禁用重叠调度避免 Mamba 异常
                        engine_kwargs['disable_radix_cache'] = False  # 修正：FP8 可启用 radix cache
                        engine_kwargs['disable_overlap_schedule'] = True
                        # 千问3.5-FP8 的 attention/mamba 后端仍用 triton
                        engine_kwargs['attention_backend'] = 'triton'
                        engine_kwargs['mamba_backend'] = 'triton'
                        # 新增：FP8 量化建议启用内存高效的注意力实现
                        engine_kwargs['enable_memory_efficient_attention'] = True
                        logger.info(
                            f"Quantization ({sglang_quant}) detected — "
                            f"forcing dtype=bfloat16, mamba_ssm_dtype=bfloat16 "
                            f"(FP8 model trained with bfloat16; L20 48G supports bfloat16 natively; "
                            f"triton backend ensures Mamba/attention compatibility)"
                        )
                    else:
                        # AWQ / bitsandbytes: 使用 bfloat16 (更大数值范围, 避免 NaN)
                        engine_kwargs['dtype'] = 'bfloat16'
                        engine_kwargs['kv_cache_dtype'] = 'bfloat16'
                        engine_kwargs['mamba_ssm_dtype'] = 'bfloat16'
                        engine_kwargs['disable_radix_cache'] = True        # 修复NaN核心
                        engine_kwargs['disable_overlap_schedule'] = True   # 避免Mamba调度异常
                        engine_kwargs['attention_backend'] = 'triton'      # 避开flashinfer Bug
                        engine_kwargs['mamba_backend'] = 'triton'          # 适配千问3.5的mamba后端
                        logger.info(
                            f"Quantization ({sglang_quant}) detected — "
                            f"forcing dtype=bfloat16, mamba_ssm_dtype=bfloat16 "
                            f"(bfloat16 range ±3.4e38 prevents Mamba SSM NaN overflow; "
                            f"causal_conv1d.py dtype alignment ensures consistency)"
                        )

                else:
                    logger.warning(
                        f"Quantization '{self.quantization}' not directly supported by SGLang, "
                        f"proceeding without quantization parameter"
                    )
            
            # ============ 安全参数过滤 ============
            # 不同版本的 SGLang 支持不同的 ServerArgs 参数
            # 动态检测 sgl.Engine (底层 ServerArgs) 接受的参数, 过滤不支持的参数
            engine_kwargs = self._filter_engine_kwargs(sgl, engine_kwargs)
            
            # ============ 创建 SGLang Engine ============
            # 
            # Engine 必须在主线程中创建:
            #   - SGLang Engine.__init__() 内部注册 signal handler (SIGQUIT 等)
            #   - Python 限制: signal.signal() 只能在主线程中调用
            #   - 在子线程中创建会触发: "signal only works in main thread"
            # 
            # Event loop 冲突的解决方案:
            #   Engine 在主线程创建后, engine.loop = uvicorn 的 uvloop (已在运行)
            #   → engine.generate() 调用 self.loop.run_until_complete() → 会失败!
            #   → 但 engine.async_generate() 不调用 run_until_complete(),
            #     直接返回 awaitable → 在 async 上下文中 await 即可, 完全安全.
            # 
            #   因此: 在 FastAPI/uvicorn async 上下文中, 必须使用 async_generate().
            #   同步的 generate() 仅供 CLI/测试 (无 running event loop) 使用.
            #   调用方 (injection_executor.py, dki_plugin.py) 已修改为
            #   优先检测并使用 async_generate() / async_forward_with_kv_injection().
            self.engine = sgl.Engine(**engine_kwargs)
            
            # ============ 加载 Tokenizer ============
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
                truncation_side="left",
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # ============ 模型配置 (仅用于元信息) ============
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            self.hidden_dim = getattr(config, 'hidden_size', 4096)
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            self.num_heads = getattr(config, 'num_attention_heads', 32)
            self.head_dim = self.hidden_dim // self.num_heads
            
            self._is_loaded = True
            
            logger.info(
                f"SGLang adapter loaded: {self.model_name} — "
                f"SGLang RadixAttention KV Injection "
                f"(schedule={self.schedule_policy}, 无 HF 模型)"
            )
            
        except ImportError as e:
            error_msg = str(e)
            # 优先检查 huggingface-hub 版本问题 (最常见的根因)
            if "huggingface-hub" in error_msg or "huggingface_hub" in error_msg:
                logger.error(
                    f"huggingface-hub 版本问题: {error_msg}\n"
                    f"  修复: pip install huggingface-hub>=1.3.0 --upgrade\n"
                    f"  或一次性升级: pip install huggingface-hub transformers "
                    f"sglang --upgrade"
                )
            elif "sglang" in error_msg.lower():
                logger.error(
                    f"SGLang 未安装: {error_msg}\n"
                    f"  修复: pip install sglang[all]\n"
                    f"  或: pip install sglang"
                )
            elif "AutoProcessor" in error_msg or "Could not import module" in error_msg:
                # SGLang 内部 lazy import 机制失败
                # 可能原因: 1) huggingface-hub 版本过低  2) SGLang 与 transformers 不兼容
                logger.error(
                    f"SGLang 内部模块加载失败: {error_msg}\n"
                    f"  可能原因:\n"
                    f"  1. huggingface-hub 版本过低 → "
                    f"pip install huggingface-hub>=1.3.0 --upgrade\n"
                    f"  2. SGLang {self._get_sglang_version()} 与 transformers "
                    f"版本不兼容 → pip install sglang --upgrade\n"
                    f"  3. 缺少多模态依赖 → "
                    f"pip install transformers[vision]"
                )
            elif "transformers" in error_msg.lower():
                logger.error(
                    f"transformers 导入失败: {error_msg}\n"
                    f"  修复: pip install transformers --upgrade"
                )
            else:
                logger.error(f"依赖导入失败: {error_msg}")
            raise
        except Exception as e:
            logger.error(f"Failed to load SGLang model: {e}")
            raise
    
    # ================================================================
    # Chat Template 处理
    # ================================================================
    
    def _has_chat_template_tokens(self, text: str) -> bool:
        """检测文本是否已包含 chat template 特殊标记 (避免双重包装)"""
        if '<|im_start|>' in text:
            return True
        if '<\uff5c' in text and '\uff5c>' in text:
            return True
        if '<|begin_of_text|>' in text or '<|start_header_id|>' in text:
            return True
        if '[INST]' in text:
            return True
        return False
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt using tokenizer's chat template.
        
        SGLang 的 engine.generate() 接收 raw prompt string,
        必须在调用前使用 tokenizer.apply_chat_template() 构造正确的 chat 格式。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                logger.warning(f"apply_chat_template failed, using ChatML fallback: {e}")
        
        # 回退: ChatML 格式 (Qwen 通用)
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts) + "\n"
    
    def _is_chat_model(self) -> bool:
        """判断是否为 Chat/Instruct 模型"""
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in ('chat', 'instruct'))
    
    def _get_stop_strings(self) -> list:
        """
        获取模型的 stop strings (防止生成越过 turn 边界后退化)
        """
        stop_strings = []
        
        name_lower = self.model_name.lower()
        if any(kw in name_lower for kw in ('deepseek', 'qwen')):
            stop_strings.append("<|im_end|>")
        
        if 'llama' in name_lower and '3' in name_lower:
            stop_strings.append("<|eot_id|>")
        
        if self.tokenizer:
            eos = getattr(self.tokenizer, 'eos_token', None)
            if eos and eos not in stop_strings:
                stop_strings.append(eos)
        
        if not stop_strings:
            stop_strings.append("<|im_end|>")
        
        logger.debug(f"Stop strings for {self.model_name}: {stop_strings}")
        return stop_strings
    
    # ================================================================
    # Logprobs 解析 (v8.0 熵门控)
    # ================================================================
    
    @staticmethod
    def _parse_sglang_logprobs(meta_info: dict) -> Optional[List[List[float]]]:
        """
        从 SGLang meta_info 中提取 logprobs 并转换为 EntropyMonitor 格式.
        
        SGLang logprobs 格式 (meta_info 中):
            - "output_token_logprobs": List[float]  — 每个生成 token 的 log prob
            - "output_top_logprobs": List[Dict[int, float]] — 每个 token 的 top-k log probs
            
        转换为:
            List[List[float]] — 每个 token 的 top-k log probabilities
            
        如果只有 output_token_logprobs (无 top-k), 则每个 token 返回 [logp] 单元素列表.
        """
        if not meta_info:
            return None
        
        # 优先使用 top-k logprobs
        top_logprobs = meta_info.get("output_top_logprobs")
        if top_logprobs and isinstance(top_logprobs, list):
            result = []
            for token_top in top_logprobs:
                if isinstance(token_top, dict):
                    # Dict[token_id, logprob] → List[float]
                    lps = []
                    for _tid, lp in token_top.items():
                        if hasattr(lp, 'logprob'):
                            lps.append(lp.logprob)
                        elif isinstance(lp, (int, float)):
                            lps.append(float(lp))
                        else:
                            try:
                                lps.append(float(lp))
                            except (TypeError, ValueError):
                                continue
                    if lps:
                        result.append(lps)
                elif isinstance(token_top, (list, tuple)):
                    # 已经是 list 格式
                    result.append([float(x) for x in token_top if isinstance(x, (int, float))])
            if result:
                return result
        
        # 回退: 使用单一 token logprobs
        token_logprobs = meta_info.get("output_token_logprobs")
        if token_logprobs and isinstance(token_logprobs, list):
            return [[float(lp)] for lp in token_logprobs if isinstance(lp, (int, float))]
        
        return None
    
    # ================================================================
    # 核心推理接口
    # ================================================================
    
    def _call_engine_generate(self, prompt: str, sampling_params: dict) -> dict:
        """
        同步调用 SGLang Engine.generate().
        
        ============ 使用场景 ============
        1. 非 async 上下文 (测试/CLI): 直接调用 engine.generate()
        2. DKISystem.chat() 通过 run_in_executor 在线程池中调用:
           engine.generate() 内部调用 self.loop.run_until_complete(),
           如果 self.loop 是 uvicorn 的 uvloop 且正在运行, 会失败.
           此时使用 asyncio.run_coroutine_threadsafe() 提交到引擎的事件循环.
        
        ============ 推荐 ============
        在 FastAPI/uvicorn async 上下文中, 请使用 _call_engine_generate_async().
        DKIPlugin.chat() (async) 已正确使用异步路径.
        """
        try:
            return self.engine.generate(prompt, sampling_params)
        except RuntimeError as e:
            if "event loop" in str(e).lower() or "already running" in str(e).lower():
                # 事件循环冲突: engine.generate() 内部的 run_until_complete() 失败
                # 尝试通过 asyncio.run_coroutine_threadsafe 提交到引擎的事件循环
                logger.warning(
                    f"engine.generate() event loop conflict: {e}. "
                    f"Falling back to run_coroutine_threadsafe via engine.async_generate(). "
                    f"Consider using DKIPlugin (async) instead of DKISystem (sync) "
                    f"for SGLang engines."
                )
                if hasattr(self.engine, 'async_generate'):
                    import concurrent.futures
                    engine_loop = getattr(self.engine, 'loop', None)
                    if engine_loop and engine_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self.engine.async_generate(prompt, sampling_params),
                            engine_loop,
                        )
                        return future.result(timeout=300)  # 5 分钟超时
                    else:
                        # 创建新的事件循环来执行
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(
                                self.engine.async_generate(prompt, sampling_params)
                            )
                        finally:
                            new_loop.close()
                raise
            raise
    
    async def _call_engine_generate_async(self, prompt: str, sampling_params: dict) -> dict:
        """
        异步调用 SGLang Engine, 解决事件循环冲突.
        
        ============ 问题根因 ============
        SGLang Engine.generate() 是同步方法, 内部实现 (engine.py line 294):
        
            def generate(self, prompt, params):
                ret = self.loop.run_until_complete(generator.__anext__())
        
        其中 self.loop 是 Engine 初始化时获取的当前线程 event loop.
        在 uvicorn 环境中, Engine 在主线程初始化, self.loop = uvicorn 的 uvloop.
        
        当 DKI 在 FastAPI async handler 中调用 model.generate() 时:
          uvicorn event loop (running)
            → async dki_plugin.chat()
              → async executor.execute()
                → model.generate()  [同步调用]
                  → engine.generate()
                    → self.loop.run_until_complete()  ← 嵌套调用! RuntimeError!
        
        ============ 为什么 vLLM 没有这个问题 ============
        vLLM LLM.generate() 内部不使用 event loop.
        vLLM 使用多进程架构, generate() 通过 zmq socket 发送请求到 worker 进程,
        然后用 threading.Event 同步等待结果. 整个过程不涉及 asyncio event loop,
        所以在 async 上下文中调用完全安全.
        
        ============ 解决方案 ============
        直接 await engine.async_generate(), 它是 SGLang 提供的异步 API.
        async_generate() 不调用 run_until_complete(), 而是直接返回 awaitable,
        可以安全地在已运行的 event loop 中 await.
        """
        if hasattr(self.engine, 'async_generate'):
            # SGLang >= 0.4.x 提供 async_generate()
            # 直接 await, 不会嵌套 run_until_complete()
            return await self.engine.async_generate(prompt, sampling_params)
        else:
            # SGLang 版本过低 (< 0.4.x), 没有 async_generate
            # 
            # 注意: 这个 fallback 在 uvicorn 环境中仍然可能失败!
            # engine.generate() 内部调用 self.loop.run_until_complete(),
            # 而 self.loop 是 uvicorn 的 uvloop (已在运行).
            # run_in_executor 虽然在新线程中执行, 但 self.loop.run_until_complete()
            # 检查的是 loop.is_running() 状态, 不管从哪个线程调用都会失败.
            # 
            # 如果遇到此问题, 请升级 SGLang: pip install sglang>=0.4.0
            logger.warning(
                "SGLang Engine 不支持 async_generate (版本过低), "
                "回退到线程池执行 engine.generate(). "
                "如果在 uvicorn 环境中报错 'event loop is already running', "
                "请升级 SGLang: pip install sglang>=0.4.0"
            )
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,  # 使用默认线程池
                self.engine.generate, prompt, sampling_params
            )
    
    def _parse_engine_output(self, output) -> Tuple[str, dict]:
        """
        解析 SGLang Engine 输出 (兼容多种返回格式)
        
        SGLang Engine.generate() / async_generate() 的返回格式因版本而异:
        
        1. dict: {"text": "...", "meta_info": {...}}
           → 直接提取 text 和 meta_info
        
        2. list of dict: [{"text": "...", "meta_info": {...}}]
           → SGLang 批量推理时返回列表, 取第一个元素
        
        3. 对象 (GenerateOutput / RequestOutput 等):
           → 尝试访问 .text / .output_text / .outputs[0].text 等属性
           → SGLang 0.5.x 可能返回此类对象
        
        4. str: 直接返回文本
           → 某些简化接口可能直接返回字符串
        
        5. None: 生成失败
           → 返回空字符串, 记录警告
        
        安全原则: 永远不会抛出异常, 最差情况返回 ("", {})
        """
        if output is None:
            logger.warning("SGLang engine returned None — model may have failed to generate")
            return "", {}
        
        # ---- Case 1: dict ----
        if isinstance(output, dict):
            output_text = output.get("text", "")
            meta_info = output.get("meta_info", {})
            if not output_text and "output_text" in output:
                output_text = output["output_text"]
            
            # ---- NaN 检测 ----
            # SGLang 在推理过程中检测到 NaN 时会立即终止生成:
            #   finish_reason = {'type': 'stop', 'matched': 'NaN happened'}
            #   text = '' (空输出)
            # 这通常发生在量化模型 (GPTQ/AWQ int4) + float16 精度下,
            # LM head 的 logits 溢出导致 softmax 产生 NaN.
            finish_reason = meta_info.get("finish_reason", {})
            if isinstance(finish_reason, dict):
                matched = finish_reason.get("matched", "")
                if "NaN" in str(matched):
                    logger.error(
                        f"[SGLang NaN detected] Model produced NaN during inference! "
                        f"finish_reason={finish_reason}, "
                        f"prompt_tokens={meta_info.get('prompt_tokens', '?')}, "
                        f"completion_tokens={meta_info.get('completion_tokens', '?')}. "
                        f"This is typically caused by numerical overflow in quantized models. "
                        f"Solutions: "
                        f"1) Use dtype=bfloat16 instead of float16 (bfloat16 range ±3.4e38 "
                        f"vs float16 ±65504, prevents Mamba SSM overflow); "
                        f"2) Ensure causal_conv1d.py dtype alignment is deployed; "
                        f"3) Try reducing prompt length."
                    )
            
            logger.debug(
                f"SGLang output (dict): text_len={len(output_text)}, "
                f"meta_keys={list(output.keys())}"
            )
            return output_text, meta_info
        
        # ---- Case 2: list (批量推理) ----
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                logger.warning("SGLang engine returned empty list")
                return "", {}
            first = output[0]
            if isinstance(first, dict):
                output_text = first.get("text", "")
                meta_info = first.get("meta_info", {})
                if not output_text and "output_text" in first:
                    output_text = first["output_text"]
                logger.debug(
                    f"SGLang output (list[0] dict): text_len={len(output_text)}, "
                    f"batch_size={len(output)}"
                )
                return output_text, meta_info
            # list[0] 是对象
            return self._extract_from_object(first, "list[0]")
        
        # ---- Case 3: str ----
        if isinstance(output, str):
            logger.debug(f"SGLang output (str): text_len={len(output)}")
            return output, {}
        
        # ---- Case 4: 对象 (GenerateOutput / RequestOutput 等) ----
        return self._extract_from_object(output, type(output).__name__)
    
    def _extract_from_object(self, obj, label: str = "object") -> Tuple[str, dict]:
        """
        从 SGLang 输出对象中提取文本和元信息
        
        SGLang 不同版本可能返回不同的对象类型:
        - GenerateOutput: 有 .text 属性
        - RequestOutput: 有 .outputs 列表, 每个元素有 .text
        - 其他: 尝试常见属性名
        """
        meta_info = {}
        output_text = ""
        
        # 策略 1: 直接 .text 属性
        if hasattr(obj, 'text'):
            output_text = obj.text or ""
            if hasattr(obj, 'meta_info'):
                meta_info = obj.meta_info if isinstance(obj.meta_info, dict) else {}
            logger.debug(
                f"SGLang output ({label}): .text len={len(output_text)}"
            )
            return output_text, meta_info
        
        # 策略 2: .output_text 属性
        if hasattr(obj, 'output_text'):
            output_text = obj.output_text or ""
            logger.debug(
                f"SGLang output ({label}): .output_text len={len(output_text)}"
            )
            return output_text, meta_info
        
        # 策略 3: .outputs 列表 (RequestOutput 风格)
        if hasattr(obj, 'outputs') and obj.outputs:
            first_output = obj.outputs[0]
            if hasattr(first_output, 'text'):
                output_text = first_output.text or ""
            elif isinstance(first_output, dict):
                output_text = first_output.get("text", "")
            logger.debug(
                f"SGLang output ({label}): .outputs[0].text len={len(output_text)}"
            )
            return output_text, meta_info
        
        # 策略 4: dict-like (支持 __getitem__)
        try:
            output_text = obj["text"]
            meta_info = obj.get("meta_info", {}) if hasattr(obj, 'get') else {}
            logger.debug(
                f"SGLang output ({label}): ['text'] len={len(output_text)}"
            )
            return output_text, meta_info
        except (KeyError, TypeError, IndexError):
            pass
        
        # 策略 5: 最终回退 — str()
        output_text = str(obj)
        logger.warning(
            f"SGLang output ({label}): unknown format, using str() — "
            f"len={len(output_text)}, type={type(obj).__name__}, "
            f"attrs={[a for a in dir(obj) if not a.startswith('_')][:10]}"
        )
        return output_text, meta_info
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """
        Generate text using SGLang.
        
        使用 tokenizer.apply_chat_template 构造符合模型官方标准的 chat 格式。
        SGLang 的 RadixAttention 会自动检测 prompt 前缀并复用 KV Cache,
        因此偏好文本作为前缀时, 首次请求 prefill 计算 KV, 后续请求直接复用。
        
        事件循环安全:
        - 在 FastAPI/uvicorn async 上下文中, 自动使用 async_generate() 或线程池
        - 在非 async 上下文中 (测试/CLI), 直接调用同步 generate()
        """
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        # Format prompt using official chat template
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # ============ logprobs 支持 (v8.0 熵门控) ============
        # 熵门控路径 (_execute_entropy_gated) 传递 logprobs=k,
        # 需要将此参数转发到 SGLang sampling_params.
        # SGLang 原生支持:
        #   return_logprob=True  — 启用 logprob 返回
        #   top_logprobs_num=k   — 返回 top-k log probs
        logprobs_k = kwargs.get('logprobs', None)
        
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop": self._get_stop_strings(),
        }
        if logprobs_k is not None:
            sampling_params["return_logprob"] = True
            sampling_params["top_logprobs_num"] = logprobs_k
        
        output = self._call_engine_generate(formatted_prompt, sampling_params)
        
        end_time = time.perf_counter()
        
        output_text, meta_info = self._parse_engine_output(output)
        
        # ============ 提取 logprobs (v8.0) ============
        parsed_logprobs = None
        if logprobs_k is not None:
            parsed_logprobs = self._parse_sglang_logprobs(meta_info)
        
        return ModelOutput(
            text=output_text,
            tokens=meta_info.get("output_ids", []),
            logprobs=parsed_logprobs,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=meta_info.get("prompt_tokens", 0),
            output_tokens=meta_info.get("completion_tokens", 0),
        )
    
    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: Union[List[KVCacheEntry], str, None] = None,
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """
        Generate with KV injection — 统一走 SGLang 原生推理.
        
        设计:
        - prompt 参数已包含偏好前缀 (由 Executor._build_preference_prefix 构造)
        - injected_kv 参数保留签名兼容, 但不再使用
        - SGLang 的 RadixAttention 自动复用相同前缀的 KV Cache
        
        与 DKI 论文一致:
        - 偏好文本 → SGLang prefill → KV Cache (attention 层)
        - 相同偏好 → 相同前缀 → KV Cache 自动复用 (RadixAttention)
        - alpha 控制由 Executor 在构造前缀时实现
        
        事件循环安全:
        - 自动检测 async 上下文, 避免 "event loop is already running" 错误
        
        Args:
            prompt: 用户输入 (已包含偏好前缀, 由 Executor 组装)
            injected_kv: 保留签名兼容, 内部不使用
            alpha: 注入强度 (由 Executor 在前缀构造时实现)
            max_new_tokens: 最大生成 token 数
        """
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        # Format prompt
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        sampling_params = {
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.9),
            "max_new_tokens": max_new_tokens,
            "stop": self._get_stop_strings(),
        }
        
        output = self._call_engine_generate(formatted_prompt, sampling_params)
        
        end_time = time.perf_counter()
        
        output_text, meta_info = self._parse_engine_output(output)
        
        return ModelOutput(
            text=output_text,
            tokens=meta_info.get("output_ids", []),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=meta_info.get("prompt_tokens", 0),
            output_tokens=meta_info.get("completion_tokens", 0),
            metadata={
                'alpha': alpha,
                'injection_mode': 'sglang_native_radix_attention',
            },
        )
    
    # ================================================================
    # 异步推理接口 (解决 event loop 冲突)
    # ================================================================
    # 
    # SGLang Engine.generate() 内部使用 self.loop.run_until_complete(),
    # 在 uvicorn async 上下文中调用会触发 "event loop is already running".
    # 
    # 以下 async 方法直接 await engine.async_generate(), 绕过 run_until_complete().
    # injection_executor.py 和 dki_plugin.py 中的 async 方法会优先调用这些异步版本.
    # 
    # vLLM 不需要这些方法, 因为 vLLM LLM.generate() 不使用 event loop.
    # ================================================================
    
    async def async_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """
        异步版本的 generate(), 用于 FastAPI/uvicorn async 上下文.
        
        直接 await engine.async_generate(), 不调用 run_until_complete(),
        完全避免 "event loop is already running" 错误.
        """
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # ============ logprobs 支持 (v8.0 熵门控) ============
        logprobs_k = kwargs.get('logprobs', None)
        
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop": self._get_stop_strings(),
        }
        if logprobs_k is not None:
            sampling_params["return_logprob"] = True
            sampling_params["top_logprobs_num"] = logprobs_k
        
        output = await self._call_engine_generate_async(formatted_prompt, sampling_params)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        output_text, meta_info = self._parse_engine_output(output)
        
        logger.info(
            f"[async_generate] latency={latency_ms:.0f}ms, "
            f"output_len={len(output_text)}, "
            f"prompt_len={len(formatted_prompt)}, "
            f"output_type={type(output).__name__}"
        )
        if not output_text:
            logger.warning(
                f"[async_generate] Empty output! "
                f"raw_output_type={type(output).__name__}, "
                f"raw_output_repr={repr(output)[:500]}"
            )
        
        # ============ 提取 logprobs (v8.0) ============
        parsed_logprobs = None
        if logprobs_k is not None:
            parsed_logprobs = self._parse_sglang_logprobs(meta_info)
        
        return ModelOutput(
            text=output_text,
            tokens=meta_info.get("output_ids", []),
            logprobs=parsed_logprobs,
            latency_ms=latency_ms,
            input_tokens=meta_info.get("prompt_tokens", 0),
            output_tokens=meta_info.get("completion_tokens", 0),
        )
    
    async def async_forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: Union[List[KVCacheEntry], str, None] = None,
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """
        异步版本的 forward_with_kv_injection(), 用于 FastAPI/uvicorn async 上下文.
        """
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        sampling_params = {
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.9),
            "max_new_tokens": max_new_tokens,
            "stop": self._get_stop_strings(),
        }
        
        output = await self._call_engine_generate_async(formatted_prompt, sampling_params)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        output_text, meta_info = self._parse_engine_output(output)
        
        logger.info(
            f"[async_forward_with_kv_injection] latency={latency_ms:.0f}ms, "
            f"output_len={len(output_text)}, "
            f"prompt_len={len(formatted_prompt)}, "
            f"alpha={alpha}, output_type={type(output).__name__}"
        )
        if not output_text:
            logger.warning(
                f"[async_forward_with_kv_injection] Empty output! "
                f"raw_output_type={type(output).__name__}, "
                f"raw_output_repr={repr(output)[:500]}"
            )
        
        return ModelOutput(
            text=output_text,
            tokens=meta_info.get("output_ids", []),
            latency_ms=latency_ms,
            input_tokens=meta_info.get("prompt_tokens", 0),
            output_tokens=meta_info.get("completion_tokens", 0),
            metadata={
                'alpha': alpha,
                'injection_mode': 'sglang_native_radix_attention',
            },
        )
    
    # ================================================================
    # 流式生成 (Streaming)
    # ================================================================
    
    async def async_stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async streaming generation using SGLang.
        
        SGLang Engine 支持两种流式模式:
        1. engine.async_generate(stream=True): 原生 async streaming (SGLang >= 0.4.x)
           → 返回 async generator, 逐 token yield
        2. 不支持 stream 参数的旧版本: 完整生成后模拟流式
        
        Yields:
            str: text chunks (tokens or groups of tokens)
        """
        if not self._is_loaded:
            self.load()
        
        # Format prompt
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop": self._get_stop_strings(),
        }
        
        # 尝试使用 SGLang 原生 streaming
        if hasattr(self.engine, 'async_generate'):
            try:
                # SGLang >= 0.4.x: async_generate 支持 stream=True 参数
                # 返回 async generator, 逐 chunk yield
                stream = self.engine.async_generate(
                    formatted_prompt, sampling_params, stream=True
                )
                
                # 检查返回值是否为 async generator
                if hasattr(stream, '__aiter__'):
                    async for chunk in stream:
                        output_text, _ = self._parse_engine_output(chunk)
                        if output_text:
                            yield output_text
                    return
                else:
                    # stream=True 不被支持, 返回值是普通 awaitable
                    # 回退到完整生成
                    output = await stream if asyncio.iscoroutine(stream) else stream
                    output_text, _ = self._parse_engine_output(output)
                    if output_text:
                        # 模拟流式: 按字符组分批 yield
                        chunk_size = 4
                        for i in range(0, len(output_text), chunk_size):
                            yield output_text[i:i + chunk_size]
                            await asyncio.sleep(0)
                    return
            except TypeError:
                # stream 参数不被接受 (旧版 SGLang)
                pass
        
        # 回退: 完整生成后模拟流式
        output = await self._call_engine_generate_async(formatted_prompt, sampling_params)
        output_text, _ = self._parse_engine_output(output)
        
        if output_text:
            chunk_size = 4
            for i in range(0, len(output_text), chunk_size):
                yield output_text[i:i + chunk_size]
                await asyncio.sleep(0)
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Synchronous streaming generation for SGLang.
        
        SGLang Engine 的同步 generate() 不支持 streaming,
        完整生成后按 chunk 分批 yield。
        
        注意: 在 FastAPI/uvicorn async 上下文中, 请使用 async_stream_generate()。
        
        Yields:
            str: text chunks
        """
        output = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        
        chunk_size = 4
        text = output.text
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    
    # ================================================================
    # BaseModelAdapter 抽象方法实现 (安全降级)
    # ================================================================
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Get embeddings — 不可用.
        
        SGLang 原生模式不提供 embedding 接口。
        建议使用独立的 embedding 服务 (如 sentence-transformers)。
        """
        raise RuntimeError(
            "embed() is not available in SGLang native KV mode. "
            "Use an independent embedding service (e.g. sentence-transformers) instead."
        )
    
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute K/V cache — 返回空列表 (安全降级).
        
        KV 注入通过 prompt 前缀 + SGLang RadixAttention 实现,
        不需要显式 compute_kv。偏好文本的 KV 在 SGLang prefill 阶段自然生成。
        """
        logger.debug(
            "compute_kv() called — returning empty KV. "
            "In SGLang adapter, KV injection is handled by RadixAttention "
            "(preferences are injected as prompt prefix)."
        )
        return [], None
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute prefill-stage entropy — 返回默认值.
        
        SGLang 原生模式不提供 attention 权重访问。
        返回 0.5 (中等熵) 作为安全降级。
        """
        return 0.5
    
    # ================================================================
    # Token 处理
    # ================================================================
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_model_len,
        )
    
    def decode(self, token_ids) -> str:
        """Decode token ids to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    
    # ================================================================
    # 模型管理与诊断
    # ================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        info = super().get_model_info()
        info.update({
            'engine': 'sglang',
            'injection_mode': self.injection_mode,
            'effective_injection_mode': self.effective_injection_mode,
            'sglang_native_kv': True,
            'radix_attention_enabled': True,
            'hf_model_loaded': False,
            'sglang_engine_loaded': self.engine is not None,
            'quantization': self.quantization,
            'schedule_policy': self.schedule_policy,
            'mem_fraction_static': self.mem_fraction_static,
            'chunked_prefill_size': self.chunked_prefill_size,
        })
        return info
    
    def unload(self) -> None:
        """Unload SGLang engine."""
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"SGLang engine shutdown warning: {e}")
            del self.engine
            self.engine = None
        
        super().unload()
        logger.info("SGLang adapter unloaded")
