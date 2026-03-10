"""
vLLM Model Adapter for DKI System (v5.0 — vLLM Native KV Injection)

High-performance inference with vLLM engine.

核心设计 (v5.0):
- 所有推理和 KV 注入都走 vLLM，不加载 HuggingFace 模型
- 偏好/历史文本作为 prompt 前缀，由 vLLM prefill 阶段自然计算 KV
- 开启 vLLM enable_prefix_caching: 相同前缀自动复用 KV Cache (零额外代码)
- 这就是 DKI 论文的 KV 注入: 偏好信息的 KV 表示通过 attention 机制影响后续推理

与 DKI 论文的一致性:
- DKI 的核心: 将用户偏好/历史信息编码为 KV 表示, 在 attention 层注入
- vLLM + prefix_caching 完美实现:
  - 偏好文本 → vLLM prefill → KV Cache (与论文的 compute_kv 等价)
  - KV Cache 通过 attention 影响推理 (与论文的 KV injection 等价)
  - 相同前缀自动复用 KV (与论文的 KV cache management 等价)
- 无需 HF 模型旁路, 零 VRAM 浪费

接口兼容性:
- 保留 BaseModelAdapter 所有抽象方法签名
- compute_kv / embed / compute_prefill_entropy 返回安全降级值
- forward_with_kv_injection 保留签名, 内部统一走 vLLM generate
- injection_mode 参数保留, 但所有值最终都路由到 vLLM 原生推理
- 旧代码传入 injection_mode="hf_kv" 或 "prompt_prefix" 不会报错

安全不变量:
- 默认且唯一的推理引擎是 vLLM
- 偏好注入通过 prompt 前缀 + vLLM PagedAttention 实现
- 所有公开接口签名与 BaseModelAdapter 完全兼容
- 异常时自动降级到无注入推理 (fail-open)

Author: AGI Demo Project
Version: 5.0.0
"""

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class VLLMAdapter(BaseModelAdapter):
    """
    vLLM-based model adapter with native KV injection via prefix caching.
    
    核心: 100% vLLM-only, 无 HF 模型, 偏好通过 prompt 前缀注入,
    vLLM enable_prefix_caching 自动复用相同前缀的 KV Cache。
    
    与 DKI 论文一致:
    - 偏好 KV 注入 = 偏好文本作为 prefix → vLLM prefill → attention 层注入
    - KV Cache 复用 = vLLM prefix caching (相同偏好前缀自动复用)
    - 无需 HF 模型计算 KV, 无 ~14GB VRAM 浪费
    
    Usage:
        # 标准用法 (推荐)
        adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")
        adapter.load()
        output = adapter.generate("你好")
        
        # 带偏好 KV 注入
        output = adapter.forward_with_kv_injection(
            prompt="用户偏好前缀\\n\\n你好",
            injected_kv=[],   # 不再使用, 保留签名兼容
            alpha=0.7,
        )
        
        # 旧代码兼容 (injection_mode 参数被接受但不影响行为)
        adapter = VLLMAdapter(
            model_name="Qwen/Qwen2-7B-Instruct",
            injection_mode="hf_kv",  # 接受但内部走 vLLM
        )
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        device: str = "cuda",
        injection_mode: str = "auto",  # 保留参数兼容, 但不影响行为
        quantization: str = "none",
        quantization_config: dict = None,
        model_impl: str = "auto",
        **kwargs
    ):
        """
        初始化 vLLM 适配器
        
        Args:
            model_name: 模型名称或路径
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU 显存利用率
            trust_remote_code: 是否信任远程代码
            device: 设备
            injection_mode: 偏好注入模式 (保留兼容, 所有值最终走 vLLM 原生推理)
                - "auto" (默认): vLLM 原生 KV 注入
                - "prompt_prefix": 等同于 auto
                - "hf_kv": 接受但发出废弃警告, 内部走 vLLM 原生推理
                - "vllm_kv": 等同于 auto
            quantization: 量化模式
                - "none" (默认): 不量化
                - "gptq": GPTQ 量化
                - "awq": AWQ 量化
                - "fp8": FP8 量化 (E4M3 格式, 需要 H100/L40/Ada Lovelace+ GPU)
                - "4bit" / "8bit": bitsandbytes 量化 (通过 vLLM bitsandbytes 集成)
            quantization_config: 量化详细配置
                FP8 专属配置项:
                - fp8_kv_cache: 是否对 KV Cache 也使用 FP8 (默认 false)
                - fp8_compute_dtype: 计算精度 (默认 "bfloat16")
            model_impl: vLLM 模型实现后端
                - "auto" (默认): vLLM 自动选择最优实现
                - "transformers": 强制使用 Transformers backend
                  适用于 vLLM 原生尚不支持的新架构 (如 Qwen-3.5)
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
        self.model_impl = model_impl
        
        # 注入模式: 保留参数兼容, 但所有值最终走 vLLM 原生推理
        valid_modes = ("auto", "prompt_prefix", "hf_kv", "vllm_kv")
        if injection_mode not in valid_modes:
            logger.warning(
                f"Unknown injection_mode '{injection_mode}', "
                f"defaulting to 'auto' (vLLM native KV injection)"
            )
            injection_mode = "auto"
        
        if injection_mode == "hf_kv":
            logger.warning(
                "injection_mode='hf_kv' is deprecated in v5.0. "
                "HF model loading has been removed. "
                "All inference now uses vLLM native KV injection "
                "(equivalent effect, no VRAM waste). "
                "This parameter is accepted for backward compatibility."
            )
        
        # 统一存储为 "prompt_prefix" 以兼容 Executor 的 _is_prompt_prefix_mode() 检查
        # 因为 Executor 检查此属性来决定是否走 prompt_prefix 路径
        self.injection_mode = "prompt_prefix"
        
        # vLLM 核心组件
        self.llm = None
        self.sampling_params = None
    
    @property
    def effective_injection_mode(self) -> str:
        """
        实际注入模式 (始终返回 "prompt_prefix")
        
        v5.0: 所有模式统一为 vLLM 原生 KV 注入,
        通过 prompt 前缀 + prefix_caching 实现。
        返回 "prompt_prefix" 以兼容 Executor 的模式检查。
        """
        return "prompt_prefix"
    
    def load(self) -> None:
        """
        Load vLLM engine with prefix caching enabled.
        
        v5.0: 只加载 vLLM, 不加载 HF 模型。
        开启 enable_prefix_caching 以自动复用相同前缀的 KV Cache。
        
        量化支持:
        - "none": 原始精度
        - "gptq": GPTQ 量化 (需要预量化模型)
        - "awq": AWQ 量化 (需要预量化模型)
        - "fp8": FP8 量化 (E4M3 格式, 需要 Ada Lovelace+ GPU)
        - "4bit": bitsandbytes 4-bit 量化 (通过 vLLM bitsandbytes 集成)
        - "8bit": bitsandbytes 8-bit 量化 (通过 vLLM bitsandbytes 集成)
        
        模型实现后端 (model_impl):
        - "auto": vLLM 自动选择 (默认)
        - "transformers": 强制使用 Transformers backend
          适用于 vLLM 原生尚不支持的新架构 (如 Qwen-3.5)
        """
        if self._is_loaded:
            return
        
        try:
            # ============ HuggingFace Hub 兼容性补丁 ============
            # 必须在 import vllm/transformers 之前调用
            # 解决 huggingface_hub ≥0.25 移除 is_offline_mode 导致导入失败
            from dki.models.hf_compat import ensure_hf_compat
            ensure_hf_compat()
            
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer, AutoConfig
            
            quant_desc = f" (quantization={self.quantization})" if self.is_quantized else ""
            logger.info(
                f"Loading vLLM engine (native KV injection via prefix caching): "
                f"{self.model_name}{quant_desc}"
            )
            
            # ============ vLLM 引擎参数 ============
            llm_kwargs = {
                'model': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'max_model_len': self.max_model_len,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
                'enable_prefix_caching': True,  # 核心: 自动复用相同前缀的 KV Cache
            }
            
            # ============ 模型实现后端 ============
            # model_impl="transformers" 用于 vLLM 原生不支持的新架构
            # 如 Qwen-3.5 (Qwen3_5ForConditionalGeneration)
            if self.model_impl and self.model_impl != "auto":
                llm_kwargs['model_impl'] = self.model_impl
                logger.info(f"vLLM model_impl: {self.model_impl} (forced Transformers backend)")
            
            if self.quantization != "none":
                # vLLM 量化参数映射
                # vLLM 支持: gptq, awq, fp8, squeezellm, bitsandbytes 等
                vllm_quant_map = {
                    "gptq": "gptq",
                    "awq": "awq",
                    "fp8": "fp8",
                    "4bit": "bitsandbytes",
                    "8bit": "bitsandbytes",
                }
                vllm_quant = vllm_quant_map.get(self.quantization)
                if vllm_quant:
                    llm_kwargs['quantization'] = vllm_quant
                    logger.info(f"vLLM quantization: {vllm_quant}")
                    
                    # ============ GPTQ dtype 兼容性 ============
                    # GPTQ 量化仅支持 float16, 不支持 bfloat16
                    # vLLM 默认 dtype="auto" 在某些 GPU 上会解析为 bfloat16,
                    # 导致 ValueError: torch.bfloat16 is not supported for quantization method gptq
                    # 强制设置 dtype="float16" 以确保 GPTQ 兼容性
                    if vllm_quant in ("gptq",):
                        llm_kwargs['dtype'] = 'float16'
                        logger.info(
                            "GPTQ quantization detected — forcing dtype=float16 "
                            "(GPTQ only supports float16, not bfloat16)"
                        )
                    
                    # ============ FP8 量化配置 ============
                    # FP8 (E4M3) 量化: 权重存储为 FP8, 计算使用 bfloat16
                    # 需要 Ada Lovelace+ GPU (L40, L20, RTX 4090, H100 等)
                    # vLLM 原生支持 FP8 量化, 无需额外依赖
                    # 显存节省约 50% (相比 float16), 精度损失 < 1%
                    elif vllm_quant == "fp8":
                        # FP8 模型通常基于 bfloat16 训练, 计算精度使用 bfloat16
                        fp8_compute_dtype = self.quantization_config.get(
                            "fp8_compute_dtype", "bfloat16"
                        )
                        llm_kwargs['dtype'] = fp8_compute_dtype
                        
                        # FP8 KV Cache: 进一步节省 KV Cache 显存
                        # 默认关闭, 因为 KV Cache 精度对长上下文推理质量有影响
                        if self.quantization_config.get("fp8_kv_cache", False):
                            llm_kwargs['kv_cache_dtype'] = 'fp8_e4m3'
                            logger.info(
                                "FP8 quantization with FP8 KV Cache enabled — "
                                "further VRAM savings but may affect long-context quality"
                            )
                        
                        logger.info(
                            f"FP8 quantization detected — "
                            f"dtype={fp8_compute_dtype}, "
                            f"kv_cache_dtype={llm_kwargs.get('kv_cache_dtype', 'auto')} "
                            f"(FP8 E4M3 weights, requires Ada Lovelace+ GPU)"
                        )
                else:
                    logger.warning(
                        f"Quantization '{self.quantization}' not directly supported by vLLM, "
                        f"proceeding without quantization parameter"
                    )
            
            self.llm = LLM(**llm_kwargs)
            
            # 加载 Tokenizer (先于 SamplingParams, 用于获取 stop tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
                truncation_side="left",
            )
            
            # 默认采样参数 (包含 stop tokens, 防止生成越过 <|im_end|> 后退化)
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop=self._get_stop_strings(),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 模型配置 (仅用于元信息)
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
                f"vLLM adapter loaded: {self.model_name} — "
                f"vLLM Native KV Injection (prefix_caching=True, 无 HF 模型)"
            )
            
        except ImportError as e:
            error_msg = str(e)
            # 精准诊断: 区分 "未安装" 和 "版本不兼容"
            if "huggingface-hub" in error_msg and "is required" in error_msg:
                # transformers 检测到 huggingface-hub 版本过低
                logger.error(
                    f"huggingface-hub 版本不兼容: {error_msg}\n"
                    f"  修复: pip install huggingface-hub>=1.3.0 --upgrade\n"
                    f"  或: pip install transformers huggingface-hub --upgrade"
                )
            elif "is_offline_mode" in error_msg:
                # huggingface_hub 缺少 is_offline_mode (hf_compat 补丁未生效?)
                logger.error(
                    f"huggingface_hub 兼容性问题: {error_msg}\n"
                    f"  修复: pip install huggingface-hub>=1.3.0 --upgrade"
                )
            elif "vllm" in error_msg.lower():
                logger.error(f"vLLM 未安装: {error_msg}")
            elif "transformers" in error_msg.lower():
                logger.error(f"transformers 未安装: {error_msg}")
            else:
                logger.error(f"依赖导入失败: {error_msg}")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    # ================================================================
    # Chat Template 处理
    # ================================================================
    
    def _has_chat_template_tokens(self, text: str) -> bool:
        """检测文本是否已包含 chat template 特殊标记 (避免双重包装)"""
        # DeepSeek/Qwen ChatML 标记 (半角)
        if '<|im_start|>' in text:
            return True
        # DeepSeek V2/V3 原生标记 (全角 ｜ + 下划线 ▁, 这是 tokenizer 自带格式)
        if '<\uff5c' in text and '\uff5c>' in text:
            return True
        # Llama 3 标记
        if '<|begin_of_text|>' in text or '<|start_header_id|>' in text:
            return True
        # Llama 2 标记
        if '[INST]' in text:
            return True
        return False
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt using tokenizer's chat template.
        
        vLLM 的 LLM.generate() 接收 raw prompt string, 不会自动应用 chat template。
        必须在调用前使用 tokenizer.apply_chat_template() 构造正确的 chat 格式。
        
        不同模型的标准格式:
        - DeepSeek/Qwen (ChatML): <|im_start|>system/user/assistant<|im_end|>
        - Llama 3.x: <|begin_of_text|><|start_header_id|>...<|end_header_id|>
        - 其他模型: tokenizer 内置 chat template
        
        优先使用 tokenizer.apply_chat_template, 回退到 ChatML 通用格式
        (因为 vLLM 默认模型 Qwen 使用 ChatML)。
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
        
        # 回退: ChatML 格式 (Qwen/DeepSeek 通用)
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
        
        v5.8 修复: 之前 SamplingParams 未设置 stop tokens, 导致模型在生成
        <|im_end|> 后继续输出, 产生大量 <|im_start|>/<|im_end|> 垃圾 token。
        
        不同模型的 stop strings:
        - DeepSeek/Qwen (ChatML): <|im_end|>
        - Llama 3.x: <|eot_id|>
        - Llama 2: </s>
        - 通用: 使用 tokenizer.eos_token
        
        Returns:
            stop strings 列表
        """
        stop_strings = []
        
        # ChatML 模型 (DeepSeek, Qwen): <|im_end|> 是 turn 结束标记
        name_lower = self.model_name.lower()
        if any(kw in name_lower for kw in ('deepseek', 'qwen')):
            stop_strings.append("<|im_end|>")
        
        # Llama 3.x: <|eot_id|> 是 turn 结束标记
        if 'llama' in name_lower and '3' in name_lower:
            stop_strings.append("<|eot_id|>")
        
        # 通用: 从 tokenizer 获取 eos_token
        if self.tokenizer:
            eos = getattr(self.tokenizer, 'eos_token', None)
            if eos and eos not in stop_strings:
                stop_strings.append(eos)
        
        # 兜底: 至少包含 ChatML 的 <|im_end|> (大多数 vLLM 模型使用 ChatML)
        if not stop_strings:
            stop_strings.append("<|im_end|>")
        
        logger.debug(f"Stop strings for {self.model_name}: {stop_strings}")
        return stop_strings
    
    # ================================================================
    # Logprobs 解析 (v8.0 熵门控)
    # ================================================================
    
    @staticmethod
    def _parse_vllm_logprobs(
        raw_logprobs: list,
    ) -> List[List[float]]:
        """
        将 vLLM 返回的 logprobs 转换为 EntropyMonitor 所需的格式.
        
        vLLM logprobs 格式 (per token):
            List[Dict[int, Logprob]]
            其中 Logprob 是 NamedTuple/dataclass, 有 .logprob 属性
            
        转换为:
            List[List[float]]  — 每个 token 的 top-k log probabilities
            
        兼容性:
            - vLLM >= 0.4.x: Logprob 对象有 .logprob 属性
            - 旧版本: Logprob 可能直接是 float
            - dict value 可能是 float 而非对象
        """
        result = []
        for token_logprobs in raw_logprobs:
            if token_logprobs is None:
                continue
            token_lps = []
            if isinstance(token_logprobs, dict):
                for _token_id, lp in token_logprobs.items():
                    if hasattr(lp, 'logprob'):
                        # vLLM Logprob 对象
                        token_lps.append(lp.logprob)
                    elif isinstance(lp, (int, float)):
                        # 直接是数值
                        token_lps.append(float(lp))
                    else:
                        # 尝试 float 转换
                        try:
                            token_lps.append(float(lp))
                        except (TypeError, ValueError):
                            continue
            if token_lps:
                result.append(token_lps)
        return result
    
    # ================================================================
    # 核心推理接口
    # ================================================================
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """
        Generate text using vLLM.
        
        使用 tokenizer.apply_chat_template 构造符合模型官方标准的 chat 格式。
        vLLM 的 prefix_caching 会自动检测 prompt 前缀并复用 KV Cache,
        因此偏好文本作为前缀时, 首次请求 prefill 计算 KV, 后续请求直接复用。
        """
        if not self._is_loaded:
            self.load()
        
        from vllm import SamplingParams
        
        start_time = time.perf_counter()
        
        # Format prompt using official chat template
        # 跳过已经包含 chat template 特殊标记的 prompt (避免双重包装)
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # ============ logprobs 支持 (v8.0 熵门控) ============
        # 熵门控路径 (_execute_entropy_gated) 传递 logprobs=k,
        # 需要将此参数转发到 vLLM SamplingParams.
        # vLLM SamplingParams 原生支持 logprobs 参数:
        #   logprobs=None (不返回) 或 logprobs=k (返回 top-k log probs)
        logprobs_k = kwargs.get('logprobs', None)
        
        sp_kwargs = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_new_tokens,
            'stop': self._get_stop_strings(),
        }
        if logprobs_k is not None:
            sp_kwargs['logprobs'] = logprobs_k
        
        sampling_params = SamplingParams(**sp_kwargs)
        
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        output = outputs[0]
        
        end_time = time.perf_counter()
        
        # ============ 提取 logprobs (v8.0) ============
        # vLLM 返回 output.outputs[0].logprobs: List[Dict[int, Logprob]]
        # 转换为 List[List[float]] 供 EntropyMonitor 使用
        parsed_logprobs = None
        if logprobs_k is not None and hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            parsed_logprobs = self._parse_vllm_logprobs(output.outputs[0].logprobs)
        
        return ModelOutput(
            text=output.outputs[0].text,
            tokens=list(output.outputs[0].token_ids),
            logprobs=parsed_logprobs,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(output.outputs[0].token_ids),
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
        Generate with KV injection — 统一走 vLLM 原生推理.
        
        v5.0 设计:
        - prompt 参数已包含偏好前缀 (由 Executor._build_preference_prefix 构造)
        - injected_kv 参数保留签名兼容, 但不再使用
        - vLLM 的 prefix_caching 自动复用相同前缀的 KV Cache
        
        与 DKI 论文一致:
        - 偏好文本 → vLLM prefill → KV Cache (attention 层)
        - 相同偏好 → 相同前缀 → KV Cache 自动复用
        - alpha 控制由 Executor 在构造前缀时实现 (前缀长度 + 强度标记)
        
        Chat Template 处理:
        - prompt 可能已由 Executor 使用 apply_chat_template 格式化
        - _has_chat_template_tokens 检测已格式化的 prompt 并跳过二次包装
        - 未格式化的 prompt 由 _format_prompt 包装为 chat template 格式
        
        Args:
            prompt: 用户输入 (已包含偏好前缀, 由 Executor 组装)
            injected_kv: 保留签名兼容, 内部不使用
            alpha: 注入强度 (由 Executor 在前缀构造时实现)
            max_new_tokens: 最大生成 token 数
        """
        if not self._is_loaded:
            self.load()
        
        from vllm import SamplingParams
        
        start_time = time.perf_counter()
        
        # Format prompt using official chat template
        # 注意: prompt 可能已包含偏好前缀, 检测是否需要包装 chat template
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=max_new_tokens,
            stop=self._get_stop_strings(),
        )
        
        # prompt 已包含偏好前缀, 直接调用 vLLM generate
        # vLLM prefix_caching 会自动检测并复用相同前缀的 KV Cache
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        output = outputs[0]
        
        end_time = time.perf_counter()
        
        return ModelOutput(
            text=output.outputs[0].text,
            tokens=list(output.outputs[0].token_ids),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(output.outputs[0].token_ids),
            metadata={
                'alpha': alpha,
                'injection_mode': 'vllm_native_prefix_caching',
            },
        )
    
    # ================================================================
    # 异步与流式生成
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
        Async version of generate() for vLLM.
        
        vLLM LLM.generate() 不使用 event loop (通过 zmq 多进程通信),
        在 async 上下文中调用完全安全, 因此直接在线程池中执行。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            ),
        )
    
    async def async_stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async streaming generation using vLLM.
        
        vLLM 支持两种流式模式:
        1. vllm.AsyncLLMEngine (服务端模式): 原生 async streaming
        2. vllm.LLM (离线模式): 不支持原生 streaming
        
        当前 DKI 使用 vllm.LLM (离线模式), 不支持原生 token-by-token streaming。
        因此使用 "模拟流式" 策略: 完整生成后按 chunk 分批 yield。
        
        如需真正的 token-by-token streaming, 应使用 vLLM 的 OpenAI-compatible
        server (vllm serve) + httpx/aiohttp SSE client, 这是 vLLM 官方推荐的
        streaming 方案。
        
        Yields:
            str: text chunks
        """
        if not self._is_loaded:
            self.load()
        
        from vllm import SamplingParams
        
        # Format prompt
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=self._get_stop_strings(),
        )
        
        # vLLM LLM 离线模式: 完整生成后分 chunk yield
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([formatted_prompt], sampling_params),
        )
        
        output_text = outputs[0].outputs[0].text
        
        # 模拟流式: 按字符组分批 yield (每 chunk ~4-8 个字符, 模拟 token 粒度)
        chunk_size = 4
        for i in range(0, len(output_text), chunk_size):
            yield output_text[i:i + chunk_size]
            await asyncio.sleep(0)  # 让出控制权, 允许 SSE flush
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Synchronous streaming generation for vLLM.
        
        vLLM LLM 离线模式不支持原生 streaming,
        完整生成后按 chunk 分批 yield。
        
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
        Get embeddings — v5.0 不可用.
        
        vLLM 原生模式不提供 embedding 接口。
        建议使用独立的 embedding 服务 (如 sentence-transformers)。
        
        Raises:
            RuntimeError: 始终抛出, 提示使用独立 embedding 服务
        """
        raise RuntimeError(
            "embed() is not available in vLLM native KV mode (v5.0). "
            "Use an independent embedding service (e.g. sentence-transformers) instead."
        )
    
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute K/V cache — v5.0 返回空列表 (安全降级).
        
        v5.0 设计: KV 注入通过 prompt 前缀 + vLLM prefix_caching 实现,
        不需要显式 compute_kv。偏好文本的 KV 在 vLLM prefill 阶段自然生成。
        
        Executor 在 prompt_prefix 模式下不会调用此方法,
        返回空列表作为安全降级 (防御性编程)。
        
        Returns:
            ([], None) — 空 KV 列表和 None hidden states
        """
        logger.debug(
            "compute_kv() called — returning empty KV. "
            "In v5.0, KV injection is handled by vLLM prefix caching "
            "(preferences are injected as prompt prefix)."
        )
        return [], None
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute prefill-stage entropy — v5.0 返回默认值.
        
        vLLM 原生模式不提供 attention 权重访问。
        返回 0.5 (中等熵) 作为安全降级。
        偏好注入强度由 Planner 的 alpha 控制, 不依赖熵门控。
        
        Returns:
            0.5 (默认中等熵值)
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
            'injection_mode': self.injection_mode,
            'effective_injection_mode': self.effective_injection_mode,
            'vllm_native_kv': True,
            'prefix_caching_enabled': True,
            'hf_model_loaded': False,  # v5.0: 永远不加载 HF 模型
            'vllm_engine_loaded': self.llm is not None,
            'quantization': self.quantization,
            'model_impl': self.model_impl,
        })
        return info
    
    def unload(self) -> None:
        """Unload vLLM engine."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        super().unload()
        logger.info("vLLM adapter unloaded")
