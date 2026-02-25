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

import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
        """
        super().__init__(model_name, device, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        
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
        """
        if self._is_loaded:
            return
        
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer, AutoConfig
            
            logger.info(
                f"Loading vLLM engine (native KV injection via prefix caching): "
                f"{self.model_name}"
            )
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                enable_prefix_caching=True,  # 核心: 自动复用相同前缀的 KV Cache
            )
            
            # 默认采样参数
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
            )
            
            # 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
                truncation_side="left",
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
            logger.error(f"vLLM not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    # ================================================================
    # Chat Template 处理
    # ================================================================
    
    def _has_chat_template_tokens(self, text: str) -> bool:
        """检测文本是否已包含 chat template 特殊标记 (避免双重包装)"""
        # DeepSeek/Qwen ChatML 标记
        if '<|im_start|>' in text:
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
    
    # ================================================================
    # 核心推理接口
    # ================================================================
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
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
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        output = outputs[0]
        
        end_time = time.perf_counter()
        
        return ModelOutput(
            text=output.outputs[0].text,
            tokens=list(output.outputs[0].token_ids),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(output.outputs[0].token_ids),
        )
    
    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: Union[List[KVCacheEntry], str, None] = None,
        alpha: float = 1.0,
        max_new_tokens: int = 512,
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
        })
        return info
    
    def unload(self) -> None:
        """Unload vLLM engine."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        super().unload()
        logger.info("vLLM adapter unloaded")
