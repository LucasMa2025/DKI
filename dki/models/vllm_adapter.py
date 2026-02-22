"""
vLLM Model Adapter for DKI System (v4.0 — Prompt Prefix Injection)

High-performance inference with vLLM engine.

改造说明 (v4.0):
- 新增 injection_mode 参数，支持三种偏好注入模式:
  - "prompt_prefix" (推荐): 偏好文本作为 prompt 前缀，由 vLLM 在 prefill 阶段自然处理
  - "hf_kv" (兼容): 使用 HuggingFace 模型计算 KV 并注入 past_key_values (原有行为)
  - "auto": 自动选择 — 默认 prompt_prefix，HF 模型仅在显式请求时加载
- prompt_prefix 模式下，所有推理走 vLLM，不加载 HF 模型，节省 ~14GB VRAM
- 保留 hf_kv 模式作为零风险回退，确保向后兼容
- compute_kv / embed / compute_prefill_entropy 在 prompt_prefix 模式下提供安全降级

安全不变量:
- 默认 injection_mode="prompt_prefix"，不加载 HF 模型
- hf_kv 模式保留完整的原有行为，零改动
- 所有公开接口签名不变，BaseModelAdapter 契约完整
- 异常时自动降级到无注入推理 (fail-open)

Author: AGI Demo Project
Version: 4.0.0
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class VLLMAdapter(BaseModelAdapter):
    """
    vLLM-based model adapter for high-performance inference.
    
    Supports two injection modes:
    - prompt_prefix: Preferences as prompt prefix → vLLM generate (recommended)
    - hf_kv: HuggingFace model for K/V computation → past_key_values injection (legacy)
    
    Usage:
        # 推荐: Prompt Prefix 模式 (默认)
        adapter = VLLMAdapter(model_name="Qwen/Qwen2-7B-Instruct")
        adapter.load()
        output = adapter.generate("你好")
        
        # 兼容: HF KV 模式 (原有行为)
        adapter = VLLMAdapter(
            model_name="Qwen/Qwen2-7B-Instruct",
            injection_mode="hf_kv",
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
        injection_mode: str = "prompt_prefix",
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
            injection_mode: 偏好注入模式
                - "prompt_prefix": 偏好文本作为 prompt 前缀 (推荐, 默认)
                - "hf_kv": HuggingFace 模型 KV 注入 (兼容模式)
                - "auto": 自动选择 (等同于 prompt_prefix)
        """
        super().__init__(model_name, device, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        
        # 注入模式
        if injection_mode in ("prompt_prefix", "hf_kv", "auto"):
            self.injection_mode = injection_mode
        else:
            logger.warning(
                f"Unknown injection_mode '{injection_mode}', "
                f"defaulting to 'prompt_prefix'"
            )
            self.injection_mode = "prompt_prefix"
        
        # vLLM specific
        self.llm = None
        self.sampling_params = None
        
        # For K/V computation (hf_kv mode only)
        self.hf_model = None
    
    @property
    def effective_injection_mode(self) -> str:
        """解析 auto 后的实际注入模式"""
        if self.injection_mode == "auto":
            return "prompt_prefix"
        return self.injection_mode
    
    def load(self) -> None:
        """
        Load vLLM engine.
        
        在 prompt_prefix 模式下，只加载 vLLM 引擎，不加载 HF 模型。
        在 hf_kv 模式下，HF 模型按需延迟加载 (首次 compute_kv 时)。
        """
        if self._is_loaded:
            return
        
        try:
            # Import vLLM
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer, AutoConfig
            
            # Load vLLM for fast inference
            logger.info(
                f"Loading vLLM engine: {self.model_name} "
                f"(injection_mode={self.effective_injection_mode})"
            )
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Default sampling params
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Get model config for architecture info
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            self.hidden_dim = getattr(config, 'hidden_size', 4096)
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            self.num_heads = getattr(config, 'num_attention_heads', 32)
            self.head_dim = self.hidden_dim // self.num_heads
            
            self._is_loaded = True
            
            mode_desc = {
                "prompt_prefix": "Prompt Prefix (偏好作为 prompt 前缀, 无 HF 模型)",
                "hf_kv": "HF KV (HF 模型按需加载, 兼容模式)",
            }
            logger.info(
                f"vLLM adapter loaded: {self.model_name} — "
                f"{mode_desc.get(self.effective_injection_mode, self.effective_injection_mode)}"
            )
            
        except ImportError as e:
            logger.error(f"vLLM not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    def _load_hf_model(self) -> None:
        """
        Load HuggingFace model for K/V computation (hf_kv mode only).
        
        在 prompt_prefix 模式下调用此方法会发出警告。
        
        When vLLM already occupies most GPU memory, loading a second full model
        on GPU will cause CUDA OOM. This method checks available GPU memory first:
        - If sufficient (>= 4GB free), load with device_map="auto" (GPU)
        - Otherwise, load to CPU to avoid OOM (slower but safe)
        """
        if self.hf_model is not None:
            return
        
        if self.effective_injection_mode == "prompt_prefix":
            logger.warning(
                "prompt_prefix 模式下不需要 HF 模型。"
                "如果需要 compute_kv，请切换到 injection_mode='hf_kv'。"
                "当前调用将被安全忽略。"
            )
            return
        
        from transformers import AutoModelForCausalLM
        
        # Check available GPU memory to decide device placement
        use_cpu = False
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0]
                free_gb = free_mem / (1024 ** 3)
                if free_gb < 4.0:
                    logger.warning(
                        f"Only {free_gb:.1f}GB free GPU memory, "
                        f"loading HF model on CPU to avoid OOM"
                    )
                    use_cpu = True
                else:
                    logger.info(
                        f"Loading HF model for K/V computation on GPU "
                        f"({free_gb:.1f}GB free): {self.model_name}"
                    )
            except Exception:
                use_cpu = True
        else:
            use_cpu = True
        
        try:
            if use_cpu:
                logger.info(
                    f"Loading HF model for K/V computation on CPU: {self.model_name}"
                )
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # CPU uses float32
                    device_map="cpu",
                    trust_remote_code=self.trust_remote_code,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    attn_implementation="eager",
                )
            self.hf_model.eval()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning(
                    f"GPU OOM loading HF model, retrying on CPU: {e}"
                )
                torch.cuda.empty_cache()
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=self.trust_remote_code,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                )
                self.hf_model.eval()
            else:
                raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """Generate text using vLLM."""
        if not self._is_loaded:
            self.load()
        
        from vllm import SamplingParams
        
        start_time = time.perf_counter()
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        end_time = time.perf_counter()
        
        return ModelOutput(
            text=output.outputs[0].text,
            tokens=list(output.outputs[0].token_ids),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(output.outputs[0].token_ids),
        )
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Get embeddings using HuggingFace model.
        
        注意: prompt_prefix 模式下此方法不可用，将抛出 RuntimeError。
        建议使用独立的 embedding 服务替代。
        """
        if self.effective_injection_mode == "prompt_prefix":
            raise RuntimeError(
                "embed() 在 prompt_prefix 模式下不可用 (无 HF 模型)。"
                "建议: 使用独立的 embedding 服务 (如 sentence-transformers)，"
                "或切换到 injection_mode='hf_kv'。"
            )
        
        self._load_hf_model()
        
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.hf_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # Use last hidden state mean as embedding
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1).detach().cpu()
        
        # Free GPU memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embeddings
    
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute K/V cache using HuggingFace model.
        
        prompt_prefix 模式下:
        - 返回空列表 ([], None)，因为偏好通过 prompt 前缀注入，不需要 KV 计算
        - 这是安全的降级行为，Executor 在 prompt_prefix 模式下不会使用返回值
        
        hf_kv 模式下:
        - 原有行为: 使用 HF 模型计算全层 KV cache
        
        GPU Memory Safety:
        - Detaches K/V tensors and moves them to CPU to avoid GPU memory accumulation
        - Clears intermediate computation tensors
        - This is critical when vLLM already occupies most GPU memory
        """
        if self.effective_injection_mode == "prompt_prefix":
            logger.debug(
                "compute_kv() called in prompt_prefix mode — "
                "returning empty KV (偏好通过 prompt 前缀注入)"
            )
            return [], None
        
        self._load_hf_model()
        
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.hf_model(
                **inputs,
                output_hidden_states=return_hidden,
                use_cache=True,
                return_dict=True,
            )
        
        # Extract K/V cache — detach and move to CPU to prevent GPU memory leak
        kv_entries = []
        past_kv = outputs.past_key_values
        
        for layer_idx, (key, value) in enumerate(past_kv):
            entry = KVCacheEntry(
                key=key.detach().cpu(),
                value=value.detach().cpu(),
                layer_idx=layer_idx,
            )
            kv_entries.append(entry)
        
        hidden_states = None
        if return_hidden and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1].detach().cpu()
        
        # Explicitly delete outputs to free GPU memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return kv_entries, hidden_states
    
    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> ModelOutput:
        """
        Generate with K/V injection — 根据 injection_mode 路由.
        
        prompt_prefix 模式:
        - prompt 已包含偏好前缀 (由 Planner 组装)
        - injected_kv 参数被忽略
        - 直接调用 vLLM generate，完整利用 vLLM 性能
        
        hf_kv 模式:
        - 原有行为: 使用 HF 模型 + past_key_values 注入
        
        Args:
            prompt: 用户输入 (prompt_prefix 模式下已包含偏好前缀)
            injected_kv: 预计算的 KV cache (仅 hf_kv 模式使用)
            alpha: 注入强度 (prompt_prefix 模式下通过前缀长度控制)
            max_new_tokens: 最大生成 token 数
        """
        if self.effective_injection_mode == "prompt_prefix":
            return self._forward_with_prompt_prefix(
                prompt=prompt,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        else:
            return self._forward_with_hf_kv(
                prompt=prompt,
                injected_kv=injected_kv,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
    
    # ================================================================
    # Prompt Prefix 模式 — 偏好通过 prompt 前缀注入
    # ================================================================
    
    def _forward_with_prompt_prefix(
        self,
        prompt: str,
        alpha: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> ModelOutput:
        """
        Prompt Prefix 模式: 偏好已由 Planner 组装到 prompt 中.
        
        在此模式下:
        - prompt 参数就是 plan.final_input，已包含偏好前缀 + 历史后缀 + 用户查询
        - 直接调用 vLLM generate，vLLM 在 prefill 阶段自然处理偏好信息
        - 偏好的 KV 自然进入 vLLM 的 PagedAttention KV Cache
        - 无需 HF 模型，无需 compute_kv，无需 past_key_values
        
        Alpha 控制:
        - 由 Planner 在构造前缀时实现 (前缀长度控制 + 权重标记)
        - alpha > 0.5: 完整偏好前缀
        - alpha 0.3-0.5: 精简偏好前缀 (只保留高优先级)
        - alpha < 0.3: 极简偏好标记
        - alpha < 0.1: 不添加前缀 (由 Planner 决定)
        """
        if not self._is_loaded:
            self.load()
        
        from vllm import SamplingParams
        
        start_time = time.perf_counter()
        
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=max_new_tokens,
        )
        
        # prompt 已包含偏好前缀 (由 Planner 组装)
        outputs = self.llm.generate([prompt], sampling_params)
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
                'injection_mode': 'prompt_prefix',
            },
        )
    
    # ================================================================
    # HF KV 模式 — 原有行为 (兼容)
    # ================================================================
    
    def _forward_with_hf_kv(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> ModelOutput:
        """
        HF KV 模式: 使用 HuggingFace 模型 + past_key_values 注入.
        
        这是原有的 forward_with_kv_injection 实现，完整保留。
        Supports FlashAttention optimization when enabled.
        """
        self._load_hf_model()
        
        start_time = time.perf_counter()
        
        inputs = self.tokenize(prompt)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get memory length
        mem_len = injected_kv[0].key.shape[2] if injected_kv else 0
        
        # Use FlashAttention optimized path if enabled
        if self.flash_attn_enabled and self._kv_injection_optimizer is not None:
            return self._forward_with_flash_attention(
                input_ids=input_ids,
                attention_mask=attention_mask,
                injected_kv=injected_kv,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                start_time=start_time,
                **kwargs
            )
        
        # Determine target device for HF model inference
        hf_device = next(self.hf_model.parameters()).device
        
        # Standard path: Prepare past_key_values with alpha scaling
        if alpha < 1.0:
            past_kv = tuple(
                (entry.key.to(hf_device), (entry.value * alpha).to(hf_device))
                for entry in injected_kv
            )
        else:
            past_kv = tuple(
                (entry.key.to(hf_device), entry.value.to(hf_device))
                for entry in injected_kv
            )
        
        # Move input tensors to HF model's device
        input_ids = input_ids.to(hf_device)
        attention_mask = attention_mask.to(hf_device)
        
        # Extend attention mask for injected K/V
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=hf_device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        # Generate with injected K/V
        with torch.no_grad():
            generated = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        end_time = time.perf_counter()
        
        # Decode output
        new_tokens = generated[0][input_ids.shape[1]:]
        output_text = self.decode(new_tokens.cpu())
        output_token_list = new_tokens.cpu().tolist()
        input_len = input_ids.shape[1]
        
        # Cleanup: free GPU memory from intermediate tensors
        del past_kv, generated, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ModelOutput(
            text=output_text,
            tokens=output_token_list,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_len,
            output_tokens=len(output_token_list),
            metadata={'alpha': alpha, 'mem_len': mem_len, 'flash_attn': False},
        )
    
    def _forward_with_flash_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        max_new_tokens: int,
        start_time: float,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass with K/V injection (FlashAttention enabled path).
        Only used in hf_kv mode.
        """
        mem_len = injected_kv[0].key.shape[2] if injected_kv else 0
        
        # Determine target device for HF model inference
        hf_device = next(self.hf_model.parameters()).device
        
        # Move KV tensors to HF model's device and apply alpha scaling
        if alpha < 1.0:
            past_kv = tuple(
                (entry.key.to(hf_device), (entry.value * alpha).to(hf_device))
                for entry in injected_kv
            )
        else:
            past_kv = tuple(
                (entry.key.to(hf_device), entry.value.to(hf_device))
                for entry in injected_kv
            )
        
        # Move input tensors to HF model's device
        input_ids = input_ids.to(hf_device)
        attention_mask = attention_mask.to(hf_device)
        
        # Extend attention mask for injected K/V
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=hf_device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        # Generate with injected K/V
        with torch.no_grad():
            generated = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        end_time = time.perf_counter()
        
        # Decode output
        new_tokens = generated[0][input_ids.shape[1]:]
        output_text = self.decode(new_tokens.cpu())
        output_token_list = new_tokens.cpu().tolist()
        input_len = input_ids.shape[1]
        
        # Cleanup: free GPU memory from intermediate tensors
        del past_kv, generated, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ModelOutput(
            text=output_text,
            tokens=output_token_list,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_len,
            output_tokens=len(output_token_list),
            metadata={
                'alpha': alpha,
                'mem_len': mem_len,
                'flash_attn': True,
                'flash_attn_backend': self._flash_attn_backend,
            },
        )
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute prefill-stage entropy for gating.
        
        prompt_prefix 模式下:
        - 返回默认值 0.5 (中等熵)
        - 偏好注入不依赖熵门控，而是由 Planner 的 alpha 控制
        
        hf_kv 模式下:
        - 原有行为: 使用 HF 模型计算 attention entropy
        """
        if self.effective_injection_mode == "prompt_prefix":
            logger.debug(
                "compute_prefill_entropy() in prompt_prefix mode — "
                "returning default 0.5"
            )
            return 0.5
        
        # Pre-check: if CUDA memory is very low, skip computation entirely
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0]
                if free_mem < 256 * 1024 * 1024:
                    logger.debug(
                        f"Skipping prefill entropy: only {free_mem / 1024 / 1024:.0f}MB free GPU memory"
                    )
                    return 0.5
            except Exception:
                pass
        
        self._load_hf_model()
        
        if self.hf_model is None:
            return 0.5
        
        inputs = self.tokenize(text)
        
        try:
            with torch.no_grad():
                outputs = self.hf_model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )
            
            if outputs.attentions is None:
                logger.warning("Attention outputs not available, returning default entropy")
                return 0.5
            
            if layer_idx >= len(outputs.attentions):
                layer_idx = len(outputs.attentions) - 1
                logger.warning(f"Layer index out of range, using layer {layer_idx}")
            
            attn_weights = outputs.attentions[layer_idx]
            attn_weights = attn_weights.clamp(min=1e-9)
            per_row_entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)
            entropy = per_row_entropy.mean()
            
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.5
    
    # ================================================================
    # 模型信息与诊断
    # ================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information with injection mode."""
        info = super().get_model_info()
        info.update({
            'injection_mode': self.injection_mode,
            'effective_injection_mode': self.effective_injection_mode,
            'hf_model_loaded': self.hf_model is not None,
            'vllm_engine_loaded': self.llm is not None,
        })
        return info
    
    def unload(self) -> None:
        """Unload models."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.hf_model is not None:
            del self.hf_model
            self.hf_model = None
        
        super().unload()
