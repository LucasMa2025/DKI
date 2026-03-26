"""
LLaMA Model Adapter for DKI System — 真正的 K/V 注入实现
HuggingFace Transformers-based adapter for LLaMA models (单模型, 无引擎)

============================================================================
⚠️  支持范围声明 — 重要
============================================================================
本实验系统 **仅支持 LLaMA 3.1 系列模型**:
  - meta-llama/Llama-3.1-8B-Instruct
  - meta-llama/Llama-3.1-70B-Instruct
  - meta-llama/Llama-3.1-405B-Instruct

不支持以下模型，传入不支持的模型名会在 load() 时抛出 ValueError:
  - LLaMA 3.2 / 3.3 及以上版本
  - 其他模型族: Qwen / Mistral / Gemma / DeepSeek 等任何非 LLaMA 3.1 模型

如需扩展到其他模型族，需重新设计 compute_kv 的语义空间对齐策略
(参见设计文档 §3 — 多模型族 KV 注入的结构性问题)。

============================================================================
核心设计 (论文 §4.2 — 偏好 K/V 注入)
============================================================================
- 偏好文本通过 model.forward(use_cache=True) 计算 K/V 表示
- K/V 通过 past_key_values 注入 self-attention 层
- 偏好 KV 占据负位置 (position 0..n_pref-1), 查询从 n_pref 开始
- 仅缩放 Value (Key 永远不缩放, 保护 attention 寻址精度)
- 通过 attention_mask 的 float bias 实现 B_alpha (论文公式)
- alpha=0 时退化为 vanilla LLM (安全保证)

负位置注入可行性分析:
============================================================================
HuggingFace LLaMA 的 RoPE 实现:
1. compute_kv 阶段: 偏好文本 forward → Key 被 RoPE 编码为 position [0, 1, ..., n_pref-1]
2. 注入阶段: past_key_values 传入 → 查询 token 的 position_ids 从 n_pref 开始
3. 效果: 偏好 KV 占据序列最前端 (概念上的"负位置"), 对查询产生隐式背景影响
4. 安全性: 偏好通常 <100 token, 在 RoPE 训练分布内, OOD 风险极低

与 DKI 论文的一致性:
- §4.2 公式: Attn_DKI = Softmax(Q[K_p;K_u]^T/√d + B_alpha) [α·V_p; V_u]
- B_alpha 通过 float attention mask 实现 (偏好位置 += log(alpha))
- 仅缩放 Value: V_aug = [α·V_p; V_u]
- Key 不缩放: K_aug = [K_p; K_u] (保护 attention addressing)

与 Executor 的兼容性:
- compute_kv(text) → List[KVCacheEntry]: Executor._get_preference_kv 调用
- forward_with_kv_injection(prompt, injected_kv, alpha) → ModelOutput: Executor 调用
- 无 injection_mode 属性 → Executor._is_prompt_prefix_mode() 返回 False → 走 HF KV 路径

Chat Template (LLaMA 3.1 单一路径):
============================================================================
- compute_kv: 偏好文本裸文本直接 tokenize (无 template wrapper)
  → 偏好 KV 作为纯语义背景, 不与推理序列的 system/user/assistant 结构冲突
- 推理: tokenizer.apply_chat_template 优先, 回退到手动 LLaMA 3 格式
- Stop: <|eot_id|> + <|end_of_text|> + EOS

KV 注入实现 (HF 5.x 稳定路径):
============================================================================
不使用 model.generate(past_key_values=...) 路径。
HF 5.x 内部引入 cache_position，会覆盖外部传入的 position_ids，
导致查询 token 的 RoPE 位置与偏好 KV 冲突，注入效果不可预测。

统一走手动 prefill + decode loop (_forward_with_bias_impl):
  1. model.forward(input_ids, past_kv=pref_kv, attention_mask=4D_bias) → prefill
  2. 逐 token 自回归 decode, 每步手动构造 attention mask
完全绕开 HF 内部推断逻辑，HF 4.x / 5.x 均稳定。

Author: AGI Demo Project
"""

import asyncio
import math
import time
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import (
    BaseModelAdapter, ModelOutput, KVCacheEntry,
    extract_kv_from_past, build_dynamic_cache_from_entries,
)

# ============ HuggingFace Hub 兼容性补丁 ============
# 必须在 import transformers 之前调用
# 解决 huggingface_hub ≥0.25 移除 is_offline_mode 导致导入失败
from dki.models.hf_compat import ensure_hf_compat
ensure_hf_compat()

# DynamicCache 兼容: Transformers >= v4.45 弃用 tuple past_key_values
# 必须使用 DynamicCache 实例, 否则 model.generate() 会失败
try:
    from transformers import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False


def _get_cache_seq_length(cache) -> int:
    """
    获取 KV cache 的序列长度 (兼容 Transformers 4.x/5.x)。

    - DynamicCache (5.x): 通过 layers[0].keys.shape[2] 或 get_seq_length() 获取
    - DynamicCache (4.x): 通过 get_seq_length() 或 key_cache[0].shape[2] 获取
    - Legacy tuple: 通过 cache[0][0].shape[2] 获取
    """
    if _HAS_DYNAMIC_CACHE and isinstance(cache, DynamicCache):
        if hasattr(cache, 'get_seq_length'):
            try:
                return cache.get_seq_length()
            except Exception:
                pass
        if hasattr(cache, 'layers') and cache.layers:
            layer0 = cache.layers[0]
            if hasattr(layer0, 'keys'):
                return layer0.keys.shape[2]
        if hasattr(cache, 'key_cache') and cache.key_cache:
            return cache.key_cache[0].shape[2]
    if isinstance(cache, (tuple, list)) and cache:
        return cache[0][0].shape[2]
    return 0


# ============================================================================
# 支持的模型关键词白名单 (仅 LLaMA 3.1)
# ============================================================================
_LLAMA31_NAME_KEYWORDS = (
    'llama-3.1',
    'llama3.1',
    'meta-llama/meta-llama-3.1',
    'meta-llama/llama-3.1',
)


class LlamaAdapter(BaseModelAdapter):
    """
    LLaMA 3.1 专用 adapter — 真正的 K/V 注入 (非提示词前缀).

    ⚠️  本实验系统仅支持 LLaMA 3.1 系列:
        meta-llama/Llama-3.1-{8B,70B,405B}-Instruct

    不支持其他模型族 (Qwen / Mistral / Gemma / DeepSeek 等)，
    也不支持 LLaMA 3.2 / 3.3 及以上版本。
    传入不支持的模型名会在 load() 时抛出 ValueError。

    核心机制:
    1. compute_kv(): 将偏好文本 (裸文本) 编码为 K/V 表示 (每层一个 KVCacheEntry)
    2. forward_with_kv_injection(): 将偏好 K/V 注入 attention 层, 配合 alpha 门控
       → 内部走手动 prefill + decode loop (_forward_with_bias_impl)
       → 完全绕开 HF generate() 的 position_ids/cache_position 内部推断
       → HF 4.x / 5.x 均稳定
    3. generate(): 标准推理 (无注入, 用于 Executor 的无注入降级路径)

    与 Executor 的交互:
    - Executor._is_prompt_prefix_mode() 返回 False (无 injection_mode 属性)
    - Executor._get_preference_kv() 调用 compute_kv() 获取偏好 KV
    - Executor._execute_with_kv_injection() 调用 forward_with_kv_injection()
    - 历史消息由 recall_v4 组装为 suffix prompt, 包含在 plan.final_input 中
    """

    # ================================================================
    # 安全常量
    # ================================================================
    ALPHA_OVERRIDE_CAP = 0.7    # alpha 安全上限 (论文 §4.2)
    MAX_PREF_TOKENS = 200       # 偏好最大 token 数 (超出则截断, 论文 §7.2 OOD 风险)
    DEFAULT_ENTROPY = 0.5       # attention 熵默认值 (计算失败时的安全降级)

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        quantization: str = "none",
        quantization_config: dict = None,
        **kwargs
    ):
        # 向后兼容: load_in_8bit=True 映射到 quantization="8bit"
        if load_in_8bit and (not quantization or quantization == "none"):
            quantization = "8bit"

        super().__init__(
            model_name, device, dtype,
            quantization=quantization,
            quantization_config=quantization_config,
            **kwargs
        )

        self.load_in_8bit = load_in_8bit or self.is_8bit
        self.trust_remote_code = trust_remote_code

    # ================================================================
    # 模型加载
    # ================================================================

    def load(self) -> None:
        """
        Load model and tokenizer.

        ⚠️  仅接受 LLaMA 3.1 系列模型名，其他模型名抛出 ValueError。

        支持量化模式:
        - "none": 原始精度 (dtype)
        - "4bit": 4-bit NF4 量化 (bitsandbytes)
        - "8bit": 8-bit LLM.int8() 量化 (bitsandbytes)
        - "gptq": GPTQ 预量化模型
        - "awq":  AWQ 预量化模型

        使用 eager attention 以支持 output_attentions (熵计算需要)。
        """
        if self._is_loaded:
            return

        # ---- 模型白名单校验 ----
        if not self._is_llama31():
            raise ValueError(
                f"[LlamaAdapter] 不支持的模型: '{self.model_name}'。\n"
                f"本实验系统仅支持 LLaMA 3.1 系列:\n"
                f"  meta-llama/Llama-3.1-8B-Instruct\n"
                f"  meta-llama/Llama-3.1-70B-Instruct\n"
                f"  meta-llama/Llama-3.1-405B-Instruct\n"
                f"不支持 LLaMA 3.2/3.3+ 及其他模型族 (Qwen/Mistral/Gemma/DeepSeek 等)。"
            )

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

            quant_desc = f" (quantization={self.quantization})" if self.is_quantized else ""
            logger.info(f"Loading model: {self.model_name}{quant_desc}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_kwargs = {
                'trust_remote_code': self.trust_remote_code,
                'device_map': 'auto',
            }

            # ============ 量化配置 ============
            if self.quantization in ("4bit", "8bit"):
                bnb_config = self._build_bnb_config()
                model_kwargs['quantization_config'] = bnb_config
                if self.is_4bit:
                    model_kwargs['torch_dtype'] = self.dtype
                logger.info(
                    f"Loading with {self.quantization} quantization "
                    f"(bitsandbytes BitsAndBytesConfig)"
                )
            elif self.quantization == "fp8":
                fp8_compute_dtype = self.quantization_config.get(
                    "fp8_compute_dtype", "bfloat16"
                )
                model_kwargs['torch_dtype'] = getattr(torch, fp8_compute_dtype, torch.bfloat16)
                logger.info(
                    f"Loading FP8 model with compute_dtype={fp8_compute_dtype}. "
                    f"For best FP8 performance, use vLLM or SGLang engine."
                )
            elif self.quantization in ("gptq", "awq"):
                model_kwargs['torch_dtype'] = self.dtype
                logger.info(
                    f"Loading pre-quantized model ({self.quantization}), "
                    f"auto-detected by AutoModelForCausalLM"
                )
            else:
                model_kwargs['torch_dtype'] = self.dtype

            # eager attention: 支持 output_attentions (熵计算) 和自定义 attention mask
            model_kwargs['attn_implementation'] = 'eager'

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self.model.eval()

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
                f"Model loaded: {self.model_name} "
                f"(layers={self.num_layers}, heads={self.num_heads}, "
                f"hidden={self.hidden_dim}, attn=eager, "
                f"quantization={self.quantization})"
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # ================================================================
    # 模型验证
    # ================================================================

    def _is_llama31(self) -> bool:
        """
        判断是否为 LLaMA 3.1 系列 (唯一支持的模型族).

        白名单关键词匹配 model_name (大小写不敏感)。
        注意: LLaMA 3.2 / 3.3 不在支持范围内。
        """
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in _LLAMA31_NAME_KEYWORDS)

    # ================================================================
    # Chat Template 处理 (LLaMA 3.1 单一路径)
    # ================================================================

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt using LLaMA 3.1 official chat template.

        优先使用 tokenizer.apply_chat_template, 回退到手动 LLaMA 3 格式:

            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>\n\n
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                logger.warning(f"apply_chat_template failed, using manual template: {e}")

        # 手动 LLaMA 3 格式回退
        parts = ["<|begin_of_text|>"]
        if system_prompt:
            parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            )
        parts.append(
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def _has_chat_template_tokens(self, text: str) -> bool:
        """检测文本是否已包含 LLaMA 3 chat template 特殊标记 (避免双重包装)"""
        return (
            '<|begin_of_text|>' in text
            or '<|start_header_id|>' in text
        )

    def _get_stop_token_ids(self) -> List[int]:
        """
        获取 LLaMA 3.1 专属 stop token IDs.

        LLaMA 3.1: <|eot_id|>  +  <|end_of_text|>  +  EOS
        """
        if self.tokenizer is None:
            return []

        stop_ids: List[int] = []

        def _add_token(token_str: str) -> None:
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if (
                tid is not None
                and tid != self.tokenizer.unk_token_id
                and tid not in stop_ids
            ):
                stop_ids.append(tid)

        _add_token("<|eot_id|>")
        _add_token("<|end_of_text|>")

        # 通用 EOS 保底
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id not in stop_ids:
            stop_ids.append(eos_id)

        logger.debug(f"Stop token IDs for {self.model_name}: {stop_ids}")
        return stop_ids

    def _format_prompt_safe(self, prompt: str) -> str:
        """
        安全地格式化 prompt (检测已有 template 标记, 避免双重包装).

        用于 generate() 和 forward_with_kv_injection()。
        """
        if self._has_chat_template_tokens(prompt):
            return prompt
        return self._format_prompt(prompt)

    # ================================================================
    # K/V 计算 (偏好编码 — 论文 §4.2)
    # ================================================================

    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute K/V cache for preference text (论文 §4.2).

        LLaMA 3.1 策略: 偏好文本裸文本直接 tokenize → forward.
        不使用任何 template 包装，偏好 KV 作为纯语义背景注入，
        不与推理序列的 system/user/assistant turn 结构冲突。

        安全措施:
        - 偏好 token 数超过 MAX_PREF_TOKENS 时截断 (OOD 风险控制, 论文 §7.2)
        - KV tensor detach + 移至 CPU (防止 GPU 内存泄漏)
        - 显式删除 outputs 并清空 CUDA cache

        Args:
            text: 偏好文本 (如 "<preference:general>\\n- 名字: Lucas\\n</preference:general>")
            return_hidden: 是否返回最后一层 hidden states

        Returns:
            (kv_entries, hidden_states):
            - kv_entries: List[KVCacheEntry], 每层一个
            - hidden_states: Optional[torch.Tensor]
        """
        if not self._is_loaded:
            self.load()

        inputs = self.tokenize(text)
        input_ids = inputs['input_ids']

        # 安全截断
        n_tokens = input_ids.shape[1]
        if n_tokens > self.MAX_PREF_TOKENS:
            logger.warning(
                f"Preference text too long ({n_tokens} tokens > {self.MAX_PREF_TOKENS}), "
                f"truncating to reduce OOD risk (论文 §7.2)"
            )
            input_ids = input_ids[:, :self.MAX_PREF_TOKENS]
            inputs['attention_mask'] = inputs['attention_mask'][:, :self.MAX_PREF_TOKENS]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                output_hidden_states=return_hidden,
                use_cache=True,
                return_dict=True,
            )

        # Extract K/V cache — detach and move to CPU
        kv_entries = []
        past_kv = outputs.past_key_values

        kv_pairs = extract_kv_from_past(past_kv)
        for layer_idx, (key_tensor, value_tensor) in enumerate(kv_pairs):
            entry = KVCacheEntry(
                key=key_tensor.detach().cpu(),
                value=value_tensor.detach().cpu(),
                layer_idx=layer_idx,
            )
            kv_entries.append(entry)

        hidden_states = None
        if return_hidden and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1].detach().cpu()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[LlamaAdapter] 偏好 KV 计算完成: tokens={n_tokens}, "
            f"layers={len(kv_entries)}, "
            f"shape={kv_entries[0].key.shape if kv_entries else 'N/A'}"
        )
        return kv_entries, hidden_states

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
        标准推理 (无 K/V 注入).

        Executor 在 alpha 太低或无偏好 KV 时调用此方法。
        """
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._get_stop_token_ids(),
            )

        end_time = time.perf_counter()

        new_tokens = outputs[0][input_ids.shape[1]:]
        output_text = self.decode(new_tokens)

        return ModelOutput(
            text=output_text,
            tokens=new_tokens.tolist(),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
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
        Async streaming generation using HuggingFace TextIteratorStreamer.

        使用 transformers.TextIteratorStreamer 实现真正的 token-by-token streaming:
        - model.generate() 在后台线程中运行
        - TextIteratorStreamer 将 token 推送到队列
        - 主协程从队列中异步读取并 yield

        Yields:
            str: decoded text chunks (token-by-token)
        """
        if not self._is_loaded:
            self.load()

        from transformers import TextIteratorStreamer

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self._get_stop_token_ids(),
            'streamer': streamer,
        }

        thread = threading.Thread(
            target=lambda: self.model.generate(**generation_kwargs),
            daemon=True,
        )
        thread.start()

        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk
            await asyncio.sleep(0)

        thread.join(timeout=5)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Synchronous streaming generation using HuggingFace TextIteratorStreamer.

        Yields:
            str: decoded text chunks (token-by-token)
        """
        if not self._is_loaded:
            self.load()

        from transformers import TextIteratorStreamer

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self._get_stop_token_ids(),
            'streamer': streamer,
        }

        thread = threading.Thread(
            target=lambda: self.model.generate(**generation_kwargs),
            daemon=True,
        )
        thread.start()

        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk

        thread.join(timeout=5)

    # ================================================================
    # 真正的 K/V 注入推理 (论文 §4.2 — 核心)
    # ================================================================

    def _build_attention_bias(
        self,
        n_pref: int,
        n_query: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构建 attention bias 矩阵 (论文 §4.2 的 B_alpha).

        论文公式:
            Attn_DKI = Softmax(Q[K_p;K_u]^T/√d + B_alpha) [α·V_p; V_u]

        B_alpha 对偏好位置施加 log(alpha) bias:
        - alpha=1.0: bias=0,    偏好与查询等权
        - alpha=0.5: bias≈-0.69, 偏好被适度抑制
        - alpha=0.3: bias≈-1.2,  偏好被较强抑制
        - alpha=0.0: bias=-inf,  偏好被完全屏蔽

        同时保持 causal mask (下三角):
        - 查询 token 只能 attend 到自身及之前的 token
        - 所有查询 token 都能 attend 到所有偏好 token (偏好是全局背景)

        Returns:
            attention_bias: [1, 1, n_query, n_pref + n_query]
        """
        total_len = n_pref + n_query
        bias = torch.zeros(1, 1, n_query, total_len, device=device, dtype=dtype)

        if alpha <= 0:
            bias[:, :, :, :n_pref] = float('-inf')
        elif alpha < 1.0:
            bias[:, :, :, :n_pref] = math.log(max(alpha, 1e-8))
        # alpha >= 1.0: bias=0, 不修改

        # causal mask: 屏蔽 query 中 i 之后的位置
        # torch.triu 向量化构建, 无 Python 循环开销
        if n_query > 1:
            query_causal = torch.triu(
                torch.full((n_query, n_query), float('-inf'), device=device, dtype=dtype),
                diagonal=1,
            )
            bias[:, :, :, n_pref:] = query_causal

        return bias

    def _prepare_kv_for_injection(
        self,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        device: torch.device,
    ) -> Tuple[Any, int]:
        """
        准备注入用的 K/V (Value 缩放, 移至 GPU).

        论文不变量:
        - Key 永远不缩放 (保护 attention 寻址精度)
        - 仅 Value 被 alpha 缩放: V_aug = [α·V_p; V_u]
        """
        clamped_alpha = min(alpha, self.ALPHA_OVERRIDE_CAP)

        cache, mem_len = build_dynamic_cache_from_entries(
            entries=injected_kv,
            device=device,
            alpha=clamped_alpha,
        )

        logger.debug(
            f"Prepared KV cache for injection: {len(injected_kv)} layers, "
            f"mem_len={mem_len}, alpha={clamped_alpha:.2f}, "
            f"type={type(cache).__name__}"
        )
        return cache, mem_len

    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """
        真正的 K/V 注入推理 (论文 §4.2).

        实现流程:
        1. 格式化 prompt (LLaMA 3.1 chat template)
        2. 将偏好 KV 移至 GPU, Value 缩放 alpha
        3. 手动 prefill + 逐 token decode (完全掌控 attention)
        4. attention bias 精确实现 B_alpha (论文公式)

        核心设计 (HF 5.x 兼容):
        ---------------------------------------------------------------
        不使用 model.generate(past_key_values=...) 路径。
        HF 5.x 重构了 generate() 内部逻辑，引入 cache_position 并
        可能覆盖外部传入的 position_ids，导致查询 token 的 RoPE 位置
        与偏好 KV 冲突，注入效果不可预测。

        统一走 _forward_with_bias_impl (手动 prefill+decode)，
        完全绕开 HF 内部推断，HF 4.x / 5.x 均稳定。
        ---------------------------------------------------------------

        安全保证:
        - alpha 被 ALPHA_OVERRIDE_CAP (0.7) 截断
        - alpha=0 → 退化为 vanilla LLM
        - 异常时自动降级到标准 generate
        """
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        clamped_alpha = min(max(alpha, 0.0), self.ALPHA_OVERRIDE_CAP)

        if not injected_kv or clamped_alpha <= 0.01:
            logger.debug("No KV or alpha too low, falling back to standard generate")
            return self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
            )

        logger.info(
            f"[LlamaAdapter] 开始 KV 注入推理: layers={len(injected_kv)}, "
            f"alpha={clamped_alpha:.2f}, max_new_tokens={max_new_tokens}"
        )
        try:
            return self._forward_with_bias_impl(
                prompt=prompt,
                injected_kv=injected_kv,
                alpha=clamped_alpha,
                max_new_tokens=max_new_tokens,
                start_time=start_time,
                **kwargs,
            )
        except Exception as e:
            logger.warning(
                f"KV injection failed ({e}), falling back to standard generate"
            )
            return self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
            )

    def _forward_with_bias_impl(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        max_new_tokens: int,
        start_time: float,
        **kwargs,
    ) -> ModelOutput:
        """
        带 Attention Bias 的 K/V 注入核心实现 (HF 4.x/5.x 稳定路径).

        步骤:
        1. Prefill: model.forward(input_ids, past_kv=pref_kv, attention_mask=4D_bias,
                                  position_ids=[mem_len .. mem_len+n_query-1])
           → 获取完整 KV cache (pref + query)
        2. Decode: 逐 token 自回归生成 (使用完整 KV cache + stop tokens)
           → 每步传入 position_ids=[mem_len + n_query + step]

        position_ids 设计:
        ---------------------------------------------------------------
        pref KV 在 compute_kv() 时以裸文本 forward，RoPE 编码为
        position [0, 1, ..., n_pref-1]。

        若 prefill 和 decode 不显式传入 position_ids，HF 默认从 0
        开始给 query token 编号，导致 query 的 Q 与 pref 的 K 在
        相同 position 上计算 QK^T，旋转角度重叠，attention 分布
        产生系统性偏差。

        修正：query 的 position_ids 从 mem_len 开始连续递增，
        确保 pref 和 query 在同一个连续 RoPE 位置空间内，
        attention 的 QK^T 点积语义正确。
        ---------------------------------------------------------------

        Attention bias 语义 (与 _build_attention_bias 完全对齐):
        - alpha=0      → 偏好位置 -inf (完全屏蔽)
        - 0 < alpha <1 → 偏好位置 log(alpha) (适度抑制)
        - alpha >= 1   → 偏好位置 0 (等权, 不修改)
        """
        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device

        past_kv, mem_len = self._prepare_kv_for_injection(
            injected_kv, alpha, device
        )

        n_query = input_ids.shape[1]

        attention_bias = self._build_attention_bias(
            n_pref=mem_len,
            n_query=n_query,
            alpha=alpha,
            device=device,
            dtype=self.dtype,
        )

        # Prefill position_ids: query token 从 mem_len 开始，紧接 pref KV
        # shape: [1, n_query]
        prefill_position_ids = torch.arange(
            mem_len, mem_len + n_query,
            device=device, dtype=torch.long,
        ).unsqueeze(0)

        # ---- Prefill ----
        with torch.no_grad():
            prefill_outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                attention_mask=attention_bias,
                position_ids=prefill_position_ids,
                use_cache=True,
                return_dict=True,
            )

        full_kv = prefill_outputs.past_key_values
        next_logits = prefill_outputs.logits[:, -1, :]
        del prefill_outputs

        # ---- Decode (逐 token 自回归) ----
        generated_ids = []
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        stop_ids = set(self._get_stop_token_ids())

        for step in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[sorted_mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)]
            else:
                next_token = next_logits.argmax(dim=-1)[0]

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if next_token_id in stop_ids:
                break

            next_input = torch.tensor([[next_token_id]], device=device)

            # Decode position_ids: 当前 token 在整个序列中的绝对位置
            # = pref(mem_len) + query(n_query) + 已生成(step)
            # shape: [1, 1]
            decode_position_ids = torch.tensor(
                [[mem_len + n_query + step]],
                device=device, dtype=torch.long,
            )

            total_past = _get_cache_seq_length(full_kv)
            decode_mask = torch.zeros(
                1, 1, 1, total_past + 1, device=device, dtype=self.dtype
            )
            # 与 prefill 阶段的 _build_attention_bias 保持语义一致
            if alpha <= 0:
                decode_mask[:, :, :, :mem_len] = float('-inf')
            elif alpha < 1.0:
                decode_mask[:, :, :, :mem_len] = math.log(max(alpha, 1e-8))
            # alpha >= 1.0: decode_mask 全零, 不修改

            with torch.no_grad():
                decode_outputs = self.model(
                    input_ids=next_input,
                    past_key_values=full_kv,
                    attention_mask=decode_mask,
                    position_ids=decode_position_ids,
                    use_cache=True,
                    return_dict=True,
                )

            full_kv = decode_outputs.past_key_values
            next_logits = decode_outputs.logits[:, -1, :]
            del decode_outputs

        end_time = time.perf_counter()
        output_text = self.decode(generated_ids)

        del full_kv, past_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ModelOutput(
            text=output_text,
            tokens=generated_ids,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=n_query,
            output_tokens=len(generated_ids),
            metadata={
                'alpha': alpha,
                'mem_len': mem_len,
                'injection_mode': 'manual_prefill_decode_bias',
                'attention_bias_applied': True,
                'position_ids_explicit': True,
            },
        )

    def forward_with_kv_injection_and_bias(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """
        带 Attention Bias 的 K/V 注入推理 (论文 §4.2 完整公式).

        forward_with_kv_injection 与本方法共用同一底层实现
        (_forward_with_bias_impl)，两者等价。
        本方法保留供向后兼容及直接调用。
        """
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        clamped_alpha = min(max(alpha, 0.0), self.ALPHA_OVERRIDE_CAP)

        if not injected_kv or clamped_alpha <= 0.01:
            return self.generate(prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)

        try:
            return self._forward_with_bias_impl(
                prompt, injected_kv, clamped_alpha, max_new_tokens, start_time, **kwargs
            )
        except Exception as e:
            logger.warning(f"KV injection with bias failed ({e}), falling back")
            return self.generate(prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)

    # ================================================================
    # Embedding
    # ================================================================

    def embed(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        if not self._is_loaded:
            self.load()

        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1).detach().cpu()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings

    # ================================================================
    # Prefill 熵计算 (论文 §4.2 — 双因子门控)
    # ================================================================

    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute prefill-stage attention entropy (论文 §4.2 门控信号).

        高熵 = 模型不确定 → 更需要偏好注入 (alpha 更高)
        低熵 = 模型确定   → 偏好注入可减弱 (alpha 更低)

        Args:
            text: 输入文本
            layer_idx: 使用哪一层的 attention (默认第 3 层)

        Returns:
            归一化 attention 熵 (scalar)
        """
        if not self._is_loaded:
            self.load()

        inputs = self.tokenize(text)

        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

            if outputs.attentions is None:
                logger.warning("Attention outputs not available, returning default entropy")
                return self.DEFAULT_ENTROPY

            layer_idx = min(layer_idx, len(outputs.attentions) - 1)
            attn_weights = outputs.attentions[layer_idx]  # [batch, heads, seq_q, seq_k]

            attn_weights = attn_weights.clamp(min=1e-9)
            per_row_entropy = -torch.sum(
                attn_weights * torch.log(attn_weights), dim=-1
            )  # [batch, heads, seq_q]
            entropy = per_row_entropy.mean()

            return entropy.item()

        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            return self.DEFAULT_ENTROPY

    # ================================================================
    # 模型信息
    # ================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        info = super().get_model_info()
        info.update({
            'adapter_type': 'llama31_hf_kv_injection',
            'supported_model_family': 'llama3.1_only',
            'injection_mode': 'manual_prefill_decode_bias',
            'kv_injection_type': 'negative_position',
            'alpha_override_cap': self.ALPHA_OVERRIDE_CAP,
            'max_pref_tokens': self.MAX_PREF_TOKENS,
            'attention_bias_available': True,
            'load_in_8bit': self.load_in_8bit,
            'quantization': self.quantization,
        })
        return info
