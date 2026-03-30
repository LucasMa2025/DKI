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

============================================================================
修复变更日志 (本版本相对 P4 修复版)
============================================================================

[本轮-1] alpha 上限开放为 1.0，支持外部传入
  - ALPHA_OVERRIDE_CAP 默认值从 0.7 提升为 1.0
  - __init__ 新增 alpha_cap: float = 1.0 参数，允许运行时配置
  - 所有 clamp 路径统一使用 self.alpha_cap
  - alpha > 1.0 时裁剪并 WARNING（不抛异常）
  - alpha = 1.0 时偏好位置 bias = 0，等权参与 attention

[本轮-2] compute_kv 改为使用 system prompt template 包裹偏好文本
  - 新增 _wrap_pref_as_system_prompt(text) → str
    将偏好文本包裹为 LLaMA 3.1 system turn 格式:
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>\\n\\n{pref}<|eot_id|>
  - compute_kv 默认走 system template 路径 (use_system_template=True)
  - 保留 use_system_template=False 选项供消融实验对比
  - MAX_PREF_TOKENS 相应提升到 300

[本轮-3] 新增 LogitBias 注入路径 (实验性)
  - compute_pref_embedding(text) → Tensor [hidden_dim]
  - _compute_logit_bias_vector(pref_emb, device) → Tensor [vocab_size]
  - _apply_logit_bias(logits, bias_vec, lambda_) → Tensor
  - forward_with_logit_bias_injection()          — 非流式
  - stream_generate_with_logit_bias()            — 同步流式
  - async_stream_generate_with_logit_bias()      — 异步流式

Author: AGI Demo Project
"""

import asyncio
import math
import time
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import (
    BaseModelAdapter, ModelOutput, KVCacheEntry,
    extract_kv_from_past, build_dynamic_cache_from_entries,
)

from dki.models.hf_compat import ensure_hf_compat
ensure_hf_compat()

try:
    from transformers import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False

# ============================================================================
# 全局安全常量
# ============================================================================
MAX_SAFE_POSITION = 8192
MAX_ROPE_POSITION = 131072

# LLaMA 3.1 system prompt 模板头尾（手动拼接，不依赖 apply_chat_template）
_LLAMA31_SYSTEM_HEADER = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
)
_LLAMA31_SYSTEM_FOOTER = "<|eot_id|>"


def _get_cache_seq_length(cache) -> int:
    """
    获取 KV cache 的序列长度 (兼容 Transformers 4.x/5.x).

    优先级:
    1. DynamicCache.get_seq_length()
    2. DynamicCache.key_cache[0].shape[2]
    3. DynamicCache.layers[0].keys.shape[2]
    4. Legacy tuple cache[0][0].shape[2]
    """
    if _HAS_DYNAMIC_CACHE and isinstance(cache, DynamicCache):
        if hasattr(cache, 'get_seq_length'):
            try:
                seq_len = cache.get_seq_length()
                if isinstance(seq_len, int):
                    return seq_len
            except Exception:
                pass
        if hasattr(cache, 'key_cache'):
            key_cache = cache.key_cache
            if key_cache and len(key_cache) > 0 and key_cache[0] is not None:
                return key_cache[0].shape[2]
        if hasattr(cache, 'layers') and cache.layers:
            layer0 = cache.layers[0]
            if hasattr(layer0, 'keys') and layer0.keys is not None:
                return layer0.keys.shape[2]
    if isinstance(cache, (tuple, list)) and cache:
        try:
            return cache[0][0].shape[2]
        except (IndexError, AttributeError):
            pass
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

    核心方法:
    - compute_kv()                                 : 偏好 (system template) → K/V
    - compute_pref_embedding()                     : 偏好 → 嵌入向量 [本轮新增]
    - forward_with_kv_injection()                  : KV 注入推理 (非流式)
    - forward_with_logit_bias_injection()          : LogitBias 注入 (非流式) [本轮新增]
    - stream_generate_with_kv_injection()          : KV 注入 (同步流式)
    - async_stream_generate_with_kv_injection()    : KV 注入 (异步流式)
    - stream_generate_with_logit_bias()            : LogitBias 注入 (同步流式) [本轮新增]
    - async_stream_generate_with_logit_bias()      : LogitBias 注入 (异步流式) [本轮新增]
    - generate() / stream_generate()               : 标准推理 (无注入)
    """

    ALPHA_OVERRIDE_CAP: float = 1.0    # [本轮-1] 从 0.7 提升为 1.0
    MAX_PREF_TOKENS: int = 300          # [本轮-2] 从 200 提升（system template 约占 10 tokens）
    DEFAULT_ENTROPY: float = 0.5
    DEFAULT_LOGIT_BIAS_LAMBDA: float = 0.1

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        quantization: str = "none",
        quantization_config: dict = None,
        alpha_cap: float = 1.0,        # [本轮-1] 支持外部传入 alpha 上限
        **kwargs
    ):
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

        # [本轮-1] 实例级 alpha 上限
        if alpha_cap > 1.0:
            logger.warning(
                f"[LlamaAdapter] alpha_cap={alpha_cap} > 1.0，已裁剪为 1.0。"
                f" alpha > 1.0 会使 log(alpha) > 0，反向放大偏好 attention，"
                f"语义不合理，如需实验请直接修改 ALPHA_OVERRIDE_CAP 常量。"
            )
            alpha_cap = 1.0
        self.alpha_cap = alpha_cap

    # ================================================================
    # 模型加载
    # ================================================================

    def load(self) -> None:
        """Load model and tokenizer. 仅接受 LLaMA 3.1 系列。"""
        if self._is_loaded:
            return

        if not self._is_llama31():
            raise ValueError(
                f"[LlamaAdapter] 不支持的模型: '{self.model_name}'。\n"
                f"本实验系统仅支持 LLaMA 3.1 系列:\n"
                f"  meta-llama/Llama-3.1-8B-Instruct\n"
                f"  meta-llama/Llama-3.1-70B-Instruct\n"
                f"  meta-llama/Llama-3.1-405B-Instruct\n"
                f"不支持 LLaMA 3.2/3.3+ 及其他模型族。"
            )

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

            quant_desc = f" (quantization={self.quantization})" if self.is_quantized else ""
            logger.info(f"Loading model: {self.model_name}{quant_desc}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                'trust_remote_code': self.trust_remote_code,
                'device_map': 'auto',
            }

            if self.quantization in ("4bit", "8bit"):
                bnb_config = self._build_bnb_config()
                model_kwargs['quantization_config'] = bnb_config
                if self.is_4bit:
                    model_kwargs['torch_dtype'] = self.dtype
            elif self.quantization == "fp8":
                fp8_dtype = (
                    self.quantization_config.get("fp8_compute_dtype", "bfloat16")
                    if self.quantization_config else "bfloat16"
                )
                model_kwargs['torch_dtype'] = getattr(torch, fp8_dtype, torch.bfloat16)
            elif self.quantization in ("gptq", "awq"):
                model_kwargs['torch_dtype'] = self.dtype
            else:
                model_kwargs['torch_dtype'] = self.dtype

            model_kwargs['attn_implementation'] = 'eager'

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs,
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
                f"quantization={self.quantization}, alpha_cap={self.alpha_cap})"
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
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in _LLAMA31_NAME_KEYWORDS)

    # ================================================================
    # [本轮-2] System Prompt Template 包裹
    # ================================================================

    def _wrap_pref_as_system_prompt(self, pref_text: str) -> str:
        """
        [本轮-2] 将偏好文本包裹为 LLaMA 3.1 system prompt 格式.

        包裹前 (裸文本):
            "用户名: Lucas，偏好: 简洁回答，语言: 中文"

        包裹后 (system template):
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n
             用户名: Lucas，偏好: 简洁回答，语言: 中文<|eot_id|>"

        语义意义:
        LLaMA 3.1 的 RLHF/SFT 训练数据中，system 指令始终以该格式呈现。
        裸文本进入 forward() 时，特殊 token 的 embedding（如 <|start_header_id|>）
        会激活模型"我在处理 system 级指令"的内部表示，使 KV 进入正确的语义子空间。
        这是解决 KV 注入偏好不被识别的最关键修复之一。

        注意:
        - 手动拼接（不用 apply_chat_template），因为后者需要完整 messages 列表
        - 不包含 user/assistant turn
        - 用于 compute_kv 和 compute_pref_embedding，不用于 query prompt 格式化

        Args:
            pref_text: 原始偏好文本（裸文本，不含 template 标记）

        Returns:
            包裹后的 system prompt 字符串
        """
        return f"{_LLAMA31_SYSTEM_HEADER}{pref_text}{_LLAMA31_SYSTEM_FOOTER}"

    # ================================================================
    # Chat Template 处理
    # ================================================================

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using LLaMA 3.1 official chat template."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception as e:
                logger.warning(f"apply_chat_template failed, using manual template: {e}")

        parts = ["<|begin_of_text|>"]
        if system_prompt:
            parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            )
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def _has_chat_template_tokens(self, text: str) -> bool:
        return '<|begin_of_text|>' in text or '<|start_header_id|>' in text

    def _get_stop_token_ids(self) -> List[int]:
        if self.tokenizer is None:
            return []
        stop_ids: List[int] = []

        def _add(token_str: str) -> None:
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id and tid not in stop_ids:
                stop_ids.append(tid)

        _add("<|eot_id|>")
        _add("<|end_of_text|>")
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id not in stop_ids:
            stop_ids.append(eos_id)
        return stop_ids

    def _format_prompt_safe(self, prompt: str) -> str:
        if self._has_chat_template_tokens(prompt):
            return prompt
        return self._format_prompt(prompt)

    # ================================================================
    # K/V 计算 (偏好编码)
    # ================================================================

    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
        use_system_template: bool = True,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute K/V cache for preference text.

        [本轮-2] 默认将偏好文本包裹为 system prompt template 后再 forward。
        这使偏好 KV 处于与 RLHF 训练数据对齐的语义分布中，
        是解决 LLaMA 3.1 无法识别注入偏好的核心修复。

        Args:
            text:                偏好文本（裸文本）
            return_hidden:       是否返回最后层 hidden states
            use_system_template: 是否包裹 system template（默认 True，推荐）
                                 False 为裸文本，仅供消融实验对比

        Returns:
            (kv_entries, hidden_states)
        """
        if not self._is_loaded:
            self.load()

        if use_system_template:
            pref_text = self._wrap_pref_as_system_prompt(text)
            logger.debug(
                f"[compute_kv] system_template=True: "
                f"{len(text)} chars → {len(pref_text)} chars (wrapped)"
            )
        else:
            pref_text = text
            logger.debug("[compute_kv] system_template=False: using raw text")

        inputs = self.tokenize(pref_text)
        input_ids = inputs['input_ids']
        n_tokens = input_ids.shape[1]

        if n_tokens > self.MAX_PREF_TOKENS:
            logger.warning(
                f"Preference text too long ({n_tokens} tokens > {self.MAX_PREF_TOKENS}), "
                f"truncating. Consider shortening preference text."
            )
            input_ids = input_ids[:, :self.MAX_PREF_TOKENS]
            inputs['attention_mask'] = inputs['attention_mask'][:, :self.MAX_PREF_TOKENS]
            n_tokens = self.MAX_PREF_TOKENS

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                output_hidden_states=return_hidden,
                use_cache=True,
                return_dict=True,
            )

        kv_entries = []
        kv_pairs = extract_kv_from_past(outputs.past_key_values)
        for layer_idx, (key_tensor, value_tensor) in enumerate(kv_pairs):
            kv_entries.append(KVCacheEntry(
                key=key_tensor.detach().cpu(),
                value=value_tensor.detach().cpu(),
                layer_idx=layer_idx,
            ))

        hidden_states = None
        if return_hidden and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1].detach().cpu()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[LlamaAdapter] 偏好 KV 计算完成: tokens={n_tokens}, "
            f"layers={len(kv_entries)}, system_template={use_system_template}, "
            f"shape={kv_entries[0].key.shape if kv_entries else 'N/A'}"
        )
        return kv_entries, hidden_states

    # ================================================================
    # [本轮-3] 偏好嵌入计算 (LogitBias 路径)
    # ================================================================

    def compute_pref_embedding(
        self,
        text: str,
        use_system_template: bool = True,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        [本轮-3] 将偏好文本编码为 hidden state 向量，用于 logit bias 注入.

        方法: forward(output_hidden_states=True) → 取最后层 → mean/last pooling

        与 compute_kv 的区别:
        - compute_kv: 保留完整 K/V 序列，注入 attention 层
        - compute_pref_embedding: 压缩为单个向量，注入 logit 层

        pool 参数:
          "mean": 加权均值池化（推荐，稳定）
          "last": 取最后一个 token（类 CLS，适合短文本）

        Args:
            text:                偏好文本（裸文本）
            use_system_template: 是否包裹 system template（默认 True）
            pool:                池化方式 ("mean" | "last")

        Returns:
            pref_emb: Tensor [hidden_dim]，CPU float32
        """
        if not self._is_loaded:
            self.load()

        pref_text = self._wrap_pref_as_system_prompt(text) if use_system_template else text

        inputs = self.tokenize(pref_text)
        input_ids = inputs['input_ids']
        if input_ids.shape[1] > self.MAX_PREF_TOKENS:
            input_ids = input_ids[:, :self.MAX_PREF_TOKENS]
            inputs['attention_mask'] = inputs['attention_mask'][:, :self.MAX_PREF_TOKENS]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        if pool == "last":
            pref_emb = last_hidden[0, -1, :].detach().cpu().float()
        else:
            mask = inputs['attention_mask'].float().unsqueeze(-1)   # [1, seq_len, 1]
            sum_h = (last_hidden * mask).sum(dim=1)                 # [1, hidden_dim]
            count = mask.sum(dim=1).clamp(min=1e-8)                 # [1, 1]
            pref_emb = (sum_h / count)[0].detach().cpu().float()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"[LlamaAdapter] 偏好嵌入计算完成: dim={pref_emb.shape[0]}, "
            f"pool={pool}, system_template={use_system_template}"
        )
        return pref_emb

    # ================================================================
    # [本轮-3] LogitBias 核心计算
    # ================================================================

    def _compute_logit_bias_vector(
        self,
        pref_emb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        [本轮-3] 计算 logit bias 向量: lm_head.weight @ pref_emb.

        公式: bias_vec = W_lm [vocab_size, hidden_dim] @ pref_emb [hidden_dim]
              → bias_vec: [vocab_size]

        每次推理只需调用一次，结果在 decode loop 内复用。

        注意:
        - 量化模型的 lm_head.weight 可能有特殊包装，统一通过 .weight 访问
        - 显式 .to(device, dtype) 确保设备和精度一致
        """
        pref_emb_gpu = pref_emb.to(device=device, dtype=self.dtype)

        lm_head = None
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
        elif hasattr(self.model, 'language_model') and hasattr(
            self.model.language_model, 'lm_head'
        ):
            lm_head = self.model.language_model.lm_head

        if lm_head is None:
            raise RuntimeError(
                "[LlamaAdapter] 无法找到 lm_head，logit bias 注入不可用。"
                " 请确认模型架构包含 model.lm_head。"
            )

        with torch.no_grad():
            # 量化兼容路径：4bit/8bit 量化下 lm_head.weight 可能是特殊类型
            # (bnb.nn.Linear4bit 等)，直接 @ 矩阵乘会报错。
            # 统一用 lm_head forward 前向通道替代 W @ v，兼容所有量化模式。
            try:
                bias_vec = lm_head(pref_emb_gpu.unsqueeze(0)).squeeze(0)  # [vocab_size]
            except Exception:
                # 回退：尝试取 .weight，对非量化模型有效
                W = lm_head.weight.to(device=device, dtype=self.dtype)  # [vocab_size, hidden_dim]
                bias_vec = W @ pref_emb_gpu                               # [vocab_size]

        return bias_vec

    def _apply_logit_bias(
        self,
        logits: torch.Tensor,
        bias_vec: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """
        [本轮-3] 将 logit bias 叠加到 logits 上.

        公式: logits_adjusted = logits + λ * bias_vec

        λ 建议范围:
          0.05 — 轻微引导，对生成影响最小
          0.10 — 适中（默认），可感知偏好倾向
          0.30 — 强引导，可能影响语言流畅性
          >0.5  — 过强，概率分布可能失真

        Args:
            logits:   [1, vocab_size] 当前步的原始 logits
            bias_vec: [vocab_size] 预计算的偏好 bias
            lambda_:  缩放系数

        Returns:
            调整后的 logits，形状与输入相同
        """
        if lambda_ == 0.0:
            return logits
        return logits + lambda_ * bias_vec

    # ================================================================
    # 核心推理接口 (标准 — 无注入)
    # ================================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """标准推理 (无 K/V 注入)."""
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

        new_tokens = outputs[0][input_ids.shape[1]:]
        return ModelOutput(
            text=self.decode(new_tokens),
            tokens=new_tokens.tolist(),
            latency_ms=(time.perf_counter() - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
        )

    # ================================================================
    # 流式生成 (标准 — 无注入)
    # ================================================================

    async def async_stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式生成 (无 K/V 注入)."""
        if not self._is_loaded:
            self.load()
        from transformers import TextIteratorStreamer

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': max_new_tokens, 'do_sample': True,
            'temperature': temperature, 'top_p': top_p,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self._get_stop_token_ids(), 'streamer': streamer,
        }
        threading.Thread(
            target=lambda: self.model.generate(**gen_kwargs), daemon=True,
        ).start()
        for chunk in streamer:
            if chunk:
                yield chunk
            await asyncio.sleep(0)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Iterator[str]:
        """同步流式生成 (无 K/V 注入)."""
        if not self._is_loaded:
            self.load()
        from transformers import TextIteratorStreamer

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': max_new_tokens, 'do_sample': True,
            'temperature': temperature, 'top_p': top_p,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self._get_stop_token_ids(), 'streamer': streamer,
        }
        t = threading.Thread(target=lambda: self.model.generate(**gen_kwargs), daemon=True)
        t.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        t.join(timeout=5)

    # ================================================================
    # KV 注入 — 内部工具
    # ================================================================

    def _validate_rope_positions(
        self, mem_len: int, n_query: int, max_new_tokens: int,
    ) -> None:
        """验证 RoPE 位置坐标，确保不越界."""
        max_position = mem_len + n_query + max_new_tokens - 1
        logger.debug(
            f"[RoPE] mem_len={mem_len}, n_query={n_query}, "
            f"max_new_tokens={max_new_tokens}, max_position_id={max_position}"
        )
        if max_position > MAX_ROPE_POSITION:
            raise ValueError(
                f"[RoPE] max_position_id={max_position} 超过 RoPE 硬上限 "
                f"{MAX_ROPE_POSITION}。请减小 mem_len/n_query/max_new_tokens。"
            )
        if max_position > MAX_SAFE_POSITION:
            logger.warning(
                f"[RoPE] max_position_id={max_position} 超过训练分布上限 "
                f"{MAX_SAFE_POSITION}，可能产生 OOD 精度下降。"
            )

    def _build_attention_bias(
        self,
        n_pref: int,
        n_query: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构建 Prefill 阶段的 4D attention bias (论文 §4.2 的 B_alpha).

        形状: [1, 1, n_query, n_pref + n_query]

        [本轮-1] alpha=1.0 时 bias=0，偏好与 query 等权。
        alpha 语义:
          <= 0        → -inf         (完全屏蔽)
          (0, 1.0)    → log(alpha)   (抑制，alpha 越小抑制越强)
          = 1.0       → 0            (等权，推荐实验值)
        """
        total_len = n_pref + n_query
        bias = torch.zeros(1, 1, n_query, total_len, device=device, dtype=dtype)

        if alpha <= 0:
            bias[:, :, :, :n_pref] = float('-inf')
        elif alpha < 1.0:
            bias[:, :, :, :n_pref] = math.log(max(alpha, 1e-8))
        # alpha == 1.0: bias=0，等权

        if n_query > 1:
            causal = torch.triu(
                torch.full((n_query, n_query), float('-inf'), device=device, dtype=dtype),
                diagonal=1,
            )
            bias[:, :, :, n_pref:] = causal

        return bias

    def _build_decode_mask(
        self,
        total_past: int,
        mem_len: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构建 Decode 阶段的 attention mask.

        [本轮-1] 与 _build_attention_bias 保持相同的 alpha=1.0 语义（bias=0）。
        形状: [1, 1, 1, total_past + 1]
        """
        mask = torch.zeros(1, 1, 1, total_past + 1, device=device, dtype=dtype)
        if alpha <= 0:
            mask[:, :, :, :mem_len] = float('-inf')
        elif alpha < 1.0:
            mask[:, :, :, :mem_len] = math.log(max(alpha, 1e-8))
        # alpha >= 1.0: 全零，等权
        return mask

    def _prepare_kv_for_injection(
        self,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        device: torch.device,
    ) -> Tuple[Any, int]:
        """
        准备注入用的 K/V.
        Key 永远不缩放；仅 Value 被 alpha 缩放: V_aug = [α·V_p; V_u]。
        [本轮-1] alpha 上限由 self.alpha_cap 控制。
        """
        clamped_alpha = min(alpha, self.alpha_cap)
        cache, mem_len = build_dynamic_cache_from_entries(
            entries=injected_kv, device=device, alpha=clamped_alpha,
        )
        logger.debug(
            f"Prepared KV: {len(injected_kv)} layers, mem_len={mem_len}, "
            f"alpha={clamped_alpha:.2f} (cap={self.alpha_cap})"
        )
        return cache, mem_len

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: float, top_p: float,
    ) -> int:
        """Top-p nucleus 采样，返回 token id。"""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)]
        else:
            next_token = logits.argmax(dim=-1)[0]
        return next_token.item()

    # ================================================================
    # KV 注入核心实现 (非流式)
    # ================================================================

    def _forward_with_bias_impl(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        max_new_tokens: int,
        start_time: float,
        **kwargs,
    ) -> ModelOutput:
        """带 Attention Bias 的 K/V 注入核心 (HF 4.x/5.x 稳定路径)."""
        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device

        past_kv, mem_len = self._prepare_kv_for_injection(injected_kv, alpha, device)
        n_query = input_ids.shape[1]
        self._validate_rope_positions(mem_len, n_query, max_new_tokens)

        attention_bias = self._build_attention_bias(
            mem_len, n_query, alpha, device, self.dtype,
        )
        prefill_pos = torch.arange(
            mem_len, mem_len + n_query, device=device, dtype=torch.long,
        ).unsqueeze(0)

        with torch.no_grad():
            pf = self.model(
                input_ids=input_ids, past_key_values=past_kv,
                attention_mask=attention_bias, position_ids=prefill_pos,
                use_cache=True, return_dict=True,
            )

        full_kv = pf.past_key_values
        next_logits = pf.logits[:, -1, :]
        del pf, past_kv

        cache_len = _get_cache_seq_length(full_kv) or (mem_len + n_query)

        generated_ids: List[int] = []
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        stop_ids = set(self._get_stop_token_ids())

        for step in range(max_new_tokens):
            next_token_id = self._sample_next_token(next_logits, temperature, top_p)
            generated_ids.append(next_token_id)
            if next_token_id in stop_ids:
                break

            next_input = torch.tensor([[next_token_id]], device=device)
            decode_mask = self._build_decode_mask(cache_len, mem_len, alpha, device, self.dtype)
            decode_pos = torch.tensor(
                [[mem_len + n_query + step]], device=device, dtype=torch.long,
            )

            with torch.no_grad():
                dc = self.model(
                    input_ids=next_input, past_key_values=full_kv,
                    attention_mask=decode_mask, position_ids=decode_pos,
                    use_cache=True, return_dict=True,
                )

            full_kv = dc.past_key_values
            next_logits = dc.logits[:, -1, :]
            cache_len += 1
            del dc

        del full_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ModelOutput(
            text=self.decode(generated_ids),
            tokens=generated_ids,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            input_tokens=n_query,
            output_tokens=len(generated_ids),
            metadata={
                'alpha': alpha, 'alpha_cap': self.alpha_cap,
                'mem_len': mem_len,
                'injection_mode': 'manual_prefill_decode_bias',
                'system_template_kv': True,
            },
        )

    def _prefill_with_kv(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float,
        max_new_tokens: int,
    ) -> Tuple[Any, int, int, int, torch.Tensor, set]:
        """执行 prefill 阶段，返回流式 decode 所需的全部状态."""
        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device

        past_kv, mem_len = self._prepare_kv_for_injection(injected_kv, alpha, device)
        n_query = input_ids.shape[1]
        self._validate_rope_positions(mem_len, n_query, max_new_tokens)

        attention_bias = self._build_attention_bias(
            mem_len, n_query, alpha, device, self.dtype,
        )
        prefill_pos = torch.arange(
            mem_len, mem_len + n_query, device=device, dtype=torch.long,
        ).unsqueeze(0)

        with torch.no_grad():
            pf = self.model(
                input_ids=input_ids, past_key_values=past_kv,
                attention_mask=attention_bias, position_ids=prefill_pos,
                use_cache=True, return_dict=True,
            )

        full_kv = pf.past_key_values
        next_logits = pf.logits[:, -1, :]
        del pf

        cache_len = _get_cache_seq_length(full_kv) or (mem_len + n_query)
        stop_ids = set(self._get_stop_token_ids())
        return full_kv, mem_len, n_query, cache_len, next_logits, stop_ids

    # ================================================================
    # KV 注入 — 公开接口 (非流式)
    # ================================================================

    def forward_with_kv_injection(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """
        真正的 K/V 注入推理 (非流式).

        [本轮-1] alpha 上限提升为 1.0（支持等权注入）。
        alpha=1.0 时 bias=0，偏好与 query 完全等权，是最强的 KV 注入模式。
        alpha=0 时退化为 vanilla LLM。
        """
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        if alpha > self.alpha_cap:
            logger.warning(
                f"alpha={alpha} > alpha_cap={self.alpha_cap}，已裁剪。"
            )
        clamped_alpha = min(max(alpha, 0.0), self.alpha_cap)

        if not injected_kv or clamped_alpha <= 0.01:
            return self.generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
            )

        logger.info(
            f"[LlamaAdapter] KV 注入推理: layers={len(injected_kv)}, "
            f"alpha={clamped_alpha:.2f} (cap={self.alpha_cap}), "
            f"max_new_tokens={max_new_tokens}"
        )
        try:
            return self._forward_with_bias_impl(
                prompt=prompt, injected_kv=injected_kv,
                alpha=clamped_alpha, max_new_tokens=max_new_tokens,
                start_time=start_time, **kwargs,
            )
        except Exception as e:
            logger.warning(f"KV injection failed ({e}), falling back to generate")
            return self.generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
            )

    def forward_with_kv_injection_and_bias(
        self,
        prompt: str,
        injected_kv: List[KVCacheEntry],
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """向后兼容接口 — 与 forward_with_kv_injection 等价."""
        return self.forward_with_kv_injection(
            prompt=prompt, injected_kv=injected_kv,
            alpha=alpha, max_new_tokens=max_new_tokens, **kwargs,
        )

    # ================================================================
    # KV 注入 — 流式接口
    # ================================================================

    def stream_generate_with_kv_injection(
        self,
        prompt: str,
        injected_kv: Optional[List[KVCacheEntry]] = None,
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Iterator[str]:
        """同步流式生成 + KV 注入. [本轮-1] alpha 上限 1.0。"""
        if not self._is_loaded:
            self.load()

        if alpha > self.alpha_cap:
            logger.warning(f"[stream_kv] alpha={alpha} > cap，已裁剪。")
        clamped_alpha = min(max(alpha, 0.0), self.alpha_cap)

        if not injected_kv or clamped_alpha <= 0.01:
            yield from self.stream_generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            return

        logger.info(
            f"[LlamaAdapter] 流式 KV 注入 (同步): "
            f"layers={len(injected_kv)}, alpha={clamped_alpha:.2f}"
        )
        try:
            full_kv, mem_len, n_query, cache_len, next_logits, stop_ids = (
                self._prefill_with_kv(prompt, injected_kv, clamped_alpha, max_new_tokens)
            )
        except Exception as e:
            logger.warning(f"Prefill failed ({e}), falling back")
            yield from self.stream_generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            return

        device = next_logits.device
        special_ids = set(self.tokenizer.all_special_ids) if skip_special_tokens else set()

        try:
            for step in range(max_new_tokens):
                next_token_id = self._sample_next_token(next_logits, temperature, top_p)
                if next_token_id in stop_ids:
                    break
                if next_token_id not in special_ids:
                    tok = self.tokenizer.decode(
                        [next_token_id], skip_special_tokens=skip_special_tokens,
                    )
                    if tok:
                        yield tok

                next_input = torch.tensor([[next_token_id]], device=device)
                decode_mask = self._build_decode_mask(
                    cache_len, mem_len, clamped_alpha, device, self.dtype,
                )
                decode_pos = torch.tensor(
                    [[mem_len + n_query + step]], device=device, dtype=torch.long,
                )
                with torch.no_grad():
                    dc = self.model(
                        input_ids=next_input, past_key_values=full_kv,
                        attention_mask=decode_mask, position_ids=decode_pos,
                        use_cache=True, return_dict=True,
                    )
                full_kv = dc.past_key_values
                next_logits = dc.logits[:, -1, :]
                cache_len += 1
                del dc
        finally:
            del full_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def async_stream_generate_with_kv_injection(
        self,
        prompt: str,
        injected_kv: Optional[List[KVCacheEntry]] = None,
        alpha: float = 1.0,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> AsyncIterator[str]:
        """异步流式生成 + KV 注入. [本轮-1] alpha 上限 1.0。"""
        if not self._is_loaded:
            self.load()

        if alpha > self.alpha_cap:
            logger.warning(f"[async_stream_kv] alpha={alpha} > cap，已裁剪。")
        clamped_alpha = min(max(alpha, 0.0), self.alpha_cap)

        if not injected_kv or clamped_alpha <= 0.01:
            async for chunk in self.async_stream_generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            ):
                yield chunk
            return

        logger.info(
            f"[LlamaAdapter] 流式 KV 注入 (异步): "
            f"layers={len(injected_kv)}, alpha={clamped_alpha:.2f}"
        )
        try:
            full_kv, mem_len, n_query, cache_len, next_logits, stop_ids = (
                self._prefill_with_kv(prompt, injected_kv, clamped_alpha, max_new_tokens)
            )
        except Exception as e:
            logger.warning(f"Prefill failed ({e}), falling back")
            async for chunk in self.async_stream_generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            ):
                yield chunk
            return

        device = next_logits.device
        special_ids = set(self.tokenizer.all_special_ids) if skip_special_tokens else set()

        try:
            for step in range(max_new_tokens):
                next_token_id = self._sample_next_token(next_logits, temperature, top_p)
                if next_token_id in stop_ids:
                    break
                if next_token_id not in special_ids:
                    tok = self.tokenizer.decode(
                        [next_token_id], skip_special_tokens=skip_special_tokens,
                    )
                    if tok:
                        yield tok

                next_input = torch.tensor([[next_token_id]], device=device)
                decode_mask = self._build_decode_mask(
                    cache_len, mem_len, clamped_alpha, device, self.dtype,
                )
                decode_pos = torch.tensor(
                    [[mem_len + n_query + step]], device=device, dtype=torch.long,
                )
                with torch.no_grad():
                    dc = self.model(
                        input_ids=next_input, past_key_values=full_kv,
                        attention_mask=decode_mask, position_ids=decode_pos,
                        use_cache=True, return_dict=True,
                    )
                full_kv = dc.past_key_values
                next_logits = dc.logits[:, -1, :]
                cache_len += 1
                del dc
                await asyncio.sleep(0)
        finally:
            del full_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ================================================================
    # [本轮-3] LogitBias 注入 — 非流式
    # ================================================================

    def forward_with_logit_bias_injection(
        self,
        prompt: str,
        pref_emb: torch.Tensor,
        lambda_logit_bias: Optional[float] = None,
        max_new_tokens: int = 2048,
        **kwargs,
    ) -> ModelOutput:
        """
        [本轮-3] LogitBias 注入推理 (非流式).

        公式: logits_adjusted = logits + λ * (lm_head.weight @ pref_emb)

        每个 decode step 均施加，实现持续约束。
        pref_emb 由 compute_pref_embedding() 预先计算。

        Args:
            prompt:             输入提示词
            pref_emb:           偏好嵌入 [hidden_dim]
            lambda_logit_bias:  偏好强度 λ，建议 [0.05, 0.3]，None 时使用类默认值
        """
        if not self._is_loaded:
            self.load()

        if lambda_logit_bias is None:
            lambda_logit_bias = self.DEFAULT_LOGIT_BIAS_LAMBDA

        start_time = time.perf_counter()
        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device
        n_input = input_ids.shape[1]

        bias_vec = self._compute_logit_bias_vector(pref_emb, device)

        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        stop_ids = set(self._get_stop_token_ids())

        with torch.no_grad():
            pf = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                use_cache=True, return_dict=True,
            )

        full_kv = pf.past_key_values
        next_logits = self._apply_logit_bias(pf.logits[:, -1, :], bias_vec, lambda_logit_bias)
        del pf

        cache_len = _get_cache_seq_length(full_kv) or n_input
        generated_ids: List[int] = []

        try:
            for step in range(max_new_tokens):
                next_token_id = self._sample_next_token(next_logits, temperature, top_p)
                generated_ids.append(next_token_id)
                if next_token_id in stop_ids:
                    break

                next_input = torch.tensor([[next_token_id]], device=device)
                decode_mask = torch.zeros(1, 1, 1, cache_len + 1, device=device, dtype=self.dtype)
                decode_pos = torch.tensor([[n_input + step]], device=device, dtype=torch.long)

                with torch.no_grad():
                    dc = self.model(
                        input_ids=next_input, past_key_values=full_kv,
                        attention_mask=decode_mask, position_ids=decode_pos,
                        use_cache=True, return_dict=True,
                    )
                full_kv = dc.past_key_values
                # [本轮-3] 每步均施加 bias — 持续约束
                next_logits = self._apply_logit_bias(
                    dc.logits[:, -1, :], bias_vec, lambda_logit_bias
                )
                cache_len += 1
                del dc
        finally:
            del full_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ModelOutput(
            text=self.decode(generated_ids),
            tokens=generated_ids,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            input_tokens=n_input,
            output_tokens=len(generated_ids),
            metadata={
                'injection_mode': 'logit_bias',
                'lambda_logit_bias': lambda_logit_bias,
                'pref_emb_dim': pref_emb.shape[0],
            },
        )

    # ================================================================
    # [本轮-3] LogitBias 注入 — 同步流式
    # ================================================================

    def stream_generate_with_logit_bias(
        self,
        prompt: str,
        pref_emb: torch.Tensor,
        lambda_logit_bias: Optional[float] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Iterator[str]:
        """
        [本轮-3] 同步流式生成 + LogitBias 注入.

        每个 decode step 均施加 λ * bias_vec，偏好信号持续约束全程输出。
        Yields str: 每个 token 解码后的文本片段。
        """
        if not self._is_loaded:
            self.load()

        if lambda_logit_bias is None:
            lambda_logit_bias = self.DEFAULT_LOGIT_BIAS_LAMBDA

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device
        n_input = input_ids.shape[1]
        stop_ids = set(self._get_stop_token_ids())
        special_ids = set(self.tokenizer.all_special_ids) if skip_special_tokens else set()

        bias_vec = self._compute_logit_bias_vector(pref_emb, device)

        with torch.no_grad():
            pf = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                use_cache=True, return_dict=True,
            )
        full_kv = pf.past_key_values
        next_logits = self._apply_logit_bias(pf.logits[:, -1, :], bias_vec, lambda_logit_bias)
        del pf

        cache_len = _get_cache_seq_length(full_kv) or n_input

        try:
            for step in range(max_new_tokens):
                next_token_id = self._sample_next_token(next_logits, temperature, top_p)
                if next_token_id in stop_ids:
                    break
                if next_token_id not in special_ids:
                    tok = self.tokenizer.decode(
                        [next_token_id], skip_special_tokens=skip_special_tokens,
                    )
                    if tok:
                        yield tok

                next_input = torch.tensor([[next_token_id]], device=device)
                decode_mask = torch.zeros(1, 1, 1, cache_len + 1, device=device, dtype=self.dtype)
                decode_pos = torch.tensor([[n_input + step]], device=device, dtype=torch.long)

                with torch.no_grad():
                    dc = self.model(
                        input_ids=next_input, past_key_values=full_kv,
                        attention_mask=decode_mask, position_ids=decode_pos,
                        use_cache=True, return_dict=True,
                    )
                full_kv = dc.past_key_values
                next_logits = self._apply_logit_bias(
                    dc.logits[:, -1, :], bias_vec, lambda_logit_bias
                )
                cache_len += 1
                del dc
        finally:
            del full_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ================================================================
    # [本轮-3] LogitBias 注入 — 异步流式
    # ================================================================

    async def async_stream_generate_with_logit_bias(
        self,
        prompt: str,
        pref_emb: torch.Tensor,
        lambda_logit_bias: Optional[float] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        [本轮-3] 异步流式生成 + LogitBias 注入.

        每个 decode step 后 await asyncio.sleep(0) 让出控制权。
        Yields str: 每个 token 解码后的文本片段。
        """
        if not self._is_loaded:
            self.load()

        if lambda_logit_bias is None:
            lambda_logit_bias = self.DEFAULT_LOGIT_BIAS_LAMBDA

        formatted_prompt = self._format_prompt_safe(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        device = input_ids.device
        n_input = input_ids.shape[1]
        stop_ids = set(self._get_stop_token_ids())
        special_ids = set(self.tokenizer.all_special_ids) if skip_special_tokens else set()

        bias_vec = self._compute_logit_bias_vector(pref_emb, device)

        with torch.no_grad():
            pf = self.model(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'],
                use_cache=True, return_dict=True,
            )
        full_kv = pf.past_key_values
        next_logits = self._apply_logit_bias(pf.logits[:, -1, :], bias_vec, lambda_logit_bias)
        del pf

        cache_len = _get_cache_seq_length(full_kv) or n_input

        try:
            for step in range(max_new_tokens):
                next_token_id = self._sample_next_token(next_logits, temperature, top_p)
                if next_token_id in stop_ids:
                    break
                if next_token_id not in special_ids:
                    tok = self.tokenizer.decode(
                        [next_token_id], skip_special_tokens=skip_special_tokens,
                    )
                    if tok:
                        yield tok

                next_input = torch.tensor([[next_token_id]], device=device)
                decode_mask = torch.zeros(1, 1, 1, cache_len + 1, device=device, dtype=self.dtype)
                decode_pos = torch.tensor([[n_input + step]], device=device, dtype=torch.long)

                with torch.no_grad():
                    dc = self.model(
                        input_ids=next_input, past_key_values=full_kv,
                        attention_mask=decode_mask, position_ids=decode_pos,
                        use_cache=True, return_dict=True,
                    )
                full_kv = dc.past_key_values
                next_logits = self._apply_logit_bias(
                    dc.logits[:, -1, :], bias_vec, lambda_logit_bias
                )
                cache_len += 1
                del dc
                await asyncio.sleep(0)
        finally:
            del full_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ================================================================
    # Embedding
    # ================================================================

    def embed(self, text: str) -> torch.Tensor:
        if not self._is_loaded:
            self.load()
        inputs = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True, return_dict=True,
            )
            emb = outputs.hidden_states[-1].mean(dim=1).detach().cpu()
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return emb

    # ================================================================
    # Prefill 熵计算 (论文 §4.2 — 双因子门控)
    # ================================================================

    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        if not self._is_loaded:
            self.load()
        inputs = self.tokenize(text)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, return_dict=True)
            if outputs.attentions is None:
                return self.DEFAULT_ENTROPY
            layer_idx = min(layer_idx, len(outputs.attentions) - 1)
            attn = outputs.attentions[layer_idx].clamp(min=1e-9)
            entropy = -torch.sum(attn * torch.log(attn), dim=-1).mean()
            return entropy.item()
        except Exception as e:
            logger.warning(f"compute_prefill_entropy failed: {e}")
            return self.DEFAULT_ENTROPY

    # ================================================================
    # 模型信息
    # ================================================================

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'adapter_type': 'llama31_hf_kv_injection',
            'supported_model_family': 'llama3.1_only',
            'injection_mode': 'manual_prefill_decode_bias',
            'kv_injection_type': 'system_template_kv',
            'alpha_cap': self.alpha_cap,
            'max_pref_tokens': self.MAX_PREF_TOKENS,
            'system_template_kv': True,
            'streaming_kv_injection': True,
            'logit_bias_injection': True,
            'load_in_8bit': self.load_in_8bit,
            'quantization': self.quantization,
        })
        return info
