"""
GLM Model Adapter for DKI System
HuggingFace Transformers-based adapter for ChatGLM/GLM-4 models
"""

import asyncio
import time
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import (
    BaseModelAdapter, ModelOutput, KVCacheEntry,
    extract_kv_from_past, build_dynamic_cache_from_entries,
)


class GLMAdapter(BaseModelAdapter):
    """
    GLM model adapter using HuggingFace Transformers.
    
    Supports ChatGLM-6B, GLM-4, and compatible models.
    """
    
    def __init__(
        self,
        model_name: str = "THUDM/glm-4-9b-chat",
        device: str = "cuda",
        dtype: str = "float16",
        trust_remote_code: bool = True,
        quantization: str = "none",
        quantization_config: dict = None,
        **kwargs
    ):
        super().__init__(
            model_name, device, dtype,
            quantization=quantization,
            quantization_config=quantization_config,
            **kwargs
        )
        
        self.trust_remote_code = trust_remote_code
    
    def load(self) -> None:
        """
        Load GLM model and tokenizer.
        
        支持量化模式:
        - "none": 原始精度 (dtype)
        - "4bit": 4-bit NF4 量化 (bitsandbytes)
        - "8bit": 8-bit LLM.int8() 量化 (bitsandbytes)
        - "gptq": GPTQ 预量化模型
        - "awq": AWQ 预量化模型
        """
        if self._is_loaded:
            return
        
        try:
            # HuggingFace Hub 兼容性补丁 (huggingface_hub ≥0.25)
            from dki.models.hf_compat import ensure_hf_compat
            ensure_hf_compat()
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            
            quant_desc = f" (quantization={self.quantization})" if self.is_quantized else ""
            logger.info(f"Loading GLM model: {self.model_name}{quant_desc}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Build model kwargs
            model_kwargs = {
                'device_map': 'auto',
                'trust_remote_code': self.trust_remote_code,
                'attn_implementation': 'eager',  # Required for output_attentions support
            }
            
            # ============ 量化配置 ============
            if self.quantization in ("4bit", "8bit"):
                bnb_config = self._build_bnb_config()
                model_kwargs['quantization_config'] = bnb_config
                if self.is_4bit:
                    model_kwargs['torch_dtype'] = self.dtype
            elif self.quantization in ("gptq", "awq"):
                model_kwargs['torch_dtype'] = self.dtype
            else:
                model_kwargs['torch_dtype'] = self.dtype
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self.model.eval()
            
            # Get architecture info
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            self.hidden_dim = getattr(config, 'hidden_size', 4096)
            self.num_layers = getattr(config, 'num_layers', getattr(config, 'num_hidden_layers', 28))
            self.num_heads = getattr(config, 'num_attention_heads', 32)
            self.head_dim = self.hidden_dim // self.num_heads
            
            self._is_loaded = True
            logger.info(
                f"GLM adapter loaded: {self.model_name} "
                f"(quantization={self.quantization})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load GLM model: {e}")
            raise
    
    def _build_chat_input(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Build chat input for GLM format."""
        # GLM-4 uses a specific chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            if history:
                for user_msg, assistant_msg in history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": query})
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Fallback for older ChatGLM
            return f"[Round 1]\n\n问：{query}\n\n答："
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """Generate text using GLM model."""
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        # Try to use model's chat method if available
        if hasattr(self.model, 'chat') and hasattr(self.tokenizer, 'apply_chat_template'):
            # Use GLM's built-in chat interface
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            end_time = time.perf_counter()
            
            return ModelOutput(
                text=response,
                latency_ms=(end_time - start_time) * 1000,
            )
        
        # Fallback to standard generation
        formatted_prompt = self._build_chat_input(prompt)
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
        Async streaming generation for GLM models.
        
        GLM-4 支持 model.stream_chat() 原生流式接口。
        对于不支持的模型, 回退到 TextIteratorStreamer。
        
        Yields:
            str: decoded text chunks
        """
        if not self._is_loaded:
            self.load()
        
        # 尝试使用 GLM 原生 stream_chat
        if hasattr(self.model, 'stream_chat') and hasattr(self.tokenizer, 'apply_chat_template'):
            loop = asyncio.get_running_loop()
            prev_text = ""
            
            def _stream_chat():
                return self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    history=[],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            
            # stream_chat 返回 generator, 在线程中逐步消费
            import queue
            q = queue.Queue()
            sentinel = object()
            
            def _producer():
                try:
                    for response, _ in _stream_chat():
                        q.put(response)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(sentinel)
            
            thread = threading.Thread(target=_producer, daemon=True)
            thread.start()
            
            prev_len = 0
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                # stream_chat 返回累积文本, 需要提取增量
                new_text = item[prev_len:]
                prev_len = len(item)
                if new_text:
                    yield new_text
                await asyncio.sleep(0)
            
            thread.join(timeout=5)
            return
        
        # 回退: TextIteratorStreamer
        from transformers import TextIteratorStreamer
        
        formatted_prompt = self._build_chat_input(prompt)
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
        Synchronous streaming generation for GLM models.
        
        Yields:
            str: decoded text chunks
        """
        if not self._is_loaded:
            self.load()
        
        # 尝试使用 GLM 原生 stream_chat
        if hasattr(self.model, 'stream_chat') and hasattr(self.tokenizer, 'apply_chat_template'):
            prev_len = 0
            for response, _ in self.model.stream_chat(
                self.tokenizer,
                prompt,
                history=[],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                new_text = response[prev_len:]
                prev_len = len(response)
                if new_text:
                    yield new_text
            return
        
        # 回退: TextIteratorStreamer
        from transformers import TextIteratorStreamer
        
        formatted_prompt = self._build_chat_input(prompt)
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
        """Compute K/V cache for text."""
        if not self._is_loaded:
            self.load()
        
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=return_hidden,
                use_cache=True,
                return_dict=True,
            )
        
        # Extract K/V cache — detach and move to CPU to prevent GPU memory leak
        # 兼容 Transformers 4.x / 5.x: 通过 extract_kv_from_past 统一提取
        kv_entries = []
        past_kv = outputs.past_key_values
        
        if past_kv is not None:
            kv_pairs = extract_kv_from_past(past_kv)
            for layer_idx, (key, value) in enumerate(kv_pairs):
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
        max_new_tokens: int = 2048,
        **kwargs
    ) -> ModelOutput:
        """Generate with K/V injection."""
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        formatted_prompt = self._build_chat_input(prompt)
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Scale K/V with alpha, 构建 DynamicCache (兼容 Transformers 4.x/5.x)
        past_kv, mem_len = build_dynamic_cache_from_entries(
            entries=injected_kv,
            device=input_ids.device,
            alpha=alpha,
        )
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=input_ids.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        end_time = time.perf_counter()
        
        new_tokens = outputs[0][input_ids.shape[1]:]
        output_text = self.decode(new_tokens.cpu())
        output_token_list = new_tokens.cpu().tolist()
        input_len = input_ids.shape[1]
        
        # Cleanup: free GPU memory from intermediate tensors
        del past_kv, outputs, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ModelOutput(
            text=output_text,
            tokens=output_token_list,
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_len,
            output_tokens=len(output_token_list),
            metadata={'alpha': alpha, 'mem_len': mem_len},
        )
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """Compute prefill-stage entropy."""
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
            
            if outputs.attentions is None or len(outputs.attentions) == 0:
                logger.warning("Attention outputs not available, returning default entropy")
                return 0.5  # Default value if attentions not available
            
            layer_idx = min(layer_idx, len(outputs.attentions) - 1)
            attn_weights = outputs.attentions[layer_idx]  # [batch, heads, seq_q, seq_k]
            attn_weights = attn_weights.clamp(min=1e-9)
            # Compute per-row entropy (each row is a probability distribution over keys)
            # then average across all rows, heads, and batch
            per_row_entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # [batch, heads, seq_q]
            entropy = per_row_entropy.mean()  # scalar: average entropy per attention row
            
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            return 0.5  # Default entropy value
