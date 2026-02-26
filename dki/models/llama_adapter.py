"""
LLaMA Model Adapter for DKI System
HuggingFace Transformers-based adapter for LLaMA models

Chat Template 支持:
- LLaMA 3 / 3.1 / 3.2 Instruct: 使用 <|begin_of_text|> + <|start_header_id|> 官方模板
- LLaMA 2 Chat: 使用 [INST] / [/INST] 官方模板  
- 优先使用 tokenizer.apply_chat_template 以确保与模型训练一致
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class LlamaAdapter(BaseModelAdapter):
    """
    LLaMA model adapter using HuggingFace Transformers.
    
    Supports LLaMA 2 Chat, LLaMA 3/3.1/3.2 Instruct, and compatible models.
    
    Chat template 遵循 Meta 官方标准:
    - Llama 3.x: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
    - Llama 2:   [INST] <<SYS>>...<</SYS>> {prompt} [/INST]
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(model_name, device, torch_dtype, **kwargs)
        
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
    
    def load(self) -> None:
        """Load LLaMA model and tokenizer."""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            
            logger.info(f"Loading LLaMA model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'torch_dtype': self.torch_dtype,
                'trust_remote_code': self.trust_remote_code,
            }
            
            if self.load_in_8bit:
                model_kwargs['load_in_8bit'] = True
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['device_map'] = 'auto'
            
            # Add eager attention for output_attentions support
            model_kwargs['attn_implementation'] = 'eager'
            
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
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            self.num_heads = getattr(config, 'num_attention_heads', 32)
            self.head_dim = self.hidden_dim // self.num_heads
            
            self._is_loaded = True
            logger.info(f"LLaMA adapter loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load LLaMA model: {e}")
            raise
    
    def _is_chat_model(self) -> bool:
        """判断是否为 Chat/Instruct 模型"""
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in ('chat', 'instruct'))
    
    def _is_llama3(self) -> bool:
        """判断是否为 Llama 3.x 系列"""
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in ('llama-3', 'llama3', 'meta-llama/meta-llama-3'))
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt using official LLaMA chat template.
        
        Llama 3.x Instruct 官方模板 (Meta 标准):
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|start_header_id|>user<|end_header_id|>
            {user_input}
            <|start_header_id|>assistant<|end_header_id|>
        
        Llama 2 Chat 官方模板:
            [INST] <<SYS>>
            {system_prompt}
            <</SYS>>
            {user_input} [/INST]
        
        优先使用 tokenizer.apply_chat_template (确保与模型训练时一致)。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 优先使用 tokenizer 内置的 chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                logger.warning(f"apply_chat_template failed, using manual template: {e}")
        
        # 回退: 根据模型版本手动构造官方模板 (标签闭合)
        if self._is_llama3():
            # Llama 3.x 官方模板 (每个消息以 <|eot_id|> 闭合)
            parts = ["<|begin_of_text|>"]
            if system_prompt:
                parts.append(
                    f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                )
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            )
            parts.append(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            return "".join(parts)
        else:
            # Llama 2 Chat 官方模板
            if system_prompt:
                return (
                    f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                    f"{prompt} [/INST]"
                )
            else:
                return f"[INST] {prompt} [/INST]"
    
    def _has_chat_template_tokens(self, text: str) -> bool:
        """检测文本是否已包含 chat template 特殊标记 (避免双重包装)"""
        # Llama 3 标记
        if '<|begin_of_text|>' in text or '<|start_header_id|>' in text:
            return True
        # Llama 2 标记
        if '[INST]' in text:
            return True
        # DeepSeek/Qwen ChatML 标记 (半角)
        if '<|im_start|>' in text:
            return True
        # DeepSeek V2/V3 原生标记 (全角 ｜, tokenizer 自带格式)
        if '<\uff5c' in text and '\uff5c>' in text:
            return True
        return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """Generate text using LLaMA model."""
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        # Format prompt using official chat template for chat/instruct models
        # 跳过已经包含 chat template 特殊标记的 prompt (避免双重包装)
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
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
        
        # Extract generated tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        output_text = self.decode(new_tokens)
        
        return ModelOutput(
            text=output_text,
            tokens=new_tokens.tolist(),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
        )
    
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
        """Generate with K/V injection."""
        if not self._is_loaded:
            self.load()
        
        start_time = time.perf_counter()
        
        # Format prompt using official chat template for chat/instruct models
        # 跳过已经包含 chat template 特殊标记的 prompt (避免双重包装)
        if self._has_chat_template_tokens(prompt):
            formatted_prompt = prompt
        elif self._is_chat_model():
            formatted_prompt = self._format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenize(formatted_prompt)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Apply alpha scaling to injected K/V
        scaled_kv = []
        for entry in injected_kv:
            if alpha < 1.0:
                # Scale the values (keys affect attention weights, values affect output)
                scaled_key = entry.key
                scaled_value = entry.value * alpha
            else:
                scaled_key = entry.key
                scaled_value = entry.value
            scaled_kv.append((scaled_key, scaled_value))
        
        past_kv = tuple(scaled_kv)
        
        # Extend attention mask
        mem_len = injected_kv[0].key.shape[2] if injected_kv else 0
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=input_ids.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        # Generate
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
        del past_kv, outputs, scaled_kv, input_ids, attention_mask
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
            
            # Check if attentions are available
            if outputs.attentions is None:
                logger.warning("Attention outputs not available, returning default entropy")
                return 0.5  # Default entropy value
            
            # Get attention weights
            layer_idx = min(layer_idx, len(outputs.attentions) - 1)
            attn_weights = outputs.attentions[layer_idx]  # [batch, heads, seq_q, seq_k]
            
            # Compute per-row entropy (each row is a probability distribution over keys)
            # then average across all rows, heads, and batch
            attn_weights = attn_weights.clamp(min=1e-9)
            per_row_entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # [batch, heads, seq_q]
            entropy = per_row_entropy.mean()  # scalar: average entropy per attention row
            
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            return 0.5  # Default entropy value
