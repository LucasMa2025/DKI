"""
LLaMA Model Adapter for DKI System
HuggingFace Transformers-based adapter for LLaMA models
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class LlamaAdapter(BaseModelAdapter):
    """
    LLaMA model adapter using HuggingFace Transformers.
    
    Supports LLaMA 2, LLaMA 3, and compatible models.
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
        
        inputs = self.tokenize(prompt)
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
            embeddings = hidden_states.mean(dim=1)
        
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
        
        # Extract K/V cache
        kv_entries = []
        past_kv = outputs.past_key_values
        
        for layer_idx, (key, value) in enumerate(past_kv):
            entry = KVCacheEntry(
                key=key,
                value=value,
                layer_idx=layer_idx,
            )
            kv_entries.append(entry)
        
        hidden_states = None
        if return_hidden and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        
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
        
        inputs = self.tokenize(prompt)
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
        output_text = self.decode(new_tokens)
        
        return ModelOutput(
            text=output_text,
            tokens=new_tokens.tolist(),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
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
            attn_weights = outputs.attentions[layer_idx]
            
            # Compute entropy
            attn_weights = attn_weights.clamp(min=1e-9)
            entropy = -torch.sum(attn_weights * torch.log(attn_weights))
            
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            return 0.5  # Default entropy value
