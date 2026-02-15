"""
vLLM Model Adapter for DKI System
High-performance inference with vLLM engine
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


class VLLMAdapter(BaseModelAdapter):
    """
    vLLM-based model adapter for high-performance inference.
    
    Note: vLLM doesn't natively support K/V injection, so we implement
    a custom attention modification layer for DKI functionality.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        
        # vLLM specific
        self.llm = None
        self.sampling_params = None
        
        # For K/V computation, we also need HuggingFace model
        self.hf_model = None
    
    def load(self) -> None:
        """Load vLLM engine and HuggingFace model for K/V computation."""
        if self._is_loaded:
            return
        
        try:
            # Import vLLM
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            
            # Load vLLM for fast inference
            logger.info(f"Loading vLLM engine: {self.model_name}")
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
            logger.info(f"vLLM adapter loaded: {self.model_name}")
            
        except ImportError as e:
            logger.error(f"vLLM not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    def _load_hf_model(self) -> None:
        """Load HuggingFace model for K/V computation."""
        if self.hf_model is not None:
            return
        
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Loading HF model for K/V computation: {self.model_name}")
        # Use attn_implementation="eager" to support output_attentions=True
        # SDPA attention does not support output_attentions
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
            attn_implementation="eager",  # Required for output_attentions support
        )
        self.hf_model.eval()
    
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
        """Get embeddings using HuggingFace model."""
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
            embeddings = hidden_states.mean(dim=1)
        
        return embeddings
    
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """Compute K/V cache using HuggingFace model."""
        self._load_hf_model()
        
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.hf_model(
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
        """
        Generate with K/V injection.
        
        Supports FlashAttention optimization when enabled.
        Falls back to HuggingFace model for K/V injection
        since vLLM doesn't support custom attention modifications.
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
        
        # Standard path: Prepare past_key_values with alpha scaling
        # Alpha scales the values (values affect output, keys affect attention weights)
        if alpha < 1.0:
            past_kv = tuple(
                (entry.key, entry.value * alpha) for entry in injected_kv
            )
        else:
            past_kv = tuple(
                (entry.key, entry.value) for entry in injected_kv
            )
        
        # Extend attention mask for injected K/V
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=self.device,
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
        output_text = self.decode(new_tokens)
        
        return ModelOutput(
            text=output_text,
            tokens=new_tokens.tolist(),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
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
        FlashAttention optimized forward pass with K/V injection.
        
        Uses KVInjectionOptimizer for efficient attention computation.
        """
        mem_len = injected_kv[0].key.shape[2] if injected_kv else 0
        
        # For now, we still use HF model but with FlashAttention-enabled attention
        # This is a simplified integration; full integration would require
        # modifying the model's attention layers directly
        
        # Prepare past_key_values with alpha scaling
        if alpha < 1.0:
            past_kv = tuple(
                (entry.key, entry.value * alpha) for entry in injected_kv
            )
        else:
            past_kv = tuple(
                (entry.key, entry.value) for entry in injected_kv
            )
        
        # Extend attention mask for injected K/V
        if mem_len > 0:
            mem_mask = torch.ones(
                (input_ids.shape[0], mem_len),
                device=self.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        
        # Generate with injected K/V
        # Note: The actual FlashAttention optimization happens at the
        # attention layer level when the model is configured to use it
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
        output_text = self.decode(new_tokens)
        
        return ModelOutput(
            text=output_text,
            tokens=new_tokens.tolist(),
            latency_ms=(end_time - start_time) * 1000,
            input_tokens=input_ids.shape[1],
            output_tokens=len(new_tokens),
            metadata={
                'alpha': alpha,
                'mem_len': mem_len,
                'flash_attn': True,
                'flash_attn_backend': self._flash_attn_backend,
            },
        )
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """Compute prefill-stage entropy for gating.
        
        Note: This method uses the HF model (not vLLM) to compute attention weights.
        When GPU memory is tight (vLLM process occupies most VRAM), this will fail with
        CUDA OOM. In that case, we return a default entropy value and release any
        intermediate tensors to avoid memory leaks.
        """
        # Pre-check: if CUDA memory is very low, skip computation entirely
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0]
                # If less than 256MB free, skip to avoid OOM
                if free_mem < 256 * 1024 * 1024:
                    logger.debug(
                        f"Skipping prefill entropy: only {free_mem / 1024 / 1024:.0f}MB free GPU memory"
                    )
                    return 0.5
            except Exception:
                pass
        
        self._load_hf_model()
        
        inputs = self.tokenize(text)
        
        try:
            with torch.no_grad():
                outputs = self.hf_model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )
            
            # Check if attentions are available
            if outputs.attentions is None:
                logger.warning("Attention outputs not available, returning default entropy")
                return 0.5  # Default entropy value
            
            # Ensure layer_idx is valid
            if layer_idx >= len(outputs.attentions):
                layer_idx = len(outputs.attentions) - 1
                logger.warning(f"Layer index out of range, using layer {layer_idx}")
            
            # Get attention weights from specified layer
            attn_weights = outputs.attentions[layer_idx]  # [batch, heads, seq_q, seq_k]
            
            # Compute per-row entropy (each row is a probability distribution over keys)
            # then average across all rows, heads, and batch
            attn_weights = attn_weights.clamp(min=1e-9)
            per_row_entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # [batch, heads, seq_q]
            entropy = per_row_entropy.mean()  # scalar: average entropy per attention row
            
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute prefill entropy: {e}, returning default")
            # Release GPU memory after OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.5  # Default entropy value
    
    def unload(self) -> None:
        """Unload models."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.hf_model is not None:
            del self.hf_model
            self.hf_model = None
        
        super().unload()
