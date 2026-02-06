"""
Fake Model Adapter for CPU-only unit testing.
Provides deterministic behavior without requiring GPU or real LLM.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry


@dataclass
class FakeQKV:
    """Fake Q/K/V tensors for testing."""
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    batch_size: int = 1
    num_heads: int = 4
    seq_len: int = 10
    head_dim: int = 32
    
    @classmethod
    def create(
        cls,
        batch_size: int = 1,
        num_heads: int = 4,
        seq_len: int = 10,
        head_dim: int = 32,
        device: str = "cpu",
    ) -> "FakeQKV":
        """Create fake QKV tensors with deterministic values."""
        torch.manual_seed(42)
        shape = (batch_size, num_heads, seq_len, head_dim)
        return cls(
            query=torch.randn(*shape, device=device),
            key=torch.randn(*shape, device=device),
            value=torch.randn(*shape, device=device),
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )


class FakeModelAdapter(BaseModelAdapter):
    """
    Fake model adapter for CPU-only testing.
    
    Provides deterministic outputs without loading real models.
    All operations are performed on CPU with fixed random seeds.
    """
    
    def __init__(
        self,
        model_name: str = "fake-model",
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        head_dim: int = 32,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._is_loaded = True
        
        # Fake tokenizer vocabulary
        self._vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
        }
        self._next_token_id = 4
    
    def load(self) -> None:
        """Fake load - already loaded."""
        self._is_loaded = True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> ModelOutput:
        """Generate fake response."""
        # Deterministic fake response based on prompt
        torch.manual_seed(hash(prompt) % 2**32)
        
        # Fake tokenization
        input_tokens = self._fake_tokenize(prompt)
        
        # Generate fake output tokens
        output_tokens = list(range(100, 100 + min(max_new_tokens, 50)))
        
        # Fake response text
        response = f"[Fake response to: {prompt[:50]}...]"
        
        return ModelOutput(
            text=response,
            tokens=output_tokens,
            input_tokens=len(input_tokens),
            output_tokens=len(output_tokens),
            latency_ms=10.0,
            metadata={"fake": True},
        )
    
    def embed(self, text: str) -> torch.Tensor:
        """Generate fake embeddings."""
        torch.manual_seed(hash(text) % 2**32)
        return torch.randn(self.hidden_dim)
    
    def compute_kv(
        self,
        text: str,
        return_hidden: bool = False,
    ) -> Tuple[List[KVCacheEntry], Optional[torch.Tensor]]:
        """
        Compute fake K/V cache entries.
        
        Returns deterministic K/V based on text hash.
        """
        torch.manual_seed(hash(text) % 2**32)
        
        # Estimate sequence length from text
        seq_len = max(1, len(text.split()))
        
        kv_entries = []
        for layer_idx in range(self.num_layers):
            key = torch.randn(1, self.num_heads, seq_len, self.head_dim)
            value = torch.randn(1, self.num_heads, seq_len, self.head_dim)
            
            kv_entries.append(KVCacheEntry(
                key=key,
                value=value,
                layer_idx=layer_idx,
            ))
        
        hidden_states = None
        if return_hidden:
            hidden_states = torch.randn(1, seq_len, self.hidden_dim)
        
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
        Generate with injected K/V cache.
        
        For testing, we verify the injection happened by checking
        the K/V dimensions and returning metadata about the injection.
        """
        torch.manual_seed(hash(prompt) % 2**32)
        
        # Calculate injected sequence length
        injected_seq_len = 0
        if injected_kv:
            injected_seq_len = injected_kv[0].key.shape[2]
        
        # Fake tokenization
        input_tokens = self._fake_tokenize(prompt)
        
        # Generate fake output
        response = f"[Fake DKI response with {injected_seq_len} injected tokens, alpha={alpha}]"
        
        return ModelOutput(
            text=response,
            tokens=list(range(100, 150)),
            input_tokens=len(input_tokens),
            output_tokens=50,
            latency_ms=15.0,
            metadata={
                "fake": True,
                "injected_seq_len": injected_seq_len,
                "alpha": alpha,
                "num_kv_layers": len(injected_kv) if injected_kv else 0,
            },
        )
    
    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """
        Compute fake prefill entropy.
        
        Returns deterministic entropy based on text characteristics.
        """
        torch.manual_seed(hash(text) % 2**32)
        
        # Simulate entropy based on text length and complexity
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        
        # Higher entropy for more complex/uncertain queries
        base_entropy = 0.5
        complexity_factor = unique_words / max(word_count, 1)
        
        entropy = base_entropy + 0.3 * complexity_factor + 0.2 * torch.rand(1).item()
        
        return min(1.0, max(0.0, entropy))
    
    def get_qkv(self, input_ids: torch.Tensor) -> FakeQKV:
        """Get fake Q/K/V tensors for input."""
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        return FakeQKV.create(
            batch_size=batch_size,
            num_heads=self.num_heads,
            seq_len=seq_len,
            head_dim=self.head_dim,
            device=self.device,
        )
    
    def _fake_tokenize(self, text: str) -> List[int]:
        """Fake tokenization."""
        tokens = [self._vocab["<bos>"]]
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_token_id
                self._next_token_id += 1
            tokens.append(self._vocab[word])
        tokens.append(self._vocab["<eos>"])
        return tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get fake model info."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": "float32",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "is_loaded": self._is_loaded,
            "is_fake": True,
        }


class FakeModelAdapterWithAttention(FakeModelAdapter):
    """
    Extended fake model that returns attention weights.
    Useful for testing attention-related components.
    """
    
    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning hidden states and attention weights.
        
        Returns:
            hidden_states: [batch, seq, hidden_dim]
            attention_weights: [batch, num_heads, seq, seq]
        """
        torch.manual_seed(42)
        
        batch_size, seq_len = input_ids.shape
        
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)
        
        # Create fake attention weights (normalized)
        attention_weights = torch.softmax(
            torch.randn(batch_size, self.num_heads, seq_len, seq_len),
            dim=-1
        )
        
        return hidden_states, attention_weights
