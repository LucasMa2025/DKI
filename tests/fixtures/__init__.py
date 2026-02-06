"""
Test fixtures for DKI unit tests.
Provides fake models, attention mechanisms, and sample data.
"""

from .fake_model import FakeModelAdapter, FakeQKV
from .fake_attention import FakeAttention, create_fake_attention_output
from .fake_embeddings import FakeEmbeddingService
from .sample_memories import (
    SAMPLE_MEMORIES,
    SAMPLE_PREFERENCES,
    SAMPLE_QUERIES,
    create_sample_memory,
    create_sample_kv_entries,
)

__all__ = [
    "FakeModelAdapter",
    "FakeQKV",
    "FakeAttention",
    "create_fake_attention_output",
    "FakeEmbeddingService",
    "SAMPLE_MEMORIES",
    "SAMPLE_PREFERENCES",
    "SAMPLE_QUERIES",
    "create_sample_memory",
    "create_sample_kv_entries",
]
