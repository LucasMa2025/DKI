"""
Sample memories and test data for DKI unit tests.
Provides realistic test scenarios without requiring external data.
"""

import torch
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from dki.models.base import KVCacheEntry


# Sample memories for testing
SAMPLE_MEMORIES = [
    {
        "id": "mem_food_1",
        "content": "用户是素食主义者，对海鲜过敏",
        "metadata": {"category": "dietary", "importance": "high"},
    },
    {
        "id": "mem_location_1",
        "content": "用户住在北京，是一名软件工程师",
        "metadata": {"category": "personal", "importance": "medium"},
    },
    {
        "id": "mem_hobby_1",
        "content": "用户周末喜欢爬山和摄影",
        "metadata": {"category": "hobby", "importance": "medium"},
    },
    {
        "id": "mem_birthday_1",
        "content": "用户的生日是3月15日",
        "metadata": {"category": "personal", "importance": "low"},
    },
    {
        "id": "mem_drink_1",
        "content": "用户喜欢咖啡胜过茶",
        "metadata": {"category": "preference", "importance": "low"},
    },
    {
        "id": "mem_work_1",
        "content": "用户在一家AI创业公司工作，负责后端开发",
        "metadata": {"category": "work", "importance": "medium"},
    },
    {
        "id": "mem_family_1",
        "content": "用户有一只叫小白的猫",
        "metadata": {"category": "family", "importance": "low"},
    },
]

# English sample memories
SAMPLE_MEMORIES_EN = [
    {
        "id": "mem_food_en_1",
        "content": "User is vegetarian and allergic to seafood",
        "metadata": {"category": "dietary", "importance": "high"},
    },
    {
        "id": "mem_location_en_1",
        "content": "User lives in Beijing and works as a software engineer",
        "metadata": {"category": "personal", "importance": "medium"},
    },
    {
        "id": "mem_hobby_en_1",
        "content": "User enjoys hiking and photography on weekends",
        "metadata": {"category": "hobby", "importance": "medium"},
    },
]

# Sample user preferences
SAMPLE_PREFERENCES = [
    {
        "user_id": "user_001",
        "content": "素食主义者，住北京，不喜欢辣，喜欢咖啡",
        "metadata": {"language": "zh"},
    },
    {
        "user_id": "user_002",
        "content": "Vegetarian, lives in Shanghai, prefers tea over coffee",
        "metadata": {"language": "en"},
    },
]

# Sample queries for testing
SAMPLE_QUERIES = [
    {
        "query": "今晚吃什么好？",
        "expected_memories": ["mem_food_1"],
        "category": "dietary",
    },
    {
        "query": "周末有什么活动推荐？",
        "expected_memories": ["mem_hobby_1"],
        "category": "hobby",
    },
    {
        "query": "我的生日是什么时候？",
        "expected_memories": ["mem_birthday_1"],
        "category": "personal",
    },
    {
        "query": "推荐一家附近的咖啡店",
        "expected_memories": ["mem_drink_1", "mem_location_1"],
        "category": "preference",
    },
]

# English queries
SAMPLE_QUERIES_EN = [
    {
        "query": "What should I eat for dinner?",
        "expected_memories": ["mem_food_en_1"],
        "category": "dietary",
    },
    {
        "query": "What activities can I do this weekend?",
        "expected_memories": ["mem_hobby_en_1"],
        "category": "hobby",
    },
]


@dataclass
class SampleMemory:
    """Sample memory data structure."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[torch.Tensor] = None


def create_sample_memory(
    memory_id: str = "test_mem",
    content: str = "This is a test memory",
    metadata: Optional[Dict[str, Any]] = None,
    embedding_dim: int = 384,
) -> SampleMemory:
    """
    Create a sample memory for testing.
    
    Args:
        memory_id: Memory identifier
        content: Memory content
        metadata: Optional metadata
        embedding_dim: Embedding dimension
        
    Returns:
        SampleMemory instance
    """
    torch.manual_seed(hash(memory_id) % 2**32)
    
    embedding = torch.randn(embedding_dim)
    embedding = embedding / embedding.norm()  # Normalize
    
    return SampleMemory(
        id=memory_id,
        content=content,
        metadata=metadata or {},
        embedding=embedding,
    )


def create_sample_kv_entries(
    num_layers: int = 4,
    batch_size: int = 1,
    num_heads: int = 4,
    seq_len: int = 10,
    head_dim: int = 32,
    device: str = "cpu",
    seed: int = 42,
) -> List[KVCacheEntry]:
    """
    Create sample K/V cache entries for testing.
    
    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device for tensors
        seed: Random seed for reproducibility
        
    Returns:
        List of KVCacheEntry for each layer
    """
    torch.manual_seed(seed)
    
    entries = []
    for layer_idx in range(num_layers):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        entries.append(KVCacheEntry(
            key=key,
            value=value,
            layer_idx=layer_idx,
        ))
    
    return entries


def create_sample_conversation() -> List[Dict[str, str]]:
    """Create sample conversation history."""
    return [
        {"role": "user", "content": "你好，我想找一家餐厅"},
        {"role": "assistant", "content": "好的，请问您有什么饮食偏好吗？"},
        {"role": "user", "content": "我是素食主义者"},
        {"role": "assistant", "content": "明白了，我为您推荐几家素食餐厅..."},
        {"role": "user", "content": "有没有离我近一点的？"},
    ]


def create_sample_gating_scenarios() -> List[Dict[str, Any]]:
    """
    Create sample scenarios for testing gating decisions.
    
    Returns scenarios with expected gating behavior.
    """
    return [
        {
            "name": "high_entropy_high_relevance",
            "entropy": 0.8,
            "relevance": 0.9,
            "expected_inject": True,
            "expected_alpha_range": (0.6, 1.0),
            "description": "High uncertainty with relevant memory → Strong injection",
        },
        {
            "name": "high_entropy_low_relevance",
            "entropy": 0.8,
            "relevance": 0.3,
            "expected_inject": False,
            "expected_alpha_range": (0.0, 0.1),
            "description": "High uncertainty but no relevant memory → No injection",
        },
        {
            "name": "low_entropy_high_relevance",
            "entropy": 0.3,
            "relevance": 0.9,
            "expected_inject": True,
            "expected_alpha_range": (0.3, 0.7),
            "description": "Model confident but relevant memory → Moderate injection",
        },
        {
            "name": "low_entropy_low_relevance",
            "entropy": 0.3,
            "relevance": 0.3,
            "expected_inject": False,
            "expected_alpha_range": (0.0, 0.1),
            "description": "Model confident and no relevant memory → No injection",
        },
    ]


def create_sample_budget_scenarios() -> List[Dict[str, Any]]:
    """
    Create sample scenarios for testing budget analysis.
    
    Returns scenarios with expected budget behavior.
    """
    return [
        {
            "name": "small_memory",
            "user_tokens": 100,
            "memory_tokens": 50,
            "context_window": 4096,
            "expected_dki_recommended": False,
            "description": "Small memory, DKI overhead not justified",
        },
        {
            "name": "large_memory_constrained_context",
            "user_tokens": 500,
            "memory_tokens": 1000,
            "context_window": 2048,
            "expected_dki_recommended": True,
            "description": "Large memory with constrained context → DKI recommended",
        },
        {
            "name": "moderate_memory",
            "user_tokens": 200,
            "memory_tokens": 300,
            "context_window": 4096,
            "expected_dki_recommended": True,
            "description": "Moderate memory size → DKI beneficial",
        },
    ]
