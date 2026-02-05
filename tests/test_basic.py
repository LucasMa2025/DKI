"""
Basic tests for DKI system components
"""

import pytest
import numpy as np
import torch

# Test configuration
def test_config_loader():
    """Test configuration loading."""
    from dki.config.config_loader import ConfigLoader, get_config
    
    # Reset singleton
    ConfigLoader.reset()
    
    config = get_config()
    
    assert config is not None
    assert config.system.name == "DKI System"
    assert config.dki.enabled == True
    assert config.rag.enabled == True


def test_database_manager():
    """Test database manager initialization."""
    from dki.database.connection import DatabaseManager
    
    # Use in-memory database for testing
    db_manager = DatabaseManager(db_path=":memory:")
    
    assert db_manager is not None
    assert db_manager.engine is not None
    
    # Test session creation
    session = db_manager.get_session()
    assert session is not None
    session.close()
    
    DatabaseManager.reset_instance()


def test_memory_repository():
    """Test memory repository operations."""
    from dki.database.connection import DatabaseManager
    from dki.database.repository import SessionRepository, MemoryRepository
    
    DatabaseManager.reset_instance()
    db_manager = DatabaseManager(db_path=":memory:")
    
    with db_manager.session_scope() as db:
        session_repo = SessionRepository(db)
        memory_repo = MemoryRepository(db)
        
        # Create session
        session = session_repo.create(session_id="test_session", user_id="test_user")
        assert session.id == "test_session"
        
        # Create memory
        memory = memory_repo.create(
            session_id="test_session",
            content="Test memory content",
            metadata={"test": True}
        )
        
        assert memory.id is not None
        assert memory.content == "Test memory content"
        
        # Retrieve memory
        retrieved = memory_repo.get(memory.id)
        assert retrieved is not None
        assert retrieved.content == memory.content
        
        # List memories
        memories = memory_repo.get_by_session("test_session")
        assert len(memories) == 1
    
    DatabaseManager.reset_instance()


def test_embedding_service():
    """Test embedding service."""
    from dki.core.embedding_service import EmbeddingService
    
    EmbeddingService.reset_instance()
    
    # Use CPU for testing
    service = EmbeddingService(device="cpu")
    
    # Test single embedding
    text = "This is a test sentence."
    embedding = service.embed(text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0
    
    # Test batch embedding
    texts = ["First sentence.", "Second sentence."]
    embeddings = service.embed(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == 2
    
    EmbeddingService.reset_instance()


def test_memory_router():
    """Test memory router."""
    from dki.core.memory_router import MemoryRouter
    from dki.core.embedding_service import EmbeddingService
    
    EmbeddingService.reset_instance()
    
    embedding_service = EmbeddingService(device="cpu")
    router = MemoryRouter(embedding_service)
    
    # Add memories
    router.add_memory("mem1", "I like vegetarian food")
    router.add_memory("mem2", "I enjoy hiking in mountains")
    router.add_memory("mem3", "I work as a software engineer")
    
    # Search
    results = router.search("What should I eat?", top_k=2)
    
    assert len(results) > 0
    assert results[0].memory_id in ["mem1", "mem2", "mem3"]
    assert results[0].score > 0
    
    # Check stats
    stats = router.get_stats()
    assert stats['total_memories'] == 3
    
    EmbeddingService.reset_instance()


def test_memory_influence_scaling():
    """Test MIS component."""
    from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling
    
    mis = MemoryInfluenceScaling(hidden_dim=128, use_learned_alpha=False)
    
    # Test alpha computation
    query_emb = torch.randn(128)
    alpha = mis.compute_alpha(query_emb, memory_relevance=0.8, entropy=0.6)
    
    assert 0.0 <= alpha <= 1.0
    
    # Test scaling
    logits = torch.randn(1, 4, 10, 20)  # [batch, heads, seq, total]
    scaled = mis.apply_scaling(logits, mem_len=5, alpha=0.5)
    
    assert scaled.shape == logits.shape


def test_query_conditioned_projection():
    """Test Query-Conditioned Projection."""
    from dki.core.components.query_conditioned_projection import QueryConditionedProjection
    
    projection = QueryConditionedProjection(hidden_dim=128, rank=16)
    
    # Test forward pass
    X_mem = torch.randn(10, 128)  # [mem_len, hidden_dim]
    X_user = torch.randn(5, 128)  # [user_len, hidden_dim]
    
    output = projection(X_mem, X_user)
    
    assert output.shape == X_mem.shape
    
    # Test with batch dimension
    X_mem_batch = torch.randn(2, 10, 128)
    X_user_batch = torch.randn(2, 5, 128)
    
    output_batch = projection(X_mem_batch, X_user_batch)
    assert output_batch.shape == X_mem_batch.shape
    
    # Check parameter count
    param_count = projection.get_parameter_count()
    assert param_count > 0
    print(f"Projection parameter count: {param_count}")
    print(f"Memory overhead: {projection.get_memory_overhead()}")


def test_dual_factor_gating():
    """Test Dual-Factor Gating (without model)."""
    from dki.core.components.dual_factor_gating import DualFactorGating, GatingDecision
    
    gating = DualFactorGating(
        entropy_threshold=0.5,
        relevance_threshold=0.7
    )
    
    # Test threshold update
    gating.update_thresholds(entropy_threshold=0.4)
    assert gating.entropy_threshold == 0.4


def test_session_kv_cache():
    """Test Session KV Cache."""
    from dki.core.components.session_kv_cache import SessionKVCache
    from dki.models.base import KVCacheEntry
    
    cache = SessionKVCache(max_size=5, strategy="lru")
    
    # Create dummy KV entries
    entries = [
        KVCacheEntry(
            key=torch.randn(1, 4, 10, 32),
            value=torch.randn(1, 4, 10, 32),
            layer_idx=i
        )
        for i in range(4)
    ]
    
    # Test put and get
    cache.put("mem1", entries, alpha=0.5)
    
    retrieved = cache.get("mem1")
    assert retrieved is not None
    assert len(retrieved) == 4
    
    # Test cache miss
    missed = cache.get("nonexistent")
    assert missed is None
    
    # Test stats
    stats = cache.get_stats()
    assert stats['size'] == 1
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    
    # Test eviction
    for i in range(10):
        cache.put(f"mem_{i}", entries, alpha=0.5)
    
    assert len(cache) <= 5


def test_position_remapper():
    """Test Position Remapper."""
    from dki.core.components.position_remapper import PositionRemapper
    
    remapper = PositionRemapper()
    
    # Test position encoding detection
    assert remapper.detect_position_encoding("meta-llama/Llama-2-7b") == "rope"
    assert remapper.detect_position_encoding("bigscience/bloom") == "alibi"
    assert remapper.detect_position_encoding("THUDM/glm-4") == "rope"
    
    # Test mask extension
    mask = torch.ones(2, 10)
    extended = remapper.get_extended_attention_mask(mask, mem_len=5, device=mask.device)
    assert extended.shape == (2, 15)


def test_metrics_calculator():
    """Test metrics calculator."""
    from dki.experiment.metrics import MetricsCalculator
    
    metrics = MetricsCalculator()
    
    # Test BLEU
    ref = "The cat sat on the mat"
    hyp = "A cat is sitting on the mat"
    bleu = metrics.compute_bleu(ref, hyp)
    assert 0.0 <= bleu <= 1.0
    
    # Test ROUGE
    rouge = metrics.compute_rouge(ref, hyp)
    assert 'rouge1' in rouge
    assert 'rouge2' in rouge
    assert 'rougeL' in rouge
    
    # Test latency stats
    latencies = [100, 150, 200, 120, 180]
    stats = metrics.compute_latency_stats(latencies)
    assert 'p50' in stats
    assert 'p95' in stats
    assert 'mean' in stats
    
    # Test memory recall
    memories = ["I like coffee", "I work in tech"]
    response = "Since you enjoy coffee, I recommend this cafe where tech workers gather."
    recall, matched = metrics.compute_memory_recall(memories, response)
    assert 0.0 <= recall <= 1.0


def test_data_generator():
    """Test experiment data generator."""
    import tempfile
    from dki.experiment.data_generator import ExperimentDataGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ExperimentDataGenerator(tmpdir)
        
        # Generate small datasets
        persona_data = generator.generate_persona_chat(n_sessions=5)
        assert len(persona_data) == 5
        
        hotpot_data = generator.generate_hotpot_qa(n_samples=5)
        assert len(hotpot_data) == 5
        
        memory_qa_data = generator.generate_memory_qa(n_samples=5)
        assert len(memory_qa_data) == 5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
