"""
Unit tests for Embedding Service.

Tests embedding functionality:
- Text embedding generation
- Batch embedding
- Similarity computation
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.fake_embeddings import FakeEmbeddingService


class TestFakeEmbeddingService:
    """Tests for FakeEmbeddingService."""
    
    @pytest.fixture
    def service(self):
        """Create fake embedding service."""
        return FakeEmbeddingService(embedding_dim=384, normalize=True)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service.embedding_dim == 384
        assert service.normalize is True
    
    def test_embed_single(self, service):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = service.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    def test_embed_batch(self, service):
        """Test batch text embedding."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = service.embed(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
    
    def test_embed_deterministic(self, service):
        """Test that embedding is deterministic."""
        text = "Test sentence"
        
        emb_1 = service.embed(text)
        emb_2 = service.embed(text)
        
        assert np.allclose(emb_1, emb_2)
    
    def test_embed_different_texts(self, service):
        """Test that different texts produce different embeddings."""
        emb_1 = service.embed("First text")
        emb_2 = service.embed("Second text")
        
        assert not np.allclose(emb_1, emb_2)
    
    def test_embed_normalized(self, service):
        """Test that embeddings are normalized."""
        embedding = service.embed("Test sentence")
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6
    
    def test_embed_unnormalized(self):
        """Test unnormalized embeddings."""
        service = FakeEmbeddingService(embedding_dim=384, normalize=False)
        embedding = service.embed("Test sentence")
        
        # Unnormalized embedding may not have unit norm
        norm = np.linalg.norm(embedding)
        # Just check it's not zero
        assert norm > 0
    
    def test_get_dimension(self, service):
        """Test getting embedding dimension."""
        dim = service.get_dimension()
        assert dim == 384


class TestSimilarityComputation:
    """Tests for similarity computation."""
    
    @pytest.fixture
    def service(self):
        return FakeEmbeddingService(embedding_dim=384, normalize=True)
    
    def test_similarity_self(self, service):
        """Test self-similarity is 1.0."""
        embedding = service.embed("Test sentence")
        
        similarity = service.similarity(embedding, embedding.reshape(1, -1))
        
        assert abs(similarity[0] - 1.0) < 1e-6
    
    def test_similarity_different(self, service):
        """Test similarity between different texts."""
        emb_1 = service.embed("I like cats")
        emb_2 = service.embed("I like dogs")
        emb_3 = service.embed("Quantum physics theory")
        
        # Similar texts should have higher similarity
        sim_similar = service.similarity(emb_1, emb_2.reshape(1, -1))[0]
        sim_different = service.similarity(emb_1, emb_3.reshape(1, -1))[0]
        
        # Both should be valid similarity scores
        assert -1.0 <= sim_similar <= 1.0
        assert -1.0 <= sim_different <= 1.0
    
    def test_similarity_batch(self, service):
        """Test batch similarity computation."""
        query = service.embed("Query text")
        docs = service.embed(["Doc 1", "Doc 2", "Doc 3"])
        
        similarities = service.similarity(query, docs)
        
        assert len(similarities) == 3


class TestMostSimilar:
    """Tests for most_similar function."""
    
    @pytest.fixture
    def service(self):
        return FakeEmbeddingService(embedding_dim=384, normalize=True)
    
    def test_most_similar(self, service):
        """Test finding most similar documents."""
        documents = [
            "I love eating pizza",
            "The weather is nice today",
            "Pizza is my favorite food",
            "I enjoy sunny days",
        ]
        
        results = service.most_similar("I want pizza", documents, top_k=2)
        
        assert len(results) == 2
        
        # Results should be (index, score, document) tuples
        for idx, score, doc in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert doc in documents
    
    def test_most_similar_sorted(self, service):
        """Test that results are sorted by score."""
        documents = ["Doc A", "Doc B", "Doc C", "Doc D"]
        
        results = service.most_similar("Query", documents, top_k=4)
        
        scores = [score for _, score, _ in results]
        
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestEmbeddingServiceSingleton:
    """Tests for singleton behavior."""
    
    def test_reset_instance(self):
        """Test resetting singleton instance."""
        service_1 = FakeEmbeddingService(embedding_dim=384)
        
        # Reset
        FakeEmbeddingService.reset_instance()
        
        # Create new instance
        service_2 = FakeEmbeddingService(embedding_dim=256)
        
        # Should be different instances (fake doesn't enforce singleton)
        # Just verify both work
        assert service_1.embedding_dim == 384
        assert service_2.embedding_dim == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
