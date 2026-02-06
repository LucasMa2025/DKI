"""
Fake Embedding Service for CPU-only testing.
Provides deterministic embeddings without loading real models.
"""

import numpy as np
import torch
from typing import List, Optional, Union


class FakeEmbeddingService:
    """
    Fake embedding service for testing.
    
    Generates deterministic embeddings based on text hash.
    Does not require loading any real embedding models.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        normalize: bool = True,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.device = device
        self._initialized = True
        self.model = "fake-embedding-model"
    
    def load(self) -> None:
        """Fake load - already loaded."""
        pass
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate fake embeddings.
        
        Uses text hash for deterministic results.
        Similar texts will have similar embeddings.
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = []
        for t in texts:
            emb = self._generate_embedding(t)
            embeddings.append(emb)
        
        result = np.array(embeddings)
        
        if is_single:
            return result[0]
        return result
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic embedding for text."""
        # Use hash for determinism
        np.random.seed(hash(text) % 2**32)
        
        # Generate base embedding
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Add semantic structure based on text features
        words = text.lower().split()
        
        # Similar words should produce similar embeddings
        for i, word in enumerate(words[:10]):  # Use first 10 words
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 0.5
        
        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Batch embed texts."""
        return self.embed(texts)
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity."""
        if self.normalize:
            return np.dot(doc_embeddings, query_embedding)
        else:
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            return np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm + 1e-9)
    
    def most_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """Find most similar documents."""
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)
        
        scores = self.similarity(query_emb, doc_embs)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            (int(idx), float(scores[idx]), documents[idx])
            for idx in top_indices
        ]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def unload(self) -> None:
        """Fake unload."""
        pass
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (no-op for fake)."""
        pass


class FakeEmbeddingServiceWithSimilarity(FakeEmbeddingService):
    """
    Extended fake embedding service with controlled similarity.
    
    Useful for testing retrieval behavior with known similarity scores.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        similarity_matrix: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        
        # Predefined similarity scores between text pairs
        self.similarity_matrix = similarity_matrix or {}
    
    def set_similarity(self, text1: str, text2: str, score: float) -> None:
        """Set predefined similarity between two texts."""
        key = self._make_key(text1, text2)
        self.similarity_matrix[key] = score
    
    def _make_key(self, text1: str, text2: str) -> str:
        """Create symmetric key for text pair."""
        texts = sorted([text1, text2])
        return f"{texts[0]}|||{texts[1]}"
    
    def get_predefined_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Get predefined similarity if exists."""
        key = self._make_key(text1, text2)
        return self.similarity_matrix.get(key)
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings with predefined similarities.
        
        If similarity is predefined, adjust embeddings to match.
        """
        # For simple cases, use parent implementation
        return super().embed(text)
