"""
Embedding Service for DKI System
Provides text embedding functionality using sentence-transformers
"""

from typing import List, Optional, Union
import numpy as np
import torch
from loguru import logger

from dki.config.config_loader import ConfigLoader


class EmbeddingService:
    """
    Embedding service using sentence-transformers.
    
    Provides:
    - Text embedding generation
    - Batch embedding
    - Similarity computation
    """
    
    _instance: Optional['EmbeddingService'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        if self._initialized:
            return
        
        config = ConfigLoader().config
        
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.normalize = normalize
        self.batch_size = config.embedding.batch_size
        self.embedding_dim = config.memory.embedding_dim
        
        self.model = None
        self._initialized = True
    
    def load(self) -> None:
        """Load the embedding model."""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Update embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Embedding model loaded, dim={self.embedding_dim}")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Embeddings as numpy array [batch, dim] or [dim]
        """
        if self.model is None:
            self.load()
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        
        if is_single:
            return embeddings[0]
        
        return embeddings
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Batch embed texts with progress tracking."""
        if self.model is None:
            self.load()
        
        batch_size = batch_size or self.batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        
        return embeddings
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding [dim]
            doc_embeddings: Document embeddings [n_docs, dim]
            
        Returns:
            Similarity scores [n_docs]
        """
        if self.normalize:
            # Already normalized, just dot product
            return np.dot(doc_embeddings, query_embedding)
        else:
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            return np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm + 1e-9)
    
    def most_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar documents to query.
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Number of results to return
            
        Returns:
            List of (index, score, document) tuples
        """
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)
        
        scores = self.similarity(query_emb, doc_embs)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (int(idx), float(scores[idx]), documents[idx])
            for idx in top_indices
        ]
        
        return results
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            self.load()
        return self.embedding_dim
    
    def unload(self) -> None:
        """Unload embedding model."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance._initialized = False
            cls._instance = None
