"""
Memory Router for DKI System
FAISS-based semantic memory retrieval
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from dki.core.embedding_service import EmbeddingService
from dki.config.config_loader import ConfigLoader


@dataclass
class MemorySearchResult:
    """Search result from memory router."""
    memory_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata or {},
        }


class MemoryRouter:
    """
    FAISS-based memory router for semantic retrieval.
    
    Provides:
    - Memory indexing
    - Semantic search
    - Top-k retrieval with similarity scores
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        index_type: str = "faiss",
    ):
        config = ConfigLoader().config
        
        self.embedding_service = embedding_service or EmbeddingService()
        self.index_type = index_type
        self.top_k = config.rag.top_k
        self.similarity_threshold = config.rag.similarity_threshold
        
        # FAISS index
        self.index = None
        self.embedding_dim = None
        
        # Memory storage
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._needs_rebuild: bool = False
    
    def _init_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            
            self.embedding_dim = self.embedding_service.get_dimension()
            
            # Create FAISS index (L2 distance, then convert to similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine sim
            
            logger.info(f"FAISS index initialized, dim={self.embedding_dim}")
            
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            raise
    
    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a memory to the router.
        
        Args:
            memory_id: Unique memory identifier
            content: Memory text content
            embedding: Pre-computed embedding (optional)
            metadata: Additional metadata
        """
        if self.index is None:
            self._init_index()
        
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embedding_service.embed(content)
        
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity with inner product
        import faiss
        faiss.normalize_L2(embedding)
        
        # Add to index
        idx = len(self._memories)
        self.index.add(embedding)
        
        # Store memory info
        self._memories[memory_id] = {
            'content': content,
            'embedding': embedding.flatten(),
            'metadata': metadata or {},
        }
        self._id_to_idx[memory_id] = idx
        self._idx_to_id[idx] = memory_id
        
        logger.debug(f"Added memory: {memory_id}")
    
    def add_memories(
        self,
        memories: List[Dict[str, Any]],
    ) -> int:
        """
        Batch add memories.
        
        Args:
            memories: List of dicts with 'id', 'content', optional 'embedding', 'metadata'
            
        Returns:
            Number of memories added
        """
        if self.index is None:
            self._init_index()
        
        # Collect texts for batch embedding
        texts_to_embed = []
        indices_to_embed = []
        
        for i, mem in enumerate(memories):
            if 'embedding' not in mem or mem['embedding'] is None:
                texts_to_embed.append(mem['content'])
                indices_to_embed.append(i)
        
        # Batch embed
        if texts_to_embed:
            embeddings = self.embedding_service.embed(texts_to_embed)
            for i, idx in enumerate(indices_to_embed):
                memories[idx]['embedding'] = embeddings[i]
        
        # Add all memories
        for mem in memories:
            self.add_memory(
                memory_id=mem['id'],
                content=mem['content'],
                embedding=mem.get('embedding'),
                metadata=mem.get('metadata'),
            )
        
        return len(memories)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
            threshold: Minimum similarity threshold
            
        Returns:
            List of MemorySearchResult sorted by score
        """
        if self.index is None or len(self._memories) == 0:
            return []
        
        # Auto-rebuild index if needed (after remove_memory)
        if self._needs_rebuild:
            self.rebuild_index()
        
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        # Embed query
        query_embedding = self.embedding_service.embed(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity with inner product
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(top_k, len(self._memories))
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            
            if score < threshold:
                continue
            
            memory_id = self._idx_to_id.get(idx)
            if memory_id is None:
                continue
            
            memory = self._memories[memory_id]
            results.append(MemorySearchResult(
                memory_id=memory_id,
                content=memory['content'],
                score=float(score),
                metadata=memory.get('metadata'),
            ))
        
        return results
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID."""
        return self._memories.get(memory_id)
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove memory from router.
        
        Note: FAISS doesn't support efficient deletion.
        We mark as deleted and rebuild index periodically.
        """
        if memory_id in self._memories:
            del self._memories[memory_id]
            # Mark for rebuild
            self._needs_rebuild = True
            return True
        return False
    
    def rebuild_index(self) -> None:
        """Rebuild FAISS index from current memories."""
        if not self._memories:
            return
        
        self._init_index()
        
        # Re-add all memories
        old_memories = list(self._memories.items())
        self._memories.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        
        for memory_id, memory in old_memories:
            self.add_memory(
                memory_id=memory_id,
                content=memory['content'],
                embedding=memory['embedding'],
                metadata=memory.get('metadata'),
            )
        
        self._needs_rebuild = False
        logger.info(f"Rebuilt index with {len(self._memories)} memories")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            'total_memories': len(self._memories),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
        }
    
    def clear(self) -> None:
        """Clear all memories and reset index."""
        self._memories.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self.index = None
        logger.info("Memory router cleared")
