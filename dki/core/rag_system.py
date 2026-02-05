"""
RAG System for DKI
Retrieval-Augmented Generation implementation as baseline
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.core.memory_router import MemoryRouter, MemorySearchResult
from dki.core.embedding_service import EmbeddingService
from dki.models.factory import ModelFactory
from dki.models.base import BaseModelAdapter, ModelOutput
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository
)
from dki.config.config_loader import ConfigLoader


@dataclass
class RAGResponse:
    """RAG system response."""
    text: str
    memories_used: List[MemorySearchResult]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'memories_used': [m.to_dict() for m in self.memories_used],
            'latency_ms': self.latency_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'metadata': self.metadata or {},
        }


class RAGSystem:
    """
    Retrieval-Augmented Generation System.
    
    Implements the standard RAG paradigm:
    1. Retrieve relevant memories
    2. Concatenate to prompt
    3. Generate response
    """
    
    def __init__(
        self,
        model_adapter: Optional[BaseModelAdapter] = None,
        memory_router: Optional[MemoryRouter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        engine: Optional[str] = None,
    ):
        self.config = ConfigLoader().config
        
        # Initialize components
        self.embedding_service = embedding_service or EmbeddingService()
        self.memory_router = memory_router or MemoryRouter(self.embedding_service)
        
        # Model adapter (lazy loaded)
        self._model_adapter = model_adapter
        self._engine = engine
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        logger.info("RAG System initialized")
    
    @property
    def model(self) -> BaseModelAdapter:
        """Get or create model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelFactory.get_or_create(engine=self._engine)
        return self._model_adapter
    
    def add_memory(
        self,
        session_id: str,
        content: str,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a memory to the system.
        
        Args:
            session_id: Session identifier
            content: Memory content
            memory_id: Optional memory ID
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        # Compute embedding
        embedding = self.embedding_service.embed(content)
        
        # Store in database
        with self.db_manager.session_scope() as db:
            session_repo = SessionRepository(db)
            memory_repo = MemoryRepository(db)
            
            # Ensure session exists
            session_repo.get_or_create(session_id)
            
            # Create memory
            memory = memory_repo.create(
                session_id=session_id,
                content=content,
                embedding=embedding,
                memory_id=memory_id,
                metadata=metadata,
            )
            memory_id = memory.id
        
        # Add to router
        self.memory_router.add_memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
        
        logger.debug(f"Added memory: {memory_id}")
        return memory_id
    
    def load_memories_from_db(self, session_id: str) -> int:
        """
        Load memories from database into router.
        
        Args:
            session_id: Session to load memories for
            
        Returns:
            Number of memories loaded
        """
        with self.db_manager.session_scope() as db:
            memory_repo = MemoryRepository(db)
            memories = memory_repo.get_by_session(session_id)
            
            count = 0
            for mem in memories:
                embedding = memory_repo.get_embedding(mem.id)
                self.memory_router.add_memory(
                    memory_id=mem.id,
                    content=mem.content,
                    embedding=embedding,
                    metadata=mem.metadata,
                )
                count += 1
        
        logger.info(f"Loaded {count} memories for session {session_id}")
        return count
    
    def _build_prompt(
        self,
        query: str,
        memories: List[MemorySearchResult],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build prompt with retrieved memories.
        
        Args:
            query: User query
            memories: Retrieved memories
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt
        """
        parts = []
        
        # System prompt
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        
        # Retrieved context
        if memories:
            parts.append("Relevant information:")
            for i, mem in enumerate(memories, 1):
                parts.append(f"[{i}] {mem.content}")
            parts.append("")
        
        # User query
        parts.append(f"User: {query}")
        parts.append("\nAssistant:")
        
        return "\n".join(parts)
    
    def chat(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> RAGResponse:
        """
        Generate response using RAG.
        
        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of memories to retrieve
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            RAGResponse with generated text and metadata
        """
        start_time = time.perf_counter()
        
        # Retrieve relevant memories
        top_k = top_k or self.config.rag.top_k
        memories = self.memory_router.search(query, top_k=top_k)
        
        # Build prompt
        prompt = self._build_prompt(query, memories, system_prompt)
        
        # Generate response
        output = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000
        
        # Log to database
        with self.db_manager.session_scope() as db:
            conv_repo = ConversationRepository(db)
            audit_repo = AuditLogRepository(db)
            
            # Store user message
            conv_repo.create(
                session_id=session_id,
                role='user',
                content=query,
            )
            
            # Store assistant response
            conv_repo.create(
                session_id=session_id,
                role='assistant',
                content=output.text,
                injection_mode='rag',
                memory_ids=[m.memory_id for m in memories],
                latency_ms=total_latency,
            )
            
            # Audit log
            audit_repo.log(
                action='rag_generate',
                session_id=session_id,
                memory_ids=[m.memory_id for m in memories],
                mode='rag',
            )
        
        return RAGResponse(
            text=output.text,
            memories_used=memories,
            latency_ms=total_latency,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            metadata={
                'prompt_length': len(prompt),
                'model': self.model.model_name,
            },
        )
    
    def search_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[MemorySearchResult]:
        """
        Search memories without generation.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        return self.memory_router.search(query, top_k=top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'router_stats': self.memory_router.get_stats(),
            'model_info': self.model.get_model_info() if self._model_adapter else None,
            'config': {
                'top_k': self.config.rag.top_k,
                'similarity_threshold': self.config.rag.similarity_threshold,
            },
        }
