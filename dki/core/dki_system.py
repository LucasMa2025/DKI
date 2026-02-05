"""
DKI System - Dynamic KV Injection
Main system implementation for attention-level memory augmentation

============================================================================
Key Distinction: DKI vs Cross-Attention (Paper Section 4.1)
============================================================================

DKI is NOT equivalent to Cross-Attention, despite both involving external K/V:

┌─────────────────────────────────────────────────────────────────────────┐
│                    Cross-Attention (e.g., Encoder-Decoder)              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Separate parameters: W_q^cross, W_k^cross, W_v^cross                 │
│  • Dedicated cross-attention layers (every N layers)                    │
│  • Training required: Cross-attention weights must be learned           │
│  • Fixed architecture: Cannot be added to decoder-only models           │
│  • Source: External encoder output (different semantic space)           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    DKI (Dynamic KV Injection)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  • Reuses existing parameters: W_k, W_v (same as self-attention)        │
│  • Injects into existing self-attention (no new layers)                 │
│  • Training-free: Works with frozen decoder-only models                 │
│  • Architecture-agnostic: Works with any transformer                    │
│  • Source: Same model's K/V computation (same semantic space)           │
│  • α-controllable: Continuous injection strength                        │
│  • Graceful degradation: α → 0 recovers vanilla behavior                │
└─────────────────────────────────────────────────────────────────────────┘

Mathematical Formulation:

Cross-Attention:
    Attn_cross(Q, K_enc, V_enc) = softmax(Q W_q^cross (K_enc W_k^cross)^T / √d) V_enc W_v^cross

DKI (Self-Attention with Injection):
    K_aug = [K_mem; K_user]  (concatenation)
    V_aug = [V_mem; V_user]
    Attn_dki(Q, K_aug, V_aug) = softmax(Q K_aug^T / √d) V_aug
    
    With α-scaling:
    Attn_dki(Q, K_aug, V_aug, α) = softmax(Q K_aug^T / √d + α·bias_mem) V_aug

Key Insight: DKI achieves memory augmentation WITHOUT:
1. Additional parameters
2. Architecture modifications
3. Model retraining
4. Separate encoder

This makes DKI a "plug-and-play" solution for any decoder-only LLM.
============================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger

from dki.core.memory_router import MemoryRouter, MemorySearchResult
from dki.core.embedding_service import EmbeddingService
from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling
from dki.core.components.query_conditioned_projection import QueryConditionedProjection
from dki.core.components.dual_factor_gating import DualFactorGating, GatingDecision
from dki.core.components.session_kv_cache import SessionKVCache
from dki.core.components.tiered_kv_cache import TieredKVCache, CacheTier
from dki.core.components.position_remapper import PositionRemapper
from dki.core.components.attention_budget import (
    AttentionBudgetAnalyzer, BudgetAnalysis, LatencyTimer, LatencyBreakdown
)
from dki.models.factory import ModelFactory
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository
)
from dki.config.config_loader import ConfigLoader


@dataclass
class DKIResponse:
    """
    DKI system response with comprehensive metadata.
    
    Includes:
    - Generation output
    - Memory usage information
    - Gating decision details
    - Budget analysis (token vs attention)
    - Latency breakdown
    - Cache statistics
    """
    text: str
    memories_used: List[MemorySearchResult]
    gating_decision: GatingDecision
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    cache_tier: str = "none"
    budget_analysis: Optional[BudgetAnalysis] = None
    latency_breakdown: Optional[LatencyBreakdown] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'text': self.text,
            'memories_used': [m.to_dict() for m in self.memories_used],
            'gating_decision': self.gating_decision.to_dict(),
            'alpha': self.gating_decision.alpha,
            'latency_ms': self.latency_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_hit': self.cache_hit,
            'cache_tier': self.cache_tier,
            'metadata': self.metadata,
        }
        
        if self.budget_analysis:
            result['budget_analysis'] = self.budget_analysis.to_dict()
        
        if self.latency_breakdown:
            result['latency_breakdown'] = self.latency_breakdown.to_dict()
        
        return result


class DKISystem:
    """
    Dynamic KV Injection System.
    
    Implements attention-level memory augmentation:
    1. Retrieve relevant memories
    2. Compute K/V representations
    3. Apply query-conditioned projection
    4. Inject into attention with α scaling
    5. Generate response
    
    Key Features:
    - Memory Influence Scaling (MIS): Continuous α ∈ [0, 1] control
    - Query-Conditioned Projection: FiLM-style adaptive projection
    - Dual-Factor Gating: Uncertainty × Relevance
    - Tiered KV Cache: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
    - Position Remapping: RoPE/ALiBi compatibility
    - Attention Budget Analysis: Token vs Attention budget tracking
    
    Design Invariants (Paper Section 5):
    1. Storage model-agnostic (text + routing vectors only)
    2. Injection model-consistent (K/V from target model)
    3. Session cache disposable (inference-time only)
    4. Graceful degradation (α → 0 recovers vanilla)
    5. Audit logging for compliance
    
    Memory Hierarchy (Paper Section 7.4):
    ┌─────────────────────────────────────────────────┐
    │  L1: GPU HBM (Hot)     - Uncompressed FP16     │
    │  L2: CPU RAM (Warm)    - Compressed (2-4×)     │
    │  L3: NVMe SSD (Cold)   - Quantized INT8 (8×)  │
    │  L4: Text Only         - Recompute on demand   │
    └─────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        model_adapter: Optional[BaseModelAdapter] = None,
        memory_router: Optional[MemoryRouter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        engine: Optional[str] = None,
    ):
        self.config = ConfigLoader().config
        
        # Core services
        self.embedding_service = embedding_service or EmbeddingService()
        self.memory_router = memory_router or MemoryRouter(self.embedding_service)
        
        # Model adapter (lazy loaded)
        self._model_adapter = model_adapter
        self._engine = engine
        
        # DKI components (lazy initialized)
        self._mis: Optional[MemoryInfluenceScaling] = None
        self._projection: Optional[QueryConditionedProjection] = None
        self._gating: Optional[DualFactorGating] = None
        self._position_remapper: Optional[PositionRemapper] = None
        
        # Cache system - supports both simple and tiered cache
        self._use_tiered_cache = self.config.dki.use_tiered_cache
        self._session_caches: Dict[str, SessionKVCache] = {}
        self._tiered_caches: Dict[str, TieredKVCache] = {}
        
        # Attention Budget Analyzer
        self._budget_analyzer = AttentionBudgetAnalyzer(
            context_window=self.config.model.engines.get(
                self.config.model.default_engine, {}
            ).max_model_len if hasattr(self.config.model.engines.get(
                self.config.model.default_engine, {}
            ), 'max_model_len') else 4096
        )
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        logger.info(f"DKI System initialized (tiered_cache={self._use_tiered_cache})")
    
    @property
    def model(self) -> BaseModelAdapter:
        """Get or create model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelFactory.get_or_create(engine=self._engine)
        return self._model_adapter
    
    @property
    def mis(self) -> MemoryInfluenceScaling:
        """Get or create MIS component."""
        if self._mis is None:
            self._mis = MemoryInfluenceScaling(
                hidden_dim=self.model.hidden_dim,
                use_learned_alpha=False,  # Start with heuristic
            )
        return self._mis
    
    @property
    def projection(self) -> QueryConditionedProjection:
        """Get or create projection component."""
        if self._projection is None:
            self._projection = QueryConditionedProjection(
                hidden_dim=self.model.hidden_dim,
            )
        return self._projection
    
    @property
    def gating(self) -> DualFactorGating:
        """Get or create gating component."""
        if self._gating is None:
            self._gating = DualFactorGating()
        return self._gating
    
    @property
    def position_remapper(self) -> PositionRemapper:
        """Get or create position remapper."""
        if self._position_remapper is None:
            pos_encoding = PositionRemapper().detect_position_encoding(self.model.model_name)
            self._position_remapper = PositionRemapper(position_encoding=pos_encoding)
        return self._position_remapper
    
    def _get_session_cache(self, session_id: str) -> Union[SessionKVCache, TieredKVCache]:
        """
        Get or create session-specific KV cache.
        
        Returns TieredKVCache if enabled, otherwise SessionKVCache.
        """
        if self._use_tiered_cache:
            if session_id not in self._tiered_caches:
                tiered_config = self.config.dki.tiered_cache
                self._tiered_caches[session_id] = TieredKVCache(
                    l1_max_size=tiered_config.l1_max_size,
                    l2_max_size=tiered_config.l2_max_size,
                    l3_path=f"{tiered_config.l3_path}/{session_id}",
                    enable_l3=tiered_config.enable_l3,
                    enable_l4=tiered_config.enable_l4,
                    ttl_seconds=tiered_config.ttl_seconds,
                )
            return self._tiered_caches[session_id]
        else:
            if session_id not in self._session_caches:
                self._session_caches[session_id] = SessionKVCache()
            return self._session_caches[session_id]
    
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
            
            session_repo.get_or_create(session_id)
            
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
        """Load memories from database into router."""
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
    
    def chat(
        self,
        query: str,
        session_id: str,
        allow_injection: bool = True,
        force_alpha: Optional[float] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        task_type: str = "reasoning",
        **kwargs
    ) -> DKIResponse:
        """
        Generate response using DKI.
        
        Args:
            query: User query
            session_id: Session identifier
            allow_injection: Whether to allow memory injection
            force_alpha: Force specific alpha value (bypasses gating)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_type: Type of task for budget analysis
            
        Returns:
            DKIResponse with generated text, metadata, and budget analysis
        """
        # Start latency timer
        with LatencyTimer() as timer:
            cache_hit = False
            cache_tier = "none"
            
            # Get session cache
            session_cache = self._get_session_cache(session_id)
            
            # Step 1: Gating decision
            timer.start_stage("gating")
            if not allow_injection:
                gating_decision = GatingDecision(
                    should_inject=False,
                    alpha=0.0,
                    entropy=0.0,
                    relevance_score=0.0,
                    margin=0.0,
                    memories=[],
                    reasoning="Injection disabled by caller",
                )
            elif force_alpha is not None:
                gating_decision = self.gating.force_inject(
                    router=self.memory_router,
                    query=query,
                    alpha=force_alpha,
                )
            else:
                gating_decision = self.gating.should_inject(
                    model=self.model,
                    query=query,
                    router=self.memory_router,
                )
            timer.end_stage()
            
            # Estimate tokens for budget analysis
            user_tokens = len(query.split()) * 1.3  # Rough estimate
            memory_tokens = sum(
                len(m.content.split()) * 1.3 
                for m in gating_decision.memories
            ) if gating_decision.memories else 0
            
            # Budget analysis
            budget_analysis = self._budget_analyzer.analyze(
                user_tokens=int(user_tokens),
                memory_tokens=int(memory_tokens),
            )
            
            # Step 2: Generate with or without injection
            if not gating_decision.should_inject or gating_decision.alpha < 0.1:
                # Fallback to vanilla generation (graceful degradation)
                timer.start_stage("prefill")
                output = self.model.generate(
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )
                timer.end_stage()
                memories_used = []
            else:
                # DKI injection flow
                output, cache_hit, cache_tier = self._generate_with_injection(
                    query=query,
                    gating_decision=gating_decision,
                    session_cache=session_cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    timer=timer,
                    **kwargs
                )
                memories_used = gating_decision.memories
            
            timer.set_cache_info(cache_hit, cache_tier)
        
        # Record latency for analysis
        self._budget_analyzer.record_latency(timer.breakdown)
        
        # Log to database
        self._log_conversation(
            session_id=session_id,
            query=query,
            response=output.text,
            gating_decision=gating_decision,
            latency_ms=timer.breakdown.total_ms,
        )
        
        return DKIResponse(
            text=output.text,
            memories_used=memories_used,
            gating_decision=gating_decision,
            latency_ms=timer.breakdown.total_ms,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            cache_hit=cache_hit,
            cache_tier=cache_tier,
            budget_analysis=budget_analysis,
            latency_breakdown=timer.breakdown,
            metadata={
                'model': self.model.model_name,
                'session_cache_stats': session_cache.get_stats(),
                'task_type': task_type,
            },
        )
    
    def _generate_with_injection(
        self,
        query: str,
        gating_decision: GatingDecision,
        session_cache: Union[SessionKVCache, TieredKVCache],
        max_new_tokens: int,
        temperature: float,
        timer: Optional[LatencyTimer] = None,
        **kwargs
    ) -> tuple:
        """
        Generate response with K/V injection.
        
        Supports both simple SessionKVCache and TieredKVCache.
        
        Returns:
            (ModelOutput, cache_hit, cache_tier)
        """
        cache_hit = False
        cache_tier = "none"
        
        # Collect K/V from memories
        all_kv_entries: List[List[KVCacheEntry]] = []
        
        for memory in gating_decision.memories:
            # Try cache first
            if timer:
                timer.start_stage("kv_load")
            
            if isinstance(session_cache, TieredKVCache):
                # Tiered cache returns (kv_entries, tier)
                cached_kv, tier = session_cache.get(
                    memory.memory_id, 
                    query=query,
                    model=self.model,  # For L4 recompute
                )
                if cached_kv is not None:
                    all_kv_entries.append(cached_kv)
                    cache_hit = True
                    cache_tier = tier.value
            else:
                # Simple cache
                cached_kv = session_cache.get(memory.memory_id, query=query)
                if cached_kv is not None:
                    all_kv_entries.append(cached_kv)
                    cache_hit = True
                    cache_tier = "session"
            
            if timer:
                timer.end_stage()
            
            if cached_kv is None:
                # Compute K/V
                if timer:
                    timer.start_stage("kv_compute")
                
                kv_entries, _ = self.model.compute_kv(memory.content)
                
                if timer:
                    timer.end_stage()
                
                # Store in cache
                if isinstance(session_cache, TieredKVCache):
                    session_cache.put(
                        memory_id=memory.memory_id,
                        kv_entries=kv_entries,
                        query=query,
                        alpha=gating_decision.alpha,
                        text_content=memory.content,  # For L4 recompute
                    )
                else:
                    session_cache.put(
                        memory_id=memory.memory_id,
                        entries=kv_entries,
                        query=query,
                        alpha=gating_decision.alpha,
                    )
                
                all_kv_entries.append(kv_entries)
        
        # Merge K/V from multiple memories
        if len(all_kv_entries) > 1:
            merged_kv = self._merge_kv_entries(all_kv_entries)
        elif len(all_kv_entries) == 1:
            merged_kv = all_kv_entries[0]
        else:
            merged_kv = []
        
        # Apply alpha scaling to K/V (MIS)
        if timer:
            timer.start_stage("projection")
        
        if merged_kv and gating_decision.alpha < 1.0:
            scaled_kv = []
            for entry in merged_kv:
                key, value = self.mis.scale_kv_values(
                    entry.key, entry.value, gating_decision.alpha
                )
                scaled_kv.append(KVCacheEntry(key=key, value=value, layer_idx=entry.layer_idx))
            merged_kv = scaled_kv
        
        if timer:
            timer.end_stage()
        
        # Generate with injected K/V
        if timer:
            timer.start_stage("prefill")
        
        output = self.model.forward_with_kv_injection(
            prompt=query,
            injected_kv=merged_kv,
            alpha=gating_decision.alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        if timer:
            timer.end_stage()
        
        return output, cache_hit, cache_tier
    
    def _merge_kv_entries(
        self,
        kv_list: List[List[KVCacheEntry]],
    ) -> List[KVCacheEntry]:
        """
        Merge K/V entries from multiple memories.
        
        Concatenates K/V along sequence dimension for each layer.
        """
        if not kv_list:
            return []
        
        num_layers = len(kv_list[0])
        merged = []
        
        for layer_idx in range(num_layers):
            # Collect K/V for this layer from all memories
            keys = [kv[layer_idx].key for kv in kv_list]
            values = [kv[layer_idx].value for kv in kv_list]
            
            # Concatenate along sequence dimension
            merged_key = torch.cat(keys, dim=2)  # [batch, heads, total_seq, head_dim]
            merged_value = torch.cat(values, dim=2)
            
            merged.append(KVCacheEntry(
                key=merged_key,
                value=merged_value,
                layer_idx=layer_idx,
            ))
        
        return merged
    
    def _log_conversation(
        self,
        session_id: str,
        query: str,
        response: str,
        gating_decision: GatingDecision,
        latency_ms: float,
    ) -> None:
        """Log conversation to database."""
        with self.db_manager.session_scope() as db:
            conv_repo = ConversationRepository(db)
            audit_repo = AuditLogRepository(db)
            
            # User message
            conv_repo.create(
                session_id=session_id,
                role='user',
                content=query,
            )
            
            # Assistant response
            conv_repo.create(
                session_id=session_id,
                role='assistant',
                content=response,
                injection_mode='dki' if gating_decision.should_inject else 'none',
                injection_alpha=gating_decision.alpha,
                memory_ids=[m.memory_id for m in gating_decision.memories],
                latency_ms=latency_ms,
            )
            
            # Audit log
            audit_repo.log(
                action='dki_generate',
                session_id=session_id,
                memory_ids=[m.memory_id for m in gating_decision.memories],
                alpha=gating_decision.alpha,
                mode='dki',
                metadata={
                    'entropy': gating_decision.entropy,
                    'relevance': gating_decision.relevance_score,
                    'reasoning': gating_decision.reasoning,
                },
            )
    
    def clear_session_cache(self, session_id: str) -> None:
        """Clear KV cache for a session."""
        if session_id in self._session_caches:
            self._session_caches[session_id].clear()
            del self._session_caches[session_id]
        
        if session_id in self._tiered_caches:
            self._tiered_caches[session_id].clear()
            del self._tiered_caches[session_id]
    
    def search_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[MemorySearchResult]:
        """Search memories without generation."""
        return self.memory_router.search(query, top_k=top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # Session cache stats
        session_stats = {
            sid: cache.get_stats()
            for sid, cache in self._session_caches.items()
        }
        
        # Tiered cache stats
        tiered_stats = {
            sid: cache.get_stats()
            for sid, cache in self._tiered_caches.items()
        }
        
        # Memory footprint for tiered caches
        tiered_footprint = {
            sid: cache.get_memory_footprint()
            for sid, cache in self._tiered_caches.items()
        }
        
        return {
            'router_stats': self.memory_router.get_stats(),
            'model_info': self.model.get_model_info() if self._model_adapter else None,
            'session_caches': session_stats,
            'tiered_caches': tiered_stats,
            'tiered_memory_footprint': tiered_footprint,
            'budget_analysis': self._budget_analyzer.get_stats(),
            'config': {
                'use_tiered_cache': self._use_tiered_cache,
                'gating': {
                    'entropy_threshold': self.config.dki.gating.entropy_threshold,
                    'relevance_threshold': self.config.dki.gating.relevance_threshold,
                },
                'projection': {
                    'rank': self.config.dki.projection.rank,
                },
                'cache': {
                    'max_size': self.config.dki.cache.max_size,
                    'strategy': self.config.dki.cache.strategy,
                },
                'tiered_cache': {
                    'l1_max_size': self.config.dki.tiered_cache.l1_max_size,
                    'l2_max_size': self.config.dki.tiered_cache.l2_max_size,
                    'enable_l3': self.config.dki.tiered_cache.enable_l3,
                    'enable_l4': self.config.dki.tiered_cache.enable_l4,
                },
            },
        }
    
    def should_use_dki(
        self,
        query: str,
        memory_count: int,
        task_type: str = "reasoning",
    ) -> Dict[str, Any]:
        """
        Recommend whether to use DKI based on budget analysis.
        
        See Paper Section 7.3 for guidelines.
        
        Args:
            query: User query
            memory_count: Number of memories to potentially inject
            task_type: Type of task
            
        Returns:
            Recommendation with reasoning
        """
        # Estimate tokens
        user_tokens = int(len(query.split()) * 1.3)
        # Assume average memory is 200 tokens
        memory_tokens = memory_count * 200
        
        return self._budget_analyzer.should_prefer_dki(
            user_tokens=user_tokens,
            memory_tokens=memory_tokens,
            task_type=task_type,
        )
    
    def get_budget_analysis(
        self,
        user_tokens: int,
        memory_tokens: int,
    ) -> BudgetAnalysis:
        """Get token/attention budget analysis for RAG vs DKI."""
        return self._budget_analyzer.analyze(user_tokens, memory_tokens)