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
from dki.core.components.hybrid_injector import (
    HybridDKIInjector, HybridInjectionConfig, UserPreference, SessionHistory
)
from dki.models.factory import ModelFactory
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository
)
from dki.config.config_loader import ConfigLoader

# Recall v4 组件 (可选, 通过配置启用)
try:
    from dki.core.recall import (
        RecallConfig,
        MultiSignalRecall,
        SuffixBuilder,
        FactRetriever,
        create_formatter,
    )
    from dki.core.recall.recall_config import FactRequest
    RECALL_V4_AVAILABLE = True
except ImportError:
    RECALL_V4_AVAILABLE = False


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
    Dynamic KV Injection System - User-Level Cross-Session Memory System.
    
    ============================================================================
    SCOPE DEFINITION (Paper Section 1.1)
    ============================================================================
    DKI is designed specifically for USER-LEVEL MEMORY:
    - User Preferences: Dietary restrictions, communication style, interests
    - Session History: Previous conversation context, established facts
    - Personal Context: Location, timezone, language preferences
    
    DKI is NOT for external knowledge bases or public data retrieval (use RAG).
    
    This focused scope enables:
    1. Short memory (50-200 tokens) → reduced position encoding risks
    2. User-owned data → simplified privacy considerations
    3. Session-coherent → effective K/V caching
    
    ============================================================================
    HYBRID INJECTION STRATEGY (Paper Section 3.9)
    ============================================================================
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Layer   │  Content     │  Injection        │  Influence      │ Analogy │
    ├──────────┼──────────────┼───────────────────┼─────────────────┼─────────┤
    │  L1      │  Preferences │  K/V (negative)   │  Implicit       │ Person. │
    │  L2      │  History     │  Suffix (positive)│  Explicit       │ Memory  │
    │  L3      │  Query       │  Input (positive) │  Primary focus  │ Current │
    └─────────────────────────────────────────────────────────────────────────┘
    
    - Preferences: Short, stable → Negative-position K/V injection (α=0.3-0.5)
    - History: Longer, dynamic → Suffix-prompt with trust-establishing guidance
    
    ============================================================================
    KEY FEATURES
    ============================================================================
    - Memory Influence Scaling (MIS): Continuous α ∈ [0, 1] control
    - Query-Conditioned Projection: FiLM-style adaptive projection
    - Dual-Factor Gating: Relevance-driven decision, entropy-modulated strength
    - Hybrid Injection: Preferences via K/V, History via suffix prompt
    - Tiered KV Cache: L1(GPU) → L2(CPU) → L3(SSD) → L4(Recompute)
    - Position Remapping: RoPE/ALiBi compatibility
    - Attention Budget Analysis: Token vs Attention budget tracking
    - Plugin Architecture: Configuration-driven, framework-agnostic
    
    Design Invariants (Paper Section 5):
    1. Storage model-agnostic (text + routing vectors only)
    2. Injection model-consistent (K/V from target model)
    3. Session cache disposable (inference-time only)
    4. Graceful degradation (α → 0 recovers vanilla)
    5. Audit logging for compliance
    
    Memory Hierarchy (Paper Section 7.4, OPTIONAL enhancement):
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
        
        # Hybrid Injector (Paper Section 3.9)
        self._hybrid_injector: Optional[HybridDKIInjector] = None
        self._use_hybrid_injection = getattr(
            getattr(self.config.dki, 'hybrid_injection', None), 
            'enabled', 
            True
        )
        
        # User preferences cache (user_id -> UserPreference)
        self._user_preferences: Dict[str, UserPreference] = {}
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        # ============ Recall v4 组件 (惰性初始化) ============
        self._recall_config: Optional[Any] = None
        self._multi_signal_recall: Optional[Any] = None
        self._suffix_builder: Optional[Any] = None
        self._fact_retriever: Optional[Any] = None
        self._recall_formatter: Optional[Any] = None
        self._use_recall_v4 = False
        
        if RECALL_V4_AVAILABLE:
            recall_dict = {}
            if hasattr(self.config, 'dki') and hasattr(self.config.dki, 'recall'):
                recall_obj = self.config.dki.recall
                if isinstance(recall_obj, dict):
                    recall_dict = recall_obj
                elif hasattr(recall_obj, '__dict__'):
                    recall_dict = self._obj_to_dict(recall_obj)
            
            if recall_dict:
                self._recall_config = RecallConfig.from_dict(recall_dict)
                self._use_recall_v4 = (
                    self._recall_config.enabled
                    and self._recall_config.strategy == "summary_with_fact_call"
                )
        
        logger.info(
            f"DKI System initialized "
            f"(tiered_cache={self._use_tiered_cache}, "
            f"hybrid_injection={self._use_hybrid_injection}, "
            f"recall_v4={self._use_recall_v4})"
        )
    
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
    
    @property
    def hybrid_injector(self) -> HybridDKIInjector:
        """Get or create hybrid injector."""
        if self._hybrid_injector is None:
            # Get config from dki.hybrid_injection if available
            hybrid_config_obj = getattr(self.config.dki, 'hybrid_injection', None)
            
            # 安全地获取配置值
            def safe_get(obj, attr, default):
                if obj is None:
                    return default
                if hasattr(obj, attr):
                    return getattr(obj, attr)
                if isinstance(obj, dict):
                    return obj.get(attr, default)
                return default
            
            # 获取偏好和历史配置
            pref_config = safe_get(hybrid_config_obj, 'preference', None)
            hist_config = safe_get(hybrid_config_obj, 'history', None)
            
            config = HybridInjectionConfig(
                preference_enabled=safe_get(pref_config, 'enabled', True),
                preference_alpha=safe_get(pref_config, 'alpha', 0.4),
                preference_max_tokens=safe_get(pref_config, 'max_tokens', 100),
                preference_position_strategy=safe_get(pref_config, 'position_strategy', 'negative'),
                history_enabled=safe_get(hist_config, 'enabled', True),
                history_max_tokens=safe_get(hist_config, 'max_tokens', 500),
                history_max_messages=safe_get(hist_config, 'max_messages', 10),
                history_method=safe_get(hist_config, 'method', 'suffix_prompt'),
            )
            
            # 获取语言设置，默认使用中文
            language = safe_get(hybrid_config_obj, 'language', 'cn')
            
            logger.info(f"Creating HybridDKIInjector with language={language}")
            
            # 获取 tokenizer: 优先从 model_adapter 获取，确保 token 估算准确
            # 如果 tokenizer=None，中文文本的 token 估算会严重偏低（因为中文不按空格分词）
            tokenizer = None
            if self._model_adapter is not None and hasattr(self._model_adapter, 'tokenizer'):
                tokenizer = self._model_adapter.tokenizer
            
            self._hybrid_injector = HybridDKIInjector(
                config=config,
                model=self._model_adapter,
                tokenizer=tokenizer,
                language=language,
            )
        return self._hybrid_injector
    
    @staticmethod
    def _obj_to_dict(obj: Any) -> Dict[str, Any]:
        """将配置对象递归转为字典"""
        if isinstance(obj, dict):
            return obj
        result = {}
        for key in dir(obj):
            if key.startswith('_'):
                continue
            val = getattr(obj, key)
            if callable(val):
                continue
            if hasattr(val, '__dict__') and not isinstance(val, (str, int, float, bool, list)):
                result[key] = DKISystem._obj_to_dict(val)
            else:
                result[key] = val
        return result
    
    def _init_recall_v4_components(self) -> None:
        """惰性初始化 Recall v4 组件 (需要 model_adapter 和 db_manager)"""
        if not self._use_recall_v4 or not RECALL_V4_AVAILABLE:
            return
        if self._multi_signal_recall is not None:
            return  # 已初始化
        
        cfg = self._recall_config
        
        # 获取模型名 (用于自动选择格式化器)
        model_name = ""
        if self._model_adapter:
            model_name = getattr(self._model_adapter, "model_name", "")
        elif hasattr(self.config, 'model'):
            engine = self.config.model.default_engine
            engine_cfg = self.config.model.engines.get(engine, {})
            model_name = getattr(engine_cfg, 'model_name', '') if hasattr(engine_cfg, 'model_name') else ''
        
        # 格式化器
        language = "cn"
        if hasattr(self.config.dki, 'hybrid_injection'):
            language = getattr(self.config.dki.hybrid_injection, 'language', 'cn')
        
        self._recall_formatter = create_formatter(
            model_name=model_name,
            formatter_type=cfg.prompt_formatter,
            language=language,
        )
        
        # ConversationRepository 的简单包装器 (recall 组件使用)
        conv_repo_wrapper = _ConversationRepoWrapper(self.db_manager)
        
        # 多信号召回器
        self._multi_signal_recall = MultiSignalRecall(
            config=cfg,
            reference_resolver=None,  # 将在 build_plan 中通过 planner 使用
            memory_router=self.memory_router,
            conversation_repo=conv_repo_wrapper,
        )
        
        # 后缀组装器
        token_counter = None
        if self._model_adapter and hasattr(self._model_adapter, 'tokenizer'):
            tokenizer = self._model_adapter.tokenizer
            token_counter = lambda text: len(tokenizer.encode(text)) if text else 0
        
        self._suffix_builder = SuffixBuilder(
            config=cfg,
            prompt_formatter=self._recall_formatter,
            token_counter=token_counter,
            model_adapter=self._model_adapter,
        )
        
        # 事实检索器
        self._fact_retriever = FactRetriever(
            config=cfg,
            conversation_repo=conv_repo_wrapper,
        )
        
        logger.info(
            f"Recall v4 components initialized: "
            f"formatter={type(self._recall_formatter).__name__}, "
            f"strategy={cfg.strategy}"
        )
    
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
                    metadata=mem.get_metadata(),
                )
                count += 1
        
        logger.info(f"Loaded {count} memories for session {session_id}")
        return count
    
    def set_user_preference(
        self,
        user_id: str,
        preference_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set user preference for hybrid injection.
        
        Preferences are:
        - Short (typically < 100 tokens)
        - Stable (rarely change)
        - Injected via K/V at negative positions
        - Cached for reuse across sessions
        
        Args:
            user_id: User identifier
            preference_text: Preference content (e.g., "素食主义者，住北京，不喜欢辣")
            metadata: Optional metadata
        """
        self._user_preferences[user_id] = UserPreference(
            content=preference_text,
            user_id=user_id,
            metadata=metadata or {},
        )
        logger.info(f"Set preference for user {user_id}: {len(preference_text)} chars")
    
    def get_user_preference(self, user_id: str) -> Optional[UserPreference]:
        """Get user preference."""
        return self._user_preferences.get(user_id)
    
    def _load_user_preferences_from_db(self, user_id: str) -> None:
        """
        从数据库加载用户偏好并设置到 DKI 系统中。
        
        这是关键的桥接方法:
        - 前端通过 /api/preferences 管理偏好 (存入 SQLite)
        - DKI 系统通过 _user_preferences 内存缓存读取偏好
        - 此方法将数据库中的偏好加载到内存缓存中
        """
        try:
            with self.db_manager.session_scope() as db:
                from sqlalchemy import text
                
                query = text("""
                    SELECT preference_text, preference_type, priority, category
                    FROM user_preferences
                    WHERE user_id = :user_id AND is_active = 1
                    ORDER BY priority DESC
                """)
                
                result = db.execute(query, {"user_id": user_id})
                rows = result.fetchall()
                
                if rows:
                    # 将所有偏好合并为一个文本
                    preference_lines = []
                    for row in rows:
                        pref_text = row[0]
                        pref_type = row[1] or "general"
                        preference_lines.append(f"- [{pref_type}] {pref_text}")
                    
                    combined_text = "\n".join(preference_lines)
                    
                    self.set_user_preference(
                        user_id=user_id,
                        preference_text=combined_text,
                    )
                    
                    logger.info(
                        f"Loaded {len(rows)} preferences from DB for user {user_id}: "
                        f"{len(combined_text)} chars"
                    )
                else:
                    logger.debug(f"No preferences found in DB for user {user_id}")
                    
        except Exception as e:
            logger.warning(f"Failed to load preferences from DB for user {user_id}: {e}")
    
    def get_session_history(self, session_id: str, max_messages: int = 10) -> SessionHistory:
        """
        Get session history for hybrid injection.
        
        History is:
        - Longer (can be 100-500 tokens)
        - Dynamic (changes each turn)
        - Injected as suffix prompt with trust-establishing guidance
        
        Args:
            session_id: Session identifier
            max_messages: Maximum messages to retrieve
            
        Returns:
            SessionHistory object
        """
        history = SessionHistory(session_id=session_id)
        
        with self.db_manager.session_scope() as db:
            conv_repo = ConversationRepository(db)
            messages = conv_repo.get_recent(session_id, limit=max_messages)
            
            for msg in messages:
                history.add_message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at.timestamp() if msg.created_at else None,
                )
        
        return history
    
    def chat(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        allow_injection: bool = True,
        use_hybrid: Optional[bool] = None,
        force_alpha: Optional[float] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        task_type: str = "reasoning",
        **kwargs
    ) -> DKIResponse:
        """
        Generate response using DKI.
        
        Supports two injection modes:
        1. Standard DKI: Memory retrieval + K/V injection
        2. Hybrid DKI: Preferences (K/V) + History (suffix prompt)
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier (for hybrid injection preferences)
            allow_injection: Whether to allow memory injection
            use_hybrid: Use hybrid injection (None = auto from config)
            force_alpha: Force specific alpha value (bypasses gating)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_type: Type of task for budget analysis
            
        Returns:
            DKIResponse with generated text, metadata, and budget analysis
        """
        # Determine injection mode
        use_hybrid_mode = use_hybrid if use_hybrid is not None else self._use_hybrid_injection
        
        # 保存原始用户输入 (用于数据库记录，避免历史递归嵌套)
        original_query = query
        
        # Start latency timer
        with LatencyTimer() as timer:
            cache_hit = False
            cache_tier = "none"
            
            # Get session cache
            session_cache = self._get_session_cache(session_id)
            
            # Step 0: Load user preferences from database (if not already cached)
            if use_hybrid_mode and allow_injection and user_id:
                if not self.get_user_preference(user_id):
                    self._load_user_preferences_from_db(user_id)
            
            # Step 0.5: History preparation (Recall v4 or Hybrid)
            hybrid_result = None
            recall_v4_suffix = None
            recall_v4_trace_ids = []
            
            if self._use_recall_v4 and use_hybrid_mode and allow_injection:
                # ============ Recall v4: 多信号召回 + 后缀组装 ============
                timer.start_stage("recall_v4")
                self._init_recall_v4_components()
                
                # Get user preference (for K/V injection, 不变)
                preference = None
                if user_id:
                    preference = self.get_user_preference(user_id)
                
                # 多信号召回
                recall_result = self._multi_signal_recall.recall(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                )
                
                # 后缀组装
                context_window = 4096
                if hasattr(self.config.model, 'engines'):
                    engine_cfg = self.config.model.engines.get(
                        self.config.model.default_engine, {}
                    )
                    context_window = getattr(engine_cfg, 'max_model_len', 4096) if hasattr(engine_cfg, 'max_model_len') else 4096
                
                pref_tokens = 0
                if preference:
                    pref_tokens = len(preference.content.split()) * 2  # 粗估
                
                assembled = self._suffix_builder.build(
                    query=query,
                    recalled_messages=recall_result.messages,
                    context_window=context_window,
                    preference_tokens=pref_tokens,
                )
                
                recall_v4_suffix = assembled.text
                recall_v4_trace_ids = assembled.trace_ids
                
                # 使用组装好的后缀替代 query
                if assembled.total_tokens > 0:
                    query = assembled.text
                
                # 仍然准备 hybrid_result 用于偏好 K/V 注入 (history 已由 recall_v4 处理)
                if preference:
                    hybrid_result = self.hybrid_injector.prepare_input(
                        user_query=original_query,
                        preference=preference,
                        history=None,
                    )
                
                timer.end_stage()
                
            elif use_hybrid_mode and allow_injection:
                # ============ 原有 Hybrid 注入 (flat_history) ============
                timer.start_stage("hybrid_prep")
                
                # Get user preference (if user_id provided)
                preference = None
                if user_id:
                    preference = self.get_user_preference(user_id)
                
                # Get session history
                history = self.get_session_history(session_id)
                
                # Prepare hybrid input
                hybrid_result = self.hybrid_injector.prepare_input(
                    user_query=query,
                    preference=preference,
                    history=history if history.messages else None,
                )
                
                # Use modified query with history suffix
                if hybrid_result.history_tokens > 0:
                    query = hybrid_result.input_text
                
                timer.end_stage()
            
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
                # But still use hybrid history suffix if available
                timer.start_stage("prefill")
                output = self.model.generate(
                    prompt=query,  # May include history suffix from hybrid
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )
                timer.end_stage()
                memories_used = []
            else:
                # DKI injection flow
                # Include preference K/V from hybrid if available
                preference_kv = None
                preference_alpha = 0.0
                if hybrid_result and hybrid_result.preference_kv is not None:
                    preference_kv = hybrid_result.preference_kv
                    preference_alpha = hybrid_result.preference_alpha
                
                output, cache_hit, cache_tier = self._generate_with_injection(
                    query=query,
                    gating_decision=gating_decision,
                    session_cache=session_cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    timer=timer,
                    preference_kv=preference_kv,
                    preference_alpha=preference_alpha,
                    **kwargs
                )
                memories_used = gating_decision.memories
            
            # ============ Recall v4: Fact Call 循环 ============
            fact_rounds_used = 0
            if (self._use_recall_v4
                    and recall_v4_trace_ids
                    and self._fact_retriever
                    and self._recall_formatter
                    and self._recall_config
                    and self._recall_config.fact_call.enabled):
                timer.start_stage("fact_call")
                output, fact_rounds_used = self._execute_fact_call_loop(
                    output=output,
                    prompt=query,
                    session_id=session_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    preference_kv=hybrid_result.preference_kv if hybrid_result else None,
                    preference_alpha=hybrid_result.preference_alpha if hybrid_result else 0.0,
                    **kwargs,
                )
                timer.end_stage()
            
            timer.set_cache_info(cache_hit, cache_tier)
        
        # Record latency for analysis
        self._budget_analyzer.record_latency(timer.breakdown)
        
        # Add hybrid injection metadata with display info
        if hybrid_result:
            metadata = {
                'model': self.model.model_name,
                'session_cache_stats': session_cache.get_stats(),
                'task_type': task_type,
                'hybrid_injection': {
                    'enabled': True,
                    'preference_tokens': hybrid_result.preference_tokens,
                    'history_tokens': hybrid_result.history_tokens,
                    'preference_alpha': hybrid_result.preference_alpha,
                    # 用于显示的明文信息
                    'preference_text': hybrid_result.preference_text,
                    'history_suffix_text': hybrid_result.history_suffix_text,
                    'history_messages': hybrid_result.history_messages,
                    'final_input': hybrid_result.input_text,
                },
            }
        else:
            metadata = {
                'model': self.model.model_name,
                'session_cache_stats': session_cache.get_stats(),
                'task_type': task_type,
                'hybrid_injection': {'enabled': False},
            }
        
        # Recall v4 metadata
        if self._use_recall_v4 and recall_v4_suffix:
            metadata['recall_v4'] = {
                'enabled': True,
                'strategy': self._recall_config.strategy if self._recall_config else '',
                'trace_ids': recall_v4_trace_ids,
                'fact_rounds_used': fact_rounds_used,
            }
        
        # Log to database (使用原始用户输入，避免历史后缀递归嵌套)
        self._log_conversation(
            session_id=session_id,
            query=original_query,
            response=output.text,
            gating_decision=gating_decision,
            latency_ms=timer.breakdown.total_ms,
            user_id=user_id,
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
            metadata=metadata,
        )
    
    def _generate_with_injection(
        self,
        query: str,
        gating_decision: GatingDecision,
        session_cache: Union[SessionKVCache, TieredKVCache],
        max_new_tokens: int,
        temperature: float,
        timer: Optional[LatencyTimer] = None,
        preference_kv: Optional[Any] = None,
        preference_alpha: float = 0.0,
        **kwargs
    ) -> tuple:
        """
        Generate response with K/V injection.
        
        Supports:
        - Simple SessionKVCache and TieredKVCache
        - Hybrid injection with preference K/V
        
        Args:
            query: User query (may include history suffix)
            gating_decision: Gating decision with memories
            session_cache: K/V cache
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            timer: Latency timer
            preference_kv: Pre-computed preference K/V (from hybrid injection)
            preference_alpha: Alpha for preference injection
        
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
        
        # Add preference K/V if available (hybrid injection)
        # Preference K/V is prepended (negative positions conceptually)
        if preference_kv is not None and preference_alpha > 0:
            if timer:
                timer.start_stage("preference_merge")
            
            # Scale preference K/V with its own alpha
            if isinstance(preference_kv, list) and len(preference_kv) > 0:
                scaled_pref_kv = []
                for entry in preference_kv:
                    key, value = self.mis.scale_kv_values(
                        entry.key, entry.value, preference_alpha
                    )
                    scaled_pref_kv.append(KVCacheEntry(
                        key=key, value=value, layer_idx=entry.layer_idx
                    ))
                
                # Prepend preference K/V to merged K/V
                if merged_kv:
                    merged_kv = self._merge_kv_entries([scaled_pref_kv, merged_kv])
                else:
                    merged_kv = scaled_pref_kv
            
            if timer:
                timer.end_stage()
        
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
    
    def _execute_fact_call_loop(
        self,
        output: ModelOutput,
        prompt: str,
        session_id: str,
        max_new_tokens: int,
        temperature: float,
        preference_kv: Optional[Any] = None,
        preference_alpha: float = 0.0,
        **kwargs,
    ) -> tuple:
        """
        Recall v4: Fact Call 循环
        
        检测模型输出中的 retrieve_fact 调用, 补充事实后重新推理。
        
        Returns:
            (final_output, fact_rounds_used)
        """
        max_rounds = self._recall_config.fact_call.max_rounds
        max_fact_tokens = self._recall_config.fact_call.max_fact_tokens
        total_fact_tokens = 0
        
        for round_idx in range(max_rounds):
            # 检查输出是否包含 fact_call
            fact_request = self._recall_formatter.detect_fact_request(output.text)
            
            if fact_request is None:
                return output, round_idx  # 不需要补充事实 ✅
            
            logger.info(
                f"DKI System fact call round {round_idx + 1}: "
                f"trace_id={fact_request.trace_id}"
            )
            
            # 检索事实
            fact_response = self._fact_retriever.retrieve(
                trace_id=fact_request.trace_id,
                session_id=session_id,
                offset=fact_request.offset,
                limit=fact_request.limit,
            )
            
            if not fact_response.messages:
                return output, round_idx
            
            # 格式化事实段落
            fact_text = self._recall_formatter.format_fact_segment(fact_response)
            
            # 粗估 token
            fact_tokens = len(fact_text) // 2
            total_fact_tokens += fact_tokens
            
            if total_fact_tokens > max_fact_tokens:
                logger.info(f"Fact token budget exhausted: {total_fact_tokens}")
                return output, round_idx
            
            # 追加事实到 prompt
            continuation = "请基于以上补充事实回答用户问题。"
            prompt = prompt + "\n\n" + fact_text + "\n\n" + continuation
            
            # 重新推理
            if preference_kv and preference_alpha > 0.1:
                output = self.model.forward_with_kv_injection(
                    prompt=prompt,
                    injected_kv=preference_kv,
                    alpha=preference_alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                output = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
        
        return output, max_rounds
    
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
        user_id: Optional[str] = None,
    ) -> None:
        """Log conversation to database."""
        with self.db_manager.session_scope() as db:
            # Ensure session exists before inserting conversation
            session_repo = SessionRepository(db)
            session_repo.get_or_create(session_id=session_id, user_id=user_id)
            
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


# ============================================================
# Recall v4 辅助类
# ============================================================

class _ConversationRepoWrapper:
    """
    ConversationRepository 的简单包装器
    
    为 recall 组件提供统一接口, 自动管理数据库 session。
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self._db_manager = db_manager
    
    def get_by_session(
        self,
        session_id: str,
        db_session: Optional[Any] = None,
        **kwargs,
    ) -> List[Any]:
        """获取会话的所有消息"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            return repo.get_by_session(session_id)
    
    def get_recent(
        self,
        session_id: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Any]:
        """获取最近的消息"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            return repo.get_recent(session_id, limit=limit)
    
    def get_by_id(self, message_id: str) -> Optional[Any]:
        """根据 ID 获取消息"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            if hasattr(repo, 'get_by_id'):
                return repo.get_by_id(message_id)
        return None