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

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger

from dki.core.text_utils import (
    strip_think_content, estimate_tokens_fast,
    detect_vague_reference, build_clarification_instruction,
)
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

# 用户隔离组件 (v3.1)
try:
    from dki.cache.user_isolation import (
        UserIsolationContext,
        UserIsolationConfig,
        CacheKeySigner,
        InferenceContextGuard,
        CacheAuditLog,
    )
    USER_ISOLATION_AVAILABLE = True
except ImportError:
    USER_ISOLATION_AVAILABLE = False


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
        
        # ============ 用户隔离 (v3.1) ============
        self._user_isolation_enabled = USER_ISOLATION_AVAILABLE
        self._cache_key_signer: Optional[Any] = None
        self._cache_audit_log: Optional[Any] = None
        
        if self._user_isolation_enabled:
            import os
            hmac_secret = os.environ.get("DKI_HMAC_SECRET", "")
            self._cache_key_signer = CacheKeySigner(secret=hmac_secret)
            self._cache_audit_log = CacheAuditLog(max_entries=10000)
            logger.info("DKI user isolation enabled (HMAC signer + audit log)")
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        # ============ Function Call Logger (v3.2) ============
        self._fc_logger: Optional[Any] = None
        try:
            from dki.core.function_call_logger import FunctionCallLogger
            self._fc_logger = FunctionCallLogger(
                db_manager=self.db_manager,
                text_log_dir="logs/function_calls",
            )
        except Exception as fc_err:
            logger.warning(f"FunctionCallLogger init failed (non-critical): {fc_err}")
        
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
                elif hasattr(recall_obj, 'model_dump'):
                    # Pydantic v2 model (RecallConfigModel)
                    recall_dict = recall_obj.model_dump()
                elif hasattr(recall_obj, 'dict'):
                    # Pydantic v1 model
                    recall_dict = recall_obj.dict()
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
            
            # Ensure model adapter is loaded before creating injector
            # This is needed for preference K/V computation
            model_adapter = self._model_adapter
            if model_adapter is None:
                try:
                    model_adapter = self.model  # triggers lazy load
                except Exception as e:
                    logger.warning(f"Failed to load model adapter for hybrid injector: {e}")
                    model_adapter = None
            
            # Get tokenizer: prefer from model_adapter for accurate token estimation
            # If tokenizer=None, Chinese text token estimation will be severely underestimated
            tokenizer = None
            if model_adapter is not None and hasattr(model_adapter, 'tokenizer'):
                tokenizer = model_adapter.tokenizer
            
            self._hybrid_injector = HybridDKIInjector(
                config=config,
                model=model_adapter,
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
        
        安全 (v3.1): user_id 必须经过验证
        
        Args:
            user_id: User identifier (must be validated)
            preference_text: Preference content (e.g., "素食主义者，住北京，不喜欢辣")
            metadata: Optional metadata
        """
        if not user_id or not user_id.strip():
            logger.warning("Attempted to set preference with empty user_id, ignoring")
            return
        
        user_id = user_id.strip()
        self._user_preferences[user_id] = UserPreference(
            content=preference_text,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        # 审计日志
        if self._cache_audit_log:
            self._cache_audit_log.record(
                user_id=user_id,
                action="put",
                cache_key=f"preference:{user_id}",
                cache_tier="memory",
                success=True,
                metadata={"text_length": len(preference_text)},
            )
        
        logger.info(f"Set preference for user {user_id}: {len(preference_text)} chars")
    
    def get_user_preference(self, user_id: str) -> Optional[UserPreference]:
        """Get user preference."""
        return self._user_preferences.get(user_id)
    
    def clear_preference_cache(self, user_id: Optional[str] = None) -> None:
        """
        清除用户偏好的内存缓存，强制下次从数据库重新加载。
        
        这是公开 API，供实验系统等外部组件安全地清除缓存，
        避免直接操作 _user_preferences 内部状态。
        
        Args:
            user_id: 指定用户 ID 则只清除该用户; None 则清除所有用户
        """
        if user_id:
            removed = self._user_preferences.pop(user_id, None)
            if removed:
                logger.debug(f"Cleared preference cache for user {user_id}")
                if self._cache_audit_log:
                    self._cache_audit_log.record(
                        user_id=user_id,
                        action="invalidate",
                        cache_key=f"preference:{user_id}",
                        cache_tier="memory",
                        success=True,
                    )
        else:
            count = len(self._user_preferences)
            self._user_preferences.clear()
            logger.debug(f"Cleared all preference caches ({count} users)")
    
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
    
    def get_session_history(
        self,
        session_id: str,
        max_messages: int = 10,
        user_id: Optional[str] = None,
    ) -> SessionHistory:
        """
        Get session history for hybrid injection, with cross-session support.
        
        v6.1: 支持跨会话记忆
        - 首先获取当前会话的历史消息
        - 如果提供了 user_id, 还会补充该用户其他会话的历史消息
        - 跨会话消息放在当前会话消息之前 (按时间顺序)
        
        History is:
        - Longer (can be 100-500 tokens)
        - Dynamic (changes each turn)
        - Injected as suffix prompt with trust-establishing guidance
        
        Args:
            session_id: Session identifier
            max_messages: Maximum messages to retrieve
            user_id: User identifier (optional, for cross-session retrieval)
            
        Returns:
            SessionHistory object
        """
        history = SessionHistory(session_id=session_id)
        
        with self.db_manager.session_scope() as db:
            conv_repo = ConversationRepository(db)
            
            # v6.1: 先添加跨会话历史 (旧的先加, 时间顺序排列)
            if user_id:
                try:
                    cross_session_limit = max_messages // 2  # 跨会话消息占一半配额
                    cross_msgs = conv_repo.get_recent_by_user_cross_session(
                        user_id=user_id,
                        current_session_id=session_id,
                        limit=cross_session_limit,
                    )
                    for msg in cross_msgs:
                        history.add_message(
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.created_at.timestamp() if msg.created_at else None,
                        )
                    if cross_msgs:
                        logger.info(
                            f"Cross-session history: added {len(cross_msgs)} messages "
                            f"from previous sessions for user {user_id}"
                        )
                except Exception as e:
                    logger.warning(f"Cross-session history retrieval failed (non-critical): {e}")
            
            # 当前会话的历史消息
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
        # ============ 用户身份验证 (v3.1) ============
        if user_id:
            user_id = user_id.strip()
            if not user_id:
                logger.warning("Empty user_id after strip, proceeding without user context")
                user_id = None
        
        # Determine injection mode
        use_hybrid_mode = use_hybrid if use_hybrid is not None else self._use_hybrid_injection
        
        # 保存原始用户输入 (用于数据库记录，避免历史递归嵌套)
        original_query = query
        
        # ============ v6.5: 模糊指代检测 ============
        # 检测用户输入中的模糊指代 (如 "前段时间说的那件事你怎么想")
        # 如果检测到且历史召回无法定位, 将在 system message 中注入澄清指令
        _vague_ref = detect_vague_reference(original_query)
        if _vague_ref.is_vague:
            logger.info(
                f"[Vague Reference Detected] "
                f"confidence={_vague_ref.confidence:.2f}, "
                f"pattern='{_vague_ref.matched_pattern}', "
                f"lang={_vague_ref.language}"
            )
        
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
            
            # Step 0.5: History + Preference preparation
            hybrid_result = None
            recall_v4_suffix = None
            recall_v4_trace_ids = []
            
            _recall_v4_failed = False
            
            # ============ v5.2: 统一偏好加载 (先于历史准备) ============
            # 偏好在 vLLM v5.0 中通过 prompt (system message) 注入, 不依赖 compute_kv。
            # 必须在所有路径中确保偏好进入 prompt, 即使没有历史消息。
            preference = None
            if use_hybrid_mode and allow_injection and user_id:
                preference = self.get_user_preference(user_id)
                if preference:
                    logger.debug(
                        f"Preference loaded for user {user_id}: "
                        f"{len(preference.content)} chars"
                    )
            
            # v5.5: 跟踪 hybrid fallback 路径中实际注入的历史 token 数
            _hybrid_fallback_hist_tokens = 0
            _hybrid_fallback_hist_messages = []
            
            if self._use_recall_v4 and use_hybrid_mode and allow_injection:
                # ============ Recall v4: 多信号召回 + 后缀组装 ============
                try:
                    timer.start_stage("recall_v4")
                    self._init_recall_v4_components()
                    
                    # v5.3: 调试日志 — 确认 recall 前数据库中的消息数
                    try:
                        _dbg_msgs = self._multi_signal_recall._conversation_repo.get_by_session(
                            session_id=session_id,
                        )
                        logger.info(
                            f"[Recall v4 pre-check] session_id={session_id}, "
                            f"user_id={user_id}, "
                            f"messages_in_db={len(_dbg_msgs)}, "
                            f"roles={[getattr(m, 'role', '?') for m in _dbg_msgs[-6:]] if _dbg_msgs else []}"
                        )
                    except Exception as _dbg_err:
                        logger.warning(f"[Recall v4 pre-check] failed: {_dbg_err}")
                    
                    # 多信号召回
                    recall_result = self._multi_signal_recall.recall(
                        query=query,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    
                    logger.debug(
                        f"Recall v4 result: {len(recall_result.messages)} messages, "
                        f"keyword={recall_result.keyword_hits}, "
                        f"bm25={recall_result.bm25_hits}, "
                        f"vector={recall_result.vector_hits}, "
                        f"recent={recall_result.recent_turns_added}"
                    )
                    
                    # 后缀组装 (使用动态上下文窗口)
                    context_window = self._get_context_window()
                    
                    pref_tokens = 0
                    if preference:
                        # 中文: 约 1.5 token/字; 英文: 约 1.3 token/word
                        import re as _re
                        _cn = len(_re.findall(r'[\u4e00-\u9fff]', preference.content))
                        _en = len(_re.findall(r'[a-zA-Z]+', preference.content))
                        pref_tokens = int(_cn * 1.5 + _en * 1.3) or 1
                    
                    assembled = self._suffix_builder.build(
                        query=query,
                        recalled_messages=recall_result.messages,
                        context_window=context_window,
                        preference_tokens=pref_tokens,
                    )
                    
                    # v5.5: 仅当实际有历史消息时才设置 recall_v4_suffix
                    # SuffixBuilder.build() 在 recalled_messages 为空时返回 text=query, total_tokens=0
                    # 此时 recall_v4_suffix 不应被设置 (它不包含历史)
                    if assembled.total_tokens > 0 and assembled.items:
                        recall_v4_suffix = assembled.text
                    else:
                        recall_v4_suffix = None  # 无历史, 不设置
                    recall_v4_trace_ids = assembled.trace_ids
                    
                    # ============ v5.2: 始终通过 chat template 构造 prompt ============
                    # 修复: 即使 assembled.total_tokens == 0 (无历史), 
                    # 只要有偏好, 也应通过 chat template 将偏好注入 system message。
                    # 在 vLLM v5.0 中, 偏好通过 prompt 前缀注入 (不依赖 compute_kv)。
                    has_history = assembled.total_tokens > 0 and assembled.items
                    has_preference = preference is not None and preference.content
                    
                    if has_history or has_preference:
                        query = self._build_recall_v4_chat_prompt(
                            items=assembled.items if has_history else [],
                            original_query=original_query,
                            preference=preference,
                            trace_ids=assembled.trace_ids if has_history else [],
                            has_fact_call_instruction=assembled.has_fact_call_instruction if has_history else False,
                            vague_reference=_vague_ref,
                        )
                        logger.debug(
                            f"Chat prompt built: history_items={len(assembled.items) if has_history else 0}, "
                            f"preference={'yes' if has_preference else 'no'}, "
                            f"history_tokens={assembled.total_tokens}"
                        )
                    
                    # 准备 hybrid_result 用于元数据显示 (偏好文本、token 统计等)
                    # 注意: 在 vLLM v5.0 中, preference_kv 为空列表 (不需要)
                    # 偏好已通过 chat template system message 注入 prompt
                    if preference:
                        hybrid_result = self.hybrid_injector.prepare_input(
                            user_query=original_query,
                            preference=preference,
                            history=None,
                        )
                    
                    # v5.5: 关键日志 — 确认 recall_v4 成功完成
                    logger.info(
                        f"[Recall v4 SUCCESS] "
                        f"recalled={len(recall_result.messages)}, "
                        f"assembled_tokens={assembled.total_tokens}, "
                        f"items={len(assembled.items)}, "
                        f"suffix_len={len(recall_v4_suffix) if recall_v4_suffix else 0}, "
                        f"has_history={has_history}, has_preference={has_preference}"
                    )
                    
                    timer.end_stage()
                
                except Exception as recall_error:
                    logger.error(
                        f"Recall v4 failed, falling back to stable (hybrid): {recall_error}",
                        exc_info=True,
                    )
                    _recall_v4_failed = True
                    # 重置 query 到原始输入
                    query = original_query
                    recall_v4_suffix = None
                    recall_v4_trace_ids = []
                    try:
                        timer.end_stage()
                    except Exception:
                        pass
            
            if (_recall_v4_failed and use_hybrid_mode and allow_injection) or \
               (not self._use_recall_v4 and use_hybrid_mode and allow_injection):
                # ============ Hybrid 注入 (回退路径或非 recall_v4 模式) ============
                timer.start_stage("hybrid_prep")
                
                # 偏好已在上方统一加载 (preference 变量)
                
                # Get session history (无论 history.enabled 配置, 仍尝试获取历史)
                # 修复: 之前 history.enabled=false 导致 HybridDKIInjector 跳过历史
                # 但这里应直接获取历史, 通过 chat template 注入
                history = self.get_session_history(session_id, user_id=user_id)
                
                # Prepare hybrid input (用于元数据: 偏好文本、token 统计)
                # 注意: hybrid_result.history_tokens 可能为 0 (因 config.history_enabled=false)
                # 但我们仍然通过 chat template 注入历史, 所以需要单独跟踪实际注入的历史 token 数
                hybrid_result = self.hybrid_injector.prepare_input(
                    user_query=query,
                    preference=preference,
                    history=history if history.messages else None,
                )
                
                # v5.2: 始终通过 chat template 构造 prompt
                # 条件: 有历史消息 或 有偏好 → 都需要 chat template 格式化
                has_history = history and history.messages and len(history.messages) > 0
                has_preference = preference is not None and preference.content
                
                if has_history or has_preference:
                    query = self._build_hybrid_chat_prompt(
                        history=history if has_history else None,
                        original_query=query,
                        preference=preference,
                        vague_reference=_vague_ref,
                    )
                
                # v5.5: 跟踪 hybrid fallback 实际注入的历史 token 数
                # (hybrid_result.history_tokens 可能因 config.history_enabled=false 而为 0)
                if has_history:
                    _hist_text = ""
                    for msg in history.messages:
                        _hist_text += getattr(msg, 'content', '') + " "
                    tokenizer = getattr(self.model, 'tokenizer', None) if self._model_adapter else None
                    if tokenizer:
                        try:
                            _hybrid_fallback_hist_tokens = len(tokenizer.encode(_hist_text))
                        except Exception:
                            _hybrid_fallback_hist_tokens = len(_hist_text) // 2
                    else:
                        _hybrid_fallback_hist_tokens = len(_hist_text) // 2
                    
                    _hybrid_fallback_hist_messages = [
                        {"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', '')}
                        for m in history.messages
                    ]
                    
                    logger.info(
                        f"[Hybrid fallback] history_msgs={len(history.messages)}, "
                        f"history_tokens={_hybrid_fallback_hist_tokens}, "
                        f"preference={'yes' if has_preference else 'no'}"
                    )
                
                timer.end_stage()
            
            # ============ v5.2: 最后防线 — 纯偏好注入 (无历史, 非 hybrid 模式) ============
            # 如果以上路径都未走 (例如 allow_injection=True 但 use_hybrid=False),
            # 但有偏好, 仍应通过 chat template 注入偏好
            if preference and preference.content and query == original_query:
                # query 未被任何路径改写, 说明偏好还没注入
                query = self._build_recall_v4_chat_prompt(
                    items=[],
                    original_query=original_query,
                    preference=preference,
                    trace_ids=[],
                    has_fact_call_instruction=False,
                    vague_reference=_vague_ref,
                )
                logger.debug("Fallback: preference injected via chat template (no history path)")
            
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
                    user_id=user_id,
                    **kwargs,
                )
                timer.end_stage()
            
            timer.set_cache_info(cache_hit, cache_tier)
        
        # Record latency for analysis
        self._budget_analyzer.record_latency(timer.breakdown)
        
        # ============ v5.2: 构建元数据 (修正 alpha/历史 token 来源) ============
        # 偏好 alpha 来自配置 (hybrid_injection.preference.alpha), 不来自门控决策
        # 历史 tokens 来自 recall_v4 assembled 或 hybrid 历史
        _pref_alpha_config = 0.0
        _pref_tokens_display = 0
        _hist_tokens_display = 0
        _pref_text_display = ""
        _hist_suffix_display = ""
        _hist_messages_display = []
        _final_input_display = query
        
        if preference and preference.content:
            _pref_alpha_config = getattr(
                getattr(getattr(self.config.dki, 'hybrid_injection', None), 'preference', None),
                'alpha', 0.4
            )
            _pref_text_display = preference.content
            # 精确计算偏好 token 数
            tokenizer = getattr(self.model, 'tokenizer', None) if self._model_adapter else None
            if tokenizer:
                try:
                    _pref_tokens_display = len(tokenizer.encode(preference.content))
                except Exception:
                    _pref_tokens_display = len(preference.content)
            else:
                import re as _re_est
                _cn = len(_re_est.findall(r'[\u4e00-\u9fff]', preference.content))
                _en = len(_re_est.findall(r'[a-zA-Z]+', preference.content))
                _pref_tokens_display = int(_cn * 1.5 + _en * 1.3) or len(preference.content)
        
        if hybrid_result:
            _hist_tokens_display = hybrid_result.history_tokens
            _hist_suffix_display = hybrid_result.history_suffix_text
            _hist_messages_display = hybrid_result.history_messages
            _final_input_display = hybrid_result.input_text
        
        # recall_v4 历史 tokens 覆盖 hybrid 的 (recall_v4 更准确)
        # v5.5: 同时提取 recall_v4 的历史消息用于可视化显示
        _recall_v4_signal_stats = {}
        if recall_v4_suffix and self._use_recall_v4:
            # recall_v4 成功: 使用 assembled 的 token 数
            # 注意: 仅当 assembled.total_tokens > 0 时才算真正有历史
            # (当 recalled_messages 为空时, assembled.text = query, total_tokens = 0)
            if 'assembled' in locals() and hasattr(assembled, 'total_tokens') and assembled.total_tokens > 0:
                _hist_tokens_display = assembled.total_tokens
                _hist_suffix_display = recall_v4_suffix
                logger.debug(f"Recall v4 history tokens (from assembled): {_hist_tokens_display}")
            else:
                # assembled.total_tokens == 0 意味着没有召回到历史消息
                # recall_v4_suffix 只是 query 文本, 不应计为历史 tokens
                logger.debug(
                    f"Recall v4 suffix exists but assembled.total_tokens=0 "
                    f"(no history recalled, suffix is just query)"
                )
        
        # v5.5: 从 recall_v4 assembled items 填充 _hist_messages_display
        if self._use_recall_v4 and recall_v4_suffix and not _hist_messages_display:
            try:
                if 'assembled' in locals() and hasattr(assembled, 'items') and assembled.items:
                    for item in assembled.items:
                        _hist_messages_display.append({
                            'role': item.role or 'user',
                            'content': item.content or '',
                            'type': item.type,
                            'trace_id': item.trace_id,
                            'confidence': item.confidence,
                        })
                    logger.debug(
                        f"Recall v4 history messages for viz: {len(_hist_messages_display)} items"
                    )
            except Exception as e:
                logger.debug(f"Failed to extract recall_v4 history items for viz: {e}")
        
        # v5.5: Hybrid fallback 历史 tokens 补充
        # 当 recall_v4 失败 (recall_v4_suffix=None) 但 hybrid fallback 注入了历史时,
        # hybrid_result.history_tokens 可能为 0 (因 config.history_enabled=false),
        # 但实际上 _build_hybrid_chat_prompt 已经将历史注入了 prompt。
        # 使用 _hybrid_fallback_hist_tokens 来修正。
        if _hist_tokens_display == 0 and _hybrid_fallback_hist_tokens > 0:
            _hist_tokens_display = _hybrid_fallback_hist_tokens
            _hist_messages_display = _hybrid_fallback_hist_messages
            logger.debug(
                f"Hybrid fallback history tokens applied: {_hybrid_fallback_hist_tokens}, "
                f"messages: {len(_hybrid_fallback_hist_messages)}"
            )
        
        # v5.4: 提取 recall_v4 多信号统计 (用于可视化)
        if self._use_recall_v4 and 'recall_result' in locals():
            try:
                _recall_v4_signal_stats = {
                    'keyword_hits': getattr(recall_result, 'keyword_hits', 0),
                    'bm25_hits': getattr(recall_result, 'bm25_hits', 0),
                    'vector_hits': getattr(recall_result, 'vector_hits', 0),
                    'recent_turns_added': getattr(recall_result, 'recent_turns_added', 0),
                    'reference_scope': getattr(recall_result, 'reference_scope', None),
                    'total_recalled': len(getattr(recall_result, 'messages', [])),
                }
            except Exception:
                pass
        
        # v5.5: _final_input_display 始终使用最终的 query
        # (query 已被 _build_recall_v4_chat_prompt 或 _build_hybrid_chat_prompt 改写)
        if query != original_query:
            _final_input_display = query  # query 已包含完整 chat template 格式
        
        # v5.4: 从 assembled 获取 summary/message count
        _summary_count = 0
        _message_count = 0
        if self._use_recall_v4 and 'assembled' in locals() and hasattr(assembled, 'summary_count'):
            _summary_count = assembled.summary_count
            _message_count = assembled.message_count
        
        # v5.5: 关键日志 — 元数据摘要
        logger.info(
            f"[DKI metadata] "
            f"pref_tokens={_pref_tokens_display}, "
            f"hist_tokens={_hist_tokens_display}, "
            f"pref_alpha={_pref_alpha_config}, "
            f"hist_messages={len(_hist_messages_display)}, "
            f"recall_v4_suffix={'yes' if recall_v4_suffix else 'no'}, "
            f"hybrid_fallback_hist={_hybrid_fallback_hist_tokens}, "
            f"query_modified={'yes' if query != original_query else 'no'}"
        )
        
        metadata = {
            'model': self.model.model_name,
            'session_cache_stats': session_cache.get_stats(),
            'task_type': task_type,
            'hybrid_injection': {
                'enabled': bool(preference) or _hist_tokens_display > 0,
                'preference_tokens': _pref_tokens_display,
                'history_tokens': _hist_tokens_display,
                'preference_alpha': _pref_alpha_config,
                # 用于显示的明文信息
                'preference_text': _pref_text_display,
                'history_suffix_text': _hist_suffix_display,
                'history_messages': _hist_messages_display,
                'final_input': _final_input_display,
            },
        }
        
        # Recall v4 metadata (v5.4: 包含完整信号统计)
        if self._use_recall_v4:
            metadata['recall_v4'] = {
                'enabled': bool(recall_v4_suffix) or self._use_recall_v4,
                'strategy': self._recall_config.strategy if self._recall_config else '',
                'trace_ids': recall_v4_trace_ids,
                'fact_rounds_used': fact_rounds_used,
                'summary_count': _summary_count,
                'message_count': _message_count,
                **_recall_v4_signal_stats,
            }
        
        # v5.7: 移除 <think> 推理内容 (用于返回和存储)
        clean_response_text, think_stripped = strip_think_content(output.text)
        if think_stripped:
            logger.debug(
                f"DKI: Think content stripped: "
                f"{len(output.text)} -> {len(clean_response_text)} chars"
            )
            metadata['think_content_stripped'] = True
        
        # Log to database (使用原始用户输入，避免历史后缀递归嵌套)
        # _log_conversation 内部也会执行 strip_think_content
        self._log_conversation(
            session_id=session_id,
            query=original_query,
            response=output.text,
            gating_decision=gating_decision,
            latency_ms=timer.breakdown.total_ms,
            user_id=user_id,
        )
        
        return DKIResponse(
            text=clean_response_text,  # v5.7: 返回清理后的响应 (无 think 内容)
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
    
    def _estimate_prompt_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量 (v5.7: 使用快速估算, 略微高估)
        
        不依赖 tokenizer, 使用 estimate_tokens_fast:
        - 中文字符: ~1.5 token/字 × 1.15 (高估 15%)
        - 英文单词: ~1.3 token/word × 1.15
        - 其他字符: ~0.5 token/char
        """
        return estimate_tokens_fast(text, overestimate_factor=1.15)
    
    def _get_context_window(self) -> int:
        """获取模型上下文窗口大小"""
        context_window = 4096
        if hasattr(self.config.model, 'engines'):
            engine_cfg = self.config.model.engines.get(
                self.config.model.default_engine, {}
            )
            context_window = getattr(engine_cfg, 'max_model_len', 4096) if hasattr(engine_cfg, 'max_model_len') else 4096
        return context_window
    
    def _get_max_prompt_tokens(self, max_new_tokens: int = 512) -> int:
        """
        v5.7: 重新设计的 token 预算分配
        
        核心变更:
        - 生成预留 = 30% 上下文 (而非固定 512)
        - 偏好: 100-200 tokens (由配置控制)
        - 标记开销: 100-150 tokens (chat template 标记)
        - 当前输入: 直接估算 (不预留, 以估算值为准)
        - 剩余: 历史消息
        
        预算 = 上下文窗口 - 生成预留(30%) - 标记开销
        """
        context_window = self._get_context_window()
        
        # v5.7: 生成预留 = 30% 上下文
        generation_reserve = int(context_window * 0.30)
        
        # 标记开销 (chat template 标记, 特殊 token 等)
        tag_overhead = self._get_tag_overhead()
        
        return context_window - generation_reserve - tag_overhead
    
    def _get_tag_overhead(self) -> int:
        """获取 chat template 标记开销 (可配置, 默认 120)"""
        # 从 recall.budget.instruction_reserve 读取, 默认 120
        try:
            recall_obj = getattr(self.config.dki, 'recall', None)
            if recall_obj:
                budget_obj = getattr(recall_obj, 'budget', None) if hasattr(recall_obj, 'budget') else (recall_obj.get('budget') if isinstance(recall_obj, dict) else None)
                if budget_obj:
                    return getattr(budget_obj, 'instruction_reserve', 120) if hasattr(budget_obj, 'instruction_reserve') else (budget_obj.get('instruction_reserve', 120) if isinstance(budget_obj, dict) else 120)
        except Exception:
            pass
        return 120
    
    def _trim_history_to_budget(
        self,
        history_messages: list,
        system_content: str,
        original_query: str,
        max_prompt_tokens: int,
    ) -> list:
        """
        v5.7: 重新设计的历史消息修剪
        
        核心原则: 用户当前输入 (original_query) 和偏好 (system_content) 
        **永远不会被截断**, 只从最旧的历史消息开始移除。
        
        预算分配 (以 4096 为例):
        ┌──────────────────────────────────────────────────┐
        │ max_model_len = 4096                             │
        ├──────────────────────────────────────────────────┤
        │ 生成预留 (30%)                   = 1228          │
        │ 标记开销 (tag_overhead)          = 120           │
        │ ─────────────────────────────────────────────    │
        │ 可用于 prompt 的总预算            = 2748         │
        │                                                  │
        │ 固定开销 (不可压缩, 直接估算):                     │
        │   偏好 (system)                  ≈ 100-200       │
        │   当前用户输入                    ≈ 按实际估算     │
        │ ─────────────────────────────────────────────    │
        │ 剩余 → 历史消息预算                               │
        └──────────────────────────────────────────────────┘
        
        Args:
            history_messages: 历史消息列表 [{"role": ..., "content": ...}]
            system_content: system message 内容
            original_query: 用户当前输入 (不可截断)
            max_prompt_tokens: prompt 最大 token 数
            
        Returns:
            修剪后的历史消息列表 (从最旧的开始移除)
        """
        if not history_messages:
            return history_messages
        
        # 计算固定开销 (不可压缩部分, 直接估算)
        fixed_tokens = 0
        if system_content:
            fixed_tokens += self._estimate_prompt_tokens(system_content) + 10  # +10 for role tags
        # 当前输入: 直接估算 (不需要预留, 以估算值为准, 已含高估)
        fixed_tokens += self._estimate_prompt_tokens(original_query) + 10  # +10 for role tags
        fixed_tokens += 20  # generation prompt + misc tags
        
        # 历史消息可用预算
        history_budget = max_prompt_tokens - fixed_tokens
        
        if history_budget <= 0:
            logger.warning(
                f"No budget for history: max_prompt={max_prompt_tokens}, "
                f"fixed={fixed_tokens} (system+query+tags)"
            )
            return []
        
        # 从最新消息开始保留 (最旧的先丢弃)
        # 使用反向遍历: 最新的消息优先保留
        kept_messages = []
        used_tokens = 0
        
        for msg in reversed(history_messages):
            msg_tokens = self._estimate_prompt_tokens(msg['content']) + 8  # +8 for role tags per message
            if used_tokens + msg_tokens > history_budget:
                break
            kept_messages.insert(0, msg)
            used_tokens += msg_tokens
        
        trimmed_count = len(history_messages) - len(kept_messages)
        if trimmed_count > 0:
            logger.info(
                f"History trimmed to fit context: "
                f"kept {len(kept_messages)}/{len(history_messages)} messages, "
                f"history_budget={history_budget}, used={used_tokens}, "
                f"fixed_overhead={fixed_tokens} (system+query)"
            )
        
        return kept_messages
    
    def _build_recall_v4_chat_prompt(
        self,
        items: list,
        original_query: str,
        preference: Optional[Any] = None,
        trace_ids: Optional[list] = None,
        has_fact_call_instruction: bool = False,
        vague_reference: Optional[Any] = None,
    ) -> str:
        """
        v5.1: 使用 chat template 构造正确的多轮对话格式
        
        v5.6: 增加 token 预算控制, 确保用户当前输入永远不被截断
        
        修复角色混乱问题:
        之前的做法是将 recall_v4 后缀 (包含多轮历史) 扁平化为一个纯文本字符串,
        然后作为单一 user message 传给 apply_chat_template。这导致历史中的
        Assistant/User 角色标记变成纯文本, 模型无法区分历史角色和当前查询。
        
        新做法:
        1. 将偏好信息作为 system message
        2. 将 recall_v4 历史 items 按原始角色还原为独立的 user/assistant messages
        3. 将当前查询作为最后一个 user message
        4. 通过 tokenizer.apply_chat_template 生成正确的多轮 prompt
        5. **token 预算检查**: 如果总长度超过预算, 从最旧历史开始修剪
        
        这样模型能正确识别每轮对话的角色, 避免把助手的历史回复当成用户消息。
        
        Args:
            items: List[HistoryItem] - recall_v4 召回的历史条目
            original_query: 原始用户查询 (不含后缀)
            preference: 用户偏好 (用于 system message)
            trace_ids: 召回的 trace_id 列表 (用于 fact call 指令)
            has_fact_call_instruction: 是否需要 fact call 指令
            
        Returns:
            格式化后的完整 prompt (已含 chat template 特殊标记)
        """
        tokenizer = None
        if self._model_adapter and hasattr(self._model_adapter, 'tokenizer'):
            tokenizer = self._model_adapter.tokenizer
        elif hasattr(self, 'model') and hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        
        # ============ 1. System message: 偏好 + 召回元数据 + 澄清指令 ============
        system_parts = []
        if preference and preference.content:
            system_parts.append(f"用户偏好:\n{preference.content}")
        
        # 如果有 fact call 指令, 加入 system message
        if has_fact_call_instruction and trace_ids:
            if self._recall_formatter:
                constraint = self._recall_formatter.format_constraint_instruction(trace_ids)
                system_parts.append(constraint)
        
        # v6.5: 模糊指代澄清指令
        # 当检测到模糊指代且历史召回不足时, 注入澄清指令引导模型主动提问
        if vague_reference and vague_reference.is_vague:
            # 判断历史召回是否足够: 如果 items 为空或很少, 说明无法定位
            history_insufficient = len(items) <= 2
            if history_insufficient:
                clarification = build_clarification_instruction(vague_reference.language)
                system_parts.append(clarification)
                logger.info(
                    f"[Clarification Injected] recall_v4 path, "
                    f"history_items={len(items)}, lang={vague_reference.language}"
                )
        
        system_content = "\n\n".join(system_parts) if system_parts else ""
        
        # ============ 2. 历史 messages: 按原始角色还原 ============
        history_messages = []
        for item in items:
            role = getattr(item, 'role', None) or 'user'
            content = getattr(item, 'content', str(item))
            item_type = getattr(item, 'type', 'message')
            
            # v5.7: 移除历史消息中的 <think> 推理内容 (防御性)
            # 数据库存储时已清理, 但旧数据或外部数据可能仍含 think 内容
            if role == 'assistant' and content:
                content, _ = strip_think_content(content)
            
            # summary 类型: 标注为摘要, 保留 trace_id 信息
            if item_type == 'summary':
                trace_id = getattr(item, 'trace_id', '')
                if trace_id:
                    content = f"[摘要 trace_id={trace_id}] {content}"
            
            # 确保 role 是 chat template 支持的标准角色
            if role not in ('user', 'assistant', 'system'):
                role = 'user'
            
            # 跳过清理后为空的消息
            if not content or not content.strip():
                continue
            
            history_messages.append({"role": role, "content": content})
        
        # ============ 2.5 Token 预算修剪 (v5.6) ============
        # 确保用户当前输入永远不被截断, 必要时从最旧历史开始修剪
        max_prompt_tokens = self._get_max_prompt_tokens()
        history_messages = self._trim_history_to_budget(
            history_messages=history_messages,
            system_content=system_content,
            original_query=original_query,
            max_prompt_tokens=max_prompt_tokens,
        )
        
        # ============ 3. 构造 messages 列表 ============
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # 添加历史消息 (保留原始角色)
        for msg in history_messages:
            messages.append(msg)
        
        # 添加当前用户查询 (始终是最后一个 user message)
        messages.append({"role": "user", "content": original_query})
        
        # ============ 4. 使用 apply_chat_template 格式化 ============
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                logger.debug(
                    f"Recall v4 chat prompt built: "
                    f"{len(messages)} messages, "
                    f"{len(history_messages)} history items, "
                    f"system={'yes' if system_content else 'no'}"
                )
                return formatted
            except Exception as e:
                logger.warning(f"apply_chat_template failed in recall_v4: {e}")
        
        # ============ 回退: ChatML 格式 (DeepSeek/Qwen 通用) ============
        parts = []
        if system_content:
            parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
        
        for msg in history_messages:
            role = msg['role']
            parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{original_query}<|im_end|>")
        parts.append("<|im_start|>assistant")
        
        formatted = "\n".join(parts) + "\n"
        logger.debug(f"Recall v4 chat prompt built (ChatML fallback): {len(messages)} messages")
        return formatted
    
    def _build_hybrid_chat_prompt(
        self,
        history,
        original_query: str,
        preference=None,
        vague_reference: Optional[Any] = None,
    ) -> str:
        """
        v5.1: Hybrid 模式下使用 chat template 构造正确的多轮对话格式
        
        v5.6: 增加 token 预算控制, 确保用户当前输入永远不被截断
        
        与 _build_recall_v4_chat_prompt 类似, 但接收的是 SessionHistory 对象
        而非 HistoryItem 列表。
        
        Args:
            history: SessionHistory 对象, 包含 .messages 列表
            original_query: 原始用户查询
            preference: 用户偏好对象
            
        Returns:
            格式化后的完整 prompt
        """
        tokenizer = None
        if self._model_adapter and hasattr(self._model_adapter, 'tokenizer'):
            tokenizer = self._model_adapter.tokenizer
        elif hasattr(self, 'model') and hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        
        # ============ 1. System message ============
        system_parts = []
        if preference:
            pref_content = getattr(preference, 'content', str(preference))
            if pref_content:
                system_parts.append(f"用户偏好:\n{pref_content}")
        
        # v6.5: 模糊指代澄清指令
        if vague_reference and vague_reference.is_vague:
            history_count = len(history.messages) if history and hasattr(history, 'messages') else 0
            history_insufficient = history_count <= 2
            if history_insufficient:
                clarification = build_clarification_instruction(vague_reference.language)
                system_parts.append(clarification)
                logger.info(
                    f"[Clarification Injected] hybrid path, "
                    f"history_msgs={history_count}, lang={vague_reference.language}"
                )
        
        system_content = "\n\n".join(system_parts) if system_parts else ""
        
        # ============ 2. 历史消息还原为独立 messages ============
        history_messages = []
        if history and hasattr(history, 'messages'):
            for msg in history.messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    role = getattr(msg, 'role', 'user')
                    content = getattr(msg, 'content', str(msg))
                
                if role not in ('user', 'assistant', 'system'):
                    role = 'user'
                
                # v5.7: 移除历史消息中的 <think> 推理内容 (防御性)
                if role == 'assistant' and content:
                    content, _ = strip_think_content(content)
                
                if content and content.strip():
                    history_messages.append({"role": role, "content": content})
        
        # ============ 2.5 Token 预算修剪 (v5.6) ============
        # 确保用户当前输入永远不被截断, 必要时从最旧历史开始修剪
        max_prompt_tokens = self._get_max_prompt_tokens()
        history_messages = self._trim_history_to_budget(
            history_messages=history_messages,
            system_content=system_content,
            original_query=original_query,
            max_prompt_tokens=max_prompt_tokens,
        )
        
        # ============ 3. 构造 messages 列表 ============
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        for msg in history_messages:
            messages.append(msg)
        
        messages.append({"role": "user", "content": original_query})
        
        # ============ 4. apply_chat_template ============
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                logger.debug(
                    f"Hybrid chat prompt built: "
                    f"{len(messages)} messages, "
                    f"{len(history_messages)} history items"
                )
                return formatted
            except Exception as e:
                logger.warning(f"apply_chat_template failed in hybrid: {e}")
        
        # ============ 回退: ChatML ============
        parts = []
        if system_content:
            parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
        
        for msg in history_messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{original_query}<|im_end|>")
        parts.append("<|im_start|>assistant")
        
        formatted = "\n".join(parts) + "\n"
        logger.debug(f"Hybrid chat prompt built (ChatML fallback): {len(messages)} messages")
        return formatted
    
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
        
        # Apply alpha scaling to memory K/V (MIS) — 仅对记忆 K/V 缩放
        # 注意: 必须在 preference merge 之前完成, 否则 preference K/V 会被双重缩放
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
        
        # Add preference K/V if available (hybrid injection)
        # Preference K/V 使用自己的 preference_alpha, 不受 gating_decision.alpha 影响
        # Preference K/V is prepended (negative positions conceptually)
        if preference_kv is not None and preference_alpha > 0:
            if timer:
                timer.start_stage("preference_merge")
            
            # Scale preference K/V with its own alpha (独立缩放)
            # Note: KV tensors may be on CPU (to save GPU memory), MIS handles this
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
        user_id: Optional[str] = None,
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
            
            # ============ Function Call 日志: 记录调用前状态 ============
            import time as _time
            fc_start = _time.time()
            prompt_before_call = prompt
            model_output_before_call = output.text
            fc_arguments = {
                "trace_id": fact_request.trace_id,
                "offset": fact_request.offset,
                "limit": fact_request.limit,
            }
            
            # 检索事实
            try:
                fact_response = self._fact_retriever.retrieve(
                    trace_id=fact_request.trace_id,
                    session_id=session_id,
                    offset=fact_request.offset,
                    limit=fact_request.limit,
                )
            except Exception as retrieve_err:
                self._log_function_call(
                    session_id=session_id, user_id=user_id,
                    round_index=round_idx, function_name="retrieve_fact",
                    arguments=fc_arguments, response_text=None,
                    status="error", error_message=str(retrieve_err),
                    prompt_before=prompt_before_call, prompt_after=prompt,
                    model_output_before=model_output_before_call,
                    latency_ms=(_time.time() - fc_start) * 1000,
                )
                return output, round_idx
            
            if not fact_response.messages:
                self._log_function_call(
                    session_id=session_id, user_id=user_id,
                    round_index=round_idx, function_name="retrieve_fact",
                    arguments=fc_arguments, response_text="(no facts found)",
                    status="success",
                    prompt_before=prompt_before_call, prompt_after=prompt,
                    model_output_before=model_output_before_call,
                    latency_ms=(_time.time() - fc_start) * 1000,
                )
                return output, round_idx
            
            # 格式化事实段落
            fact_text = self._recall_formatter.format_fact_segment(fact_response)
            
            # 粗估 token
            fact_tokens = len(fact_text) // 2
            total_fact_tokens += fact_tokens
            
            if total_fact_tokens > max_fact_tokens:
                logger.info(f"Fact token budget exhausted: {total_fact_tokens}")
                self._log_function_call(
                    session_id=session_id, user_id=user_id,
                    round_index=round_idx, function_name="retrieve_fact",
                    arguments=fc_arguments, response_text=fact_text,
                    response_tokens=fact_tokens, status="budget_exceeded",
                    error_message=f"Total {total_fact_tokens} > max {max_fact_tokens}",
                    prompt_before=prompt_before_call, prompt_after=prompt,
                    model_output_before=model_output_before_call,
                    latency_ms=(_time.time() - fc_start) * 1000,
                )
                return output, round_idx
            
            # 追加事实到 prompt
            continuation = "请基于以上补充事实回答用户问题。"
            prompt = prompt + "\n\n" + fact_text + "\n\n" + continuation
            
            # ============ Function Call 日志: 记录成功调用 ============
            self._log_function_call(
                session_id=session_id, user_id=user_id,
                round_index=round_idx, function_name="retrieve_fact",
                arguments=fc_arguments, response_text=fact_text,
                response_tokens=fact_tokens, status="success",
                prompt_before=prompt_before_call, prompt_after=prompt,
                model_output_before=model_output_before_call,
                latency_ms=(_time.time() - fc_start) * 1000,
            )
            
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
    
    def _log_function_call(
        self,
        session_id: str,
        user_id: Optional[str],
        round_index: int,
        function_name: str,
        arguments: Dict[str, Any],
        response_text: Optional[str],
        response_tokens: int = 0,
        status: str = "success",
        error_message: Optional[str] = None,
        prompt_before: Optional[str] = None,
        prompt_after: Optional[str] = None,
        model_output_before: Optional[str] = None,
        latency_ms: float = 0,
    ) -> None:
        """记录 function call 日志 (委托给 FunctionCallLogger)"""
        if not hasattr(self, '_fc_logger') or not self._fc_logger:
            return
        
        try:
            self._fc_logger.log(
                session_id=session_id,
                user_id=user_id,
                round_index=round_index,
                function_name=function_name,
                arguments=arguments,
                response_text=response_text,
                response_tokens=response_tokens,
                status=status,
                error_message=error_message,
                prompt_before=prompt_before,
                prompt_after=prompt_after,
                model_output_before=model_output_before,
                latency_ms=latency_ms,
            )
        except Exception as log_err:
            logger.warning(f"Failed to log function call: {log_err}")
    
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
        # v5.7: 存储前移除 <think> 推理内容 (节省历史 token 预算)
        clean_response, think_stripped = strip_think_content(response)
        if think_stripped:
            logger.debug(
                f"Think content stripped before DB storage: "
                f"{len(response)} -> {len(clean_response)} chars"
            )
        
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
            
            # Assistant response (存储清理后的内容)
            conv_repo.create(
                session_id=session_id,
                role='assistant',
                content=clean_response,
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
        
        # 用户隔离统计
        isolation_stats = {}
        if self._cache_audit_log:
            isolation_stats = self._cache_audit_log.get_stats()
        
        return {
            'router_stats': self.memory_router.get_stats(),
            'model_info': self.model.get_model_info() if self._model_adapter else None,
            'session_caches': session_stats,
            'tiered_caches': tiered_stats,
            'tiered_memory_footprint': tiered_footprint,
            'budget_analysis': self._budget_analyzer.get_stats(),
            'user_isolation': {
                'enabled': self._user_isolation_enabled,
                'audit': isolation_stats,
            },
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

class _DetachedMessage:
    """
    脱管消息的纯 Python 副本 (解决 SQLAlchemy DetachedInstanceError)
    
    SQLAlchemy ORM 对象在 session.close() 后变为 detached 状态,
    访问其属性会失败。此类在 session 活跃时提取所有需要的属性,
    创建独立于 session 的纯 Python 对象。
    """
    __slots__ = ('id', 'message_id', 'session_id', 'role', 'content',
                 'created_at', 'injection_mode', 'injection_alpha',
                 'memory_ids', 'latency_ms')
    
    def __init__(self, orm_obj):
        self.id = getattr(orm_obj, 'id', None)
        self.message_id = getattr(orm_obj, 'id', None)  # 兼容 recall 组件
        self.session_id = getattr(orm_obj, 'session_id', None)
        self.role = getattr(orm_obj, 'role', 'user')
        self.content = getattr(orm_obj, 'content', '')
        self.created_at = getattr(orm_obj, 'created_at', None)
        self.injection_mode = getattr(orm_obj, 'injection_mode', None)
        self.injection_alpha = getattr(orm_obj, 'injection_alpha', None)
        self.memory_ids = getattr(orm_obj, 'memory_ids', None)
        self.latency_ms = getattr(orm_obj, 'latency_ms', None)


class _ConversationRepoWrapper:
    """
    ConversationRepository 的安全包装器
    
    为 recall 组件提供统一接口, 自动管理数据库 session。
    
    关键修复 (v5.1):
    返回 _DetachedMessage 纯 Python 副本, 而非 SQLAlchemy ORM 对象。
    ORM 对象在 session_scope 退出后会变为 detached 状态,
    导致后续属性访问静默失败 (role/content 返回 None),
    这是 DKI 模式历史消息注入为 0 的根因。
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self._db_manager = db_manager
    
    @staticmethod
    def _to_detached(orm_objects: List[Any]) -> List[_DetachedMessage]:
        """将 ORM 对象列表转换为脱管安全的纯 Python 副本"""
        return [_DetachedMessage(obj) for obj in orm_objects]
    
    def get_by_session(
        self,
        session_id: str,
        db_session: Optional[Any] = None,
        **kwargs,
    ) -> List[_DetachedMessage]:
        """获取会话的所有消息 (返回脱管安全副本)"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            orm_results = repo.get_by_session(session_id)
            # 在 session 活跃时提取属性, 避免 DetachedInstanceError
            return self._to_detached(orm_results)
    
    def get_recent(
        self,
        session_id: str,
        limit: int = 10,
        **kwargs,
    ) -> List[_DetachedMessage]:
        """获取最近的消息 (返回脱管安全副本)"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            orm_results = repo.get_recent(session_id, limit=limit)
            return self._to_detached(orm_results)
    
    def get_by_id(self, message_id: str) -> Optional[_DetachedMessage]:
        """根据 ID 获取消息 (返回脱管安全副本)"""
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            if hasattr(repo, 'get_by_id'):
                result = repo.get_by_id(message_id)
                if result:
                    return _DetachedMessage(result)
        return None
    
    def get_cross_session_history(
        self,
        user_id: str,
        current_session_id: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> List[_DetachedMessage]:
        """
        获取用户跨会话的历史消息 (返回脱管安全副本)
        
        通过 Session.user_id 关联, 检索该用户所有历史会话的消息,
        排除当前会话 (当前会话消息由 get_by_session 获取)。
        
        Args:
            user_id: 用户 ID
            current_session_id: 当前会话 ID (排除, 避免重复)
            limit: 最大消息数
            
        Returns:
            其他会话的历史消息 (时间顺序, 旧到新)
        """
        with self._db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            orm_results = repo.get_recent_by_user_cross_session(
                user_id=user_id,
                current_session_id=current_session_id,
                limit=limit,
            )
            return self._to_detached(orm_results)