"""
DKI Plugin - Dynamic KV Injection Plugin for LLM

DKI 作为 LLM 的插件，接收原始用户输入，通过配置驱动的适配器读取外部数据，
执行 K/V 注入后调用 LLM 推理。

核心职责:
1. 通过配置驱动的适配器读取上层应用的数据库 (用户偏好 + 历史消息)
2. 执行 DKI 注入 (偏好 K/V 负位置 + 历史后缀正位置)
3. 调用 LLM 推理
4. 记录工作数据供监控

上层应用集成方式:
1. 提供适配器配置文件 (指定数据库连接和字段映射)
2. 删除 RAG/Prompt 工程代码
3. 传递 user_id + 原始用户输入给 DKI

架构 (v3.0 重构):
    DKIPlugin (瘦 Facade, 对外接口不变)
       ↓
    InjectionPlanner (纯决策, 不碰模型)
       ↓ InjectionPlan (中间产物)
    InjectionExecutor (纯执行, 不做决策)
       ↓
    ModelAdapter (LLM 推理)

Author: AGI Demo Project
Version: 3.0.0
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

from loguru import logger

from dki.core.exceptions import (
    DKIError,
    AdapterConnectionError,
    AdapterSchemaError,
    AdapterTimeoutError,
    KVComputeError,
    KVOOMError,
    ModelError,
    ModelOOMError,
    RecallError,
    BM25InitError,
)

from dki.core.text_utils import (
    strip_think_content,
    estimate_tokens_fast,
    detect_vague_reference,
    build_clarification_instruction,
    init_think_filter,
    get_think_filter,
    create_stream_filter,
    create_stream_detector,
    get_show_thinking,
)

from dki.adapters.base import (
    IUserDataAdapter,
    UserPreference as AdapterUserPreference,
    ChatMessage as AdapterChatMessage,
    UserProfile,
)
from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling
from dki.core.components.dual_factor_gating import DualFactorGating, GatingDecision
from dki.core.components.memory_trigger import (
    MemoryTrigger,
    MemoryTriggerConfig,
    TriggerType,
    TriggerResult,
)
from dki.core.components.reference_resolver import (
    ReferenceResolver,
    ReferenceResolverConfig,
    ReferenceType,
    ResolvedReference,
    Message as ResolverMessage,
)
from dki.config.config_loader import ConfigLoader
from dki.cache import (
    PreferenceCacheManager,
    CacheConfig,
    DKIRedisClient,
    RedisConfig,
    REDIS_AVAILABLE,
)
from dki.api.visualization_routes import record_visualization

# 新架构组件
from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    SafetyEnvelope,
    QueryContext,
    ExecutionResult,
)
from dki.core.plugin.injection_planner import InjectionPlanner
from dki.core.plugin.injection_executor import InjectionExecutor


@dataclass
class InjectionMetadata:
    """DKI 注入元数据 (用于监控)"""
    # 注入状态
    injection_enabled: bool = False
    alpha: float = 0.0
    
    # 注入策略 (recall_v4)
    injection_strategy: str = "recall_v4"
    
    # Token 统计
    preference_tokens: int = 0
    history_tokens: int = 0
    query_tokens: int = 0
    total_tokens: int = 0
    
    # 缓存状态
    preference_cache_hit: bool = False
    preference_cache_tier: str = "none"
    
    # 性能
    latency_ms: float = 0.0
    adapter_latency_ms: float = 0.0
    injection_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    
    # 门控决策
    gating_decision: Optional[Dict[str, Any]] = None
    
    # 数据来源
    preferences_count: int = 0
    history_messages_count: int = 0
    relevant_history_count: int = 0
    
    # Memory Trigger 信息
    memory_triggered: bool = False
    trigger_type: Optional[str] = None
    
    # Reference Resolver 信息
    reference_resolved: bool = False
    reference_type: Optional[str] = None
    reference_scope: Optional[str] = None
    
    # v7.0: 向量检索能力信息
    retrieval_mode: str = "unknown"  # bm25_only | bm25_embedding | keyword | unknown
    
    # v9.0: 注入明文详情 (供上层应用 / 实验系统获取)
    # 之前通过 _last_injection_detail hack 暴露, 现在作为正式 metadata 字段
    preference_text: Optional[str] = None
    history_suffix_text: Optional[str] = None
    history_messages: Optional[List[Dict[str, str]]] = None
    final_input: Optional[str] = None
    
    # Alpha Profile (v3.0)
    alpha_profile: Optional[Dict[str, Any]] = None
    
    # 安全违规 (v3.0)
    safety_violations: Optional[List[str]] = None
    
    # 时间戳
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "injection_enabled": self.injection_enabled,
            "injection_strategy": self.injection_strategy,
            "alpha": self.alpha,
            "alpha_profile": self.alpha_profile,
            "tokens": {
                "preference": self.preference_tokens,
                "history": self.history_tokens,
                "query": self.query_tokens,
                "total": self.total_tokens,
            },
            "cache": {
                "preference_hit": self.preference_cache_hit,
                "preference_tier": self.preference_cache_tier,
            },
            "latency": {
                "total_ms": self.latency_ms,
                "adapter_ms": self.adapter_latency_ms,
                "injection_ms": self.injection_latency_ms,
                "inference_ms": self.inference_latency_ms,
            },
            "data_source": {
                "preferences_count": self.preferences_count,
                "history_messages_count": self.history_messages_count,
                "relevant_history_count": self.relevant_history_count,
            },
            "gating_decision": self.gating_decision,
            "memory_trigger": {
                "triggered": self.memory_triggered,
                "type": self.trigger_type,
            },
            "reference_resolver": {
                "resolved": self.reference_resolved,
                "type": self.reference_type,
                "scope": self.reference_scope,
            },
            "retrieval_mode": self.retrieval_mode,
            "safety_violations": self.safety_violations or [],
            # v9.0: 注入明文详情
            "injection_detail": {
                "preference_text": self.preference_text,
                "history_suffix_text": self.history_suffix_text,
                "history_messages": self.history_messages or [],
                "final_input": self.final_input,
            },
        }


@dataclass
class DKIPluginResponse:
    """DKI 插件响应"""
    # 生成结果
    text: str
    
    # Token 统计
    input_tokens: int = 0
    output_tokens: int = 0
    
    # 注入元数据 (用于监控)
    metadata: InjectionMetadata = field(default_factory=InjectionMetadata)
    
    # 原始模型输出 (可选)
    raw_output: Optional[ModelOutput] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata.to_dict(),
        }


class DKIPlugin:
    """
    DKI 插件核心 (瘦 Facade)
    
    v3.0 架构:
    - 对外接口不变 (chat, get_stats, from_config, ...)
    - 内部拆分为 Planner + Executor
    - 决策与执行分离, 可独立测试
    
    上层应用集成方式:
    1. 提供适配器配置文件 (指定数据库连接和字段映射)
    2. 删除 RAG/Prompt 工程代码
    3. 传递 user_id + 原始用户输入给 DKI
    
    使用方式 1: 从配置文件创建 (推荐)
    ```python
    dki = await DKIPlugin.from_config(
        model_adapter=vllm_adapter,
        adapter_config_path="config/adapter_config.yaml",
    )
    
    response = await dki.chat(
        query="推荐一家餐厅",
        user_id="user_123",
        session_id="session_456",
    )
    ```
    
    使用方式 2: 从配置字典创建
    ```python
    dki = await DKIPlugin.from_config(
        model_adapter=vllm_adapter,
        adapter_config={
            "database": {"type": "postgresql", "host": "localhost", ...},
            "preferences": {"table": "user_preferences", "fields": {...}},
            "messages": {"table": "chat_messages", "fields": {...}},
        },
    )
    ```
    
    高级用法: 直接访问 Planner / Executor (用于测试)
    ```python
    # 生成计划但不执行
    context = dki.planner.analyze_query("推荐餐厅")
    plan = dki.planner.build_plan(query, user_id, prefs, history, context)
    print(plan.to_dict())  # 检查决策
    
    # 手动执行
    result = await dki.executor.execute(plan)
    ```
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        user_data_adapter: IUserDataAdapter,
        config: Optional[Any] = None,
        language: str = "en",
        memory_trigger_config: Optional[MemoryTriggerConfig] = None,
        reference_resolver_config: Optional[ReferenceResolverConfig] = None,
        redis_client: Optional[DKIRedisClient] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        初始化 DKI 插件
        
        Args:
            model_adapter: LLM 模型适配器
            user_data_adapter: 外部数据适配器 (读取上层应用的数据库)
            config: 配置 (可选，默认从 config.yaml 加载)
            language: 语言 ("en" | "cn")
            memory_trigger_config: Memory Trigger 配置 (可选)
            reference_resolver_config: Reference Resolver 配置 (可选)
            redis_client: Redis 客户端 (可选，用于分布式缓存)
            cache_config: 缓存配置 (可选)
        """
        self.model = model_adapter
        self.data_adapter = user_data_adapter
        self.config = config or ConfigLoader().config
        self.language = language
        
        # ============ v5.8: 初始化思考内容过滤器 (外置正则) ============
        think_filter_config = None
        if hasattr(self.config, 'dki') and hasattr(self.config.dki, 'think_filter'):
            tf_obj = self.config.dki.think_filter
            if isinstance(tf_obj, dict):
                think_filter_config = tf_obj
            elif hasattr(tf_obj, 'model_dump'):
                think_filter_config = tf_obj.model_dump()
            elif hasattr(tf_obj, 'dict'):
                think_filter_config = tf_obj.dict()
            elif hasattr(tf_obj, '__dict__'):
                think_filter_config = vars(tf_obj)
        init_think_filter(think_filter_config)
        
        # DKI 组件 (延迟初始化)
        self._mis: Optional[MemoryInfluenceScaling] = None
        self._gating: Optional[DualFactorGating] = None
        
        # ============ Recall v4 组件 ============
        self._fact_retriever = None
        self._prompt_formatter = None
        self._recall_config = None
        self._suffix_builder = None
        try:
            from dki.core.recall import (
                RecallConfig,
                SuffixBuilder,
                FactRetriever,
                create_formatter,
            )
            # 从配置加载 recall config
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
            self._recall_config = RecallConfig.from_dict(recall_dict)
            
            # 创建 PromptFormatter
            model_name = getattr(model_adapter, 'model_name', '') or ''
            self._prompt_formatter = create_formatter(
                model_name=model_name,
                formatter_type=self._recall_config.prompt_formatter,
                language=language,
            )
            
            # v6.2: 创建 SuffixBuilder (修复历史消息压缩缺失的 Bug)
            # DKI Plugin 使用 adapter 做召回, 但仍需 SuffixBuilder 做 token 预算分配和压缩
            token_counter = None
            if model_adapter and hasattr(model_adapter, 'tokenizer') and model_adapter.tokenizer:
                tokenizer = model_adapter.tokenizer
                token_counter = lambda text: len(tokenizer.encode(text)) if text else 0
            self._suffix_builder = SuffixBuilder(
                config=self._recall_config,
                prompt_formatter=self._prompt_formatter,
                token_counter=token_counter,
                model_adapter=model_adapter,
            )
            
            # FactRetriever 需要 conversation_repo, 延迟初始化
            self._fact_retriever = FactRetriever(
                config=self._recall_config,
                conversation_repo=None,  # 延迟设置
            )
            
            logger.info("Recall v4 components initialized (SuffixBuilder + FactRetriever)")
        except ImportError:
            logger.info("Recall v4 components not available, fact resolution disabled")
        except Exception as recall_err:
            logger.warning(f"Recall v4 init failed (non-critical): {recall_err}")
        
        # ============ 获取 context_window ============
        self._context_window = self._resolve_context_window()
        
        # ============ Planner (纯决策, v6.2: 含 SuffixBuilder) ============
        self._planner = InjectionPlanner(
            config=self.config,
            language=language,
            injection_strategy="recall_v4",
            memory_trigger_config=memory_trigger_config,
            reference_resolver_config=reference_resolver_config,
            recall_config=self._recall_config,
            suffix_builder=self._suffix_builder,  # v6.2: 传递 SuffixBuilder
            fact_retriever=self._fact_retriever,
            prompt_formatter=self._prompt_formatter,
        )
        
        # ============ Function Call Logger (v3.2) ============
        self._fc_logger = None
        try:
            from dki.core.function_call_logger import FunctionCallLogger
            self._fc_logger = FunctionCallLogger(text_log_dir="logs/function_calls")
        except Exception as fc_err:
            logger.warning(f"FunctionCallLogger init failed (non-critical): {fc_err}")
        
        # ============ Executor (纯执行, v3.3: O(1) forward pass) ============
        # v7.1: 传递 fact_retriever, prompt_formatter, recall_config 以支持 inline_intercept
        self._executor = InjectionExecutor(
            model_adapter=model_adapter,
            function_call_logger=self._fc_logger,
            fact_retriever=self._fact_retriever,
            prompt_formatter=self._prompt_formatter,
            recall_config=self._recall_config,
        )
        
        # ============ 偏好 K/V 缓存 (支持 Redis 分布式) ============
        self._redis_client = redis_client
        self._cache_config = cache_config or CacheConfig()
        self._preference_cache = PreferenceCacheManager(
            redis_client=redis_client,
            config=self._cache_config,
        )
        
        # ============ P1-3: 偏好文本缓存 (5 分钟 TTL, 有界 LRU) ============
        # 偏好是低频变更数据，但每次 chat 都查 DB → P95 延迟来源
        # KV (衍生物) 有三级缓存，偏好文本 (源数据) 也应有缓存
        # 修复: 使用 OrderedDict 实现有界 LRU, 防止内存无限增长
        from collections import OrderedDict
        self._preference_text_cache: OrderedDict = OrderedDict()
        self._preference_text_cache_maxsize: int = 1000
        self._preference_cache_ttl: float = 300.0  # 5 分钟 TTL (秒)
        
        # P0-4: AsyncSingleFlight 防止偏好缓存 thundering herd
        self._preference_single_flight: Dict[str, asyncio.Future] = {}
        
        # 工作日志 (用于监控 API)
        self._injection_logs: List[InjectionMetadata] = []
        self._max_logs = 1000
        
        # 统计数据
        self._stats = {
            "total_requests": 0,
            "injection_enabled_count": 0,
            "cache_hits": 0,
            "total_latency_ms": 0.0,
            "avg_alpha": 0.0,
        }
        
        # 日志输出缓存状态
        cache_status = "L1 only"
        if redis_client and redis_client.is_available:
            cache_status = "L1 + L2 (Redis)"
        elif self._cache_config.l2_enabled:
            cache_status = "L1 + L2 (Redis not connected)"
        
        logger.info(
            f"DKI Plugin initialized "
            f"(strategy=recall_v4, language={language}, "
            f"cache={cache_status}, architecture=planner+executor)"
        )
    
    # ================================================================
    # 内部方法
    # ================================================================
    
    def _detect_retrieval_mode(self) -> str:
        """
        v7.0: 检测适配器的检索模式
        
        根据 ConfigDrivenAdapter 的状态判断当前使用的检索策略:
        - bm25_embedding: 有 vector_index_config → BM25 + Embedding 混合检索
        - bm25_only: 无 vector_index_config → 仅 BM25 召回
        - pgvector: 使用 PostgreSQL 预计算向量
        - keyword: 简单关键词匹配
        - unknown: 非 ConfigDrivenAdapter 或无法判断
        """
        try:
            from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
            
            if not isinstance(self.data_adapter, ConfigDrivenAdapter):
                return "unknown"
            
            adapter = self.data_adapter
            vs_config = adapter.adapter_config.vector_search
            
            if not vs_config.enabled:
                return "keyword"
            
            # 检查是否有向量能力
            if vs_config.has_vector_capability:
                if vs_config.type.value == "pgvector":
                    return "pgvector"
                return "bm25_embedding"
            
            # 无向量能力但 type=dynamic → BM25-only
            if getattr(adapter, '_bm25_only_mode', False):
                return "bm25_only"
            
            return "bm25_only"
            
        except Exception:
            return "unknown"
    
    def _get_max_recent_turns(self) -> int:
        """
        v7.2: 获取近轮对话的最大轮次数
        
        从配置 dki.recall.budget.max_recent_turns 读取, 默认 5
        """
        try:
            if self._recall_config and hasattr(self._recall_config, 'budget'):
                return getattr(self._recall_config.budget, 'max_recent_turns', 5)
        except Exception:
            pass
        return 5
    
    def _get_stream_buffer_max(self) -> int:
        """
        v5.8: 获取流式过滤缓冲区最大字符数
        
        从配置 dki.think_filter.stream_buffer_max_chars 读取, 默认 500
        """
        try:
            if hasattr(self.config, 'dki') and hasattr(self.config.dki, 'think_filter'):
                tf = self.config.dki.think_filter
                if isinstance(tf, dict):
                    return tf.get('stream_buffer_max_chars', 500)
                return getattr(tf, 'stream_buffer_max_chars', 500)
        except Exception:
            pass
        return 500
    
    def _merge_recent_and_recalled(
        self,
        recent_messages: List[AdapterChatMessage],
        recalled_messages: List[AdapterChatMessage],
    ) -> List[AdapterChatMessage]:
        """
        v7.2: 合并近轮对话与 BM25 召回结果
        v7.3: 合并后按时间排序, 确保成对注入
        v7.4: 过滤末尾无 assistant 回复的 user 消息
              (用户当前查询已在 prompt 最后, 不需要在历史中重复)
        
        策略: 近轮优先, 去重, 按时间排序, 成对补全, 过滤孤立末尾 user
        - recent_messages: 按时间正序 (最旧在前), 确保多轮连贯
        - recalled_messages: 按 BM25 相关性排序, 可能来自任意会话
        
        合并后: 按时间正序排列 (近轮 + BM25 混合, 去重)
        """
        seen_ids = set()
        merged = []
        
        # 1. 近轮消息优先 (保持时间序)
        for msg in recent_messages:
            msg_id = getattr(msg, 'message_id', None) or id(msg)
            msg_id = str(msg_id)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                merged.append(msg)
        
        # 2. BM25 召回补充 (去重)
        bm25_added = 0
        for msg in recalled_messages:
            msg_id = getattr(msg, 'message_id', None) or id(msg)
            msg_id = str(msg_id)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                merged.append(msg)
                bm25_added += 1
        
        # v7.3: 合并后按时间戳排序 (确保时间正序)
        merged.sort(
            key=lambda m: (
                m.timestamp.isoformat() if hasattr(getattr(m, 'timestamp', None), 'isoformat')
                else str(getattr(m, 'timestamp', ''))
            )
        )
        
        # v7.4: 过滤末尾无 assistant 回复的 user 消息
        # 用户当前查询已经写入 DB 但尚无 assistant 回复, 会被 get_recent_messages 拉出
        # 这条消息已在 prompt 最后作为独立查询, 历史中不应重复
        merged = self._remove_trailing_unpaired_user(merged)
        
        if bm25_added > 0:
            logger.debug(
                f"Merged history: {len(recent_messages)} recent + "
                f"{bm25_added} BM25 recalled = {len(merged)} total "
                f"(sorted by timestamp)"
            )
        
        return merged
    
    @staticmethod
    def _remove_trailing_unpaired_user(
        messages: List[AdapterChatMessage],
    ) -> List[AdapterChatMessage]:
        """
        v7.4: 从末尾移除没有 assistant 回复的 user 消息
        
        原因:
        - Demo 流程: 先写入 user 消息 → 再调用 DKI Plugin
        - get_recent_messages 会拉出刚写入的 user 消息 (无 assistant 回复)
        - 该消息已在 prompt 最后作为用户当前输入, 历史中重复注入会导致:
          1. 浪费 context budget
          2. 模型混淆 (同一问题出现两次)
        
        策略:
        - 从列表末尾向前扫描, 移除连续的 role="user" 消息
        - 一旦遇到 role="assistant" 停止 (之前的 user 消息是有配对的)
        """
        if not messages:
            return messages
        
        # 从末尾向前扫描
        cut_index = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            role = getattr(messages[i], 'role', 'user')
            if role == 'user':
                cut_index = i
            else:
                break  # 遇到 assistant, 停止
        
        if cut_index < len(messages):
            removed_count = len(messages) - cut_index
            logger.debug(
                f"Removed {removed_count} trailing unpaired user message(s) "
                f"from history (already in current query)"
            )
            return messages[:cut_index]
        
        return messages
    
    def _resolve_context_window(self) -> int:
        """
        v6.2: 从配置或模型中获取上下文窗口大小
        
        优先级:
        1. config.model.engines.{engine}.max_model_len
        2. model_adapter.tokenizer.model_max_length
        3. 默认 4096
        """
        context_window = 4096
        
        # 尝试从配置获取
        try:
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'engines'):
                engine_name = getattr(self.config.model, 'default_engine', None)
                if engine_name:
                    engine_cfg = self.config.model.engines.get(engine_name, {})
                    if hasattr(engine_cfg, 'max_model_len'):
                        context_window = engine_cfg.max_model_len
                    elif isinstance(engine_cfg, dict) and 'max_model_len' in engine_cfg:
                        context_window = engine_cfg['max_model_len']
        except Exception:
            pass
        
        # 回退: 从 tokenizer 获取
        if context_window <= 4096:
            try:
                if self.model and hasattr(self.model, 'tokenizer') and self.model.tokenizer:
                    max_len = getattr(self.model.tokenizer, 'model_max_length', None)
                    if max_len and max_len < 1_000_000:  # 排除 sentinel 值
                        context_window = max_len
            except Exception:
                pass
        
        logger.info(f"DKI Plugin context_window resolved: {context_window}")
        return context_window
    
    # ================================================================
    # 内部组件访问器 (高级用法 / 测试)
    # ================================================================
    
    @property
    def planner(self) -> InjectionPlanner:
        """获取 Planner (用于测试/调试)"""
        return self._planner
    
    @property
    def executor(self) -> InjectionExecutor:
        """获取 Executor (用于测试/调试)"""
        return self._executor
    
    @property
    def mis(self) -> MemoryInfluenceScaling:
        """获取 MIS 组件"""
        if self._mis is None:
            self._mis = MemoryInfluenceScaling(
                hidden_dim=self.model.hidden_dim,
                use_learned_alpha=False,
            )
        return self._mis
    
    @property
    def gating(self) -> DualFactorGating:
        """获取门控组件"""
        if self._gating is None:
            self._gating = DualFactorGating()
        return self._gating
    
    # ================================================================
    # P1-3: 偏好文本缓存
    # ================================================================
    
    async def _get_cached_preferences(
        self, user_id: str
    ) -> List[AdapterUserPreference]:
        """
        带 TTL 的偏好文本缓存 (P1-3) + AsyncSingleFlight (P0-4)
        
        偏好是低频变更数据 (通常数天不变)，但每次 chat 都查 DB。
        5 分钟 TTL 可显著降低 P95 延迟，偏好更新后最多 5 分钟生效。
        
        P0-4: 使用 single-flight 模式防止 thundering herd:
        当 100 个并发请求同时缓存未命中时, 只有第一个请求查 DB,
        其余请求等待第一个请求的结果。
        
        Args:
            user_id: 用户 ID
            
        Returns:
            用户偏好列表 (可能来自缓存)
        """
        now = time.time()
        
        # 1. 检查 TTL 缓存 (有界 LRU)
        if user_id in self._preference_text_cache:
            cached_prefs, cached_at = self._preference_text_cache[user_id]
            if now - cached_at < self._preference_cache_ttl:
                # 命中: 移到末尾 (LRU)
                self._preference_text_cache.move_to_end(user_id)
                logger.debug(f"Preference text cache hit for user {user_id}")
                return cached_prefs
            else:
                # 已过期, 移除
                self._preference_text_cache.pop(user_id, None)
        
        # 2. P0-4: Single-flight — 合并并发请求 (shield 保护 + 异常降级)
        flight_key = f"pref:{user_id}"
        if flight_key in self._preference_single_flight:
            # 已有 in-flight 请求, 等待其结果
            logger.debug(f"Preference single-flight join for user {user_id}")
            try:
                return await asyncio.shield(self._preference_single_flight[flight_key])
            except Exception:
                # 降级: 主请求异常时, 等待方返回空列表而不是向上传播
                return []
        
        # 创建 future, 标记 in-flight
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._preference_single_flight[flight_key] = future
        
        try:
            # 缓存未命中，查询 DB
            preferences = await self.data_adapter.get_user_preferences(user_id)
            self._preference_text_cache[user_id] = (preferences, time.time())
            self._preference_text_cache.move_to_end(user_id)
            # 有界淘汰: 超过上限时移除最久未使用的条目
            while len(self._preference_text_cache) > self._preference_text_cache_maxsize:
                self._preference_text_cache.popitem(last=False)
            
            # 通知所有等待者
            if not future.done():
                future.set_result(preferences)
            return preferences
        except Exception as e:
            # 通知所有等待者异常
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            # 清理 in-flight 标记
            self._preference_single_flight.pop(flight_key, None)
    
    def invalidate_preference_text_cache(
        self, user_id: Optional[str] = None
    ):
        """
        使偏好文本缓存失效
        
        当用户偏好更新时调用此方法以立即生效
        
        Args:
            user_id: 指定用户则只清除该用户，None 则清除所有
        """
        if user_id:
            self._preference_text_cache.pop(user_id, None)
        else:
            self._preference_text_cache.clear()
    
    # ================================================================
    # 工厂方法
    # ================================================================
    
    @classmethod
    async def from_config(
        cls,
        model_adapter: BaseModelAdapter,
        adapter_config: Optional[Union[str, Dict[str, Any]]] = None,
        adapter_config_path: Optional[str] = None,
        config: Optional[Any] = None,
        language: str = "cn",
        memory_trigger_config: Optional[Union[Dict[str, Any], MemoryTriggerConfig]] = None,
        reference_resolver_config: Optional[Union[Dict[str, Any], ReferenceResolverConfig]] = None,
        enable_redis: Optional[bool] = None,
        redis_config: Optional[Union[Dict[str, Any], RedisConfig]] = None,
    ) -> "DKIPlugin":
        """
        从配置创建 DKI 插件 (推荐方式)
        
        上层应用只需提供配置文件，无需实现任何接口
        
        Args:
            model_adapter: LLM 模型适配器
            adapter_config: 适配器配置字典
            adapter_config_path: 适配器配置文件路径 (YAML)
            config: DKI 配置
            language: 语言
            memory_trigger_config: Memory Trigger 配置 (可选)
            reference_resolver_config: Reference Resolver 配置 (可选)
            enable_redis: 是否启用 Redis (可选)
            redis_config: Redis 配置 (可选)
            
        Returns:
            初始化完成的 DKI 插件
        """
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        import yaml
        
        # 加载全局配置
        config_loader = ConfigLoader()
        global_config = config_loader.config
        
        # 加载原始配置字典
        try:
            with open(config_loader._config_path, 'r', encoding='utf-8') as f:
                _raw_config = yaml.safe_load(f) or {}
        except Exception:
            _raw_config = {}
        
        # 创建配置驱动的适配器
        if adapter_config_path:
            user_adapter = ConfigDrivenAdapter.from_yaml(adapter_config_path)
        elif adapter_config:
            if isinstance(adapter_config, str):
                user_adapter = ConfigDrivenAdapter.from_yaml(adapter_config)
            else:
                user_adapter = ConfigDrivenAdapter.from_dict(adapter_config)
        else:
            import os
            default_paths = [
                "config/adapter_config.yaml",
                "adapter_config.yaml",
                "../config/adapter_config.yaml",
            ]
            for path in default_paths:
                if os.path.exists(path):
                    user_adapter = ConfigDrivenAdapter.from_yaml(path)
                    break
            else:
                raise ValueError(
                    "No adapter configuration provided. "
                    "Please provide adapter_config or adapter_config_path."
                )
        
        # 连接数据库
        await user_adapter.connect()
        
        # 处理 Memory Trigger 配置
        mt_config = None
        if memory_trigger_config:
            if isinstance(memory_trigger_config, dict):
                mt_config = MemoryTriggerConfig(**memory_trigger_config)
            else:
                mt_config = memory_trigger_config
        elif _raw_config.get('memory_trigger'):
            mt_config = MemoryTriggerConfig.from_dict(_raw_config['memory_trigger'])
        
        # 处理 Reference Resolver 配置
        rr_config = None
        if reference_resolver_config:
            if isinstance(reference_resolver_config, dict):
                rr_config = ReferenceResolverConfig(**reference_resolver_config)
            else:
                rr_config = reference_resolver_config
        elif _raw_config.get('reference_resolver'):
            rr_config = ReferenceResolverConfig.from_dict(_raw_config['reference_resolver'])
        
        # ============ 处理 Redis 配置 ============
        redis_client = None
        cache_config = CacheConfig()
        
        if _raw_config.get('preference_cache'):
            cache_config = CacheConfig.from_dict(_raw_config['preference_cache'])
        
        should_enable_redis = enable_redis
        if should_enable_redis is None:
            should_enable_redis = _raw_config.get('redis', {}).get('enabled', False)
        
        if should_enable_redis and REDIS_AVAILABLE:
            if redis_config:
                if isinstance(redis_config, dict):
                    r_config = RedisConfig.from_dict(redis_config)
                else:
                    r_config = redis_config
            elif _raw_config.get('redis'):
                r_config = RedisConfig.from_dict(_raw_config['redis'])
            else:
                r_config = RedisConfig(enabled=True)
            
            r_config.enabled = True
            redis_client = DKIRedisClient(r_config)
            connected = await redis_client.connect()
            
            if connected:
                logger.info("Redis connected for distributed cache")
                cache_config.l2_enabled = True
            else:
                logger.warning("Redis connection failed, falling back to L1 only")
                redis_client = None
        elif should_enable_redis and not REDIS_AVAILABLE:
            logger.warning(
                "Redis requested but redis library not installed. "
                "Install with: pip install redis"
            )
        
        # 创建插件
        plugin = cls(
            model_adapter=model_adapter,
            user_data_adapter=user_adapter,
            config=config,
            language=language,
            memory_trigger_config=mt_config,
            reference_resolver_config=rr_config,
            redis_client=redis_client,
            cache_config=cache_config,
        )
        
        logger.info("DKI Plugin created from configuration")
        return plugin
    
    # ================================================================
    # 核心 chat 方法 (对外接口不变)
    # ================================================================
    
    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_alpha: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        DKI 增强的聊天
        
        v3.0 流程:
        1. Planner.analyze_query() → QueryContext (确定召回范围)
        2. 通过适配器读取数据 (偏好 + 历史)
        3. Planner.build_plan() → InjectionPlan (决策)
        4. Executor.execute() → ExecutionResult (执行)
        5. 记录工作数据 → InjectionMetadata
        
        Args:
            query: 原始用户输入 (不含任何 prompt 构造)
            user_id: 用户标识 (用于读取偏好)
            session_id: 会话标识 (用于读取历史)
            force_alpha: 强制 alpha 值 (可选，跳过门控)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            DKIPluginResponse 包含生成结果和注入元数据
        """
        # 从配置读取 max_new_tokens 默认值 (config.yaml → model.max_new_tokens)
        if max_new_tokens is None:
            max_new_tokens = getattr(
                getattr(self.config, 'model', None), 'max_new_tokens', 2048
            )
        
        start_time = time.time()
        metadata = InjectionMetadata()
        
        try:
            # ============ Step 1: 分析查询 (Planner Phase 1) ============
            context = self._planner.analyze_query(query)
            
            metadata.memory_triggered = context.memory_triggered
            metadata.trigger_type = context.trigger_type
            metadata.reference_resolved = context.reference_resolved
            metadata.reference_type = context.reference_type
            metadata.reference_scope = context.reference_scope
            
            # ============ Step 2: 通过适配器读取外部数据 ============
            adapter_start = time.time()
            
            # P1-3: 使用带 TTL 的偏好文本缓存
            preferences = await self._get_cached_preferences(user_id)
            metadata.preferences_count = len(preferences)
            
            # v7.0: 检测适配器的向量检索能力
            # ConfigDrivenAdapter 会根据 vector_index_config 的有无
            # 自动选择 BM25+Embedding 或 BM25-only
            retrieval_mode = self._detect_retrieval_mode()
            metadata.retrieval_mode = retrieval_mode
            
            # v6.1: 不限制 session_id, 支持跨会话记忆检索
            # 适配器会在所有该用户的会话中检索相关历史
            relevant_history = await self.data_adapter.search_relevant_history(
                user_id=user_id,
                query=query,
                session_id=None,  # 跨会话检索
                limit=context.recall_limit,
            )
            
            # v7.2: 获取近轮对话 (跨会话, 按时间近度)
            # BM25 只做语义相关性召回, 可能遗漏最近的对话内容
            # 近轮对话确保模型能看到最近的上下文, 维持多轮连贯性
            recent_messages = []
            try:
                max_recent = self._get_max_recent_turns()
                recent_messages = await self.data_adapter.get_recent_messages(
                    user_id=user_id,
                    limit=max_recent * 2,  # 每轮 user+assistant
                )
                if recent_messages:
                    logger.debug(
                        f"Recent messages loaded: {len(recent_messages)} "
                        f"(max_recent_turns={max_recent})"
                    )
            except Exception as e:
                logger.warning(f"Failed to get recent messages (non-critical): {e}")
            
            # v7.2: 合并近轮对话 + BM25 召回 (近轮优先, 去重)
            if recent_messages:
                relevant_history = self._merge_recent_and_recalled(
                    recent_messages=recent_messages,
                    recalled_messages=relevant_history,
                )
            
            metadata.relevant_history_count = len(relevant_history)
            
            metadata.adapter_latency_ms = (time.time() - adapter_start) * 1000
            
            logger.debug(
                f"Adapter data loaded: {len(preferences)} preferences, "
                f"{len(relevant_history)} relevant history messages "
                f"(recent={len(recent_messages)}, retrieval_mode={retrieval_mode})"
            )
            
            # ============ Step 3: 构建注入计划 (Planner Phase 2) ============
            plan = self._planner.build_plan(
                query=query,
                user_id=user_id,
                preferences=preferences,
                relevant_history=relevant_history,
                context=context,
                force_alpha=force_alpha,
                session_id=session_id,
                context_window=self._context_window,  # v6.2: 传递实际上下文窗口
            )
            
            # 从 plan 填充 metadata
            metadata.injection_strategy = plan.strategy
            metadata.injection_enabled = plan.injection_enabled
            metadata.alpha = plan.alpha_profile.effective_preference_alpha
            metadata.alpha_profile = plan.alpha_profile.to_dict()
            metadata.preference_tokens = plan.preference_tokens
            metadata.history_tokens = plan.history_tokens
            metadata.query_tokens = plan.query_tokens
            metadata.total_tokens = plan.total_tokens
            metadata.gating_decision = plan.gating_decision
            metadata.safety_violations = plan.safety_violations
            
            # ============ Step 3.5: 模糊指代澄清 (v6.5) ============
            _vague_ref = detect_vague_reference(query)
            if _vague_ref.is_vague:
                # 历史不足时注入澄清指令
                history_insufficient = len(relevant_history) <= 2
                if history_insufficient:
                    plan.clarification_instruction = build_clarification_instruction(
                        _vague_ref.language
                    )
                    logger.info(
                        f"[Clarification] Plugin path: "
                        f"confidence={_vague_ref.confidence:.2f}, "
                        f"pattern='{_vague_ref.matched_pattern}', "
                        f"history_count={len(relevant_history)}"
                    )
            
            # ============ Step 4: 执行注入计划 (Executor) ============
            injection_start = time.time()
            
            # v8.0: fact_retrieve_method 路由
            fact_method = self._executor._get_fact_retrieve_method(plan)
            
            if (fact_method == "entropy_gated"
                    and plan.has_fact_call_instruction
                    and self._fact_retriever):
                # Entropy-Gated: 两阶段生成 (probe + grounding)
                logger.info(
                    "[chat] entropy_gated mode: "
                    "using two-stage generation with entropy monitoring"
                )
                result = await self._executor._execute_entropy_gated(
                    plan=plan,
                    prompt=plan.final_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            elif (fact_method == "inline_intercept"
                    and plan.has_fact_call_instruction
                    and self._fact_retriever):
                # Inline Intercept: stop 拦截
                logger.info(
                    "[chat] inline_intercept mode: "
                    "using stop-token interception"
                )
                result = await self._executor._execute_inline_intercept(
                    plan=plan,
                    prompt=plan.final_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                # post_hoc / native_tool_calls / 无 fact call 指令
                result = await self._executor.execute(
                    plan=plan,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            
            # 从 result 填充 metadata
            metadata.inference_latency_ms = result.inference_latency_ms
            metadata.preference_cache_hit = result.preference_cache_hit
            metadata.preference_cache_tier = result.preference_cache_tier
            
            total_execution_ms = (time.time() - injection_start) * 1000
            metadata.injection_latency_ms = (
                total_execution_ms - result.inference_latency_ms
            )
            
            # ============ Step 5: 记录工作数据 ============
            metadata.latency_ms = (time.time() - start_time) * 1000
            self._record_injection_log(
                metadata=metadata,
                query=query,
                user_id=user_id,
                session_id=session_id,
                final_input=plan.final_input,
                plan=plan,
            )
            
            # v5.7: 移除 <think> 推理内容
            clean_text, _ = strip_think_content(result.text)
            
            return DKIPluginResponse(
                text=clean_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                metadata=metadata,
                raw_output=result.raw_output,
            )
            
        except AdapterConnectionError as e:
            # P0-1 方案A: 暂时性适配器连接错误 → 重试 1 次后降级
            logger.warning(
                f"Adapter connection failed (retryable): {e} "
                f"[code={e.error_code}]"
            )
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.injection_strategy = "adapter_retry_fallback"
            
            # 重试 1 次
            try:
                await asyncio.sleep(0.5)
                return await self._fallback_without_adapter(
                    query=query,
                    user_id=user_id,
                    metadata=metadata,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    start_time=start_time,
                    **kwargs,
                )
            except Exception as retry_err:
                logger.error(f"Adapter retry fallback failed: {retry_err}")
                return await self._fallback_no_injection(
                    query=query,
                    metadata=metadata,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    start_time=start_time,
                    **kwargs,
                )
        
        except AdapterSchemaError as e:
            # P0-1 方案A: 永久性 Schema 错误 → 直接降级 + 告警
            logger.error(
                f"Adapter schema error (permanent): {e} "
                f"[code={e.error_code}]"
            )
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.injection_strategy = "schema_error_fallback"
            return await self._fallback_no_injection(
                query=query,
                metadata=metadata,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                start_time=start_time,
                **kwargs,
            )
        
        except (KVOOMError, ModelOOMError) as e:
            # P0-1 方案A: GPU OOM → 清理缓存后降级
            logger.critical(
                f"GPU OOM during DKI: {e} [code={e.error_code}]"
            )
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared after OOM")
            except Exception:
                pass
            
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.injection_strategy = "oom_fallback"
            return await self._fallback_no_injection(
                query=query,
                metadata=metadata,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                start_time=start_time,
                **kwargs,
            )
        
        except DKIError as e:
            # P0-1 方案A: 其他 DKI 结构化异常
            logger.error(
                f"DKI error: {e} [code={e.error_code}, "
                f"retryable={e.retryable}]"
            )
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.injection_strategy = "dki_error_fallback"
            return await self._fallback_stable_then_none(
                query=query,
                user_id=user_id,
                metadata=metadata,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                start_time=start_time,
                **kwargs,
            )
        
        except Exception as e:
            # 未分类异常: 保持原有三级降级逻辑
            logger.error(f"DKI Plugin unexpected error: {e}")
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.injection_strategy = "stable_fallback"
            return await self._fallback_stable_then_none(
                query=query,
                user_id=user_id,
                metadata=metadata,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                start_time=start_time,
                **kwargs,
            )
    
    # ================================================================
    # P0-1: 分级降级辅助方法
    # ================================================================
    
    async def _fallback_without_adapter(
        self,
        query: str,
        user_id: str,
        metadata: InjectionMetadata,
        max_new_tokens: int,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        降级: 不使用适配器数据, 仅偏好 K/V + 原始查询
        
        适用于适配器暂时性故障 (连接超时等)
        """
        try:
            stable_plan = InjectionPlan(
                strategy="stable",
                original_query=query,
                final_input=query,
                user_id=user_id,
                injection_enabled=False,
            )
            
            result = await self._executor.execute(
                plan=stable_plan,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            clean_text, _ = strip_think_content(result.text)
            metadata.latency_ms = (time.time() - start_time) * 1000
            
            return DKIPluginResponse(
                text=clean_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                metadata=metadata,
                raw_output=result.raw_output,
            )
        except Exception as e:
            logger.error(f"Fallback without adapter failed: {e}")
            return await self._fallback_no_injection(
                query=query,
                metadata=metadata,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                start_time=start_time,
                **kwargs,
            )
    
    async def _fallback_no_injection(
        self,
        query: str,
        metadata: InjectionMetadata,
        max_new_tokens: int,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        最终降级: 直接调用 LLM, 无任何注入
        
        适用于所有其他降级失败的情况
        """
        try:
            if hasattr(self.model, 'async_generate'):
                output = await self.model.async_generate(
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                output = self.model.generate(
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            
            clean_text, _ = strip_think_content(output.text)
            metadata.injection_strategy = "none_fallback"
            metadata.latency_ms = (time.time() - start_time) * 1000
            
            return DKIPluginResponse(
                text=clean_text,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata=metadata,
            )
        except Exception as fallback_error:
            logger.error(f"Final fallback generation failed: {fallback_error}")
            raise
    
    async def _fallback_stable_then_none(
        self,
        query: str,
        user_id: str,
        metadata: InjectionMetadata,
        max_new_tokens: int,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        两级降级: stable → 无注入
        
        1. 尝试 stable 策略 (偏好 K/V + 原始查询)
        2. 失败则直接调用 LLM (无注入)
        """
        # 第一级: stable 策略
        try:
            logger.info("Falling back to stable strategy in DKI Plugin")
            stable_plan = InjectionPlan(
                strategy="stable",
                original_query=query,
                final_input=query,
                user_id=user_id,
                injection_enabled=False,
            )
            
            try:
                preferences = await self.data_adapter.get_user_preferences(user_id)
                if preferences:
                    pref_text = "\n".join(
                        f"- {p.preference_type}: {p.preference_text}"
                        for p in sorted(
                            preferences, key=lambda x: x.priority, reverse=True
                        )
                        if not p.is_expired()
                    )
                    stable_plan.preference_text = pref_text
                    stable_plan.preferences_count = len(preferences)
                    stable_plan.injection_enabled = True
                    stable_plan.alpha_profile = AlphaProfile(
                        preference_alpha=0.4, history_alpha=1.0
                    )
            except Exception:
                pass  # 偏好加载失败, 继续无注入推理
            
            result = await self._executor.execute(
                plan=stable_plan,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            clean_text, _ = strip_think_content(result.text)
            metadata.latency_ms = (time.time() - start_time) * 1000
            
            return DKIPluginResponse(
                text=clean_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                metadata=metadata,
                raw_output=result.raw_output,
            )
        except Exception as stable_error:
            logger.error(f"Stable fallback failed: {stable_error}")
        
        # 第二级: 无注入
        return await self._fallback_no_injection(
            query=query,
            metadata=metadata,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            start_time=start_time,
            **kwargs,
        )
    
    # ================================================================
    # 流式生成 (Streaming)
    # ================================================================
    
    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_alpha: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        DKI 增强的流式聊天
        
        与 chat() 相同的 Planner → Adapter → Executor 流程,
        但在推理阶段使用流式生成, 逐 token 返回。
        
        Yields:
            字典, 包含:
            - type: "token" | "metadata" | "done" | "error"
            - content: token 文本 (type=token 时)
            - metadata: InjectionMetadata (type=metadata 时)
            - text: 完整文本 (type=done 时)
        """
        if max_new_tokens is None:
            max_new_tokens = getattr(
                getattr(self.config, 'model', None), 'max_new_tokens', 2048
            )
        
        start_time = time.time()
        metadata = InjectionMetadata()
        
        try:
            # ============ Step 1-3: 与 chat() 相同的 Planner 流程 ============
            context = self._planner.analyze_query(query)
            metadata.memory_triggered = context.memory_triggered
            metadata.trigger_type = context.trigger_type
            metadata.reference_resolved = context.reference_resolved
            metadata.reference_type = context.reference_type
            metadata.reference_scope = context.reference_scope
            
            adapter_start = time.time()
            preferences = await self._get_cached_preferences(user_id)
            metadata.preferences_count = len(preferences)
            
            retrieval_mode = self._detect_retrieval_mode()
            metadata.retrieval_mode = retrieval_mode
            
            relevant_history = await self.data_adapter.search_relevant_history(
                user_id=user_id,
                query=query,
                session_id=None,
                limit=context.recall_limit,
            )
            
            recent_messages = []
            try:
                max_recent = self._get_max_recent_turns()
                recent_messages = await self.data_adapter.get_recent_messages(
                    user_id=user_id,
                    limit=max_recent * 2,
                )
            except Exception:
                pass
            
            if recent_messages:
                relevant_history = self._merge_recent_and_recalled(
                    recent_messages=recent_messages,
                    recalled_messages=relevant_history,
                )
            
            metadata.relevant_history_count = len(relevant_history)
            metadata.adapter_latency_ms = (time.time() - adapter_start) * 1000
            
            plan = self._planner.build_plan(
                query=query,
                user_id=user_id,
                preferences=preferences,
                relevant_history=relevant_history,
                context=context,
                force_alpha=force_alpha,
                session_id=session_id,
                context_window=self._context_window,
            )
            
            metadata.injection_strategy = plan.strategy
            metadata.injection_enabled = plan.injection_enabled
            metadata.alpha = plan.alpha_profile.effective_preference_alpha
            metadata.alpha_profile = plan.alpha_profile.to_dict()
            metadata.preference_tokens = plan.preference_tokens
            metadata.history_tokens = plan.history_tokens
            metadata.query_tokens = plan.query_tokens
            metadata.total_tokens = plan.total_tokens
            
            # 先发送 metadata
            yield {
                "type": "metadata",
                "metadata": metadata.to_dict(),
            }
            
            # ============ Step 4: 流式执行 ============
            # ---- v8.0: entropy_gated / inline_intercept 路由 ----
            # 两者均不支持真正的流式, 回退到非流式后模拟流式输出
            fact_method = self._executor._get_fact_retrieve_method(plan)
            if (fact_method in ("entropy_gated", "inline_intercept")
                    and plan.has_fact_call_instruction
                    and self._fact_retriever):
                logger.info(
                    f"[chat_stream] {fact_method} mode: "
                    "falling back to non-streaming execution"
                )
                if fact_method == "entropy_gated":
                    result = await self._executor._execute_entropy_gated(
                        plan=plan,
                        prompt=plan.final_input,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                else:
                    result = await self._executor._execute_inline_intercept(
                        plan=plan,
                        prompt=plan.final_input,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                clean_text, _ = strip_think_content(result.text)
                metadata.latency_ms = (time.time() - start_time) * 1000
                
                # 模拟流式输出
                yield {"type": "token", "content": clean_text}
                yield {
                    "type": "done",
                    "text": clean_text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "metadata": metadata.to_dict(),
                }
                return
            
            # 检查模型是否支持流式生成
            has_stream = (
                hasattr(self.model, 'stream_generate')
                or hasattr(self.model, 'async_stream_generate')
            )
            
            if has_stream:
                # 使用模型的流式生成
                # v5.9: 使用 StreamThinkDetector 检测思考内容
                #   - 思考内容通过 type="thinking" 事件发送
                #   - 正常内容通过 type="token" 事件发送
                #   - 客户端根据 show_thinking 配置决定是否显示
                detector = create_stream_detector(buffer_max_chars=200)
                show_thinking = get_show_thinking()
                full_text = ""
                input_tokens = 0
                output_tokens = 0
                
                prompt = plan.final_input
                if not prompt:
                    prompt = query
                
                if hasattr(self.model, 'async_stream_generate'):
                    stream = self.model.async_stream_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                    async for chunk in stream:
                        token_text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                        # v5.9: 通过 StreamThinkDetector 检测
                        for evt_type, evt_text in detector.feed(token_text):
                            if evt_type == "token":
                                yield {"type": "token", "content": evt_text}
                            elif evt_type == "thinking" and show_thinking:
                                yield {"type": "thinking", "content": evt_text}
                elif hasattr(self.model, 'stream_generate'):
                    for chunk in self.model.stream_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    ):
                        token_text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                        # v5.9: 通过 StreamThinkDetector 检测
                        for evt_type, evt_text in detector.feed(token_text):
                            if evt_type == "token":
                                yield {"type": "token", "content": evt_text}
                            elif evt_type == "thinking" and show_thinking:
                                yield {"type": "thinking", "content": evt_text}
                
                # v5.9: 释放缓冲区残留内容
                for evt_type, evt_text in detector.flush():
                    if evt_type == "token":
                        yield {"type": "token", "content": evt_text}
                    elif evt_type == "thinking" and show_thinking:
                        yield {"type": "thinking", "content": evt_text}
                
                # v5.9: 最终使用全量正则清理 (确保存储的文本干净)
                clean_text = detector.get_clean_text()
                metadata.latency_ms = (time.time() - start_time) * 1000
                
                # 流式模式下估算 token (修复 token 统计为 0 的问题)
                input_tokens = estimate_tokens_fast(prompt, overestimate_factor=1.15)
                output_tokens = estimate_tokens_fast(clean_text, overestimate_factor=1.15)
                
                yield {
                    "type": "done",
                    "text": clean_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "metadata": metadata.to_dict(),
                }
            else:
                # 模型不支持流式: 回退到非流式, 一次性返回
                result = await self._executor.execute(
                    plan=plan,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                
                clean_text, _ = strip_think_content(result.text)
                metadata.latency_ms = (time.time() - start_time) * 1000
                
                # 模拟流式: 逐句发送
                yield {"type": "token", "content": clean_text}
                yield {
                    "type": "done",
                    "text": clean_text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "metadata": metadata.to_dict(),
                }
        
        except Exception as e:
            logger.error(f"DKI stream error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_code": getattr(e, 'error_code', 'UNKNOWN'),
            }
    
    # ================================================================
    # 日志与监控
    # ================================================================
    
    def _record_injection_log(
        self,
        metadata: InjectionMetadata,
        query: str = "",
        user_id: str = "",
        session_id: str = "",
        final_input: str = "",
        plan: Optional[InjectionPlan] = None,
    ):
        """记录注入日志 (用于监控和可视化)"""
        self._injection_logs.append(metadata)
        
        # 限制日志数量
        if len(self._injection_logs) > self._max_logs:
            self._injection_logs = self._injection_logs[-self._max_logs:]
        
        # 更新统计
        self._stats["total_requests"] += 1
        if metadata.injection_enabled:
            self._stats["injection_enabled_count"] += 1
        if metadata.preference_cache_hit:
            self._stats["cache_hits"] += 1
        self._stats["total_latency_ms"] += metadata.latency_ms
        
        # 计算平均 alpha
        if self._stats["injection_enabled_count"] > 0:
            total_alpha = sum(
                log.alpha for log in self._injection_logs
                if log.injection_enabled
            )
            self._stats["avg_alpha"] = (
                total_alpha / self._stats["injection_enabled_count"]
            )
        
        # v6.2: 从 plan 提取可视化所需的明文信息
        preference_text = ""
        history_suffix_text = ""
        history_messages = []
        recall_v4_info = {}
        
        if plan:
            preference_text = plan.preference_text or ""
            history_suffix_text = plan.assembled_suffix or plan.history_suffix or ""
            
            # 从 plan.history_items 构造历史消息列表 (用于 UI 显示)
            for item in (plan.history_items or []):
                role = getattr(item, 'role', 'user') or 'user'
                content = getattr(item, 'content', str(item))
                if content and content.strip():
                    history_messages.append({"role": role, "content": content})
            
            # Recall v4 元数据
            if plan.strategy == "recall_v4":
                recall_v4_info = {
                    "enabled": True,
                    "strategy": plan.recall_strategy,
                    "trace_ids": plan.trace_ids or [],
                    "fact_rounds_used": plan.fact_rounds_used,
                    "summary_count": plan.summary_count,
                    "message_count": plan.message_count,
                }
        
        # v9.0: 写入 InjectionMetadata 正式字段 (供上层应用 / 实验系统使用)
        metadata.preference_text = preference_text or None
        metadata.history_suffix_text = history_suffix_text or None
        metadata.history_messages = history_messages if history_messages else None
        metadata.final_input = final_input or None
        
        # 记录可视化数据
        try:
            record_visualization({
                "request_id": metadata.request_id,
                "timestamp": metadata.timestamp.isoformat(),
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "mode": "dki",
                "injection_enabled": metadata.injection_enabled,
                "alpha": metadata.alpha,
                "alpha_profile": metadata.alpha_profile,
                "preference_tokens": metadata.preference_tokens,
                "history_tokens": metadata.history_tokens,
                "query_tokens": metadata.query_tokens,
                "total_tokens": metadata.total_tokens,
                "cache_hit": metadata.preference_cache_hit,
                "cache_tier": metadata.preference_cache_tier,
                "latency_ms": metadata.latency_ms,
                "adapter_latency_ms": metadata.adapter_latency_ms,
                "injection_latency_ms": metadata.injection_latency_ms,
                "inference_latency_ms": metadata.inference_latency_ms,
                "preferences_count": metadata.preferences_count,
                "relevant_history_count": metadata.relevant_history_count,
                "memory_triggered": metadata.memory_triggered,
                "trigger_type": metadata.trigger_type,
                "reference_resolved": metadata.reference_resolved,
                "reference_type": metadata.reference_type,
                "safety_violations": metadata.safety_violations,
                "final_input": final_input,
                # v6.2: 注入明文信息 (用于可视化完整显示)
                "preference_text": preference_text,
                "history_suffix_text": history_suffix_text,
                "history_messages": history_messages,
                # v6.2: Recall v4 元数据
                "recall_v4": recall_v4_info,
                "recall_strategy": recall_v4_info.get("strategy", ""),
                "trace_ids": recall_v4_info.get("trace_ids", []),
                "fact_rounds_used": recall_v4_info.get("fact_rounds_used", 0),
                "summary_count": recall_v4_info.get("summary_count", 0),
                "message_count": recall_v4_info.get("message_count", 0),
            })
        except Exception as e:
            logger.debug(f"Failed to record visualization: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据 (用于监控 API)"""
        cache_stats = self._preference_cache.get_stats()
        planner_stats = self._planner.get_stats()
        executor_stats = self._executor.get_stats()
        
        return {
            "total_requests": self._stats["total_requests"],
            "injection_enabled_count": self._stats["injection_enabled_count"],
            "injection_rate": (
                self._stats["injection_enabled_count"]
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"]
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "avg_latency_ms": (
                self._stats["total_latency_ms"]
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "avg_alpha": self._stats["avg_alpha"],
            # Planner 统计 (含 memory_trigger + reference_resolver)
            "planner": planner_stats,
            # Executor 统计
            "executor": executor_stats,
            # 缓存统计 (包含 Redis)
            "cache": cache_stats,
            # 向后兼容: 扁平化旧字段
            "memory_trigger_count": planner_stats.get(
                "memory_trigger_count", 0
            ),
            "memory_trigger_rate": (
                planner_stats.get("memory_trigger_count", 0)
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "reference_resolved_count": planner_stats.get(
                "reference_resolved_count", 0
            ),
            "reference_resolved_rate": (
                planner_stats.get("reference_resolved_count", 0)
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "memory_trigger_config": planner_stats.get(
                "memory_trigger", {}
            ),
            "reference_resolver_config": planner_stats.get(
                "reference_resolver", {}
            ),
        }
    
    def get_injection_logs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取注入日志 (用于监控 API)"""
        logs = self._injection_logs[-(limit + offset):]
        if offset > 0:
            logs = logs[:-offset]
        return [log.to_dict() for log in logs[-limit:]]
    
    # ================================================================
    # 缓存管理
    # ================================================================
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """清除偏好缓存"""
        self._executor.clear_preference_cache(user_id)
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """
        使用户缓存失效
        
        当用户偏好更新时调用此方法:
        - 使偏好文本缓存失效 (P1-3)
        - 使偏好 KV 缓存失效
        """
        # P1-3: 同时清除偏好文本缓存
        self.invalidate_preference_text_cache(user_id)
        return await self._preference_cache.invalidate(user_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._preference_cache.get_stats()
    
    # ================================================================
    # 组件配置更新 (委托给 Planner / Executor)
    # ================================================================
    
    def update_reference_resolver_config(
        self,
        just_now_turns: Optional[int] = None,
        recently_turns: Optional[int] = None,
        last_topic_turns: Optional[int] = None,
        assistant_stance_turns: Optional[int] = None,
    ):
        """运行时更新 Reference Resolver 配置"""
        self._planner.update_reference_resolver_config(
            just_now_turns=just_now_turns,
            recently_turns=recently_turns,
            last_topic_turns=last_topic_turns,
            assistant_stance_turns=assistant_stance_turns,
        )
        logger.info(
            f"Reference Resolver config updated: "
            f"{self._planner.get_stats().get('reference_resolver', {})}"
        )
    
    def update_memory_trigger_config(
        self,
        enabled: Optional[bool] = None,
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
    ):
        """运行时更新 Memory Trigger 配置"""
        self._planner.update_memory_trigger_config(
            enabled=enabled,
            custom_patterns=custom_patterns,
        )
        logger.info(
            f"Memory Trigger config updated: "
            f"{self._planner.get_stats().get('memory_trigger', {})}"
        )
    
    def get_component_configs(self) -> Dict[str, Any]:
        """获取所有组件的当前配置"""
        planner_stats = self._planner.get_stats()
        configs = {
            "memory_trigger": planner_stats.get("memory_trigger", {}),
            "reference_resolver": planner_stats.get("reference_resolver", {}),
            "injection_strategy": planner_stats.get("strategy", "recall_v4"),
        }
        
        return configs
    
    # ================================================================
    # 生命周期
    # ================================================================
    
    async def close(self):
        """
        关闭 DKI 插件
        
        清理资源，包括:
        - 关闭 Redis 连接
        - 关闭数据库连接
        """
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
        
        if hasattr(self.data_adapter, 'disconnect'):
            await self.data_adapter.disconnect()
            logger.info("Database connection closed")
        elif hasattr(self.data_adapter, 'close'):
            await self.data_adapter.close()
            logger.info("Database connection closed")
        
        logger.info("DKI Plugin closed")
