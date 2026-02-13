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

Author: AGI Demo Project
Version: 2.0.0
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import torch
from loguru import logger

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
from dki.core.injection import (
    FullAttentionInjector,
    FullAttentionConfig,
    InjectionResult as FullAttentionResult,
)


@dataclass
class InjectionMetadata:
    """DKI 注入元数据 (用于监控)"""
    # 注入状态
    injection_enabled: bool = False
    alpha: float = 0.0
    
    # 注入策略 (stable | full_attention)
    injection_strategy: str = "stable"
    
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
    
    # Full Attention 策略特有信息
    full_attention_fallback: bool = False
    history_kv_tokens: int = 0
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "injection_enabled": self.injection_enabled,
            "injection_strategy": self.injection_strategy,
            "alpha": self.alpha,
            "tokens": {
                "preference": self.preference_tokens,
                "history": self.history_tokens,
                "history_kv": self.history_kv_tokens,  # Full Attention 特有
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
            "full_attention": {
                "fallback_triggered": self.full_attention_fallback,
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
    DKI 插件核心
    
    作为 LLM 的插件，接收原始用户输入，通过配置驱动的适配器读取外部数据，
    执行 K/V 注入后调用 LLM 推理。
    
    上层应用集成方式:
    1. 提供适配器配置文件 (指定数据库连接和字段映射)
    2. 删除 RAG/Prompt 工程代码
    3. 传递 user_id + 原始用户输入给 DKI
    
    使用方式 1: 从配置文件创建 (推荐)
    ```python
    # 上层应用只需提供配置文件，无需实现任何接口
    dki = await DKIPlugin.from_config(
        model_adapter=vllm_adapter,
        adapter_config_path="config/adapter_config.yaml",
    )
    
    # 调用时只需传递 user_id 和原始输入
    response = await dki.chat(
        query="推荐一家餐厅",  # 原始用户输入，无需任何 prompt 构造
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
    """
    
    # 历史后缀提示词模板
    HISTORY_PREFIX_EN = """
[Session History Reference]
Before responding, please refer to the following session history.
These are real conversation records between you and the user, and are trustworthy.
---
"""
    
    HISTORY_SUFFIX_EN = """
---
[End of Session History]
Please respond based on the above history and the user's current question.
"""
    
    HISTORY_PREFIX_CN = """
[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
---
"""
    
    HISTORY_SUFFIX_CN = """
---
[会话历史结束]
请基于以上历史和用户当前问题给出回复。
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
        
        # DKI 组件 (延迟初始化)
        self._mis: Optional[MemoryInfluenceScaling] = None
        self._gating: Optional[DualFactorGating] = None
        
        # Memory Trigger (可配置)
        self._memory_trigger = MemoryTrigger(
            config=memory_trigger_config,
            language="auto",
        )
        
        # Reference Resolver (召回轮数可配置)
        self._reference_resolver = ReferenceResolver(
            config=reference_resolver_config,
            language="auto",
        )
        
        # 短期历史缓存 (用于指代解析)
        self._short_term_history: Dict[str, List[ResolverMessage]] = {}
        self._max_history_per_session = 50
        
        # ============ 注入策略选择 ============
        # stable: 偏好 K/V + 历史 Suffix Prompt (默认，生产推荐)
        # full_attention: 偏好 K/V + 历史 K/V (研究，方案 C)
        self._injection_strategy = self._get_injection_strategy(config)
        
        # Full Attention 注入器 (仅在 full_attention 策略时使用)
        self._full_attention_injector: Optional[FullAttentionInjector] = None
        if self._injection_strategy == "full_attention":
            fa_config = self._get_full_attention_config(config)
            self._full_attention_injector = FullAttentionInjector(
                config=fa_config,
                language=language,
            )
        
        # ============ 偏好 K/V 缓存 (支持 Redis 分布式) ============
        # 重要: 多实例部署时启用 Redis 可保持缓存命中率
        # - 没有 Redis: 缓存命中率 = 70%/N (N = 实例数)
        # - 有 Redis: 缓存命中率 = 70% (恒定)
        self._redis_client = redis_client
        self._cache_config = cache_config or CacheConfig()
        self._preference_cache = PreferenceCacheManager(
            redis_client=redis_client,
            config=self._cache_config,
        )
        
        # 偏好 K/V 本地缓存 (用于 _get_preference_kv 的快速查找)
        self._preference_kv_cache: Dict[str, Tuple[Any, str]] = {}
        
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
            "memory_trigger_count": 0,
            "reference_resolved_count": 0,
        }
        
        # 日志输出缓存状态
        cache_status = "L1 only"
        if redis_client and redis_client.is_available:
            cache_status = "L1 + L2 (Redis)"
        elif self._cache_config.l2_enabled:
            cache_status = "L1 + L2 (Redis not connected)"
        
        logger.info(
            f"DKI Plugin initialized "
            f"(strategy={self._injection_strategy}, language={language}, cache={cache_status})"
        )
    
    def _get_injection_strategy(self, config: Any) -> str:
        """获取注入策略"""
        if config and hasattr(config, 'dki'):
            return getattr(config.dki, 'injection_strategy', 'stable')
        if config and isinstance(config, dict):
            return config.get('dki', {}).get('injection_strategy', 'stable')
        return 'stable'
    
    def _get_full_attention_config(self, config: Any) -> FullAttentionConfig:
        """获取 Full Attention 配置"""
        fa_dict = {}
        
        if config and hasattr(config, 'dki') and hasattr(config.dki, 'full_attention'):
            fa_dict = config.dki.full_attention
        elif config and isinstance(config, dict):
            fa_dict = config.get('dki', {}).get('full_attention', {})
        
        if fa_dict:
            return FullAttentionConfig.from_dict(fa_dict)
        return FullAttentionConfig()
    
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
            reference_resolver_config: Reference Resolver 配置 (可选，支持外置召回轮数)
            enable_redis: 是否启用 Redis (可选，覆盖配置文件)
            redis_config: Redis 配置 (可选)
            
        Returns:
            初始化完成的 DKI 插件
            
        Example:
            ```python
            # 从配置文件创建
            dki = await DKIPlugin.from_config(
                model_adapter=vllm_adapter,
                adapter_config_path="config/adapter_config.yaml",
            )
            
            # 从配置字典创建，带自定义 Reference Resolver 配置
            dki = await DKIPlugin.from_config(
                model_adapter=vllm_adapter,
                adapter_config={
                    "database": {"type": "postgresql", ...},
                    "preferences": {"table": "user_preferences", ...},
                    "messages": {"table": "chat_messages", ...},
                },
                reference_resolver_config={
                    "just_now_turns": 5,    # "刚刚" 召回 5 轮
                    "recently_turns": 20,   # "最近" 召回 20 轮
                    "last_topic_turns": 15, # "那件事" 召回 15 轮
                },
            )
            
            # 启用 Redis 分布式缓存 (多实例部署推荐)
            dki = await DKIPlugin.from_config(
                model_adapter=vllm_adapter,
                adapter_config_path="config/adapter_config.yaml",
                enable_redis=True,
                redis_config={"host": "redis.example.com", "port": 6379},
            )
            ```
        """
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        import yaml
        
        # 加载全局配置 (原始 YAML dict，用于访问非 pydantic 模型的扩展配置节)
        config_loader = ConfigLoader()
        global_config = config_loader.config
        
        # 加载原始配置字典 (memory_trigger, redis 等扩展配置不在 pydantic Config 中)
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
            # 尝试从默认路径加载
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
        
        # 处理 Reference Resolver 配置 (支持外置召回轮数)
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
        
        # 从全局配置加载缓存配置
        if _raw_config.get('preference_cache'):
            cache_config = CacheConfig.from_dict(_raw_config['preference_cache'])
        
        # 确定是否启用 Redis
        should_enable_redis = enable_redis
        if should_enable_redis is None:
            # 从配置文件读取
            should_enable_redis = _raw_config.get('redis', {}).get('enabled', False)
        
        if should_enable_redis and REDIS_AVAILABLE:
            # 创建 Redis 配置
            if redis_config:
                if isinstance(redis_config, dict):
                    r_config = RedisConfig.from_dict(redis_config)
                else:
                    r_config = redis_config
            elif _raw_config.get('redis'):
                r_config = RedisConfig.from_dict(_raw_config['redis'])
            else:
                r_config = RedisConfig(enabled=True)
            
            # 确保启用
            r_config.enabled = True
            
            # 创建并连接 Redis 客户端
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
    
    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_alpha: Optional[float] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        DKI 增强的聊天
        
        核心流程:
        1. 通过适配器读取上层应用的数据库 (用户偏好 + 历史消息)
        2. 执行 DKI 注入 (偏好 K/V 负位置 + 历史后缀正位置)
        3. 调用 LLM 推理
        4. 记录工作数据供监控
        
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
        start_time = time.time()
        metadata = InjectionMetadata()
        
        try:
            # ============ Step 0: Memory Trigger & Reference Resolver ============
            # 检测是否触发记忆存储/更新
            trigger_result = self._memory_trigger.detect(query)
            metadata.memory_triggered = trigger_result.triggered
            metadata.trigger_type = trigger_result.trigger_type.value if trigger_result.triggered else None
            
            if trigger_result.triggered:
                self._stats["memory_trigger_count"] += 1
                logger.debug(
                    f"Memory trigger detected: {trigger_result.trigger_type.value}, "
                    f"confidence={trigger_result.confidence:.2f}"
                )
            
            # 解析指代表达，确定历史召回范围
            reference_result = self._reference_resolver.resolve(query)
            metadata.reference_resolved = reference_result.reference_type != ReferenceType.NONE
            metadata.reference_type = reference_result.reference_type.value
            metadata.reference_scope = reference_result.scope.value if reference_result.scope else None
            
            # 根据指代类型确定召回轮数
            recall_limit = 10  # 默认
            if reference_result.reference_type != ReferenceType.NONE:
                self._stats["reference_resolved_count"] += 1
                recall_limit = reference_result.recall_turns or recall_limit
                logger.debug(
                    f"Reference resolved: {reference_result.reference_type.value}, "
                    f"recall_turns={recall_limit}"
                )
            
            # ============ Step 1: 通过适配器读取外部数据 ============
            adapter_start = time.time()
            
            # 读取用户偏好 (从上层应用的数据库)
            preferences = await self.data_adapter.get_user_preferences(user_id)
            metadata.preferences_count = len(preferences)
            
            # 读取相关历史 (从上层应用的数据库)
            # 使用 Reference Resolver 确定的召回轮数
            relevant_history = await self.data_adapter.search_relevant_history(
                user_id=user_id,
                query=query,
                session_id=session_id,
                limit=recall_limit,
            )
            metadata.relevant_history_count = len(relevant_history)
            
            metadata.adapter_latency_ms = (time.time() - adapter_start) * 1000
            
            logger.debug(
                f"Adapter data loaded: {len(preferences)} preferences, "
                f"{len(relevant_history)} relevant history messages"
            )
            
            # ============ Step 2: DKI 注入处理 (策略分支) ============
            injection_start = time.time()
            metadata.injection_strategy = self._injection_strategy
            
            # 根据策略选择注入方式
            if self._injection_strategy == "full_attention" and self._full_attention_injector:
                # ============ Full Attention 策略 (方案 C) ============
                # 偏好 + 历史均通过 K/V 注入
                output, metadata = await self._inject_full_attention(
                    query=query,
                    user_id=user_id,
                    preferences=preferences,
                    relevant_history=relevant_history,
                    metadata=metadata,
                    force_alpha=force_alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                # ============ Stable 策略 (默认) ============
                # 偏好 K/V + 历史 Suffix Prompt
                output, metadata = await self._inject_stable(
                    query=query,
                    user_id=user_id,
                    preferences=preferences,
                    relevant_history=relevant_history,
                    metadata=metadata,
                    force_alpha=force_alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            
            # injection_latency_ms 应排除 inference 时间
            total_inject_ms = (time.time() - injection_start) * 1000
            metadata.injection_latency_ms = total_inject_ms - metadata.inference_latency_ms
            
            # ============ Step 3: 记录工作数据 ============
            metadata.latency_ms = (time.time() - start_time) * 1000
            self._record_injection_log(
                metadata=metadata,
                query=query,
                user_id=user_id,
                session_id=session_id,
                final_input=query,  # 记录原始查询
            )
            
            return DKIPluginResponse(
                text=output.text,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata=metadata,
                raw_output=output,
            )
            
        except Exception as e:
            logger.error(f"DKI Plugin error: {e}")
            metadata.latency_ms = (time.time() - start_time) * 1000
            
            # 降级: 直接调用 LLM (无注入)
            try:
                output = self.model.generate(
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return DKIPluginResponse(
                    text=output.text,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    metadata=metadata,
                )
            except Exception as fallback_error:
                logger.error(f"Fallback generation failed: {fallback_error}")
                raise
    
    async def _inject_stable(
        self,
        query: str,
        user_id: str,
        preferences: List[AdapterUserPreference],
        relevant_history: List[AdapterChatMessage],
        metadata: InjectionMetadata,
        force_alpha: Optional[float],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[ModelOutput, InjectionMetadata]:
        """
        Stable 策略注入 (默认)
        
        偏好: K/V 注入 (负位置)
        历史: Suffix Prompt (正位置，占用 Context)
        """
        # 2.1 处理用户偏好 (K/V 注入，负位置)
        preference_kv = None
        preference_alpha = 0.0
        
        if preferences:
            preference_text = self._format_preferences(preferences)
            preference_kv, cache_hit, cache_tier = await self._get_preference_kv(
                user_id=user_id,
                preference_text=preference_text,
            )
            metadata.preference_cache_hit = cache_hit
            metadata.preference_cache_tier = cache_tier
            metadata.preference_tokens = self._estimate_tokens(preference_text)
            
            # 偏好使用较低的 alpha (0.3-0.5)
            preference_alpha = self.config.dki.hybrid_injection.preference.alpha \
                if hasattr(self.config.dki, 'hybrid_injection') else 0.4
        
        # 2.2 处理历史消息 (后缀提示词，正位置)
        history_suffix = ""
        if relevant_history:
            history_suffix = self._format_history_suffix(relevant_history)
            metadata.history_tokens = self._estimate_tokens(history_suffix)
        
        # 2.3 构造最终输入
        # 正确的顺序: 历史后缀 → 用户查询
        # 这样模型先看到历史上下文，再处理当前问题
        # 注入 = 偏好 K/V (负位置)
        final_input = query
        if history_suffix:
            final_input = history_suffix + "\n\n" + query
        
        metadata.query_tokens = self._estimate_tokens(query)
        metadata.total_tokens = metadata.query_tokens + metadata.history_tokens
        
        # 2.4 门控决策
        if force_alpha is not None:
            alpha = force_alpha
            metadata.injection_enabled = alpha > 0.1
        else:
            # 基于偏好和历史决定是否注入
            should_inject = len(preferences) > 0 or len(relevant_history) > 0
            alpha = preference_alpha if should_inject else 0.0
            metadata.injection_enabled = should_inject
        
        metadata.alpha = alpha
        metadata.gating_decision = {
            "should_inject": metadata.injection_enabled,
            "alpha": alpha,
            "preference_alpha": preference_alpha,
            "has_preferences": len(preferences) > 0,
            "has_history": len(relevant_history) > 0,
            "strategy": "stable",
        }
        
        # 2.5 LLM 推理
        inference_start = time.time()
        
        if metadata.injection_enabled and preference_kv:
            # 带 K/V 注入的推理
            output = self.model.forward_with_kv_injection(
                prompt=final_input,
                injected_kv=preference_kv,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            # 普通推理 (无注入或 alpha 太低)
            output = self.model.generate(
                prompt=final_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        metadata.inference_latency_ms = (time.time() - inference_start) * 1000
        
        return output, metadata
    
    async def _inject_full_attention(
        self,
        query: str,
        user_id: str,
        preferences: List[AdapterUserPreference],
        relevant_history: List[AdapterChatMessage],
        metadata: InjectionMetadata,
        force_alpha: Optional[float],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[ModelOutput, InjectionMetadata]:
        """
        Full Attention 策略注入 (研究 - 方案 C)
        
        偏好: K/V 注入 (负位置)
        历史: K/V 注入 (负位置，NEW!)
        全局指示: 极简提示 (约 3-5 tokens)
        
        目标: 0% Context 占用
        """
        # 准备数据
        preference_text = self._format_preferences(preferences) if preferences else ""
        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in relevant_history
        ] if relevant_history else []
        
        # 调用 Full Attention 注入器
        fa_result = self._full_attention_injector.inject(
            model_adapter=self.model,
            preference_text=preference_text,
            history_messages=history_messages,
            query=query,
        )
        
        # 检查是否需要 fallback 到 Stable 策略
        if not fa_result.success or fa_result.fallback_triggered:
            metadata.full_attention_fallback = True
            logger.info(
                f"Full attention fallback to stable: {fa_result.error_message}"
            )
            # 回退到 Stable 策略
            return await self._inject_stable(
                query=query,
                user_id=user_id,
                preferences=preferences,
                relevant_history=relevant_history,
                metadata=metadata,
                force_alpha=force_alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        # 更新元数据
        metadata.preference_tokens = fa_result.preference_tokens
        metadata.history_kv_tokens = fa_result.history_tokens
        metadata.history_tokens = 0  # Full Attention 不使用 suffix prompt
        metadata.query_tokens = self._estimate_tokens(query)
        metadata.total_tokens = metadata.query_tokens + len(fa_result.global_indication.split())
        
        # 构造最终输入 (仅包含全局指示 + 查询)
        final_input = query
        if fa_result.global_indication:
            final_input = fa_result.global_indication + "\n" + query
        
        # 门控决策
        if force_alpha is not None:
            alpha = force_alpha
        else:
            # Full Attention 使用配置的 alpha
            alpha = self._full_attention_injector.config.preference_alpha
        
        metadata.injection_enabled = fa_result.merged_kv is not None
        metadata.alpha = alpha
        metadata.gating_decision = {
            "should_inject": metadata.injection_enabled,
            "alpha": alpha,
            "preference_alpha": self._full_attention_injector.config.preference_alpha,
            "history_alpha": self._full_attention_injector.config.history_alpha,
            "has_preferences": fa_result.preference_tokens > 0,
            "has_history": fa_result.history_tokens > 0,
            "strategy": "full_attention",
            "position_mode": fa_result.position_mode,
        }
        
        # LLM 推理
        inference_start = time.time()
        
        if metadata.injection_enabled and fa_result.merged_kv:
            # 带合并 K/V 注入的推理
            output = self.model.forward_with_kv_injection(
                prompt=final_input,
                injected_kv=fa_result.merged_kv,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            # 普通推理
            output = self.model.generate(
                prompt=final_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        metadata.inference_latency_ms = (time.time() - inference_start) * 1000
        
        logger.debug(
            f"Full attention injection completed: "
            f"pref_kv={fa_result.preference_tokens}, hist_kv={fa_result.history_tokens}, "
            f"context={metadata.total_tokens} tokens"
        )
        
        return output, metadata
    
    def _format_preferences(self, preferences: List[AdapterUserPreference]) -> str:
        """格式化用户偏好为文本"""
        if not preferences:
            return ""
        
        # 按优先级排序
        sorted_prefs = sorted(preferences, key=lambda p: p.priority, reverse=True)
        
        lines = []
        for pref in sorted_prefs:
            if pref.is_expired():
                continue
            lines.append(f"- {pref.preference_type}: {pref.preference_text}")
        
        return "\n".join(lines)
    
    def _format_history_suffix(self, messages: List[AdapterChatMessage]) -> str:
        """格式化历史消息为后缀提示词"""
        if not messages:
            return ""
        
        # 选择语言模板
        prefix = self.HISTORY_PREFIX_CN if self.language == "cn" else self.HISTORY_PREFIX_EN
        suffix = self.HISTORY_SUFFIX_CN if self.language == "cn" else self.HISTORY_SUFFIX_EN
        
        # 格式化消息
        lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            if self.language == "en":
                role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        
        history_content = "\n".join(lines)
        
        return prefix + history_content + suffix
    
    async def _get_preference_kv(
        self,
        user_id: str,
        preference_text: str,
    ) -> Tuple[Optional[List[KVCacheEntry]], bool, str]:
        """
        获取偏好的 K/V 表示
        
        Returns:
            (kv_entries, cache_hit, cache_tier)
        """
        import hashlib
        content_hash = hashlib.md5(preference_text.encode()).hexdigest()
        cache_key = f"{user_id}:{content_hash}"
        
        # 检查缓存
        if cache_key in self._preference_kv_cache:
            kv_entries, cached_hash = self._preference_kv_cache[cache_key]
            if cached_hash == content_hash:
                return kv_entries, True, "memory"
        
        # 计算 K/V
        try:
            kv_entries, _ = self.model.compute_kv(preference_text)
            self._preference_kv_cache[cache_key] = (kv_entries, content_hash)
            return kv_entries, False, "compute"
        except Exception as e:
            logger.error(f"Failed to compute preference K/V: {e}")
            return None, False, "error"
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        # 粗略估算: 1.3 tokens per word
        return int(len(text.split()) * 1.3)
    
    def _record_injection_log(
        self,
        metadata: InjectionMetadata,
        query: str = "",
        user_id: str = "",
        session_id: str = "",
        final_input: str = "",
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
            self._stats["avg_alpha"] = total_alpha / self._stats["injection_enabled_count"]
        
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
                "final_input": final_input,
            })
        except Exception as e:
            logger.debug(f"Failed to record visualization: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据 (用于监控 API)"""
        # 获取缓存统计
        cache_stats = self._preference_cache.get_stats()
        
        return {
            "total_requests": self._stats["total_requests"],
            "injection_enabled_count": self._stats["injection_enabled_count"],
            "injection_rate": (
                self._stats["injection_enabled_count"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "avg_alpha": self._stats["avg_alpha"],
            # Memory Trigger 统计
            "memory_trigger_count": self._stats["memory_trigger_count"],
            "memory_trigger_rate": (
                self._stats["memory_trigger_count"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            # Reference Resolver 统计
            "reference_resolved_count": self._stats["reference_resolved_count"],
            "reference_resolved_rate": (
                self._stats["reference_resolved_count"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            # 组件配置信息
            "memory_trigger_config": self._memory_trigger.get_stats(),
            "reference_resolver_config": self._reference_resolver.get_stats(),
            # 缓存统计 (包含 Redis)
            "cache": cache_stats,
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
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """清除偏好缓存"""
        if user_id:
            keys_to_remove = [
                k for k in self._preference_kv_cache 
                if k.startswith(f"{user_id}:")
            ]
            for key in keys_to_remove:
                del self._preference_kv_cache[key]
        else:
            self._preference_kv_cache.clear()
    
    # ============ 组件配置更新方法 ============
    
    def update_reference_resolver_config(
        self,
        just_now_turns: Optional[int] = None,
        recently_turns: Optional[int] = None,
        last_topic_turns: Optional[int] = None,
        assistant_stance_turns: Optional[int] = None,
    ):
        """
        运行时更新 Reference Resolver 配置
        
        允许外部动态调整召回轮数
        
        Args:
            just_now_turns: "刚刚" 指代的召回轮数
            recently_turns: "最近" 指代的召回轮数
            last_topic_turns: "那件事" 指代的召回轮数
            assistant_stance_turns: "你之前说的" 指代的召回轮数
        """
        self._reference_resolver.update_config(
            just_now_turns=just_now_turns,
            recently_turns=recently_turns,
            last_topic_turns=last_topic_turns,
            assistant_stance_turns=assistant_stance_turns,
        )
        logger.info(f"Reference Resolver config updated: {self._reference_resolver.get_stats()}")
    
    def update_memory_trigger_config(
        self,
        enabled: Optional[bool] = None,
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        运行时更新 Memory Trigger 配置
        
        Args:
            enabled: 是否启用
            custom_patterns: 自定义规则列表
        """
        self._memory_trigger.update_config(
            enabled=enabled,
            custom_patterns=custom_patterns,
        )
        logger.info(f"Memory Trigger config updated: {self._memory_trigger.get_stats()}")
    
    def get_component_configs(self) -> Dict[str, Any]:
        """获取所有组件的当前配置"""
        configs = {
            "memory_trigger": self._memory_trigger.get_stats(),
            "reference_resolver": self._reference_resolver.get_stats(),
            "injection_strategy": self._injection_strategy,
        }
        
        # 添加 Full Attention 配置 (如果启用)
        if self._full_attention_injector:
            configs["full_attention"] = self._full_attention_injector.get_stats()
        
        return configs
    
    def switch_injection_strategy(self, strategy: str) -> bool:
        """
        运行时切换注入策略
        
        Args:
            strategy: "stable" | "full_attention"
            
        Returns:
            是否切换成功
        """
        if strategy not in ("stable", "full_attention"):
            logger.error(f"Invalid injection strategy: {strategy}")
            return False
        
        old_strategy = self._injection_strategy
        self._injection_strategy = strategy
        
        # 如果切换到 full_attention 但注入器未初始化，则初始化
        if strategy == "full_attention" and not self._full_attention_injector:
            fa_config = self._get_full_attention_config(self.config)
            self._full_attention_injector = FullAttentionInjector(
                config=fa_config,
                language=self.language,
            )
        
        logger.info(f"Injection strategy switched: {old_strategy} -> {strategy}")
        return True
    
    def get_full_attention_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取 Full Attention 注入器统计
        
        Returns:
            统计数据，如果未启用则返回 None
        """
        if self._full_attention_injector:
            return self._full_attention_injector.get_stats()
        return None
    
    def get_full_attention_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取 Full Attention attention pattern 日志
        
        用于研究分析
        """
        if self._full_attention_injector:
            return self._full_attention_injector.get_attention_logs(limit)
        return []
    
    def update_full_attention_config(
        self,
        position_mode: Optional[str] = None,
        preference_alpha: Optional[float] = None,
        history_alpha: Optional[float] = None,
        history_position_start: Optional[int] = None,
        max_total_kv_tokens: Optional[int] = None,
    ):
        """
        运行时更新 Full Attention 配置
        
        用于研究实验调参
        """
        if self._full_attention_injector:
            self._full_attention_injector.update_config(
                position_mode=position_mode,
                preference_alpha=preference_alpha,
                history_alpha=history_alpha,
                history_position_start=history_position_start,
                max_total_kv_tokens=max_total_kv_tokens,
            )
        else:
            logger.warning("Full attention injector not initialized")
    
    async def close(self):
        """
        关闭 DKI 插件
        
        清理资源，包括:
        - 关闭 Redis 连接
        - 关闭数据库连接
        """
        # 关闭 Redis 连接
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
        
        # 关闭数据库连接
        if hasattr(self.data_adapter, 'close'):
            await self.data_adapter.close()
            logger.info("Database connection closed")
        
        logger.info("DKI Plugin closed")
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """
        使用户缓存失效
        
        当用户偏好更新时调用此方法
        
        Args:
            user_id: 用户 ID
            
        Returns:
            失效的缓存条目数
        """
        return await self._preference_cache.invalidate(user_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计
        
        Returns:
            缓存统计信息，包括:
            - L1 (Memory) 命中率
            - L2 (Redis) 命中率
            - 总体命中率
            - Redis 连接状态
        """
        return self._preference_cache.get_stats()