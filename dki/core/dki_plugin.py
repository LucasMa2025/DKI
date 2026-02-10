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


@dataclass
class InjectionMetadata:
    """DKI 注入元数据 (用于监控)"""
    # 注入状态
    injection_enabled: bool = False
    alpha: float = 0.0
    
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
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "injection_enabled": self.injection_enabled,
            "alpha": self.alpha,
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
        
        logger.info(f"DKI Plugin initialized (language={language}, cache={cache_status})")
    
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
        
        # 加载全局配置
        config_loader = ConfigLoader()
        global_config = config_loader.config
        
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
        elif global_config.get('memory_trigger'):
            mt_config = MemoryTriggerConfig.from_dict(global_config['memory_trigger'])
        
        # 处理 Reference Resolver 配置 (支持外置召回轮数)
        rr_config = None
        if reference_resolver_config:
            if isinstance(reference_resolver_config, dict):
                rr_config = ReferenceResolverConfig(**reference_resolver_config)
            else:
                rr_config = reference_resolver_config
        elif global_config.get('reference_resolver'):
            rr_config = ReferenceResolverConfig.from_dict(global_config['reference_resolver'])
        
        # ============ 处理 Redis 配置 ============
        redis_client = None
        cache_config = CacheConfig()
        
        # 从全局配置加载缓存配置
        if global_config.get('preference_cache'):
            cache_config = CacheConfig.from_dict(global_config['preference_cache'])
        
        # 确定是否启用 Redis
        should_enable_redis = enable_redis
        if should_enable_redis is None:
            # 从配置文件读取
            should_enable_redis = global_config.get('redis', {}).get('enabled', False)
        
        if should_enable_redis and REDIS_AVAILABLE:
            # 创建 Redis 配置
            if redis_config:
                if isinstance(redis_config, dict):
                    r_config = RedisConfig.from_dict(redis_config)
                else:
                    r_config = redis_config
            elif global_config.get('redis'):
                r_config = RedisConfig.from_dict(global_config['redis'])
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
            
            # ============ Step 2: DKI 注入处理 ============
            injection_start = time.time()
            
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
            # 输入 = 用户查询 + 历史后缀 (正位置)
            # 注入 = 偏好 K/V (负位置)
            final_input = query
            if history_suffix:
                final_input = query + "\n\n" + history_suffix
            
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
            }
            
            metadata.injection_latency_ms = (time.time() - injection_start) * 1000
            
            # ============ Step 3: LLM 推理 ============
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
            
            # ============ Step 4: 记录工作数据 ============
            metadata.latency_ms = (time.time() - start_time) * 1000
            self._record_injection_log(metadata)
            
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
    
    def _record_injection_log(self, metadata: InjectionMetadata):
        """记录注入日志 (用于监控)"""
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
        return {
            "memory_trigger": self._memory_trigger.get_stats(),
            "reference_resolver": self._reference_resolver.get_stats(),
        }
    
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