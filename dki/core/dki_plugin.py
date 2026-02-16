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
    
    # Alpha Profile (v3.0)
    alpha_profile: Optional[Dict[str, Any]] = None
    
    # 安全违规 (v3.0)
    safety_violations: Optional[List[str]] = None
    
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
            "safety_violations": self.safety_violations or [],
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
        
        # DKI 组件 (延迟初始化)
        self._mis: Optional[MemoryInfluenceScaling] = None
        self._gating: Optional[DualFactorGating] = None
        
        # ============ Planner (纯决策) ============
        self._planner = InjectionPlanner(
            config=self.config,
            language=language,
            injection_strategy="recall_v4",
            memory_trigger_config=memory_trigger_config,
            reference_resolver_config=reference_resolver_config,
        )
        
        # ============ Executor (纯执行) ============
        self._executor = InjectionExecutor(
            model_adapter=model_adapter,
        )
        
        # ============ 偏好 K/V 缓存 (支持 Redis 分布式) ============
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
        max_new_tokens: int = 512,
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
            
            preferences = await self.data_adapter.get_user_preferences(user_id)
            metadata.preferences_count = len(preferences)
            
            relevant_history = await self.data_adapter.search_relevant_history(
                user_id=user_id,
                query=query,
                session_id=session_id,
                limit=context.recall_limit,
            )
            metadata.relevant_history_count = len(relevant_history)
            
            metadata.adapter_latency_ms = (time.time() - adapter_start) * 1000
            
            logger.debug(
                f"Adapter data loaded: {len(preferences)} preferences, "
                f"{len(relevant_history)} relevant history messages"
            )
            
            # ============ Step 3: 构建注入计划 (Planner Phase 2) ============
            plan = self._planner.build_plan(
                query=query,
                user_id=user_id,
                preferences=preferences,
                relevant_history=relevant_history,
                context=context,
                force_alpha=force_alpha,
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
            
            # ============ Step 4: 执行注入计划 (Executor) ============
            injection_start = time.time()
            
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
            )
            
            return DKIPluginResponse(
                text=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                metadata=metadata,
                raw_output=result.raw_output,
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
        
        当用户偏好更新时调用此方法
        """
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
        
        if hasattr(self.data_adapter, 'close'):
            await self.data_adapter.close()
            logger.info("Database connection closed")
        
        logger.info("DKI Plugin closed")
