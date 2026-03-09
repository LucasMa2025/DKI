"""
EnhancedDKIPlugin — DKIPlugin 的增强包装层

在 DKIPlugin 基础上增加:
1. dynamic_router: 自动在 RAG 和 DKI (recall_v4) 之间切换
2. message_management: DKI 内部完成消息写入和偏好写入
3. 统一的生命周期管理

设计原则:
- EnhancedDKIPlugin 是 DKIPlugin 的超集, 不修改 DKIPlugin 本身
- 所有增强功能通过配置开关控制, 默认关闭 (向后兼容)
- 上层应用可以选择性启用增强功能

Author: AGI Demo Project
Version: 4.0.0
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from loguru import logger

from dki.core.dki_plugin import DKIPlugin, DKIPluginResponse, InjectionMetadata


# ============================================================
# 配置
# ============================================================

@dataclass
class DynamicRouterConfig:
    """
    动态路由配置
    
    控制 EnhancedDKIPlugin 在 RAG 和 DKI 之间的自动切换。
    路由决策基于 ConversationRouter 的五维评分模型。
    """
    enabled: bool = False
    # 路由器配置 (传递给 ConversationRouter)
    dki_threshold: float = 0.45
    rag_threshold: float = 0.25
    default_mode: str = "dki"  # 路由器禁用时的默认模式
    # 权重
    weight_history: float = 0.25
    weight_preference: float = 0.20
    weight_trigger: float = 0.20
    weight_session_depth: float = 0.20
    weight_cross_session: float = 0.15
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicRouterConfig":
        if not d:
            return cls()
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class MessageManagementConfig:
    """
    消息管理配置
    
    控制 DKI 是否内部完成消息写入和偏好写入。
    启用后, 上层应用不需要自己写入消息到数据库。
    """
    enabled: bool = False
    # 写入用户消息
    write_user_message: bool = True
    # 写入助手回复
    write_assistant_message: bool = True
    # 写入偏好 (从对话中自动提取偏好, 未来功能)
    auto_extract_preference: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MessageManagementConfig":
        if not d:
            return cls()
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class EnhancedPluginConfig:
    """EnhancedDKIPlugin 的完整配置"""
    dynamic_router: DynamicRouterConfig = field(default_factory=DynamicRouterConfig)
    message_management: MessageManagementConfig = field(default_factory=MessageManagementConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnhancedPluginConfig":
        if not d:
            return cls()
        return cls(
            dynamic_router=DynamicRouterConfig.from_dict(d.get("dynamic_router", {})),
            message_management=MessageManagementConfig.from_dict(d.get("message_management", {})),
        )


# ============================================================
# EnhancedDKIPlugin
# ============================================================

class EnhancedDKIPlugin:
    """
    DKI Plugin 增强包装层
    
    在 DKIPlugin 基础上提供:
    1. 动态路由 (dynamic_router): 自动选择 RAG 或 DKI
    2. 消息管理 (message_management): 内部完成消息读写
    3. 统一生命周期管理
    
    使用方式:
    ```python
    # 最简用法 (等同于 DKIPlugin)
    enhanced = EnhancedDKIPlugin(dki_plugin)
    response = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
    
    # 启用动态路由
    enhanced = EnhancedDKIPlugin(
        dki_plugin,
        rag_system=rag,
        config=EnhancedPluginConfig(
            dynamic_router=DynamicRouterConfig(enabled=True),
        ),
    )
    response = await enhanced.chat("什么是量子计算", user_id="u1", session_id="s1")
    # → 自动路由到 RAG (知识检索型查询)
    
    # 启用消息管理
    enhanced = EnhancedDKIPlugin(
        dki_plugin,
        store=chat_store,
        config=EnhancedPluginConfig(
            message_management=MessageManagementConfig(enabled=True),
        ),
    )
    response = await enhanced.chat("推荐餐厅", user_id="u1", session_id="s1")
    # → DKI 内部自动写入用户消息和助手回复到 store
    ```
    """
    
    def __init__(
        self,
        dki_plugin: DKIPlugin,
        rag_system: Optional[Any] = None,
        store: Optional[Any] = None,
        config: Optional[EnhancedPluginConfig] = None,
    ):
        """
        Args:
            dki_plugin: 核心 DKI Plugin 实例
            rag_system: RAG 系统实例 (动态路由需要)
            store: IChatStore 实例 (消息管理需要)
            config: 增强配置
        """
        self._dki_plugin = dki_plugin
        self._rag_system = rag_system
        self._store = store
        self._config = config or EnhancedPluginConfig()
        
        # 动态路由器 (延迟初始化)
        self._router = None
        if self._config.dynamic_router.enabled:
            self._init_router()
        
        # 统计
        self._enhanced_stats = {
            "total_requests": 0,
            "dki_routes": 0,
            "rag_routes": 0,
            "messages_written": 0,
        }
        
        features = []
        if self._config.dynamic_router.enabled:
            features.append("dynamic_router")
        if self._config.message_management.enabled:
            features.append("message_management")
        
        logger.info(
            f"EnhancedDKIPlugin initialized "
            f"(features={features or ['none']})"
        )
    
    def _init_router(self):
        """初始化动态路由器"""
        try:
            from dki.core.conversation_router import ConversationRouter, RouterConfig
            
            rc = self._config.dynamic_router
            router_config = RouterConfig(
                enabled=rc.enabled,
                default_mode=rc.default_mode,
                dki_threshold=rc.dki_threshold,
                rag_threshold=rc.rag_threshold,
                weight_history=rc.weight_history,
                weight_preference=rc.weight_preference,
                weight_trigger=rc.weight_trigger,
                weight_session_depth=rc.weight_session_depth,
                weight_cross_session=rc.weight_cross_session,
            )
            self._router = ConversationRouter(config=router_config)
            logger.info("Dynamic router initialized for EnhancedDKIPlugin")
        except ImportError:
            logger.warning("ConversationRouter not available, dynamic routing disabled")
            self._config.dynamic_router.enabled = False
    
    # ================================================================
    # 属性代理 (透传到 DKIPlugin)
    # ================================================================
    
    @property
    def dki_plugin(self) -> DKIPlugin:
        """获取底层 DKI Plugin"""
        return self._dki_plugin
    
    @property
    def model(self):
        """代理: 模型适配器"""
        return self._dki_plugin.model
    
    @property
    def data_adapter(self):
        """代理: 数据适配器"""
        return self._dki_plugin.data_adapter
    
    @property
    def planner(self):
        """代理: Planner"""
        return self._dki_plugin.planner
    
    @property
    def executor(self):
        """代理: Executor"""
        return self._dki_plugin.executor
    
    # ================================================================
    # 核心 chat 方法 (增强版)
    # ================================================================
    
    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_alpha: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        force_mode: Optional[str] = None,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        增强版 chat — 支持动态路由和消息管理
        
        Args:
            query: 用户查询
            user_id: 用户 ID
            session_id: 会话 ID
            force_alpha: 强制 alpha 值
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            force_mode: 强制模式 ("dki" | "rag" | None=自动)
            **kwargs: 额外参数
            
        Returns:
            DKIPluginResponse (与 DKIPlugin.chat() 返回格式一致)
        """
        self._enhanced_stats["total_requests"] += 1
        
        # ============ Step 1: 消息管理 — 写入用户消息 ============
        if self._config.message_management.enabled and self._store:
            await self._write_user_message(user_id, session_id, query)
        
        # ============ Step 2: 动态路由 ============
        route_mode = self._resolve_route_mode(
            query=query,
            user_id=user_id,
            session_id=session_id,
            force_mode=force_mode,
        )
        
        # ============ Step 3: 执行 ============
        if route_mode == "rag" and self._rag_system:
            response = await self._execute_rag(
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            self._enhanced_stats["rag_routes"] += 1
        else:
            response = await self._dki_plugin.chat(
                query=query,
                user_id=user_id,
                session_id=session_id,
                force_alpha=force_alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            self._enhanced_stats["dki_routes"] += 1
        
        # ============ Step 4: 消息管理 — 写入助手回复 ============
        if self._config.message_management.enabled and self._store:
            await self._write_assistant_message(
                user_id=user_id,
                session_id=session_id,
                content=response.text,
                metadata={
                    "injection_enabled": response.metadata.injection_enabled,
                    "alpha": response.metadata.alpha,
                    "latency_ms": response.metadata.latency_ms,
                    "route_mode": route_mode,
                },
            )
        
        return response
    
    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_alpha: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        force_mode: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        增强版流式 chat — 支持动态路由和消息管理
        """
        self._enhanced_stats["total_requests"] += 1
        
        # 写入用户消息
        if self._config.message_management.enabled and self._store:
            await self._write_user_message(user_id, session_id, query)
        
        # 路由决策
        route_mode = self._resolve_route_mode(
            query=query,
            user_id=user_id,
            session_id=session_id,
            force_mode=force_mode,
        )
        
        if route_mode == "rag" and self._rag_system:
            # RAG 不支持流式, 模拟流式返回
            response = await self._execute_rag(
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            self._enhanced_stats["rag_routes"] += 1
            
            yield {"type": "metadata", "metadata": response.metadata.to_dict(), "route_mode": "rag"}
            yield {"type": "token", "content": response.text}
            yield {
                "type": "done",
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "metadata": response.metadata.to_dict(),
            }
            
            # 写入助手回复
            if self._config.message_management.enabled and self._store:
                await self._write_assistant_message(
                    user_id=user_id,
                    session_id=session_id,
                    content=response.text,
                    metadata={"route_mode": "rag"},
                )
        else:
            # DKI 流式
            self._enhanced_stats["dki_routes"] += 1
            full_text = ""
            
            async for chunk in self._dki_plugin.chat_stream(
                query=query,
                user_id=user_id,
                session_id=session_id,
                force_alpha=force_alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            ):
                # 注入路由信息
                if isinstance(chunk, dict):
                    chunk["route_mode"] = "dki"
                    if chunk.get("type") == "token":
                        full_text += chunk.get("content", "")
                    elif chunk.get("type") == "done":
                        full_text = chunk.get("text", full_text)
                yield chunk
            
            # 写入助手回复
            if self._config.message_management.enabled and self._store and full_text:
                await self._write_assistant_message(
                    user_id=user_id,
                    session_id=session_id,
                    content=full_text,
                    metadata={"route_mode": "dki"},
                )
    
    # ================================================================
    # 动态路由
    # ================================================================
    
    def _resolve_route_mode(
        self,
        query: str,
        user_id: str,
        session_id: str,
        force_mode: Optional[str] = None,
    ) -> str:
        """
        解析路由模式
        
        Returns:
            "dki" | "rag"
        """
        # 强制模式
        if force_mode:
            return force_mode
        
        # ============ 闭源模型强制 RAG ============
        # 闭源模型无法进行 K/V 注入, 必须走 RAG 路由 (prompt 拼接)
        if self._is_closed_source_model():
            logger.debug(
                "[Router] Closed-source model detected, forcing RAG route"
            )
            return "rag"
        
        # 路由器未启用 → 默认 DKI
        if not self._config.dynamic_router.enabled or not self._router:
            return "dki"
        
        # RAG 不可用 → DKI
        if not self._rag_system:
            return "dki"
        
        # 使用路由器决策
        try:
            # 收集信号 (轻量级, 不查 DB)
            # 从 DKI Plugin 的统计中获取偏好信息
            has_prefs = False
            pref_count = 0
            try:
                # 检查偏好文本缓存
                if user_id in self._dki_plugin._preference_text_cache:
                    cached_prefs, _ = self._dki_plugin._preference_text_cache[user_id]
                    has_prefs = len(cached_prefs) > 0
                    pref_count = len(cached_prefs)
            except Exception:
                pass
            
            decision = self._router.route(
                query=query,
                session_id=session_id,
                user_id=user_id,
                has_user_preferences=has_prefs,
                preference_count=pref_count,
                dki_available=True,
                rag_available=True,
                forced_mode=force_mode,
            )
            
            return decision.mode.value
            
        except Exception as e:
            logger.warning(f"Router decision failed, defaulting to DKI: {e}")
            return "dki"
    
    def _is_closed_source_model(self) -> bool:
        """
        检测当前模型是否为闭源模型

        检测逻辑 (优先级从高到低):
        1. 模型适配器自身标记 (is_closed_source=True)
        2. ModelFactory.is_closed_source_engine() (基于 config)
        """
        try:
            model = self._dki_plugin.model
            if getattr(model, "is_closed_source", False):
                return True
        except Exception:
            pass
        try:
            from dki.models.factory import ModelFactory
            return ModelFactory.is_closed_source_engine()
        except Exception:
            pass
        return False

    async def _execute_rag(
        self,
        query: str,
        user_id: str,
        session_id: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> DKIPluginResponse:
        """
        执行 RAG 并将结果包装为 DKIPluginResponse (统一返回格式)
        """
        start_time = time.time()
        
        try:
            # RAGSystem.chat() 是同步的
            import asyncio
            rag_response = await asyncio.to_thread(
                self._rag_system.chat,
                query=query,
                session_id=session_id,
                user_id=user_id,
                max_new_tokens=max_new_tokens or 2048,
                temperature=temperature,
            )
            
            # 包装为 DKIPluginResponse
            metadata = InjectionMetadata(
                injection_enabled=False,
                injection_strategy="rag",
                latency_ms=(time.time() - start_time) * 1000,
                retrieval_mode="rag",
            )
            
            return DKIPluginResponse(
                text=rag_response.text,
                input_tokens=rag_response.input_tokens,
                output_tokens=rag_response.output_tokens,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"RAG execution failed, falling back to DKI: {e}")
            # RAG 失败 → 降级到 DKI
            return await self._dki_plugin.chat(
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
    
    # ================================================================
    # 消息管理
    # ================================================================
    
    async def _write_user_message(
        self,
        user_id: str,
        session_id: str,
        content: str,
    ):
        """写入用户消息到 store"""
        if not self._config.message_management.write_user_message:
            return
        
        try:
            _async = hasattr(self._store, 'a_add_message')
            
            # 确保会话存在
            if _async:
                session = await self._store.a_get_session(session_id)
                if not session:
                    await self._store.a_create_session(
                        user_id=user_id,
                        title="New Chat",
                        session_id=session_id,
                    )
                await self._store.a_add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=content,
                )
            else:
                session = self._store.get_session(session_id)
                if not session:
                    self._store.create_session(
                        user_id=user_id,
                        title="New Chat",
                        session_id=session_id,
                    )
                self._store.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=content,
                )
            
            self._enhanced_stats["messages_written"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to write user message (non-critical): {e}")
    
    async def _write_assistant_message(
        self,
        user_id: str,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """写入助手回复到 store"""
        if not self._config.message_management.write_assistant_message:
            return
        
        try:
            _async = hasattr(self._store, 'a_add_message')
            
            if _async:
                await self._store.a_add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=content,
                    metadata=metadata,
                )
            else:
                self._store.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=content,
                    metadata=metadata,
                )
            
            self._enhanced_stats["messages_written"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to write assistant message (non-critical): {e}")
    
    # ================================================================
    # 偏好管理 (委托给 store)
    # ================================================================
    
    async def add_preference(
        self,
        user_id: str,
        preference_text: str,
        preference_type: str = "general",
        priority: int = 5,
    ):
        """
        添加用户偏好
        
        同时:
        1. 写入 store (持久化)
        2. 使 DKI Plugin 的偏好缓存失效 (立即生效)
        """
        if not self._store:
            raise RuntimeError("Store not configured, cannot add preference")
        
        _async = hasattr(self._store, 'a_add_preference')
        
        if _async:
            pref = await self._store.a_add_preference(
                user_id=user_id,
                preference_text=preference_text,
                preference_type=preference_type,
                priority=priority,
            )
        else:
            pref = self._store.add_preference(
                user_id=user_id,
                preference_text=preference_text,
                preference_type=preference_type,
                priority=priority,
            )
        
        # 使 DKI 偏好缓存失效
        self._dki_plugin.invalidate_preference_text_cache(user_id)
        
        return pref
    
    # ================================================================
    # 代理方法 (透传到 DKIPlugin)
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计 (合并 DKI + 增强层)"""
        dki_stats = self._dki_plugin.get_stats()
        
        router_stats = {}
        if self._router:
            router_stats = self._router.get_stats()
        
        return {
            **dki_stats,
            "enhanced": {
                **self._enhanced_stats,
                "dynamic_router_enabled": self._config.dynamic_router.enabled,
                "message_management_enabled": self._config.message_management.enabled,
                "router": router_stats,
            },
        }
    
    def get_injection_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """代理: 获取注入日志"""
        return self._dki_plugin.get_injection_logs(limit=limit, offset=offset)
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """代理: 清除偏好缓存"""
        self._dki_plugin.clear_preference_cache(user_id)
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """代理: 使用户缓存失效"""
        return await self._dki_plugin.invalidate_user_cache(user_id)
    
    def get_component_configs(self) -> Dict[str, Any]:
        """代理: 获取组件配置"""
        configs = self._dki_plugin.get_component_configs()
        configs["dynamic_router"] = {
            "enabled": self._config.dynamic_router.enabled,
            "dki_threshold": self._config.dynamic_router.dki_threshold,
            "rag_threshold": self._config.dynamic_router.rag_threshold,
        }
        configs["message_management"] = {
            "enabled": self._config.message_management.enabled,
            "write_user_message": self._config.message_management.write_user_message,
            "write_assistant_message": self._config.message_management.write_assistant_message,
        }
        return configs
    
    async def close(self):
        """关闭所有资源"""
        await self._dki_plugin.close()
        logger.info("EnhancedDKIPlugin closed")
