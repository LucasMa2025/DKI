"""
DKI Plugin Manager — 多实例管理器 (P2-M3)

管理多个 DKI Plugin 实例, 支持:
1. A/B 测试: 同时运行两个不同模型的 DKI Plugin
2. 蓝绿部署: 新旧版本 Plugin 并行运行, 逐步切换流量
3. 多租户: 不同租户使用不同的 Plugin 配置

核心设计:
- DKIPluginManager 是 DKIPlugin 的上层管理器
- 每个 Plugin 实例有唯一的 instance_id
- 支持按权重路由请求 (A/B 测试)
- 支持按用户路由 (多租户)
- 向后兼容: 单实例场景下行为与直接使用 DKIPlugin 完全一致

Usage:
    # 单实例 (向后兼容)
    manager = DKIPluginManager()
    await manager.register("default", plugin)
    response = await manager.chat(query, user_id, session_id)
    
    # A/B 测试
    manager = DKIPluginManager()
    await manager.register("model_a", plugin_a, weight=50)
    await manager.register("model_b", plugin_b, weight=50)
    response = await manager.chat(query, user_id, session_id)  # 按权重路由
    
    # 指定实例
    response = await manager.chat(query, user_id, session_id, instance_id="model_a")

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class PluginInstance:
    """DKI Plugin 实例元数据"""
    instance_id: str
    plugin: Any  # DKIPlugin (避免循环导入)
    weight: int = 100  # 路由权重 (0-100)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.request_count, 1)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.request_count, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "weight": self.weight,
            "is_active": self.is_active,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": self.error_rate,
        }


class DKIPluginManager:
    """
    DKI Plugin 多实例管理器
    
    路由策略:
    1. weighted_random: 按权重随机路由 (默认, 适合 A/B 测试)
    2. user_hash: 按 user_id 哈希路由 (同一用户始终路由到同一实例)
    3. round_robin: 轮询路由
    4. lowest_latency: 路由到平均延迟最低的实例
    """
    
    ROUTING_STRATEGIES = ("weighted_random", "user_hash", "round_robin", "lowest_latency")
    
    def __init__(
        self,
        routing_strategy: str = "weighted_random",
        fallback_on_error: bool = True,
    ):
        """
        Args:
            routing_strategy: 路由策略
            fallback_on_error: 实例出错时是否自动切换到其他实例
        """
        if routing_strategy not in self.ROUTING_STRATEGIES:
            raise ValueError(
                f"Unknown routing strategy: {routing_strategy}. "
                f"Available: {self.ROUTING_STRATEGIES}"
            )
        
        self._instances: Dict[str, PluginInstance] = {}
        self._routing_strategy = routing_strategy
        self._fallback_on_error = fallback_on_error
        self._round_robin_idx = 0
    
    # ================================================================
    # 实例管理
    # ================================================================
    
    async def register(
        self,
        instance_id: str,
        plugin: Any,
        weight: int = 100,
    ) -> None:
        """
        注册 DKI Plugin 实例
        
        Args:
            instance_id: 实例标识 (如 "model_a", "model_b")
            plugin: DKIPlugin 实例
            weight: 路由权重 (0-100, 仅 weighted_random 策略使用)
        """
        self._instances[instance_id] = PluginInstance(
            instance_id=instance_id,
            plugin=plugin,
            weight=weight,
        )
        logger.info(
            f"DKI Plugin instance registered: {instance_id} "
            f"(weight={weight}, total_instances={len(self._instances)})"
        )
    
    async def unregister(self, instance_id: str, close: bool = True) -> None:
        """
        注销 DKI Plugin 实例
        
        Args:
            instance_id: 实例标识
            close: 是否关闭 Plugin (释放资源)
        """
        if instance_id not in self._instances:
            return
        
        inst = self._instances.pop(instance_id)
        if close and hasattr(inst.plugin, 'close'):
            await inst.plugin.close()
        
        logger.info(f"DKI Plugin instance unregistered: {instance_id}")
    
    def set_weight(self, instance_id: str, weight: int) -> None:
        """动态调整实例权重 (用于灰度发布)"""
        if instance_id in self._instances:
            self._instances[instance_id].weight = weight
            logger.info(f"Instance {instance_id} weight updated to {weight}")
    
    def set_active(self, instance_id: str, active: bool) -> None:
        """启用/禁用实例"""
        if instance_id in self._instances:
            self._instances[instance_id].is_active = active
            logger.info(f"Instance {instance_id} active={active}")
    
    def get_instance(self, instance_id: str) -> Optional[Any]:
        """获取指定实例的 DKIPlugin"""
        inst = self._instances.get(instance_id)
        return inst.plugin if inst else None
    
    @property
    def default_instance(self) -> Optional[Any]:
        """获取默认实例 (第一个注册的)"""
        if not self._instances:
            return None
        return next(iter(self._instances.values())).plugin
    
    # ================================================================
    # 路由
    # ================================================================
    
    def _get_active_instances(self) -> List[PluginInstance]:
        """获取所有活跃实例"""
        return [inst for inst in self._instances.values() if inst.is_active]
    
    def _route(self, user_id: str = "") -> Optional[PluginInstance]:
        """
        根据路由策略选择实例
        
        Returns:
            选中的 PluginInstance, 或 None (无可用实例)
        """
        active = self._get_active_instances()
        if not active:
            return None
        
        if len(active) == 1:
            return active[0]
        
        if self._routing_strategy == "weighted_random":
            weights = [inst.weight for inst in active]
            total = sum(weights)
            if total == 0:
                return random.choice(active)
            r = random.uniform(0, total)
            cumulative = 0
            for inst in active:
                cumulative += inst.weight
                if r <= cumulative:
                    return inst
            return active[-1]
        
        elif self._routing_strategy == "user_hash":
            # 同一用户始终路由到同一实例
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            idx = hash_val % len(active)
            return active[idx]
        
        elif self._routing_strategy == "round_robin":
            idx = self._round_robin_idx % len(active)
            self._round_robin_idx += 1
            return active[idx]
        
        elif self._routing_strategy == "lowest_latency":
            return min(active, key=lambda inst: inst.avg_latency_ms)
        
        return active[0]
    
    # ================================================================
    # 核心 API (代理到 DKIPlugin)
    # ================================================================
    
    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        instance_id: Optional[str] = None,
        **kwargs,
    ):
        """
        路由到合适的 DKI Plugin 实例执行 chat
        
        Args:
            query: 用户查询
            user_id: 用户 ID
            session_id: 会话 ID
            instance_id: 指定实例 (可选, 跳过路由)
            **kwargs: 传递给 DKIPlugin.chat() 的额外参数
        """
        # 指定实例
        if instance_id:
            inst = self._instances.get(instance_id)
            if not inst:
                raise ValueError(f"Instance not found: {instance_id}")
            return await self._execute_chat(inst, query, user_id, session_id, **kwargs)
        
        # 路由选择
        inst = self._route(user_id)
        if not inst:
            raise RuntimeError("No active DKI Plugin instances available")
        
        try:
            return await self._execute_chat(inst, query, user_id, session_id, **kwargs)
        except Exception as e:
            inst.error_count += 1
            
            if self._fallback_on_error:
                # 尝试其他实例
                for fallback in self._get_active_instances():
                    if fallback.instance_id != inst.instance_id:
                        logger.warning(
                            f"Instance {inst.instance_id} failed, "
                            f"falling back to {fallback.instance_id}: {e}"
                        )
                        try:
                            return await self._execute_chat(
                                fallback, query, user_id, session_id, **kwargs
                            )
                        except Exception:
                            fallback.error_count += 1
                            continue
            
            raise
    
    async def _execute_chat(
        self,
        inst: PluginInstance,
        query: str,
        user_id: str,
        session_id: str,
        **kwargs,
    ):
        """执行 chat 并记录统计"""
        start = time.time()
        try:
            response = await inst.plugin.chat(
                query=query,
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )
            inst.request_count += 1
            inst.total_latency_ms += (time.time() - start) * 1000
            
            # 在 response metadata 中标记实例 ID
            if hasattr(response, 'metadata') and hasattr(response.metadata, '__dict__'):
                response.metadata.__dict__['_plugin_instance_id'] = inst.instance_id
            
            return response
        except Exception:
            inst.request_count += 1
            inst.total_latency_ms += (time.time() - start) * 1000
            raise
    
    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """路由到合适的实例执行流式 chat"""
        if instance_id:
            inst = self._instances.get(instance_id)
            if not inst:
                raise ValueError(f"Instance not found: {instance_id}")
        else:
            inst = self._route(user_id)
            if not inst:
                raise RuntimeError("No active DKI Plugin instances available")
        
        inst.request_count += 1
        async for chunk in inst.plugin.chat_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        ):
            # 在 metadata chunk 中注入实例 ID
            if isinstance(chunk, dict) and chunk.get("type") == "metadata":
                chunk["instance_id"] = inst.instance_id
            yield chunk
    
    # ================================================================
    # 统计与管理
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有实例的统计信息"""
        instances = {}
        for inst_id, inst in self._instances.items():
            instances[inst_id] = inst.to_dict()
            # 合并 Plugin 自身的统计
            try:
                instances[inst_id]["plugin_stats"] = inst.plugin.get_stats()
            except Exception:
                instances[inst_id]["plugin_stats"] = {}
        
        return {
            "routing_strategy": self._routing_strategy,
            "total_instances": len(self._instances),
            "active_instances": len(self._get_active_instances()),
            "instances": instances,
        }
    
    async def close_all(self) -> None:
        """关闭所有 Plugin 实例"""
        for inst_id in list(self._instances.keys()):
            await self.unregister(inst_id, close=True)
        logger.info("All DKI Plugin instances closed")
