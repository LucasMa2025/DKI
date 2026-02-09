"""
DKI Monitoring API Routes

DKI 监控 API 路由

职责: 提供 DKI 工作数据的监控接口
- 统计数据 (stats)
- 注入日志 (logs)
- 健康检查 (health)
- 缓存状态 (cache)

注意: 这不是对外聊天服务，而是监控 DKI 插件的工作状态

Author: AGI Demo Project
Version: 2.0.0
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger


# ============ Response Models ============

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="健康状态: healthy | degraded | unhealthy")
    version: str = Field(..., description="DKI 版本")
    uptime_seconds: float = Field(..., description="运行时间 (秒)")
    components: Dict[str, str] = Field(default_factory=dict, description="组件状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatsResponse(BaseModel):
    """统计数据响应"""
    # 请求统计
    total_requests: int = Field(0, description="总请求数")
    injection_enabled_count: int = Field(0, description="启用注入的请求数")
    injection_rate: float = Field(0.0, description="注入率")
    
    # 缓存统计
    cache_hits: int = Field(0, description="缓存命中数")
    cache_hit_rate: float = Field(0.0, description="缓存命中率")
    preference_cache_size: int = Field(0, description="偏好缓存大小")
    
    # 性能统计
    avg_latency_ms: float = Field(0.0, description="平均延迟 (ms)")
    avg_alpha: float = Field(0.0, description="平均 alpha 值")
    
    # 数据源统计
    adapter_type: str = Field("", description="适配器类型")
    adapter_connected: bool = Field(False, description="适配器连接状态")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InjectionLogEntry(BaseModel):
    """注入日志条目"""
    request_id: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # 注入状态
    injection_enabled: bool
    alpha: float
    
    # Token 统计
    preference_tokens: int = 0
    history_tokens: int = 0
    query_tokens: int = 0
    
    # 缓存状态
    preference_cache_hit: bool = False
    preference_cache_tier: str = "none"
    
    # 性能
    latency_ms: float = 0.0
    adapter_latency_ms: float = 0.0
    injection_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0


class InjectionLogsResponse(BaseModel):
    """注入日志响应"""
    logs: List[InjectionLogEntry]
    total_count: int
    offset: int
    limit: int


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""
    preference_cache: Dict[str, Any] = Field(default_factory=dict)
    session_cache: Dict[str, Any] = Field(default_factory=dict)
    total_memory_mb: float = 0.0


class AdapterStatsResponse(BaseModel):
    """适配器统计响应"""
    adapter_type: str
    connected: bool
    stats: Dict[str, Any] = Field(default_factory=dict)


# ============ Global State ============

_startup_time: float = 0.0
_dki_plugin = None
_user_adapter = None


def set_startup_time(t: float):
    global _startup_time
    _startup_time = t


def get_startup_time() -> float:
    return _startup_time


def set_dki_plugin(plugin):
    global _dki_plugin
    _dki_plugin = plugin


def get_dki_plugin():
    return _dki_plugin


def set_user_adapter(adapter):
    global _user_adapter
    _user_adapter = adapter


def get_user_adapter():
    return _user_adapter


# ============ Router ============

def create_monitoring_router() -> APIRouter:
    """创建监控 API 路由"""
    
    router = APIRouter(prefix="/v1", tags=["Monitoring"])
    
    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="健康检查",
        description="检查 DKI 系统健康状态",
    )
    async def health_check():
        """
        健康检查
        
        检查 DKI 插件及其依赖组件的健康状态
        """
        components = {}
        
        # 检查 DKI 插件
        dki = get_dki_plugin()
        if dki:
            components["dki_plugin"] = "ok"
        else:
            components["dki_plugin"] = "not_initialized"
        
        # 检查适配器
        adapter = get_user_adapter()
        if adapter:
            try:
                if await adapter.health_check():
                    components["user_adapter"] = "ok"
                else:
                    components["user_adapter"] = "degraded"
            except Exception as e:
                components["user_adapter"] = f"error: {str(e)}"
        else:
            components["user_adapter"] = "not_configured"
        
        # 检查模型
        if dki and hasattr(dki, 'model') and dki.model:
            components["model"] = "ok"
        else:
            components["model"] = "not_loaded"
        
        # 确定整体状态
        status = "healthy"
        if any("error" in v for v in components.values()):
            status = "unhealthy"
        elif any(v in ["degraded", "not_initialized", "not_configured"] for v in components.values()):
            status = "degraded"
        
        uptime = time.time() - get_startup_time() if get_startup_time() > 0 else 0
        
        return HealthResponse(
            status=status,
            version="2.0.0",
            uptime_seconds=uptime,
            components=components,
        )
    
    @router.get(
        "/stats",
        response_model=StatsResponse,
        summary="系统统计",
        description="获取 DKI 系统统计数据",
    )
    async def get_stats():
        """
        获取统计数据
        
        返回 DKI 插件的工作统计，包括:
        - 请求统计
        - 缓存统计
        - 性能统计
        """
        dki = get_dki_plugin()
        adapter = get_user_adapter()
        
        if not dki:
            return StatsResponse(
                adapter_type=type(adapter).__name__ if adapter else "none",
                adapter_connected=adapter.is_connected if adapter else False,
            )
        
        stats = dki.get_stats()
        
        return StatsResponse(
            total_requests=stats.get("total_requests", 0),
            injection_enabled_count=stats.get("injection_enabled_count", 0),
            injection_rate=stats.get("injection_rate", 0.0),
            cache_hits=stats.get("cache_hits", 0),
            cache_hit_rate=stats.get("cache_hit_rate", 0.0),
            preference_cache_size=stats.get("preference_cache_size", 0),
            avg_latency_ms=stats.get("avg_latency_ms", 0.0),
            avg_alpha=stats.get("avg_alpha", 0.0),
            adapter_type=type(adapter).__name__ if adapter else "none",
            adapter_connected=adapter.is_connected if adapter else False,
        )
    
    @router.get(
        "/logs",
        response_model=InjectionLogsResponse,
        summary="注入日志",
        description="获取 DKI 注入日志",
    )
    async def get_injection_logs(
        limit: int = Query(default=100, ge=1, le=1000, description="返回条数"),
        offset: int = Query(default=0, ge=0, description="偏移量"),
    ):
        """
        获取注入日志
        
        返回 DKI 注入操作的详细日志，用于监控和调试
        """
        dki = get_dki_plugin()
        
        if not dki:
            return InjectionLogsResponse(
                logs=[],
                total_count=0,
                offset=offset,
                limit=limit,
            )
        
        logs_data = dki.get_injection_logs(limit=limit, offset=offset)
        
        logs = []
        for log in logs_data:
            logs.append(InjectionLogEntry(
                request_id=log.get("request_id", ""),
                timestamp=datetime.fromisoformat(log.get("timestamp", datetime.utcnow().isoformat())),
                injection_enabled=log.get("injection_enabled", False),
                alpha=log.get("alpha", 0.0),
                preference_tokens=log.get("tokens", {}).get("preference", 0),
                history_tokens=log.get("tokens", {}).get("history", 0),
                query_tokens=log.get("tokens", {}).get("query", 0),
                preference_cache_hit=log.get("cache", {}).get("preference_hit", False),
                preference_cache_tier=log.get("cache", {}).get("preference_tier", "none"),
                latency_ms=log.get("latency", {}).get("total_ms", 0.0),
                adapter_latency_ms=log.get("latency", {}).get("adapter_ms", 0.0),
                injection_latency_ms=log.get("latency", {}).get("injection_ms", 0.0),
                inference_latency_ms=log.get("latency", {}).get("inference_ms", 0.0),
            ))
        
        return InjectionLogsResponse(
            logs=logs,
            total_count=len(logs),
            offset=offset,
            limit=limit,
        )
    
    @router.get(
        "/cache/stats",
        response_model=CacheStatsResponse,
        summary="缓存统计",
        description="获取缓存统计数据",
    )
    async def get_cache_stats():
        """
        获取缓存统计
        
        返回偏好 K/V 缓存和会话缓存的统计数据
        """
        dki = get_dki_plugin()
        
        if not dki:
            return CacheStatsResponse()
        
        stats = dki.get_stats()
        
        return CacheStatsResponse(
            preference_cache={
                "size": stats.get("preference_cache_size", 0),
                "hit_rate": stats.get("cache_hit_rate", 0.0),
            },
            session_cache={},
            total_memory_mb=0.0,  # TODO: 实现内存统计
        )
    
    @router.post(
        "/cache/clear",
        summary="清除缓存",
        description="清除 DKI 缓存",
    )
    async def clear_cache(
        user_id: Optional[str] = Query(default=None, description="用户 ID (可选，不指定则清除全部)"),
    ):
        """
        清除缓存
        
        清除偏好 K/V 缓存，可指定用户或清除全部
        """
        dki = get_dki_plugin()
        
        if not dki:
            raise HTTPException(status_code=503, detail="DKI plugin not initialized")
        
        dki.clear_preference_cache(user_id)
        
        return {
            "status": "ok",
            "message": f"Cache cleared for user: {user_id}" if user_id else "All cache cleared",
        }
    
    @router.get(
        "/adapter/stats",
        response_model=AdapterStatsResponse,
        summary="适配器统计",
        description="获取外部数据适配器统计",
    )
    async def get_adapter_stats():
        """
        获取适配器统计
        
        返回外部数据适配器的统计数据
        """
        adapter = get_user_adapter()
        
        if not adapter:
            return AdapterStatsResponse(
                adapter_type="none",
                connected=False,
            )
        
        stats = {}
        if hasattr(adapter, 'get_stats'):
            stats = adapter.get_stats()
        
        return AdapterStatsResponse(
            adapter_type=type(adapter).__name__,
            connected=adapter.is_connected,
            stats=stats,
        )
    
    return router
