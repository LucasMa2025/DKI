"""
DKI Plugin API Routes

DKI 插件 API 路由

核心设计:
- 上层应用只需传递 user_id 和原始输入
- DKI 自动通过适配器读取用户偏好和历史消息
- DKI 执行 K/V 注入后调用 LLM 推理

Author: AGI Demo Project
Version: 2.0.0
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

# 导入可视化记录函数
from dki.api.visualization_routes import record_visualization
# 导入统计记录函数
from dki.api.stats_routes import record_dki_request


# ============ Request/Response Models ============

class DKIChatRequest(BaseModel):
    """
    DKI 聊天请求
    
    上层应用只需传递:
    - query: 原始用户输入 (不含任何 prompt 构造)
    - user_id: 用户标识 (DKI 用于读取偏好和历史)
    - session_id: 会话标识 (DKI 用于读取会话历史)
    
    DKI 会自动:
    1. 通过适配器读取用户偏好 → K/V 注入 (负位置)
    2. 通过适配器检索相关历史 → 后缀提示词 (正位置)
    3. 调用 LLM 推理
    """
    # 原始用户输入 (不含任何 prompt 构造)
    query: str = Field(..., description="原始用户输入，不含任何 prompt 构造")
    
    # 用户标识 - DKI 用于读取偏好和历史
    user_id: str = Field(..., description="用户标识，DKI 用于读取偏好和历史")
    
    # 会话标识 - DKI 用于读取会话历史
    session_id: Optional[str] = Field(None, description="会话标识，DKI 用于读取会话历史")
    
    # 可选参数
    model: Optional[str] = Field(None, description="模型名称")
    temperature: float = Field(0.7, ge=0, le=2, description="采样温度")
    max_tokens: int = Field(512, ge=1, le=4096, description="最大生成 token 数")
    
    # 高级参数 (可选)
    force_alpha: Optional[float] = Field(None, ge=0, le=1, description="强制 alpha 值")
    use_hybrid: bool = Field(True, description="是否使用混合注入策略")


class DKIMetadataResponse(BaseModel):
    """DKI 元数据响应"""
    injection_enabled: bool = Field(False, description="是否启用注入")
    alpha: float = Field(0.0, description="注入强度")
    preference_tokens: int = Field(0, description="偏好 token 数")
    history_tokens: int = Field(0, description="历史 token 数")
    cache_hit: bool = Field(False, description="缓存是否命中")
    cache_tier: str = Field("none", description="缓存层级")
    latency_ms: float = Field(0, description="延迟 (ms)")


class DKIChatResponse(BaseModel):
    """DKI 聊天响应"""
    id: str = Field(..., description="响应 ID")
    text: str = Field(..., description="生成的文本")
    
    # Token 统计
    input_tokens: int = Field(0, description="输入 token 数")
    output_tokens: int = Field(0, description="输出 token 数")
    
    # DKI 元数据 (用于调试和监控)
    dki_metadata: DKIMetadataResponse = Field(default_factory=DKIMetadataResponse, description="DKI 元数据")
    
    # 兼容 OpenAI 格式
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 时间戳
    created: int = Field(..., description="创建时间戳")


# ============ Global State ============

_dki_system = None
_executor = ThreadPoolExecutor(max_workers=4)


def set_dki_plugin(dki_system):
    """设置 DKI 系统实例"""
    global _dki_system
    _dki_system = dki_system


def get_dki_plugin():
    """获取 DKI 系统实例"""
    return _dki_system


# ============ Router ============

def create_dki_router() -> APIRouter:
    """创建 DKI API 路由"""
    
    router = APIRouter(prefix="/v1/dki", tags=["DKI"])
    
    @router.post(
        "/chat",
        response_model=DKIChatResponse,
        summary="DKI 增强聊天",
        description="""
        DKI 增强的聊天接口
        
        上层应用只需传递:
        - query: 原始用户输入 (不含任何 prompt 构造)
        - user_id: 用户标识
        - session_id: 会话标识 (可选)
        
        DKI 会自动:
        1. 通过适配器读取用户偏好 → K/V 注入 (负位置)
        2. 通过适配器检索相关历史 → 后缀提示词 (正位置)
        3. 调用 LLM 推理
        """,
    )
    async def dki_chat(request: DKIChatRequest):
        """
        DKI 增强聊天
        
        核心流程:
        1. 接收原始用户输入 (不含任何 prompt 构造)
        2. DKI 通过适配器读取用户偏好和历史
        3. DKI 执行 K/V 注入
        4. DKI 调用 LLM 推理
        5. 返回响应
        """
        dki_system = get_dki_plugin()
        
        if not dki_system:
            raise HTTPException(
                status_code=503,
                detail="DKI system not initialized. Please check configuration."
            )
        
        try:
            # 使用线程池执行同步的 DKI chat 方法
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: dki_system.chat(
                    query=request.query,  # 原始用户输入
                    session_id=request.session_id or request.user_id,
                    user_id=request.user_id,
                    force_alpha=request.force_alpha,
                    use_hybrid=request.use_hybrid,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            
            # 构造响应
            # 从 DKIResponse 中提取元数据
            hybrid_info = response.metadata.get("hybrid_injection", {})
            preference_tokens = hybrid_info.get("preference_tokens", 0)
            history_tokens = hybrid_info.get("history_tokens", 0)
            
            dki_metadata = DKIMetadataResponse(
                injection_enabled=response.gating_decision.should_inject if response.gating_decision else False,
                alpha=response.gating_decision.alpha if response.gating_decision else 0.0,
                preference_tokens=preference_tokens,
                history_tokens=history_tokens,
                cache_hit=response.cache_hit,
                cache_tier=response.cache_tier or "none",
                latency_ms=response.latency_ms,
            )
            
            request_id = f"dki-{uuid.uuid4().hex[:8]}"
            
            # 记录可视化数据
            try:
                # LatencyBreakdown 属性: router_ms, gating_ms, kv_compute_ms,
                # kv_load_ms, projection_ms, prefill_ms, decode_ms, total_ms
                lb = response.latency_breakdown
                viz_data = {
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "mode": "dki",
                    "query": request.query,
                    "user_id": request.user_id,
                    "session_id": request.session_id or request.user_id,
                    "injection_enabled": response.gating_decision.should_inject if response.gating_decision else False,
                    "alpha": response.gating_decision.alpha if response.gating_decision else 0.0,
                    "preference_tokens": preference_tokens,
                    "history_tokens": history_tokens,
                    "query_tokens": max(0, response.input_tokens - preference_tokens - history_tokens),
                    "total_tokens": response.input_tokens,
                    "cache_hit": response.cache_hit,
                    "cache_tier": response.cache_tier or "none",
                    "latency_ms": response.latency_ms,
                    # 注入明文信息 (用于显示)
                    "preference_text": hybrid_info.get("preference_text", ""),
                    "history_suffix_text": hybrid_info.get("history_suffix_text", ""),
                    "history_messages": hybrid_info.get("history_messages", []),
                    "final_input": hybrid_info.get("final_input", request.query),
                    "rag_prompt_text": "",
                    "rag_context_text": "",
                    # 延迟分解 (使用 LatencyBreakdown 正确属性)
                    "adapter_latency_ms": (lb.router_ms if lb else 0),
                    "injection_latency_ms": ((lb.kv_compute_ms + lb.projection_ms) if lb else 0),
                    "inference_latency_ms": ((lb.prefill_ms + lb.decode_ms) if lb else 0),
                }
                record_visualization(viz_data)
                logger.debug(f"Recorded visualization data for request {request_id}")
            except Exception as viz_error:
                logger.warning(f"Failed to record visualization: {viz_error}")
            
            # 记录 DKI 统计数据
            try:
                _alpha = response.gating_decision.alpha if response.gating_decision else 0.0
                _injected = response.gating_decision.should_inject if response.gating_decision else False
                record_dki_request(
                    cache_tier=response.cache_tier or "L3",
                    alpha=_alpha,
                    injected=_injected,
                )
            except Exception as stats_error:
                logger.warning(f"Failed to record stats: {stats_error}")
            
            return DKIChatResponse(
                id=request_id,
                text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                dki_metadata=dki_metadata,
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop",
                }],
                created=int(datetime.utcnow().timestamp()),
            )
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"DKI chat error: {e}\n{error_detail}")
            raise HTTPException(status_code=500, detail=f"DKI chat error: {str(e)}")
    
    @router.get(
        "/info",
        summary="DKI 插件信息",
        description="获取 DKI 插件的配置和状态信息",
    )
    async def dki_info():
        """获取 DKI 插件信息"""
        dki_system = get_dki_plugin()
        
        if not dki_system:
            return {
                "status": "not_initialized",
                "message": "DKI system not initialized",
            }
        
        try:
            stats = dki_system.get_stats()
            return {
                "status": "ready",
                "version": "2.0.0",
                "stats": stats,
                "config": {
                    "hybrid_injection_enabled": True,
                    "preference_kv_injection": True,
                    "history_suffix_injection": True,
                },
            }
        except Exception as e:
            logger.error(f"DKI info error: {e}")
            return {
                "status": "error",
                "message": str(e),
            }
    
    return router
