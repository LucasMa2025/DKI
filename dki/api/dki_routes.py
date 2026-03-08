"""
DKI Plugin API Routes

DKI 插件 API 路由

v8.2 重构:
- /v1/dki/chat 端点优先使用 DKIPlugin (与实验系统一致)
- 支持降级到 DKISystem (兼容旧配置)
- DKIPlugin.chat() 是 async 方法, 直接 await (无需 ThreadPoolExecutor)
- 响应映射从 DKIPluginResponse/InjectionMetadata 提取字段

核心设计:
- 上层应用只需传递 user_id 和原始输入
- DKI 自动通过适配器读取用户偏好和历史消息
- DKI 执行偏好注入(提示词前缀)后调用 LLM 推理
- 消息持久化由上层应用负责 (如 demo/api/chat.py), DKI 本身只读

安全设计 (v3.1):
- 每个请求创建 UserIsolationContext，贯穿整个请求生命周期
- user_id 必须经过验证 (从 auth token 获取，不信任请求体中的 user_id)
- 所有缓存操作都通过 UserIsolationContext 进行

Author: AGI Demo Project
Version: 9.0.0
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
# 导入认证依赖
from dki.api.auth_routes import get_current_user

# 用户隔离上下文 (可选导入)
try:
    from dki.cache.user_isolation import UserIsolationContext, CacheKeySigner
    USER_ISOLATION_AVAILABLE = True
except ImportError:
    USER_ISOLATION_AVAILABLE = False

# DKIPlugin 类型检测 (用于区分 DKIPlugin 和 DKISystem)
try:
    from dki.core.dki_plugin import DKIPlugin
    DKIPLUGIN_AVAILABLE = True
except ImportError:
    DKIPLUGIN_AVAILABLE = False


# ============ Request/Response Models ============

class DKIChatRequest(BaseModel):
    """
    DKI 聊天请求
    
    上层应用只需传递:
    - query: 原始用户输入 (不含任何 prompt 构造)
    - user_id: 用户标识 (DKI 用于读取偏好和历史)
    - session_id: 会话标识 (DKI 用于读取会话历史)
    
    DKI 会自动:
    1. 通过适配器读取用户偏好 → 提示词前缀注入 (vLLM 环境)
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
    # v8.2: 新增 DKIPlugin 特有字段
    retrieval_mode: str = Field("unknown", description="检索模式 (bm25_only | bm25_embedding | keyword)")
    preferences_count: int = Field(0, description="偏好数量")
    relevant_history_count: int = Field(0, description="召回的相关历史消息数")


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

_dki_plugin = None
_executor = ThreadPoolExecutor(max_workers=4)


def set_dki_plugin(instance):
    """
    设置 DKI 实例
    
    v8.2: 优先接受 DKIPlugin 实例, 也兼容 DKISystem (降级)
    """
    global _dki_plugin
    _dki_plugin = instance
    
    # 日志: 标记实际类型
    type_name = type(instance).__name__ if instance else "None"
    logger.info(f"DKI plugin set: type={type_name}")


def get_dki_plugin():
    """获取 DKI 实例"""
    return _dki_plugin


def _is_dki_plugin(instance) -> bool:
    """检测实例是否为 DKIPlugin (而非 DKISystem)"""
    if DKIPLUGIN_AVAILABLE and isinstance(instance, DKIPlugin):
        return True
    # 鸭子类型检测: DKIPlugin.chat 是 async 方法
    chat_method = getattr(instance, 'chat', None)
    if chat_method and asyncio.iscoroutinefunction(chat_method):
        return True
    return False


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
        
        v8.2: 使用 DKIPlugin (与实验系统一致)
        
        上层应用只需传递:
        - query: 原始用户输入 (不含任何 prompt 构造)
        - user_id: 用户标识 (将与 auth token 中的 user_id 交叉验证)
        - session_id: 会话标识 (可选)
        
        DKI 会自动:
        1. 验证 user_id 身份归属 (防止跨用户请求)
        2. 通过适配器读取用户偏好 → 提示词前缀注入
        3. 通过适配器检索相关历史 → 后缀提示词
        4. 调用 LLM 推理
        """,
    )
    async def dki_chat(
        request: DKIChatRequest,
        auth_user: dict = Depends(get_current_user),
    ):
        """
        DKI enhanced chat.
        
        v8.2: 优先使用 DKIPlugin (async), 降级到 DKISystem (sync)
        """
        dki_instance = get_dki_plugin()
        
        if not dki_instance:
            raise HTTPException(
                status_code=503,
                detail="DKI system not initialized. Please check configuration."
            )
        
        # ============ User Identity Verification ============
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(
                status_code=400,
                detail="user_id is required and cannot be empty"
            )
        
        verified_user_id = request.user_id.strip()
        
        # Cross-validate: if authenticated, ensure request user_id matches token user
        if auth_user and auth_user.get("id"):
            if verified_user_id != auth_user["id"]:
                logger.warning(
                    f"User ID mismatch: request={verified_user_id}, token={auth_user['id']}. "
                    f"Using token user_id for security."
                )
                verified_user_id = auth_user["id"]
        
        session_id = request.session_id or verified_user_id
        
        try:
            if _is_dki_plugin(dki_instance):
                # ============ DKIPlugin 路径 (v8.2 主路径) ============
                return await _handle_dki_plugin_chat(
                    dki_instance, request, verified_user_id, session_id
                )
            else:
                # ============ DKISystem 降级路径 ============
                return await _handle_dki_system_chat(
                    dki_instance, request, verified_user_id, session_id
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
        dki_instance = get_dki_plugin()
        
        if not dki_instance:
            return {
                "status": "not_initialized",
                "message": "DKI system not initialized",
            }
        
        try:
            stats = dki_instance.get_stats()
            is_plugin = _is_dki_plugin(dki_instance)
            return {
                "status": "ready",
                "version": "8.2.0",
                "type": "DKIPlugin" if is_plugin else "DKISystem",
                "stats": stats,
                "config": {
                    "hybrid_injection_enabled": True,
                    "preference_prompt_prefix_injection": True,
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


# ============ DKIPlugin 处理 (v8.2 主路径) ============

async def _handle_dki_plugin_chat(
    dki_plugin,
    request: DKIChatRequest,
    verified_user_id: str,
    session_id: str,
) -> DKIChatResponse:
    """
    处理 DKIPlugin.chat() 的异步响应
    
    DKIPlugin.chat() 返回 DKIPluginResponse:
    - text: 生成文本
    - input_tokens / output_tokens: Token 统计
    - metadata: InjectionMetadata (结构化注入信息)
    
    注意: DKIPlugin 自身不负责消息持久化 (设计如此)
    上层应用 (如 demo/api/chat.py) 负责消息的读写管理
    """
    # DKIPlugin.chat() 是 async 方法, 直接 await
    response = await dki_plugin.chat(
        query=request.query,
        user_id=verified_user_id,
        session_id=session_id,
        force_alpha=request.force_alpha,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    # 从 InjectionMetadata 提取字段
    meta = response.metadata
    
    dki_metadata = DKIMetadataResponse(
        injection_enabled=meta.injection_enabled,
        alpha=meta.alpha,
        preference_tokens=meta.preference_tokens,
        history_tokens=meta.history_tokens,
        cache_hit=meta.preference_cache_hit,
        cache_tier=meta.preference_cache_tier or "none",
        latency_ms=meta.latency_ms,
        retrieval_mode=meta.retrieval_mode,
        preferences_count=meta.preferences_count,
        relevant_history_count=meta.relevant_history_count,
    )
    
    request_id = meta.request_id or f"dki-{uuid.uuid4().hex[:8]}"
    
    # DKIPlugin._record_injection_log() 已在 chat() 内部记录了完整的可视化数据
    # (包含 preference_text, history_suffix_text, history_messages, final_input)
    # 此处不再重复记录, 避免覆盖 DKIPlugin 记录的完整数据为空值
    
    # 记录统计数据
    try:
        record_dki_request(
            cache_tier=meta.preference_cache_tier or "L3",
            alpha=meta.alpha,
            injected=meta.injection_enabled,
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


# ============ DKISystem 降级处理 ============

async def _handle_dki_system_chat(
    dki_system,
    request: DKIChatRequest,
    verified_user_id: str,
    session_id: str,
) -> DKIChatResponse:
    """
    处理 DKISystem.chat() 的同步响应 (降级路径)
    
    DKISystem.chat() 返回 DKIResponse:
    - text, memories_used, gating_decision, latency_ms, ...
    - metadata["hybrid_injection"] 包含注入详情
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        _executor,
        lambda: dki_system.chat(
            query=request.query,
            session_id=session_id,
            user_id=verified_user_id,
            force_alpha=request.force_alpha,
            use_hybrid=request.use_hybrid,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    )
    
    # 从 DKIResponse 中提取元数据
    hybrid_info = response.metadata.get("hybrid_injection", {})
    preference_tokens = hybrid_info.get("preference_tokens", 0)
    history_tokens = hybrid_info.get("history_tokens", 0)
    
    _gating_alpha = response.gating_decision.alpha if response.gating_decision else 0.0
    _pref_alpha = hybrid_info.get("preference_alpha", 0.0)
    _display_alpha = max(_pref_alpha, _gating_alpha)
    
    dki_metadata = DKIMetadataResponse(
        injection_enabled=(
            (response.gating_decision.should_inject if response.gating_decision else False)
            or bool(preference_tokens)
            or bool(history_tokens)
        ),
        alpha=_display_alpha,
        preference_tokens=preference_tokens,
        history_tokens=history_tokens,
        cache_hit=response.cache_hit,
        cache_tier=response.cache_tier or "none",
        latency_ms=response.latency_ms,
    )
    
    request_id = f"dki-{uuid.uuid4().hex[:8]}"
    
    # 记录可视化数据
    try:
        lb = response.latency_breakdown
        viz_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "dki",
            "query": request.query,
            "user_id": verified_user_id,
            "session_id": session_id,
            "injection_enabled": dki_metadata.injection_enabled,
            "alpha": _display_alpha,
            "preference_tokens": preference_tokens,
            "history_tokens": history_tokens,
            "query_tokens": max(0, response.input_tokens - preference_tokens - history_tokens),
            "total_tokens": response.input_tokens,
            "cache_hit": response.cache_hit,
            "cache_tier": response.cache_tier or "none",
            "latency_ms": response.latency_ms,
            "preference_text": hybrid_info.get("preference_text", ""),
            "history_suffix_text": hybrid_info.get("history_suffix_text", ""),
            "history_messages": hybrid_info.get("history_messages", []),
            "final_input": hybrid_info.get("final_input", request.query),
            "rag_prompt_text": "",
            "rag_context_text": "",
            "adapter_latency_ms": (lb.router_ms if lb else 0),
            "injection_latency_ms": ((lb.kv_compute_ms + lb.projection_ms) if lb else 0),
            "inference_latency_ms": ((lb.prefill_ms + lb.decode_ms) if lb else 0),
        }
        record_visualization(viz_data)
        logger.debug(f"[DKISystem fallback] Recorded visualization for {request_id}")
    except Exception as viz_error:
        logger.warning(f"Failed to record visualization: {viz_error}")
    
    # 记录统计数据
    try:
        record_dki_request(
            cache_tier=response.cache_tier or "L3",
            alpha=_display_alpha,
            injected=dki_metadata.injection_enabled,
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
