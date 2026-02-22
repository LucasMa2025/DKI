"""
DKI Injection Visualization API Routes

DKI 注入可视化 API 路由

提供注入过程的详细可视化数据，用于调试和理解 DKI 的工作机制。

v3.2 新增:
- Function Call 日志 API (获取会话的 function call 记录)
- 可视化详情中包含 function call 信息

Author: AGI Demo Project
Version: 3.2.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from dki.api.auth_routes import get_current_user, require_auth


# ============ Visualization Data Models ============

class TokenInfo(BaseModel):
    """Token 信息"""
    text: str = Field(..., description="Token 文本")
    position: int = Field(..., description="位置索引")
    type: str = Field(..., description="类型: query/preference/history/suffix")
    attention_weight: Optional[float] = Field(None, description="注意力权重")


class InjectionLayerInfo(BaseModel):
    """注入层信息"""
    layer_name: str = Field(..., description="层名称")
    layer_type: str = Field(..., description="层类型: preference_kv/history_suffix")
    token_count: int = Field(0, description="Token 数量")
    position_range: List[int] = Field(default_factory=list, description="位置范围 [start, end]")
    alpha: float = Field(0.0, description="注入强度")
    cache_status: str = Field("none", description="缓存状态")


class AttentionVisualization(BaseModel):
    """注意力可视化数据"""
    query_tokens: List[TokenInfo] = Field(default_factory=list, description="查询 tokens")
    memory_tokens: List[TokenInfo] = Field(default_factory=list, description="记忆 tokens (偏好 K/V)")
    history_tokens: List[TokenInfo] = Field(default_factory=list, description="历史 tokens (后缀)")
    attention_matrix: Optional[List[List[float]]] = Field(None, description="注意力矩阵 (简化版)")
    

class InjectionFlowStep(BaseModel):
    """注入流程步骤"""
    step_id: int = Field(..., description="步骤 ID")
    step_name: str = Field(..., description="步骤名称")
    description: str = Field(..., description="步骤描述")
    status: str = Field("pending", description="状态: pending/running/completed/skipped")
    duration_ms: float = Field(0.0, description="耗时 (ms)")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")


class InjectionVisualizationResponse(BaseModel):
    """注入可视化响应"""
    request_id: str = Field(..., description="请求 ID")
    timestamp: str = Field(..., description="时间戳")
    
    # 模式
    mode: str = Field("dki", description="模式: dki/rag/baseline")
    
    # 输入信息
    original_query: str = Field(..., description="原始查询")
    user_id: str = Field(..., description="用户 ID")
    session_id: str = Field(..., description="会话 ID")
    
    # 注入层信息
    injection_layers: List[InjectionLayerInfo] = Field(default_factory=list, description="注入层列表")
    
    # 注入流程
    flow_steps: List[InjectionFlowStep] = Field(default_factory=list, description="注入流程步骤")
    
    # Token 分布
    token_distribution: Dict[str, int] = Field(default_factory=dict, description="Token 分布")
    
    # 注意力可视化
    attention_viz: Optional[AttentionVisualization] = Field(None, description="注意力可视化")
    
    # 最终输入
    final_input_preview: str = Field("", description="最终输入预览 (截断)")
    
    # 注入明文信息 (用于显示, 不显示实际 K/V)
    preference_text: str = Field("", description="偏好原文")
    history_suffix_text: str = Field("", description="历史后缀原文")
    history_messages: List[Dict[str, str]] = Field(default_factory=list, description="历史消息列表")
    
    # RAG 专用字段
    rag_prompt_text: str = Field("", description="RAG 完整提示词")
    rag_context_text: str = Field("", description="RAG 检索上下文")
    
    # Recall v4 信息
    recall_v4_enabled: bool = Field(False, description="是否使用 recall_v4")
    recall_strategy: str = Field("", description="召回策略")
    recall_trace_ids: List[str] = Field(default_factory=list, description="召回的 trace IDs")
    recall_fact_rounds: int = Field(0, description="fact call 轮次")
    recall_summary_count: int = Field(0, description="summary 条目数")
    recall_message_count: int = Field(0, description="原文消息数")
    
    # 性能指标
    total_latency_ms: float = Field(0.0, description="总延迟 (ms)")
    injection_overhead_ms: float = Field(0.0, description="注入开销 (ms)")


class InjectionHistoryItem(BaseModel):
    """注入历史记录"""
    request_id: str
    timestamp: str
    user_id: str
    query_preview: str
    injection_enabled: bool
    alpha: float
    preference_tokens: int
    history_tokens: int
    latency_ms: float


class InjectionHistoryResponse(BaseModel):
    """注入历史响应"""
    items: List[InjectionHistoryItem]
    total: int
    page: int
    page_size: int


# ============ Global State ============

_visualization_history: List[Dict[str, Any]] = []
_max_history = 100


def record_visualization(data: Dict[str, Any]):
    """记录可视化数据"""
    global _visualization_history
    _visualization_history.append(data)
    if len(_visualization_history) > _max_history:
        _visualization_history = _visualization_history[-_max_history:]


def get_visualization_history() -> List[Dict[str, Any]]:
    """获取可视化历史"""
    return _visualization_history


def clear_visualization_history():
    """清除可视化历史"""
    global _visualization_history
    _visualization_history = []


# ============ Router ============

def create_visualization_router() -> APIRouter:
    """创建可视化 API 路由"""
    
    router = APIRouter(prefix="/v1/dki/visualization", tags=["DKI Visualization"])
    
    @router.get(
        "/history",
        response_model=InjectionHistoryResponse,
        summary="Get injection history",
        description="Get recent DKI injection history records, filtered by authenticated user",
    )
    async def get_injection_history(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Page size"),
        user: Optional[dict] = Depends(get_current_user),
    ):
        """Get injection history filtered by authenticated user."""
        history = get_visualization_history()
        
        # User-level isolation: filter by authenticated user_id
        if user and user.get("id"):
            history = [h for h in history if h.get("user_id") == user["id"]]
        
        # Pagination
        total = len(history)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = history[start:end]
        
        items = []
        for item in page_items:
            items.append(InjectionHistoryItem(
                request_id=item.get("request_id", ""),
                timestamp=item.get("timestamp", ""),
                user_id=item.get("user_id", ""),
                query_preview=item.get("query", "")[:50] + "..." if len(item.get("query", "")) > 50 else item.get("query", ""),
                injection_enabled=item.get("injection_enabled", False),
                alpha=item.get("alpha", 0.0),
                preference_tokens=item.get("preference_tokens", 0),
                history_tokens=item.get("history_tokens", 0),
                latency_ms=item.get("latency_ms", 0.0),
            ))
        
        return InjectionHistoryResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    
    @router.get(
        "/detail/{request_id}",
        response_model=InjectionVisualizationResponse,
        summary="Get injection detail",
        description="Get detailed injection visualization data for a specific request",
    )
    async def get_injection_detail(
        request_id: str,
        user: Optional[dict] = Depends(get_current_user),
    ):
        """Get injection detail with user-level access check."""
        history = get_visualization_history()
        
        for item in history:
            if item.get("request_id") == request_id:
                # User isolation: verify ownership via authenticated user
                if user and user.get("id") and item.get("user_id") != user["id"]:
                    raise HTTPException(status_code=403, detail="Access denied: this record belongs to another user")
                return build_visualization_response(item)
        
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    @router.get(
        "/latest",
        response_model=InjectionVisualizationResponse,
        summary="Get latest injection detail",
        description="Get the most recent DKI injection visualization data, filtered by authenticated user",
    )
    async def get_latest_injection(
        user: Optional[dict] = Depends(get_current_user),
    ):
        """Get latest injection detail filtered by authenticated user."""
        history = get_visualization_history()
        
        # User-level isolation: filter by authenticated user_id
        if user and user.get("id"):
            history = [h for h in history if h.get("user_id") == user["id"]]
        
        if not history:
            raise HTTPException(status_code=404, detail="No injection history available")
        
        return build_visualization_response(history[-1])
    
    @router.get(
        "/flow-diagram",
        summary="Get injection flow diagram data",
        description="Get DKI injection flow chart data",
    )
    async def get_flow_diagram():
        """Get injection flow diagram data."""
        return {
            "nodes": [
                {"id": "input", "label": "User Input", "type": "input"},
                {"id": "adapter", "label": "External Data Adapter", "type": "process"},
                {"id": "preferences", "label": "User Preferences", "type": "data"},
                {"id": "history", "label": "History Messages", "type": "data"},
                {"id": "memory_trigger", "label": "Memory Trigger", "type": "process"},
                {"id": "reference_resolver", "label": "Reference Resolver", "type": "process"},
                {"id": "kv_injection", "label": "K/V Injection (neg pos)", "type": "injection"},
                {"id": "suffix_injection", "label": "Suffix Injection (pos pos)", "type": "injection"},
                {"id": "llm", "label": "LLM Inference", "type": "process"},
                {"id": "output", "label": "Output Response", "type": "output"},
            ],
            "edges": [
                {"from": "input", "to": "adapter", "label": "query + user_id"},
                {"from": "adapter", "to": "preferences", "label": "Load preferences"},
                {"from": "adapter", "to": "history", "label": "Retrieve history"},
                {"from": "input", "to": "memory_trigger", "label": "Detect trigger"},
                {"from": "input", "to": "reference_resolver", "label": "Resolve references"},
                {"from": "preferences", "to": "kv_injection", "label": "Encode K/V"},
                {"from": "history", "to": "suffix_injection", "label": "Format suffix"},
                {"from": "kv_injection", "to": "llm", "label": "Inject K/V"},
                {"from": "suffix_injection", "to": "llm", "label": "Append input"},
                {"from": "input", "to": "llm", "label": "Original query"},
                {"from": "llm", "to": "output", "label": "Generate response"},
            ],
            "description": {
                "title": "DKI Injection Flow",
                "summary": "DKI integrates user memory into LLM inference via hybrid injection strategy",
                "layers": [
                    {
                        "name": "L1 - Preference Layer",
                        "method": "K/V Injection (negative positions)",
                        "description": "Short-term stable user preferences, injected into attention computation via K/V cache",
                    },
                    {
                        "name": "L2 - History Layer",
                        "method": "Suffix Prompting (positive positions)",
                        "description": "Dynamic conversation history, appended to input as a suffix",
                    },
                    {
                        "name": "L3 - Query Layer",
                        "method": "Original Input",
                        "description": "The user's current query input",
                    },
                ],
            },
        }
    
    @router.delete(
        "/history",
        summary="清除注入历史",
        description="清除所有注入历史记录",
    )
    async def clear_history():
        """清除注入历史"""
        clear_visualization_history()
        return {"message": "History cleared", "success": True}
    
    # ================================================================
    # Function Call 日志 API (v3.2)
    # ================================================================
    
    @router.get(
        "/function-calls/{session_id}",
        summary="获取会话的 Function Call 日志",
        description="""
        获取指定会话的所有 Function Call 日志记录。
        
        返回每次 function call 的:
        - 函数名、参数、返回文本
        - 调用前后的 prompt 快照
        - 触发 function call 的模型输出
        - 调用状态 (success/error/budget_exceeded)
        - 耗时统计
        
        用于可视化和调试 recall_v4 的 fact call 循环。
        """,
    )
    async def get_session_function_calls(
        session_id: str,
        include_prompts: bool = Query(False, description="是否包含完整 prompt (大文本)"),
        limit: int = Query(100, ge=1, le=500, description="最大返回数量"),
    ):
        """获取会话的 Function Call 日志"""
        fc_logger = _get_function_call_logger()
        
        if fc_logger:
            try:
                logs = fc_logger.get_by_session(
                    session_id=session_id,
                    limit=limit,
                    include_prompts=include_prompts,
                )
                return {
                    "session_id": session_id,
                    "function_calls": logs,
                    "total": len(logs),
                    "include_prompts": include_prompts,
                }
            except Exception as e:
                logger.error(f"Failed to get function call logs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 无 FunctionCallLogger — 尝试从数据库直接查询
        try:
            db_logs = _query_function_calls_from_db(session_id, limit)
            return {
                "session_id": session_id,
                "function_calls": db_logs,
                "total": len(db_logs),
                "include_prompts": include_prompts,
            }
        except Exception as e:
            logger.warning(f"Failed to query function call logs from DB: {e}")
            return {
                "session_id": session_id,
                "function_calls": [],
                "total": 0,
                "include_prompts": include_prompts,
                "message": "Function call logging not available",
            }
    
    @router.get(
        "/function-calls/{session_id}/stats",
        summary="获取会话 Function Call 统计",
        description="获取指定会话的 Function Call 统计数据",
    )
    async def get_session_function_call_stats(session_id: str):
        """获取会话 Function Call 统计"""
        try:
            db_manager = _get_db_manager()
            if db_manager:
                from dki.database.repository import FunctionCallLogRepository
                with db_manager.session_scope() as db:
                    repo = FunctionCallLogRepository(db)
                    stats = repo.get_stats(session_id=session_id)
                    return {
                        "session_id": session_id,
                        "stats": stats,
                    }
        except Exception as e:
            logger.warning(f"Failed to get function call stats: {e}")
        
        return {
            "session_id": session_id,
            "stats": {
                "total": 0,
                "success": 0,
                "error": 0,
                "budget_exceeded": 0,
                "success_rate": 0,
            },
        }
    
    @router.get(
        "/detail-with-fc/{request_id}",
        summary="获取注入详情 (含 Function Call)",
        description="获取指定请求的详细注入可视化数据，包含 function call 日志",
    )
    async def get_injection_detail_with_fc(request_id: str):
        """获取注入详情 (含 Function Call 日志)"""
        history = get_visualization_history()
        
        viz_data = None
        for item in history:
            if item.get("request_id") == request_id:
                viz_data = item
                break
        
        if not viz_data:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        
        response = build_visualization_response(viz_data)
        
        # 附加 function call 日志
        session_id = viz_data.get("session_id", "")
        fc_logs = []
        
        fc_logger = _get_function_call_logger()
        if fc_logger and session_id:
            try:
                fc_logs = fc_logger.get_by_session(
                    session_id=session_id,
                    include_prompts=True,
                )
            except Exception as e:
                logger.warning(f"Failed to get FC logs for detail: {e}")
        
        # 转为 dict 并附加 fc_logs
        response_dict = response.dict()
        response_dict["function_calls"] = fc_logs
        
        return response_dict
    
    return router


# ============ Function Call Logger 辅助函数 ============

_function_call_logger = None
_db_manager_ref = None


def set_function_call_logger(fc_logger):
    """设置全局 FunctionCallLogger 引用 (由 app 启动时调用)"""
    global _function_call_logger
    _function_call_logger = fc_logger


def set_db_manager(db_manager):
    """设置全局 DatabaseManager 引用 (由 app 启动时调用)"""
    global _db_manager_ref
    _db_manager_ref = db_manager


def _get_function_call_logger():
    """获取 FunctionCallLogger"""
    return _function_call_logger


def _get_db_manager():
    """获取 DatabaseManager"""
    return _db_manager_ref


def _query_function_calls_from_db(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """从数据库直接查询 function call 日志"""
    db_manager = _get_db_manager()
    if not db_manager:
        return []
    
    from dki.database.repository import FunctionCallLogRepository
    with db_manager.session_scope() as db:
        repo = FunctionCallLogRepository(db)
        logs = repo.get_by_session(session_id, limit=limit)
        return [log.to_dict() for log in logs]


def build_visualization_response(data: Dict[str, Any]) -> InjectionVisualizationResponse:
    """构建可视化响应"""
    
    # 构建注入层信息
    injection_layers = []
    
    # 偏好层 (K/V 注入)
    if data.get("preference_tokens", 0) > 0:
        injection_layers.append(InjectionLayerInfo(
            layer_name="L1 - 用户偏好",
            layer_type="preference_kv",
            token_count=data.get("preference_tokens", 0),
            position_range=[-data.get("preference_tokens", 0), 0],  # 负位置
            alpha=data.get("alpha", 0.0),
            cache_status=data.get("cache_tier", "none"),
        ))
    
    # 历史层 (后缀注入)
    if data.get("history_tokens", 0) > 0:
        query_tokens = data.get("query_tokens", 0)
        injection_layers.append(InjectionLayerInfo(
            layer_name="L2 - 会话历史",
            layer_type="history_suffix",
            token_count=data.get("history_tokens", 0),
            position_range=[query_tokens, query_tokens + data.get("history_tokens", 0)],  # 正位置
            alpha=1.0,  # 后缀注入 alpha = 1
            cache_status="none",  # 历史不缓存
        ))
    
    # 构建流程步骤
    flow_steps = [
        InjectionFlowStep(
            step_id=1,
            step_name="接收输入",
            description="接收用户原始查询和标识",
            status="completed",
            duration_ms=0.1,
            details={"query_length": len(data.get("query", ""))},
        ),
        InjectionFlowStep(
            step_id=2,
            step_name="Memory Trigger 检测",
            description="检测是否触发记忆存储/更新",
            status="completed" if data.get("memory_triggered") else "skipped",
            duration_ms=0.5,
            details={
                "triggered": data.get("memory_triggered", False),
                "trigger_type": data.get("trigger_type"),
            },
        ),
        InjectionFlowStep(
            step_id=3,
            step_name="Reference Resolver 解析",
            description="解析指代表达，确定召回范围",
            status="completed" if data.get("reference_resolved") else "skipped",
            duration_ms=0.5,
            details={
                "resolved": data.get("reference_resolved", False),
                "reference_type": data.get("reference_type"),
            },
        ),
        InjectionFlowStep(
            step_id=4,
            step_name="读取外部数据",
            description="通过适配器读取用户偏好和历史消息",
            status="completed",
            duration_ms=data.get("adapter_latency_ms", 0),
            details={
                "preferences_count": data.get("preferences_count", 0),
                "history_count": data.get("relevant_history_count", 0),
            },
        ),
        InjectionFlowStep(
            step_id=5,
            step_name="偏好 K/V 注入",
            description="将用户偏好编码为 K/V 并注入到负位置",
            status="completed" if data.get("preference_tokens", 0) > 0 else "skipped",
            duration_ms=data.get("injection_latency_ms", 0) * 0.6,
            details={
                "tokens": data.get("preference_tokens", 0),
                "cache_hit": data.get("cache_hit", False),
                "alpha": data.get("alpha", 0),
            },
        ),
        InjectionFlowStep(
            step_id=6,
            step_name="历史后缀注入",
            description="将相关历史格式化为后缀并拼接到输入",
            status="completed" if data.get("history_tokens", 0) > 0 else "skipped",
            duration_ms=data.get("injection_latency_ms", 0) * 0.4,
            details={
                "tokens": data.get("history_tokens", 0),
                "messages_count": data.get("relevant_history_count", 0),
            },
        ),
        InjectionFlowStep(
            step_id=7,
            step_name="LLM 推理",
            description="调用 LLM 进行推理生成",
            status="completed",
            duration_ms=data.get("inference_latency_ms", 0),
            details={
                "input_tokens": data.get("total_tokens", 0),
            },
        ),
    ]
    
    # Token 分布
    token_distribution = {
        "query": data.get("query_tokens", 0),
        "preference": data.get("preference_tokens", 0),
        "history": data.get("history_tokens", 0),
        "total": data.get("total_tokens", 0),
    }
    
    # 最终输入预览 (保留更多内容以便调试分析)
    final_input = data.get("final_input", data.get("query", ""))
    final_input_preview = final_input[:2000] + "..." if len(final_input) > 2000 else final_input
    
    # 判断模式
    request_id = data.get("request_id", "")
    mode = "dki"
    if request_id.startswith("chat-rag-") or request_id.startswith("rag-"):
        mode = "rag"
    elif request_id.startswith("chat-baseline-") or request_id.startswith("baseline-"):
        mode = "baseline"
    # 允许显式设置
    mode = data.get("mode", mode)
    
    # Recall v4 信息
    recall_v4_info = data.get("recall_v4", {})
    recall_v4_enabled = recall_v4_info.get("enabled", False)
    recall_strategy = recall_v4_info.get("strategy", data.get("recall_strategy", ""))
    recall_trace_ids = recall_v4_info.get("trace_ids", data.get("trace_ids", []))
    recall_fact_rounds = recall_v4_info.get("fact_rounds_used", data.get("fact_rounds_used", 0))
    recall_summary_count = data.get("summary_count", 0)
    recall_message_count = data.get("message_count", 0)
    
    return InjectionVisualizationResponse(
        request_id=request_id,
        timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        mode=mode,
        original_query=data.get("query", ""),
        user_id=data.get("user_id", ""),
        session_id=data.get("session_id", ""),
        injection_layers=injection_layers,
        flow_steps=flow_steps,
        token_distribution=token_distribution,
        attention_viz=None,  # 简化版不包含完整注意力矩阵
        final_input_preview=final_input_preview,
        # 注入明文信息
        preference_text=data.get("preference_text", ""),
        history_suffix_text=data.get("history_suffix_text", ""),
        history_messages=data.get("history_messages", []),
        # RAG 专用
        rag_prompt_text=data.get("rag_prompt_text", data.get("final_input", "")),
        rag_context_text=data.get("rag_context_text", ""),
        # Recall v4 信息
        recall_v4_enabled=recall_v4_enabled,
        recall_strategy=recall_strategy,
        recall_trace_ids=recall_trace_ids,
        recall_fact_rounds=recall_fact_rounds,
        recall_summary_count=recall_summary_count,
        recall_message_count=recall_message_count,
        # 性能指标
        total_latency_ms=data.get("latency_ms", 0),
        injection_overhead_ms=data.get("adapter_latency_ms", 0) + data.get("injection_latency_ms", 0),
    )
