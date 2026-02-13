"""
DKI Injection Visualization API Routes

DKI 注入可视化 API 路由

提供注入过程的详细可视化数据，用于调试和理解 DKI 的工作机制

Author: AGI Demo Project
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger


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
        summary="获取注入历史",
        description="获取最近的 DKI 注入历史记录",
    )
    async def get_injection_history(
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    ):
        """获取注入历史"""
        history = get_visualization_history()
        
        # 分页
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
        summary="获取注入详情",
        description="获取指定请求的详细注入可视化数据",
    )
    async def get_injection_detail(request_id: str):
        """获取注入详情"""
        history = get_visualization_history()
        
        for item in history:
            if item.get("request_id") == request_id:
                return build_visualization_response(item)
        
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    @router.get(
        "/latest",
        response_model=InjectionVisualizationResponse,
        summary="获取最新注入详情",
        description="获取最近一次 DKI 注入的详细可视化数据",
    )
    async def get_latest_injection():
        """获取最新注入详情"""
        history = get_visualization_history()
        
        if not history:
            raise HTTPException(status_code=404, detail="No injection history available")
        
        return build_visualization_response(history[-1])
    
    @router.get(
        "/flow-diagram",
        summary="获取注入流程图数据",
        description="获取 DKI 注入流程的图表数据",
    )
    async def get_flow_diagram():
        """获取注入流程图数据"""
        return {
            "nodes": [
                {"id": "input", "label": "用户输入", "type": "input"},
                {"id": "adapter", "label": "外部数据适配器", "type": "process"},
                {"id": "preferences", "label": "用户偏好", "type": "data"},
                {"id": "history", "label": "历史消息", "type": "data"},
                {"id": "memory_trigger", "label": "Memory Trigger", "type": "process"},
                {"id": "reference_resolver", "label": "Reference Resolver", "type": "process"},
                {"id": "kv_injection", "label": "K/V 注入 (负位置)", "type": "injection"},
                {"id": "suffix_injection", "label": "后缀注入 (正位置)", "type": "injection"},
                {"id": "llm", "label": "LLM 推理", "type": "process"},
                {"id": "output", "label": "输出响应", "type": "output"},
            ],
            "edges": [
                {"from": "input", "to": "adapter", "label": "query + user_id"},
                {"from": "adapter", "to": "preferences", "label": "读取偏好"},
                {"from": "adapter", "to": "history", "label": "检索历史"},
                {"from": "input", "to": "memory_trigger", "label": "检测触发"},
                {"from": "input", "to": "reference_resolver", "label": "解析指代"},
                {"from": "preferences", "to": "kv_injection", "label": "编码 K/V"},
                {"from": "history", "to": "suffix_injection", "label": "格式化后缀"},
                {"from": "kv_injection", "to": "llm", "label": "注入 K/V"},
                {"from": "suffix_injection", "to": "llm", "label": "拼接输入"},
                {"from": "input", "to": "llm", "label": "原始查询"},
                {"from": "llm", "to": "output", "label": "生成响应"},
            ],
            "description": {
                "title": "DKI 注入流程",
                "summary": "DKI 通过混合注入策略将用户记忆融入 LLM 推理",
                "layers": [
                    {
                        "name": "L1 - 偏好层",
                        "method": "K/V 注入 (负位置)",
                        "description": "短期稳定的用户偏好，通过 K/V 缓存注入到注意力计算中",
                    },
                    {
                        "name": "L2 - 历史层",
                        "method": "后缀提示词 (正位置)",
                        "description": "动态的会话历史，作为后缀拼接到输入中",
                    },
                    {
                        "name": "L3 - 查询层",
                        "method": "原始输入",
                        "description": "用户当前的查询输入",
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
    
    return router


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
    
    # 最终输入预览
    final_input = data.get("final_input", data.get("query", ""))
    final_input_preview = final_input[:200] + "..." if len(final_input) > 200 else final_input
    
    return InjectionVisualizationResponse(
        request_id=data.get("request_id", ""),
        timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
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
        # 性能指标
        total_latency_ms=data.get("latency_ms", 0),
        injection_overhead_ms=data.get("adapter_latency_ms", 0) + data.get("injection_latency_ms", 0),
    )
