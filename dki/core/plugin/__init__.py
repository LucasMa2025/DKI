"""
DKI Plugin Sub-Components

DKI 插件内部子组件，实现"决策-执行"分离架构。

架构:
    上层应用
       ↓ (稳定通信协议: chat(query, user_id, session_id))
    DKIPlugin (瘦 Facade, 对外接口不变)
       ↓
    InjectionPlanner (纯决策, 不碰模型)
       ↓ InjectionPlan (中间产物)
    InjectionExecutor (纯执行, 不做决策)
       ↓
    ModelAdapter (LLM 推理)

对外不需要直接使用这些子组件，它们由 DKIPlugin 内部编排。
仅供高级用户或测试使用。
"""

from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    SafetyEnvelope,
    QueryContext,
    ExecutionResult,
)

from dki.core.plugin.injection_planner import InjectionPlanner
from dki.core.plugin.injection_executor import InjectionExecutor

__all__ = [
    # 数据结构
    "InjectionPlan",
    "AlphaProfile",
    "SafetyEnvelope",
    "QueryContext",
    "ExecutionResult",
    # 组件
    "InjectionPlanner",
    "InjectionExecutor",
]
