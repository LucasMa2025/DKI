"""
DKI Injection Plan - 注入计划数据结构

决策层与执行层之间的中间产物 (Intermediate Representation)。
将 DKI 的"决定做什么"与"怎么做"彻底分离。

设计原则:
- InjectionPlan 是纯数据结构，不包含任何逻辑
- Planner 只负责生成 InjectionPlan
- Executor 只负责执行 InjectionPlan
- 任何环节都可以独立测试

核心数据结构:
- AlphaProfile: 分层 alpha 控制 (偏好/历史/安全上限)
- SafetyEnvelope: 策略安全边界
- InjectionPlan: 决策层输出
- ExecutionResult: 执行层输出
- QueryContext: 查询分析上下文

Author: AGI Demo Project
Version: 3.0.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# AlphaProfile - 分层 Alpha 控制
# ============================================================

@dataclass
class AlphaProfile:
    """
    分层 Alpha 控制
    
    将单一 alpha float 升级为结构化的分层控制，
    匹配 DKI 的三层语义模型:
    
    | 层         | 语义               | 稳定性要求 |
    |-----------|-------------------|-----------|
    | Preference | Personality / Bias | 极高      |
    | History    | Episodic Memory    | 中        |
    | Query      | Current Intent     | 最高优先级 |
    
    安全不变量:
    - preference_alpha 永远不超过 override_cap
    - Key tensor 永远不被 alpha 缩放 (由 Executor 保证)
    """
    preference_alpha: float = 0.4    # 偏好注入强度 (K/V)
    history_alpha: float = 1.0       # 历史注入强度 (suffix=1.0, kv<1.0)
    override_cap: float = 0.7        # 偏好 alpha 安全上限
    
    @property
    def effective_preference_alpha(self) -> float:
        """确保偏好 alpha 不超过安全上限"""
        return min(self.preference_alpha, self.override_cap)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_alpha": self.preference_alpha,
            "history_alpha": self.history_alpha,
            "override_cap": self.override_cap,
            "effective_preference_alpha": self.effective_preference_alpha,
        }


# ============================================================
# SafetyEnvelope - 策略安全边界
# ============================================================

@dataclass
class SafetyEnvelope:
    """
    安全边界
    
    不同策略下的参数安全约束。
    违规情况会被记录到 InjectionPlan.safety_violations，
    但不会阻止执行 (仅告警)。
    
    用途:
    - 防止实验参数意外泄漏到生产
    - 防止 preference 覆盖 query
    - 限制 K/V token 总量
    """
    # Stable 策略安全限制
    stable_max_preference_alpha: float = 0.5
    stable_history_alpha_must_be: float = 1.0   # suffix 必须为 1.0
    
    # Full Attention 策略安全限制
    full_attention_max_preference_alpha: float = 0.7
    full_attention_max_history_alpha: float = 0.5
    
    # 通用限制
    max_total_kv_tokens: int = 600
    
    def validate(self, plan: "InjectionPlan") -> List[str]:
        """
        验证计划是否在安全边界内
        
        Returns:
            违规说明列表 (空 = 通过)
        """
        violations = []
        
        if plan.strategy == "stable":
            if plan.alpha_profile.preference_alpha > self.stable_max_preference_alpha:
                violations.append(
                    f"Stable: preference_alpha={plan.alpha_profile.preference_alpha:.2f} "
                    f"> max={self.stable_max_preference_alpha:.2f}"
                )
        elif plan.strategy == "full_attention":
            if plan.alpha_profile.preference_alpha > self.full_attention_max_preference_alpha:
                violations.append(
                    f"FullAttention: preference_alpha="
                    f"{plan.alpha_profile.preference_alpha:.2f} "
                    f"> max={self.full_attention_max_preference_alpha:.2f}"
                )
            if plan.alpha_profile.history_alpha > self.full_attention_max_history_alpha:
                violations.append(
                    f"FullAttention: history_alpha="
                    f"{plan.alpha_profile.history_alpha:.2f} "
                    f"> max={self.full_attention_max_history_alpha:.2f}"
                )
        
        return violations


# ============================================================
# QueryContext - 查询分析上下文
# ============================================================

@dataclass
class QueryContext:
    """
    查询分析结果
    
    由 Planner.analyze_query() 生成，
    用于在数据加载之前确定召回范围。
    
    典型流程:
    1. context = planner.analyze_query(query)
    2. history = adapter.search(limit=context.recall_limit)
    3. plan = planner.build_plan(query, prefs, history, context)
    """
    recall_limit: int = 10
    
    # Memory Trigger
    memory_triggered: bool = False
    trigger_type: Optional[str] = None
    trigger_confidence: float = 0.0
    
    # Reference Resolver
    reference_resolved: bool = False
    reference_type: Optional[str] = None
    reference_scope: Optional[str] = None


# ============================================================
# InjectionPlan - 注入计划 (核心中间产物)
# ============================================================

@dataclass
class InjectionPlan:
    """
    注入计划 - 决策与执行之间的中间产物
    
    Planner 生成 InjectionPlan，Executor 消费 InjectionPlan。
    
    核心不变量:
    - strategy 决定执行路径
    - injection_enabled 决定是否注入
    - alpha_profile 控制注入强度 (分层)
    - final_input 是模型最终接收的输入文本
    
    可独立测试:
    ```python
    plan = InjectionPlan(
        strategy="stable",
        injection_enabled=True,
        preference_text="素食主义者",
        alpha_profile=AlphaProfile(preference_alpha=0.3),
    )
    assert plan.alpha_profile.effective_preference_alpha == 0.3
    ```
    """
    # ============ 策略 ============
    strategy: str = "stable"          # stable | full_attention | none
    
    # ============ 偏好数据 (K/V 注入) ============
    preference_text: str = ""
    preferences_count: int = 0
    preference_tokens: int = 0
    
    # ============ 历史数据 ============
    # Stable: 格式化后的 suffix prompt (始终准备, 用于 fallback)
    history_suffix: str = ""
    # Full Attention: 原始消息列表 (供 executor 转换为 K/V)
    history_messages: List[Dict[str, Any]] = field(default_factory=list)
    history_tokens: int = 0
    relevant_history_count: int = 0
    
    # ============ 查询 ============
    user_id: str = ""                 # 用于 K/V 缓存键
    original_query: str = ""          # 原始用户输入
    final_input: str = ""             # 最终发给模型的输入
    query_tokens: int = 0
    total_tokens: int = 0
    
    # ============ Alpha 控制 (分层) ============
    alpha_profile: AlphaProfile = field(default_factory=AlphaProfile)
    
    # ============ 注入决策 ============
    injection_enabled: bool = False
    gating_decision: Dict[str, Any] = field(default_factory=dict)
    
    # ============ Memory Trigger ============
    memory_triggered: bool = False
    trigger_type: Optional[str] = None
    
    # ============ Reference Resolver ============
    reference_resolved: bool = False
    reference_type: Optional[str] = None
    reference_scope: Optional[str] = None
    recall_limit: int = 10
    
    # ============ Full Attention 特有 ============
    global_indication: str = ""
    full_attention_fallback: bool = False
    history_kv_tokens: int = 0
    
    # ============ Recall v4 特有 ============
    # 由 recall_v4 策略生成的组装后缀 (替代 history_suffix)
    assembled_suffix: str = ""
    recall_strategy: str = ""             # summary_with_fact_call | flat_history
    summary_count: int = 0                # suffix 中的 summary 条目数
    message_count: int = 0                # suffix 中的原文消息数
    trace_ids: List[str] = field(default_factory=list)  # 所有 trace_id
    has_fact_call_instruction: bool = False  # 是否包含 fact call 指导
    fact_rounds_used: int = 0             # 实际使用的 fact call 轮次
    session_id: str = ""                  # 会话 ID (用于 fact retriever)
    
    # ============ 安全 ============
    safety_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于日志/调试)"""
        return {
            "strategy": self.strategy,
            "injection_enabled": self.injection_enabled,
            "alpha_profile": self.alpha_profile.to_dict(),
            "preferences_count": self.preferences_count,
            "preference_tokens": self.preference_tokens,
            "history_tokens": self.history_tokens,
            "history_kv_tokens": self.history_kv_tokens,
            "query_tokens": self.query_tokens,
            "total_tokens": self.total_tokens,
            "relevant_history_count": self.relevant_history_count,
            "memory_triggered": self.memory_triggered,
            "trigger_type": self.trigger_type,
            "reference_resolved": self.reference_resolved,
            "reference_type": self.reference_type,
            "reference_scope": self.reference_scope,
            "full_attention_fallback": self.full_attention_fallback,
            "gating_decision": self.gating_decision,
            "safety_violations": self.safety_violations,
            # Recall v4
            "recall_strategy": self.recall_strategy,
            "summary_count": self.summary_count,
            "message_count": self.message_count,
            "has_fact_call_instruction": self.has_fact_call_instruction,
            "fact_rounds_used": self.fact_rounds_used,
            "trace_ids_count": len(self.trace_ids),
        }


# ============================================================
# ExecutionResult - 执行结果
# ============================================================

@dataclass
class ExecutionResult:
    """
    执行结果
    
    Executor 的输出，包含模型输出和执行性能数据。
    """
    # 模型输出
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_output: Optional[Any] = None
    
    # 执行时间
    inference_latency_ms: float = 0.0
    
    # 缓存信息 (由 executor 填充)
    preference_cache_hit: bool = False
    preference_cache_tier: str = "none"
    
    # Full Attention 执行信息
    full_attention_fallback: bool = False
    full_attention_position_mode: str = ""
    full_attention_preference_tokens: int = 0
    full_attention_history_tokens: int = 0
    
    # 降级
    fallback_used: bool = False
    error_message: Optional[str] = None
    
    # Recall v4 fact call
    fact_rounds_used: int = 0
    fact_tokens_total: int = 0