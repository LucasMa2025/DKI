"""
DKI Recall v4 — 数据结构与配置

定义召回与组装流程中的所有数据结构:
- RecallConfig: 从 config.yaml 读取的配置
- HistoryItem: 历史条目 (summary 或原文)
- RecallResult: 多信号召回结果
- AssembledSuffix: 组装完成的后缀
- FactRequest / FactResponse: 事实请求/响应

Author: AGI Demo Project
Version: 4.0.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# RecallConfig — 配置数据结构
# ============================================================

@dataclass
class RecallSignalsConfig:
    """多信号召回配置"""
    keyword_enabled: bool = True
    keyword_topk: int = 5
    keyword_method: str = "tfidf"  # tfidf | textrank
    vector_enabled: bool = True
    vector_top_k: int = 10
    vector_threshold: float = 0.5
    reference_enabled: bool = True


@dataclass
class RecallBudgetConfig:
    """动态 History 预算配置"""
    generation_reserve: int = 512
    instruction_reserve: int = 150
    min_recent_turns: int = 2
    max_recent_turns: int = 5


@dataclass
class RecallSummaryConfig:
    """逐消息 Summary 配置"""
    per_message_threshold: int = 200
    max_tokens_per_summary: int = 150
    strategy: str = "extractive"  # extractive | llm


@dataclass
class RecallFactCallConfig:
    """Function Call 配置"""
    enabled: bool = True
    max_rounds: int = 3
    max_fact_tokens: int = 800
    batch_size: int = 5


@dataclass
class RecallScoreWeights:
    """召回分数融合权重 (补充建议: 必须显式归一化)"""
    keyword_weight: float = 0.4   # w1: 关键词权重 (w1 >= w2, 事实准确优先)
    vector_weight: float = 0.35   # w2: 向量相似度权重
    recency_weight: float = 0.25  # w3: 时间近度权重


# ============================================================
# F1-1: EpistemicModeConfig — 认知态模式配置
# ============================================================

@dataclass
class EpistemicModeProfile:
    """单个认知态模式的权重预设"""
    keyword_weight: float = 0.4
    vector_weight: float = 0.35
    recency_weight: float = 0.25
    description: str = ""


@dataclass
class EpistemicModeConfig:
    """
    认知态模式配置 (F1-1)
    
    基于 QueryContext 选择权重预设（离散、可审计），
    模式数量有限（4-6 种），工程可控。
    
    预设从 YAML 配置加载，完全可编辑。
    """
    enabled: bool = False
    fallback_mode: str = "direct_lookup"
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "clarification": {
            "keyword_weight": 0.50, "vector_weight": 0.10,
            "recency_weight": 0.40, "description": "用户在回忆或澄清过去的信息",
        },
        "continuation": {
            "keyword_weight": 0.20, "vector_weight": 0.30,
            "recency_weight": 0.50, "description": "用户在延展当前话题",
        },
        "semantic_search": {
            "keyword_weight": 0.20, "vector_weight": 0.55,
            "recency_weight": 0.25, "description": "用户在进行语义相关的检索",
        },
        "correction": {
            "keyword_weight": 0.15, "vector_weight": 0.15,
            "recency_weight": 0.70, "description": "用户在修正之前的信息",
        },
        "direct_lookup": {
            "keyword_weight": 0.60, "vector_weight": 0.25,
            "recency_weight": 0.15, "description": "用户直接搜索特定关键词",
        },
    })


# ============================================================
# F1-2: SignalGatingConfig — 信号门控配置
# ============================================================

@dataclass
class SignalGatingConfig:
    """
    信号置信度门控配置 (F1-2)
    
    低置信度信号退出加权（不是降权，是剔除），
    剩余信号动态归一化。消除低置信度信号的"拖累效应"。
    """
    enabled: bool = True
    confidence_threshold: float = 0.15   # 低于此值的信号不参与融合
    keyword_min_hits: int = 1            # keyword 至少命中 1 个词才有效
    vector_min_similarity: float = 0.2   # vector 最高相似度低于此值视为无效


@dataclass
class RecallConfig:
    """
    记忆召回策略 v4 完整配置
    
    从 config.yaml 的 dki.recall 段读取。
    """
    enabled: bool = True
    strategy: str = "summary_with_fact_call"  # summary_with_fact_call | flat_history

    signals: RecallSignalsConfig = field(default_factory=RecallSignalsConfig)
    budget: RecallBudgetConfig = field(default_factory=RecallBudgetConfig)
    summary: RecallSummaryConfig = field(default_factory=RecallSummaryConfig)
    fact_call: RecallFactCallConfig = field(default_factory=RecallFactCallConfig)
    score_weights: RecallScoreWeights = field(default_factory=RecallScoreWeights)
    prompt_formatter: str = "auto"  # auto | generic | deepseek | glm
    
    # F1-1: 认知态模式配置
    epistemic_modes: EpistemicModeConfig = field(default_factory=EpistemicModeConfig)
    # F1-2: 信号门控配置
    signal_gating: SignalGatingConfig = field(default_factory=SignalGatingConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RecallConfig":
        """从字典构建配置"""
        if not d:
            return cls()
        
        signals = RecallSignalsConfig(**d.get("signals", {})) if "signals" in d else RecallSignalsConfig()
        budget = RecallBudgetConfig(**d.get("budget", {})) if "budget" in d else RecallBudgetConfig()
        summary = RecallSummaryConfig(**d.get("summary", {})) if "summary" in d else RecallSummaryConfig()
        fact_call = RecallFactCallConfig(**d.get("fact_call", {})) if "fact_call" in d else RecallFactCallConfig()
        
        weights_dict = d.get("score_weights", {})
        score_weights = RecallScoreWeights(**weights_dict) if weights_dict else RecallScoreWeights()

        # F1-1: 认知态模式配置
        em_dict = d.get("epistemic_modes", {})
        epistemic_modes = EpistemicModeConfig(
            enabled=em_dict.get("enabled", False),
            fallback_mode=em_dict.get("fallback_mode", "direct_lookup"),
            profiles=em_dict.get("profiles", EpistemicModeConfig().profiles),
        ) if em_dict else EpistemicModeConfig()
        
        # F1-2: 信号门控配置
        sg_dict = d.get("signal_gating", {})
        signal_gating = SignalGatingConfig(**sg_dict) if sg_dict else SignalGatingConfig()

        return cls(
            enabled=d.get("enabled", True),
            strategy=d.get("strategy", "summary_with_fact_call"),
            signals=signals,
            budget=budget,
            summary=summary,
            fact_call=fact_call,
            score_weights=score_weights,
            prompt_formatter=d.get("prompt_formatter", "auto"),
            epistemic_modes=epistemic_modes,
            signal_gating=signal_gating,
        )


# ============================================================
# HistoryItem — 历史条目
# ============================================================

@dataclass
class HistoryItem:
    """
    可见区历史条目 (可以是原始消息或 summary)
    
    type="summary": content 是 summary 文本, trace_id 用于 function call 溯源
    type="message": content 是原始消息, trace_id 同样可用于溯源
    """
    type: str              # "summary" | "message"
    content: str           # 文本内容 (summary 文本 或 原始消息)
    trace_id: str          # 可溯源 ID (message_id, 用于 function call 检索原文)
    role: Optional[str] = None     # "user" | "assistant"
    token_count: int = 0   # token 数
    confidence: str = "high"  # "high" (原文) | "medium" (summary) | "low"
    # 结构化认知标记 (补充建议: 机器可读)
    facts_covered: List[str] = field(default_factory=list)
    facts_missing: List[str] = field(default_factory=list)


# ============================================================
# RecallResult — 多信号召回结果
# ============================================================

@dataclass
class RecallResult:
    """多信号召回结果"""
    messages: List[Any] = field(default_factory=list)  # 召回的完整消息列表 (已排序)
    keyword_hits: int = 0
    vector_hits: int = 0
    reference_scope: Optional[str] = None
    recent_turns_added: int = 0
    scores: Dict[str, float] = field(default_factory=dict)  # message_id -> final_score


# ============================================================
# AssembledSuffix — 组装后的后缀
# ============================================================

@dataclass
class AssembledSuffix:
    """组装完成的后缀"""
    text: str = ""                              # 最终拼装文本
    items: List[HistoryItem] = field(default_factory=list)
    total_tokens: int = 0
    message_count: int = 0                      # 原文消息数量
    summary_count: int = 0                      # summary 数量
    has_fact_call_instruction: bool = False      # 是否包含 function call 指导
    trace_ids: List[str] = field(default_factory=list)


# ============================================================
# FactRequest / FactResponse — 事实请求/响应
# ============================================================

@dataclass
class FactRequest:
    """模型发出的事实检索请求"""
    trace_id: str
    query: Optional[str] = None
    offset: int = 0
    limit: int = 5


@dataclass
class FactResponse:
    """事实检索响应 (mini-batch)"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    trace_id: str = ""
    total_count: int = 0
    offset: int = 0
    has_more: bool = False
