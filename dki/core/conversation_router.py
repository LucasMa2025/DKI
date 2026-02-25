"""
Conversation Router — RAG/DKI 动态路由系统

根据会话上下文自动选择最优推理模式:
- RAG: 知识检索增强 (短会话、首次查询、知识密集型)
- DKI: 动态记忆注入 (多轮长会话、长期记忆、偏好依赖型)

============================================================================
设计哲学
============================================================================

DKI 和 RAG 不是替代关系, 而是互补关系:

┌────────────────────────────────────────────────────────────────────────┐
│  维度          │  RAG 优势                │  DKI 优势                   │
├────────────────┼──────────────────────────┼─────────────────────────────┤
│  会话长度       │  1-3 轮 (短)              │  5+ 轮 (长, 跨会话)          │
│  记忆类型       │  外部知识库               │  用户偏好 + 会话历史           │
│  个性化        │  无/弱                    │  强 (偏好 K/V 注入)           │
│  历史依赖       │  低 (单次检索)             │  高 (多信号融合召回)           │
│  上下文压力     │  低 (top-k 检索)           │  高 (长历史压缩 + 摘要)        │
│  延迟          │  低 (单次检索)             │  较高 (多信号 + 事实补充)       │
│  首次交互       │  ★ 强                   │  弱 (无历史可召回)             │
│  跨会话连续性   │  弱                      │  ★ 强 (跨会话记忆)            │
│  偏好遵循       │  弱 (system prompt)      │  ★ 强 (attention 级注入)      │
│  事实准确性     │  依赖知识库               │  ★ 自洽验证 (Fact Call Loop)   │
└────────────────────────────────────────────────────────────────────────┘

路由策略: 根据 5 个维度的评分加权决策

    Score_DKI = w₁·S_history + w₂·S_preference + w₃·S_trigger 
              + w₄·S_session_depth + w₅·S_cross_session

    Route = DKI  if Score_DKI > θ_dki
          = RAG  if Score_DKI < θ_rag
          = DKI  otherwise (with reduced confidence)

============================================================================
Author: AGI Demo Project  
Version: 1.0.0
============================================================================
"""

import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from loguru import logger


# ============================================================
# 数据结构
# ============================================================

class RouteMode(str, Enum):
    """路由目标模式"""
    RAG = "rag"
    DKI = "dki"
    HYBRID = "hybrid"  # 保留: 未来可能的混合模式


class RouteReason(str, Enum):
    """路由原因"""
    SHORT_SESSION = "short_session"               # 短会话, RAG 足够
    FIRST_INTERACTION = "first_interaction"         # 首次交互, 无历史
    KNOWLEDGE_QUERY = "knowledge_query"             # 知识检索型查询
    MULTI_TURN_DEEP = "multi_turn_deep"             # 多轮深度对话
    PREFERENCE_DEPENDENT = "preference_dependent"   # 偏好依赖型
    MEMORY_TRIGGER = "memory_trigger"               # 记忆触发信号
    CROSS_SESSION = "cross_session"                 # 跨会话记忆需求
    CONTEXT_PRESSURE = "context_pressure"           # 上下文压力大
    FORCED = "forced"                               # 强制指定
    DEFAULT = "default"                             # 默认路由


@dataclass
class RoutingSignals:
    """路由信号集合 — 所有用于路由决策的原始信号"""
    
    # ---- 会话维度 ----
    session_turn_count: int = 0          # 当前会话轮次
    session_total_tokens: int = 0        # 当前会话总 token 数
    session_duration_seconds: float = 0  # 当前会话持续时间 (秒)
    
    # ---- 历史维度 ----
    user_total_sessions: int = 0         # 用户历史会话总数
    user_total_messages: int = 0         # 用户历史消息总数
    has_cross_session_history: bool = False  # 是否有跨会话历史
    
    # ---- 偏好维度 ----
    has_user_preferences: bool = False   # 是否设置了用户偏好
    preference_count: int = 0            # 偏好数量
    preference_relevance: float = 0.0    # 偏好与当前查询的相关度 (0-1)
    
    # ---- 查询维度 ----
    query_length: int = 0                # 查询长度 (字符)
    has_memory_trigger: bool = False      # 是否触发了记忆信号
    trigger_type: Optional[str] = None   # 触发类型
    has_reference: bool = False           # 是否包含指代引用
    reference_type: Optional[str] = None # 指代类型
    is_knowledge_query: bool = False     # 是否为知识检索型查询
    
    # ---- 系统维度 ----
    dki_available: bool = True           # DKI 系统是否可用
    rag_available: bool = True           # RAG 系统是否可用
    forced_mode: Optional[str] = None    # 强制指定的模式


@dataclass
class RoutingDecision:
    """路由决策结果"""
    mode: RouteMode                      # 选择的模式
    confidence: float = 0.0              # 决策置信度 (0-1)
    score_dki: float = 0.0               # DKI 得分 (0-1)
    score_rag: float = 0.0               # RAG 得分 (0-1)
    reason: RouteReason = RouteReason.DEFAULT
    reasoning: str = ""                  # 可读的决策理由
    signals: Optional[RoutingSignals] = None
    latency_ms: float = 0.0             # 路由决策耗时
    
    # 子维度得分 (用于调试和可视化)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 4),
            "score_dki": round(self.score_dki, 4),
            "score_rag": round(self.score_rag, 4),
            "reason": self.reason.value,
            "reasoning": self.reasoning,
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "latency_ms": round(self.latency_ms, 3),
        }


# ============================================================
# 路由配置
# ============================================================

@dataclass
class RouterConfig:
    """
    路由器配置
    
    可通过 config.yaml 的 routing 段加载, 也可编程设置。
    所有阈值和权重都可调, 支持 A/B 测试。
    """
    # ---- 全局开关 ----
    enabled: bool = True                 # 路由器是否启用 (关闭时使用 default_mode)
    default_mode: str = "rag"            # 路由器禁用时的默认模式
    
    # ---- DKI 激活阈值 ----
    dki_threshold: float = 0.45          # Score_DKI > 此值 → 选择 DKI
    rag_threshold: float = 0.25          # Score_DKI < 此值 → 确定选择 RAG
    # 中间区域 (rag_threshold ≤ Score ≤ dki_threshold): 取决于其他启发式规则
    
    # ---- 维度权重 (总和 = 1.0) ----
    weight_history: float = 0.25         # w₁: 会话历史深度
    weight_preference: float = 0.20      # w₂: 偏好依赖度
    weight_trigger: float = 0.20         # w₃: 记忆触发信号
    weight_session_depth: float = 0.20   # w₄: 会话轮次深度
    weight_cross_session: float = 0.15   # w₅: 跨会话连续性
    
    # ---- 维度评分参数 ----
    # 会话轮次
    turn_threshold_low: int = 2          # 低于此值 → 短会话 (倾向 RAG)
    turn_threshold_high: int = 6         # 高于此值 → 长会话 (倾向 DKI)
    
    # 会话 token 数
    token_threshold_low: int = 500       # 低 token → RAG
    token_threshold_high: int = 3000     # 高 token → DKI
    
    # 用户历史
    session_count_threshold: int = 3     # 超过此值有跨会话记忆价值
    message_count_threshold: int = 20    # 超过此值有长期记忆价值
    
    # ---- 惩罚与奖励 ----
    first_turn_rag_bonus: float = 0.3    # 首轮对话给 RAG 额外加分
    knowledge_query_rag_bonus: float = 0.2  # 知识检索查询给 RAG 加分
    preference_dki_bonus: float = 0.15   # 有偏好时给 DKI 加分
    memory_trigger_dki_bonus: float = 0.25  # 触发记忆信号给 DKI 加分
    cross_session_dki_bonus: float = 0.2 # 有跨会话历史给 DKI 加分
    
    # ---- 知识检索查询关键词 (用于简单分类) ----
    knowledge_query_patterns_cn: List[str] = field(default_factory=lambda: [
        r"什么是", r"介绍一下", r"解释", r"定义",
        r"如何.*?(?:使用|操作|实现|配置)",
        r"怎么.*?(?:做|弄|搞|办)",
        r".*?的(?:区别|差异|比较)",
        r"推荐.*?(?:工具|框架|方案)",
        r".*?(?:教程|指南|入门)",
    ])
    knowledge_query_patterns_en: List[str] = field(default_factory=lambda: [
        r"what is", r"explain", r"define", r"how to",
        r"difference between", r"compare",
        r"recommend.*?(?:tool|framework|approach)",
        r"tutorial", r"guide",
    ])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouterConfig":
        """从字典构建配置"""
        if not d:
            return cls()
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# ============================================================
# 信号采集器
# ============================================================

class SignalCollector:
    """
    信号采集器 — 从会话上下文中提取路由信号
    
    不依赖 GPU, 仅使用轻量级规则和数据库查询。
    设计为 O(1) 复杂度, 不影响请求延迟。
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        
        # 编译正则 (一次性)
        self._knowledge_patterns_cn = [
            re.compile(p, re.IGNORECASE) for p in config.knowledge_query_patterns_cn
        ]
        self._knowledge_patterns_en = [
            re.compile(p, re.IGNORECASE) for p in config.knowledge_query_patterns_en
        ]
        
        # 记忆触发模式 (简化版, 与 MemoryTrigger 兼容)
        self._memory_trigger_patterns_cn = [
            re.compile(r"我们(刚刚|刚才|之前|上次).*?(讨论|聊|说|谈)"),
            re.compile(r"(之前|上次|前几天)你(说|提到|建议)"),
            re.compile(r"你(还记得|记得).*?吗"),
            re.compile(r"我(之前|上次)(说|提到|问)过"),
            re.compile(r"(那件事|那个问题|那个话题)"),
        ]
        self._memory_trigger_patterns_en = [
            re.compile(r"we (just|recently|previously) (discussed|talked|mentioned)", re.I),
            re.compile(r"(earlier|last time|before) you (said|mentioned|suggested)", re.I),
            re.compile(r"do you (remember|recall)", re.I),
            re.compile(r"I (previously|earlier) (said|mentioned|asked)", re.I),
        ]
        
        # 指代/引用模式
        self._reference_patterns_cn = [
            re.compile(r"(那个|这个|上面|前面)(说的|提到的|讲的)"),
            re.compile(r"你(刚才|刚刚|之前)说的"),
            re.compile(r"继续(讲|说|聊)"),
        ]
        self._reference_patterns_en = [
            re.compile(r"(that|this|the above|the previous) (thing|topic|point)", re.I),
            re.compile(r"you (just|previously) (said|mentioned)", re.I),
            re.compile(r"continue (with|from)", re.I),
        ]
    
    def collect(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_turn_count: int = 0,
        session_total_tokens: int = 0,
        session_duration_seconds: float = 0,
        user_total_sessions: int = 0,
        user_total_messages: int = 0,
        has_user_preferences: bool = False,
        preference_count: int = 0,
        dki_available: bool = True,
        rag_available: bool = True,
        forced_mode: Optional[str] = None,
    ) -> RoutingSignals:
        """
        采集路由信号
        
        Args:
            query: 当前用户查询
            session_id: 会话 ID (可选, 用于数据库查询)
            user_id: 用户 ID (可选, 用于数据库查询)
            session_turn_count: 当前会话轮次 (由调用方提供, 避免 DB 查询)
            session_total_tokens: 当前会话总 token 数 (由调用方提供)
            session_duration_seconds: 当前会话持续时间 (由调用方提供)
            user_total_sessions: 用户历史会话总数 (由调用方提供)
            user_total_messages: 用户历史消息总数 (由调用方提供)
            has_user_preferences: 是否有用户偏好 (由调用方提供)
            preference_count: 偏好数量 (由调用方提供)
            dki_available: DKI 是否可用
            rag_available: RAG 是否可用
            forced_mode: 强制模式
            
        Returns:
            RoutingSignals
        """
        signals = RoutingSignals()
        
        # ---- 会话维度 (直接赋值, 由调用方提供) ----
        signals.session_turn_count = session_turn_count
        signals.session_total_tokens = session_total_tokens
        signals.session_duration_seconds = session_duration_seconds
        
        # ---- 历史维度 ----
        signals.user_total_sessions = user_total_sessions
        signals.user_total_messages = user_total_messages
        signals.has_cross_session_history = (
            user_total_sessions >= self.config.session_count_threshold
        )
        
        # ---- 偏好维度 ----
        signals.has_user_preferences = has_user_preferences
        signals.preference_count = preference_count
        
        # ---- 查询维度 (轻量级规则分析) ----
        signals.query_length = len(query)
        signals.has_memory_trigger = self._check_memory_trigger(query)
        signals.has_reference = self._check_reference(query)
        signals.is_knowledge_query = self._check_knowledge_query(query)
        
        # ---- 系统维度 ----
        signals.dki_available = dki_available
        signals.rag_available = rag_available
        signals.forced_mode = forced_mode
        
        return signals
    
    def _check_memory_trigger(self, query: str) -> bool:
        """检查是否包含记忆触发信号"""
        for pattern in self._memory_trigger_patterns_cn:
            if pattern.search(query):
                return True
        for pattern in self._memory_trigger_patterns_en:
            if pattern.search(query):
                return True
        return False
    
    def _check_reference(self, query: str) -> bool:
        """检查是否包含指代引用"""
        for pattern in self._reference_patterns_cn:
            if pattern.search(query):
                return True
        for pattern in self._reference_patterns_en:
            if pattern.search(query):
                return True
        return False
    
    def _check_knowledge_query(self, query: str) -> bool:
        """检查是否为知识检索型查询"""
        for pattern in self._knowledge_patterns_cn:
            if pattern.search(query):
                return True
        for pattern in self._knowledge_patterns_en:
            if pattern.search(query):
                return True
        return False


# ============================================================
# 路由评分引擎
# ============================================================

class RoutingScoringEngine:
    """
    路由评分引擎 — 将信号转化为 DKI/RAG 得分
    
    五维评分模型:
    
    D₁: 会话历史深度
        - 当前会话轮次 + 总 token 数 → 归一化分数
        - 高分 → DKI (多轮上下文压缩优势)
    
    D₂: 偏好依赖度
        - 有无偏好 + 偏好数量 → 归一化分数
        - 高分 → DKI (attention 级偏好注入优势)
    
    D₃: 记忆触发信号
        - 元认知表达 / 指代引用 → 二值 + 权重
        - 高分 → DKI (多信号召回优势)
    
    D₄: 会话轮次深度
        - 当前轮次的阶梯函数
        - 高分 → DKI (结构化摘要优势)
    
    D₅: 跨会话连续性
        - 历史会话数 + 消息数 → 归一化分数
        - 高分 → DKI (跨会话记忆优势)
    
    最终:
        Score_DKI = w₁·D₁ + w₂·D₂ + w₃·D₃ + w₄·D₄ + w₅·D₅
                  + bonuses - penalties
        Score_RAG = 1.0 - Score_DKI + rag_bonuses
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
    
    def score(self, signals: RoutingSignals) -> Tuple[float, float, Dict[str, float]]:
        """
        计算 DKI 和 RAG 的得分
        
        Returns:
            (score_dki, score_rag, dimension_scores)
        """
        dims = {}
        
        # ---- D₁: 会话历史深度 ----
        # 使用 sigmoid-like 函数将 token 数映射到 [0, 1]
        if signals.session_total_tokens <= 0:
            d1_tokens = 0.0
        elif signals.session_total_tokens >= self.config.token_threshold_high:
            d1_tokens = 1.0
        else:
            d1_tokens = (signals.session_total_tokens - self.config.token_threshold_low) / max(
                self.config.token_threshold_high - self.config.token_threshold_low, 1
            )
            d1_tokens = max(0.0, min(1.0, d1_tokens))
        dims["history_depth"] = d1_tokens
        
        # ---- D₂: 偏好依赖度 ----
        if signals.has_user_preferences and signals.preference_count > 0:
            # 偏好存在 → 基础分 0.5, 每个偏好加 0.1, 最高 1.0
            d2 = min(1.0, 0.5 + signals.preference_count * 0.1)
        else:
            d2 = 0.0
        dims["preference"] = d2
        
        # ---- D₃: 记忆触发信号 ----
        d3 = 0.0
        if signals.has_memory_trigger:
            d3 += 0.7
        if signals.has_reference:
            d3 += 0.3
        d3 = min(1.0, d3)
        dims["trigger"] = d3
        
        # ---- D₄: 会话轮次深度 (阶梯函数) ----
        if signals.session_turn_count <= self.config.turn_threshold_low:
            d4 = 0.0
        elif signals.session_turn_count >= self.config.turn_threshold_high:
            d4 = 1.0
        else:
            d4 = (signals.session_turn_count - self.config.turn_threshold_low) / max(
                self.config.turn_threshold_high - self.config.turn_threshold_low, 1
            )
        dims["session_depth"] = d4
        
        # ---- D₅: 跨会话连续性 ----
        d5 = 0.0
        if signals.has_cross_session_history:
            # 有跨会话历史 → 基础分 0.5
            d5 = 0.5
            # 历史消息越多, 分数越高
            if signals.user_total_messages >= self.config.message_count_threshold:
                d5 += 0.3
            # 历史会话越多, 分数越高
            if signals.user_total_sessions >= self.config.session_count_threshold * 2:
                d5 += 0.2
        d5 = min(1.0, d5)
        dims["cross_session"] = d5
        
        # ---- 加权求和 ----
        score_dki = (
            self.config.weight_history * dims["history_depth"]
            + self.config.weight_preference * dims["preference"]
            + self.config.weight_trigger * dims["trigger"]
            + self.config.weight_session_depth * dims["session_depth"]
            + self.config.weight_cross_session * dims["cross_session"]
        )
        
        # ---- 奖励与惩罚 ----
        bonuses_dki = 0.0
        bonuses_rag = 0.0
        
        # 首轮 → RAG 加分
        if signals.session_turn_count <= 1:
            bonuses_rag += self.config.first_turn_rag_bonus
        
        # 知识检索查询 → RAG 加分
        if signals.is_knowledge_query:
            bonuses_rag += self.config.knowledge_query_rag_bonus
        
        # 有偏好 → DKI 加分
        if signals.has_user_preferences:
            bonuses_dki += self.config.preference_dki_bonus
        
        # 记忆触发 → DKI 加分
        if signals.has_memory_trigger:
            bonuses_dki += self.config.memory_trigger_dki_bonus
        
        # 跨会话 → DKI 加分
        if signals.has_cross_session_history:
            bonuses_dki += self.config.cross_session_dki_bonus
        
        score_dki += bonuses_dki
        score_rag = (1.0 - score_dki) + bonuses_rag
        
        # 归一化到 [0, 1]
        total = score_dki + score_rag
        if total > 0:
            score_dki = score_dki / total
            score_rag = score_rag / total
        else:
            score_dki = 0.5
            score_rag = 0.5
        
        dims["bonuses_dki"] = bonuses_dki
        dims["bonuses_rag"] = bonuses_rag
        
        return score_dki, score_rag, dims


# ============================================================
# ConversationRouter — 路由器主类
# ============================================================

class ConversationRouter:
    """
    会话路由器 — 自动选择 RAG 或 DKI
    
    使用方式:
    ```python
    router = ConversationRouter()
    
    # 快速路由 (推荐: 由 app.py 调用, 传入已知信号)
    decision = router.route(
        query="你还记得我的名字吗?",
        session_turn_count=5,
        has_user_preferences=True,
        preference_count=3,
        user_total_sessions=8,
    )
    
    if decision.mode == RouteMode.DKI:
        response = dki_system.chat(...)
    else:
        response = rag_system.chat(...)
    ```
    
    设计原则:
    1. 零 GPU 依赖 — 纯规则 + 数据库统计
    2. O(1) 延迟 — 不做向量检索, 不调用模型
    3. 可解释 — 每个决策都有完整的评分理由
    4. 可配置 — 所有阈值通过 config.yaml 调整
    5. 可回退 — 任一系统不可用时自动降级
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._collector = SignalCollector(self.config)
        self._scorer = RoutingScoringEngine(self.config)
        
        # 路由统计
        self._stats = {
            "total_routes": 0,
            "rag_routes": 0,
            "dki_routes": 0,
            "avg_score_dki": 0.0,
            "avg_latency_ms": 0.0,
        }
        
        logger.info(
            f"ConversationRouter initialized: "
            f"dki_threshold={self.config.dki_threshold}, "
            f"rag_threshold={self.config.rag_threshold}, "
            f"enabled={self.config.enabled}"
        )
    
    def route(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_turn_count: int = 0,
        session_total_tokens: int = 0,
        session_duration_seconds: float = 0,
        user_total_sessions: int = 0,
        user_total_messages: int = 0,
        has_user_preferences: bool = False,
        preference_count: int = 0,
        dki_available: bool = True,
        rag_available: bool = True,
        forced_mode: Optional[str] = None,
    ) -> RoutingDecision:
        """
        执行路由决策
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            user_id: 用户 ID
            session_turn_count: 当前会话轮次
            session_total_tokens: 当前会话总 token 数
            session_duration_seconds: 当前会话持续时间
            user_total_sessions: 用户历史会话总数
            user_total_messages: 用户历史消息总数
            has_user_preferences: 是否有用户偏好
            preference_count: 偏好数量
            dki_available: DKI 系统是否可用
            rag_available: RAG 系统是否可用
            forced_mode: 强制模式 (跳过评分)
            
        Returns:
            RoutingDecision
        """
        start_time = time.perf_counter()
        
        # ---- 快速路径: 路由器禁用 ----
        if not self.config.enabled:
            default = RouteMode(self.config.default_mode)
            return RoutingDecision(
                mode=default,
                confidence=1.0,
                reason=RouteReason.FORCED,
                reasoning=f"Router disabled, using default mode: {default.value}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # ---- 快速路径: 强制模式 ----
        if forced_mode:
            mode = RouteMode(forced_mode)
            return RoutingDecision(
                mode=mode,
                confidence=1.0,
                reason=RouteReason.FORCED,
                reasoning=f"Forced mode: {mode.value}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # ---- 快速路径: 系统不可用 ----
        if not dki_available and not rag_available:
            return RoutingDecision(
                mode=RouteMode.RAG,
                confidence=0.0,
                reason=RouteReason.DEFAULT,
                reasoning="Neither DKI nor RAG available, defaulting to RAG",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        if not dki_available:
            return RoutingDecision(
                mode=RouteMode.RAG,
                confidence=1.0,
                reason=RouteReason.DEFAULT,
                reasoning="DKI not available, using RAG",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        if not rag_available:
            return RoutingDecision(
                mode=RouteMode.DKI,
                confidence=1.0,
                reason=RouteReason.DEFAULT,
                reasoning="RAG not available, using DKI",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # ---- 主路径: 信号采集 ----
        signals = self._collector.collect(
            query=query,
            session_id=session_id,
            user_id=user_id,
            session_turn_count=session_turn_count,
            session_total_tokens=session_total_tokens,
            session_duration_seconds=session_duration_seconds,
            user_total_sessions=user_total_sessions,
            user_total_messages=user_total_messages,
            has_user_preferences=has_user_preferences,
            preference_count=preference_count,
            dki_available=dki_available,
            rag_available=rag_available,
            forced_mode=forced_mode,
        )
        
        # ---- 主路径: 评分 ----
        score_dki, score_rag, dimension_scores = self._scorer.score(signals)
        
        # ---- 主路径: 决策 ----
        mode, reason, confidence, reasoning = self._decide(
            score_dki, score_rag, signals, dimension_scores
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # ---- 更新统计 ----
        self._update_stats(mode, score_dki, latency_ms)
        
        decision = RoutingDecision(
            mode=mode,
            confidence=confidence,
            score_dki=score_dki,
            score_rag=score_rag,
            reason=reason,
            reasoning=reasoning,
            signals=signals,
            dimension_scores=dimension_scores,
            latency_ms=latency_ms,
        )
        
        logger.info(
            f"[Router] → {mode.value} (conf={confidence:.2f}, "
            f"dki={score_dki:.3f}, rag={score_rag:.3f}) "
            f"reason={reason.value} | turns={session_turn_count}, "
            f"prefs={preference_count}, trigger={signals.has_memory_trigger}"
        )
        
        return decision
    
    def _decide(
        self,
        score_dki: float,
        score_rag: float,
        signals: RoutingSignals,
        dims: Dict[str, float],
    ) -> Tuple[RouteMode, RouteReason, float, str]:
        """
        基于得分和信号做最终决策
        
        Returns:
            (mode, reason, confidence, reasoning)
        """
        # ---- 高置信度 DKI ----
        if score_dki >= self.config.dki_threshold:
            # 确定最显著的原因
            reason = self._identify_primary_reason_dki(signals, dims)
            confidence = min(1.0, score_dki / self.config.dki_threshold)
            reasoning = self._build_reasoning_dki(score_dki, dims, signals)
            return RouteMode.DKI, reason, confidence, reasoning
        
        # ---- 高置信度 RAG ----
        if score_dki <= self.config.rag_threshold:
            reason = self._identify_primary_reason_rag(signals)
            confidence = min(1.0, (1.0 - score_dki) / (1.0 - self.config.rag_threshold))
            reasoning = self._build_reasoning_rag(score_rag, signals)
            return RouteMode.RAG, reason, confidence, reasoning
        
        # ---- 中间区域: 启发式决策 ----
        # 记忆触发 → DKI (用户明确要求回忆)
        if signals.has_memory_trigger:
            return (
                RouteMode.DKI, 
                RouteReason.MEMORY_TRIGGER, 
                0.7,
                f"Memory trigger detected in ambiguous zone (dki={score_dki:.3f}), "
                f"routing to DKI for memory recall"
            )
        
        # 有偏好 + 非知识查询 → DKI
        if signals.has_user_preferences and not signals.is_knowledge_query:
            return (
                RouteMode.DKI,
                RouteReason.PREFERENCE_DEPENDENT,
                0.6,
                f"User preferences exist and query is not knowledge-type "
                f"(dki={score_dki:.3f}), routing to DKI"
            )
        
        # 默认: RAG (更安全, 延迟更低)
        return (
            RouteMode.RAG,
            RouteReason.DEFAULT,
            0.5,
            f"Ambiguous zone (dki={score_dki:.3f}, rag={score_rag:.3f}), "
            f"defaulting to RAG (lower latency, simpler)"
        )
    
    def _identify_primary_reason_dki(
        self, signals: RoutingSignals, dims: Dict[str, float]
    ) -> RouteReason:
        """识别 DKI 路由的主要原因"""
        # 按维度得分排序, 取最高的
        scores = {
            RouteReason.MULTI_TURN_DEEP: dims.get("session_depth", 0),
            RouteReason.PREFERENCE_DEPENDENT: dims.get("preference", 0),
            RouteReason.MEMORY_TRIGGER: dims.get("trigger", 0),
            RouteReason.CROSS_SESSION: dims.get("cross_session", 0),
            RouteReason.CONTEXT_PRESSURE: dims.get("history_depth", 0),
        }
        return max(scores, key=scores.get)
    
    def _identify_primary_reason_rag(self, signals: RoutingSignals) -> RouteReason:
        """识别 RAG 路由的主要原因"""
        if signals.session_turn_count <= 1:
            return RouteReason.FIRST_INTERACTION
        if signals.is_knowledge_query:
            return RouteReason.KNOWLEDGE_QUERY
        return RouteReason.SHORT_SESSION
    
    def _build_reasoning_dki(
        self, score: float, dims: Dict[str, float], signals: RoutingSignals
    ) -> str:
        """构建 DKI 路由的可读理由"""
        parts = [f"DKI score {score:.3f} > threshold {self.config.dki_threshold}"]
        
        if dims.get("session_depth", 0) > 0.5:
            parts.append(f"deep session ({signals.session_turn_count} turns)")
        if dims.get("preference", 0) > 0.3:
            parts.append(f"preferences ({signals.preference_count})")
        if dims.get("trigger", 0) > 0:
            parts.append("memory trigger detected")
        if dims.get("cross_session", 0) > 0.3:
            parts.append(f"cross-session history ({signals.user_total_sessions} sessions)")
        
        return "; ".join(parts)
    
    def _build_reasoning_rag(self, score: float, signals: RoutingSignals) -> str:
        """构建 RAG 路由的可读理由"""
        parts = [f"RAG score {score:.3f} (DKI below threshold)"]
        
        if signals.session_turn_count <= 1:
            parts.append("first interaction")
        if signals.is_knowledge_query:
            parts.append("knowledge-type query")
        if not signals.has_user_preferences:
            parts.append("no preferences set")
        
        return "; ".join(parts)
    
    def _update_stats(self, mode: RouteMode, score_dki: float, latency_ms: float):
        """更新路由统计"""
        self._stats["total_routes"] += 1
        if mode == RouteMode.DKI:
            self._stats["dki_routes"] += 1
        else:
            self._stats["rag_routes"] += 1
        
        # 滑动平均
        n = self._stats["total_routes"]
        self._stats["avg_score_dki"] = (
            self._stats["avg_score_dki"] * (n - 1) + score_dki
        ) / n
        self._stats["avg_latency_ms"] = (
            self._stats["avg_latency_ms"] * (n - 1) + latency_ms
        ) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self._stats["total_routes"]
        return {
            **self._stats,
            "dki_ratio": self._stats["dki_routes"] / max(total, 1),
            "rag_ratio": self._stats["rag_routes"] / max(total, 1),
        }
    
    def update_config(self, **kwargs) -> None:
        """动态更新配置 (用于 A/B 测试)"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old = getattr(self.config, key)
                setattr(self.config, key, value)
                logger.info(f"Router config updated: {key} = {old} → {value}")
        
        # 重新初始化子组件
        self._collector = SignalCollector(self.config)
        self._scorer = RoutingScoringEngine(self.config)
