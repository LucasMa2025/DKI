"""
Reference Resolver - 指代解析器

处理用户输入中的指代表达:
- "刚刚" / "刚才" → 最近 1-3 轮
- "最近" → 当前会话
- "那件事" / "那个问题" → 上一个共享主题
- "之前你说的" / "你上次说" → 助手的历史立场

设计原则:
- 召回轮数可配置 (外置)
- 支持中英文
- 规则优先，LLM 辅助 (可选)

Author: AGI Demo Project
Version: 1.0.0
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import yaml
from loguru import logger


class ReferenceType(str, Enum):
    """指代类型"""
    TEMPORAL = "temporal"  # 时间指代 (刚刚、最近)
    REFERENTIAL = "referential"  # 实体指代 (那件事、那个问题)
    STANCE = "stance"  # 立场指代 (你之前说的)
    NONE = "none"  # 无指代


class ReferenceScope(str, Enum):
    """指代范围"""
    LAST_1_3_TURNS = "last_1_3_turns"  # 最近 1-3 轮
    LAST_5_10_TURNS = "last_5_10_turns"  # 最近 5-10 轮
    CURRENT_SESSION = "current_session"  # 当前会话
    LAST_SHARED_TOPIC = "last_shared_topic"  # 上一个共享主题
    ASSISTANT_LAST_STANCE = "assistant_last_stance"  # 助手最后立场
    FIRST_OCCURRENCE_SESSION = "first_occurrence_session"  # v6.3: 首次出现 (全会话范围, 从头搜索)
    FIRST_TOPIC_GENESIS = "first_topic_genesis"  # v6.3: 主题起源 (语义成立点定位)
    CUSTOM = "custom"  # 自定义范围


@dataclass
class Message:
    """消息结构"""
    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[datetime] = None
    turn_id: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class ResolvedReference:
    """解析后的指代"""
    reference_type: ReferenceType
    scope: ReferenceScope
    resolved_content: Optional[str] = None
    confidence: float = 1.0
    source_turns: List[int] = field(default_factory=list)
    matched_keyword: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def recall_turns(self) -> Optional[int]:
        """获取建议的召回轮数"""
        return self.metadata.get('recall_turns')


@dataclass
class ReferenceResolverConfig:
    """
    Reference Resolver 配置
    
    支持从 YAML 文件加载，实现参数外置
    """
    # 召回轮数配置 (核心可配置项)
    last_few_turns: int = 3  # "刚刚/刚才" 召回轮数
    recent_turns: int = 10  # "最近" 召回轮数
    session_max_turns: int = 50  # 会话最大召回轮数
    
    # 指代关键词映射 (中文)
    reference_mappings_cn: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "刚刚": {"scope": "last_1_3_turns", "type": "temporal"},
        "刚才": {"scope": "last_1_3_turns", "type": "temporal"},
        "最近": {"scope": "current_session", "type": "temporal"},
        "那件事": {"scope": "last_shared_topic", "type": "referential"},
        "那个问题": {"scope": "last_shared_topic", "type": "referential"},
        "那个话题": {"scope": "last_shared_topic", "type": "referential"},
        "之前你说的": {"scope": "assistant_last_stance", "type": "stance"},
        "你上次说": {"scope": "assistant_last_stance", "type": "stance"},
        "你之前提到": {"scope": "assistant_last_stance", "type": "stance"},
        "上次": {"scope": "last_5_10_turns", "type": "temporal"},
        "前几天": {"scope": "last_5_10_turns", "type": "temporal"},
        # v6.3: 首次/起始类指代 → 当前会话全范围 (无主题限定)
        "第一次": {"scope": "first_occurrence_session", "type": "temporal"},
        "首次": {"scope": "first_occurrence_session", "type": "temporal"},
        "开始": {"scope": "first_occurrence_session", "type": "temporal"},
        # v6.3: 主题起源指代 → 语义成立点定位 (带主题暗示)
        "怎么聊起": {"scope": "first_topic_genesis", "type": "referential"},
        "怎么说起": {"scope": "first_topic_genesis", "type": "referential"},
        "最早提到": {"scope": "first_topic_genesis", "type": "referential"},
        "什么时候开始说": {"scope": "first_topic_genesis", "type": "temporal"},
        "聊起": {"scope": "current_session", "type": "referential"},
        "说起": {"scope": "current_session", "type": "referential"},
        # v6.3: "最后" → 最近 1-3 轮 (指最近一次提及)
        "最后": {"scope": "last_1_3_turns", "type": "temporal"},
    })
    
    # 指代关键词映射 (英文)
    reference_mappings_en: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "just now": {"scope": "last_1_3_turns", "type": "temporal"},
        "just": {"scope": "last_1_3_turns", "type": "temporal"},
        "recently": {"scope": "current_session", "type": "temporal"},
        "that thing": {"scope": "last_shared_topic", "type": "referential"},
        "that issue": {"scope": "last_shared_topic", "type": "referential"},
        "that topic": {"scope": "last_shared_topic", "type": "referential"},
        "you said earlier": {"scope": "assistant_last_stance", "type": "stance"},
        "you mentioned": {"scope": "assistant_last_stance", "type": "stance"},
        "last time": {"scope": "last_5_10_turns", "type": "temporal"},
        "earlier": {"scope": "last_5_10_turns", "type": "temporal"},
        # v6.3: first-occurrence references → full session scope (no topic filter)
        "first time": {"scope": "first_occurrence_session", "type": "temporal"},
        "at the beginning": {"scope": "first_occurrence_session", "type": "temporal"},
        "when we started": {"scope": "first_occurrence_session", "type": "temporal"},
        # v6.3: topic genesis references → semantic commitment point (topic-seeking)
        "how did we start talking about": {"scope": "first_topic_genesis", "type": "referential"},
        "when did we first discuss": {"scope": "first_topic_genesis", "type": "referential"},
        "when was the first mention of": {"scope": "first_topic_genesis", "type": "referential"},
        "when we talked about": {"scope": "first_topic_genesis", "type": "referential"},
        "when you first mentioned": {"scope": "first_topic_genesis", "type": "referential"},
        # v6.3: "last" / "finally" → most recent 1-3 turns
        "at last": {"scope": "last_1_3_turns", "type": "temporal"},
        "in the end": {"scope": "last_1_3_turns", "type": "temporal"},
    })
    
    # v6.3: Topic Genesis 判定配置
    # 三层递进判定: Level 1 (命名+定义, 强) → Level 2 (概念收敛, 中) → Level 3 (角色切换, 弱)
    genesis_context_window: int = 1  # Genesis 命中后, 前后各取 N 条作为叙事上下文
    genesis_confidence_levels: Dict[str, float] = field(default_factory=lambda: {
        "naming_definition": 1.0,      # Level 1: 命名+定义 (强, e.g. "DKI 是 Dynamic K/V Injection")
        "concept_convergence": 0.75,   # Level 2: 概念收敛 (中, e.g. 从类比过渡到实体)
        "role_shift": 0.5,             # Level 3: 讨论角色切换 (弱, 从背景→前台)
    })
    # Level 1 判定模式: 命名+定义 (正则)
    genesis_naming_patterns: List[str] = field(default_factory=lambda: [
        r"(.+?)(?:是|叫做|叫|称为|全称是|的全称|stands?\s+for|means?)\s*(.+)",
        r"(?:我把|我将|我们把|我们将)(?:它|这个|这套|这个系统)(?:叫做?|称为|命名为)\s*(.+)",
    ])
    # Level 2 判定模式: 概念收敛 (从类比/联想 → 实体)
    genesis_convergence_markers: List[str] = field(default_factory=lambda: [
        "想起了", "想到了", "联想到", "说到", "提到",
        "reminds me of", "made me think of", "speaking of",
    ])
    # Level 3 判定模式: 角色切换 (从背景→前台)
    genesis_role_shift_markers: List[str] = field(default_factory=lambda: [
        "你知道为什么", "为什么", "怎么实现", "能不能",
        "how does", "why does", "can you explain", "what about",
    ])
    
    # 是否启用 LLM 辅助解析
    use_llm_fallback: bool = False
    llm_fallback_threshold: float = 0.5  # 规则置信度低于此值时使用 LLM
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReferenceResolverConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('reference_resolver', data))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceResolverConfig":
        """从字典创建配置"""
        defaults = cls()
        return cls(
            last_few_turns=data.get('last_few_turns', 3),
            recent_turns=data.get('recent_turns', 10),
            session_max_turns=data.get('session_max_turns', 50),
            reference_mappings_cn=data.get('reference_mappings_cn', defaults.reference_mappings_cn),
            reference_mappings_en=data.get('reference_mappings_en', defaults.reference_mappings_en),
            # v6.3: Topic Genesis 配置
            genesis_context_window=data.get('genesis_context_window', 1),
            genesis_confidence_levels=data.get('genesis_confidence_levels', defaults.genesis_confidence_levels),
            genesis_naming_patterns=data.get('genesis_naming_patterns', defaults.genesis_naming_patterns),
            genesis_convergence_markers=data.get('genesis_convergence_markers', defaults.genesis_convergence_markers),
            genesis_role_shift_markers=data.get('genesis_role_shift_markers', defaults.genesis_role_shift_markers),
            use_llm_fallback=data.get('use_llm_fallback', False),
            llm_fallback_threshold=data.get('llm_fallback_threshold', 0.5),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'last_few_turns': self.last_few_turns,
            'recent_turns': self.recent_turns,
            'session_max_turns': self.session_max_turns,
            'reference_mappings_cn': self.reference_mappings_cn,
            'reference_mappings_en': self.reference_mappings_en,
            'genesis_context_window': self.genesis_context_window,
            'genesis_confidence_levels': self.genesis_confidence_levels,
            'genesis_naming_patterns': self.genesis_naming_patterns,
            'genesis_convergence_markers': self.genesis_convergence_markers,
            'genesis_role_shift_markers': self.genesis_role_shift_markers,
            'use_llm_fallback': self.use_llm_fallback,
            'llm_fallback_threshold': self.llm_fallback_threshold,
        }


class ReferenceResolver:
    """
    指代解析器
    
    核心功能:
    1. 检测用户输入中的指代表达
    2. 根据指代类型确定召回范围
    3. 从历史消息中提取相关内容
    
    使用方式:
    ```python
    # 使用默认配置
    resolver = ReferenceResolver()
    
    # 使用自定义配置 (召回轮数外置)
    config = ReferenceResolverConfig(
        last_few_turns=5,  # 自定义 "刚刚" 召回轮数
        recent_turns=20,   # 自定义 "最近" 召回轮数
    )
    resolver = ReferenceResolver(config=config)
    
    # 从配置文件加载
    config = ReferenceResolverConfig.from_yaml("config/reference_resolver.yaml")
    resolver = ReferenceResolver(config=config)
    
    # 解析指代
    result = resolver.resolve(
        query="刚才你说的那个方案是什么？",
        history=[...],
    )
    ```
    """
    
    def __init__(
        self,
        config: Optional[ReferenceResolverConfig] = None,
        language: str = "auto",
        llm_resolver: Optional[Callable] = None,
    ):
        """
        初始化指代解析器
        
        Args:
            config: 解析器配置 (可选)
            language: 语言 ("cn", "en", "auto")
            llm_resolver: LLM 辅助解析函数 (可选)
        """
        self.config = config or ReferenceResolverConfig()
        self.language = language
        self.llm_resolver = llm_resolver
        
        logger.info(
            f"ReferenceResolver initialized "
            f"(last_few_turns={self.config.last_few_turns}, "
            f"recent_turns={self.config.recent_turns})"
        )
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        return 'cn' if chinese_ratio > 0.3 else 'en'
    
    def resolve(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        stance_cache: Optional[Dict[str, Any]] = None,
    ) -> ResolvedReference:
        """
        解析查询中的指代
        
        Args:
            query: 用户查询
            history: 历史消息列表 (按时间顺序，最新在后)，可选
            stance_cache: 立场缓存 (可选，用于立场指代解析)
            
        Returns:
            ResolvedReference 包含解析结果
        """
        if not query or not query.strip():
            return ResolvedReference(
                reference_type=ReferenceType.NONE,
                scope=ReferenceScope.CUSTOM,
            )
        
        # 确定语言
        if self.language == "auto":
            lang = self._detect_language(query)
        else:
            lang = self.language
        
        # 获取对应语言的映射
        mappings = (
            self.config.reference_mappings_cn if lang == 'cn'
            else self.config.reference_mappings_en
        )
        
        # 规则匹配
        for keyword, mapping in mappings.items():
            if keyword in query.lower():
                scope = ReferenceScope(mapping['scope'])
                ref_type = ReferenceType(mapping['type'])
                
                # 计算召回轮数
                recall_turns = self._get_recall_turns_for_scope(scope)
                
                # 如果有历史，根据范围提取内容
                resolved_content = None
                source_turns = []
                extra_metadata = {}
                confidence = 1.0
                if history:
                    resolved_content, source_turns, extra_metadata = self._resolve_by_scope(
                        scope=scope,
                        ref_type=ref_type,
                        history=history,
                        query=query,
                        stance_cache=stance_cache,
                    )
                    # Genesis 判定可能降低 confidence
                    if 'genesis_confidence' in extra_metadata:
                        confidence = extra_metadata['genesis_confidence']
                
                return ResolvedReference(
                    reference_type=ref_type,
                    scope=scope,
                    resolved_content=resolved_content,
                    confidence=confidence,
                    source_turns=source_turns,
                    matched_keyword=keyword,
                    metadata={
                        'language': lang,
                        'recall_turns': recall_turns,
                        **extra_metadata,
                    },
                )
        
        # 如果规则匹配失败，尝试 LLM 辅助
        if self.config.use_llm_fallback and self.llm_resolver and history:
            return self._llm_resolve(query, history)
        
        return ResolvedReference(
            reference_type=ReferenceType.NONE,
            scope=ReferenceScope.CUSTOM,
        )
    
    def _get_recall_turns_for_scope(self, scope: ReferenceScope) -> int:
        """根据范围获取召回轮数"""
        if scope == ReferenceScope.LAST_1_3_TURNS:
            return self.config.last_few_turns
        elif scope == ReferenceScope.LAST_5_10_TURNS:
            return self.config.recent_turns
        elif scope == ReferenceScope.CURRENT_SESSION:
            return self.config.session_max_turns
        elif scope == ReferenceScope.LAST_SHARED_TOPIC:
            return self.config.last_few_turns * 2  # 主题上下文
        elif scope == ReferenceScope.ASSISTANT_LAST_STANCE:
            return self.config.recent_turns
        elif scope == ReferenceScope.FIRST_OCCURRENCE_SESSION:
            # v6.3: 首次出现需要全会话范围 (从头搜索)
            return self.config.session_max_turns
        elif scope == ReferenceScope.FIRST_TOPIC_GENESIS:
            # v6.3: 主题起源需要全会话范围 (从头扫描找语义成立点)
            return self.config.session_max_turns
        return self.config.recent_turns  # 默认
    
    def _resolve_by_scope(
        self,
        scope: ReferenceScope,
        ref_type: ReferenceType,
        history: List[Message],
        query: str = "",
        stance_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[int], Dict[str, Any]]:
        """
        根据范围解析内容
        
        Returns:
            (resolved_content, source_turns, extra_metadata)
            extra_metadata: 额外元数据 (如 genesis_level, genesis_confidence)
        """
        if not history:
            return None, [], {}
        
        if scope == ReferenceScope.LAST_1_3_TURNS:
            # 最近 N 轮 (可配置)
            n = self.config.last_few_turns
            recent = history[-n*2:] if len(history) >= n*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - n*2), len(history)))
            return content, turns, {}
        
        elif scope == ReferenceScope.LAST_5_10_TURNS:
            # 最近 5-10 轮
            n = self.config.recent_turns
            recent = history[-n*2:] if len(history) >= n*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - n*2), len(history)))
            return content, turns, {}
        
        elif scope == ReferenceScope.CURRENT_SESSION:
            # 当前会话 (最多 session_max_turns)
            max_turns = self.config.session_max_turns
            recent = history[-max_turns*2:] if len(history) >= max_turns*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - max_turns*2), len(history)))
            return content, turns, {}
        
        elif scope == ReferenceScope.LAST_SHARED_TOPIC:
            # 上一个共享主题
            content, turns = self._find_last_topic(history)
            return content, turns, {}
        
        elif scope == ReferenceScope.ASSISTANT_LAST_STANCE:
            # 助手最后立场
            content, turns = self._find_assistant_stance(history, stance_cache)
            return content, turns, {}
        
        elif scope == ReferenceScope.FIRST_OCCURRENCE_SESSION:
            # v6.3: 首次出现 — 从会话开头搜索 (无主题过滤)
            content, turns = self._find_first_occurrence(history)
            return content, turns, {}
        
        elif scope == ReferenceScope.FIRST_TOPIC_GENESIS:
            # v6.3: 主题起源 — 语义成立点定位 (带主题过滤)
            return self._find_topic_genesis(query, history)
        
        return None, [], {}
    
    def _format_messages(self, messages: List[Message]) -> str:
        """格式化消息列表"""
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
    
    def _find_last_topic(self, history: List[Message]) -> Tuple[Optional[str], List[int]]:
        """
        查找上一个共享主题
        
        简单实现: 查找最近一个有明确主题的消息
        """
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if msg.topic:
                # 找到主题后，获取相关的上下文
                start = max(0, i - 2)
                end = min(len(history), i + 3)
                context = history[start:end]
                return self._format_messages(context), list(range(start, end))
        
        # 如果没有明确主题，返回最近几轮
        n = self.config.last_few_turns
        recent = history[-n*2:] if len(history) >= n*2 else history
        return self._format_messages(recent), list(range(max(0, len(history) - n*2), len(history)))
    
    def _find_first_occurrence(
        self,
        history: List[Message],
    ) -> Tuple[Optional[str], List[int]]:
        """
        v6.3: 查找会话中的首次出现
        
        用于处理 "第一次"、"首次"、"开始" 等**不带主题限定**的指代。
        从会话开头开始搜索，返回最早的几条消息作为上下文。
        
        语义: 用户想知道会话一开始说了什么 (无主题过滤)。
        
        Returns:
            (resolved_content, source_turns)
        """
        if not history:
            return None, []
        
        # 从会话开头取前 N 条消息 (N = last_few_turns * 2, 即 user+assistant 对)
        n = self.config.last_few_turns
        first_messages = history[:n * 2] if len(history) >= n * 2 else history
        content = self._format_messages(first_messages)
        turns = list(range(len(first_messages)))
        return content, turns
    
    def _find_topic_genesis(
        self,
        query: str,
        history: List[Message],
    ) -> Tuple[Optional[str], List[int], Dict[str, Any]]:
        """
        v6.3: 主题起源定位 (Topic Genesis)
        
        与 _find_first_occurrence 的区别:
        - first_occurrence: 取会话开头 N 条 (无主题过滤)
        - topic_genesis: 扫描全会话, 找主题 T 首次被当作讨论对象的那一轮
        
        三层递进判定 (只读, 不写入, 不画像):
        - Level 1 (confidence=1.0):  命名+定义 — "DKI 是 Dynamic K/V Injection"
        - Level 2 (confidence=0.75): 概念收敛 — 从类比/联想过渡到实体讨论
        - Level 3 (confidence=0.5):  角色切换 — 从背景提及变成正式讨论对象
        
        如果三层都未命中, 降级到 _find_first_occurrence。
        
        Args:
            query: 用户查询 (用于提取主题实体)
            history: 完整会话历史
            
        Returns:
            (resolved_content, source_turns, genesis_metadata)
            genesis_metadata 包含: genesis_level, genesis_confidence, topic_entity
        """
        if not history:
            return None, [], {}
        
        # ============ Phase 0: 从查询中提取主题实体 ============
        topic_entity = self._extract_topic_from_query(query)
        
        if not topic_entity:
            # 无法提取主题 → 降级到 first_occurrence
            logger.debug("Topic genesis: no topic entity extracted, falling back to first_occurrence")
            content, turns = self._find_first_occurrence(history)
            return content, turns, {
                "genesis_level": "fallback",
                "genesis_confidence": 0.3,
                "topic_entity": None,
            }
        
        logger.debug(f"Topic genesis: searching for entity '{topic_entity}'")
        
        ctx_window = self.config.genesis_context_window
        confidence_levels = self.config.genesis_confidence_levels
        
        # ============ Phase 1: 从头扫描, 三层判定 ============
        for i, msg in enumerate(history):
            content_lower = msg.content.lower()
            topic_lower = topic_entity.lower()
            
            # 主题实体必须出现在该消息中
            if topic_lower not in content_lower:
                continue
            
            # --- Level 1: 命名+定义 (强) ---
            for pattern in self.config.genesis_naming_patterns:
                match = re.search(pattern, msg.content, re.IGNORECASE)
                if match and topic_lower in match.group(0).lower():
                    return self._build_genesis_result(
                        history, i, ctx_window,
                        level="naming_definition",
                        confidence=confidence_levels.get("naming_definition", 1.0),
                        topic_entity=topic_entity,
                    )
            
            # --- Level 2: 概念收敛 (中) ---
            # 同一条消息中同时出现收敛标记和主题实体
            for marker in self.config.genesis_convergence_markers:
                if marker in content_lower:
                    return self._build_genesis_result(
                        history, i, ctx_window,
                        level="concept_convergence",
                        confidence=confidence_levels.get("concept_convergence", 0.75),
                        topic_entity=topic_entity,
                    )
            
            # --- Level 3: 角色切换 (弱) ---
            # 主题出现后, 下一条消息(或本条)出现深入讨论标记
            next_msg = history[i + 1] if i + 1 < len(history) else None
            check_msgs = [msg]
            if next_msg:
                check_msgs.append(next_msg)
            
            for check_msg in check_msgs:
                for marker in self.config.genesis_role_shift_markers:
                    if marker in check_msg.content.lower():
                        return self._build_genesis_result(
                            history, i, ctx_window,
                            level="role_shift",
                            confidence=confidence_levels.get("role_shift", 0.5),
                            topic_entity=topic_entity,
                        )
        
        # ============ Phase 2: 三层都未命中 → 找第一次出现主题实体的位置 ============
        for i, msg in enumerate(history):
            if topic_entity.lower() in msg.content.lower():
                return self._build_genesis_result(
                    history, i, ctx_window,
                    level="keyword_only",
                    confidence=0.4,
                    topic_entity=topic_entity,
                )
        
        # ============ Phase 3: 主题实体在历史中完全不存在 → 降级 ============
        logger.debug(f"Topic genesis: entity '{topic_entity}' not found in history, falling back")
        content, turns = self._find_first_occurrence(history)
        return content, turns, {
            "genesis_level": "fallback",
            "genesis_confidence": 0.3,
            "topic_entity": topic_entity,
        }
    
    def _extract_topic_from_query(self, query: str) -> Optional[str]:
        """
        从查询中提取主题实体
        
        策略: 去除已知的指代关键词, 提取剩余的核心名词/实体。
        这是一个轻量级规则提取, 不依赖 NER 或 LLM。
        
        Examples:
            "我们怎么聊起 DKI 的？" → "DKI"
            "最早提到机器学习是什么时候？" → "机器学习"
            "how did we start talking about DKI?" → "DKI"
        """
        # 收集所有指代关键词 (用于剥离)
        all_keywords = set()
        for keyword in self.config.reference_mappings_cn:
            all_keywords.add(keyword)
        for keyword in self.config.reference_mappings_en:
            all_keywords.add(keyword)
        
        # 额外的停用词/语气词
        stopwords_cn = {"的", "了", "吗", "呢", "吧", "啊", "是", "在", "我们", "你", "我", 
                        "什么时候", "什么", "怎么", "哪里", "为什么", "这个", "那个"}
        stopwords_en = {"the", "a", "an", "is", "was", "were", "are", "we", "you", "i",
                        "about", "of", "did", "do", "how", "when", "where", "what", "why",
                        "first", "start", "talking", "discuss", "mention", "talked"}
        
        text = query.strip()
        
        # 去除指代关键词 (按长度降序, 避免短词误删长词的一部分)
        sorted_keywords = sorted(all_keywords, key=len, reverse=True)
        for kw in sorted_keywords:
            text = text.replace(kw, " ")
        
        # 去除标点
        text = re.sub(r'[？?！!。.，,、；;：:""''「」【】()（）]', ' ', text)
        
        # 检测语言
        lang = self._detect_language(query)
        
        if lang == 'cn':
            # 中文: 去除停用词后, 取最长的连续非空白片段
            for sw in sorted(stopwords_cn, key=len, reverse=True):
                text = text.replace(sw, " ")
            parts = [p.strip() for p in text.split() if p.strip()]
            if parts:
                # 取最长的片段作为主题实体
                return max(parts, key=len)
        else:
            # 英文: 去除停用词后, 取剩余词组
            words = text.split()
            filtered = [w for w in words if w.lower() not in stopwords_en and len(w) > 1]
            if filtered:
                return " ".join(filtered).strip()
        
        return None
    
    def _build_genesis_result(
        self,
        history: List[Message],
        genesis_index: int,
        ctx_window: int,
        level: str,
        confidence: float,
        topic_entity: str,
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        """
        构建 Genesis 结果: 命中轮 + 前后 ctx_window 条上下文
        
        返回的不是单条消息, 而是 [genesis_index - ctx_window, genesis_index + ctx_window] 
        范围内的消息, 保证叙事完整性。
        """
        start = max(0, genesis_index - ctx_window)
        end = min(len(history), genesis_index + ctx_window + 1)
        context_msgs = history[start:end]
        
        content = self._format_messages(context_msgs)
        turns = list(range(start, end))
        
        logger.info(
            f"Topic genesis found: level={level}, confidence={confidence:.2f}, "
            f"entity='{topic_entity}', turn={genesis_index}"
        )
        
        metadata = {
            "genesis_level": level,
            "genesis_confidence": confidence,
            "genesis_turn_index": genesis_index,
            "topic_entity": topic_entity,
        }
        
        return content, turns, metadata
    
    def _find_assistant_stance(
        self,
        history: List[Message],
        stance_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[int]]:
        """
        查找助手的历史立场
        
        优先使用 stance_cache，否则从历史中提取
        """
        # 优先使用 stance_cache
        if stance_cache:
            # 获取最新的立场
            latest_stance = None
            for topic, stance in stance_cache.items():
                if latest_stance is None or stance.get('updated_at', '') > latest_stance.get('updated_at', ''):
                    latest_stance = stance
            
            if latest_stance:
                return f"关于「{latest_stance.get('topic', '未知')}」: {latest_stance.get('stance', '')}", []
        
        # 从历史中查找助手的观点表达
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if msg.role == "assistant":
                # 简单判断是否包含观点表达
                opinion_markers = ['我认为', '我觉得', '我建议', '我的看法是', 'I think', 'I believe', 'I suggest']
                for marker in opinion_markers:
                    if marker in msg.content:
                        return msg.content, [i]
        
        return None, []
    
    def _llm_resolve(
        self,
        query: str,
        history: List[Message],
    ) -> ResolvedReference:
        """
        使用 LLM 辅助解析
        
        后续扩展: 调用 LLM 进行复杂指代消解
        """
        if not self.llm_resolver:
            return ResolvedReference(
                reference_type=ReferenceType.NONE,
                scope=ReferenceScope.CUSTOM,
            )
        
        try:
            result = self.llm_resolver(query, history)
            return result
        except Exception as e:
            logger.error(f"LLM resolve failed: {e}")
            return ResolvedReference(
                reference_type=ReferenceType.NONE,
                scope=ReferenceScope.CUSTOM,
            )
    
    def update_config(
        self,
        just_now_turns: Optional[int] = None,
        recently_turns: Optional[int] = None,
        last_topic_turns: Optional[int] = None,
        assistant_stance_turns: Optional[int] = None,
        # 兼容旧参数名
        last_few_turns: Optional[int] = None,
        recent_turns: Optional[int] = None,
        session_max_turns: Optional[int] = None,
    ):
        """
        动态更新配置
        
        支持两种参数命名风格:
        - 新风格: just_now_turns, recently_turns (更直观)
        - 旧风格: last_few_turns, recent_turns (兼容)
        
        Args:
            just_now_turns: "刚刚/刚才" 召回轮数
            recently_turns: "最近" 召回轮数
            last_topic_turns: "那件事" 召回轮数 (预留)
            assistant_stance_turns: "你之前说的" 召回轮数 (预留)
            last_few_turns: (兼容) "刚刚/刚才" 召回轮数
            recent_turns: (兼容) "最近" 召回轮数
            session_max_turns: 会话最大召回轮数
        """
        # 新参数优先
        if just_now_turns is not None:
            self.config.last_few_turns = just_now_turns
        elif last_few_turns is not None:
            self.config.last_few_turns = last_few_turns
        
        if recently_turns is not None:
            self.config.recent_turns = recently_turns
        elif recent_turns is not None:
            self.config.recent_turns = recent_turns
        
        if session_max_turns is not None:
            self.config.session_max_turns = session_max_turns
        
        # last_topic_turns 和 assistant_stance_turns 预留，后续实现
        
        logger.info(
            f"ReferenceResolver config updated: "
            f"last_few_turns={self.config.last_few_turns}, "
            f"recent_turns={self.config.recent_turns}"
        )
    
    def add_mapping(
        self,
        keyword: str,
        scope: str,
        ref_type: str,
        language: str = "cn",
    ):
        """
        动态添加指代映射
        
        Args:
            keyword: 关键词
            scope: 范围 (ReferenceScope 的值)
            ref_type: 类型 (ReferenceType 的值)
            language: 语言
        """
        mapping = {"scope": scope, "type": ref_type}
        
        if language == "cn":
            self.config.reference_mappings_cn[keyword] = mapping
        else:
            self.config.reference_mappings_en[keyword] = mapping
        
        logger.info(f"Added reference mapping: {keyword} -> {mapping} ({language})")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'language': self.language,
            'last_few_turns': self.config.last_few_turns,
            'recent_turns': self.config.recent_turns,
            'session_max_turns': self.config.session_max_turns,
            'use_llm_fallback': self.config.use_llm_fallback,
            'mappings_cn_count': len(self.config.reference_mappings_cn),
            'mappings_en_count': len(self.config.reference_mappings_en),
        }


# 便捷函数
def create_reference_resolver(
    config_path: Optional[str] = None,
    last_few_turns: int = 3,
    recent_turns: int = 10,
    language: str = "auto",
) -> ReferenceResolver:
    """
    创建 Reference Resolver 实例
    
    Args:
        config_path: 配置文件路径 (可选)
        last_few_turns: "刚刚" 召回轮数 (如果不使用配置文件)
        recent_turns: "最近" 召回轮数 (如果不使用配置文件)
        language: 语言
        
    Returns:
        ReferenceResolver 实例
    """
    if config_path:
        config = ReferenceResolverConfig.from_yaml(config_path)
    else:
        config = ReferenceResolverConfig(
            last_few_turns=last_few_turns,
            recent_turns=recent_turns,
        )
    
    return ReferenceResolver(config=config, language=language)
