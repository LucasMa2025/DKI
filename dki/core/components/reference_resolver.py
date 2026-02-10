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
from typing import Dict, List, Optional, Any, Callable
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
    })
    
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
        return cls(
            last_few_turns=data.get('last_few_turns', 3),
            recent_turns=data.get('recent_turns', 10),
            session_max_turns=data.get('session_max_turns', 50),
            reference_mappings_cn=data.get('reference_mappings_cn', cls().reference_mappings_cn),
            reference_mappings_en=data.get('reference_mappings_en', cls().reference_mappings_en),
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
                if history:
                    resolved_content, source_turns = self._resolve_by_scope(
                        scope=scope,
                        ref_type=ref_type,
                        history=history,
                        stance_cache=stance_cache,
                    )
                
                return ResolvedReference(
                    reference_type=ref_type,
                    scope=scope,
                    resolved_content=resolved_content,
                    confidence=1.0,
                    source_turns=source_turns,
                    matched_keyword=keyword,
                    metadata={
                        'language': lang,
                        'recall_turns': recall_turns,
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
        return self.config.recent_turns  # 默认
    
    def _resolve_by_scope(
        self,
        scope: ReferenceScope,
        ref_type: ReferenceType,
        history: List[Message],
        stance_cache: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], List[int]]:
        """
        根据范围解析内容
        
        Returns:
            (resolved_content, source_turns)
        """
        if not history:
            return None, []
        
        if scope == ReferenceScope.LAST_1_3_TURNS:
            # 最近 N 轮 (可配置)
            n = self.config.last_few_turns
            recent = history[-n*2:] if len(history) >= n*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - n*2), len(history)))
            return content, turns
        
        elif scope == ReferenceScope.LAST_5_10_TURNS:
            # 最近 5-10 轮
            n = self.config.recent_turns
            recent = history[-n*2:] if len(history) >= n*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - n*2), len(history)))
            return content, turns
        
        elif scope == ReferenceScope.CURRENT_SESSION:
            # 当前会话 (最多 session_max_turns)
            max_turns = self.config.session_max_turns
            recent = history[-max_turns*2:] if len(history) >= max_turns*2 else history
            content = self._format_messages(recent)
            turns = list(range(max(0, len(history) - max_turns*2), len(history)))
            return content, turns
        
        elif scope == ReferenceScope.LAST_SHARED_TOPIC:
            # 上一个共享主题
            return self._find_last_topic(history)
        
        elif scope == ReferenceScope.ASSISTANT_LAST_STANCE:
            # 助手最后立场
            return self._find_assistant_stance(history, stance_cache)
        
        return None, []
    
    def _format_messages(self, messages: List[Message]) -> str:
        """格式化消息列表"""
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
    
    def _find_last_topic(self, history: List[Message]) -> tuple[Optional[str], List[int]]:
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
    
    def _find_assistant_stance(
        self,
        history: List[Message],
        stance_cache: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], List[int]]:
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
