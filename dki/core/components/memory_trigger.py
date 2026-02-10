"""
Memory Trigger - 记忆触发器

核心原则: 不是每句话都进记忆，只在出现特定信号时触发。

触发信号分类:
1. 元认知表达 - "我们刚刚讨论的"、"之前你说过"
2. 状态变化 - "我现在改变想法了"、"补充一点"
3. 长期价值信号 - 偏好、立场、计划、定义

设计原则:
- 规则可配置 (支持外置配置文件)
- 支持中英文
- 后续可用分类器增强

Author: AGI Demo Project
Version: 1.0.0
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import yaml
from loguru import logger


class TriggerType(str, Enum):
    """触发类型"""
    META_COGNITIVE = "meta_cognitive"  # 元认知表达
    STATE_CHANGE = "state_change"  # 状态变化
    LONG_TERM_VALUE = "long_term_value"  # 长期价值信号
    RECALL_REQUEST = "recall_request"  # 回顾请求
    OPINION_QUERY = "opinion_query"  # 观点查询
    NONE = "none"  # 无触发


@dataclass
class TriggerResult:
    """触发结果"""
    triggered: bool
    trigger_type: TriggerType
    matched_pattern: Optional[str] = None
    confidence: float = 1.0
    extracted_topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryTriggerConfig:
    """
    Memory Trigger 配置
    
    支持从 YAML 文件加载，实现规则外置
    """
    # 元认知表达模式 (中文)
    meta_cognitive_patterns_cn: List[str] = field(default_factory=lambda: [
        r"我们(刚刚|刚才|之前|上次).*?(讨论|聊|说|谈)",
        r"(之前|上次|前几天)你(说|提到|建议)",
        r"(那件事|那个问题|那个话题)",
        r"你(还记得|记得).*?吗",
        r"我(之前|上次)(说|提到|问)过",
    ])
    
    # 元认知表达模式 (英文)
    meta_cognitive_patterns_en: List[str] = field(default_factory=lambda: [
        r"we (just|recently|previously) (discussed|talked|mentioned)",
        r"(earlier|last time|before) you (said|mentioned|suggested)",
        r"(that thing|that issue|that topic)",
        r"do you (remember|recall)",
        r"I (previously|earlier) (said|mentioned|asked)",
    ])
    
    # 状态变化模式 (中文)
    state_change_patterns_cn: List[str] = field(default_factory=lambda: [
        r"我(现在|已经)?(改变|改了|变了).*?(想法|主意|看法)",
        r"(补充|更正|纠正)一(下|点)",
        r"其实我(想|觉得|认为)",
        r"我的(偏好|喜好|习惯)是",
        r"我(不再|不想|不要).*?了",
    ])
    
    # 状态变化模式 (英文)
    state_change_patterns_en: List[str] = field(default_factory=lambda: [
        r"I('ve| have)? (changed|updated) my (mind|opinion|view)",
        r"(let me|I want to) (add|correct|clarify)",
        r"actually I (think|believe|want)",
        r"my (preference|habit) is",
        r"I (no longer|don't want to)",
    ])
    
    # 长期价值信号 (中文)
    long_term_value_patterns_cn: List[str] = field(default_factory=lambda: [
        r"请(记住|记下|保存)",
        r"我(喜欢|不喜欢|偏好|讨厌)",
        r"我的(计划|目标|打算)是",
        r"(定义|规定|约定).*?为",
        r"我(总是|通常|一般|习惯)",
        r"我是.*?(素食|过敏|不吃)",
    ])
    
    # 长期价值信号 (英文)
    long_term_value_patterns_en: List[str] = field(default_factory=lambda: [
        r"please (remember|note|save)",
        r"I (like|dislike|prefer|hate)",
        r"my (plan|goal|intention) is",
        r"(define|set|establish).*?as",
        r"I (always|usually|typically|tend to)",
        r"I am.*?(vegetarian|allergic|don't eat)",
    ])
    
    # 回顾请求模式 (中文)
    recall_request_patterns_cn: List[str] = field(default_factory=lambda: [
        r"(最近|刚才)我们(聊|讨论|谈)了(什么|哪些)",
        r"我们(讨论|聊)过(什么|哪些)(话题|问题)",
        r"(总结|回顾)一下我们的(对话|讨论)",
        r"我们(之前|上次)说(了|过)什么",
    ])
    
    # 回顾请求模式 (英文)
    recall_request_patterns_en: List[str] = field(default_factory=lambda: [
        r"what (did|have) we (discussed|talked about) (recently|earlier)",
        r"what (topics|issues) have we (discussed|covered)",
        r"(summarize|recap) our (conversation|discussion)",
        r"what did we (say|talk about) (before|earlier)",
    ])
    
    # 观点查询模式 (中文)
    opinion_query_patterns_cn: List[str] = field(default_factory=lambda: [
        r"你(对|关于).*?(有|有没有)(新|什么)(看法|想法|观点)",
        r"你(现在|目前)(怎么|如何)(看|想|认为)",
        r"你的(看法|观点|想法)(变|改变)了吗",
        r"你(还是|仍然)(认为|觉得)",
    ])
    
    # 观点查询模式 (英文)
    opinion_query_patterns_en: List[str] = field(default_factory=lambda: [
        r"do you have (any|new) (thoughts|views|opinions) (on|about)",
        r"what do you (think|believe) (now|currently)",
        r"has your (view|opinion|thought) changed",
        r"do you still (think|believe)",
    ])
    
    # 是否启用分类器增强 (后续扩展)
    use_classifier: bool = False
    classifier_model: Optional[str] = None
    classifier_threshold: float = 0.7
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MemoryTriggerConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('memory_trigger', data))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTriggerConfig":
        """从字典创建配置"""
        return cls(
            meta_cognitive_patterns_cn=data.get('meta_cognitive_patterns_cn', cls().meta_cognitive_patterns_cn),
            meta_cognitive_patterns_en=data.get('meta_cognitive_patterns_en', cls().meta_cognitive_patterns_en),
            state_change_patterns_cn=data.get('state_change_patterns_cn', cls().state_change_patterns_cn),
            state_change_patterns_en=data.get('state_change_patterns_en', cls().state_change_patterns_en),
            long_term_value_patterns_cn=data.get('long_term_value_patterns_cn', cls().long_term_value_patterns_cn),
            long_term_value_patterns_en=data.get('long_term_value_patterns_en', cls().long_term_value_patterns_en),
            recall_request_patterns_cn=data.get('recall_request_patterns_cn', cls().recall_request_patterns_cn),
            recall_request_patterns_en=data.get('recall_request_patterns_en', cls().recall_request_patterns_en),
            opinion_query_patterns_cn=data.get('opinion_query_patterns_cn', cls().opinion_query_patterns_cn),
            opinion_query_patterns_en=data.get('opinion_query_patterns_en', cls().opinion_query_patterns_en),
            use_classifier=data.get('use_classifier', False),
            classifier_model=data.get('classifier_model'),
            classifier_threshold=data.get('classifier_threshold', 0.7),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'meta_cognitive_patterns_cn': self.meta_cognitive_patterns_cn,
            'meta_cognitive_patterns_en': self.meta_cognitive_patterns_en,
            'state_change_patterns_cn': self.state_change_patterns_cn,
            'state_change_patterns_en': self.state_change_patterns_en,
            'long_term_value_patterns_cn': self.long_term_value_patterns_cn,
            'long_term_value_patterns_en': self.long_term_value_patterns_en,
            'recall_request_patterns_cn': self.recall_request_patterns_cn,
            'recall_request_patterns_en': self.recall_request_patterns_en,
            'opinion_query_patterns_cn': self.opinion_query_patterns_cn,
            'opinion_query_patterns_en': self.opinion_query_patterns_en,
            'use_classifier': self.use_classifier,
            'classifier_model': self.classifier_model,
            'classifier_threshold': self.classifier_threshold,
        }


class MemoryTrigger:
    """
    记忆触发器
    
    核心功能:
    1. 检测用户输入中的记忆相关信号
    2. 分类触发类型
    3. 提取相关主题 (可选)
    
    使用方式:
    ```python
    # 使用默认配置
    trigger = MemoryTrigger()
    
    # 使用自定义配置
    config = MemoryTriggerConfig.from_yaml("config/memory_trigger.yaml")
    trigger = MemoryTrigger(config=config)
    
    # 检测触发
    result = trigger.check("我们刚才聊了什么？")
    if result.triggered:
        print(f"触发类型: {result.trigger_type}")
    ```
    """
    
    def __init__(
        self,
        config: Optional[MemoryTriggerConfig] = None,
        language: str = "auto",  # "cn", "en", "auto"
    ):
        """
        初始化记忆触发器
        
        Args:
            config: 触发器配置 (可选，默认使用内置配置)
            language: 语言 ("cn", "en", "auto" 自动检测)
        """
        self.config = config or MemoryTriggerConfig()
        self.language = language
        
        # 编译正则表达式 (提高性能)
        self._compiled_patterns: Dict[TriggerType, Dict[str, List[re.Pattern]]] = {}
        self._compile_patterns()
        
        # 分类器 (后续扩展)
        self._classifier: Optional[Callable] = None
        if self.config.use_classifier:
            self._init_classifier()
        
        logger.info(f"MemoryTrigger initialized (language={language})")
    
    def _compile_patterns(self):
        """编译正则表达式"""
        pattern_groups = {
            TriggerType.META_COGNITIVE: {
                'cn': self.config.meta_cognitive_patterns_cn,
                'en': self.config.meta_cognitive_patterns_en,
            },
            TriggerType.STATE_CHANGE: {
                'cn': self.config.state_change_patterns_cn,
                'en': self.config.state_change_patterns_en,
            },
            TriggerType.LONG_TERM_VALUE: {
                'cn': self.config.long_term_value_patterns_cn,
                'en': self.config.long_term_value_patterns_en,
            },
            TriggerType.RECALL_REQUEST: {
                'cn': self.config.recall_request_patterns_cn,
                'en': self.config.recall_request_patterns_en,
            },
            TriggerType.OPINION_QUERY: {
                'cn': self.config.opinion_query_patterns_cn,
                'en': self.config.opinion_query_patterns_en,
            },
        }
        
        for trigger_type, lang_patterns in pattern_groups.items():
            self._compiled_patterns[trigger_type] = {}
            for lang, patterns in lang_patterns.items():
                self._compiled_patterns[trigger_type][lang] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
    
    def _init_classifier(self):
        """初始化分类器 (后续扩展)"""
        # TODO: 集成轻量级分类器 (如 DeBERTa)
        logger.warning("Classifier not implemented yet, using rule-based only")
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的中文检测: 如果包含中文字符，认为是中文
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        return 'cn' if chinese_ratio > 0.3 else 'en'
    
    def check(self, message: str) -> TriggerResult:
        """
        检查消息是否触发记忆信号
        
        Args:
            message: 用户消息
            
        Returns:
            TriggerResult 包含触发信息
        """
        if not message or not message.strip():
            return TriggerResult(triggered=False, trigger_type=TriggerType.NONE)
        
        # 确定语言
        if self.language == "auto":
            lang = self._detect_language(message)
        else:
            lang = self.language
        
        # 按优先级检查各类触发
        # 优先级: RECALL_REQUEST > OPINION_QUERY > META_COGNITIVE > STATE_CHANGE > LONG_TERM_VALUE
        check_order = [
            TriggerType.RECALL_REQUEST,
            TriggerType.OPINION_QUERY,
            TriggerType.META_COGNITIVE,
            TriggerType.STATE_CHANGE,
            TriggerType.LONG_TERM_VALUE,
        ]
        
        for trigger_type in check_order:
            result = self._check_patterns(message, trigger_type, lang)
            if result.triggered:
                return result
        
        # 如果启用分类器，进行二次检查
        if self.config.use_classifier and self._classifier:
            return self._classifier_check(message)
        
        return TriggerResult(triggered=False, trigger_type=TriggerType.NONE)
    
    def detect(self, message: str) -> TriggerResult:
        """
        检测消息是否触发记忆信号 (check 的别名)
        
        Args:
            message: 用户消息
            
        Returns:
            TriggerResult 包含触发信息
        """
        return self.check(message)
    
    def _check_patterns(
        self,
        message: str,
        trigger_type: TriggerType,
        lang: str,
    ) -> TriggerResult:
        """检查特定类型的模式"""
        patterns = self._compiled_patterns.get(trigger_type, {}).get(lang, [])
        
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                # 尝试提取主题
                topic = self._extract_topic(message, match, trigger_type)
                
                return TriggerResult(
                    triggered=True,
                    trigger_type=trigger_type,
                    matched_pattern=pattern.pattern,
                    confidence=1.0,
                    extracted_topic=topic,
                    metadata={
                        'language': lang,
                        'match_start': match.start(),
                        'match_end': match.end(),
                        'matched_text': match.group(),
                    },
                )
        
        return TriggerResult(triggered=False, trigger_type=TriggerType.NONE)
    
    def _extract_topic(
        self,
        message: str,
        match: re.Match,
        trigger_type: TriggerType,
    ) -> Optional[str]:
        """
        从消息中提取主题
        
        简单实现: 提取匹配位置之后的内容
        后续可用 NER 或 LLM 增强
        """
        # 获取匹配后的文本
        after_match = message[match.end():].strip()
        
        # 简单清理
        if after_match:
            # 移除常见的连接词
            connectors = ['的', '是', '吗', '呢', '啊', '了', '?', '？', '!', '！']
            for conn in connectors:
                after_match = after_match.lstrip(conn)
            
            # 截取前 50 个字符作为主题
            if len(after_match) > 50:
                after_match = after_match[:50] + "..."
            
            return after_match if after_match else None
        
        return None
    
    def _classifier_check(self, message: str) -> TriggerResult:
        """使用分类器检查 (后续扩展)"""
        # TODO: 实现分类器检查
        return TriggerResult(triggered=False, trigger_type=TriggerType.NONE)
    
    def add_pattern(
        self,
        trigger_type: TriggerType,
        pattern: str,
        language: str = "cn",
    ):
        """
        动态添加模式
        
        Args:
            trigger_type: 触发类型
            pattern: 正则表达式模式
            language: 语言
        """
        if trigger_type not in self._compiled_patterns:
            self._compiled_patterns[trigger_type] = {'cn': [], 'en': []}
        
        if language not in self._compiled_patterns[trigger_type]:
            self._compiled_patterns[trigger_type][language] = []
        
        self._compiled_patterns[trigger_type][language].append(
            re.compile(pattern, re.IGNORECASE)
        )
        
        logger.info(f"Added pattern for {trigger_type.value} ({language}): {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pattern_counts = {}
        for trigger_type, lang_patterns in self._compiled_patterns.items():
            pattern_counts[trigger_type.value] = {
                lang: len(patterns) for lang, patterns in lang_patterns.items()
            }
        
        return {
            'language': self.language,
            'use_classifier': self.config.use_classifier,
            'pattern_counts': pattern_counts,
        }
    
    def update_config(
        self,
        enabled: Optional[bool] = None,
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        运行时更新配置
        
        Args:
            enabled: 是否启用 (预留，当前始终启用)
            custom_patterns: 自定义规则列表，格式:
                [{"trigger_type": "meta_cognitive", "pattern": "...", "language": "cn"}, ...]
        """
        if custom_patterns:
            for rule in custom_patterns:
                trigger_type_str = rule.get('trigger_type', '')
                pattern = rule.get('pattern', '')
                language = rule.get('language', 'cn')
                
                try:
                    trigger_type = TriggerType(trigger_type_str)
                    self.add_pattern(trigger_type, pattern, language)
                except ValueError:
                    logger.warning(f"Unknown trigger type: {trigger_type_str}")
        
        logger.info(f"MemoryTrigger config updated")


# 便捷函数
def create_memory_trigger(
    config_path: Optional[str] = None,
    language: str = "auto",
) -> MemoryTrigger:
    """
    创建 Memory Trigger 实例
    
    Args:
        config_path: 配置文件路径 (可选)
        language: 语言
        
    Returns:
        MemoryTrigger 实例
    """
    if config_path:
        config = MemoryTriggerConfig.from_yaml(config_path)
    else:
        config = MemoryTriggerConfig()
    
    return MemoryTrigger(config=config, language=language)
