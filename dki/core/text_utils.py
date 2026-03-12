"""
DKI 文本处理工具

共享工具函数:
- ThinkContentFilter: 可配置的思考内容过滤器 (支持外置正则)
- strip_think_content: 兼容旧接口的过滤函数
- StreamThinkDetector: 流式输出思考内容检测器 (v5.9, 推荐)
- StreamThinkFilter: 向后兼容的流式过滤器
- estimate_tokens_fast: 快速 token 估算 (不依赖 tokenizer, 略微高估)
- detect_vague_reference: 检测用户输入中的模糊指代 (v6.5)
- build_clarification_prompt: 构建澄清提示词 (v6.5)

关于思考内容过滤 (v5.9 架构):
================================
不同模型使用不同的格式输出思考/推理过程:

1. DeepSeek-R1 系列: <think>推理内容</think>最终回复
   - 完整格式: <think>...</think>
   - 截断格式: 推理内容</think> (无起始标记)
   
2. 通义千问 Qwen 系列: Thinking Process:\n分析内容...\n最终回复
   - 纯文本格式, 无标签包裹
   - 可能包含 "Analyze the Request", "Drafting the Response" 等英文分析块

3. Qwen3/3.5 系列: <|think|>推理内容<|/think|>最终回复
   - 特殊标签格式

4. Claude 系列: <antThinking>推理内容</antThinking>

5. 其他模型: 可能使用 "思考过程:", "推理过程:" 等中文标记

v5.9 架构变更:
- 流式输出时不再实时过滤, 而是检测并标记思考内容
- 通过 show_thinking 配置控制客户端是否显示
- 最终存储时对完整文本应用全量正则过滤
- StreamThinkDetector.feed() 返回 [(event_type, text), ...] 事件列表
  event_type: "token" (正常内容) | "thinking" (思考内容)

过滤规则来源:
- 内置规则: DeepSeek-R1 的 <think>...</think> (始终生效)
- 外置规则: 通过 config.dki.think_filter.rules 配置

Author: AGI Demo Project
Version: 5.9.0
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
# Think 内容过滤规则
# ============================================================

@dataclass
class ThinkFilterRule:
    """单条思考内容过滤规则"""
    name: str                          # 规则名称 (用于日志)
    pattern: re.Pattern                # 编译后的正则
    action: str = "remove"             # "remove" | "remove_to"
    scope: str = "all"                 # "output" | "history" | "all"
    stream_aware: bool = False         # 是否支持流式过滤
    # 原始 pattern 字符串 (用于序列化/调试)
    raw_pattern: str = ""
    
    def matches(self, text: str) -> bool:
        """检查文本是否匹配此规则"""
        return bool(self.pattern.search(text))
    
    def apply(self, text: str) -> Tuple[str, bool]:
        """
        应用规则到文本
        
        Returns:
            (cleaned_text, was_applied)
        """
        if self.action == "remove":
            result = self.pattern.sub('', text)
            applied = result != text
            return result, applied
        elif self.action == "remove_to":
            # 从文本开头移除到匹配位置 (含匹配内容)
            match = self.pattern.search(text)
            if match:
                result = text[match.end():]
                return result, True
            return text, False
        return text, False


# ============================================================
# 内置规则 (始终生效)
# ============================================================

# DeepSeek-R1: 完整格式 <think>...</think>
_BUILTIN_THINK_FULL = ThinkFilterRule(
    name="builtin_deepseek_think_full",
    pattern=re.compile(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
        re.DOTALL | re.IGNORECASE,
    ),
    action="remove",
    scope="all",
    stream_aware=True,
    raw_pattern=r'<\s*think\s*>.*?<\s*/\s*think\s*>',
)

# DeepSeek-R1: 仅有结束标记 (从文本开头到 </think>)
_BUILTIN_THINK_TAIL = ThinkFilterRule(
    name="builtin_deepseek_think_tail",
    pattern=re.compile(
        r'^.*?<\s*/\s*think\s*>',
        re.DOTALL | re.IGNORECASE,
    ),
    action="remove",  # 特殊处理: 仅当无 <think> 时才应用
    scope="all",
    stream_aware=True,
    raw_pattern=r'^.*?<\s*/\s*think\s*>',
)

_BUILTIN_RULES = [_BUILTIN_THINK_FULL, _BUILTIN_THINK_TAIL]

# 用于检测是否存在 <think> 开标签
_HAS_THINK_OPEN_RE = re.compile(r'<\s*think\s*>', re.IGNORECASE)
_HAS_THINK_CLOSE_RE = re.compile(r'<\s*/\s*think\s*>', re.IGNORECASE)


# ============================================================
# 正则标志解析
# ============================================================

_FLAG_MAP = {
    "DOTALL": re.DOTALL,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "VERBOSE": re.VERBOSE,
    "S": re.DOTALL,
    "I": re.IGNORECASE,
    "M": re.MULTILINE,
    "X": re.VERBOSE,
}


def _parse_flags(flags_str: str) -> int:
    """解析正则标志字符串, 如 'DOTALL|IGNORECASE'"""
    if not flags_str:
        return 0
    combined = 0
    for part in flags_str.split("|"):
        part = part.strip().upper()
        if part in _FLAG_MAP:
            combined |= _FLAG_MAP[part]
        else:
            logger.warning(f"Unknown regex flag: {part}")
    return combined


def parse_filter_rules(rules_config: List[Dict[str, Any]]) -> List[ThinkFilterRule]:
    """
    从配置字典列表解析过滤规则
    
    Args:
        rules_config: 配置中的 rules 列表, 每个元素包含:
            - name: str
            - pattern: str (正则表达式)
            - flags: str (如 "DOTALL|IGNORECASE")
            - action: str ("remove" | "remove_to")
            - scope: str ("output" | "history" | "all")
            - stream_aware: bool
    
    Returns:
        编译后的 ThinkFilterRule 列表
    """
    parsed = []
    for rule_dict in rules_config:
        name = rule_dict.get("name", "unnamed_rule")
        raw_pattern = rule_dict.get("pattern", "")
        if not raw_pattern:
            logger.warning(f"Think filter rule '{name}' has empty pattern, skipping")
            continue
        
        flags_str = rule_dict.get("flags", "")
        flags = _parse_flags(flags_str)
        
        try:
            compiled = re.compile(raw_pattern, flags)
        except re.error as e:
            logger.error(
                f"Think filter rule '{name}' has invalid regex: {e}. "
                f"Pattern: {raw_pattern}"
            )
            continue
        
        parsed.append(ThinkFilterRule(
            name=name,
            pattern=compiled,
            action=rule_dict.get("action", "remove"),
            scope=rule_dict.get("scope", "all"),
            stream_aware=rule_dict.get("stream_aware", False),
            raw_pattern=raw_pattern,
        ))
        logger.debug(f"Loaded think filter rule: {name}")
    
    return parsed


# ============================================================
# ThinkContentFilter — 核心过滤器
# ============================================================

class ThinkContentFilter:
    """
    可配置的思考内容过滤器
    
    支持:
    1. 内置规则 (DeepSeek-R1 <think>...</think>)
    2. 外置规则 (从配置文件加载)
    3. 按 scope 过滤 (output / history / all)
    4. 流式感知标记 (供 StreamThinkFilter 使用)
    
    使用方式:
        # 方式 1: 全局单例 (推荐)
        from dki.core.text_utils import get_think_filter
        f = get_think_filter()
        clean, stripped = f.strip(text, scope="output")
        
        # 方式 2: 从配置初始化
        f = ThinkContentFilter.from_config(config_dict)
        clean, stripped = f.strip(text)
    """
    
    def __init__(
        self,
        custom_rules: Optional[List[ThinkFilterRule]] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._builtin_rules = list(_BUILTIN_RULES)
        self._custom_rules: List[ThinkFilterRule] = custom_rules or []
        
        total = len(self._builtin_rules) + len(self._custom_rules)
        logger.info(
            f"ThinkContentFilter initialized: "
            f"{len(self._builtin_rules)} builtin + "
            f"{len(self._custom_rules)} custom = {total} rules"
        )
    
    @classmethod
    def from_config(cls, think_filter_config: Dict[str, Any]) -> "ThinkContentFilter":
        """
        从配置字典创建过滤器
        
        Args:
            think_filter_config: config.dki.think_filter 节点
                {
                    "enabled": true,
                    "rules": [
                        {"name": "...", "pattern": "...", "flags": "...", ...},
                        ...
                    ]
                }
        """
        enabled = think_filter_config.get("enabled", True)
        rules_config = think_filter_config.get("rules", [])
        custom_rules = parse_filter_rules(rules_config) if rules_config else []
        return cls(custom_rules=custom_rules, enabled=enabled)
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @property
    def all_rules(self) -> List[ThinkFilterRule]:
        """所有规则 (内置 + 自定义)"""
        return self._builtin_rules + self._custom_rules
    
    @property
    def custom_rules(self) -> List[ThinkFilterRule]:
        return self._custom_rules
    
    def get_rules_for_scope(self, scope: str = "all") -> List[ThinkFilterRule]:
        """获取适用于指定 scope 的规则"""
        return [
            r for r in self.all_rules
            if r.scope == "all" or r.scope == scope
        ]
    
    def get_stream_aware_rules(self) -> List[ThinkFilterRule]:
        """获取支持流式过滤的规则"""
        return [r for r in self.all_rules if r.stream_aware]
    
    def strip(
        self,
        text: str,
        scope: str = "all",
    ) -> Tuple[str, bool]:
        """
        过滤文本中的思考内容
        
        Args:
            text: 原始文本
            scope: 过滤范围 ("output" | "history" | "all")
        
        Returns:
            (cleaned_text, was_stripped)
        """
        if not self._enabled or not text:
            return text, False
        
        original_len = len(text)
        result = text
        any_stripped = False
        applied_rules = []
        
        # Step 1: 应用内置 DeepSeek-R1 规则 (特殊逻辑)
        result, builtin_stripped = self._apply_builtin_rules(result)
        if builtin_stripped:
            any_stripped = True
            applied_rules.append("builtin_deepseek")
        
        # Step 2: 应用自定义规则 (按配置顺序)
        for rule in self._custom_rules:
            if rule.scope != "all" and rule.scope != scope:
                continue
            
            new_result, applied = rule.apply(result)
            if applied:
                result = new_result
                any_stripped = True
                applied_rules.append(rule.name)
        
        # Step 3: 清理
        if any_stripped:
            result = result.lstrip('\n\r\t ')
            result = re.sub(r'\n{3,}', '\n\n', result)
            
            removed_len = original_len - len(result)
            logger.debug(
                f"Think content stripped: removed {removed_len} chars "
                f"({original_len} -> {len(result)}), "
                f"rules: {applied_rules}"
            )
        
        return result, any_stripped
    
    def _apply_builtin_rules(self, text: str) -> Tuple[str, bool]:
        """
        应用内置 DeepSeek-R1 规则
        
        保持原有的两阶段逻辑:
        1. 先尝试移除完整的 <think>...</think> 块
        2. 如果没找到完整块, 检查仅有 </think> 的情况
        """
        result = text
        stripped = False
        
        # Step 1: 完整 <think>...</think>
        if _BUILTIN_THINK_FULL.pattern.search(result):
            result = _BUILTIN_THINK_FULL.pattern.sub('', result)
            stripped = True
        
        # Step 2: 仅有 </think> (无对应 <think>)
        if not stripped and _HAS_THINK_CLOSE_RE.search(result):
            if not _HAS_THINK_OPEN_RE.search(result):
                result = _BUILTIN_THINK_TAIL.pattern.sub('', result)
                stripped = True
        
        return result, stripped
    
    def info(self) -> Dict[str, Any]:
        """返回过滤器信息 (用于调试/API)"""
        return {
            "enabled": self._enabled,
            "builtin_rules": len(self._builtin_rules),
            "custom_rules": [
                {
                    "name": r.name,
                    "scope": r.scope,
                    "action": r.action,
                    "stream_aware": r.stream_aware,
                    "pattern": r.raw_pattern[:80] + "..." if len(r.raw_pattern) > 80 else r.raw_pattern,
                }
                for r in self._custom_rules
            ],
            "total_rules": len(self.all_rules),
        }


# ============================================================
# StreamThinkDetector — 流式输出思考内容检测器
# ============================================================

class StreamThinkDetector:
    """
    流式输出的思考内容检测器 (v5.9)
    
    设计理念:
      不再在流式过程中过滤内容, 而是检测每个 token 是否属于思考块。
      调用方根据检测结果决定:
      - show_thinking=true  → 思考内容用 type="thinking" 发送, 客户端可折叠显示
      - show_thinking=false → 思考内容不发送给客户端
      最终存储时, 对完整文本应用全量正则过滤。
    
    状态机:
    1. NORMAL    — 正常输出内容
    2. BUFFERING — 缓冲中 (检测到可能的思考标记前缀, 尚未确认)
    3. THINKING  — 思考块中 (已确认进入思考块)
    
    使用方式:
        detector = StreamThinkDetector()
        for token in stream:
            events = detector.feed(token)
            for evt_type, evt_text in events:
                if evt_type == "token":
                    yield {"type": "token", "content": evt_text}
                elif evt_type == "thinking":
                    if show_thinking:
                        yield {"type": "thinking", "content": evt_text}
        # 流结束
        events = detector.flush()
        for evt_type, evt_text in events:
            ...
        # 获取过滤后的干净文本
        clean = detector.get_clean_text()
    """
    
    # 确认进入思考块的完整标记
    _ENTER_MARKS = [
        re.compile(r'<\s*think\s*>', re.IGNORECASE),
        re.compile(r'<\|think\|>', re.IGNORECASE),
        re.compile(r'(?:^|\n)Thinking Process:\s*\n', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?:^|\n)(?:思考过程|推理过程|分析过程)[：:]\s*\n', re.MULTILINE),
    ]
    
    # 思考块结束标记
    _EXIT_MARKS = [
        re.compile(r'<\s*/\s*think\s*>', re.IGNORECASE),
        re.compile(r'<\|/think\|>', re.IGNORECASE),
    ]
    
    # 可能是思考标记前缀的模式 (用于缓冲判断)
    # 注意: 必须覆盖从最短前缀开始的所有可能
    # 例如 "<" → "<t" → "<th" → ... → "<think" → "<think>" 全链路
    _PREFIXES = [
        # DeepSeek-R1: <think> — 从 "<" 开始, t 是可选的 (覆盖纯 "<")
        re.compile(r'<\s*(?:t(?:h(?:i(?:n(?:k\s*)?)?)?)?)?$', re.IGNORECASE),
        # Qwen3: <|think|> — 从 "<|" 开始
        re.compile(r'<\|(?:t(?:h(?:i(?:n(?:k(?:\|)?)?)?)?)?)?$', re.IGNORECASE),
        # 纯文本: Thinking Process: — 需要覆盖完整前缀链
        # T → Th → ... → Thinking → Thinking  → Thinking P → ... → Thinking Process:
        re.compile(r'(?:^|\n)T(?:h(?:i(?:n(?:k(?:i(?:n(?:g(?:\s(?:P(?:r(?:o(?:c(?:e(?:s(?:s(?::)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?$', re.IGNORECASE),
        # 中文: 思考过程: — 覆盖 思 → 思考 → 思考过 → 思考过程 → 思考过程:
        re.compile(r'(?:^|\n)(?:思(?:考(?:过(?:程(?:[：:])?)?)?)?)?$'),
        # 中文: 推理过程: — 覆盖 推 → 推理 → 推理过 → 推理过程 → 推理过程:
        re.compile(r'(?:^|\n)(?:推(?:理(?:过(?:程(?:[：:])?)?)?)?)?$'),
    ]
    
    def __init__(self, buffer_max_chars: int = 200):
        self._state = "NORMAL"
        self._buffer = ""
        self._full_text = ""
        self._buffer_max = buffer_max_chars
    
    @property
    def state(self) -> str:
        return self._state
    
    @property
    def full_text(self) -> str:
        """完整的原始文本 (未过滤)"""
        return self._full_text
    
    def feed(self, token: str) -> List[Tuple[str, str]]:
        """
        输入一个 token, 返回事件列表
        
        每个事件是 (event_type, text):
        - ("token", text)    — 正常输出内容
        - ("thinking", text) — 思考内容
        
        Returns:
            事件列表 (可能为空, 表示正在缓冲)
        """
        if not token:
            return []
        
        self._full_text += token
        events: List[Tuple[str, str]] = []
        
        if self._state == "NORMAL":
            self._buffer += token
            events = self._check_normal()
        elif self._state == "BUFFERING":
            self._buffer += token
            events = self._check_buffering()
        elif self._state == "THINKING":
            self._buffer += token
            events = self._check_thinking()
        
        return events
    
    def flush(self) -> List[Tuple[str, str]]:
        """
        流结束后释放缓冲区
        
        Returns:
            剩余事件列表
        """
        events: List[Tuple[str, str]] = []
        if self._buffer:
            if self._state == "THINKING":
                events.append(("thinking", self._buffer))
            elif self._state == "BUFFERING":
                # 未确认为思考内容, 作为正常输出
                events.append(("token", self._buffer))
            else:
                events.append(("token", self._buffer))
            self._buffer = ""
        self._state = "NORMAL"
        return events
    
    def get_clean_text(self) -> str:
        """
        获取过滤后的干净文本 (流结束后调用)
        
        对完整累积文本应用全量正则过滤。
        """
        f = get_think_filter()
        clean, _ = f.strip(self._full_text, scope="output")
        return clean
    
    def _check_normal(self) -> List[Tuple[str, str]]:
        """NORMAL 状态: 检查是否可能进入思考块"""
        events: List[Tuple[str, str]] = []
        
        # 检查是否完整匹配思考块入口
        for mark in self._ENTER_MARKS:
            m = mark.search(self._buffer)
            if m:
                # 入口标记之前的内容作为正常输出
                before = self._buffer[:m.start()]
                if before:
                    events.append(("token", before))
                # 入口标记本身作为思考内容
                events.append(("thinking", self._buffer[m.start():]))
                self._buffer = ""
                self._state = "THINKING"
                return events
        
        # 检查尾部是否可能是思考标记前缀
        tail = self._buffer[-80:] if len(self._buffer) > 80 else self._buffer
        for prefix_re in self._PREFIXES:
            if prefix_re.search(tail):
                # 可能是前缀, 进入缓冲
                self._state = "BUFFERING"
                return events  # 暂不输出
        
        # 安全: 输出缓冲区内容
        if self._buffer:
            events.append(("token", self._buffer))
            self._buffer = ""
        return events
    
    def _check_buffering(self) -> List[Tuple[str, str]]:
        """BUFFERING 状态: 继续缓冲, 判断是否确认进入思考块"""
        events: List[Tuple[str, str]] = []
        
        # 检查是否匹配思考块入口
        for mark in self._ENTER_MARKS:
            m = mark.search(self._buffer)
            if m:
                before = self._buffer[:m.start()]
                if before:
                    events.append(("token", before))
                events.append(("thinking", self._buffer[m.start():]))
                self._buffer = ""
                self._state = "THINKING"
                return events
        
        # 缓冲区过大: 不是思考标记, 释放
        if len(self._buffer) > self._buffer_max:
            events.append(("token", self._buffer))
            self._buffer = ""
            self._state = "NORMAL"
            return events
        
        # 检查是否仍可能匹配前缀
        tail = self._buffer[-80:] if len(self._buffer) > 80 else self._buffer
        still_possible = any(
            prefix_re.search(tail) for prefix_re in self._PREFIXES
        )
        
        if not still_possible:
            events.append(("token", self._buffer))
            self._buffer = ""
            self._state = "NORMAL"
        
        return events
    
    def _check_thinking(self) -> List[Tuple[str, str]]:
        """THINKING 状态: 检测是否到达思考块结尾"""
        events: List[Tuple[str, str]] = []
        
        for mark in self._EXIT_MARKS:
            m = mark.search(self._buffer)
            if m:
                # 结束标记之前 (含标记) 作为思考内容
                think_part = self._buffer[:m.end()]
                after = self._buffer[m.end():]
                if think_part:
                    events.append(("thinking", think_part))
                self._buffer = ""
                self._state = "NORMAL"
                # 结束标记之后的内容继续处理
                if after:
                    self._buffer = after
                    events.extend(self._check_normal())
                return events
        
        # 未结束: 定期释放思考内容 (避免缓冲区过大)
        if len(self._buffer) > 2000:
            events.append(("thinking", self._buffer))
            self._buffer = ""
        
        return events


# ============================================================
# StreamThinkFilter — 向后兼容包装器
# ============================================================

class StreamThinkFilter:
    """
    向后兼容的 StreamThinkFilter
    
    内部使用 StreamThinkDetector, 行为与旧版一致:
    - feed() 返回应发送给客户端的文本 (过滤掉思考内容)
    - flush() 返回缓冲区残留
    - get_clean_full_text() 返回全量过滤后的文本
    
    新代码建议直接使用 StreamThinkDetector。
    """
    
    def __init__(
        self,
        think_filter: Optional[ThinkContentFilter] = None,
        buffer_max_chars: int = 500,
    ):
        self._detector = StreamThinkDetector(buffer_max_chars=buffer_max_chars)
        self._filter = think_filter
    
    @property
    def state(self) -> str:
        return self._detector.state
    
    @property
    def full_text(self) -> str:
        return self._detector.full_text
    
    def feed(self, token: str) -> str:
        events = self._detector.feed(token)
        # 只返回 "token" 类型, 过滤掉 "thinking"
        return "".join(text for evt_type, text in events if evt_type == "token")
    
    def flush(self) -> str:
        events = self._detector.flush()
        return "".join(text for evt_type, text in events if evt_type == "token")
    
    def get_clean_full_text(self) -> str:
        if self._filter:
            clean, _ = self._filter.strip(self._detector.full_text, scope="output")
            return clean
        return self._detector.full_text


# ============================================================
# 全局单例管理
# ============================================================

_global_filter: Optional[ThinkContentFilter] = None
_global_show_thinking: bool = True


def init_think_filter(config: Optional[Dict[str, Any]] = None) -> ThinkContentFilter:
    """
    初始化全局思考内容过滤器
    
    应在应用启动时调用一次。
    
    Args:
        config: dki.think_filter 配置节点
            如果为 None, 仅使用内置规则
    
    Returns:
        初始化后的 ThinkContentFilter
    """
    global _global_filter, _global_show_thinking
    
    if config:
        _global_filter = ThinkContentFilter.from_config(config)
        # 读取 show_thinking 配置
        show_val = config.get("show_thinking", True)
        if isinstance(show_val, str):
            _global_show_thinking = show_val.lower() in ("true", "1", "yes")
        else:
            _global_show_thinking = bool(show_val)
    else:
        _global_filter = ThinkContentFilter()
        _global_show_thinking = True
    
    logger.info(
        f"Global ThinkContentFilter initialized: "
        f"{_global_filter.info()}, show_thinking={_global_show_thinking}"
    )
    return _global_filter


def get_think_filter() -> ThinkContentFilter:
    """
    获取全局思考内容过滤器
    
    如果尚未初始化, 返回仅包含内置规则的默认实例。
    """
    global _global_filter
    if _global_filter is None:
        _global_filter = ThinkContentFilter()
    return _global_filter


def get_show_thinking() -> bool:
    """
    获取是否在客户端显示思考过程的配置
    
    Returns:
        True — 思考内容通过 SSE event "thinking" 发送给客户端
        False — 思考内容不发送, 静默过滤
    """
    return _global_show_thinking


def create_stream_filter(
    buffer_max_chars: int = 500,
) -> StreamThinkFilter:
    """
    创建流式思考内容过滤器 (向后兼容)
    
    新代码建议使用 create_stream_detector()。
    
    Args:
        buffer_max_chars: 缓冲区最大字符数
    
    Returns:
        StreamThinkFilter 实例
    """
    return StreamThinkFilter(
        think_filter=get_think_filter(),
        buffer_max_chars=buffer_max_chars,
    )


def create_stream_detector(
    buffer_max_chars: int = 200,
) -> StreamThinkDetector:
    """
    创建流式思考内容检测器 (v5.9 推荐)
    
    配合 get_show_thinking() 使用:
        detector = create_stream_detector()
        show = get_show_thinking()
        for token in stream:
            for evt_type, evt_text in detector.feed(token):
                if evt_type == "token":
                    yield {"type": "token", "content": evt_text}
                elif evt_type == "thinking" and show:
                    yield {"type": "thinking", "content": evt_text}
        # 流结束
        for evt_type, evt_text in detector.flush():
            ...
        clean_text = detector.get_clean_text()
    
    Args:
        buffer_max_chars: 缓冲区最大字符数
    
    Returns:
        StreamThinkDetector 实例
    """
    return StreamThinkDetector(buffer_max_chars=buffer_max_chars)


# ============================================================
# 兼容旧接口
# ============================================================

def strip_think_content(text: str, scope: str = "all") -> Tuple[str, bool]:
    """
    移除模型输出中的思考/推理内容 (兼容旧接口)
    
    使用全局 ThinkContentFilter, 支持内置 + 外置规则。
    
    Args:
        text: 原始文本 (可能包含思考内容)
        scope: 过滤范围 ("output" | "history" | "all")
        
    Returns:
        (cleaned_text, was_stripped) — 清理后的文本和是否执行了移除
    """
    return get_think_filter().strip(text, scope=scope)


# ============================================================
# v6.5: 模糊指代检测与澄清提示词
# ============================================================

# 模糊指代模式 (中文) — 这些表达无法通过简单的历史召回定位到具体事件
# 特征: 时间模糊 + 事件模糊, 两者同时缺失
_VAGUE_REFERENCE_PATTERNS_CN = [
    # "前段时间/上次/之前 + 说的/聊的/提到的 + 那件事/那个"
    re.compile(r'(?:前段时间|上次|之前|以前|前几天|好久之前|很久以前|有一次).*?(?:说的?|聊的?|提到的?|讨论的?|谈的?|问的?).*?(?:那[件个]|什么|怎么)', re.DOTALL),
    # "你还记得 + 模糊事件"
    re.compile(r'你还?记得.*?(?:那[件个]事|那次|那个|吗)'),
    # "上次和你说的那件事 + 你怎么看/想"
    re.compile(r'(?:上次|之前|前段时间).*?(?:和你|跟你).*?(?:说|聊|提).*?(?:怎么[看想]|什么[看想]法|有什么|进展)'),
    # "那件事你现在怎么想" (无上下文的 "那件事")
    re.compile(r'^(?:那[件个]事|那个问题|那个话题).*?(?:怎么[看想办]|什么[看想]法|进展|结果|后来)'),
    # "我们之前讨论的" (无具体主题)
    re.compile(r'(?:我们|咱们?)(?:之前|上次|前段时间)(?:讨论|聊|说|谈)(?:的|过的?)(?:那[件个]|什么|怎么样)'),
]

# 模糊指代模式 (英文)
_VAGUE_REFERENCE_PATTERNS_EN = [
    re.compile(r'(?:last time|before|earlier|a while ago|some time ago).*?(?:we|you|I).*?(?:talked|discussed|mentioned|said).*?(?:that thing|that|about)', re.IGNORECASE),
    re.compile(r'(?:do you|you)\s+remember.*?(?:that thing|what we|that time)', re.IGNORECASE),
    re.compile(r'(?:that thing|that matter|that issue)\s+(?:we|you|I).*?(?:how|what|any)', re.IGNORECASE),
    re.compile(r'^(?:that thing|that matter|that issue).*?(?:think|opinion|progress|update)', re.IGNORECASE),
    re.compile(r'(?:what|how)\s+(?:do you|about).*?(?:that thing|what we discussed|that matter)', re.IGNORECASE),
]

# 具体事件关键词 — 如果用户提到了具体主题, 则不算模糊
_SPECIFIC_TOPIC_INDICATORS_CN = re.compile(
    r'(?:关于|有关|涉及|针对|对于|说到|提到)\s*(?!的?[那这][件个]|的?什么|的?哪)(?:[\u4e00-\u9fff]{2,}|[a-zA-Z]\w{2,})'
)
_SPECIFIC_TOPIC_INDICATORS_EN = re.compile(
    r'(?:about|regarding|concerning|related to)\s+(?!that\b|this\b|what\b|which\b|it\b|the\s+thing)\w{3,}',
    re.IGNORECASE,
)


@dataclass
class VagueReferenceResult:
    """模糊指代检测结果"""
    is_vague: bool = False
    confidence: float = 0.0
    matched_pattern: str = ""
    language: str = "cn"
    
    def __bool__(self):
        return self.is_vague


def detect_vague_reference(query: str) -> VagueReferenceResult:
    """
    检测用户输入中的模糊指代
    
    模糊指代的定义:
    - 用户引用了过去的对话/事件, 但没有给出足够的上下文来定位具体内容
    - 例如: "前段时间和你说的那件事你现在怎么想"
    - 这类查询即使通过多信号召回也无法准确定位, 因为缺少关键词和语义锚点
    
    与 ReferenceResolver 的区别:
    - ReferenceResolver 处理可解析的指代 (如 "刚才说的" → 最近 1-3 轮)
    - detect_vague_reference 检测无法解析的模糊指代 (需要用户澄清)
    
    Args:
        query: 用户输入
        
    Returns:
        VagueReferenceResult 包含检测结果
    """
    if not query or not query.strip():
        return VagueReferenceResult()
    
    query = query.strip()
    
    # 检测语言
    chinese_chars = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    total_chars = len(query.replace(' ', ''))
    is_chinese = (chinese_chars / max(total_chars, 1)) > 0.3
    
    # 如果用户已经给出了具体主题, 不算模糊
    if is_chinese and _SPECIFIC_TOPIC_INDICATORS_CN.search(query):
        return VagueReferenceResult(language="cn")
    if not is_chinese and _SPECIFIC_TOPIC_INDICATORS_EN.search(query):
        return VagueReferenceResult(language="en")
    
    # 匹配模糊指代模式
    patterns = _VAGUE_REFERENCE_PATTERNS_CN if is_chinese else _VAGUE_REFERENCE_PATTERNS_EN
    lang = "cn" if is_chinese else "en"
    
    for pattern in patterns:
        match = pattern.search(query)
        if match:
            return VagueReferenceResult(
                is_vague=True,
                confidence=0.85,
                matched_pattern=match.group(0),
                language=lang,
            )
    
    return VagueReferenceResult(language=lang)


# 澄清提示词模板
_CLARIFICATION_SYSTEM_PROMPT_CN = (
    "【重要指令】当用户的提问中包含模糊的时间或事件指代"
    "（如『前段时间说的那件事』、『上次聊的那个』等），"
    "而你无法从已有的对话历史中确定用户具体指的是哪件事或哪次对话时，"
    "请不要猜测或编造内容。你应该：\n"
    "1. 礼貌地告知用户你不确定具体指的是哪件事\n"
    "2. 请用户提供更多线索，例如：\n"
    "   - 大概的时间范围（哪天、哪周）\n"
    "   - 讨论的主题或关键词\n"
    "   - 当时的具体场景或结论\n"
    "3. 如果对话历史中有多个可能匹配的话题，可以列出让用户确认\n\n"
    "示例回复：\n"
    "「我不太确定您指的是哪次对话。我们之前聊过 [话题A]、[话题B] 等内容，"
    "您能告诉我大概是什么时候、关于什么主题的吗？这样我可以更准确地回答您。」"
)

_CLARIFICATION_SYSTEM_PROMPT_EN = (
    "[IMPORTANT INSTRUCTION] When the user's question contains vague time or event references "
    "(e.g., 'that thing we discussed before', 'what we talked about last time'), "
    "and you cannot determine the specific conversation or event from the available history, "
    "DO NOT guess or fabricate content. Instead:\n"
    "1. Politely let the user know you're not sure which specific conversation they're referring to\n"
    "2. Ask the user to provide more context, such as:\n"
    "   - Approximate time frame (which day, which week)\n"
    "   - The topic or keywords discussed\n"
    "   - Specific conclusions or outcomes from that conversation\n"
    "3. If there are multiple possible matching topics in the history, list them for confirmation\n\n"
    "Example response:\n"
    "\"I'm not sure which conversation you're referring to. We've previously discussed [Topic A], "
    "[Topic B], etc. Could you tell me approximately when it was and what topic it was about? "
    "That way I can give you a more accurate answer.\""
)


def build_clarification_instruction(language: str = "cn") -> str:
    """
    构建澄清指令 (注入到 system message 中)
    
    当检测到模糊指代时, 将此指令追加到 system message,
    引导模型主动向用户澄清而非猜测。
    
    Args:
        language: "cn" 或 "en"
        
    Returns:
        澄清指令文本
    """
    if language == "cn":
        return _CLARIFICATION_SYSTEM_PROMPT_CN
    return _CLARIFICATION_SYSTEM_PROMPT_EN


def estimate_tokens_fast(text: str, overestimate_factor: float = 1.15) -> int:
    """
    快速 token 估算 (不依赖 tokenizer)
    
    略微高估以确保安全 (不超出上下文窗口):
    - 中文字符: ~1.5 token/字 × overestimate_factor
    - 英文单词: ~1.3 token/word × overestimate_factor
    - 其他字符 (标点/数字/空格): ~0.5 token/char
    - 特殊标记 (chat template): 按原始长度计算
    
    Args:
        text: 待估算文本
        overestimate_factor: 高估系数 (默认 1.15, 即高估 15%)
        
    Returns:
        估算的 token 数 (整数, 向上取整)
    """
    if not text:
        return 0
    
    # 统计字符类型
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    # 英文单词 (连续字母序列)
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    
    # 其他字符 (标点, 数字, 空格, 特殊符号)
    other_chars = len(text) - chinese_chars - sum(
        len(w) for w in re.findall(r'[a-zA-Z]+', text)
    )
    
    # 基础估算
    base_tokens = (
        chinese_chars * 1.5
        + english_words * 1.3
        + other_chars * 0.5
    )
    
    # 应用高估系数
    return max(1, int(base_tokens * overestimate_factor + 0.5))
