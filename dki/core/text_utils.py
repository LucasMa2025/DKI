"""
DKI 文本处理工具

共享工具函数:
- strip_think_content: 移除模型输出中的 <think>...</think> 推理内容
- estimate_tokens_fast: 快速 token 估算 (不依赖 tokenizer, 略微高估)
- detect_vague_reference: 检测用户输入中的模糊指代 (v6.5)
- build_clarification_prompt: 构建澄清提示词 (v6.5)

关于 <think> 标记:
===================
DeepSeek-R1 系列模型 (包括 distill 版本) 在生成回复时,
会先输出 <think>...推理过程...</think>, 然后才输出最终回复。

特殊情况:
1. 完整格式: <think>推理内容</think>最终回复
2. 仅有结束标记: 推理内容</think>最终回复
   (当 <think> 标记被截断或 vLLM tokenizer 不输出起始标记时)
3. 标记变体: 可能出现空格, 如 < think > 或 <think >
4. 嵌套情况: 不存在嵌套 <think>, 但可能有多段

为什么 UI 只看到 </think> 没有 <think>:
- DeepSeek-R1 distill 模型的 chat template 会将 <think> 作为
  assistant 回复的起始部分自动添加, 但 vLLM 输出时可能只返回
  生成的 token (不含 prompt 中的 assistant 起始标记)
- 因此存储到数据库的内容可能以推理过程开头, 以 </think> 结尾

Author: AGI Demo Project
Version: 5.7.0
"""

import re
from dataclasses import dataclass
from typing import Tuple

from loguru import logger


# ============================================================
# Think 内容过滤正则
# ============================================================

# 完整格式: <think>...</think>
# 支持标记内的空格变体
_THINK_FULL_RE = re.compile(
    r'<\s*think\s*>.*?<\s*/\s*think\s*>',
    re.DOTALL | re.IGNORECASE,
)

# 仅有结束标记: 从文本开头到 </think> (当 <think> 被截断或不存在时)
# 仅匹配文本开头的情况 (避免误删中间内容)
_THINK_TAIL_ONLY_RE = re.compile(
    r'^.*?<\s*/\s*think\s*>',
    re.DOTALL | re.IGNORECASE,
)


def strip_think_content(text: str) -> Tuple[str, bool]:
    """
    移除模型输出中的 <think>...</think> 推理内容
    
    过滤策略:
    1. 先尝试移除完整的 <think>...</think> 块 (可能有多个)
    2. 如果没找到完整块, 检查是否以推理内容开头 + </think> 结尾
       (DeepSeek-R1 模型有时只输出 </think> 不输出 <think>)
    3. 清理结果: 去除多余空行, 保持文本整洁
    
    Args:
        text: 原始文本 (可能包含 think 内容)
        
    Returns:
        (cleaned_text, was_stripped) — 清理后的文本 和 是否执行了移除
    """
    if not text:
        return text, False
    
    original_len = len(text)
    result = text
    stripped = False
    
    # Step 1: 移除完整的 <think>...</think> 块
    if _THINK_FULL_RE.search(result):
        result = _THINK_FULL_RE.sub('', result)
        stripped = True
    
    # Step 2: 如果没有完整块, 检查仅有 </think> 的情况
    # 仅当文本中存在 </think> 但没有对应的 <think> 时
    if not stripped and re.search(r'<\s*/\s*think\s*>', result, re.IGNORECASE):
        # 检查是否有未匹配的 <think>
        has_open = bool(re.search(r'<\s*think\s*>', result, re.IGNORECASE))
        if not has_open:
            # 仅有 </think>, 删除从开头到 </think> 的内容
            result = _THINK_TAIL_ONLY_RE.sub('', result)
            stripped = True
    
    # Step 3: 清理
    if stripped:
        # 去除开头的空白行
        result = result.lstrip('\n\r\t ')
        # 合并连续空行为单个空行
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        removed_len = original_len - len(result)
        logger.debug(
            f"Think content stripped: removed {removed_len} chars "
            f"({original_len} -> {len(result)})"
        )
    
    return result, stripped


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
# 这些词表明用户已经给出了足够的上下文
# 注意: 排除代词性质的词 (那个/那件/这个/什么) 以避免误判
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
