"""
DKI 文本处理工具

共享工具函数:
- strip_think_content: 移除模型输出中的 <think>...</think> 推理内容
- estimate_tokens_fast: 快速 token 估算 (不依赖 tokenizer, 略微高估)

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
