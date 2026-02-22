"""
DKI Recall v4 — 模型特定提示格式化器

职责:
1. 格式化 history list 中的 summary 和 message 条目
2. 生成可信+推理限定提示 (含 function call 调用指令)
3. 格式化 function call 返回的事实段落
4. 从模型输出中检测 function call 触发

支持的格式化器:
- GenericFormatter: 通用纯文本标记 (适用于任何模型)
- DeepSeekFormatter: DeepSeek 特殊 token
- GLMFormatter: ChatGLM observation 格式

补充建议已集成:
- 结构化 epistemic marker ([SUMMARY]...[/SUMMARY])
- 强约束句式 (未调用 retrieve_fact → 回答视为无效)

Author: AGI Demo Project
Version: 4.0.0
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from dki.core.recall.recall_config import HistoryItem, FactRequest, FactResponse


# ============================================================
# 抽象基类
# ============================================================

class PromptFormatter(ABC):
    """
    模型特定提示格式化器 (抽象基类)
    """

    def __init__(self, language: str = "cn"):
        self.language = language

    @abstractmethod
    def format_summary_item(self, item: HistoryItem) -> str:
        """格式化 summary 条目 (含 trace_id + 结构化认知标记)"""

    @abstractmethod
    def format_message_item(self, item: HistoryItem) -> str:
        """格式化原文消息条目"""

    @abstractmethod
    def format_constraint_instruction(self, trace_ids: List[str]) -> str:
        """生成可信+推理限定提示"""

    @abstractmethod
    def format_fact_segment(self, response: FactResponse) -> str:
        """格式化事实段落 (function call 返回)"""

    @abstractmethod
    def detect_fact_request(self, model_output: str) -> Optional[FactRequest]:
        """从模型输出中检测 function call 请求"""

    def format_history_list(self, items: List[HistoryItem]) -> str:
        """格式化完整的 history list"""
        parts = []
        for item in items:
            if item.type == "summary":
                parts.append(self.format_summary_item(item))
            else:
                parts.append(self.format_message_item(item))
        return "\n\n".join(parts)

    def format_full_suffix(
        self,
        items: List[HistoryItem],
        trace_ids: List[str],
        query: str,
    ) -> str:
        """
        格式化完整后缀: history list + 限定提示 + query
        """
        parts = []

        # History list header
        if self.language == "cn":
            parts.append("[会话历史参考]")
        else:
            parts.append("[Session History Reference]")

        # History items
        parts.append(self.format_history_list(items))

        # 限定提示
        has_summaries = any(i.type == "summary" for i in items)
        if has_summaries:
            parts.append(self.format_constraint_instruction(trace_ids))

        # 用户 query
        if self.language == "cn":
            parts.append(f"用户当前问题: {query}")
        else:
            parts.append(f"Current question: {query}")

        return "\n\n".join(parts)


# ============================================================
# GenericFormatter — 通用格式化器
# ============================================================

class GenericFormatter(PromptFormatter):
    """
    通用格式化器 (默认, 纯文本标记)
    
    适用于任何模型, 不依赖特殊 token。
    Function Call 触发格式:
      retrieve_fact(trace_id="msg-xxx", offset=0, limit=5)
    检测: 正则匹配 retrieve_fact\\(.*?\\)
    """

    # 检测 function call 的正则
    _FACT_CALL_PATTERN = re.compile(
        r'retrieve_fact\s*\(\s*'
        r'trace_id\s*=\s*["\']([^"\']+)["\']\s*'
        r'(?:,\s*offset\s*=\s*(\d+))?\s*'
        r'(?:,\s*limit\s*=\s*(\d+))?\s*'
        r'\)',
        re.IGNORECASE,
    )

    def format_summary_item(self, item: HistoryItem) -> str:
        """
        结构化 epistemic marker (补充建议: 机器可读)
        
        [SUMMARY]
        facts_covered: [...]
        facts_missing: [...]
        confidence: medium
        trace_id: msg-005
        [/SUMMARY]
        """
        role_label = self._role_label(item.role)

        # 构造结构化标记
        lines = [f"[SUMMARY trace_id=\"{item.trace_id}\" confidence=\"{item.confidence}\"]"]
        lines.append(f"{role_label}: {item.content}")

        if item.facts_covered:
            covered = ", ".join(f'"{f}"' for f in item.facts_covered)
            lines.append(f"facts_covered: [{covered}]")
        if item.facts_missing:
            missing = ", ".join(f'"{f}"' for f in item.facts_missing)
            lines.append(f"facts_missing: [{missing}]")

        lines.append("[/SUMMARY]")
        return "\n".join(lines)

    def format_message_item(self, item: HistoryItem) -> str:
        """格式化原文消息"""
        role_label = self._role_label(item.role)
        return f"{role_label}: {item.content}"

    def format_constraint_instruction(self, trace_ids: List[str]) -> str:
        """
        可信+推理限定提示
        
        补充建议集成:
        - 强约束: 未调用 retrieve_fact 即基于 summary 给出数值/时间/原话 → 回答无效
        """
        trace_list = ", ".join(f'"{tid}"' for tid in trace_ids)

        if self.language == "cn":
            return (
                "[可信+推理限定]\n"
                "以上标记为 [SUMMARY] 的内容为摘要，非完整事实记录。\n"
                "若回答需要精确原话、具体数值、时间、因果关系，\n"
                "且摘要中未明确包含，\n"
                f"请调用 retrieve_fact(trace_id=\"...\", offset=0, limit=5) 获取原始记录。\n"
                f"可用 trace_id: {trace_list}\n"
                "\n"
                "【强制约束】若在未调用 retrieve_fact 的情况下，\n"
                "直接基于 summary 给出具体数值/时间/引用原话，\n"
                "该回答被视为无效。\n"
                "\n"
                "不得基于摘要进行推理或补全。\n"
                "[/可信+推理限定]"
            )
        else:
            return (
                "[Trustworthy + Reasoning Constraint]\n"
                "Sections marked [SUMMARY] above are summaries, not complete factual records.\n"
                "If your answer requires exact quotes, specific numbers, dates, or causal relationships,\n"
                "and the summary does not explicitly contain them,\n"
                f"call retrieve_fact(trace_id=\"...\", offset=0, limit=5) to get original records.\n"
                f"Available trace_ids: {trace_list}\n"
                "\n"
                "[MANDATORY] If you provide specific numbers/dates/quotes based on a summary\n"
                "WITHOUT calling retrieve_fact first, the answer is considered INVALID.\n"
                "\n"
                "Do not reason or extrapolate from summaries.\n"
                "[/Trustworthy + Reasoning Constraint]"
            )

    def format_fact_segment(self, response: FactResponse) -> str:
        """格式化事实段落"""
        lines = [f"[FACT_SEGMENT trace_id=\"{response.trace_id}\" "
                 f"offset={response.offset} total={response.total_count}]"]

        for msg in response.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_label = self._role_label(role)
            lines.append(f"{role_label}: {content}")

        if response.has_more:
            if self.language == "cn":
                lines.append(f"(还有更多内容, 可调用 retrieve_fact(trace_id=\"{response.trace_id}\", "
                           f"offset={response.offset + len(response.messages)}, limit=5))")
            else:
                lines.append(f"(More available, call retrieve_fact(trace_id=\"{response.trace_id}\", "
                           f"offset={response.offset + len(response.messages)}, limit=5))")

        lines.append("[/FACT_SEGMENT]")
        return "\n".join(lines)

    def detect_fact_request(self, model_output: str) -> Optional[FactRequest]:
        """从模型输出中检测 function call"""
        match = self._FACT_CALL_PATTERN.search(model_output)
        if not match:
            return None

        trace_id = match.group(1)
        offset = int(match.group(2)) if match.group(2) else 0
        limit = int(match.group(3)) if match.group(3) else 5

        return FactRequest(
            trace_id=trace_id,
            offset=offset,
            limit=limit,
        )

    def _role_label(self, role: Optional[str]) -> str:
        if self.language == "cn":
            return "用户" if role == "user" else "助手"
        return "User" if role == "user" else "Assistant"


# ============================================================
# DeepSeekFormatter — DeepSeek 格式化器
# ============================================================

class DeepSeekFormatter(GenericFormatter):
    """
    DeepSeek 格式化器
    
    利用 DeepSeek 的 function call 特殊 token:
      <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>retrieve_fact
      {"trace_id": "...", "offset": 0, "limit": 5}
      <｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """

    _DS_TOOL_CALL_PATTERN = re.compile(
        r'<｜tool▁call▁begin｜>\s*retrieve_fact\s*\n'
        r'\s*\{[^}]*"trace_id"\s*:\s*"([^"]+)"'
        r'(?:[^}]*"offset"\s*:\s*(\d+))?'
        r'(?:[^}]*"limit"\s*:\s*(\d+))?'
        r'[^}]*\}\s*\n\s*<｜tool▁call▁end｜>',
        re.DOTALL,
    )

    def format_constraint_instruction(self, trace_ids: List[str]) -> str:
        """DeepSeek 版本: 在 system prompt 中注册工具定义"""
        base = super().format_constraint_instruction(trace_ids)

        tool_def = (
            "\n\n[TOOL_DEFINITION]\n"
            "name: retrieve_fact\n"
            "description: 检索指定 trace_id 的原始消息内容\n"
            "parameters:\n"
            "  trace_id: string (必填) - 消息溯源 ID\n"
            "  offset: int (默认 0) - 分页偏移\n"
            "  limit: int (默认 5) - 返回条数\n"
            "[/TOOL_DEFINITION]"
        )
        return base + tool_def

    def detect_fact_request(self, model_output: str) -> Optional[FactRequest]:
        """优先检测 DeepSeek 特殊 token, 回退到通用正则"""
        match = self._DS_TOOL_CALL_PATTERN.search(model_output)
        if match:
            return FactRequest(
                trace_id=match.group(1),
                offset=int(match.group(2)) if match.group(2) else 0,
                limit=int(match.group(3)) if match.group(3) else 5,
            )
        return super().detect_fact_request(model_output)


# ============================================================
# GLMFormatter — ChatGLM 格式化器
# ============================================================

class GLMFormatter(GenericFormatter):
    """
    ChatGLM 格式化器
    使用 <|tool_call|> / <|observation|> 格式
    """

    _GLM_TOOL_PATTERN = re.compile(
        r'<\|tool_call\|>\s*retrieve_fact\s*\n'
        r'\s*\{[^}]*"trace_id"\s*:\s*"([^"]+)"'
        r'(?:[^}]*"offset"\s*:\s*(\d+))?'
        r'(?:[^}]*"limit"\s*:\s*(\d+))?'
        r'[^}]*\}',
        re.DOTALL,
    )

    def format_fact_segment(self, response: FactResponse) -> str:
        """GLM 格式: 使用 <|observation|> 标记"""
        base = super().format_fact_segment(response)
        return f"<|observation|>\n{base}\n<|/observation|>"

    def detect_fact_request(self, model_output: str) -> Optional[FactRequest]:
        """优先检测 GLM 格式, 回退到通用正则"""
        match = self._GLM_TOOL_PATTERN.search(model_output)
        if match:
            return FactRequest(
                trace_id=match.group(1),
                offset=int(match.group(2)) if match.group(2) else 0,
                limit=int(match.group(3)) if match.group(3) else 5,
            )
        return super().detect_fact_request(model_output)


# ============================================================
# 工厂函数
# ============================================================

def create_formatter(
    model_name: str = "",
    formatter_type: str = "auto",
    language: str = "cn",
) -> PromptFormatter:
    """
    通过配置或模型名自动选择格式化器
    
    Args:
        model_name: 模型名称 (用于 auto 检测)
        formatter_type: 格式化器类型 (auto | generic | deepseek | glm)
        language: 语言 (cn | en)
    """
    if formatter_type == "auto":
        name = model_name.lower()
        if "deepseek" in name:
            return DeepSeekFormatter(language=language)
        elif "glm" in name or "chatglm" in name:
            return GLMFormatter(language=language)
        else:
            return GenericFormatter(language=language)
    elif formatter_type == "deepseek":
        return DeepSeekFormatter(language=language)
    elif formatter_type == "glm":
        return GLMFormatter(language=language)
    else:
        return GenericFormatter(language=language)
