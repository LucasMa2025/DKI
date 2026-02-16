"""
DKI Recall v4 — 记忆召回与后缀组装模块

核心流程:
    用户 query
        ↓
    MultiSignalRecall (多信号召回: 关键词+指代+向量)
        ↓ RecallResult (完整关联消息列表)
    SuffixBuilder (逐消息阈值判断 → summary/原文 → 组装后缀)
        ↓ AssembledSuffix (组装好的历史 + 限定提示 + query)
    FactRetriever (trace_id → 原始消息分页检索)
        ↓ FactResponse
    PromptFormatter (模型特定格式化器)

Author: AGI Demo Project
Version: 4.0.0
"""

from dki.core.recall.recall_config import (
    RecallConfig,
    HistoryItem,
    RecallResult,
    AssembledSuffix,
    FactRequest,
    FactResponse,
)

from dki.core.recall.multi_signal_recall import MultiSignalRecall
from dki.core.recall.suffix_builder import SuffixBuilder
from dki.core.recall.fact_retriever import FactRetriever
from dki.core.recall.prompt_formatter import (
    PromptFormatter,
    GenericFormatter,
    DeepSeekFormatter,
    GLMFormatter,
    create_formatter,
)

__all__ = [
    # 数据结构
    "RecallConfig",
    "HistoryItem",
    "RecallResult",
    "AssembledSuffix",
    "FactRequest",
    "FactResponse",
    # 组件
    "MultiSignalRecall",
    "SuffixBuilder",
    "FactRetriever",
    # 格式化器
    "PromptFormatter",
    "GenericFormatter",
    "DeepSeekFormatter",
    "GLMFormatter",
    "create_formatter",
]
