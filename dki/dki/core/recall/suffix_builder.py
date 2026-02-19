"""
DKI Recall v4 — 后缀组装器

输入: 召回的消息列表 + context 预算
输出: 组装好的后缀文本 (history list + 限定提示 + query)

核心逻辑:
1. 遍历消息列表
2. 逐条判断: token > 阈值 → summary + trace_id, 否则保留原文
3. 累计 token, 超预算停止
4. 按模型适配格式化 (通过 PromptFormatter)
5. 追加可信+推理限定提示
6. 追加用户 query

Author: AGI Demo Project
Version: 4.0.0
"""

import re
from typing import Any, Callable, List, Optional

from loguru import logger

from dki.core.recall.recall_config import (
    RecallConfig,
    HistoryItem,
    AssembledSuffix,
)
from dki.core.recall.prompt_formatter import PromptFormatter

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class SuffixBuilder:
    """
    后缀组装器
    
    输入: 召回的消息列表 + context 预算
    输出: AssembledSuffix (组装好的后缀)
    """

    def __init__(
        self,
        config: RecallConfig,
        prompt_formatter: PromptFormatter,
        token_counter: Optional[Callable[[str], int]] = None,
        model_adapter: Optional[Any] = None,
    ):
        """
        Args:
            config: 召回配置
            prompt_formatter: 模型特定格式化器
            token_counter: token 计数函数 (text -> count)
                          如果 None, 使用 model_adapter.tokenizer 或粗估
            model_adapter: 模型适配器 (用于 LLM summary 或 token 计数)
        """
        self.config = config
        self.formatter = prompt_formatter
        self._model_adapter = model_adapter

        # token 计数器
        if token_counter:
            self._count_tokens = token_counter
        elif model_adapter and hasattr(model_adapter, "tokenizer"):
            self._count_tokens = lambda text: len(
                model_adapter.tokenizer.encode(text)
            )
        else:
            # 粗估: 中文约 1.5 token/字, 英文约 1.3 token/word
            self._count_tokens = self._rough_token_count

    def build(
        self,
        query: str,
        recalled_messages: List[Any],
        context_window: int = 4096,
        preference_tokens: int = 0,
    ) -> AssembledSuffix:
        """
        组装后缀
        
        Args:
            query: 用户查询
            recalled_messages: 召回的消息列表
            context_window: 上下文窗口大小
            preference_tokens: 偏好占用的 token 数
        """
        result = AssembledSuffix()

        if not recalled_messages:
            result.text = query
            return result

        # ============ 计算可用预算 ============
        budget_cfg = self.config.budget
        query_tokens = self._count_tokens(query)
        context_budget = (
            context_window
            - preference_tokens
            - query_tokens
            - budget_cfg.generation_reserve
            - budget_cfg.instruction_reserve
        )

        if context_budget <= 0:
            logger.warning(
                f"Context budget exhausted: window={context_window}, "
                f"pref={preference_tokens}, query={query_tokens}"
            )
            result.text = query
            return result

        # ============ 逐消息阈值判断 ============
        items, used_tokens = self._process_messages(
            recalled_messages, context_budget
        )

        result.items = items
        result.total_tokens = used_tokens
        result.message_count = sum(1 for i in items if i.type == "message")
        result.summary_count = sum(1 for i in items if i.type == "summary")
        result.trace_ids = [i.trace_id for i in items if i.trace_id]
        result.has_fact_call_instruction = result.summary_count > 0

        # ============ 格式化完整后缀 ============
        result.text = self.formatter.format_full_suffix(
            items=items,
            trace_ids=result.trace_ids,
            query=query,
        )

        logger.debug(
            f"Suffix built: {result.message_count} msgs + "
            f"{result.summary_count} summaries, "
            f"{result.total_tokens} tokens"
        )

        return result

    # ================================================================
    # 逐消息处理
    # ================================================================

    def _process_messages(
        self,
        messages: List[Any],
        context_budget: int,
    ) -> tuple:
        """
        逐消息阈值判断
        
        超阈值 → summary + trace_id
        阈值内 → 保留原文
        
        Returns:
            (items, used_tokens)
        """
        items: List[HistoryItem] = []
        used_tokens = 0
        threshold = self.config.summary.per_message_threshold

        for msg in messages:
            content = getattr(msg, "content", str(msg))
            role = getattr(msg, "role", "user")
            msg_id = str(
                getattr(msg, "id", None)
                or getattr(msg, "message_id", None)
                or str(id(msg))
            )

            msg_tokens = self._count_tokens(content)

            if msg_tokens > threshold:
                # 超阈值: 生成 summary
                summary_text = self._summarize(content)
                summary_tokens = self._count_tokens(summary_text)

                if used_tokens + summary_tokens > context_budget:
                    break  # 预算耗尽

                # 提取 facts_covered 和 facts_missing
                facts_covered, facts_missing = self._extract_epistemic_markers(
                    content, summary_text
                )

                items.append(HistoryItem(
                    type="summary",
                    content=summary_text,
                    trace_id=msg_id,
                    role=role,
                    token_count=summary_tokens,
                    confidence="medium",
                    facts_covered=facts_covered,
                    facts_missing=facts_missing,
                ))
                used_tokens += summary_tokens
            else:
                # 阈值内: 保留原文
                if used_tokens + msg_tokens > context_budget:
                    break  # 预算耗尽

                items.append(HistoryItem(
                    type="message",
                    content=content,
                    trace_id=msg_id,
                    role=role,
                    token_count=msg_tokens,
                    confidence="high",
                ))
                used_tokens += msg_tokens

        return items, used_tokens

    # ================================================================
    # Summary 生成
    # ================================================================

    def _summarize(self, text: str) -> str:
        """
        生成 summary
        
        策略由配置决定:
        - extractive: jieba TextRank 抽取式 (快, 可预测)
        - llm: 调用系统 LLM (慢, 高质量)
        """
        strategy = self.config.summary.strategy
        max_tokens = self.config.summary.max_tokens_per_summary

        if strategy == "llm" and self._model_adapter:
            return self._llm_summarize(text, max_tokens)
        else:
            return self._extractive_summarize(text, max_tokens)

    def _extractive_summarize(self, text: str, max_tokens: int) -> str:
        """
        抽取式摘要 (jieba TextRank)
        
        1. 按句子切分
        2. jieba TextRank 对每句打分
        3. 选取得分最高的句子
        4. 按原文顺序排列
        5. 截断到 max_tokens
        """
        if not JIEBA_AVAILABLE:
            # 无 jieba, 简单截断
            return self._truncate_to_tokens(text, max_tokens)

        # 切句
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if not sentences:
            return self._truncate_to_tokens(text, max_tokens)

        # TextRank 关键词打分
        try:
            keywords = jieba.analyse.textrank(
                text, topK=20, withWeight=True,
            )
            keyword_dict = dict(keywords)
        except Exception:
            keyword_dict = {}

        # 为每个句子计算分数
        scored = []
        for i, sent in enumerate(sentences):
            words = jieba.lcut(sent)
            score = sum(keyword_dict.get(w, 0) for w in words)
            scored.append((i, sent, score))

        # 选取 top 句子
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = sorted(scored[:5], key=lambda x: x[0])  # 恢复原文顺序

        summary = "。".join(s[1] for s in selected)

        # 截断
        if self._count_tokens(summary) > max_tokens:
            summary = self._truncate_to_tokens(summary, max_tokens)

        return summary

    def _llm_summarize(self, text: str, max_tokens: int) -> str:
        """
        LLM 生成式 summary
        
        调用系统 LLM 生成摘要, 明确约束:
        - 这是摘要，不是事实
        - 不确定处必须标注
        - 不得推理、不得补全
        """
        prompt = (
            f"请将以下对话内容压缩为不超过{max_tokens}字的摘要。\n"
            "要求:\n"
            "- 仅提取事实性陈述，不得推理或补充\n"
            "- 标注不确定或可能遗漏的信息\n"
            "- 使用\"提到了\"\"讨论了\"等不确定措辞\n"
            f"\n原文:\n{text}\n\n摘要:"
        )

        try:
            output = self._model_adapter.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.3,  # 低温度确保一致性
            )
            return output.text.strip()
        except Exception as e:
            logger.warning(f"LLM summarize failed, falling back to extractive: {e}")
            return self._extractive_summarize(text, max_tokens)

    # ================================================================
    # 认知标记提取
    # ================================================================

    def _extract_epistemic_markers(
        self,
        original: str,
        summary: str,
    ) -> tuple:
        """
        提取结构化认知标记 (补充建议: 机器可读)
        
        粗略方法: 对比原文和 summary 中的实体类型
        """
        facts_covered = []
        facts_missing = []

        # 简单启发式: 检查常见事实类别是否在 summary 中保留
        fact_patterns = {
            "日期/时间": r'\d{4}[-/年]\d{1,2}[-/月]\d{0,2}|[上下]午|早上|晚上|\d{1,2}[点时]',
            "价格/数字": r'\d+[元块万千百]|\d+\.\d+',
            "人名": None,  # 需要 NER, 暂跳过
            "地名": None,  # 需要 NER, 暂跳过
        }

        for fact_type, pattern in fact_patterns.items():
            if pattern is None:
                continue
            orig_matches = re.findall(pattern, original)
            summ_matches = re.findall(pattern, summary)

            if orig_matches:
                if summ_matches:
                    facts_covered.append(fact_type)
                else:
                    facts_missing.append(fact_type)

        # 如果原文很长但 summary 很短, 标注可能遗漏
        if len(original) > len(summary) * 3:
            if "详细内容" not in facts_missing:
                facts_missing.append("详细内容")

        return facts_covered, facts_missing

    # ================================================================
    # 工具函数
    # ================================================================

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数"""
        if self._count_tokens(text) <= max_tokens:
            return text

        # 二分截断
        lo, hi = 0, len(text)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._count_tokens(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid
        return text[:lo] + "..."

    @staticmethod
    def _rough_token_count(text: str) -> int:
        """粗估 token 数"""
        if not text:
            return 0
        # 中文字符每字约 1.5 token, 英文 word 约 1.3 token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other = len(text) - chinese_chars - english_words
        return int(chinese_chars * 1.5 + english_words * 1.3 + other * 0.5)
