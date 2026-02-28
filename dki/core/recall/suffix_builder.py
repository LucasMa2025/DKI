"""
DKI Recall v4 — 后缀组装器

输入: 召回的消息列表 + context 预算
输出: 组装好的后缀文本 (history list + 限定提示 + query)

核心逻辑 (v6.0 — 两阶段全局预算分配):
Phase 1: 完整收集 — 遍历所有消息, 保留原文 + 标记 msg_id + 计算 token
Phase 2: 全局分配 — Planner 根据总预算决定每条消息保留/压缩
    - 低于阈值: 优先保留全文
    - 高于阈值: 如果预算够 → 保留全文; 如果不够 → 压缩
    - 只有真正放不下的才做 summary + trace_id
Phase 3: 格式化 — 按模型适配格式化 + 追加限定提示 + query

Author: AGI Demo Project
Version: 6.0.0
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from dki.core.text_utils import strip_think_content, estimate_tokens_fast
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
        组装后缀 (v6.0: 两阶段全局预算分配)
        
        Phase 1: 完整收集所有消息 (不压缩)
        Phase 2: 全局预算分配 — 智能决定保留/压缩
        Phase 3: 格式化输出
        
        预算分配策略:
        - 生成预留 = 30% 上下文窗口
        - 标记开销 = instruction_reserve (默认 120)
        - 偏好 = 直接估算 (100-200 tokens)
        - 当前输入 = 直接估算 (不预留, 高估 15%)
        - 剩余 → 历史消息 (全局分配)
        
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
        
        generation_reserve = int(context_window * 0.30)
        tag_overhead = budget_cfg.instruction_reserve  # 默认 120-150
        
        context_budget = (
            context_window
            - generation_reserve
            - tag_overhead
            - preference_tokens     # 偏好 (100-200 tokens)
            - query_tokens           # 当前输入 (直接估算)
        )

        if context_budget <= 0:
            logger.warning(
                f"Context budget exhausted: window={context_window}, "
                f"pref={preference_tokens}, query={query_tokens}"
            )
            result.text = query
            return result

        # ============ Phase 1: 完整收集 (不压缩) ============
        collected = self._collect_messages(recalled_messages)
        
        # ============ Phase 2: 全局预算分配 ============
        items, used_tokens = self._global_budget_allocate(
            collected, context_budget
        )

        result.items = items
        result.total_tokens = used_tokens
        result.message_count = sum(1 for i in items if i.type == "message")
        result.summary_count = sum(1 for i in items if i.type == "summary")
        result.trace_ids = [i.trace_id for i in items if i.trace_id]
        result.has_fact_call_instruction = result.summary_count > 0

        # ============ Phase 3: 格式化完整后缀 ============
        result.text = self.formatter.format_full_suffix(
            items=items,
            trace_ids=result.trace_ids,
            query=query,
        )

        logger.debug(
            f"Suffix built (v6.0): {result.message_count} msgs + "
            f"{result.summary_count} summaries, "
            f"{result.total_tokens} tokens, "
            f"budget={context_budget}"
        )

        return result

    # ================================================================
    # Phase 1: 完整收集 (不压缩)
    # ================================================================

    def _collect_messages(
        self,
        messages: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: 完整收集所有消息, 不做任何压缩决策
        
        每条消息记录:
        - msg_id: 消息 ID (用于溯源)
        - content: 清理后的原文 (已移除 <think>)
        - role: 角色
        - tokens: 原文 token 数
        
        Returns:
            收集到的消息列表 (含 token 统计)
        """
        collected = []

        for msg in messages:
            content = getattr(msg, "content", str(msg))
            role = getattr(msg, "role", "user")
            msg_id = str(
                getattr(msg, "id", None)
                or getattr(msg, "message_id", None)
                or str(id(msg))
            )

            # 移除 assistant 消息中的 <think> 推理内容
            if role == 'assistant' and content:
                content, think_stripped = strip_think_content(content)
                if think_stripped:
                    logger.debug(f"Think content stripped from history msg {msg_id}")
                if not content or not content.strip():
                    continue  # 清理后为空, 跳过

            msg_tokens = self._count_tokens(content)
            
            collected.append({
                'msg_id': msg_id,
                'content': content,
                'role': role,
                'tokens': msg_tokens,
            })

        return collected

    # ================================================================
    # Phase 2: 全局预算分配
    # ================================================================

    def _global_budget_allocate(
        self,
        collected: List[Dict[str, Any]],
        context_budget: int,
    ) -> Tuple[List[HistoryItem], int]:
        """
        Phase 2: 全局预算分配 (v6.0 核心改动)
        
        策略:
        1. 先计算如果所有消息都保留原文, 总共需要多少 token
        2. Case A: 总 token <= 预算 → 全部保留原文 (最优)
        3. Case B: 总 token > 预算 → 智能分配:
           a. 按 token 数将消息分为"短消息"(≤阈值) 和"长消息"(>阈值)
           b. 短消息全部保留原文 (优先级最高)
           c. 剩余预算分配给长消息:
              - 从前到后遍历, 如果预算够 → 保留原文
              - 如果预算不够保留原文但够放 summary → 压缩
              - 如果连 summary 都放不下 → 跳过
        
        这样的优势:
        - 不是简单的逐条阈值判断, 而是全局最优
        - 短消息永远保留原文 (它们本来就不占多少空间)
        - 长消息只有在预算真正不够时才压缩
        - 最大化保留事实信息
        
        Returns:
            (items, used_tokens)
        """
        if not collected:
            return [], 0
        
        threshold = self.config.summary.per_message_threshold
        total_tokens = sum(m['tokens'] for m in collected)
        
        # ============ Case A: 全部放得下 → 全部保留原文 ============
        if total_tokens <= context_budget:
            items = []
            for m in collected:
                items.append(HistoryItem(
                    type="message",
                    content=m['content'],
                    trace_id=m['msg_id'],
                    role=m['role'],
                    token_count=m['tokens'],
                    confidence="high",
                ))
            logger.debug(
                f"Global budget: all {len(items)} msgs fit "
                f"({total_tokens}/{context_budget} tokens)"
            )
            return items, total_tokens
        
        # ============ Case B: 需要智能分配 ============
        # 分类: 短消息 vs 长消息
        short_msgs = []  # (idx, msg_dict)
        long_msgs = []   # (idx, msg_dict)
        for idx, m in enumerate(collected):
            if m['tokens'] <= threshold:
                short_msgs.append((idx, m))
            else:
                long_msgs.append((idx, m))
        
        # 短消息优先: 计算短消息总 token
        short_total = sum(m['tokens'] for _, m in short_msgs)
        
        # 分配给长消息的预算
        long_budget = context_budget - short_total
        
        if long_budget <= 0:
            # 极端情况: 短消息就已经超预算了, 按顺序截断短消息
            items = []
            used = 0
            for _, m in short_msgs:
                if used + m['tokens'] > context_budget:
                    break
                items.append(HistoryItem(
                    type="message",
                    content=m['content'],
                    trace_id=m['msg_id'],
                    role=m['role'],
                    token_count=m['tokens'],
                    confidence="high",
                ))
                used += m['tokens']
            logger.debug(
                f"Global budget: short msgs exceeded budget, "
                f"kept {len(items)}/{len(short_msgs)} short msgs"
            )
            return items, used
        
        # 长消息分配: 尽量保留原文, 不够则压缩
        long_items = []  # (original_idx, HistoryItem)
        long_used = 0
        
        for idx, m in long_msgs:
            remaining = long_budget - long_used
            
            if remaining <= 0:
                break  # 预算耗尽
            
            if m['tokens'] <= remaining:
                # 预算够: 保留原文
                long_items.append((idx, HistoryItem(
                    type="message",
                    content=m['content'],
                    trace_id=m['msg_id'],
                    role=m['role'],
                    token_count=m['tokens'],
                    confidence="high",
                )))
                long_used += m['tokens']
            else:
                # 预算不够原文: 尝试压缩
                summary_text = self._summarize(m['content'])
                summary_tokens = self._count_tokens(summary_text)
                
                if summary_tokens <= remaining:
                    # summary 放得下
                    facts_covered, facts_missing = self._extract_epistemic_markers(
                        m['content'], summary_text
                    )
                    long_items.append((idx, HistoryItem(
                        type="summary",
                        content=summary_text,
                        trace_id=m['msg_id'],
                        role=m['role'],
                        token_count=summary_tokens,
                        confidence="medium",
                        facts_covered=facts_covered,
                        facts_missing=facts_missing,
                    )))
                    long_used += summary_tokens
                else:
                    # 连 summary 都放不下, 跳过
                    logger.debug(
                        f"Global budget: skipping msg {m['msg_id']} "
                        f"(summary {summary_tokens} > remaining {remaining})"
                    )
                    continue
        
        # 合并: 按原始顺序排列 (短消息 + 长消息)
        all_indexed = [(idx, HistoryItem(
            type="message",
            content=m['content'],
            trace_id=m['msg_id'],
            role=m['role'],
            token_count=m['tokens'],
            confidence="high",
        )) for idx, m in short_msgs]
        all_indexed.extend(long_items)
        
        # 按原始顺序排序
        all_indexed.sort(key=lambda x: x[0])
        items = [item for _, item in all_indexed]
        used_tokens = short_total + long_used
        
        n_full = sum(1 for i in items if i.type == "message")
        n_summary = sum(1 for i in items if i.type == "summary")
        logger.debug(
            f"Global budget: {n_full} full + {n_summary} summarized, "
            f"{used_tokens}/{context_budget} tokens, "
            f"({len(collected) - len(items)} msgs dropped)"
        )
        
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
