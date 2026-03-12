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

v7.0 优化 (不依赖 LLM):
- Query-Aware 句子评分: 压缩时优先保留与当前 query 相关的句子
- 增强认知标记检测: 扩展事实类别 (地址/邮箱/电话/URL/专有名词/量词等)
- TF-IDF 事实密度评分: 事实密集句优先保留
- Summary 完整性校验: 遗漏的事实自动加入 facts_missing

Author: AGI Demo Project
Version: 7.0.0
"""

import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
        
        # ============ Phase 2: 全局预算分配 (v7.0: 传入 query 用于 Query-Aware 评分) ============
        items, used_tokens = self._global_budget_allocate(
            collected, context_budget, query=query
        )

        result.items = items
        result.total_tokens = used_tokens
        result.message_count = sum(1 for i in items if i.type == "message")
        result.summary_count = sum(1 for i in items if i.type == "summary")
        # 只收集 summary 类型条目的 trace_id (原文消息不需要 retrieve_fact 回溯)
        # 每条 summary 对应一个唯一 trace_id, 用于 [可信+推理限定] 块
        result.trace_ids = [
            i.trace_id for i in items
            if i.type == "summary" and i.trace_id
        ]
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
        
        v7.3 改进:
        - 收集 timestamp 字段 (来自 ChatMessage.timestamp)
        - 按时间戳排序 (确保时间正序)
        - 成对处理: 确保 user 问题和 assistant 回复成对出现
        
        每条消息记录:
        - msg_id: 消息 ID (用于溯源)
        - content: 清理后的原文 (已移除 <think>)
        - role: 角色
        - tokens: 原文 token 数
        - timestamp: 消息时间戳 (ISO 格式字符串)
        - pair_id: 配对 ID (同一轮 user-assistant 共享)
        
        Returns:
            收集到的消息列表 (含 token 统计, 按时间排序, 成对组织)
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
            
            # v7.3: 提取时间戳
            timestamp = getattr(msg, "timestamp", None)
            if timestamp is None:
                timestamp = getattr(msg, "created_at", None)
            # 统一转为 ISO 字符串
            ts_str = ""
            ts_sort_key = ""
            if timestamp is not None:
                try:
                    if hasattr(timestamp, 'isoformat'):
                        ts_str = timestamp.strftime("%Y-%m-%d %H:%M")
                        ts_sort_key = timestamp.isoformat()
                    else:
                        ts_str = str(timestamp)
                        ts_sort_key = str(timestamp)
                except Exception:
                    ts_str = str(timestamp)
                    ts_sort_key = str(timestamp)
            
            # v7.3: 提取 parent_id (用于配对)
            parent_id = getattr(msg, "parent_id", None)

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
                'timestamp': ts_str,
                'ts_sort_key': ts_sort_key,
                'parent_id': parent_id,
            })

        # v7.3: 按时间戳排序 (确保时间正序)
        collected.sort(key=lambda m: m.get('ts_sort_key', ''))
        
        # v7.3: 成对处理 — 确保 user 问题和 assistant 回复成对出现
        collected = self._ensure_paired_messages(collected)
        
        # v7.4: 移除末尾无配对的 user 消息
        # 用户当前查询已在 prompt 最后, 历史中不应重复
        collected = self._remove_trailing_unpaired_user(collected)

        return collected
    
    @staticmethod
    def _remove_trailing_unpaired_user(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        v7.4: 移除末尾没有 assistant 回复的 user 消息
        
        用户当前查询已在 prompt 最后作为独立输入, 不需要在历史中重复。
        从末尾向前扫描, 移除连续的 role="user" 消息, 遇到 assistant 停止。
        """
        if not messages:
            return messages
        
        cut_index = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get('role') == 'user':
                cut_index = i
            else:
                break
        
        if cut_index < len(messages):
            removed = len(messages) - cut_index
            logger.debug(
                f"SuffixBuilder: removed {removed} trailing unpaired "
                f"user message(s) from collected history"
            )
            return messages[:cut_index]
        
        return messages

    def _ensure_paired_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        v7.3: 确保消息成对出现 (user 问题 + assistant 回复)
        
        策略:
        1. 如果 assistant 消息有 parent_id, 通过 parent_id 找到对应的 user 消息
        2. 否则按时间序列, 相邻的 user→assistant 自然配对
        3. 为每对消息分配相同的 pair_id
        4. 孤立的 assistant 消息 (无对应 user) 仍然保留, 但标记无配对
        
        Returns:
            成对组织的消息列表 (保持时间序)
        """
        if not messages:
            return messages
        
        # 构建 msg_id → message 索引
        id_to_msg = {m['msg_id']: m for m in messages}
        paired_ids = set()
        pair_counter = 0
        
        # Pass 1: 通过 parent_id 配对
        for m in messages:
            if m['role'] == 'assistant' and m.get('parent_id'):
                parent_id = str(m['parent_id'])
                if parent_id in id_to_msg:
                    pair_counter += 1
                    pair_id = f"pair_{pair_counter}"
                    id_to_msg[parent_id]['pair_id'] = pair_id
                    m['pair_id'] = pair_id
                    paired_ids.add(parent_id)
                    paired_ids.add(m['msg_id'])
        
        # Pass 2: 按时间序列配对未配对的消息 (相邻 user→assistant)
        for i in range(len(messages) - 1):
            curr = messages[i]
            next_msg = messages[i + 1]
            if (curr['msg_id'] not in paired_ids
                    and next_msg['msg_id'] not in paired_ids
                    and curr['role'] == 'user'
                    and next_msg['role'] == 'assistant'):
                pair_counter += 1
                pair_id = f"pair_{pair_counter}"
                curr['pair_id'] = pair_id
                next_msg['pair_id'] = pair_id
                paired_ids.add(curr['msg_id'])
                paired_ids.add(next_msg['msg_id'])
        
        # Pass 3: 标记未配对的消息
        for m in messages:
            if m['msg_id'] not in paired_ids:
                m.setdefault('pair_id', None)
        
        return messages

    # ================================================================
    # Phase 2: 全局预算分配
    # ================================================================

    def _global_budget_allocate(
        self,
        collected: List[Dict[str, Any]],
        context_budget: int,
        query: str = "",
    ) -> Tuple[List[HistoryItem], int]:
        """
        Phase 2: 全局预算分配 (v7.0 核心改动)
        
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
        
        v7.0 优化:
        - 压缩时使用 Query-Aware 句子评分 (与 query 相关的句子优先保留)
        - 增强的认知标记检测 + Summary 完整性校验
        
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
                    timestamp=m.get('timestamp', ''),
                    pair_id=m.get('pair_id'),
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
                    timestamp=m.get('timestamp', ''),
                    pair_id=m.get('pair_id'),
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
                    timestamp=m.get('timestamp', ''),
                    pair_id=m.get('pair_id'),
                )))
                long_used += m['tokens']
            else:
                # 预算不够原文: 尝试压缩 (v7.0: Query-Aware)
                summary_text = self._summarize(m['content'], query=query)
                summary_tokens = self._count_tokens(summary_text)
                
                if summary_tokens <= remaining:
                    # summary 放得下
                    # v7.0: 增强认知标记 + 完整性校验
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
                        timestamp=m.get('timestamp', ''),
                        pair_id=m.get('pair_id'),
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
            timestamp=m.get('timestamp', ''),
            pair_id=m.get('pair_id'),
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
    # Summary 生成 (v7.0: Query-Aware + TF-IDF 事实密度)
    # ================================================================

    def _summarize(self, text: str, query: str = "") -> str:
        """
        生成 summary (v7.0: 支持 Query-Aware 评分)
        
        策略由配置决定:
        - extractive: jieba TextRank + Query-Aware + TF-IDF 事实密度 (快, 可预测)
        - llm: 调用系统 LLM (慢, 高质量)
        
        Args:
            text: 原文
            query: 当前用户查询 (v7.0: 用于 Query-Aware 评分)
        """
        strategy = self.config.summary.strategy
        max_tokens = self.config.summary.max_tokens_per_summary

        if strategy == "llm" and self._model_adapter:
            return self._llm_summarize(text, max_tokens)
        else:
            return self._extractive_summarize(text, max_tokens, query=query)

    # ================================================================
    # v7.0: 增强切句策略
    # ================================================================

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        v7.0: 增强切句策略
        
        相比原版 re.split(r'[。！？\\n]+', text) 的改进:
        1. 支持中英文混合标点: 。！？!?
        2. 支持分号/冒号后的语义分割: ；;：:
        3. 支持省略号: ……、...
        4. 保留短句 (≥3 字符, 原版 ≥5)
        5. 多行文本按换行分割
        6. 避免在数字小数点处误切 (如 3.14)
        
        Returns:
            切分后的句子列表 (已去除空白)
        """
        # 先按换行切分 (保留段落结构)
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 增强的句子切分正则:
            # - 中文句末: 。！？
            # - 英文句末: !? (句号需排除小数点)
            # - 分号: ；;
            # - 省略号: ……、...
            # - 冒号后如果跟换行也算 (已在外层处理)
            parts = re.split(
                r'(?<=[。！？!?])'           # 中英文句末标点后切分
                r'|(?<=[；;])'               # 分号后切分
                r'|(?<=……)'                  # 中文省略号后切分
                r'|(?<=\.\.\.)'              # 英文省略号后切分
                r'|(?<=[.])(?=\s+[A-Z\u4e00-\u9fff])',  # 英文句号后跟空格+大写/中文
                line,
            )
            
            for part in parts:
                part = part.strip()
                if len(part) >= 3:  # 降低阈值: 3 字符 (原版 5)
                    sentences.append(part)
        
        return sentences

    # ================================================================
    # v7.0: Query-Aware 句子评分
    # ================================================================

    def _compute_query_relevance(
        self,
        sentence: str,
        query_words: Set[str],
        query_bigrams: Set[str],
    ) -> float:
        """
        v7.0: 计算句子与 query 的相关性分数
        
        评分维度:
        1. 词汇重叠: 句子中包含 query 词的比例 (unigram)
        2. 二元组重叠: 句子中包含 query 二元组的比例 (bigram, 捕获短语)
        3. 长度归一化: 避免长句因词多而天然高分
        
        Args:
            sentence: 待评分句子
            query_words: query 的分词集合
            query_bigrams: query 的二元组集合
            
        Returns:
            相关性分数 (0.0 ~ 1.0)
        """
        if not query_words:
            return 0.0
        
        if JIEBA_AVAILABLE:
            sent_words = set(jieba.lcut(sentence))
        else:
            # 无 jieba: 按字符切分 (中文) + 按空格切分 (英文)
            sent_words = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', sentence))
        
        if not sent_words:
            return 0.0
        
        # Unigram 重叠
        overlap = len(query_words & sent_words)
        unigram_score = overlap / len(query_words) if query_words else 0.0
        
        # Bigram 重叠 (捕获短语级相关性)
        bigram_score = 0.0
        if query_bigrams and JIEBA_AVAILABLE:
            sent_word_list = list(sent_words)
            sent_bigrams = set()
            for i in range(len(sent_word_list) - 1):
                sent_bigrams.add(sent_word_list[i] + sent_word_list[i + 1])
            bigram_overlap = len(query_bigrams & sent_bigrams)
            bigram_score = bigram_overlap / len(query_bigrams) if query_bigrams else 0.0
        
        # 加权融合: unigram 60% + bigram 40%
        return unigram_score * 0.6 + bigram_score * 0.4

    # ================================================================
    # v7.0: TF-IDF 事实密度评分
    # ================================================================

    @staticmethod
    def _compute_fact_density(sentence: str) -> float:
        """
        v7.0: 计算句子的事实密度分数
        
        事实密度 = 句子中包含的事实性信息量 / 句子长度
        
        事实信号:
        1. 数字 (日期、价格、数量、编号等)
        2. 专有名词标记 (引号内的名称、品牌等)
        3. URL / 邮箱 / 电话
        4. 精确量词 (3个、5次、100元等)
        5. 命名实体启发式 (大写英文词、中文地名后缀等)
        
        Returns:
            事实密度分数 (0.0 ~ 1.0, 归一化)
        """
        if not sentence or len(sentence) < 3:
            return 0.0
        
        fact_signals = 0
        sent_len = max(len(sentence), 1)
        
        # 1. 数字 (日期、价格、数量、编号)
        numbers = re.findall(
            r'\d{4}[-/年]\d{1,2}[-/月]?\d{0,2}日?'  # 日期
            r'|\d{1,2}[点时:]\d{0,2}[分]?'           # 时间
            r'|\d+\.?\d*[元块万千百亿%％度℃]'          # 带单位的数字
            r'|\d{3,}',                                # 3位以上纯数字 (编号等)
            sentence,
        )
        fact_signals += len(numbers) * 2  # 数字权重 x2
        
        # 2. 引号内的专有名词 (《》「」""'' 等)
        quoted = re.findall(
            r'[《「"\'](.*?)[》」"\']',
            sentence,
        )
        fact_signals += len(quoted) * 1.5
        
        # 3. URL
        urls = re.findall(r'https?://\S+|www\.\S+', sentence)
        fact_signals += len(urls) * 2
        
        # 4. 邮箱
        emails = re.findall(r'\S+@\S+\.\S+', sentence)
        fact_signals += len(emails) * 2
        
        # 5. 电话号码
        phones = re.findall(r'1[3-9]\d{9}|\d{3,4}-\d{7,8}', sentence)
        fact_signals += len(phones) * 2
        
        # 6. 精确量词 (中文)
        measures = re.findall(
            r'\d+[个只条件套份层楼间位天月年周岁公里米厘米毫米吨斤克升毫升]',
            sentence,
        )
        fact_signals += len(measures) * 1.5
        
        # 7. 英文大写词 (可能是专有名词, 排除句首)
        upper_words = re.findall(r'(?<!\A)(?<![.!?]\s)[A-Z][a-zA-Z]{2,}', sentence)
        fact_signals += len(upper_words) * 0.5
        
        # 8. 中文地名后缀
        place_suffixes = re.findall(
            r'[\u4e00-\u9fff]{1,4}(?:省|市|区|县|镇|村|路|街|号|大厦|广场|公园|医院|学校|大学)',
            sentence,
        )
        fact_signals += len(place_suffixes) * 1.5
        
        # 归一化: 使用 sigmoid 映射到 [0, 1]
        # 经验值: 5 个事实信号 → ~0.73, 10 个 → ~0.95
        raw_density = fact_signals / max(sent_len / 10, 1)
        normalized = 1.0 / (1.0 + math.exp(-2.0 * (raw_density - 0.5)))
        
        return min(normalized, 1.0)

    def _extractive_summarize(
        self, text: str, max_tokens: int, query: str = ""
    ) -> str:
        """
        抽取式摘要 (v7.0: Query-Aware + TF-IDF 事实密度)
        
        v7.0 改进:
        1. 增强切句: 支持更多标点, 降低最小长度阈值
        2. Query-Aware 评分: 与当前 query 相关的句子获得加分
        3. TF-IDF 事实密度: 包含数字/专有名词/地名等事实的句子获得加分
        4. 多维融合: TextRank + Query 相关性 + 事实密度
        5. 动态 top-k: 根据预算和句子数量动态选取
        
        Args:
            text: 原文
            max_tokens: 摘要最大 token 数
            query: 当前用户查询 (v7.0)
        """
        if not JIEBA_AVAILABLE:
            # 无 jieba, 简单截断
            return self._truncate_to_tokens(text, max_tokens)

        # v7.0: 增强切句
        sentences = self._split_sentences(text)

        if not sentences:
            return self._truncate_to_tokens(text, max_tokens)

        # ============ TextRank 关键词打分 (原有) ============
        try:
            keywords = jieba.analyse.textrank(
                text, topK=20, withWeight=True,
            )
            keyword_dict = dict(keywords)
        except Exception:
            keyword_dict = {}

        # ============ v7.0: Query-Aware 预计算 ============
        query_words: Set[str] = set()
        query_bigrams: Set[str] = set()
        if query and JIEBA_AVAILABLE:
            q_words = jieba.lcut(query)
            # 过滤停用词和单字符
            query_words = {w for w in q_words if len(w) > 1}
            # 构造二元组
            for i in range(len(q_words) - 1):
                if len(q_words[i]) > 1 or len(q_words[i + 1]) > 1:
                    query_bigrams.add(q_words[i] + q_words[i + 1])

        # ============ 多维评分 ============
        scored = []
        for i, sent in enumerate(sentences):
            # 维度 1: TextRank 关键词得分
            words = jieba.lcut(sent)
            textrank_score = sum(keyword_dict.get(w, 0) for w in words)
            
            # 维度 2: Query-Aware 相关性 (v7.0)
            query_score = self._compute_query_relevance(
                sent, query_words, query_bigrams
            ) if query_words else 0.0
            
            # 维度 3: TF-IDF 事实密度 (v7.0)
            fact_density = self._compute_fact_density(sent)
            
            # 多维融合:
            # - TextRank: 40% (文本重要性)
            # - Query 相关性: 35% (与当前问题的关联)
            # - 事实密度: 25% (事实信息量)
            #
            # 注意: textrank_score 需要归一化
            combined_score = textrank_score  # 基础分 (后续归一化)
            scored.append((i, sent, textrank_score, query_score, fact_density, combined_score))

        # TextRank 分数归一化
        max_textrank = max((s[2] for s in scored), default=1.0)
        if max_textrank > 0:
            for j in range(len(scored)):
                i, sent, tr, qs, fd, _ = scored[j]
                tr_norm = tr / max_textrank
                # 加权融合
                combined = tr_norm * 0.40 + qs * 0.35 + fd * 0.25
                scored[j] = (i, sent, tr, qs, fd, combined)

        # 按综合分排序
        scored.sort(key=lambda x: x[5], reverse=True)
        
        # v7.0: 动态 top-k (根据预算和句子数量)
        # 至少选 3 句, 最多选 min(8, 句子数/2)
        dynamic_k = max(3, min(8, len(sentences) // 2))
        selected = sorted(scored[:dynamic_k], key=lambda x: x[0])  # 恢复原文顺序

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
        
        v6.1 修复: 使用 chat template (apply_chat_template) 构造 prompt,
        避免 instruct 模型 (如 llama3.1-8b-instruct) 因纯文本 prompt 中
        混入角色标签而输出无意义 token (如 "assistant")。
        
        策略:
        1. 优先使用 tokenizer.apply_chat_template (system + user messages)
        2. 回退: 使用 ChatML 格式手动构造
        3. 最终回退: 纯文本 prompt (兼容非 chat 模型)
        4. 输出验证: 如果生成结果太短或无意义, 回退到抽取式摘要
        """
        system_instruction = (
            "你是一个摘要助手。请严格按照要求生成摘要。\n"
            "要求:\n"
            "- 仅提取事实性陈述，不得推理或补充\n"
            "- 标注不确定或可能遗漏的信息\n"
            "- 使用\"提到了\"\"讨论了\"等不确定措辞\n"
            f"- 摘要不超过{max_tokens}字"
        )
        user_message = f"请将以下对话内容压缩为摘要:\n\n{text}"
        
        try:
            # 优先使用 chat template 构造正确的多轮格式
            tokenizer = getattr(self._model_adapter, 'tokenizer', None)
            
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message},
                ]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception as e:
                    logger.debug(f"apply_chat_template failed for summary, using fallback: {e}")
                    # 回退: ChatML 格式
                    prompt = (
                        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                        f"<|im_start|>user\n{user_message}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
            else:
                # 无 tokenizer: 纯文本 prompt (兼容非 chat 模型)
                prompt = (
                    f"{system_instruction}\n\n{user_message}\n\n摘要:"
                )
            
            output = self._model_adapter.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.3,  # 低温度确保一致性
            )
            result = output.text.strip()
            
            # 输出验证: 检测无意义输出
            # llama3.1-8b-instruct 等小模型可能输出角色标签或重复 token
            if self._is_summary_invalid(result, text):
                logger.warning(
                    f"LLM summary invalid (len={len(result)}, "
                    f"content='{result[:50]}...'), falling back to extractive"
                )
                return self._extractive_summarize(text, max_tokens)
            
            return result
        except Exception as e:
            logger.warning(f"LLM summarize failed, falling back to extractive: {e}")
            return self._extractive_summarize(text, max_tokens)
    
    @staticmethod
    def _is_summary_invalid(summary: str, original: str) -> bool:
        """
        验证 LLM 生成的 summary 是否有效
        
        无效条件:
        1. 空或太短 (< 10 字符)
        2. 只包含角色标签 (assistant, user, system 等)
        3. 重复 token 模式 (如 "assistant。assistant。assistant...")
        4. 包含 chat template 特殊标记 (模型输出了格式标记而非内容)
        """
        if not summary or len(summary) < 10:
            return True
        
        # 检测只包含角色标签的无意义输出
        cleaned = summary.lower().strip()
        invalid_tokens = {'assistant', 'user', 'system', '助手', '用户'}
        
        # 去除标点后检查
        cleaned_no_punct = re.sub(r'[。.，,！!？?\s]+', '', cleaned)
        if cleaned_no_punct in invalid_tokens or len(cleaned_no_punct) < 5:
            return True
        
        # 检测重复 token 模式: 将文本按标点分割, 如果所有片段都是同一个词则无效
        # 典型案例: "assistant。assistant。assistant。assistant。assistant"
        segments = re.split(r'[。.，,！!？?\s\n]+', cleaned)
        segments = [s.strip() for s in segments if s.strip()]
        if segments and len(segments) >= 2:
            unique_segments = set(segments)
            # 如果所有片段都相同, 且这个片段是角色标签或无意义词
            if len(unique_segments) == 1:
                the_word = unique_segments.pop()
                if the_word in invalid_tokens or len(the_word) < 4:
                    return True
        
        # 检测 chat template 特殊标记泄漏
        template_markers = [
            '<|im_start|>', '<|im_end|>', '<|begin_of_text|>',
            '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
            '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>',
        ]
        for marker in template_markers:
            if marker in summary:
                return True
        
        return False

    # ================================================================
    # 认知标记提取 (v7.0: 增强版)
    # ================================================================

    # v7.0: 扩展的事实类别正则表 (从 2 类扩展到 10+ 类)
    _FACT_PATTERNS: Dict[str, str] = {
        # 时间类
        "日期": r'\d{4}[-/年]\d{1,2}[-/月]\d{0,2}日?',
        "时间": r'[上下]午|早上|晚上|凌晨|\d{1,2}[点时:]\d{0,2}[分]?',
        "时间段": r'[今昨前明后]天|上[周个]月|下[周个]月|本[周月年]|\d+天前|\d+[年月周]后',
        # 数值类
        "价格": r'\d+\.?\d*[元块万千百亿]',
        "数量": r'\d+[个只条件套份层楼间位次回趟]',
        "百分比": r'\d+\.?\d*[%％]',
        "度量": r'\d+\.?\d*(?:公里|千米|米|厘米|毫米|公斤|斤|克|升|毫升|度|℃)',
        # 联系方式
        "电话": r'1[3-9]\d{9}|\d{3,4}-\d{7,8}|\d{5,11}',
        "邮箱": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "URL": r'https?://\S+|www\.\S+',
        # 地理信息
        "地址": (
            r'[\u4e00-\u9fff]{1,6}(?:省|市|区|县|镇|村|街道|路|街|巷|号|弄|栋|幢|单元|室)'
        ),
        # 专有名词
        "书名/标题": r'[《「](.*?)[》」]',
        "引用名称": r'[""]([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9\s]{1,20})["""]',
        # 编号
        "编号": r'[A-Z]{0,3}\d{5,}|[A-Z]{2,}\d{2,}',
        "订单号": r'(?:订单|编号|单号|工单)[：:]\s*\S+',
    }

    def _extract_epistemic_markers(
        self,
        original: str,
        summary: str,
    ) -> Tuple[List[str], List[str]]:
        """
        v7.0: 增强的认知标记提取 + Summary 完整性校验
        
        相比原版的改进:
        1. 事实类别从 2 类扩展到 10+ 类
        2. 每类事实精确计数 (不仅是有/无)
        3. 部分覆盖检测: 原文有 3 个日期, summary 只保留了 1 个 → 标注
        4. Summary 完整性校验: 检查关键事实是否被遗漏
        5. 事实值级别对比: 不仅检查类别是否存在, 还检查具体值是否保留
        
        Returns:
            (facts_covered, facts_missing)
            - facts_covered: 在 summary 中保留的事实类别列表
            - facts_missing: 在 summary 中遗漏的事实类别列表 (含具体遗漏数量)
        """
        facts_covered: List[str] = []
        facts_missing: List[str] = []

        for fact_type, pattern in self._FACT_PATTERNS.items():
            try:
                orig_matches = set(re.findall(pattern, original))
                summ_matches = set(re.findall(pattern, summary))
            except re.error:
                continue

            if not orig_matches:
                continue  # 原文中没有此类事实, 跳过
            
            if not summ_matches:
                # 完全遗漏
                facts_missing.append(
                    f"{fact_type}({len(orig_matches)}项)"
                )
            elif summ_matches >= orig_matches:
                # 完全覆盖
                facts_covered.append(fact_type)
            else:
                # 部分覆盖: 标记已覆盖, 同时标注遗漏数量
                missing_count = len(orig_matches - summ_matches)
                facts_covered.append(fact_type)
                if missing_count > 0:
                    facts_missing.append(
                        f"{fact_type}(部分遗漏{missing_count}项)"
                    )

        # ============ v7.0: Summary 完整性校验 ============
        # 检查 1: 压缩比过高 (原文很长但 summary 很短)
        if len(original) > len(summary) * 4:
            facts_missing.append("详细内容(压缩比>4x)")
        elif len(original) > len(summary) * 3:
            if "详细内容" not in [f.split("(")[0] for f in facts_missing]:
                facts_missing.append("详细内容(压缩比>3x)")
        
        # 检查 2: jieba 关键词覆盖率
        if JIEBA_AVAILABLE:
            try:
                orig_keywords = jieba.analyse.textrank(
                    original, topK=10, withWeight=False,
                )
                if orig_keywords:
                    covered_kw = sum(
                        1 for kw in orig_keywords if kw in summary
                    )
                    coverage = covered_kw / len(orig_keywords)
                    if coverage < 0.3:
                        facts_missing.append(
                            f"关键词(覆盖率{coverage:.0%})"
                        )
            except Exception:
                pass

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
