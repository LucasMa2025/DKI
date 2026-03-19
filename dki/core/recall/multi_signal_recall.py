"""
DKI Recall v4 — 多信号召回器

四路信号融合:
1. 关键词+权重: jieba 分词 → TF-IDF/TextRank → 数据库匹配
2. BM25 全文检索: rank_bm25 → 基于词频的相关性排序 (P1)
3. 指代解析: ReferenceResolver (已有) → 确定召回范围
4. 向量相似度: EmbeddingService + FAISS (已有) → 语义匹配

合并策略:
- 去重 (按 message_id)
- F1-1: 认知态模式选择 (Epistemic Mode) → 动态权重预设
- F1-2: 信号置信度门控 (Signal Confidence Gating) → 低置信度信号退出
- F1-3: 统一 min-max 归一化 (Score Normalization)
- P0 修复: recency 分数参与 final_score 融合 (指数时间衰减)
- 补充固定近期轮数

P1 BM25 设计说明:
- BM25 补充关键词信号: jieba TF-IDF 侧重"关键词权重", BM25 侧重"文档相关性排序"
- 两者互补: TF-IDF 对短查询敏感, BM25 对长查询和词频分布更鲁棒
- BM25 使用 jieba 分词作为 tokenizer, 与关键词信号共享分词结果
- BM25 独立参与信号融合 (有自己的权重和置信度门控)
- P1 优化: keyword 和 BM25 共享同一次 DB 拉取, 避免重复 IO
- P1 优化: BM25 索引按 (session_id, last_msg_id) 缓存, 避免每次重建

Author: AGI Demo Project
Version: 4.4.0
"""

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from dki.core.recall.recall_config import (
    RecallConfig,
    RecallResult,
    RecallScoreWeights,
    EpistemicModeConfig,
    SignalGatingConfig,
)

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba not installed. Keyword recall disabled. pip install jieba")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.info("rank_bm25 not installed. BM25 recall disabled. pip install rank-bm25")


@dataclass
class SignalConfidence:
    """
    单路信号的置信度评估 (F1-2)
    
    用于信号置信度门控：低置信度信号退出加权，
    剩余信号动态归一化。
    """
    score: float = 0.0         # 归一化后的分数
    confidence: float = 0.0    # 置信度 [0, 1]
    coverage: float = 0.0      # 覆盖率 (keyword: 命中词数/查询词数)
    active: bool = True        # 是否参与融合


def select_epistemic_mode(
    context: Optional[Any] = None,
    config: Optional[EpistemicModeConfig] = None,
) -> str:
    """
    基于 QueryContext 选择认知态模式 (F1-1)
    
    纯规则，无 ML 依赖。规则可审计、可调试、可热更新。
    
    规则优先级（从高到低）：
    1. reference_resolved → clarification
    2. memory_triggered + high confidence → semantic_search
    3. trigger_type == correction → correction
    4. fallback → direct_lookup
    
    Args:
        context: QueryContext (可选，来自 Planner.analyze_query)
        config: EpistemicModeConfig (可选)
        
    Returns:
        模式名称（对应 config.profiles 的 key）
    """
    if not config or not config.enabled:
        return config.fallback_mode if config else "direct_lookup"
    
    if not context:
        return config.fallback_mode
    
    # 规则 1: 指代解析成功 → clarification
    reference_resolved = getattr(context, "reference_resolved", False)
    if reference_resolved:
        return "clarification"
    
    # 规则 2: 高置信度记忆触发 → semantic_search
    memory_triggered = getattr(context, "memory_triggered", False)
    trigger_confidence = getattr(context, "trigger_confidence", 0.0)
    if memory_triggered and trigger_confidence > 0.7:
        return "semantic_search"
    
    # 规则 3: 修正类触发 → correction
    trigger_type = getattr(context, "trigger_type", None)
    if trigger_type and "correct" in str(trigger_type).lower():
        return "correction"
    
    # 规则 4: 默认
    return config.fallback_mode


def get_weights_for_mode(
    mode: str,
    config: EpistemicModeConfig,
    default_weights: RecallScoreWeights,
) -> RecallScoreWeights:
    """
    获取指定模式的权重预设 (F1-1)
    
    Args:
        mode: 模式名称
        config: EpistemicModeConfig
        default_weights: 默认权重 (当模式不存在时使用)
        
    Returns:
        RecallScoreWeights
    """
    if mode in config.profiles:
        profile = config.profiles[mode]
        return RecallScoreWeights(
            keyword_weight=profile.get("keyword_weight", default_weights.keyword_weight),
            bm25_weight=profile.get("bm25_weight", default_weights.bm25_weight),
            vector_weight=profile.get("vector_weight", default_weights.vector_weight),
            recency_weight=profile.get("recency_weight", default_weights.recency_weight),
        )
    return default_weights


class MultiSignalRecall:
    """
    多信号融合召回器
    
    四路信号:
    1. 关键词+权重: jieba 分词 → TF-IDF 关键词 → 会话历史匹配
    2. BM25 全文检索: rank_bm25 → 基于词频的文档相关性排序 (P1)
    3. 指代解析: ReferenceResolver → 召回范围
    4. 向量相似度: MemoryRouter → 语义匹配
    
    合并: 归一化分数加权融合 + 去重 + 补充近期轮次
    
    P1 优化:
    - keyword 和 BM25 共享同一次 DB 拉取 (避免双倍 IO)
    - BM25 索引按 (session_id, last_msg_id) 缓存 (避免每次重建)
    - P0 修复: recency 分数参与 final_score 融合
    """

    # BM25 索引缓存容量 (LRU)
    _BM25_CACHE_MAXSIZE = 32

    def __init__(
        self,
        config: RecallConfig,
        reference_resolver: Optional[Any] = None,
        memory_router: Optional[Any] = None,
        conversation_repo: Optional[Any] = None,
    ):
        self.config = config
        self._reference_resolver = reference_resolver
        self._memory_router = memory_router
        self._conversation_repo = conversation_repo

        # P1: BM25 索引 LRU 缓存 key=(session_id, last_msg_id)
        self._bm25_cache: OrderedDict = OrderedDict()

        self._stats = {
            "recalls": 0,
            "keyword_total_hits": 0,
            "bm25_total_hits": 0,
            "vector_total_hits": 0,
            "bm25_cache_hits": 0,
        }

    def recall(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        db_session: Optional[Any] = None,
        max_results: int = 50,
        query_context: Optional[Any] = None,
        retrieval_mode: str = "unknown",
    ) -> RecallResult:
        """
        执行多信号召回
        
        1. 指代解析 → 确定范围
        2. 共享 DB 拉取 (P1: keyword + BM25 复用同一次查询)
        3. 关键词+权重检索
        4. BM25 全文检索 (P1: LRU 索引缓存)
        5. 向量相似度检索
        6. F1-1: 认知态模式选择 → 动态权重
        7. P0: recency 时间衰减分数计算
        8. F1-3: 统一 min-max 归一化 (含 recency)
        9. F1-2: 信号置信度门控 → 低置信度退出
        10. 加权融合排序 (含 recency 分量)
        11. 补充固定近期轮次
        12. 跨会话历史召回 (BM25 相关性过滤)
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            user_id: 用户 ID
            db_session: 数据库 session
            max_results: 最大结果数
            query_context: QueryContext (可选, 用于 F1-1 认知态模式选择)
            retrieval_mode: 检索模式 (bm25_only | bm25_embedding | keyword | unknown)
        """
        self._stats["recalls"] += 1
        result = RecallResult()

        # ============ 1. 指代解析 ============
        recall_turns = 10  # 默认
        if self.config.signals.reference_enabled and self._reference_resolver:
            try:
                ref_result = self._reference_resolver.resolve(query)
                if ref_result.recall_turns:
                    recall_turns = ref_result.recall_turns
                result.reference_scope = (
                    ref_result.scope.value if ref_result.scope else None
                )
            except Exception as e:
                logger.warning(f"Reference resolver error: {e}")

        # ============ 2. P1: 共享 DB 拉取 (keyword + BM25 复用) ============
        # 避免 keyword 和 BM25 各自独立拉取同一 session 的消息 (双倍 IO)
        shared_messages: Optional[List[Any]] = None
        if (self._conversation_repo
                and (self.config.signals.keyword_enabled or self.config.signals.bm25_enabled)
                and JIEBA_AVAILABLE):
            try:
                raw = (
                    self._conversation_repo.get_by_session(
                        session_id=session_id,
                        db_session=db_session,
                    ) if db_session
                    else self._conversation_repo.get_by_session(session_id=session_id)
                )
                max_messages = recall_turns * 2
                shared_messages = raw[-max_messages:] if raw and len(raw) > max_messages else (raw or [])
            except Exception:
                try:
                    raw = self._conversation_repo.get_by_session(session_id=session_id)
                    max_messages = recall_turns * 2
                    shared_messages = raw[-max_messages:] if raw and len(raw) > max_messages else (raw or [])
                except Exception as e:
                    logger.warning(f"Shared DB fetch failed: {e}")
                    shared_messages = []

        # ============ 3. 关键词+权重检索 ============
        keyword_scored: Dict[str, float] = {}
        keyword_query_terms = 0
        keyword_hit_terms = 0
        if (self.config.signals.keyword_enabled
                and JIEBA_AVAILABLE
                and self._conversation_repo):
            keyword_scored, keyword_query_terms, keyword_hit_terms = (
                self._keyword_recall_with_confidence(
                    query, session_id, db_session,
                    max_turns=recall_turns,
                    shared_messages=shared_messages,
                )
            )
            result.keyword_hits = len(keyword_scored)
            self._stats["keyword_total_hits"] += len(keyword_scored)

        # ============ 4. BM25 全文检索 (P1: LRU 缓存) ============
        bm25_scored: Dict[str, float] = {}
        if (self.config.signals.bm25_enabled
                and BM25_AVAILABLE
                and JIEBA_AVAILABLE
                and self._conversation_repo):
            bm25_scored = self._bm25_recall(
                query, session_id, db_session,
                max_turns=recall_turns,
                shared_messages=shared_messages,
            )
            result.bm25_hits = len(bm25_scored)
            self._stats["bm25_total_hits"] += len(bm25_scored)

        # ============ 5. 向量相似度检索 ============
        vector_scored: Dict[str, float] = {}
        if self.config.signals.vector_enabled and self._memory_router:
            vector_scored = self._vector_recall(
                query, session_id, user_id
            )
            result.vector_hits = len(vector_scored)
            self._stats["vector_total_hits"] += len(vector_scored)

        # ============ 6. F1-1: 认知态模式选择 → 动态权重 ============
        em_config = self.config.epistemic_modes
        selected_mode = select_epistemic_mode(
            context=query_context,
            config=em_config,
        )
        active_weights = get_weights_for_mode(
            mode=selected_mode,
            config=em_config,
            default_weights=self.config.score_weights,
        )

        # BM25-only 模式: 覆盖权重，大幅提升 recency
        if retrieval_mode == "bm25_only":
            bm25_cfg = self.config.bm25_only_tuning
            active_weights = RecallScoreWeights(
                keyword_weight=bm25_cfg.keyword_weight,
                bm25_weight=bm25_cfg.bm25_weight,
                vector_weight=0.0,
                recency_weight=bm25_cfg.recency_weight,
            )
            logger.debug(
                f"BM25-only mode: overriding weights → "
                f"kw={bm25_cfg.keyword_weight}, bm25={bm25_cfg.bm25_weight}, "
                f"recency={bm25_cfg.recency_weight}"
            )

        self._stats.setdefault("epistemic_mode_counts", {})
        self._stats["epistemic_mode_counts"][selected_mode] = (
            self._stats["epistemic_mode_counts"].get(selected_mode, 0) + 1
        )

        # ============ 7. P0: recency 时间衰减分数 ============
        # 收集所有候选消息的时间戳，计算指数衰减分数
        all_candidate_msgs: Dict[str, Any] = {}
        for msgs in (shared_messages or []):
            mid = str(getattr(msgs, "id", None) or getattr(msgs, "message_id", id(msgs)))
            all_candidate_msgs[mid] = msgs
        recency_scored = self._compute_recency_scores(all_candidate_msgs)

        # ============ 8. F1-3: 统一 min-max 归一化 (含 recency) ============
        all_msg_ids = (
            set(keyword_scored.keys())
            | set(bm25_scored.keys())
            | set(vector_scored.keys())
            | set(recency_scored.keys())
        )
        norm_keyword = self._min_max_normalize(keyword_scored)
        norm_bm25 = self._min_max_normalize(bm25_scored)
        norm_vector = self._min_max_normalize(vector_scored)
        norm_recency = self._min_max_normalize(recency_scored)

        # ============ 9. F1-2: 信号置信度门控 ============
        sg_config = self.config.signal_gating

        kw_confidence = self._compute_keyword_confidence(
            keyword_scored, keyword_query_terms, keyword_hit_terms,
        )
        bm25_confidence = self._compute_bm25_confidence(bm25_scored)
        vec_confidence = self._compute_vector_confidence(vector_scored)
        rec_confidence = 1.0  # recency 始终活跃

        signals_dropped = 0
        kw_active = True
        bm25_active = True
        vec_active = True
        rec_active = True

        if sg_config.enabled:
            if kw_confidence < sg_config.confidence_threshold:
                kw_active = False
                signals_dropped += 1
            if bm25_confidence < sg_config.confidence_threshold:
                bm25_active = False
                signals_dropped += 1
            if vec_confidence < sg_config.confidence_threshold:
                vec_active = False
                signals_dropped += 1
            # BM25-only 模式下 vector 强制不活跃
            if retrieval_mode == "bm25_only":
                vec_active = False

        self._stats.setdefault("signal_gating_dropped_total", 0)
        self._stats["signal_gating_dropped_total"] += signals_dropped

        # ============ 10. 加权融合排序 (含 recency 分量, P0 修复) ============
        active_signal_weights = []
        if kw_active:
            active_signal_weights.append(("keyword", active_weights.keyword_weight))
        if bm25_active:
            active_signal_weights.append(("bm25", active_weights.bm25_weight))
        if vec_active:
            active_signal_weights.append(("vector", active_weights.vector_weight))
        if rec_active:
            active_signal_weights.append(("recency", active_weights.recency_weight))

        w_sum = sum(w for _, w in active_signal_weights)
        if w_sum <= 0:
            w_sum = 1.0

        norm_weights = {name: w / w_sum for name, w in active_signal_weights}

        final_scores: Dict[str, float] = {}
        for msg_id in all_msg_ids:
            score = 0.0
            if kw_active:
                score += norm_weights.get("keyword", 0.0) * norm_keyword.get(msg_id, 0.0)
            if bm25_active:
                score += norm_weights.get("bm25", 0.0) * norm_bm25.get(msg_id, 0.0)
            if vec_active:
                score += norm_weights.get("vector", 0.0) * norm_vector.get(msg_id, 0.0)
            if rec_active:
                # P0 修复: recency 分量正式参与 final_score
                score += norm_weights.get("recency", 0.0) * norm_recency.get(msg_id, 0.0)
            final_scores[msg_id] = score

        sorted_ids = sorted(
            final_scores.keys(),
            key=lambda x: final_scores[x],
            reverse=True,
        )[:max_results]

        result.scores = {mid: final_scores[mid] for mid in sorted_ids}

        # ============ 11. 获取完整消息对象 ============
        recalled_messages = []
        if sorted_ids and self._conversation_repo:
            try:
                recalled_messages = self._fetch_messages_by_ids(
                    sorted_ids, session_id, db_session
                )
            except Exception as e:
                logger.error(f"Failed to fetch recalled messages: {e}")

        # ============ 12. 补充近期轮次 ============
        recent_messages = []
        min_turns = self.config.budget.min_recent_turns
        if min_turns > 0 and self._conversation_repo:
            try:
                recent_messages = self._get_recent_turns(
                    session_id, min_turns, db_session
                )
                result.recent_turns_added = len(recent_messages)
            except Exception as e:
                logger.warning(f"Failed to get recent turns: {e}")

        # ============ 13. 合并 (近期优先, 去重) ============
        seen_ids = set()
        final_messages = []

        for msg in recent_messages:
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                final_messages.append(msg)

        for msg in recalled_messages:
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                final_messages.append(msg)

        # ============ 14. 跨会话历史召回 (P2: BM25 相关性过滤) ============
        cross_session_count = 0
        if user_id and self._conversation_repo and hasattr(self._conversation_repo, 'get_cross_session_history'):
            try:
                bm25_cfg = self.config.bm25_only_tuning
                cross_session_limit = (
                    bm25_cfg.cross_session_limit
                    if retrieval_mode == "bm25_only"
                    else (
                        self.config.budget.cross_session_limit
                        if hasattr(self.config.budget, 'cross_session_limit')
                        else 10
                    )
                )
                cross_session_msgs = self._conversation_repo.get_cross_session_history(
                    user_id=user_id,
                    current_session_id=session_id,
                    limit=cross_session_limit * 3,  # 多拉一些，后面过滤
                )
                # P2: BM25-only 模式下对跨会话消息做相关性 + 时间衰减过滤
                if retrieval_mode == "bm25_only" and cross_session_msgs and bm25_scored:
                    cross_session_msgs = self._filter_cross_session_by_relevance(
                        cross_session_msgs=cross_session_msgs,
                        query=query,
                        bm25_min_score=bm25_cfg.cross_session_min_bm25_score,
                        decay_rate=bm25_cfg.cross_session_time_decay_rate,
                        limit=cross_session_limit,
                    )

                for msg in cross_session_msgs:
                    msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))
                    if msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        final_messages.append(msg)
                        cross_session_count += 1

                if cross_session_count > 0:
                    logger.info(
                        f"Cross-session recall: added {cross_session_count} messages "
                        f"from previous sessions for user {user_id}"
                    )
            except Exception as e:
                logger.warning(f"Cross-session recall failed (non-critical): {e}")

        result.messages = final_messages
        final_msg_ids = {
            str(getattr(m, "id", None) or getattr(m, "message_id", id(m)))
            for m in final_messages
        }
        result.scores = {k: v for k, v in final_scores.items() if k in final_msg_ids}

        logger.debug(
            f"Recall complete: mode={selected_mode}, retrieval={retrieval_mode}, "
            f"kw={result.keyword_hits}(active={kw_active}), "
            f"bm25={result.bm25_hits}(active={bm25_active}), "
            f"vec={result.vector_hits}(active={vec_active}), "
            f"recency_active={rec_active}, "
            f"recent={result.recent_turns_added}, "
            f"cross_session={cross_session_count}, "
            f"gating_dropped={signals_dropped}, "
            f"total={len(final_messages)}"
        )

        return result

    # ================================================================
    # 关键词召回
    # ================================================================

    def _keyword_recall(
        self,
        query: str,
        session_id: str,
        db_session: Optional[Any],
        max_turns: int = 10,
    ) -> Dict[str, float]:
        """关键词+权重检索
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            db_session: 数据库 session (可选)
            max_turns: 最大回溯轮数 (由指代解析决定)
        """
        scored = {}

        # 提取关键词
        topk = self.config.signals.keyword_topk
        method = self.config.signals.keyword_method

        try:
            if method == "textrank":
                keywords = jieba.analyse.textrank(
                    query, topK=topk, withWeight=True,
                )
            else:
                keywords = jieba.analyse.extract_tags(
                    query, topK=topk, withWeight=True,
                    allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn'),
                )
        except Exception as e:
            logger.warning(f"jieba keyword extraction failed: {e}")
            return scored

        if not keywords:
            return scored

        logger.debug(f"Keywords extracted: {keywords}")

        # 从会话历史中匹配关键词
        try:
            messages = self._conversation_repo.get_by_session(
                session_id=session_id,
                db_session=db_session,
            ) if db_session else self._conversation_repo.get_by_session(
                session_id=session_id,
            )
        except Exception:
            try:
                messages = self._conversation_repo.get_by_session(
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to get session messages: {e}")
                return scored

        if not messages:
            return scored

        # 限制回溯范围: 只搜索最近 max_turns 轮 (每轮 user+assistant)
        max_messages = max_turns * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        # 对每条消息计算关键词命中分数
        for msg in messages:
            content = getattr(msg, "content", "")
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))

            score = 0.0
            for kw, weight in keywords:
                if kw in content:
                    score += weight

            if score > 0:
                scored[msg_id] = score

        return scored

    # ================================================================
    # 向量召回
    # ================================================================

    def _vector_recall(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
    ) -> Dict[str, float]:
        """向量相似度检索"""
        scored = {}

        try:
            top_k = self.config.signals.vector_top_k
            threshold = self.config.signals.vector_threshold

            results = self._memory_router.search(
                query=query,
                top_k=top_k,
            )

            for r in results:
                score = getattr(r, "score", 0.0)
                if score >= threshold:
                    msg_id = str(
                        getattr(r, "id", None)
                        or getattr(r, "message_id", None)
                        or getattr(r, "memory_id", id(r))
                    )
                    scored[msg_id] = score

        except Exception as e:
            logger.warning(f"Vector recall failed: {e}")

        return scored

    # ================================================================
    # BM25 全文检索 (P1)
    # ================================================================

    def _bm25_recall(
        self,
        query: str,
        session_id: str,
        db_session: Optional[Any],
        max_turns: int = 10,
        shared_messages: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        BM25 全文检索 (P1: LRU 索引缓存 + 共享 DB 拉取)
        
        使用 rank_bm25 (BM25Okapi) 对会话历史进行全文检索。
        与关键词信号互补:
        - jieba TF-IDF: 提取关键词 → 精确匹配 → 侧重"关键词权重"
        - BM25: 全文分词 → 文档相关性排序 → 侧重"整体相关性"
        
        P1 优化:
        - 接受 shared_messages 参数，避免重复 DB 拉取
        - 按 (session_id, last_msg_id) 缓存 BM25 索引，同一 session 消息未变时复用
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            db_session: 数据库 session (可选)
            max_turns: 最大回溯轮数 (由指代解析决定)
            shared_messages: 共享的消息列表 (P1: 避免重复 DB 拉取)
            
        Returns:
            msg_id → BM25 score 的字典
        """
        scored: Dict[str, float] = {}

        if not BM25_AVAILABLE or not JIEBA_AVAILABLE:
            return scored

        # P1: 优先使用共享消息列表
        if shared_messages is not None:
            messages = shared_messages
        else:
            try:
                raw = (
                    self._conversation_repo.get_by_session(
                        session_id=session_id,
                        db_session=db_session,
                    ) if db_session
                    else self._conversation_repo.get_by_session(session_id=session_id)
                )
            except Exception:
                try:
                    raw = self._conversation_repo.get_by_session(session_id=session_id)
                except Exception as e:
                    logger.warning(f"BM25: Failed to get session messages: {e}")
                    return scored

            if not raw:
                return scored

            max_messages = max_turns * 2
            messages = raw[-max_messages:] if len(raw) > max_messages else raw

        if not messages:
            return scored

        # P1: BM25 索引 LRU 缓存 — key=(session_id, last_msg_id)
        last_msg_id = str(
            getattr(messages[-1], "id", None)
            or getattr(messages[-1], "message_id", id(messages[-1]))
        )
        cache_key = (session_id, last_msg_id)

        if cache_key in self._bm25_cache:
            bm25, msg_ids = self._bm25_cache[cache_key]
            self._bm25_cache.move_to_end(cache_key)
            self._stats["bm25_cache_hits"] += 1
            logger.debug(f"BM25 index cache hit: session={session_id}, last={last_msg_id}")
        else:
            # 构建 BM25 语料库
            corpus: List[List[str]] = []
            msg_ids: List[str] = []
            for msg in messages:
                content = getattr(msg, "content", "")
                msg_id = str(
                    getattr(msg, "id", None)
                    or getattr(msg, "message_id", id(msg))
                )
                if not content.strip():
                    continue
                tokens = list(jieba.cut(content))
                tokens = [t.strip() for t in tokens if len(t.strip()) > 1]
                if tokens:
                    corpus.append(tokens)
                    msg_ids.append(msg_id)

            if not corpus:
                return scored

            try:
                bm25 = BM25Okapi(corpus)
            except Exception as e:
                logger.warning(f"BM25: Failed to build index: {e}")
                return scored

            # 写入 LRU 缓存
            self._bm25_cache[cache_key] = (bm25, msg_ids)
            self._bm25_cache.move_to_end(cache_key)
            while len(self._bm25_cache) > self._BM25_CACHE_MAXSIZE:
                self._bm25_cache.popitem(last=False)

        # 对查询分词
        query_tokens = [t.strip() for t in jieba.cut(query) if len(t.strip()) > 1]
        if not query_tokens:
            return scored

        try:
            scores = bm25.get_scores(query_tokens)
        except Exception as e:
            logger.warning(f"BM25: Failed to compute scores: {e}")
            return scored

        top_k = self.config.signals.bm25_top_k
        scored_pairs = sorted(
            zip(msg_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        for msg_id, score in scored_pairs:
            if score > 0:
                scored[msg_id] = float(score)

        logger.debug(
            f"BM25 recall: {len(scored)} hits from {len(msg_ids)} docs, "
            f"query_tokens={query_tokens[:5]}..."
        )

        return scored

    @staticmethod
    def _compute_bm25_confidence(
        scored: Dict[str, float],
    ) -> float:
        """
        计算 BM25 信号的置信度 (F1-2)
        
        置信度策略:
        - 基于 top-3 分数的均值, 归一化到 [0, 1]
        - BM25 分数没有固定上界, 使用 sigmoid 映射
        - sigmoid(mean_top3 / 5.0) 使得 mean_top3=5 时置信度约 0.73
        
        Returns:
            confidence ∈ [0, 1]
        """
        if not scored:
            return 0.0
        
        top_scores = sorted(scored.values(), reverse=True)[:3]
        mean_top3 = sum(top_scores) / len(top_scores)
        
        # BM25 分数没有固定上界, 使用 sigmoid 归一化
        # sigmoid(x/5) 使得 x=5 → 0.73, x=10 → 0.88, x=2 → 0.60
        try:
            confidence = 1.0 / (1.0 + math.exp(-mean_top3 / 5.0))
        except OverflowError:
            confidence = 1.0 if mean_top3 > 0 else 0.0
        
        # 调整: sigmoid(0) = 0.5, 我们希望 0 分数 → 0 置信度
        # 使用 2 * (sigmoid - 0.5) 映射到 [0, 1]
        confidence = max(0.0, 2.0 * (confidence - 0.5))
        
        return confidence

    # ================================================================
    # 分数归一化 (补充建议: 必须显式归一化)
    # ================================================================

    @staticmethod
    def _normalize_scores(
        raw_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        对 TF-IDF 权重做 sigmoid 归一化到 [0, 1]
        
        sigmoid(x) = 1 / (1 + exp(-k*(x - x_mid)))
        k=5, x_mid=median 使得中位数映射到 0.5
        """
        if not raw_scores:
            return {}

        values = list(raw_scores.values())
        median_val = sorted(values)[len(values) // 2]

        normalized = {}
        for msg_id, score in raw_scores.items():
            # sigmoid 归一化, k=5
            try:
                sig = 1.0 / (1.0 + math.exp(-5.0 * (score - median_val)))
            except OverflowError:
                sig = 0.0 if score < median_val else 1.0
            normalized[msg_id] = sig

        return normalized

    @staticmethod
    def _clip_scores(
        raw_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """clip 向量分数到 [0, 1]"""
        return {
            k: max(0.0, min(1.0, v))
            for k, v in raw_scores.items()
        }

    # ================================================================
    # F1-3: 统一 min-max 归一化
    # ================================================================

    @staticmethod
    def _min_max_normalize(
        raw_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        统一 min-max 归一化到 [0, 1] (F1-3)
        
        所有信号使用相同的归一化方法:
        - 单元素或所有分数相同: 映射到 0.5 (P2 修复: 避免单命中消息排名虚高)
        - 多元素: (x - min) / (max - min)
        
        优点: 简单、可解释、各信号可比
        """
        if not raw_scores:
            return {}

        values = list(raw_scores.values())
        v_min = min(values)
        v_max = max(values)
        spread = v_max - v_min

        if spread <= 0:
            # 单元素或所有分数相同 → 映射到中间值 0.5，避免单命中消息排名虚高
            return {k: 0.5 for k in raw_scores}

        return {
            k: (v - v_min) / spread
            for k, v in raw_scores.items()
        }

    # ================================================================
    # F1-2: 信号置信度门控 — 置信度计算
    # ================================================================

    def _keyword_recall_with_confidence(
        self,
        query: str,
        session_id: str,
        db_session: Optional[Any],
        max_turns: int = 10,
        shared_messages: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, float], int, int]:
        """
        关键词召回 + 返回置信度所需的元数据 (F1-2)
        
        P1 优化: 接受 shared_messages 参数，避免重复 DB 拉取。
        
        Returns:
            (scored, query_terms, hit_terms):
            - scored: msg_id → raw_score
            - query_terms: 查询关键词数
            - hit_terms: 至少命中一条消息的关键词数
        """
        scored: Dict[str, float] = {}
        query_terms = 0
        hit_terms = 0

        topk = self.config.signals.keyword_topk
        method = self.config.signals.keyword_method

        try:
            if method == "textrank":
                keywords = jieba.analyse.textrank(
                    query, topK=topk, withWeight=True,
                )
            else:
                keywords = jieba.analyse.extract_tags(
                    query, topK=topk, withWeight=True,
                    allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn'),
                )
        except Exception as e:
            logger.warning(f"jieba keyword extraction failed: {e}")
            return scored, 0, 0

        if not keywords:
            return scored, 0, 0

        query_terms = len(keywords)
        logger.debug(f"Keywords extracted: {keywords}")

        # P1: 优先使用共享消息列表，避免重复 DB 拉取
        if shared_messages is not None:
            messages = shared_messages
        else:
            try:
                messages = self._conversation_repo.get_by_session(
                    session_id=session_id,
                    db_session=db_session,
                ) if db_session else self._conversation_repo.get_by_session(
                    session_id=session_id,
                )
            except Exception:
                try:
                    messages = self._conversation_repo.get_by_session(
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to get session messages: {e}")
                    return scored, query_terms, 0

            if not messages:
                return scored, query_terms, 0

            max_messages = max_turns * 2
            if len(messages) > max_messages:
                messages = messages[-max_messages:]

        if not messages:
            return scored, query_terms, 0

        hit_keyword_set = set()
        for msg in messages:
            content = getattr(msg, "content", "")
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))

            score = 0.0
            for kw, weight in keywords:
                if kw in content:
                    score += weight
                    hit_keyword_set.add(kw)

            if score > 0:
                scored[msg_id] = score

        hit_terms = len(hit_keyword_set)
        return scored, query_terms, hit_terms

    @staticmethod
    def _compute_keyword_confidence(
        scored: Dict[str, float],
        query_terms: int,
        hit_terms: int,
    ) -> float:
        """
        计算关键词信号的置信度 (F1-2)
        
        置信度 = coverage × density
        - coverage = hit_terms / query_terms  (命中关键词占查询关键词比例)
        - density = min(1.0, len(scored) / 3)  (命中消息数, 3 条封顶)
        
        Returns:
            confidence ∈ [0, 1]
        """
        if query_terms <= 0 or not scored:
            return 0.0
        
        coverage = hit_terms / query_terms
        density = min(1.0, len(scored) / 3.0)
        return coverage * density

    @staticmethod
    def _compute_vector_confidence(
        scored: Dict[str, float],
    ) -> float:
        """
        计算向量信号的置信度 (F1-2)
        
        置信度 = mean(top-3 scores)
        - 向量分数本身就是 cosine similarity ∈ [0, 1]
        - 取 top-3 均值作为整体置信度
        
        Returns:
            confidence ∈ [0, 1]
        """
        if not scored:
            return 0.0
        
        top_scores = sorted(scored.values(), reverse=True)[:3]
        return sum(top_scores) / len(top_scores)

    # ================================================================
    # 消息获取
    # ================================================================

    def _fetch_messages_by_ids(
        self,
        msg_ids: List[str],
        session_id: str,
        db_session: Optional[Any],
    ) -> List[Any]:
        """根据 message_id 列表获取完整消息"""
        try:
            all_messages = self._conversation_repo.get_by_session(
                session_id=session_id,
            )
        except Exception:
            return []

        id_set = set(msg_ids)
        matched = []
        for msg in all_messages:
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))
            if msg_id in id_set:
                matched.append(msg)

        # 按原始排序顺序返回
        id_order = {mid: i for i, mid in enumerate(msg_ids)}
        matched.sort(
            key=lambda m: id_order.get(
                str(getattr(m, "id", None) or getattr(m, "message_id", id(m))), 999
            )
        )
        return matched

    def _get_recent_turns(
        self,
        session_id: str,
        n_turns: int,
        db_session: Optional[Any],
    ) -> List[Any]:
        """获取最近 N 轮会话消息"""
        try:
            if hasattr(self._conversation_repo, "get_recent"):
                return self._conversation_repo.get_recent(
                    session_id=session_id,
                    limit=n_turns * 2,  # 每轮 user+assistant
                )
            else:
                all_msgs = self._conversation_repo.get_by_session(
                    session_id=session_id,
                )
                return all_msgs[-(n_turns * 2):] if all_msgs else []
        except Exception as e:
            logger.warning(f"get_recent_turns failed: {e}")
            return []

    # ================================================================
    # P0: recency 时间衰减分数
    # ================================================================

    @staticmethod
    def _compute_recency_scores(
        msg_map: Dict[str, Any],
        decay_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        P0 修复: 计算每条消息的时间近度分数 (指数衰减)
        
        score(msg) = exp(-decay_rate * hours_ago)
        
        - 越新的消息分数越高 (趋近 1.0)
        - 越旧的消息分数越低 (趋近 0.0)
        - decay_rate=0.1 时: 1 小时前 ≈ 0.90, 24 小时前 ≈ 0.09
        
        Args:
            msg_map: msg_id → 消息对象
            decay_rate: 每小时衰减率 (默认 0.1)
            
        Returns:
            msg_id → recency_score ∈ [0, 1]
        """
        if not msg_map:
            return {}

        now = datetime.now(timezone.utc)
        scored: Dict[str, float] = {}

        for msg_id, msg in msg_map.items():
            ts = getattr(msg, "timestamp", None) or getattr(msg, "created_at", None)
            if ts is None:
                scored[msg_id] = 0.5  # 无时间戳时给中间值
                continue

            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                hours_ago = max(0.0, (now - ts).total_seconds() / 3600.0)
                scored[msg_id] = math.exp(-decay_rate * hours_ago)
            except Exception:
                scored[msg_id] = 0.5

        return scored

    # ================================================================
    # P2: 跨会话消息相关性过滤
    # ================================================================

    def _filter_cross_session_by_relevance(
        self,
        cross_session_msgs: List[Any],
        query: str,
        bm25_min_score: float = 1.0,
        decay_rate: float = 0.05,
        limit: int = 5,
    ) -> List[Any]:
        """
        P2: 对跨会话消息做 BM25 相关性 + 时间衰减过滤
        
        在 BM25-only 模式下，跨会话消息容易引入噪声（老但关键词撞车）。
        此方法对候选跨会话消息进行二次评分：
        - BM25 分数低于阈值的直接丢弃
        - 剩余消息按 BM25 × 时间衰减综合排序
        
        Args:
            cross_session_msgs: 候选跨会话消息列表
            query: 当前查询
            bm25_min_score: BM25 最低分数阈值
            decay_rate: 每天时间衰减率 (默认 0.05, 即每天衰减 5%)
            limit: 最终保留条数
            
        Returns:
            过滤并排序后的跨会话消息列表
        """
        if not cross_session_msgs or not BM25_AVAILABLE or not JIEBA_AVAILABLE:
            return cross_session_msgs[:limit]

        # 对跨会话消息构建临时 BM25 索引
        corpus: List[List[str]] = []
        msg_ids: List[str] = []
        for msg in cross_session_msgs:
            content = getattr(msg, "content", "")
            msg_id = str(getattr(msg, "id", None) or getattr(msg, "message_id", id(msg)))
            if not content.strip():
                continue
            tokens = [t.strip() for t in jieba.cut(content) if len(t.strip()) > 1]
            if tokens:
                corpus.append(tokens)
                msg_ids.append(msg_id)

        if not corpus:
            return cross_session_msgs[:limit]

        try:
            bm25 = BM25Okapi(corpus)
            query_tokens = [t.strip() for t in jieba.cut(query) if len(t.strip()) > 1]
            if not query_tokens:
                return cross_session_msgs[:limit]
            scores = bm25.get_scores(query_tokens)
        except Exception as e:
            logger.warning(f"Cross-session BM25 filter failed: {e}")
            return cross_session_msgs[:limit]

        # 构建 msg_id → (msg, bm25_score) 映射
        id_to_msg = {
            str(getattr(m, "id", None) or getattr(m, "message_id", id(m))): m
            for m in cross_session_msgs
        }
        id_to_bm25 = dict(zip(msg_ids, scores))

        now = datetime.now(timezone.utc)
        scored_msgs: List[Tuple[float, Any]] = []

        for msg_id, bm25_score in id_to_bm25.items():
            if bm25_score < bm25_min_score:
                continue  # 相关性不足，丢弃
            msg = id_to_msg.get(msg_id)
            if msg is None:
                continue

            # 时间衰减 (按天)
            ts = getattr(msg, "timestamp", None) or getattr(msg, "created_at", None)
            time_factor = 1.0
            if ts is not None:
                try:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    days_ago = max(0.0, (now - ts).total_seconds() / 86400.0)
                    time_factor = math.exp(-decay_rate * days_ago)
                except Exception:
                    pass

            combined_score = bm25_score * time_factor
            scored_msgs.append((combined_score, msg))

        # 按综合分数降序排序，取 top-limit
        scored_msgs.sort(key=lambda x: x[0], reverse=True)
        result = [m for _, m in scored_msgs[:limit]]

        logger.debug(
            f"Cross-session filter: {len(cross_session_msgs)} candidates → "
            f"{len(result)} kept (bm25_min={bm25_min_score}, limit={limit})"
        )
        return result

    # ================================================================
    # 统计
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
