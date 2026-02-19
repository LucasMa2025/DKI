"""
DKI Recall v4 — 多信号召回器

三路信号融合:
1. 关键词+权重: jieba 分词 → TF-IDF/TextRank → 数据库匹配
2. 指代解析: ReferenceResolver (已有) → 确定召回范围
3. 向量相似度: EmbeddingService + FAISS (已有) → 语义匹配

合并策略:
- 去重 (按 message_id)
- F1-1: 认知态模式选择 (Epistemic Mode) → 动态权重预设
- F1-2: 信号置信度门控 (Signal Confidence Gating) → 低置信度信号退出
- F1-3: 统一 min-max 归一化 (Score Normalization)
- 补充固定近期轮数

Author: AGI Demo Project
Version: 4.1.0
"""

import math
from dataclasses import dataclass
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
            vector_weight=profile.get("vector_weight", default_weights.vector_weight),
            recency_weight=profile.get("recency_weight", default_weights.recency_weight),
        )
    return default_weights


class MultiSignalRecall:
    """
    多信号融合召回器
    
    三路信号:
    1. 关键词+权重: jieba 分词 → TF-IDF 关键词 → 会话历史匹配
    2. 指代解析: ReferenceResolver → 召回范围
    3. 向量相似度: MemoryRouter → 语义匹配
    
    合并: 归一化分数加权融合 + 去重 + 补充近期轮次
    """

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

        self._stats = {
            "recalls": 0,
            "keyword_total_hits": 0,
            "vector_total_hits": 0,
        }

    def recall(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        db_session: Optional[Any] = None,
        max_results: int = 50,
        query_context: Optional[Any] = None,
    ) -> RecallResult:
        """
        执行多信号召回
        
        1. 指代解析 → 确定范围
        2. 关键词+权重检索
        3. 向量相似度检索
        4. F1-1: 认知态模式选择 → 动态权重
        5. F1-3: 统一 min-max 归一化
        6. F1-2: 信号置信度门控 → 低置信度退出
        7. 加权融合排序
        8. 补充固定近期轮数
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            user_id: 用户 ID
            db_session: 数据库 session
            max_results: 最大结果数
            query_context: QueryContext (可选, 用于 F1-1 认知态模式选择)
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

        # ============ 2. 关键词+权重检索 ============
        keyword_scored: Dict[str, float] = {}  # msg_id -> raw_score
        keyword_query_terms = 0   # 查询关键词数 (用于 F1-2 置信度)
        keyword_hit_terms = 0     # 命中关键词数
        if (self.config.signals.keyword_enabled
                and JIEBA_AVAILABLE
                and self._conversation_repo):
            keyword_scored, keyword_query_terms, keyword_hit_terms = self._keyword_recall_with_confidence(
                query, session_id, db_session,
                max_turns=recall_turns,
            )
            result.keyword_hits = len(keyword_scored)
            self._stats["keyword_total_hits"] += len(keyword_scored)

        # ============ 3. 向量相似度检索 ============
        vector_scored: Dict[str, float] = {}  # msg_id -> raw_score
        if self.config.signals.vector_enabled and self._memory_router:
            vector_scored = self._vector_recall(
                query, session_id, user_id
            )
            result.vector_hits = len(vector_scored)
            self._stats["vector_total_hits"] += len(vector_scored)

        # ============ 4. F1-1: 认知态模式选择 → 动态权重 ============
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
        
        # 记录选择的模式
        self._stats.setdefault("epistemic_mode_counts", {})
        self._stats["epistemic_mode_counts"][selected_mode] = (
            self._stats["epistemic_mode_counts"].get(selected_mode, 0) + 1
        )

        # ============ 5. F1-3: 统一 min-max 归一化 ============
        all_msg_ids = set(keyword_scored.keys()) | set(vector_scored.keys())
        norm_keyword = self._min_max_normalize(keyword_scored)
        norm_vector = self._min_max_normalize(vector_scored)

        # ============ 6. F1-2: 信号置信度门控 ============
        sg_config = self.config.signal_gating
        
        # 计算各信号的置信度
        kw_confidence = self._compute_keyword_confidence(
            keyword_scored, keyword_query_terms, keyword_hit_terms,
        )
        vec_confidence = self._compute_vector_confidence(vector_scored)
        # recency 置信度: 如果有 recent_turns 配置，始终活跃
        rec_confidence = 1.0
        
        # 门控: 低置信度信号退出
        signals_dropped = 0
        kw_active = True
        vec_active = True
        rec_active = True
        
        if sg_config.enabled:
            if kw_confidence < sg_config.confidence_threshold:
                kw_active = False
                signals_dropped += 1
            if vec_confidence < sg_config.confidence_threshold:
                vec_active = False
                signals_dropped += 1
        
        self._stats.setdefault("signal_gating_dropped_total", 0)
        self._stats["signal_gating_dropped_total"] += signals_dropped

        # ============ 7. 加权融合排序 (门控后动态归一化) ============
        active_signal_weights = []
        if kw_active:
            active_signal_weights.append(("keyword", active_weights.keyword_weight))
        if vec_active:
            active_signal_weights.append(("vector", active_weights.vector_weight))
        if rec_active:
            active_signal_weights.append(("recency", active_weights.recency_weight))
        
        # 动态归一化: 参与信号的权重归一化到 1.0
        w_sum = sum(w for _, w in active_signal_weights)
        if w_sum <= 0:
            w_sum = 1.0
        
        norm_weights = {name: w / w_sum for name, w in active_signal_weights}
        
        final_scores: Dict[str, float] = {}
        for msg_id in all_msg_ids:
            score = 0.0
            if kw_active:
                score += norm_weights.get("keyword", 0.0) * norm_keyword.get(msg_id, 0.0)
            if vec_active:
                score += norm_weights.get("vector", 0.0) * norm_vector.get(msg_id, 0.0)
            final_scores[msg_id] = score

        # 按 final_score 降序排序
        sorted_ids = sorted(
            final_scores.keys(),
            key=lambda x: final_scores[x],
            reverse=True,
        )[:max_results]

        result.scores = {mid: final_scores[mid] for mid in sorted_ids}

        # ============ 8. 获取完整消息对象 ============
        recalled_messages = []
        if sorted_ids and self._conversation_repo:
            try:
                recalled_messages = self._fetch_messages_by_ids(
                    sorted_ids, session_id, db_session
                )
            except Exception as e:
                logger.error(f"Failed to fetch recalled messages: {e}")

        # ============ 9. 补充近期轮次 ============
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

        # ============ 10. 合并 (近期优先, 去重) ============
        seen_ids = set()
        final_messages = []

        # 近期消息优先 (recency bonus 由门控后权重控制)
        for msg in recent_messages:
            msg_id = getattr(msg, "id", None) or getattr(msg, "message_id", id(msg))
            msg_id = str(msg_id)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                final_messages.append(msg)
                # 添加时间近度 bonus (使用门控后的 recency 权重)
                if msg_id in final_scores and rec_active:
                    final_scores[msg_id] += norm_weights.get("recency", 0.0)

        # 召回消息
        for msg in recalled_messages:
            msg_id = getattr(msg, "id", None) or getattr(msg, "message_id", id(msg))
            msg_id = str(msg_id)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                final_messages.append(msg)

        result.messages = final_messages
        # 只保留 final_messages 中出现的 msg_id 的分数
        final_msg_ids = set()
        for msg in final_messages:
            msg_id = getattr(msg, "id", None) or getattr(msg, "message_id", id(msg))
            final_msg_ids.add(str(msg_id))
        result.scores = {k: v for k, v in final_scores.items() if k in final_msg_ids}

        logger.debug(
            f"Recall complete: mode={selected_mode}, "
            f"kw={result.keyword_hits}(active={kw_active}), "
            f"vec={result.vector_hits}(active={vec_active}), "
            f"recent={result.recent_turns_added}, "
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
        - 单元素: 映射到 1.0
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
            # 所有分数相同 → 全部映射到 1.0
            return {k: 1.0 for k in raw_scores}
        
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
    ) -> Tuple[Dict[str, float], int, int]:
        """
        关键词召回 + 返回置信度所需的元数据 (F1-2)
        
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
                return scored, query_terms, 0

        if not messages:
            return scored, query_terms, 0

        # 限制回溯范围
        max_messages = max_turns * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        # 对每条消息计算关键词命中分数, 同时追踪命中关键词数
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
    # 统计
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
