"""
DKI Recall v4 — 多信号召回器

三路信号融合:
1. 关键词+权重: jieba 分词 → TF-IDF/TextRank → 数据库匹配
2. 指代解析: ReferenceResolver (已有) → 确定召回范围
3. 向量相似度: EmbeddingService + FAISS (已有) → 语义匹配

合并策略:
- 去重 (按 message_id)
- 归一化分数融合: final_score = w1*norm_keyword + w2*norm_vector + w3*recency
- 补充固定近期轮数

Author: AGI Demo Project
Version: 4.0.0
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from dki.core.recall.recall_config import RecallConfig, RecallResult

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba not installed. Keyword recall disabled. pip install jieba")


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
    ) -> RecallResult:
        """
        执行多信号召回
        
        1. 指代解析 → 确定范围
        2. 关键词+权重检索
        3. 向量相似度检索
        4. 归一化融合排序
        5. 补充固定近期轮数
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
        if (self.config.signals.keyword_enabled
                and JIEBA_AVAILABLE
                and self._conversation_repo):
            keyword_scored = self._keyword_recall(
                query, session_id, db_session
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

        # ============ 4. 归一化融合排序 ============
        all_msg_ids = set(keyword_scored.keys()) | set(vector_scored.keys())

        # 归一化 keyword 分数 (sigmoid)
        norm_keyword = self._normalize_scores(keyword_scored)
        # 归一化 vector 分数 (已在 [0,1] 范围, 但 clip 确保)
        norm_vector = self._clip_scores(vector_scored)

        # 加权融合
        w = self.config.score_weights
        final_scores: Dict[str, float] = {}
        for msg_id in all_msg_ids:
            kw_score = norm_keyword.get(msg_id, 0.0)
            vec_score = norm_vector.get(msg_id, 0.0)
            final_scores[msg_id] = (
                w.keyword_weight * kw_score
                + w.vector_weight * vec_score
            )

        # 按 final_score 降序排序
        sorted_ids = sorted(
            final_scores.keys(),
            key=lambda x: final_scores[x],
            reverse=True,
        )[:max_results]

        result.scores = {mid: final_scores[mid] for mid in sorted_ids}

        # ============ 5. 获取完整消息对象 ============
        recalled_messages = []
        if sorted_ids and self._conversation_repo:
            try:
                recalled_messages = self._fetch_messages_by_ids(
                    sorted_ids, session_id, db_session
                )
            except Exception as e:
                logger.error(f"Failed to fetch recalled messages: {e}")

        # ============ 6. 补充近期轮次 ============
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

        # ============ 7. 合并 (近期优先, 去重) ============
        seen_ids = set()
        final_messages = []

        # 近期消息优先
        for msg in recent_messages:
            msg_id = getattr(msg, "id", None) or getattr(msg, "message_id", id(msg))
            msg_id = str(msg_id)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                final_messages.append(msg)
                # 添加时间近度 bonus
                if msg_id in final_scores:
                    final_scores[msg_id] += w.recency_weight

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
            f"Recall complete: kw={result.keyword_hits}, "
            f"vec={result.vector_hits}, "
            f"recent={result.recent_turns_added}, "
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
    ) -> Dict[str, float]:
        """关键词+权重检索"""
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
