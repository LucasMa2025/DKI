"""
SQLite Data Adapter for Experiment System

将现有 SQLite DatabaseManager + Repository 包装为 IUserDataAdapter，
使实验系统可以通过 DKIPlugin 运行 DKI 模式。

设计原则:
- 仅在 experiment 目录下，不修改 dki.database 模块
- 实现 IUserDataAdapter 接口的所有抽象方法
- 支持跨会话检索 (session_id=None)
- 同步操作包装为 async (SQLite 不需要真正的异步)

v7.2 更新:
- 实现 get_recent_messages() — 恢复近轮对话注入
- search_relevant_history() 升级到 jieba + BM25 — 提升检索质量
"""

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# jieba 分词 (可选依赖, 降级到 bigram)
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning(
        "jieba not installed, falling back to bigram tokenization. "
        "Install with: pip install jieba"
    )

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    ChatMessage,
    UserPreference,
)
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository,
    MemoryRepository,
    ConversationRepository,
    DemoUserRepository,
    UserPreferenceRepository,
)


class SQLiteDataAdapter(IUserDataAdapter):
    """
    SQLite 数据适配器 — 将现有 DKI 数据库包装为 IUserDataAdapter

    用途: 实验系统 (ExperimentRunner) 通过 DKIPlugin 运行 DKI 模式时，
    需要一个实现 IUserDataAdapter 接口的适配器来读取实验数据。

    数据流:
        ExperimentRunner
          → DKIPlugin(user_data_adapter=SQLiteDataAdapter)
            → SQLiteDataAdapter.get_user_preferences()  → UserPreferenceRepository
            → SQLiteDataAdapter.search_relevant_history() → ConversationRepository
            → SQLiteDataAdapter.get_session_history()     → ConversationRepository
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        keyword_top_k: int = 10,
    ):
        db_path = getattr(db_manager, '_db_path', getattr(db_manager, 'db_path', ':memory:'))
        config = AdapterConfig(
            adapter_type=AdapterType.SQLITE,
            connection_string=f"sqlite:///{db_path}",
        )
        super().__init__(config)
        self._db_manager = db_manager
        self._keyword_top_k = keyword_top_k

    # ============ 连接管理 (SQLite 无需真正的异步连接) ============

    async def connect(self) -> None:
        """SQLite 通过 DatabaseManager 已连接，标记为已连接。"""
        self._connected = True
        logger.debug("SQLiteDataAdapter: connected (SQLite via DatabaseManager)")

    async def disconnect(self) -> None:
        """SQLite 连接由 DatabaseManager 管理，此处仅标记。"""
        self._connected = False
        logger.debug("SQLiteDataAdapter: disconnected")

    async def health_check(self) -> bool:
        """检查 SQLite 连接健康。"""
        try:
            from sqlalchemy import text
            with self._db_manager.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ============ 用户画像 ============

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """从 demo_users 表获取用户画像。"""
        try:
            with self._db_manager.session_scope() as session:
                repo = DemoUserRepository(session)
                user = repo.get(user_id)
                if not user:
                    return None
                return UserProfile(
                    user_id=user.id,
                    username=user.username,
                    display_name=user.display_name,
                    email=user.email,
                    created_at=user.created_at,
                    is_active=user.is_active,
                )
        except Exception as e:
            logger.error(f"SQLiteDataAdapter.get_user_profile failed: {e}")
            return None

    # ============ 用户偏好 ============

    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """从 user_preferences 表获取用户偏好。"""
        try:
            with self._db_manager.session_scope() as session:
                repo = UserPreferenceRepository(session)
                db_prefs = repo.get_by_user(
                    user_id=user_id,
                    active_only=not include_expired,
                )

                # 类型过滤
                if preference_types:
                    db_prefs = [
                        p for p in db_prefs
                        if p.preference_type in preference_types
                    ]

                return [
                    UserPreference(
                        user_id=p.user_id,
                        preference_text=p.preference_text,
                        preference_type=p.preference_type,
                        preference_id=p.id,
                        priority=p.priority,
                        category=p.category,
                        created_at=p.created_at,
                        updated_at=p.updated_at,
                        is_active=p.is_active,
                    )
                    for p in db_prefs
                ]
        except Exception as e:
            logger.error(f"SQLiteDataAdapter.get_user_preferences failed: {e}")
            return []

    # ============ 会话历史 ============

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """从 conversations 表获取会话历史。"""
        try:
            with self._db_manager.session_scope() as session:
                repo = ConversationRepository(session)
                convs = repo.get_by_session(session_id, limit=limit)

                # 时间过滤
                if before:
                    convs = [c for c in convs if c.created_at and c.created_at < before]
                if after:
                    convs = [c for c in convs if c.created_at and c.created_at > after]

                return self._conversations_to_chat_messages(convs, session_id)
        except Exception as e:
            logger.error(f"SQLiteDataAdapter.get_session_history failed: {e}")
            return []

    # ============ 近轮对话获取 (v7.2: 恢复近轮对话注入) ============

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[ChatMessage]:
        """
        获取用户最近的消息 (跨会话, 按时间正序)

        v7.2 修复: DKIPlugin.chat() 在 Step 2 中调用此方法获取近轮对话,
        与 BM25 召回结果合并后注入到提示词后缀。

        之前 SQLiteDataAdapter 未实现此方法, 导致 DKIPlugin 始终拿到空列表,
        近轮对话注入被跳过, 严重低估了 DKI 的历史召回能力。

        实现策略:
        - 查询该 user_id 关联的所有 session 的 conversations
        - 按 created_at DESC 取最近 limit 条
        - 反转为时间正序 (最旧在前), 与 _merge_recent_and_recalled 的预期一致

        Args:
            user_id: 用户标识
            limit: 最大消息数 (建议 10-20, 即 5-10 轮对话)

        Returns:
            List[ChatMessage]: 按时间正序排列的近轮消息
        """
        try:
            with self._db_manager.session_scope() as db_session:
                from dki.database.models import Session as SessionModel, Conversation
                from sqlalchemy import desc

                # 跨会话: 先找该用户的所有 session_id
                user_sessions = (
                    db_session.query(SessionModel.id)
                    .filter(SessionModel.user_id == user_id)
                    .all()
                )
                session_ids = [s.id for s in user_sessions]

                if not session_ids:
                    return []

                # 获取这些 session 中最近的 limit 条消息 (时间降序)
                conversations = (
                    db_session.query(Conversation)
                    .filter(Conversation.session_id.in_(session_ids))
                    .order_by(desc(Conversation.created_at), desc(Conversation.id))
                    .limit(limit)
                    .all()
                )

                if not conversations:
                    return []

                # 反转为时间正序 (最旧在前)
                conversations = list(reversed(conversations))

                return self._conversations_to_chat_messages(
                    conversations,
                    session_id="cross_session",
                    user_id=user_id,
                )
        except Exception as e:
            logger.error(f"SQLiteDataAdapter.get_recent_messages failed: {e}")
            return []

    # ============ 相关历史检索 (核心: 仅搜索 conversations 表) ============

    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        检索与查询相关的历史对话消息 (v7.2: jieba + BM25)。

        当 session_id=None 时，执行跨会话检索 (跨该 user_id 的所有会话)。
        当 session_id 有值时，仅在该会话内检索。

        ============================================================
        重要: 此方法仅搜索 conversations 表 (历史对话)。
        ============================================================
        
        DKI 的提示词构造是两步分离的:
        
        Step 1 — 用户偏好 (提示词前缀注入):
            DKIPlugin.chat() → get_user_preferences() → preference_text
            → InjectionExecutor 放入 system message (显式文本前缀)
            偏好数据来源: user_preferences 表
        
        Step 2 — 历史召回 (多路融合):
            DKIPlugin.chat() → search_relevant_history() [本方法]
            → InjectionPlanner._build_suffix_only_plan() → history items
            历史数据来源: conversations 表
        
        v7.2 检索策略升级: jieba 分词 + BM25 评分
        - 使用 jieba 进行中文分词 (降级: bigram 滑窗)
        - 使用 BM25 (Okapi BM25) 算法替代简单关键词计数
        - BM25 考虑了词频 (TF)、逆文档频率 (IDF) 和文档长度归一化
        - 显著提升中文语义检索质量
        """
        try:
            with self._db_manager.session_scope() as db_session:
                conv_repo = ConversationRepository(db_session)

                # ============ 1. 获取候选对话消息 ============
                if session_id is None:
                    conversations = conv_repo.get_by_user_cross_session(
                        user_id=user_id,
                        current_session_id=None,
                        limit=200,
                    )
                else:
                    conversations = conv_repo.get_by_session(
                        session_id=session_id,
                        limit=100,
                    )

                if not conversations:
                    return []

                # ============ 2. 分词 ============
                query_tokens = self._tokenize(query)
                if not query_tokens:
                    recent = conversations[-limit:]
                    return self._conversations_to_chat_messages(
                        recent,
                        session_id=session_id or "cross_session",
                        user_id=user_id,
                    )

                # ============ 3. 构建文档语料库并计算 BM25 ============
                doc_tokens_list: List[List[str]] = []
                for conv in conversations:
                    doc_tokens_list.append(self._tokenize(conv.content))

                scored = self._bm25_score(
                    query_tokens=query_tokens,
                    doc_tokens_list=doc_tokens_list,
                )

                # ============ 4. 排序 + 近轮兜底 ============
                scored_with_conv = [
                    (conversations[i], score)
                    for i, score in enumerate(scored)
                    if score > 0.0
                ]

                if scored_with_conv:
                    scored_with_conv.sort(key=lambda x: x[1], reverse=True)
                    top_convs = [item[0] for item in scored_with_conv[:limit]]
                else:
                    top_convs = conversations[-limit:]

                # ============ 5. 转换为 ChatMessage ============
                results = self._conversations_to_chat_messages(
                    top_convs,
                    session_id=session_id or "cross_session",
                    user_id=user_id,
                )
                return results

        except Exception as e:
            logger.error(f"SQLiteDataAdapter.search_relevant_history failed: {e}")
            return []

    # ============ 写入方法 (实验系统需要写入记忆和对话) ============

    def add_memory(self, session_id: str, content: str, user_id: Optional[str] = None) -> str:
        """
        添加记忆 (同步方法，供实验系统使用)。

        在 sessions 表创建/获取 session，然后在 memories 表中创建记忆。
        """
        with self._db_manager.session_scope() as db_session:
            session_repo = SessionRepository(db_session)
            session_repo.get_or_create(session_id, user_id=user_id)

            memory_repo = MemoryRepository(db_session)
            memory = memory_repo.create(
                session_id=session_id,
                content=content,
            )
            return memory.id

    def add_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        injection_mode: Optional[str] = None,
        injection_alpha: Optional[float] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加对话记录 (同步方法，供实验系统使用)。

        用于 LongMemEval 历史播放: 直接写入 expected_response，
        而不是通过 chat() 生成低质量短响应。
        """
        with self._db_manager.session_scope() as db_session:
            session_repo = SessionRepository(db_session)
            session_repo.get_or_create(session_id, user_id=user_id)

            conv_repo = ConversationRepository(db_session)
            conv = conv_repo.create(
                session_id=session_id,
                role=role,
                content=content,
                injection_mode=injection_mode,
                injection_alpha=injection_alpha,
                latency_ms=latency_ms,
                metadata=metadata,
            )
            return conv.id

    # ============ 辅助方法 ============

    # ---- 停用词表 ----
    _EN_STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'shall',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'this', 'that', 'these', 'those',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'and', 'or', 'but', 'not', 'if', 'so', 'than', 'too', 'very',
        'just', 'about', 'also', 'more', 'some', 'any', 'only',
    })

    _CN_STOPWORDS = frozenset({
        '的', '了', '是', '在', '我', '你', '他', '她', '们', '有',
        '这', '那', '个', '也', '就', '都', '不', '吗', '呢', '吧',
        '啊', '啦', '呀', '请', '和', '与', '什么', '怎么', '哪',
        '哪里', '为什么', '还', '又', '被', '把', '让', '给', '从',
        '到', '对', '着', '过', '会', '能', '要', '想', '可以',
        '一', '二', '三', '上', '下', '中', '大', '小',
    })

    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词 (jieba 优先, 降级到 bigram)。

        返回去停用词后的 token 列表 (小写)。
        """
        tokens: List[str] = []

        if JIEBA_AVAILABLE:
            # jieba 精确模式分词
            words = jieba.lcut(text)
            for w in words:
                w_stripped = w.strip()
                if not w_stripped:
                    continue
                w_lower = w_stripped.lower()
                # 过滤停用词和单字符
                if w_lower in self._CN_STOPWORDS or w_lower in self._EN_STOPWORDS:
                    continue
                if len(w_lower) < 2:
                    continue
                tokens.append(w_lower)
        else:
            # 降级: 英文按空格分词 + 中文 bigram
            # 英文
            en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
            tokens.extend(
                w for w in en_words
                if w not in self._EN_STOPWORDS and len(w) > 1
            )
            # 中文 bigram
            cn_segments = re.findall(r'[\u4e00-\u9fff]+', text)
            for seg in cn_segments:
                filtered = ''.join(
                    c for c in seg if c not in self._CN_STOPWORDS
                )
                for i in range(len(filtered) - 1):
                    bigram = filtered[i:i + 2]
                    if len(bigram) == 2:
                        tokens.append(bigram)

        return tokens

    @staticmethod
    def _bm25_score(
        query_tokens: List[str],
        doc_tokens_list: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[float]:
        """
        Okapi BM25 评分。

        对每个文档计算 BM25 分数, 返回与 doc_tokens_list 等长的分数列表。

        BM25(D, Q) = Σ_{q ∈ Q} IDF(q) · (tf(q,D) · (k1+1)) / (tf(q,D) + k1 · (1 - b + b · |D|/avgdl))

        Args:
            query_tokens: 查询分词结果
            doc_tokens_list: 文档分词结果列表
            k1: 词频饱和参数 (默认 1.5)
            b: 文档长度归一化参数 (默认 0.75)

        Returns:
            List[float]: 每个文档的 BM25 分数
        """
        n_docs = len(doc_tokens_list)
        if n_docs == 0:
            return []

        # 计算平均文档长度
        doc_lengths = [len(dt) for dt in doc_tokens_list]
        avgdl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0

        # 计算每个 query token 的 IDF
        # IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        query_token_set = set(query_tokens)
        df: Dict[str, int] = {}  # document frequency
        for dt in doc_tokens_list:
            seen_in_doc = set(dt)
            for token in query_token_set:
                if token in seen_in_doc:
                    df[token] = df.get(token, 0) + 1

        idf: Dict[str, float] = {}
        for token in query_token_set:
            n_q = df.get(token, 0)
            idf[token] = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

        # 计算每个文档的 BM25 分数
        scores: List[float] = []
        for i, doc_tokens in enumerate(doc_tokens_list):
            dl = doc_lengths[i]
            tf_counter = Counter(doc_tokens)
            score = 0.0
            for q_token in query_tokens:
                tf_val = tf_counter.get(q_token, 0)
                if tf_val == 0:
                    continue
                numerator = tf_val * (k1 + 1.0)
                denominator = tf_val + k1 * (1.0 - b + b * dl / avgdl)
                score += idf.get(q_token, 0.0) * numerator / denominator
            scores.append(score)

        return scores

    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词 (向后兼容, 内部使用 _tokenize)。

        保留此方法以兼容可能的外部调用。
        """
        return self._tokenize(text)

    def _conversations_to_chat_messages(
        self,
        conversations,
        session_id: str = "",
        user_id: str = "",
    ) -> List[ChatMessage]:
        """将 Conversation ORM 对象转换为 ChatMessage 数据结构。"""
        messages = []
        for conv in conversations:
            # 从 session 获取 user_id (如果可用)
            conv_user_id = user_id
            if hasattr(conv, 'session') and conv.session and conv.session.user_id:
                conv_user_id = conv.session.user_id

            messages.append(ChatMessage(
                message_id=conv.id,
                session_id=conv.session_id or session_id,
                user_id=conv_user_id,
                role=conv.role,
                content=conv.content,
                timestamp=conv.created_at or datetime.utcnow(),
                metadata=conv.get_metadata() if hasattr(conv, 'get_metadata') else {},
            ))
        return messages

    def __repr__(self) -> str:
        return f"SQLiteDataAdapter(db={self._db_manager._db_path}, connected={self._connected})"
