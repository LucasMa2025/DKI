"""
SQLite Data Adapter for Experiment System

将现有 SQLite DatabaseManager + Repository 包装为 IUserDataAdapter，
使实验系统可以通过 DKIPlugin 运行 DKI 模式。

设计原则:
- 仅在 experiment 目录下，不修改 dki.database 模块
- 实现 IUserDataAdapter 接口的所有抽象方法
- 支持跨会话检索 (session_id=None)
- 同步操作包装为 async (SQLite 不需要真正的异步)
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

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

    # ============ 相关历史检索 (核心: 仅搜索 conversations 表) ============

    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        检索与查询相关的历史对话消息。

        当 session_id=None 时，执行跨会话检索 (跨该 user_id 的所有会话)。
        当 session_id 有值时，仅在该会话内检索。

        ============================================================
        重要: 此方法仅搜索 conversations 表 (历史对话)。
        ============================================================
        
        DKI 的提示词构造是两步分离的:
        
        Step 1 — 用户偏好 (提示词前缀注入):
            DKIPlugin.chat() → get_user_preferences() → preference_text
            → InjectionExecutor 放入 system message (显式文本前缀)
            (vLLM 环境下为 prompt prefix, 非 past_key_value 注入)
            偏好数据来源: user_preferences 表
        
        Step 2 — 历史召回 (多路融合):
            DKIPlugin.chat() → search_relevant_history() [本方法]
            → InjectionPlanner._build_suffix_only_plan() → history items
            历史数据来源: conversations 表
        
        persona/记忆信息存储在 memories 表中，但在实验系统中，
        这些信息已通过 _write_session_preferences() 同步写入
        user_preferences 表，由 get_user_preferences() 路径处理。
        
        如果将 memories 表数据混入此方法，会导致 persona 同时出现在
        preferences (system message) 和 history (suffix) 两个通道中，
        这违反了 DKI 的两步分离原则。
        
        检索策略: 关键词匹配 (SQLite 不支持向量检索)
        - 从 query 中提取关键词
        - 在 conversations 表中搜索包含这些关键词的消息
        - 按匹配度排序，返回 top-k
        """
        try:
            with self._db_manager.session_scope() as db_session:
                conv_repo = ConversationRepository(db_session)

                # ============ 1. 获取候选对话消息 ============
                if session_id is None:
                    # 跨会话检索: 获取该用户所有会话的历史
                    conversations = conv_repo.get_by_user_cross_session(
                        user_id=user_id,
                        current_session_id=None,
                        limit=200,  # 获取较多消息用于过滤
                    )
                else:
                    # 单会话检索
                    conversations = conv_repo.get_by_session(
                        session_id=session_id,
                        limit=100,
                    )

                if not conversations:
                    return []

                # ============ 2. 关键词提取 ============
                keywords = self._extract_keywords(query)
                if not keywords:
                    # 无法提取关键词时，返回最近的消息 (近轮兜底)
                    recent = conversations[-limit:]
                    return self._conversations_to_chat_messages(
                        recent,
                        session_id=session_id or "cross_session",
                        user_id=user_id,
                    )

                # ============ 3. 关键词匹配评分 ============
                scored = []
                for conv in conversations:
                    content_lower = conv.content.lower()
                    score = sum(1 for kw in keywords if kw in content_lower)
                    if score > 0:
                        scored.append((conv, score))

                # ============ 4. 排序 + 近轮兜底 ============
                if scored:
                    # 按匹配度降序排序
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top_convs = [item[0] for item in scored[:limit]]
                else:
                    # 关键词匹配无结果时，返回最近消息 (近轮兜底)
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

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词 (支持中文 bigram 滑窗分词)。"""
        # 英文: 按空格/标点分词
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        en_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
            'this', 'that', 'these', 'those',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'and', 'or', 'but', 'not', 'if', 'so', 'than', 'too', 'very',
        }
        en_keywords = [w for w in en_words if w not in en_stopwords and len(w) > 1]

        # 中文: 提取连续中文段落, bigram 滑窗分词
        cn_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        cn_stopchars = set('的了是在我你他她们有这那个也就都不吗呢吧啊啦呀请和与什么怎么哪哪里为什么')
        cn_keywords = []
        for seg in cn_segments:
            filtered = ''.join(c for c in seg if c not in cn_stopchars)
            for i in range(len(filtered) - 1):
                bigram = filtered[i:i+2]
                if len(bigram) == 2:
                    cn_keywords.append(bigram)

        # 去重保序
        seen = set()
        unique = []
        for kw in en_keywords + cn_keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique

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
