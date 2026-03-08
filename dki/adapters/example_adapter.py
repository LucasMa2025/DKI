"""
Example Adapter - 示例适配器

用于 DKI 示例 Chat UI 的内存适配器
实现 IUserDataAdapter 接口，同时提供写入方法供 demo 使用

注意:
- 这是 demo/dev 环境使用的适配器
- 数据存储在内存中，重启后丢失
- 检索算法与 ConfigDrivenAdapter 保持一致 (BM25 + jieba 中文分词)
- 生产环境应使用 ConfigDrivenAdapter 连接真实数据库

Author: AGI Demo Project
Version: 3.0.0 (v7.1: BM25+jieba, 缓存, 跨会话检索)
"""

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    UserPreference,
    ChatMessage,
)


@dataclass
class ExampleDataStore:
    """
    示例数据存储
    
    模拟上层应用的数据库
    """
    # 用户表
    users: Dict[str, UserProfile] = field(default_factory=dict)
    
    # 偏好表
    preferences: Dict[str, List[UserPreference]] = field(default_factory=dict)
    
    # 消息表 (key=session_id, value=messages)
    messages: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    
    # 会话表
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ExampleAdapter(IUserDataAdapter):
    """
    示例适配器 (v3.0)
    
    用于 DKI 示例 Chat UI，实现 IUserDataAdapter 接口
    
    v3.0 改进:
    - search_relevant_history: BM25 + jieba 中文分词 (与 ConfigDrivenAdapter 一致)
    - get_user_preferences: 添加内存缓存 (TTL 可配)
    - get_session_history: 完善时间过滤和排序
    - 新增 clear_cache / invalidate_user_cache 方法
    - 新增 get_all_user_messages 跨会话消息收集
    
    使用方式:
    ```python
    adapter = ExampleAdapter()
    await adapter.connect()
    
    # 添加示例数据 (模拟上层应用写入)
    adapter.add_user("user_001", "张三")
    adapter.add_preference("user_001", "dietary", "素食主义者，不吃辣")
    adapter.add_message("session_001", "user_001", "user", "推荐一家餐厅")
    
    # DKI 读取数据
    preferences = await adapter.get_user_preferences("user_001")
    history = await adapter.search_relevant_history("user_001", "餐厅")
    ```
    """
    
    # ============ BM25 中文停用词表 (与 ConfigDrivenAdapter 一致) ============
    _CN_STOPWORDS = frozenset({
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        '什么', '吗', '呢', '吧', '啊', '哦', '嗯', '呀', '哈', '哪', '嘛',
        '可以', '没', '还', '对', '把', '让', '被', '从', '给', '用', '但',
        '而', '又', '所以', '因为', '如果', '这个', '那个', '怎么', '为什么',
        '哪个', '多少', '几', '谁', '怎样', '这样', '那样',
    })
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config or AdapterConfig(adapter_type=AdapterType.MEMORY))
        
        # 内存数据存储 (模拟上层应用的数据库)
        self._store = ExampleDataStore()
        
        # ============ 缓存 (与 ConfigDrivenAdapter 对齐) ============
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl: int = config.cache_ttl if config else 300  # 默认 5 分钟
        self._cache_enabled: bool = config.enable_cache if config else True
        
        # ============ jieba 可用性检测 (启动时一次) ============
        self._jieba_available = False
        try:
            import jieba  # noqa: F401
            self._jieba_available = True
        except ImportError:
            logger.warning(
                "jieba not installed, falling back to char+bigram tokenization. "
                "Install with: pip install jieba"
            )
    
    # ============ 连接管理 ============
    
    async def connect(self) -> None:
        """连接 (内存适配器无需实际连接)"""
        self._connected = True
        logger.info("ExampleAdapter connected (in-memory)")
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        self.clear_cache()
        logger.info("ExampleAdapter disconnected")
    
    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected
    
    # ============ 读取接口 (DKI 调用) ============
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self._store.users.get(user_id)
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """
        获取用户偏好 (带缓存)
        
        与 ConfigDrivenAdapter 对齐:
        - 支持 preference_types 过滤
        - 支持 include_expired 控制
        - 按优先级降序排序
        - 内存缓存 (TTL 可配)
        """
        # 检查缓存
        cache_key = f"prefs:{user_id}:{preference_types}:{include_expired}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        preferences = self._store.preferences.get(user_id, [])
        
        # 过滤
        result = []
        for pref in preferences:
            if not pref.is_active:
                continue
            if not include_expired and pref.is_expired():
                continue
            if preference_types and pref.preference_type not in preference_types:
                continue
            result.append(pref)
        
        # 按优先级排序
        result.sort(key=lambda p: p.priority, reverse=True)
        
        # 写入缓存
        self._set_cached(cache_key, result)
        
        return result
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """
        获取会话历史
        
        与 ConfigDrivenAdapter 对齐:
        - 支持 before/after 时间过滤
        - 按时间正序返回 (最早在前)
        - limit 限制返回数量 (取最近的 N 条)
        """
        messages = self._store.messages.get(session_id, [])
        
        # 时间过滤
        result = []
        for msg in messages:
            if before and msg.timestamp and msg.timestamp >= before:
                continue
            if after and msg.timestamp and msg.timestamp <= after:
                continue
            result.append(msg)
        
        # 按时间正序排序
        result.sort(key=lambda m: m.timestamp if m.timestamp else datetime.min)
        
        # 取最近的 limit 条 (保持时间正序)
        return result[-limit:]
    
    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[ChatMessage]:
        """
        获取用户最近的消息 (跨会话, 按时间正序)
        """
        all_messages = []
        for session_msgs in self._store.messages.values():
            for msg in session_msgs:
                if msg.user_id == user_id or msg.role == "assistant":
                    all_messages.append(msg)
        
        # 按时间降序排序, 取最近 limit 条
        all_messages.sort(
            key=lambda m: m.timestamp if m.timestamp else datetime.min,
            reverse=True,
        )
        recent = all_messages[:limit]
        
        # 反转为时间正序 (最旧在前)
        recent.reverse()
        return recent
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        检索相关历史 (v7.1: BM25 + jieba 中文分词)
        
        与 ConfigDrivenAdapter._search_with_bm25_only 算法完全一致:
        - BM25 评分 (TF-IDF 加权, 文档长度归一化)
        - 优先使用 jieba 分词 (中文语义分词)
        - 过滤停用词 (避免高频词稀释权重)
        - 回退: jieba 不可用时使用单字+bigram
        - 跨会话检索 (session_id=None 时搜索用户所有会话)
        - score=0 过滤 + 最近消息兜底
        """
        # 收集消息
        all_messages = self._collect_user_messages(user_id, session_id)
        
        if not all_messages:
            return []
        
        # BM25 评分
        scored_messages = self._bm25_score(query, all_messages)
        
        # 过滤 score=0, 按分数降序排序
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        relevant = [(msg, score) for msg, score in scored_messages if score > 0]
        
        if relevant:
            logger.debug(
                f"ExampleAdapter BM25 recall: {len(relevant)} messages with score > 0 "
                f"(top score={relevant[0][1]:.3f})"
            )
            return [msg for msg, score in relevant[:limit]]
        
        # BM25 无结果时回退: 返回最近的消息
        fallback_count = min(limit, 5)
        logger.info(
            f"ExampleAdapter BM25: no messages scored > 0 for '{query[:50]}', "
            f"falling back to {fallback_count} most recent messages"
        )
        all_messages.sort(
            key=lambda m: m.timestamp if m.timestamp else datetime.min,
            reverse=True,
        )
        return all_messages[:fallback_count]
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        获取用户会话列表
        
        返回按更新时间倒序排列的会话列表
        """
        sessions = []
        for session in self._store.sessions.values():
            if session.get("user_id") != user_id:
                continue
            if active_only and not session.get("is_active", True):
                continue
            sessions.append(session)
        
        # 按更新时间排序
        sessions.sort(
            key=lambda s: s.get("updated_at", datetime.min),
            reverse=True,
        )
        return sessions[:limit]
    
    # ============ 消息收集 (内部方法) ============
    
    def _collect_user_messages(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[ChatMessage]:
        """
        收集用户消息 (与 ConfigDrivenAdapter._get_user_messages 对齐)
        
        Args:
            user_id: 用户 ID
            session_id: 可选会话 ID (None 表示跨会话)
            limit: 最大消息数
            
        Returns:
            消息列表 (按时间倒序)
        """
        all_messages: List[ChatMessage] = []
        
        if session_id:
            # 仅搜索指定会话
            all_messages = list(self._store.messages.get(session_id, []))
        else:
            # 搜索用户的所有会话 (跨会话检索)
            for sid, messages in self._store.messages.items():
                for msg in messages:
                    if msg.user_id == user_id:
                        all_messages.append(msg)
        
        # 按时间倒序排序 (最近的在前, 与 ConfigDrivenAdapter 一致)
        all_messages.sort(
            key=lambda m: m.timestamp if m.timestamp else datetime.min,
            reverse=True,
        )
        
        return all_messages[:limit]
    
    # ============ BM25 分词与评分 (与 ConfigDrivenAdapter 完全一致) ============
    
    def _tokenize(self, text: str) -> List[str]:
        """
        中英文混合分词 (v7.1)
        
        与 ConfigDrivenAdapter._bm25_score 内部 tokenize 完全一致:
        - 有 jieba: jieba 分词 + 英文单词 + 停用词过滤
        - 无 jieba: 单字 + bigram + 英文单词 + 停用词过滤
        """
        tokens: List[str] = []
        text_lower = text.lower()
        
        # 英文单词
        en_tokens = re.findall(r'[a-zA-Z0-9]+', text_lower)
        tokens.extend(en_tokens)
        
        if self._jieba_available:
            import jieba
            # jieba 分词: 产出有语义的词组 (如 "村上春树", "挪威", "森林")
            cn_text = re.sub(r'[a-zA-Z0-9]+', ' ', text_lower)
            words = jieba.lcut(cn_text)
            for w in words:
                w = w.strip()
                if len(w) >= 1 and any('\u4e00' <= c <= '\u9fff' for c in w):
                    if w not in self._CN_STOPWORDS:
                        tokens.append(w)
        else:
            # 回退: 单字 + bigram (过滤停用词)
            cn_chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
            for i in range(len(cn_chars)):
                if cn_chars[i] not in self._CN_STOPWORDS:
                    tokens.append(cn_chars[i])
                if i + 1 < len(cn_chars):
                    bigram = cn_chars[i] + cn_chars[i + 1]
                    if bigram not in self._CN_STOPWORDS:
                        tokens.append(bigram)
        
        return tokens
    
    def _bm25_score(
        self,
        query: str,
        messages: List[ChatMessage],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Tuple[ChatMessage, float]]:
        """
        BM25 评分 (与 ConfigDrivenAdapter._bm25_score 算法完全一致)
        
        参数:
        - k1: 词频饱和参数 (默认 1.5)
        - b: 文档长度归一化参数 (默认 0.75)
        
        返回: [(message, score), ...]
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(msg, 0.0) for msg in messages]
        
        # 文档分词
        doc_tokens_list = [self._tokenize(msg.content) for msg in messages]
        
        # 计算平均文档长度
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)
        
        # 计算 IDF
        N = len(messages)
        idf: Dict[str, float] = {}
        for qt in set(query_tokens):
            df = sum(1 for dt in doc_tokens_list if qt in dt)
            idf[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        # 计算每个文档的 BM25 分数
        results: List[Tuple[ChatMessage, float]] = []
        for msg, doc_tokens in zip(messages, doc_tokens_list):
            score = 0.0
            dl = len(doc_tokens)
            
            # 词频统计
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                score += idf.get(qt, 0) * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
                )
            
            results.append((msg, score))
        
        return results
    
    # ============ 缓存方法 (与 ConfigDrivenAdapter 对齐) ============
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self._cache_enabled:
            return None
        
        if key not in self._cache:
            return None
        
        # 检查过期
        timestamp = self._cache_timestamps.get(key)
        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self._cache_ttl:
                del self._cache[key]
                del self._cache_timestamps[key]
                return None
        
        return self._cache[key]
    
    def _set_cached(self, key: str, value: Any) -> None:
        """设置缓存"""
        if not self._cache_enabled:
            return
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.utcnow()
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def invalidate_user_cache(self, user_id: str) -> int:
        """
        清除指定用户的缓存
        
        当用户偏好变更时调用，确保下次读取获取最新数据
        与 ConfigDrivenAdapter 的缓存失效策略对齐
        
        Returns:
            清除的缓存条目数
        """
        keys_to_remove = [k for k in self._cache if k.startswith(f"prefs:{user_id}:")]
        for k in keys_to_remove:
            del self._cache[k]
            self._cache_timestamps.pop(k, None)
        return len(keys_to_remove)
    
    # ============ 写入接口 (Demo 专用, 非 DKI 职责) ============
    
    def add_user(
        self,
        user_id: str,
        username: str,
        display_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        添加用户 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        user = UserProfile(
            user_id=user_id,
            username=username,
            display_name=display_name or username,
            preferences=preferences or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._store.users[user_id] = user
        return user
    
    def add_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_text: str,
        priority: int = 0,
        category: Optional[str] = None,
    ) -> UserPreference:
        """
        添加用户偏好 (模拟上层应用写入)
        
        注意: 添加后自动清除该用户的偏好缓存
        """
        pref = UserPreference(
            user_id=user_id,
            preference_id=uuid.uuid4().hex[:8],
            preference_type=preference_type,
            preference_text=preference_text,
            priority=priority,
            category=category,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        if user_id not in self._store.preferences:
            self._store.preferences[user_id] = []
        self._store.preferences[user_id].append(pref)
        
        # 缓存失效: 偏好变更后清除缓存
        self.invalidate_user_cache(user_id)
        
        return pref
    
    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        添加消息 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        msg = ChatMessage(
            message_id=uuid.uuid4().hex[:8],
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        
        if session_id not in self._store.messages:
            self._store.messages[session_id] = []
        self._store.messages[session_id].append(msg)
        
        # 更新会话
        if session_id not in self._store.sessions:
            self._store.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "message_count": 0,
            }
        self._store.sessions[session_id]["message_count"] += 1
        self._store.sessions[session_id]["updated_at"] = datetime.utcnow()
        
        return msg
    
    def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建会话 (模拟上层应用写入)
        
        注意: 这是上层应用的操作，不是 DKI 的职责
        """
        sid = session_id or uuid.uuid4().hex[:8]
        session = {
            "session_id": sid,
            "user_id": user_id,
            "title": title or f"会话 {sid}",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0,
            "is_active": True,
        }
        self._store.sessions[sid] = session
        self._store.messages[sid] = []
        return session
    
    def update_preference(
        self,
        user_id: str,
        preference_id: str,
        preference_text: Optional[str] = None,
        priority: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """
        更新偏好 (模拟上层应用写入)
        
        注意: 更新后自动清除该用户的偏好缓存
        """
        preferences = self._store.preferences.get(user_id, [])
        for pref in preferences:
            if pref.preference_id == preference_id:
                if preference_text is not None:
                    pref.preference_text = preference_text
                if priority is not None:
                    pref.priority = priority
                if is_active is not None:
                    pref.is_active = is_active
                pref.updated_at = datetime.utcnow()
                # 缓存失效
                self.invalidate_user_cache(user_id)
                return True
        return False
    
    def delete_preference(self, user_id: str, preference_id: str) -> bool:
        """
        删除偏好 (模拟上层应用写入)
        
        注意: 删除后自动清除该用户的偏好缓存
        """
        preferences = self._store.preferences.get(user_id, [])
        for i, pref in enumerate(preferences):
            if pref.preference_id == preference_id:
                preferences.pop(i)
                # 缓存失效
                self.invalidate_user_cache(user_id)
                return True
        return False
    
    # ============ 统计与调试 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据 (与 ConfigDrivenAdapter.get_stats 对齐)"""
        total_messages = sum(len(msgs) for msgs in self._store.messages.values())
        total_preferences = sum(len(prefs) for prefs in self._store.preferences.values())
        
        return {
            "connected": self._connected,
            "adapter_type": "memory",
            "users_count": len(self._store.users),
            "sessions_count": len(self._store.sessions),
            "messages_count": total_messages,
            "preferences_count": total_preferences,
            "cache_size": len(self._cache),
            "cache_enabled": self._cache_enabled,
            "jieba_available": self._jieba_available,
        }
    
    def clear(self):
        """清空所有数据和缓存"""
        self._store = ExampleDataStore()
        self.clear_cache()