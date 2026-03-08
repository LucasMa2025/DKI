"""
RAG System for DKI
Retrieval-Augmented Generation implementation as baseline

v6.0 更新:
- 增加异步 chat (async_chat) 和流式 chat (chat_stream)
- 增加偏好缓存 (AsyncSingleFlight + TTL)
- 细化异常处理 (使用 DKI 结构化异常)
- 从提示词构造中移除 <think> 推理内容
- 使用新的 token 预算分配 (30% 生成预留)
- 历史轮次从外置配置读取 (与 DKI 一致)
- 使用快速 token 估算 (estimate_tokens_fast)
- 存储响应前移除 think 内容
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from loguru import logger

from dki.core.text_utils import (
    strip_think_content, estimate_tokens_fast,
    detect_vague_reference, build_clarification_instruction,
)
from dki.core.memory_router import MemoryRouter, MemorySearchResult
from dki.core.embedding_service import EmbeddingService
from dki.core.exceptions import (
    DKIError,
    AdapterConnectionError,
    AdapterTimeoutError,
    ModelError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelOOMError,
    VectorSearchError,
)
from dki.models.factory import ModelFactory
from dki.models.base import BaseModelAdapter, ModelOutput
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository,
    UserPreferenceRepository,
)
from dki.config.config_loader import ConfigLoader


# ============================================================
# RAG 专用异常
# ============================================================

class RAGError(DKIError):
    """RAG 系统基础异常"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="RAG_ERROR", retryable=False, cause=cause)


class RAGMemorySearchError(RAGError):
    """RAG 记忆检索失败"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause=cause)
        self.error_code = "RAG_MEMORY_SEARCH"
        self.retryable = True


class RAGHistoryError(RAGError):
    """RAG 历史加载失败 (非致命, 可降级)"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause=cause)
        self.error_code = "RAG_HISTORY"
        self.retryable = False


class RAGPreferenceError(RAGError):
    """RAG 偏好加载失败 (非致命, 可降级)"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause=cause)
        self.error_code = "RAG_PREFERENCE"
        self.retryable = False


class RAGPromptBuildError(RAGError):
    """RAG 提示词构造失败"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause=cause)
        self.error_code = "RAG_PROMPT_BUILD"
        self.retryable = False


class RAGGenerationError(RAGError):
    """RAG 生成失败"""
    def __init__(self, message: str, retryable: bool = False, cause: Optional[Exception] = None):
        super().__init__(message, cause=cause)
        self.error_code = "RAG_GENERATION"
        self.retryable = retryable


@dataclass
class RAGPromptInfo:
    """RAG 提示词构造信息 - 用于显示"""
    original_query: str = ""
    system_prompt: str = ""
    retrieved_context: str = ""  # 检索到的上下文
    history_text: str = ""  # 历史对话文本
    history_messages: List[Dict[str, str]] = None  # 历史消息列表
    final_prompt: str = ""  # 最终构造的提示词
    
    def __post_init__(self):
        if self.history_messages is None:
            self.history_messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_query': self.original_query,
            'system_prompt': self.system_prompt,
            'retrieved_context': self.retrieved_context,
            'history_text': self.history_text,
            'history_messages': self.history_messages,
            'final_prompt': self.final_prompt,
        }


@dataclass
class RAGResponse:
    """RAG system response."""
    text: str
    memories_used: List[MemorySearchResult]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = None
    # 新增: 提示词构造信息
    prompt_info: Optional[RAGPromptInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'memories_used': [m.to_dict() for m in self.memories_used],
            'latency_ms': self.latency_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'metadata': self.metadata or {},
            'prompt_info': self.prompt_info.to_dict() if self.prompt_info else None,
        }


class RAGSystem:
    """
    Retrieval-Augmented Generation System.
    
    Implements the standard RAG paradigm:
    1. Retrieve relevant memories
    2. Concatenate to prompt
    3. Generate response
    
    v6.0 新增:
    - async_chat(): 异步版本的 chat
    - chat_stream(): 流式生成 (SSE 兼容)
    - 偏好缓存 (TTL + AsyncSingleFlight 防惊群)
    - 细化异常处理 (RAGError 层次)
    """
    
    # 偏好缓存 TTL (秒)
    _PREFERENCE_CACHE_TTL: float = 300.0  # 5 分钟
    
    def __init__(
        self,
        model_adapter: Optional[BaseModelAdapter] = None,
        memory_router: Optional[MemoryRouter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        engine: Optional[str] = None,
        preference_cache_ttl: Optional[float] = None,
    ):
        self.config = ConfigLoader().config
        
        # Initialize components
        self.embedding_service = embedding_service or EmbeddingService()
        self.memory_router = memory_router or MemoryRouter(self.embedding_service)
        
        # Model adapter (lazy loaded)
        self._model_adapter = model_adapter
        self._engine = engine
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        # v6.0: 偏好缓存 (TTL + SingleFlight)
        self._preference_cache: Dict[str, Tuple[Optional[str], float]] = {}
        self._preference_cache_ttl = preference_cache_ttl or self._PREFERENCE_CACHE_TTL
        self._preference_single_flight: Dict[str, asyncio.Future] = {}
        
        # 统计
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "async_requests": 0,
            "stream_requests": 0,
            "preference_cache_hits": 0,
            "preference_cache_misses": 0,
            "errors": 0,
        }
        
        logger.info("RAG System initialized (v6.0: async + streaming + preference cache)")
    
    @property
    def model(self) -> BaseModelAdapter:
        """Get or create model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelFactory.get_or_create(engine=self._engine)
        return self._model_adapter
    
    def add_memory(
        self,
        session_id: str,
        content: str,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_db: bool = False,
    ) -> str:
        """
        Add a memory to the system.
        
        Args:
            session_id: Session identifier
            content: Memory content
            memory_id: Optional memory ID
            metadata: Optional metadata
            skip_db: If True, skip database insertion (memory already in DB,
                      only add to in-memory router). This avoids UNIQUE constraint
                      violations when DKI and RAG share the same database.
            
        Returns:
            Memory ID
        """
        # Compute embedding
        embedding = self.embedding_service.embed(content)
        
        if not skip_db:
            # Store in database
            with self.db_manager.session_scope() as db:
                session_repo = SessionRepository(db)
                memory_repo = MemoryRepository(db)
                
                # Ensure session exists
                session_repo.get_or_create(session_id)
                
                # Check if memory already exists (avoid UNIQUE constraint violation)
                if memory_id:
                    existing = memory_repo.get(memory_id)
                    if existing:
                        logger.debug(f"Memory {memory_id} already exists in DB, skipping insert")
                        memory_id = existing.id
                    else:
                        memory = memory_repo.create(
                            session_id=session_id,
                            content=content,
                            embedding=embedding,
                            memory_id=memory_id,
                            metadata=metadata,
                        )
                        memory_id = memory.id
                else:
                    memory = memory_repo.create(
                        session_id=session_id,
                        content=content,
                        embedding=embedding,
                        memory_id=memory_id,
                        metadata=metadata,
                    )
                    memory_id = memory.id
        else:
            # skip_db mode: memory_id must be provided
            if not memory_id:
                from dki.database.repository import BaseRepository
                memory_id = BaseRepository.generate_id("mem_")
        
        # Add to router (in-memory index)
        self.memory_router.add_memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
        
        logger.debug(f"Added memory: {memory_id} (skip_db={skip_db})")
        return memory_id
    
    def load_memories_from_db(self, session_id: str) -> int:
        """
        Load memories from database into router.
        
        Args:
            session_id: Session to load memories for
            
        Returns:
            Number of memories loaded
        """
        with self.db_manager.session_scope() as db:
            memory_repo = MemoryRepository(db)
            memories = memory_repo.get_by_session(session_id)
            
            count = 0
            for mem in memories:
                embedding = memory_repo.get_embedding(mem.id)
                self.memory_router.add_memory(
                    memory_id=mem.id,
                    content=mem.content,
                    embedding=embedding,
                    metadata=mem.get_metadata(),
                )
                count += 1
        
        logger.info(f"Loaded {count} memories for session {session_id}")
        return count
    
    def _estimate_tokens(self, text: str) -> int:
        """
        v5.7: 使用快速估算 (不依赖 tokenizer, 略微高估 15%)
        与 DKI 系统使用相同的估算方法, 确保对比公平
        """
        return estimate_tokens_fast(text, overestimate_factor=1.15)
    
    def _get_max_context_length(self) -> int:
        """获取模型最大上下文长度"""
        if self.model:
            if hasattr(self.model, 'max_model_len'):
                return self.model.max_model_len
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer:
                try:
                    return self.model.tokenizer.model_max_length
                except Exception:
                    pass
        return 4096  # 默认安全长度
    
    def _build_prompt(
        self,
        query: str,
        memories: List[MemorySearchResult],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """
        Build prompt with retrieved memories and conversation history.
        
        使用 tokenizer.apply_chat_template 构造符合各模型官方标准的 chat 格式:
        - DeepSeek/Qwen: <|im_start|>system/user/assistant<|im_end|>
        - Llama 3.x:     <|begin_of_text|><|start_header_id|>...<|end_header_id|>
        - GLM:           GLM 原生 chat template
        - 其他模型:      tokenizer 内置的 chat template
        
        如果 tokenizer 不支持 apply_chat_template, 则回退到通用格式。
        
        Includes automatic truncation to prevent exceeding model context length.
        
        Args:
            query: User query
            memories: Retrieved memories
            system_prompt: Optional system prompt
            history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Tuple of (formatted_prompt, RAGPromptInfo)
        """
        # v5.7: 获取模型最大上下文长度, 生成预留 = 30% 上下文
        max_context = self._get_max_context_length()
        generation_reserve = int(max_context * 0.30)
        tag_overhead = 120  # chat template 标记开销
        max_prompt_tokens = max_context - generation_reserve - tag_overhead
        
        # 用于记录的信息
        prompt_info = RAGPromptInfo(
            original_query=query,
            system_prompt=system_prompt or "",
            history_messages=history or [],
        )
        
        # === 1. 构建 retrieved context (作为 system prompt 的一部分) ===
        context_parts = []
        context_text = ""
        if memories:
            for i, mem in enumerate(memories, 1):
                line = f"[{i}] {mem.content}"
                context_parts.append(line)
            context_text = "\n".join(context_parts)
        
        prompt_info.retrieved_context = context_text
        
        # === 2. 构建 system prompt (含检索到的上下文) ===
        full_system_prompt = ""
        if system_prompt and context_text:
            full_system_prompt = (
                f"{system_prompt}\n\n"
                f"Relevant information:\n{context_text}"
            )
        elif system_prompt:
            full_system_prompt = system_prompt
        elif context_text:
            full_system_prompt = f"Relevant information:\n{context_text}"
        
        # === 3. 构建 conversation history (截断最旧的) ===
        history_parts = []
        selected_history_msgs = []
        if history:
            # v5.7: 移除历史消息中的 <think> 推理内容
            cleaned_history = []
            for msg in history:
                cleaned_msg = dict(msg)  # 浅拷贝
                if cleaned_msg.get('role') == 'assistant' and cleaned_msg.get('content'):
                    cleaned_content, _ = strip_think_content(cleaned_msg['content'])
                    if cleaned_content and cleaned_content.strip():
                        cleaned_msg['content'] = cleaned_content
                    else:
                        continue  # 清理后为空, 跳过
                cleaned_history.append(cleaned_msg)
            
            # 粗估可用 token 预算 (直接估算, 不预留)
            system_tokens = self._estimate_tokens(full_system_prompt) if full_system_prompt else 0
            query_tokens = self._estimate_tokens(query)
            remaining_tokens = max_prompt_tokens - system_tokens - query_tokens - 40
            
            # 从最新开始，保留尽可能多的历史
            used_tokens = 0
            for msg in reversed(cleaned_history):
                msg_tokens = self._estimate_tokens(msg['content']) + 8  # +8 for role tags
                if used_tokens + msg_tokens > remaining_tokens:
                    break
                selected_history_msgs.insert(0, msg)
                used_tokens += msg_tokens
            
            if selected_history_msgs:
                for msg in selected_history_msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg['content']}")
            
            if len(selected_history_msgs) < len(cleaned_history):
                logger.info(
                    f"RAG prompt truncated: kept {len(selected_history_msgs)}/{len(cleaned_history)} "
                    f"history messages to fit model context ({max_context} tokens)"
                )
        
        prompt_info.history_text = "\n".join(history_parts)
        
        # === 4. 构造标准 messages 列表, 使用 apply_chat_template ===
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        
        # 添加历史对话
        for msg in selected_history_msgs:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前查询
        messages.append({"role": "user", "content": query})
        
        # 尝试使用 tokenizer.apply_chat_template (适配所有模型)
        tokenizer = getattr(self.model, 'tokenizer', None)
        use_chat_template = False
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            try:
                final_prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                use_chat_template = True
            except Exception as e:
                logger.warning(f"apply_chat_template failed, using fallback format: {e}")
        
        if not use_chat_template:
            # 回退: ChatML 格式 (半角标记, 标签闭合, Qwen/DeepSeek/通用)
            # 确保所有 <|im_start|> 都有对应的 <|im_end|> 闭合
            parts = []
            if full_system_prompt:
                parts.append(f"<|im_start|>system\n{full_system_prompt}<|im_end|>")
            for msg in selected_history_msgs:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            parts.append(f"<|im_start|>user\n{query}<|im_end|>")
            parts.append("<|im_start|>assistant")
            final_prompt = "\n".join(parts) + "\n"
        
        prompt_info.final_prompt = final_prompt
        
        # 最终安全检查
        final_tokens = self._estimate_tokens(final_prompt)
        if final_tokens > max_prompt_tokens:
            logger.warning(
                f"RAG prompt still too long ({final_tokens} > {max_prompt_tokens}), "
                f"forcefully truncating"
            )
            # 强制截断: 只保留 query (ChatML 格式, 半角标记, 标签闭合)
            final_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            prompt_info.final_prompt = final_prompt
            prompt_info.history_text = ""
            prompt_info.retrieved_context = ""
        
        return final_prompt, prompt_info
    
    def _get_conversation_history(
        self,
        session_id: str,
        max_turns: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session, with cross-session support.
        
        v6.1: 支持跨会话记忆
        - 首先获取该用户其他会话的历史消息 (跨会话记忆)
        - 然后获取当前会话的历史消息
        - 跨会话消息在前, 当前会话消息在后
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of conversation turns to retrieve
            user_id: User identifier (optional, for cross-session retrieval)
            
        Returns:
            List of conversation messages
        """
        result = []
        
        with self.db_manager.session_scope() as db:
            conv_repo = ConversationRepository(db)
            
            # v6.1: 跨会话历史 (放在当前会话之前)
            if user_id:
                try:
                    cross_session_limit = max_turns  # 跨会话消息量 = 当前会话消息量
                    cross_msgs = conv_repo.get_recent_by_user_cross_session(
                        user_id=user_id,
                        current_session_id=session_id,
                        limit=cross_session_limit * 2,  # 每轮 user+assistant
                    )
                    for msg in cross_msgs:
                        result.append({"role": msg.role, "content": msg.content})
                    if cross_msgs:
                        logger.info(
                            f"RAG cross-session: added {len(cross_msgs)} messages "
                            f"from previous sessions for user {user_id}"
                        )
                except Exception as e:
                    logger.warning(f"RAG cross-session retrieval failed (non-critical): {e}")
            
            # 当前会话历史
            messages = conv_repo.get_recent(session_id, n_turns=max_turns)
            for msg in messages:
                result.append({"role": msg.role, "content": msg.content})
        
        return result
    
    def _load_user_preferences(self, user_id: str) -> Optional[str]:
        """
        从数据库加载用户偏好文本 (同步版本)
        
        Args:
            user_id: 用户标识
            
        Returns:
            偏好文本 (多条偏好合并), 无偏好时返回 None
            
        Raises:
            RAGPreferenceError: 偏好加载失败 (非致命)
        """
        if not user_id:
            return None
        
        # v6.0: 检查缓存
        cached = self._preference_cache.get(user_id)
        if cached:
            pref_text, cached_time = cached
            if (time.time() - cached_time) < self._preference_cache_ttl:
                self._stats["preference_cache_hits"] += 1
                logger.debug(f"RAG preference cache hit for user {user_id}")
                return pref_text
        
        self._stats["preference_cache_misses"] += 1
        
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                preferences = pref_repo.get_by_user(user_id, active_only=True)
                
                if not preferences:
                    self._preference_cache[user_id] = (None, time.time())
                    return None
                
                # 按优先级合并偏好文本
                pref_texts = []
                for p in preferences:
                    text = getattr(p, 'preference_text', '') or ''
                    if text.strip():
                        pref_texts.append(text.strip())
                
                if pref_texts:
                    combined = "\n".join(pref_texts)
                    self._preference_cache[user_id] = (combined, time.time())
                    logger.debug(
                        f"RAG loaded {len(pref_texts)} preferences for user {user_id}: "
                        f"{len(combined)} chars"
                    )
                    return combined
                
                self._preference_cache[user_id] = (None, time.time())
        except Exception as e:
            logger.warning(f"RAG failed to load preferences for user {user_id}: {e}")
            raise RAGPreferenceError(
                f"Failed to load preferences for user {user_id}: {e}",
                cause=e,
            )
        
        return None
    
    async def _load_user_preferences_async(self, user_id: str) -> Optional[str]:
        """
        异步加载用户偏好文本 (带 SingleFlight 防惊群)
        
        Args:
            user_id: 用户标识
            
        Returns:
            偏好文本 (多条偏好合并), 无偏好时返回 None
        """
        if not user_id:
            return None
        
        # v6.0: 检查缓存
        cached = self._preference_cache.get(user_id)
        if cached:
            pref_text, cached_time = cached
            if (time.time() - cached_time) < self._preference_cache_ttl:
                self._stats["preference_cache_hits"] += 1
                logger.debug(f"RAG preference cache hit for user {user_id}")
                return pref_text
        
        # SingleFlight: 防止并发请求同一用户偏好
        if user_id in self._preference_single_flight:
            logger.debug(f"RAG preference single-flight hit for user {user_id}")
            return await self._preference_single_flight[user_id]
        
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._preference_single_flight[user_id] = future
        
        self._stats["preference_cache_misses"] += 1
        
        try:
            # 在线程池中执行同步 DB 操作
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._load_user_preferences_sync, user_id
            )
            self._preference_cache[user_id] = (result, time.time())
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._preference_single_flight.pop(user_id, None)
    
    def _load_user_preferences_sync(self, user_id: str) -> Optional[str]:
        """同步加载偏好 (供 run_in_executor 调用)"""
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                preferences = pref_repo.get_by_user(user_id, active_only=True)
                
                if not preferences:
                    return None
                
                pref_texts = []
                for p in preferences:
                    text = getattr(p, 'preference_text', '') or ''
                    if text.strip():
                        pref_texts.append(text.strip())
                
                if pref_texts:
                    combined = "\n".join(pref_texts)
                    logger.debug(
                        f"RAG loaded {len(pref_texts)} preferences for user {user_id}: "
                        f"{len(combined)} chars"
                    )
                    return combined
        except Exception as e:
            logger.warning(f"RAG failed to load preferences for user {user_id}: {e}")
        return None
    
    def invalidate_preference_cache(self, user_id: Optional[str] = None):
        """
        使偏好缓存失效
        
        Args:
            user_id: 指定用户 ID, None 则清除所有
        """
        if user_id:
            self._preference_cache.pop(user_id, None)
            logger.debug(f"RAG preference cache invalidated for user {user_id}")
        else:
            self._preference_cache.clear()
            logger.debug("RAG preference cache fully cleared")
    
    def _get_max_history_turns(self) -> int:
        """
        v5.7: 从外置配置读取最大历史轮次 (与 DKI 一致)
        
        配置路径: dki.recall.budget.max_recent_turns (默认 5)
        """
        try:
            recall_obj = getattr(self.config.dki, 'recall', None)
            if recall_obj:
                budget_obj = getattr(recall_obj, 'budget', None) if hasattr(recall_obj, 'budget') else (recall_obj.get('budget') if isinstance(recall_obj, dict) else None)
                if budget_obj:
                    val = getattr(budget_obj, 'max_recent_turns', 5) if hasattr(budget_obj, 'max_recent_turns') else (budget_obj.get('max_recent_turns', 5) if isinstance(budget_obj, dict) else 5)
                    return val
        except Exception:
            pass
        return 5
    
    def _prepare_chat_context(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
        top_k: Optional[int],
        system_prompt: Optional[str],
        max_history_turns: Optional[int],
        include_history: bool,
        preference_text: Optional[str] = None,
    ) -> Tuple[str, RAGPromptInfo, List[MemorySearchResult], Optional[List[Dict[str, str]]], Optional[str]]:
        """
        准备 chat 上下文 (共享逻辑, 供 chat / async_chat / chat_stream 使用)
        
        Returns:
            (prompt, prompt_info, memories, history, preference_text)
        """
        if max_history_turns is None:
            max_history_turns = self._get_max_history_turns()
        
        # 加载偏好 (如果未提供)
        if preference_text is None:
            try:
                preference_text = self._load_user_preferences(user_id)
            except RAGPreferenceError as e:
                logger.warning(f"Preference loading failed, degrading: {e}")
                preference_text = None
        
        # 构造 system prompt: 用户偏好 + 自定义 system prompt
        effective_system_prompt = system_prompt or ""
        if preference_text:
            pref_section = f"用户偏好:\n{preference_text}"
            if effective_system_prompt:
                effective_system_prompt = f"{pref_section}\n\n{effective_system_prompt}"
            else:
                effective_system_prompt = pref_section
            logger.debug(f"RAG injected preference into system prompt: {len(preference_text)} chars")
        
        # 检索相关记忆
        top_k = top_k or self.config.rag.top_k
        try:
            memories = self.memory_router.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Memory search failed, degrading to no-context: {e}")
            memories = []
        
        # 获取对话历史
        history = None
        if include_history:
            try:
                history = self._get_conversation_history(
                    session_id, max_turns=max_history_turns, user_id=user_id
                )
                logger.debug(
                    f"Retrieved {len(history)} history messages for session "
                    f"{session_id} (user={user_id})"
                )
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}")
                history = None
        
        # 模糊指代澄清
        _vague_ref = detect_vague_reference(query)
        if _vague_ref.is_vague:
            history_count = len(history) if history else 0
            history_insufficient = history_count <= 2
            if history_insufficient:
                clarification = build_clarification_instruction(_vague_ref.language)
                if effective_system_prompt:
                    effective_system_prompt = effective_system_prompt + "\n\n" + clarification
                else:
                    effective_system_prompt = clarification
                logger.info(
                    f"[Clarification] RAG path: "
                    f"confidence={_vague_ref.confidence:.2f}, "
                    f"pattern='{_vague_ref.matched_pattern}', "
                    f"history_count={history_count}"
                )
        
        # 构建提示词
        try:
            prompt, prompt_info = self._build_prompt(
                query, memories,
                effective_system_prompt if effective_system_prompt else None,
                history,
            )
        except Exception as e:
            raise RAGPromptBuildError(
                f"Failed to build RAG prompt: {e}", cause=e
            )
        
        return prompt, prompt_info, memories, history, preference_text
    
    def _generate_and_process(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, bool, ModelOutput]:
        """
        调用模型生成并后处理 (共享逻辑)
        
        Returns:
            (clean_response, think_stripped, raw_output)
            
        Raises:
            RAGGenerationError: 生成失败
        """
        try:
            output = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            # 区分不同的模型错误
            err_msg = str(e).lower()
            if "oom" in err_msg or "out of memory" in err_msg:
                raise RAGGenerationError(
                    f"Model OOM during RAG generation: {e}",
                    retryable=True, cause=e,
                )
            elif "timeout" in err_msg:
                raise RAGGenerationError(
                    f"Model timeout during RAG generation: {e}",
                    retryable=True, cause=e,
                )
            elif "connection" in err_msg or "connect" in err_msg:
                raise RAGGenerationError(
                    f"Model connection error during RAG generation: {e}",
                    retryable=True, cause=e,
                )
            else:
                raise RAGGenerationError(
                    f"RAG generation failed: {e}",
                    retryable=False, cause=e,
                )
        
        # 移除 <think> 推理内容
        clean_response, think_stripped = strip_think_content(output.text)
        if think_stripped:
            logger.debug(
                f"RAG: Think content stripped: "
                f"{len(output.text)} -> {len(clean_response)} chars"
            )
        
        return clean_response, think_stripped, output
    
    def _log_conversation(
        self,
        session_id: str,
        user_id: Optional[str],
        query: str,
        clean_response: str,
        memories: List[MemorySearchResult],
        total_latency: float,
    ):
        """记录对话到数据库 (非致命, 失败不影响返回)"""
        try:
            with self.db_manager.session_scope() as db:
                session_repo = SessionRepository(db)
                session_repo.get_or_create(session_id=session_id, user_id=user_id)
                
                conv_repo = ConversationRepository(db)
                audit_repo = AuditLogRepository(db)
                
                conv_repo.create(
                    session_id=session_id,
                    role='user',
                    content=query,
                )
                
                conv_repo.create(
                    session_id=session_id,
                    role='assistant',
                    content=clean_response,
                    injection_mode='rag',
                    memory_ids=[m.memory_id for m in memories],
                    latency_ms=total_latency,
                )
                
                audit_repo.log(
                    action='rag_generate',
                    session_id=session_id,
                    memory_ids=[m.memory_id for m in memories],
                    mode='rag',
                )
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    def chat(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        include_history: bool = True,
        max_history_turns: Optional[int] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Generate response using RAG with conversation history (同步版本).
        
        v6.0: 细化异常处理
        - RAGPreferenceError → 降级 (无偏好注入)
        - RAGMemorySearchError → 降级 (无上下文)
        - RAGGenerationError → 向上抛出 (可重试标记)
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier (optional, used for preference loading)
            top_k: Number of memories to retrieve
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            include_history: Whether to include conversation history
            max_history_turns: Maximum history turns (None = 从配置读取)
            
        Returns:
            RAGResponse with generated text and metadata
            
        Raises:
            RAGGenerationError: 模型生成失败
            RAGPromptBuildError: 提示词构造失败
        """
        start_time = time.perf_counter()
        self._stats["total_requests"] += 1
        
        try:
            prompt, prompt_info, memories, history, preference_text = self._prepare_chat_context(
                query=query,
                session_id=session_id,
                user_id=user_id,
                top_k=top_k,
                system_prompt=system_prompt,
                max_history_turns=max_history_turns,
                include_history=include_history,
            )
            
            clean_response, think_stripped, output = self._generate_and_process(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            end_time = time.perf_counter()
            total_latency = (end_time - start_time) * 1000
            
            # 记录对话
            self._log_conversation(
                session_id=session_id,
                user_id=user_id,
                query=query,
                clean_response=clean_response,
                memories=memories,
                total_latency=total_latency,
            )
            
            return RAGResponse(
                text=clean_response,
                memories_used=memories,
                latency_ms=total_latency,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata={
                    'prompt_length': len(prompt),
                    'model': self.model.model_name,
                    'history_turns': len(history) if history else 0,
                    'preference_injected': bool(preference_text),
                    'preference_text': preference_text or "",
                    'think_content_stripped': think_stripped,
                },
                prompt_info=prompt_info,
            )
        
        except (RAGGenerationError, RAGPromptBuildError):
            self._stats["errors"] += 1
            raise
        except Exception as e:
            self._stats["errors"] += 1
            raise RAGError(f"Unexpected RAG error: {e}", cause=e)
    
    async def async_chat(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        include_history: bool = True,
        max_history_turns: Optional[int] = None,
        **kwargs
    ) -> RAGResponse:
        """
        异步版本的 RAG chat.
        
        与 chat() 相同的逻辑, 但:
        - 偏好加载使用 AsyncSingleFlight 防惊群
        - 模型推理在线程池中执行 (避免阻塞事件循环)
        - 数据库操作在线程池中执行
        
        Args:
            同 chat()
            
        Returns:
            RAGResponse
            
        Raises:
            RAGGenerationError: 模型生成失败
            RAGPromptBuildError: 提示词构造失败
        """
        start_time = time.perf_counter()
        self._stats["total_requests"] += 1
        self._stats["async_requests"] += 1
        
        try:
            # 异步加载偏好
            preference_text = None
            try:
                preference_text = await self._load_user_preferences_async(user_id)
            except Exception as e:
                logger.warning(f"Async preference loading failed, degrading: {e}")
                preference_text = None
            
            # 准备上下文 (在线程池中执行同步 DB 操作)
            loop = asyncio.get_event_loop()
            prompt, prompt_info, memories, history, preference_text = await loop.run_in_executor(
                None,
                lambda: self._prepare_chat_context(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    top_k=top_k,
                    system_prompt=system_prompt,
                    max_history_turns=max_history_turns,
                    include_history=include_history,
                    preference_text=preference_text,
                ),
            )
            
            # 模型推理 (在线程池中执行)
            clean_response, think_stripped, output = await loop.run_in_executor(
                None,
                lambda: self._generate_and_process(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                ),
            )
            
            end_time = time.perf_counter()
            total_latency = (end_time - start_time) * 1000
            
            # 记录对话 (在线程池中执行, 不阻塞返回)
            loop.run_in_executor(
                None,
                lambda: self._log_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    clean_response=clean_response,
                    memories=memories,
                    total_latency=total_latency,
                ),
            )
            
            return RAGResponse(
                text=clean_response,
                memories_used=memories,
                latency_ms=total_latency,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                metadata={
                    'prompt_length': len(prompt),
                    'model': self.model.model_name,
                    'history_turns': len(history) if history else 0,
                    'preference_injected': bool(preference_text),
                    'preference_text': preference_text or "",
                    'think_content_stripped': think_stripped,
                    'async': True,
                },
                prompt_info=prompt_info,
            )
        
        except (RAGGenerationError, RAGPromptBuildError):
            self._stats["errors"] += 1
            raise
        except Exception as e:
            self._stats["errors"] += 1
            raise RAGError(f"Unexpected async RAG error: {e}", cause=e)
    
    async def chat_stream(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        include_history: bool = True,
        max_history_turns: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        RAG 流式聊天 (SSE 兼容)
        
        Yields:
            字典, 包含:
            - type: "token" | "metadata" | "done" | "error"
            - content: token 文本 (type=token 时)
            - text: 完整文本 (type=done 时)
            - error: 错误信息 (type=error 时)
        """
        start_time = time.perf_counter()
        self._stats["total_requests"] += 1
        self._stats["stream_requests"] += 1
        
        try:
            # 异步加载偏好
            preference_text = None
            try:
                preference_text = await self._load_user_preferences_async(user_id)
            except Exception:
                preference_text = None
            
            # 准备上下文
            loop = asyncio.get_event_loop()
            prompt, prompt_info, memories, history, preference_text = await loop.run_in_executor(
                None,
                lambda: self._prepare_chat_context(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    top_k=top_k,
                    system_prompt=system_prompt,
                    max_history_turns=max_history_turns,
                    include_history=include_history,
                    preference_text=preference_text,
                ),
            )
            
            # 发送 metadata
            yield {
                "type": "metadata",
                "memories_count": len(memories),
                "history_turns": len(history) if history else 0,
                "preference_injected": bool(preference_text),
            }
            
            # 检查模型是否支持流式生成
            has_stream = (
                hasattr(self.model, 'stream_generate')
                or hasattr(self.model, 'async_stream_generate')
            )
            
            if has_stream:
                full_text = ""
                input_tokens = 0
                output_tokens = 0
                
                if hasattr(self.model, 'async_stream_generate'):
                    stream = self.model.async_stream_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                    async for chunk in stream:
                        token_text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                        full_text += token_text
                        yield {"type": "token", "content": token_text}
                elif hasattr(self.model, 'stream_generate'):
                    for chunk in self.model.stream_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    ):
                        token_text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                        full_text += token_text
                        yield {"type": "token", "content": token_text}
                
                clean_text, _ = strip_think_content(full_text)
                total_latency = (time.perf_counter() - start_time) * 1000
                
                # 记录对话
                loop.run_in_executor(
                    None,
                    lambda: self._log_conversation(
                        session_id=session_id,
                        user_id=user_id,
                        query=query,
                        clean_response=clean_text,
                        memories=memories,
                        total_latency=total_latency,
                    ),
                )
                
                yield {
                    "type": "done",
                    "text": clean_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": total_latency,
                }
            else:
                # 模型不支持流式: 回退到非流式, 一次性返回
                clean_response, think_stripped, output = await loop.run_in_executor(
                    None,
                    lambda: self._generate_and_process(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    ),
                )
                
                total_latency = (time.perf_counter() - start_time) * 1000
                
                # 记录对话
                loop.run_in_executor(
                    None,
                    lambda: self._log_conversation(
                        session_id=session_id,
                        user_id=user_id,
                        query=query,
                        clean_response=clean_response,
                        memories=memories,
                        total_latency=total_latency,
                    ),
                )
                
                # 模拟流式: 一次性返回全部
                yield {"type": "token", "content": clean_response}
                yield {
                    "type": "done",
                    "text": clean_response,
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "latency_ms": total_latency,
                }
        
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"RAG stream error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_code": getattr(e, 'error_code', 'RAG_UNKNOWN'),
                "retryable": getattr(e, 'retryable', False),
            }
    
    def search_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[MemorySearchResult]:
        """
        Search memories without generation.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        return self.memory_router.search(query, top_k=top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        pref_cache_total = (
            self._stats["preference_cache_hits"]
            + self._stats["preference_cache_misses"]
        )
        return {
            'router_stats': self.memory_router.get_stats(),
            'model_info': self.model.get_model_info() if self._model_adapter else None,
            'config': {
                'top_k': self.config.rag.top_k,
                'similarity_threshold': self.config.rag.similarity_threshold,
            },
            # v6.0: 新增统计
            'requests': {
                'total': self._stats["total_requests"],
                'async': self._stats["async_requests"],
                'stream': self._stats["stream_requests"],
                'errors': self._stats["errors"],
            },
            'preference_cache': {
                'size': len(self._preference_cache),
                'ttl': self._preference_cache_ttl,
                'hits': self._stats["preference_cache_hits"],
                'misses': self._stats["preference_cache_misses"],
                'hit_rate': (
                    self._stats["preference_cache_hits"] / pref_cache_total
                    if pref_cache_total > 0 else 0
                ),
            },
        }
