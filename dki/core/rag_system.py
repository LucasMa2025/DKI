"""
RAG System for DKI
Retrieval-Augmented Generation implementation as baseline

v5.7 更新:
- 从提示词构造中移除 <think> 推理内容
- 使用新的 token 预算分配 (30% 生成预留)
- 历史轮次从外置配置读取 (与 DKI 一致)
- 使用快速 token 估算 (estimate_tokens_fast)
- 存储响应前移除 think 内容
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.core.text_utils import strip_think_content, estimate_tokens_fast
from dki.core.memory_router import MemoryRouter, MemorySearchResult
from dki.core.embedding_service import EmbeddingService
from dki.models.factory import ModelFactory
from dki.models.base import BaseModelAdapter, ModelOutput
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository, MemoryRepository, ConversationRepository, AuditLogRepository,
    UserPreferenceRepository,
)
from dki.config.config_loader import ConfigLoader


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
    """
    
    def __init__(
        self,
        model_adapter: Optional[BaseModelAdapter] = None,
        memory_router: Optional[MemoryRouter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        engine: Optional[str] = None,
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
        
        logger.info("RAG System initialized")
    
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
        从数据库加载用户偏好文本 (v5.3: 为 RAG 增加偏好注入, 确保对比公平)
        
        Args:
            user_id: 用户标识
            
        Returns:
            偏好文本 (多条偏好合并), 无偏好时返回 None
        """
        if not user_id:
            return None
        
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                preferences = pref_repo.get_by_user(user_id, active_only=True)
                
                if not preferences:
                    return None
                
                # 按优先级合并偏好文本
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
    
    def chat(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        include_history: bool = True,
        max_history_turns: Optional[int] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Generate response using RAG with conversation history.
        
        v5.7: 增强偏好注入和历史轮次配置
        - 偏好通过 system prompt 注入 (与 DKI 一致)
        - 历史轮次从外置配置读取 (dki.recall.budget.max_recent_turns)
        - 移除 <think> 推理内容 (存储和召回时双重过滤)
        - 使用 30% 上下文生成预留 (与 DKI 一致)
        
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
        """
        start_time = time.perf_counter()
        
        # v5.7: 历史轮次从外置配置读取 (与 DKI 一致)
        if max_history_turns is None:
            max_history_turns = self._get_max_history_turns()
        
        # v5.3: 加载用户偏好 (与 DKI 公平对比)
        preference_text = self._load_user_preferences(user_id)
        
        # 构造 system prompt: 用户偏好 + 自定义 system prompt
        effective_system_prompt = system_prompt or ""
        if preference_text:
            pref_section = f"用户偏好:\n{preference_text}"
            if effective_system_prompt:
                effective_system_prompt = f"{pref_section}\n\n{effective_system_prompt}"
            else:
                effective_system_prompt = pref_section
            logger.debug(f"RAG injected preference into system prompt: {len(preference_text)} chars")
        
        # Retrieve relevant memories
        top_k = top_k or self.config.rag.top_k
        memories = self.memory_router.search(query, top_k=top_k)
        
        # Get conversation history if enabled
        history = None
        if include_history:
            try:
                history = self._get_conversation_history(session_id, max_turns=max_history_turns, user_id=user_id)
                logger.debug(f"Retrieved {len(history)} history messages for session {session_id} (user={user_id})")
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}")
                history = None
        
        # Build prompt with history (now returns tuple)
        prompt, prompt_info = self._build_prompt(
            query, memories,
            effective_system_prompt if effective_system_prompt else None,
            history,
        )
        
        # Generate response
        output = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000
        
        # v5.7: 存储前移除 <think> 推理内容
        clean_response, think_stripped = strip_think_content(output.text)
        if think_stripped:
            logger.debug(
                f"RAG: Think content stripped before DB storage: "
                f"{len(output.text)} -> {len(clean_response)} chars"
            )
        
        # Log to database
        try:
            with self.db_manager.session_scope() as db:
                # Ensure session exists before inserting conversation
                session_repo = SessionRepository(db)
                session_repo.get_or_create(session_id=session_id, user_id=user_id)
                
                conv_repo = ConversationRepository(db)
                audit_repo = AuditLogRepository(db)
                
                # Store user message
                conv_repo.create(
                    session_id=session_id,
                    role='user',
                    content=query,
                )
                
                # Store assistant response (存储清理后的内容)
                conv_repo.create(
                    session_id=session_id,
                    role='assistant',
                    content=clean_response,
                    injection_mode='rag',
                    memory_ids=[m.memory_id for m in memories],
                    latency_ms=total_latency,
                )
                
                # Audit log
                audit_repo.log(
                    action='rag_generate',
                    session_id=session_id,
                    memory_ids=[m.memory_id for m in memories],
                    mode='rag',
                )
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
        
        return RAGResponse(
            text=clean_response,  # v5.7: 返回清理后的响应 (无 think 内容)
            memories_used=memories,
            latency_ms=total_latency,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            metadata={
                'prompt_length': len(prompt),
                'model': self.model.model_name,
                'history_turns': len(history) if history else 0,
                # v5.3: 偏好注入信息 (与 DKI 对比公平)
                'preference_injected': bool(preference_text),
                'preference_text': preference_text or "",
                # v5.7: think 内容过滤信息
                'think_content_stripped': think_stripped,
            },
            prompt_info=prompt_info,
        )
    
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
        return {
            'router_stats': self.memory_router.get_stats(),
            'model_info': self.model.get_model_info() if self._model_adapter else None,
            'config': {
                'top_k': self.config.rag.top_k,
                'similarity_threshold': self.config.rag.similarity_threshold,
            },
        }
