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
import json
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
    KVOOMError,
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
from dki.core.recall.recall_config import RecallConfig, RecallFactCallConfig, FactResponse

from collections import OrderedDict


# ============================================================
# retrieve_fact 工具定义 (闭源模型 native tool_calls 用)
# ============================================================

RETRIEVE_FACT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "retrieve_fact",
        "description": (
            "当回答需要精确原话、具体数值、时间、因果关系，"
            "且当前摘要中未明确包含时，调用此函数获取原始记录。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trace_id": {
                    "type": "string",
                    "description": "消息溯源 ID (从 [SUMMARY] 标记中获取)",
                },
                "offset": {
                    "type": "integer",
                    "description": "分页偏移 (默认 0)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数 (默认 5)",
                    "default": 5,
                },
            },
            "required": ["trace_id"],
        },
    },
}


# ============================================================
# 有界 TTL 缓存 (修复无上限 dict 内存泄漏风险)
# ============================================================

class BoundedTTLCache:
    """
    有界 TTL 缓存 (LRU 淘汰 + TTL 过期)
    
    解决原始 dict 缓存无上限的问题:
    - maxsize: 最大缓存条目数 (超过后淘汰最久未使用的条目)
    - ttl: 条目存活时间 (秒)
    """
    
    def __init__(self, maxsize: int = 1000, ttl: float = 300.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[tuple]:
        """获取缓存条目, 过期返回 None"""
        item = self._cache.get(key)
        if item is None:
            return None
        value, cached_time = item
        if (time.time() - cached_time) >= self._ttl:
            # 已过期, 移除
            self._cache.pop(key, None)
            return None
        # 命中: 移到末尾 (LRU)
        self._cache.move_to_end(key)
        return item
    
    def set(self, key: str, value, timestamp: Optional[float] = None):
        """设置缓存条目"""
        ts = timestamp or time.time()
        self._cache[key] = (value, ts)
        self._cache.move_to_end(key)
        # 超过上限: 淘汰最久未使用
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
    
    def pop(self, key: str, default=None):
        """移除缓存条目"""
        return self._cache.pop(key, default)
    
    def clear(self):
        """清空所有缓存"""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        item = self._cache.get(key)
        if item is None:
            return False
        _, cached_time = item
        if (time.time() - cached_time) >= self._ttl:
            self._cache.pop(key, None)
            return False
        return True


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
        
        # v6.0: 偏好缓存 (TTL + SingleFlight + LRU 有界)
        _ttl = preference_cache_ttl or self._PREFERENCE_CACHE_TTL
        self._preference_cache = BoundedTTLCache(maxsize=1000, ttl=_ttl)
        self._preference_cache_ttl = _ttl
        self._preference_single_flight: Dict[str, asyncio.Future] = {}
        
        # v7.1: 历史消息压缩组件 (仅 compress 模式初始化)
        self._suffix_builder = None
        self._recall_config_obj = None
        self._prompt_formatter_obj = None
        if getattr(self.config.rag, 'history_compression', 'truncate') == 'compress':
            self._init_suffix_builder()
        
        # 统计
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "async_requests": 0,
            "stream_requests": 0,
            "preference_cache_hits": 0,
            "preference_cache_misses": 0,
            "errors": 0,
        }
        
        _mode = getattr(self.config.rag, 'history_compression', 'truncate')
        logger.info(
            f"RAG System initialized (v7.1: async + streaming + preference cache, "
            f"history_compression={_mode})"
        )
    
    @property
    def model(self) -> BaseModelAdapter:
        """Get or create model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelFactory.get_or_create(engine=self._engine)
        return self._model_adapter
    
    def _init_suffix_builder(self):
        """
        v7.1: 初始化 SuffixBuilder (仅 compress 模式使用)
        
        复用 DKI 的 recall 组件:
        - RecallConfig: 从 config.dki.recall 读取
        - PromptFormatter: 根据模型名自动选择
        - SuffixBuilder: 智能压缩 + trace_id 溯源
        """
        try:
            from dki.core.recall import (
                RecallConfig,
                SuffixBuilder,
                create_formatter,
            )
            
            # 从配置加载 recall config
            recall_dict = {}
            if hasattr(self.config, 'dki') and hasattr(self.config.dki, 'recall'):
                recall_obj = self.config.dki.recall
                if isinstance(recall_obj, dict):
                    recall_dict = recall_obj
                elif hasattr(recall_obj, 'model_dump'):
                    recall_dict = recall_obj.model_dump()
                elif hasattr(recall_obj, 'dict'):
                    recall_dict = recall_obj.dict()
            self._recall_config_obj = RecallConfig.from_dict(recall_dict)
            
            # 创建 PromptFormatter
            model_name = ''
            if self._model_adapter:
                model_name = getattr(self._model_adapter, 'model_name', '') or ''
            self._prompt_formatter_obj = create_formatter(
                model_name=model_name,
                formatter_type=self._recall_config_obj.prompt_formatter,
                language='cn',
            )
            
            # 创建 SuffixBuilder
            token_counter = None
            if self._model_adapter and hasattr(self._model_adapter, 'tokenizer'):
                tokenizer = self._model_adapter.tokenizer
                if tokenizer:
                    token_counter = lambda text: len(tokenizer.encode(text)) if text else 0
            self._suffix_builder = SuffixBuilder(
                config=self._recall_config_obj,
                prompt_formatter=self._prompt_formatter_obj,
                token_counter=token_counter,
                model_adapter=self._model_adapter,
            )
            
            logger.info(
                "RAG compress mode: SuffixBuilder initialized "
                f"(formatter={type(self._prompt_formatter_obj).__name__})"
            )
        except ImportError:
            logger.warning(
                "RAG compress mode: recall components not available, "
                "falling back to truncate mode"
            )
        except Exception as e:
            logger.warning(
                f"RAG compress mode: SuffixBuilder init failed ({e}), "
                "falling back to truncate mode"
            )
    
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
    ) -> Tuple[str, "RAGPromptInfo", Dict[str, Any]]:
        """
        Build prompt with retrieved memories and conversation history.
        
        使用 tokenizer.apply_chat_template 构造符合各模型官方标准的 chat 格式:
        - DeepSeek/Qwen: <|im_start|>system/user/assistant<|im_end|>
        - Llama 3.x:     <|begin_of_text|><|start_header_id|>...<|end_header_id|>
        - GLM:           GLM 原生 chat template
        - 其他模型:      tokenizer 内置的 chat template
        
        如果 tokenizer 不支持 apply_chat_template, 则回退到通用格式。
        
        Includes automatic truncation to prevent exceeding model context length.
        
        v7.1: 新增 compress 模式 (history_compression="compress")
        - 超预算的历史消息使用 SuffixBuilder 智能压缩
        - 压缩后的消息保留 trace_id, 支持 retrieve_fact 溯源
        - 返回值扩展为三元组: (prompt, prompt_info, compression_meta)
        
        Args:
            query: User query
            memories: Retrieved memories
            system_prompt: Optional system prompt
            history: Optional conversation history
                     [{"role": "user/assistant", "content": "...", "id": "..."}]
            
        Returns:
            Tuple of (formatted_prompt, RAGPromptInfo, compression_meta)
            compression_meta 包含:
            - trace_ids: List[str] — 被压缩消息的 trace_id 列表
            - has_fact_call_instruction: bool — 是否包含 fact call 指导
            - summary_count: int — 被压缩的消息数量
            - compression_mode: str — "truncate" | "compress"
        """
        # v5.7: 获取模型最大上下文长度, 生成预留 = 30% 上下文
        max_context = self._get_max_context_length()
        generation_reserve = int(max_context * 0.30)
        tag_overhead = 120  # chat template 标记开销
        max_prompt_tokens = max_context - generation_reserve - tag_overhead
        
        # 压缩元数据 (默认: 无压缩)
        compression_meta: Dict[str, Any] = {
            "trace_ids": [],
            "has_fact_call_instruction": False,
            "summary_count": 0,
            "compression_mode": "truncate",
        }
        
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
        
        # === 3. 构建 conversation history ===
        # 计算可用 token 预算
        system_tokens = self._estimate_tokens(full_system_prompt) if full_system_prompt else 0
        query_tokens = self._estimate_tokens(query)
        remaining_tokens = max(max_prompt_tokens - system_tokens - query_tokens - 40, 0)
        
        history_parts = []
        selected_history_msgs = []
        
        # 判断压缩模式
        compression_mode = getattr(self.config.rag, 'history_compression', 'truncate')
        use_compress = (
            compression_mode == 'compress'
            and self._suffix_builder is not None
            and history
        )
        
        if history and use_compress:
            # ---- compress 模式: 使用 SuffixBuilder 智能压缩 ----
            compression_meta["compression_mode"] = "compress"
            selected_history_msgs, compression_meta = self._compress_history_with_suffix_builder(
                history=history,
                query=query,
                remaining_tokens=remaining_tokens,
                compression_meta=compression_meta,
            )
            
            if selected_history_msgs:
                for msg in selected_history_msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg['content']}")
        
        elif history:
            # ---- truncate 模式 (默认/baseline): 整条丢弃最旧消息 ----
            compression_meta["compression_mode"] = "truncate"
            
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
        
        return final_prompt, prompt_info, compression_meta
    
    def _compress_history_with_suffix_builder(
        self,
        history: List[Dict[str, str]],
        query: str,
        remaining_tokens: int,
        compression_meta: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        v7.1: 使用 SuffixBuilder 智能压缩历史消息
        
        将 RAG 的 history dict 列表适配为 SuffixBuilder 可接受的输入,
        调用 SuffixBuilder.build() 获得压缩结果, 然后还原为 messages 列表。
        
        Args:
            history: 历史消息列表 [{"role", "content", "id"(可选)}]
            query: 当前查询 (用于 Query-Aware 评分)
            remaining_tokens: 可用 token 预算
            compression_meta: 压缩元数据 (将被更新)
            
        Returns:
            (selected_msgs, updated_compression_meta)
        """
        from types import SimpleNamespace
        
        try:
            # 将 dict 包装为 SimpleNamespace (SuffixBuilder 通过 getattr 访问)
            wrapped_messages = []
            for msg in history:
                ns = SimpleNamespace(
                    content=msg.get('content', ''),
                    role=msg.get('role', 'user'),
                    id=msg.get('id', str(id(msg))),
                    message_id=msg.get('id', str(id(msg))),
                )
                wrapped_messages.append(ns)
            
            # 调用 SuffixBuilder
            suffix_result = self._suffix_builder.build(
                query=query,
                recalled_messages=wrapped_messages,
                context_window=remaining_tokens + 200,  # +200 补偿 SuffixBuilder 内部的预留
                preference_tokens=0,
            )
            
            # 从 AssembledSuffix.items 还原为 messages 列表
            selected_msgs = []
            for item in suffix_result.items:
                selected_msgs.append({
                    "role": item.role or "user",
                    "content": item.content,
                })
            
            # 更新压缩元数据
            compression_meta["trace_ids"] = suffix_result.trace_ids
            compression_meta["has_fact_call_instruction"] = suffix_result.has_fact_call_instruction
            compression_meta["summary_count"] = suffix_result.summary_count
            
            if suffix_result.summary_count > 0:
                logger.info(
                    f"RAG compress mode: {suffix_result.message_count} msgs kept, "
                    f"{suffix_result.summary_count} msgs compressed, "
                    f"trace_ids={suffix_result.trace_ids}, "
                    f"total_tokens={suffix_result.total_tokens}"
                )
            
            return selected_msgs, compression_meta
        
        except Exception as e:
            logger.warning(
                f"RAG compress mode failed ({e}), falling back to truncate"
            )
            # 降级: 使用 truncate 逻辑
            compression_meta["compression_mode"] = "truncate"
            cleaned = []
            for msg in history:
                cleaned_msg = dict(msg)
                if cleaned_msg.get('role') == 'assistant' and cleaned_msg.get('content'):
                    cleaned_content, _ = strip_think_content(cleaned_msg['content'])
                    if cleaned_content and cleaned_content.strip():
                        cleaned_msg['content'] = cleaned_content
                    else:
                        continue
                cleaned.append(cleaned_msg)
            
            selected = []
            used = 0
            for msg in reversed(cleaned):
                t = self._estimate_tokens(msg['content']) + 8
                if used + t > remaining_tokens:
                    break
                selected.insert(0, msg)
                used += t
            
            return selected, compression_meta
    
    def _get_conversation_history(
        self,
        session_id: str,
        max_turns: int = 5,
        user_id: Optional[str] = None,
        include_ids: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session, with cross-session support.
        
        v6.1: 支持跨会话记忆
        - 首先获取该用户其他会话的历史消息 (跨会话记忆)
        - 然后获取当前会话的历史消息
        - 跨会话消息在前, 当前会话消息在后
        
        v7.1: include_ids=True 时, 每条消息额外携带 "id" 字段
              用于 compress 模式下 SuffixBuilder 的 trace_id 溯源
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of conversation turns to retrieve
            user_id: User identifier (optional, for cross-session retrieval)
            include_ids: If True, include message ID in each dict
            
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
                        entry = {"role": msg.role, "content": msg.content}
                        if include_ids:
                            entry["id"] = getattr(msg, 'id', str(id(msg)))
                        result.append(entry)
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
                entry = {"role": msg.role, "content": msg.content}
                if include_ids:
                    entry["id"] = getattr(msg, 'id', str(id(msg)))
                result.append(entry)
        
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
        
        # v6.0: 检查缓存 (BoundedTTLCache 内部处理 TTL 过期)
        cached = self._preference_cache.get(user_id)
        if cached:
            pref_text, _cached_time = cached
            self._stats["preference_cache_hits"] += 1
            logger.debug(f"RAG preference cache hit for user {user_id}")
            return pref_text
        
        self._stats["preference_cache_misses"] += 1
        
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                preferences = pref_repo.get_by_user(user_id, active_only=True)
                
                if not preferences:
                    self._preference_cache.set(user_id, None)
                    return None
                
                # 按优先级合并偏好文本
                pref_texts = []
                for p in preferences:
                    text = getattr(p, 'preference_text', '') or ''
                    if text.strip():
                        pref_texts.append(text.strip())
                
                if pref_texts:
                    combined = "\n".join(pref_texts)
                    self._preference_cache.set(user_id, combined)
                    logger.debug(
                        f"RAG loaded {len(pref_texts)} preferences for user {user_id}: "
                        f"{len(combined)} chars"
                    )
                    return combined
                
                self._preference_cache.set(user_id, None)
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
        
        修复:
        - 使用 asyncio.get_running_loop() 替代废弃的 get_event_loop()
        - 等待方使用 asyncio.shield() 防止被取消影响 future
        - 等待方异常时降级为 None, 不向上传播
        
        Args:
            user_id: 用户标识
            
        Returns:
            偏好文本 (多条偏好合并), 无偏好时返回 None
        """
        if not user_id:
            return None
        
        # v6.0: 检查缓存 (BoundedTTLCache 内部处理 TTL 过期)
        cached = self._preference_cache.get(user_id)
        if cached:
            pref_text, _cached_time = cached
            self._stats["preference_cache_hits"] += 1
            logger.debug(f"RAG preference cache hit for user {user_id}")
            return pref_text
        
        # SingleFlight: 等待已有的 in-flight 请求 (shield 保护 + 异常降级)
        if user_id in self._preference_single_flight:
            logger.debug(f"RAG preference single-flight hit for user {user_id}")
            try:
                return await asyncio.shield(self._preference_single_flight[user_id])
            except Exception:
                return None  # 降级: 不向上传播, 避免等待方因主请求异常而崩溃
        
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._preference_single_flight[user_id] = future
        
        self._stats["preference_cache_misses"] += 1
        
        try:
            # 在线程池中执行同步 DB 操作
            result = await loop.run_in_executor(
                None, self._load_user_preferences_sync, user_id
            )
            self._preference_cache.set(user_id, result)
            future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
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
        
        v6.1 修复: 简化嵌套 getattr/isinstance 判断, 使用 try/except + 属性链
        """
        try:
            return self.config.dki.recall.budget.max_recent_turns
        except (AttributeError, TypeError):
            pass
        
        # 回退: 尝试 dict 风格访问
        try:
            recall_obj = getattr(self.config.dki, 'recall', None)
            if isinstance(recall_obj, dict):
                budget = recall_obj.get('budget', {})
                if isinstance(budget, dict):
                    return budget.get('max_recent_turns', 5)
        except (AttributeError, TypeError):
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
    ) -> Tuple[str, "RAGPromptInfo", List[MemorySearchResult], Optional[List[Dict[str, str]]], Optional[str], Dict[str, Any]]:
        """
        准备 chat 上下文 (共享逻辑, 供 chat / async_chat / chat_stream 使用)
        
        v7.1: 返回值扩展为六元组, 新增 compression_meta
        
        Returns:
            (prompt, prompt_info, memories, history, preference_text, compression_meta)
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
        # v7.1: compress 模式需要 include_ids=True 以保留 trace_id
        history = None
        _use_compress = (
            getattr(self.config.rag, 'history_compression', 'truncate') == 'compress'
            and self._suffix_builder is not None
        )
        if include_history:
            try:
                history = self._get_conversation_history(
                    session_id, max_turns=max_history_turns,
                    user_id=user_id, include_ids=_use_compress,
                )
                logger.debug(
                    f"Retrieved {len(history)} history messages for session "
                    f"{session_id} (user={user_id}, include_ids={_use_compress})"
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
            prompt, prompt_info, compression_meta = self._build_prompt(
                query, memories,
                effective_system_prompt if effective_system_prompt else None,
                history,
            )
        except Exception as e:
            raise RAGPromptBuildError(
                f"Failed to build RAG prompt: {e}", cause=e
            )
        
        return prompt, prompt_info, memories, history, preference_text, compression_meta
    
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
        except (ModelOOMError, KVOOMError) as e:
            # 结构化异常: GPU OOM (优先匹配, 可重试)
            raise RAGGenerationError(
                f"Model OOM during RAG generation: {e}",
                retryable=True, cause=e,
            )
        except ModelTimeoutError as e:
            # 结构化异常: 超时 (可重试)
            raise RAGGenerationError(
                f"Model timeout during RAG generation: {e}",
                retryable=True, cause=e,
            )
        except ModelConnectionError as e:
            # 结构化异常: 连接错误 (可重试)
            raise RAGGenerationError(
                f"Model connection error during RAG generation: {e}",
                retryable=True, cause=e,
            )
        except ModelError as e:
            # 结构化异常: 其他模型错误 (不可重试)
            raise RAGGenerationError(
                f"RAG generation failed: {e}",
                retryable=False, cause=e,
            )
        except Exception as e:
            # 回退: 未被结构化异常捕获的错误, 使用字符串匹配兜底
            err_msg = str(e).lower()
            retryable = (
                "oom" in err_msg or "out of memory" in err_msg
                or "timeout" in err_msg
                or "connection" in err_msg or "connect" in err_msg
            )
            raise RAGGenerationError(
                f"RAG generation failed: {e}",
                retryable=retryable, cause=e,
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
    
    async def _log_conversation_async(
        self,
        session_id: str,
        user_id: Optional[str],
        query: str,
        clean_response: str,
        memories: List[MemorySearchResult],
        total_latency: float,
    ):
        """
        异步记录对话到数据库 (非致命, 失败只记录日志)
        
        修复 fire-and-forget 问题: 使用 ensure_future + done_callback
        确保异常不会被静默吞掉, 也不会阻塞调用方
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
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
        except Exception as e:
            logger.error(f"Async conversation log failed: {e}")
    
    def _fire_and_forget_log(
        self,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        user_id: Optional[str],
        query: str,
        clean_response: str,
        memories: List[MemorySearchResult],
        total_latency: float,
    ):
        """
        Fire-and-forget 日志记录 (带异常回调, 防止内存泄漏)
        
        替代直接调用 loop.run_in_executor() 不 await 的模式
        """
        task = asyncio.ensure_future(
            self._log_conversation_async(
                session_id=session_id,
                user_id=user_id,
                query=query,
                clean_response=clean_response,
                memories=memories,
                total_latency=total_latency,
            )
        )
        task.add_done_callback(
            lambda t: (
                logger.error(f"Log task failed: {t.exception()}")
                if t.exception() else None
            )
        )
    
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
            prompt, prompt_info, memories, history, preference_text, compression_meta = self._prepare_chat_context(
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
                    'compression_mode': compression_meta.get('compression_mode', 'truncate'),
                    'trace_ids': compression_meta.get('trace_ids', []),
                    'has_fact_call_instruction': compression_meta.get('has_fact_call_instruction', False),
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
            loop = asyncio.get_running_loop()
            prompt, prompt_info, memories, history, preference_text, compression_meta = await loop.run_in_executor(
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
            
            # 模型推理: 检查 fact_retrieve_method 路由
            fact_method = self._get_fact_retrieve_method()
            
            if fact_method == "native_tool_calls":
                # 闭源模型: 使用原生 tool_calls 进行事实检索
                fact_cfg = self._get_fact_call_config()
                clean_response, think_stripped, output = await self._generate_with_tool_calls(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    session_id=session_id,
                    max_rounds=fact_cfg.inline_intercept_max_rounds,
                    max_fact_tokens=fact_cfg.max_fact_tokens,
                    **kwargs,
                )
            else:
                # post_hoc (默认) / inline_intercept (开源): 在线程池中执行
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
            
            # 记录对话 (fire-and-forget, 带异常回调, 不阻塞返回)
            self._fire_and_forget_log(
                loop=loop,
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
                    'async': True,
                    'fact_retrieve_method': fact_method,
                    'compression_mode': compression_meta.get('compression_mode', 'truncate'),
                    'trace_ids': compression_meta.get('trace_ids', []),
                    'has_fact_call_instruction': compression_meta.get('has_fact_call_instruction', False),
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
            loop = asyncio.get_running_loop()
            prompt, prompt_info, memories, history, preference_text, compression_meta = await loop.run_in_executor(
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
            
            # 发送 metadata (v7.1: 包含压缩模式信息)
            yield {
                "type": "metadata",
                "memories_count": len(memories),
                "history_turns": len(history) if history else 0,
                "preference_injected": bool(preference_text),
                "compression_mode": compression_meta.get("compression_mode", "truncate"),
                "trace_ids": compression_meta.get("trace_ids", []),
                "has_fact_call_instruction": compression_meta.get("has_fact_call_instruction", False),
            }
            
            # 检查 fact_retrieve_method 路由
            fact_method = self._get_fact_retrieve_method()
            
            if fact_method == "native_tool_calls":
                # ---- 闭源模型 native_tool_calls 流式路径 ----
                # Phase 1: 非流式 tool_calls 循环 (需要完整响应判断 finish_reason)
                fact_cfg = self._get_fact_call_config()
                clean_response, think_stripped, output = await self._generate_with_tool_calls(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    session_id=session_id,
                    max_rounds=fact_cfg.inline_intercept_max_rounds,
                    max_fact_tokens=fact_cfg.max_fact_tokens,
                    **kwargs,
                )
                
                total_latency = (time.perf_counter() - start_time) * 1000
                
                # 记录对话 (fire-and-forget)
                self._fire_and_forget_log(
                    loop=loop,
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    clean_response=clean_response,
                    memories=memories,
                    total_latency=total_latency,
                )
                
                # 模拟流式输出 (tool_calls 循环已完成, 结果一次性返回)
                yield {"type": "token", "content": clean_response}
                yield {
                    "type": "done",
                    "text": clean_response,
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "latency_ms": total_latency,
                    "fact_retrieve_method": fact_method,
                }
            else:
                # ---- post_hoc (默认) 流式路径 ----
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
                    
                    # 记录对话 (fire-and-forget, 带异常回调)
                    self._fire_and_forget_log(
                        loop=loop,
                        session_id=session_id,
                        user_id=user_id,
                        query=query,
                        clean_response=clean_text,
                        memories=memories,
                        total_latency=total_latency,
                    )
                    
                    # 流式模式下估算 token (修复 token 统计为 0 的问题)
                    input_tokens = self._estimate_tokens(prompt)
                    output_tokens = self._estimate_tokens(clean_text)
                    
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
                    
                    # 记录对话 (fire-and-forget, 带异常回调)
                    self._fire_and_forget_log(
                        loop=loop,
                        session_id=session_id,
                        user_id=user_id,
                        query=query,
                        clean_response=clean_response,
                        memories=memories,
                        total_latency=total_latency,
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
    
    # ================================================================
    # Native Tool Calls (闭源模型 fact retrieval)
    # ================================================================

    def _get_fact_retrieve_method(self) -> str:
        """
        获取 fact_retrieve_method 配置值
        
        优先从 config.dki.recall.fact_call 读取;
        如果配置为 "auto", 则根据模型类型自动推断:
        - 闭源模型 → "native_tool_calls"
        - 开源模型 → "inline_intercept"
        
        Returns:
            "post_hoc" | "inline_intercept" | "native_tool_calls"
        """
        method = "post_hoc"
        try:
            fact_call_cfg = self.config.dki.recall.fact_call
            if isinstance(fact_call_cfg, dict):
                method = fact_call_cfg.get("fact_retrieve_method", "post_hoc")
            elif hasattr(fact_call_cfg, "fact_retrieve_method"):
                method = fact_call_cfg.fact_retrieve_method
        except (AttributeError, TypeError):
            pass
        
        if method == "auto":
            if getattr(self.model, 'is_closed_source', False):
                method = "native_tool_calls"
            else:
                method = "inline_intercept"
        
        return method

    def _get_fact_call_config(self) -> RecallFactCallConfig:
        """获取 fact_call 配置"""
        try:
            fact_call_cfg = self.config.dki.recall.fact_call
            if isinstance(fact_call_cfg, RecallFactCallConfig):
                return fact_call_cfg
            if isinstance(fact_call_cfg, dict):
                return RecallFactCallConfig(**{
                    k: v for k, v in fact_call_cfg.items()
                    if k in RecallFactCallConfig.__dataclass_fields__
                })
        except (AttributeError, TypeError):
            pass
        return RecallFactCallConfig()

    def _prompt_to_messages(self, prompt: str) -> List[Dict[str, Any]]:
        """
        将格式化后的 prompt 字符串反解析为 messages 列表
        
        支持 ChatML 格式 (<|im_start|>...<|im_end|>) 和纯文本格式。
        闭源模型 API 需要 messages 列表而非拼接的 prompt 字符串。
        """
        messages = []
        
        # 尝试 ChatML 格式解析
        import re
        chatml_pattern = re.compile(
            r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',
            re.DOTALL,
        )
        matches = chatml_pattern.findall(prompt)
        
        if matches:
            for role, content in matches:
                messages.append({"role": role, "content": content.strip()})
            return messages
        
        # 尝试 Llama3 格式解析
        llama3_pattern = re.compile(
            r'<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n*(.*?)(?=<\|start_header_id\|>|<\|eot_id\|>|$)',
            re.DOTALL,
        )
        matches = llama3_pattern.findall(prompt)
        if matches:
            for role, content in matches:
                content = content.strip()
                if content:
                    messages.append({"role": role, "content": content})
            return messages
        
        # 回退: 整个 prompt 作为 user message
        messages.append({"role": "user", "content": prompt.strip()})
        return messages

    async def _call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        tools: List[Dict],
        **kwargs,
    ) -> ModelOutput:
        """
        调用闭源模型 API (带 tools 参数)
        
        优先使用 adapter 的 async_generate_with_tools 方法;
        如果不支持, 回退到普通 generate (tools 信息丢失)。
        """
        adapter = self.model
        
        if hasattr(adapter, 'async_generate_with_tools'):
            return await adapter.async_generate_with_tools(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
        
        # 回退: 不支持 tools, 用普通 generate
        logger.warning(
            "Model adapter does not support async_generate_with_tools, "
            "falling back to generate without tools"
        )
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: adapter.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            ),
        )

    async def _call_without_tools(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ModelOutput:
        """
        调用闭源模型 API (不带 tools, 最终回答轮)
        """
        adapter = self.model
        
        if hasattr(adapter, 'async_generate_with_tools'):
            # 复用同一方法, 不传 tools
            return await adapter.async_generate_with_tools(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                tools=None,
                **kwargs,
            )
        
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: adapter.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            ),
        )

    def _retrieve_fact_for_rag(
        self,
        trace_id: str,
        session_id: str,
    ) -> Optional[FactResponse]:
        """
        为 RAG 路径检索事实
        
        使用 FactRetriever (如果可用), 否则直接查数据库。
        """
        # 尝试使用 FactRetriever
        try:
            from dki.core.recall.fact_retriever import FactRetriever
            from dki.core.recall.recall_config import RecallConfig
            
            recall_cfg = None
            try:
                recall_cfg = RecallConfig.from_dict(
                    getattr(self.config.dki, 'recall', {})
                    if hasattr(self.config, 'dki') and hasattr(self.config.dki, 'recall')
                    else {}
                )
            except Exception:
                recall_cfg = RecallConfig()
            
            retriever = FactRetriever(config=recall_cfg)
            # 提供 conversation_repo
            with self.db_manager.session_scope() as db:
                conv_repo = ConversationRepository(db)
                retriever._conversation_repo = conv_repo
                return retriever.retrieve(
                    trace_id=trace_id,
                    session_id=session_id,
                    db_session=db,
                )
        except Exception as e:
            logger.warning(f"RAG fact retrieval failed for trace_id={trace_id}: {e}")
            return None

    def _format_fact_result(self, fact_response: Optional[FactResponse]) -> str:
        """
        格式化事实检索结果为文本
        
        用于 tool result message 的 content。
        """
        if not fact_response or not fact_response.messages:
            return "未找到相关事实记录。"
        
        parts = []
        for msg in fact_response.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        
        result = "\n".join(parts)
        
        if fact_response.has_more:
            result += (
                f"\n(还有更多内容, 可继续调用 retrieve_fact, "
                f"trace_id=\"{fact_response.trace_id}\", "
                f"offset={fact_response.offset + len(fact_response.messages)})"
            )
        
        return result

    async def _generate_with_tool_calls(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        session_id: str,
        max_rounds: int = 3,
        max_fact_tokens: int = 800,
        **kwargs,
    ) -> Tuple[str, bool, ModelOutput]:
        """
        带 native tool_calls 的生成 (闭源模型)
        
        流程:
        1. 调用 API (tools=[retrieve_fact_tool_def])
        2. 响应 finish_reason == "tool_calls" → 解析参数 → 检索事实
        3. 拼接 tool result message → 再次调用 API
        4. 循环直到正常结束或达到 max_rounds
        
        Args:
            prompt: 格式化后的 prompt 字符串
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            session_id: 会话 ID
            max_rounds: 最大 tool_call 循环轮次
            max_fact_tokens: 事实 token 预算上限
            
        Returns:
            (clean_response, think_stripped, raw_output)
        """
        total_fact_tokens = 0
        messages = self._prompt_to_messages(prompt)
        
        for round_idx in range(max_rounds):
            # 调用 API (带 tools)
            try:
                output = await self._call_with_tools(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    tools=[RETRIEVE_FACT_TOOL_DEF],
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"Native tool_calls API call failed (round {round_idx + 1}): {e}, "
                    f"falling back to post_hoc"
                )
                # 降级: 不带 tools 重试
                output = await self._call_without_tools(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                clean_text, think_stripped = strip_think_content(output.text)
                return clean_text, think_stripped, output
            
            response_message = output.metadata.get("raw_message", {})
            finish_reason = output.metadata.get("finish_reason", "stop")
            
            # 正常结束 (非 tool_calls)
            if finish_reason != "tool_calls":
                clean_text, think_stripped = strip_think_content(output.text)
                return clean_text, think_stripped, output
            
            # 解析 tool_calls
            tool_calls = response_message.get("tool_calls", [])
            if not tool_calls:
                clean_text, think_stripped = strip_think_content(output.text)
                return clean_text, think_stripped, output
            
            # 将 assistant message (含 tool_calls) 加入 messages
            messages.append(response_message)
            
            # 逐个处理 tool_calls
            for tc in tool_calls:
                func = tc.get("function", {})
                if func.get("name") != "retrieve_fact":
                    # 未知 tool, 返回空结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "Unknown tool.",
                    })
                    continue
                
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                
                trace_id = args.get("trace_id", "")
                offset = args.get("offset", 0)
                limit = args.get("limit", 5)
                
                if not trace_id:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "Error: trace_id is required.",
                    })
                    continue
                
                # 检索事实
                fact_response = self._retrieve_fact_for_rag(trace_id, session_id)
                fact_text = self._format_fact_result(fact_response)
                fact_tokens = estimate_tokens_fast(fact_text)
                total_fact_tokens += fact_tokens
                
                # 拼接 tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": fact_text,
                })
                
                logger.info(
                    f"Native tool_calls round {round_idx + 1}: "
                    f"trace_id={trace_id}, fact_tokens={fact_tokens}, "
                    f"total_fact_tokens={total_fact_tokens}"
                )
            
            # 检查 fact token 预算
            if total_fact_tokens > max_fact_tokens:
                logger.info(
                    f"Native tool_calls: fact token budget exhausted "
                    f"({total_fact_tokens} > {max_fact_tokens})"
                )
                # 最后一轮: 不带 tools, 让模型正常回答
                final_output = await self._call_without_tools(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                clean_text, think_stripped = strip_think_content(final_output.text)
                return clean_text, think_stripped, final_output
        
        # 达到 max_rounds, 强制最后一轮不带 tools
        logger.info(
            f"Native tool_calls: max rounds reached ({max_rounds}), "
            f"forcing final generation without tools"
        )
        final_output = await self._call_without_tools(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        clean_text, think_stripped = strip_think_content(final_output.text)
        return clean_text, think_stripped, final_output

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
