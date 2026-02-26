"""
DKI Service — 独立 API 服务模式

============================================================================
DKI 从插件到服务的演进
============================================================================

DKI 可以以两种形态存在:

┌─────────────────────────────────────────────────────────────────────────┐
│  形态 1: 插件模式 (DKIPlugin)                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  适用: LLM 应用直接集成 DKI                                             │
│  部署: DKI 与 LLM 在同一进程                                            │
│  接口: Python API (dki.chat())                                          │
│  延迟: 最低 (无网络开销)                                                 │
│  耦合: 高 (需要 ModelAdapter, vLLM 依赖)                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  形态 2: 服务模式 (DKI Service)       ← 本文件实现                       │
├─────────────────────────────────────────────────────────────────────────┤
│  适用: 多应用共享 DKI 能力, 或 DKI 与 LLM 分离部署                       │
│  部署: DKI 作为独立 HTTP/gRPC 服务                                       │
│  接口: REST API (/api/dki/prompt, /api/dki/callback)                     │
│  延迟: 额外 ~1-5ms 网络开销 (可忽略, 相对 LLM 推理的 1-10s)              │
│  耦合: 低 (DKI 不依赖 ModelAdapter)                                      │
└─────────────────────────────────────────────────────────────────────────┘

============================================================================
服务模式架构
============================================================================

    ┌─────────────┐         ┌──────────────────┐        ┌──────────────┐
    │  上层应用     │ ──(1)─→ │  DKI Service      │        │  LLM Engine  │
    │  (任意语言)   │         │  (FastAPI)        │        │  (vLLM/TGI)  │
    │             │ ←─(4)── │                   │        │              │
    └─────────────┘         └──────────────────┘        └──────────────┘
                                │       ↑
                           (2) prompt  (3) fact_callback
                                ↓       │
                            ┌───────────────┐
                            │  Database      │
                            │  (偏好+历史)    │
                            └───────────────┘

    流程:
    (1) 上层应用 POST /api/dki/prompt 
        → 传入 query, user_id, session_id
        ← 返回 DKI 增强后的完整 prompt

    (2) 上层应用拿到 prompt, 自行调用 LLM Engine 推理
    
    (3) 如果 LLM 输出包含 function call (事实检索请求),
        上层应用 POST /api/dki/callback
        ← 返回事实数据
    
    (4) 上层应用将最终结果返回用户

============================================================================
核心价值
============================================================================

1. DKI 的核心功能是 **提示词生成** (偏好注入 + 历史召回 + 后缀组装)
   这一步不需要 GPU, 不需要 ModelAdapter, 是纯 CPU 操作。

2. DKI 生成的 prompt 可以被任何 LLM 使用:
   - vLLM, TGI, Ollama, OpenAI API, Claude API...
   - 不再绑定到特定模型适配器

3. Function Call 回调是可选的:
   - 只有当 LLM 输出包含 retrieve_fact(...) 时才需要
   - 上层应用可以选择忽略或替换为自己的实现

4. 偏好和历史管理 API 也可以独立暴露:
   - POST /api/dki/preferences — 管理偏好
   - GET /api/dki/history — 查询历史

============================================================================
Author: AGI Demo Project
Version: 1.0.0
============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.core.text_utils import strip_think_content

# DKI 核心组件 (不依赖 ModelAdapter)
from dki.database.connection import DatabaseManager
from dki.database.repository import (
    ConversationRepository,
    UserPreferenceRepository,
    SessionRepository,
)
from dki.config.config_loader import ConfigLoader

# Recall v4 组件 (可选)
try:
    from dki.core.recall import (
        RecallConfig,
        MultiSignalRecall,
        SuffixBuilder,
        FactRetriever,
    )
    from dki.core.recall.recall_config import FactRequest, HistoryItem, AssembledSuffix
    RECALL_V4_AVAILABLE = True
except ImportError:
    RECALL_V4_AVAILABLE = False

# 记忆触发器 (可选)
try:
    from dki.core.components.memory_trigger import MemoryTrigger, MemoryTriggerConfig
    TRIGGER_AVAILABLE = True
except ImportError:
    TRIGGER_AVAILABLE = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DKIPromptRequest:
    """DKI 提示词生成请求"""
    query: str                          # 用户原始输入
    user_id: str                        # 用户 ID
    session_id: str                     # 会话 ID
    system_prompt: Optional[str] = None # 自定义 system prompt
    max_history_tokens: int = 2000      # 历史最大 token 数
    include_preferences: bool = True    # 是否注入偏好
    include_history: bool = True        # 是否注入历史
    chat_template: str = "chatml"       # 模板格式: chatml, deepseek, llama3, raw


@dataclass
class DKIPromptResponse:
    """DKI 提示词生成响应"""
    prompt: str                         # 完整的 prompt (可直接发给 LLM)
    messages: List[Dict[str, str]]      # 结构化 messages (可用于 OpenAI API)
    
    # 元数据
    preference_injected: bool = False
    preference_text: str = ""
    history_injected: bool = False
    history_token_count: int = 0
    history_message_count: int = 0
    recall_strategy: str = ""
    trace_ids: List[str] = field(default_factory=list)
    has_fact_call_instruction: bool = False
    
    # 性能
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactCallbackRequest:
    """事实检索回调请求"""
    trace_id: str                       # 溯源 ID (来自 LLM 输出的 retrieve_fact 调用)
    query: Optional[str] = None         # 事实查询 (可选, 如果 LLM 指定了)
    session_id: Optional[str] = None    # 会话 ID (用于定位原文)
    offset: int = 0
    limit: int = 5


@dataclass
class FactCallbackResponse:
    """事实检索回调响应"""
    messages: List[Dict[str, str]]      # 原文消息列表
    trace_id: str = ""
    total_count: int = 0
    has_more: bool = False


# ============================================================
# DKI Service 核心
# ============================================================

class DKIService:
    """
    DKI 独立服务 — 提供提示词生成和事实回调 API
    
    不依赖 ModelAdapter, 不需要 GPU。
    核心功能:
    1. 加载用户偏好
    2. 多信号召回历史消息
    3. 后缀组装
    4. 提示词生成 (支持多种 chat template)
    5. 事实回调 (Function Call)
    """
    
    def __init__(self):
        self.config = ConfigLoader().config
        
        # 数据库
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
            echo=self.config.database.echo,
        )
        
        # Recall v4 (如果可用)
        self._recall_config: Optional[RecallConfig] = None
        self._multi_signal_recall: Optional[MultiSignalRecall] = None
        self._suffix_builder: Optional[SuffixBuilder] = None
        self._fact_retriever: Optional[FactRetriever] = None
        
        if RECALL_V4_AVAILABLE:
            recall_dict = {}
            if hasattr(self.config.dki, 'recall'):
                recall_attr = self.config.dki.recall
                recall_dict = recall_attr.__dict__ if hasattr(recall_attr, '__dict__') else {}
            self._recall_config = RecallConfig.from_dict(recall_dict)
            
            if self._recall_config.enabled:
                self._init_recall_v4()
        
        logger.info(
            f"DKI Service initialized: "
            f"recall_v4={'enabled' if self._multi_signal_recall else 'disabled'}"
        )
    
    def _init_recall_v4(self):
        """初始化 Recall v4 组件"""
        try:
            self._multi_signal_recall = MultiSignalRecall(
                config=self._recall_config,
                db_manager=self.db_manager,
            )
            self._suffix_builder = SuffixBuilder(
                config=self._recall_config,
            )
            self._fact_retriever = FactRetriever(
                db_manager=self.db_manager,
            )
            logger.info("Recall v4 components initialized for DKI Service")
        except Exception as e:
            logger.warning(f"Failed to initialize Recall v4: {e}")
    
    def generate_prompt(self, request: DKIPromptRequest) -> DKIPromptResponse:
        """
        生成 DKI 增强的提示词
        
        这是 DKI Service 的核心方法。它:
        1. 加载用户偏好
        2. 通过 Recall v4 召回相关历史
        3. 组装后缀
        4. 按指定 chat template 格式化完整 prompt
        
        返回的 prompt 可以直接发送给任何 LLM。
        """
        start_time = time.perf_counter()
        
        messages = []  # 结构化消息列表
        preference_text = ""
        preference_injected = False
        history_injected = False
        history_token_count = 0
        history_message_count = 0
        recall_strategy = ""
        trace_ids = []
        has_fact_call_instruction = False
        
        # ---- 1. 加载偏好 ----
        if request.include_preferences:
            preference_text = self._load_preferences(request.user_id)
            if preference_text:
                preference_injected = True
        
        # ---- 2. 构建 system message ----
        system_content = ""
        if request.system_prompt:
            system_content = request.system_prompt
        if preference_text:
            pref_section = f"请严格遵循以下用户偏好:\n{preference_text}"
            if system_content:
                system_content = f"{system_content}\n\n{pref_section}"
            else:
                system_content = pref_section
        
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # ---- 3. 召回历史 ----
        if request.include_history and self._multi_signal_recall:
            try:
                recalled_items, meta = self._recall_history(
                    query=request.query,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    max_tokens=request.max_history_tokens,
                )
                
                if recalled_items:
                    for item in recalled_items:
                        role = getattr(item, 'role', 'user') or 'user'
                        content = getattr(item, 'content', '')
                        if content.strip():
                            messages.append({"role": role, "content": content})
                    
                    history_injected = True
                    history_message_count = len(recalled_items)
                    history_token_count = meta.get("total_tokens", 0)
                    recall_strategy = meta.get("strategy", "")
                    trace_ids = meta.get("trace_ids", [])
                    has_fact_call_instruction = meta.get("has_fact_call_instruction", False)
                    
            except Exception as e:
                logger.warning(f"History recall failed: {e}")
        
        # ---- 4. 添加当前查询 ----
        messages.append({"role": "user", "content": request.query})
        
        # ---- 5. 格式化 prompt ----
        prompt = self._format_prompt(messages, request.chat_template)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return DKIPromptResponse(
            prompt=prompt,
            messages=messages,
            preference_injected=preference_injected,
            preference_text=preference_text,
            history_injected=history_injected,
            history_token_count=history_token_count,
            history_message_count=history_message_count,
            recall_strategy=recall_strategy,
            trace_ids=trace_ids,
            has_fact_call_instruction=has_fact_call_instruction,
            latency_ms=latency_ms,
        )
    
    def handle_fact_callback(self, request: FactCallbackRequest) -> FactCallbackResponse:
        """
        处理事实回调 (Function Call)
        
        当 LLM 输出包含 retrieve_fact(trace_id=...) 时,
        上层应用调用此方法获取原文。
        """
        if not self._fact_retriever:
            return FactCallbackResponse(
                messages=[],
                trace_id=request.trace_id,
                total_count=0,
            )
        
        try:
            fact_req = FactRequest(
                trace_id=request.trace_id,
                query=request.query,
                offset=request.offset,
                limit=request.limit,
            )
            
            fact_resp = self._fact_retriever.retrieve(fact_req)
            
            return FactCallbackResponse(
                messages=[
                    {"role": msg.get("role", "unknown"), "content": msg.get("content", "")}
                    for msg in fact_resp.messages
                ],
                trace_id=fact_resp.trace_id,
                total_count=fact_resp.total_count,
                has_more=fact_resp.has_more,
            )
        except Exception as e:
            logger.error(f"Fact callback failed: {e}")
            return FactCallbackResponse(
                messages=[],
                trace_id=request.trace_id,
            )
    
    def log_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        记录对话消息 (供后续召回使用)
        
        上层应用在拿到 LLM 回复后, 应调用此方法记录 assistant 消息。
        user 消息在 generate_prompt 时已自动记录。
        """
        try:
            with self.db_manager.session_scope() as db:
                conv_repo = ConversationRepository(db)
                conv_repo.create(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                )
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    # ---- 私有方法 ----
    
    def _load_preferences(self, user_id: str) -> str:
        """加载用户偏好文本"""
        if not user_id:
            return ""
        
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                prefs = pref_repo.get_by_user(user_id, active_only=True)
                
                if not prefs:
                    return ""
                
                texts = []
                for p in prefs:
                    text = getattr(p, 'preference_text', '') or ''
                    if text.strip():
                        texts.append(text.strip())
                
                return "\n".join(texts) if texts else ""
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
            return ""
    
    def _recall_history(
        self,
        query: str,
        session_id: str,
        user_id: str,
        max_tokens: int = 2000,
    ) -> tuple:
        """
        使用 Recall v4 召回历史
        
        Returns:
            (items: List[HistoryItem], meta: dict)
        """
        # Phase 1: 多信号召回
        recall_result = self._multi_signal_recall.recall(
            query=query,
            session_id=session_id,
            user_id=user_id,
        )
        
        if not recall_result.messages:
            return [], {"total_tokens": 0}
        
        # Phase 2: 后缀组装
        assembled = self._suffix_builder.build(
            query=query,
            recalled_messages=recall_result.messages,
            context_window=max_tokens + 500,  # 给 query 和 prompt 留余量
            preference_tokens=0,
        )
        
        meta = {
            "total_tokens": assembled.total_tokens,
            "strategy": self._recall_config.strategy if self._recall_config else "",
            "trace_ids": assembled.trace_ids,
            "has_fact_call_instruction": assembled.has_fact_call_instruction,
        }
        
        return assembled.items, meta
    
    def _format_prompt(
        self,
        messages: List[Dict[str, str]],
        template: str = "chatml",
    ) -> str:
        """
        格式化 prompt
        
        支持多种 chat template:
        - chatml: ChatML 格式 (Qwen/DeepSeek/通用)
        - deepseek: DeepSeek 专用格式
        - llama3: Llama 3 格式
        - raw: 原始文本 (不加特殊标记)
        """
        if template == "raw":
            parts = []
            for msg in messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            return "\n\n".join(parts)
        
        if template == "deepseek":
            return self._format_deepseek(messages)
        
        if template == "llama3":
            return self._format_llama3(messages)
        
        # 默认: ChatML (兼容 Qwen/DeepSeek/通用)
        return self._format_chatml(messages)
    
    def _format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """
        ChatML 格式 (Qwen/DeepSeek 通用)
        
        标准格式 (半角符号, 标签闭合):
            <|im_start|>system
            {system_content}<|im_end|>
            <|im_start|>user
            {user_content}<|im_end|>
            <|im_start|>assistant
        """
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts) + "\n"
    
    def _format_deepseek(self, messages: List[Dict[str, str]]) -> str:
        """
        DeepSeek 格式 (ChatML 兼容)
        
        DeepSeek V2/V3 tokenizer 在 apply_chat_template 中使用全角 ｜ 和 ▁,
        但手动回退时统一使用 ChatML 半角格式, 因为:
        1. DeepSeek 官方支持 ChatML 格式
        2. 半角标记更通用且不依赖特殊 Unicode 字符
        3. 确保标签闭合一致性
        
        标准格式 (半角符号, 标签闭合):
            <|im_start|>system
            {system_content}<|im_end|>
            <|im_start|>user
            {user_content}<|im_end|>
            <|im_start|>assistant
        """
        # DeepSeek 兼容 ChatML, 统一使用半角标准格式
        return self._format_chatml(messages)
    
    def _format_llama3(self, messages: List[Dict[str, str]]) -> str:
        """
        Llama 3 格式
        
        标准格式 (半角符号, 标签闭合):
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_content}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_content}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            parts.append(
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
