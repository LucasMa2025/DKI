"""
Hybrid DKI Injector - Layered Memory Injection Strategy

Implements the hybrid injection approach from DKI Paper Section 3.9:
- User Preferences: Negative-position K/V injection (short, stable, implicit)
- Session History: Suffix-prompt injection (longer, dynamic, explicit)

This layered approach:
1. Minimizes OOD risk (preferences are short, history in positive positions)
2. Maximizes flexibility (history can be long, preferences cached)
3. Enables explicit citation (history visible in prompt)
4. Reduces hallucination (prompt establishes trust)

Information Layering (mirrors human cognition):
┌─────────────────────────────────────────────────────────────┐
│  Layer   │  Content     │  Injection        │  Analogy      │
├──────────┼──────────────┼───────────────────┼───────────────┤
│  L1      │  Preferences │  K/V (negative)   │  Personality  │
│  L2      │  History     │  Suffix (positive)│  Memory       │
│  L3      │  Query       │  Input (positive) │  Current      │
└─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
from loguru import logger


@dataclass
class UserPreference:
    """User preference data structure."""
    content: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Cached K/V (computed once, reused across sessions)
    kv_cache: Optional[Any] = None
    token_count: int = 0


@dataclass
class SessionMessage:
    """Single message in session history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class SessionHistory:
    """Session history container."""
    messages: List[SessionMessage] = field(default_factory=list)
    session_id: str = ""
    max_tokens: int = 500
    
    def add_message(self, role: str, content: str, timestamp: Optional[float] = None):
        self.messages.append(SessionMessage(role=role, content=content, timestamp=timestamp))
    
    def get_recent(self, max_messages: int = 10) -> List[SessionMessage]:
        """Get most recent messages."""
        return self.messages[-max_messages:] if self.messages else []


@dataclass
class HybridInjectionConfig:
    """Configuration for hybrid injection strategy."""
    # Preference injection settings
    preference_enabled: bool = True
    preference_alpha: float = 0.4  # Lower α for background influence
    preference_max_tokens: int = 100
    preference_position_strategy: str = "negative"  # negative | actual_prefix
    
    # History injection settings
    history_enabled: bool = True
    history_max_tokens: int = 500
    history_max_messages: int = 10
    history_method: str = "suffix_prompt"  # suffix_prompt | kv_injection
    
    # Prompt templates
    history_prefix_template: str = ""
    history_suffix_template: str = ""


@dataclass
class HybridInjectionResult:
    """Result of hybrid injection preparation."""
    input_text: str
    preference_kv: Optional[Any] = None
    preference_alpha: float = 0.0
    preference_tokens: int = 0
    history_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增: 用于显示的明文信息
    preference_text: str = ""  # 偏好原文
    history_suffix_text: str = ""  # 历史后缀原文
    history_messages: List[Dict[str, str]] = field(default_factory=list)  # 历史消息列表


class HybridDKIInjector:
    """
    Hybrid DKI Injector implementing layered memory injection.
    
    Strategy:
    - Preferences → Negative-position K/V injection (implicit influence)
    - History → Suffix-prompt injection (explicit reference)
    
    This design:
    - Minimizes OOD risk for preferences (short content, safe positions)
    - Enables history citation (visible in prompt)
    - Supports caching (preference K/V reusable)
    - Reduces hallucination (trust-establishing prompts)
    """
    
    # Default prompt templates (Chinese + English)
    DEFAULT_HISTORY_PREFIX_EN = """
[Session History Reference]
Before responding, please refer to the following session history.
These are real conversation records between you and the user, and are trustworthy.
Please provide a coherent response after understanding the historical context.
---
"""
    
    DEFAULT_HISTORY_SUFFIX_EN = """
---
[End of Session History]
Please respond based on the above history and the user's current question.
Note: Historical information is for reference; please answer comprehensively.
"""
    
    DEFAULT_HISTORY_PREFIX_CN = """
[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
请在理解历史上下文后，给出连贯的整体回复。
重要：请使用中文回复用户。
---
"""
    
    DEFAULT_HISTORY_SUFFIX_CN = """
---
[会话历史结束]
请基于以上历史和用户当前问题，使用中文给出回复。
注意：历史信息仅供参考，请综合回答。
"""
    
    def __init__(
        self,
        config: Optional[HybridInjectionConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        language: str = "en",  # "en" or "cn"
    ):
        self.config = config or HybridInjectionConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.language = language
        
        # Set default templates based on language
        if not self.config.history_prefix_template:
            self.config.history_prefix_template = (
                self.DEFAULT_HISTORY_PREFIX_CN if language == "cn" 
                else self.DEFAULT_HISTORY_PREFIX_EN
            )
        if not self.config.history_suffix_template:
            self.config.history_suffix_template = (
                self.DEFAULT_HISTORY_SUFFIX_CN if language == "cn"
                else self.DEFAULT_HISTORY_SUFFIX_EN
            )
        
        # Preference K/V cache (user_id -> kv_cache)
        self._preference_cache: Dict[str, Any] = {}
        
        logger.info(f"HybridDKIInjector initialized (language={language})")
    
    def prepare_input(
        self,
        user_query: str,
        preference: Optional[UserPreference] = None,
        history: Optional[SessionHistory] = None,
        system_prompt: str = "",
    ) -> HybridInjectionResult:
        """
        Prepare hybrid injection input.
        
        Args:
            user_query: Current user query
            preference: User preference (for K/V injection)
            history: Session history (for suffix prompt)
            system_prompt: Optional system prompt
            
        Returns:
            HybridInjectionResult with prepared input and K/V cache
        """
        result = HybridInjectionResult(
            input_text="",
            metadata={"query": user_query}
        )
        
        # Build text input parts
        # 正确的顺序: System prompt → History → User query
        # 这样模型先看到历史上下文，然后回答当前问题
        text_parts = []
        
        # 1. System prompt (if provided)
        if system_prompt:
            text_parts.append(system_prompt)
        
        # 2. History as prefix/context (if enabled)
        # 历史消息应该在用户查询之前，作为上下文
        history_text = ""
        if self.config.history_enabled and history and history.messages:
            history_text = self._format_history(history)
            if history_text:
                text_parts.append(history_text)
                result.history_tokens = self._estimate_tokens(history_text)
                # 保存用于显示的历史信息
                result.history_suffix_text = history_text
                result.history_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in history.get_recent(self.config.history_max_messages)
                ]
        
        # 3. User query (当前问题)
        text_parts.append(f"User: {user_query}")
        
        result.input_text = "\n\n".join(text_parts)
        result.total_tokens = self._estimate_tokens(result.input_text)
        
        # 4. Preference K/V (if enabled)
        if self.config.preference_enabled and preference and preference.content:
            kv_cache = self._get_or_compute_preference_kv(preference)
            if kv_cache is not None:
                result.preference_kv = kv_cache
                result.preference_alpha = self.config.preference_alpha
                result.preference_tokens = preference.token_count
                result.metadata["preference_cached"] = preference.kv_cache is not None
            # 保存用于显示的偏好文本 (无论 K/V 是否成功计算)
            result.preference_text = preference.content
        
        logger.debug(
            f"Hybrid input prepared: "
            f"pref_tokens={result.preference_tokens}, "
            f"history_tokens={result.history_tokens}, "
            f"total_tokens={result.total_tokens}"
        )
        
        return result
    
    def _format_history(self, history: SessionHistory) -> str:
        """
        Format session history with trust-establishing prompts.
        
        The prompt template:
        1. Establishes trust ("real conversation records, trustworthy")
        2. Guides processing ("understand context, then respond")
        3. Sets expectations ("comprehensive answer")
        """
        if not history.messages:
            return ""
        
        # Get recent messages (respect max_messages)
        recent = history.get_recent(self.config.history_max_messages)
        
        # Format messages
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            if self.language == "cn":
                role_label = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_label}: {msg.content}")
        
        history_content = "\n".join(lines)
        
        # Truncate if too long
        if self._estimate_tokens(history_content) > self.config.history_max_tokens:
            # Simple truncation - keep most recent
            while (self._estimate_tokens(history_content) > self.config.history_max_tokens 
                   and len(lines) > 1):
                lines.pop(0)
                history_content = "\n".join(lines)
        
        # Wrap with prompt template
        return (
            self.config.history_prefix_template +
            history_content +
            self.config.history_suffix_template
        )
    
    def _get_or_compute_preference_kv(
        self,
        preference: UserPreference,
    ) -> Optional[Any]:
        """
        Get or compute K/V cache for user preference.
        
        Preferences are ideal for caching because:
        1. Short content (typically < 100 tokens)
        2. Stable over time (rarely changes)
        3. Reused across sessions
        """
        # Check if already cached
        if preference.kv_cache is not None:
            return preference.kv_cache
        
        # Check instance cache
        cache_key = f"{preference.user_id}:{hash(preference.content)}"
        if cache_key in self._preference_cache:
            return self._preference_cache[cache_key]
        
        # Compute K/V if model available
        if self.model is None:
            logger.warning("Model not set, cannot compute preference K/V")
            return None
        
        try:
            with torch.no_grad():
                # Tokenize preference
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(
                        preference.content, 
                        return_tensors="pt"
                    )
                    preference.token_count = tokens.shape[1]
                else:
                    # Estimate token count
                    preference.token_count = int(len(preference.content.split()) * 1.3)
                
                # Compute K/V through model
                if hasattr(self.model, 'compute_kv'):
                    kv_entries, _ = self.model.compute_kv(preference.content)
                    preference.kv_cache = kv_entries
                    self._preference_cache[cache_key] = kv_entries
                    return kv_entries
                elif hasattr(self.model, 'forward') and hasattr(self.model, 'model'):
                    # HuggingFace style
                    outputs = self.model.model(
                        input_ids=tokens,
                        use_cache=True,
                        return_dict=True,
                    )
                    kv_cache = outputs.past_key_values
                    preference.kv_cache = kv_cache
                    self._preference_cache[cache_key] = kv_cache
                    return kv_cache
                else:
                    logger.warning("Model does not support K/V computation")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to compute preference K/V: {e}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Rough estimate: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """Clear preference K/V cache."""
        if user_id:
            keys_to_remove = [k for k in self._preference_cache if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._preference_cache[key]
        else:
            self._preference_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get injector statistics."""
        return {
            "preference_cache_size": len(self._preference_cache),
            "config": {
                "preference_enabled": self.config.preference_enabled,
                "preference_alpha": self.config.preference_alpha,
                "history_enabled": self.config.history_enabled,
                "history_max_tokens": self.config.history_max_tokens,
                "language": self.language,
            }
        }


# Convenience functions for common use cases

def create_hybrid_injector(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    preference_alpha: float = 0.4,
    history_max_tokens: int = 500,
    language: str = "en",
) -> HybridDKIInjector:
    """
    Create a hybrid injector with common settings.
    
    Args:
        model: LLM model for K/V computation
        tokenizer: Tokenizer for token counting
        preference_alpha: Injection strength for preferences (0.3-0.5 recommended)
        history_max_tokens: Maximum tokens for history
        language: "en" or "cn" for prompt templates
        
    Returns:
        Configured HybridDKIInjector
    """
    config = HybridInjectionConfig(
        preference_enabled=True,
        preference_alpha=preference_alpha,
        history_enabled=True,
        history_max_tokens=history_max_tokens,
    )
    
    return HybridDKIInjector(
        config=config,
        model=model,
        tokenizer=tokenizer,
        language=language,
    )
