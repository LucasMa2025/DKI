"""
API Request/Response Models
Pydantic models for DKI API endpoints

Author: AGI Demo Project
Version: 1.0.0
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ============ OpenAI Compatible Models ============

class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    
    Extended with DKI-specific parameters.
    """
    # Standard OpenAI parameters
    model: str = "default"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=512, ge=1, le=32768)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    # DKI Extension parameters
    dki_enabled: bool = Field(default=True, description="Enable DKI injection")
    dki_user_id: Optional[str] = Field(default=None, description="User ID for preference lookup")
    dki_session_id: Optional[str] = Field(default=None, description="Session ID for history lookup")
    dki_force_alpha: Optional[float] = Field(default=None, ge=0, le=1, description="Force injection strength")
    dki_use_hybrid: Optional[bool] = Field(default=None, description="Use hybrid injection strategy")
    
    class Config:
        extra = "allow"  # Allow additional fields


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class DKIMetadata(BaseModel):
    """DKI-specific metadata in response."""
    injection_enabled: bool = False
    alpha: Optional[float] = None
    memories_used: int = 0
    preference_tokens: int = 0
    history_tokens: int = 0
    cache_hit: bool = False
    cache_tier: Optional[str] = None
    latency_ms: float = 0
    gating_decision: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    
    Extended with DKI metadata.
    """
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    
    # DKI Extension
    dki_metadata: Optional[DKIMetadata] = None
    
    class Config:
        extra = "allow"


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


# ============ DKI Specific Models ============

class DKIInjectRequest(BaseModel):
    """
    Direct DKI injection request.
    
    Allows external systems to directly call DKI injection
    without going through OpenAI-compatible interface.
    """
    query: str = Field(..., description="User query")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Optional: directly provide data instead of fetching
    preferences: Optional[str] = Field(default=None, description="Direct preference text")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Direct history messages")
    
    # Injection parameters
    force_alpha: Optional[float] = Field(default=None, ge=0, le=1)
    use_hybrid: bool = True
    
    # Generation parameters
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    
    class Config:
        extra = "allow"


class DKIInjectResponse(BaseModel):
    """Response from direct DKI injection."""
    response: str
    dki_metadata: DKIMetadata
    
    # Additional info
    request_id: str = Field(default_factory=lambda: f"dki-{uuid.uuid4().hex[:16]}")
    created: int = Field(default_factory=lambda: int(time.time()))


# ============ Preference Management Models ============

class PreferenceItem(BaseModel):
    """Single user preference."""
    preference_id: Optional[str] = None
    preference_text: str
    preference_type: str = "custom"
    priority: int = 0
    category: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreferenceRequest(BaseModel):
    """Request to update user preferences."""
    user_id: str
    preferences: List[PreferenceItem]
    replace_all: bool = Field(default=False, description="Replace all existing preferences")


class PreferenceResponse(BaseModel):
    """Response from preference update."""
    user_id: str
    updated_count: int
    cache_invalidated: bool
    preferences: List[PreferenceItem]


class PreferenceListResponse(BaseModel):
    """Response listing user preferences."""
    user_id: str
    preferences: List[PreferenceItem]
    total_count: int


# ============ Session History Models ============

class SessionMessage(BaseModel):
    """Message in session history."""
    message_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    """Response with session history."""
    session_id: str
    user_id: Optional[str]
    messages: List[SessionMessage]
    total_count: int


# ============ Health and Stats Models ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    components: Dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """System statistics response."""
    dki_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    adapter_stats: Dict[str, Any]
    uptime_seconds: float


# ============ Error Models ============

class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str
    param: Optional[str] = None
    type: str = "invalid_request_error"


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail
