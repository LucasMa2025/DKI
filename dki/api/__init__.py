"""
DKI API Module
Provides REST API endpoints for DKI system

This module provides:
- OpenAI-compatible chat completions API
- DKI-specific injection API
- User preference management API
- Health and stats endpoints
"""

from dki.api.routes import create_api_router
from dki.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DKIInjectRequest,
    DKIInjectResponse,
    PreferenceRequest,
    PreferenceResponse,
)
from dki.api.dependencies import (
    get_dki_system,
    get_user_adapter,
    get_preference_cache,
)

__all__ = [
    "create_api_router",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "DKIInjectRequest",
    "DKIInjectResponse",
    "PreferenceRequest",
    "PreferenceResponse",
    "get_dki_system",
    "get_user_adapter",
    "get_preference_cache",
]
