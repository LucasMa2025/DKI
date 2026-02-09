"""
API Routes
FastAPI routes for DKI API endpoints

Author: AGI Demo Project
Version: 1.0.0
"""

import time
import uuid
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from dki.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
    DKIMetadata,
    DKIInjectRequest,
    DKIInjectResponse,
    PreferenceRequest,
    PreferenceResponse,
    PreferenceItem,
    PreferenceListResponse,
    SessionHistoryResponse,
    SessionMessage,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    ErrorDetail,
)
from dki.api.dependencies import (
    get_dki_system,
    get_user_adapter,
    get_preference_cache,
    get_startup_time,
)


def create_api_router() -> APIRouter:
    """Create and configure API router with all endpoints."""
    
    router = APIRouter()
    
    # ============ OpenAI Compatible Endpoints ============
    
    @router.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        tags=["OpenAI Compatible"],
        summary="Chat Completions",
        description="OpenAI-compatible chat completions endpoint with DKI injection",
    )
    async def chat_completions(
        request: ChatCompletionRequest,
        dki_system=Depends(get_dki_system),
        user_adapter=Depends(get_user_adapter),
        preference_cache=Depends(get_preference_cache),
    ):
        """
        Generate chat completion with optional DKI injection.
        
        This endpoint is compatible with OpenAI's chat completions API
        and extends it with DKI-specific parameters.
        """
        try:
            # Extract user query from messages
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                raise HTTPException(
                    status_code=400,
                    detail="No user message found in request"
                )
            
            query = user_messages[-1].content
            
            # Build system prompt from system messages
            system_messages = [m for m in request.messages if m.role == "system"]
            system_prompt = "\n".join(m.content for m in system_messages) if system_messages else ""
            
            # Generate session ID if not provided
            session_id = request.dki_session_id or f"session_{uuid.uuid4().hex[:8]}"
            user_id = request.dki_user_id or request.user
            
            # Check if DKI is enabled
            if request.dki_enabled and user_id:
                # Fetch user preferences if adapter available
                preferences_text = None
                if user_adapter:
                    try:
                        preferences = await user_adapter.get_user_preferences(user_id)
                        if preferences:
                            preferences_text = "\n".join(
                                f"- {p.preference_type}: {p.preference_text}"
                                for p in preferences
                            )
                    except Exception as e:
                        logger.warning(f"Failed to fetch preferences: {e}")
                
                # Set user preference in DKI system
                if preferences_text:
                    dki_system.set_user_preference(
                        user_id=user_id,
                        preference_text=preferences_text,
                    )
                
                # Generate with DKI
                response = dki_system.chat(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    force_alpha=request.dki_force_alpha,
                    use_hybrid=request.dki_use_hybrid,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature,
                )
                
                # Build DKI metadata
                dki_metadata = DKIMetadata(
                    injection_enabled=response.gating_decision.should_inject,
                    alpha=response.gating_decision.alpha,
                    memories_used=len(response.memories_used),
                    preference_tokens=response.metadata.get("hybrid_injection", {}).get("preference_tokens", 0),
                    history_tokens=response.metadata.get("hybrid_injection", {}).get("history_tokens", 0),
                    cache_hit=response.cache_hit,
                    cache_tier=response.cache_tier,
                    latency_ms=response.latency_ms,
                    gating_decision={
                        "should_inject": response.gating_decision.should_inject,
                        "relevance_score": response.gating_decision.relevance_score,
                        "entropy": response.gating_decision.entropy,
                        "reasoning": response.gating_decision.reasoning,
                    },
                )
                
                generated_text = response.text
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                
            else:
                # Generate without DKI
                output = dki_system.model.generate(
                    prompt=query,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature,
                )
                
                generated_text = output.text
                input_tokens = output.input_tokens
                output_tokens = output.output_tokens
                dki_metadata = DKIMetadata(injection_enabled=False)
            
            # Build response
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=generated_text,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                dki_metadata=dki_metadata,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============ DKI Specific Endpoints ============
    
    @router.post(
        "/v1/dki/inject",
        response_model=DKIInjectResponse,
        tags=["DKI"],
        summary="Direct DKI Injection",
        description="Direct DKI injection endpoint for external systems",
    )
    async def dki_inject(
        request: DKIInjectRequest,
        dki_system=Depends(get_dki_system),
        user_adapter=Depends(get_user_adapter),
    ):
        """
        Direct DKI injection without OpenAI-compatible wrapper.
        
        Allows external systems to directly call DKI injection
        with full control over parameters.
        """
        try:
            # Get preferences (from request or adapter)
            preferences_text = request.preferences
            if preferences_text is None and user_adapter:
                try:
                    preferences = await user_adapter.get_user_preferences(request.user_id)
                    if preferences:
                        preferences_text = "\n".join(
                            f"- {p.preference_type}: {p.preference_text}"
                            for p in preferences
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch preferences: {e}")
            
            # Set user preference
            if preferences_text:
                dki_system.set_user_preference(
                    user_id=request.user_id,
                    preference_text=preferences_text,
                )
            
            # Generate with DKI
            response = dki_system.chat(
                query=request.query,
                session_id=request.session_id,
                user_id=request.user_id,
                force_alpha=request.force_alpha,
                use_hybrid=request.use_hybrid,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            # Build response
            return DKIInjectResponse(
                response=response.text,
                dki_metadata=DKIMetadata(
                    injection_enabled=response.gating_decision.should_inject,
                    alpha=response.gating_decision.alpha,
                    memories_used=len(response.memories_used),
                    preference_tokens=response.metadata.get("hybrid_injection", {}).get("preference_tokens", 0),
                    history_tokens=response.metadata.get("hybrid_injection", {}).get("history_tokens", 0),
                    cache_hit=response.cache_hit,
                    cache_tier=response.cache_tier,
                    latency_ms=response.latency_ms,
                    gating_decision={
                        "should_inject": response.gating_decision.should_inject,
                        "relevance_score": response.gating_decision.relevance_score,
                        "entropy": response.gating_decision.entropy,
                        "reasoning": response.gating_decision.reasoning,
                    },
                ),
            )
            
        except Exception as e:
            logger.error(f"DKI inject error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============ Preference Management Endpoints ============
    
    @router.get(
        "/v1/dki/preferences/{user_id}",
        response_model=PreferenceListResponse,
        tags=["Preferences"],
        summary="Get User Preferences",
    )
    async def get_preferences(
        user_id: str,
        preference_types: Optional[str] = Query(default=None, description="Comma-separated types"),
        user_adapter=Depends(get_user_adapter),
    ):
        """Get all preferences for a user."""
        try:
            types_list = preference_types.split(",") if preference_types else None
            
            preferences = await user_adapter.get_user_preferences(
                user_id=user_id,
                preference_types=types_list,
            )
            
            return PreferenceListResponse(
                user_id=user_id,
                preferences=[
                    PreferenceItem(
                        preference_id=p.preference_id,
                        preference_text=p.preference_text,
                        preference_type=p.preference_type,
                        priority=p.priority,
                        category=p.category,
                        expires_at=p.expires_at,
                        metadata=p.metadata,
                    )
                    for p in preferences
                ],
                total_count=len(preferences),
            )
            
        except Exception as e:
            logger.error(f"Get preferences error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post(
        "/v1/dki/preferences/{user_id}",
        response_model=PreferenceResponse,
        tags=["Preferences"],
        summary="Update User Preferences",
    )
    async def update_preferences(
        user_id: str,
        request: PreferenceRequest,
        user_adapter=Depends(get_user_adapter),
        preference_cache=Depends(get_preference_cache),
    ):
        """Update or create user preferences."""
        try:
            from dki.adapters.base import UserPreference as AdapterPreference
            
            updated_count = 0
            
            for pref_item in request.preferences:
                pref = AdapterPreference(
                    user_id=user_id,
                    preference_id=pref_item.preference_id,
                    preference_text=pref_item.preference_text,
                    preference_type=pref_item.preference_type,
                    priority=pref_item.priority,
                    category=pref_item.category,
                    expires_at=pref_item.expires_at,
                    metadata=pref_item.metadata,
                )
                
                if await user_adapter.update_user_preference(pref):
                    updated_count += 1
            
            # Invalidate cache
            cache_invalidated = False
            if preference_cache:
                await preference_cache.invalidate(user_id)
                cache_invalidated = True
            
            return PreferenceResponse(
                user_id=user_id,
                updated_count=updated_count,
                cache_invalidated=cache_invalidated,
                preferences=request.preferences,
            )
            
        except Exception as e:
            logger.error(f"Update preferences error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete(
        "/v1/dki/preferences/{user_id}/{preference_id}",
        tags=["Preferences"],
        summary="Delete User Preference",
    )
    async def delete_preference(
        user_id: str,
        preference_id: str,
        user_adapter=Depends(get_user_adapter),
        preference_cache=Depends(get_preference_cache),
    ):
        """Delete a specific user preference."""
        try:
            success = await user_adapter.delete_user_preference(user_id, preference_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Preference not found")
            
            # Invalidate cache
            if preference_cache:
                await preference_cache.invalidate(user_id)
            
            return {"status": "deleted", "preference_id": preference_id}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete preference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============ Session History Endpoints ============
    
    @router.get(
        "/v1/dki/sessions/{session_id}/history",
        response_model=SessionHistoryResponse,
        tags=["Sessions"],
        summary="Get Session History",
    )
    async def get_session_history(
        session_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        user_adapter=Depends(get_user_adapter),
    ):
        """Get conversation history for a session."""
        try:
            messages = await user_adapter.get_session_history(
                session_id=session_id,
                limit=limit,
            )
            
            return SessionHistoryResponse(
                session_id=session_id,
                user_id=messages[0].user_id if messages else None,
                messages=[
                    SessionMessage(
                        message_id=m.message_id,
                        role=m.role,
                        content=m.content,
                        timestamp=m.timestamp,
                        metadata=m.metadata,
                    )
                    for m in messages
                ],
                total_count=len(messages),
            )
            
        except Exception as e:
            logger.error(f"Get session history error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============ Health and Stats Endpoints ============
    
    @router.get(
        "/v1/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health Check",
    )
    async def health_check(
        user_adapter=Depends(get_user_adapter),
    ):
        """Check system health."""
        components = {}
        
        # Check DKI engine
        components["dki_engine"] = "ok"
        
        # Check user adapter
        try:
            if await user_adapter.health_check():
                components["user_adapter"] = "ok"
            else:
                components["user_adapter"] = "degraded"
        except Exception:
            components["user_adapter"] = "error"
        
        # Check cache
        components["cache"] = "ok"
        
        # Overall status
        status = "healthy"
        if "error" in components.values():
            status = "unhealthy"
        elif "degraded" in components.values():
            status = "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            components=components,
        )
    
    @router.get(
        "/v1/stats",
        response_model=StatsResponse,
        tags=["Health"],
        summary="System Statistics",
    )
    async def get_stats(
        dki_system=Depends(get_dki_system),
        preference_cache=Depends(get_preference_cache),
        user_adapter=Depends(get_user_adapter),
    ):
        """Get system statistics."""
        try:
            startup_time = get_startup_time()
            uptime = time.time() - startup_time if startup_time > 0 else 0
            
            return StatsResponse(
                dki_stats=dki_system.get_stats() if dki_system else {},
                cache_stats=preference_cache.get_stats() if preference_cache else {},
                adapter_stats={"type": type(user_adapter).__name__} if user_adapter else {},
                uptime_seconds=uptime,
            )
            
        except Exception as e:
            logger.error(f"Get stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get(
        "/v1/models",
        tags=["Models"],
        summary="List Available Models",
    )
    async def list_models():
        """List available models (OpenAI compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "dki-default",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "dki",
                    "permission": [],
                    "root": "dki-default",
                    "parent": None,
                }
            ]
        }
    
    return router
