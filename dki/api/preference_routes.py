"""
Preference Management Routes
FastAPI routes for user preference management

Author: AGI Demo Project
Version: 1.0.0
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from dki.api.auth_routes import require_auth


# Simple in-memory preference store for demo purposes
_preferences_db = {}


class PreferenceCreate(BaseModel):
    user_id: str
    preference_text: str
    preference_type: str = "general"
    priority: int = 5
    category: Optional[str] = None
    metadata: Optional[dict] = None
    is_active: bool = True


class PreferenceUpdate(BaseModel):
    preference_text: Optional[str] = None
    preference_type: Optional[str] = None
    priority: Optional[int] = None
    category: Optional[str] = None
    metadata: Optional[dict] = None
    is_active: Optional[bool] = None


class PreferenceResponse(BaseModel):
    id: str
    user_id: str
    preference_text: str
    preference_type: str
    priority: int
    category: Optional[str] = None
    metadata: Optional[dict] = None
    is_active: bool
    created_at: str
    updated_at: str


def create_preference_router() -> APIRouter:
    """Create preference management router."""
    
    router = APIRouter(prefix="/api/preferences", tags=["Preferences"])
    
    @router.get("", response_model=List[PreferenceResponse])
    async def list_preferences(
        user_id: str = Query(..., description="User ID to get preferences for"),
        preference_type: Optional[str] = Query(default=None),
        category: Optional[str] = Query(default=None),
        user: dict = Depends(require_auth),
    ):
        """
        List all preferences for a user.
        """
        # Verify user can access these preferences
        if user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        user_prefs = [
            p for p in _preferences_db.values()
            if p["user_id"] == user_id
        ]
        
        # Filter by type
        if preference_type:
            user_prefs = [p for p in user_prefs if p["preference_type"] == preference_type]
        
        # Filter by category
        if category:
            user_prefs = [p for p in user_prefs if p.get("category") == category]
        
        # Sort by priority descending, then by created_at
        user_prefs.sort(key=lambda x: (-x["priority"], x["created_at"]))
        
        return [PreferenceResponse(**p) for p in user_prefs]
    
    @router.post("", response_model=PreferenceResponse)
    async def create_preference(
        request: PreferenceCreate,
        user: dict = Depends(require_auth),
    ):
        """
        Create a new preference.
        """
        # Verify user can create this preference
        if request.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        pref_id = f"pref_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()
        
        preference = {
            "id": pref_id,
            "user_id": request.user_id,
            "preference_text": request.preference_text,
            "preference_type": request.preference_type,
            "priority": request.priority,
            "category": request.category,
            "metadata": request.metadata,
            "is_active": request.is_active,
            "created_at": now,
            "updated_at": now,
        }
        
        _preferences_db[pref_id] = preference
        
        logger.info(f"Created preference: {pref_id} for user: {request.user_id}")
        
        return PreferenceResponse(**preference)
    
    @router.get("/{preference_id}", response_model=PreferenceResponse)
    async def get_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Get a specific preference.
        """
        preference = _preferences_db.get(preference_id)
        
        if not preference:
            raise HTTPException(status_code=404, detail="Preference not found")
        
        if preference["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return PreferenceResponse(**preference)
    
    @router.patch("/{preference_id}", response_model=PreferenceResponse)
    async def update_preference(
        preference_id: str,
        request: PreferenceUpdate,
        user: dict = Depends(require_auth),
    ):
        """
        Update a preference.
        """
        preference = _preferences_db.get(preference_id)
        
        if not preference:
            raise HTTPException(status_code=404, detail="Preference not found")
        
        if preference["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        if request.preference_text is not None:
            preference["preference_text"] = request.preference_text
        if request.preference_type is not None:
            preference["preference_type"] = request.preference_type
        if request.priority is not None:
            preference["priority"] = request.priority
        if request.category is not None:
            preference["category"] = request.category
        if request.metadata is not None:
            preference["metadata"] = request.metadata
        if request.is_active is not None:
            preference["is_active"] = request.is_active
        
        preference["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Updated preference: {preference_id}")
        
        return PreferenceResponse(**preference)
    
    @router.delete("/{preference_id}")
    async def delete_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Delete a preference.
        """
        preference = _preferences_db.get(preference_id)
        
        if not preference:
            raise HTTPException(status_code=404, detail="Preference not found")
        
        if preference["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        del _preferences_db[preference_id]
        
        logger.info(f"Deleted preference: {preference_id}")
        
        return {"status": "deleted", "preference_id": preference_id}
    
    return router


def get_user_preferences_for_dki(user_id: str) -> List[dict]:
    """
    Get active preferences for DKI injection.
    """
    return [
        p for p in _preferences_db.values()
        if p["user_id"] == user_id and p["is_active"]
    ]
