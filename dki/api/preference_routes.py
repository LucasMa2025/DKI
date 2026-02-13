"""
Preference Management Routes
FastAPI routes for user preference management

Author: AGI Demo Project
Version: 1.0.0

修正: 使用实际的 SQLite 数据库而非内存存储
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from dki.api.auth_routes import require_auth
from dki.database.connection import DatabaseManager
from dki.config.config_loader import ConfigLoader


# Database manager singleton
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager."""
    global _db_manager
    if _db_manager is None:
        config = ConfigLoader().config
        _db_manager = DatabaseManager(
            db_path=config.database.path,
            echo=config.database.echo,
        )
    return _db_manager


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
    userId: str  # 前端期望 camelCase
    preferenceText: str
    preferenceType: str
    priority: int
    category: Optional[str] = None
    metadata: Optional[dict] = None
    isActive: bool
    createdAt: str
    updatedAt: str


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
        
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            from sqlalchemy import text
            
            # Query preferences from database
            query = text("""
                SELECT id, user_id, preference_text, preference_type, priority, 
                       category, metadata, is_active, created_at, updated_at
                FROM user_preferences
                WHERE user_id = :user_id AND is_active = 1
                ORDER BY priority DESC, created_at DESC
            """)
            
            result = db.execute(query, {"user_id": user_id})
            rows = result.fetchall()
            
            preferences = []
            for row in rows:
                # Filter by type if specified
                if preference_type and row[3] != preference_type:
                    continue
                # Filter by category if specified
                if category and row[5] != category:
                    continue
                
                preferences.append(PreferenceResponse(
                    id=row[0],
                    userId=row[1],
                    preferenceText=row[2],
                    preferenceType=row[3],
                    priority=row[4],
                    category=row[5],
                    metadata=None,  # JSON parsing would be needed
                    isActive=bool(row[7]),
                    createdAt=row[8] if isinstance(row[8], str) else row[8].isoformat() if row[8] else datetime.now().isoformat(),
                    updatedAt=row[9] if isinstance(row[9], str) else row[9].isoformat() if row[9] else datetime.now().isoformat(),
                ))
            
            return preferences
    
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
        
        db_manager = get_db_manager()
        pref_id = f"pref_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        with db_manager.session_scope() as db:
            from sqlalchemy import text
            
            # Insert preference into database
            query = text("""
                INSERT INTO user_preferences 
                (id, user_id, preference_text, preference_type, priority, category, is_active, created_at, updated_at)
                VALUES (:id, :user_id, :preference_text, :preference_type, :priority, :category, :is_active, :created_at, :updated_at)
            """)
            
            db.execute(query, {
                "id": pref_id,
                "user_id": request.user_id,
                "preference_text": request.preference_text,
                "preference_type": request.preference_type,
                "priority": request.priority,
                "category": request.category,
                "is_active": 1 if request.is_active else 0,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            })
            
            logger.info(f"Created preference: {pref_id} for user: {request.user_id}")
            
            return PreferenceResponse(
                id=pref_id,
                userId=request.user_id,
                preferenceText=request.preference_text,
                preferenceType=request.preference_type,
                priority=request.priority,
                category=request.category,
                metadata=request.metadata,
                isActive=request.is_active,
                createdAt=now.isoformat(),
                updatedAt=now.isoformat(),
            )
    
    @router.get("/{preference_id}", response_model=PreferenceResponse)
    async def get_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Get a specific preference.
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            from sqlalchemy import text
            
            query = text("""
                SELECT id, user_id, preference_text, preference_type, priority, 
                       category, metadata, is_active, created_at, updated_at
                FROM user_preferences
                WHERE id = :id
            """)
            
            result = db.execute(query, {"id": preference_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Preference not found")
            
            if row[1] != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return PreferenceResponse(
                id=row[0],
                userId=row[1],
                preferenceText=row[2],
                preferenceType=row[3],
                priority=row[4],
                category=row[5],
                metadata=None,
                isActive=bool(row[7]),
                createdAt=row[8] if isinstance(row[8], str) else row[8].isoformat() if row[8] else datetime.now().isoformat(),
                updatedAt=row[9] if isinstance(row[9], str) else row[9].isoformat() if row[9] else datetime.now().isoformat(),
            )
    
    @router.patch("/{preference_id}", response_model=PreferenceResponse)
    async def update_preference(
        preference_id: str,
        request: PreferenceUpdate,
        user: dict = Depends(require_auth),
    ):
        """
        Update a preference.
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            from sqlalchemy import text
            
            # First check if preference exists and belongs to user
            check_query = text("SELECT user_id FROM user_preferences WHERE id = :id")
            result = db.execute(check_query, {"id": preference_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Preference not found")
            
            if row[0] != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Build update query dynamically
            updates = []
            params = {"id": preference_id, "updated_at": datetime.now().isoformat()}
            
            if request.preference_text is not None:
                updates.append("preference_text = :preference_text")
                params["preference_text"] = request.preference_text
            if request.preference_type is not None:
                updates.append("preference_type = :preference_type")
                params["preference_type"] = request.preference_type
            if request.priority is not None:
                updates.append("priority = :priority")
                params["priority"] = request.priority
            if request.category is not None:
                updates.append("category = :category")
                params["category"] = request.category
            if request.is_active is not None:
                updates.append("is_active = :is_active")
                params["is_active"] = 1 if request.is_active else 0
            
            updates.append("updated_at = :updated_at")
            
            if updates:
                update_query = text(f"UPDATE user_preferences SET {', '.join(updates)} WHERE id = :id")
                db.execute(update_query, params)
            
            logger.info(f"Updated preference: {preference_id}")
            
            # Return updated preference
            return await get_preference(preference_id, user)
    
    @router.delete("/{preference_id}")
    async def delete_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
    ):
        """
        Delete a preference.
        """
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as db:
            from sqlalchemy import text
            
            # First check if preference exists and belongs to user
            check_query = text("SELECT user_id FROM user_preferences WHERE id = :id")
            result = db.execute(check_query, {"id": preference_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Preference not found")
            
            if row[0] != user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Soft delete
            delete_query = text("UPDATE user_preferences SET is_active = 0 WHERE id = :id")
            db.execute(delete_query, {"id": preference_id})
            
            logger.info(f"Deleted preference: {preference_id}")
            
            return {"status": "deleted", "preference_id": preference_id}
    
    return router


def get_user_preferences_for_dki(user_id: str) -> List[dict]:
    """
    Get active preferences for DKI injection.
    """
    db_manager = get_db_manager()
    
    with db_manager.session_scope() as db:
        from sqlalchemy import text
        
        query = text("""
            SELECT id, user_id, preference_text, preference_type, priority, category, is_active
            FROM user_preferences
            WHERE user_id = :user_id AND is_active = 1
            ORDER BY priority DESC
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return [
            {
                "id": row[0],
                "user_id": row[1],
                "preference_text": row[2],
                "preference_type": row[3],
                "priority": row[4],
                "category": row[5],
                "is_active": bool(row[6]),
            }
            for row in rows
        ]
