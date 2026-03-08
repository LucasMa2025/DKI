"""
Demo Preferences API — 偏好管理路由

支持同步和异步 store 操作。

Author: AGI Demo Project
Version: 2.1.0
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from demo.api.deps import get_store, get_dki_plugin, require_auth, is_async_store
from demo.store.base import IChatStore


# ============ Request/Response Models ============

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
    """响应格式: camelCase (与前端兼容)"""
    id: str
    userId: str
    preferenceText: str
    preferenceType: str
    priority: int
    category: Optional[str] = None
    metadata: Optional[dict] = None
    isActive: bool
    createdAt: str
    updatedAt: str


def _pref_to_response(pref) -> PreferenceResponse:
    """将 DemoPreference 转为响应"""
    return PreferenceResponse(
        id=pref.id,
        userId=pref.user_id,
        preferenceText=pref.preference_text,
        preferenceType=pref.preference_type,
        priority=pref.priority,
        category=pref.category,
        metadata=pref.get_metadata() if hasattr(pref, 'get_metadata') else None,
        isActive=pref.is_active,
        createdAt=pref.created_at.isoformat() if pref.created_at else "",
        updatedAt=pref.updated_at.isoformat() if pref.updated_at else "",
    )


# ============ Router ============

def create_preference_router() -> APIRouter:
    """创建偏好管理路由"""
    
    router = APIRouter(prefix="/api/preferences", tags=["Preferences"])
    
    @router.get("", response_model=List[PreferenceResponse])
    async def list_preferences(
        user_id: str = Query(..., description="User ID"),
        preference_type: Optional[str] = Query(default=None),
        category: Optional[str] = Query(default=None),
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """列出用户偏好"""
        if user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        pref_types = [preference_type] if preference_type else None

        if is_async_store(store):
            prefs = await store.a_get_preferences(user_id, preference_types=pref_types)
        else:
            prefs = store.get_preferences(user_id, preference_types=pref_types)
        
        # 按 category 过滤 (如果指定)
        if category:
            prefs = [p for p in prefs if p.category == category]
        
        return [_pref_to_response(p) for p in prefs]
    
    @router.post("", response_model=PreferenceResponse)
    async def create_preference(
        request: PreferenceCreate,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
        dki_plugin = Depends(get_dki_plugin),
    ):
        """创建偏好"""
        if request.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if is_async_store(store):
            pref = await store.a_add_preference(
                user_id=request.user_id,
                preference_text=request.preference_text,
                preference_type=request.preference_type,
                priority=request.priority,
                category=request.category,
            )
        else:
            pref = store.add_preference(
                user_id=request.user_id,
                preference_text=request.preference_text,
                preference_type=request.preference_type,
                priority=request.priority,
                category=request.category,
            )
        
        logger.info(f"Created preference: {pref.id} for user: {request.user_id}")
        
        # ★ 使 DKI Plugin 偏好缓存失效 (确保下次 chat 读取最新偏好)
        _invalidate_dki_cache(dki_plugin, request.user_id)
        
        return _pref_to_response(pref)
    
    @router.get("/{preference_id}", response_model=PreferenceResponse)
    async def get_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
    ):
        """获取单个偏好"""
        if is_async_store(store):
            pref = await store.a_get_preference(preference_id)
        else:
            pref = store.get_preference(preference_id)

        if not pref:
            raise HTTPException(status_code=404, detail="Preference not found")
        if pref.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return _pref_to_response(pref)
    
    @router.patch("/{preference_id}", response_model=PreferenceResponse)
    async def update_preference(
        preference_id: str,
        request: PreferenceUpdate,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
        dki_plugin = Depends(get_dki_plugin),
    ):
        """更新偏好"""
        _async = is_async_store(store)

        if _async:
            pref = await store.a_get_preference(preference_id)
        else:
            pref = store.get_preference(preference_id)

        if not pref:
            raise HTTPException(status_code=404, detail="Preference not found")
        if pref.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        kwargs = {}
        if request.preference_text is not None:
            kwargs["preference_text"] = request.preference_text
        if request.preference_type is not None:
            kwargs["preference_type"] = request.preference_type
        if request.priority is not None:
            kwargs["priority"] = request.priority
        if request.category is not None:
            kwargs["category"] = request.category
        if request.is_active is not None:
            kwargs["is_active"] = request.is_active
        
        if _async:
            updated = await store.a_update_preference(preference_id, **kwargs)
        else:
            updated = store.update_preference(preference_id, **kwargs)
        logger.info(f"Updated preference: {preference_id}")
        
        # ★ 使 DKI Plugin 偏好缓存失效
        _invalidate_dki_cache(dki_plugin, pref.user_id)
        
        return _pref_to_response(updated)
    
    @router.delete("/{preference_id}")
    async def delete_preference(
        preference_id: str,
        user: dict = Depends(require_auth),
        store: IChatStore = Depends(get_store),
        dki_plugin = Depends(get_dki_plugin),
    ):
        """删除偏好 (软删除)"""
        _async = is_async_store(store)

        if _async:
            pref = await store.a_get_preference(preference_id)
        else:
            pref = store.get_preference(preference_id)

        if not pref:
            raise HTTPException(status_code=404, detail="Preference not found")
        if pref.user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if _async:
            await store.a_delete_preference(preference_id)
        else:
            store.delete_preference(preference_id)
        logger.info(f"Deleted preference: {preference_id}")
        
        # ★ 使 DKI Plugin 偏好缓存失效
        _invalidate_dki_cache(dki_plugin, pref.user_id)
        
        return {"status": "deleted", "preference_id": preference_id}
    
    return router


def _invalidate_dki_cache(dki_plugin, user_id: str) -> None:
    """使 DKI Plugin 偏好缓存失效"""
    try:
        if hasattr(dki_plugin, 'invalidate_preference_text_cache'):
            dki_plugin.invalidate_preference_text_cache(user_id)
            logger.debug(f"Invalidated DKI preference cache for user: {user_id}")
    except Exception as e:
        logger.warning(f"Failed to invalidate DKI cache: {e}")
