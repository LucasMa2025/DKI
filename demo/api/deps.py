"""
Demo API Dependencies — 依赖注入

提供 store, dki_plugin 等依赖给 API 路由。
支持同步和异步 store 操作。

Author: AGI Demo Project
Version: 2.1.0
"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from demo.store.base import IChatStore


security = HTTPBearer(auto_error=False)

# Token → user_id 映射 (内存, tokens 是临时的)
_tokens_db: dict = {}


def get_store(request: Request) -> IChatStore:
    """获取 Chat Store 实例"""
    store = getattr(request.app.state, "store", None)
    if not store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat store not initialized",
        )
    return store


def is_async_store(store: IChatStore) -> bool:
    """检查 store 是否为异步实现"""
    return hasattr(store, 'a_get_user')


def get_dki_plugin(request: Request):
    """获取 DKI Plugin 实例"""
    plugin = getattr(request.app.state, "dki_plugin", None)
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DKI Plugin not initialized",
        )
    return plugin


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[str]:
    """从 token 获取 user_id (可选认证)"""
    if not credentials:
        return None
    token = credentials.credentials
    return _tokens_db.get(token)


def require_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """从 token 获取 user_id (必须认证)"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    user_id = _tokens_db.get(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


async def require_auth(
    user_id: str = Depends(require_user_id),
    store: IChatStore = Depends(get_store),
) -> dict:
    """
    完整认证: 返回用户字典 (与现有前端兼容)
    
    支持同步和异步 store。
    返回格式与 dki/api/auth_routes.py 一致:
    {"id": ..., "username": ..., "email": ..., ...}
    """
    if is_async_store(store):
        user = await store.a_get_user(user_id)
    else:
        user = store.get_user(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user.to_dict()
