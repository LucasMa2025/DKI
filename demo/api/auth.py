"""
Demo Auth API — 认证与用户管理路由

功能:
- 登录 (支持密码验证, 向后兼容 demo mode)
- 注册 (支持密码设置)
- 登出
- 获取当前用户信息
- 修改密码
- 找回密码 (通过邮箱)
- 更新用户资料
- 列出所有用户 (管理接口)

Author: AGI Demo Project
Version: 3.0.0
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from demo.api.deps import (
    get_store, require_auth, require_user_id,
    is_async_store, _tokens_db,
)
from demo.store.base import IChatStore


# ============ Request/Response Models ============

class LoginRequest(BaseModel):
    username: str
    password: str = ""  # 空密码 = demo mode
    remember: bool = False


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field("", min_length=0, max_length=128)
    email: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    """修改密码"""
    old_password: str = Field("", description="旧密码 (demo mode 用户可为空)")
    new_password: str = Field(..., min_length=4, max_length=128, description="新密码")


class RecoverPasswordRequest(BaseModel):
    """找回密码 (通过邮箱)"""
    email: str = Field(..., description="注册时使用的邮箱")
    new_password: str = Field(..., min_length=4, max_length=128, description="新密码")


class UpdateProfileRequest(BaseModel):
    """更新用户资料"""
    display_name: Optional[str] = Field(None, max_length=128)
    email: Optional[str] = Field(None, max_length=128)
    avatar: Optional[str] = Field(None, max_length=256)


class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    avatar: Optional[str] = None
    has_password: bool = False
    created_at: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


class MessageResponse(BaseModel):
    """通用消息响应"""
    message: str
    success: bool = True


# ============ Router ============

def create_auth_router() -> APIRouter:
    """创建认证路由"""
    
    router = APIRouter(prefix="/api/auth", tags=["Authentication"])
    
    @router.post("/login", response_model=LoginResponse)
    async def login(
        request: LoginRequest,
        store: IChatStore = Depends(get_store),
    ):
        """
        登录
        
        密码验证逻辑:
        - 用户不存在: 自动创建 (demo mode, 密码可选)
        - 用户存在且未设密码: 任意密码通过 (demo mode)
        - 用户存在且已设密码: 验证密码
        """
        if is_async_store(store):
            user, created = await store.a_get_or_create_user(
                username=request.username,
                display_name=request.username,
            )
        else:
            user, created = store.get_or_create_user(
                username=request.username,
                display_name=request.username,
            )
        
        if created:
            # 新用户: 如果提供了密码则设置
            if request.password:
                if is_async_store(store):
                    await store.a_update_user(user.id, password=request.password)
                elif hasattr(store, 'update_user'):
                    store.update_user(user.id, password=request.password)
            logger.info(f"Created new user: {request.username} (id={user.id})")
        else:
            # 已有用户: 验证密码
            if not user.verify_password(request.password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect password",
                )
            logger.info(f"User logged in: {request.username} (id={user.id})")
        
        # 更新登录时间
        if is_async_store(store):
            await store.a_update_user_login(user.id)
        else:
            store.update_user_login(user.id)
        
        # 生成 token
        token = f"demo_{uuid.uuid4().hex}"
        _tokens_db[token] = user.id
        
        return LoginResponse(
            token=token,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                avatar=user.avatar,
                has_password=user.has_password,
                created_at=user.created_at.isoformat() if user.created_at else None,
            ),
        )
    
    @router.post("/register", response_model=UserResponse)
    async def register(
        request: RegisterRequest,
        store: IChatStore = Depends(get_store),
    ):
        """
        注册新用户
        
        - 用户名必须唯一
        - 密码可选 (不设密码 = demo mode, 任意密码可登录)
        """
        if is_async_store(store):
            existing = await store.a_get_user_by_username(request.username)
        else:
            existing = store.get_user_by_username(request.username)

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )
        
        if is_async_store(store):
            user = await store.a_create_user(
                username=request.username,
                display_name=request.username,
                email=request.email,
                password=request.password if request.password else None,
            )
        else:
            user = store.create_user(
                username=request.username,
                display_name=request.username,
                email=request.email,
                password=request.password if request.password else None,
            )
        
        logger.info(f"Registered user: {request.username} (id={user.id})")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            has_password=user.has_password,
            created_at=user.created_at.isoformat() if user.created_at else None,
        )
    
    @router.post("/logout")
    async def logout(user: dict = Depends(require_auth)):
        """登出"""
        tokens_to_remove = [
            token for token, uid in _tokens_db.items()
            if uid == user["id"]
        ]
        for token in tokens_to_remove:
            del _tokens_db[token]
        
        return {"status": "logged_out"}
    
    @router.get("/me", response_model=UserResponse)
    async def get_current_user_info(user: dict = Depends(require_auth)):
        """获取当前用户信息"""
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user.get("email"),
            avatar=user.get("avatar"),
            has_password=user.get("hasPassword", False),
            created_at=user.get("createdAt"),
        )
    
    @router.put("/password", response_model=MessageResponse)
    async def change_password(
        request: ChangePasswordRequest,
        user_id: str = Depends(require_user_id),
        store: IChatStore = Depends(get_store),
    ):
        """
        修改密码
        
        - 已设密码的用户: 需要验证旧密码
        - 未设密码的用户 (demo mode): 旧密码可为空, 直接设置新密码
        """
        # 获取用户
        if is_async_store(store):
            user = await store.a_get_user(user_id)
        else:
            user = store.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # 验证旧密码 (已设密码的用户必须验证)
        if user.has_password:
            if not user.verify_password(request.old_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect old password",
                )
        
        # 设置新密码
        if is_async_store(store):
            await store.a_update_user(user_id, password=request.new_password)
        elif hasattr(store, 'update_user'):
            store.update_user(user_id, password=request.new_password)
        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Password change not supported by current store",
            )
        
        logger.info(f"Password changed for user: {user.username} (id={user_id})")
        
        return MessageResponse(
            message="Password changed successfully",
            success=True,
        )
    
    @router.post("/recover-password", response_model=MessageResponse)
    async def recover_password(
        request: RecoverPasswordRequest,
        store: IChatStore = Depends(get_store),
    ):
        """
        找回密码 (通过邮箱)
        
        演示系统: 直接通过邮箱匹配用户并重置密码。
        生产环境应发送重置邮件, 此处简化处理。
        """
        # 通过邮箱查找用户
        if is_async_store(store):
            user = await store.a_get_user_by_email(request.email) \
                if hasattr(store, 'a_get_user_by_email') else None
        elif hasattr(store, 'get_user_by_email'):
            user = store.get_user_by_email(request.email)
        else:
            user = None
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No user found with this email address",
            )
        
        # 重置密码
        if is_async_store(store):
            await store.a_update_user(user.id, password=request.new_password)
        elif hasattr(store, 'update_user'):
            store.update_user(user.id, password=request.new_password)
        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Password recovery not supported by current store",
            )
        
        logger.info(f"Password recovered for user: {user.username} (email={request.email})")
        
        return MessageResponse(
            message="Password has been reset successfully. Please login with your new password.",
            success=True,
        )
    
    @router.put("/profile", response_model=UserResponse)
    async def update_profile(
        request: UpdateProfileRequest,
        user_id: str = Depends(require_user_id),
        store: IChatStore = Depends(get_store),
    ):
        """更新用户资料 (显示名称、邮箱、头像)"""
        updates = {}
        if request.display_name is not None:
            updates['display_name'] = request.display_name
        if request.email is not None:
            updates['email'] = request.email
        if request.avatar is not None:
            updates['avatar'] = request.avatar
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update",
            )
        
        if is_async_store(store):
            user = await store.a_update_user(user_id, **updates) \
                if hasattr(store, 'a_update_user') else None
        elif hasattr(store, 'update_user'):
            user = store.update_user(user_id, **updates)
        else:
            user = None
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found or update not supported",
            )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            avatar=user.avatar,
            has_password=user.has_password,
            created_at=user.created_at.isoformat() if user.created_at else None,
        )
    
    @router.get("/users")
    async def list_users(store: IChatStore = Depends(get_store)):
        """列出所有用户 (管理接口)"""
        if is_async_store(store):
            users = await store.a_list_users(limit=100)
        else:
            users = store.list_users(limit=100)
        return [u.to_dict() for u in users]
    
    return router
