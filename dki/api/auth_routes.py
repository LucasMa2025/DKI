"""
Authentication Routes
FastAPI routes for user authentication

Author: AGI Demo Project
Version: 1.0.0

演示系统认证:
- 只查询用户账号，不验证密码
- 登录时查询用户，如不存在则创建
- 用户数据持久化到数据库，确保测试过程中的偏好及会话历史可管理
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger

from dki.database.connection import DatabaseManager
from dki.database.repository import DemoUserRepository


# Token store (in-memory, tokens are transient)
# 用户数据现在存储在数据库中，只有 token 是临时的
_tokens_db = {}


class LoginRequest(BaseModel):
    """
    登录请求 - 演示系统简化版
    
    注意: 演示系统不验证密码，只查询用户名
    """
    username: str
    password: str = ""  # 演示系统不验证密码，设为可选
    remember: bool = False


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    avatar: Optional[str] = None
    created_at: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Simple password hashing for demo."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token() -> str:
    """Generate a simple token."""
    return f"dki_{uuid.uuid4().hex}"


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[dict]:
    """Get current user from token."""
    if not credentials:
        return None
    
    token = credentials.credentials
    user_id = _tokens_db.get(token)
    
    if not user_id:
        return None
    
    # 从数据库获取用户
    db_manager = DatabaseManager.get_instance()
    with db_manager.session_scope() as db:
        user_repo = DemoUserRepository(db)
        user = user_repo.get(user_id)
        if user:
            return user.to_dict()
    return None


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Require authentication."""
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
    
    # 从数据库获取用户
    db_manager = DatabaseManager.get_instance()
    with db_manager.session_scope() as db:
        user_repo = DemoUserRepository(db)
        user = user_repo.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user.to_dict()


def create_auth_router() -> APIRouter:
    """Create authentication router."""
    
    router = APIRouter(prefix="/api/auth", tags=["Authentication"])
    
    @router.post("/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        """
        User login endpoint.
        
        演示系统登录:
        - 只查询用户账号，不验证密码
        - 如果用户不存在则自动创建
        - 用户数据持久化到数据库
        """
        db_manager = DatabaseManager.get_instance()
        
        with db_manager.session_scope() as db:
            user_repo = DemoUserRepository(db)
            
            # 查询或创建用户 (不验证密码)
            user, created = user_repo.get_or_create(
                username=request.username,
                display_name=request.username,
            )
            
            if created:
                logger.info(f"Created new demo user: {request.username} (id={user.id})")
            else:
                logger.info(f"Demo user logged in: {request.username} (id={user.id})")
            
            user_id = user.id
            user_dict = user.to_dict()
        
        # Generate token (token 仍然是临时的)
        token = generate_token()
        _tokens_db[token] = user_id
        
        return LoginResponse(
            token=token,
            user=UserResponse(
                id=user_dict["id"],
                username=user_dict["username"],
                email=user_dict.get("email"),
                avatar=user_dict.get("avatar"),
                created_at=user_dict.get("createdAt"),
            ),
        )
    
    @router.post("/register", response_model=UserResponse)
    async def register(request: RegisterRequest):
        """
        User registration endpoint.
        
        演示系统注册 (实际上与登录相同，因为不验证密码)
        """
        db_manager = DatabaseManager.get_instance()
        
        with db_manager.session_scope() as db:
            user_repo = DemoUserRepository(db)
            
            # Check if username exists
            existing_user = user_repo.get_by_username(request.username)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists",
                )
            
            # Create user
            user = user_repo.create(
                username=request.username,
                display_name=request.username,
                email=request.email,
            )
            
            logger.info(f"Registered user: {request.username} (id={user.id})")
            user_dict = user.to_dict()
        
        return UserResponse(
            id=user_dict["id"],
            username=user_dict["username"],
            email=user_dict.get("email"),
            created_at=user_dict.get("createdAt"),
        )
    
    @router.post("/logout")
    async def logout(user: dict = Depends(require_auth)):
        """
        User logout endpoint.
        """
        # Remove token
        tokens_to_remove = [
            token for token, uid in _tokens_db.items()
            if uid == user["id"]
        ]
        for token in tokens_to_remove:
            del _tokens_db[token]
        
        return {"status": "logged_out"}
    
    @router.get("/me", response_model=UserResponse)
    async def get_current_user_info(user: dict = Depends(require_auth)):
        """
        Get current user information.
        """
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user.get("email"),
            avatar=user.get("avatar"),
            created_at=user.get("createdAt"),
        )
    
    @router.get("/users", response_model=list)
    async def list_users():
        """
        List all demo users.
        
        用于演示系统管理界面，查看所有已创建的用户
        """
        db_manager = DatabaseManager.get_instance()
        
        with db_manager.session_scope() as db:
            user_repo = DemoUserRepository(db)
            users = user_repo.list_all(limit=100)
            return [user.to_dict() for user in users]
    
    return router
