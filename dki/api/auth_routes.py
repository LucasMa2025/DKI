"""
Authentication Routes
FastAPI routes for user authentication

Author: AGI Demo Project
Version: 1.0.0
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger


# Simple in-memory user store for demo purposes
# In production, use a proper database
_users_db = {}
_tokens_db = {}


class LoginRequest(BaseModel):
    username: str
    password: str
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
    
    return _users_db.get(user_id)


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
    
    user = _users_db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def create_auth_router() -> APIRouter:
    """Create authentication router."""
    
    router = APIRouter(prefix="/api/auth", tags=["Authentication"])
    
    @router.post("/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        """
        User login endpoint.
        
        For demo purposes, accepts any username/password and creates
        a user if it doesn't exist.
        """
        # Demo mode: create user if not exists
        user_id = None
        for uid, user in _users_db.items():
            if user["username"] == request.username:
                user_id = uid
                break
        
        if not user_id:
            # Create new user for demo
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            _users_db[user_id] = {
                "id": user_id,
                "username": request.username,
                "password_hash": hash_password(request.password),
                "email": None,
                "avatar": None,
                "created_at": datetime.now().isoformat(),
            }
            logger.info(f"Created demo user: {request.username}")
        
        # Generate token
        token = generate_token()
        _tokens_db[token] = user_id
        
        user = _users_db[user_id]
        
        return LoginResponse(
            token=token,
            user=UserResponse(
                id=user["id"],
                username=user["username"],
                email=user.get("email"),
                avatar=user.get("avatar"),
                created_at=user.get("created_at"),
            ),
        )
    
    @router.post("/register", response_model=UserResponse)
    async def register(request: RegisterRequest):
        """
        User registration endpoint.
        """
        # Check if username exists
        for user in _users_db.values():
            if user["username"] == request.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists",
                )
        
        # Create user
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        _users_db[user_id] = {
            "id": user_id,
            "username": request.username,
            "password_hash": hash_password(request.password),
            "email": request.email,
            "avatar": None,
            "created_at": datetime.now().isoformat(),
        }
        
        logger.info(f"Registered user: {request.username}")
        
        return UserResponse(
            id=user_id,
            username=request.username,
            email=request.email,
            created_at=_users_db[user_id]["created_at"],
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
            created_at=user.get("created_at"),
        )
    
    return router
