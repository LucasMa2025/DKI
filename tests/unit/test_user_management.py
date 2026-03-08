"""
Unit Tests for User Management (注册, 密码修改, 找回密码, 资料更新)

测试 DemoUser 模型的密码功能和 Auth API 端点。

Author: AGI Demo Project
"""

import pytest
import hashlib
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from demo.store.models import DemoUser


# ============================================================
# 1. DemoUser 密码功能
# ============================================================

class TestDemoUserPassword:
    """测试 DemoUser 密码相关方法"""

    def _make_user(self, **kwargs) -> DemoUser:
        """创建测试用户 (使用正常构造函数)"""
        user = DemoUser(
            id=kwargs.get("id", "test-user-id"),
            username=kwargs.get("username", "testuser"),
            display_name=kwargs.get("display_name", "Test User"),
            email=kwargs.get("email", "test@example.com"),
            avatar=kwargs.get("avatar", None),
            password_hash=kwargs.get("password_hash", None),
            is_active=kwargs.get("is_active", True),
            created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
            last_login_at=kwargs.get("last_login_at", None),
        )
        return user

    def test_hash_password(self):
        """SHA-256 密码哈希"""
        hashed = DemoUser.hash_password("mypassword")
        expected = hashlib.sha256("mypassword".encode("utf-8")).hexdigest()
        assert hashed == expected
        assert len(hashed) == 64  # SHA-256 hex digest length

    def test_set_password(self):
        """设置密码"""
        user = self._make_user()
        assert user.password_hash is None
        user.set_password("secret123")
        assert user.password_hash is not None
        assert user.password_hash == DemoUser.hash_password("secret123")

    def test_verify_password_correct(self):
        """验证正确密码"""
        user = self._make_user()
        user.set_password("correct_password")
        assert user.verify_password("correct_password") is True

    def test_verify_password_wrong(self):
        """验证错误密码"""
        user = self._make_user()
        user.set_password("correct_password")
        assert user.verify_password("wrong_password") is False

    def test_verify_password_demo_mode(self):
        """Demo mode: 无密码时任意密码通过"""
        user = self._make_user(password_hash=None)
        assert user.verify_password("anything") is True
        assert user.verify_password("") is True

    def test_has_password_true(self):
        """已设密码"""
        user = self._make_user()
        user.set_password("pass")
        assert user.has_password is True

    def test_has_password_false(self):
        """未设密码"""
        user = self._make_user(password_hash=None)
        assert user.has_password is False

    def test_has_password_empty_string(self):
        """空字符串密码哈希"""
        user = self._make_user(password_hash="")
        assert user.has_password is False

    def test_to_dict_includes_has_password(self):
        """to_dict 包含 hasPassword 字段"""
        user = self._make_user()
        user.set_password("pass")
        d = user.to_dict()
        assert "hasPassword" in d
        assert d["hasPassword"] is True

    def test_to_dict_no_password_hash_exposed(self):
        """to_dict 不暴露 password_hash"""
        user = self._make_user()
        user.set_password("secret")
        d = user.to_dict()
        assert "password_hash" not in d
        assert "passwordHash" not in d

    def test_password_different_inputs_different_hashes(self):
        """不同密码产生不同哈希"""
        h1 = DemoUser.hash_password("password1")
        h2 = DemoUser.hash_password("password2")
        assert h1 != h2

    def test_password_same_input_same_hash(self):
        """相同密码产生相同哈希 (确定性)"""
        h1 = DemoUser.hash_password("same_password")
        h2 = DemoUser.hash_password("same_password")
        assert h1 == h2


# ============================================================
# 2. Auth API Pydantic Models
# ============================================================

class TestAuthModels:
    """测试 Auth API 的 Pydantic 模型"""

    def test_login_request_defaults(self):
        from demo.api.auth import LoginRequest
        req = LoginRequest(username="alice")
        assert req.username == "alice"
        assert req.password == ""
        assert req.remember is False

    def test_register_request_validation(self):
        from demo.api.auth import RegisterRequest
        req = RegisterRequest(username="bob", password="pass123", email="bob@test.com")
        assert req.username == "bob"
        assert req.password == "pass123"
        assert req.email == "bob@test.com"

    def test_register_request_min_length(self):
        from demo.api.auth import RegisterRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RegisterRequest(username="a")  # min_length=2

    def test_change_password_request(self):
        from demo.api.auth import ChangePasswordRequest
        req = ChangePasswordRequest(old_password="old", new_password="newpass")
        assert req.old_password == "old"
        assert req.new_password == "newpass"

    def test_change_password_min_length(self):
        from demo.api.auth import ChangePasswordRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChangePasswordRequest(new_password="ab")  # min_length=4

    def test_recover_password_request(self):
        from demo.api.auth import RecoverPasswordRequest
        req = RecoverPasswordRequest(email="user@test.com", new_password="newpass")
        assert req.email == "user@test.com"

    def test_update_profile_request(self):
        from demo.api.auth import UpdateProfileRequest
        req = UpdateProfileRequest(display_name="New Name", email="new@test.com")
        assert req.display_name == "New Name"
        assert req.email == "new@test.com"
        assert req.avatar is None

    def test_user_response(self):
        from demo.api.auth import UserResponse
        resp = UserResponse(
            id="u1",
            username="alice",
            email="alice@test.com",
            has_password=True,
        )
        assert resp.id == "u1"
        assert resp.has_password is True

    def test_login_response(self):
        from demo.api.auth import LoginResponse, UserResponse
        user = UserResponse(id="u1", username="alice")
        resp = LoginResponse(token="demo_abc123", user=user)
        assert resp.token == "demo_abc123"
        assert resp.user.username == "alice"

    def test_message_response(self):
        from demo.api.auth import MessageResponse
        resp = MessageResponse(message="OK", success=True)
        assert resp.message == "OK"
        assert resp.success is True


# ============================================================
# 3. Auth Router 端点 (使用 FastAPI TestClient)
# ============================================================

class TestAuthEndpoints:
    """测试 Auth API 端点 (集成测试级别, 使用 mock store)"""

    @pytest.fixture
    def mock_store(self):
        """创建 mock store (同步, 无 a_ 前缀方法)"""
        store = MagicMock(spec=[
            'get_or_create_user', 'get_user_by_username', 'create_user',
            'get_user', 'update_user', 'update_user_login',
            'get_user_by_email', 'list_users',
        ])
        store.get_or_create_user = MagicMock()
        store.get_user_by_username = MagicMock(return_value=None)
        store.create_user = MagicMock()
        store.get_user = MagicMock()
        store.update_user = MagicMock()
        store.update_user_login = MagicMock()
        store.get_user_by_email = MagicMock()
        store.list_users = MagicMock(return_value=[])
        return store

    @pytest.fixture
    def app(self, mock_store):
        """创建 FastAPI 测试应用"""
        from fastapi import FastAPI
        from demo.api.auth import create_auth_router

        app = FastAPI()
        app.state.store = mock_store
        app.include_router(create_auth_router())
        return app

    @pytest.fixture
    def client(self, app):
        """创建 TestClient"""
        from fastapi.testclient import TestClient
        return TestClient(app)

    def _make_mock_user(self, **kwargs):
        """创建 mock 用户对象"""
        user = MagicMock()
        user.id = kwargs.get("id", "user-123")
        user.username = kwargs.get("username", "testuser")
        user.display_name = kwargs.get("display_name", "Test User")
        user.email = kwargs.get("email", "test@example.com")
        user.avatar = kwargs.get("avatar", None)
        user.password_hash = kwargs.get("password_hash", None)
        user.has_password = bool(kwargs.get("password_hash"))
        user.created_at = datetime.now(timezone.utc)
        user.last_login_at = None
        user.verify_password = MagicMock(return_value=True)
        user.to_dict = MagicMock(return_value={
            "id": user.id,
            "username": user.username,
            "displayName": user.display_name,
            "email": user.email,
            "avatar": user.avatar,
            "hasPassword": user.has_password,
            "createdAt": user.created_at.isoformat(),
        })
        return user

    def test_login_new_user(self, client, mock_store):
        """登录 - 新用户自动创建"""
        mock_user = self._make_mock_user()
        mock_store.get_or_create_user.return_value = (mock_user, True)

        resp = client.post("/api/auth/login", json={
            "username": "newuser",
            "password": "",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["username"] == "testuser"

    def test_login_existing_user_correct_password(self, client, mock_store):
        """登录 - 已有用户正确密码"""
        mock_user = self._make_mock_user(password_hash="somehash")
        mock_user.has_password = True
        mock_user.verify_password.return_value = True
        mock_store.get_or_create_user.return_value = (mock_user, False)

        resp = client.post("/api/auth/login", json={
            "username": "existing",
            "password": "correct",
        })
        assert resp.status_code == 200

    def test_login_existing_user_wrong_password(self, client, mock_store):
        """登录 - 已有用户错误密码"""
        mock_user = self._make_mock_user(password_hash="somehash")
        mock_user.has_password = True
        mock_user.verify_password.return_value = False
        mock_store.get_or_create_user.return_value = (mock_user, False)

        resp = client.post("/api/auth/login", json={
            "username": "existing",
            "password": "wrong",
        })
        assert resp.status_code == 401

    def test_register_success(self, client, mock_store):
        """注册 - 成功"""
        mock_user = self._make_mock_user(username="newuser")
        mock_store.get_user_by_username.return_value = None
        mock_store.create_user.return_value = mock_user

        resp = client.post("/api/auth/register", json={
            "username": "newuser",
            "password": "pass1234",
            "email": "new@test.com",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "newuser"

    def test_register_duplicate_username(self, client, mock_store):
        """注册 - 用户名已存在"""
        mock_store.get_user_by_username.return_value = self._make_mock_user()

        resp = client.post("/api/auth/register", json={
            "username": "existing",
            "password": "pass1234",
        })
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_recover_password_success(self, client, mock_store):
        """找回密码 - 成功"""
        mock_user = self._make_mock_user(email="user@test.com")
        mock_store.get_user_by_email.return_value = mock_user

        resp = client.post("/api/auth/recover-password", json={
            "email": "user@test.com",
            "new_password": "newpass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_recover_password_email_not_found(self, client, mock_store):
        """找回密码 - 邮箱不存在"""
        mock_store.get_user_by_email.return_value = None

        resp = client.post("/api/auth/recover-password", json={
            "email": "nonexistent@test.com",
            "new_password": "newpass123",
        })
        assert resp.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
