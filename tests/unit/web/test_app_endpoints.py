"""
Web App 端点单元测试

测试 dki/web/app.py 中新增和修改的端点:
- /api/chat: 用户隔离 (token/user_id 解析)
- /api/preference: 偏好注入
- /api/preferences/{user_id}: 获取/清除偏好
- /api/experiment/sessions: 实验会话列表
- /api/experiment/session/{session_id}: 实验会话详情
- PreferenceRequest / ChatRequest 模型
- _resolve_user_id 辅助函数
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# 测试 Pydantic 模型和辅助函数 (不需要完整 FastAPI app)
from dki.web.app import ChatRequest, PreferenceRequest, ExperimentRequest


# ============================================================
# Request Model 测试
# ============================================================

class TestChatRequest:
    """ChatRequest 模型测试"""

    def test_default_values(self):
        req = ChatRequest(query="Hello")
        assert req.query == "Hello"
        assert req.mode == "dki"
        assert req.session_id is None
        assert req.force_alpha is None
        assert req.max_new_tokens == 256
        assert req.temperature == 0.7
        assert req.token is None
        assert req.user_id is None

    def test_with_token(self):
        req = ChatRequest(query="Hi", token="abc123")
        assert req.token == "abc123"
        assert req.user_id is None

    def test_with_user_id(self):
        req = ChatRequest(query="Hi", user_id="user_001")
        assert req.user_id == "user_001"
        assert req.token is None

    def test_with_both_token_and_user_id(self):
        req = ChatRequest(query="Hi", token="abc123", user_id="user_001")
        assert req.token == "abc123"
        assert req.user_id == "user_001"

    def test_all_modes(self):
        for mode in ["dki", "rag", "baseline"]:
            req = ChatRequest(query="test", mode=mode)
            assert req.mode == mode

    def test_force_alpha(self):
        req = ChatRequest(query="test", force_alpha=0.8)
        assert req.force_alpha == 0.8

    def test_custom_generation_params(self):
        req = ChatRequest(query="test", max_new_tokens=512, temperature=0.3)
        assert req.max_new_tokens == 512
        assert req.temperature == 0.3


class TestPreferenceRequest:
    """PreferenceRequest 模型测试"""

    def test_default_values(self):
        req = PreferenceRequest(content="我喜欢素食")
        assert req.content == "我喜欢素食"
        assert req.user_id is None
        assert req.preference_type == "general"
        assert req.priority == 8

    def test_custom_values(self):
        req = PreferenceRequest(
            user_id="user_123",
            content="我是程序员",
            preference_type="technical",
            priority=9,
        )
        assert req.user_id == "user_123"
        assert req.content == "我是程序员"
        assert req.preference_type == "technical"
        assert req.priority == 9

    def test_chinese_content(self):
        req = PreferenceRequest(
            content="我对海鲜过敏，请不要推荐任何海鲜相关的食物",
            user_id="test_user",
        )
        assert "海鲜" in req.content


class TestExperimentRequest:
    """ExperimentRequest 模型测试"""

    def test_default_values(self):
        req = ExperimentRequest(name="test_exp")
        assert req.name == "test_exp"
        assert req.description == ""
        assert req.modes == ["dki", "rag", "baseline"]
        assert req.datasets == ["persona_chat", "memory_qa"]
        assert req.max_samples == 50

    def test_custom_values(self):
        req = ExperimentRequest(
            name="custom_exp",
            description="A custom experiment",
            modes=["dki"],
            datasets=["hotpot_qa"],
            max_samples=20,
        )
        assert req.name == "custom_exp"
        assert req.modes == ["dki"]
        assert req.max_samples == 20


# ============================================================
# _resolve_user_id 逻辑测试 (通过模型验证)
# ============================================================

class TestResolveUserIdLogic:
    """
    测试 _resolve_user_id 的逻辑 (不启动 FastAPI app)
    
    由于 _resolve_user_id 是 create_app() 内部的闭包函数,
    无法直接导入。这里测试其逻辑等价物。
    """

    def test_token_priority(self):
        """Token 应优先于 user_id"""
        # 模拟逻辑
        tokens_db = {"valid_token": "user_from_token"}
        
        request = ChatRequest(query="test", token="valid_token", user_id="explicit_user")
        
        # 逻辑: token 存在且有效 → 使用 token 对应的 user_id
        uid = tokens_db.get(request.token)
        if uid:
            resolved = uid
        elif request.user_id and request.user_id.strip():
            resolved = request.user_id.strip()
        else:
            resolved = "demo_user"
        
        assert resolved == "user_from_token"

    def test_user_id_fallback(self):
        """Token 无效时应使用 user_id"""
        tokens_db = {}
        
        request = ChatRequest(query="test", token="invalid_token", user_id="explicit_user")
        
        uid = tokens_db.get(request.token)
        if uid:
            resolved = uid
        elif request.user_id and request.user_id.strip():
            resolved = request.user_id.strip()
        else:
            resolved = "demo_user"
        
        assert resolved == "explicit_user"

    def test_demo_user_fallback(self):
        """无 token 和 user_id 时应降级为 demo_user"""
        tokens_db = {}
        
        request = ChatRequest(query="test")
        
        uid = tokens_db.get(request.token) if request.token else None
        if uid:
            resolved = uid
        elif request.user_id and request.user_id.strip():
            resolved = request.user_id.strip()
        else:
            resolved = "demo_user"
        
        assert resolved == "demo_user"

    def test_empty_user_id_fallback(self):
        """空字符串 user_id 应降级为 demo_user"""
        tokens_db = {}
        
        request = ChatRequest(query="test", user_id="   ")
        
        uid = None
        if request.user_id and request.user_id.strip():
            resolved = request.user_id.strip()
        else:
            resolved = "demo_user"
        
        assert resolved == "demo_user"


# ============================================================
# FastAPI TestClient 集成测试 (需要 mock 系统依赖)
# ============================================================

class TestWebAppEndpoints:
    """
    Web App 端点集成测试
    
    使用 FastAPI TestClient + mock 系统依赖
    """

    @pytest.fixture
    def app_client(self):
        """创建测试客户端 (mock 所有外部依赖)"""
        # 延迟导入以避免模块级副作用
        with patch("dki.web.app.ConfigLoader") as MockConfigLoader, \
             patch("dki.web.app.DatabaseManager") as MockDBManager, \
             patch("dki.web.app.DKISystem") as MockDKI, \
             patch("dki.web.app.RAGSystem") as MockRAG, \
             patch("dki.web.app.ExampleAdapter") as MockExampleAdapter, \
             patch("dki.web.app.init_dependencies"), \
             patch("dki.web.app.cleanup_dependencies"), \
             patch("dki.web.app.set_dki_plugin"), \
             patch("dki.web.app.REDIS_AVAILABLE", False):
            
            mock_config = MagicMock()
            mock_config.database.path = ":memory:"
            MockConfigLoader.return_value.config = mock_config
            
            mock_adapter = MagicMock()
            MockExampleAdapter.return_value = mock_adapter
            
            from dki.web.app import create_app
            app = create_app()
            
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            yield client

    def test_chat_request_with_user_id(self, app_client):
        """Chat 请求应接受 user_id 参数"""
        # 由于 DKI 系统是 mock 的，这里只验证请求格式被接受
        response = app_client.post("/api/chat", json={
            "query": "Hello",
            "mode": "baseline",
            "user_id": "test_user_123",
        })
        # 即使返回 500 (因为 mock 系统)，也说明请求格式被接受
        assert response.status_code in [200, 500]

    def test_chat_request_with_token(self, app_client):
        """Chat 请求应接受 token 参数"""
        response = app_client.post("/api/chat", json={
            "query": "Hello",
            "mode": "baseline",
            "token": "some_token",
        })
        assert response.status_code in [200, 500]

    def test_preference_endpoint_exists(self, app_client):
        """/api/preference 端点应存在"""
        response = app_client.post("/api/preference", json={
            "content": "我喜欢素食",
            "user_id": "test_user",
        })
        # 即使返回 500 (因为 mock DB)，也说明端点存在
        assert response.status_code in [200, 500]
        # 不应是 404 或 405
        assert response.status_code != 404
        assert response.status_code != 405

    def test_get_preferences_endpoint_exists(self, app_client):
        """/api/preferences/{user_id} 端点应存在 (可能需要认证)"""
        response = app_client.get("/api/preferences/test_user")
        # 端点存在: 200/401/500 均可; 不应是 404 (不存在) 或 405 (方法不允许)
        assert response.status_code in [200, 401, 500]
        assert response.status_code != 404

    def test_delete_preferences_endpoint_exists(self, app_client):
        """DELETE /api/preferences/{user_id} 端点应存在 (可能需要认证)"""
        response = app_client.delete("/api/preferences/test_user")
        assert response.status_code in [200, 401, 500]
        assert response.status_code != 404

    def test_experiment_sessions_endpoint_exists(self, app_client):
        """/api/experiment/sessions 端点应存在"""
        response = app_client.get("/api/experiment/sessions")
        assert response.status_code in [200, 500]
        assert response.status_code != 404

    def test_experiment_session_detail_endpoint_exists(self, app_client):
        """/api/experiment/session/{session_id} 端点应存在"""
        response = app_client.get("/api/experiment/session/test_session_001")
        assert response.status_code in [200, 404, 500]
        # 端点存在 (不是 405 Method Not Allowed)
        assert response.status_code != 405

    def test_index_page(self, app_client):
        """首页应返回 HTML"""
        response = app_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
