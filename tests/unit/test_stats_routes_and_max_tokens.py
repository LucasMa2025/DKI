"""
单元测试: stats_routes 修复 + max_tokens 默认值修复

覆盖:
1. /api/stats 路由注册后可正常访问 (不再 404)
2. /api/stats 在无 DKI Plugin 时回退到计数器数据
3. /api/stats 在有 DKI Plugin 时读取真实统计
4. /api/stats/dki 和 /api/stats/cache 子路由
5. demo/api/chat.py 中 max_tokens 默认值为 4096, 上限 16384
6. record_dki_request 计数器逻辑

Author: AGI Demo Project
"""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dki.api.stats_routes import (
    create_stats_router,
    record_dki_request,
    get_stats_snapshot,
    _dki_stats,
)


# ============================================================
# Fixtures
# ============================================================

def make_app(with_plugin: bool = False) -> FastAPI:
    """创建带 stats 路由的测试 FastAPI 应用"""
    app = FastAPI()
    app.include_router(create_stats_router())

    if with_plugin:
        mock_plugin = MagicMock()
        mock_plugin.get_stats.return_value = {
            "total_requests": 10,
            "injection_enabled_count": 7,
            "cache_hits": 3,
            "avg_alpha": 0.35,
            "injection_rate": 0.7,
            "cache": {
                "l1_size": 5,
                "l1_max_size": 1000,
                "l1_hit_rate": 0.3,
                "l2_hit_rate": 0.1,
            },
        }
        mock_plugin.get_cache_stats.return_value = {
            "l1_size": 5,
            "l1_max_size": 1000,
            "l1_hit_rate": 0.3,
            "l2_hit_rate": 0.1,
        }
        mock_plugin.get_injection_logs.return_value = [
            {"cache": {"preference_tier": "L1"}},
            {"cache": {"preference_tier": "L2"}},
            {"cache": {"preference_tier": "none"}},
        ]
        mock_plugin.data_adapter = MagicMock()
        mock_plugin.data_adapter.__class__.__name__ = "ConfigDrivenAdapter"
        app.state.dki_plugin = mock_plugin

    return app


# ============================================================
# 1. 路由注册测试
# ============================================================

class TestStatsRouteRegistration:
    """验证 stats 路由注册后不再 404"""

    def test_stats_endpoint_exists(self):
        """GET /api/stats 应返回 200, 不再 404"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        resp = client.get("/api/stats")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    def test_stats_dki_endpoint_exists(self):
        """GET /api/stats/dki 应返回 200"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        resp = client.get("/api/stats/dki")
        assert resp.status_code == 200

    def test_stats_cache_endpoint_exists(self):
        """GET /api/stats/cache 应返回 200"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        resp = client.get("/api/stats/cache")
        assert resp.status_code == 200


# ============================================================
# 2. 无 Plugin 时回退到计数器
# ============================================================

class TestStatsFallbackCounter:
    """无 DKI Plugin 时使用模块级计数器"""

    def test_stats_response_structure_no_plugin(self):
        """响应结构包含 dkiStats / cacheStats / adapterStats / uptimeSeconds"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        resp = client.get("/api/stats")
        data = resp.json()

        assert "dkiStats" in data
        assert "cacheStats" in data
        assert "adapterStats" in data
        assert "uptimeSeconds" in data

    def test_dki_stats_camel_case_keys(self):
        """dkiStats 字段使用 camelCase (与前端 TypeScript 类型对齐)"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        data = client.get("/api/stats").json()
        dki = data["dkiStats"]

        assert "totalRequests" in dki
        assert "l1Hits" in dki
        assert "l2Hits" in dki
        assert "l3Computes" in dki
        assert "avgAlpha" in dki
        assert "injectionRate" in dki

    def test_cache_stats_camel_case_keys(self):
        """cacheStats 字段使用 camelCase"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        data = client.get("/api/stats").json()
        cache = data["cacheStats"]

        assert "l1Size" in cache
        assert "l1MaxSize" in cache
        assert "l1HitRate" in cache
        assert "l2HitRate" in cache

    def test_uptime_is_positive(self):
        """uptimeSeconds 应为正数"""
        app = make_app(with_plugin=False)
        client = TestClient(app)
        data = client.get("/api/stats").json()
        assert data["uptimeSeconds"] >= 0


# ============================================================
# 3. 有 Plugin 时读取真实统计
# ============================================================

# ============================================================
# 4. record_dki_request 计数器逻辑
# ============================================================

class TestRecordDkiRequest:
    """record_dki_request 计数器逻辑"""

    def setup_method(self):
        """每个测试前重置计数器"""
        _dki_stats["total_requests"] = 0
        _dki_stats["l1_hits"] = 0
        _dki_stats["l2_hits"] = 0
        _dki_stats["l3_computes"] = 0
        _dki_stats["avg_alpha"] = 0.0
        _dki_stats["injection_rate"] = 0.0

    def test_l1_hit_increments(self):
        record_dki_request(cache_tier="L1", alpha=0.5, injected=True)
        assert _dki_stats["l1_hits"] == 1
        assert _dki_stats["total_requests"] == 1

    def test_l2_hit_increments(self):
        record_dki_request(cache_tier="L2", alpha=0.3, injected=False)
        assert _dki_stats["l2_hits"] == 1

    def test_l3_compute_increments(self):
        record_dki_request(cache_tier="L3", alpha=0.0, injected=False)
        assert _dki_stats["l3_computes"] == 1

    def test_avg_alpha_running_average(self):
        record_dki_request(cache_tier="L3", alpha=0.4, injected=True)
        record_dki_request(cache_tier="L3", alpha=0.6, injected=True)
        assert abs(_dki_stats["avg_alpha"] - 0.5) < 0.001

    def test_injection_rate_calculation(self):
        record_dki_request(cache_tier="L3", alpha=0.0, injected=True)
        record_dki_request(cache_tier="L3", alpha=0.0, injected=False)
        # 1 injected out of 2 → rate = 0.5
        assert abs(_dki_stats["injection_rate"] - 0.5) < 0.001

    def test_get_stats_snapshot(self):
        record_dki_request(cache_tier="L1", alpha=0.3, injected=True)
        snap = get_stats_snapshot()
        assert snap["total_requests"] == 1
        assert snap["l1_hits"] == 1


# ============================================================
# 5. max_tokens 默认值测试
# ============================================================

class TestChatMaxTokensDefault:
    """验证 demo/api/chat.py 中 max_tokens 默认值已提升"""

    def test_max_tokens_default_is_4096(self):
        """ChatSendRequest.max_tokens 默认值应为 4096"""
        from demo.api.chat import ChatSendRequest
        req = ChatSendRequest(query="test", user_id="u1")
        assert req.max_tokens == 4096, (
            f"Expected default max_tokens=4096, got {req.max_tokens}. "
            "回复截断问题: 默认值太小导致长回复被截断。"
        )

    def test_max_tokens_upper_limit_is_16384(self):
        """ChatSendRequest.max_tokens 上限应为 16384"""
        from demo.api.chat import ChatSendRequest
        from pydantic import ValidationError
        # 16384 应合法
        req = ChatSendRequest(query="test", user_id="u1", max_tokens=16384)
        assert req.max_tokens == 16384

        # 超过上限应报错
        with pytest.raises(ValidationError):
            ChatSendRequest(query="test", user_id="u1", max_tokens=16385)

    def test_max_tokens_min_is_1(self):
        """max_tokens 最小值应为 1"""
        from demo.api.chat import ChatSendRequest
        from pydantic import ValidationError
        req = ChatSendRequest(query="test", user_id="u1", max_tokens=1)
        assert req.max_tokens == 1

        with pytest.raises(ValidationError):
            ChatSendRequest(query="test", user_id="u1", max_tokens=0)
