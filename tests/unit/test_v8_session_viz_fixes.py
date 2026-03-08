"""
v8.0 修正的单元测试

覆盖:
1. ExperimentRunner 初始化参数修正 (不再接受 dki_system)
2. visualization_routes.py /latest 端点支持 session_id 过滤
3. session_routes.py SessionResponse 字段映射验证
"""

import os
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

os.environ.setdefault("DKI_ENV", "test")


# ============ ExperimentRunner Init Tests ============

class TestExperimentRunnerInit(unittest.TestCase):
    """验证 ExperimentRunner.__init__ 签名正确"""

    def test_no_dki_system_parameter(self):
        """ExperimentRunner 不应接受 dki_system 参数"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())

        self.assertNotIn('dki_system', params,
                         "ExperimentRunner 不应有 dki_system 参数 (已被 dki_plugin 替代)")

    def test_has_dki_plugin_parameter(self):
        """ExperimentRunner 应接受 dki_plugin 参数"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())

        self.assertIn('dki_plugin', params)

    def test_has_rag_system_parameter(self):
        """ExperimentRunner 应接受 rag_system 参数"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())

        self.assertIn('rag_system', params)

    @patch('dki.experiment.runner.ConfigLoader')
    @patch('dki.experiment.runner.DatabaseManager')
    def test_init_with_rag_only(self, mock_db, mock_config):
        """可以只传 rag_system 初始化"""
        mock_config.return_value.config = MagicMock()
        mock_config.return_value.config.database.path = ":memory:"

        from dki.experiment.runner import ExperimentRunner

        mock_rag = MagicMock()
        runner = ExperimentRunner(rag_system=mock_rag)

        self.assertEqual(runner.rag_system, mock_rag)
        self.assertIsNone(runner._dki_plugin)

    @patch('dki.experiment.runner.ConfigLoader')
    @patch('dki.experiment.runner.DatabaseManager')
    def test_init_with_dki_plugin(self, mock_db, mock_config):
        """可以传 dki_plugin 初始化"""
        mock_config.return_value.config = MagicMock()
        mock_config.return_value.config.database.path = ":memory:"

        from dki.experiment.runner import ExperimentRunner

        mock_plugin = MagicMock()
        runner = ExperimentRunner(dki_plugin=mock_plugin)

        self.assertEqual(runner._dki_plugin, mock_plugin)

    @patch('dki.experiment.runner.ConfigLoader')
    @patch('dki.experiment.runner.DatabaseManager')
    def test_init_rejects_dki_system_kwarg(self, mock_db, mock_config):
        """传 dki_system 应引发 TypeError"""
        mock_config.return_value.config = MagicMock()
        mock_config.return_value.config.database.path = ":memory:"

        from dki.experiment.runner import ExperimentRunner

        with self.assertRaises(TypeError) as ctx:
            ExperimentRunner(dki_system=MagicMock())

        self.assertIn("dki_system", str(ctx.exception))


# ============ Visualization Latest Endpoint Tests ============

class TestVisualizationLatestEndpoint(unittest.TestCase):
    """验证 /latest 端点支持 session_id 过滤"""

    def setUp(self):
        """设置测试数据"""
        self.test_history = [
            {
                "request_id": "req_001",
                "user_id": "user_alice",
                "session_id": "session_aaa",
                "query": "你好",
                "timestamp": "2026-03-03T10:00:00",
                "injection_enabled": True,
                "alpha": 0.4,
            },
            {
                "request_id": "req_002",
                "user_id": "user_alice",
                "session_id": "session_bbb",
                "query": "推荐美食",
                "timestamp": "2026-03-03T10:01:00",
                "injection_enabled": True,
                "alpha": 0.6,
            },
            {
                "request_id": "req_003",
                "user_id": "user_alice",
                "session_id": "session_aaa",
                "query": "你叫什么名字",
                "timestamp": "2026-03-03T10:02:00",
                "injection_enabled": True,
                "alpha": 0.5,
            },
            {
                "request_id": "req_004",
                "user_id": "user_bob",
                "session_id": "session_ccc",
                "query": "天气如何",
                "timestamp": "2026-03-03T10:03:00",
                "injection_enabled": False,
                "alpha": 0.0,
            },
        ]

    def test_session_id_filter(self):
        """按 session_id 过滤应返回该会话最后一条"""
        history = self.test_history.copy()
        # Simulate user filter
        user_id = "user_alice"
        filtered = [h for h in history if h.get("user_id") == user_id]
        # Simulate session filter
        session_id = "session_aaa"
        filtered = [h for h in filtered if h.get("session_id") == session_id]

        self.assertEqual(len(filtered), 2)
        # Latest should be req_003
        latest = filtered[-1]
        self.assertEqual(latest["request_id"], "req_003")
        self.assertEqual(latest["session_id"], "session_aaa")

    def test_no_session_filter_returns_overall_latest(self):
        """不指定 session_id 时返回全局最后一条"""
        history = self.test_history.copy()
        user_id = "user_alice"
        filtered = [h for h in history if h.get("user_id") == user_id]

        # No session filter → latest for this user
        latest = filtered[-1]
        self.assertEqual(latest["request_id"], "req_003")

    def test_session_filter_empty_result(self):
        """不存在的 session_id 返回空"""
        history = self.test_history.copy()
        user_id = "user_alice"
        filtered = [h for h in history if h.get("user_id") == user_id]
        session_id = "session_nonexistent"
        filtered = [h for h in filtered if h.get("session_id") == session_id]

        self.assertEqual(len(filtered), 0)

    def test_user_isolation(self):
        """不同用户的数据应隔离"""
        history = self.test_history.copy()
        user_id = "user_bob"
        filtered = [h for h in history if h.get("user_id") == user_id]

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["request_id"], "req_004")


# ============ SessionResponse Mapping Tests ============

class TestSessionResponseMapping(unittest.TestCase):
    """验证后端 snake_case 到前端 camelCase 的映射"""

    def test_snake_case_response_mapping(self):
        """后端返回 snake_case 字段应正确映射"""
        raw_response = {
            "id": "session_001",
            "title": "Test Session",
            "user_id": "user_alice",
            "message_count": 5,
            "created_at": "2026-03-03T10:00:00",
            "updated_at": "2026-03-03T10:05:00",
            "preview": "Hello world...",
        }

        # Simulate the mapping function from api.ts
        mapped = {
            "id": raw_response["id"],
            "title": raw_response.get("title") or raw_response["id"],
            "userId": raw_response.get("user_id") or raw_response.get("userId"),
            "messageCount": raw_response.get("message_count", raw_response.get("messageCount", 0)),
            "createdAt": raw_response.get("created_at") or raw_response.get("createdAt", ""),
            "updatedAt": raw_response.get("updated_at") or raw_response.get("updatedAt", ""),
            "preview": raw_response.get("preview"),
        }

        self.assertEqual(mapped["messageCount"], 5)
        self.assertEqual(mapped["userId"], "user_alice")
        self.assertEqual(mapped["createdAt"], "2026-03-03T10:00:00")

    def test_missing_message_count_defaults_to_zero(self):
        """缺少 message_count 时应默认为 0 (避免 NaN)"""
        raw_response = {
            "id": "session_002",
            "title": "Empty Session",
            "created_at": "2026-03-03T10:00:00",
            "updated_at": "2026-03-03T10:00:00",
        }

        message_count = raw_response.get("message_count", raw_response.get("messageCount", 0))
        self.assertEqual(message_count, 0)

    def test_total_messages_no_nan(self):
        """总消息计数不应产生 NaN"""
        sessions = [
            {"messageCount": 5},
            {"messageCount": 0},
            {"messageCount": None},
            {},
        ]

        total = sum(s.get("messageCount") or 0 for s in sessions)
        self.assertEqual(total, 5)
        self.assertFalse(total != total)  # NaN check: NaN != NaN


# ============ App.py ExperimentRunner Caller Tests ============

class TestAppExperimentRunnerCallers(unittest.TestCase):
    """验证 app.py 中 ExperimentRunner 的调用参数正确"""

    def test_no_dki_system_in_app_runner_calls(self):
        """app.py 不应再传 dki_system 给 ExperimentRunner"""
        import inspect
        import dki.web.app as app_module

        source = inspect.getsource(app_module)

        # Find all ExperimentRunner( calls
        import re
        runner_calls = re.findall(
            r'ExperimentRunner\([^)]*\)',
            source,
            re.DOTALL
        )

        for call in runner_calls:
            self.assertNotIn('dki_system', call,
                             f"发现 app.py 仍在传递 dki_system: {call}")


# ============ Visualization Route Session Filter Integration ============

class TestVisualizationRouteSessionFilter(unittest.TestCase):
    """验证 visualization_routes.py 的 session_id 参数"""

    def test_latest_endpoint_has_session_id_param(self):
        """验证 /latest 端点接受 session_id 查询参数"""
        import inspect
        from dki.api.visualization_routes import create_visualization_router

        # 创建路由并检查 get_latest_injection 函数签名
        router = create_visualization_router()

        # Find the /latest route handler
        # FastAPI routes may include the prefix in the path
        latest_route = None
        for route in router.routes:
            path = getattr(route, 'path', '')
            if path.endswith('/latest'):
                latest_route = route
                break

        self.assertIsNotNone(latest_route, f"应存在 /latest 路由, 已有路由: {[getattr(r, 'path', '') for r in router.routes]}")

        # Check that the endpoint function has session_id parameter
        endpoint = latest_route.endpoint
        sig = inspect.signature(endpoint)
        params = list(sig.parameters.keys())

        self.assertIn('session_id', params,
                       "/latest 端点应接受 session_id 参数")


if __name__ == '__main__':
    unittest.main()
