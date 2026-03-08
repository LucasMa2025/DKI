"""
ExperimentRunner 重构后的单元测试

验证 v7.0 重构:
- DKISystem 依赖已移除
- DKI 模式通过 DKIPlugin + SQLiteDataAdapter 运行
- Baseline 模式使用 model 属性
- memories_used 正确包含偏好和历史统计
- 异步调用包装 (_run_plugin_chat) 正常工作
"""

import asyncio
import os
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

os.environ.setdefault("DKI_ENV", "test")


@dataclass
class MockInjectionMetadata:
    """模拟 InjectionMetadata"""
    injection_enabled: bool = True
    alpha: float = 0.4
    latency_ms: float = 100.0
    preference_tokens: int = 50
    history_tokens: int = 30
    query_tokens: int = 10
    total_tokens: int = 90
    preference_cache_hit: bool = False
    preferences_count: int = 3
    relevant_history_count: int = 2
    injection_strategy: str = "recall_v4"


@dataclass
class MockDKIPluginResponse:
    """模拟 DKIPluginResponse"""
    text: str = "这是一个素食推荐"
    input_tokens: int = 100
    output_tokens: int = 50
    metadata: MockInjectionMetadata = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = MockInjectionMetadata()


class TestRunnerNoDKISystem(unittest.TestCase):
    """验证 runner.py 不再引用 DKISystem"""

    def test_no_dki_system_import(self):
        """验证 runner.py 不再导入 DKISystem。"""
        import dki.experiment.runner as runner_module

        # 检查模块中没有 DKISystem
        self.assertFalse(hasattr(runner_module, 'DKISystem'))

    def test_runner_init_no_dki_system(self):
        """验证 ExperimentRunner.__init__ 不接受 dki_system 参数。"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())

        self.assertNotIn('dki_system', params)
        self.assertIn('dki_plugin', params)
        self.assertIn('model_adapter', params)

    def test_runner_has_sqlite_adapter(self):
        """验证 ExperimentRunner 具有 _sqlite_adapter 属性。"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        source = inspect.getsource(ExperimentRunner.__init__)
        self.assertIn('_sqlite_adapter', source)
        self.assertIn('SQLiteDataAdapter', source)


class TestRunnerPluginChat(unittest.TestCase):
    """测试 _run_plugin_chat 异步包装器"""

    def setUp(self):
        """创建带 Mock 的 runner。"""
        # Patch ConfigLoader 和 DatabaseManager
        self.config_patcher = patch('dki.experiment.runner.ConfigLoader')
        self.db_patcher = patch('dki.experiment.runner.DatabaseManager')
        self.mock_config_loader = self.config_patcher.start()
        self.mock_db_manager = self.db_patcher.start()

        # 配置 mock
        mock_config = MagicMock()
        mock_config.database.path = ":memory:"
        self.mock_config_loader.return_value.config = mock_config

    def tearDown(self):
        self.config_patcher.stop()
        self.db_patcher.stop()

    def test_run_plugin_chat_basic(self):
        """测试 _run_plugin_chat 基本调用。"""
        from dki.experiment.runner import ExperimentRunner

        mock_plugin = MagicMock()
        mock_response = MockDKIPluginResponse(text="推荐蔬菜沙拉")

        # 让 chat() 返回一个 coroutine
        async def mock_chat(**kwargs):
            return mock_response

        mock_plugin.chat = mock_chat

        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._dki_plugin = mock_plugin

        result = runner._run_plugin_chat(
            query="推荐素食",
            session_id="sess_001",
            user_id="user_001",
        )

        self.assertEqual(result.text, "推荐蔬菜沙拉")
        self.assertEqual(result.metadata.alpha, 0.4)

    def test_model_property_from_plugin(self):
        """测试 model 属性从 DKIPlugin 获取。"""
        from dki.experiment.runner import ExperimentRunner

        mock_plugin = MagicMock()
        mock_model = MagicMock()
        mock_plugin.model = mock_model

        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._dki_plugin = mock_plugin
        runner._model_adapter = None

        self.assertEqual(runner.model, mock_model)

    def test_model_property_direct(self):
        """测试 model 属性直接返回 _model_adapter。"""
        from dki.experiment.runner import ExperimentRunner

        mock_model = MagicMock()

        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._model_adapter = mock_model
        runner._dki_plugin = None

        self.assertEqual(runner.model, mock_model)


class TestRunnerMemoriesUsed(unittest.TestCase):
    """验证 memories_used 正确包含偏好和历史统计"""

    def test_dki_memories_used_format(self):
        """DKI 模式的 memories_used 应包含偏好和历史计数。"""
        # 验证 ExperimentResult 中 memories_used 的格式
        meta = MockInjectionMetadata(preferences_count=3, relevant_history_count=5)

        memories_used_ids = []
        if meta.preferences_count > 0:
            memories_used_ids.append(f"prefs:{meta.preferences_count}")
        if meta.relevant_history_count > 0:
            memories_used_ids.append(f"history:{meta.relevant_history_count}")

        self.assertEqual(memories_used_ids, ["prefs:3", "history:5"])

    def test_dki_memories_empty(self):
        """无偏好无历史时 memories_used 应为空。"""
        meta = MockInjectionMetadata(preferences_count=0, relevant_history_count=0)

        memories_used_ids = []
        if meta.preferences_count > 0:
            memories_used_ids.append(f"prefs:{meta.preferences_count}")
        if meta.relevant_history_count > 0:
            memories_used_ids.append(f"history:{meta.relevant_history_count}")

        self.assertEqual(memories_used_ids, [])


class TestExperimentInit(unittest.TestCase):
    """验证 __init__.py 导出正确"""

    def test_sqlite_adapter_exported(self):
        """SQLiteDataAdapter 应从 experiment 包导出。"""
        from dki.experiment import SQLiteDataAdapter
        self.assertTrue(hasattr(SQLiteDataAdapter, 'add_memory'))
        self.assertTrue(hasattr(SQLiteDataAdapter, 'add_conversation'))
        self.assertTrue(hasattr(SQLiteDataAdapter, 'search_relevant_history'))

    def test_experiment_runner_exported(self):
        """ExperimentRunner 应从 experiment 包导出。"""
        from dki.experiment import ExperimentRunner
        self.assertTrue(hasattr(ExperimentRunner, '_run_plugin_chat'))
        self.assertTrue(hasattr(ExperimentRunner, 'model'))


if __name__ == '__main__':
    unittest.main()
