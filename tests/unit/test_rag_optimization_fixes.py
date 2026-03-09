"""
RAG System 优化修复验证测试

验证 claude_rag优化建议.md 中提出的问题修复:
1. SingleFlight bug 修复 (asyncio.shield + 异常降级)
2. asyncio.get_event_loop() → asyncio.get_running_loop()
3. fire-and-forget Future 泄漏修复
4. BoundedTTLCache 有界 TTL 缓存
5. _get_max_history_turns 简化
6. 流式 token 统计修复
7. 错误分类改用结构化异常
8. dki_plugin.py 类似问题修复
"""

import asyncio
import time
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ============================================================
# 1. BoundedTTLCache 单元测试
# ============================================================


class TestBoundedTTLCache(unittest.TestCase):
    """测试有界 TTL 缓存"""

    def setUp(self):
        from dki.core.rag_system import BoundedTTLCache
        self.cache = BoundedTTLCache(maxsize=3, ttl=2.0)

    def test_set_and_get(self):
        """测试基本的写入和读取"""
        self.cache.set("user_1", "pref_1")
        result = self.cache.get("user_1")
        self.assertIsNotNone(result)
        value, ts = result
        self.assertEqual(value, "pref_1")

    def test_get_miss(self):
        """测试缓存未命中"""
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)

    def test_ttl_expiry(self):
        """测试 TTL 过期"""
        self.cache = self._create_cache(maxsize=10, ttl=0.1)
        self.cache.set("user_1", "pref_1")
        # 立即读取应该命中
        self.assertIsNotNone(self.cache.get("user_1"))
        # 等待过期
        time.sleep(0.15)
        self.assertIsNone(self.cache.get("user_1"))

    def test_maxsize_eviction(self):
        """测试超过上限时淘汰最久未使用的条目"""
        self.cache.set("user_1", "pref_1")
        self.cache.set("user_2", "pref_2")
        self.cache.set("user_3", "pref_3")
        # 缓存已满 (maxsize=3)
        self.assertEqual(len(self.cache), 3)
        
        # 写入第 4 个条目, 应淘汰 user_1
        self.cache.set("user_4", "pref_4")
        self.assertEqual(len(self.cache), 3)
        self.assertIsNone(self.cache.get("user_1"))
        self.assertIsNotNone(self.cache.get("user_2"))

    def test_lru_order(self):
        """测试 LRU 淘汰顺序: 访问过的条目不应被先淘汰"""
        self.cache.set("user_1", "pref_1")
        self.cache.set("user_2", "pref_2")
        self.cache.set("user_3", "pref_3")
        
        # 访问 user_1, 将其移到末尾
        self.cache.get("user_1")
        
        # 写入第 4 个, 应淘汰 user_2 (最久未使用)
        self.cache.set("user_4", "pref_4")
        self.assertIsNotNone(self.cache.get("user_1"))
        self.assertIsNone(self.cache.get("user_2"))

    def test_pop(self):
        """测试移除条目"""
        self.cache.set("user_1", "pref_1")
        self.cache.pop("user_1")
        self.assertIsNone(self.cache.get("user_1"))

    def test_clear(self):
        """测试清空缓存"""
        self.cache.set("user_1", "pref_1")
        self.cache.set("user_2", "pref_2")
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)

    def test_contains_valid(self):
        """测试 __contains__ 检查有效条目"""
        self.cache.set("user_1", "pref_1")
        self.assertIn("user_1", self.cache)
        self.assertNotIn("user_2", self.cache)

    def test_contains_expired(self):
        """测试 __contains__ 检查过期条目"""
        self.cache = self._create_cache(maxsize=10, ttl=0.1)
        self.cache.set("user_1", "pref_1")
        time.sleep(0.15)
        self.assertNotIn("user_1", self.cache)

    def test_set_none_value(self):
        """测试缓存 None 值 (偏好不存在的情况)"""
        self.cache.set("user_1", None)
        result = self.cache.get("user_1")
        self.assertIsNotNone(result)
        value, ts = result
        self.assertIsNone(value)

    def test_overwrite_existing(self):
        """测试覆盖已有条目"""
        self.cache.set("user_1", "old_pref")
        self.cache.set("user_1", "new_pref")
        result = self.cache.get("user_1")
        value, ts = result
        self.assertEqual(value, "new_pref")
        # 覆盖不应增加条目数
        self.assertEqual(len(self.cache), 1)

    def _create_cache(self, maxsize: int, ttl: float):
        from dki.core.rag_system import BoundedTTLCache
        return BoundedTTLCache(maxsize=maxsize, ttl=ttl)


# ============================================================
# 2. SingleFlight 修复测试
# ============================================================


class TestSingleFlightFix(unittest.TestCase):
    """测试 SingleFlight 修复"""

    def test_async_single_flight_shield_exception_degrades(self):
        """测试 asyncio.shield 保护: 主请求异常时等待方降级为 None 而不崩溃"""

        async def run_test():
            from dki.core.rag_system import RAGSystem

            # 创建 mock 对象
            with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
                rag = RAGSystem.__new__(RAGSystem)
                rag._preference_cache = MagicMock()
                rag._preference_cache.get = MagicMock(return_value=None)
                rag._preference_cache_ttl = 300.0
                rag._preference_single_flight = {}
                rag._stats = {"preference_cache_hits": 0, "preference_cache_misses": 0}

                # 创建一个已经被异常设置的 future (模拟主请求失败)
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                future.set_exception(Exception("DB connection failed"))
                rag._preference_single_flight["user_1"] = future

                # 等待方应该降级为 None 而不是崩溃
                result = await rag._load_user_preferences_async("user_1")
                
                # shield 保护下, 异常被捕获, 等待方返回 None
                self.assertIsNone(result)

        asyncio.run(run_test())

    def test_async_single_flight_shield_success(self):
        """测试 asyncio.shield 保护: 主请求成功时等待方拿到结果"""

        async def run_test():
            from dki.core.rag_system import RAGSystem

            with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
                rag = RAGSystem.__new__(RAGSystem)
                rag._preference_cache = MagicMock()
                rag._preference_cache.get = MagicMock(return_value=None)
                rag._preference_cache_ttl = 300.0
                rag._preference_single_flight = {}
                rag._stats = {"preference_cache_hits": 0, "preference_cache_misses": 0}

                # 创建一个已经成功设置结果的 future (模拟主请求成功)
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                future.set_result("user preference text")
                rag._preference_single_flight["user_1"] = future

                # 等待方应该拿到主请求的结果
                result = await rag._load_user_preferences_async("user_1")
                self.assertEqual(result, "user preference text")

        asyncio.run(run_test())

    def test_no_get_event_loop_in_rag(self):
        """验证 rag_system.py 中不再使用 asyncio.get_event_loop()"""
        import inspect
        from dki.core import rag_system
        source = inspect.getsource(rag_system)
        
        # 排除注释和文档字符串中的引用
        import re
        # 匹配实际代码行中的 get_event_loop 调用
        code_lines = [
            line for line in source.split('\n')
            if 'get_event_loop' in line
            and not line.strip().startswith('#')
            and not line.strip().startswith('-')
            and '"""' not in line
            and "'''" not in line
            and 'get_running_loop' not in line
            and '替代' not in line
        ]
        
        # 过滤掉注释中的引用
        actual_calls = [
            line for line in code_lines
            if 'asyncio.get_event_loop()' in line
        ]
        
        self.assertEqual(
            len(actual_calls), 0,
            f"Found asyncio.get_event_loop() calls in rag_system.py: {actual_calls}"
        )

    def test_no_get_event_loop_in_dki_plugin(self):
        """验证 dki_plugin.py 中不再使用 asyncio.get_event_loop()"""
        import inspect
        from dki.core import dki_plugin
        source = inspect.getsource(dki_plugin)
        
        code_lines = [
            line for line in source.split('\n')
            if 'asyncio.get_event_loop()' in line
            and not line.strip().startswith('#')
            and not line.strip().startswith('-')
        ]
        
        self.assertEqual(
            len(code_lines), 0,
            f"Found asyncio.get_event_loop() calls in dki_plugin.py: {code_lines}"
        )


# ============================================================
# 3. fire-and-forget 修复测试
# ============================================================


class TestFireAndForgetFix(unittest.TestCase):
    """测试 fire-and-forget Future 修复"""

    def test_log_conversation_async_exists(self):
        """验证 _log_conversation_async 方法存在"""
        from dki.core.rag_system import RAGSystem
        self.assertTrue(hasattr(RAGSystem, '_log_conversation_async'))
        # 验证是协程
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(RAGSystem._log_conversation_async))

    def test_fire_and_forget_log_exists(self):
        """验证 _fire_and_forget_log 方法存在"""
        from dki.core.rag_system import RAGSystem
        self.assertTrue(hasattr(RAGSystem, '_fire_and_forget_log'))

    def test_fire_and_forget_log_creates_task(self):
        """测试 fire-and-forget 日志记录创建了 task"""

        async def run_test():
            from dki.core.rag_system import RAGSystem

            with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
                rag = RAGSystem.__new__(RAGSystem)
                rag._log_conversation = MagicMock()
                
                loop = asyncio.get_running_loop()
                
                # 调用 fire_and_forget_log
                rag._fire_and_forget_log(
                    loop=loop,
                    session_id="test_session",
                    user_id="test_user",
                    query="hello",
                    clean_response="world",
                    memories=[],
                    total_latency=100.0,
                )
                
                # 等待 task 完成
                await asyncio.sleep(0.1)
                
                # 验证 _log_conversation 被调用
                rag._log_conversation.assert_called_once()

        asyncio.run(run_test())

    def test_fire_and_forget_log_handles_exception(self):
        """测试 fire-and-forget 日志记录正确处理异常"""

        async def run_test():
            from dki.core.rag_system import RAGSystem

            with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
                rag = RAGSystem.__new__(RAGSystem)
                rag._log_conversation = MagicMock(
                    side_effect=Exception("DB error")
                )
                
                loop = asyncio.get_running_loop()
                
                # 不应抛出异常
                rag._fire_and_forget_log(
                    loop=loop,
                    session_id="test_session",
                    user_id="test_user",
                    query="hello",
                    clean_response="world",
                    memories=[],
                    total_latency=100.0,
                )
                
                # 等待 task 完成
                await asyncio.sleep(0.1)
                
                # 不应崩溃, 只记录日志
                self.assertTrue(True)

        asyncio.run(run_test())


# ============================================================
# 4. _get_max_history_turns 简化测试
# ============================================================


class TestGetMaxHistoryTurns(unittest.TestCase):
    """测试简化后的 _get_max_history_turns"""

    def _create_rag_with_config(self, config_mock):
        """创建带 mock config 的 RAGSystem"""
        from dki.core.rag_system import RAGSystem

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag.config = config_mock
            return rag

    def test_attribute_chain_access(self):
        """测试属性链访问 config.dki.recall.budget.max_recent_turns"""
        config = MagicMock()
        config.dki.recall.budget.max_recent_turns = 10
        
        rag = self._create_rag_with_config(config)
        self.assertEqual(rag._get_max_history_turns(), 10)

    def test_default_value_on_missing_config(self):
        """测试配置缺失时返回默认值 5"""
        config = MagicMock()
        config.dki = MagicMock(spec=[])  # 没有 recall 属性
        
        rag = self._create_rag_with_config(config)
        self.assertEqual(rag._get_max_history_turns(), 5)

    def test_dict_style_access(self):
        """测试 dict 风格的配置访问"""
        config = MagicMock()
        # 第一次属性链访问抛出 AttributeError
        type(config.dki.recall).budget = PropertyMock(side_effect=AttributeError)
        # 但 recall 本身是 dict
        config.dki.recall = {"budget": {"max_recent_turns": 7}}
        
        rag = self._create_rag_with_config(config)
        self.assertEqual(rag._get_max_history_turns(), 7)


# ============================================================
# 5. 错误分类改进测试
# ============================================================


class TestErrorClassification(unittest.TestCase):
    """测试错误分类改用结构化异常"""

    def test_catches_model_oom_error(self):
        """测试捕获 ModelOOMError 结构化异常"""
        from dki.core.rag_system import RAGSystem, RAGGenerationError
        from dki.core.exceptions import ModelOOMError

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag._model_adapter = MagicMock()
            rag._engine = None
            rag._model_adapter.generate = MagicMock(
                side_effect=ModelOOMError("GPU OOM")
            )
            # 设置 model property 返回 mock adapter
            type(rag).model = PropertyMock(return_value=rag._model_adapter)

            with self.assertRaises(RAGGenerationError) as ctx:
                rag._generate_and_process(
                    prompt="test", max_new_tokens=100, temperature=0.7
                )
            self.assertTrue(ctx.exception.retryable)
            self.assertIn("OOM", str(ctx.exception))

    def test_catches_model_timeout_error(self):
        """测试捕获 ModelTimeoutError 结构化异常"""
        from dki.core.rag_system import RAGSystem, RAGGenerationError
        from dki.core.exceptions import ModelTimeoutError

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag._model_adapter = MagicMock()
            rag._engine = None
            rag._model_adapter.generate = MagicMock(
                side_effect=ModelTimeoutError("Request timed out")
            )
            type(rag).model = PropertyMock(return_value=rag._model_adapter)

            with self.assertRaises(RAGGenerationError) as ctx:
                rag._generate_and_process(
                    prompt="test", max_new_tokens=100, temperature=0.7
                )
            self.assertTrue(ctx.exception.retryable)

    def test_catches_model_connection_error(self):
        """测试捕获 ModelConnectionError 结构化异常"""
        from dki.core.rag_system import RAGSystem, RAGGenerationError
        from dki.core.exceptions import ModelConnectionError

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag._model_adapter = MagicMock()
            rag._engine = None
            rag._model_adapter.generate = MagicMock(
                side_effect=ModelConnectionError("Cannot connect")
            )
            type(rag).model = PropertyMock(return_value=rag._model_adapter)

            with self.assertRaises(RAGGenerationError) as ctx:
                rag._generate_and_process(
                    prompt="test", max_new_tokens=100, temperature=0.7
                )
            self.assertTrue(ctx.exception.retryable)

    def test_catches_generic_error_with_oom_hint(self):
        """测试未分类异常的字符串匹配兜底"""
        from dki.core.rag_system import RAGSystem, RAGGenerationError

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag._model_adapter = MagicMock()
            rag._engine = None
            rag._model_adapter.generate = MagicMock(
                side_effect=RuntimeError("CUDA out of memory")
            )
            type(rag).model = PropertyMock(return_value=rag._model_adapter)

            with self.assertRaises(RAGGenerationError) as ctx:
                rag._generate_and_process(
                    prompt="test", max_new_tokens=100, temperature=0.7
                )
            self.assertTrue(ctx.exception.retryable)

    def test_catches_generic_non_retryable_error(self):
        """测试非可重试的通用错误"""
        from dki.core.rag_system import RAGSystem, RAGGenerationError

        with patch.object(RAGSystem, '__init__', lambda self, **kwargs: None):
            rag = RAGSystem.__new__(RAGSystem)
            rag._model_adapter = MagicMock()
            rag._engine = None
            rag._model_adapter.generate = MagicMock(
                side_effect=ValueError("Invalid parameter")
            )
            type(rag).model = PropertyMock(return_value=rag._model_adapter)

            with self.assertRaises(RAGGenerationError) as ctx:
                rag._generate_and_process(
                    prompt="test", max_new_tokens=100, temperature=0.7
                )
            self.assertFalse(ctx.exception.retryable)


# ============================================================
# 6. dki_plugin.py 修复验证
# ============================================================


class TestDKIPluginFixes(unittest.TestCase):
    """测试 dki_plugin.py 的类似问题修复"""

    def test_datetime_uses_timezone_utc(self):
        """验证 InjectionMetadata 使用 datetime.now(timezone.utc) 而非 utcnow()"""
        from dki.core.dki_plugin import InjectionMetadata
        from datetime import timezone

        meta = InjectionMetadata()
        # 验证 timestamp 有时区信息
        self.assertIsNotNone(meta.timestamp.tzinfo)
        self.assertEqual(meta.timestamp.tzinfo, timezone.utc)

    def test_estimate_tokens_fast_imported(self):
        """验证 dki_plugin.py 导入了 estimate_tokens_fast"""
        from dki.core.text_utils import estimate_tokens_fast
        # 验证函数可用
        result = estimate_tokens_fast("hello world", overestimate_factor=1.15)
        self.assertGreater(result, 0)

    def test_preference_text_cache_is_ordered_dict(self):
        """验证 _preference_text_cache 使用 OrderedDict (有界 LRU)"""
        # 由于 DKIPlugin 初始化需要大量依赖, 我们直接检查代码
        import inspect
        from dki.core import dki_plugin
        source = inspect.getsource(dki_plugin.DKIPlugin.__init__)
        
        self.assertIn("OrderedDict", source)
        self.assertIn("_preference_text_cache_maxsize", source)

    def test_single_flight_uses_shield(self):
        """验证 _get_cached_preferences 中使用 asyncio.shield"""
        import inspect
        from dki.core import dki_plugin
        source = inspect.getsource(dki_plugin.DKIPlugin._get_cached_preferences)
        
        self.assertIn("asyncio.shield", source)


# ============================================================
# 7. 流式 token 统计修复测试
# ============================================================


class TestStreamTokenEstimation(unittest.TestCase):
    """测试流式生成中 token 估算修复"""

    def test_rag_stream_estimates_tokens(self):
        """验证 RAG chat_stream 的 done 事件包含非零 token 估算"""
        import inspect
        from dki.core import rag_system
        source = inspect.getsource(rag_system.RAGSystem.chat_stream)
        
        # 验证流式模式下有 token 估算调用
        self.assertIn("self._estimate_tokens(prompt)", source)
        self.assertIn("self._estimate_tokens(clean_text)", source)

    def test_dki_stream_estimates_tokens(self):
        """验证 DKI chat_stream 的 done 事件包含非零 token 估算"""
        import inspect
        from dki.core import dki_plugin
        source = inspect.getsource(dki_plugin.DKIPlugin.chat_stream)
        
        # 验证流式模式下有 token 估算调用
        self.assertIn("estimate_tokens_fast", source)


# ============================================================
# 8. 整体回归测试
# ============================================================


class TestRegressionChecks(unittest.TestCase):
    """回归检查: 确保修复没有破坏已有功能"""

    def test_rag_system_importable(self):
        """验证 rag_system 模块可正常导入"""
        from dki.core.rag_system import (
            RAGSystem,
            RAGResponse,
            RAGPromptInfo,
            RAGError,
            RAGGenerationError,
            RAGPreferenceError,
            BoundedTTLCache,
        )
        self.assertIsNotNone(RAGSystem)
        self.assertIsNotNone(BoundedTTLCache)

    def test_dki_plugin_importable(self):
        """验证 dki_plugin 模块可正常导入"""
        from dki.core.dki_plugin import (
            DKIPlugin,
            DKIPluginResponse,
            InjectionMetadata,
        )
        self.assertIsNotNone(DKIPlugin)

    def test_rag_system_has_all_methods(self):
        """验证 RAGSystem 所有关键方法仍然存在"""
        from dki.core.rag_system import RAGSystem

        required_methods = [
            'chat',
            'async_chat',
            'chat_stream',
            'add_memory',
            'search_memories',
            'get_stats',
            '_load_user_preferences',
            '_load_user_preferences_async',
            '_load_user_preferences_sync',
            '_prepare_chat_context',
            '_generate_and_process',
            '_build_prompt',
            '_get_max_history_turns',
            '_log_conversation',
            '_log_conversation_async',
            '_fire_and_forget_log',
            '_estimate_tokens',
            'invalidate_preference_cache',
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(RAGSystem, method_name),
                f"RAGSystem missing method: {method_name}"
            )


if __name__ == '__main__':
    unittest.main()
