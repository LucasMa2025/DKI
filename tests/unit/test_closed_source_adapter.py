"""
Unit Tests for Closed-Source Model Adapter

测试覆盖:
1. ClosedSourceAdapter — 初始化、API 调用、生成
2. Prompt 解析 — ChatML / Llama3 / 纯文本
3. SSE 解析 — 流式响应
4. ModelFactory — 闭源适配器注册与创建
5. EnhancedDKIPlugin — 闭源模型自动 RAG 路由
6. 配置兼容 — EngineConfig 闭源字段

Author: AGI Demo Project
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any, Dict, Optional


# ============================================================
# 1. ClosedSourceAdapter 基础测试
# ============================================================

class TestClosedSourceAdapterInit:
    """测试 ClosedSourceAdapter 初始化"""

    def test_default_init(self):
        """默认参数初始化"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter()
        assert adapter.model_name == "gpt-4o"
        assert adapter.api_base == "https://api.openai.com/v1"
        assert adapter.is_closed_source is True
        assert adapter.is_loaded is False
        assert adapter.device == "cpu"
        assert adapter.max_model_len == 8192

    def test_custom_init(self):
        """自定义参数初始化"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(
            model_name="deepseek-chat",
            api_key="sk-test-key",
            api_base="https://api.deepseek.com/v1",
            max_model_len=32768,
            timeout=60.0,
            max_retries=3,
        )
        assert adapter.model_name == "deepseek-chat"
        assert adapter.api_key == "sk-test-key"
        assert adapter.api_base == "https://api.deepseek.com/v1"
        assert adapter.max_model_len == 32768
        assert adapter.timeout == 60.0
        assert adapter.max_retries == 3

    def test_ignores_gpu_params(self):
        """忽略 GPU 相关参数但不报错"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(
            model_name="gpt-4o",
            device="cuda:0",
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            load_in_8bit=True,
        )
        # 设备强制为 cpu (闭源模型不需要 GPU)
        assert adapter.device == "cpu"
        assert adapter.is_closed_source is True


class TestClosedSourceAdapterLoadUnload:
    """测试 load / unload 生命周期"""

    def test_load_with_api_key(self):
        """提供 API key 时可以 load"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(
            model_name="test-model",
            api_key="sk-test",
        )
        # httpx 可能未安装, load 也不应失败 (有 fallback)
        adapter.load()
        assert adapter.is_loaded is True

    def test_load_without_api_key_raises(self):
        """没有 API key 且环境变量也没有时应 raise"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(model_name="test-model")
        # 清除可能存在的环境变量
        import os
        env_keys = [
            "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY",
            "ZHIPUAI_API_KEY", "MOONSHOT_API_KEY", "CLOSED_SOURCE_API_KEY",
        ]
        original_values = {}
        for key in env_keys:
            if key in os.environ:
                original_values[key] = os.environ.pop(key)

        try:
            with pytest.raises(ValueError, match="API key not provided"):
                adapter.load()
        finally:
            os.environ.update(original_values)

    def test_load_from_env(self):
        """从环境变量自动获取 API key"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        import os

        os.environ["CLOSED_SOURCE_API_KEY"] = "sk-env-test"
        try:
            adapter = ClosedSourceAdapter(model_name="test-model")
            adapter.load()
            assert adapter.is_loaded is True
            assert adapter.api_key == "sk-env-test"
        finally:
            del os.environ["CLOSED_SOURCE_API_KEY"]

    def test_unload(self):
        """unload 正确释放资源"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(
            model_name="test-model",
            api_key="sk-test",
        )
        adapter.load()
        assert adapter.is_loaded is True

        adapter.unload()
        assert adapter.is_loaded is False


# ============================================================
# 2. Prompt 解析测试
# ============================================================

class TestPromptParsing:
    """测试不同 prompt 格式的解析"""

    def setup_method(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.adapter = ClosedSourceAdapter(
            model_name="test-model",
            api_key="sk-test",
        )

    def test_plain_text_prompt(self):
        """纯文本 → user message"""
        messages = self.adapter._parse_prompt_to_messages("你好，请推荐餐厅")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好，请推荐餐厅"

    def test_plain_text_with_system_prompt(self):
        """纯文本 + default_system_prompt"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        adapter = ClosedSourceAdapter(
            model_name="test",
            api_key="sk-test",
            default_system_prompt="你是一个乐于助人的AI助手。",
        )
        messages = adapter._parse_prompt_to_messages("你好")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个乐于助人的AI助手。"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "你好"

    def test_chatml_prompt(self):
        """ChatML 格式解析"""
        prompt = (
            "<|im_start|>system\n你是一个助手<|im_end|>\n"
            "<|im_start|>user\n推荐餐厅<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = self.adapter._parse_chatml(prompt)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个助手"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "推荐餐厅"

    def test_llama3_prompt(self):
        """Llama 3 格式解析"""
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Hello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        messages = self.adapter._parse_llama3_format(prompt)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "You are a helpful assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "Hello" in messages[1]["content"]


# ============================================================
# 3. SSE 解析测试
# ============================================================

class TestSSEParsing:
    """测试 SSE 行解析"""

    def test_parse_normal_line(self):
        """正常 SSE 行"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        data = {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        line = f"data: {json.dumps(data)}"
        result = ClosedSourceAdapter._parse_sse_line(line)
        assert result == "Hello"

    def test_parse_done_line(self):
        """[DONE] 行"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        result = ClosedSourceAdapter._parse_sse_line("data: [DONE]")
        assert result is None

    def test_parse_empty_line(self):
        """空行"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        result = ClosedSourceAdapter._parse_sse_line("")
        assert result is None

    def test_parse_no_content_delta(self):
        """delta 中无 content"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        data = {"choices": [{"delta": {"role": "assistant"}}]}
        line = f"data: {json.dumps(data)}"
        result = ClosedSourceAdapter._parse_sse_line(line)
        assert result is None


# ============================================================
# 4. K/V 注入相关方法测试
# ============================================================

class TestClosedSourceKVMethods:
    """测试闭源模型不支持 K/V 注入"""

    def setup_method(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.adapter = ClosedSourceAdapter(
            model_name="test-model",
            api_key="sk-test",
        )

    def test_compute_kv_raises(self):
        """compute_kv 应 raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="compute_kv"):
            self.adapter.compute_kv("test text")

    def test_forward_with_kv_injection_raises(self):
        """forward_with_kv_injection 应 raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="K/V injection"):
            self.adapter.forward_with_kv_injection("test", None)

    def test_embed_raises(self):
        """embed 应 raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="embed"):
            self.adapter.embed("test")

    def test_compute_prefill_entropy_raises(self):
        """compute_prefill_entropy 应 raise NotImplementedError"""
        with pytest.raises(NotImplementedError, match="compute_prefill_entropy"):
            self.adapter.compute_prefill_entropy("test")


# ============================================================
# 5. 兼容性属性测试
# ============================================================

class TestClosedSourceCompatibility:
    """测试闭源模型与 BaseModelAdapter 接口的兼容性"""

    def setup_method(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.adapter = ClosedSourceAdapter(
            model_name="deepseek-chat",
            api_key="sk-test",
            api_base="https://api.deepseek.com/v1",
        )

    def test_is_quantized(self):
        assert self.adapter.is_quantized is False

    def test_is_4bit(self):
        assert self.adapter.is_4bit is False

    def test_is_8bit(self):
        assert self.adapter.is_8bit is False

    def test_flash_attn_enabled(self):
        assert self.adapter.flash_attn_enabled is False

    def test_get_model_info(self):
        info = self.adapter.get_model_info()
        assert info["model_name"] == "deepseek-chat"
        assert info["is_closed_source"] is True
        assert info["device"] == "api"
        assert info["api_base"] == "https://api.deepseek.com/v1"

    def test_repr(self):
        s = repr(self.adapter)
        assert "ClosedSourceAdapter" in s
        assert "deepseek-chat" in s

    def test_tokenize_estimation(self):
        """tokenize 应该返回估算值"""
        result = self.adapter.tokenize("Hello world 你好世界")
        assert "input_ids" in result
        assert len(result["input_ids"]) > 0


# ============================================================
# 6. API 请求构建测试
# ============================================================

class TestAPIRequestBuilding:
    """测试 API 请求构建"""

    def setup_method(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.adapter = ClosedSourceAdapter(
            model_name="gpt-4o",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )

    def test_build_headers(self):
        headers = self.adapter._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_build_chat_request(self):
        body = self.adapter._build_chat_request(
            prompt="Hello",
            max_new_tokens=100,
            temperature=0.5,
        )
        assert body["model"] == "gpt-4o"
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.5
        assert body["stream"] is False
        assert len(body["messages"]) >= 1
        assert body["messages"][-1]["role"] == "user"
        assert body["messages"][-1]["content"] == "Hello"

    def test_build_stream_request(self):
        body = self.adapter._build_chat_request(
            prompt="Hello",
            stream=True,
        )
        assert body["stream"] is True

    def test_get_api_url(self):
        url = self.adapter._get_api_url()
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_get_api_url_already_has_path(self):
        """已经包含路径时不重复添加"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        adapter = ClosedSourceAdapter(
            model_name="test",
            api_key="sk-test",
            api_base="https://example.com/v1/chat/completions",
        )
        url = adapter._get_api_url()
        assert url == "https://example.com/v1/chat/completions"

    def test_extra_params_passthrough(self):
        """额外参数透传"""
        body = self.adapter._build_chat_request(
            prompt="Hello",
            presence_penalty=0.5,
            frequency_penalty=0.3,
            seed=42,
        )
        assert body["presence_penalty"] == 0.5
        assert body["frequency_penalty"] == 0.3
        assert body["seed"] == 42


# ============================================================
# 7. Generate 方法测试 (通过 Mock HTTP)
# ============================================================

class TestGenerate:
    """测试生成方法 (mock HTTP 调用)"""

    def setup_method(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.adapter = ClosedSourceAdapter(
            model_name="test-model",
            api_key="sk-test",
        )

    def test_sync_generate(self):
        """同步生成"""
        mock_response = {
            "choices": [{"message": {"content": "Hello! 你好！"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "test-model",
        }
        self.adapter._sync_api_call = MagicMock(return_value=mock_response)

        output = self.adapter.generate("你好")
        assert output.text == "Hello! 你好！"
        assert output.input_tokens == 10
        assert output.output_tokens == 5
        assert output.latency_ms > 0

    @pytest.mark.asyncio
    async def test_async_generate(self):
        """异步生成"""
        mock_response = {
            "choices": [{"message": {"content": "Async hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            "model": "test-model",
        }
        self.adapter._async_api_call = AsyncMock(return_value=mock_response)

        output = await self.adapter.async_generate("你好")
        assert output.text == "Async hello!"
        assert output.input_tokens == 8
        assert output.output_tokens == 3

    def test_generate_empty_response(self):
        """空响应处理"""
        mock_response = {"choices": [], "usage": {}}
        self.adapter._sync_api_call = MagicMock(return_value=mock_response)

        output = self.adapter.generate("你好")
        assert output.text == ""
        assert output.input_tokens == 0
        assert output.output_tokens == 0


# ============================================================
# 8. ModelFactory 注册测试
# ============================================================

class TestModelFactoryClosedSource:
    """测试 ModelFactory 闭源适配器注册"""

    def test_closed_source_registered(self):
        """closed_source 已注册"""
        from dki.models.factory import ModelFactory
        assert "closed_source" in ModelFactory._adapters

    def test_closed_source_adapter_class(self):
        """注册的是 ClosedSourceAdapter"""
        from dki.models.factory import ModelFactory
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        assert ModelFactory._adapters["closed_source"] is ClosedSourceAdapter

    def test_is_closed_source_engine(self):
        """is_closed_source_engine 辅助方法"""
        from dki.models.factory import ModelFactory
        from dki.config.config_loader import ConfigLoader

        # 备份并重置
        ConfigLoader.reset()

        with patch.object(ConfigLoader, '_find_config_path', return_value="dummy"):
            with patch.object(ConfigLoader, '_load_config'):
                loader = ConfigLoader.__new__(ConfigLoader)
                ConfigLoader._instance = loader
                loader._config_path = "dummy"
                loader._check_interval = 30
                loader._last_mtime = 0
                loader._last_check_time = 0

                from dki.config.config_loader import Config, ModelConfig
                loader._config = Config(
                    model=ModelConfig(default_engine="closed_source")
                )
                ConfigLoader._config = loader._config

                assert ModelFactory.is_closed_source_engine() is True
                assert ModelFactory.is_closed_source_engine("vllm") is False
                assert ModelFactory.is_closed_source_engine("closed_source") is True

        ConfigLoader.reset()

    def test_get_adapter_is_closed_source(self):
        """get_adapter_is_closed_source 辅助方法"""
        from dki.models.factory import ModelFactory
        from dki.models.closed_source_adapter import ClosedSourceAdapter

        adapter = ClosedSourceAdapter(model_name="test", api_key="sk-test")
        assert ModelFactory.get_adapter_is_closed_source(adapter) is True

        mock_adapter = MagicMock()
        mock_adapter.is_closed_source = False
        assert ModelFactory.get_adapter_is_closed_source(mock_adapter) is False

        assert ModelFactory.get_adapter_is_closed_source(None) is False


# ============================================================
# 9. EnhancedDKIPlugin 闭源模型路由测试
# ============================================================

class MockClosedSourceDKIPlugin:
    """模拟使用闭源模型的 DKI Plugin"""

    def __init__(self):
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        self.model = ClosedSourceAdapter(
            model_name="deepseek-chat",
            api_key="sk-test",
        )
        self.data_adapter = MagicMock()
        self.planner = MagicMock()
        self.executor = MagicMock()
        self._preference_text_cache = {}
        self._closed = False

    async def chat(self, query, user_id, session_id, **kwargs):
        from dki.core.dki_plugin import DKIPluginResponse, InjectionMetadata
        metadata = InjectionMetadata(
            injection_enabled=False,
            injection_strategy="rag",
            latency_ms=100.0,
        )
        return DKIPluginResponse(
            text=f"DKI response: {query}",
            input_tokens=10,
            output_tokens=20,
            metadata=metadata,
        )

    async def chat_stream(self, query, user_id, session_id, **kwargs):
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata(
            injection_enabled=False,
            injection_strategy="rag",
            latency_ms=100.0,
        )
        yield {"type": "metadata", "metadata": metadata.to_dict()}
        yield {"type": "token", "content": f"DKI: {query}"}
        yield {"type": "done", "text": f"DKI: {query}"}

    def get_stats(self):
        return {"total": 0}

    def invalidate_preference_text_cache(self, user_id):
        pass

    async def close(self):
        self._closed = True


class MockRAGSystem:
    """模拟 RAG 系统"""

    def chat(self, query, session_id, user_id, **kwargs):
        from dki.core.dki_plugin import DKIPluginResponse, InjectionMetadata

        # 返回 RAGResponse-like 对象
        return MagicMock(
            text=f"RAG response: {query}",
            input_tokens=15,
            output_tokens=25,
            metadata={"model": "deepseek-chat"},
        )


class TestEnhancedPluginClosedSourceRoute:
    """测试 EnhancedDKIPlugin 对闭源模型的自动路由"""

    def setup_method(self):
        self.mock_dki = MockClosedSourceDKIPlugin()
        self.mock_rag = MockRAGSystem()

    def test_is_closed_source_model_detection(self):
        """检测闭源模型"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
        )

        enhanced = EnhancedDKIPlugin(
            dki_plugin=self.mock_dki,
            rag_system=self.mock_rag,
        )
        assert enhanced._is_closed_source_model() is True

    def test_closed_source_forces_rag_route(self):
        """闭源模型强制 RAG 路由"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
            DynamicRouterConfig,
        )

        enhanced = EnhancedDKIPlugin(
            dki_plugin=self.mock_dki,
            rag_system=self.mock_rag,
            config=EnhancedPluginConfig(
                dynamic_router=DynamicRouterConfig(enabled=True),
            ),
        )

        mode = enhanced._resolve_route_mode(
            query="推荐餐厅",
            user_id="u1",
            session_id="s1",
        )
        assert mode == "rag"

    def test_closed_source_forces_rag_even_without_router(self):
        """即使未启用动态路由, 闭源模型也强制 RAG"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
            DynamicRouterConfig,
        )

        enhanced = EnhancedDKIPlugin(
            dki_plugin=self.mock_dki,
            rag_system=self.mock_rag,
            config=EnhancedPluginConfig(
                dynamic_router=DynamicRouterConfig(enabled=False),
            ),
        )

        mode = enhanced._resolve_route_mode(
            query="推荐餐厅",
            user_id="u1",
            session_id="s1",
        )
        assert mode == "rag"

    def test_force_mode_overrides_closed_source(self):
        """force_mode 优先级高于闭源检测"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
        )

        enhanced = EnhancedDKIPlugin(
            dki_plugin=self.mock_dki,
            rag_system=self.mock_rag,
        )

        mode = enhanced._resolve_route_mode(
            query="test",
            user_id="u1",
            session_id="s1",
            force_mode="dki",
        )
        assert mode == "dki"

    @pytest.mark.asyncio
    async def test_chat_uses_rag_for_closed_source(self):
        """chat() 对闭源模型走 RAG 路由"""
        from dki.integration.enhanced_plugin import (
            EnhancedDKIPlugin,
            EnhancedPluginConfig,
        )

        enhanced = EnhancedDKIPlugin(
            dki_plugin=self.mock_dki,
            rag_system=self.mock_rag,
        )

        response = await enhanced.chat(
            query="推荐餐厅",
            user_id="u1",
            session_id="s1",
        )

        assert response.text == "RAG response: 推荐餐厅"
        assert enhanced._enhanced_stats["rag_routes"] == 1
        assert enhanced._enhanced_stats["dki_routes"] == 0


# ============================================================
# 10. EngineConfig 闭源字段测试
# ============================================================

class TestEngineConfigClosedSource:
    """测试 EngineConfig 闭源模型配置字段"""

    def test_closed_source_fields_defaults(self):
        """闭源配置字段默认值"""
        from dki.config.config_loader import EngineConfig

        config = EngineConfig(model_name="gpt-4o")
        assert config.api_key is None
        assert config.api_base is None
        assert config.api_version is None
        assert config.timeout == 120.0
        assert config.max_retries == 2
        assert config.default_system_prompt is None

    def test_closed_source_fields_custom(self):
        """闭源配置字段自定义值"""
        from dki.config.config_loader import EngineConfig

        config = EngineConfig(
            model_name="deepseek-chat",
            api_key="sk-xxx",
            api_base="https://api.deepseek.com/v1",
            timeout=60.0,
            max_retries=5,
            default_system_prompt="你是一个AI助手",
        )
        assert config.api_key == "sk-xxx"
        assert config.api_base == "https://api.deepseek.com/v1"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.default_system_prompt == "你是一个AI助手"

    def test_closed_source_yaml_dict(self):
        """从 YAML 字典构建闭源配置"""
        from dki.config.config_loader import EngineConfig

        yaml_dict = {
            "enabled": True,
            "model_name": "glm-4-flash",
            "api_key": "sk-glm-key",
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "max_model_len": 16384,
            "timeout": 30.0,
        }
        config = EngineConfig(**yaml_dict)
        assert config.model_name == "glm-4-flash"
        assert config.api_key == "sk-glm-key"
        assert config.api_base == "https://open.bigmodel.cn/api/paas/v4"


# ============================================================
# 11. ModelOutput 兼容性测试
# ============================================================

class TestModelOutputCompat:
    """测试闭源适配器的输出与 ModelOutput 兼容"""

    def test_generate_returns_model_output(self):
        """generate 应返回 ModelOutput 兼容对象"""
        from dki.models.closed_source_adapter import ClosedSourceAdapter
        from dki.models.base import ModelOutput

        adapter = ClosedSourceAdapter(
            model_name="test",
            api_key="sk-test",
        )

        mock_response = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "model": "test",
        }
        adapter._sync_api_call = MagicMock(return_value=mock_response)

        output = adapter.generate("hello")

        # 应该是 ModelOutput 实例
        assert isinstance(output, ModelOutput)
        assert output.text == "test"
        assert output.input_tokens == 5
        assert output.output_tokens == 2
        assert output.tokens is None  # 闭源模型不返回 token IDs
        assert output.logits is None
        assert output.hidden_states is None
        assert output.kv_cache is None


# ============================================================
# 12. __init__.py 导出测试
# ============================================================

class TestModuleExports:
    """测试模块导出"""

    def test_closed_source_adapter_in_models(self):
        """ClosedSourceAdapter 在 dki.models 中导出"""
        from dki.models import ClosedSourceAdapter
        assert ClosedSourceAdapter is not None

    def test_closed_source_adapter_in_all(self):
        """ClosedSourceAdapter 在 __all__ 中"""
        from dki import models
        assert "ClosedSourceAdapter" in models.__all__
