"""
单元测试: LLaMA 引擎配置修复验证

验证 config.yaml 和 config_env.yaml 中 llama 引擎配置的正确性,
以及 ModelFactory 能正确识别和创建 llama 引擎适配器。

修复内容:
1. config.yaml 新增 llama 引擎配置块
2. config_env.yaml 新增 llama 引擎配置块 (环境变量驱动)
3. start_dki_with_model.sh llama_8b 补充 CUDA_VISIBLE_DEVICES
"""

import os
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.config.config_loader import ConfigLoader, EngineConfig, Config
from dki.models.factory import ModelFactory
from dki.models.llama_adapter import LlamaAdapter


# ============================================================================
# 配置文件结构验证
# ============================================================================

class TestConfigYamlHasLlamaEngine:
    """验证 config.yaml 中存在 llama 引擎配置"""

    @pytest.fixture
    def config_yaml_path(self):
        """获取 config.yaml 路径"""
        return Path(__file__).parent.parent.parent / "config" / "config.yaml"

    @pytest.fixture
    def config_env_yaml_path(self):
        """获取 config_env.yaml 路径"""
        return Path(__file__).parent.parent.parent / "config" / "config_env.yaml"

    def test_config_yaml_has_llama_engine(self, config_yaml_path):
        """config.yaml 应包含 llama 引擎配置"""
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        engines = config.get('model', {}).get('engines', {})
        assert 'llama' in engines, (
            "config.yaml model.engines 应包含 'llama' 配置块"
        )

    def test_config_yaml_llama_has_required_fields(self, config_yaml_path):
        """config.yaml llama 引擎应包含必要字段"""
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        llama_cfg = config['model']['engines']['llama']
        assert 'enabled' in llama_cfg
        assert 'model_name' in llama_cfg
        assert llama_cfg['enabled'] is True

    def test_config_yaml_llama_model_path_is_llama31(self, config_yaml_path):
        """config.yaml llama 引擎 model_name 应指向 llama-3.1"""
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_name = config['model']['engines']['llama']['model_name']
        assert 'llama' in model_name.lower(), (
            f"llama engine model_name 应包含 'llama', 实际: {model_name}"
        )

    def test_config_env_yaml_has_llama_engine(self, config_env_yaml_path):
        """config_env.yaml 应包含 llama 引擎配置"""
        with open(config_env_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        engines = config.get('model', {}).get('engines', {})
        assert 'llama' in engines, (
            "config_env.yaml model.engines 应包含 'llama' 配置块"
        )

    def test_config_env_yaml_llama_uses_env_vars(self, config_env_yaml_path):
        """config_env.yaml llama 引擎 model_name 应使用环境变量"""
        with open(config_env_yaml_path, 'r') as f:
            raw_content = f.read()

        # 确保 llama 配置块中使用了 ${DKI_MODEL_PATH} 环境变量
        # 通过原始文本检查 (yaml.safe_load 会解析掉 ${} 语法)
        assert '${DKI_MODEL_PATH' in raw_content, (
            "config_env.yaml llama.model_name 应使用 ${DKI_MODEL_PATH} 环境变量"
        )


# ============================================================================
# EngineConfig 解析验证
# ============================================================================

class TestEngineConfigParsing:
    """验证 EngineConfig 能正确解析 llama 引擎配置"""

    def test_engine_config_accepts_llama_fields(self):
        """EngineConfig 应能解析 llama 引擎的所有字段"""
        cfg = EngineConfig(
            enabled=True,
            model_name="/opt/ai-demo/models/llama-3.1-8b-instruct",
            device="cuda",
            dtype="float16",
            trust_remote_code=True,
            load_in_8bit=False,
        )
        assert cfg.enabled is True
        assert cfg.model_name == "/opt/ai-demo/models/llama-3.1-8b-instruct"
        assert cfg.load_in_8bit is False

    def test_engine_config_load_in_8bit_default(self):
        """EngineConfig load_in_8bit 默认应为 False"""
        cfg = EngineConfig(model_name="test-model")
        assert cfg.load_in_8bit is False

    def test_engine_config_load_in_8bit_true(self):
        """EngineConfig 应支持 load_in_8bit=True"""
        cfg = EngineConfig(
            model_name="test-model",
            load_in_8bit=True,
        )
        assert cfg.load_in_8bit is True


# ============================================================================
# ModelFactory llama 引擎验证
# ============================================================================

class TestModelFactoryLlamaEngine:
    """验证 ModelFactory 能正确处理 llama 引擎"""

    def test_llama_registered_in_factory(self):
        """llama 应在 ModelFactory 注册表中"""
        assert 'llama' in ModelFactory._adapters

    def test_llama_adapter_class_correct(self):
        """llama 注册的适配器类应为 LlamaAdapter"""
        assert ModelFactory._adapters['llama'] is LlamaAdapter

    def test_factory_create_llama_with_config(self):
        """ModelFactory.create 应能创建 llama 适配器 (带配置)"""
        # Mock ConfigLoader 返回包含 llama 引擎的配置
        mock_engine_config = EngineConfig(
            enabled=True,
            model_name="/opt/ai-demo/models/llama-3.1-8b-instruct",
            device="cuda",
            dtype="float16",
            trust_remote_code=True,
            load_in_8bit=False,
        )

        mock_config = MagicMock()
        mock_config.model.default_engine = "llama"
        mock_config.model.engines = {'llama': mock_engine_config}

        with patch.object(ConfigLoader, '__init__', lambda self: None):
            with patch.object(ConfigLoader, 'config', new_callable=lambda: property(lambda self: mock_config)):
                adapter = ModelFactory.create(engine="llama")

        assert isinstance(adapter, LlamaAdapter)
        assert adapter.model_name == "/opt/ai-demo/models/llama-3.1-8b-instruct"
        assert adapter.load_in_8bit is False

    def test_factory_create_llama_with_8bit(self):
        """ModelFactory.create 应能传递 load_in_8bit 参数"""
        mock_engine_config = EngineConfig(
            enabled=True,
            model_name="/opt/ai-demo/models/llama-3.1-8b-instruct",
            device="cuda",
            dtype="float16",
            trust_remote_code=True,
            load_in_8bit=True,
        )

        mock_config = MagicMock()
        mock_config.model.default_engine = "llama"
        mock_config.model.engines = {'llama': mock_engine_config}

        with patch.object(ConfigLoader, '__init__', lambda self: None):
            with patch.object(ConfigLoader, 'config', new_callable=lambda: property(lambda self: mock_config)):
                adapter = ModelFactory.create(engine="llama")

        assert isinstance(adapter, LlamaAdapter)
        assert adapter.load_in_8bit is True


# ============================================================================
# 启动脚本环境变量对齐验证
# ============================================================================

class TestStartScriptLlamaAlignment:
    """验证启动脚本 llama_8b 配置与 config_env.yaml 对齐"""

    def test_llama_engine_resolves_from_env(self):
        """模拟 start_dki_with_model.sh llama_8b 的环境变量"""
        env_vars = {
            "DKI_MODEL_PATH": "/opt/ai-demo/models/llama-3.1-8b-instruct",
            "DKI_MODEL_ENGINE": "llama",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        with patch.dict(os.environ, env_vars):
            engine = ConfigLoader._resolve_env_vars("${DKI_MODEL_ENGINE:-vllm}")
            model_path = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_PATH:-/opt/ai-demo/models/llama-3.1-8b-instruct}"
            )
            assert engine == "llama"
            assert model_path == "/opt/ai-demo/models/llama-3.1-8b-instruct"

    def test_llama_default_fallback_when_no_env(self):
        """无环境变量时, llama 配置应使用默认值"""
        for var in ["DKI_MODEL_PATH", "DKI_MODEL_ENGINE"]:
            os.environ.pop(var, None)

        model_path = ConfigLoader._resolve_env_vars(
            "${DKI_MODEL_PATH:-/opt/ai-demo/models/llama-3.1-8b-instruct}"
        )
        assert model_path == "/opt/ai-demo/models/llama-3.1-8b-instruct"

    def test_start_script_has_cuda_for_llama(self):
        """启动脚本 llama_8b 应设置 CUDA_VISIBLE_DEVICES"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "start_dki_with_model.sh"
        with open(script_path, 'r') as f:
            content = f.read()

        # 查找 llama_8b case 块
        llama_block_start = content.find('"llama_8b")')
        assert llama_block_start != -1, "启动脚本应包含 llama_8b 配置"

        # 获取 llama_8b 到下一个 ;; 之间的内容
        llama_block_end = content.find(';;', llama_block_start)
        llama_block = content[llama_block_start:llama_block_end]

        assert 'CUDA_VISIBLE_DEVICES' in llama_block, (
            "llama_8b 配置块应包含 CUDA_VISIBLE_DEVICES 设置"
        )
        assert 'DKI_MODEL_ENGINE="llama"' in llama_block, (
            "llama_8b 配置块应设置 DKI_MODEL_ENGINE=llama"
        )


# ============================================================================
# 端到端: ConfigLoader 加载 llama 引擎
# ============================================================================

class TestConfigLoaderLlamaEndToEnd:
    """端到端验证: ConfigLoader 能从 config.yaml 正确加载 llama 引擎"""

    def test_load_config_yaml_with_llama(self):
        """ConfigLoader 加载 config.yaml 后应包含 llama 引擎"""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        # 重置单例
        ConfigLoader.reset()

        with patch.dict(os.environ, {"DKI_CONFIG_PATH": str(config_path)}):
            loader = ConfigLoader()
            config = loader.config

        engines = config.model.engines
        assert 'llama' in engines, (
            f"ConfigLoader 加载后应包含 llama 引擎, 实际引擎: {list(engines.keys())}"
        )

        llama_cfg = engines['llama']
        assert llama_cfg.enabled is True
        assert 'llama' in llama_cfg.model_name.lower()

        # 清理单例
        ConfigLoader.reset()

    def test_load_config_env_yaml_with_llama(self):
        """ConfigLoader 加载 config_env.yaml 后应包含 llama 引擎"""
        config_path = Path(__file__).parent.parent.parent / "config" / "config_env.yaml"

        # 重置单例
        ConfigLoader.reset()

        with patch.dict(os.environ, {"DKI_CONFIG_PATH": str(config_path)}):
            loader = ConfigLoader()
            config = loader.config

        engines = config.model.engines
        assert 'llama' in engines, (
            f"ConfigLoader 加载 config_env.yaml 后应包含 llama 引擎, "
            f"实际引擎: {list(engines.keys())}"
        )

        # 清理单例
        ConfigLoader.reset()
