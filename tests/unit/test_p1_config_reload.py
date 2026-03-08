"""
Unit Tests for Config Hot-Reload (P1-4)

测试 ConfigLoader 的 mtime 检测和热加载机制。

Author: AGI Demo Project
"""

import os
import time
import tempfile
import pytest
import yaml

from dki.config.config_loader import ConfigLoader, Config


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前重置 ConfigLoader 单例"""
    ConfigLoader.reset()
    yield
    ConfigLoader.reset()


@pytest.fixture
def tmp_config(tmp_path):
    """创建临时配置文件"""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "system": {"name": "Test System", "version": "1.0.0", "debug": True},
        "model": {
            "default_engine": "vllm",
            "max_new_tokens": 1024,
            "engines": {},
        },
    }
    config_file.write_text(yaml.dump(config_data, allow_unicode=True), encoding="utf-8")
    return str(config_file)


# ============================================================
# 基本加载
# ============================================================

class TestConfigLoaderBasic:
    """测试基本配置加载"""

    def test_load_from_file(self, tmp_config):
        """从文件加载配置"""
        loader = ConfigLoader(config_path=tmp_config)
        cfg = loader.config
        assert isinstance(cfg, Config)
        assert cfg.system.name == "Test System"
        assert cfg.system.version == "1.0.0"

    def test_singleton_pattern(self, tmp_config):
        """单例模式"""
        loader1 = ConfigLoader(config_path=tmp_config)
        loader2 = ConfigLoader()
        assert loader1 is loader2

    def test_get_dot_notation(self, tmp_config):
        """点分隔键获取配置"""
        loader = ConfigLoader(config_path=tmp_config)
        assert loader.get("system.name") == "Test System"
        assert loader.get("system.nonexistent", "default") == "default"

    def test_reload(self, tmp_config):
        """手动 reload"""
        loader = ConfigLoader(config_path=tmp_config)
        assert loader.config.system.name == "Test System"

        # 修改文件
        with open(tmp_config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["system"]["name"] = "Updated System"
        with open(tmp_config, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        loader.reload()
        assert loader.config.system.name == "Updated System"

    def test_missing_file_uses_defaults(self):
        """文件不存在时使用默认配置"""
        loader = ConfigLoader(config_path="/nonexistent/path/config.yaml")
        cfg = loader.config
        assert isinstance(cfg, Config)
        # 使用默认值
        assert cfg.system.name == "DKI System"


# ============================================================
# P1-4: 热加载
# ============================================================

class TestConfigHotReload:
    """测试基于 mtime 的热加载"""

    def test_auto_reload_on_file_change(self, tmp_path):
        """文件变更后自动重新加载"""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "system": {"name": "Original", "version": "1.0.0"},
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = ConfigLoader(
            config_path=str(config_file),
            check_interval=0,  # 每次都检查
        )
        assert loader.config.system.name == "Original"

        # 修改文件 (确保 mtime 变化)
        time.sleep(0.1)
        config_data["system"]["name"] = "Reloaded"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        # 访问 config 属性触发热加载
        assert loader.config.system.name == "Reloaded"

    def test_no_reload_within_interval(self, tmp_path):
        """check_interval 内不重复检查"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "Original", "version": "1.0.0"}}
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = ConfigLoader(
            config_path=str(config_file),
            check_interval=9999,  # 很长的间隔
        )
        assert loader.config.system.name == "Original"

        # 修改文件
        time.sleep(0.1)
        config_data["system"]["name"] = "Changed"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        # 由于 check_interval 很长, 不会重新加载
        assert loader.config.system.name == "Original"

    def test_file_deleted_keeps_current_config(self, tmp_path):
        """文件被删除时保持当前配置"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "Persistent", "version": "1.0.0"}}
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = ConfigLoader(
            config_path=str(config_file),
            check_interval=0,
        )
        assert loader.config.system.name == "Persistent"

        # 删除文件
        os.remove(str(config_file))

        # 仍然返回旧配置
        assert loader.config.system.name == "Persistent"


# ============================================================
# 环境变量替换
# ============================================================

class TestConfigEnvVars:
    """测试环境变量替换"""

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """${VAR} 语法替换"""
        monkeypatch.setenv("TEST_MODEL_NAME", "my-custom-model")

        config_file = tmp_path / "config.yaml"
        config_data = {
            "system": {"name": "${TEST_MODEL_NAME}", "version": "1.0.0"},
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = ConfigLoader(config_path=str(config_file))
        assert loader.config.system.name == "my-custom-model"

    def test_env_var_with_default(self, tmp_path):
        """${VAR:-default} 语法"""
        config_file = tmp_path / "config.yaml"
        # 使用不存在的环境变量
        raw_yaml = 'system:\n  name: "${NONEXISTENT_VAR:-fallback_name}"\n  version: "1.0.0"\n'
        config_file.write_text(raw_yaml, encoding="utf-8")

        loader = ConfigLoader(config_path=str(config_file))
        assert loader.config.system.name == "fallback_name"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
