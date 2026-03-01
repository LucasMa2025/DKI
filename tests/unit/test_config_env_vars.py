"""
Unit tests for ConfigLoader environment variable substitution.

Tests the ${VAR} and ${VAR:-default} syntax support added to ConfigLoader.
"""

import os
import pytest
from unittest.mock import patch

# Import the class under test
from dki.config.config_loader import ConfigLoader


class TestResolveEnvVars:
    """Test ConfigLoader._resolve_env_vars static method."""

    def test_simple_string_no_vars(self):
        """Plain string without env vars should be returned as-is."""
        assert ConfigLoader._resolve_env_vars("hello world") == "hello world"

    def test_simple_env_var_exists(self):
        """${VAR} should be replaced when the env var exists."""
        with patch.dict(os.environ, {"MY_TEST_VAR": "replaced_value"}):
            result = ConfigLoader._resolve_env_vars("${MY_TEST_VAR}")
            assert result == "replaced_value"

    def test_simple_env_var_not_exists(self):
        """${VAR} should be kept as-is when the env var does not exist."""
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        result = ConfigLoader._resolve_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_env_var_with_default_exists(self):
        """${VAR:-default} should use env var value when it exists."""
        with patch.dict(os.environ, {"MY_VAR": "actual_value"}):
            result = ConfigLoader._resolve_env_vars("${MY_VAR:-fallback}")
            assert result == "actual_value"

    def test_env_var_with_default_not_exists(self):
        """${VAR:-default} should use default when env var does not exist."""
        os.environ.pop("MISSING_VAR_99999", None)
        result = ConfigLoader._resolve_env_vars("${MISSING_VAR_99999:-my_default}")
        assert result == "my_default"

    def test_env_var_with_empty_default(self):
        """${VAR:-} should resolve to empty string when env var is missing."""
        os.environ.pop("MISSING_VAR_EMPTY", None)
        result = ConfigLoader._resolve_env_vars("${MISSING_VAR_EMPTY:-}")
        assert result == ""

    def test_mixed_text_and_var(self):
        """Env var in the middle of a string should be replaced inline."""
        with patch.dict(os.environ, {"MODEL_PATH": "/opt/models/llama"}):
            result = ConfigLoader._resolve_env_vars("path=${MODEL_PATH}/config")
            assert result == "path=/opt/models/llama/config"

    def test_multiple_vars_in_string(self):
        """Multiple env vars in one string should all be replaced."""
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}):
            result = ConfigLoader._resolve_env_vars("${HOST}:${PORT}")
            assert result == "0.0.0.0:8080"

    def test_dict_recursive(self):
        """Env vars in nested dicts should be resolved recursively."""
        with patch.dict(os.environ, {"DB_PATH": "/data/test.db"}):
            data = {
                "database": {
                    "path": "${DB_PATH}",
                    "type": "sqlite",
                }
            }
            result = ConfigLoader._resolve_env_vars(data)
            assert result["database"]["path"] == "/data/test.db"
            assert result["database"]["type"] == "sqlite"

    def test_list_recursive(self):
        """Env vars in lists should be resolved recursively."""
        with patch.dict(os.environ, {"ITEM_A": "alpha"}):
            data = ["${ITEM_A}", "beta", "${MISSING_ITEM:-gamma}"]
            result = ConfigLoader._resolve_env_vars(data)
            assert result == ["alpha", "beta", "gamma"]

    def test_non_string_passthrough(self):
        """Non-string types (int, float, bool, None) should pass through unchanged."""
        assert ConfigLoader._resolve_env_vars(42) == 42
        assert ConfigLoader._resolve_env_vars(3.14) == 3.14
        assert ConfigLoader._resolve_env_vars(True) is True
        assert ConfigLoader._resolve_env_vars(None) is None

    def test_deeply_nested_structure(self):
        """Deeply nested mixed structure should be fully resolved."""
        with patch.dict(os.environ, {"ENGINE": "vllm", "MODEL": "/opt/model"}):
            data = {
                "model": {
                    "default_engine": "${ENGINE}",
                    "engines": {
                        "vllm": {
                            "model_name": "${MODEL}",
                            "enabled": True,
                            "params": [1, "${ENGINE}", "${MISSING:-default}"],
                        }
                    }
                }
            }
            result = ConfigLoader._resolve_env_vars(data)
            assert result["model"]["default_engine"] == "vllm"
            assert result["model"]["engines"]["vllm"]["model_name"] == "/opt/model"
            assert result["model"]["engines"]["vllm"]["enabled"] is True
            assert result["model"]["engines"]["vllm"]["params"] == [1, "vllm", "default"]

    def test_model_config_alignment_with_script(self):
        """
        Simulate the environment set by start_dki_with_model.sh
        and verify config_env.yaml would resolve correctly.
        """
        # Simulate: bash start_dki_with_model.sh deepseek_14b
        env_vars = {
            "DKI_MODEL_PATH": "/opt/ai-demo/models/deepseek-r1-distill-qwen-14b",
            "DKI_MODEL_ENGINE": "vllm",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        with patch.dict(os.environ, env_vars):
            # These are the key config values from config_env.yaml
            model_name = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_PATH:-/opt/ai-demo/models/deepseek-r1-distill-qwen-7b}"
            )
            engine = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_ENGINE:-vllm}"
            )
            assert model_name == "/opt/ai-demo/models/deepseek-r1-distill-qwen-14b"
            assert engine == "vllm"

    def test_model_config_default_fallback(self):
        """
        When no env vars are set, config_env.yaml should fall back to defaults
        (same as original config.yaml).
        """
        # Clear relevant env vars
        for var in ["DKI_MODEL_PATH", "DKI_MODEL_ENGINE"]:
            os.environ.pop(var, None)

        model_name = ConfigLoader._resolve_env_vars(
            "${DKI_MODEL_PATH:-/opt/ai-demo/models/deepseek-r1-distill-qwen-7b}"
        )
        engine = ConfigLoader._resolve_env_vars(
            "${DKI_MODEL_ENGINE:-vllm}"
        )
        assert model_name == "/opt/ai-demo/models/deepseek-r1-distill-qwen-7b"
        assert engine == "vllm"

    def test_qianwen_model_alignment(self):
        """Simulate qianwen_14b model from start script."""
        env_vars = {
            "DKI_MODEL_PATH": "/opt/ai-demo/models/qianwen-14b-chat",
            "DKI_MODEL_ENGINE": "vllm",
        }
        with patch.dict(os.environ, env_vars):
            model_name = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_PATH:-/opt/ai-demo/models/deepseek-r1-distill-qwen-7b}"
            )
            assert model_name == "/opt/ai-demo/models/qianwen-14b-chat"

    def test_llama_model_alignment(self):
        """Simulate llama_8b model from start script."""
        env_vars = {
            "DKI_MODEL_PATH": "/opt/ai-demo/models/llama-3.1-8b-instruct",
            "DKI_MODEL_ENGINE": "llama",
        }
        with patch.dict(os.environ, env_vars):
            model_name = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_PATH:-/opt/ai-demo/models/deepseek-r1-distill-qwen-7b}"
            )
            engine = ConfigLoader._resolve_env_vars(
                "${DKI_MODEL_ENGINE:-vllm}"
            )
            assert model_name == "/opt/ai-demo/models/llama-3.1-8b-instruct"
            assert engine == "llama"
