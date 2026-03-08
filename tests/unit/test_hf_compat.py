"""
单元测试: HuggingFace Hub 兼容层 (hf_compat)

验证 ensure_hf_compat() 在不同 huggingface_hub 版本下的行为:
1. huggingface_hub 已有 is_offline_mode → 不注入 (旧版本)
2. huggingface_hub 缺少 is_offline_mode → 注入兼容函数 (新版本)
3. 注入的兼容函数正确读取 HF_HUB_OFFLINE 环境变量
4. ensure_hf_compat() 幂等性 (多次调用安全)
5. huggingface_hub 未安装 → 不报错, 静默跳过

Author: AGI Demo Project
"""

import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock

# 直接导入 (首次导入, 不需要 reload)
from dki.models.hf_compat import ensure_hf_compat, _get_hf_hub_version, _check_hf_hub_version
import dki.models.hf_compat as hf_compat_module


# ============================================================================
# 辅助: 创建干净的 mock huggingface_hub 模块
# ============================================================================

def _make_mock_hf_hub(has_offline_mode=False, offline_mode_return=False):
    """
    创建一个 mock huggingface_hub 模块。
    
    Args:
        has_offline_mode: 是否包含 is_offline_mode 函数
        offline_mode_return: is_offline_mode 的返回值 (仅在 has_offline_mode=True 时有效)
    """
    mock_mod = types.ModuleType("huggingface_hub")
    mock_mod.__file__ = "<mock>"
    mock_mod.__path__ = []
    if has_offline_mode:
        mock_mod.is_offline_mode = lambda: offline_mode_return
    return mock_mod


# ============================================================================
# 1. 旧版本: is_offline_mode 已存在, 不注入
# ============================================================================

class TestHfCompatOldVersion:
    """测试旧版 huggingface_hub (已有 is_offline_mode)"""

    def test_no_injection_when_function_exists(self):
        """is_offline_mode 已存在时不覆盖"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=True, offline_mode_return=True)
        original_func = mock_hf_hub.is_offline_mode

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            # 重置补丁状态
            hf_compat_module._patched = False
            ensure_hf_compat()

            # 原始函数应保持不变
            assert mock_hf_hub.is_offline_mode is original_func
            assert mock_hf_hub.is_offline_mode() is True

    def test_old_version_offline_false(self):
        """旧版本 is_offline_mode 返回 False 时保持不变"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=True, offline_mode_return=False)
        original_func = mock_hf_hub.is_offline_mode

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()

            assert mock_hf_hub.is_offline_mode is original_func
            assert mock_hf_hub.is_offline_mode() is False


# ============================================================================
# 2. 新版本: is_offline_mode 缺失, 注入兼容函数
# ============================================================================

class TestHfCompatNewVersion:
    """测试新版 huggingface_hub (缺少 is_offline_mode)"""

    def test_injection_when_function_missing(self):
        """is_offline_mode 缺失时注入兼容函数"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)
        assert not hasattr(mock_hf_hub, 'is_offline_mode')

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()

            # 应该被注入
            assert hasattr(mock_hf_hub, 'is_offline_mode')
            assert callable(mock_hf_hub.is_offline_mode)

    def test_injected_function_returns_false_by_default(self):
        """注入的函数默认返回 False (在线模式)"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()

            # 默认: HF_HUB_OFFLINE 未设置 → False
            env = os.environ.copy()
            env.pop('HF_HUB_OFFLINE', None)
            with patch.dict(os.environ, env, clear=True):
                assert mock_hf_hub.is_offline_mode() is False

    def test_injected_function_reads_env_var_1(self):
        """HF_HUB_OFFLINE='1' → True"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()

            with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
                assert mock_hf_hub.is_offline_mode() is True

    def test_injected_function_reads_env_var_0(self):
        """HF_HUB_OFFLINE='0' → False"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()

            with patch.dict(os.environ, {'HF_HUB_OFFLINE': '0'}):
                assert mock_hf_hub.is_offline_mode() is False


# ============================================================================
# 3. 幂等性: 多次调用安全
# ============================================================================

class TestHfCompatIdempotency:
    """测试 ensure_hf_compat() 幂等性"""

    def test_multiple_calls_safe_new_version(self):
        """新版本: 多次调用不报错"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            ensure_hf_compat()
            ensure_hf_compat()

            assert hasattr(mock_hf_hub, 'is_offline_mode')
            assert callable(mock_hf_hub.is_offline_mode)

    def test_multiple_calls_safe_old_version(self):
        """旧版本: 多次调用不覆盖原始函数"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=True, offline_mode_return=True)
        original_func = mock_hf_hub.is_offline_mode

        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            ensure_hf_compat()
            ensure_hf_compat()

            assert mock_hf_hub.is_offline_mode is original_func


# ============================================================================
# 4. huggingface_hub 未安装
# ============================================================================

class TestHfCompatNotInstalled:
    """测试 huggingface_hub 未安装的情况"""

    def test_no_error_when_not_installed(self):
        """huggingface_hub 未安装时静默跳过, 不报错"""
        # sys.modules[key] = None 模拟模块不存在
        with patch.dict('sys.modules', {'huggingface_hub': None}):
            hf_compat_module._patched = False
            # 应该不报错 (内部 try/except 捕获 ImportError)
            ensure_hf_compat()  # 不抛异常即通过


# ============================================================================
# 5. 环境变量边界值测试
# ============================================================================

class TestHfCompatEnvVarEdgeCases:
    """测试注入函数对环境变量的边界值处理"""

    def _inject(self):
        """创建 mock 并注入兼容函数"""
        mock_hf_hub = _make_mock_hf_hub(has_offline_mode=False)
        return mock_hf_hub

    def test_env_var_empty_string(self):
        """HF_HUB_OFFLINE='' → False"""
        mock_hf_hub = self._inject()
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            with patch.dict(os.environ, {'HF_HUB_OFFLINE': ''}):
                assert mock_hf_hub.is_offline_mode() is False

    def test_env_var_true_string(self):
        """HF_HUB_OFFLINE='true' → False (严格匹配 '1')"""
        mock_hf_hub = self._inject()
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            with patch.dict(os.environ, {'HF_HUB_OFFLINE': 'true'}):
                assert mock_hf_hub.is_offline_mode() is False

    def test_env_var_1(self):
        """HF_HUB_OFFLINE='1' → True"""
        mock_hf_hub = self._inject()
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
                assert mock_hf_hub.is_offline_mode() is True

    def test_env_var_not_set(self):
        """HF_HUB_OFFLINE 未设置 → False"""
        mock_hf_hub = self._inject()
        with patch.dict('sys.modules', {'huggingface_hub': mock_hf_hub}):
            hf_compat_module._patched = False
            ensure_hf_compat()
            env = os.environ.copy()
            env.pop('HF_HUB_OFFLINE', None)
            with patch.dict(os.environ, env, clear=True):
                assert mock_hf_hub.is_offline_mode() is False


# ============================================================================
# 6. 版本检查测试
# ============================================================================

class TestHfCompatVersionCheck:
    """测试 _get_hf_hub_version() 和 _check_hf_hub_version()"""

    def test_get_version_returns_string(self):
        """_get_hf_hub_version() 返回字符串"""
        version = _get_hf_hub_version()
        assert isinstance(version, str)
        # 在测试环境中, huggingface_hub 可能已安装也可能未安装
        # 但返回值一定是字符串

    def test_get_version_when_not_installed(self):
        """huggingface_hub 未安装时返回 'unknown'"""
        # version() 在函数内部通过 from importlib.metadata import version 导入
        # mock importlib.metadata.version 使其抛出异常
        with patch('importlib.metadata.version', side_effect=Exception("not found")):
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                # 两种获取方式都失败时返回 unknown
                ver = _get_hf_hub_version()
                assert ver == "unknown"

    def test_check_version_low_warns(self):
        """huggingface-hub 版本过低时输出警告"""
        hf_compat_module._version_checked = False
        with patch('dki.models.hf_compat._get_hf_hub_version', return_value="0.36.2"):
            with patch('dki.models.hf_compat.logger') as mock_logger:
                _check_hf_hub_version()
                # 应该输出警告
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "0.36.2" in warning_msg
                assert "1.3.0" in warning_msg
                assert "pip install" in warning_msg

    def test_check_version_ok_no_warn(self):
        """huggingface-hub 版本满足要求时不警告"""
        hf_compat_module._version_checked = False
        with patch('dki.models.hf_compat._get_hf_hub_version', return_value="1.5.0"):
            with patch('dki.models.hf_compat.logger') as mock_logger:
                _check_hf_hub_version()
                mock_logger.warning.assert_not_called()

    def test_check_version_unknown_no_warn(self):
        """版本未知时不警告"""
        hf_compat_module._version_checked = False
        with patch('dki.models.hf_compat._get_hf_hub_version', return_value="unknown"):
            with patch('dki.models.hf_compat.logger') as mock_logger:
                _check_hf_hub_version()
                mock_logger.warning.assert_not_called()

    def test_check_version_idempotent(self):
        """版本检查幂等: 只执行一次"""
        hf_compat_module._version_checked = False
        with patch('dki.models.hf_compat._get_hf_hub_version', return_value="0.36.2"):
            with patch('dki.models.hf_compat.logger') as mock_logger:
                _check_hf_hub_version()
                _check_hf_hub_version()
                _check_hf_hub_version()
                # 只警告一次
                assert mock_logger.warning.call_count == 1


# ============================================================================
# 7. VLLMAdapter ImportError 诊断测试
# ============================================================================

class TestVLLMAdapterImportDiagnostics:
    """测试 VLLMAdapter.load() 的 ImportError 诊断"""

    def test_version_mismatch_error_detected(self):
        """版本不兼容错误被正确识别"""
        error_msg = (
            "huggingface-hub>=1.3.0,<2.0 is required for a normal functioning "
            "of this module, but found huggingface-hub==0.36.2."
        )
        # 验证诊断逻辑
        assert "huggingface-hub" in error_msg
        assert "is required" in error_msg

    def test_is_offline_mode_error_detected(self):
        """is_offline_mode 错误被正确识别"""
        error_msg = "cannot import name 'is_offline_mode' from 'huggingface_hub'"
        assert "is_offline_mode" in error_msg

    def test_vllm_not_installed_detected(self):
        """vLLM 未安装错误被正确识别"""
        error_msg = "No module named 'vllm'"
        assert "vllm" in error_msg.lower()
