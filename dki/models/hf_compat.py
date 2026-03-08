"""
HuggingFace Hub 兼容层

解决 huggingface_hub / transformers / vLLM 之间的版本兼容性问题。

已知问题:
  1. huggingface_hub ≥0.25 移除了 is_offline_mode()
     - vLLM (部分版本) 内部依赖 from huggingface_hub import is_offline_mode
     - 导致 ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
     
  2. transformers ≥4.48 要求 huggingface-hub ≥1.3.0
     - 如果 huggingface-hub 版本太低, transformers 导入时直接抛出 ImportError
     - 错误信息: "huggingface-hub>=1.3.0,<2.0 is required ... but found huggingface-hub==0.36.2"

解决方案:
  在导入 vllm/transformers 之前调用 ensure_hf_compat():
  - 检测 huggingface_hub 版本, 给出清晰的诊断信息
  - 如果 is_offline_mode 缺失, 注入兼容实现
  - 补丁是幂等的 (多次调用安全)

使用方式:
  from dki.models.hf_compat import ensure_hf_compat
  ensure_hf_compat()  # 必须在 from vllm import ... 之前调用

Author: AGI Demo Project
"""

import os
from loguru import logger

# 补丁状态标记 (避免重复日志)
_patched = False
_version_checked = False


def _get_hf_hub_version() -> str:
    """
    获取 huggingface_hub 的安装版本。
    
    Returns:
        版本号字符串 (如 "0.36.2"), 获取失败时返回 "unknown"
    """
    try:
        from importlib.metadata import version
        return version("huggingface-hub")
    except Exception:
        try:
            import huggingface_hub
            return getattr(huggingface_hub, '__version__', 'unknown')
        except Exception:
            return "unknown"


def _check_hf_hub_version() -> None:
    """
    检测 huggingface-hub 版本是否满足 transformers 的要求。
    
    如果版本过低, 输出警告和修复建议 (但不阻止执行,
    让 transformers 自己抛出明确的错误信息)。
    """
    global _version_checked
    if _version_checked:
        return
    _version_checked = True
    
    hf_version = _get_hf_hub_version()
    if hf_version == "unknown":
        return
    
    try:
        from packaging.version import Version
        current = Version(hf_version)
        
        # transformers ≥4.48 要求 huggingface-hub ≥1.3.0
        min_required = Version("1.3.0")
        if current < min_required:
            logger.warning(
                f"huggingface-hub 版本过低: {hf_version} "
                f"(transformers 要求 ≥1.3.0)\n"
                f"  修复方案:\n"
                f"    pip install huggingface-hub>=1.3.0 --upgrade\n"
                f"  或一次性升级所有依赖:\n"
                f"    pip install transformers huggingface-hub vllm --upgrade"
            )
    except ImportError:
        # packaging 未安装, 跳过版本检查
        pass
    except Exception:
        # 版本解析失败, 跳过
        pass


def ensure_hf_compat() -> None:
    """
    确保 huggingface_hub 兼容性。
    
    执行以下检查和修复:
    1. 检测 huggingface-hub 版本, 如果过低则输出警告和修复建议
    2. 检测 is_offline_mode 是否存在, 如果缺失则注入兼容实现
    
    此函数是幂等的, 多次调用安全。
    必须在 import vllm / import transformers 之前调用。
    """
    global _patched
    
    # ============ 版本检查 ============
    _check_hf_hub_version()
    
    # ============ is_offline_mode 兼容 ============
    try:
        import huggingface_hub
    except ImportError:
        # huggingface_hub 未安装 — 这是真实错误, 不做补丁
        # 后续导入 vllm/transformers 时会自然报错
        return
    
    # 检查 is_offline_mode 是否存在
    if hasattr(huggingface_hub, 'is_offline_mode'):
        # 函数已存在, 无需补丁
        return
    
    # ============ 注入兼容实现 ============
    # 新版 huggingface_hub 使用 HF_HUB_OFFLINE 环境变量控制离线模式
    # 注入一个读取该环境变量的兼容函数
    
    def _is_offline_mode() -> bool:
        """
        兼容实现: 检查 HF_HUB_OFFLINE 环境变量。
        
        huggingface_hub ≥0.25 移除了 is_offline_mode(),
        此函数作为兼容替代, 供 vLLM 等依赖方使用。
        
        环境变量:
          HF_HUB_OFFLINE=1  → 离线模式 (True)
          HF_HUB_OFFLINE=0  → 在线模式 (False, 默认)
        """
        return os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    # 注入到 huggingface_hub 模块
    huggingface_hub.is_offline_mode = _is_offline_mode
    
    if not _patched:
        logger.info(
            "huggingface_hub compat: injected is_offline_mode() "
            "(huggingface_hub removed this function, "
            "using HF_HUB_OFFLINE env var as fallback)"
        )
        _patched = True
