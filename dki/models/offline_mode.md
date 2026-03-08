# ==================== 最顶部的猴子补丁代码 ====================

# 第一步：先导入核心模块（必须在所有其他导入之前）

import sys
import os

# 第二步：强制创建/修改 huggingface_hub 模块的 is_offline_mode 函数

def \_patch_huggingface_hub_offline_mode():
"""
猴子补丁：提前注入 is_offline_mode 函数到 huggingface_hub，解决版本兼容问题
执行时机：必须在导入 transformers/vllm 之前
""" # 1. 确保 huggingface_hub 模块已加载（未加载则手动导入）
try:
import huggingface_hub
except ImportError: # 如果 huggingface_hub 没安装，直接抛出异常（这是真实错误，无法补丁）
raise ImportError("huggingface_hub 库未安装，请先执行 pip install huggingface_hub")

    # 2. 定义兼容的 is_offline_mode 函数（永远返回 False/0）
    def is_offline_mode() -> bool:
        """兼容新版本 huggingface_hub，强制返回非离线模式（0/False）"""
        return False

    # 3. 注入函数到 huggingface_hub 模块（覆盖/新增）
    setattr(huggingface_hub, "is_offline_mode", is_offline_mode)
    # 额外注入到 huggingface_hub.__init__ （防止从 __init__ 导入时找不到）
    if hasattr(huggingface_hub, "__init__"):
        setattr(huggingface_hub.__init__, "is_offline_mode", is_offline_mode)

    print("[Patch] 已成功注入 is_offline_mode 到 huggingface_hub 模块")

# 第三步：立即执行补丁（这行代码必须是 main.py 中第一个执行的逻辑）

\_patch_huggingface_hub_offline_mode()

# ==================== 补丁代码结束 ====================

# 以下才是你原来的 main.py 代码（正常的导入和业务逻辑）

import argparse
import asyncio
from demo.config import load_demo_config
from demo.app import create_app

# ... 其他原有导入和代码 ...

def main(): # 你的原有业务逻辑
pass

if **name** == "**main**":
main()
