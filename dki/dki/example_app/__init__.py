"""
DKI Example Application

DKI 示例应用

职责: 演示如何使用 DKI 插件
- 提供简单的 Chat UI
- 自带简单的数据存储 (供 DKI 适配器读取)
- 展示 DKI 注入效果

注意:
- 这是一个示例应用，不是 DKI 的核心功能
- 生产环境中，上层应用应该有自己的用户系统和数据库
- DKI 通过适配器读取上层应用的数据

Author: AGI Demo Project
Version: 2.0.0
"""

from .app import create_example_app
from .service import ExampleAppService

__all__ = [
    "create_example_app",
    "ExampleAppService",
]
