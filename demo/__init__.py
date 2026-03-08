"""
DKI Demo — 独立上层应用

Demo 系统是 DKI Plugin 的一个"客户"，展示如何作为独立上层应用
通过 dki_plugin.chat() 与 DKI 交互。

核心设计:
1. 独立持久化层 (demo/store/) — 不依赖 DKI/dki/database
2. 独立 API 层 (demo/api/) — 不依赖 DKI/dki/web/app.py
3. 通过 ConfigDrivenAdapter 桥接 — Demo DB 映射给 DKI Plugin 只读

Author: AGI Demo Project
Version: 2.0.0
"""

__version__ = "2.0.0"
