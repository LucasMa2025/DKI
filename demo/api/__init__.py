"""
Demo API — 独立 API 路由

路由设计与现有前端 (DKI/ui) 完全兼容:
- /api/auth/*        → 认证
- /api/chat/send     → 对话 (调用 dki_plugin.chat())
- /api/sessions/*    → 会话管理
- /api/preferences/* → 偏好管理
- /v1/dki/*          → DKI 可视化 (透传)
"""
