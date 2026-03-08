# DKI Web 测试界面

## 概述

FastAPI Web 应用，提供 REST API 和内嵌 Web UI，用于 DKI 系统的交互式测试。

## 架构

```
web/
├── app.py           # FastAPI 应用 (API + 内嵌 HTML/JS UI)
├── __init__.py      # 导出 create_app()
├── 日志.md          # 运行日志示例
└── README.md        # 本文档
```

## 功能清单

### REST API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 核心对话端点 (支持 DKI/RAG/baseline/auto 模式) |
| `/api/memory` | POST | 添加记忆 (写入 memories 表, RAG 检索用) |
| `/api/preference` | POST | 添加用户偏好 (写入 user_preferences 表, DKI K/V 注入用) |
| `/api/preferences/{user_id}` | GET | 查看用户偏好列表 |
| `/api/preferences/{user_id}` | DELETE | 清空用户偏好 |
| `/api/sessions` | GET | 查看所有会话 |
| `/api/sessions/{session_id}` | GET | 查看会话详情 (含对话历史和偏好) |
| `/api/router/stats` | GET | 路由器统计 (v6.0) |
| `/api/router/config` | POST | 动态更新路由器配置 |
| `/api/router/test` | POST | 测试路由决策 (不执行推理) |
| `/api/experiments/run` | POST | 运行实验 |
| `/api/experiments/comparison` | POST | 运行 DKI vs RAG 对比 |
| `/api/experiments/results` | GET | 查看实验结果 |
| `/api/visualization/*` | GET | 可视化数据 |
| `/api/stats/*` | GET | 系统统计 |

### Web UI 功能

- **对话测试**: 支持 DKI / RAG / Baseline / Auto 四种模式实时切换
- **偏好管理**: 添加/查看/清空用户偏好，偏好通过 K/V 注入影响 DKI 推理
- **Alpha 控制**: 滑块实时调节 DKI 注入强度 α (0.0 ~ 1.0)
- **会话管理**: 查看所有会话历史和详情
- **可视化面板**: 查看注入信息、延迟分析、召回统计

## 用户系统

### 用户创建与识别

1. **自动创建**: 首次聊天时自动在 `demo_users` 表创建用户
2. **身份识别优先级**:
   - `request.token` → 从 auth 系统查找 user_id
   - `request.user_id` → 直接使用
   - 默认 → `"demo_user"`
3. **前端存储**: `localStorage` 保存 `dki_user_id` 和 `dki_token`

### user_id 传递链路

```
前端 (JS)
  └─ currentUserId = localStorage.get('dki_user_id') || 'demo_user'
     └─ POST /api/chat { user_id: currentUserId, ... }
        └─ _resolve_user_id(request) → resolved_user_id
           └─ DemoUserRepository.get_or_create(username=resolved_user_id)  [v7.0]
              └─ DKISystem.chat(user_id=resolved_user_id, ...)
                 └─ _log_conversation() → SessionRepository.get_or_create(session_id, user_id)
                    └─ Session 表关联 user_id (跨会话检索依赖此关联)
```

### 跨会话记忆

- `Session.user_id` 将会话与用户关联
- `ConversationRepository.get_recent_by_user_cross_session()` 通过 `JOIN Session` 检索同一用户的历史会话
- `MultiSignalRecall.recall()` 的第 12 步自动调用跨会话检索

## v7.0 修复记录

1. **Auto-routing Bug 修复**: `/api/chat` 中 RAG 分支原使用 `request.mode == "rag"` 判断，auto 路由后 `request.mode` 仍为 `"auto"`，导致 auto→RAG 路径实际进入 baseline。修正为 `actual_mode == "rag"`。

2. **用户自动创建**: `/api/chat` 端点新增 `DemoUserRepository.get_or_create()` 调用，确保首次聊天用户在数据库中存在，保障后续偏好查询和跨会话检索的可靠性。

## 启动方式

```bash
cd DKI
python -m uvicorn dki.web.app:create_app --factory --host 0.0.0.0 --port 8000
```

或使用 `create_app()` 工厂函数:

```python
from dki.web.app import create_app
app = create_app()
```

## 依赖

- FastAPI + Uvicorn
- DKI Core (dki_system, rag_system, conversation_router)
- DKI Database (SQLite/PostgreSQL)
- DKI Config (config.yaml)
- 可选: Redis (偏好缓存加速)
