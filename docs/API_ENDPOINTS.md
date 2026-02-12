# DKI API 端点文档

## 端点前缀规范

| 前缀 | 用途 | 说明 |
|------|------|------|
| `/api/*` | 内部 API | 认证、会话、偏好、统计等内部功能 |
| `/v1/*` | OpenAI 兼容 API | 聊天、模型列表等 OpenAI 兼容端点 |
| `/v1/dki/*` | DKI 特定 API | DKI 增强聊天、可视化等 DKI 特定功能 |

## 完整端点列表

### 认证 API (`/api/auth`)

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/logout` | 用户登出 |
| POST | `/api/auth/register` | 用户注册 |
| GET | `/api/auth/me` | 获取当前用户信息 |

### 会话管理 API (`/api/sessions`)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/sessions` | 列出所有会话 |
| POST | `/api/sessions` | 创建新会话 |
| GET | `/api/sessions/{id}` | 获取会话详情 |
| PATCH | `/api/sessions/{id}` | 更新会话 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/sessions/{id}/messages` | 获取会话消息 |

### 偏好管理 API (`/api/preferences`)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/preferences?user_id=xxx` | 列出用户偏好 |
| POST | `/api/preferences` | 创建偏好 |
| GET | `/api/preferences/{id}` | 获取偏好详情 |
| PATCH | `/api/preferences/{id}` | 更新偏好 |
| DELETE | `/api/preferences/{id}` | 删除偏好 |

### 统计 API (`/api/stats`)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 系统统计 |
| GET | `/api/stats/dki` | DKI 统计 |
| GET | `/api/stats/cache` | 缓存统计 |

### OpenAI 兼容 API (`/v1`)

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | OpenAI 兼容聊天 |
| GET | `/v1/models` | 模型列表 |
| GET | `/v1/health` | 健康检查 |
| GET | `/v1/stats` | 统计信息 |

### DKI 特定 API (`/v1/dki`)

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/dki/chat` | DKI 增强聊天 (推荐) |
| POST | `/v1/dki/inject` | 直接 DKI 注入 |
| GET | `/v1/dki/info` | DKI 插件信息 |
| GET | `/v1/dki/preferences/{user_id}` | 获取用户偏好 (DKI 内部) |
| POST | `/v1/dki/preferences/{user_id}` | 更新用户偏好 (DKI 内部) |
| DELETE | `/v1/dki/preferences/{user_id}/{pref_id}` | 删除偏好 (DKI 内部) |
| GET | `/v1/dki/sessions/{session_id}/history` | 获取会话历史 (DKI 内部) |

### 可视化 API (`/v1/dki/visualization`)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/v1/dki/visualization/latest` | 最新注入可视化 |
| GET | `/v1/dki/visualization/history` | 历史记录 |
| GET | `/v1/dki/visualization/detail/{request_id}` | 详情 |
| GET | `/v1/dki/visualization/flow-diagram` | 流程图 |
| DELETE | `/v1/dki/visualization/history` | 清除历史 |

## 前端 API 调用示例

### 聊天 (推荐使用 `/v1/dki/chat`)

```typescript
// 使用 DKI 增强聊天
const response = await api.chat.send({
    query: "用户原始输入",
    dkiUserId: "user_123",
    dkiSessionId: "session_456",
    temperature: 0.7,
    maxTokens: 2048,
});
```

### 认证

```typescript
// 登录
const { token, user } = await api.auth.login({
    username: "admin",
    password: "password",
});

// 获取当前用户
const currentUser = await api.auth.getCurrentUser();
```

### 会话管理

```typescript
// 列出会话
const sessions = await api.sessions.list();

// 创建会话
const newSession = await api.sessions.create("新对话");

// 获取会话消息
const messages = await api.sessions.getMessages(sessionId);
```

## 注意事项

1. **认证**: 除了 `/api/auth/login` 和 `/api/auth/register`，其他端点都需要 Bearer Token 认证
2. **CORS**: 后端已配置 CORS，允许跨域请求
3. **错误处理**: 所有错误返回 `{ detail: "错误信息" }` 格式
4. **分页**: 列表端点支持 `limit` 和 `offset` 参数
