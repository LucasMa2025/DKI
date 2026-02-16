# Plugin Interface 插件接口模块说明书

> 源文件: `DKI/dki/core/plugin_interface.py`  
> 模块路径: `dki.core.plugin_interface`  
> 文件行数: 574 行

---

## 1. 模块概述

`plugin_interface.py` 实现了 DKI 论文 Section 8.2 中定义的**标准化插件架构**，提供配置驱动、框架无关的 DKI 集成方案。该模块包含:

- **7 个配置数据类**: 完整的 DKI 插件配置体系
- **1 个抽象接口**: `DKIPluginInterface` (ABC)
- **1 个默认实现**: `DKIPlugin`
- **1 个中间件**: `DKIMiddleware` (FastAPI 集成)

设计目标: 零代码部署、框架无关、渐进式采用、A/B 测试就绪。

---

## 2. 配置数据类体系

### 2.1 DKIPluginConfig (顶层配置)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `True` | 插件总开关 |
| `version` | `str` | `"1.0"` | 配置版本 |
| `memory_source` | `MemorySourceConfig` | — | 记忆数据源配置 |
| `preference_injection` | `PreferenceInjectionConfig` | — | 偏好注入配置 |
| `history_injection` | `HistoryInjectionConfig` | — | 历史注入配置 |
| `gating` | `GatingConfig` | — | 门控配置 |
| `cache` | `CacheConfig` | — | 缓存配置 |
| `safety` | `SafetyConfig` | — | 安全配置 |
| `ab_test` | `ABTestConfig` | — | A/B 测试配置 |

### 2.2 子配置详情

**MemorySourceConfig** — 记忆数据源:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `type` | `"sqlite"` | 数据源类型: sqlite/postgresql/redis/api/file |
| `connection` | `""` | 连接字符串 |
| `table` | `"user_memories"` | 数据表名 |
| `endpoint` | `""` | API 端点 (type=api 时) |
| `auth` | `""` | 认证信息 |

**PreferenceInjectionConfig** — 偏好注入:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用偏好注入 |
| `position_strategy` | `"negative"` | 位置策略: negative/actual_prefix |
| `alpha` | `0.4` | 注入强度 |
| `max_tokens` | `100` | 最大 token 数 |

**HistoryInjectionConfig** — 历史注入:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用历史注入 |
| `method` | `"suffix_prompt"` | 注入方式: suffix_prompt/kv_injection |
| `max_tokens` | `500` | 最大 token 数 |
| `prompt_template` | `"default"` | 提示词模板 |

**GatingConfig** — 门控:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `relevance_threshold` | `0.7` | 相关性阈值 |
| `entropy_ceiling` | `1.0` | 熵上限 |
| `entropy_floor` | `0.5` | 熵下限 |

**SafetyConfig** — 安全:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_alpha` | `0.8` | alpha 最大值 (安全限制) |
| `fallback_on_error` | `True` | 错误时降级 |
| `audit_logging` | `True` | 审计日志开关 |
| `log_path` | `"./dki_audit.log"` | 审计日志路径 |

**ABTestConfig** — A/B 测试:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `False` | 是否启用 A/B 测试 |
| `dki_percentage` | `50` | DKI 流量百分比 |

---

## 3. DKIPluginInterface (抽象接口)

定义了 DKI 插件必须实现的 4 个抽象方法:

| 方法 | 签名 | 说明 |
|------|------|------|
| `load_config` | `(config_path: str) -> None` | 加载 YAML 配置 |
| `connect_memory_source` | `(source_config) -> None` | 连接记忆数据源 |
| `get_user_memory` | `(user_id: str) -> (str, str)` | 获取用户记忆 (偏好, 历史) |
| `compute_memory_kv` | `(text, model) -> (K, V)` | 计算记忆的 K/V 表示 |
| `inject_memory` | `(K_user, V_user, K_mem, V_mem, alpha) -> (K, V)` | 执行 K/V 注入 |

---

## 4. DKIPlugin (默认实现)

### 4.1 初始化流程

```
DKIPlugin.__init__(config)
  ├─ 存储配置
  ├─ 初始化内部状态 (_model, _tokenizer, _memory_connection, _kv_cache, _audit_log)
  ├─ 创建 HybridDKIInjector (从配置构建 HybridInjectionConfig)
  └─ 日志记录
```

### 4.2 `connect_memory_source()` — 连接数据源

```
connect_memory_source(source_config)
  ├─ type == "sqlite" → sqlite3.connect(connection)
  ├─ type == "postgresql" → psycopg2.connect(connection)
  ├─ type == "redis" → redis.from_url(connection)
  ├─ type == "api" → 存储 {endpoint, auth}
  ├─ type == "file" → 存储文件路径
  └─ 其他 → 警告日志
```

### 4.3 `get_user_memory()` — 获取用户记忆

```
get_user_memory(user_id)
  ├─ 连接为 dict (API)?
  │   └─ GET {endpoint}/users/{user_id}/memory → (preferences, history)
  ├─ 连接有 cursor (SQL)?
  │   └─ SELECT preferences, history FROM user_memories WHERE user_id = ?
  ├─ 连接有 get (Redis)?
  │   └─ HGETALL user:{user_id} → (preferences, history)
  └─ 返回 (preferences, history)
```

### 4.4 `compute_memory_kv()` — 计算 K/V

```
compute_memory_kv(memory_text, model)
  ├─ 检查缓存: hash(memory_text) in _kv_cache? → 返回缓存
  ├─ model 有 compute_kv 方法 (DKI 适配器)?
  │   ├─ kv_entries, _ = model.compute_kv(memory_text)
  │   ├─ K = cat([e.key for e in kv_entries])
  │   └─ V = cat([e.value for e in kv_entries])
  ├─ model 有 model 属性 (HuggingFace)?
  │   ├─ tokens = tokenizer.encode(memory_text)
  │   ├─ outputs = model.model(input_ids=tokens, use_cache=True)
  │   ├─ K = stack([layer[0] for layer in outputs.past_key_values])
  │   └─ V = stack([layer[1] for layer in outputs.past_key_values])
  ├─ 缓存结果 (如果启用)
  └─ 返回 (K, V)
```

### 4.5 `inject_memory()` — K/V 注入

```
inject_memory(K_user, V_user, K_mem, V_mem, alpha)
  ├─ alpha = min(alpha, safety.max_alpha)  (安全限制)
  ├─ K_combined = cat([K_mem, K_user], dim=-2)
  ├─ V_combined = cat([V_mem, V_user], dim=-2)
  └─ 返回 (K_combined, V_combined)
  注意: alpha 缩放在注意力计算时通过 logit bias 应用，不在此处
```

### 4.6 `should_use_dki()` — A/B 测试路由

```
should_use_dki(user_id)
  ├─ 插件未启用? → False
  ├─ A/B 测试启用?
  │   ├─ user_hash = hash(user_id) % 100
  │   └─ 返回 user_hash < dki_percentage
  └─ 返回 True
```

**A/B 测试算法:** 基于 user_id 的哈希值取模，确保同一用户始终被路由到同一组 (一致性体验)。

---

## 5. DKIMiddleware (FastAPI 中间件)

```python
# 使用方式
from fastapi import FastAPI
from dki.core.plugin_interface import DKIMiddleware

app = FastAPI()
app.add_middleware(DKIMiddleware, config_path="./dki_config.yaml")
```

**工作原理:**
- 初始化时从 YAML 加载配置并创建 DKIPlugin 实例
- 每个 HTTP 请求到达时，将 plugin 注入到 `scope["state"]["dki_plugin"]`
- 路由处理函数可通过 `request.state.dki_plugin` 访问 DKI 功能

---

## 6. YAML 配置文件示例

```yaml
dki:
  enabled: true
  version: "1.0"
  
  memory_source:
    type: sqlite
    connection: "./data/user_data.db"
    table: user_memories
  
  injection:
    preference_injection:
      enabled: true
      position_strategy: negative
      alpha: 0.4
      max_tokens: 100
    history_injection:
      enabled: true
      method: suffix_prompt
      max_tokens: 500
  
  gating:
    relevance_threshold: 0.7
    entropy_ceiling: 1.0
    entropy_floor: 0.5
  
  cache:
    enabled: true
    max_size: 100
    strategy: weighted
    ttl_seconds: 3600
  
  safety:
    max_alpha: 0.8
    fallback_on_error: true
    audit_logging: true
    log_path: ./dki_audit.log
  
  ab_test:
    enabled: false
    dki_percentage: 50
```

---

## 7. 数据库交互

### 7.1 user_memories 表 (通过 connect_memory_source 访问)

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | `TEXT` | 用户标识 |
| `preferences` | `TEXT` | 用户偏好文本 |
| `history` | `TEXT` | 历史摘要文本 |

### 7.2 审计日志文件 (JSONL 格式)

每条日志包含:
```json
{
  "timestamp": 1708000000.0,
  "user_id": "user_123",
  "memory_ids": ["mem_1", "mem_2"],
  "alpha": 0.4,
  "status": "success",
  "metadata": {}
}
```

---

## 8. 设计说明

- **配置驱动**: 通过 YAML 文件配置所有参数，实现零代码部署
- **框架无关**: 抽象接口支持 vLLM、HuggingFace、TensorRT-LLM 等
- **渐进采用**: 可通过 `enabled` 开关逐步启用，`ab_test` 支持灰度发布
- **安全限制**: `max_alpha` 防止过度注入，`fallback_on_error` 确保降级
- **K/V 缓存**: 基于文本哈希的内存缓存，避免重复计算
