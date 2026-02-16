# redis_client.py 程序说明书

**模块路径**: `dki/cache/redis_client.py`  
**版本**: 1.0.0  
**编写日期**: 2026-02-16  
**所属系统**: DKI (Dynamic Knowledge Injection) 缓存子系统

---

## 1. 模块概述

### 1.1 功能定位

`redis_client.py` 是 DKI 系统的**异步 Redis 客户端封装层**，为上层缓存管理器 (`PreferenceCacheManager`) 提供统一的分布式缓存访问接口。该模块封装了连接池管理、自动压缩/解压缩、健康检查、优雅降级等生产级特性。

### 1.2 核心能力

- **连接池管理**: 基于 `redis.asyncio.ConnectionPool` 的高并发连接池
- **自动压缩**: 对超过阈值的数据自动进行 zlib 压缩，使用标记字节协议
- **健康检查**: 后台定期 PING 检测，连接断开时自动重连
- **优雅降级**: Redis 不可用时所有操作返回默认值，不抛出异常
- **全局单例**: 提供 `get_redis_client()` / `close_redis_client()` 全局访问接口
- **原始字节操作**: 提供 `get_raw()` / `set_raw()` 跳过 pickle 序列化的接口

### 1.3 外部依赖

| 依赖 | 用途 | 必需 |
|------|------|------|
| `redis.asyncio` | 异步 Redis 客户端库 | 否 (可选) |
| `loguru` | 日志记录 | 是 |
| `zlib` (标准库) | 数据压缩 | 是 |
| `pickle` (标准库) | 对象序列化 | 是 |

### 1.4 Redis 库可用性检测

```python
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
```

模块加载时自动检测 `redis` 库是否已安装，通过 `REDIS_AVAILABLE` 常量暴露检测结果。

---

## 2. 数据结构定义

### 2.1 RedisConfig (Redis 配置数据类)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **连接设置** | | | |
| `enabled` | `bool` | `False` | 是否启用 Redis |
| `host` | `str` | `"localhost"` | Redis 服务器地址 |
| `port` | `int` | `6379` | Redis 服务器端口 |
| `password` | `str` | `""` | Redis 密码 |
| `db` | `int` | `0` | Redis 数据库编号 |
| **连接池设置** | | | |
| `max_connections` | `int` | `50` | 连接池最大连接数 |
| `socket_timeout` | `float` | `5.0` | Socket 读写超时 (秒) |
| `socket_connect_timeout` | `float` | `5.0` | Socket 连接超时 (秒) |
| `retry_on_timeout` | `bool` | `True` | 超时后是否自动重试 |
| **键设置** | | | |
| `key_prefix` | `str` | `"dki"` | 全局键前缀 |
| `default_ttl` | `int` | `86400` | 默认 TTL (24小时) |
| **压缩设置** | | | |
| `enable_compression` | `bool` | `True` | 是否启用压缩 |
| `compression_level` | `int` | `6` | zlib 压缩级别 (1-9) |
| `compression_threshold` | `int` | `1024` | 压缩阈值 (字节)，仅压缩 > 1KB 的数据 |
| **健康检查** | | | |
| `health_check_interval` | `int` | `30` | 健康检查间隔 (秒) |

#### 配置加载方法

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig"
```

从字典创建配置实例，支持从 YAML 配置文件加载。所有字段均有默认值，缺失字段使用默认值。

---

## 3. 类设计

### 3.1 DKIRedisClient 类

#### 3.1.1 内部数据结构

```
DKIRedisClient
├── config: RedisConfig                       # Redis 配置
├── _pool: Optional[ConnectionPool]           # 连接池
├── _client: Optional[Redis]                  # Redis 客户端实例
├── _connected: bool                          # 连接状态标志
├── _health_task: Optional[asyncio.Task]      # 健康检查后台任务
└── _stats: Dict                              # 操作统计
    ├── connections: int                      # 连接次数
    ├── disconnections: int                   # 断开次数
    ├── operations: int                       # 操作次数
    ├── errors: int                           # 错误次数
    ├── compressions: int                     # 压缩次数
    └── decompressions: int                   # 解压次数
```

#### 3.1.2 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_available` | `bool` | Redis 是否可用 (库已安装 AND 已启用 AND 已连接) |
| `client` | `Optional[Redis]` | 底层 Redis 客户端实例 |

#### 3.1.3 公共方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `connect` | `async () → bool` | 连接 Redis，创建连接池 |
| `close` | `async () → None` | 关闭连接和连接池 |
| `get` | `async (key) → Optional[Any]` | 获取值 (自动反序列化) |
| `set` | `async (key, value, ttl) → bool` | 设置值 (自动序列化) |
| `delete` | `async (key) → bool` | 删除单个键 |
| `delete_pattern` | `async (pattern) → int` | 模式匹配批量删除 |
| `exists` | `async (key) → bool` | 检查键是否存在 |
| `ttl` | `async (key) → int` | 获取键的剩余 TTL |
| `expire` | `async (key, ttl) → bool` | 设置键的 TTL |
| `get_raw` | `async (key) → Optional[bytes]` | 获取原始字节 (不反序列化) |
| `set_raw` | `async (key, value, ttl) → bool` | 设置原始字节 (不序列化) |
| `ping` | `async () → bool` | 测试连接 |
| `info` | `async () → Dict` | 获取 Redis 服务器信息 |
| `get_stats` | `() → Dict` | 获取客户端统计 |

---

## 4. 核心流程说明

### 4.1 连接流程 (connect)

```
connect()
│
├── 1. 检查 REDIS_AVAILABLE
│      └── False → 日志警告 "Redis library not installed" → 返回 False
│
├── 2. 检查 config.enabled
│      └── False → 日志信息 "Redis is disabled" → 返回 False
│
├── 3. 创建连接池 ConnectionPool:
│      ├── host, port, password, db
│      ├── max_connections (默认 50)
│      ├── socket_timeout (默认 5s)
│      ├── socket_connect_timeout (默认 5s)
│      ├── retry_on_timeout (默认 True)
│      └── decode_responses = False  ← 处理二进制数据
│
├── 4. 创建 Redis 客户端:
│      └── Redis(connection_pool=pool)
│
├── 5. 测试连接:
│      └── await client.ping()
│          ├── 成功 → _connected = True
│          └── 失败 → 抛出异常 → _connected = False
│
├── 6. 启动健康检查 (如果 health_check_interval > 0):
│      └── asyncio.create_task(_health_check_loop())
│
└── 返回 True/False
```

### 4.2 关闭流程 (close)

```
close()
│
├── 1. 取消健康检查任务:
│      ├── _health_task.cancel()
│      └── await _health_task (捕获 CancelledError)
│
├── 2. 关闭 Redis 客户端:
│      └── await _client.close()
│
├── 3. 断开连接池:
│      └── await _pool.disconnect()
│
├── 4. 更新状态:
│      ├── _connected = False
│      └── _stats["disconnections"] += 1
│
└── 日志: "Redis connection closed"
```

### 4.3 健康检查循环 (_health_check_loop)

```
_health_check_loop() [后台异步任务]
│
└── while True:
    │
    ├── await asyncio.sleep(health_check_interval)  # 默认 30 秒
    │
    ├── await _client.ping()
    │   ├── 成功 → 继续循环
    │   └── 失败 → _connected = False
    │              └── 尝试重连: await self.connect()
    │                  ├── 成功 → _connected = True (在 connect 内设置)
    │                  └── 失败 → 下次循环再试
    │
    └── CancelledError → 退出循环
```

### 4.4 数据读取流程 (get)

```
get(key)
│
├── 1. 检查 is_available → False 则返回 None
│
├── 2. 构建完整键: full_key = "{key_prefix}:{key}"
│      例: "dki:some_key"
│
├── 3. await _client.get(full_key)
│      └── None → 返回 None
│
├── 4. 解压缩: _decompress(data)
│      ├── 读取标记字节 data[0:1]
│      ├── 0x01 → zlib.decompress(data[1:])
│      └── 0x00 → data[1:] (无压缩)
│
├── 5. 反序列化: pickle.loads(decompressed)
│
└── 返回: 反序列化后的 Python 对象
```

### 4.5 数据写入流程 (set)

```
set(key, value, ttl=None)
│
├── 1. 检查 is_available → False 则返回 False
│
├── 2. 构建完整键: full_key = "{key_prefix}:{key}"
│
├── 3. 确定 TTL: ttl = ttl or config.default_ttl
│
├── 4. 序列化: serialized = pickle.dumps(value)
│
├── 5. 压缩: compressed = _compress(serialized)
│      ├── len(serialized) > compression_threshold?
│      │   ├── YES → 0x01 + zlib.compress(serialized, level)
│      │   └── NO  → 0x00 + serialized
│
├── 6. await _client.setex(full_key, ttl, compressed)
│
└── 返回 True/False
```

### 4.6 模式匹配删除流程 (delete_pattern)

```
delete_pattern(pattern)
│
├── 1. 构建完整模式: full_pattern = "{key_prefix}:{pattern}"
│      例: "dki:dki:pref_kv:user_123:*"
│
├── 2. SCAN 循环 (安全遍历，不阻塞 Redis):
│      │
│      └── while True:
│          ├── cursor, keys = await _client.scan(
│          │       cursor=cursor,
│          │       match=full_pattern,
│          │       count=100          # 每次扫描 100 个键
│          │   )
│          │
│          ├── if keys:
│          │   └── await _client.delete(*keys)  # 批量删除
│          │       └── count += len(keys)
│          │
│          └── if cursor == 0: break  # 扫描完成
│
└── 返回: count (删除的键数量)
```

**为什么使用 SCAN 而非 KEYS**:
- `KEYS pattern` 命令会阻塞 Redis 直到扫描完所有键，在大数据集上可能导致服务不可用
- `SCAN` 命令是增量式的，每次只处理少量键，不会阻塞 Redis
- 这是生产环境的最佳实践

---

## 5. 关键算法说明

### 5.1 压缩/解压缩协议

#### 压缩 (_compress)

```
输入: data (原始字节)
│
├── 检查条件:
│   ├── enable_compression == True?
│   └── len(data) > compression_threshold (默认 1024)?
│
├── 条件满足:
│   ├── compressed = zlib.compress(data, level=compression_level)
│   └── 返回: b'\x01' + compressed
│              ↑ 标记字节: 表示已压缩
│
└── 条件不满足:
    └── 返回: b'\x00' + data
               ↑ 标记字节: 表示未压缩
```

#### 解压缩 (_decompress)

```
输入: data (带标记字节的数据)
│
├── 空数据 → 直接返回
│
├── 读取标记字节: marker = data[0:1]
├── 提取负载: payload = data[1:]
│
├── marker == b'\x01':
│   └── 返回: zlib.decompress(payload)
│
└── marker == b'\x00':
    └── 返回: payload (原始数据)
```

#### 标记字节协议图

```
┌──────────────────────────────────────────┐
│ Redis 中存储的数据格式                      │
├──────────┬───────────────────────────────┤
│ Byte 0   │ Byte 1 ~ N                    │
│ (标记)    │ (负载)                         │
├──────────┼───────────────────────────────┤
│ 0x00     │ 原始数据 (未压缩)               │
│ 0x01     │ zlib 压缩数据                   │
└──────────┴───────────────────────────────┘
```

### 5.2 键前缀机制

所有 Redis 键都会自动添加配置的前缀，实现命名空间隔离：

```
用户传入 key: "pref_kv:user_123:abc123"
实际 Redis key: "dki:pref_kv:user_123:abc123"
                 ↑ key_prefix (默认 "dki")
```

### 5.3 get_raw / set_raw 与 get / set 的区别

```
get/set 流程:
  写入: value → pickle.dumps → _compress → Redis
  读取: Redis → _decompress → pickle.loads → value

get_raw/set_raw 流程:
  写入: bytes → _compress → Redis
  读取: Redis → _decompress → bytes

区别: get_raw/set_raw 跳过 pickle 序列化/反序列化步骤
用途: PreferenceCacheManager 自行处理序列化，使用 raw 接口避免双重序列化
```

---

## 6. 与 Redis 数据库的交互说明

### 6.1 使用的 Redis 命令

| Redis 命令 | 对应方法 | 说明 |
|-----------|---------|------|
| `PING` | `connect()`, `ping()`, `_health_check_loop()` | 连接测试和健康检查 |
| `GET` | `get()`, `get_raw()` | 读取键值 |
| `SETEX` | `set()`, `set_raw()` | 写入键值 (带 TTL) |
| `DEL` | `delete()`, `delete_pattern()` | 删除键 |
| `SCAN` | `delete_pattern()` | 增量式键扫描 |
| `EXISTS` | `exists()` | 检查键存在性 |
| `TTL` | `ttl()` | 查询键的剩余生存时间 |
| `EXPIRE` | `expire()` | 设置键的生存时间 |
| `INFO` | `info()` | 获取服务器信息 |

### 6.2 Redis 键命名规范

```
{key_prefix}:{业务键}

示例:
  dki:dki:pref_kv:user_123:a1b2c3d4    # 偏好缓存
  dki:some_other_key                     # 其他业务数据
```

### 6.3 连接池参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_connections` | 50 | 最大并发连接数，超出时排队等待 |
| `socket_timeout` | 5.0s | 单次读写操作超时 |
| `socket_connect_timeout` | 5.0s | 建立连接超时 |
| `retry_on_timeout` | True | 超时后自动重试一次 |
| `decode_responses` | False | 不自动解码，处理二进制数据 |

---

## 7. 全局单例管理

### 7.1 get_redis_client()

```python
async def get_redis_client(config: Optional[RedisConfig] = None) -> DKIRedisClient
```

**流程**:
```
get_redis_client(config)
│
├── _global_redis_client 存在?
│   ├── YES → 直接返回现有实例
│   └── NO  → 创建新实例
│             ├── DKIRedisClient(config)
│             ├── await connect()
│             └── 保存为全局单例
│
└── 返回: DKIRedisClient 实例
```

**注意**: `config` 参数仅在首次调用时生效，后续调用忽略。

### 7.2 close_redis_client()

```python
async def close_redis_client() → None
```

**流程**:
```
close_redis_client()
│
├── _global_redis_client 存在?
│   ├── YES → await close()
│   │         _global_redis_client = None
│   └── NO  → 无操作
```

---

## 8. 异常处理策略

### 8.1 总体原则

**优雅降级**: 所有公共方法在 Redis 不可用或操作失败时返回安全的默认值，不向上层抛出异常。

### 8.2 各方法异常处理

| 方法 | 异常时返回值 | 日志级别 | 说明 |
|------|------------|---------|------|
| `connect()` | `False` | ERROR | 连接失败 |
| `get()` | `None` | WARNING | 读取失败 |
| `set()` | `False` | WARNING | 写入失败 |
| `delete()` | `False` | WARNING | 删除失败 |
| `delete_pattern()` | `0` | WARNING | 模式删除失败 |
| `exists()` | `False` | WARNING | 存在性检查失败 |
| `ttl()` | `-1` | WARNING | TTL 查询失败 |
| `expire()` | `False` | WARNING | 设置 TTL 失败 |
| `get_raw()` | `None` | WARNING | 原始读取失败 |
| `set_raw()` | `False` | WARNING | 原始写入失败 |
| `ping()` | `False` | 无 | 静默失败 |
| `info()` | `{}` | WARNING | 信息查询失败 |

### 8.3 健康检查异常处理

```
健康检查失败:
  1. _connected = False (标记为不可用)
  2. 尝试重连 connect()
  3. 重连失败 → 静默处理，等待下次检查
  4. 重连成功 → _connected = True (在 connect 内恢复)
```

---

## 9. 统计指标说明

### 9.1 get_stats() 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | `bool` | 是否启用 |
| `connected` | `bool` | 是否已连接 |
| `host` | `str` | 连接地址 (host:port) |
| `db` | `int` | 数据库编号 |
| `connections` | `int` | 累计连接次数 |
| `disconnections` | `int` | 累计断开次数 |
| `operations` | `int` | 累计操作次数 |
| `errors` | `int` | 累计错误次数 |
| `compressions` | `int` | 累计压缩次数 |
| `decompressions` | `int` | 累计解压次数 |

### 9.2 info() 返回的 Redis 服务器信息

| 字段 | 说明 |
|------|------|
| `redis_version` | Redis 版本号 |
| `connected_clients` | 当前连接的客户端数 |
| `used_memory_human` | 已使用内存 (人类可读格式) |
| `total_connections_received` | 累计接收的连接数 |
| `total_commands_processed` | 累计处理的命令数 |

---

## 10. 配置示例

### 10.1 YAML 配置

```yaml
redis:
  enabled: true
  host: "redis-server"
  port: 6379
  password: "your_password"
  db: 0
  max_connections: 50
  socket_timeout: 5.0
  socket_connect_timeout: 5.0
  retry_on_timeout: true
  key_prefix: "dki"
  default_ttl: 86400
  enable_compression: true
  compression_level: 6
  compression_threshold: 1024
  health_check_interval: 30
```

### 10.2 代码使用示例

```python
# 方式1: 直接创建
config = RedisConfig(enabled=True, host="redis-server", password="secret")
client = DKIRedisClient(config)
await client.connect()

# 使用
await client.set("my_key", {"data": "value"}, ttl=3600)
data = await client.get("my_key")

# 关闭
await client.close()

# 方式2: 全局单例
client = await get_redis_client(config)
# ... 使用 ...
await close_redis_client()
```
