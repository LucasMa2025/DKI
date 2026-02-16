# preference_cache.py 程序说明书

**模块路径**: `dki/cache/preference_cache.py`  
**版本**: 2.0.0  
**编写日期**: 2026-02-16  
**所属系统**: DKI (Dynamic Knowledge Injection) 缓存子系统

---

## 1. 模块概述

### 1.1 功能定位

`preference_cache.py` 实现了 DKI 系统的**分层偏好 K/V 缓存管理器**，负责对用户偏好文本经模型计算后产生的 Key-Value 缓存数据进行多级缓存管理。该模块是 DKI 系统性能优化的核心组件，通过三级缓存架构（L1 内存 → L2 Redis → L3 按需计算）大幅降低重复计算开销。

### 1.2 核心能力

- **三级缓存查找**: L1(内存) → L2(Redis) → L3(按需计算)，逐级降级
- **LRU 淘汰策略**: 基于 OrderedDict 的 O(1) 时间复杂度 LRU 缓存
- **自动压缩**: 对大型 K/V 张量数据进行 zlib 压缩存储
- **分布式共享**: 通过 Redis L2 层实现多实例间缓存共享
- **优雅降级**: Redis 不可用时自动降级为 L1+L3 模式
- **缓存预热**: 支持批量预热高频用户的缓存数据

### 1.3 外部依赖

| 依赖 | 用途 | 必需 |
|------|------|------|
| `torch` | 张量数据处理 | 是 |
| `numpy` | 张量序列化/反序列化 | 是 |
| `loguru` | 日志记录 | 是 |
| `zlib` (标准库) | 数据压缩 | 是 |
| `pickle` (标准库) | 对象序列化 | 是 |
| `hashlib` (标准库) | 偏好文本哈希 | 是 |
| `dki.cache.redis_client.DKIRedisClient` | Redis 客户端 | 否 (TYPE_CHECKING) |
| `dki.models.base.KVCacheEntry` | K/V 缓存条目数据结构 | 是 (反序列化时) |

---

## 2. 数据结构定义

### 2.1 枚举类型

#### CacheTier (缓存层级枚举)

```python
class CacheTier(str, Enum):
    L1_MEMORY = "L1_MEMORY"    # 内存缓存层
    L2_REDIS  = "L2_REDIS"     # Redis 分布式缓存层
    L3_COMPUTE = "L3_COMPUTE"  # 按需计算层
    MISS      = "MISS"         # 缓存未命中
```

### 2.2 数据类

#### CacheTierInfo (缓存命中信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| `tier` | `CacheTier` | 命中的缓存层级 |
| `latency_ms` | `float` | 查询延迟 (毫秒) |
| `hit` | `bool` | 是否命中缓存 |
| `user_id` | `str` | 用户标识 |
| `preference_hash` | `str` | 偏好文本哈希值 |
| `size_bytes` | `int` | 数据大小 (字节)，默认 0 |

#### CacheEntry (缓存条目)

| 字段 | 类型 | 说明 |
|------|------|------|
| `kv_data` | `Any` | K/V 缓存数据 (List[KVCacheEntry] 或序列化字节) |
| `preference_hash` | `str` | 偏好文本的 MD5 哈希 |
| `created_at` | `float` | 创建时间戳 |
| `last_accessed` | `float` | 最后访问时间戳 |
| `access_count` | `int` | 访问次数，默认 1 |
| `size_bytes` | `int` | 数据大小 (字节)，默认 0 |

#### CacheConfig (缓存配置)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `l1_max_size` | `int` | 1000 | L1 内存缓存最大用户数 |
| `l1_max_memory_mb` | `int` | 5000 | L1 最大内存限制 (MB) |
| `l2_enabled` | `bool` | False | 是否启用 L2 Redis 缓存 |
| `l2_ttl_seconds` | `int` | 86400 | L2 缓存过期时间 (24小时) |
| `l2_key_prefix` | `str` | "dki:pref_kv" | L2 Redis 键前缀 |
| `enable_compression` | `bool` | True | 是否启用 zlib 压缩 |
| `compression_level` | `int` | 6 | zlib 压缩级别 (1-9) |

### 2.3 关联外部数据结构

#### KVCacheEntry (来自 `dki/models/base.py`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | `torch.Tensor` | Key 张量，形状 [batch, num_heads, seq_len, head_dim] |
| `value` | `torch.Tensor` | Value 张量，形状 [batch, num_heads, seq_len, head_dim] |
| `layer_idx` | `int` | 模型层索引 |

---

## 3. 类设计

### 3.1 LRUCache 类

#### 3.1.1 设计说明

基于 `OrderedDict` 实现的**线程安全 LRU (最近最少使用) 缓存**，所有核心操作均为 O(1) 时间复杂度。

#### 3.1.2 内部数据结构

```
LRUCache
├── maxsize: int                              # 最大容量
├── _cache: OrderedDict[str, CacheEntry]      # 有序字典 (核心存储)
├── _lock: asyncio.Lock                       # 异步锁 (并发安全)
└── _stats: Dict                              # 统计信息
    ├── hits: int                             # 命中次数
    ├── misses: int                           # 未命中次数
    └── evictions: int                        # 淘汰次数
```

#### 3.1.3 方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `get` | `async get(key: str) → Optional[CacheEntry]` | 获取缓存项，更新访问顺序 |
| `put` | `async put(key: str, entry: CacheEntry) → None` | 存入缓存项，满时淘汰最旧 |
| `delete` | `async delete(key: str) → bool` | 删除指定缓存项 |
| `delete_prefix` | `async delete_prefix(prefix: str) → int` | 删除指定前缀的所有缓存项 |
| `clear` | `async clear() → None` | 清空所有缓存 |
| `get_stats` | `get_stats() → Dict[str, Any]` | 获取缓存统计信息 |

#### 3.1.4 LRU 淘汰算法详解

```
OrderedDict 内部结构 (双向链表 + 哈希表):

  HEAD (最旧) ←→ ... ←→ ... ←→ TAIL (最新)
  ┌────────┐    ┌────────┐    ┌────────┐
  │ user_A │ ←→ │ user_B │ ←→ │ user_C │
  └────────┘    └────────┘    └────────┘

操作说明:
  get(user_B):
    1. 在哈希表中 O(1) 查找 user_B
    2. move_to_end(user_B) → O(1) 移到尾部
    3. 更新 last_accessed 和 access_count

  put(user_D) 且缓存已满:
    1. popitem(last=False) → O(1) 淘汰头部 (user_A)
    2. 将 user_D 插入尾部

  delete_prefix("user_"):
    1. 遍历所有 key，收集匹配前缀的 key → O(n)
    2. 逐个删除 → O(k), k 为匹配数量
```

### 3.2 PreferenceCacheManager 类

#### 3.2.1 设计说明

**分层缓存管理器**，协调 L1(内存)、L2(Redis)、L3(计算) 三级缓存的读写操作。

#### 3.2.2 内部数据结构

```
PreferenceCacheManager
├── config: CacheConfig                       # 缓存配置
├── _redis_client: Optional[DKIRedisClient]   # Redis 客户端 (可选)
├── _l1_cache: LRUCache                       # L1 内存缓存
└── _stats: Dict                              # 统计信息
    ├── l1_hits: int                          # L1 命中次数
    ├── l2_hits: int                          # L2 命中次数
    ├── l3_computes: int                      # L3 计算次数
    ├── total_requests: int                   # 总请求数
    ├── invalidations: int                    # 失效操作次数
    ├── l2_errors: int                        # L2 错误次数
    └── total_bytes_cached: int               # 总缓存字节数
```

#### 3.2.3 方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_preference_kv` | `async (user_id, preference_text, model, force_recompute) → (kv_entries, CacheTierInfo)` | 核心方法：获取偏好 K/V 缓存 |
| `invalidate` | `async (user_id) → int` | 失效指定用户的所有缓存 |
| `clear_all` | `async () → None` | 清空所有缓存 |
| `warm_cache` | `async (user_ids, preference_getter, model) → Dict` | 批量预热缓存 |
| `get_stats` | `() → Dict[str, Any]` | 获取缓存统计 |
| `_compute_kv` | `async (preference_text, model) → Any` | 调用模型计算 K/V |
| `_store_in_caches` | `async (cache_key, kv_entries, preference_hash) → None` | 存入 L1+L2 缓存 |
| `_serialize_kv` | `(kv_entries) → bytes` | 序列化 K/V 数据 |
| `_deserialize_kv` | `(data: bytes) → Any` | 反序列化 K/V 数据 |
| `_compute_preference_hash` | `(preference_text) → str` | 计算偏好文本哈希 |
| `_make_cache_key` | `(user_id, preference_hash) → str` | 生成缓存键 |
| `_make_redis_key` | `(cache_key) → str` | 生成 Redis 键 |
| `_is_l2_available` | `() → bool` | 检查 L2 是否可用 |

---

## 4. 核心流程说明

### 4.1 获取偏好 K/V 缓存 (get_preference_kv) — 主流程

这是本模块最核心的方法，实现了三级缓存的逐级查找逻辑。

```
get_preference_kv(user_id, preference_text, model, force_recompute)
│
├── 1. 计算 preference_hash = MD5(preference_text)[:16]
├── 2. 生成 cache_key = "{user_id}:{preference_hash}"
│
├── [force_recompute = True?]
│   ├── YES → 3a. 调用 _compute_kv() 计算 K/V
│   │         3b. 调用 _store_in_caches() 写入 L1+L2
│   │         3c. 返回 (kv_entries, CacheTierInfo(L3_COMPUTE))
│   │
│   └── NO → 继续逐级查找
│
├── 4. L1 查找: _l1_cache.get(cache_key)
│   ├── 命中 → 返回 (entry.kv_data, CacheTierInfo(L1_MEMORY, hit=True))
│   └── 未命中 → 继续
│
├── 5. L2 查找 (仅当 Redis 可用):
│   │   redis_key = "{l2_key_prefix}:{cache_key}"
│   │
│   ├── 5a. redis_client.get_raw(redis_key)
│   ├── 5b. 命中 → _deserialize_kv(cached_data)
│   │         5c. 提升到 L1: _l1_cache.put(cache_key, ...)
│   │         5d. 返回 (kv_entries, CacheTierInfo(L2_REDIS, hit=True))
│   └── 未命中或异常 → 继续
│
├── 6. L3 计算:
│   ├── 6a. _compute_kv(preference_text, model)
│   ├── 6b. _store_in_caches(cache_key, kv_entries, preference_hash)
│   └── 6c. 返回 (kv_entries, CacheTierInfo(L3_COMPUTE, hit=False))
│
└── 返回 (kv_entries, tier_info)
```

### 4.2 缓存存储流程 (_store_in_caches)

```
_store_in_caches(cache_key, kv_entries, preference_hash)
│
├── 1. 序列化: serialized = _serialize_kv(kv_entries)
│      └── 计算 size_bytes = len(serialized)
│
├── 2. L1 存储:
│      └── _l1_cache.put(cache_key, CacheEntry(...))
│
└── 3. L2 存储 (仅当 Redis 可用):
       ├── redis_key = "{l2_key_prefix}:{cache_key}"
       ├── redis_client.set_raw(redis_key, serialized, ttl=l2_ttl_seconds)
       └── 异常时记录 l2_errors，不影响主流程
```

### 4.3 缓存失效流程 (invalidate)

```
invalidate(user_id)
│
├── 1. L1 失效:
│      ├── prefix = "{user_id}:"
│      └── _l1_cache.delete_prefix(prefix)
│          └── 遍历所有 key，删除以 prefix 开头的条目
│
├── 2. L2 失效 (仅当 Redis 可用):
│      ├── pattern = "{l2_key_prefix}:{user_id}:*"
│      └── redis_client.delete_pattern(pattern)
│          └── 使用 SCAN 命令安全遍历并删除匹配键
│
└── 返回: 总删除条目数
```

### 4.4 缓存预热流程 (warm_cache)

```
warm_cache(user_ids, preference_getter, model)
│
├── 遍历 user_ids:
│   │
│   ├── 1. preference_text = await preference_getter(user_id)
│   │      └── 为空 → 跳过 (skipped++)
│   │
│   ├── 2. (kv, tier_info) = await get_preference_kv(user_id, preference_text, model)
│   │
│   ├── 3. tier_info.hit == True?
│   │      ├── YES → 已缓存，跳过 (skipped++)
│   │      └── NO  → 新计算并缓存 (success++)
│   │
│   └── 4. 异常 → failed++
│
└── 返回统计: {total, success, failed, skipped}
```

---

## 5. 关键算法说明

### 5.1 偏好文本哈希算法

```python
def _compute_preference_hash(self, preference_text: str) -> str:
    return hashlib.md5(preference_text.encode()).hexdigest()[:16]
```

**算法说明**:
- 使用 MD5 对偏好文本进行哈希
- 取前 16 个十六进制字符 (64 位)
- 碰撞概率: 在 10^9 个不同偏好文本下，碰撞概率约 2.7 × 10^-11
- 用途: 作为缓存键的一部分，判断偏好文本是否变化

### 5.2 缓存键生成规则

```
L1 缓存键格式:
  {user_id}:{preference_hash}
  例: user_123:a1b2c3d4e5f67890

L2 Redis 键格式:
  {redis_key_prefix}:{l2_key_prefix}:{user_id}:{preference_hash}
  例: dki:dki:pref_kv:user_123:a1b2c3d4e5f67890
  注: 存在双重 "dki:" 前缀 (redis_client.key_prefix + cache.l2_key_prefix)
```

### 5.3 K/V 数据序列化算法 (_serialize_kv)

```
输入: List[KVCacheEntry] (GPU 张量列表)
│
├── 1. 遍历每个 KVCacheEntry:
│      ├── key_np = entry.key.cpu().numpy()     # GPU → CPU → NumPy
│      ├── value_np = entry.value.cpu().numpy()
│      └── 构建字典:
│          {
│            'key': key_np.tobytes(),            # NumPy → 原始字节
│            'value': value_np.tobytes(),
│            'layer_idx': entry.layer_idx,
│            'key_shape': list(key_np.shape),    # 保存形状信息
│            'value_shape': list(value_np.shape),
│            'key_dtype': str(key_np.dtype),     # 保存数据类型
│            'value_dtype': str(value_np.dtype),
│          }
│
├── 2. pickle.dumps(serializable_list)           # 序列化为字节流
│
├── 3. [enable_compression?]
│      ├── YES → zlib.compress(data, level=compression_level)
│      └── NO  → 原始数据
│
└── 输出: bytes (压缩后的序列化字节)
```

### 5.4 K/V 数据反序列化算法 (_deserialize_kv)

```
输入: bytes (压缩后的序列化字节)
│
├── 1. [enable_compression?]
│      ├── YES → zlib.decompress(data)
│      └── NO  → 原始数据
│
├── 2. pickle.loads(data) → serializable_list
│
├── 3. 遍历每个字典项:
│      ├── 读取 key_shape, value_shape (兼容旧格式 'shape')
│      ├── 读取 key_dtype, value_dtype (默认 float16)
│      ├── np.frombuffer(item['key'], dtype=key_dtype)
│      │   .reshape(key_shape).copy()             # copy() 避免只读数组
│      ├── torch.from_numpy(key_array)             # NumPy → PyTorch 张量
│      └── 构建 KVCacheEntry(key, value, layer_idx)
│
└── 输出: List[KVCacheEntry]
```

**向后兼容说明**: 反序列化支持两种格式:
- **旧格式**: 使用单一 `shape` 和 `dtype` 字段
- **新格式**: 使用分离的 `key_shape`/`value_shape` 和 `key_dtype`/`value_dtype` 字段

### 5.5 L2 提升 (Promotion) 机制

当数据在 L2 (Redis) 命中时，自动提升到 L1 (内存):

```
L2 命中 → 反序列化 → 写入 L1 → 返回数据

目的:
- 后续相同请求直接命中 L1 (< 1ms)，无需再访问 Redis (1-5ms)
- 实现热数据自动上浮的效果
```

---

## 6. 与 Redis 的交互说明

### 6.1 Redis 键结构

| 操作 | Redis 键模式 | 说明 |
|------|-------------|------|
| 存储/读取 | `dki:dki:pref_kv:{user_id}:{pref_hash}` | 单个用户偏好的 K/V 数据 |
| 失效 (用户级) | `dki:dki:pref_kv:{user_id}:*` | 删除用户所有偏好缓存 |
| 失效 (全局) | `dki:dki:pref_kv:*` | 清空所有偏好缓存 |

### 6.2 Redis 操作清单

| 操作 | 调用方法 | Redis 命令 | 说明 |
|------|---------|-----------|------|
| 读取 | `redis_client.get_raw(key)` | `GET` | 读取原始字节数据 |
| 写入 | `redis_client.set_raw(key, data, ttl)` | `SETEX` | 写入带 TTL 的原始字节 |
| 模式删除 | `redis_client.delete_pattern(pattern)` | `SCAN` + `DEL` | 安全的模式匹配删除 |

### 6.3 数据在 Redis 中的存储格式

```
Redis Value 结构:
┌──────────┬─────────────────────────────────────────┐
│ 标记字节  │ 数据负载                                  │
│ (1 byte) │ (变长)                                    │
├──────────┼─────────────────────────────────────────┤
│ 0x00     │ 未压缩的 [pickle + zlib 压缩的 KV 数据]   │
│ 0x01     │ zlib 压缩的 [pickle + zlib 压缩的 KV 数据]│
└──────────┴─────────────────────────────────────────┘

注: 存在双重压缩:
  第1层: preference_cache._serialize_kv() 的 zlib 压缩
  第2层: redis_client._compress() 的 zlib 压缩
  (第2层对已压缩数据的压缩收益极低)
```

---

## 7. 统计指标说明

### 7.1 get_stats() 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_requests` | `int` | 总请求数 |
| `l1_hits` | `int` | L1 命中次数 |
| `l2_hits` | `int` | L2 命中次数 |
| `l3_computes` | `int` | L3 计算次数 |
| `invalidations` | `int` | 失效操作次数 |
| `l2_errors` | `int` | L2 错误次数 |
| `l1_hit_rate` | `float` | L1 命中率 (l1_hits / total) |
| `l2_hit_rate` | `float` | L2 命中率 (l2_hits / total) |
| `overall_hit_rate` | `float` | 总命中率 ((l1+l2) / total) |
| `total_bytes_cached` | `int` | 总缓存字节数 |
| `total_mb_cached` | `float` | 总缓存 MB 数 |
| `l1_cache` | `Dict` | L1 LRU 缓存统计 |
| `l2_enabled` | `bool` | L2 是否启用 |
| `l2_available` | `bool` | L2 是否可用 |
| `redis` | `Dict` | Redis 客户端统计 |

---

## 8. 异常处理策略

| 场景 | 处理方式 | 影响 |
|------|---------|------|
| Redis 连接失败 | 跳过 L2，继续 L1+L3 | 降级运行，命中率下降 |
| Redis GET 异常 | 记录 l2_errors，继续 L3 计算 | 单次请求延迟增加 |
| Redis SET 异常 | 记录 l2_errors，不影响返回 | 数据未写入 L2 |
| 模型 compute_kv 异常 | 返回空列表 [] | 无 K/V 注入 |
| 序列化异常 | 返回空 pickle (pickle.dumps([])) | 缓存空数据 |
| 反序列化异常 | 返回空列表 [] | 等同于缓存未命中 |

---

## 9. 配置示例

### 9.1 YAML 配置

```yaml
cache:
  l1_max_size: 1000          # L1 最大用户数
  l1_max_memory_mb: 5000     # L1 最大内存 (MB)
  l2_enabled: true           # 启用 Redis L2
  l2_ttl_seconds: 86400      # L2 TTL: 24小时
  l2_key_prefix: "dki:pref_kv"
  enable_compression: true   # 启用压缩
  compression_level: 6       # 压缩级别
```

### 9.2 代码使用示例

```python
# 基础使用 (仅 L1)
cache = PreferenceCacheManager()

# 生产环境使用 (L1 + L2)
from dki.cache import DKIRedisClient, RedisConfig
redis_client = DKIRedisClient(RedisConfig(enabled=True))
await redis_client.connect()

config = CacheConfig(l2_enabled=True, l1_max_size=2000)
cache = PreferenceCacheManager(redis_client=redis_client, config=config)

# 获取偏好 K/V
kv_entries, tier_info = await cache.get_preference_kv(
    user_id="user_123",
    preference_text="素食主义者，不吃辣",
    model=model_adapter,
)
print(f"命中层级: {tier_info.tier}, 延迟: {tier_info.latency_ms:.1f}ms")

# 用户偏好更新时失效缓存
await cache.invalidate(user_id="user_123")
```
