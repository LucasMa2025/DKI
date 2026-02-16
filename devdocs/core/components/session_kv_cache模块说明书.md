# Session KV Cache 模块说明书

> 文件路径: `DKI/dki/core/components/session_kv_cache.py`

## 1. 模块概述

本模块实现了 DKI 系统的**会话级 K/V 缓存 (Session KV Cache)**，管理单个会话内计算过的 K/V 表示。这是将 DKI 从**无状态记忆注入机制**转变为**有状态时序算子**的关键组件。

### 核心洞察: 记忆摊销

$$\text{摊销成本} = \frac{C_{\text{compute}} + (T-1) \cdot C_{\text{load}}}{T} \xrightarrow{T \to \infty} C_{\text{load}}$$

- 第 1 轮: 计算 K/V (昂贵)
- 第 2~T 轮: 从缓存加载 K/V (廉价)
- 随着对话轮次增加，平均成本趋近于加载成本

### 与 RAG 的对比

| 特性 | RAG | DKI + Session Cache |
|------|-----|---------------------|
| 每轮处理 | 重新处理记忆 token | 加载缓存 K/V |
| 长对话效率 | 线性增长 | 摊销递减 |
| 时序一致性 | 每轮可能不同 | 同一记忆表示一致 |

### 设计不变量

- 会话缓存在会话结束后可丢弃
- 它是推理时的性能增强，不是持久化记忆

## 2. 数据结构

### 2.1 CacheMetadata

缓存条目元数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `memory_id` | `str` | 记忆标识符 |
| `created_at` | `float` | 创建时间戳 |
| `last_accessed` | `float` | 最后访问时间戳 |
| `access_count` | `int` | 访问次数 |
| `alpha` | `float` | 最后使用的 α 值 |
| `query_hash` | `str` | 查询哈希 (用于查询条件化缓存) |

### 2.2 CachedKV

缓存的 K/V 表示。

| 字段 | 类型 | 说明 |
|------|------|------|
| `entries` | `List[KVCacheEntry]` | K/V 缓存条目列表 (每层一个) |
| `metadata` | `CacheMetadata` | 元数据 |

## 3. 核心类: SessionKVCache

### 3.1 初始化

```python
cache = SessionKVCache(
    max_size=100,          # 最大缓存条目数
    strategy="weighted",   # 淘汰策略: "lru", "lfu", "weighted"
    ttl_seconds=3600,      # 生存时间 (秒)
)
```

内部使用 `OrderedDict` 存储，支持 O(1) 的 LRU 操作。

### 3.2 get() — 获取缓存

**流程:**

```
输入: memory_id, query? (可选)
  │
  ├── 1. 构建缓存键
  │     query_hash = hash(query) % 10000000 (如有 query)
  │     cache_key = "{memory_id}:{query_hash}" 或 "{memory_id}"
  │
  ├── 2. 查找缓存
  │     cache_key not in _cache → miss, 返回 None
  │
  ├── 3. TTL 检查
  │     now - created_at > ttl_seconds → 过期, 删除, 返回 None
  │
  ├── 4. 更新元数据
  │     last_accessed = now
  │     access_count += 1
  │     move_to_end(cache_key)  # LRU 更新
  │
  └── 5. 返回 cached.entries (hit)
```

### 3.3 put() — 存储缓存

**流程:**

```
输入: memory_id, entries (KVCacheEntry 列表), query?, alpha
  │
  ├── 1. 构建缓存键
  │
  ├── 2. 淘汰检查
  │     while len(_cache) >= max_size:
  │       _evict()  # 按策略淘汰
  │
  ├── 3. 创建元数据
  │     CacheMetadata(memory_id, now, now, 1, alpha, query_hash)
  │
  └── 4. 存储
        _cache[cache_key] = CachedKV(entries, metadata)
```

### 3.4 get_or_compute() — 获取或计算

**流程:**

```
输入: memory_id, text, model, projection, X_user, alpha
  │
  ├── 1. 尝试缓存
  │     cached = get(memory_id)
  │     命中 → 返回 (cached, True)
  │
  ├── 2. 计算 K/V
  │     kv_entries, hidden_states = model.compute_kv(text, return_hidden=True)
  │
  ├── 3. 查询条件化投影 (如有)
  │     if projection and hidden_states:
  │       X_mem_proj = projection(hidden_states, X_user)
  │       (注: 当前为简化实现，未重新计算 K/V)
  │
  ├── 4. 存入缓存
  │     put(memory_id, kv_entries, alpha=alpha)
  │
  └── 5. 返回 (kv_entries, False)
```

### 3.5 _evict() — 淘汰策略

**三种淘汰策略:**

#### LRU (Least Recently Used)

```python
self._cache.popitem(last=False)  # 移除最早的条目
```

O(1) 操作，利用 OrderedDict 的有序特性。

#### LFU (Least Frequently Used)

```python
min_key = min(keys, key=lambda k: cache[k].metadata.access_count)
del cache[min_key]
```

O(n) 操作，遍历所有条目找最小访问次数。

#### Weighted (多因子加权，默认)

**评分公式:**

$$\text{score} = 0.4 \times \text{frequency} + 0.3 \times \text{recency} + 0.3 \times \text{importance}$$

其中:
- $\text{frequency} = \frac{\text{access\_count}}{100}$ — 访问频率 (归一化)
- $\text{recency} = \frac{1}{\text{now} - \text{last\_accessed} + 1}$ — 最近访问度
- $\text{importance} = \alpha$ — 注入强度 (α 高的记忆更重要)

淘汰得分最低的条目。

**权重说明:** (0.4, 0.3, 0.3) 为启发式选择，在 ±20% 变化范围内性能稳健。

### 3.6 invalidate() — 失效指定记忆

```python
count = cache.invalidate(memory_id)
```

删除所有以 `memory_id` 开头的缓存键 (包括不同 query_hash 的变体)。

### 3.7 统计方法

#### get_stats() — 基础统计

| 指标 | 说明 |
|------|------|
| `size` | 当前缓存大小 |
| `max_size` | 最大容量 |
| `hits` | 命中次数 |
| `misses` | 未命中次数 |
| `hit_rate` | 命中率 |
| `strategy` | 淘汰策略 |

#### get_amortization_stats() — 摊销统计

| 指标 | 说明 |
|------|------|
| `compute_saved` | 节省的 K/V 计算次数 (= hits) |
| `amortization_ratio` | 摊销比率 (= hits / total) |
| `avg_reuse_count` | 平均复用次数 |
| `total_turns` | 总访问次数 |
| `efficiency_gain` | 理论效率提升 = reuse / (reuse + 1) |

## 4. 缓存键设计

```
缓存键 = "{memory_id}" 或 "{memory_id}:{query_hash}"
```

- 无 query 时: 同一记忆的 K/V 在所有查询间共享
- 有 query 时: 不同查询可能产生不同的投影结果，需要区分

`query_hash` 使用 `hash(query) % 10000000`，是简单的数值哈希。

## 5. 配置依赖

| 配置路径 | 说明 |
|---------|------|
| `config.dki.cache.max_size` | 最大缓存大小 |
| `config.dki.cache.strategy` | 淘汰策略 |
| `config.dki.cache.ttl_seconds` | 生存时间 |

## 6. 数据库交互

本模块不涉及数据库交互。所有数据存储在内存中，会话结束后自动丢弃。

## 7. 注意事项

1. **get_or_compute 的投影简化**: 当前 `get_or_compute()` 中的查询条件化投影是简化实现，计算了投影但未用投影结果重新生成 K/V
2. **query_hash 碰撞**: 使用 `hash() % 10000000` 可能产生碰撞，但对于会话级缓存 (通常 < 100 条目) 碰撞概率极低
3. **LFU 策略的 O(n) 复杂度**: 每次淘汰需要遍历所有条目。对于 max_size=100 的默认值，性能影响可忽略
4. **Weighted 策略中 frequency 的归一化**: `access_count / 100` 假设最大访问次数约为 100，超过时 frequency 分量会超过 1.0
5. **TTL 仅在 get() 时检查**: 过期条目不会主动清理，只在被访问时惰性删除。长时间不访问的过期条目会占用内存
6. **`__contains__` 使用 `any()`**: `memory_id in cache` 检查所有键是否以 memory_id 开头，是 O(n) 操作
