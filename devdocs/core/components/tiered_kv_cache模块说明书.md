# Tiered KV Cache 模块说明书

> 文件路径: `DKI/dki/core/components/tiered_kv_cache.py`

## 1. 模块概述

本模块实现了 DKI 论文 Section 7.4 中描述的**分层 K/V 缓存 (Tiered KV Cache)**，是一个可选的生产环境性能优化组件。它模拟计算机存储层次结构，将 K/V 缓存按访问频率分布在不同存储介质上。

### 重要说明

> **这是可选增强组件**。DKI 在简单缓存下即可正确运行；分层缓存是性能优化，不是功能需求。系统在 L2/L3/L4 禁用时仍保持正确性。

### 存储层次架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DKI Memory Hierarchy                      │
├─────────────────────────────────────────────────────────────┤
│  L1: GPU HBM (Hot)                                          │
│  ├── Top-k 最近使用的记忆                                    │
│  ├── 未压缩 FP16                                            │
│  └── 容量: 5-10 memories/session                            │
│                                                              │
│  L2: CPU RAM (Warm)                                          │
│  ├── 会话活跃记忆                                            │
│  ├── 压缩 (2-4×)                                            │
│  └── 容量: 50-100 memories/session                          │
│                                                              │
│  L3: NVMe SSD (Cold)                                         │
│  ├── 所有会话记忆                                            │
│  ├── 量化 INT8 + 压缩 (8×)                                  │
│  └── 容量: 无限制                                            │
│                                                              │
│  L4: Recompute on Demand                                     │
│  ├── 仅存储文本 + 路由向量                                   │
│  └── 需要时重新计算 K/V                                      │
└─────────────────────────────────────────────────────────────┘
```

### 关键洞察

DKI 的内存占用与**活跃记忆数**成正比，而非总语料库大小。这使得 DKI 在大语料库 + 稀疏相关性场景下更高效。

### 误差组合说明

不假设压缩技术间的误差可加性。激进压缩仅应用于 α 已经很低的冷记忆。

## 2. 数据结构

### 2.1 CacheTier (枚举)

| 层级 | 值 | 存储介质 | 压缩 |
|------|-----|---------|------|
| `L1_GPU` | `"l1_gpu"` | GPU HBM | 无 (FP16) |
| `L2_CPU` | `"l2_cpu"` | CPU RAM | pickle (2×) |
| `L3_SSD` | `"l3_ssd"` | SSD 文件 | INT8 量化 + pickle (8×) |
| `L4_RECOMPUTE` | `"l4_recompute"` | 仅文本 | 按需重算 |

### 2.2 TieredCacheMetadata

分层缓存元数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `memory_id` | `str` | 记忆标识符 |
| `tier` | `CacheTier` | 当前层级 |
| `created_at` | `float` | 创建时间 |
| `last_accessed` | `float` | 最后访问时间 |
| `access_count` | `int` | 访问次数 |
| `alpha` | `float` | α 值 |
| `size_bytes` | `int` | 数据大小 (字节) |
| `compressed` | `bool` | 是否已压缩 |
| `quantized` | `bool` | 是否已量化 |

**score() 方法 — 重要性评分:**

$$\text{score} = 0.4 \times \min\left(\frac{\text{access\_count}}{100}, 1\right) + 0.3 \times \frac{1}{\text{now} - \text{last\_accessed} + 1} + 0.3 \times \alpha$$

用于淘汰和降级决策。

### 2.3 TieredCacheEntry

缓存条目。

| 字段 | 类型 | 说明 |
|------|------|------|
| `kv_entries` | `Optional[List[KVCacheEntry]]` | K/V 数据 (L4 为 None) |
| `metadata` | `TieredCacheMetadata` | 元数据 |
| `compressed_data` | `Optional[bytes]` | 压缩数据 (L2/L3) |
| `text_content` | `Optional[str]` | 文本内容 (L4) |

## 3. 压缩器

### 3.1 KVCompressor (抽象基类)

定义压缩/解压接口：
- `compress(kv_entries) → bytes`
- `decompress(data) → List[KVCacheEntry]`
- `get_compression_ratio() → float`

### 3.2 SimpleCompressor

简单压缩实现，支持两种模式：

#### 非量化模式 (L2 使用)

```
压缩: tensor.half().cpu() → pickle.dumps()
解压: pickle.loads() → tensor
压缩比: ~2× (FP32 → FP16)
```

#### 量化模式 (L3 使用)

```
压缩:
  scale = tensor.abs().max()
  quantized = (tensor / scale × 127).to(int8)
  存储: {key_int8, value_int8, scale, layer_idx}
  → pickle.dumps()

解压:
  tensor = (int8_tensor.float() / 127 × scale)
  
压缩比: ~4× (FP16 → INT8)
```

**INT8 量化算法:**

$$q = \text{round}\left(\frac{x}{\max(|x|)} \times 127\right) \in [-127, 127]$$

$$\hat{x} = \frac{q}{127} \times \max(|x|)$$

> 生产环境建议集成 GEAR [14] 实现 4× 压缩且近乎无损。

## 4. 核心类: TieredKVCache

### 4.1 初始化

```python
cache = TieredKVCache(
    l1_max_size=10,        # L1 最大条目数
    l2_max_size=100,       # L2 最大条目数
    l3_path="./data/kv_cache",  # L3 存储路径
    enable_l3=True,        # 启用 L3
    enable_l4=True,        # 启用 L4
    ttl_seconds=3600,      # 生存时间
    compressor=None,       # 自定义 L2 压缩器
)
```

### 4.2 get() — 分层查找

**完整流程:**

```
输入: memory_id, query?, model? (L4 重算用)
  │
  ├── L1: GPU HBM (最快)
  │   ├── 命中 → 更新元数据 → 返回 (entries, L1_GPU)
  │   └── 过期 → 删除 → 继续
  │
  ├── L2: CPU RAM (压缩)
  │   ├── 命中 → 解压 → 提升到 L1 → 返回 (entries, L2_CPU)
  │   └── 过期 → 删除 → 继续
  │
  ├── L3: SSD (量化+压缩)
  │   ├── 文件存在 → 读取 → 解压 → 提升到 L2 → 返回 (entries, L3_SSD)
  │   └── 文件不存在/读取失败 → 继续
  │
  ├── L4: Recompute (文本重算)
  │   ├── 有文本 + 有模型 → 重算 K/V → 存入 L2 → 返回 (entries, L4_RECOMPUTE)
  │   └── 无文本/无模型 → 继续
  │
  └── 全部未命中 → 返回 (None, L4_RECOMPUTE)
```

**数据流向 (查找时的提升):**

```
L3 → L2 → L1 (逐级提升)
L4 → L2 (重算后存入 L2)
L2 → L1 (解压后提升)
```

### 4.3 put() — 存储

**流程:**

```
输入: memory_id, kv_entries, query?, alpha, text_content?
  │
  ├── 1. 存储文本到 L4 (如启用且有文本)
  │     _l4_text_store[memory_id] = text_content
  │
  ├── 2. 尝试存入 L1
  │     ├── L1 有空间 → 直接存入 → 返回 L1_GPU
  │     └── L1 已满 → 降级最不重要的条目到 L2
  │         → 存入 L1 → 返回 L1_GPU
  │
  └── 新数据始终先进入 L1 (最热)
```

### 4.4 提升与降级

#### _promote_to_l1() — 提升到 L1

```
1. 确保 L1 有空间 (必要时降级)
2. 将张量移到 GPU (如可用)
   entry.key.cuda() / entry.value.cuda()
3. 更新元数据: tier=L1, compressed=False
4. 存入 _l1_cache
```

#### _promote_to_l2() — 提升到 L2

```
1. 确保 L2 有空间 (必要时降级到 L3)
2. 压缩 K/V: l2_compressor.compress(kv_entries)
3. 更新元数据: tier=L2, compressed=True
4. 存入 _l2_cache (仅存压缩数据，不存原始 K/V)
```

#### _demote_from_l1() — 从 L1 降级到 L2

```
1. 找到 L1 中 score 最低的条目
2. 从 L1 移除
3. 压缩 → 存入 L2
4. 如 L2 已满 → 触发 L2 降级
```

#### _demote_from_l2() — 从 L2 降级到 L3

```
1. 找到 L2 中 score 最低的条目
2. 从 L2 移除
3. 解压 L2 数据 → 用 L3 压缩器重新压缩 (INT8 量化)
4. 写入 SSD 文件: {l3_path}/{cache_key}.kv
```

### 4.5 完整数据流

```
新数据 ──put──→ L1 (GPU, FP16)
                 │
            降级 (score 最低)
                 ↓
                L2 (CPU, pickle 压缩)
                 │
            降级 (score 最低)
                 ↓
                L3 (SSD, INT8 量化 + pickle)
                 
L4 (文本) ──重算──→ L2 ──提升──→ L1

查找路径: L1 → L2 → L3 → L4 → None
提升路径: L3 → L2 → L1 (命中时逐级提升)
```

### 4.6 invalidate() — 失效

删除指定记忆在所有层级的缓存：
- L1: 删除内存条目
- L2: 删除内存条目
- L3: 删除 SSD 文件
- L4: 删除文本存储

### 4.7 统计方法

#### get_stats() — 缓存统计

| 指标 | 说明 |
|------|------|
| `l1_size` / `l1_max_size` | L1 当前/最大大小 |
| `l1_hit_rate` | L1 命中率 |
| `l2_size` / `l2_max_size` | L2 当前/最大大小 |
| `l2_hit_rate` | L2 命中率 |
| `l3_hit_rate` | L3 命中率 |
| `l4_text_count` | L4 文本存储数 |
| `l4_recomputes` | L4 重算次数 |
| `promotions` | 提升次数 |
| `demotions` | 降级次数 |

#### get_memory_footprint() — 内存占用

| 指标 | 说明 |
|------|------|
| `l1_gpu_bytes` | L1 GPU 内存占用 |
| `l2_cpu_bytes` | L2 CPU 内存占用 |
| `l3_ssd_bytes` | L3 SSD 存储占用 |
| `total_bytes` | 总占用 |

## 5. 配置依赖

通过 `ConfigLoader` 加载配置 (当前未直接使用配置值，参数通过构造函数传入)。

## 6. 数据库交互

本模块不涉及数据库交互。L3 层使用本地文件系统存储。

**L3 文件格式:**
- 路径: `{l3_path}/{cache_key}.kv`
- 内容: pickle 序列化的 INT8 量化 K/V 数据
- 清理: `clear()` 或 `invalidate()` 时删除

## 7. 注意事项

1. **L2 降级到 L3 的双重压缩**: 从 L2 降级时，先解压 L2 数据 (FP16)，再用 L3 压缩器重新压缩 (INT8)。这确保 L3 始终使用最高压缩比
2. **GPU 可用性检查**: `_promote_to_l1()` 使用 `torch.cuda.is_available()` 检查 GPU，无 GPU 时张量保持在 CPU
3. **L3 文件命名**: 使用 `cache_key` 作为文件名，如果 cache_key 包含特殊字符 (如 `:`) 可能在某些文件系统上出问题
4. **L4 文本存储不持久化**: `_l4_text_store` 是内存字典，进程重启后丢失
5. **并发安全**: 当前实现不是线程安全的，多线程环境需要外部加锁
6. **L3 miss 统计**: 即使 L3 未启用 (`enable_l3=False`)，`l3_misses` 仍会递增
7. **SimpleCompressor 的量化精度**: INT8 量化使用全局 max 缩放，对于数值分布不均匀的张量可能损失较大。生产环境建议使用分组量化或 GEAR
