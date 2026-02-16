# __init__.py 程序说明书

**模块路径**: `dki/cache/__init__.py`  
**版本**: 2.0.0  
**编写日期**: 2026-02-16  
**所属系统**: DKI (Dynamic Knowledge Injection) 缓存子系统

---

## 1. 模块概述

### 1.1 功能定位

`__init__.py` 是 DKI 缓存子系统的**包初始化模块**，负责统一导出缓存模块的所有公共 API，为上层调用方提供简洁的导入接口。

### 1.2 设计目标

- 提供统一的包级别导入入口
- 隐藏内部实现细节，只暴露公共接口
- 通过模块文档字符串说明缓存系统的整体架构

---

## 2. 模块架构

### 2.1 缓存系统三级架构

本模块在文档字符串中定义了缓存系统的三级架构：

```
┌─────────────────────────────────────────────────────┐
│                  DKI Cache System                     │
├─────────────────────────────────────────────────────┤
│  L1 (Memory)    │ 每实例热缓存     │ 延迟 < 1ms     │
│  L2 (Redis)     │ 分布式温缓存     │ 延迟 1-5ms     │
│  L3 (Recompute) │ 冷数据按需计算   │ 延迟 50-200ms  │
└─────────────────────────────────────────────────────┘
```

### 2.2 导出组件关系图

```
dki.cache (本模块)
│
├── preference_cache.py ──────────────────────────────┐
│   ├── PreferenceCacheManager  ← 分层偏好 K/V 缓存管理器 │
│   ├── CacheTierInfo           ← 缓存层级命中信息       │
│   ├── CacheConfig             ← 缓存配置数据类         │
│   └── CacheTier               ← 缓存层级枚举           │
│                                                        │
├── non_vectorized_handler.py ────────────────────────┤
│   └── NonVectorizedDataHandler ← 非向量化数据处理器    │
│                                                        │
└── redis_client.py ──────────────────────────────────┤
    ├── DKIRedisClient          ← Redis 客户端封装       │
    ├── RedisConfig             ← Redis 配置数据类       │
    ├── get_redis_client()      ← 获取全局 Redis 客户端  │
    ├── close_redis_client()    ← 关闭全局 Redis 客户端  │
    └── REDIS_AVAILABLE         ← Redis 库可用性标志     │
```

---

## 3. 导出接口清单

### 3.1 `__all__` 导出列表

| 导出名称 | 来源模块 | 类型 | 说明 |
|----------|---------|------|------|
| `PreferenceCacheManager` | `preference_cache` | 类 | 分层偏好 K/V 缓存管理器 |
| `CacheTierInfo` | `preference_cache` | 数据类 | 缓存层级命中信息 |
| `CacheConfig` | `preference_cache` | 数据类 | 缓存配置 |
| `CacheTier` | `preference_cache` | 枚举 | 缓存层级标识 (L1/L2/L3/MISS) |
| `NonVectorizedDataHandler` | `non_vectorized_handler` | 类 | 非向量化消息数据处理器 |
| `DKIRedisClient` | `redis_client` | 类 | 异步 Redis 客户端 |
| `RedisConfig` | `redis_client` | 数据类 | Redis 连接配置 |
| `get_redis_client` | `redis_client` | 异步函数 | 获取/创建全局 Redis 单例 |
| `close_redis_client` | `redis_client` | 异步函数 | 关闭全局 Redis 连接 |
| `REDIS_AVAILABLE` | `redis_client` | 布尔常量 | Redis 库是否已安装 |

---

## 4. 使用方式

### 4.1 标准导入

```python
# 推荐: 从包级别导入
from dki.cache import PreferenceCacheManager, CacheConfig
from dki.cache import DKIRedisClient, RedisConfig
from dki.cache import NonVectorizedDataHandler

# 也可以直接导入子模块
from dki.cache.preference_cache import PreferenceCacheManager
```

### 4.2 Redis 可用性检查

```python
from dki.cache import REDIS_AVAILABLE

if REDIS_AVAILABLE:
    # Redis 库已安装，可以使用分布式缓存
    from dki.cache import get_redis_client
    client = await get_redis_client(config)
else:
    # Redis 库未安装，仅使用 L1 内存缓存
    pass
```

---

## 5. 关键设计说明

### 5.1 Redis 对缓存命中率的影响

模块文档中特别说明了 Redis 在多实例部署中的重要性：

| 部署模式 | 缓存命中率 | 说明 |
|---------|-----------|------|
| 单实例 (无 Redis) | ~70% | L1 内存缓存即可满足 |
| N 实例 (无 Redis) | ~70%/N | 每个实例独立缓存，命中率线性下降 |
| N 实例 (有 Redis) | ~70% | Redis 共享缓存，命中率保持恒定 |

### 5.2 模块间依赖关系

```
__init__.py
    ├── import preference_cache  (无外部依赖)
    ├── import non_vectorized_handler  (依赖 dki.adapters.base.ChatMessage)
    └── import redis_client  (可选依赖 redis.asyncio)
```

---

## 6. 流程说明

本模块为纯导入模块，无业务逻辑流程。Python 解释器在首次 `import dki.cache` 时执行以下步骤：

```
1. 执行 __init__.py
   │
   ├── 2. 导入 preference_cache 模块
   │      └── 加载 PreferenceCacheManager, CacheTierInfo, CacheConfig, CacheTier
   │
   ├── 3. 导入 non_vectorized_handler 模块
   │      └── 加载 NonVectorizedDataHandler
   │
   ├── 4. 导入 redis_client 模块
   │      ├── 尝试 import redis.asyncio
   │      ├── 成功 → REDIS_AVAILABLE = True
   │      └── 失败 → REDIS_AVAILABLE = False (优雅降级)
   │
   └── 5. 定义 __all__ 导出列表
```
