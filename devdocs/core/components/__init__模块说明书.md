# DKI Core Components 包初始化模块说明书

> 文件路径: `DKI/dki/core/components/__init__.py`

## 1. 模块概述

本文件是 `dki.core.components` 包的入口文件，负责统一导出所有核心组件的公共 API。通过此文件，上层模块可以直接从 `dki.core.components` 导入所需的类和工厂函数，无需关心具体子模块路径。

## 2. 导出结构

### 2.1 核心计算组件

| 导出名称 | 来源模块 | 说明 |
|---------|---------|------|
| `MemoryInfluenceScaling` | `memory_influence_scaling` | 记忆影响力缩放 (MIS)，连续 α 控制 |
| `QueryConditionedProjection` | `query_conditioned_projection` | 查询条件化投影 (FiLM 风格) |
| `DualFactorGating` | `dual_factor_gating` | 双因子门控 (不确定性 × 相关性) |
| `SessionKVCache` | `session_kv_cache` | 会话级 K/V 缓存 |
| `PositionRemapper` | `position_remapper` | 位置编码重映射 (RoPE/ALiBi 兼容) |

### 2.2 混合注入器

| 导出名称 | 来源模块 | 说明 |
|---------|---------|------|
| `HybridDKIInjector` | `hybrid_injector` | 混合注入器主类 |
| `HybridInjectionConfig` | `hybrid_injector` | 混合注入配置 |
| `HybridInjectionResult` | `hybrid_injector` | 注入结果数据结构 |
| `UserPreference` | `hybrid_injector` | 用户偏好数据结构 |
| `SessionHistory` | `hybrid_injector` | 会话历史容器 |
| `SessionMessage` | `hybrid_injector` | 单条会话消息 |
| `create_hybrid_injector` | `hybrid_injector` | 工厂函数 |

### 2.3 记忆触发器

| 导出名称 | 来源模块 | 说明 |
|---------|---------|------|
| `MemoryTrigger` | `memory_trigger` | 记忆触发器主类 |
| `MemoryTriggerConfig` | `memory_trigger` | 触发器配置 |
| `TriggerType` | `memory_trigger` | 触发类型枚举 |
| `TriggerResult` | `memory_trigger` | 触发结果 |
| `create_memory_trigger` | `memory_trigger` | 工厂函数 |

### 2.4 指代解析器

| 导出名称 | 来源模块 | 说明 |
|---------|---------|------|
| `ReferenceResolver` | `reference_resolver` | 指代解析器主类 |
| `ReferenceResolverConfig` | `reference_resolver` | 解析器配置 |
| `ReferenceType` | `reference_resolver` | 指代类型枚举 |
| `ReferenceScope` | `reference_resolver` | 指代范围枚举 |
| `ResolvedReference` | `reference_resolver` | 解析结果 |
| `Message` | `reference_resolver` | 消息数据结构 |
| `create_reference_resolver` | `reference_resolver` | 工厂函数 |

## 3. 未导出的模块

以下模块中的类**未**在 `__init__.py` 中导出，需要直接从子模块导入：

| 模块 | 未导出类 | 说明 |
|------|---------|------|
| `attention_budget` | `AttentionBudgetAnalyzer`, `BudgetAnalysis`, `LatencyBreakdown`, `LatencyTimer` | 注意力预算分析器 |
| `tiered_kv_cache` | `TieredKVCache`, `CacheTier`, `SimpleCompressor`, `KVCompressor` | 分层 K/V 缓存 |

> **注意**: 这些类在 `dki_system.py` 等上层模块中被直接引用，建议后续补充到 `__all__` 导出列表中。

## 4. 组件依赖关系

```
dki_system.py / dki_plugin.py
    ├── HybridDKIInjector (混合注入)
    │     ├── UserPreference (偏好数据)
    │     └── SessionHistory (历史数据)
    ├── DualFactorGating (门控决策)
    │     └── MemoryRouter (记忆检索)
    ├── MemoryInfluenceScaling (α 计算)
    ├── QueryConditionedProjection (投影)
    ├── SessionKVCache (会话缓存)
    ├── PositionRemapper (位置编码)
    ├── MemoryTrigger (触发检测)
    ├── ReferenceResolver (指代解析)
    ├── AttentionBudgetAnalyzer (预算分析)
    └── TieredKVCache (分层缓存)
```

## 5. 数据库交互

本模块自身不涉及数据库交互。各组件均为纯计算/逻辑组件，数据库操作由上层 `dki_system.py` 和 `dki_plugin.py` 通过 Repository 层完成。
