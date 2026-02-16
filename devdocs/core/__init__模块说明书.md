# DKI Core 模块初始化说明书

> 源文件: `DKI/dki/core/__init__.py`  
> 模块路径: `dki.core`  
> 文件行数: 24 行

---

## 1. 模块概述

`__init__.py` 是 `dki.core` 包的入口文件，负责统一导出核心模块的公共 API。该文件通过 `__all__` 列表定义了 8 个对外暴露的核心类，构成 DKI 系统的核心层接口。

---

## 2. 导出清单

| 序号 | 导出名称 | 来源模块 | 说明 |
|------|----------|----------|------|
| 1 | `DKISystem` | `dki.core.dki_system` | DKI 核心系统，实现 K/V 注入的完整流程 |
| 2 | `RAGSystem` | `dki.core.rag_system` | RAG 基线系统，用于对比实验 |
| 3 | `MemoryRouter` | `dki.core.memory_router` | FAISS 语义检索路由器 |
| 4 | `EmbeddingService` | `dki.core.embedding_service` | 文本嵌入服务 (sentence-transformers) |
| 5 | `DKIPlugin` | `dki.core.plugin_interface` | DKI 插件默认实现 |
| 6 | `DKIPluginInterface` | `dki.core.plugin_interface` | DKI 插件抽象接口 (ABC) |
| 7 | `DKIPluginConfig` | `dki.core.plugin_interface` | DKI 插件配置数据类 |
| 8 | `DKIMiddleware` | `dki.core.plugin_interface` | FastAPI 中间件集成 |

---

## 3. 模块依赖关系

```
dki.core
├── dki_system.py        → DKISystem (核心系统)
│   ├── memory_router.py → MemoryRouter
│   ├── embedding_service.py → EmbeddingService
│   └── components/      → MIS, QCP, Gating, Cache, HybridInjector 等
├── rag_system.py        → RAGSystem (基线对比)
│   ├── memory_router.py → MemoryRouter
│   └── embedding_service.py → EmbeddingService
├── plugin_interface.py  → DKIPlugin, DKIPluginInterface, DKIPluginConfig, DKIMiddleware
├── memory_router.py     → MemoryRouter (FAISS 检索)
├── embedding_service.py → EmbeddingService (向量化)
├── architecture.py      → 架构文档 (纯注释)
├── injection/           → 注入策略实现
│   ├── full_attention_injector.py → FullAttentionInjector
│   └── engram_inspired_injector.py → EngramInspiredFullAttentionInjector
└── components/          → 核心组件
    ├── memory_influence_scaling.py → MIS (α 缩放)
    ├── query_conditioned_projection.py → QCP (FiLM 投影)
    ├── dual_factor_gating.py → 双因子门控
    ├── session_kv_cache.py → 会话 KV 缓存
    ├── tiered_kv_cache.py → 分层 KV 缓存
    ├── position_remapper.py → 位置重映射
    ├── attention_budget.py → 注意力预算分析
    ├── hybrid_injector.py → 杂化注入器
    ├── memory_trigger.py → 记忆触发器
    └── reference_resolver.py → 指代解析器
```

---

## 4. 使用方式

```python
# 方式 1: 直接导入核心系统
from dki.core import DKISystem, RAGSystem

# 方式 2: 导入插件接口
from dki.core import DKIPlugin, DKIPluginConfig

# 方式 3: 导入基础组件
from dki.core import MemoryRouter, EmbeddingService
```

---

## 5. 设计说明

- **统一入口**: 所有核心类通过 `dki.core` 统一导出，外部使用者无需了解内部模块结构
- **最小暴露**: 仅导出 8 个核心类，内部组件 (如 MIS、QCP、Gating) 不在此层导出
- **内部组件**: `components/` 和 `injection/` 子包的类由核心系统内部使用，不对外暴露
