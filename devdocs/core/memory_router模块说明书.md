# MemoryRouter 记忆路由器模块说明书

> 源文件: `DKI/dki/core/memory_router.py`  
> 模块路径: `dki.core.memory_router`  
> 文件行数: 284 行

---

## 1. 模块概述

`MemoryRouter` 是 DKI 系统的**语义记忆检索引擎**，基于 FAISS (Facebook AI Similarity Search) 实现高效的向量相似度搜索。它负责将用户记忆索引化，并在推理时根据查询语义检索最相关的记忆片段。

---

## 2. 数据结构

### 2.1 MemorySearchResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `memory_id` | `str` | 记忆唯一标识 |
| `content` | `str` | 记忆文本内容 |
| `score` | `float` | 相似度分数 (0~1) |
| `metadata` | `Dict[str, Any]` | 附加元数据 |

### 2.2 MemoryRouter 内部状态

| 属性 | 类型 | 说明 |
|------|------|------|
| `embedding_service` | `EmbeddingService` | 嵌入服务实例 |
| `index` | `faiss.IndexFlatIP` | FAISS 内积索引 |
| `embedding_dim` | `int` | 嵌入维度 |
| `top_k` | `int` | 默认检索数量 (来自配置) |
| `similarity_threshold` | `float` | 最低相似度阈值 |
| `_memories` | `Dict[str, Dict]` | 记忆存储 {id → {content, embedding, metadata}} |
| `_id_to_idx` | `Dict[str, int]` | ID → FAISS 索引映射 |
| `_idx_to_id` | `Dict[int, str]` | FAISS 索引 → ID 映射 |
| `_needs_rebuild` | `bool` | 是否需要重建索引 |

---

## 3. 核心方法流程

### 3.1 `_init_index()` — 初始化 FAISS 索引

```
调用 _init_index()
    ├─ import faiss
    ├─ embedding_dim = embedding_service.get_dimension()
    ├─ 创建 IndexFlatIP(embedding_dim)  ← 内积索引 (用于余弦相似度)
    └─ 日志记录
```

**为什么使用 IndexFlatIP:**
- `IndexFlatIP` = Flat Index with Inner Product
- 配合 L2 归一化的向量，内积等价于余弦相似度
- Flat 索引提供精确搜索 (无近似)，适合中小规模记忆集

### 3.2 `add_memory(memory_id, content, embedding, metadata)` — 添加记忆

```
调用 add_memory(memory_id, content, embedding=None, metadata=None)
    ├─ index 为 None? → 调用 _init_index()
    ├─ embedding 为 None? → embedding = embedding_service.embed(content)
    ├─ embedding 转为 float32, reshape 为 (1, dim)
    ├─ faiss.normalize_L2(embedding)  ← L2 归一化 (确保内积 = 余弦相似度)
    ├─ idx = len(_memories)  ← 分配 FAISS 索引号
    ├─ index.add(embedding)  ← 添加到 FAISS 索引
    ├─ 存储到 _memories[memory_id]
    ├─ 更新 _id_to_idx, _idx_to_id 映射
    └─ 日志记录
```

### 3.3 `add_memories(memories)` — 批量添加

```
调用 add_memories(memories: List[Dict])
    ├─ index 为 None? → 调用 _init_index()
    ├─ 收集需要计算嵌入的文本
    │   └─ 遍历 memories, 找出 embedding 为 None 的项
    ├─ 批量嵌入: embeddings = embedding_service.embed(texts_to_embed)
    ├─ 将计算结果回填到 memories 列表
    ├─ 逐个调用 add_memory() 添加
    └─ 返回添加数量
```

### 3.4 `search(query, top_k, threshold)` — 语义搜索 ⭐核心方法

```
调用 search(query, top_k=None, threshold=None)
    ├─ 检查: index 为 None 或 _memories 为空? → 返回 []
    ├─ _needs_rebuild? → 调用 rebuild_index()
    ├─ top_k = top_k or self.top_k (配置默认值)
    ├─ threshold = threshold or self.similarity_threshold
    │
    ├─ 查询嵌入:
    │   ├─ query_embedding = embedding_service.embed(query)
    │   ├─ 转为 float32, reshape (1, dim)
    │   └─ faiss.normalize_L2(query_embedding)
    │
    ├─ FAISS 搜索:
    │   ├─ k = min(top_k, len(_memories))
    │   └─ scores, indices = index.search(query_embedding, k)
    │
    ├─ 结果过滤:
    │   ├─ 跳过 idx < 0 (FAISS 空槽位)
    │   ├─ 跳过 score < threshold (低于阈值)
    │   └─ 通过 _idx_to_id 映射获取 memory_id
    │
    └─ 返回 List[MemorySearchResult] (按分数降序)
```

### 3.5 `remove_memory(memory_id)` — 删除记忆

```
调用 remove_memory(memory_id)
    ├─ memory_id 在 _memories 中?
    │   ├─ 是 → 从 _memories 删除, 标记 _needs_rebuild = True
    │   └─ 否 → 返回 False
    └─ 返回 True
```

**注意:** FAISS 不支持高效的单条删除。删除操作仅标记为需要重建，实际重建在下次 `search()` 时触发。

### 3.6 `rebuild_index()` — 重建索引

```
调用 rebuild_index()
    ├─ _memories 为空? → 返回
    ├─ 调用 _init_index()  ← 重新创建空索引
    ├─ 保存旧记忆列表
    ├─ 清空 _memories, _id_to_idx, _idx_to_id
    ├─ 逐个重新添加所有记忆 (add_memory)
    ├─ _needs_rebuild = False
    └─ 日志记录
```

---

## 4. 关键算法

### 4.1 FAISS 内积搜索 (IndexFlatIP)

FAISS `IndexFlatIP` 使用暴力搜索计算查询向量与所有索引向量的内积:

```
score_i = query · memory_i = Σ(query_j × memory_i_j)  (j = 1..dim)
```

配合 L2 归一化 (||query|| = ||memory_i|| = 1):
```
score_i = cos(query, memory_i)
```

**时间复杂度:** O(n × d)，其中 n = 记忆数量，d = 嵌入维度

### 4.2 延迟重建策略

删除操作不立即重建索引，而是标记 `_needs_rebuild = True`，在下次搜索时才重建。这是因为:
- FAISS Flat 索引不支持原地删除
- 批量删除后一次重建比逐次重建更高效
- 搜索是读操作的热路径，删除是低频操作

---

## 5. 配置依赖

| 配置项 | 来源 | 说明 |
|--------|------|------|
| `config.rag.top_k` | rag.top_k | 默认检索数量 |
| `config.rag.similarity_threshold` | rag.similarity_threshold | 最低相似度阈值 |

---

## 6. 与其他模块的关系

| 调用方 | 使用方式 |
|--------|----------|
| `DKISystem` | 语义检索相关记忆，用于 K/V 注入 |
| `RAGSystem` | 语义检索相关记忆，拼接到 prompt |
| `DualFactorGating` | 通过 router 搜索评估相关性分数 |

---

## 7. 设计说明

- **精确搜索**: 使用 `IndexFlatIP` 而非近似索引 (如 IVF)，因为用户级记忆规模通常较小 (< 10000)
- **归一化保证**: 添加和搜索时都进行 L2 归一化，确保内积等价于余弦相似度
- **双向映射**: `_id_to_idx` 和 `_idx_to_id` 提供 O(1) 的 ID ↔ 索引转换
