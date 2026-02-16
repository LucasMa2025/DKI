# EmbeddingService 嵌入服务模块说明书

> 源文件: `DKI/dki/core/embedding_service.py`  
> 模块路径: `dki.core.embedding_service`  
> 文件行数: 200 行

---

## 1. 模块概述

`EmbeddingService` 提供基于 `sentence-transformers` 的文本嵌入服务，是 DKI 系统中语义检索的基础组件。采用**单例模式**确保全局只有一个模型实例，支持单文本嵌入、批量嵌入和相似度计算。

---

## 2. 类定义

### 2.1 EmbeddingService (单例)

| 属性 | 类型 | 说明 |
|------|------|------|
| `_instance` | `Optional[EmbeddingService]` | 类级别单例引用 |
| `model_name` | `str` | 模型名称 (来自配置) |
| `device` | `str` | 运行设备 (cpu/cuda) |
| `normalize` | `bool` | 是否 L2 归一化 (默认 True) |
| `batch_size` | `int` | 批量编码大小 |
| `embedding_dim` | `int` | 嵌入维度 |
| `model` | `Optional[SentenceTransformer]` | 模型实例 (延迟加载) |

---

## 3. 核心方法流程

### 3.1 `__new__()` — 单例创建

```
调用 __new__()
    ├─ cls._instance 为 None?
    │   ├─ 是 → 创建新实例, 设置 _initialized = False
    │   └─ 否 → 返回已有实例
    └─ 返回 cls._instance
```

### 3.2 `load()` — 模型加载

```
调用 load()
    ├─ self.model 不为 None? → 直接返回 (已加载)
    ├─ import SentenceTransformer
    ├─ 加载模型: SentenceTransformer(model_name, device=device)
    ├─ 更新 embedding_dim = model.get_sentence_embedding_dimension()
    └─ 日志记录
```

### 3.3 `embed(text)` — 文本嵌入

```
调用 embed(text)
    ├─ model 为 None? → 调用 load()
    ├─ 判断输入类型: str → 单文本, List[str] → 批量
    ├─ 调用 model.encode(texts, batch_size, normalize_embeddings, show_progress_bar=False)
    ├─ 单文本 → 返回 embeddings[0] (shape: [dim])
    └─ 批量 → 返回 embeddings (shape: [batch, dim])
```

### 3.4 `similarity(query_embedding, doc_embeddings)` — 相似度计算

**算法说明:**

- **已归一化** (`normalize=True`): 直接点积
  ```
  similarity = doc_embeddings · query_embedding
  ```
  因为 L2 归一化后，点积等价于余弦相似度。

- **未归一化** (`normalize=False`): 标准余弦相似度
  ```
  similarity = (doc_embeddings · query_embedding) / (||doc_embeddings|| × ||query_embedding|| + ε)
  ```
  其中 ε = 1e-9 防止除零。

### 3.5 `most_similar(query, documents, top_k)` — Top-K 检索

```
调用 most_similar(query, documents, top_k=5)
    ├─ query_emb = embed(query)
    ├─ doc_embs = embed(documents)
    ├─ scores = similarity(query_emb, doc_embs)
    ├─ top_indices = argsort(scores)[::-1][:top_k]  (降序取前 K)
    └─ 返回 [(index, score, document), ...]
```

### 3.6 `unload()` / `reset_instance()` — 资源释放

```
unload():
    ├─ del self.model
    ├─ self.model = None
    └─ torch.cuda.empty_cache()  (释放 GPU 显存)

reset_instance():
    ├─ 调用 unload()
    ├─ _initialized = False
    └─ _instance = None  (允许重新创建)
```

---

## 4. 关键算法

### 4.1 余弦相似度

对于两个向量 a 和 b:
```
cos_sim(a, b) = (a · b) / (||a|| × ||b||)
```

当向量已 L2 归一化 (||a|| = ||b|| = 1) 时:
```
cos_sim(a, b) = a · b  (简化为点积)
```

### 4.2 Top-K 检索

使用 `numpy.argsort` 进行全量排序后取前 K 个。对于小规模文档集 (< 10000) 这是高效的；大规模检索由 `MemoryRouter` 的 FAISS 索引处理。

---

## 5. 配置依赖

从 `ConfigLoader().config` 读取:

| 配置项 | 来源 | 说明 |
|--------|------|------|
| `config.embedding.model_name` | embedding.model_name | 模型名称 |
| `config.embedding.device` | embedding.device | 运行设备 |
| `config.embedding.batch_size` | embedding.batch_size | 批量大小 |
| `config.memory.embedding_dim` | memory.embedding_dim | 嵌入维度 |

---

## 6. 设计说明

- **单例模式**: 避免重复加载大型嵌入模型，节省内存
- **延迟加载**: 模型在首次调用 `embed()` 时才加载，加速系统启动
- **归一化优化**: 默认 L2 归一化，使相似度计算简化为点积运算
