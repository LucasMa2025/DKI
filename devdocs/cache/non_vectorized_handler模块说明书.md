# non_vectorized_handler.py 程序说明书

**模块路径**: `dki/cache/non_vectorized_handler.py`  
**版本**: 1.0.0  
**编写日期**: 2026-02-16  
**所属系统**: DKI (Dynamic Knowledge Injection) 缓存子系统

---

## 1. 模块概述

### 1.1 功能定位

`non_vectorized_handler.py` 实现了 DKI 系统的**非向量化消息数据处理器**，负责对没有预计算 embedding 向量的聊天消息进行相关性搜索。该模块提供三种搜索策略（Lazy、Batch、Hybrid），根据消息数据量自动选择最优策略，并通过内置的 Embedding 缓存避免重复计算。

### 1.2 核心能力

- **三种搜索策略**: Lazy (实时计算)、Batch (批量预计算)、Hybrid (BM25 初筛 + Embedding 精排)
- **自动策略选择**: 根据消息数量自动选择最优搜索策略
- **Embedding 缓存**: 基于 LRU 的 Embedding 向量缓存，避免重复计算
- **BM25 文本检索**: 基于 BM25Okapi 算法的快速文本初筛
- **余弦相似度排序**: 基于向量余弦相似度的精确排序
- **批量预计算**: 支持大规模消息的批量 Embedding 预计算

### 1.3 外部依赖

| 依赖 | 用途 | 必需 |
|------|------|------|
| `loguru` | 日志记录 | 是 |
| `dki.adapters.base.ChatMessage` | 聊天消息数据结构 | 是 |
| `rank_bm25` (BM25Okapi) | BM25 文本检索算法 | 否 (可选，Hybrid 策略需要) |
| `EmbeddingService` (外部传入) | Embedding 向量计算服务 | 是 |

---

## 2. 数据结构定义

### 2.1 枚举类型

#### SearchStrategy (搜索策略枚举)

```python
class SearchStrategy(str, Enum):
    LAZY   = "lazy"    # 实时 Embedding 计算
    BATCH  = "batch"   # 批量预计算 Embedding
    HYBRID = "hybrid"  # BM25 初筛 + Embedding 精排
```

### 2.2 数据类

#### HandlerConfig (处理器配置)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **策略选择** | | | |
| `default_strategy` | `SearchStrategy` | `HYBRID` | 默认搜索策略 |
| **自动策略阈值** | | | |
| `lazy_max_messages` | `int` | `100` | 消息数 ≤ 100 时使用 Lazy 策略 |
| `batch_trigger_count` | `int` | `1000` | 消息数 ≥ 1000 时使用 Batch 策略 |
| **BM25 设置** | | | |
| `bm25_candidates_multiplier` | `int` | `4` | BM25 候选数 = top_k × 此倍数 |
| `bm25_min_candidates` | `int` | `20` | BM25 最少候选数 |
| **Embedding 缓存** | | | |
| `cache_embeddings` | `bool` | `True` | 是否缓存 Embedding |
| `cache_max_size` | `int` | `100000` | 最大缓存 Embedding 数量 |
| `cache_ttl_hours` | `int` | `168` | 缓存 TTL (7天) |
| **批量计算** | | | |
| `batch_size` | `int` | `100` | 每批计算的消息数 |
| `max_concurrent_batches` | `int` | `4` | 最大并发批次数 |

#### SearchResult (搜索结果)

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | `ChatMessage` | 匹配的聊天消息 |
| `score` | `float` | 相关性分数 (0.0 ~ 1.0) |
| `strategy_used` | `SearchStrategy` | 使用的搜索策略 |
| `metadata` | `Dict[str, Any]` | 附加元数据 |

### 2.3 关联外部数据结构

#### ChatMessage (来自 `dki/adapters/base.py`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `message_id` | `str` | 消息唯一标识 |
| `session_id` | `str` | 会话标识 |
| `user_id` | `str` | 用户标识 |
| `role` | `str` | 角色 ("user" / "assistant" / "system") |
| `content` | `str` | 消息内容 |
| `timestamp` | `datetime` | 消息时间戳 |
| `embedding` | `Optional[List[float]]` | 预计算的 Embedding 向量 (可选) |
| `metadata` | `Dict[str, Any]` | 消息元数据 |
| `token_count` | `Optional[int]` | Token 数量 |
| `parent_id` | `Optional[str]` | 父消息 ID |

**关键方法**:
- `content_hash() → str`: 返回消息内容的 MD5 哈希值，用作缓存键

#### 关联数据库表

ChatMessage 数据来源于上层应用的 **messages 表**:

| 字段 | 数据库列 | 类型 | 说明 |
|------|---------|------|------|
| `message_id` | `message_id` | VARCHAR/UUID | 主键 |
| `session_id` | `session_id` | VARCHAR/UUID | 会话外键 |
| `user_id` | `user_id` | VARCHAR | 用户标识 |
| `role` | `role` | VARCHAR | 消息角色 |
| `content` | `content` | TEXT | 消息内容 |
| `timestamp` | `timestamp` | TIMESTAMP | 创建时间 |
| `embedding` | `embedding` | VECTOR/BLOB | 向量数据 (可选) |
| `metadata` | `metadata` | JSON | 元数据 |
| `token_count` | `token_count` | INTEGER | Token 数 |
| `parent_id` | `parent_id` | VARCHAR/UUID | 父消息引用 |

---

## 3. 类设计

### 3.1 EmbeddingCache 类

#### 3.1.1 设计说明

基于 `OrderedDict` 实现的 **LRU Embedding 向量缓存**，所有核心操作均为 O(1) 时间复杂度。用于缓存已计算的 Embedding 向量，避免对相同消息重复计算。

#### 3.1.2 内部数据结构

```
EmbeddingCache
├── max_size: int                                    # 最大容量 (默认 100,000)
├── _cache: OrderedDict[str, List[float]]            # 有序字典 (核心存储)
│   └── key: message_id 或 content_hash
│       value: Embedding 向量 (浮点数列表)
└── _lock: asyncio.Lock                              # 异步锁 (并发安全)
```

#### 3.1.3 方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `get` | `async (key: str) → Optional[List[float]]` | 获取缓存的 Embedding，更新访问顺序 |
| `put` | `async (key: str, embedding: List[float]) → None` | 存入 Embedding，满时淘汰最旧 |
| `clear` | `async () → None` | 清空所有缓存 |

#### 3.1.4 缓存键策略

```
缓存键 = msg.message_id or msg.content_hash()

优先级:
  1. message_id (如果存在) → 使用消息唯一标识
  2. content_hash() → 使用内容 MD5 哈希 (message_id 为 None 时)

目的: 确保相同内容的消息共享 Embedding 缓存
```

### 3.2 NonVectorizedDataHandler 类

#### 3.2.1 设计说明

**非向量化消息搜索处理器**，提供多种策略对没有预计算 Embedding 的消息进行相关性搜索。

#### 3.2.2 内部数据结构

```
NonVectorizedDataHandler
├── embedding_service: Any                    # Embedding 计算服务
├── config: HandlerConfig                     # 处理器配置
├── _embedding_cache: EmbeddingCache          # Embedding 缓存
├── _bm25_index: None                         # BM25 索引 (懒初始化)
├── _bm25_messages: List[ChatMessage]         # BM25 索引对应的消息列表
└── _stats: Dict                              # 统计信息
    ├── lazy_searches: int                    # Lazy 搜索次数
    ├── batch_searches: int                   # Batch 搜索次数
    ├── hybrid_searches: int                  # Hybrid 搜索次数
    ├── cache_hits: int                       # 缓存命中次数
    └── embeddings_computed: int              # Embedding 计算次数
```

#### 3.2.3 公共方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `search_relevant_messages` | `async (messages, query, top_k, strategy) → List[SearchResult]` | 核心方法：搜索相关消息 |
| `batch_precompute_embeddings` | `async (messages, progress_callback) → int` | 批量预计算 Embedding |
| `clear_cache` | `async () → None` | 清空 Embedding 缓存 |
| `get_stats` | `() → Dict[str, Any]` | 获取统计信息 |

#### 3.2.4 私有方法清单

| 方法 | 签名 | 说明 |
|------|------|------|
| `_select_strategy` | `(message_count) → SearchStrategy` | 自动选择搜索策略 |
| `_lazy_search` | `async (messages, query, top_k) → List[SearchResult]` | Lazy 策略搜索 |
| `_batch_search` | `async (messages, query, top_k) → List[SearchResult]` | Batch 策略搜索 |
| `_hybrid_search` | `async (messages, query, top_k) → List[SearchResult]` | Hybrid 策略搜索 |
| `_bm25_filter` | `async (messages, query, top_n) → List[ChatMessage]` | BM25 初筛 |
| `_tokenize` | `(text) → List[str]` | 简单分词 |
| `_cosine_similarity` | `(vec1, vec2) → float` | 余弦相似度计算 |

---

## 4. 核心流程说明

### 4.1 搜索相关消息 (search_relevant_messages) — 主流程

```
search_relevant_messages(messages, query, top_k=5, strategy=None)
│
├── 1. 空消息列表检查 → 返回 []
│
├── 2. 策略选择:
│      ├── strategy 已指定 → 使用指定策略
│      └── strategy = None → 自动选择:
│          ├── len(messages) ≤ 100   → LAZY
│          ├── len(messages) ≥ 1000  → BATCH
│          └── 100 < len < 1000     → HYBRID
│
├── 3. 执行搜索:
│      ├── LAZY   → _lazy_search(messages, query, top_k)
│      ├── BATCH  → _batch_search(messages, query, top_k)
│      └── HYBRID → _hybrid_search(messages, query, top_k)
│
└── 返回: List[SearchResult] (按相关性降序排列)
```

### 4.2 自动策略选择算法 (_select_strategy)

```
消息数量区间与策略映射:

  0 ──── 100 ──── 1000 ──── ∞
  │  LAZY  │  HYBRID  │  BATCH  │
  
  ≤ 100:   LAZY   — 数据量小，直接实时计算所有 Embedding
  101~999: HYBRID — 中等数据量，BM25 初筛减少计算量
  ≥ 1000:  BATCH  — 大数据量，先批量预计算再搜索
```

### 4.3 Lazy 搜索流程 (_lazy_search)

**适用场景**: 消息数 ≤ 100，实时计算所有消息的 Embedding

```
_lazy_search(messages, query, top_k)
│
├── 1. 计算查询 Embedding:
│      query_embedding = embedding_service.embed(query)
│
├── 2. 遍历所有消息，计算相似度:
│      for msg in messages:
│      │
│      ├── 2a. 生成缓存键: cache_key = msg.message_id or msg.content_hash()
│      │
│      ├── 2b. 查缓存: cached = await _embedding_cache.get(cache_key)
│      │   ├── 命中 → 使用缓存的 Embedding (cache_hits++)
│      │   └── 未命中 → 计算 Embedding:
│      │       ├── msg_embedding = embedding_service.embed(msg.content)
│      │       ├── embeddings_computed++
│      │       └── 写入缓存 (如果 cache_embeddings=True)
│      │
│      └── 2c. 计算余弦相似度:
│              score = cosine_similarity(query_embedding, msg_embedding)
│              scored_messages.append((msg, score))
│
├── 3. 按分数降序排序:
│      scored_messages.sort(key=score, reverse=True)
│
└── 4. 返回 top_k 个 SearchResult
```

### 4.4 Batch 搜索流程 (_batch_search)

**适用场景**: 消息数 ≥ 1000，先批量预计算缺失的 Embedding

```
_batch_search(messages, query, top_k)
│
├── 1. 找出缺失 Embedding 的消息:
│      messages_without_embeddings = [
│          msg for msg in messages
│          if msg.embedding is None
│          and (msg.message_id or msg.content_hash()) not in _embedding_cache
│      ]
│
├── 2. 批量预计算缺失的 Embedding:
│      if messages_without_embeddings:
│          await batch_precompute_embeddings(messages_without_embeddings)
│
└── 3. 调用 Lazy 搜索 (此时所有 Embedding 已缓存):
       return await _lazy_search(messages, query, top_k)
```

### 4.5 Hybrid 搜索流程 (_hybrid_search)

**适用场景**: 消息数 100~1000，BM25 快速初筛 + Embedding 精确排序

```
_hybrid_search(messages, query, top_k)
│
├── 1. 计算 BM25 候选数量:
│      bm25_candidates = max(
│          bm25_min_candidates,        # 默认 20
│          top_k × bm25_candidates_multiplier  # 默认 top_k × 4
│      )
│      例: top_k=5 → candidates = max(20, 5×4) = 20
│
├── 2. BM25 初筛:
│      candidates = await _bm25_filter(messages, query, bm25_candidates)
│      │
│      ├── 2a. 分词: tokenize(msg.content) for each msg
│      ├── 2b. 构建 BM25 索引: BM25Okapi(tokenized_messages)
│      ├── 2c. 查询评分: scores = bm25.get_scores(tokenize(query))
│      └── 2d. 取 top-N 候选消息
│
├── 3. 空候选检查 → 返回 []
│
└── 4. Embedding 精排:
       return await _lazy_search(candidates, query, top_k)
       (仅对 BM25 筛选出的候选消息计算 Embedding)
```

**Hybrid 策略的优势**:
```
假设: 500 条消息, top_k=5

纯 Lazy:  计算 500 个 Embedding → 500 次 embed() 调用
Hybrid:   BM25 筛选 20 候选 → 计算 20 个 Embedding → 20 次 embed() 调用
          
计算量减少: (500-20)/500 = 96%
```

### 4.6 BM25 初筛流程 (_bm25_filter)

```
_bm25_filter(messages, query, top_n)
│
├── 1. 检查 rank_bm25 库:
│      ├── 已安装 → 继续
│      └── 未安装 → 降级: 返回 messages[:top_n] (取前 N 条)
│
├── 2. 分词:
│      tokenized_messages = [tokenize(msg.content) for msg in messages]
│      query_tokens = tokenize(query)
│
├── 3. 构建 BM25 索引:
│      bm25 = BM25Okapi(tokenized_messages)
│
├── 4. 计算 BM25 分数:
│      scores = bm25.get_scores(query_tokens)
│
├── 5. 排序取 top-N:
│      scored_indices = sorted(range(len(scores)),
│                              key=lambda i: scores[i],
│                              reverse=True)[:top_n]
│
└── 返回: [messages[i] for i in scored_indices]
```

### 4.7 批量预计算 Embedding 流程 (batch_precompute_embeddings)

```
batch_precompute_embeddings(messages, progress_callback=None)
│
├── 遍历消息 (按 batch_size 分批, 默认 100):
│   │
│   ├── batch = messages[i : i+batch_size]
│   ├── texts = [msg.content for msg in batch]
│   │
│   ├── 计算 Embedding:
│   │   ├── embedding_service 有 embed_batch 方法?
│   │   │   ├── YES → embeddings = embed_batch(texts)  # 批量计算
│   │   │   └── NO  → embeddings = [embed(t) for t in texts]  # 逐条回退
│   │
│   ├── 缓存 Embedding:
│   │   for msg, embedding in zip(batch, embeddings):
│   │       cache_key = msg.message_id or msg.content_hash()
│   │       await _embedding_cache.put(cache_key, embedding)
│   │       computed++
│   │
│   └── 进度回调 (如果提供):
│       progress_callback(computed, total)
│
└── 返回: computed (计算的 Embedding 数量)
```

---

## 5. 关键算法说明

### 5.1 余弦相似度算法 (_cosine_similarity)

```python
def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
```

**数学公式**:

$$\text{cosine\_similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}$$

**特性**:
- 输出范围: [-1.0, 1.0]（对于 Embedding 向量通常在 [0.0, 1.0]）
- 零向量保护: 任一向量为零向量时返回 0.0
- 时间复杂度: O(d)，d 为向量维度

### 5.2 BM25 (Best Matching 25) 算法

BM25 是一种基于概率的文本检索算法，用于 Hybrid 策略的初筛阶段。

**核心公式**:

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$$

其中:
- \( Q \) = 查询词列表
- \( D \) = 文档 (消息内容)
- \( f(q_i, D) \) = 词 \( q_i \) 在文档 \( D \) 中的出现频率
- \( |D| \) = 文档长度
- \( avgdl \) = 平均文档长度
- \( k_1 \), \( b \) = BM25 参数 (BM25Okapi 默认 k1=1.5, b=0.75)

**本模块使用的 BM25Okapi 实现** (来自 `rank_bm25` 库):
- 自动计算 IDF 权重
- 自动处理文档长度归一化
- 返回每个文档的 BM25 分数数组

### 5.3 分词算法 (_tokenize)

```python
def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\w+', text.lower())
    return tokens
```

**算法说明**:
- 转换为小写 (大小写不敏感)
- 使用正则 `\w+` 匹配连续的字母/数字/下划线
- 自动过滤标点符号和特殊字符
- 适用于英文和基本的中文分词 (中文每个字符作为独立 token)

### 5.4 自动策略选择算法

```
_select_strategy(message_count: int) → SearchStrategy

决策逻辑:
  if message_count ≤ lazy_max_messages (100):
      return LAZY
      理由: 数据量小，实时计算全部 Embedding 的开销可接受
      
  elif message_count ≥ batch_trigger_count (1000):
      return BATCH
      理由: 数据量大，需要批量预计算以利用批处理优化
      
  else:  # 100 < count < 1000
      return HYBRID
      理由: 中等数据量，BM25 初筛可大幅减少 Embedding 计算量
```

---

## 6. Embedding 缓存机制

### 6.1 缓存键生成规则

```
缓存键 = msg.message_id  (优先使用消息 ID)
       or msg.content_hash()  (回退到内容 MD5 哈希)

示例:
  message_id = "msg_abc123"  → 缓存键 = "msg_abc123"
  message_id = None, content = "Hello" → 缓存键 = MD5("Hello")
```

### 6.2 缓存查找与写入流程

```
┌─────────────────────────────────────────────────┐
│ Embedding 缓存查找流程                            │
├─────────────────────────────────────────────────┤
│                                                   │
│  1. 生成 cache_key                                │
│     ↓                                             │
│  2. _embedding_cache.get(cache_key)               │
│     ├── 命中 → 返回缓存的 Embedding               │
│     │         (move_to_end, O(1))                 │
│     └── 未命中 → 3. embedding_service.embed()     │
│                    ↓                               │
│                 4. _embedding_cache.put(key, emb)  │
│                    ├── 缓存未满 → 直接插入          │
│                    └── 缓存已满 → 淘汰最旧 + 插入   │
│                                                   │
└─────────────────────────────────────────────────┘
```

### 6.3 缓存容量与性能

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_max_size` | 100,000 | 最大缓存条目数 |
| 单条 Embedding 大小 | ~3KB (768维 float32) | 取决于模型维度 |
| 最大内存占用 | ~300MB | 100,000 × 3KB |
| 查找/插入时间复杂度 | O(1) | OrderedDict 实现 |

---

## 7. 与外部服务的交互

### 7.1 EmbeddingService 接口要求

本模块依赖外部传入的 `embedding_service` 对象，需实现以下接口：

| 方法 | 签名 | 必需 | 说明 |
|------|------|------|------|
| `embed` | `(text: str) → List[float]` | 是 | 计算单条文本的 Embedding |
| `embed_batch` | `(texts: List[str]) → List[List[float]]` | 否 | 批量计算 Embedding (可选优化) |

### 7.2 与 ChatMessage 数据源的关系

```
上层应用 (messages 表)
    │
    ├── 查询消息列表 → List[ChatMessage]
    │   (部分消息可能没有 embedding 字段)
    │
    └── 传入 NonVectorizedDataHandler.search_relevant_messages()
        │
        ├── 有 embedding 的消息 → 可直接使用 (但本模块不使用此字段)
        └── 无 embedding 的消息 → 本模块实时计算并缓存
```

**注意**: 本模块始终通过 `embedding_service.embed()` 计算 Embedding，不直接使用 `ChatMessage.embedding` 字段。`ChatMessage.embedding` 字段仅在 `_batch_search` 中用于判断消息是否已有预计算的 Embedding。

---

## 8. 异常处理策略

| 场景 | 处理方式 | 影响 |
|------|---------|------|
| `rank_bm25` 未安装 | 降级: 返回前 N 条消息 | Hybrid 策略退化为截断 |
| `embed()` 调用失败 | 异常向上传播 | 搜索失败 |
| `embed_batch()` 不存在 | 回退到逐条 `embed()` | 性能下降但功能正常 |
| 空消息列表 | 直接返回 [] | 无影响 |

---

## 9. 统计指标说明

### 9.1 get_stats() 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `lazy_searches` | `int` | Lazy 策略搜索次数 |
| `batch_searches` | `int` | Batch 策略搜索次数 |
| `hybrid_searches` | `int` | Hybrid 策略搜索次数 |
| `cache_hits` | `int` | Embedding 缓存命中次数 |
| `embeddings_computed` | `int` | Embedding 计算次数 |
| `cache_size` | `int` | 当前缓存大小 |
| `cache_max_size` | `int` | 最大缓存容量 |

---

## 10. 便捷工厂函数

### create_non_vectorized_handler()

```python
def create_non_vectorized_handler(
    embedding_service: Any,
    strategy: str = "hybrid",
    cache_embeddings: bool = True,
    **kwargs,
) -> NonVectorizedDataHandler
```

**参数说明**:
- `embedding_service`: Embedding 计算服务实例
- `strategy`: 默认策略字符串 ("lazy" / "batch" / "hybrid")
- `cache_embeddings`: 是否启用 Embedding 缓存
- `**kwargs`: 传递给 `HandlerConfig` 的其他参数

**使用示例**:

```python
# 快速创建
handler = create_non_vectorized_handler(
    embedding_service=my_embedding_service,
    strategy="hybrid",
    cache_embeddings=True,
    bm25_candidates_multiplier=6,
)

# 搜索
results = await handler.search_relevant_messages(
    messages=chat_messages,
    query="关于食物偏好的讨论",
    top_k=5,
)

for result in results:
    print(f"[{result.score:.3f}] {result.message.content[:50]}...")
```

---

## 11. 性能特征

### 11.1 各策略时间复杂度

| 策略 | 时间复杂度 | 说明 |
|------|-----------|------|
| LAZY | O(N × d) | N=消息数, d=Embedding 维度 |
| BATCH | O(N × d) (首次) + O(N × d) (搜索) | 首次批量计算后搜索 |
| HYBRID | O(N × L) + O(K × d) | N=消息数, L=平均词数, K=BM25 候选数 |

### 11.2 Hybrid 策略性能优势

```
场景: 500 条消息, top_k=5, Embedding 计算耗时 10ms/条

LAZY:   500 × 10ms = 5000ms (5秒)
HYBRID: BM25 筛选 (~50ms) + 20 × 10ms = 250ms (0.25秒)

加速比: 5000/250 = 20x
```
