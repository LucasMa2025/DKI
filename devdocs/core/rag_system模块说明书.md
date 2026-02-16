# RAGSystem 检索增强生成系统模块说明书

> 源文件: `DKI/dki/core/rag_system.py`  
> 模块路径: `dki.core.rag_system`  
> 文件行数: 503 行

---

## 1. 模块概述

`RAGSystem` 实现了标准的 **Retrieval-Augmented Generation (检索增强生成)** 范式，作为 DKI 系统的**基线对比系统**。核心流程: 检索相关记忆 → 拼接到 Prompt → 调用 LLM 生成。

与 DKI 的关键区别: RAG 将检索到的记忆**以文本形式拼接到 Prompt 中**，占用 Token Budget；DKI 将记忆**以 K/V 形式注入注意力层**，不占用 Token Budget。

---

## 2. 数据结构

### 2.1 RAGPromptInfo

| 字段                | 类型         | 说明                 |
| ------------------- | ------------ | -------------------- |
| `original_query`    | `str`        | 原始用户查询         |
| `system_prompt`     | `str`        | 系统提示词           |
| `retrieved_context` | `str`        | 检索到的记忆上下文   |
| `history_text`      | `str`        | 历史对话文本         |
| `history_messages`  | `List[Dict]` | 历史消息列表         |
| `final_prompt`      | `str`        | 最终构造的完整提示词 |

### 2.2 RAGResponse

| 字段            | 类型                       | 说明           |
| --------------- | -------------------------- | -------------- |
| `text`          | `str`                      | 生成的回复文本 |
| `memories_used` | `List[MemorySearchResult]` | 使用的记忆列表 |
| `latency_ms`    | `float`                    | 总延迟 (毫秒)  |
| `input_tokens`  | `int`                      | 输入 Token 数  |
| `output_tokens` | `int`                      | 输出 Token 数  |
| `metadata`      | `Dict`                     | 元数据         |
| `prompt_info`   | `RAGPromptInfo`            | 提示词构造详情 |

---

## 3. 核心方法流程

### 3.1 `chat()` — 主对话方法

```
chat(query, session_id, user_id, top_k, system_prompt, ...)
  │
  ├─ Step 1: 检索相关记忆
  │   └─ memories = memory_router.search(query, top_k)
  │
  ├─ Step 2: 获取会话历史 (如果 include_history=True)
  │   └─ history = _get_conversation_history(session_id, max_turns)
  │
  ├─ Step 3: 构建 Prompt (含自动截断)
  │   └─ prompt, prompt_info = _build_prompt(query, memories, system_prompt, history)
  │
  ├─ Step 4: 调用 LLM 生成
  │   └─ output = model.generate(prompt, max_new_tokens, temperature)
  │
  ├─ Step 5: 记录到数据库
  │   ├─ 确保 session 存在 (get_or_create)
  │   ├─ 存储用户消息 (role='user')
  │   ├─ 存储助手回复 (role='assistant', injection_mode='rag')
  │   └─ 写入审计日志 (action='rag_generate')
  │
  └─ 返回 RAGResponse
```

### 3.2 `_build_prompt()` — 提示词构建 (核心算法)

```
_build_prompt(query, memories, system_prompt, history)
  │
  ├─ Step 1: 计算可用空间
  │   ├─ max_context = _get_max_context_length()
  │   ├─ max_prompt_tokens = max_context - 512  (预留生成空间)
  │   ├─ 计算固定部分 token 数 (system_prompt + query + footer)
  │   └─ remaining_tokens = max_prompt_tokens - fixed_tokens
  │
  ├─ Step 2: 构建检索上下文 (优先保留，最多占 50% 剩余空间)
  │   ├─ 遍历 memories, 逐条添加 "[i] content"
  │   ├─ 累计 token 数 > remaining_tokens * 0.5 时停止
  │   └─ remaining_tokens -= context_tokens
  │
  ├─ Step 3: 构建会话历史 (从最新开始，截断最旧的)
  │   ├─ 从 history 末尾开始遍历 (reversed)
  │   ├─ 逐条添加 "User/AI: content"
  │   ├─ 累计 token 数 > remaining_tokens 时停止
  │   └─ 如有截断，记录日志
  │
  ├─ Step 4: 组装最终 Prompt (按顺序拼接)
  │
  └─ Step 5: 最终安全检查
      ├─ 如果仍超长 → 强制截断，仅保留 query
      └─ 返回 (final_prompt, RAGPromptInfo)
```

### 3.3 `_estimate_tokens()` — Token 估算

优先使用 tokenizer 精确计算，回退时使用粗略估算 (中文约 1.5 字/token, 英文约 4 字符/token, 取平均约 `len(text) // 2`)。

### 3.4 `_get_conversation_history()` — 获取会话历史

从数据库 `conversations` 表中获取指定 session 的最近 N 轮对话，返回 `[{"role": ..., "content": ...}]` 格式。

---

## 4. 提示词组装示例

### 4.1 完整示例 (System Prompt + 记忆 + 历史 + 查询)

**输入参数:**

-   `query` = "推荐一家北京的餐厅"
-   `system_prompt` = "你是一个美食推荐助手"
-   `memories` = 3 条检索到的记忆
-   `history` = 2 轮历史对话

**最终组装的 Prompt (final_prompt):**

```
System: 你是一个美食推荐助手

Relevant information:
[1] 用户偏好: 素食主义者，不吃辣，喜欢日料
[2] 用户常去区域: 朝阳区、海淀区
[3] 用户预算偏好: 人均 100-200 元

Previous conversation:
User: 我最近想尝试一些新的餐厅
AI: 好的，您有什么特别的偏好吗？比如菜系、价位或者地区？
User: 主要在北京，预算不要太高
AI: 明白了，我可以帮您推荐一些性价比高的餐厅。

User: 推荐一家北京的餐厅

AI:
```

**Prompt 结构分析:**

| 区域                 | 内容            | 估算 Token 数 | 占比 |
| -------------------- | --------------- | ------------- | ---- |
| System Prompt        | 系统角色定义    | ~15           | 3%   |
| Retrieved Context    | 3 条检索记忆    | ~80           | 16%  |
| Conversation History | 2 轮 (4 条消息) | ~120          | 24%  |
| User Query           | 当前查询        | ~15           | 3%   |
| **总计**             |                 | **~230**      |      |
| 预留生成空间         |                 | 512           |      |

### 4.2 仅查询示例 (无记忆、无历史)

```
User: 今天天气怎么样？

AI:
```

### 4.3 上下文截断示例

当模型上下文窗口为 4096 tokens，预留 512 给生成，可用 3584 tokens:

```
Step 1: 固定部分 (system + query + footer) = 50 tokens
        remaining = 3534 tokens

Step 2: 检索上下文最多占 50% = 1767 tokens
        实际使用 3 条记忆 = 80 tokens
        remaining = 3454 tokens

Step 3: 会话历史从最新开始填充
        保留最近 20 轮 (约 2000 tokens)
        截断更早的 30 轮 → 日志记录 "kept 20/50 history messages"

Step 4: 组装 → 总计约 2130 tokens < 3584 ✓
```

### 4.4 强制截断示例

极端情况下，如果组装后仍超长:

```
final_tokens = 4000 > max_prompt_tokens = 3584
→ 强制截断: 仅保留 "User: {query}\nAI:"
→ 清空 history_text 和 retrieved_context
→ 记录 WARNING 日志
```

---

## 5. 关键算法

### 5.1 空间分配策略

Prompt 空间分配采用**优先级策略**:

1. **固定部分** (最高优先级): System Prompt + User Query + Footer → 必须保留
2. **检索上下文** (高优先级): 最多占剩余空间的 50% → 保证记忆信息
3. **会话历史** (中优先级): 占用剩余空间 → 从最新开始，截断最旧的
4. **生成空间** (预留): 固定 512 tokens → 确保模型有足够生成空间

### 5.2 历史截断算法

```python
# 从最新消息开始，逆序遍历
selected_history = []
used_tokens = header_tokens

for msg in reversed(history):
    line_tokens = estimate_tokens(f"{role}: {content}")
    if used_tokens + line_tokens > remaining_tokens:
        break  # 空间不足，停止添加
    selected_history.insert(0, line)  # 插入到头部保持时序
    used_tokens += line_tokens
```

**特点:** 保留最近的对话，丢弃最早的对话，符合"近因效应"。

---

## 6. 数据库交互

### 6.1 涉及的数据库表

| 表名            | 操作  | 说明                                                    |
| --------------- | ----- | ------------------------------------------------------- |
| `sessions`      | 读/写 | `get_or_create(session_id)` 确保会话存在                |
| `memories`      | 读    | `get_by_session()` 加载记忆, `get_embedding()` 获取向量 |
| `conversations` | 读/写 | `get_recent()` 获取历史, `create()` 存储新消息          |
| `audit_logs`    | 写    | `log(action='rag_generate')` 记录审计日志               |

### 6.2 conversations 表写入字段

**用户消息:**

```python
conv_repo.create(
    session_id=session_id,
    role='user',
    content=query,
)
```

**助手回复:**

```python
conv_repo.create(
    session_id=session_id,
    role='assistant',
    content=output.text,
    injection_mode='rag',
    memory_ids=[m.memory_id for m in memories],
    latency_ms=total_latency,
)
```

### 6.3 audit_logs 表写入字段

```python
audit_repo.log(
    action='rag_generate',
    session_id=session_id,
    memory_ids=[m.memory_id for m in memories],
    mode='rag',
)
```

---

## 7. 其他方法

### 7.1 `add_memory()` — 添加记忆

计算嵌入 → 存入数据库 → 添加到 MemoryRouter 索引。

### 7.2 `load_memories_from_db()` — 从数据库加载记忆

从 `memories` 表加载指定 session 的所有记忆到 MemoryRouter。

### 7.3 `search_memories()` — 仅检索不生成

直接调用 `memory_router.search()`，用于调试和测试。

### 7.4 `get_stats()` — 获取统计信息

返回路由器统计、模型信息和配置参数。

---

## 8. 配置依赖

| 配置项                            | 来源                     | 说明         |
| --------------------------------- | ------------------------ | ------------ |
| `config.rag.top_k`                | rag.top_k                | 默认检索数量 |
| `config.rag.similarity_threshold` | rag.similarity_threshold | 相似度阈值   |
| `config.database.path`            | database.path            | 数据库路径   |
| `config.database.echo`            | database.echo            | SQL 日志开关 |

---

## 9. 设计说明

-   **基线对比**: RAGSystem 作为 DKI 的对比基线，两者共享 MemoryRouter 和 EmbeddingService
-   **自动截断**: `_build_prompt()` 自动处理上下文长度限制，避免超出模型窗口
-   **空间分配**: 检索上下文最多占 50% 剩余空间，避免记忆过多挤占历史和查询
-   **审计追踪**: 每次生成都记录到 conversations 和 audit_logs 表
-   **延迟加载**: 模型适配器通过 `@property` 延迟创建
