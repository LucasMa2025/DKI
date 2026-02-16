# Hybrid DKI Injector 模块说明书

> 文件路径: `DKI/dki/core/components/hybrid_injector.py`

## 1. 模块概述

本模块实现了 DKI 论文 Section 3.9 中提出的**混合注入策略 (Hybrid Injection Strategy)**，是 DKI 系统的核心注入组件。该策略将用户偏好和会话历史分层注入，模拟人类认知的分层记忆结构。

### 信息分层架构 (类比人类认知)

```
┌─────────────────────────────────────────────────────────────┐
│  层级   │  内容       │  注入方式          │  认知类比      │
├─────────┼─────────────┼───────────────────┼───────────────┤
│  L1     │  偏好       │  K/V (负位置)      │  人格/性格    │
│  L2     │  历史       │  后缀提示 (正位置) │  记忆/经验    │
│  L3     │  查询       │  输入 (正位置)     │  当前思考     │
└─────────────────────────────────────────────────────────────┘
```

### 设计优势

| 优势 | 说明 |
|------|------|
| **最小化 OOD 风险** | 偏好短小，历史在正位置 |
| **最大化灵活性** | 历史可长可短，偏好可缓存 |
| **支持显式引用** | 历史在 prompt 中可见，可被引用 |
| **减少幻觉** | 信任建立提示引导模型正确使用历史 |

## 2. 数据结构

### 2.1 UserPreference

用户偏好数据结构。

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `str` | 偏好文本内容 |
| `user_id` | `str` | 用户 ID |
| `metadata` | `Dict[str, Any]` | 元数据 |
| `kv_cache` | `Optional[Any]` | 缓存的 K/V (计算一次，跨会话复用) |
| `token_count` | `int` | token 数量 |

### 2.2 SessionMessage

单条会话消息。

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | `str` | 角色 ("user" 或 "assistant") |
| `content` | `str` | 消息内容 |
| `timestamp` | `Optional[float]` | 时间戳 |

### 2.3 SessionHistory

会话历史容器。

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[SessionMessage]` | 消息列表 |
| `session_id` | `str` | 会话 ID |
| `max_tokens` | `int` | 最大 token 数 (默认 500) |

方法:
- `add_message(role, content, timestamp)` — 添加消息
- `get_recent(max_messages=10)` — 获取最近 N 条消息

### 2.4 HybridInjectionConfig

混合注入配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `preference_enabled` | `bool` | `True` | 启用偏好注入 |
| `preference_alpha` | `float` | `0.4` | 偏好注入强度 (建议 0.3-0.5) |
| `preference_max_tokens` | `int` | `100` | 偏好最大 token 数 |
| `preference_position_strategy` | `str` | `"negative"` | 位置策略 |
| `history_enabled` | `bool` | `True` | 启用历史注入 |
| `history_max_tokens` | `int` | `500` | 历史最大 token 数 |
| `history_max_messages` | `int` | `10` | 历史最大消息数 |
| `history_method` | `str` | `"suffix_prompt"` | 历史注入方式 |
| `history_prefix_template` | `str` | `""` | 历史前缀模板 |
| `history_suffix_template` | `str` | `""` | 历史后缀模板 |

### 2.5 HybridInjectionResult

注入准备结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `input_text` | `str` | 组装后的完整输入文本 |
| `preference_kv` | `Optional[Any]` | 偏好 K/V 缓存 |
| `preference_alpha` | `float` | 偏好注入强度 |
| `preference_tokens` | `int` | 偏好 token 数 |
| `history_tokens` | `int` | 历史 token 数 |
| `total_tokens` | `int` | 总 token 数 |
| `metadata` | `Dict` | 元数据 |
| `preference_text` | `str` | 偏好原文 (用于显示) |
| `history_suffix_text` | `str` | 历史后缀原文 (用于显示) |
| `history_messages` | `List[Dict]` | 历史消息列表 (用于显示) |

## 3. 核心类: HybridDKIInjector

### 3.1 初始化

```python
injector = HybridDKIInjector(
    config=HybridInjectionConfig(),
    model=model,          # LLM 模型 (用于 K/V 计算)
    tokenizer=tokenizer,  # 分词器 (用于 token 计数)
    language="cn",        # "en" 或 "cn"
)
```

**初始化流程:**
1. 设置配置 (使用默认或自定义)
2. 根据语言选择默认提示模板 (中文/英文)
3. 初始化偏好 K/V 缓存字典 `_preference_cache`

### 3.2 默认提示模板

**中文模板:**

```
[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
请在理解历史上下文后，给出连贯的整体回复。
重要：请使用中文回复用户。
---
{历史消息}
---
[会话历史结束]
请基于以上历史和用户当前问题，使用中文给出回复。
注意：历史信息仅供参考，请综合回答。
```

**英文模板:**

```
[Session History Reference]
Before responding, please refer to the following session history.
These are real conversation records between you and the user, and are trustworthy.
Please provide a coherent response after understanding the historical context.
---
{history messages}
---
[End of Session History]
Please respond based on the above history and the user's current question.
Note: Historical information is for reference; please answer comprehensively.
```

### 3.3 prepare_input() — 核心方法: 准备混合注入输入

**完整流程:**

```
输入: user_query, preference?, history?, system_prompt?
  │
  ├── 1. 初始化结果对象
  │
  ├── 2. 组装文本部分 (顺序: System → History → Query)
  │     │
  │     ├── 2a. System Prompt (如有)
  │     │     text_parts.append(system_prompt)
  │     │
  │     ├── 2b. 历史消息 (如启用且有消息)
  │     │     history_text = _format_history(history)
  │     │     text_parts.append(history_text)
  │     │     记录 history_tokens, history_suffix_text, history_messages
  │     │
  │     └── 2c. 用户查询
  │           text_parts.append("User: {user_query}")
  │
  ├── 3. 拼接文本
  │     input_text = "\n\n".join(text_parts)
  │     total_tokens = _estimate_tokens(input_text)
  │
  ├── 4. 安全检查: Prompt 长度限制
  │     max_prompt_tokens = max_model_len - 512 (预留生成空间)
  │     如果超长 → 截断历史，只保留 system + query
  │
  ├── 5. 偏好 K/V 处理 (如启用且有偏好)
  │     kv_cache = _get_or_compute_preference_kv(preference)
  │     设置 preference_kv, preference_alpha, preference_tokens
  │     保存 preference_text
  │
  └── 6. 返回 HybridInjectionResult
```

**组装示例 (中文场景):**

假设输入:
- system_prompt = "你是一个有帮助的AI助手"
- preference.content = "用户偏好简洁回答，喜欢Python"
- history = [User: "Python怎么排序？", Assistant: "可以用sorted()函数"]
- user_query = "那列表推导式呢？"

组装结果:

```
你是一个有帮助的AI助手

[会话历史参考]
在回复用户之前，请参考以下历史会话信息。
这些是用户与你之前的真实对话记录，内容可信。
请在理解历史上下文后，给出连贯的整体回复。
重要：请使用中文回复用户。
---
用户: Python怎么排序？
助手: 可以用sorted()函数
---
[会话历史结束]
请基于以上历史和用户当前问题，使用中文给出回复。
注意：历史信息仅供参考，请综合回答。

User: 那列表推导式呢？
```

同时，偏好 "用户偏好简洁回答，喜欢Python" 被转换为 K/V 对，以 α=0.4 在负位置注入。

### 3.4 _format_history() — 历史格式化

**流程:**

```
输入: SessionHistory
  │
  ├── 1. 获取最近消息 (respect max_messages)
  │
  ├── 2. 安全过滤: 跳过包含注入模板标记的消息
  │     标记: "[会话历史参考]", "[Session History Reference]" 等
  │     (防止递归嵌套: 历史中包含之前注入的历史后缀)
  │
  ├── 3. 格式化消息
  │     中文: "用户: {content}" / "助手: {content}"
  │     英文: "User: {content}" / "Assistant: {content}"
  │     跳过空消息
  │
  ├── 4. 截断 (如超过 history_max_tokens)
  │     从最早的消息开始移除，直到 token 数在限制内
  │
  └── 5. 包装模板
        prefix_template + history_content + suffix_template
```

**关键设计 — 防递归嵌套:**

当历史消息中包含之前注入的历史后缀文本时（例如 assistant 的回复中包含了 `[会话历史参考]` 标记），这些消息会被过滤掉，避免历史提示无限嵌套。

### 3.5 _get_or_compute_preference_kv() — 偏好 K/V 计算与缓存

**流程:**

```
输入: UserPreference
  │
  ├── 1. 检查偏好对象自身缓存
  │     preference.kv_cache is not None → 直接返回
  │
  ├── 2. 检查实例级缓存
  │     cache_key = "{user_id}:{hash(content)}"
  │     _preference_cache[cache_key] → 直接返回
  │
  ├── 3. 计算 K/V (需要 model)
  │     ├── 分词: tokenizer.encode(content)
  │     │   或估算: len(words) × 1.3
  │     │
  │     ├── 方式 A: model.compute_kv(content)
  │     │   (自定义适配器接口)
  │     │
  │     └── 方式 B: model.model(input_ids, use_cache=True)
  │         (HuggingFace 风格)
  │
  ├── 4. 缓存结果
  │     preference.kv_cache = kv_entries
  │     _preference_cache[cache_key] = kv_entries
  │
  └── 5. 返回 kv_entries
```

**缓存策略说明:**

偏好特别适合缓存，因为：
1. **内容短** — 通常 < 100 tokens
2. **变化慢** — 用户偏好很少改变
3. **跨会话复用** — 同一用户的偏好在所有会话中通用

### 3.6 _estimate_tokens() — Token 数量估算

**两种模式:**

1. **精确模式** (有 tokenizer): 调用 `tokenizer.encode(text)` 获取精确 token 数
2. **启发式模式** (无 tokenizer): 混合中英文估算
   - 中文字符: ~1.5 tokens/字符 (子词分词)
   - 英文单词: ~1.3 tokens/单词
   - 最少返回 1

### 3.7 其他方法

| 方法 | 说明 |
|------|------|
| `clear_preference_cache(user_id?)` | 清除偏好缓存 (指定用户或全部) |
| `get_stats()` | 获取注入器统计信息 |

## 4. 工厂函数: create_hybrid_injector()

```python
injector = create_hybrid_injector(
    model=model,
    tokenizer=tokenizer,
    preference_alpha=0.4,
    history_max_tokens=500,
    language="cn",
)
```

快速创建配置好的 `HybridDKIInjector` 实例。

## 5. 数据库交互

本模块不直接与数据库交互。偏好和历史数据由上层 (`dki_system.py`, `dki_plugin.py`) 从数据库获取后传入。

## 6. 注意事项

1. **Prompt 长度安全检查**: `prepare_input()` 会检查总 prompt 长度，超过 `max_model_len - 512` 时自动截断历史
2. **tokenizer.model_max_length 过滤**: 某些 tokenizer 返回极大值 (如 1e30)，代码过滤了 > 1,000,000 的值
3. **偏好缓存键使用 hash()**: `hash(preference.content)` 在不同 Python 进程间不稳定 (PYTHONHASHSEED)，分布式环境下可能导致缓存不一致
4. **历史截断策略**: 当前采用简单的从头移除策略，可能丢失重要的早期上下文。后续可考虑基于相关性的选择性截断
