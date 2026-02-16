# Reference Resolver 模块说明书

> 文件路径: `DKI/dki/core/components/reference_resolver.py`

## 1. 模块概述

本模块实现了 DKI 系统的**指代解析器 (Reference Resolver)**，用于处理用户输入中的指代表达，确定历史消息的召回范围。当用户说"刚刚"、"那件事"、"你之前说的"等指代词时，解析器会判断应该回溯多少轮对话来获取相关上下文。

### 指代类型

| 类型 | 枚举值 | 说明 | 示例 |
|------|--------|------|------|
| **时间指代** | `TEMPORAL` | 基于时间的回溯 | "刚刚"、"最近"、"上次" |
| **实体指代** | `REFERENTIAL` | 指向特定话题/事物 | "那件事"、"那个问题" |
| **立场指代** | `STANCE` | 指向助手的历史观点 | "你之前说的"、"你上次说" |
| **无指代** | `NONE` | 无指代表达 | 普通对话 |

### 指代范围

| 范围 | 枚举值 | 召回轮数 | 说明 |
|------|--------|---------|------|
| 最近 1-3 轮 | `LAST_1_3_TURNS` | `last_few_turns` (默认 3) | "刚刚"、"刚才" |
| 最近 5-10 轮 | `LAST_5_10_TURNS` | `recent_turns` (默认 10) | "上次"、"前几天" |
| 当前会话 | `CURRENT_SESSION` | `session_max_turns` (默认 50) | "最近" |
| 上一共享主题 | `LAST_SHARED_TOPIC` | `last_few_turns × 2` | "那件事"、"那个话题" |
| 助手最后立场 | `ASSISTANT_LAST_STANCE` | `recent_turns` | "你之前说的" |
| 自定义 | `CUSTOM` | — | 默认/回退 |

## 2. 数据结构

### 2.1 Message

消息数据结构。

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | `str` | 角色 ("user" 或 "assistant") |
| `content` | `str` | 消息内容 |
| `timestamp` | `Optional[datetime]` | 时间戳 |
| `turn_id` | `Optional[str]` | 轮次 ID |
| `topic` | `Optional[str]` | 主题标签 |

### 2.2 ResolvedReference

解析结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `reference_type` | `ReferenceType` | 指代类型 |
| `scope` | `ReferenceScope` | 指代范围 |
| `resolved_content` | `Optional[str]` | 解析出的相关内容 |
| `confidence` | `float` | 置信度 |
| `source_turns` | `List[int]` | 来源轮次索引 |
| `matched_keyword` | `Optional[str]` | 匹配的关键词 |
| `metadata` | `Dict` | 元数据 (语言、召回轮数等) |

属性:
- `recall_turns` — 建议的召回轮数 (从 metadata 获取)

### 2.3 ReferenceResolverConfig

解析器配置，支持 YAML 文件加载。

**核心可配置项:**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `last_few_turns` | 3 | "刚刚/刚才" 召回轮数 |
| `recent_turns` | 10 | "最近/上次" 召回轮数 |
| `session_max_turns` | 50 | 会话最大召回轮数 |
| `use_llm_fallback` | `False` | 是否启用 LLM 辅助 |
| `llm_fallback_threshold` | 0.5 | LLM 回退阈值 |

**指代关键词映射 (中文):**

| 关键词 | 范围 | 类型 |
|--------|------|------|
| "刚刚" | `last_1_3_turns` | `temporal` |
| "刚才" | `last_1_3_turns` | `temporal` |
| "最近" | `current_session` | `temporal` |
| "那件事" | `last_shared_topic` | `referential` |
| "那个问题" | `last_shared_topic` | `referential` |
| "那个话题" | `last_shared_topic` | `referential` |
| "之前你说的" | `assistant_last_stance` | `stance` |
| "你上次说" | `assistant_last_stance` | `stance` |
| "你之前提到" | `assistant_last_stance` | `stance` |
| "上次" | `last_5_10_turns` | `temporal` |
| "前几天" | `last_5_10_turns` | `temporal` |

**指代关键词映射 (英文):**

| 关键词 | 范围 | 类型 |
|--------|------|------|
| "just now" | `last_1_3_turns` | `temporal` |
| "just" | `last_1_3_turns` | `temporal` |
| "recently" | `current_session` | `temporal` |
| "that thing" | `last_shared_topic` | `referential` |
| "you said earlier" | `assistant_last_stance` | `stance` |
| "last time" | `last_5_10_turns` | `temporal` |

## 3. 核心类: ReferenceResolver

### 3.1 初始化

```python
resolver = ReferenceResolver(
    config=ReferenceResolverConfig(),  # 可选
    language="auto",                   # "cn", "en", "auto"
    llm_resolver=None,                 # LLM 辅助函数 (可选)
)
```

### 3.2 resolve() — 核心解析方法

**完整流程:**

```
输入: query (用户查询), history? (历史消息列表), stance_cache? (立场缓存)
  │
  ├── 1. 空查询检查
  │     空或纯空白 → 返回 (NONE, CUSTOM)
  │
  ├── 2. 语言检测
  │     language == "auto" → _detect_language(query)
  │
  ├── 3. 获取对应语言的关键词映射
  │     cn → reference_mappings_cn
  │     en → reference_mappings_en
  │
  ├── 4. 规则匹配 (遍历关键词)
  │     for keyword, mapping in mappings:
  │       if keyword in query.lower():
  │         │
  │         ├── 确定范围 scope = ReferenceScope(mapping['scope'])
  │         ├── 确定类型 ref_type = ReferenceType(mapping['type'])
  │         ├── 计算召回轮数 recall_turns
  │         │
  │         ├── 如果有历史 → 根据范围提取内容
  │         │   _resolve_by_scope(scope, ref_type, history, stance_cache)
  │         │
  │         └── 返回 ResolvedReference
  │
  ├── 5. 规则匹配失败 → LLM 辅助 (如启用)
  │     use_llm_fallback && llm_resolver → _llm_resolve(query, history)
  │
  └── 6. 无匹配 → 返回 (NONE, CUSTOM)
```

### 3.3 _resolve_by_scope() — 按范围解析内容

根据不同的指代范围，从历史消息中提取相关内容。

**各范围的解析策略:**

#### LAST_1_3_TURNS (最近 1-3 轮)

```
n = config.last_few_turns (默认 3)
recent = history[-n*2:]  # 每轮 2 条消息 (user + assistant)
返回格式化的消息文本
```

#### LAST_5_10_TURNS (最近 5-10 轮)

```
n = config.recent_turns (默认 10)
recent = history[-n*2:]
返回格式化的消息文本
```

#### CURRENT_SESSION (当前会话)

```
max_turns = config.session_max_turns (默认 50)
recent = history[-max_turns*2:]
返回格式化的消息文本
```

#### LAST_SHARED_TOPIC (上一个共享主题)

```
从后向前遍历历史:
  找到有 topic 标签的消息 → 获取前后各 2 条上下文
  未找到 → 回退到最近 last_few_turns 轮
```

#### ASSISTANT_LAST_STANCE (助手最后立场)

```
优先使用 stance_cache:
  获取最新的立场记录 → 返回 "关于「{topic}」: {stance}"

否则从历史中查找:
  从后向前遍历助手消息
  查找包含观点标记的消息:
    中文: "我认为"、"我觉得"、"我建议"、"我的看法是"
    英文: "I think"、"I believe"、"I suggest"
  返回第一个匹配的消息
```

### 3.4 解析示例

**示例 1: 时间指代**

```python
query = "刚才你说的那个方案是什么？"
history = [
    Message(role="user", content="有什么好的排序算法？"),
    Message(role="assistant", content="推荐使用快速排序..."),
    Message(role="user", content="还有呢？"),
    Message(role="assistant", content="归并排序也不错..."),
]

result = resolver.resolve(query, history)
# result.reference_type = TEMPORAL
# result.scope = LAST_1_3_TURNS
# result.matched_keyword = "刚才"
# result.recall_turns = 3
# result.resolved_content = "用户: 有什么好的排序算法？\n助手: 推荐使用快速排序...\n..."
```

**示例 2: 立场指代**

```python
query = "之前你说的那个建议还有效吗？"
history = [
    Message(role="assistant", content="我建议使用Redis作为缓存层"),
    ...
]

result = resolver.resolve(query, history)
# result.reference_type = STANCE
# result.scope = ASSISTANT_LAST_STANCE
# result.matched_keyword = "之前你说的"
# result.resolved_content = "我建议使用Redis作为缓存层"
```

### 3.5 update_config() — 动态更新配置

支持两种参数命名风格：

```python
# 新风格 (更直观)
resolver.update_config(
    just_now_turns=5,      # "刚刚" 召回轮数
    recently_turns=20,     # "最近" 召回轮数
)

# 旧风格 (兼容)
resolver.update_config(
    last_few_turns=5,
    recent_turns=20,
    session_max_turns=100,
)
```

新参数优先于旧参数。

### 3.6 add_mapping() — 动态添加映射

```python
resolver.add_mapping(
    keyword="前天",
    scope="last_5_10_turns",
    ref_type="temporal",
    language="cn",
)
```

### 3.7 _detect_language() — 语言检测

与 MemoryTrigger 相同的算法：中文字符占比 > 30% → 中文。

## 4. 工厂函数: create_reference_resolver()

```python
resolver = create_reference_resolver(
    config_path="config/reference_resolver.yaml",  # 可选
    last_few_turns=3,
    recent_turns=10,
    language="auto",
)
```

## 5. 数据库交互

本模块不直接与数据库交互。历史消息由上层模块 (`dki_plugin.py`) 从数据库获取后传入 `resolve()` 方法。

## 6. 注意事项

1. **关键词匹配使用 `in` 操作符**: `keyword in query.lower()`，这意味着：
   - 子串匹配可能产生误匹配 (如 "just" 可能匹配 "adjust")
   - 匹配顺序取决于字典遍历顺序 (Python 3.7+ 保证插入顺序)
   - 建议后续改为正则匹配或添加词边界检查
2. **召回轮数 × 2**: 代码中使用 `n*2` 是因为每轮对话包含 user 和 assistant 两条消息
3. **stance_cache 的时间比较**: 使用字符串比较 `updated_at`，假设时间格式为可排序的字符串 (如 ISO 8601)
4. **LLM 辅助解析**: 接口已预留 (`llm_resolver` 回调函数)，但未内置实现
5. **_find_last_topic 的回退**: 当没有消息带有 `topic` 标签时，回退到最近 `last_few_turns` 轮，这可能不是最优策略
