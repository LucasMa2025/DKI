# Memory Trigger 模块说明书

> 文件路径: `DKI/dki/core/components/memory_trigger.py`

## 1. 模块概述

本模块实现了 DKI 系统的**记忆触发器 (Memory Trigger)**，用于检测用户输入中的记忆相关信号。核心原则是：**不是每句话都进记忆，只在出现特定信号时触发**。

### 触发信号分类

| 触发类型 | 枚举值 | 说明 | 示例 |
|---------|--------|------|------|
| **元认知表达** | `META_COGNITIVE` | 用户提及之前的对话 | "我们刚刚讨论的..." |
| **状态变化** | `STATE_CHANGE` | 用户改变想法或补充信息 | "我改变主意了" |
| **长期价值信号** | `LONG_TERM_VALUE` | 偏好、习惯、计划等 | "请记住我喜欢..." |
| **回顾请求** | `RECALL_REQUEST` | 请求回顾对话内容 | "我们聊了什么？" |
| **观点查询** | `OPINION_QUERY` | 查询助手的观点变化 | "你现在怎么看？" |
| **无触发** | `NONE` | 无记忆相关信号 | 普通对话 |

### 设计原则

- **规则可配置**: 支持外置 YAML 配置文件
- **双语支持**: 中文和英文模式
- **可扩展**: 预留分类器增强接口

## 2. 数据结构

### 2.1 TriggerType (枚举)

```python
class TriggerType(str, Enum):
    META_COGNITIVE = "meta_cognitive"
    STATE_CHANGE = "state_change"
    LONG_TERM_VALUE = "long_term_value"
    RECALL_REQUEST = "recall_request"
    OPINION_QUERY = "opinion_query"
    NONE = "none"
```

### 2.2 TriggerResult

触发检测结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `triggered` | `bool` | 是否触发 |
| `trigger_type` | `TriggerType` | 触发类型 |
| `matched_pattern` | `Optional[str]` | 匹配的正则模式 |
| `confidence` | `float` | 置信度 (规则匹配固定为 1.0) |
| `extracted_topic` | `Optional[str]` | 提取的主题 |
| `metadata` | `Dict` | 元数据 (语言、匹配位置等) |

### 2.3 MemoryTriggerConfig

触发器配置，支持从 YAML 文件加载。

**内置正则模式 (部分示例):**

| 类型 | 中文模式 | 英文模式 |
|------|---------|---------|
| 元认知 | `我们(刚刚\|刚才).*?(讨论\|聊)` | `we (just\|recently) (discussed\|talked)` |
| 状态变化 | `我(现在)?(改变\|改了).*?(想法\|主意)` | `I('ve\| have)? (changed\|updated) my (mind)` |
| 长期价值 | `请(记住\|记下\|保存)` | `please (remember\|note\|save)` |
| 回顾请求 | `(最近)我们(聊\|讨论)了(什么)` | `what (did\|have) we (discussed)` |
| 观点查询 | `你(对).*?(有)(新)(看法)` | `do you have (any\|new) (thoughts)` |

**配置加载方式:**

```python
# 从 YAML 文件加载
config = MemoryTriggerConfig.from_yaml("config/memory_trigger.yaml")

# 从字典创建
config = MemoryTriggerConfig.from_dict(data)

# 使用默认配置
config = MemoryTriggerConfig()
```

## 3. 核心类: MemoryTrigger

### 3.1 初始化

```python
trigger = MemoryTrigger(
    config=MemoryTriggerConfig(),  # 可选，默认使用内置配置
    language="auto",               # "cn", "en", "auto" (自动检测)
)
```

**初始化流程:**

```
1. 设置配置和语言
2. 编译正则表达式 (_compile_patterns)
   将所有模式按 TriggerType × Language 分组编译
   使用 re.IGNORECASE 忽略大小写
3. 初始化分类器 (如启用，当前为预留)
```

### 3.2 check() / detect() — 触发检测 (核心方法)

`detect()` 是 `check()` 的别名。

**完整流程:**

```
输入: message (用户消息)
  │
  ├── 1. 空消息检查
  │     空或纯空白 → 返回 (triggered=False, NONE)
  │
  ├── 2. 语言检测
  │     language == "auto" → _detect_language(message)
  │     中文字符占比 > 30% → "cn"，否则 → "en"
  │
  ├── 3. 按优先级检查各类触发
  │     优先级顺序:
  │     ① RECALL_REQUEST (回顾请求)
  │     ② OPINION_QUERY (观点查询)
  │     ③ META_COGNITIVE (元认知)
  │     ④ STATE_CHANGE (状态变化)
  │     ⑤ LONG_TERM_VALUE (长期价值)
  │     │
  │     对每种类型:
  │     └── _check_patterns(message, trigger_type, lang)
  │           遍历该类型的所有编译模式
  │           匹配成功 → 提取主题 → 返回 TriggerResult
  │
  ├── 4. 分类器二次检查 (如启用)
  │     规则匹配失败时，使用分类器补充检测
  │     (当前未实现)
  │
  └── 5. 无匹配 → 返回 (triggered=False, NONE)
```

**优先级设计说明:**

回顾请求 > 观点查询 > 元认知 > 状态变化 > 长期价值

- 回顾请求最优先：用户明确要求回顾，应立即响应
- 长期价值最低：因为偏好类信号可能与其他类型重叠

### 3.3 _extract_topic() — 主题提取

**流程:**

```
输入: message, match (正则匹配对象), trigger_type
  │
  ├── 1. 获取匹配位置之后的文本
  │     after_match = message[match.end():]
  │
  ├── 2. 清理连接词
  │     移除: 的、是、吗、呢、啊、了、?、？、!、！
  │
  ├── 3. 截断
  │     超过 50 字符 → 截取前 50 + "..."
  │
  └── 4. 返回主题 (或 None)
```

**示例:**

输入: "我们刚刚讨论的Python排序问题"
- 匹配: "我们刚刚讨论的"
- after_match: "Python排序问题"
- 清理后: "Python排序问题"
- 返回: "Python排序问题"

### 3.4 _detect_language() — 语言检测

简单的中文检测算法：
- 统计 Unicode 范围 `\u4e00`-`\u9fff` 内的字符数
- 中文字符占比 > 30% → 中文
- 否则 → 英文

### 3.5 add_pattern() — 动态添加模式

```python
trigger.add_pattern(
    trigger_type=TriggerType.LONG_TERM_VALUE,
    pattern=r"我的名字是",
    language="cn",
)
```

运行时添加新的正则模式，立即编译并生效。

### 3.6 update_config() — 运行时更新配置

```python
trigger.update_config(
    custom_patterns=[
        {"trigger_type": "meta_cognitive", "pattern": "你还记得吗", "language": "cn"},
        {"trigger_type": "long_term_value", "pattern": "my name is", "language": "en"},
    ]
)
```

批量添加自定义规则。

## 4. 工厂函数: create_memory_trigger()

```python
trigger = create_memory_trigger(
    config_path="config/memory_trigger.yaml",  # 可选
    language="auto",
)
```

## 5. 触发检测示例

| 用户输入 | 触发类型 | 匹配模式 | 提取主题 |
|---------|---------|---------|---------|
| "我们刚才聊了什么？" | RECALL_REQUEST | `(最近\|刚才)我们(聊)了(什么)` | — |
| "请记住我喜欢Python" | LONG_TERM_VALUE | `请(记住)` | "我喜欢Python" |
| "我改变想法了，用Java吧" | STATE_CHANGE | `我(改变).*?(想法)` | "用Java吧" |
| "之前你说过的那个方案" | META_COGNITIVE | `(之前)你(说)` | "那个方案" |
| "你现在怎么看这个问题？" | OPINION_QUERY | `你(现在)(怎么)(看)` | "这个问题？" |
| "今天天气真好" | NONE | — | — |

## 6. 数据库交互

本模块不涉及数据库交互。触发结果由上层模块 (`dki_plugin.py`) 用于决定是否存储/检索记忆。

## 7. 注意事项

1. **正则表达式性能**: 所有模式在初始化时预编译 (`re.compile`)，运行时直接使用编译后的对象
2. **分类器预留**: `use_classifier` 和 `_classifier` 已预留接口，但未实现。计划集成轻量级分类器 (如 DeBERTa)
3. **主题提取的局限性**: 当前使用简单的"匹配后文本截取"，对复杂句式效果有限。后续可用 NER 或 LLM 增强
4. **优先级固定**: 触发类型的检查优先级硬编码在 `check()` 方法中，不可配置
