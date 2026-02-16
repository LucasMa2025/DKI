# FullAttentionInjector 全注意力注入器模块说明书

> 源文件: `DKI/dki/core/injection/full_attention_injector.py`  
> 模块路径: `dki.core.injection.full_attention_injector`  
> 文件行数: 557 行

---

## 1. 模块概述

`FullAttentionInjector` 是 DKI 论文**方案 C (Plan C)** 的研究实现，核心创新在于将**历史消息也通过 K/V 注入**（而非 Suffix Prompt），实现 **0% Context 占用**。

**与 Stable 策略对比:**

| 维度 | Stable 策略 | Full Attention 策略 |
|------|------------|-------------------|
| 偏好注入 | K/V (负位置) | K/V (负位置) |
| 历史注入 | Suffix Prompt (正位置) | K/V (负位置) ← 核心区别 |
| Context 占用 | 中等 (历史占用) | 极小 (仅全局指示 3-5 tokens) |
| 可引用性 | 历史可被模型引用 | 历史不可直接引用 |
| 稳定性 | 生产推荐 | 研究实验 |
| 降级机制 | 无 | 自动回退到 Stable |

---

## 2. 数据结构

### 2.1 PositionMode (位置编码模式)

| 模式 | 值 | 说明 |
|------|-----|------|
| `FIXED_NEGATIVE` | `"fixed_negative"` | C1: 固定负位置 RoPE |
| `CONSTANT` | `"constant"` | C1 变体: 所有 KV 使用相同位置 |
| `NOPE` | `"nope"` | C1 变体: 不应用位置编码 (NoPE) |

### 2.2 FullAttentionConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用 |
| `position_mode` | `FIXED_NEGATIVE` | 位置编码模式 |
| `preference_position_start` | `-100` | 偏好起始位置 |
| `preference_alpha` | `0.4` | 偏好注入强度 |
| `preference_max_tokens` | `100` | 偏好最大 token |
| `history_position_start` | `-500` | 历史起始位置 |
| `history_alpha` | `0.3` | 历史注入强度 |
| `history_max_tokens` | `400` | 历史最大 token |
| `history_max_messages` | `10` | 最大消息数 |
| `global_indication_enabled` | `True` | 启用全局指示 |
| `global_indication_en` | `"[Memory Context Available]"` | 英文全局指示 |
| `global_indication_cn` | `"[记忆上下文可用]"` | 中文全局指示 |
| `max_total_kv_tokens` | `600` | K/V 总 token 上限 |
| `fallback_to_stable` | `True` | 超限时降级到 Stable |
| `log_attention_patterns` | `True` | 记录 attention pattern |

### 2.3 InjectionResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | 是否成功 |
| `preference_kv` | `Optional[List[Tuple[Tensor, Tensor]]]` | 偏好 K/V |
| `history_kv` | `Optional[List[Tuple[Tensor, Tensor]]]` | 历史 K/V |
| `merged_kv` | `Optional[List[Tuple[Tensor, Tensor]]]` | 合并后 K/V |
| `preference_positions` | `Optional[List[int]]` | 偏好位置序列 |
| `history_positions` | `Optional[List[int]]` | 历史位置序列 |
| `preference_tokens` | `int` | 偏好 token 数 |
| `history_tokens` | `int` | 历史 token 数 |
| `total_kv_tokens` | `int` | 总 K/V token 数 |
| `global_indication` | `str` | 全局指示文本 |
| `position_mode` | `str` | 位置模式 |
| `fallback_triggered` | `bool` | 是否触发降级 |
| `error_message` | `str` | 错误信息 |
| `compute_time_ms` | `float` | 计算耗时 |

---

## 3. 位置布局设计

### 3.1 注意力序列位置分布

```
位置轴:
  -500        -100         -1    0         L
   │           │            │    │         │
   ├───────────┼────────────┤    ├─────────┤
   │ History   │ Preference │    │ Query + │
   │ K/V       │ K/V        │    │ Global  │
   │ α=0.3     │ α=0.4      │    │ α=1.0   │
   └───────────┴────────────┘    └─────────┘
       负位置区域                    正位置区域
```

### 3.2 位置计算算法

```python
def _compute_positions(start, num_tokens):
    """
    从 start 开始，生成 num_tokens 个连续位置
    
    示例:
      start=-100, num_tokens=50 → [-100, -99, ..., -51]
      start=-500, num_tokens=200 → [-500, -499, ..., -301]
    """
    return list(range(start, start + num_tokens))
```

### 3.3 三种位置编码模式

**FIXED_NEGATIVE (默认):**
```
偏好位置: [-100, -99, -98, ..., -51]  (50 tokens)
历史位置: [-500, -499, -498, ..., -301]  (200 tokens)
→ 使用负位置的 RoPE 编码
```

**CONSTANT:**
```
所有 K/V token 使用相同位置: -10500
→ 消除位置信息的影响，纯语义注意力
```

**NOPE (No Position Encoding):**
```
不应用任何位置编码
→ 完全依赖内容语义，无位置偏置
```

---

## 4. inject() 主流程

```
FullAttentionInjector.inject(model_adapter, preference_text, history_messages, query)
  │
  ├─ Step 1: 估算 token 数量
  │   ├─ preference_tokens = _estimate_tokens(preference_text)
  │   ├─ history_text = _format_history_for_kv(history_messages)
  │   └─ history_tokens = _estimate_tokens(history_text)
  │
  ├─ Step 2: 安全检查
  │   ├─ total_tokens > max_total_kv_tokens?
  │   │   ├─ fallback_to_stable=True → 触发降级，返回 fallback_triggered=True
  │   │   └─ fallback_to_stable=False → 截断历史消息
  │   └─ 通过 → 继续
  │
  ├─ Step 3: 计算偏好 K/V
  │   ├─ model_adapter.compute_kv(preference_text) → kv_entries
  │   ├─ _compute_positions(-100, preference_tokens) → [-100, -99, ...]
  │   └─ _apply_position_encoding(kv, positions, model) → 位置编码后的 K/V
  │
  ├─ Step 4: 计算历史 K/V (核心创新)
  │   ├─ model_adapter.compute_kv(history_text) → kv_entries
  │   ├─ _compute_positions(-500, history_tokens) → [-500, -499, ...]
  │   └─ _apply_position_encoding(kv, positions, model) → 位置编码后的 K/V
  │
  ├─ Step 5: 合并 K/V
  │   └─ _merge_kv(preference_kv, history_kv, pref_alpha, hist_alpha)
  │       ├─ 仅缩放 Value (Key 不变)
  │       │   ├─ h_v = h_v * history_alpha
  │       │   └─ p_v = p_v * preference_alpha
  │       └─ 拼接: [History_KV, Preference_KV]
  │           ├─ merged_k = cat([h_k, p_k], dim=2)
  │           └─ merged_v = cat([h_v, p_v], dim=2)
  │
  ├─ Step 6: 生成全局指示
  │   └─ "[记忆上下文可用]" (cn) 或 "[Memory Context Available]" (en)
  │
  └─ Step 7: 返回 InjectionResult
```

---

## 5. 关键算法详解

### 5.1 K/V 合并与 Alpha 缩放

```python
def _merge_kv(preference_kv, history_kv, preference_alpha, history_alpha):
    """
    合并策略: 历史在前，偏好在后
    
    关键设计: 仅缩放 Value, Key 不变
    
    原因:
    - Key 作为注意力寻址地址，缩放会干扰匹配精度
    - Value 承载输出贡献，通过 α 调制记忆影响强度
    - 参考: Engram 论文 (arXiv:2601.07372) Section 2.3
    
    数学表示:
      K_merged = [K_history ; K_preference]  (不缩放)
      V_merged = [α_hist * V_history ; α_pref * V_preference]  (缩放)
    """
    for layer_idx in range(num_layers):
        h_k, h_v = history_kv[layer_idx]
        p_k, p_v = preference_kv[layer_idx]
        
        # 仅缩放 Value
        h_v = h_v * history_alpha    # 历史影响强度
        p_v = p_v * preference_alpha  # 偏好影响强度
        
        # 拼接 (历史在前，偏好在后)
        merged_k = torch.cat([h_k, p_k], dim=2)
        merged_v = torch.cat([h_v, p_v], dim=2)
```

**为什么仅缩放 Value?**

在注意力计算中:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

- `Q * K^T` 计算注意力权重 (哪些位置与查询相关)
- 如果缩放 K，会改变注意力权重分布，可能导致模型"找不到"记忆
- 缩放 V 只改变输出贡献的强度，不影响注意力寻址

### 5.2 位置编码应用

```python
def _apply_position_encoding(kv_list, positions, model_adapter):
    """
    根据 position_mode 应用位置编码:
    
    NOPE: 直接返回原始 K/V (不应用位置编码)
    CONSTANT: 所有 token 使用相同位置 (消除位置信息)
    FIXED_NEGATIVE: 使用负位置的 RoPE
    
    如果模型不支持 apply_rope_to_kv 方法，返回原始 K/V
    """
    if position_mode == NOPE:
        return kv_list  # 无位置编码
    
    if position_mode == CONSTANT:
        positions = [constant_pos] * len(positions)  # 统一位置
    
    if hasattr(model_adapter, 'apply_rope_to_kv'):
        return model_adapter.apply_rope_to_kv(kv_list, positions)
    
    return kv_list  # 模型不支持，返回原始
```

### 5.3 历史消息格式化 (K/V 专用)

```python
def _format_history_for_kv(messages):
    """
    K/V 注入专用格式化 (不使用 suffix prompt 模板)
    
    与 Stable 策略的区别:
    - Stable: 包含 [会话历史参考]...[会话历史结束] 模板
    - Full Attention: 仅简洁的角色+内容格式
    
    原因: K/V 注入不需要显式提示词，模型通过注意力机制隐式理解
    """
    lines = []
    for msg in messages[-max_messages:]:
        role_label = "用户" if role == "user" else "助手"
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)
```

**格式化示例:**
```
用户: 推荐一家北京的餐厅
助手: 推荐海底捞，他们可以根据过敏情况定制菜单。
用户: 那家店在哪里?
助手: 海底捞在朝阳区有多家分店，最近的在望京。
```

### 5.4 历史截断算法

```python
def _truncate_history(messages, max_tokens):
    """
    从最近的消息开始保留，直到达到 token 上限
    
    策略: 保留最近的消息 (时间优先)
    """
    result = []
    total_tokens = 0
    
    for msg in reversed(messages):  # 从最近开始
        tokens = _estimate_tokens(msg.content)
        if total_tokens + tokens > max_tokens:
            break
        result.insert(0, msg)
        total_tokens += tokens
    
    return result
```

---

## 6. 安全机制

### 6.1 Token 上限检查

```
total_tokens = preference_tokens + history_tokens
if total_tokens > max_total_kv_tokens (默认 600):
    ├─ fallback_to_stable=True → 触发降级，返回给 DKIPlugin
    │   └─ DKIPlugin._inject_full_attention() 检测到 fallback_triggered
    │       └─ 自动回退到 _inject_stable()
    └─ fallback_to_stable=False → 截断历史消息
```

### 6.2 异常处理

```
inject() 中的任何异常:
  └─ 返回 InjectionResult(success=False, error_message=str(e))
      └─ DKIPlugin 检测到 success=False → 回退到 Stable 策略
```

---

## 7. 研究数据收集

### 7.1 Attention Pattern 日志

每次成功注入后记录:

```json
{
  "timestamp": 1708000000.0,
  "query": "推荐一家餐厅...",
  "position_mode": "fixed_negative",
  "preference_tokens": 50,
  "history_tokens": 200,
  "preference_positions": [-100, -99, -98, -97, -96],
  "history_positions": [-500, -499, -498, -497, -496],
  "compute_time_ms": 15.3
}
```

### 7.2 统计数据

| 指标 | 说明 |
|------|------|
| `total_injections` | 总注入次数 |
| `successful_injections` | 成功次数 |
| `fallback_count` | 降级次数 |
| `avg_preference_tokens` | 平均偏好 token 数 |
| `avg_history_tokens` | 平均历史 token 数 |
| `avg_compute_time_ms` | 平均计算耗时 |

---

## 8. YAML 配置示例

```yaml
dki:
  injection_strategy: full_attention
  
  full_attention:
    enabled: true
    position_mode: fixed_negative
    
    preference:
      position_start: -100
      alpha: 0.4
      max_tokens: 100
    
    history:
      position_start: -500
      alpha: 0.3
      max_tokens: 400
      max_messages: 10
    
    global_indication:
      enabled: true
      text_en: "[Memory Context Available]"
      text_cn: "[记忆上下文可用]"
    
    safety:
      max_total_kv_tokens: 600
      fallback_to_stable: true
      log_attention_patterns: true
```

---

## 9. 运行时配置更新

```python
# 通过 DKIPlugin 更新
plugin.update_full_attention_config(
    position_mode="nope",           # 切换到 NoPE 模式
    preference_alpha=0.3,           # 降低偏好影响
    history_alpha=0.5,              # 提高历史影响
    history_position_start=-300,    # 调整历史起始位置
    max_total_kv_tokens=800,        # 提高 token 上限
)
```
