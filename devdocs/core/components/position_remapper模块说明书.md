# Position Remapper 模块说明书

> 文件路径: `DKI/dki/core/components/position_remapper.py`

## 1. 模块概述

本模块实现了 DKI 系统的**位置编码重映射层 (Position Remapper)**，解决 K/V 注入时的位置编码兼容性问题。不同的 LLM 使用不同的位置编码方案 (RoPE, ALiBi, Absolute)，注入外部 K/V 时需要确保位置编码的一致性。

### 支持的位置编码方案

| 方案 | 代表模型 | 处理方式 |
|------|---------|---------|
| **RoPE** | LLaMA, Qwen, Mistral, Yi, DeepSeek, Baichuan, GLM | 重映射到负/前缀位置 |
| **ALiBi** | BLOOM, MPT, Falcon | 调整注意力偏置 |
| **Absolute** | GPT 系列 | 默认处理 |

### 两种策略对比

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **Virtual Prefix** (策略 1) | 记忆映射到负位置 [-mem_len, -1] | 实现简单 | 可能超出训练分布 |
| **Actual Prefix** (策略 2) | 记忆插入位置 0，用户输入后移 | 在训练分布内 | 需要重新计算用户 RoPE |

> 当前实现使用策略 1 (Virtual Prefix)，建议生产环境通过 A/B 测试确定最优策略。

## 2. 核心类: PositionRemapper

### 2.1 初始化

```python
remapper = PositionRemapper(
    strategy="virtual_prefix",  # "virtual_prefix" 或 "actual_prefix"
    position_encoding="rope",   # "rope", "alibi", "absolute"
)
```

### 2.2 remap_for_rope() — RoPE 位置重映射

**流程:**

```
输入: K_mem [batch, heads, seq, head_dim], V_mem, mem_len, rope_cache?
  │
  ├── strategy == "virtual_prefix"
  │     记忆映射到负位置 [-mem_len, -1]
  │     对于 RoPE，这意味着计算负位置的 cos/sin
  │     大多数实现基于相对距离工作，因此直接返回
  │     → 返回 (K_mem, V_mem) 不变
  │
  ├── strategy == "actual_prefix"
  │     记忆插入位置 0，用户位置后移
  │     需要重新计算用户输入的 RoPE
  │     (当前简化实现: 直接返回)
  │     → 返回 (K_mem, V_mem) 不变
  │
  └── 未知策略 → 警告日志，返回不变
```

> **注意**: 当前两种策略的实现都是简化版本 (直接返回原始张量)。完整实现需要根据具体模型架构重新计算 RoPE 旋转。

### 2.3 remap_for_alibi() — ALiBi 注意力偏置计算

**关键算法 — ALiBi 偏置计算:**

ALiBi (Attention with Linear Biases) 通过在注意力分数上加线性偏置来编码位置信息。

**流程:**

```
输入: mem_len, user_len, num_heads, device
  │
  ├── 1. 计算 ALiBi 斜率 (每个头一个)
  │     slopes = _get_alibi_slopes(num_heads)
  │     斜率按指数递减: 2^(-8/n × i) for head i
  │
  ├── 2. 构建位置向量
  │     positions = [-mem_len, ..., -1, 0, 1, ..., user_len-1]
  │     记忆在负位置，用户在正位置
  │
  ├── 3. 计算相对距离矩阵
  │     distances[i, j] = positions[i] - positions[j]
  │     shape: [total_len, total_len]
  │
  └── 4. 计算 ALiBi 偏置
        alibi_bias = slopes × distances
        shape: [heads, total_len, total_len]
```

**ALiBi 斜率计算公式:**

对于 n 个注意力头，斜率为几何序列：

$$m_i = 2^{-\frac{8}{n} \cdot i}, \quad i = 0, 1, \ldots, n-1$$

当 n 不是 2 的幂时，使用最近的 2 的幂计算基础斜率，再补充额外斜率。

**数值示例 (8 heads):**

```
slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
```

对于 mem_len=3, user_len=4:
```
positions = [-3, -2, -1, 0, 1, 2, 3]
distances (7×7 矩阵):
  [[ 0, -1, -2, -3, -4, -5, -6],
   [ 1,  0, -1, -2, -3, -4, -5],
   [ 2,  1,  0, -1, -2, -3, -4],
   [ 3,  2,  1,  0, -1, -2, -3],
   [ 4,  3,  2,  1,  0, -1, -2],
   [ 5,  4,  3,  2,  1,  0, -1],
   [ 6,  5,  4,  3,  2,  1,  0]]
```

### 2.4 get_extended_attention_mask() — 扩展注意力掩码

**流程:**

```
输入: attention_mask [batch, user_len], mem_len, device
  │
  ├── 1. 创建记忆掩码 (全 1，始终关注记忆)
  │     mem_mask = ones(batch, mem_len)
  │
  └── 2. 拼接
        extended_mask = cat([mem_mask, attention_mask], dim=1)
        shape: [batch, mem_len + user_len]
```

### 2.5 detect_position_encoding() — 自动检测位置编码类型

根据模型名称自动推断位置编码方案：

| 模型名包含 | 检测结果 |
|-----------|---------|
| llama, qwen, mistral, yi, deepseek, baichuan, glm | `rope` |
| bloom, mpt, falcon | `alibi` |
| 其他 | `absolute` |

## 3. 配置依赖

| 配置路径 | 说明 |
|---------|------|
| `config.dki.position.strategy` | 位置重映射策略 |

## 4. 数据库交互

本模块不涉及数据库交互。

## 5. 注意事项

1. **RoPE 重映射的简化实现**: 当前 `remap_for_rope()` 直接返回原始张量，未实际执行位置重映射。完整实现需要：
   - Virtual Prefix: 为记忆 K 计算负位置的 cos/sin 旋转
   - Actual Prefix: 重新计算用户输入的 RoPE 旋转 (位置偏移 mem_len)
2. **ALiBi 斜率的非 2 幂处理**: 当 num_heads 不是 2 的幂时，使用交错采样策略补充额外斜率
3. **num_heads=0 边界情况**: `_get_alibi_slopes(0)` 返回空张量，已处理
4. **扩展掩码假设**: `get_extended_attention_mask()` 假设所有记忆位置都应被关注 (全 1 掩码)，不支持选择性记忆屏蔽
