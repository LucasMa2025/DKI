# Engram-Inspired DKI Full-Attention 融合分析报告

> **Date**: 2026-02-14  
> **Paper**: _Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models_ (arXiv:2601.07372)  
> **Subject**: DKI Full-Attention 注入策略借鉴 Engram 记忆处理方案的可行性分析与实现

---

## 1. 分析视角说明

本分析 **不** 将预训练模型级静态记忆 (Engram) 与用户级动态记忆 (DKI) 视为对立概念，而是从 **记忆处理的统一视角** 出发:

| 维度         | Engram (模型级)              | DKI (用户级)                | 共性                                        |
| ------------ | ---------------------------- | --------------------------- | ------------------------------------------- |
| **本质**     | 高频 N-gram 模式的结构化存储 | 用户偏好/历史的结构化存储   | 都是 "将重复模式从注意力计算中卸载"         |
| **目标**     | 释放注意力容量给全局推理     | 释放 Context 窗口给当前推理 | 都是 "让注意力专注于真正需要动态计算的部分" |
| **数据特征** | 稳定、高频、可预测           | 偏好稳定、历史动态          | 用户偏好 ≈ 个人级 "N-gram"                  |
| **挑战**     | 与模型参数无缝融合           | 与 Query 上下文自适应融合   | 都需要门控/调制机制                         |

**核心洞察**: Engram 解决的是 "如何在 Transformer 中高效注入外部记忆且不破坏原有计算"，这与 DKI 的目标完全一致。区别仅在于记忆来源 (训练语料 vs 用户数据) 和粒度 (token 级模式 vs 语义级偏好)。

---

## 2. Engram 论文核心机制

### 2.1 整体架构

```
Input Tokens → Tokenizer Compression → Multi-Head Hashing → Engram Table Lookup
                                                                    ↓
                                                          Contextualized Gating
                                                                    ↓
                                                          Conv1D Refinement
                                                                    ↓
                                                    Residual Addition to FFN Output
```

### 2.2 关键机制

#### 2.2.1 上下文门控 (Eq. 4)

```
α_t = σ(RMSNorm(h_t)^T · RMSNorm(k_t) / √d)
```

-   每个记忆 token 独立计算与当前上下文的对齐程度
-   使用 sigmoid (非 softmax)，允许多个记忆同时激活
-   RMSNorm 稳定门控计算

#### 2.2.2 卷积精化 (Eq. 5)

```
Y = SiLU(DWConv1D(RMSNorm(V̄))) + V̄
```

-   因果 Depthwise Conv1D 扩展局部感受野
-   SiLU 激活提供平滑非线性
-   残差连接确保初始阶段不破坏信号 (零初始化)

#### 2.2.3 设计原则

-   **Key 不缩放**: Key 作为注意力寻址地址，必须保持精度
-   **仅 Value 门控**: 通过调制 Value 来控制输出贡献
-   **层级选择性**: 并非所有层都需要记忆注入

### 2.3 实验关键发现

1. **有效深度增加** (Section 6.1): 注入 Engram 后，模型的 LogitLens 收敛提前约 10 层
2. **注意力容量释放** (Section 6.2): 移除低层 FFN 对性能影响减小 (Engram 补偿了 FFN 的记忆功能)
3. **CKA 跨层对应** (Section 6.3): Engram + 14 层 ≈ 无 Engram + 24 层

---

## 3. DKI 借鉴方案: Engram-Inspired Full Attention

### 3.1 借鉴映射关系

| Engram 机制                   | DKI 映射                           | 借鉴价值             | 适配难度 |
| ----------------------------- | ---------------------------------- | -------------------- | -------- |
| **Per-Token Gating**          | 偏好/历史中每个 token 的上下文门控 | ★★★★★ (核心)         | 中       |
| **Value-Only Scaling**        | 仅缩放 Value, Key 不变             | ★★★★★ (修正原有设计) | 低       |
| **Conv1D Refinement**         | 门控后 Value 的卷积精化            | ★★★★☆ (增强局部模式) | 中       |
| **Layer-Selective Injection** | 偏好 → 早期层, 历史 → 中期层       | ★★★★☆ (效率提升)     | 高       |
| **RMSNorm**                   | 门控计算中的归一化                 | ★★★☆☆ (稳定性)       | 低       |
| **Multi-Head Hashing**        | 不适用 (DKI 无需 O(1) lookup)      | N/A                  | N/A      |
| **Tokenizer Compression**     | 不适用 (DKI 使用标准分词)          | N/A                  | N/A      |

### 3.2 方案总体架构

```
┌───────────────────────────────────────────────────────────────────┐
│  Engram-Inspired DKI Full-Attention 注入流程                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: 记忆收集                                                │
│     ├─ 用户偏好 → tokenize → model.compute_kv() → K_pref, V_pref │
│     └─ 会话历史 → tokenize → model.compute_kv() → K_hist, V_hist │
│                                                                   │
│  Phase 2: 上下文感知门控 (Engram Eq. 4 适配)                      │
│     ├─ 查询表示: h_q = mean_pool(model.embed(query))              │
│     ├─ 偏好门控: α_i = σ(norm(h_q)·norm(K_pref[i])/√d) × α_base │
│     ├─ 历史门控: α_j = σ(norm(h_q)·norm(K_hist[j])/√d) × α_base │
│     └─ Value 门控: V'[i] = α_i · V[i]  (Key 不变)               │
│                                                                   │
│  Phase 3: 卷积精化 (Engram Eq. 5 适配)                            │
│     ├─ V'' = SiLU(CausalConv1D(RMSNorm(V'))) + V'               │
│     └─ 残差连接，零初始化 (初始阶段 V'' ≈ V')                     │
│                                                                   │
│  Phase 4: 层级选择性注入 (Engram Section 2.5 适配)                │
│     ├─ 早期层: 注入偏好 K/V (卸载"静态模式重建")                   │
│     ├─ 中期层: 注入历史 K/V (提供"推理上下文")                     │
│     └─ 其余层: 原始注意力 (无注入，释放计算容量)                   │
│                                                                   │
│  Phase 5: 注意力计算                                              │
│     ├─ Attn_l = softmax(Q · [K_mem; K_input]^T / √d)             │
│     │           × [V_gated_refined; V_input]                      │
│     └─ Key 保持原始精度 → 注意力分数准确                           │
│        Value 经门控+精化 → 输出贡献受控                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.3 详细注入流程

#### Step 1: 记忆收集

```python
# 与原始 DKI 相同
preference_kv = model_adapter.compute_kv(preference_text)  # List[KVCacheEntry]
history_kv = model_adapter.compute_kv(history_text)        # List[KVCacheEntry]
```

#### Step 2: 查询表示计算

```python
# 新增: 获取查询的隐藏状态表示
query_embedding = model_adapter.embed_text(query)  # [1, seq_len, hidden_dim]
query_hidden = query_embedding.mean(dim=1)         # [1, hidden_dim]
query_hidden = query_hidden.view(1, num_heads, head_dim)  # [1, heads, head_dim]
```

#### Step 3: 上下文感知门控

对每层 K/V 独立计算门控:

```python
for layer_idx, entry in enumerate(kv_entries):
    K_mem = entry.key    # [1, heads, mem_len, head_dim]
    V_mem = entry.value  # [1, heads, mem_len, head_dim]

    # RMSNorm 归一化 (Engram 推荐)
    q_norm = rmsnorm(query_hidden)        # [1, heads, head_dim]
    k_norm = rmsnorm(K_mem)               # [1, heads, mem_len, head_dim]

    # 逐 token 对齐分数
    q_expanded = q_norm.unsqueeze(2)      # [1, heads, 1, head_dim]
    scores = (q_expanded * k_norm).sum(-1)  # [1, heads, mem_len]
    scores = scores / (sqrt(head_dim) * temperature)

    # Sigmoid 门控 (非 softmax)
    gates = sigmoid(scores) * base_alpha  # [1, heads, mem_len]

    # 仅门控 Value (Key 保持不变)
    V_gated = V_mem * gates.unsqueeze(-1)  # [1, heads, mem_len, head_dim]
    # K_mem 不变
```

**为什么用 sigmoid 而非 softmax**:

-   Softmax 是 "竞争性" 门控 (所有 token 争夺注意力份额)
-   Sigmoid 是 "独立性" 门控 (每个 token 独立决定是否激活)
-   用户偏好中 "喜欢辣食" 和 "偏爱中餐" 都可能同时与食物查询高度相关

#### Step 4: 卷积精化

```python
# 对门控后的 V_gated 进行因果卷积
V_normed = rmsnorm(V_gated)

# Reshape 为 Conv1D 期望的格式
V_conv = V_normed.reshape(batch*heads, mem_len, dim).transpose(1, 2)

# 因果左填充 + 卷积
V_conv = F.pad(V_conv, ((kernel_size-1)*dilation, 0))
V_conv = conv1d(V_conv)  # Depthwise Conv1D

# SiLU 激活 + 残差
V_refined = silu(V_conv) + V_gated  # 零初始化确保初始 V_refined ≈ V_gated
```

**为什么需要卷积精化**:

-   记忆中 "用户" "喜欢" "辣食" 三个 token 有局部依赖
-   单独的门控是逐 token 独立的，无法捕获这种依赖
-   轻量因果卷积 (kernel=4) 以最小开销扩展感受野

#### Step 5: 层级注入

```python
# 偏好 → 早期层 (如 L0, L3, L6)
# 历史 → 中期层 (如 L9, L15)
# 其余层: 原始注意力

for layer_idx in range(num_layers):
    if layer_idx in preference_layers:
        # 将偏好 K/V 拼接到该层的 KV Cache
        K_layer = concat([K_pref_gated, K_original], dim=seq_len)
        V_layer = concat([V_pref_refined, V_original], dim=seq_len)
    elif layer_idx in history_layers:
        # 将历史 K/V 拼接到该层的 KV Cache
        K_layer = concat([K_hist_gated, K_original], dim=seq_len)
        V_layer = concat([V_hist_refined, V_original], dim=seq_len)
    else:
        # 原始注意力 (无注入)
        K_layer = K_original
        V_layer = V_original
```

**层级策略的理论基础** (来自 Engram Section 6):

-   **早期层** 主要负责 pattern matching 和 "记忆查找"
    → 适合注入偏好 (稳定的、类似查找表的信息)
-   **中期层** 主要负责 reasoning 和 "上下文整合"
    → 适合注入历史 (需要推理的上下文信息)
-   **后期层** 主要负责 generation 和 "输出格式化"
    → 不注入，避免干扰生成质量

---

## 4. 与原始 DKI Full-Attention 的对比

| 维度           | 原始 DKI         | Engram-Inspired DKI          |
| -------------- | ---------------- | ---------------------------- |
| **α 缩放粒度** | 全局统一 α       | 逐 token 上下文门控          |
| **α 作用对象** | K 和 V 都缩放 ❌ | 仅缩放 V (Key 不变) ✅       |
| **Value 处理** | 原始直出         | 卷积精化 + 残差              |
| **注入层策略** | 所有层统一注入   | 层级选择性注入               |
| **门控归一化** | 无               | RMSNorm                      |
| **门控函数**   | N/A (直接乘 α)   | Sigmoid (逐 token)           |
| **感受野**     | 单 token         | 扩展 (Conv kernel=4)         |
| **参数开销**   | 0                | ~1KB (RMSNorm + Conv1D 参数) |

### 4.1 关键修正: Value-Only Scaling

原始实现中 `_merge_kv` 和 `_scale_kv` 同时缩放 K 和 V:

```python
# 原始 (有问题)
h_k = h_k * history_alpha  # ❌ Key 被缩放，降低注意力匹配精度
h_v = h_v * history_alpha
```

修正后 (借鉴 Engram 设计原则):

```python
# 修正 (Value-Only)
# h_k 不变 (保持注意力寻址精度)
h_v = h_v * history_alpha  # ✅ 仅 Value 被缩放
```

**理论依据**: 在标准注意力 `Attn = softmax(QK^T/√d) · V` 中:

-   `QK^T` 计算的是 "注意力分数" (寻址匹配度)
-   缩放 K 会人为降低匹配分数，使记忆 token 难以被正确寻址
-   缩放 V 仅调制匹配后的输出贡献，不影响寻址过程

---

## 5. 实现文件说明

### 5.1 新增文件

-   **`engram_inspired_injector.py`**: 完整的 Engram-Inspired 注入器实现
    -   `RMSNorm`: 归一化模块
    -   `ContextAwareGating`: 上下文感知逐 token 门控
    -   `ConvolutionRefinement`: 因果卷积精化
    -   `EngramInspiredFullAttentionInjector`: 主注入器
    -   `EngramInspiredConfig`: 配置类
    -   `EngramInjectionResult`: 结果类 (含门控统计)
    -   `LayerInjectionPolicy`: 层级注入策略

### 5.2 修改文件

-   **`full_attention_injector.py`**:
    -   `_merge_kv`: 修正为 Value-Only Scaling
    -   `_scale_kv`: 修正为 Value-Only Scaling
-   **`__init__.py`**: 导出新增模块

---

## 6. 使用方式

### 6.1 基础使用

```python
from dki.core.injection import (
    EngramInspiredFullAttentionInjector,
    EngramInspiredConfig,
    GatingMode,
    RefinementMode,
)

# 创建配置
config = EngramInspiredConfig(
    gating_mode=GatingMode.CONTEXT_AWARE,
    refinement_mode=RefinementMode.CONV1D,
    preference_base_alpha=0.4,
    history_base_alpha=0.3,
)

# 初始化注入器
injector = EngramInspiredFullAttentionInjector(
    config=config,
    language="cn",
    head_dim=128,    # hidden_dim / num_heads
    num_heads=32,    # 模型注意力头数
    num_layers=30,   # 模型总层数
)

# 执行注入
result = injector.inject(
    model_adapter=model_adapter,
    preference_text="用户喜欢辣食，偏爱中餐",
    history_messages=[
        {"role": "user", "content": "推荐一家餐厅"},
        {"role": "assistant", "content": "推荐火锅店..."},
    ],
    query="还有什么好吃的吗？",
)

# 使用结果
if result.success and result.layer_kv:
    for layer_idx, (k_mem, v_mem) in result.layer_kv.items():
        # 将 k_mem, v_mem 注入到对应层的 KV Cache
        model.inject_kv_at_layer(layer_idx, k_mem, v_mem)
```

### 6.2 三种门控模式

```python
# 模式 1: UNIFORM (原始 DKI 行为，全局统一 α)
config = EngramInspiredConfig(gating_mode=GatingMode.UNIFORM)

# 模式 2: CONTEXT_AWARE (Engram-inspired，逐 token 门控)
config = EngramInspiredConfig(gating_mode=GatingMode.CONTEXT_AWARE)

# 模式 3: HYBRID (全局 α × 逐 token 门控)
config = EngramInspiredConfig(gating_mode=GatingMode.HYBRID)
```

### 6.3 从 YAML 配置

```yaml
injection:
    strategy: engram_inspired
    engram_inspired:
        enabled: true
        gating_mode: context_aware
        gating_temperature: 1.0
        use_rmsnorm: true
        refinement_mode: conv1d
        conv_kernel_size: 4
        preference:
            position_start: -100
            base_alpha: 0.4
            max_tokens: 100
        history:
            position_start: -500
            base_alpha: 0.3
            max_tokens: 400
            max_messages: 10
        layer_policy:
            preference_layer_ratios: [0.0, 0.1, 0.2]
            history_layer_ratios: [0.3, 0.5]
            per_layer_gating: true
        safety:
            max_total_kv_tokens: 600
            fallback_to_stable: true
```

---

## 7. 性能分析

### 7.1 额外计算开销

| 组件                 | 操作             | 复杂度             | 估算延迟 (GPU) |
| -------------------- | ---------------- | ------------------ | -------------- |
| Query Representation | Mean Pooling     | O(seq_q × d)       | < 0.1ms        |
| RMSNorm (门控)       | 2× (query + key) | O(mem_len × d)     | < 0.1ms        |
| Dot Product (门控)   | QK^T per-token   | O(mem_len × d)     | < 0.1ms        |
| Sigmoid              | Element-wise     | O(mem_len)         | < 0.01ms       |
| Conv1D (精化)        | Depthwise Causal | O(mem_len × d × k) | < 0.2ms        |
| **总计**             |                  |                    | **< 0.5ms**    |

### 7.2 参数开销

| 模块               | 参数量 (head_dim=128, kernel=4) |
| ------------------ | ------------------------------- |
| RMSNorm (query)    | 128                             |
| RMSNorm (key)      | 128                             |
| RMSNorm (conv)     | 128                             |
| Conv1D (depthwise) | 128 × 4 + 128 = 640             |
| **总计**           | **~1,024 (4 KB)**               |

**结论**: 额外开销可以忽略不计 (< 0.5ms 延迟, 4KB 内存)。

---

## 8. 客观评估: 借鉴限制与注意事项

### 8.1 不适用的 Engram 机制

1. **Multi-Head Hashing**: Engram 使用哈希实现 O(1) 查找，DKI 的记忆量级 (百~千 token) 不需要
2. **Tokenizer Compression**: Engram 合并相邻 token 以增大 N-gram 跨度，DKI 使用标准分词即可
3. **Sparsity Allocation**: Engram 与 MoE 的容量分配问题不适用于 DKI 的推理时注入场景

### 8.2 潜在风险

1. **层级注入需要模型修改**: 当前 HuggingFace API 不直接支持 per-layer K/V 注入

    - **缓解**: 可通过 monkey-patching attention layers 或使用 `transformers` 的 `output_hidden_states` hook
    - **降级方案**: 不使用层级注入，退回到全层注入 + Value-Only Gating

2. **门控模块需要训练**: `ContextAwareGating` 的 RMSNorm 参数需要微调

    - **缓解**: 使用零初始化 + 残差连接，初始行为等价于原始 DKI
    - **降级方案**: 使用 HYBRID 模式，门控作为软性调制

3. **卷积精化的适用性**: Conv1D 对极短记忆 (< 4 token) 效果有限
    - **缓解**: kernel_size 应 ≤ 记忆最小长度

### 8.3 建议的渐进式实施路径

```
Phase 0 (已完成): 修正 Value-Only Scaling  ← 零成本，立即改善
Phase 1: 启用 CONTEXT_AWARE 门控          ← 核心价值，中等难度
Phase 2: 添加 Conv1D 精化                 ← 增量改善，低难度
Phase 3: 实现层级选择性注入               ← 需要模型层 hook，高难度
```

---

## 9. 结论

从记忆处理的统一视角来看，Engram 论文的以下机制对 DKI Full-Attention 注入具有 **明确的借鉴价值**:

1. ✅ **上下文感知门控**: 逐 token 门控比全局 α 更精细，能自动抑制无关记忆 token
2. ✅ **Value-Only Scaling**: 修正了原始 DKI 同时缩放 K/V 的设计缺陷
3. ✅ **卷积精化**: 扩展记忆 token 的局部感受野，捕获 N-gram 模式
4. ⚠️ **层级注入**: 理论价值高，但需要模型层级 hook 支持，建议作为 Phase 3 实现

**总体评价**: 融合方案是 **合理且有价值的**，额外开销极小 (< 0.5ms, 4KB)，核心改进 (Value-Only Scaling) 可以立即应用。Engram 的记忆处理智慧——"让记忆根据上下文自适应调制，而非一刀切"——完全适用于 DKI 的用户级记忆场景。
