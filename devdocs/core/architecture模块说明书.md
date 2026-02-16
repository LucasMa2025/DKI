# DKI Architecture 架构文档说明书

> 源文件: `DKI/dki/core/architecture.py`  
> 模块路径: `dki.core.architecture`  
> 文件行数: 263 行

---

## 1. 模块概述

`architecture.py` 是 DKI 系统的**活文档 (Living Documentation)** 模块，以 Python 注释和工具函数的形式记录了 DKI 论文中的核心架构设计决策。该文件不包含业务逻辑，仅提供两个文档查询函数。

---

## 2. 核心架构概念

### 2.1 DKI vs Cross-Attention 的本质区别

| 维度 | Cross-Attention | DKI |
|------|----------------|-----|
| 参数 | 独立的 W_q^cross, W_k^cross, W_v^cross | 复用自注意力的 W_k, W_v |
| 层结构 | 需要专用的交叉注意力层 | 注入到现有自注意力层 |
| 训练 | 需要训练交叉注意力权重 | 免训练，适用于冻结模型 |
| 架构 | 仅适用于 Encoder-Decoder | 适用于任何 Transformer |
| 来源 | 外部编码器输出 (不同语义空间) | 同模型的 K/V 计算 (同语义空间) |
| 控制 | 学习得到的固定权重 | 连续 α ∈ [0, 1] 控制 |
| 降级 | 需要屏蔽交叉注意力层 | α → 0 平滑恢复原始行为 |

**数学公式对比:**

- Cross-Attention:
  ```
  Attn_cross(Q, K_enc, V_enc) = softmax(Q·W_q^cross·(K_enc·W_k^cross)^T / √d) · V_enc·W_v^cross
  ```

- DKI (带 α 缩放):
  ```
  K_aug = [K_mem; K_user]  (拼接)
  V_aug = [V_mem; V_user]
  Attn_dki(Q, K_aug, V_aug, α) = softmax(Q·K_aug^T / √d + α·bias_mem) · V_aug
  ```

### 2.2 注意力预算重分配假说 (Paper Section 3.2)

**预算类型定义:**

| 预算类型 | 约束性质 | 超出后果 |
|----------|----------|----------|
| Token Budget (B_t) | 硬约束 — 上下文窗口大小 | 截断 → 信息丢失 |
| Attention Budget (B_a = n_q × n_k) | 软约束 — 计算成本 | 延迟增加但无信息丢失 |

**RAG vs DKI 预算对比:**

| 指标 | RAG | DKI |
|------|-----|-----|
| Token 占用 | B_t = n_m + n_u | B_t = n_u (记忆不占 prompt) |
| 注意力预算 | B_a = (n_m + n_u)² | B_a = n_u × (n_m + n_u) |

**核心洞察:** DKI 用注意力预算换取 Token 预算。当上下文窗口受限、记忆量大、需要推理时，释放 Token 预算的边际收益大于增加注意力预算的边际成本。

### 2.3 记忆层次与摊销 (Paper Section 7.4)

```
┌─────────────────────────────────────────────────────────────┐
│                    DKI Memory Hierarchy                      │
├─────────────────────────────────────────────────────────────┤
│  L1: GPU HBM (Hot)    - Top-k 最近使用, 未压缩 FP16        │
│  L2: CPU RAM (Warm)   - 会话活跃记忆, GEAR 压缩 (2-4×)     │
│  L3: NVMe SSD (Cold)  - 全部会话记忆, INT8 量化 (8×)       │
│  L4: Recompute        - 仅存文本+路由向量, 按需重算 K/V     │
└─────────────────────────────────────────────────────────────┘
```

**摊销公式:**
```
Amortized Cost = (C_compute + (T-1)·C_load) / T → C_load  (当 T → ∞)
```

### 2.4 设计不变量 (Paper Section 5)

1. **存储模型无关**: 记忆以文本 + 路由向量存储，K/V 按需计算
2. **注入模型一致**: K/V 必须来自目标模型 (同语义空间)
3. **会话缓存可丢弃**: 缓存仅为推理优化，清除不影响正确性
4. **优雅降级**: α → 0 平滑恢复原始 LLM 行为
5. **审计日志**: 所有注入决策均记录 (查询、记忆、α 值、门控原因)

### 2.5 查询条件投影 (Paper Section 3.4.2)

采用 **FiLM (Feature-wise Linear Modulation)** 风格的记忆中心设计:

```
X_proj = γ(X_user) ⊙ X_mem_low + β(X_user)
```

**结构约束 (隐式正则化):**
- γ 有界: 由初始化和训练动态约束
- β 零均值: 初始化为零，偏移应保持较小
- 残差主导: X_proj ≈ X_mem + ε·f(X_mem, X_user)

### 2.6 熵作为启发式代理 (Paper Section 3.5.2)

使用注意力熵作为模型不确定性的**启发式代理**:

| 场景 | 熵表现 | 是否适合 |
|------|--------|----------|
| 事实问答 | 高熵 → 认知不确定 | ✅ 适合 |
| 知识检索 | 分散注意力 → 需要锚定 | ✅ 适合 |
| 头脑风暴/创意 | 高熵是任务固有属性 | ❌ 可能误导 |
| 摘要生成 | 分散注意力是正常的 | ❌ 可能误导 |

因此系统使用**双因子门控** (熵 × 相关性) 而非单一熵门控。

---

## 3. 公共函数

### 3.1 `get_architecture_summary() -> str`

返回 DKI 架构摘要文本，包含核心机制、关键组件、设计原则和使用场景建议。

### 3.2 `get_comparison_table() -> str`

返回 DKI vs RAG vs Cross-Attention 的对比表格文本，涵盖 Token 预算、注意力预算、训练需求、架构变更、注入控制、降级方式、多轮效率、Decoder-Only 兼容性等维度。

---

## 4. 设计说明

该文件作为"活文档"，将论文中的关键理论直接嵌入代码库，确保开发者在阅读代码时能直接理解设计决策的理论依据，避免理论与实现脱节。
