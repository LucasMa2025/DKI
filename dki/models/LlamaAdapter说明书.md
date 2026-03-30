# LlamaAdapter 适配器说明书

**版本**: 本轮修复版（alpha_cap=1.0 / system-template KV / LogitBias 注入）  
**文件**: `dki/models/llama_adapter.py`  
**依赖**: `dki/models/base.py`（`BaseModelAdapter` / `KVCacheEntry` / `build_dynamic_cache_from_entries`）

---

## 一、设计定位

`LlamaAdapter` 是 DKI 系统中**唯一实现真正 K/V 注入**的模型适配器。

区别于 vLLM/SGLang 适配器（通过提示词前缀间接实现），`LlamaAdapter` 直接操作 HuggingFace Transformers 的 `past_key_values` 机制，将用户偏好的 K/V 表示注入到自注意力层，**绕过 context window token 预算**，实现论文 §4.2 描述的偏好隐式背景影响。

**支持范围（严格限定）**：仅 LLaMA 3.1 系列
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`  
- `meta-llama/Llama-3.1-405B-Instruct`

不支持 LLaMA 3.2/3.3+、Qwen、Mistral 等其他模型族。

---

## 二、架构概览

```
用户偏好文本
     │
     ▼
compute_kv()                          ← 偏好编码（system template 包裹）
     │  K/V 存储 CPU，跨 session 复用
     ▼
forward_with_kv_injection()           ← KV 注入推理（主路径）
     ├── _prepare_kv_for_injection()  ← Value 缩放 α，Key 不变
     ├── _build_attention_bias()      ← B_alpha 矩阵（prefill）
     ├── _build_decode_mask()         ← decode 阶段 alpha bias
     └── model.forward() × (1 + T)   ← 1 次 prefill + T 次 decode

或：
compute_pref_embedding()              ← 偏好压缩为单向量
     │
     ▼
forward_with_logit_bias_injection()   ← LogitBias 注入（实验性）
     └── _compute_logit_bias_vector() ← lm_head.weight @ pref_emb
```

---

## 三、核心机制详解

### 3.1 偏好 KV 计算（`compute_kv`）

**System Template 包裹**是本版本最关键的修复：

```python
# 包裹前（裸文本，旧版行为）
"用户名: Lucas，偏好: 简洁回答"

# 包裹后（system template，当前行为）
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
 用户名: Lucas，偏好: 简洁回答<|eot_id|>"
```

**语义意义**：LLaMA 3.1 的 RLHF/SFT 训练数据中，system 级指令始终以该格式呈现。加入特殊 token（`<|start_header_id|>system<|end_header_id|>`）后，模型的 embedding 层会激活"处理 system 级指令"的内部表示，使 K/V 进入与推理时 chat template 对齐的语义子空间。裸文本 KV 与 chat template 推理序列存在分布漂移，导致注入偏好不被模型识别。

**安全截断**：超过 `MAX_PREF_TOKENS=300` 时截断，防止 OOD 风险。

**返回值**：`List[KVCacheEntry]`，每层一个，存储在 CPU，支持跨 session 复用。

**消融实验**：通过 `use_system_template=False` 可切换为裸文本路径，用于对比实验。

### 3.2 K/V 注入推理（`_forward_with_bias_impl`）

这是论文 §4.2 公式的完整实现：

$$\text{Attn}_{\text{DKI}} = \text{Softmax}\!\left(\frac{Q\,[K_p;\,K_u]^\top}{\sqrt{d}} + \mathbf{B}_\alpha\right)\![\alpha \cdot V_p;\, V_u]$$

**执行流程**：

```
Step 1 — 准备偏好 KV
  KVCacheEntry list  →  build_dynamic_cache_from_entries()
  Value  ×= α（截断到 alpha_cap）
  Key 不变（保护 attention 寻址精度）

Step 2 — 构建 Prefill attention bias（B_alpha）
  bias: [1, 1, n_query, n_pref + n_query]
  偏好位置 [0..n_pref-1]: += log(α)   （α<1 时抑制，α=1 时 bias=0）
  查询内部:   causal mask（torch.triu 向量化，非 for 循环）

Step 3 — Prefill
  model.forward(
      input_ids     = query_tokens,
      past_key_values = pref_kv,         # 偏好 KV 作为前缀
      attention_mask  = bias_4d,         # 4D float bias
      position_ids    = [mem_len .. mem_len+n_query-1],  # 显式指定，5.x 兼容
  )
  → 得到 full_kv（pref + query 合并）和 first_logits

Step 4 — Decode loop（逐 token）
  for step in range(max_new_tokens):
      next_token = top_p_sample(logits)
      decode_mask: [1,1,1, cache_len+1]，偏好位置施加 log(α)
      decode_pos:  [[mem_len + n_query + step]]   # 显式指定
      model.forward(next_token, full_kv, decode_mask, decode_pos)
      → 追加新 KV，更新 logits
```

**关键设计决策**：

| 决策 | 理由 |
|------|------|
| 手动 prefill+decode，不用 `model.generate()` | 避免 Transformers 5.x 内部推断 position_ids 失效 |
| Key 不缩放，只缩放 Value | Key 决定"模型关注哪里"，缩放会破坏 attention 寻址 |
| 4D float bias（非 bool mask） | 允许连续强度控制；bool mask 只能二值屏蔽 |
| 显式 position_ids | 解决 5.x generate() 不自动推断 past_kv_length 的 bug |
| `torch.triu` 向量化 causal mask | 替代 O(n²) Python for 循环，n_query 大时性能显著提升 |

### 3.3 Alpha 控制语义

| alpha 值 | B_alpha 偏好位置 bias | Value 缩放 | 效果 |
|---------|----------------------|-----------|------|
| 0.0 | -∞ | 0× | 偏好完全屏蔽，退化为 vanilla LLM |
| 0.3 | log(0.3) ≈ -1.20 | 0.3× | 强抑制，偏好影响较弱 |
| 0.5 | log(0.5) ≈ -0.69 | 0.5× | 中等强度（论文推荐基础值） |
| 0.7 | log(0.7) ≈ -0.36 | 0.7× | 较强，论文建议上限 |
| 1.0 | 0（等权） | 1× | 最强，偏好与 query 完全等权 |

`alpha_cap`（实例级，默认 1.0）可在构造时指定，超出则 WARNING 并裁剪。

### 3.4 LogitBias 注入（实验性，`forward_with_logit_bias_injection`）

不同于 KV 注入（在 attention 层注入），LogitBias 在**输出词表层**注入偏好影响：

$$\text{logits\_adjusted} = \text{logits} + \lambda \cdot (\mathbf{W}_{\text{lm}} \cdot \mathbf{e}_{\text{pref}})$$

其中 $\mathbf{e}_{\text{pref}} \in \mathbb{R}^{H}$ 是偏好嵌入（mean pooling），$\mathbf{W}_{\text{lm}} \in \mathbb{R}^{V \times H}$ 是 lm_head 权重。

**优点**：计算代价极低（每 step 一次矩阵向量乘），无需修改 attention 结构。  
**缺点**：只影响词表分布，不能引导 attention 关注模式，偏好表达力弱于 KV 注入。  
**适用场景**：快速实验、对比验证、或在不支持 KV 注入的场景下的降级选项。

**λ 建议范围**：0.05（轻微）/ 0.10（默认）/ 0.30（强引导）/ >0.5（可能破坏流畅性）

---

## 四、公开接口列表

### 4.1 偏好编码

| 方法 | 返回 | 说明 |
|------|------|------|
| `compute_kv(text, return_hidden, use_system_template)` | `(List[KVCacheEntry], Optional[Tensor])` | 偏好→K/V，主路径 |
| `compute_pref_embedding(text, use_system_template, pool)` | `Tensor [hidden_dim]` | 偏好→嵌入向量，LogitBias 路径 |

### 4.2 K/V 注入推理

| 方法 | 类型 | 说明 |
|------|------|------|
| `forward_with_kv_injection(prompt, injected_kv, alpha, max_new_tokens)` | 非流式 | 主接口，Executor 调用 |
| `forward_with_kv_injection_and_bias(...)` | 非流式 | 向后兼容别名，等价于上方法 |
| `stream_generate_with_kv_injection(prompt, injected_kv, alpha, ...)` | 同步流式 | yield str chunks |
| `async_stream_generate_with_kv_injection(prompt, injected_kv, alpha, ...)` | 异步流式 | async yield str chunks |

### 4.3 LogitBias 注入推理

| 方法 | 类型 | 说明 |
|------|------|------|
| `forward_with_logit_bias_injection(prompt, pref_emb, lambda_logit_bias, ...)` | 非流式 | — |
| `stream_generate_with_logit_bias(prompt, pref_emb, lambda_logit_bias, ...)` | 同步流式 | — |
| `async_stream_generate_with_logit_bias(prompt, pref_emb, lambda_logit_bias, ...)` | 异步流式 | — |

### 4.4 标准推理（无注入）

| 方法 | 说明 |
|------|------|
| `generate(prompt, max_new_tokens, temperature, top_p)` | 无注入推理，alpha≤0.01 时自动降级 |
| `stream_generate(...)` | 同步流式，无注入 |
| `async_stream_generate(...)` | 异步流式，无注入 |

### 4.5 辅助接口

| 方法 | 说明 |
|------|------|
| `load()` | 加载模型，仅接受 LLaMA 3.1 系列 |
| `embed(text)` | 文本嵌入（mean pooling） |
| `compute_prefill_entropy(text, layer_idx)` | 计算 prefill 注意力熵（门控信号） |
| `get_model_info()` | 返回模型架构和配置信息 |

---

## 五、量化支持

| 量化模式 | BitsAndBytes | 备注 |
|---------|-------------|------|
| `none` | 否 | 原始精度（dtype） |
| `4bit` | 是（NF4） | 推荐，内存减半 |
| `8bit` | 是（LLM.int8()） | — |
| `gptq` | 否 | 预量化模型，AutoModel 自动检测 |
| `awq` | 否 | 预量化模型，AutoModel 自动检测 |
| `fp8` | 否 | compute_dtype=bfloat16 |

> **注意**：量化模式下 `_compute_logit_bias_vector` 使用 `lm_head forward` 前向通道代替矩阵直接乘，避免 4bit 权重类型不兼容问题。

---

## 六、偏好注入的实际效果分析

### 6.1 什么时候有效

KV 注入在以下条件下效果最稳定：

1. **偏好文本 ≤100 tokens**：RoPE 位置在训练分布内，OOD 风险低。
2. **alpha ∈ [0.4, 0.7]**：论文敏感性实验的最优区间，质量峰值且幻觉受控。
3. **使用 system template 包裹**：KV 语义空间与推理时 chat template 对齐，是偏好被识别的前提。
4. **LLaMA 3.1 系列**：其他模型族未经验证，不在支持范围内。

### 6.2 什么时候效果有限

1. **alpha=1.0（等权）**：偏好 KV 与查询 KV 等权参与 attention，可能过强干扰生成逻辑。推荐从 0.5 开始调试。
2. **偏好文本过长（>200 tokens）**：截断到 300 token 上限，截断后的 KV 可能丢失关键信息；更好的做法是提前对偏好文本做结构化压缩。
3. **多轮对话中历史信息很长**：KV 注入仅影响偏好层，历史信息通过 suffix prompt 注入，当 suffix 过长时 token 预算竞争仍存在。

### 6.3 与 Prompt Prefix 方式的对比

| 维度 | KV 注入（LlamaAdapter） | Prompt Prefix（vLLM/SGLang） |
|------|------------------------|------------------------------|
| Token 消耗 | 0（不占 context window） | 偏好文本长度（例如 50~100 tokens） |
| 延迟 | 首次：KV 计算约 50ms；复用：<2ms | 首次：正常推理；复用：prefix_cache 命中 |
| 可控性 | alpha 连续控制 + B_alpha bias | 通过 strength modifier 文本控制，粒度粗 |
| 模型依赖 | 必须是开源 HF 模型 | 任何 vLLM/SGLang 支持的模型 |
| 理论保证 | 论文 Theorem 1（有界分布重加权） | 无明确理论界 |
| 生产稳定性 | 受 HF API 版本影响（5.x 需显式 position_ids） | 稳定，依赖 vLLM 成熟接口 |

### 6.4 方案的工程意义

1. **验证了 K/V 注入的理论可行性**：在 LLaMA 3.1 上实验证明，system-template 包裹的偏好 KV 可以被模型正确识别并在 attention 层产生可测量的影响，为论文 §4.2 提供了实验基础。

2. **提供了零 token 代价的偏好影响机制**：在 context 资源极度受限的场景（如嵌入式设备、4K context 模型），偏好不占 token 是关键优势。

3. **alpha 的连续可控性**：不同于 prompt 工程（只能靠词语强度离散调整），alpha 允许精确的连续控制，支持基于熵的动态门控（`compute_prefill_entropy`）。

4. **LogitBias 路径提供轻量备选**：在不需要 attention 级别精确影响时，LogitBias 方案提供了仅需 hidden_dim 大小向量的极低代价偏好注入，适合快速迭代实验。

5. **Transformers 版本兼容路径**：完整兼容 4.x 和 5.x（通过显式 position_ids、`build_dynamic_cache_from_entries`、`extract_kv_from_past` 多策略降级），提供了跨版本 KV 注入的工程参考实现。

---

## 七、已知限制与注意事项

| 限制 | 影响 | 缓解方案 |
|------|------|----------|
| 仅支持 LLaMA 3.1 | 不可用于生产多模型场景 | vLLM/SGLang adapter 作为通用路径 |
| 手动 decode loop 速度慢 | 比 model.generate() 慢约 15~30% | 流式接口下用户感知延迟不变 |
| 量化模型 KV 精度 | 4bit/8bit 量化后 KV 精度损失 | 实验场景可接受；生产用无量化模型 |
| 4D attention mask 版本依赖 | Transformers 不同版本行为有差异 | 使用 `eager` attention 实现，最稳定 |
| LogitBias 对词表的影响不可预测 | λ 过大时生成文本可能出现重复或非自然词语 | 严格限制 λ ≤ 0.3 |

---

## 八、典型使用示例

```python
from dki.models.llama_adapter import LlamaAdapter

# 初始化（alpha_cap=0.7 对应论文推荐上限）
adapter = LlamaAdapter(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    dtype="bfloat16",
    quantization="4bit",
    alpha_cap=0.7,
)
adapter.load()

# 1. 计算偏好 KV（一次计算，多轮复用）
pref_text = "用户名: Lucas，语言偏好: 中文，回答风格: 简洁"
kv_entries, _ = adapter.compute_kv(pref_text)   # system template 自动包裹

# 2. KV 注入推理（非流式）
result = adapter.forward_with_kv_injection(
    prompt="今天天气怎么样？",
    injected_kv=kv_entries,
    alpha=0.5,
    max_new_tokens=200,
)
print(result.text)

# 3. KV 注入推理（同步流式）
for chunk in adapter.stream_generate_with_kv_injection(
    prompt="请推荐一本书",
    injected_kv=kv_entries,
    alpha=0.5,
    max_new_tokens=300,
):
    print(chunk, end="", flush=True)

# 4. LogitBias 注入（实验性）
pref_emb = adapter.compute_pref_embedding(pref_text)
result = adapter.forward_with_logit_bias_injection(
    prompt="今天天气怎么样？",
    pref_emb=pref_emb,
    lambda_logit_bias=0.1,
    max_new_tokens=200,
)

# 5. 熵计算（用于门控）
entropy = adapter.compute_prefill_entropy("今天天气怎么样？", layer_idx=3)
dynamic_alpha = 0.3 + 0.4 * entropy   # 高熵 → 更强注入
```

---

## 九、错误修复记录（本版本）

| 问题 | 文件位置 | 修复方式 |
|------|---------|----------|
| `DEFAULT_LOGIT_BIAS_LAMBDA` 类常量在方法默认参数中引用导致 `NameError` | L1198/1295/1374 | 改为 `Optional[float] = None`，方法体内读 `self.DEFAULT_LOGIT_BIAS_LAMBDA` |
| `_compute_logit_bias_vector` 量化模型直接 `@ weight` 报错 | L553-557 | 优先用 `lm_head(pref_emb.unsqueeze(0))` 前向，回退才用矩阵乘 |
| `extract_kv_from_past` 策略1空列表降级 | `base.py` L300 | 加 `kc[0] is not None` 检查 |
| `PackedKV.scale_values` inplace 导致缓存复用 alpha 累乘 | `base.py` L244 | 改为返回新 `PackedKV` 对象 |
| 5.x `generate()` position_ids 推断失效 | `llama_adapter.py` L963 | 所有 prefill/decode 均显式传入 `position_ids` |
| `_build_attention_bias` causal mask Python for 循环 | `llama_adapter.py` L807 | 改用 `torch.triu` 向量化 |
