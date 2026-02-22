# DKI VLLMAdapter 改造方案

**日期**: 2026-02-22  
**状态**: 设计完成，待实施  
**影响范围**: `dki/models/vllm_adapter.py`, `dki/core/plugin/injection_executor.py`

---

## 0. 设计目标：DKI + AGA 混合部署

### 0.1 用户的真实需求

用户可能同时需要：

```
用户请求 → vLLM 推理引擎 (单一模型，高效推理)
              ├── DKI: 用户级记忆注入 (偏好 → KV 负位置, 历史 → recall 召回)
              └── AGA: 全局知识扩展 (高熵时注入领域知识 100-500 tokens/次)
```

**这两个系统的注入层次不同、目标不同、时机不同，但必须共享同一个推理引擎。**

| 维度     | DKI                          | AGA                                 |
| -------- | ---------------------------- | ----------------------------------- |
| 注入对象 | 用户级偏好 + 会话历史        | 全局领域知识                        |
| 注入时机 | 推理前（prefill 阶段）       | 推理中（每个 token 的高熵时刻）     |
| 注入方式 | `past_key_values` 负位置拼接 | `register_forward_hook` delta       |
| 数据规模 | 小（偏好文本 ~200 tokens）   | 大（知识库 100K+ tokens，分片注入） |
| 控制方式 | Alpha 缩放（0.2-0.5）        | 熵门控（自动决策）                  |
| 用户隔离 | 必须（per-user）             | 不需要（全局知识）                  |

### 0.2 为什么不能放弃 vLLM

如果放弃 vLLM，用户被迫二选一：

-   **选 HF 模型**：DKI 偏好注入精确（`past_key_values`），但无高吞吐量
-   **选 vLLM**：AGA 可以 hook 注入，但 DKI 的 `past_key_values` 不可用

**混合部署的核心价值：一个 vLLM 引擎同时承载 DKI + AGA，不浪费资源。**

### 0.3 改造后的目标架构

```
=== 目标架构: DKI + AGA 混合部署 ===

                    ┌─────────────────────────────────┐
                    │         vLLM 推理引擎            │
                    │    (PagedAttention, 连续批处理)   │
                    │                                  │
                    │  ┌──────────────────────────┐    │
                    │  │   Transformer Layer N     │    │
                    │  │                          │    │
                    │  │  ┌─ self_attn ──────────┐│    │
                    │  │  │  Q·K^T → softmax → V ││    │
                    │  │  │  ↑                    ││    │
                    │  │  │  DKI: past_key_values ││    │  ← DKI 偏好注入
                    │  │  │  (KV Cache 负位置)    ││    │    (推理前, prefill)
                    │  │  └───────────────────────┘│    │
                    │  │           ↓                │    │
                    │  │    attn_output             │    │
                    │  │           ↓                │    │
                    │  │  ┌─ AGA hook ────────────┐│    │
                    │  │  │  熵评估 → 门控决策     ││    │  ← AGA 知识注入
                    │  │  │  高熵: +delta          ││    │    (推理中, per-token)
                    │  │  │  低熵: 旁路            ││    │
                    │  │  └───────────────────────┘│    │
                    │  │           ↓                │    │
                    │  │    final_output            │    │
                    │  └──────────────────────────┘    │
                    └─────────────────────────────────┘

VRAM: ~14GB (vLLM) + ~8MB (AGA KVStore) = ~14GB
推理: vLLM 原生 (PagedAttention + 连续批处理)
DKI: 偏好通过 vLLM 内部模型的 past_key_values 注入
AGA: 知识通过 register_forward_hook 注入
```

**关键洞察：DKI 和 AGA 的注入点不冲突！**

-   DKI 注入在 **attention 计算之前**（KV Cache 负位置）
-   AGA 注入在 **attention 计算之后**（hidden_states delta）
-   两者可以在同一个 Transformer 层上共存

---

## 1. 问题诊断

### 1.1 核心问题：vLLM 引擎占用显存但几乎不被使用

DKI 的 `VLLMAdapter` 在 `load()` 时加载了两个模型：

```
VLLMAdapter.load()
  ├── self.llm = LLM(model=...)          # vLLM 引擎 (~14GB VRAM for 7B)
  └── self.hf_model = AutoModelForCausalLM(...)  # HF 模型 (按需加载)
```

但 DKI 的实际调用链是：

```
dki.chat() → executor.execute()
  ├── 有偏好 (绝大多数) → forward_with_kv_injection() → 用 self.hf_model
  └── 无偏好 (极少数)   → generate()                  → 用 self.llm (vLLM)
```

**证据链：**

1. `injection_executor.py:456` — 有偏好且 alpha > 0.1 时走 `forward_with_kv_injection()`
2. `vllm_adapter.py:296` — `forward_with_kv_injection()` 注释明确写着 "Falls back to HuggingFace model"
3. `vllm_adapter.py:298` — 第一行就是 `self._load_hf_model()`
4. `vllm_adapter.py:353` — 实际推理用 `self.hf_model.generate()`

**结果：**

-   vLLM 引擎占用 ~14GB VRAM，但 DKI 的核心功能（偏好注入）永远不走 vLLM
-   HF 模型在 VRAM 不足时被迫加载到 CPU，推理速度下降 10-100x
-   两个模型总共需要 ~28GB VRAM（7B 模型），或 14GB VRAM + 大量 CPU 内存

### 1.2 为什么会这样？

vLLM 通过 PagedAttention 管理 KV Cache，**不暴露 `past_key_values` 接口给外部**。
DKI 的偏好注入需要将预计算的 KV 对插入到 KV Cache 负位置，这在 vLLM 中无法实现。

### 1.3 其他适配器没有这个问题

| 适配器          | 模型数量     | KV 注入方式             | 推理方式               |
| --------------- | ------------ | ----------------------- | ---------------------- |
| LlamaAdapter    | 1 个 HF 模型 | `past_key_values`       | 同一模型               |
| DeepSeekAdapter | 1 个 HF 模型 | `past_key_values`       | 同一模型               |
| GLMAdapter      | 1 个 HF 模型 | `past_key_values`       | 同一模型               |
| **VLLMAdapter** | **2 个模型** | HF 的 `past_key_values` | **HF 模型（非 vLLM）** |

---

## 2. 改造方案：统一引擎 + 双层注入

### 2.1 核心思路

**不是"用 AGA 替代 DKI 的偏好注入"，而是"让 DKI 和 AGA 共享同一个 vLLM 引擎"。**

DKI 的偏好注入有两种可行路径：

| 路径       | 方式                                   | 保真度 | vLLM 兼容        | 适用场景           |
| ---------- | -------------------------------------- | ------ | ---------------- | ------------------ |
| **路径 A** | 通过 vLLM 内部模型的 `past_key_values` | 高     | 需要访问内部 API | DKI 独立部署       |
| **路径 B** | 通过 AGA hook 将偏好转为 delta 注入    | 中     | 完全兼容         | DKI + AGA 混合部署 |

**推荐：路径 A 为主，路径 B 为备选。**

### 2.2 路径 A 详解：vLLM 内部模型 + past_key_values

vLLM 的内部模型仍然是标准的 `nn.Module`，AGA 的 `VLLMAdapter.extract_model()`
已经证明可以提取它。关键问题是：**提取出的内部模型是否支持 `past_key_values`？**

```python
# AGA 已验证: 可以从 vLLM 提取内部模型
from aga.adapter.vllm import VLLMAdapter as AGAVLLMAdapter
internal_model = AGAVLLMAdapter.extract_model(llm_engine)
# internal_model 是标准 nn.Module
```

**技术分析：**

1. vLLM 内部模型的 `forward()` 方法签名与 HF 不同
    - HF: `model(input_ids, attention_mask, past_key_values=...)`
    - vLLM: `model(input_ids, positions, kv_caches, attn_metadata)`
2. vLLM 的 KV Cache 由 `CacheEngine` 管理，不接受外部 `past_key_values`
3. **但是**：vLLM 的 `generate()` 在 prefill 阶段会处理完整的 prompt
    - 如果将偏好文本作为 **prompt 前缀** 注入，效果等价于 `past_key_values`
    - 这就是 **Prompt Prefix Injection** 方案

### 2.3 Prompt Prefix Injection（推荐方案）

**核心思想：将 DKI 的偏好文本作为 prompt 前缀，让 vLLM 在 prefill 阶段自然处理。**

```
=== 当前 DKI 流程 ===
偏好文本 → HF model.forward() → KVCacheEntry → past_key_values → HF model.generate()

=== 改造后流程 ===
偏好文本 → 构造前缀 prompt → vLLM.generate(prefix + user_query)
                                    ↑
                                    AGA hook 同时在高熵时注入领域知识
```

**为什么这是可行的：**

1. `past_key_values` 的本质就是"让模型看到偏好文本但不输出它"
2. Prompt 前缀实现了完全相同的语义——模型在 prefill 阶段处理偏好文本，
   其 KV 自然进入 vLLM 的 PagedAttention KV Cache
3. Alpha 缩放可以通过 **Soft Prompt** 或 **Attention Bias** 近似实现
4. vLLM 原生支持 prompt 前缀，无需任何 hack

**Alpha 缩放的处理：**

DKI 的 alpha 控制偏好影响强度（0.2-0.5）。在 Prompt Prefix 方案中：

| Alpha 实现方式      | 描述                                            | 精度 | 复杂度 |
| ------------------- | ----------------------------------------------- | ---- | ------ |
| **Prompt 权重标记** | 在前缀中添加 "以下是用户偏好（参考权重: 30%）:" | 低   | 极低   |
| **前缀长度控制**    | alpha 越低，保留的偏好文本越少                  | 中   | 低     |
| **AGA hook alpha**  | 将偏好也注册到 AGA，由 AGA 的 reliability 控制  | 中   | 中     |
| **Attention Bias**  | 在 attention 计算中对前缀位置施加 bias          | 高   | 高     |

**推荐：前缀长度控制 + Prompt 权重标记（简单有效）**

### 2.4 架构对比

```
=== 改造前 (当前) ===

VLLMAdapter
  ├── self.llm (vLLM)        → 仅用于 generate() (极少调用)
  └── self.hf_model (HF)     → 用于 compute_kv() + forward_with_kv_injection()
                                (几乎所有请求)

VRAM: ~14GB (vLLM) + ~14GB (HF) = ~28GB
实际推理: HF 模型 (无 PagedAttention, 无连续批处理)
DKI: 精确 past_key_values 注入，但用 HF 推理
AGA: 无法使用（HF 模型没有 AGA hook）


=== 改造后: 统一引擎 + 双层注入 ===

VLLMAdapter
  └── self.llm (vLLM)        → 用于所有推理
      ├── DKI: Prompt Prefix  → 偏好文本作为 prompt 前缀 (prefill 阶段)
      └── AGA: forward hook   → 领域知识在高熵时 delta 注入 (decode 阶段)

VRAM: ~14GB (vLLM) + ~8MB (AGA KVStore) = ~14GB
实际推理: vLLM (PagedAttention + 连续批处理)
DKI: 偏好通过 prompt 前缀自然进入 KV Cache
AGA: 知识通过 register_forward_hook 注入
```

### 2.5 关键技术映射

| DKI 概念     | 当前实现                               | 改造后                                  |
| ------------ | -------------------------------------- | --------------------------------------- |
| 偏好 KV 计算 | `compute_kv()` → HF 前向传播 → 全层 KV | 不再需要 — 偏好文本直接作为 prompt 前缀 |
| KV 注入      | `past_key_values` 负位置拼接           | Prompt Prefix (vLLM prefill 自然处理)   |
| Alpha 缩放   | `entry.value * alpha`                  | 前缀长度控制 + prompt 权重标记          |
| 用户隔离     | `BoundedUserKVCache` per-user 分区     | per-request prompt 构造 (天然隔离)      |
| 推理         | `self.hf_model.generate()`             | `self.llm.generate()` (vLLM 原生)       |
| 知识扩展     | 不支持                                 | AGA hook 自动注入                       |

### 2.6 DKI + AGA 混合部署的数据流

```
用户请求: "推荐一本关于量子计算的书"
用户偏好: ["喜欢通俗易懂的解释", "偏好中文书籍"]
AGA 知识库: [量子计算领域知识, 书籍数据库, ...]

Step 1: DKI Planner 分析查询
  → 触发偏好注入 (alpha=0.3)
  → 召回相关历史

Step 2: 构造 Prompt Prefix
  prefix = "[用户偏好 (参考权重: 30%)]\n- 喜欢通俗易懂的解释\n- 偏好中文书籍\n\n"
  prefix += "[相关历史]\n- 之前讨论过《时间简史》\n\n"
  final_prompt = prefix + "推荐一本关于量子计算的书"

Step 3: vLLM 推理 (DKI + AGA 同时工作)
  vLLM.generate(final_prompt)
    ├── Prefill: 处理 prefix + query → KV Cache 自然包含偏好信息
    └── Decode: 逐 token 生成
        ├── Token "我" → AGA 评估熵 → 低熵 → 旁路
        ├── Token "推荐" → AGA 评估熵 → 低熵 → 旁路
        ├── Token "《" → AGA 评估熵 → 高熵 → 注入量子计算书籍知识
        ├── Token "量子" → AGA 评估熵 → 中熵 → 部分注入
        └── ...

Step 4: 返回结果
  "我推荐《量子计算：从入门到精通》（中文版），这本书以通俗易懂的方式..."
       ↑ DKI 偏好影响: 推荐中文、通俗风格
       ↑ AGA 知识影响: 提供具体书名和内容
```

---

## 3. 实施计划

### Phase 1: VLLMAdapter 改造 — Prompt Prefix 模式

**目标：消除 HF 模型依赖，所有推理走 vLLM，DKI 偏好通过 prompt 前缀注入**

#### 3.1 修改 `VLLMAdapter.__init__`

```python
class VLLMAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        injection_mode: str = "prompt_prefix",
        aga_enabled: bool = False,
        aga_config: dict = None,
        **kwargs
    ):
        """
        Args:
            injection_mode: DKI 偏好注入模式
                - "prompt_prefix": 偏好文本作为 prompt 前缀 (推荐)
                - "aga_bridge": 偏好通过 AGA KVStore 注入 (需要 AGA)
                - "hf_fallback": HF 模型回退 (兼容模式，与当前行为一致)
            aga_enabled: 是否启用 AGA 全局知识注入
            aga_config: AGA 配置 (仅 aga_enabled=True 时生效)
        """
        super().__init__(model_name, **kwargs)
        self.injection_mode = injection_mode
        self.aga_enabled = aga_enabled
        self._aga_config = aga_config or {}

        # AGA 插件 (可选)
        self._aga_plugin = None
        self._aga_adapter = None
        self._aga_attached = False

        # HF 回退模式 (保留兼容性)
        self.hf_model = None
```

#### 3.2 修改 `load()` — 统一引擎

```python
def load(self) -> None:
    from vllm import LLM

    # 加载 vLLM 引擎 (始终加载)
    enforce_eager = self.aga_enabled  # AGA 需要 eager 模式
    self.llm = LLM(
        model=self.model_name,
        enforce_eager=enforce_eager,
        tensor_parallel_size=self.tensor_parallel_size,
        max_model_len=self.max_model_len,
        gpu_memory_utilization=self.gpu_memory_utilization,
        trust_remote_code=self.trust_remote_code,
    )

    # 如果启用 AGA，初始化并挂载
    if self.aga_enabled:
        self._init_aga()

    self._is_loaded = True
    # 不再加载 HF 模型 (除非 hf_fallback 模式)
```

#### 3.3 新增 `_init_aga()` — AGA 初始化

```python
def _init_aga(self):
    """初始化 AGA 插件并挂载到 vLLM 内部模型"""
    from aga import AGAPlugin, AGAConfig
    from aga.adapter.vllm import VLLMAdapter as AGAVLLMAdapter

    internal_model = AGAVLLMAdapter.extract_model(self.llm)

    aga_config = AGAConfig(
        hidden_dim=self.hidden_dim,
        bottleneck_dim=self._aga_config.get("bottleneck_dim", 64),
        max_slots=self._aga_config.get("max_slots", 256),
        device="cuda",
        fail_open=True,
    )

    self._aga_plugin = AGAPlugin(aga_config)
    self._aga_adapter = AGAVLLMAdapter(enforce_eager=True)
    self._aga_adapter.set_llm_engine_ref(self.llm)

    layer_indices = self._aga_config.get("layer_indices", [-1, -2, -3])
    self._aga_plugin.attach(
        internal_model,
        layer_indices=layer_indices,
        adapter=self._aga_adapter,
    )
    self._aga_attached = True
    logger.info(f"AGA 已挂载到 vLLM 内部模型, layers={layer_indices}")
```

#### 3.4 修改 `forward_with_kv_injection()` — 三模式路由

```python
def forward_with_kv_injection(
    self,
    prompt: str,
    injected_kv: List[KVCacheEntry],
    alpha: float = 1.0,
    max_new_tokens: int = 512,
    **kwargs
) -> ModelOutput:
    """
    带偏好注入的推理 — 三种模式

    prompt_prefix: 偏好文本 → prompt 前缀 → vLLM 推理
    aga_bridge:    偏好 KV → AGA KVStore → vLLM 推理
    hf_fallback:   偏好 KV → past_key_values → HF 推理 (兼容)
    """
    if self.injection_mode == "prompt_prefix":
        return self._forward_with_prompt_prefix(
            prompt, alpha, max_new_tokens, **kwargs
        )
    elif self.injection_mode == "aga_bridge" and self._aga_attached:
        return self._forward_with_aga_bridge(
            prompt, injected_kv, alpha, max_new_tokens, **kwargs
        )
    else:
        return self._forward_with_hf_fallback(
            prompt, injected_kv, alpha, max_new_tokens, **kwargs
        )
```

#### 3.5 新增 `_forward_with_prompt_prefix()` — Prompt Prefix 模式

```python
def _forward_with_prompt_prefix(
    self,
    prompt: str,
    alpha: float = 1.0,
    max_new_tokens: int = 512,
    **kwargs
) -> ModelOutput:
    """
    Prompt Prefix 模式: 偏好已由 Planner 组装到 prompt 中

    注意: 在此模式下，injected_kv 参数不再使用。
    偏好文本由 DKI Planner 在 plan.final_input 中组装为 prompt 前缀。
    Executor 传入的 prompt 参数就是 plan.final_input，已包含偏好前缀。

    Alpha 控制: 通过 Planner 的前缀构造策略实现
    - alpha > 0.5: 完整偏好前缀
    - alpha 0.3-0.5: 精简偏好前缀 (只保留高优先级)
    - alpha < 0.3: 极简偏好标记
    """
    from vllm import SamplingParams

    start_time = time.perf_counter()

    sampling_params = SamplingParams(
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.9),
        max_tokens=max_new_tokens,
    )

    # prompt 已包含偏好前缀 (由 Planner 组装)
    outputs = self.llm.generate([prompt], sampling_params)
    output = outputs[0]

    end_time = time.perf_counter()

    return ModelOutput(
        text=output.outputs[0].text,
        tokens=list(output.outputs[0].token_ids),
        latency_ms=(end_time - start_time) * 1000,
        input_tokens=len(output.prompt_token_ids),
        output_tokens=len(output.outputs[0].token_ids),
        metadata={
            'alpha': alpha,
            'injection_mode': 'prompt_prefix',
            'aga_active': self._aga_attached,
            'aga_knowledge': (
                self._aga_plugin.knowledge_count if self._aga_plugin else 0
            ),
        },
    )
```

#### 3.6 Executor 适配 — 最小改动

`injection_executor.py` 的核心改动：

```python
# injection_executor.py — 仅需修改 _execute_with_kv_injection

async def _execute_with_kv_injection(self, plan, ...):
    """
    改造点: prompt_prefix 模式下不再需要 compute_kv
    """
    if self.model.injection_mode == "prompt_prefix":
        # Prompt Prefix 模式: 偏好已在 plan.final_input 中
        # 不需要 compute_kv，直接用 vLLM 推理
        output = self.model.forward_with_kv_injection(
            prompt=plan.final_input,  # 已包含偏好前缀
            injected_kv=[],           # 不使用
            alpha=plan.alpha_profile.effective_preference_alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
    else:
        # 原有逻辑: compute_kv → forward_with_kv_injection
        preference_kv, cache_hit, cache_tier = self._get_preference_kv(...)
        output = self.model.forward_with_kv_injection(
            prompt=plan.final_input,
            injected_kv=preference_kv,
            alpha=...,
            ...
        )
```

**关键：`plan.final_input` 已经由 Planner 组装好了偏好前缀 + 历史后缀 + 用户查询。**

当前 DKI 的 `recall_v4` 策略已经在 `plan.final_input` 中组装了历史后缀，
偏好前缀只需要在 Planner 中增加一个步骤。

### Phase 2: Planner 适配 — 偏好前缀构造

```python
# injection_planner.py — 新增偏好前缀构造

def _build_preference_prefix(
    self,
    preferences: List[Preference],
    alpha: float,
) -> str:
    """
    构造偏好前缀 (用于 prompt_prefix 模式)

    Alpha 控制策略:
    - alpha > 0.5: 完整偏好列表
    - alpha 0.3-0.5: 只保留 top-3 高优先级偏好
    - alpha < 0.3: 单行偏好摘要
    - alpha < 0.1: 不添加前缀
    """
    if alpha < 0.1 or not preferences:
        return ""

    sorted_prefs = sorted(preferences, key=lambda p: p.priority, reverse=True)

    if alpha > 0.5:
        pref_lines = [f"- {p.preference_text}" for p in sorted_prefs]
    elif alpha > 0.3:
        pref_lines = [f"- {p.preference_text}" for p in sorted_prefs[:3]]
    else:
        pref_lines = [sorted_prefs[0].preference_text]

    prefix = "[用户偏好]\n" + "\n".join(pref_lines) + "\n\n"
    return prefix
```

### Phase 3: AGA 知识管理集成 (可选)

如果用户同时使用 AGA，需要在 VLLMAdapter 中暴露 AGA 管理接口：

```python
# VLLMAdapter 新增方法

@property
def aga_plugin(self) -> Optional["AGAPlugin"]:
    """获取 AGA 插件实例 (用于知识管理)"""
    return self._aga_plugin

def load_aga_knowledge(self, source: str, **kwargs) -> int:
    """加载 AGA 知识"""
    if self._aga_plugin is None:
        raise RuntimeError("AGA 未启用，请设置 aga_enabled=True")
    return self._aga_plugin.load_from(source, **kwargs)

def get_aga_diagnostics(self) -> dict:
    """获取 AGA 诊断信息"""
    if self._aga_plugin is None:
        return {"aga_enabled": False}
    return self._aga_plugin.get_diagnostics()
```

### Phase 4: compute_kv 优化

在 `prompt_prefix` 模式下，`compute_kv()` 不再被 Executor 调用。
但为了保持 `BaseModelAdapter` 接口兼容，仍需实现：

```python
def compute_kv(self, text: str, return_hidden: bool = False):
    """
    计算 KV — 在 prompt_prefix 模式下使用 vLLM 内部模型

    注意: prompt_prefix 模式下此方法不会被 Executor 调用。
    保留实现是为了接口兼容和其他可能的使用场景。
    """
    if self.injection_mode == "prompt_prefix":
        # 使用 vLLM 内部模型 (零额外模型)
        return self._compute_kv_via_vllm_internal(text, return_hidden)
    else:
        # 原有逻辑: 使用 HF 模型
        self._load_hf_model()
        return self._compute_kv_via_hf(text, return_hidden)

def _compute_kv_via_vllm_internal(self, text, return_hidden):
    """使用 vLLM 内部模型计算 KV (零额外模型)"""
    from aga.adapter.vllm import VLLMAdapter as AGAVLLMAdapter

    internal_model = AGAVLLMAdapter.extract_model(self.llm)
    inputs = self.tokenize(text)

    # 将输入移到模型设备
    device = next(internal_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = internal_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=True,
            return_dict=True,
        )

    kv_entries = []
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
        for layer_idx, (key, value) in enumerate(outputs.past_key_values):
            kv_entries.append(KVCacheEntry(
                key=key.detach().cpu(),
                value=value.detach().cpu(),
                layer_idx=layer_idx,
            ))

    hidden_states = None
    if return_hidden and hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states[-1].detach().cpu()

    del outputs
    torch.cuda.empty_cache()
    return kv_entries, hidden_states
```

---

## 4. 配置变更

### 4.1 DKI 配置文件

```yaml
model:
    default_engine: vllm
    engines:
        vllm:
            enabled: true
            model_name: "Qwen/Qwen2-7B-Instruct"

            # DKI 偏好注入模式
            injection_mode: "prompt_prefix" # "prompt_prefix" | "aga_bridge" | "hf_fallback"

            # vLLM 配置
            tensor_parallel_size: 1
            max_model_len: 8192
            gpu_memory_utilization: 0.9

            # AGA 全局知识注入 (可选)
            aga_enabled: true # 是否启用 AGA
            aga:
                bottleneck_dim: 64
                max_slots: 256
                layer_indices: [-1, -2, -3]
                knowledge_sources: # 启动时自动加载的知识
                    - "knowledge/domain.jsonl"
```

### 4.2 三种部署模式

```yaml
# === 模式 1: DKI 独立部署 (无 AGA) ===
injection_mode: "prompt_prefix"
aga_enabled: false
# 效果: vLLM 推理 + 偏好前缀, 无知识扩展

# === 模式 2: DKI + AGA 混合部署 (推荐) ===
injection_mode: "prompt_prefix"
aga_enabled: true
# 效果: vLLM 推理 + 偏好前缀 + AGA 知识注入

# === 模式 3: 兼容模式 (与当前行为一致) ===
injection_mode: "hf_fallback"
aga_enabled: false
# 效果: HF 推理 + past_key_values 注入, 与当前完全一致
```

---

## 5. 效果预估

### 5.1 资源对比

| 指标      | 当前 (双模型)          | prompt_prefix   | prompt_prefix + AGA | hf_fallback    |
| --------- | ---------------------- | --------------- | ------------------- | -------------- |
| VRAM (7B) | ~28GB (vLLM + HF)      | ~14GB (仅 vLLM) | ~14GB (vLLM + AGA)  | ~14GB (仅 HF)  |
| CPU 内存  | 可能 ~14GB (HF on CPU) | ~0              | ~0                  | ~0             |
| 推理引擎  | HF (无优化)            | vLLM            | vLLM                | HF             |
| 吞吐量    | ~30 tok/s              | ~100+ tok/s     | ~90+ tok/s          | ~30 tok/s      |
| DKI 偏好  | 精确 (past_kv)         | 近似 (prompt)   | 近似 (prompt)       | 精确 (past_kv) |
| AGA 知识  | 不支持                 | 不支持          | **支持**            | 不支持         |

### 5.2 功能对比

| 功能           | 当前           | prompt_prefix + AGA       |
| -------------- | -------------- | ------------------------- |
| DKI 偏好注入   | 精确           | 近似 (prompt 前缀)        |
| AGA 知识注入   | 不支持         | **支持 (高熵门控)**       |
| 连续批处理     | 不支持 (HF)    | 支持 (vLLM)               |
| PagedAttention | 不支持 (HF)    | 支持 (vLLM)               |
| 用户隔离       | per-user cache | per-request prompt (天然) |
| 模型数量       | 2 个           | 1 个                      |
| VRAM 节省      | —              | **~14GB**                 |

---

## 6. 偏好注入精度分析

### 6.1 Prompt Prefix vs past_key_values

**Prompt Prefix 的语义等价性：**

`past_key_values` 注入的本质是：

```
attention(Q, [K_pref; K_input], [V_pref; V_input])
```

Prompt Prefix 的本质是：

```
attention(Q, K_[pref_tokens; input_tokens], V_[pref_tokens; input_tokens])
```

**区别：**

1. `past_key_values` 的 K/V 是预计算的，可以被 alpha 缩放
2. Prompt Prefix 的 K/V 是 vLLM 在 prefill 时实时计算的，alpha 只能通过间接方式控制

**对偏好注入的影响：**

-   偏好是"软约束"（如"喜欢简洁"），不需要精确的 alpha 控制
-   Prompt Prefix 实际上比 `past_key_values` 更自然——模型直接"阅读"偏好文本
-   在 LLM 的 instruction following 能力下，prompt 中的偏好指令通常比 KV 注入更有效

### 6.2 Alpha 精度损失评估

| Alpha 范围 | past_key_values 效果 | Prompt Prefix 效果 | 差异 |
| ---------- | -------------------- | ------------------ | ---- |
| 0.5-1.0    | 强偏好影响           | 完整偏好前缀       | 小   |
| 0.3-0.5    | 中等偏好影响         | 精简偏好前缀       | 中   |
| 0.1-0.3    | 弱偏好影响           | 极简偏好标记       | 中   |
| < 0.1      | 几乎无影响           | 无前缀             | 无   |

**结论：对于 DKI 的偏好场景，Prompt Prefix 的精度损失在可接受范围内。**

---

## 7. 风险与缓解

### 7.1 偏好前缀增加 prompt 长度

**风险**: 偏好前缀占用 token 预算，减少可用的生成长度。

**缓解**:

1. 偏好文本通常很短（~50-200 tokens），相对于 `max_model_len=8192` 影响极小
2. Planner 可以根据 alpha 动态控制前缀长度
3. 高优先级偏好优先保留

### 7.2 AGA CUDA Graph 兼容性

**风险**: AGA 的熵门控可能导致 CUDA Graph 失效。

**缓解**:

1. `enforce_eager=True`（AGA 模式默认启用）
2. AGA 内部使用 `torch.where` 替代 `if/else`（已实现）
3. 仅在 `aga_enabled=True` 时启用 eager 模式，不影响纯 DKI 部署

### 7.3 vLLM 内部 API 变化

**风险**: `extract_model()` 依赖 vLLM 内部结构，版本更新可能失效。

**缓解**:

1. AGA 的 `VLLMAdapter` 已实现多路径探测（支持 vLLM 0.4.x - 0.6.x）
2. `check_compatibility()` 可在启动时验证兼容性
3. 保留 `hf_fallback` 模式作为终极回退

---

## 8. 实施步骤

### Step 1: VLLMAdapter 改造 (向后兼容)

-   [ ] 添加 `injection_mode`, `aga_enabled`, `aga_config` 参数
-   [ ] 实现 `_init_aga()` 方法
-   [ ] 实现 `_forward_with_prompt_prefix()` 方法
-   [ ] 实现 `_compute_kv_via_vllm_internal()` 方法
-   [ ] 将当前 `forward_with_kv_injection()` 重命名为 `_forward_with_hf_fallback()`
-   [ ] 新的 `forward_with_kv_injection()` 根据 `injection_mode` 路由
-   [ ] 暴露 `aga_plugin` 属性和 `load_aga_knowledge()` 方法

### Step 2: Planner 适配

-   [ ] 新增 `_build_preference_prefix()` 方法
-   [ ] 在 `build_plan()` 中根据 `injection_mode` 决定是否构造前缀
-   [ ] `plan.final_input` 在 prompt_prefix 模式下包含偏好前缀

### Step 3: Executor 适配

-   [ ] `_execute_with_kv_injection()` 在 prompt_prefix 模式下跳过 `compute_kv`
-   [ ] 保留原有逻辑作为 `hf_fallback` 路径

### Step 4: Factory 和配置

-   [ ] `factory.py` 传递新参数
-   [ ] 配置文件增加 `injection_mode`, `aga_enabled`, `aga` 配置项

### Step 5: 测试

-   [ ] 单元测试: prompt_prefix 模式
-   [ ] 单元测试: AGA 混合模式
-   [ ] 集成测试: DKI chat 流程 (三种模式)
-   [ ] A/B 测试: 偏好遵循率对比 (prompt_prefix vs past_key_values)

### Step 6: 文档

-   [ ] 更新 MODELS_ANALYSIS_REPORT.md
-   [ ] 更新 DKI 配置文档
-   [ ] 新增混合部署指南

---

## 9. 推荐决策

| 场景                      | 推荐配置                                                 |
| ------------------------- | -------------------------------------------------------- |
| DKI + AGA 混合部署 (推荐) | `injection_mode: "prompt_prefix"` + `aga_enabled: true`  |
| DKI 独立部署 (高吞吐)     | `injection_mode: "prompt_prefix"` + `aga_enabled: false` |
| DKI 独立部署 (精确偏好)   | `injection_mode: "hf_fallback"` + `aga_enabled: false`   |
| 过渡期                    | `injection_mode: "hf_fallback"` (当前行为，零风险)       |

**最终推荐**: `prompt_prefix` + `aga_enabled: true`

理由：

1. 单一 vLLM 引擎，节省 ~14GB VRAM
2. DKI 偏好通过 prompt 前缀自然注入，利用 LLM 的 instruction following 能力
3. AGA 知识通过 hook 在高熵时自动注入，提供领域知识扩展
4. 两者注入点不冲突（prefill vs decode），可以完美共存
5. 保留 `hf_fallback` 作为零风险回退
