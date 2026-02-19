# DKI 系统完整优化方案

> **基于 `kv_cache优化建议.md` + `优化建议.md` 两份文档，结合现有 `dki/cache` + `dki/core` 实际代码的客观分析与落地方案**

---

## 〇、评估方法论

本优化方案遵循以下原则：

1. **代码事实优先**：所有分析基于已审查过的代码实际状态，而非假设
2. **ROI 排序**：按「收益 / 改动风险」排序，而非按难度
3. **客观取舍**：两份文档中有重叠、矛盾或已过时的建议，逐条标注
4. **可落地**：每条建议给出具体修改位置和预期效果

---

## 一、当前系统状态总结（审查后）

经过完整审查，系统已修复的问题：

| 类别 | 已修复数量 | 关键修复 |
|------|-----------|---------|
| GPU 显存泄漏 | 12+ | `compute_kv` / `embed` / `forward` 的 `.detach().cpu()` + `del` |
| bfloat16 序列化 | 3 | `preference_cache` / `user_isolation` / `base.py` |
| 并发安全 | 2 | `UserScopedCacheStore` 的 `asyncio.Lock` |
| 数值正确性 | 2 | 双重 alpha 缩放修复、score fusion 权重归一化 |
| 架构缺陷 | 5+ | `close()` 接口、marker 过滤、token 估算等 |

**当前系统是"正确但保守"的状态**，优化空间集中在性能和工程效率。

---

## 二、优化项完整清单（按优先级排序）

### 🔴 P0 — 必做（上线前 / 高并发前）

---

#### P0-1：`torch.cuda.empty_cache()` 策略性裁剪

**来源**：`kv_cache优化建议.md` 问题 1

**现状分析**：

当前 `empty_cache()` 分布（经 grep 统计）：

| 文件 | 调用次数 | 位置 |
|------|---------|------|
| `vllm_adapter.py` | 6 | `compute_kv` / `embed` / `forward_with_kv_injection` |
| `deepseek_adapter.py` | 3 | `compute_kv` / `embed` / `forward` |
| `glm_adapter.py` | 3 | 同上 |
| `llama_adapter.py` | 3 | 同上 |
| `preference_cache.py` | 2 | `_compute_kv` 正常 + 异常路径 |
| `user_isolation.py` | 2 | 同上 |
| `hybrid_injector.py` | 2 | `_get_or_compute_preference_kv` |
| `injection_executor.py` | 1 | `_get_preference_kv` |
| `embedding_service.py` | 1 | embedding 计算后 |

**总计约 24 处调用**，绝大多数在正常路径上。

**问题**：PyTorch caching allocator 的设计是 `del tensor` 后显存回到 allocator 池，`empty_cache()` 将池中块归还 CUDA driver。频繁调用导致：
- allocator 频繁抖动 → 后续 `to(device)` 重新 malloc
- 隐式 PCIe / driver sync → 拉长 latency
- 多并发时 allocator 锁竞争

**优化方案**：

```
保留规则:
✅ del tensor              → 必须保留（所有位置）
✅ empty_cache()           → 仅保留在以下位置:
   1. OOM recovery 的 except 块中
   2. Executor.execute() 的 finally 块（请求边界统一清理）
   3. 模型卸载 / 系统关闭时

删除规则:
❌ compute_kv 正常路径中的 empty_cache    → 删除
❌ embed 正常路径中的 empty_cache          → 删除
❌ forward_with_kv_injection 正常路径      → 删除
❌ _get_preference_kv 正常路径             → 删除
❌ _compute_kv (preference_cache) 正常路径 → 删除
```

**具体修改**：

1. **所有 model adapter** (`vllm/deepseek/glm/llama_adapter.py`)：
   - `compute_kv`：保留 `del outputs`，删除正常路径 `empty_cache()`，保留 except 块中的
   - `embed`：同上
   - `forward_with_kv_injection`：保留 `del past_kv, outputs`，删除正常路径 `empty_cache()`

2. **`injection_executor.py`**：
   - `_get_preference_kv`：删除 L780 的 `empty_cache()`
   - **新增**：在 `execute()` 方法的 `finally` 块中添加统一清理点：
   ```python
   async def execute(self, plan, ...):
       try:
           # ... 现有逻辑 ...
       except Exception as e:
           # ... 现有降级逻辑 ...
       finally:
           # 请求边界统一清理
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
   ```

3. **`preference_cache.py` / `user_isolation.py`**：
   - `_compute_kv`：删除正常路径 `empty_cache()`，保留 except 块中的

**预期效果**：
- 单请求 latency 降低 5-15ms（减少 sync 点）
- 并发场景下 allocator 锁竞争显著减少
- GPU 显存利用率更稳定（allocator 复用更高效）

**风险**：低。`del tensor` 已确保引用释放，allocator 自然复用。

---

#### P0-2：Executor 内置缓存加 LRU / 容量上限

**来源**：`kv_cache优化建议.md` 建议 4

**现状分析**：

`injection_executor.py` L127：
```python
self._preference_kv_cache: Dict[str, Dict[str, Tuple[Any, str]]] = {}
```

这是一个**无界字典**，结构为 `{user_id: {content_hash: (kv_entries, content_hash)}}`。

**问题**：
- 无 LRU 淘汰 → 用户数增长后 CPU 内存无限增长
- 无容量上限 → 无法预测内存占用
- `clear_preference_cache` 只提供全量清除，无自动淘汰

**优化方案**：

为 Executor 的 `_preference_kv_cache` 添加双层限制：

```python
from collections import OrderedDict

class BoundedUserKVCache:
    """
    有界的用户级 KV 缓存
    
    双层限制:
    - max_users: 最大用户数 (LRU 淘汰最久未访问的用户)
    - max_entries_per_user: 每用户最大条目数
    """
    def __init__(self, max_users: int = 500, max_entries_per_user: int = 5):
        self._max_users = max_users
        self._max_entries_per_user = max_entries_per_user
        self._cache: OrderedDict[str, OrderedDict[str, Tuple]] = OrderedDict()
    
    def get(self, user_id: str, content_hash: str):
        if user_id in self._cache:
            self._cache.move_to_end(user_id)  # LRU touch
            return self._cache[user_id].get(content_hash)
        return None
    
    def put(self, user_id: str, content_hash: str, value):
        if user_id not in self._cache:
            if len(self._cache) >= self._max_users:
                # 淘汰最久未访问的用户
                evicted_uid, evicted_data = self._cache.popitem(last=False)
                del evicted_data  # 释放 CPU tensor
            self._cache[user_id] = OrderedDict()
        
        self._cache.move_to_end(user_id)
        user_cache = self._cache[user_id]
        
        if len(user_cache) >= self._max_entries_per_user:
            user_cache.popitem(last=False)
        
        user_cache[content_hash] = value
    
    def clear(self, user_id=None):
        if user_id:
            self._cache.pop(user_id, None)
        else:
            self._cache.clear()
```

**修改位置**：`injection_executor.py` L127

**预期效果**：
- CPU 内存占用可预测（max_users × max_entries × kv_size）
- 自动淘汰冷用户的 KV 数据
- 防止长期运行后内存泄漏

**风险**：极低。纯增量改动。

---

#### P0-3：Fact Call 重构为 Planner-only

**来源**：`优化建议.md` 最终建议 #1 + Fact Call 重构详细方案

**现状分析**：

`injection_executor.py` L459-661 的 `_execute_fact_call_loop` 实现：
- `for round_idx in range(max_rounds)` 循环
- 每轮：detect → retrieve → format → re-infer（完整推理）
- 每轮推理都需要 `_get_preference_kv` + `forward_with_kv_injection`

**问题**（三个层面）：

1. **架构职责错位**：Executor 变成了控制器，违反 Planner/Executor 分离原则
2. **GPU 成本不可预测**：forward 次数 = 1~N，KV 注入次数 = N，峰值显存不可预测
3. **监控/可解释性被破坏**：一次用户请求 → N 次 `model.forward`，latency 不可预测

**优化方案（分两阶段）**：

**阶段 A（短期，低风险）：硬限制 + 监控**

不重构架构，但加强约束：

```python
# injection_executor.py - _execute_fact_call_loop 修改
# 1. 添加总 latency 硬限制
MAX_FACT_CALL_LATENCY_MS = 2000  # 2 秒硬限制

for round_idx in range(max_rounds):
    elapsed_ms = (time.time() - loop_start) * 1000
    if elapsed_ms > MAX_FACT_CALL_LATENCY_MS:
        logger.warning(f"Fact call latency budget exceeded: {elapsed_ms:.0f}ms")
        break
    # ... 现有逻辑 ...

# 2. 在 ExecutionResult 中添加监控字段
result.metadata["fact_call_latency_ms"] = elapsed_ms
result.metadata["fact_call_rounds_actual"] = round_idx + 1
```

**阶段 B（中期，高收益）：Planner-only 重构**

完整实现 Planner-side Fact Detection：

1. **新增数据结构**（`injection_plan.py`）：
```python
@dataclass
class FactRequirement:
    trace_id: str
    source: str           # "history" | "memory"
    query: str
    offset: int = 0
    limit: int = 5
    confidence: float = 1.0

# InjectionPlan 扩展
@dataclass
class InjectionPlan:
    # ... 现有字段 ...
    fact_blocks: List[str] = field(default_factory=list)
    fact_tokens: int = 0
    fact_strategy: str = "none"  # "planner_only" | "none"
```

2. **Planner 新增方法**（`injection_planner.py`）：
```python
def _detect_fact_requirements(
    self, query: str, recall_results: List[ChatMessage]
) -> List[FactRequirement]:
    """基于规则 + 召回信号检测事实需求（替代 LLM detect）"""
    requirements = []
    
    # 规则 1：明确时间指代
    if any(kw in query for kw in ["上次", "之前", "前面", "earlier", "last time"]):
        requirements.append(FactRequirement(
            trace_id=generate_trace_id(),
            source="history",
            query=query,
            limit=5,
        ))
    
    # 规则 2：recall_v4 召回置信度不足
    if recall_results and self._low_recall_confidence(recall_results):
        requirements.append(FactRequirement(
            trace_id=generate_trace_id(),
            source="history",
            query="补充相关上下文",
            limit=3,
            confidence=0.6,
        ))
    
    return requirements
```

3. **Executor 简化**：删除整个 `_execute_fact_call_loop`，`execute()` 中移除 Fact Call 分支

**预期效果**：
- GPU 成本：forward 次数从 1~N 降为 **恒定 1**
- P99 latency：消灭 Fact Call 导致的长尾
- 架构清晰：Executor 回归纯执行层

**风险**：中。需要验证 Planner-side 规则检测的召回质量不低于 LLM-based 检测。建议 A/B 测试。

**建议**：先实施阶段 A（1 天），再规划阶段 B（3-5 天）。

---

#### P0-4：`history_alpha` 动态衰减

**来源**：`优化建议.md` 问题 1

**现状分析**：

`injection_planner.py` L538：
```python
hist_alpha = 1.0  # 固定值
```

`AlphaProfile` 中 `history_alpha: float = 1.0` 是硬编码。

**问题**：
- 历史 suffix token 很多时，模型 100% 信任历史 → 过度"顺着旧话说"
- 新 query 的创新性/独立性下降
- 长对话中"人格粘滞"风险

**优化方案**：

```python
# injection_planner.py - _compute_alpha_profile 修改
import math

def _compute_alpha_profile(self, strategy, preferences, relevant_history, force_alpha):
    # ... 现有 force_alpha 和 pref_alpha 逻辑 ...
    
    # 动态 history_alpha：历史越长，权重越低
    history_tokens = sum(
        self._estimate_tokens(msg.content) for msg in relevant_history
    )
    
    if history_tokens > 0:
        # 对数衰减：512 token 以下 alpha=1.0，之后逐渐衰减到 0.3
        hist_alpha = max(
            0.3,
            min(1.0, 1.0 - math.log(max(history_tokens / 512, 1.0)) * 0.3)
        )
    else:
        hist_alpha = 1.0
    
    return AlphaProfile(
        preference_alpha=pref_alpha,
        history_alpha=hist_alpha,
    )
```

**衰减曲线**：

| history_tokens | history_alpha |
|---------------|---------------|
| 0-512 | 1.0 |
| 1024 | ~0.79 |
| 2048 | ~0.58 |
| 4096 | ~0.37 |
| 8192+ | 0.30 (下限) |

**注意**：当前 `history_alpha` 在 `recall_v4` 策略中语义为"后缀组装权重"，实际上 suffix 是直接拼接到 prompt 的，alpha 并不用于缩放 suffix 文本。因此，此优化的实际效果取决于 `history_alpha` 是否被 Executor 用于某种形式的加权。

**客观评估**：如果 `history_alpha` 仅在 stable 策略的 K/V 注入中使用，而 recall_v4 的历史是纯 suffix 拼接（alpha=1.0 意味着不做任何缩放），那么此优化**仅对 stable 回退路径有效**。对于 recall_v4，更有效的方式是在 `SuffixBuilder` 中通过 **token budget** 来控制历史长度（见 P1-1）。

**修改位置**：`injection_planner.py` L515-555

**风险**：低。衰减函数有上下限保护。

---

### 🟡 P1 — 推荐（规模化前）

---

#### P1-1：`recall_limit` → Token Budget（软预算）

**来源**：`优化建议.md` 建议 4

**现状分析**：

`injection_planner.py` L228：
```python
context.recall_limit = reference_result.recall_turns or 10  # 离散条数
```

`MultiSignalRecall` 和 `SuffixBuilder` 使用条数限制。

**问题**：
- 条数限制是粗粒度的：5 条短消息 vs 5 条长消息，token 差异巨大
- 无法精确控制最终 prompt 长度
- 可能导致 context window 溢出或浪费

**优化方案**：

```python
# QueryContext 扩展
@dataclass
class QueryContext:
    recall_limit: int = 10
    recall_token_budget: int = 2048  # 新增：token 软预算
    # ... 其他字段 ...

# injection_planner.py - analyze_query 修改
def analyze_query(self, query):
    context = QueryContext()
    # ... 现有逻辑 ...
    
    # Token budget 计算
    base_budget = 2048  # 默认
    if context.reference_resolved:
        # 明确引用时给更多预算
        base_budget = int(base_budget * 1.5)
    if context.trigger_confidence > 0.8:
        # 高置信度 trigger 时给更多预算
        base_budget = int(base_budget * 1.2)
    
    context.recall_token_budget = base_budget
    return context
```

然后在 `SuffixBuilder` 中使用 `recall_token_budget` 而非 `recall_limit` 来控制组装。

**预期效果**：
- Prompt 长度可预测
- 更精确的 context window 利用
- 与模型 max_length 对齐

**风险**：低-中。需要调整 `SuffixBuilder` 和 `MultiSignalRecall` 的接口。

---

#### P1-2：MemoryTrigger confidence 影响 AlphaProfile

**来源**：`优化建议.md` 建议 5

**现状分析**：

`injection_planner.py` L206-210：
```python
context.trigger_confidence = trigger_result.confidence
# 记录了 confidence，但从未用于影响 alpha
```

**问题**：MemoryTrigger 检测到高置信度的记忆触发（如"还记得我喜欢什么吗"），但偏好注入强度不变。

**优化方案**：

```python
# _compute_alpha_profile 中添加
if context and context.memory_triggered and context.trigger_confidence > 0.5:
    # 高置信度 trigger 时增强偏好注入
    confidence_boost = min(context.trigger_confidence, 1.0)
    pref_alpha = min(
        pref_alpha * (0.7 + 0.3 * confidence_boost),  # 最多提升 30%
        0.7  # 仍受 override_cap 约束
    )
```

**预期效果**：
- 用户主动询问偏好时，偏好注入更强
- 普通对话时保持默认强度
- 提升偏好相关对话的质量

**风险**：极低。受 `override_cap` 保护。

---

#### P1-3：偏好文本本身缓存（Adapter 层）

**来源**：`优化建议.md` 建议 2

**现状分析**：

`dki_plugin.py` L579：
```python
preferences = await self.data_adapter.get_user_preferences(user_id)
```

每次 `chat()` 调用都会查询数据库获取偏好文本。偏好是**低频变更数据**，但每次都走 DB 查询。

**问题**：
- 偏好 KV 有三级缓存（L1/L2/L3），但偏好**文本本身**没有缓存
- 每次 chat 都查 DB → P95 延迟来源
- 反直觉：KV（衍生物）有缓存，文本（源数据）没有

**优化方案**：

在 `DKIPlugin` 中添加偏好文本缓存：

```python
# dki_plugin.py
from functools import lru_cache
import time

class DKIPlugin:
    def __init__(self, ...):
        # ... 现有初始化 ...
        self._preference_text_cache: Dict[str, Tuple[List, float]] = {}
        self._preference_cache_ttl = 300  # 5 分钟 TTL
    
    async def _get_cached_preferences(self, user_id: str):
        """带 TTL 的偏好文本缓存"""
        now = time.time()
        if user_id in self._preference_text_cache:
            cached, cached_at = self._preference_text_cache[user_id]
            if now - cached_at < self._preference_cache_ttl:
                return cached
        
        preferences = await self.data_adapter.get_user_preferences(user_id)
        self._preference_text_cache[user_id] = (preferences, now)
        return preferences
```

**修改位置**：`dki_plugin.py` L579，将 `get_user_preferences` 替换为 `_get_cached_preferences`

**预期效果**：
- 5 分钟内同一用户的重复查询不走 DB
- P95 延迟降低 5-20ms（取决于 DB 延迟）
- 偏好更新后最多 5 分钟生效（可接受）

**风险**：极低。TTL 保证最终一致性。

---

#### P1-4：KV 监控指标增强

**来源**：`kv_cache优化建议.md` 建议 12 + `优化建议.md` 建议 6

**现状分析**：

`ExecutionResult` 已有 `inference_latency_ms`、`preference_cache_hit`、`preference_cache_tier`。

**缺失指标**：
- `kv_bytes_cpu`：CPU 上的 KV 缓存大小
- `kv_bytes_gpu_peak`：推理期间 GPU KV 峰值
- `kv_transfer_latency_ms`：CPU→GPU 传输耗时
- `allocator_fragmentation`：allocator 碎片率

**优化方案**：

```python
# injection_plan.py - ExecutionResult 扩展
@dataclass
class ExecutionResult:
    # ... 现有字段 ...
    
    # KV 监控 (新增)
    kv_bytes_cpu: int = 0
    kv_bytes_gpu_peak: int = 0
    kv_transfer_latency_ms: float = 0.0
    kv_layers_count: int = 0

# injection_executor.py - _execute_with_kv_injection 中添加
if preference_kv:
    result.kv_layers_count = len(preference_kv)
    result.kv_bytes_cpu = sum(
        e.key.nelement() * e.key.element_size() +
        e.value.nelement() * e.value.element_size()
        for e in preference_kv
    )
```

**风险**：极低。纯观测性改动。

---

### 🟢 P2 — 中期优化（收益高，改动大）

---

#### P2-1：KV 打包（KV Packing）

**来源**：`kv_cache优化建议.md` 建议 2

**现状分析**：

当前 KV 存储为 `List[KVCacheEntry]`，每个 entry 包含一层的 key 和 value：
```python
# 32 层模型 → 32 个 KVCacheEntry
# 每个 entry: key=[1, H, T, D], value=[1, H, T, D]
```

CPU ⇄ GPU 传输是逐层进行的：
```python
# injection_executor.py L764-769
cpu_kv_entries = [
    KVCacheEntry(key=e.key.cpu(), value=e.value.cpu(), layer_idx=e.layer_idx)
    for e in kv_entries
]
```

**问题**：
- 32 层 × 2 tensor × memcpy = 64 次 CPU⇄GPU 传输
- 每次 `.to()` 是一次 launch + sync 点
- Redis 序列化也是逐层 metadata

**优化方案**：

新增 `PackedKV` 数据结构：

```python
# models/base.py 新增
@dataclass
class PackedKV:
    """
    打包的 KV 缓存 - 将所有层的 KV 合并为单一 tensor
    
    Shape:
    - keys:   [L, H, T, D]  (L=layers, H=heads, T=tokens, D=head_dim)
    - values: [L, H, T, D]
    
    优势:
    - CPU→GPU 拷贝: 64 次 → 2 次
    - Redis 序列化: 64 次 metadata → 1 次
    - Alpha scaling: per-layer → 一次 vectorized
    """
    keys: torch.Tensor      # [L, H, T, D]
    values: torch.Tensor     # [L, H, T, D]
    num_layers: int
    dtype: torch.dtype
    
    @classmethod
    def from_entries(cls, entries: List[KVCacheEntry]) -> "PackedKV":
        """从 List[KVCacheEntry] 打包"""
        if not entries:
            raise ValueError("Empty entries")
        
        sorted_entries = sorted(entries, key=lambda e: e.layer_idx)
        keys = torch.stack([e.key.squeeze(0) for e in sorted_entries])
        values = torch.stack([e.value.squeeze(0) for e in sorted_entries])
        
        return cls(
            keys=keys,
            values=values,
            num_layers=len(sorted_entries),
            dtype=keys.dtype,
        )
    
    def to_entries(self) -> List[KVCacheEntry]:
        """解包为 List[KVCacheEntry]"""
        return [
            KVCacheEntry(
                key=self.keys[i].unsqueeze(0),
                value=self.values[i].unsqueeze(0),
                layer_idx=i,
            )
            for i in range(self.num_layers)
        ]
    
    def to(self, device) -> "PackedKV":
        """整体搬移到目标设备（单次传输）"""
        return PackedKV(
            keys=self.keys.to(device),
            values=self.values.to(device),
            num_layers=self.num_layers,
            dtype=self.dtype,
        )
    
    def scale_values(self, alpha: float) -> "PackedKV":
        """Vectorized alpha scaling（inplace）"""
        self.values.mul_(alpha)
        return self
```

**改动范围**：
1. `models/base.py`：新增 `PackedKV`
2. `injection_executor.py`：`_get_preference_kv` 返回 `PackedKV`，缓存 `PackedKV`
3. `preference_cache.py`：序列化/反序列化 `PackedKV`
4. Model adapters：`forward_with_kv_injection` 接受 `PackedKV` 或 `List[KVCacheEntry]`

**预期效果**：

| 项目 | 现在 | 打包后 |
|------|------|--------|
| CPU→GPU 拷贝 | 64 次 | **2 次** |
| Redis 序列化 | 64 次 metadata | **1 次** |
| allocator 碎片 | 高 | **显著降低** |
| alpha scaling | per-layer loop | **一次 `mul_`** |

**风险**：中。需要修改多个文件的接口，建议新旧格式并存过渡。

---

#### P2-2：CPU KV Tensor 改为 Pinned Memory

**来源**：`kv_cache优化建议.md` 建议 6

**现状分析**：

当前 CPU tensor 使用默认的 pageable memory。每次 `.to(device)` 需要：
1. CPU pageable → CPU pinned（隐式拷贝）
2. CPU pinned → GPU HBM（DMA 传输）

**优化方案**：

在缓存时直接使用 pinned memory：

```python
# injection_executor.py - _get_preference_kv 修改
cpu_kv_entries = [
    KVCacheEntry(
        key=e.key.cpu().pin_memory(),
        value=e.value.cpu().pin_memory(),
        layer_idx=e.layer_idx,
    )
    for e in kv_entries
]
```

或配合 PackedKV：

```python
packed = PackedKV.from_entries(kv_entries)
packed.keys = packed.keys.cpu().pin_memory()
packed.values = packed.values.cpu().pin_memory()
```

**预期效果**：
- CPU→GPU 传输速度提升 ~2x（跳过 pageable→pinned 的隐式拷贝）
- 与 `non_blocking=True` 配合效果更佳

**风险**：低。pinned memory 占用物理内存且不可被 swap，需确保总量可控。配合 P0-2 的 LRU 限制使用。

---

#### P2-3：CPU→GPU 拷贝使用 `non_blocking=True`

**来源**：`kv_cache优化建议.md` 建议 7

**现状分析**：

当前所有 `.to(device)` 调用都是同步的（`non_blocking` 默认为 `False`）。grep 确认代码中没有任何 `non_blocking` 使用。

**优化方案**：

在 KV 从 CPU 搬移到 GPU 时使用非阻塞传输：

```python
# injection_executor.py - _get_preference_kv 中
# 从缓存取出时
kv_entries = [
    KVCacheEntry(
        key=e.key.to(self.model.device, non_blocking=True),
        value=e.value.to(self.model.device, non_blocking=True),
        layer_idx=e.layer_idx,
    )
    for e in cached_entries
]
# 在实际使用前同步
torch.cuda.current_stream().synchronize()
```

**注意**：`non_blocking=True` 需要配合 pinned memory 才有意义。如果源 tensor 不在 pinned memory 中，CUDA runtime 会自动退化为同步拷贝。

**预期效果**：
- 配合 pinned memory，传输与计算可重叠
- 单请求 latency 降低 2-5ms

**风险**：低。需要在正确位置添加 synchronize。

---

#### P2-4：Redis KV 序列化格式优化

**来源**：`kv_cache优化建议.md` 建议 3 + 5

**现状分析**：

`preference_cache.py` L569-606 的 `_serialize_kv`：
```python
# 当前流程:
# 1. entry.key.cpu() → numpy
# 2. bfloat16 → float32 转换
# 3. numpy.tobytes()
# 4. pickle.dumps(serializable_list)
# 5. zlib.compress(data)
```

**问题**：
- pickle 有 GIL 限制
- numpy → torch → reshape → copy 链路长
- bfloat16 → float32 → 再 cast 回来（精度损失 + 体积翻倍）
- pickle + zlib 双重开销

**优化方案（保守版，推荐）**：

使用 `torch.save` + `io.BytesIO` 替代 pickle + numpy：

```python
import io
import zlib

def _serialize_kv_v2(self, kv_entries):
    """优化的 KV 序列化 - 使用 torch.save 直接序列化"""
    buffer = io.BytesIO()
    
    # 直接保存 tensor，保留原始 dtype（包括 bfloat16）
    save_data = {
        'version': 2,
        'entries': [
            {
                'key': entry.key.cpu(),
                'value': entry.value.cpu(),
                'layer_idx': entry.layer_idx,
            }
            for entry in kv_entries
        ]
    }
    
    torch.save(save_data, buffer)
    data = buffer.getvalue()
    
    if self.config.enable_compression:
        data = zlib.compress(data, level=self.config.compression_level)
    
    return data

def _deserialize_kv_v2(self, data):
    """优化的 KV 反序列化"""
    if self.config.enable_compression:
        data = zlib.decompress(data)
    
    buffer = io.BytesIO(data)
    save_data = torch.load(buffer, weights_only=True)
    
    from dki.models.base import KVCacheEntry
    return [
        KVCacheEntry(
            key=e['key'],
            value=e['value'],
            layer_idx=e['layer_idx'],
        )
        for e in save_data['entries']
    ]
```

**优势**：
- 保留 bfloat16 原始精度（无 float32 中转）
- 无 numpy 转换开销
- `torch.save` 内部使用高效的 pickle protocol + tensor 特化路径
- 向后兼容：通过 `version` 字段区分新旧格式

**预期效果**：
- 序列化速度提升 ~2-3x
- bfloat16 模型存储体积减半（无需转 float32）
- 反序列化无精度损失

**风险**：低。新旧格式可并存。

---

### 🔵 P3 — 高级优化（前瞻性）

---

#### P3-1：KV 注入支持 Position Remap

**来源**：`kv_cache优化建议.md` 建议 9

**现状分析**：

当前设计隐含前提：injected KV 的 position = prefix 的 position。

**问题**：
- 不同 prompt length 时，注入 KV 的位置编码可能不一致
- 未来如果支持 negative position / virtual prefix，需要 RoPE rebase

**评估**：当前系统使用 HuggingFace `generate` 的 `past_key_values` 接口，位置编码由模型内部处理。只要 `past_key_values` 的 sequence length 与 `position_ids` 一致，就不会出问题。

**建议**：暂不实施。记录为技术债务，在以下场景触发时实施：
- 支持 variable-length prefix
- 支持多源 KV concat
- 切换到自定义 attention kernel

**风险**：N/A（暂不实施）

---

#### P3-2：KV Segment 化（多注入源）

**来源**：`kv_cache优化建议.md` 建议 10

**现状分析**：

当前只有一个注入源：preference KV。未来可能有：
- preference A / preference B
- memory KV
- tool KV

**评估**：当前单源设计足够。多源 concat 涉及：
- concat 顺序语义
- alpha 不可交换性
- attention 饱和问题

**建议**：暂不实施。在需要多源注入时设计 `KVSegment` 抽象。

---

#### P3-3：`compute_kv` 和 `forward_with_kv` Pipeline 化

**来源**：`kv_cache优化建议.md` 建议 8

**评估**：在当前架构中，`compute_kv` 的结果被缓存，通常只在首次请求时计算。后续请求直接从缓存读取 KV。因此 pipeline 化的收益仅在**首次请求**（cold start）时有效。

**建议**：优先级低。如果 cold start latency 成为瓶颈，可以考虑在 `DKIPlugin.chat()` 中并行发起 `compute_kv` 和数据加载。

---

#### P3-4：用户偏好 Embedding → Selective KV

**来源**：`优化建议.md` 建议 7

**评估**：这是一个研究方向，而非工程优化。需要：
- 对偏好文本做 embedding
- 基于 query-preference 相似度选择性注入部分 KV
- 需要实验验证效果

**建议**：作为实验项目，不纳入工程优化计划。

---

## 三、两份文档建议的交叉分析

| 建议 | kv_cache文档 | 优化文档 | 本方案评估 | 纳入优先级 |
|------|-------------|---------|-----------|-----------|
| 移除 empty_cache | ✅ 强烈建议 | — | **采纳** | P0-1 |
| KV Packing | ✅ 建议 | — | **采纳** | P2-1 |
| Executor 缓存 LRU | ✅ 建议 | — | **采纳** | P0-2 |
| Redis 序列化优化 | ✅ 建议 | — | **采纳（保守版）** | P2-4 |
| Pinned Memory | ✅ 建议 | — | **采纳** | P2-2 |
| non_blocking | ✅ 建议 | — | **采纳（配合 pinned）** | P2-3 |
| Position Remap | ⚠️ 前瞻 | — | **暂不实施** | P3-1 |
| KV Segment | ⚠️ 前瞻 | — | **暂不实施** | P3-2 |
| Pipeline compute+forward | ✅ 建议 | — | **优先级低** | P3-3 |
| Fact Call 重构 | — | 🔥 强烈建议 | **采纳（分阶段）** | P0-3 |
| history_alpha 动态衰减 | — | 🔥 必做 | **采纳（需验证语义）** | P0-4 |
| 偏好文本缓存 | — | 🔥 必做 | **采纳** | P1-3 |
| recall_limit → token budget | — | 推荐 | **采纳** | P1-1 |
| MemoryTrigger → Alpha | — | 推荐 | **采纳** | P1-2 |
| KV 监控指标 | ✅ 建议 | ✅ 建议 | **采纳** | P1-4 |
| Selective KV | — | 可选 | **暂不实施** | P3-4 |

---

## 四、实施路线图

### Phase 1：安全加固（1-2 天）

| 任务 | 优先级 | 预计耗时 | 影响范围 |
|------|--------|---------|---------|
| P0-1：裁剪 empty_cache | P0 | 2h | 8 文件 |
| P0-2：Executor LRU 缓存 | P0 | 3h | 1 文件 |
| P0-3A：Fact Call 硬限制 | P0 | 2h | 1 文件 |
| P1-4：KV 监控指标 | P1 | 2h | 2 文件 |

### Phase 2：性能优化（3-5 天）

| 任务 | 优先级 | 预计耗时 | 影响范围 |
|------|--------|---------|---------|
| P0-4：history_alpha 衰减 | P0 | 2h | 1 文件 |
| P1-1：Token Budget | P1 | 4h | 3 文件 |
| P1-2：MemoryTrigger → Alpha | P1 | 1h | 1 文件 |
| P1-3：偏好文本缓存 | P1 | 2h | 1 文件 |

### Phase 3：架构升级（5-10 天）

| 任务 | 优先级 | 预计耗时 | 影响范围 |
|------|--------|---------|---------|
| P0-3B：Fact Call Planner-only | P0 | 5d | 4 文件 |
| P2-1：KV Packing | P2 | 3d | 6 文件 |
| P2-2+P2-3：Pinned + non_blocking | P2 | 1d | 2 文件 |
| P2-4：Redis 序列化优化 | P2 | 2d | 2 文件 |

---

## 五、未纳入方案的建议及原因

| 建议 | 来源 | 不纳入原因 |
|------|------|-----------|
| KV cache allocator awareness (paged/block) | kv_cache文档 | 需要自定义 CUDA allocator，投入产出比低 |
| 用户偏好 embedding → selective KV | 优化文档 | 研究方向，需实验验证，非工程优化 |
| 历史检索异步化 | 优化文档 | 当前已是 async 接口，实际瓶颈不在此 |
| KV 二进制格式（自定义 header） | kv_cache文档 | `torch.save` 方案（P2-4）已足够，自定义格式维护成本高 |

---

## 六、总结

**一句话**：这个系统的正确性已经过充分验证，现在需要的是**从"学术正确"走向"系统最优"**。

**核心三件事**：
1. **停止滥用 `empty_cache()`** — 让 PyTorch allocator 做它该做的事
2. **给 Executor 加上边界** — LRU 缓存 + Fact Call 硬限制
3. **让 KV 数据流更粗粒度** — Packing + Pinned + non_blocking

这三件事做完，系统就从"能跑"变成"能扛"。

---

*文档生成时间：2026-02-18*
*基于代码审查版本：v3.2.0（含所有已修复的 bug）*
