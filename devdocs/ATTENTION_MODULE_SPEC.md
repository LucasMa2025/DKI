# DKI Attention 模块程序说明书

> 版本: 1.0.0  
> 最后更新: 2026-02-18  
> 作者: AGI Demo Project

---

## 1. 模块概述

### 1.1 定位

`dki/attention/` 是 DKI 系统的**注意力计算优化层**，负责将预计算的记忆 K/V 张量高效注入到 Transformer 的注意力计算中。它是 DKI 论文核心公式的底层实现：

```
Attention(Q, [K_mem; K_input], [V_mem; V_input]) = softmax(Q·[K_mem; K_input]ᵀ / √d) · [V_mem; V_input]
```

### 1.2 设计目标

| 目标 | 实现方式 |
|------|----------|
| **高性能** | FlashAttention-3/2 硬件加速，减少 ~50-70% 延迟 |
| **自动降级** | FA3 → FA2 → Standard 三级降级，保证兼容性 |
| **软注入** | Alpha 混合机制，支持 α ∈ [0, 1] 连续强度控制 |
| **大记忆支持** | 分块注入（Chunked Injection），处理超长 K/V 序列 |
| **可观测性** | 内置性能分析器，支持延迟/内存/FLOPS 监控 |

### 1.3 在 DKI 系统中的位置

```
用户输入
  ↓
DKIPlugin / DKISystem
  ↓
InjectionExecutor / HybridInjector
  ↓ compute_kv() → KVCacheEntry
  ↓
BaseModelAdapter.enable_flash_attention()
  ↓ 创建 KVInjectionOptimizer
  ↓
┌─────────────────────────────────┐
│  dki/attention/ (本模块)         │
│                                 │
│  FlashAttentionConfig  (配置)    │
│  FlashAttentionBackend (后端)    │
│  KVInjectionOptimizer  (注入)    │
│  AttentionProfiler     (监控)    │
└─────────────────────────────────┘
  ↓
模型推理输出
```

---

## 2. 文件结构

```
dki/attention/
├── __init__.py          # 统一导出接口
├── config.py            # 配置管理 (197 行)
├── backend.py           # GPU 检测与后端选择 (286 行)
├── kv_injection.py      # K/V 注入优化器 (457 行)
└── profiler.py          # 性能分析器 (309 行)
```

**总计**: ~1,294 行代码

---

## 3. 各文件详细说明

### 3.1 config.py — 配置管理

#### 职责
管理 FlashAttention 的所有配置参数，支持从 YAML 文件、字典和直接构造三种方式创建。

#### 类结构

```
FlashAttentionConfig          # 顶层配置（总控）
├── FA3Config                 # FlashAttention-3 特定配置 (H100+)
├── FA2Config                 # FlashAttention-2 配置 (A100/4090)
├── KVInjectionConfig         # K/V 注入优化配置
└── ProfilingConfig           # 性能监控配置
```

#### 类说明

| 类 | 字段 | 默认值 | 说明 |
|---|---|---|---|
| **FA3Config** | `use_fp8` | `False` | 是否启用 FP8 精度（降低内存，可能影响精度） |
| | `enable_async` | `True` | 是否启用异步执行（利用 TMA） |
| | `enable_warp_specialization` | `True` | 是否启用 Warp 特化 |
| | `num_stages` | `2` | 软件流水线阶段数 |
| **FA2Config** | `causal` | `False` | 是否启用 causal masking |
| | `dropout` | `0.0` | Dropout 率（推理时应为 0） |
| | `softmax_scale` | `None` | Softmax scale（None = 1/√d） |
| | `return_softmax` | `False` | 是否返回 softmax 统计（调试用） |
| **KVInjectionConfig** | `enabled` | `True` | 是否启用优化的 K/V 注入 |
| | `strategy` | `"prepend"` | 注入位置策略: `prepend`(前置) / `interleave`(交错) |
| | `chunked` | `True` | 是否使用分块注入 |
| | `chunk_size` | `1024` | 分块大小（token 数） |
| | `alpha_blending` | `True` | 是否启用 alpha 混合（软注入） |
| **ProfilingConfig** | `enabled` | `False` | 是否启用性能监控 |
| | `log_memory` | `True` | 是否记录内存使用 |
| | `log_latency` | `True` | 是否记录延迟 |
| | `log_flops` | `False` | 是否记录 FLOPS |
| | `log_path` | `None` | 日志输出路径 |

#### 配置加载方式

```python
# 方式 1: 从 YAML 文件
config = FlashAttentionConfig.from_yaml("config/config.yaml")

# 方式 2: 从字典
config = FlashAttentionConfig.from_dict({
    "enabled": True,
    "backend": "auto",
    "kv_injection": {"strategy": "prepend", "chunk_size": 2048},
})

# 方式 3: 直接构造
config = FlashAttentionConfig(
    enabled=True,
    backend="fa3",
    fa3=FA3Config(use_fp8=True),
)

# 序列化
config_dict = config.to_dict()
```

#### YAML 配置示例

```yaml
flash_attention:
  enabled: true
  backend: auto           # auto | fa3 | fa2 | standard
  
  fa3:
    use_fp8: false
    enable_async: true
    enable_warp_specialization: true
    num_stages: 2
  
  fa2:
    causal: false
    dropout: 0.0
    softmax_scale: null
  
  kv_injection:
    enabled: true
    strategy: prepend      # prepend | interleave
    chunked: true
    chunk_size: 1024
    alpha_blending: true
  
  profiling:
    enabled: false
    log_memory: true
    log_latency: true
    log_path: logs/attention_profile.json
```

---

### 3.2 backend.py — GPU 检测与后端选择

#### 职责
1. 检测当前 GPU 硬件能力
2. 选择最佳 FlashAttention 后端
3. 提供标准注意力和 FlashAttention 的统一前向接口

#### 后端降级策略

```
FlashAttention-3 (FA3)          ← H100/H200 (compute capability ≥ 9.0)
       │ 不支持
       ↓
FlashAttention-2 (FA2)          ← A100/RTX 4090/A10 (compute capability ≥ 8.0)
       │ 不支持
       ↓
Standard PyTorch                ← V100/RTX 20xx 或无 GPU
```

#### GPU 能力映射表

| GPU | Compute Capability | 后端 |
|-----|-------------------|------|
| H100/H200 | (9, 0) | FA3 |
| RTX 4090 | (8, 9) | FA2 |
| A100 | (8, 0) | FA2 |
| A10/A30 | (8, 6) | FA2 |
| RTX 20xx | (7, 5) | Standard |
| V100 | (7, 0) | Standard |

#### 全局常量

| 常量 | 类型 | 说明 |
|------|------|------|
| `FLASH_ATTN_AVAILABLE` | `bool` | FlashAttention 库是否已安装 |
| `FLASH_ATTN_VERSION` | `str \| None` | FlashAttention 版本号 |

#### 类: FlashAttentionBackend

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `detect_best_backend()` | `str` | 自动检测最佳后端（`"fa3"` / `"fa2"` / `"standard"`） |
| `supports_fa3()` | `bool` | 是否支持 FA3 |
| `supports_fa2()` | `bool` | 是否支持 FA2 |
| `get_gpu_info()` | `Dict` | 获取完整 GPU 信息（名称、显存、计算能力等） |
| `validate_backend(backend)` | `str` | 验证请求的后端是否可用，不可用则自动降级 |

#### 函数: flash_attention_forward

```python
def flash_attention_forward(
    query: Tensor,    # [batch, seq_q, heads, head_dim]  (FlashAttention 格式)
    key: Tensor,      # [batch, seq_k, heads, head_dim]
    value: Tensor,    # [batch, seq_k, heads, head_dim]
    backend: str = "auto",
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
) -> Tensor:          # [batch, seq_q, heads, head_dim]
```

**统一前向接口**。根据 `backend` 参数自动选择 FlashAttention 或标准实现。内部处理张量格式转换：

- FlashAttention 格式: `[B, S, H, D]`
- 标准 PyTorch 格式: `[B, H, S, D]`

当 FlashAttention 调用失败时，自动降级到标准实现。

#### 函数: scaled_dot_product_attention_standard

```python
def scaled_dot_product_attention_standard(
    query: Tensor,    # [batch, heads, seq_q, head_dim]  (标准格式)
    key: Tensor,      # [batch, heads, seq_k, head_dim]
    value: Tensor,    # [batch, heads, seq_k, head_dim]
    attn_mask: Tensor = None,
    dropout_p: float = 0.0,
    scale: float = None,
) -> Tensor:          # [batch, heads, seq_q, head_dim]
```

**标准 Scaled Dot-Product Attention 实现**。作为 FlashAttention 不可用时的兼容后备方案。

---

### 3.3 kv_injection.py — K/V 注入优化器

#### 职责
将预计算的记忆 K/V 张量注入到注意力计算中，是 DKI 系统的核心算法实现。

#### 数据结构: InjectionResult

```python
@dataclass
class InjectionResult:
    output: Tensor       # 注入后的注意力输出
    latency_ms: float    # 计算延迟（毫秒）
    memory_key_len: int  # 记忆 K/V 序列长度
    input_key_len: int   # 输入 K/V 序列长度
    total_key_len: int   # 总 K/V 序列长度 (memory + input)
    backend_used: str    # 使用的后端
    alpha: float         # 注入强度
```

#### 类: KVInjectionOptimizer

##### 初始化

```python
optimizer = KVInjectionOptimizer(
    config=FlashAttentionConfig(),   # 可选，使用默认配置
    backend="auto",                   # 可选，覆盖配置中的后端
)
```

##### 核心方法

**1. `inject()` — 标准 K/V 注入**

```python
result = optimizer.inject(
    query=query,              # [B, seq_q, H, D]
    key=key,                  # [B, seq_k, H, D]
    value=value,              # [B, seq_k, H, D]
    memory_key=memory_k,      # [B, seq_m, H, D]
    memory_value=memory_v,    # [B, seq_m, H, D]
    alpha=0.5,                # 注入强度
    causal=False,             # 是否 causal masking
)

output = result.output        # [B, seq_q, H, D]
```

**内部流程:**

```
1. 确保 memory K/V 与 query 在同一设备
2. 拼接: full_key = [memory_key; key], full_value = [memory_value; value]
3. 调用 flash_attention_forward(query, full_key, full_value)
4. 如果 alpha < 1.0 且启用 alpha_blending:
   a. 计算原始输出: original = flash_attention_forward(query, key, value)
   b. 混合: output = α * injected + (1-α) * original
5. 清理中间张量
```

**2. `inject_chunked()` — 分块 K/V 注入**

```python
result = optimizer.inject_chunked(
    query=query,
    key=key,
    value=value,
    memory_key=large_memory_k,    # 可能非常大
    memory_value=large_memory_v,
    alpha=0.5,
    chunk_size=1024,              # 可选，默认从配置读取
)
```

**适用场景:** 当 memory K/V 序列非常长时（> chunk_size），分块处理以避免单次拼接过大。

**算法:**

```
1. 如果 seq_m ≤ chunk_size → 直接调用 inject()
2. 否则:
   a. 将 memory K/V 按 chunk_size 分块
   b. 对每个 chunk 计算 Q @ K_chunk 的注意力分数
   c. 将所有 memory chunk 分数 + input 分数拼接
   d. 统一做 softmax + matmul
   e. 结果数学上等价于完整拼接的注入
3. 如果 alpha < 1.0 → 执行 alpha 混合
```

**数学等价性保证:**

```
inject_chunked(Q, K, V, K_mem, V_mem) ≡ inject(Q, K, V, K_mem, V_mem)
```

> **注意:** 当前实现通过拼接所有分块分数后统一 softmax，保证数学正确性，但内存峰值与不分块相同。真正的内存优化需实现 online softmax (log-sum-exp)。

**3. `inject_multi_layer()` — 多层 K/V 注入**

```python
results = optimizer.inject_multi_layer(
    queries=[q_layer0, q_layer1, ...],       # 每层的 Q
    keys=[k_layer0, k_layer1, ...],          # 每层的 K
    values=[v_layer0, v_layer1, ...],        # 每层的 V
    memory_keys=[mk_layer0, mk_layer1, ...], # 每层的记忆 K
    memory_values=[mv_layer0, mv_layer1, ...],
    alpha=0.5,
    layer_alphas=[0.3, 0.4, 0.5, ...],       # 可选，每层独立 alpha
)
```

**适用场景:** Transformer 多层注入。支持全局 alpha 或每层独立 alpha。

##### 统计与监控

```python
# 获取统计
stats = optimizer.get_stats()
# {
#   "total_injections": 100,
#   "total_latency_ms": 1234.5,
#   "avg_latency_ms": 12.3,
#   "avg_memory_len": 128.0,
#   "backend": "fa2",
#   "backend_usage": {"fa3": 0, "fa2": 100, "standard": 0},
#   "config": {"strategy": "prepend", "chunked": true, ...}
# }

# 重置统计
optimizer.reset_stats()
```

---

### 3.4 profiler.py — 性能分析器

#### 职责
监控和记录注意力计算的性能数据，支持延迟、内存、FLOPS 跟踪，生成分析报告。

#### 数据结构: ProfileRecord

```python
@dataclass
class ProfileRecord:
    operation: str              # 操作名称
    latency_ms: float           # 延迟（毫秒）
    memory_allocated_mb: float  # GPU 已分配内存 (MB)
    memory_reserved_mb: float   # GPU 已保留内存 (MB)
    input_shape: str            # 输入张量形状
    output_shape: str           # 输出张量形状
    backend: str                # 使用的后端
    timestamp: datetime         # 记录时间
    metadata: Dict              # 额外元数据
```

#### 类: AttentionProfiler

##### 使用方式 1: 上下文管理器（推荐）

```python
profiler = AttentionProfiler(config=ProfilingConfig(enabled=True))

with profiler.profile("kv_injection", backend="fa2"):
    result = optimizer.inject(query, key, value, memory_k, memory_v)

with profiler.profile("chunked_injection", backend="fa2", metadata={"chunks": 4}):
    result = optimizer.inject_chunked(...)
```

即使 `yield` 块内抛出异常，也会正确记录性能数据。

##### 使用方式 2: 手动记录

```python
start = time.perf_counter()
result = optimizer.inject(...)
latency = (time.perf_counter() - start) * 1000

profiler.record(
    operation="kv_injection",
    latency_ms=latency,
    input_tensor=query,
    output_tensor=result.output,
    backend="fa2",
)
```

##### 生成报告

```python
report = profiler.get_report()
# {
#   "total_records": 100,
#   "time_range": {"start": "...", "end": "..."},
#   "operations": {
#     "kv_injection": {
#       "count": 80,
#       "total_latency_ms": 960.0,
#       "avg_latency_ms": 12.0,
#       "min_latency_ms": 8.5,
#       "max_latency_ms": 25.3,
#       "p50_latency_ms": 11.2,
#       "p90_latency_ms": 18.7,
#       "p99_latency_ms": 24.1,
#     },
#     "chunked_injection": { ... }
#   },
#   "backend_usage": {"fa2": 80, "standard": 20},
#   "memory": {
#     "avg_allocated_mb": 1024.5,
#     "max_allocated_mb": 2048.0,
#     "avg_reserved_mb": 1536.0,
#   }
# }
```

##### 导出到文件

```python
profiler.export_to_file("logs/attention_profile.json")
```

##### 控制

```python
profiler.enabled = True    # 动态启用
profiler.enabled = False   # 动态禁用（profile() 变为 no-op）
profiler.clear()           # 清除所有记录
len(profiler)              # 返回记录数量
```

---

## 4. 集成点

### 4.1 被调用方

本模块被以下组件调用：

| 调用方 | 文件 | 使用方式 |
|--------|------|----------|
| `BaseModelAdapter` | `dki/models/base.py` | `enable_flash_attention()` 创建 `KVInjectionOptimizer` 实例 |
| `VLLMAdapter` | `dki/models/vllm_adapter.py` | 文档中引用（当前使用 HuggingFace `past_key_values` 路径） |

### 4.2 典型调用流程

```python
# 1. 模型适配器启用 FlashAttention
model_adapter.enable_flash_attention(FlashAttentionConfig(backend="auto"))

# 2. 内部创建 KVInjectionOptimizer
#    self._kv_injection_optimizer = KVInjectionOptimizer(config, backend)

# 3. 推理时注入记忆 K/V
result = model_adapter._kv_injection_optimizer.inject(
    query=q, key=k, value=v,
    memory_key=mem_k, memory_value=mem_v,
    alpha=gating_decision.alpha,
)
```

### 4.3 当前限制

| 限制 | 说明 |
|------|------|
| vLLM 集成 | `vllm_adapter.py` 目前使用 HuggingFace `generate()` + `past_key_values` 进行 K/V 注入，尚未直接调用 `KVInjectionOptimizer`。这是因为 vLLM 不暴露逐层注意力钩子。 |
| 分块内存优化 | `inject_chunked()` 保证数学正确性，但内存峰值与不分块相同。真正的在线 softmax 优化尚未实现。 |
| Interleave 策略 | `KVInjectionConfig.strategy = "interleave"` 当前回退到 `prepend` 实现。 |

---

## 5. 张量格式约定

### 5.1 FlashAttention 格式（本模块的标准格式）

```
[batch_size, sequence_length, num_heads, head_dim]
简写: [B, S, H, D]
```

### 5.2 标准 PyTorch 格式

```
[batch_size, num_heads, sequence_length, head_dim]
简写: [B, H, S, D]
```

### 5.3 格式转换

`flash_attention_forward()` 内部自动处理格式转换：

```python
# FlashAttention → Standard: tensor.transpose(1, 2)
# Standard → FlashAttention: tensor.transpose(1, 2)
```

**调用者无需关心格式转换**，只需统一使用 FlashAttention 格式 `[B, S, H, D]`。

---

## 6. Alpha 混合机制

### 6.1 公式

当 `alpha < 1.0` 且 `alpha_blending = True` 时：

```
output = α × Attention(Q, [K_mem; K], [V_mem; V]) + (1-α) × Attention(Q, K, V)
```

- `α = 1.0`: 完全注入（记忆完全参与注意力）
- `α = 0.5`: 半注入（记忆影响减半）
- `α = 0.0`: 无注入（等价于原始模型）

### 6.2 设计不变量

> **Key 张量永远不被 alpha 缩放**（保护 attention addressing）

Alpha 仅作用于最终输出的线性混合，不修改注意力分数的计算过程。

### 6.3 与 MIS 的关系

`KVInjectionOptimizer` 的 `alpha` 参数由上层的 **Memory Influence Scaling (MIS)** 和 **Dual-Factor Gating** 决定：

```
MIS → alpha_profile → Gating → final_alpha → KVInjectionOptimizer.inject(alpha=final_alpha)
```

---

## 7. 性能参考

### 7.1 延迟对比（典型场景）

| 后端 | seq_q=128, seq_m=64 | seq_q=512, seq_m=128 | seq_q=2048, seq_m=256 |
|------|---------------------|----------------------|-----------------------|
| FA3 (H100) | ~0.8ms | ~2.1ms | ~6.5ms |
| FA2 (A100) | ~1.2ms | ~3.5ms | ~11.0ms |
| Standard | ~2.5ms | ~8.0ms | ~35.0ms |

### 7.2 内存对比

| 后端 | 相对内存使用 |
|------|------------|
| FA3 | ~60% of Standard |
| FA2 | ~70% of Standard |
| Standard | 100% (baseline) |

---

## 8. 错误处理与降级

### 8.1 降级链

```
FA3 调用失败 → 自动降级到 FA2
FA2 调用失败 → 自动降级到 Standard
Standard 调用失败 → 抛出异常（由上层捕获）
```

### 8.2 FlashAttention 未安装

```python
FLASH_ATTN_AVAILABLE = False  # 自动检测

# detect_best_backend() → "standard"
# supports_fa3() → False
# supports_fa2() → False
# flash_attention_forward() → 自动使用 standard 路径
```

### 8.3 无 GPU

```python
torch.cuda.is_available() == False

# detect_best_backend() → "standard"
# _get_memory_stats() → {} (空字典)
# 所有计算在 CPU 上执行
```

---

## 9. 配置推荐

### 9.1 生产环境（H100）

```yaml
flash_attention:
  enabled: true
  backend: auto          # 自动检测为 fa3
  fa3:
    use_fp8: false       # 精度优先
    enable_async: true
  kv_injection:
    strategy: prepend
    chunked: true
    chunk_size: 2048
    alpha_blending: true
  profiling:
    enabled: false       # 生产环境关闭
```

### 9.2 生产环境（A100/4090）

```yaml
flash_attention:
  enabled: true
  backend: auto          # 自动检测为 fa2
  kv_injection:
    strategy: prepend
    chunked: true
    chunk_size: 1024
    alpha_blending: true
  profiling:
    enabled: false
```

### 9.3 开发/调试环境

```yaml
flash_attention:
  enabled: true
  backend: standard      # 强制标准模式，便于调试
  kv_injection:
    strategy: prepend
    chunked: false       # 关闭分块，简化调试
    alpha_blending: true
  profiling:
    enabled: true        # 启用性能监控
    log_memory: true
    log_latency: true
    log_path: logs/attention_debug.json
```

### 9.4 无 GPU 环境

```yaml
flash_attention:
  enabled: false          # 完全禁用
  backend: standard
  profiling:
    enabled: false
```

---

## 10. 公共 API 速查表

### 导入

```python
from dki.attention import (
    FlashAttentionConfig,     # 配置
    FlashAttentionBackend,    # 后端检测
    KVInjectionOptimizer,     # K/V 注入
    AttentionProfiler,        # 性能分析
    FLASH_ATTN_AVAILABLE,     # 是否可用
    FLASH_ATTN_VERSION,       # 版本号
)
```

### FlashAttentionConfig

| 方法 | 说明 |
|------|------|
| `from_dict(data)` | 从字典创建 |
| `from_yaml(path)` | 从 YAML 文件创建 |
| `to_dict()` | 序列化为字典 |

### FlashAttentionBackend

| 方法 | 说明 |
|------|------|
| `detect_best_backend()` | 自动检测最佳后端 |
| `supports_fa3()` | 是否支持 FA3 |
| `supports_fa2()` | 是否支持 FA2 |
| `get_gpu_info()` | 获取 GPU 信息 |
| `validate_backend(backend)` | 验证并降级后端 |

### KVInjectionOptimizer

| 方法 | 说明 |
|------|------|
| `inject(q, k, v, mem_k, mem_v, alpha)` | 标准 K/V 注入 |
| `inject_chunked(q, k, v, mem_k, mem_v, alpha, chunk_size)` | 分块 K/V 注入 |
| `inject_multi_layer(qs, ks, vs, mem_ks, mem_vs, alpha)` | 多层 K/V 注入 |
| `get_stats()` | 获取统计 |
| `reset_stats()` | 重置统计 |
| `backend` (property) | 当前后端 |

### AttentionProfiler

| 方法 | 说明 |
|------|------|
| `profile(op, backend, metadata)` | 上下文管理器 |
| `record(op, latency_ms, ...)` | 手动记录 |
| `get_report()` | 生成报告（含 P50/P90/P99） |
| `get_records()` | 获取所有原始记录 |
| `export_to_file(path)` | 导出到 JSON 文件 |
| `clear()` | 清除记录 |
| `enabled` (property) | 启用/禁用 |

### 独立函数

| 函数 | 说明 |
|------|------|
| `flash_attention_forward(q, k, v, backend, ...)` | 统一注意力前向（自动选择后端） |
| `scaled_dot_product_attention_standard(q, k, v, ...)` | 标准注意力实现 |

---

## 11. 测试覆盖

本模块有 **42 个单元测试**（位于 `tests/unit/test_attention_review_2026_02_18.py`），覆盖：

| 测试类别 | 数量 | 覆盖内容 |
|----------|------|----------|
| 数学等价性 | 6 | inject 与手动拼接等价、inject_chunked 与 inject 等价、alpha 混合正确性 |
| 内存清理 | 4 | 中间张量释放、GPU 内存清理 |
| 后端检测 | 8 | GPU 能力映射、降级策略、无 GPU 处理 |
| 配置边界 | 6 | from_dict/to_dict 序列化、from_yaml、默认值 |
| Profiler | 10 | 上下文管理器、手动记录、P50/P90/P99、异常处理 |
| 统计 | 4 | 平均记忆长度、后端使用统计、配置报告 |
| 设备一致性 | 2 | 跨设备张量自动迁移 |
| Alpha 混合 | 2 | alpha_blending 禁用时的行为 |
