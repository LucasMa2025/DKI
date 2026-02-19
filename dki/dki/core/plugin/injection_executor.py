"""
DKI Injection Executor - 注入计划执行器

职责: 只做执行, 不做决策
- 根据 InjectionPlan 执行 recall_v4 或 stable 策略
- 计算偏好 K/V (含缓存)
- 调用模型推理 (带 K/V 注入)
- 处理降级 (fallback)

输入: InjectionPlan + ModelAdapter
输出: ExecutionResult

不变量:
- Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
- Preference alpha 永远不超过 AlphaProfile.override_cap
- 任何异常自动降级到无注入推理
- Executor 只做一次 O(1) forward pass (v3.3: Fact Call 已移至 Planner)

安全不变量 (v3.1):
- 偏好 K/V 缓存按用户物理分区，不同用户的 K/V 不共享数据结构
- 每次缓存访问都验证 user_id 归属权
- 推理前后有上下文隔离检查，防止 K/V 残留泄露

策略 (v3.3):
- recall_v4 (默认): 偏好 K/V + 后缀组装 (事实已由 Planner 内联)
- stable (回退): recall_v4 失败时自动降级到 stable (偏好 K/V + 平铺历史后缀)
- full_attention 已移除

Author: AGI Demo Project
Version: 3.3.0
"""

import re
import time
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry, PackedKV
from dki.core.plugin.injection_plan import (
    InjectionPlan,
    ExecutionResult,
)


# ============================================================
# F1-4: retrieve_fact 工具调用过滤正则
# ============================================================
# v3.3 将 Fact Call 移至 Planner, Executor 不再循环推理。
# 但模型仍可能在输出中生成 retrieve_fact(...) 调用 (因为 prompt
# 中可能残留 constraint instruction)。这些调用对最终用户无意义,
# 应被静默剥离。
#
# 支持三种格式:
# 1. Generic:  retrieve_fact(trace_id="...", ...)
# 2. DeepSeek: <｜tool▁call▁begin｜>retrieve_fact\n{...}<｜tool▁call▁end｜>
# 3. GLM:      <|tool_call|>retrieve_fact\n{...}

_RETRIEVE_FACT_GENERIC_RE = re.compile(
    r'retrieve_fact\s*\([^)]*\)',
    re.IGNORECASE,
)

_RETRIEVE_FACT_DEEPSEEK_RE = re.compile(
    r'<｜tool▁calls?▁begin｜>.*?<｜tool▁calls?▁end｜>',
    re.DOTALL,
)

_RETRIEVE_FACT_GLM_RE = re.compile(
    r'<\|tool_call\|>\s*retrieve_fact\s*\n\s*\{[^}]*\}',
    re.DOTALL,
)

_ALL_FACT_CALL_PATTERNS = [
    _RETRIEVE_FACT_DEEPSEEK_RE,
    _RETRIEVE_FACT_GLM_RE,
    _RETRIEVE_FACT_GENERIC_RE,
]


def _strip_retrieve_fact_calls(text: str) -> Tuple[str, int]:
    """
    F1-4: 从模型输出中剥离残留的 retrieve_fact 工具调用
    
    Returns:
        (cleaned_text, stripped_count)
    """
    stripped = 0
    for pattern in _ALL_FACT_CALL_PATTERNS:
        matches = pattern.findall(text)
        stripped += len(matches)
        text = pattern.sub("", text)
    
    # 清理多余空行 (剥离后可能留下连续空行)
    if stripped > 0:
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    return text, stripped


# ============================================================
# BoundedUserKVCache - 有界的用户级 KV 缓存 (P0-2)
# ============================================================

class BoundedUserKVCache:
    """
    有界的用户级 KV 缓存 (P0-2 优化)
    
    替换原来的无界 Dict[str, Dict[str, Tuple]]，提供双层 LRU 限制:
    - max_users: 最大用户数 (LRU 淘汰最久未访问的用户)
    - max_entries_per_user: 每用户最大条目数 (LRU 淘汰最旧的偏好 KV)
    
    安全不变量:
    - 不同用户的 KV 存储在独立的 OrderedDict 中 (物理分区)
    - 每次 get/put 都更新 LRU 顺序
    - 淘汰时 del evicted_data 释放 CPU tensor 引用
    
    内存可预测性:
    - 最大 CPU 内存 ≈ max_users × max_entries_per_user × kv_size_per_entry
    - 例: 500 用户 × 5 条目 × 10MB/条目 = 25GB (上限)
    """
    
    def __init__(
        self,
        max_users: int = 500,
        max_entries_per_user: int = 5,
    ):
        self._max_users = max_users
        self._max_entries_per_user = max_entries_per_user
        # OrderedDict 保证 LRU 顺序: 最近访问的在末尾
        self._cache: OrderedDict[str, OrderedDict[str, Tuple]] = OrderedDict()
        # 统计
        self._evictions_user = 0
        self._evictions_entry = 0
    
    def get(
        self, user_id: str, content_hash: str
    ) -> Optional[Tuple]:
        """
        获取缓存条目
        
        Args:
            user_id: 用户 ID
            content_hash: 偏好内容哈希
            
        Returns:
            (kv_entries, content_hash) 元组，未命中返回 None
        """
        if user_id not in self._cache:
            return None
        
        # LRU touch: 将用户移到末尾 (最近访问)
        self._cache.move_to_end(user_id)
        user_partition = self._cache[user_id]
        
        entry = user_partition.get(content_hash)
        if entry is not None:
            # LRU touch: 将条目移到末尾
            user_partition.move_to_end(content_hash)
        
        return entry
    
    def put(
        self, user_id: str, content_hash: str, value: Tuple
    ) -> None:
        """
        存入缓存条目
        
        如果用户数超限，淘汰最久未访问的用户。
        如果该用户的条目数超限，淘汰该用户最旧的条目。
        
        Args:
            user_id: 用户 ID
            content_hash: 偏好内容哈希
            value: (kv_entries, content_hash) 元组
        """
        if user_id not in self._cache:
            # 检查用户数上限
            if len(self._cache) >= self._max_users:
                # 淘汰最久未访问的用户 (OrderedDict 头部)
                evicted_uid, evicted_partition = self._cache.popitem(last=False)
                # 释放该用户所有 KV tensor 引用
                evicted_partition.clear()
                del evicted_partition
                self._evictions_user += 1
                logger.debug(
                    f"BoundedUserKVCache: evicted user {evicted_uid} "
                    f"(total user evictions: {self._evictions_user})"
                )
            self._cache[user_id] = OrderedDict()
        
        # LRU touch: 将用户移到末尾
        self._cache.move_to_end(user_id)
        user_partition = self._cache[user_id]
        
        # 检查条目数上限
        if content_hash not in user_partition and len(user_partition) >= self._max_entries_per_user:
            # 淘汰该用户最旧的条目
            evicted_hash, evicted_data = user_partition.popitem(last=False)
            del evicted_data
            self._evictions_entry += 1
        
        user_partition[content_hash] = value
    
    def clear(self, user_id: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            user_id: 指定用户 ID 则只清除该用户，None 则清除所有
        """
        if user_id:
            partition = self._cache.pop(user_id, None)
            if partition:
                partition.clear()
                del partition
        else:
            for partition in self._cache.values():
                partition.clear()
            self._cache.clear()
    
    def __contains__(self, user_id: str) -> bool:
        return user_id in self._cache
    
    @property
    def user_count(self) -> int:
        """当前缓存的用户数"""
        return len(self._cache)
    
    @property
    def total_entries(self) -> int:
        """所有用户的总缓存条目数"""
        return sum(len(p) for p in self._cache.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "max_users": self._max_users,
            "max_entries_per_user": self._max_entries_per_user,
            "current_users": self.user_count,
            "total_entries": self.total_entries,
            "evictions_user": self._evictions_user,
            "evictions_entry": self._evictions_entry,
        }

# 用户隔离组件
try:
    from dki.cache.user_isolation import (
        UserIsolationContext,
        InferenceContextGuard,
        CacheKeySigner,
    )
    USER_ISOLATION_AVAILABLE = True
except ImportError:
    USER_ISOLATION_AVAILABLE = False

# Recall v4 组件不再需要 (v3.3: Fact Call 已移至 Planner)
# 保留标志位以兼容旧代码
RECALL_V4_AVAILABLE = True


class InjectionExecutor:
    """
    注入计划执行器 (v3.3)
    
    纯执行层，根据 InjectionPlan 调用模型。
    支持 recall_v4 和 stable 两种策略:
    - recall_v4: 偏好 K/V 注入 + 后缀组装 (事实已由 Planner 内联)
    - stable: 偏好 K/V 注入 + 平铺历史后缀 (recall_v4 失败时的回退策略)
    
    不变量:
    - Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
    - Preference alpha 永远不超过 AlphaProfile.override_cap
    - 任何异常自动降级到无注入推理
    - Executor 只做一次 O(1) forward pass (v3.3: 无 Fact Call 循环)
    
    安全不变量 (v3.1):
    - 偏好 K/V 缓存按 user_id 物理分区
    - 每次缓存访问验证 user_id 归属
    - 推理上下文隔离 (InferenceContextGuard)
    
    用法:
        executor = InjectionExecutor(model_adapter=model)
        result = await executor.execute(plan, max_new_tokens=512)
        
        # result.text 是生成结果
        # result.inference_latency_ms 是推理耗时
        # result.preference_cache_hit 是缓存命中情况
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        # Function Call 日志记录器 (v3.2, 保留用于记录 Planner 侧事实解析)
        function_call_logger: Optional[Any] = None,
    ):
        """
        初始化执行器
        
        Args:
            model_adapter: LLM 模型适配器
            function_call_logger: Function Call 日志记录器 (可选)
        """
        self.model = model_adapter
        
        # Function Call 日志记录器
        self._fc_logger = function_call_logger
        
        # ============ 用户级隔离的偏好 K/V 缓存 (P0-2: BoundedUserKVCache) ============
        # 改进: 从无界 Dict 升级为有界 LRU 缓存
        # 双层限制: max_users (用户级 LRU) + max_entries_per_user (条目级 LRU)
        self._preference_kv_cache = BoundedUserKVCache(
            max_users=500,
            max_entries_per_user=5,
        )
        
        # 推理上下文守卫 (防止 K/V 残留泄露)
        self._inference_guard: Optional[InferenceContextGuard] = None
        if USER_ISOLATION_AVAILABLE:
            self._inference_guard = InferenceContextGuard()
        
        # 统计
        self._stats = {
            "executions": 0,
            "recall_v4_executions": 0,
            "plain_executions": 0,
            "fallbacks": 0,
            "cache_hits": 0,
            "cache_user_isolation_denials": 0,
        }
    
    # ================================================================
    # 主执行入口
    # ================================================================
    
    async def execute(
        self,
        plan: InjectionPlan,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> ExecutionResult:
        """
        执行注入计划 (v3.3: O(1) forward pass, 无 Fact Call 循环)
        
        执行路径:
        - recall_v4 + 有偏好 + 注入启用: K/V 注入推理 (事实已由 Planner 内联)
        - stable + 有偏好 + 注入启用: K/V 注入推理 (偏好 + 历史后缀)
        - 无偏好 / 无注入: 直接生成
        - 异常: 降级到 stable → 再失败则降级到无注入
        
        Args:
            plan: 注入计划 (来自 Planner, 事实已预解析)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            ExecutionResult 包含生成结果和执行时间
        """
        self._stats["executions"] += 1
        
        try:
            if plan.injection_enabled and plan.preference_text:
                # K/V 注入推理 (recall_v4 和 stable 共用)
                # v3.3: 不再有 Fact Call 循环, Planner 已将事实内联到 plan.final_input
                result = await self._execute_with_kv_injection(
                    plan, max_new_tokens, temperature, **kwargs
                )
                
                # 填充 Planner 侧事实解析信息到结果
                result.fact_blocks_resolved = len(plan.fact_blocks)
                result.fact_tokens_total = plan.fact_tokens
                result.fact_strategy = plan.fact_strategy
                
                if plan.strategy == "recall_v4":
                    self._stats["recall_v4_executions"] += 1
                else:
                    self._stats.setdefault("stable_executions", 0)
                    self._stats["stable_executions"] += 1
            else:
                result = await self._execute_plain(
                    plan, max_new_tokens, temperature, **kwargs
                )
                self._stats["plain_executions"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            
            # 第一级降级: 尝试 stable 策略 (如果当前是 recall_v4)
            if plan.strategy == "recall_v4" and plan.preference_text:
                logger.info("Falling back to stable strategy")
                try:
                    return await self._execute_stable_fallback(
                        plan, max_new_tokens, temperature, str(e), **kwargs
                    )
                except Exception as stable_error:
                    logger.error(f"Stable fallback also failed: {stable_error}")
            
            # 第二级降级: 无注入推理
            return await self._execute_fallback(
                plan, max_new_tokens, temperature, str(e), **kwargs
            )
    
    # ================================================================
    # K/V 注入推理 (recall_v4 统一策略)
    # ================================================================
    
    async def _execute_with_kv_injection(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """
        执行 K/V 注入推理 (recall_v4 统一策略)
        
        偏好: K/V 注入 (负位置)
        历史: 由 recall_v4 后缀组装 (已包含在 plan.final_input 中)
        
        安全: 使用 InferenceContextGuard 确保推理上下文隔离
        """
        result = ExecutionResult()
        
        # 获取偏好 K/V (含用户级隔离缓存)
        preference_kv, cache_hit, cache_tier = self._get_preference_kv(
            user_id=plan.user_id,
            preference_text=plan.preference_text,
        )
        result.preference_cache_hit = cache_hit
        result.preference_cache_tier = cache_tier
        
        # ============ P1-4: KV 监控指标 ============
        if preference_kv and isinstance(preference_kv, list) and len(preference_kv) > 0:
            result.kv_layers_count = len(preference_kv)
            try:
                result.kv_bytes_cpu = sum(
                    e.key.nelement() * e.key.element_size()
                    + e.value.nelement() * e.value.element_size()
                    for e in preference_kv
                )
            except Exception:
                pass  # 非关键路径，不影响推理
        
        # 使用 effective alpha (受 override_cap 约束)
        alpha = plan.alpha_profile.effective_preference_alpha
        
        # 推理 (带上下文隔离守卫)
        inference_start = time.time()
        
        if preference_kv and alpha > 0.1:
            # 带 K/V 注入的推理 (使用推理隔离守卫)
            if self._inference_guard:
                with self._inference_guard.scoped_inference(
                    user_id=plan.user_id,
                    kv_entries=preference_kv,
                ):
                    output = self.model.forward_with_kv_injection(
                        prompt=plan.final_input,
                        injected_kv=preference_kv,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
            else:
                output = self.model.forward_with_kv_injection(
                    prompt=plan.final_input,
                    injected_kv=preference_kv,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
        else:
            # 无注入推理 (alpha 太低或无 K/V)
            output = self.model.generate(
                prompt=plan.final_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        result.inference_latency_ms = (time.time() - inference_start) * 1000
        result.text = output.text
        result.input_tokens = output.input_tokens
        result.output_tokens = output.output_tokens
        result.raw_output = output
        
        # F1-4: 防御性拦截 — 剥离残留的 retrieve_fact 工具调用
        result.text, fact_stripped = _strip_retrieve_fact_calls(result.text)
        if fact_stripped > 0:
            logger.info(
                f"F1-4: Stripped {fact_stripped} residual retrieve_fact call(s) from output"
            )
            self._stats.setdefault("fact_calls_stripped", 0)
            self._stats["fact_calls_stripped"] += fact_stripped
        
        return result
    
    # ================================================================
    # 无注入执行
    # ================================================================
    
    async def _execute_plain(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """执行无注入推理"""
        result = ExecutionResult()
        
        inference_start = time.time()
        output = self.model.generate(
            prompt=plan.final_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        result.inference_latency_ms = (time.time() - inference_start) * 1000
        result.text = output.text
        result.input_tokens = output.input_tokens
        result.output_tokens = output.output_tokens
        result.raw_output = output
        
        # F1-4: 防御性拦截
        result.text, fact_stripped = _strip_retrieve_fact_calls(result.text)
        if fact_stripped > 0:
            logger.info(f"F1-4: Stripped {fact_stripped} residual retrieve_fact call(s) from plain output")
            self._stats.setdefault("fact_calls_stripped", 0)
            self._stats["fact_calls_stripped"] += fact_stripped
        
        return result
    
    # ================================================================
    # Stable 策略降级 (recall_v4 失败时的第一级降级)
    # ================================================================
    
    async def _execute_stable_fallback(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        error_message: str,
        **kwargs,
    ) -> ExecutionResult:
        """
        Stable 策略降级: 使用偏好 K/V + 平铺历史后缀
        
        当 recall_v4 执行失败时, 回退到 stable 策略:
        - 偏好: K/V 注入 (不变)
        - 历史: 使用 plan.history_suffix (平铺历史后缀)
        """
        self._stats.setdefault("stable_fallbacks", 0)
        self._stats["stable_fallbacks"] += 1
        
        result = ExecutionResult()
        result.fallback_used = True
        result.error_message = f"stable_fallback: {error_message}"
        
        # 构造 stable 输入: 历史后缀 + 原始查询
        # 注意: recall_v4 会将 assembled_suffix 同步到 history_suffix,
        # stable 降级应使用历史后缀 (即使它等于 assembled_suffix)
        if plan.history_suffix:
            stable_input = plan.history_suffix + "\n\n" + plan.original_query
        else:
            stable_input = plan.original_query
        
        # 获取偏好 K/V
        preference_kv, cache_hit, cache_tier = self._get_preference_kv(
            user_id=plan.user_id,
            preference_text=plan.preference_text,
        )
        result.preference_cache_hit = cache_hit
        result.preference_cache_tier = cache_tier
        
        alpha = plan.alpha_profile.effective_preference_alpha
        
        inference_start = time.time()
        
        if preference_kv and alpha > 0.1:
            if self._inference_guard:
                with self._inference_guard.scoped_inference(
                    user_id=plan.user_id,
                    kv_entries=preference_kv,
                ):
                    output = self.model.forward_with_kv_injection(
                        prompt=stable_input,
                        injected_kv=preference_kv,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
            else:
                output = self.model.forward_with_kv_injection(
                    prompt=stable_input,
                    injected_kv=preference_kv,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
        else:
            output = self.model.generate(
                prompt=stable_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        result.inference_latency_ms = (time.time() - inference_start) * 1000
        result.text = output.text
        result.input_tokens = output.input_tokens
        result.output_tokens = output.output_tokens
        result.raw_output = output
        
        # F1-4: 防御性拦截
        result.text, fact_stripped = _strip_retrieve_fact_calls(result.text)
        if fact_stripped > 0:
            logger.info(f"F1-4: Stripped {fact_stripped} residual retrieve_fact call(s) from stable output")
            self._stats.setdefault("fact_calls_stripped", 0)
            self._stats["fact_calls_stripped"] += fact_stripped
        
        self._stats.setdefault("stable_executions", 0)
        self._stats["stable_executions"] += 1
        
        logger.info("Stable fallback executed successfully")
        return result
    
    # ================================================================
    # 降级执行 (无注入)
    # ================================================================
    
    async def _execute_fallback(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        error_message: str,
        **kwargs,
    ) -> ExecutionResult:
        """降级执行 (无注入)"""
        self._stats["fallbacks"] += 1
        result = ExecutionResult()
        result.fallback_used = True
        result.error_message = error_message
        
        try:
            inference_start = time.time()
            output = self.model.generate(
                prompt=plan.original_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            result.inference_latency_ms = (time.time() - inference_start) * 1000
            result.text = output.text
            result.input_tokens = output.input_tokens
            result.output_tokens = output.output_tokens
            result.raw_output = output
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            raise
        
        return result
    
    # ================================================================
    # K/V 缓存 (用户级隔离)
    # ================================================================
    
    def _get_preference_kv(
        self,
        user_id: str,
        preference_text: str,
    ) -> Tuple[Optional[List[KVCacheEntry]], bool, str]:
        """
        获取偏好 K/V (含用户级隔离内存缓存)
        
        安全改进 (v3.1):
        - 缓存按 user_id 物理分区，不同用户的 K/V 存储在独立字典中
        - 访问时验证 user_id 归属，防止跨用户数据泄露
        - 即使 content_hash 相同，不同用户也不会共享缓存条目
        
        Returns:
            (kv_entries, cache_hit, cache_tier)
        """
        if not user_id or not user_id.strip():
            logger.warning("Empty user_id in _get_preference_kv, skipping cache")
            try:
                kv_entries, _ = self.model.compute_kv(preference_text)
                return kv_entries, False, "compute_no_cache"
            except Exception as e:
                logger.error(f"Failed to compute preference K/V: {e}")
                return None, False, "error"
        
        content_hash = hashlib.md5(preference_text.encode()).hexdigest()
        
        # 查找 BoundedUserKVCache (P0-2: 自带 LRU 淘汰)
        cached_entry = self._preference_kv_cache.get(user_id, content_hash)
        
        if cached_entry is not None:
            cached_kv, cached_hash = cached_entry
            if cached_hash == content_hash:
                self._stats["cache_hits"] += 1
                device = getattr(self.model, 'device', 'cpu')
                
                # P2-1: PackedKV 缓存 → 单次 to(device) → 解包
                if isinstance(cached_kv, PackedKV):
                    try:
                        if str(device) != "cpu":
                            gpu_packed = cached_kv.to(device)
                        else:
                            gpu_packed = cached_kv
                        return gpu_packed.to_entries(), True, "memory_packed"
                    except Exception:
                        # fallback: 直接返回解包
                        return cached_kv.to_entries(), True, "memory_packed"
                
                # 兼容旧格式: List[KVCacheEntry]
                kv_entries = cached_kv
                if str(device) != "cpu" and kv_entries and isinstance(kv_entries, list):
                    try:
                        kv_entries = [
                            KVCacheEntry(
                                key=e.key.to(device), value=e.value.to(device),
                                layer_idx=e.layer_idx,
                            )
                            for e in kv_entries
                        ]
                    except Exception:
                        pass
                return kv_entries, True, "memory"
        
        # 计算 K/V
        try:
            kv_entries, _ = self.model.compute_kv(preference_text)
            
            # P2-1: 打包为 PackedKV 后存入 CPU 缓存
            # 优势: CPU→GPU 传输从 64 次降为 2 次
            if kv_entries and isinstance(kv_entries, list) and len(kv_entries) > 0:
                try:
                    packed = PackedKV.from_entries(kv_entries).cpu()
                    self._preference_kv_cache.put(
                        user_id, content_hash, (packed, content_hash)
                    )
                except Exception:
                    # fallback: 逐层 CPU 缓存 (兼容非标准 tensor shape)
                    try:
                        cpu_entries = [
                            KVCacheEntry(
                                key=e.key.cpu(), value=e.value.cpu(),
                                layer_idx=e.layer_idx,
                            )
                            for e in kv_entries
                        ]
                        self._preference_kv_cache.put(
                            user_id, content_hash, (cpu_entries, content_hash)
                        )
                    except Exception:
                        pass
            
            # 释放 GPU 显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return kv_entries, False, "compute"
        except Exception as e:
            logger.error(f"Failed to compute preference K/V: {e}")
            return None, False, "error"
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """
        清除偏好缓存
        
        Args:
            user_id: 指定用户 ID 则只清除该用户的缓存，
                     None 则清除所有用户的缓存
        """
        self._preference_kv_cache.clear(user_id)
        if user_id:
            logger.debug(f"Cleared preference cache for user {user_id}")
        else:
            logger.debug("Cleared all preference caches")
    
    # ================================================================
    # 统计
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Executor 统计"""
        cache_stats = self._preference_kv_cache.get_stats()
        
        return {
            **self._stats,
            "stable_executions": self._stats.get("stable_executions", 0),
            "stable_fallbacks": self._stats.get("stable_fallbacks", 0),
            "kv_cache_user_partitions": cache_stats["current_users"],
            "kv_cache_total_entries": cache_stats["total_entries"],
            "kv_cache_isolation": "per_user_partition_lru",
            "kv_cache_bounded": cache_stats,
            "inference_guard_available": self._inference_guard is not None,
        }
