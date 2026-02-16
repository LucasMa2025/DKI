"""
DKI Injection Executor - 注入计划执行器

职责: 只做执行, 不做决策
- 根据 InjectionPlan 执行 recall_v4 策略
- 计算偏好 K/V (含缓存)
- 调用模型推理 (带 K/V 注入)
- 执行 Fact Call 循环 (事实补充)
- 处理降级 (fallback)

输入: InjectionPlan + ModelAdapter
输出: ExecutionResult

不变量:
- Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
- Preference alpha 永远不超过 AlphaProfile.override_cap
- 任何异常自动降级到无注入推理

安全不变量 (v3.1):
- 偏好 K/V 缓存按用户物理分区，不同用户的 K/V 不共享数据结构
- 每次缓存访问都验证 user_id 归属权
- 推理前后有上下文隔离检查，防止 K/V 残留泄露

注意 (v3.2):
- stable 和 full_attention 策略已移除
- 系统统一使用 recall_v4 (偏好 K/V + 后缀组装 + Fact Call)

Author: AGI Demo Project
Version: 3.2.0
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
from dki.core.plugin.injection_plan import (
    InjectionPlan,
    ExecutionResult,
)

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

# Recall v4 组件 (可选)
try:
    from dki.core.recall import FactRetriever, PromptFormatter as RecallPromptFormatter
    from dki.core.recall.recall_config import RecallConfig
    RECALL_V4_AVAILABLE = True
except ImportError:
    RECALL_V4_AVAILABLE = False


class InjectionExecutor:
    """
    注入计划执行器 (v3.2)
    
    纯执行层，根据 InjectionPlan 调用模型。
    统一使用 recall_v4 策略: 偏好 K/V 注入 + 后缀组装 + Fact Call 循环。
    
    不变量:
    - Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
    - Preference alpha 永远不超过 AlphaProfile.override_cap
    - 任何异常自动降级到无注入推理
    
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
        # Recall v4 组件
        fact_retriever: Optional[Any] = None,
        prompt_formatter: Optional[Any] = None,
        recall_config: Optional[Any] = None,
    ):
        """
        初始化执行器
        
        Args:
            model_adapter: LLM 模型适配器
            fact_retriever: 事实检索器 (Recall v4)
            prompt_formatter: 提示格式化器 (Recall v4)
            recall_config: 召回配置 (Recall v4)
        """
        self.model = model_adapter
        
        # Recall v4 组件
        self._fact_retriever = fact_retriever
        self._prompt_formatter = prompt_formatter
        self._recall_config = recall_config
        
        # ============ 用户级隔离的偏好 K/V 缓存 ============
        # 改进: 从全局 Dict 改为用户级分区 Dict
        # 结构: {user_id: {content_hash: (kv_entries, content_hash)}}
        self._preference_kv_cache: Dict[str, Dict[str, Tuple[Any, str]]] = {}
        
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
            "fact_call_rounds": 0,
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
        执行注入计划 (v3.2: 统一 recall_v4 策略)
        
        执行路径:
        - 有偏好 + 注入启用: K/V 注入推理 + Fact Call 循环
        - 无偏好 / 无注入: 直接生成
        
        Args:
            plan: 注入计划 (来自 Planner)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            ExecutionResult 包含生成结果和执行时间
        """
        self._stats["executions"] += 1
        
        try:
            if plan.injection_enabled and plan.preference_text:
                # recall_v4: 偏好 K/V 注入 + 后缀组装
                result = await self._execute_with_kv_injection(
                    plan, max_new_tokens, temperature, **kwargs
                )
                # Fact Call 循环 (事实补充)
                if (plan.has_fact_call_instruction
                        and self._fact_retriever
                        and self._prompt_formatter
                        and self._recall_config
                        and self._recall_config.fact_call.enabled):
                    result = await self._execute_fact_call_loop(
                        plan, result, max_new_tokens, temperature, **kwargs
                    )
                self._stats["recall_v4_executions"] += 1
            else:
                result = await self._execute_plain(
                    plan, max_new_tokens, temperature, **kwargs
                )
                self._stats["plain_executions"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed, falling back: {e}")
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
        
        return result
    
    # ================================================================
    # 降级执行
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
    # Recall v4: Fact Call 循环
    # ================================================================
    
    async def _execute_fact_call_loop(
        self,
        plan: InjectionPlan,
        initial_result: ExecutionResult,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """
        Fact Call 循环: 检测模型输出中的 retrieve_fact 调用, 补充事实后重新推理
        
        最多执行 max_rounds 轮, 总事实 token 不超过 max_fact_tokens
        """
        if not self._prompt_formatter or not self._fact_retriever:
            return initial_result
        
        max_rounds = self._recall_config.fact_call.max_rounds
        max_fact_tokens = self._recall_config.fact_call.max_fact_tokens
        
        result = initial_result
        prompt = plan.final_input
        total_fact_tokens = 0
        
        for round_idx in range(max_rounds):
            # 检查输出是否包含 fact_call
            fact_request = self._prompt_formatter.detect_fact_request(result.text)
            
            if fact_request is None:
                break  # 不需要补充事实, 结束循环 ✅
            
            logger.info(
                f"Fact call round {round_idx + 1}: "
                f"trace_id={fact_request.trace_id}, "
                f"offset={fact_request.offset}"
            )
            
            # 检索事实
            fact_response = self._fact_retriever.retrieve(
                trace_id=fact_request.trace_id,
                session_id=plan.session_id,
                offset=fact_request.offset,
                limit=fact_request.limit,
            )
            
            if not fact_response.messages:
                logger.warning(f"No facts found for trace_id={fact_request.trace_id}")
                break
            
            # 格式化事实段落
            fact_text = self._prompt_formatter.format_fact_segment(fact_response)
            
            # 粗估事实 token
            fact_tokens = len(fact_text.split()) * 2  # 粗估
            total_fact_tokens += fact_tokens
            
            if total_fact_tokens > max_fact_tokens:
                logger.info(f"Fact token budget exhausted: {total_fact_tokens} > {max_fact_tokens}")
                break
            
            # 追加事实段落到 prompt, 继续推理
            if self._prompt_formatter.language == "cn":
                continuation = "请基于以上补充事实回答用户问题。"
            else:
                continuation = "Please answer based on the supplementary facts above."
            
            prompt = prompt + "\n\n" + fact_text + "\n\n" + continuation
            
            # 重新推理
            inference_start = time.time()
            
            # 获取偏好 K/V
            preference_kv, cache_hit, cache_tier = self._get_preference_kv(
                user_id=plan.user_id,
                preference_text=plan.preference_text,
            )
            alpha = plan.alpha_profile.effective_preference_alpha
            
            if preference_kv and alpha > 0.1:
                # Fact call 循环中的推理也需要隔离守卫
                if self._inference_guard:
                    with self._inference_guard.scoped_inference(
                        user_id=plan.user_id,
                        kv_entries=preference_kv,
                    ):
                        output = self.model.forward_with_kv_injection(
                            prompt=prompt,
                            injected_kv=preference_kv,
                            alpha=alpha,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                else:
                    output = self.model.forward_with_kv_injection(
                        prompt=prompt,
                        injected_kv=preference_kv,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
            else:
                output = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            
            result.inference_latency_ms += (time.time() - inference_start) * 1000
            result.text = output.text
            result.input_tokens = output.input_tokens
            result.output_tokens = output.output_tokens
            result.raw_output = output
            result.fact_rounds_used = round_idx + 1
            result.fact_tokens_total = total_fact_tokens
            
            self._stats["fact_call_rounds"] += 1
        
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
        
        # 获取用户专属分区
        user_partition = self._preference_kv_cache.get(user_id)
        
        # 检查用户分区中的缓存
        if user_partition is not None and content_hash in user_partition:
            kv_entries, cached_hash = user_partition[content_hash]
            if cached_hash == content_hash:
                self._stats["cache_hits"] += 1
                return kv_entries, True, "memory"
        
        # 计算 K/V
        try:
            kv_entries, _ = self.model.compute_kv(preference_text)
            
            # 存入用户专属分区
            if user_id not in self._preference_kv_cache:
                self._preference_kv_cache[user_id] = {}
            self._preference_kv_cache[user_id][content_hash] = (kv_entries, content_hash)
            
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
        if user_id:
            # 只清除指定用户的分区
            if user_id in self._preference_kv_cache:
                del self._preference_kv_cache[user_id]
                logger.debug(f"Cleared preference cache for user {user_id}")
        else:
            self._preference_kv_cache.clear()
            logger.debug("Cleared all preference caches")
    
    # ================================================================
    # 统计
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Executor 统计"""
        # 统计用户分区信息
        total_entries = sum(
            len(partition) for partition in self._preference_kv_cache.values()
        )
        
        return {
            **self._stats,
            "kv_cache_user_partitions": len(self._preference_kv_cache),
            "kv_cache_total_entries": total_entries,
            "kv_cache_isolation": "per_user_partition",
            "inference_guard_available": self._inference_guard is not None,
        }
