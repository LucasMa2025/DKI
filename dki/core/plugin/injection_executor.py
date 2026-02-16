"""
DKI Injection Executor - 注入计划执行器

职责: 只做执行, 不做决策
- 根据 InjectionPlan 选择执行路径
- 计算偏好 K/V (含缓存)
- 调用 Full Attention 注入器
- 调用模型推理
- 处理降级 (fallback)

输入: InjectionPlan + ModelAdapter
输出: ExecutionResult

不变量:
- Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
- Preference alpha 永远不超过 AlphaProfile.override_cap
- 任何异常自动降级到无注入推理

Author: AGI Demo Project
Version: 3.0.0
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
from dki.core.injection import (
    FullAttentionInjector,
    FullAttentionConfig,
)

# Recall v4 组件 (可选)
try:
    from dki.core.recall import FactRetriever, PromptFormatter as RecallPromptFormatter
    from dki.core.recall.recall_config import RecallConfig
    RECALL_V4_AVAILABLE = True
except ImportError:
    RECALL_V4_AVAILABLE = False


class InjectionExecutor:
    """
    注入计划执行器
    
    纯执行层，根据 InjectionPlan 调用模型。
    
    不变量:
    - Key tensor 永远不被 alpha 缩放 (保护 attention addressing)
    - Preference alpha 永远不超过 AlphaProfile.override_cap
    - 任何异常自动降级到无注入推理
    
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
        full_attention_injector: Optional[FullAttentionInjector] = None,
        # Recall v4 组件 (可选)
        fact_retriever: Optional[Any] = None,
        prompt_formatter: Optional[Any] = None,
        recall_config: Optional[Any] = None,
    ):
        """
        初始化执行器
        
        Args:
            model_adapter: LLM 模型适配器
            full_attention_injector: Full Attention 注入器 (可选)
            fact_retriever: 事实检索器 (Recall v4, 可选)
            prompt_formatter: 提示格式化器 (Recall v4, 可选)
            recall_config: 召回配置 (Recall v4, 可选)
        """
        self.model = model_adapter
        self._full_attention_injector = full_attention_injector
        
        # Recall v4 组件
        self._fact_retriever = fact_retriever
        self._prompt_formatter = prompt_formatter
        self._recall_config = recall_config
        
        # 偏好 K/V 缓存 (内存级)
        self._preference_kv_cache: Dict[str, Tuple[Any, str]] = {}
        
        # 统计
        self._stats = {
            "executions": 0,
            "stable_executions": 0,
            "full_attention_executions": 0,
            "plain_executions": 0,
            "recall_v4_executions": 0,
            "fallbacks": 0,
            "cache_hits": 0,
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
        执行注入计划
        
        根据 plan.strategy 选择执行路径:
        - stable: 偏好 K/V 注入 + suffix prompt
        - full_attention: 偏好 K/V + 历史 K/V
        - none / 无注入: 直接生成
        
        Args:
            plan: 注入计划 (来自 Planner)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            ExecutionResult 包含生成结果和执行时间
        """
        self._stats["executions"] += 1
        
        try:
            if (plan.strategy == "full_attention"
                    and self._full_attention_injector
                    and plan.injection_enabled):
                result = await self._execute_full_attention(
                    plan, max_new_tokens, temperature, **kwargs
                )
                self._stats["full_attention_executions"] += 1
            elif plan.injection_enabled and plan.preference_text:
                # Stable 策略 (含 recall_v4 的 fact call 循环)
                result = await self._execute_stable(
                    plan, max_new_tokens, temperature, **kwargs
                )
                # 如果启用了 recall_v4 fact call, 执行循环
                if (plan.has_fact_call_instruction
                        and self._fact_retriever
                        and self._prompt_formatter
                        and self._recall_config
                        and self._recall_config.fact_call.enabled):
                    result = await self._execute_fact_call_loop(
                        plan, result, max_new_tokens, temperature, **kwargs
                    )
                self._stats["stable_executions"] += 1
                if plan.recall_strategy:
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
    # Stable 策略执行
    # ================================================================
    
    async def _execute_stable(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """
        执行 Stable 策略
        
        偏好: K/V 注入 (负位置)
        历史: Suffix Prompt (正位置，已包含在 plan.final_input 中)
        """
        result = ExecutionResult()
        
        # 获取偏好 K/V (含缓存)
        preference_kv, cache_hit, cache_tier = self._get_preference_kv(
            user_id=plan.user_id,
            preference_text=plan.preference_text,
        )
        result.preference_cache_hit = cache_hit
        result.preference_cache_tier = cache_tier
        
        # 使用 effective alpha (受 override_cap 约束)
        alpha = plan.alpha_profile.effective_preference_alpha
        
        # 推理
        inference_start = time.time()
        
        if preference_kv and alpha > 0.1:
            # 带 K/V 注入的推理
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
    # Full Attention 策略执行
    # ================================================================
    
    async def _execute_full_attention(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """
        执行 Full Attention 策略 (研究 - 方案 C)
        
        偏好: K/V 注入 (负位置)
        历史: K/V 注入 (负位置)
        全局指示: 极简提示 (约 3-5 tokens)
        
        如果 Full Attention 失败或触发 fallback，自动回退到 Stable。
        """
        result = ExecutionResult()
        
        # 调用 Full Attention 注入器
        fa_result = self._full_attention_injector.inject(
            model_adapter=self.model,
            preference_text=plan.preference_text,
            history_messages=plan.history_messages,
            query=plan.original_query,
        )
        
        # 检查是否需要 fallback 到 Stable
        if not fa_result.success or fa_result.fallback_triggered:
            result.full_attention_fallback = True
            logger.info(
                f"Full attention fallback to stable: "
                f"{fa_result.error_message}"
            )
            # 使用已准备好的 history_suffix 回退到 Stable
            return await self._execute_stable_fallback(
                plan, max_new_tokens, temperature, **kwargs
            )
        
        # 记录 Full Attention 信息
        result.full_attention_position_mode = fa_result.position_mode
        result.full_attention_preference_tokens = fa_result.preference_tokens
        result.full_attention_history_tokens = fa_result.history_tokens
        
        # 构造最终输入 (仅包含全局指示 + 查询)
        final_input = plan.original_query
        if fa_result.global_indication:
            final_input = fa_result.global_indication + "\n" + plan.original_query
        
        # 使用 effective alpha
        alpha = plan.alpha_profile.effective_preference_alpha
        
        # 推理
        inference_start = time.time()
        
        if fa_result.merged_kv:
            output = self.model.forward_with_kv_injection(
                prompt=final_input,
                injected_kv=fa_result.merged_kv,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            output = self.model.generate(
                prompt=final_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        
        result.inference_latency_ms = (time.time() - inference_start) * 1000
        result.text = output.text
        result.input_tokens = output.input_tokens
        result.output_tokens = output.output_tokens
        result.raw_output = output
        
        logger.debug(
            f"Full attention executed: "
            f"pref_kv={fa_result.preference_tokens}, "
            f"hist_kv={fa_result.history_tokens}"
        )
        
        return result
    
    async def _execute_stable_fallback(
        self,
        plan: InjectionPlan,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ExecutionResult:
        """
        Full Attention 回退到 Stable 策略
        
        使用 plan 中已准备好的 history_suffix 构造 stable 输入。
        """
        # 重新构造 stable 的 final_input
        if plan.history_suffix:
            stable_input = plan.history_suffix + "\n\n" + plan.original_query
        else:
            stable_input = plan.original_query
        
        # 创建一个临时 stable plan
        stable_plan = InjectionPlan(
            strategy="stable",
            preference_text=plan.preference_text,
            user_id=plan.user_id,
            original_query=plan.original_query,
            final_input=stable_input,
            injection_enabled=plan.injection_enabled,
            alpha_profile=plan.alpha_profile,
        )
        
        result = await self._execute_stable(
            stable_plan, max_new_tokens, temperature, **kwargs
        )
        result.full_attention_fallback = True
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
    # K/V 缓存
    # ================================================================
    
    def _get_preference_kv(
        self,
        user_id: str,
        preference_text: str,
    ) -> Tuple[Optional[List[KVCacheEntry]], bool, str]:
        """
        获取偏好 K/V (含内存缓存)
        
        Returns:
            (kv_entries, cache_hit, cache_tier)
        """
        content_hash = hashlib.md5(preference_text.encode()).hexdigest()
        cache_key = f"{user_id}:{content_hash}"
        
        # 检查缓存
        if cache_key in self._preference_kv_cache:
            kv_entries, cached_hash = self._preference_kv_cache[cache_key]
            if cached_hash == content_hash:
                self._stats["cache_hits"] += 1
                return kv_entries, True, "memory"
        
        # 计算 K/V
        try:
            kv_entries, _ = self.model.compute_kv(preference_text)
            self._preference_kv_cache[cache_key] = (kv_entries, content_hash)
            return kv_entries, False, "compute"
        except Exception as e:
            logger.error(f"Failed to compute preference K/V: {e}")
            return None, False, "error"
    
    def clear_preference_cache(self, user_id: Optional[str] = None):
        """清除偏好缓存"""
        if user_id:
            keys_to_remove = [
                k for k in self._preference_kv_cache
                if k.startswith(f"{user_id}:")
            ]
            for key in keys_to_remove:
                del self._preference_kv_cache[key]
        else:
            self._preference_kv_cache.clear()
    
    # ================================================================
    # Full Attention 管理
    # ================================================================
    
    def set_full_attention_injector(
        self,
        injector: FullAttentionInjector,
    ):
        """设置 Full Attention 注入器"""
        self._full_attention_injector = injector
    
    @property
    def full_attention_injector(self) -> Optional[FullAttentionInjector]:
        """获取 Full Attention 注入器"""
        return self._full_attention_injector
    
    # ================================================================
    # 统计
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Executor 统计"""
        return {
            **self._stats,
            "kv_cache_size": len(self._preference_kv_cache),
            "full_attention_available": (
                self._full_attention_injector is not None
            ),
        }
