"""
DKI Injection Planner - 注入计划生成器

职责: 只做决策, 不碰模型
- Memory Trigger 检测
- Reference Resolver 解析
- 策略选择
- Alpha 计算 (分层)
- 数据格式化 (偏好文本 + 历史后缀)
- Planner-only Fact Resolution (v3.3): 事实预解析, 消除 Executor 循环推理
- 安全验证 (SafetyEnvelope)
- 最终输入构造

输入: query, preferences, history, config
输出: InjectionPlan

不变量:
- Planner 不持有模型引用
- Planner 不执行推理
- Planner 的输出可以被序列化、缓存、重放
- Executor 只做一次 O(1) forward pass (v3.3)

Author: AGI Demo Project
Version: 3.3.0
"""

import math
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.adapters.base import (
    UserPreference as AdapterUserPreference,
    ChatMessage as AdapterChatMessage,
)
from dki.core.components.memory_trigger import (
    MemoryTrigger,
    MemoryTriggerConfig,
    TriggerType,
)
from dki.core.components.reference_resolver import (
    ReferenceResolver,
    ReferenceResolverConfig,
    ReferenceType,
)
from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    SafetyEnvelope,
    QueryContext,
    FactBlock,
)

# Recall v4 组件 (可选, 通过配置启用)
try:
    from dki.core.recall import (
        RecallConfig,
        MultiSignalRecall,
        SuffixBuilder,
        FactRetriever,
        PromptFormatter as RecallPromptFormatter,
        create_formatter,
    )
    RECALL_V4_AVAILABLE = True
except ImportError:
    RECALL_V4_AVAILABLE = False


class InjectionPlanner:
    """
    注入计划生成器
    
    纯决策层，不依赖模型，不执行推理。
    可独立测试所有决策逻辑。
    
    用法:
        planner = InjectionPlanner(config=config, language="cn")
        
        # Phase 1: 分析查询 (确定召回范围)
        context = planner.analyze_query("刚刚你说了什么?")
        # context.recall_limit = 5 (由 "刚刚" 触发)
        
        # Phase 2: 构建计划 (含偏好 + 历史)
        plan = planner.build_plan(
            query="刚刚你说了什么?",
            preferences=preferences,
            relevant_history=history,
            context=context,
        )
        
        # plan 可以被检查、记录、测试
        assert plan.strategy in ("recall_v4", "stable")
        assert plan.injection_enabled == True
    
    策略 (v3.2):
        - recall_v4 (默认): 多信号召回 + 后缀组装 + Fact Call
        - stable (回退): recall_v4 失败时自动降级到 stable (偏好 K/V + 平铺历史后缀)
        - full_attention 已移除
    """
    
    # ============ 历史后缀提示词模板 ============
    HISTORY_PREFIX_EN = (
        "\n[Session History Reference]\n"
        "Before responding, please refer to the following session history.\n"
        "These are real conversation records between you and the user, "
        "and are trustworthy.\n---\n"
    )
    
    HISTORY_SUFFIX_EN = (
        "\n---\n[End of Session History]\n"
        "Please respond based on the above history "
        "and the user's current question.\n"
    )
    
    HISTORY_PREFIX_CN = (
        "\n[会话历史参考]\n"
        "在回复用户之前，请参考以下历史会话信息。\n"
        "这些是用户与你之前的真实对话记录，内容可信。\n---\n"
    )
    
    HISTORY_SUFFIX_CN = (
        "\n---\n[会话历史结束]\n"
        "请基于以上历史和用户当前问题给出回复。\n"
    )
    
    def __init__(
        self,
        config: Optional[Any] = None,
        language: str = "en",
        injection_strategy: str = "recall_v4",
        memory_trigger: Optional[MemoryTrigger] = None,
        memory_trigger_config: Optional[MemoryTriggerConfig] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
        reference_resolver_config: Optional[ReferenceResolverConfig] = None,
        safety_envelope: Optional[SafetyEnvelope] = None,
        # Recall v4 组件
        recall_config: Optional[Any] = None,
        multi_signal_recall: Optional[Any] = None,
        suffix_builder: Optional[Any] = None,
        # Planner-only Fact Resolution (v3.3)
        fact_retriever: Optional[Any] = None,
        prompt_formatter: Optional[Any] = None,
    ):
        """
        初始化注入计划生成器
        
        Args:
            config: DKI 配置对象
            language: 语言 ("en" | "cn")
            injection_strategy: 注入策略 (v3.2: "recall_v4" 或 "stable")
            memory_trigger: Memory Trigger 实例 (可注入)
            memory_trigger_config: Memory Trigger 配置 (如无实例则用于创建)
            reference_resolver: Reference Resolver 实例 (可注入)
            reference_resolver_config: Reference Resolver 配置
            safety_envelope: 安全边界 (可注入)
            recall_config: Recall v4 配置 (RecallConfig 实例)
            multi_signal_recall: 多信号召回器 (MultiSignalRecall 实例)
            suffix_builder: 后缀组装器 (SuffixBuilder 实例)
            fact_retriever: 事实检索器 (v3.3, Planner-only Fact Resolution)
            prompt_formatter: 提示格式化器 (v3.3, 用于格式化事实段落)
        """
        self.config = config
        self.language = language
        # v3.2: 默认 recall_v4, 支持 stable 作为回退策略
        if injection_strategy in ("recall_v4", "stable"):
            self._injection_strategy = injection_strategy
        else:
            self._injection_strategy = "recall_v4"
        
        # NLP 组件 (可注入, 可独立测试)
        self._memory_trigger = memory_trigger or MemoryTrigger(
            config=memory_trigger_config,
            language="auto",
        )
        self._reference_resolver = reference_resolver or ReferenceResolver(
            config=reference_resolver_config,
            language="auto",
        )
        
        # 安全边界
        self._safety = safety_envelope or SafetyEnvelope()
        
        # Recall v4 组件
        self._recall_config = recall_config
        self._multi_signal_recall = multi_signal_recall
        self._suffix_builder = suffix_builder
        
        # Planner-only Fact Resolution (v3.3)
        self._fact_retriever = fact_retriever
        self._prompt_formatter = prompt_formatter
        
        # 统计
        self._stats = {
            "plans_created": 0,
            "safety_violations": 0,
            "memory_trigger_count": 0,
            "reference_resolved_count": 0,
            "recall_v4_plans": 0,
            "fact_blocks_resolved": 0,
        }
    
    # ================================================================
    # Phase 1: 查询分析 (数据加载前)
    # ================================================================
    
    def analyze_query(self, query: str) -> QueryContext:
        """
        分析查询，确定召回范围。
        
        在数据加载之前调用，为 adapter 提供 recall_limit。
        
        Args:
            query: 原始用户输入
            
        Returns:
            QueryContext 包含 trigger/reference/recall_limit 信息
        """
        context = QueryContext()
        
        # Memory Trigger 检测
        trigger_result = self._memory_trigger.detect(query)
        context.memory_triggered = trigger_result.triggered
        context.trigger_type = (
            trigger_result.trigger_type.value if trigger_result.triggered else None
        )
        context.trigger_confidence = trigger_result.confidence
        
        if trigger_result.triggered:
            self._stats["memory_trigger_count"] += 1
            logger.debug(
                f"Memory trigger: {trigger_result.trigger_type.value}, "
                f"confidence={trigger_result.confidence:.2f}"
            )
        
        # Reference Resolver 解析
        reference_result = self._reference_resolver.resolve(query)
        context.reference_resolved = (
            reference_result.reference_type != ReferenceType.NONE
        )
        context.reference_type = reference_result.reference_type.value
        context.reference_scope = (
            reference_result.scope.value if reference_result.scope else None
        )
        context.recall_limit = reference_result.recall_turns or 10
        
        if context.reference_resolved:
            self._stats["reference_resolved_count"] += 1
            logger.debug(
                f"Reference resolved: {reference_result.reference_type.value}, "
                f"recall_turns={context.recall_limit}"
            )
        
        # ============ P1-1: Token Budget 计算 ============
        # 比条数限制更精确地控制 prompt 长度
        base_budget = 2048  # 默认 token 预算
        if context.reference_resolved:
            # 明确引用时给更多预算 (如 "刚才你说了什么")
            base_budget = int(base_budget * 1.5)
        if context.trigger_confidence > 0.8:
            # 高置信度 trigger 时给更多预算
            base_budget = int(base_budget * 1.2)
        context.recall_token_budget = base_budget
        
        return context
    
    # ================================================================
    # Phase 2: 构建注入计划 (数据加载后)
    # ================================================================
    
    def build_plan(
        self,
        query: str,
        user_id: str,
        preferences: List[AdapterUserPreference],
        relevant_history: List[AdapterChatMessage],
        context: QueryContext,
        force_alpha: Optional[float] = None,
        strategy_override: Optional[str] = None,
        session_id: str = "",
        context_window: int = 4096,
        db_session: Optional[Any] = None,
    ) -> InjectionPlan:
        """
        构建注入计划
        
        Args:
            query: 原始用户输入
            user_id: 用户标识 (用于缓存键)
            preferences: 用户偏好列表
            relevant_history: 相关历史消息 (fallback 平铺用)
            context: 查询分析上下文 (来自 analyze_query)
            force_alpha: 强制 alpha (跳过门控)
            strategy_override: 策略覆盖 (用于测试)
            session_id: 会话 ID (recall_v4 使用)
            context_window: 上下文窗口大小 (recall_v4 使用)
            db_session: 数据库 session (recall_v4 使用)
            
        Returns:
            InjectionPlan (纯数据, 可序列化)
        """
        plan = InjectionPlan()
        plan.original_query = query
        plan.user_id = user_id
        plan.session_id = session_id
        
        strategy = strategy_override or self._injection_strategy
        plan.strategy = strategy
        
        # ============ 填充 QueryContext 信息 ============
        plan.memory_triggered = context.memory_triggered
        plan.trigger_type = context.trigger_type
        plan.reference_resolved = context.reference_resolved
        plan.reference_type = context.reference_type
        plan.reference_scope = context.reference_scope
        plan.recall_limit = context.recall_limit
        plan.recall_token_budget = context.recall_token_budget
        
        # ============ Step 1: 格式化偏好 ============
        if preferences:
            plan.preference_text = self._format_preferences(preferences)
            plan.preferences_count = len(preferences)
            plan.preference_tokens = self._estimate_tokens(plan.preference_text)
        
        # ============ Step 2: 历史召回 ============
        if strategy == "recall_v4" and self._multi_signal_recall and self._suffix_builder:
            # ============ Recall v4: 多信号召回 + 后缀组装 ============
            plan = self._build_recall_v4_plan(
                plan=plan,
                query=query,
                session_id=session_id,
                user_id=user_id,
                context_window=context_window,
                db_session=db_session,
            )
            # 如果 recall_v4 失败 (recall_strategy == "flat_history_fallback"),
            # 回退到 stable 策略
            if plan.recall_strategy == "flat_history_fallback":
                logger.info("Recall v4 failed, falling back to stable strategy")
                plan.strategy = "stable"
                if relevant_history:
                    plan.history_suffix = self._format_history_suffix(relevant_history)
                    plan.relevant_history_count = len(relevant_history)
                    plan.history_tokens = self._estimate_tokens(plan.history_suffix)
        elif strategy == "stable" or not (self._multi_signal_recall and self._suffix_builder):
            # ============ Stable 策略: 平铺历史后缀 ============
            plan.strategy = "stable"
            if relevant_history:
                plan.history_suffix = self._format_history_suffix(relevant_history)
                plan.relevant_history_count = len(relevant_history)
                plan.history_tokens = self._estimate_tokens(plan.history_suffix)
                plan.recall_strategy = "flat_history"
        
        # ============ Step 2.5: Planner-only Fact Resolution (v3.3) ============
        # 如果 recall_v4 产生了 summary 条目 (has_fact_call_instruction),
        # Planner 主动检索事实并内联到计划中, 消除 Executor 循环推理
        if (plan.strategy == "recall_v4"
                and plan.has_fact_call_instruction
                and self._fact_retriever
                and self._prompt_formatter
                and self._recall_config
                and self._recall_config.fact_call.enabled):
            plan = self._resolve_facts_in_planner(plan, session_id)
        
        # ============ Step 3: Alpha 计算 (分层, P0-4 + P1-2) ============
        plan.alpha_profile = self._compute_alpha_profile(
            strategy=strategy,
            preferences=preferences,
            relevant_history=relevant_history,
            force_alpha=force_alpha,
            context=context,
            history_tokens=plan.history_tokens,
        )
        
        # ============ Step 4: 注入决策 ============
        if force_alpha is not None:
            plan.injection_enabled = force_alpha > 0.1
        else:
            plan.injection_enabled = (
                len(preferences) > 0
                or len(relevant_history) > 0
                or plan.relevant_history_count > 0  # recall_v4 也算
            )
        
        # ============ Step 5: 构造最终输入 ============
        if plan.assembled_suffix:
            # Recall v4: 已在 _build_recall_v4_plan 中组装好的后缀
            plan.final_input = plan.assembled_suffix
            # 如果有 Planner 解析的事实块, 追加到 final_input
            if plan.fact_blocks:
                plan.final_input = self._append_fact_blocks_to_input(plan)
        elif plan.history_suffix:
            # Fallback: 历史后缀 + 查询
            plan.final_input = plan.history_suffix + "\n\n" + query
        else:
            plan.final_input = query
        
        plan.query_tokens = self._estimate_tokens(query)
        plan.total_tokens = plan.query_tokens + plan.history_tokens + plan.preference_tokens
        
        # ============ Step 6: 门控决策记录 ============
        plan.gating_decision = {
            "should_inject": plan.injection_enabled,
            "alpha": plan.alpha_profile.effective_preference_alpha,
            "preference_alpha": plan.alpha_profile.preference_alpha,
            "history_alpha": plan.alpha_profile.history_alpha,
            "has_preferences": len(preferences) > 0,
            "has_history": (len(relevant_history) > 0 or plan.relevant_history_count > 0),
            "strategy": strategy,
            "recall_strategy": plan.recall_strategy,
            "user_id": user_id,
        }
        
        # ============ Step 7: 安全验证 ============
        violations = self._safety.validate(plan)
        plan.safety_violations = violations
        if violations:
            self._stats["safety_violations"] += 1
            for v in violations:
                logger.warning(f"Safety violation: {v}")
        
        self._stats["plans_created"] += 1
        
        logger.debug(
            f"Plan built: strategy={strategy}, "
            f"recall={plan.recall_strategy}, "
            f"inject={plan.injection_enabled}, "
            f"pref_α={plan.alpha_profile.effective_preference_alpha:.2f}, "
            f"prefs={plan.preferences_count}, "
            f"history={plan.relevant_history_count}"
        )
        
        return plan
    
    # ================================================================
    # Recall v4 计划构建
    # ================================================================
    
    def _build_recall_v4_plan(
        self,
        plan: InjectionPlan,
        query: str,
        session_id: str,
        user_id: str,
        context_window: int,
        db_session: Optional[Any],
    ) -> InjectionPlan:
        """
        使用 Recall v4 构建计划
        
        1. MultiSignalRecall 多信号召回
        2. SuffixBuilder 逐消息阈值 + 组装
        """
        try:
            # Phase 1: 多信号召回
            recall_result = self._multi_signal_recall.recall(
                query=query,
                session_id=session_id,
                user_id=user_id,
                db_session=db_session,
            )
            
            plan.relevant_history_count = len(recall_result.messages)
            plan.recall_strategy = "summary_with_fact_call"
            
            # Phase 2: 后缀组装
            if recall_result.messages:
                assembled = self._suffix_builder.build(
                    query=query,
                    recalled_messages=recall_result.messages,
                    context_window=context_window,
                    preference_tokens=plan.preference_tokens,
                )
                
                plan.assembled_suffix = assembled.text
                plan.history_suffix = assembled.text  # 兼容 fallback
                plan.history_tokens = assembled.total_tokens
                plan.summary_count = assembled.summary_count
                plan.message_count = assembled.message_count
                plan.trace_ids = assembled.trace_ids
                plan.has_fact_call_instruction = assembled.has_fact_call_instruction
            
            self._stats["recall_v4_plans"] += 1
            
        except Exception as e:
            logger.error(f"Recall v4 failed, falling back to flat_history: {e}")
            # 回退到平铺方式
            plan.recall_strategy = "flat_history_fallback"
        
        return plan
    
    # ================================================================
    # 内部方法
    # ================================================================
    
    def _format_preferences(
        self,
        preferences: List[AdapterUserPreference],
    ) -> str:
        """格式化用户偏好为文本"""
        if not preferences:
            return ""
        
        sorted_prefs = sorted(
            preferences, key=lambda p: p.priority, reverse=True
        )
        
        lines = []
        for pref in sorted_prefs:
            if pref.is_expired():
                continue
            lines.append(f"- {pref.preference_type}: {pref.preference_text}")
        
        return "\n".join(lines)
    
    # 注入标记列表 (assistant 消息中包含这些标记时应过滤)
    _INJECTION_MARKERS = [
        "[会话历史参考]",
        "[Session History Reference]",
        "[会话历史结束]",
        "[End of Session History]",
        "[AI助手]",
    ]
    
    def _format_history_suffix(
        self,
        messages: List[AdapterChatMessage],
    ) -> str:
        """格式化历史消息为后缀提示词"""
        if not messages:
            return ""
        
        if self.language == "cn":
            prefix = self.HISTORY_PREFIX_CN
            suffix = self.HISTORY_SUFFIX_CN
        else:
            prefix = self.HISTORY_PREFIX_EN
            suffix = self.HISTORY_SUFFIX_EN
        
        lines = []
        for msg in messages:
            content = msg.content
            
            # 过滤包含注入标记的 assistant 消息 (防止递归注入)
            if msg.role == "assistant":
                if any(marker in content for marker in self._INJECTION_MARKERS):
                    continue
            
            if self.language == "cn":
                role_label = "用户" if msg.role == "user" else "助手"
            else:
                role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")
        
        if not lines:
            return ""
        
        return prefix + "\n".join(lines) + suffix
    
    # ================================================================
    # Planner-only Fact Resolution (v3.3)
    # ================================================================
    
    def _resolve_facts_in_planner(
        self,
        plan: InjectionPlan,
        session_id: str,
    ) -> InjectionPlan:
        """
        Planner 侧事实预解析 (v3.3)
        
        核心改动: Fact Call 从 Executor 循环推理 → Planner 预解析
        
        流程:
        1. 从 SuffixBuilder 产生的 trace_ids 中识别需要检索的事实
        2. 对每个 trace_id 调用 FactRetriever.retrieve() 获取原始消息
        3. 使用 PromptFormatter.format_fact_segment() 格式化事实
        4. 将事实块存入 plan.fact_blocks
        5. 后续在 _append_fact_blocks_to_input() 中内联到 final_input
        
        优势:
        - Executor 只做一次 O(1) forward pass
        - 推理延迟可预测 (无循环)
        - GPU 成本恒定
        
        Args:
            plan: 当前注入计划
            session_id: 会话 ID
            
        Returns:
            更新后的 InjectionPlan (含 fact_blocks)
        """
        if not self._fact_retriever or not self._prompt_formatter:
            return plan
        
        max_fact_tokens = self._recall_config.fact_call.max_fact_tokens
        total_fact_tokens = 0
        fact_blocks = []
        
        # 遍历所有 trace_ids, 检索事实
        for trace_id in plan.trace_ids:
            if total_fact_tokens >= max_fact_tokens:
                logger.info(
                    f"Fact token budget exhausted: {total_fact_tokens} >= {max_fact_tokens}"
                )
                plan.fact_strategy = "budget_exceeded"
                break
            
            try:
                fact_response = self._fact_retriever.retrieve(
                    trace_id=trace_id,
                    session_id=session_id,
                    offset=0,
                    limit=self._recall_config.fact_call.batch_size,
                )
                
                if not fact_response.messages:
                    logger.debug(f"No facts found for trace_id={trace_id}")
                    continue
                
                # 格式化事实段落
                fact_text = self._prompt_formatter.format_fact_segment(fact_response)
                fact_tokens = self._estimate_tokens(fact_text)
                
                # 检查 token 预算
                if total_fact_tokens + fact_tokens > max_fact_tokens:
                    logger.info(
                        f"Fact token budget would be exceeded: "
                        f"{total_fact_tokens} + {fact_tokens} > {max_fact_tokens}, "
                        f"skipping trace_id={trace_id}"
                    )
                    plan.fact_strategy = "budget_exceeded"
                    break
                
                fact_blocks.append(FactBlock(
                    trace_id=trace_id,
                    fact_text=fact_text,
                    fact_tokens=fact_tokens,
                    source="retriever",
                ))
                total_fact_tokens += fact_tokens
                
                logger.debug(
                    f"Fact resolved for trace_id={trace_id}: "
                    f"{fact_tokens} tokens"
                )
                
            except Exception as e:
                logger.warning(f"Fact retrieval failed for trace_id={trace_id}: {e}")
                continue
        
        plan.fact_blocks = fact_blocks
        plan.fact_tokens = total_fact_tokens
        plan.fact_rounds_used = len(fact_blocks)
        
        if plan.fact_strategy == "budget_exceeded":
            # 保持 budget_exceeded (即使 0 个 block 也保留, 说明预算不足)
            pass
        elif fact_blocks:
            plan.fact_strategy = "planner_resolved"
        else:
            plan.fact_strategy = "none"
        
        # 更新总 token 计数
        plan.total_tokens += total_fact_tokens
        
        self._stats["fact_blocks_resolved"] += len(fact_blocks)
        
        logger.info(
            f"Planner fact resolution: {len(fact_blocks)} blocks, "
            f"{total_fact_tokens} tokens, strategy={plan.fact_strategy}"
        )
        
        return plan
    
    def _append_fact_blocks_to_input(self, plan: InjectionPlan) -> str:
        """
        将 Planner 解析的事实块追加到 final_input
        
        格式:
        [原始 assembled_suffix]
        
        [补充事实]
        [FACT_SEGMENT ...]
        ...
        [/补充事实]
        
        [不需要再次请求事实的指令]
        """
        parts = [plan.assembled_suffix]
        
        # 事实块
        if self.language == "cn":
            parts.append("\n[补充事实 - 以下为原始记录，内容可信]")
        else:
            parts.append("\n[Supplementary Facts - Original records below, trustworthy]")
        
        for block in plan.fact_blocks:
            parts.append(block.fact_text)
        
        if self.language == "cn":
            parts.append("[/补充事实]")
        else:
            parts.append("[/Supplementary Facts]")
        
        # 添加 "不需要再次请求事实" 的指令
        if self.language == "cn":
            parts.append(
                "\n【重要】以上补充事实已包含所需的原始记录。"
                "请直接基于这些事实回答用户问题，"
                "不需要再调用 retrieve_fact，不需要请求更多事实。"
            )
        else:
            parts.append(
                "\n[IMPORTANT] The supplementary facts above contain the required original records. "
                "Please answer the user's question based on these facts directly. "
                "Do NOT call retrieve_fact again. Do NOT request more facts."
            )
        
        return "\n\n".join(parts)
    
    def _compute_alpha_profile(
        self,
        strategy: str,
        preferences: List[AdapterUserPreference],
        relevant_history: List[AdapterChatMessage],
        force_alpha: Optional[float],
        context: Optional[QueryContext] = None,
        history_tokens: int = 0,
    ) -> AlphaProfile:
        """
        计算分层 Alpha
        
        规则 (v3.4):
        - force_alpha 覆盖所有 (实验用)
        - recall_v4: preference 从配置读取
        - P0-4: history_alpha 动态衰减 (对数衰减, 512 token 以下不衰减)
        - P1-2: MemoryTrigger confidence 增强偏好注入 (最多 +30%)
        - 无偏好时 preference_alpha = 0.0
        
        history_alpha 衰减曲线:
        | history_tokens | history_alpha |
        |---------------|---------------|
        | 0-512         | 1.0           |
        | 1024          | ~0.79         |
        | 2048          | ~0.58         |
        | 4096          | ~0.37         |
        | 8192+         | 0.30 (下限)   |
        """
        if force_alpha is not None:
            return AlphaProfile(
                preference_alpha=force_alpha,
                history_alpha=1.0,
            )
        
        # 默认值
        pref_alpha = 0.4
        
        # 从配置读取偏好 alpha
        if self.config and hasattr(self.config, 'dki'):
            dki_cfg = self.config.dki
            if hasattr(dki_cfg, 'hybrid_injection'):
                hi_cfg = dki_cfg.hybrid_injection
                if hasattr(hi_cfg, 'preference') and hasattr(hi_cfg.preference, 'alpha'):
                    pref_alpha = hi_cfg.preference.alpha
        
        # 没有偏好时，alpha = 0
        if not preferences:
            pref_alpha = 0.0
        
        # ============ P1-2: MemoryTrigger confidence → Alpha ============
        # 高置信度 trigger 时增强偏好注入 (最多提升 30%)
        if (context
                and context.memory_triggered
                and context.trigger_confidence > 0.5
                and pref_alpha > 0):
            confidence_boost = min(context.trigger_confidence, 1.0)
            pref_alpha = min(
                pref_alpha * (0.7 + 0.3 * confidence_boost),  # 最多提升 30%
                0.7,  # 仍受 override_cap 约束
            )
        
        # ============ P0-4: history_alpha 动态衰减 ============
        # 对数衰减：512 token 以下 alpha=1.0，之后逐渐衰减到 0.3
        if history_tokens <= 0:
            # 如果没有传入 history_tokens, 从 relevant_history 估算
            if relevant_history:
                history_tokens = sum(
                    self._estimate_tokens(msg.content) for msg in relevant_history
                )
        
        if history_tokens > 0:
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
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量 (粗略: 1.3 tokens per word)"""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
    
    # ================================================================
    # 策略管理
    # ================================================================
    
    @property
    def injection_strategy(self) -> str:
        """当前注入策略"""
        return self._injection_strategy
    
    @injection_strategy.setter
    def injection_strategy(self, value: str):
        if value in ("recall_v4", "stable"):
            self._injection_strategy = value
        else:
            logger.warning(
                f"Invalid strategy: {value}. "
                f"v3.2 supports 'recall_v4' and 'stable'. Ignoring."
            )
    
    # ================================================================
    # NLP 组件配置更新 (运行时)
    # ================================================================
    
    def update_reference_resolver_config(
        self,
        just_now_turns: Optional[int] = None,
        recently_turns: Optional[int] = None,
        last_topic_turns: Optional[int] = None,
        assistant_stance_turns: Optional[int] = None,
    ):
        """运行时更新 Reference Resolver 配置"""
        self._reference_resolver.update_config(
            just_now_turns=just_now_turns,
            recently_turns=recently_turns,
            last_topic_turns=last_topic_turns,
            assistant_stance_turns=assistant_stance_turns,
        )
    
    def update_memory_trigger_config(
        self,
        enabled: Optional[bool] = None,
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
    ):
        """运行时更新 Memory Trigger 配置"""
        self._memory_trigger.update_config(
            enabled=enabled,
            custom_patterns=custom_patterns,
        )
    
    # ================================================================
    # 统计
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Planner 统计"""
        return {
            **self._stats,
            "strategy": self._injection_strategy,
            "memory_trigger": self._memory_trigger.get_stats(),
            "reference_resolver": self._reference_resolver.get_stats(),
        }
