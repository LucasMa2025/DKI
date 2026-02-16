"""
DKI Injection Planner - 注入计划生成器

职责: 只做决策, 不碰模型
- Memory Trigger 检测
- Reference Resolver 解析
- 策略选择
- Alpha 计算 (分层)
- 数据格式化 (偏好文本 + 历史后缀)
- 安全验证 (SafetyEnvelope)
- 最终输入构造

输入: query, preferences, history, config
输出: InjectionPlan

不变量:
- Planner 不持有模型引用
- Planner 不执行推理
- Planner 的输出可以被序列化、缓存、重放

Author: AGI Demo Project
Version: 3.0.0
"""

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
)

# Recall v4 组件 (可选, 通过配置启用)
try:
    from dki.core.recall import (
        RecallConfig,
        MultiSignalRecall,
        SuffixBuilder,
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
        assert plan.strategy == "stable"
        assert plan.injection_enabled == True
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
        injection_strategy: str = "stable",
        memory_trigger: Optional[MemoryTrigger] = None,
        memory_trigger_config: Optional[MemoryTriggerConfig] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
        reference_resolver_config: Optional[ReferenceResolverConfig] = None,
        safety_envelope: Optional[SafetyEnvelope] = None,
        # Recall v4 组件 (可选)
        recall_config: Optional[Any] = None,
        multi_signal_recall: Optional[Any] = None,
        suffix_builder: Optional[Any] = None,
    ):
        """
        初始化注入计划生成器
        
        Args:
            config: DKI 配置对象
            language: 语言 ("en" | "cn")
            injection_strategy: 默认注入策略 ("stable" | "full_attention" | "recall_v4")
            memory_trigger: Memory Trigger 实例 (可注入)
            memory_trigger_config: Memory Trigger 配置 (如无实例则用于创建)
            reference_resolver: Reference Resolver 实例 (可注入)
            reference_resolver_config: Reference Resolver 配置
            safety_envelope: 安全边界 (可注入)
            recall_config: Recall v4 配置 (RecallConfig 实例)
            multi_signal_recall: 多信号召回器 (MultiSignalRecall 实例)
            suffix_builder: 后缀组装器 (SuffixBuilder 实例)
        """
        self.config = config
        self.language = language
        self._injection_strategy = injection_strategy
        
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
        
        # 统计
        self._stats = {
            "plans_created": 0,
            "safety_violations": 0,
            "memory_trigger_count": 0,
            "reference_resolved_count": 0,
            "recall_v4_plans": 0,
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
            relevant_history: 相关历史消息 (用于 stable/full_attention)
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
        
        # ============ Step 1: 格式化偏好 ============
        if preferences:
            plan.preference_text = self._format_preferences(preferences)
            plan.preferences_count = len(preferences)
            plan.preference_tokens = self._estimate_tokens(plan.preference_text)
        
        # ============ Step 2: 格式化历史 (策略分支) ============
        use_recall_v4 = (
            strategy == "recall_v4"
            or (strategy == "stable"
                and self._recall_config
                and self._recall_config.enabled
                and self._recall_config.strategy == "summary_with_fact_call")
        )
        
        if use_recall_v4 and self._multi_signal_recall and self._suffix_builder:
            # ============ Recall v4: 多信号召回 + 后缀组装 ============
            plan = self._build_recall_v4_plan(
                plan=plan,
                query=query,
                session_id=session_id,
                user_id=user_id,
                context_window=context_window,
                db_session=db_session,
            )
        else:
            # ============ 原有策略: stable / full_attention ============
            # 始终准备 history_suffix (用于 fallback)
            if relevant_history:
                plan.history_suffix = self._format_history_suffix(relevant_history)
                plan.relevant_history_count = len(relevant_history)
            
            if strategy == "full_attention":
                # Full Attention: 额外准备原始消息列表
                plan.history_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in relevant_history
                ] if relevant_history else []
                total_hist_text = " ".join(
                    m["content"] for m in plan.history_messages
                )
                plan.history_tokens = self._estimate_tokens(total_hist_text)
            else:
                # Stable: 使用 suffix token 计数
                plan.history_tokens = self._estimate_tokens(plan.history_suffix)
        
        # ============ Step 3: Alpha 计算 (分层) ============
        plan.alpha_profile = self._compute_alpha_profile(
            strategy=strategy,
            preferences=preferences,
            relevant_history=relevant_history,
            force_alpha=force_alpha,
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
        if use_recall_v4 and plan.assembled_suffix:
            # Recall v4: 已在 _build_recall_v4_plan 中组装好的后缀
            plan.final_input = plan.assembled_suffix
        elif strategy == "full_attention":
            # Full Attention: 仅查询 (全局指示由 executor 添加)
            plan.final_input = query
        else:
            # Stable: 历史后缀 + 查询
            if plan.history_suffix:
                plan.final_input = plan.history_suffix + "\n\n" + query
            else:
                plan.final_input = query
        
        plan.query_tokens = self._estimate_tokens(query)
        plan.total_tokens = plan.query_tokens + plan.history_tokens
        
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
            if self.language == "cn":
                role_label = "用户" if msg.role == "user" else "助手"
            else:
                role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        
        return prefix + "\n".join(lines) + suffix
    
    def _compute_alpha_profile(
        self,
        strategy: str,
        preferences: List[AdapterUserPreference],
        relevant_history: List[AdapterChatMessage],
        force_alpha: Optional[float],
    ) -> AlphaProfile:
        """
        计算分层 Alpha
        
        规则:
        - force_alpha 覆盖所有 (实验用)
        - Stable: preference 从配置读取, history 固定为 1.0 (suffix)
        - Full Attention: preference/history 从 full_attention 配置读取
        - 无偏好时 preference_alpha = 0.0
        """
        if force_alpha is not None:
            return AlphaProfile(
                preference_alpha=force_alpha,
                history_alpha=1.0 if strategy == "stable" else force_alpha,
            )
        
        # 默认值
        pref_alpha = 0.4
        hist_alpha = 1.0 if strategy == "stable" else 0.3
        
        # 从配置读取
        if self.config and hasattr(self.config, 'dki'):
            dki_cfg = self.config.dki
            
            if strategy == "full_attention":
                if hasattr(dki_cfg, 'full_attention'):
                    fa_cfg = dki_cfg.full_attention
                    if hasattr(fa_cfg, 'preference_alpha'):
                        pref_alpha = fa_cfg.preference_alpha
                    if hasattr(fa_cfg, 'history_alpha'):
                        hist_alpha = fa_cfg.history_alpha
            else:
                if hasattr(dki_cfg, 'hybrid_injection'):
                    pref_alpha = dki_cfg.hybrid_injection.preference.alpha
        
        # 没有偏好时，alpha = 0
        if not preferences:
            pref_alpha = 0.0
        
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
        if value in ("stable", "full_attention", "recall_v4"):
            self._injection_strategy = value
        else:
            logger.warning(f"Invalid strategy: {value}")
    
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
