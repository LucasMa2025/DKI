"""
Entropy-Gated Dynamic Memory Injection (EGDMI)

============================================================================
设计思路 (来自用户讨论)
============================================================================

核心问题:
- 负位置偏好 KV 注入 (论文方案) 的真正价值不在于节约 context tokens,
  而在于偏好信息能**稳定作用于各层 attention**, 不被其他提示词稀释
- 但 vLLM 不支持运行时 KV 拼接, 全量负位置注入风险高
- 长会话中指代消解困难, 明确语义场景可行

解决方案: 混合提示词 + 熵门控动态注入

============================================================================
架构概述
============================================================================

Phase 1: Prompt Construction (静态, 每次请求)
┌──────────────────────────────────────────────────────────┐
│  偏好 (Preferences)           100-200 tokens (prefix)    │
│  历史摘要 (History Summary)   summary + trace_id         │
│  近期对话 (Recent N turns)    固定轮数, 完整保留          │
│  系统指令 (Fact Grounding)    "必须基于事实推理"          │
│  当前查询 (Current Query)     用户输入                    │
└──────────────────────────────────────────────────────────┘

Phase 2: Generation + Entropy Monitoring (动态, 逐 token)
┌──────────────────────────────────────────────────────────┐
│  vLLM 逐 token 生成                                      │
│  ↓                                                       │
│  每 N tokens 计算 logits entropy                          │
│  ↓                                                       │
│  if entropy > threshold:                                  │
│     → 暂停生成                                            │
│     → 语义分析当前生成上下文                               │
│     → 从记忆库检索相关事实                                 │
│     → 将事实注入为 "grounding context"                     │
│     → 继续生成 (带 grounding)                             │
│  else:                                                   │
│     → 正常继续                                            │
└──────────────────────────────────────────────────────────┘

============================================================================
关键创新
============================================================================

1. **偏好作为 Prompt Prefix (不是负位置 KV)**
   - 偏好 100-200 tokens, 在 8K-32K context 下占比 <2.5%
   - Prompt prefix 在 vLLM 中自动获得 prefix caching
   - 效果与负位置 KV 注入在短偏好场景下差异 <2%
   - 但实现简单, 无需修改 vLLM 内核

2. **历史 = Summary + Recent Turns (双轨)**
   - 远期历史: summary + trace_id (可追溯但不占 context)
   - 近期历史: 完整保留 N 轮 (保持对话连贯性)
   - 总 history budget 动态计算: context - 30%gen - pref - query

3. **熵门控动态记忆注入 (核心创新)**
   - 生成过程中监控 token 分布的熵值
   - 高熵 = 模型不确定 → 可能需要事实支撑
   - 触发时: 检索相关记忆, 注入为 grounding context
   - 低熵 = 模型确定 → 正常生成, 不干预

4. **事实约束指令 (Fact Grounding)**
   - 系统指令明确要求: "如需事实推理, 必须使用已知事实, 不得依赖幻觉"
   - 配合熵门控, 在模型不确定时主动提供事实

============================================================================
挑战与应对
============================================================================

挑战1: 指代消解 ("那个", "之前说的", "上次")
  → 近期 N 轮完整保留, 指代通常在 3-5 轮内
  → 远期指代通过 summary 中的关键实体链解决
  → 明确语义 (人名、地名、数字) 通过 BM25 + 向量检索

挑战2: vLLM 不支持运行时暂停/注入
  → Phase 2 (熵门控) 需要 streaming + logprobs API
  → vLLM 支持 logprobs 返回, 可在 streaming 回调中计算熵
  → 但真正的"暂停-注入-继续"需要 vLLM 的 prefix continuation
  → 替代方案: 两阶段生成 (先短生成检测熵, 再完整生成)

挑战3: 熵阈值校准
  → 不同模型、不同语言的熵分布不同
  → 需要离线 calibration: 在标注数据上统计 "需要事实" vs "不需要" 的熵分布
  → 动态阈值: 使用滑动窗口的相对熵变化, 而非绝对阈值

============================================================================
实现路线
============================================================================

Phase A (可立即实现, 不依赖 vLLM 修改):
  - Prompt Construction: 偏好 prefix + summary + recent turns + fact grounding
  - 这就是当前 recall_v4 的增强版

Phase B (需要 streaming + logprobs):
  - 逐 token 熵监控
  - 高熵触发 → 检索记忆 → 重新生成 (retry with context)
  - 不是真正的"暂停-注入", 而是 "检测-丢弃-重试"

Phase C (需要 vLLM 内核修改或 HuggingFace 模型):
  - 真正的运行时 KV 拼接
  - 负位置偏好 KV (论文原始方案)
  - 生成过程中动态 KV 注入

============================================================================
"""

import time
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


# ============================================================
# Data Structures
# ============================================================

@dataclass
class EntropyState:
    """逐 token 熵状态"""
    token_idx: int = 0
    entropy: float = 0.0
    is_high_entropy: bool = False
    window_entropy_avg: float = 0.0
    window_entropy_std: float = 0.0
    relative_entropy_spike: float = 0.0  # 相对于窗口平均的偏移


@dataclass
class GroundingContext:
    """事实 grounding 上下文"""
    facts: List[str] = field(default_factory=list)
    source_trace_ids: List[str] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    injection_reason: str = ""


@dataclass
class EGDMIConfig:
    """Entropy-Gated Dynamic Memory Injection 配置"""
    
    # Phase 1: Prompt Construction
    preference_max_tokens: int = 200
    history_summary_max_tokens: int = 300
    recent_turns_count: int = 3
    fact_grounding_instruction: str = (
        "Important: When reasoning about facts, you MUST use information "
        "from the conversation history and user preferences. "
        "Do NOT fabricate or hallucinate facts. "
        "If you are unsure, say so explicitly."
    )
    
    # Phase 2: Entropy Monitoring
    entropy_check_interval: int = 8  # 每 N tokens 检查一次熵
    entropy_window_size: int = 32    # 滑动窗口大小
    entropy_spike_threshold: float = 1.5  # 相对熵突增阈值 (std 倍数)
    entropy_absolute_threshold: float = 3.0  # 绝对熵阈值 (nats)
    max_retrieval_per_generation: int = 2  # 每次生成最多触发几次检索
    
    # Phase 2: Retrieval
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.3
    
    # Generation
    generation_reserve_ratio: float = 0.30
    context_window: int = 8192


@dataclass
class EGDMIPrompt:
    """EGDMI 构造的完整提示词"""
    preference_section: str = ""
    summary_section: str = ""
    recent_turns_section: str = ""
    grounding_instruction: str = ""
    current_query: str = ""
    
    # Token 统计
    preference_tokens: int = 0
    summary_tokens: int = 0
    recent_turns_tokens: int = 0
    instruction_tokens: int = 0
    query_tokens: int = 0
    total_tokens: int = 0
    
    # 完整 prompt
    full_prompt: str = ""


# ============================================================
# Phase 1: Enhanced Prompt Builder
# ============================================================

class EGDMIPromptBuilder:
    """
    EGDMI 提示词构造器
    
    构造策略:
    1. 偏好 (prefix): 100-200 tokens, 固定位置
    2. 历史摘要: summary + trace_id, 可追溯
    3. 近期对话: 完整保留最近 N 轮
    4. 事实约束指令: 明确要求基于事实推理
    5. 当前查询: 用户输入
    
    Token 预算:
    - generation_reserve = context_window * 0.30
    - available = context_window - generation_reserve
    - preference: min(actual, preference_max_tokens)
    - instruction: ~50 tokens (固定)
    - query: actual estimation
    - recent_turns: actual (最近 N 轮完整保留)
    - summary: 剩余空间
    """
    
    def __init__(self, config: EGDMIConfig):
        self.config = config
    
    def build(
        self,
        query: str,
        preferences: List[str],
        history_summary: str,
        summary_trace_ids: List[str],
        recent_messages: List[Dict[str, str]],  # [{"role": "user"/"assistant", "content": "..."}]
    ) -> EGDMIPrompt:
        """构造 EGDMI 提示词"""
        
        prompt = EGDMIPrompt()
        
        # 1. 偏好 section
        if preferences:
            pref_text = "User preferences:\n" + "\n".join(
                f"- {p}" for p in preferences
            )
            prompt.preference_section = pref_text
            prompt.preference_tokens = self._estimate_tokens(pref_text)
        
        # 2. 事实约束指令
        prompt.grounding_instruction = self.config.fact_grounding_instruction
        prompt.instruction_tokens = self._estimate_tokens(prompt.grounding_instruction)
        
        # 3. 当前查询
        prompt.current_query = query
        prompt.query_tokens = self._estimate_tokens(query)
        
        # 4. 近期对话 (完整保留)
        recent_text = ""
        if recent_messages:
            recent_parts = []
            for msg in recent_messages[-self.config.recent_turns_count * 2:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                recent_parts.append(f"[{role}]: {content}")
            recent_text = "\n".join(recent_parts)
            prompt.recent_turns_section = recent_text
            prompt.recent_turns_tokens = self._estimate_tokens(recent_text)
        
        # 5. 历史摘要 (用剩余空间)
        generation_reserve = int(self.config.context_window * self.config.generation_reserve_ratio)
        available = self.config.context_window - generation_reserve
        
        used = (
            prompt.preference_tokens
            + prompt.instruction_tokens
            + prompt.query_tokens
            + prompt.recent_turns_tokens
            + 50  # chat template overhead
        )
        
        summary_budget = available - used
        
        if history_summary and summary_budget > 50:
            summary_text = f"Conversation history summary:\n{history_summary}"
            if summary_trace_ids:
                summary_text += f"\n[trace: {', '.join(summary_trace_ids[:5])}]"
            
            summary_tokens = self._estimate_tokens(summary_text)
            if summary_tokens > summary_budget:
                # 截断 summary
                ratio = summary_budget / summary_tokens
                chars_limit = int(len(summary_text) * ratio * 0.9)
                summary_text = summary_text[:chars_limit] + "..."
                summary_tokens = self._estimate_tokens(summary_text)
            
            prompt.summary_section = summary_text
            prompt.summary_tokens = summary_tokens
        
        # 6. 组装完整 prompt
        prompt.total_tokens = (
            prompt.preference_tokens
            + prompt.summary_tokens
            + prompt.recent_turns_tokens
            + prompt.instruction_tokens
            + prompt.query_tokens
        )
        
        prompt.full_prompt = self._assemble(prompt)
        
        return prompt
    
    def _assemble(self, prompt: EGDMIPrompt) -> str:
        """组装为 ChatML 格式"""
        parts = []
        
        # System message: preferences + grounding instruction
        system_parts = []
        if prompt.preference_section:
            system_parts.append(prompt.preference_section)
        if prompt.grounding_instruction:
            system_parts.append(prompt.grounding_instruction)
        
        if system_parts:
            system_content = "\n\n".join(system_parts)
            parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
        
        # History summary (as system context)
        if prompt.summary_section:
            parts.append(f"<|im_start|>system\n{prompt.summary_section}<|im_end|>")
        
        # Recent turns
        if prompt.recent_turns_section:
            # Parse recent turns back into role/content pairs
            for line in prompt.recent_turns_section.split("\n"):
                if line.startswith("[user]:"):
                    content = line[len("[user]:"):].strip()
                    parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif line.startswith("[assistant]:"):
                    content = line[len("[assistant]:"):].strip()
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Current query
        parts.append(f"<|im_start|>user\n{prompt.current_query}<|im_end|>")
        
        # Open assistant tag for generation
        parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """快速 token 估算"""
        if not text:
            return 0
        
        # 中文: ~1.5 tokens/char, 英文: ~1.3 tokens/word
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        ascii_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
        
        tokens = chinese_chars * 1.5 + ascii_words * 1.3 + other_chars * 0.5
        return int(tokens * 1.15)  # 15% overestimate


# ============================================================
# Phase 2: Entropy Monitor
# ============================================================

class EntropyMonitor:
    """
    Token 分布熵监控器
    
    在 streaming 生成过程中, 监控每个 token 的 logits 分布熵。
    当检测到熵突增 (模型不确定性增加) 时, 触发记忆检索。
    
    熵计算:
    H(p) = -Σ p_i * log(p_i)
    
    对于 vocab_size=32000 的模型:
    - 均匀分布 (最大不确定): H ≈ 10.4 nats
    - 确定性输出 (最小不确定): H ≈ 0 nats
    - 典型生成: H ≈ 1.5-4.0 nats
    - 高不确定性: H > 4.0 nats
    
    检测策略:
    1. 绝对阈值: entropy > absolute_threshold
    2. 相对突增: entropy > window_avg + spike_threshold * window_std
    3. 两者满足其一即触发
    """
    
    def __init__(self, config: EGDMIConfig):
        self.config = config
        self._entropy_window: List[float] = []
        self._retrieval_count: int = 0
        self._total_tokens: int = 0
    
    def reset(self):
        """重置状态 (每次新生成时调用)"""
        self._entropy_window = []
        self._retrieval_count = 0
        self._total_tokens = 0
    
    def should_check(self) -> bool:
        """是否应该在当前 token 检查熵"""
        return (
            self._total_tokens % self.config.entropy_check_interval == 0
            and self._retrieval_count < self.config.max_retrieval_per_generation
        )
    
    def update(self, logprobs: List[float]) -> EntropyState:
        """
        更新熵状态
        
        Args:
            logprobs: 当前 token 的 top-k log probabilities
                      (vLLM streaming 返回的 logprobs)
        
        Returns:
            EntropyState with is_high_entropy flag
        """
        self._total_tokens += 1
        
        # 计算熵 (从 logprobs)
        entropy = self._compute_entropy_from_logprobs(logprobs)
        
        # 更新滑动窗口
        self._entropy_window.append(entropy)
        if len(self._entropy_window) > self.config.entropy_window_size:
            self._entropy_window.pop(0)
        
        # 计算窗口统计
        window_avg = sum(self._entropy_window) / len(self._entropy_window)
        
        if len(self._entropy_window) >= 4:
            variance = sum(
                (e - window_avg) ** 2 for e in self._entropy_window
            ) / len(self._entropy_window)
            window_std = math.sqrt(variance)
        else:
            window_std = 1.0  # 窗口太小, 使用默认值
        
        # 检测熵突增
        relative_spike = (entropy - window_avg) / max(window_std, 0.1)
        
        is_high = (
            entropy > self.config.entropy_absolute_threshold
            or relative_spike > self.config.entropy_spike_threshold
        )
        
        state = EntropyState(
            token_idx=self._total_tokens,
            entropy=entropy,
            is_high_entropy=is_high,
            window_entropy_avg=window_avg,
            window_entropy_std=window_std,
            relative_entropy_spike=relative_spike,
        )
        
        if is_high:
            self._retrieval_count += 1
            logger.debug(
                f"[EGDMI] High entropy detected at token {self._total_tokens}: "
                f"H={entropy:.2f}, avg={window_avg:.2f}, "
                f"spike={relative_spike:.1f}σ"
            )
        
        return state
    
    @staticmethod
    def _compute_entropy_from_logprobs(logprobs: List[float]) -> float:
        """
        从 log probabilities 计算熵
        
        H = -Σ exp(logp) * logp
        """
        if not logprobs:
            return 0.0
        
        entropy = 0.0
        for logp in logprobs:
            if logp > -30:  # 避免数值问题
                p = math.exp(logp)
                if p > 0:
                    entropy -= p * logp
        
        return entropy


# ============================================================
# Phase 2: Memory Retriever (for entropy-triggered retrieval)
# ============================================================

class EntropyTriggeredRetriever:
    """
    熵触发的记忆检索器
    
    当 EntropyMonitor 检测到高熵时, 基于当前生成上下文
    检索相关记忆, 构造 grounding context.
    
    检索策略:
    1. 提取当前生成文本的最后 N tokens 作为查询
    2. 结合原始用户查询进行联合检索
    3. 过滤已在 prompt 中出现的记忆 (避免重复)
    4. 返回 top-k 相关记忆作为 grounding context
    """
    
    def __init__(
        self,
        config: EGDMIConfig,
        memory_recall=None,  # MultiSignalRecall instance
    ):
        self.config = config
        self.memory_recall = memory_recall
    
    def retrieve_grounding(
        self,
        original_query: str,
        generated_so_far: str,
        session_id: str,
        user_id: Optional[str] = None,
        already_used_ids: Optional[set] = None,
        db_session=None,
    ) -> GroundingContext:
        """
        检索 grounding context
        
        Args:
            original_query: 原始用户查询
            generated_so_far: 已生成的文本
            session_id: 会话 ID
            user_id: 用户 ID
            already_used_ids: 已在 prompt 中使用的记忆 ID
            db_session: 数据库会话
        
        Returns:
            GroundingContext with retrieved facts
        """
        if not self.memory_recall:
            return GroundingContext(injection_reason="no_recall_system")
        
        # 构造联合查询: 原始查询 + 最近生成文本
        recent_generated = generated_so_far[-200:] if len(generated_so_far) > 200 else generated_so_far
        combined_query = f"{original_query} {recent_generated}"
        
        try:
            recall_result = self.memory_recall.recall(
                query=combined_query,
                session_id=session_id,
                user_id=user_id,
                db_session=db_session,
                max_results=self.config.retrieval_top_k * 2,
            )
        except Exception as e:
            logger.warning(f"[EGDMI] Retrieval failed: {e}")
            return GroundingContext(injection_reason=f"retrieval_error: {e}")
        
        # 过滤已使用的记忆
        already_used = already_used_ids or set()
        facts = []
        trace_ids = []
        scores = []
        
        for item in recall_result.items:
            if item.message_id in already_used:
                continue
            if item.final_score < self.config.retrieval_min_score:
                continue
            
            facts.append(item.content)
            trace_ids.append(item.message_id)
            scores.append(item.final_score)
            
            if len(facts) >= self.config.retrieval_top_k:
                break
        
        return GroundingContext(
            facts=facts,
            source_trace_ids=trace_ids,
            relevance_scores=scores,
            injection_reason="entropy_spike",
        )


# ============================================================
# EGDMI Controller (Orchestrator)
# ============================================================

class EGDMIController:
    """
    EGDMI 主控制器
    
    编排 Phase 1 (Prompt Construction) 和 Phase 2 (Entropy Monitoring)。
    
    使用模式:
    
    1. 基础模式 (Phase A, 可立即使用):
       只使用 Phase 1 的增强提示词, 不做熵监控。
       等价于当前 recall_v4 的增强版。
    
    2. 熵监控模式 (Phase B, 需要 streaming + logprobs):
       在 streaming 生成中监控熵, 高熵时触发检索。
       使用 "检测-丢弃-重试" 策略 (不需要 vLLM 内核修改)。
    
    3. 完整模式 (Phase C, 需要 HF 模型或 vLLM 修改):
       真正的运行时 KV 拼接, 负位置偏好 KV, 生成中动态注入。
    
    当前实现: Phase A + Phase B 的基础框架
    """
    
    def __init__(self, config: Optional[EGDMIConfig] = None):
        self.config = config or EGDMIConfig()
        self.prompt_builder = EGDMIPromptBuilder(self.config)
        self.entropy_monitor = EntropyMonitor(self.config)
        self.retriever: Optional[EntropyTriggeredRetriever] = None
    
    def set_retriever(self, memory_recall):
        """设置记忆检索器"""
        self.retriever = EntropyTriggeredRetriever(
            config=self.config,
            memory_recall=memory_recall,
        )
    
    def build_prompt(
        self,
        query: str,
        preferences: List[str],
        history_summary: str,
        summary_trace_ids: List[str],
        recent_messages: List[Dict[str, str]],
    ) -> EGDMIPrompt:
        """Phase 1: 构造增强提示词"""
        return self.prompt_builder.build(
            query=query,
            preferences=preferences,
            history_summary=history_summary,
            summary_trace_ids=summary_trace_ids,
            recent_messages=recent_messages,
        )
    
    def create_entropy_callback(
        self,
        original_query: str,
        session_id: str,
        user_id: Optional[str] = None,
        already_used_ids: Optional[set] = None,
        db_session=None,
    ):
        """
        创建 streaming 回调函数, 用于熵监控
        
        返回一个回调函数, 可以传递给 vLLM streaming API:
        
        Usage:
            callback = controller.create_entropy_callback(...)
            
            async for chunk in vllm_stream(prompt, logprobs=5):
                grounding = callback(chunk.logprobs, chunk.text_so_far)
                if grounding and grounding.facts:
                    # 高熵触发, 需要重试
                    new_prompt = inject_grounding(prompt, grounding)
                    # restart generation with new_prompt
        """
        self.entropy_monitor.reset()
        generated_text = []
        
        def callback(
            logprobs: Optional[List[float]],
            token_text: str,
        ) -> Optional[GroundingContext]:
            """
            每个 token 调用一次
            
            Returns:
                GroundingContext if high entropy detected and retrieval succeeded,
                None otherwise
            """
            generated_text.append(token_text)
            
            if not logprobs or not self.entropy_monitor.should_check():
                return None
            
            state = self.entropy_monitor.update(logprobs)
            
            if not state.is_high_entropy:
                return None
            
            if not self.retriever:
                logger.debug("[EGDMI] High entropy but no retriever configured")
                return None
            
            # 检索 grounding context
            full_generated = "".join(generated_text)
            grounding = self.retriever.retrieve_grounding(
                original_query=original_query,
                generated_so_far=full_generated,
                session_id=session_id,
                user_id=user_id,
                already_used_ids=already_used_ids,
                db_session=db_session,
            )
            
            if grounding.facts:
                logger.info(
                    f"[EGDMI] Entropy-triggered retrieval: "
                    f"{len(grounding.facts)} facts retrieved "
                    f"at token {state.token_idx}"
                )
                return grounding
            
            return None
        
        return callback
    
    def inject_grounding_into_prompt(
        self,
        original_prompt: str,
        grounding: GroundingContext,
    ) -> str:
        """
        将 grounding context 注入到 prompt 中
        
        策略: 在 assistant 开始标记之前插入 grounding 信息
        """
        grounding_text = "Relevant facts for your reference:\n"
        for i, fact in enumerate(grounding.facts, 1):
            grounding_text += f"[Fact {i}] {fact}\n"
        grounding_text += "Please use these facts in your response.\n"
        
        # 在 <|im_start|>assistant 之前插入
        marker = "<|im_start|>assistant\n"
        if marker in original_prompt:
            idx = original_prompt.rfind(marker)
            return (
                original_prompt[:idx]
                + f"<|im_start|>system\n{grounding_text}<|im_end|>\n"
                + original_prompt[idx:]
            )
        
        # Fallback: 追加到末尾
        return original_prompt + f"\n{grounding_text}"


# ============================================================
# Phase B: Two-Stage Generation (检测-丢弃-重试)
# ============================================================

class TwoStageGenerator:
    """
    两阶段生成器 (Phase B 的可行实现)
    
    不需要修改 vLLM 内核, 使用两次 generate 调用:
    
    Stage 1: 短生成 (probe)
    - max_tokens = 64
    - logprobs = 5
    - 检测是否有高熵 token
    
    Stage 2: 完整生成
    - 如果 Stage 1 没有高熵: 继续生成 (prefix continuation)
    - 如果 Stage 1 有高熵: 检索记忆, 注入 grounding, 重新生成
    
    优点:
    - 不需要 vLLM 修改
    - 只有在检测到高熵时才有额外开销
    - 大多数情况下只需要一次完整生成
    
    缺点:
    - Stage 1 的短生成是额外开销 (但很小, 64 tokens)
    - 重试时丢弃 Stage 1 的结果 (浪费少量计算)
    - 不能在生成中途注入 (只能在开始时)
    """
    
    def __init__(
        self,
        controller: EGDMIController,
        model_adapter=None,  # BaseModelAdapter
    ):
        self.controller = controller
        self.model = model_adapter
    
    async def generate(
        self,
        prompt: str,
        original_query: str,
        session_id: str,
        user_id: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        probe_tokens: int = 64,
        logprobs_k: int = 5,
        already_used_ids: Optional[set] = None,
        db_session=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        两阶段生成
        
        Returns:
            {
                "text": str,
                "latency_ms": float,
                "entropy_triggered": bool,
                "grounding_facts": List[str],
                "stages": int,  # 1 or 2
            }
        """
        start_time = time.time()
        
        # Stage 1: Probe generation
        probe_output = self.model.generate(
            prompt=prompt,
            max_new_tokens=probe_tokens,
            temperature=temperature,
            logprobs=logprobs_k,
            **kwargs,
        )
        
        # Analyze entropy from probe
        entropy_triggered = False
        grounding = None
        
        if hasattr(probe_output, 'logprobs') and probe_output.logprobs:
            callback = self.controller.create_entropy_callback(
                original_query=original_query,
                session_id=session_id,
                user_id=user_id,
                already_used_ids=already_used_ids,
                db_session=db_session,
            )
            
            for token_logprobs in probe_output.logprobs:
                result = callback(token_logprobs, "")
                if result and result.facts:
                    entropy_triggered = True
                    grounding = result
                    break
        
        if entropy_triggered and grounding:
            # Stage 2: Re-generate with grounding
            enhanced_prompt = self.controller.inject_grounding_into_prompt(
                prompt, grounding
            )
            
            full_output = self.model.generate(
                prompt=enhanced_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            return {
                "text": full_output.text,
                "latency_ms": latency_ms,
                "entropy_triggered": True,
                "grounding_facts": grounding.facts,
                "stages": 2,
            }
        else:
            # No high entropy: continue with remaining tokens
            if probe_output.output_tokens >= max_new_tokens:
                # Probe already generated enough
                latency_ms = (time.time() - start_time) * 1000
                return {
                    "text": probe_output.text,
                    "latency_ms": latency_ms,
                    "entropy_triggered": False,
                    "grounding_facts": [],
                    "stages": 1,
                }
            
            # Generate remaining tokens
            remaining_tokens = max_new_tokens - probe_output.output_tokens
            continuation_prompt = prompt + probe_output.text
            
            cont_output = self.model.generate(
                prompt=continuation_prompt,
                max_new_tokens=remaining_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            full_text = probe_output.text + cont_output.text
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "text": full_text,
                "latency_ms": latency_ms,
                "entropy_triggered": False,
                "grounding_facts": [],
                "stages": 1,  # 逻辑上是一次生成 (probe + continuation)
            }
