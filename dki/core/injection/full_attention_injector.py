"""
Full Attention Injector - Research Implementation of Plan C

基于 plan.md 方案 C 的实现：
- C1: 固定负位置 (Fixed Negative Position)
- 用户偏好 + 历史消息均通过 K/V 注入
- 目标: 0% Context 占用 (仅保留极简全局指示)

研究目的:
1. 验证历史消息 K/V 注入的可行性
2. 对比 Stable 策略的输出质量
3. 收集 attention pattern 数据

Author: AGI Demo Project
Version: 1.0.0
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
from loguru import logger


class PositionMode(Enum):
    """位置编码模式 (对应 Plan C 的子方案)"""
    FIXED_NEGATIVE = "fixed_negative"  # C1: 固定负位置
    CONSTANT = "constant"              # C1 变体: 所有 KV 使用相同位置
    NOPE = "nope"                      # C1 变体: 不应用位置编码 (NoPE)


@dataclass
class FullAttentionConfig:
    """Full Attention 策略配置"""
    enabled: bool = True
    
    # 位置模式
    position_mode: PositionMode = PositionMode.FIXED_NEGATIVE
    
    # 偏好 K/V 配置
    preference_position_start: int = -100
    preference_alpha: float = 0.4
    preference_max_tokens: int = 100
    
    # 历史 K/V 配置
    history_position_start: int = -500
    history_alpha: float = 0.3
    history_max_tokens: int = 400
    history_max_messages: int = 10
    
    # 全局指示
    global_indication_enabled: bool = True
    global_indication_en: str = "[Memory Context Available]"
    global_indication_cn: str = "[记忆上下文可用]"
    
    # 安全设置
    max_total_kv_tokens: int = 600
    fallback_to_stable: bool = True
    log_attention_patterns: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FullAttentionConfig":
        """从字典创建配置"""
        position_mode = config.get("position_mode", "fixed_negative")
        if isinstance(position_mode, str):
            position_mode = PositionMode(position_mode)
        
        return cls(
            enabled=config.get("enabled", True),
            position_mode=position_mode,
            preference_position_start=config.get("preference", {}).get("position_start", -100),
            preference_alpha=config.get("preference", {}).get("alpha", 0.4),
            preference_max_tokens=config.get("preference", {}).get("max_tokens", 100),
            history_position_start=config.get("history", {}).get("position_start", -500),
            history_alpha=config.get("history", {}).get("alpha", 0.3),
            history_max_tokens=config.get("history", {}).get("max_tokens", 400),
            history_max_messages=config.get("history", {}).get("max_messages", 10),
            global_indication_enabled=config.get("global_indication", {}).get("enabled", True),
            global_indication_en=config.get("global_indication", {}).get("text_en", "[Memory Context Available]"),
            global_indication_cn=config.get("global_indication", {}).get("text_cn", "[记忆上下文可用]"),
            max_total_kv_tokens=config.get("safety", {}).get("max_total_kv_tokens", 600),
            fallback_to_stable=config.get("safety", {}).get("fallback_to_stable", True),
            log_attention_patterns=config.get("safety", {}).get("log_attention_patterns", True),
        )


@dataclass
class InjectionResult:
    """注入结果"""
    success: bool = False
    
    # K/V 数据
    preference_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    history_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    merged_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    
    # 位置信息
    preference_positions: Optional[List[int]] = None
    history_positions: Optional[List[int]] = None
    
    # Token 统计
    preference_tokens: int = 0
    history_tokens: int = 0
    total_kv_tokens: int = 0
    
    # 全局指示 (唯一的 context 占用)
    global_indication: str = ""
    
    # 元数据
    position_mode: str = ""
    fallback_triggered: bool = False
    error_message: str = ""
    
    # 性能
    compute_time_ms: float = 0.0


class FullAttentionInjector:
    """
    Full Attention 注入器 - 方案 C 实现
    
    核心思想:
    1. 用户偏好: K/V 注入到负位置 (与 Stable 策略相同)
    2. 历史消息: K/V 注入到负位置 (NEW: 替代 suffix prompt)
    3. 全局指示: 极简提示 (约 3-5 tokens)
    
    位置布局:
    ┌─────────────────────────────────────────────────────────────────┐
    │  [History KV]     │  [Preference KV]   │  [Query + Indication] │
    │  pos: -500~-101   │  pos: -100~-1      │  pos: 0~L             │
    │  α: 0.3           │  α: 0.4            │  α: 1.0               │
    └─────────────────────────────────────────────────────────────────┘
    
    与 Stable 策略对比:
    - Stable:  偏好(K/V) + 历史(Suffix Prompt) → Context 占用中等
    - Full:    偏好(K/V) + 历史(K/V) + 指示    → Context 占用极小
    """
    
    def __init__(
        self,
        config: Optional[FullAttentionConfig] = None,
        language: str = "en",
    ):
        """
        初始化 Full Attention 注入器
        
        Args:
            config: 配置
            language: 语言 ("en" | "cn")
        """
        self.config = config or FullAttentionConfig()
        self.language = language
        
        # 统计数据
        self._stats = {
            "total_injections": 0,
            "successful_injections": 0,
            "fallback_count": 0,
            "avg_preference_tokens": 0.0,
            "avg_history_tokens": 0.0,
            "avg_compute_time_ms": 0.0,
        }
        
        # Attention pattern 日志 (用于研究分析)
        self._attention_logs: List[Dict[str, Any]] = []
        self._max_attention_logs = 100
        
        logger.info(
            f"FullAttentionInjector initialized "
            f"(mode={self.config.position_mode.value}, language={language})"
        )
    
    def inject(
        self,
        model_adapter: Any,
        preference_text: str,
        history_messages: List[Dict[str, str]],
        query: str,
    ) -> InjectionResult:
        """
        执行 Full Attention 注入
        
        Args:
            model_adapter: 模型适配器 (用于计算 K/V)
            preference_text: 用户偏好文本
            history_messages: 历史消息列表 [{"role": "user/assistant", "content": "..."}]
            query: 用户查询
            
        Returns:
            InjectionResult 包含注入结果
        """
        start_time = time.time()
        result = InjectionResult()
        result.position_mode = self.config.position_mode.value
        
        self._stats["total_injections"] += 1
        
        try:
            # ============ Step 1: 估算 token 数量 ============
            preference_tokens = self._estimate_tokens(preference_text)
            history_text = self._format_history_for_kv(history_messages)
            history_tokens = self._estimate_tokens(history_text)
            total_tokens = preference_tokens + history_tokens
            
            result.preference_tokens = preference_tokens
            result.history_tokens = history_tokens
            result.total_kv_tokens = total_tokens
            
            # ============ Step 2: 安全检查 ============
            if total_tokens > self.config.max_total_kv_tokens:
                if self.config.fallback_to_stable:
                    result.fallback_triggered = True
                    result.error_message = (
                        f"Total K/V tokens ({total_tokens}) exceeds limit "
                        f"({self.config.max_total_kv_tokens}), fallback to stable"
                    )
                    self._stats["fallback_count"] += 1
                    logger.warning(result.error_message)
                    return result
                else:
                    # 截断历史
                    history_messages = self._truncate_history(
                        history_messages,
                        self.config.max_total_kv_tokens - preference_tokens,
                    )
                    history_text = self._format_history_for_kv(history_messages)
                    history_tokens = self._estimate_tokens(history_text)
            
            # ============ Step 3: 计算偏好 K/V ============
            preference_kv = None
            preference_positions = []
            
            if preference_text and preference_tokens > 0:
                preference_kv, _ = model_adapter.compute_kv(preference_text)
                
                # 计算位置
                preference_positions = self._compute_positions(
                    start=self.config.preference_position_start,
                    num_tokens=preference_tokens,
                )
                
                # 应用位置编码 (根据模式)
                preference_kv = self._apply_position_encoding(
                    preference_kv,
                    preference_positions,
                    model_adapter,
                )
                
                result.preference_kv = preference_kv
                result.preference_positions = preference_positions
            
            # ============ Step 4: 计算历史 K/V (核心创新) ============
            history_kv = None
            history_positions = []
            
            if history_text and history_tokens > 0:
                history_kv, _ = model_adapter.compute_kv(history_text)
                
                # 计算位置 (在偏好之前)
                history_positions = self._compute_positions(
                    start=self.config.history_position_start,
                    num_tokens=history_tokens,
                )
                
                # 应用位置编码
                history_kv = self._apply_position_encoding(
                    history_kv,
                    history_positions,
                    model_adapter,
                )
                
                result.history_kv = history_kv
                result.history_positions = history_positions
            
            # ============ Step 5: 合并 K/V ============
            if preference_kv or history_kv:
                result.merged_kv = self._merge_kv(
                    preference_kv=preference_kv,
                    history_kv=history_kv,
                    preference_alpha=self.config.preference_alpha,
                    history_alpha=self.config.history_alpha,
                )
            
            # ============ Step 6: 生成全局指示 ============
            if self.config.global_indication_enabled:
                if self.language == "cn":
                    result.global_indication = self.config.global_indication_cn
                else:
                    result.global_indication = self.config.global_indication_en
            
            # ============ Step 7: 记录成功 ============
            result.success = True
            result.compute_time_ms = (time.time() - start_time) * 1000
            
            self._stats["successful_injections"] += 1
            self._update_avg_stats(result)
            
            # 记录 attention pattern (用于研究)
            if self.config.log_attention_patterns:
                self._log_attention_pattern(result, query)
            
            logger.debug(
                f"Full attention injection completed: "
                f"pref={preference_tokens} tokens, hist={history_tokens} tokens, "
                f"time={result.compute_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.compute_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Full attention injection failed: {e}")
            return result
    
    def _format_history_for_kv(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化历史消息为 K/V 注入文本
        
        注意: 这里不使用 suffix prompt 模板，而是简洁格式
        因为 K/V 注入不需要显式提示词
        """
        if not messages:
            return ""
        
        lines = []
        for msg in messages[-self.config.history_max_messages:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if self.language == "cn":
                role_label = "用户" if role == "user" else "助手"
            else:
                role_label = "User" if role == "user" else "Assistant"
            
            lines.append(f"{role_label}: {content}")
        
        return "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        if not text:
            return 0
        # 粗略估算: 1.3 tokens per word (中文按字符)
        if self.language == "cn":
            return int(len(text) * 0.7)  # 中文字符约 0.7 tokens
        return int(len(text.split()) * 1.3)
    
    def _compute_positions(self, start: int, num_tokens: int) -> List[int]:
        """
        计算位置序列
        
        Args:
            start: 起始位置 (负数)
            num_tokens: token 数量
            
        Returns:
            位置列表，从 start 递增到 start + num_tokens - 1
        """
        return list(range(start, start + num_tokens))
    
    def _apply_position_encoding(
        self,
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
        positions: List[int],
        model_adapter: Any,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        应用位置编码
        
        根据 position_mode 选择不同的策略:
        - FIXED_NEGATIVE: 使用负位置的 RoPE
        - CONSTANT: 所有 token 使用相同位置
        - NOPE: 不应用位置编码
        """
        if self.config.position_mode == PositionMode.NOPE:
            # NoPE: 不应用位置编码，直接返回原始 K/V
            return kv_list
        
        if self.config.position_mode == PositionMode.CONSTANT:
            # 常量位置: 所有 token 使用相同位置 (如 -10000)
            constant_pos = self.config.history_position_start - 10000
            positions = [constant_pos] * len(positions)
        
        # FIXED_NEGATIVE 或 CONSTANT: 应用 RoPE
        # 注意: 这里假设 model_adapter 支持 apply_rope_to_kv 方法
        # 如果不支持，则跳过位置编码
        if hasattr(model_adapter, 'apply_rope_to_kv'):
            return model_adapter.apply_rope_to_kv(kv_list, positions)
        
        # 如果模型不支持自定义位置编码，返回原始 K/V
        logger.debug("Model adapter does not support custom position encoding, using raw K/V")
        return kv_list
    
    def _merge_kv(
        self,
        preference_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        history_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        preference_alpha: float,
        history_alpha: float,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        合并偏好和历史的 K/V
        
        位置布局: [History KV] [Preference KV]
        """
        if not preference_kv and not history_kv:
            return []
        
        if not preference_kv:
            # 只有历史
            return self._scale_kv(history_kv, history_alpha)
        
        if not history_kv:
            # 只有偏好
            return self._scale_kv(preference_kv, preference_alpha)
        
        # 合并: 历史在前，偏好在后
        merged = []
        num_layers = len(preference_kv)
        
        for layer_idx in range(num_layers):
            h_k, h_v = history_kv[layer_idx]
            p_k, p_v = preference_kv[layer_idx]
            
            # 应用 α 缩放
            h_k = h_k * history_alpha
            h_v = h_v * history_alpha
            p_k = p_k * preference_alpha
            p_v = p_v * preference_alpha
            
            # 拼接: [History, Preference]
            merged_k = torch.cat([h_k, p_k], dim=2)
            merged_v = torch.cat([h_v, p_v], dim=2)
            
            merged.append((merged_k, merged_v))
        
        return merged
    
    def _scale_kv(
        self,
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
        alpha: float,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """应用 α 缩放"""
        return [(k * alpha, v * alpha) for k, v in kv_list]
    
    def _truncate_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """截断历史消息以满足 token 限制"""
        if not messages:
            return []
        
        result = []
        total_tokens = 0
        
        # 从最近的消息开始
        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = self._estimate_tokens(content)
            
            if total_tokens + tokens > max_tokens:
                break
            
            result.insert(0, msg)
            total_tokens += tokens
        
        return result
    
    def _update_avg_stats(self, result: InjectionResult):
        """更新平均统计"""
        n = self._stats["successful_injections"]
        
        # 增量平均
        self._stats["avg_preference_tokens"] = (
            (self._stats["avg_preference_tokens"] * (n - 1) + result.preference_tokens) / n
        )
        self._stats["avg_history_tokens"] = (
            (self._stats["avg_history_tokens"] * (n - 1) + result.history_tokens) / n
        )
        self._stats["avg_compute_time_ms"] = (
            (self._stats["avg_compute_time_ms"] * (n - 1) + result.compute_time_ms) / n
        )
    
    def _log_attention_pattern(self, result: InjectionResult, query: str):
        """记录 attention pattern (用于研究分析)"""
        log_entry = {
            "timestamp": time.time(),
            "query": query[:100],  # 截断
            "position_mode": result.position_mode,
            "preference_tokens": result.preference_tokens,
            "history_tokens": result.history_tokens,
            "preference_positions": (
                result.preference_positions[:5] if result.preference_positions else []
            ),
            "history_positions": (
                result.history_positions[:5] if result.history_positions else []
            ),
            "compute_time_ms": result.compute_time_ms,
        }
        
        self._attention_logs.append(log_entry)
        
        # 限制日志数量
        if len(self._attention_logs) > self._max_attention_logs:
            self._attention_logs = self._attention_logs[-self._max_attention_logs:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        return {
            **self._stats,
            "config": {
                "position_mode": self.config.position_mode.value,
                "preference_alpha": self.config.preference_alpha,
                "history_alpha": self.config.history_alpha,
                "max_total_kv_tokens": self.config.max_total_kv_tokens,
            },
        }
    
    def get_attention_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取 attention pattern 日志"""
        return self._attention_logs[-limit:]
    
    def update_config(
        self,
        position_mode: Optional[str] = None,
        preference_alpha: Optional[float] = None,
        history_alpha: Optional[float] = None,
        history_position_start: Optional[int] = None,
        max_total_kv_tokens: Optional[int] = None,
    ):
        """运行时更新配置"""
        if position_mode:
            self.config.position_mode = PositionMode(position_mode)
        if preference_alpha is not None:
            self.config.preference_alpha = preference_alpha
        if history_alpha is not None:
            self.config.history_alpha = history_alpha
        if history_position_start is not None:
            self.config.history_position_start = history_position_start
        if max_total_kv_tokens is not None:
            self.config.max_total_kv_tokens = max_total_kv_tokens
        
        logger.info(f"FullAttentionInjector config updated: {self.get_stats()['config']}")
