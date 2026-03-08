"""
DKI Rate Limiter & Circuit Breaker (P1-2)

提供用户级限流和适配器熔断能力:
1. UserRateLimiter: 令牌桶 + 并发控制
2. CircuitBreaker: 三态熔断器 (CLOSED → OPEN → HALF_OPEN)

配置外置: 所有参数从 config.yaml 的 rate_limit 节读取,
不修改 dki_system.py。

Author: AGI Demo Project
Version: 1.0.0
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.core.exceptions import RateLimitError, CircuitBreakerOpenError


# ============================================================
# Rate Limiter Configuration
# ============================================================

class RateLimitConfig:
    """
    限流配置 (外置)
    
    从 config.yaml 的 rate_limit 节加载:
    ```yaml
    rate_limit:
      enabled: true
      max_rpm: 30
      max_concurrent: 3
      burst_size: 5
    ```
    """
    
    def __init__(
        self,
        enabled: bool = True,
        max_rpm: int = 30,
        max_concurrent: int = 3,
        burst_size: int = 5,
    ):
        self.enabled = enabled
        self.max_rpm = max_rpm
        self.max_concurrent = max_concurrent
        self.burst_size = burst_size
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitConfig":
        return cls(
            enabled=data.get("enabled", True),
            max_rpm=data.get("max_rpm", 30),
            max_concurrent=data.get("max_concurrent", 3),
            burst_size=data.get("burst_size", 5),
        )


class CircuitBreakerConfig:
    """
    熔断器配置 (外置)
    
    从 config.yaml 的 circuit_breaker 节加载:
    ```yaml
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      recovery_timeout: 30
      half_open_max_calls: 1
    ```
    """
    
    def __init__(
        self,
        enabled: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
    ):
        self.enabled = enabled
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitBreakerConfig":
        return cls(
            enabled=data.get("enabled", True),
            failure_threshold=data.get("failure_threshold", 5),
            recovery_timeout=data.get("recovery_timeout", 30),
            half_open_max_calls=data.get("half_open_max_calls", 1),
        )


# ============================================================
# User Rate Limiter
# ============================================================

class UserRateLimiter:
    """
    用户级限流器 (令牌桶 + 并发控制)
    
    特性:
    - RPM (Requests Per Minute) 限制
    - 并发请求数限制
    - 突发容量 (burst) 支持
    - 线程安全 (asyncio 单线程模型)
    
    使用方式:
    ```python
    limiter = UserRateLimiter(config)
    
    # 检查是否允许请求
    limiter.check_or_raise(user_id)
    
    # 请求开始
    limiter.acquire(user_id)
    try:
        # ... 处理请求 ...
    finally:
        limiter.release(user_id)
    ```
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self._config = config or RateLimitConfig()
        self._request_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._concurrent: Dict[str, int] = defaultdict(int)
        # 统计
        self._total_checked = 0
        self._total_rejected = 0
    
    @property
    def enabled(self) -> bool:
        return self._config.enabled
    
    def check(self, user_id: str) -> bool:
        """
        检查用户是否可以发送请求
        
        Returns:
            True 如果允许, False 如果被限流
        """
        if not self._config.enabled:
            return True
        
        now = time.time()
        self._total_checked += 1
        
        # 清理 1 分钟前的记录
        timestamps = self._request_timestamps[user_id]
        self._request_timestamps[user_id] = [
            t for t in timestamps if now - t < 60
        ]
        
        # RPM 检查
        if len(self._request_timestamps[user_id]) >= self._config.max_rpm:
            self._total_rejected += 1
            return False
        
        # 并发检查
        if self._concurrent[user_id] >= self._config.max_concurrent:
            self._total_rejected += 1
            return False
        
        return True
    
    def check_or_raise(self, user_id: str) -> None:
        """检查限流, 超限则抛出 RateLimitError"""
        if not self.check(user_id):
            raise RateLimitError(
                message=f"Rate limit exceeded for user {user_id}",
                user_id=user_id,
                retry_after=60.0 / max(self._config.max_rpm, 1),
            )
    
    def acquire(self, user_id: str) -> None:
        """记录请求开始 (RPM + 并发)"""
        now = time.time()
        self._request_timestamps[user_id].append(now)
        self._concurrent[user_id] += 1
    
    def release(self, user_id: str) -> None:
        """记录请求结束 (释放并发槽)"""
        if self._concurrent[user_id] > 0:
            self._concurrent[user_id] -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取限流统计"""
        return {
            "enabled": self._config.enabled,
            "max_rpm": self._config.max_rpm,
            "max_concurrent": self._config.max_concurrent,
            "total_checked": self._total_checked,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                self._total_rejected / max(self._total_checked, 1)
            ),
            "active_users": len(
                [u for u, c in self._concurrent.items() if c > 0]
            ),
        }
    
    def update_config(self, config: RateLimitConfig) -> None:
        """运行时更新配置 (支持热加载)"""
        self._config = config
        logger.info(
            f"Rate limiter config updated: "
            f"max_rpm={config.max_rpm}, "
            f"max_concurrent={config.max_concurrent}"
        )


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitBreaker:
    """
    三态熔断器: CLOSED → OPEN → HALF_OPEN
    
    状态转换:
    - CLOSED: 正常状态, 所有请求通过
    - OPEN: 熔断状态, 所有请求被拒绝
    - HALF_OPEN: 半开状态, 允许少量探测请求
    
    转换条件:
    - CLOSED → OPEN: 连续失败次数 >= failure_threshold
    - OPEN → HALF_OPEN: 距离上次失败 >= recovery_timeout 秒
    - HALF_OPEN → CLOSED: 探测请求成功
    - HALF_OPEN → OPEN: 探测请求失败
    
    使用方式:
    ```python
    breaker = CircuitBreaker(config)
    
    if not breaker.allow_request():
        raise CircuitBreakerOpenError()
    
    try:
        result = await adapter.query(...)
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise
    ```
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self._config = config or CircuitBreakerConfig()
        self._state = "CLOSED"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        # 统计
        self._total_success = 0
        self._total_failure = 0
        self._total_rejected = 0
        self._state_changes: List[Dict[str, Any]] = []
    
    @property
    def enabled(self) -> bool:
        return self._config.enabled
    
    @property
    def state(self) -> str:
        return self._state
    
    def allow_request(self) -> bool:
        """
        检查是否允许请求通过
        
        Returns:
            True 如果允许, False 如果被熔断
        """
        if not self._config.enabled:
            return True
        
        if self._state == "CLOSED":
            return True
        
        if self._state == "OPEN":
            # 检查是否可以转为 HALF_OPEN
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._config.recovery_timeout:
                self._transition_to("HALF_OPEN")
                return True
            self._total_rejected += 1
            return False
        
        # HALF_OPEN: 允许有限的探测请求
        if self._half_open_calls < self._config.half_open_max_calls:
            self._half_open_calls += 1
            return True
        
        self._total_rejected += 1
        return False
    
    def allow_or_raise(self) -> None:
        """检查熔断, 熔断则抛出 CircuitBreakerOpenError"""
        if not self.allow_request():
            raise CircuitBreakerOpenError(
                message=f"Circuit breaker is {self._state}",
                recovery_timeout=self._config.recovery_timeout,
            )
    
    def record_success(self) -> None:
        """记录成功"""
        self._total_success += 1
        self._failure_count = 0
        
        if self._state == "HALF_OPEN":
            self._transition_to("CLOSED")
    
    def record_failure(self) -> None:
        """记录失败"""
        self._total_failure += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "HALF_OPEN":
            self._transition_to("OPEN")
        elif (
            self._state == "CLOSED"
            and self._failure_count >= self._config.failure_threshold
        ):
            self._transition_to("OPEN")
    
    def _transition_to(self, new_state: str) -> None:
        """状态转换"""
        old_state = self._state
        self._state = new_state
        
        if new_state == "HALF_OPEN":
            self._half_open_calls = 0
        elif new_state == "CLOSED":
            self._failure_count = 0
        
        change = {
            "from": old_state,
            "to": new_state,
            "timestamp": time.time(),
            "failure_count": self._failure_count,
        }
        self._state_changes.append(change)
        
        # 保留最近 100 条状态变更
        if len(self._state_changes) > 100:
            self._state_changes = self._state_changes[-100:]
        
        logger.warning(
            f"Circuit breaker state: {old_state} → {new_state} "
            f"(failures={self._failure_count})"
        )
    
    def reset(self) -> None:
        """手动重置熔断器"""
        self._transition_to("CLOSED")
        self._failure_count = 0
        logger.info("Circuit breaker manually reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取熔断器统计"""
        return {
            "enabled": self._config.enabled,
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self._config.failure_threshold,
            "recovery_timeout": self._config.recovery_timeout,
            "total_success": self._total_success,
            "total_failure": self._total_failure,
            "total_rejected": self._total_rejected,
            "recent_state_changes": self._state_changes[-5:],
        }
    
    def update_config(self, config: CircuitBreakerConfig) -> None:
        """运行时更新配置 (支持热加载)"""
        self._config = config
        logger.info(
            f"Circuit breaker config updated: "
            f"threshold={config.failure_threshold}, "
            f"timeout={config.recovery_timeout}s"
        )
