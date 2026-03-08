"""
Unit Tests for Rate Limiter & Circuit Breaker (P1-2)

测试令牌桶限流、并发控制、三态熔断器。

Author: AGI Demo Project
"""

import time
import pytest

from dki.core.rate_limiter import (
    RateLimitConfig,
    CircuitBreakerConfig,
    UserRateLimiter,
    CircuitBreaker,
)
from dki.core.exceptions import RateLimitError, CircuitBreakerOpenError


# ============================================================
# RateLimitConfig
# ============================================================

class TestRateLimitConfig:
    """测试限流配置"""

    def test_defaults(self):
        cfg = RateLimitConfig()
        assert cfg.enabled is True
        assert cfg.max_rpm == 30
        assert cfg.max_concurrent == 3
        assert cfg.burst_size == 5

    def test_from_dict(self):
        cfg = RateLimitConfig.from_dict({
            "enabled": False,
            "max_rpm": 60,
            "max_concurrent": 10,
            "burst_size": 20,
        })
        assert cfg.enabled is False
        assert cfg.max_rpm == 60
        assert cfg.max_concurrent == 10
        assert cfg.burst_size == 20

    def test_from_dict_partial(self):
        """部分字段使用默认值"""
        cfg = RateLimitConfig.from_dict({"max_rpm": 100})
        assert cfg.enabled is True  # default
        assert cfg.max_rpm == 100
        assert cfg.max_concurrent == 3  # default


# ============================================================
# CircuitBreakerConfig
# ============================================================

class TestCircuitBreakerConfig:
    """测试熔断器配置"""

    def test_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.enabled is True
        assert cfg.failure_threshold == 5
        assert cfg.recovery_timeout == 30
        assert cfg.half_open_max_calls == 1

    def test_from_dict(self):
        cfg = CircuitBreakerConfig.from_dict({
            "enabled": False,
            "failure_threshold": 10,
            "recovery_timeout": 60,
            "half_open_max_calls": 3,
        })
        assert cfg.enabled is False
        assert cfg.failure_threshold == 10
        assert cfg.recovery_timeout == 60
        assert cfg.half_open_max_calls == 3


# ============================================================
# UserRateLimiter
# ============================================================

class TestUserRateLimiter:
    """测试用户级限流器"""

    def test_disabled_always_allows(self):
        """禁用时所有请求通过"""
        limiter = UserRateLimiter(RateLimitConfig(enabled=False))
        for _ in range(100):
            assert limiter.check("user1") is True

    def test_rpm_limit(self):
        """RPM 限制生效"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=3, max_concurrent=100))
        # 前 3 个请求通过
        for _ in range(3):
            assert limiter.check("user1") is True
            limiter.acquire("user1")
            limiter.release("user1")
        # 第 4 个被拒绝
        assert limiter.check("user1") is False

    def test_concurrent_limit(self):
        """并发限制生效"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=100, max_concurrent=2))
        # 占用 2 个并发槽
        limiter.acquire("user1")
        limiter.acquire("user1")
        # 第 3 个被拒绝
        assert limiter.check("user1") is False
        # 释放 1 个后可以通过
        limiter.release("user1")
        assert limiter.check("user1") is True

    def test_check_or_raise(self):
        """check_or_raise 超限抛出 RateLimitError"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=1, max_concurrent=100))
        limiter.acquire("u1")
        limiter.release("u1")
        # 第 2 个请求超限
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_or_raise("u1")
        assert exc_info.value.user_id == "u1"
        assert exc_info.value.retry_after > 0

    def test_per_user_isolation(self):
        """不同用户独立限流"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=2, max_concurrent=100))
        limiter.acquire("user_a")
        limiter.acquire("user_a")
        # user_a 超限
        assert limiter.check("user_a") is False
        # user_b 不受影响
        assert limiter.check("user_b") is True

    def test_stats(self):
        """统计数据正确"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=1, max_concurrent=100))
        limiter.check("u1")
        limiter.acquire("u1")
        limiter.check("u1")  # rejected

        stats = limiter.get_stats()
        assert stats["total_checked"] == 2
        assert stats["total_rejected"] == 1
        assert stats["active_users"] == 1

    def test_update_config(self):
        """运行时更新配置"""
        limiter = UserRateLimiter(RateLimitConfig(max_rpm=1))
        limiter.acquire("u1")
        assert limiter.check("u1") is False

        # 放宽限制
        limiter.update_config(RateLimitConfig(max_rpm=100))
        assert limiter.check("u1") is True

    def test_release_no_negative(self):
        """release 不会导致负数并发"""
        limiter = UserRateLimiter()
        limiter.release("u1")  # 没有 acquire 就 release
        assert limiter._concurrent["u1"] == 0


# ============================================================
# CircuitBreaker
# ============================================================

class TestCircuitBreaker:
    """测试三态熔断器"""

    def test_initial_state_closed(self):
        """初始状态为 CLOSED"""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.allow_request() is True

    def test_disabled_always_allows(self):
        """禁用时所有请求通过"""
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        for _ in range(100):
            cb.record_failure()
        assert cb.allow_request() is True

    def test_closed_to_open(self):
        """连续失败达到阈值 → OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()  # 第 3 次
        assert cb.state == "OPEN"

    def test_open_rejects_requests(self):
        """OPEN 状态拒绝请求"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=9999,  # 不会自动恢复
        ))
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.allow_request() is False

    def test_open_to_half_open_after_timeout(self):
        """超过 recovery_timeout 后 → HALF_OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0,  # 立即恢复
        ))
        cb.record_failure()
        assert cb.state == "OPEN"
        # recovery_timeout=0, 立即转为 HALF_OPEN
        assert cb.allow_request() is True
        assert cb.state == "HALF_OPEN"

    def test_half_open_success_to_closed(self):
        """HALF_OPEN 成功 → CLOSED"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0,
        ))
        cb.record_failure()
        cb.allow_request()  # → HALF_OPEN
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_half_open_failure_to_open(self):
        """HALF_OPEN 失败 → OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0,
        ))
        cb.record_failure()
        cb.allow_request()  # → HALF_OPEN
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_half_open_limited_calls(self):
        """HALF_OPEN 只允许有限的探测请求"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0,
            half_open_max_calls=1,
        ))
        cb.record_failure()
        # 第 1 次 allow_request: OPEN → HALF_OPEN (transition, _half_open_calls=0)
        assert cb.allow_request() is True
        assert cb.state == "HALF_OPEN"
        # 第 2 次 allow_request: HALF_OPEN, _half_open_calls 0→1 (允许)
        assert cb.allow_request() is True
        # 第 3 次 allow_request: HALF_OPEN, _half_open_calls=1 >= max=1 (拒绝)
        assert cb.allow_request() is False

    def test_allow_or_raise(self):
        """allow_or_raise 熔断时抛出 CircuitBreakerOpenError"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=9999,
        ))
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.allow_or_raise()
        assert exc_info.value.recovery_timeout == 9999

    def test_success_resets_failure_count(self):
        """成功重置失败计数"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # 重置
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"  # 还没到 3

    def test_manual_reset(self):
        """手动重置熔断器"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == "OPEN"
        cb.reset()
        assert cb.state == "CLOSED"

    def test_stats(self):
        """统计数据正确"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        cb.record_success()
        cb.record_failure()
        cb.record_failure()  # → OPEN

        stats = cb.get_stats()
        assert stats["state"] == "OPEN"
        assert stats["total_success"] == 1
        assert stats["total_failure"] == 2
        assert stats["failure_count"] == 2

    def test_state_changes_recorded(self):
        """状态变更被记录"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0,
        ))
        cb.record_failure()  # CLOSED → OPEN
        cb.allow_request()   # OPEN → HALF_OPEN
        cb.record_success()  # HALF_OPEN → CLOSED

        stats = cb.get_stats()
        changes = stats["recent_state_changes"]
        assert len(changes) == 3
        assert changes[0]["from"] == "CLOSED"
        assert changes[0]["to"] == "OPEN"
        assert changes[1]["to"] == "HALF_OPEN"
        assert changes[2]["to"] == "CLOSED"

    def test_update_config(self):
        """运行时更新配置"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == "OPEN"
        cb.reset()
        # 提高阈值
        cb.update_config(CircuitBreakerConfig(failure_threshold=10))
        cb.record_failure()
        assert cb.state == "CLOSED"  # 阈值提高了


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
