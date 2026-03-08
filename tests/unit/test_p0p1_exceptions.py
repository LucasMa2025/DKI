"""
Unit Tests for DKI Structured Exceptions (P0-1 方案A)

测试异常层次结构、属性、分级降级策略。

Author: AGI Demo Project
"""

import pytest

from dki.core.exceptions import (
    DKIError,
    AdapterError,
    AdapterConnectionError,
    AdapterSchemaError,
    AdapterTimeoutError,
    KVComputeError,
    KVOOMError,
    KVSerializationError,
    ModelError,
    ModelOOMError,
    ModelConnectionError,
    ModelTimeoutError,
    RecallError,
    BM25InitError,
    VectorSearchError,
    RateLimitError,
    CircuitBreakerOpenError,
)


# ============================================================
# 1. 异常层次结构
# ============================================================

class TestExceptionHierarchy:
    """测试异常继承关系"""

    def test_all_inherit_from_dki_error(self):
        """所有 DKI 异常都继承自 DKIError"""
        exceptions = [
            AdapterError("test"),
            AdapterConnectionError("test"),
            AdapterSchemaError("test"),
            AdapterTimeoutError("test"),
            KVComputeError("test"),
            KVOOMError("test"),
            KVSerializationError("test"),
            ModelError("test"),
            ModelOOMError("test"),
            ModelConnectionError("test"),
            ModelTimeoutError("test"),
            RecallError("test"),
            BM25InitError("test"),
            VectorSearchError("test"),
            RateLimitError("test"),
            CircuitBreakerOpenError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, DKIError), f"{type(exc).__name__} should inherit DKIError"
            assert isinstance(exc, Exception)

    def test_adapter_hierarchy(self):
        """Adapter 异常继承 AdapterError"""
        assert isinstance(AdapterConnectionError("x"), AdapterError)
        assert isinstance(AdapterSchemaError("x"), AdapterError)
        assert isinstance(AdapterTimeoutError("x"), AdapterError)

    def test_kv_hierarchy(self):
        """KV 异常继承 KVComputeError"""
        assert isinstance(KVOOMError("x"), KVComputeError)
        assert isinstance(KVSerializationError("x"), KVComputeError)

    def test_model_hierarchy(self):
        """Model 异常继承 ModelError"""
        assert isinstance(ModelOOMError("x"), ModelError)
        assert isinstance(ModelConnectionError("x"), ModelError)
        assert isinstance(ModelTimeoutError("x"), ModelError)

    def test_recall_hierarchy(self):
        """Recall 异常继承 RecallError"""
        assert isinstance(BM25InitError("x"), RecallError)
        assert isinstance(VectorSearchError("x"), RecallError)


# ============================================================
# 2. 异常属性
# ============================================================

class TestExceptionAttributes:
    """测试异常的 error_code 和 retryable 属性"""

    def test_dki_error_defaults(self):
        """DKIError 默认属性"""
        e = DKIError("test error")
        assert e.error_code == "DKI_UNKNOWN"
        assert e.retryable is False
        assert e.cause is None
        assert str(e) == "test error"

    def test_dki_error_custom(self):
        """DKIError 自定义属性"""
        cause = ValueError("original")
        e = DKIError("msg", error_code="CUSTOM", retryable=True, cause=cause)
        assert e.error_code == "CUSTOM"
        assert e.retryable is True
        assert e.cause is cause

    # --- Adapter ---
    def test_adapter_connection_error(self):
        e = AdapterConnectionError("connection lost")
        assert e.error_code == "ADAPTER_CONNECTION"
        assert e.retryable is True

    def test_adapter_schema_error(self):
        e = AdapterSchemaError("table not found")
        assert e.error_code == "ADAPTER_SCHEMA"
        assert e.retryable is False

    def test_adapter_timeout_error(self):
        e = AdapterTimeoutError("query timeout")
        assert e.error_code == "ADAPTER_TIMEOUT"
        assert e.retryable is True

    # --- KV ---
    def test_kv_oom_error(self):
        e = KVOOMError("GPU OOM during KV compute")
        assert e.error_code == "KV_COMPUTE_OOM"
        assert e.retryable is True

    def test_kv_serialization_error(self):
        e = KVSerializationError("corrupt cache")
        assert e.error_code == "KV_COMPUTE_SERIALIZATION"
        assert e.retryable is False

    # --- Model ---
    def test_model_oom_error(self):
        e = ModelOOMError("inference OOM")
        assert e.error_code == "MODEL_OOM"
        assert e.retryable is True

    def test_model_connection_error(self):
        e = ModelConnectionError("vLLM unreachable")
        assert e.error_code == "MODEL_CONNECTION"
        assert e.retryable is True

    def test_model_timeout_error(self):
        e = ModelTimeoutError("inference timeout")
        assert e.error_code == "MODEL_TIMEOUT"
        assert e.retryable is True

    # --- Recall ---
    def test_bm25_init_error(self):
        e = BM25InitError("jieba dict not found")
        assert e.error_code == "RECALL_BM25_INIT"
        assert e.retryable is False

    def test_vector_search_error(self):
        e = VectorSearchError("FAISS index corrupt")
        assert e.error_code == "RECALL_VECTOR"
        assert e.retryable is True

    # --- Rate Limit ---
    def test_rate_limit_error(self):
        e = RateLimitError(user_id="u1", retry_after=2.0)
        assert e.error_code == "RATE_LIMIT"
        assert e.retryable is True
        assert e.user_id == "u1"
        assert e.retry_after == 2.0

    def test_circuit_breaker_open_error(self):
        e = CircuitBreakerOpenError(recovery_timeout=30)
        assert e.error_code == "CIRCUIT_BREAKER_OPEN"
        assert e.retryable is True
        assert e.recovery_timeout == 30


# ============================================================
# 3. 异常可被 except 正确捕获
# ============================================================

class TestExceptionCatching:
    """测试异常可被正确的 except 块捕获"""

    def test_catch_adapter_connection_as_dki_error(self):
        with pytest.raises(DKIError):
            raise AdapterConnectionError("test")

    def test_catch_adapter_connection_as_adapter_error(self):
        with pytest.raises(AdapterError):
            raise AdapterConnectionError("test")

    def test_catch_kv_oom_as_kv_compute_error(self):
        with pytest.raises(KVComputeError):
            raise KVOOMError("test")

    def test_catch_model_oom_as_model_error(self):
        with pytest.raises(ModelError):
            raise ModelOOMError("test")

    def test_catch_bm25_as_recall_error(self):
        with pytest.raises(RecallError):
            raise BM25InitError("test")

    def test_retryable_classification(self):
        """验证可重试/不可重试分类"""
        retryable = [
            AdapterConnectionError("x"),
            AdapterTimeoutError("x"),
            KVOOMError("x"),
            ModelOOMError("x"),
            ModelConnectionError("x"),
            ModelTimeoutError("x"),
            VectorSearchError("x"),
            RateLimitError("x"),
            CircuitBreakerOpenError("x"),
        ]
        non_retryable = [
            AdapterSchemaError("x"),
            KVSerializationError("x"),
            BM25InitError("x"),
        ]

        for exc in retryable:
            assert exc.retryable is True, f"{type(exc).__name__} should be retryable"
        for exc in non_retryable:
            assert exc.retryable is False, f"{type(exc).__name__} should NOT be retryable"


# ============================================================
# 4. cause 链传播
# ============================================================

class TestExceptionCauseChain:
    """测试异常原因链"""

    def test_cause_preserved(self):
        original = ConnectionError("TCP reset")
        e = AdapterConnectionError("adapter failed", cause=original)
        assert e.cause is original
        assert str(e.cause) == "TCP reset"

    def test_cause_none_by_default(self):
        e = AdapterSchemaError("schema mismatch")
        assert e.cause is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
