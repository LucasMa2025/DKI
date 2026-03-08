"""
DKI Structured Exception Types (P0-1 方案A)

引入结构化异常类型，区分暂时性/永久性错误，
支持分级降级策略和精确的监控告警。

异常层次:
    DKIError (基类)
    ├── AdapterError (适配器层)
    │   ├── AdapterConnectionError (暂时性, 可重试)
    │   ├── AdapterSchemaError (永久性, 不可重试)
    │   └── AdapterTimeoutError (暂时性, 可重试)
    ├── KVComputeError (K/V 计算层)
    │   ├── KVOOMError (GPU OOM, 可恢复)
    │   └── KVSerializationError (序列化失败)
    ├── ModelError (模型推理层)
    │   ├── ModelOOMError (GPU OOM)
    │   ├── ModelConnectionError (远程模型连接失败)
    │   └── ModelTimeoutError (推理超时)
    ├── RecallError (召回层)
    │   ├── BM25InitError (BM25/jieba 初始化失败)
    │   └── VectorSearchError (向量检索失败)
    └── RateLimitError (限流)

设计原则:
1. 每个异常携带 error_code (用于 Prometheus label)
2. retryable 标记是否可重试 (用于上层决策)
3. 异常消息人类可读, error_code 机器可读

Author: AGI Demo Project
Version: 1.0.0
"""

from typing import Optional


class DKIError(Exception):
    """
    DKI 基础异常
    
    所有 DKI 异常的基类，携带结构化错误信息。
    
    Attributes:
        error_code: 机器可读的错误码 (用于 Prometheus label)
        retryable: 是否可重试
        cause: 原始异常 (可选)
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "DKI_UNKNOWN",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.cause = cause


# ============================================================
# Adapter 层异常
# ============================================================

class AdapterError(DKIError):
    """适配器层基础异常"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "ADAPTER_ERROR",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, retryable, cause)


class AdapterConnectionError(AdapterError):
    """
    适配器连接异常 (暂时性, 可重试)
    
    场景: 数据库连接超时、网络抖动、连接池耗尽
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="ADAPTER_CONNECTION",
            retryable=True,
            cause=cause,
        )


class AdapterSchemaError(AdapterError):
    """
    适配器 Schema 异常 (永久性, 不可重试)
    
    场景: 表不存在、字段映射错误、SQL 语法错误
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="ADAPTER_SCHEMA",
            retryable=False,
            cause=cause,
        )


class AdapterTimeoutError(AdapterError):
    """
    适配器超时异常 (暂时性, 可重试)
    
    场景: 数据库查询超时、连接池等待超时
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="ADAPTER_TIMEOUT",
            retryable=True,
            cause=cause,
        )


# ============================================================
# K/V 计算层异常
# ============================================================

class KVComputeError(DKIError):
    """K/V 计算层基础异常"""
    
    def __init__(
        self,
        message: str,
        cause_type: str = "unknown",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            error_code=f"KV_COMPUTE_{cause_type.upper()}",
            retryable=retryable,
            cause=cause,
        )


class KVOOMError(KVComputeError):
    """
    K/V 计算 GPU OOM (可恢复)
    
    场景: 偏好 K/V 计算时 GPU 显存不足
    恢复策略: 清理 GPU 缓存后降级到无注入推理
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            cause_type="oom",
            retryable=True,
            cause=cause,
        )


class KVSerializationError(KVComputeError):
    """
    K/V 序列化异常 (不可重试)
    
    场景: KV tensor 序列化/反序列化失败、缓存损坏
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            cause_type="serialization",
            retryable=False,
            cause=cause,
        )


# ============================================================
# 模型推理层异常
# ============================================================

class ModelError(DKIError):
    """模型推理层基础异常"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MODEL_ERROR",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, retryable, cause)


class ModelOOMError(ModelError):
    """
    模型推理 GPU OOM
    
    场景: 推理时 GPU 显存不足 (通常是上下文过长)
    恢复策略: 清理 GPU 缓存, 截断上下文后重试
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="MODEL_OOM",
            retryable=True,
            cause=cause,
        )


class ModelConnectionError(ModelError):
    """
    远程模型连接异常 (暂时性, 可重试)
    
    场景: vLLM/SGLang 服务不可达、API 超时
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="MODEL_CONNECTION",
            retryable=True,
            cause=cause,
        )


class ModelTimeoutError(ModelError):
    """
    模型推理超时
    
    场景: LLM 推理时间过长 (模型卡住或负载过高)
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="MODEL_TIMEOUT",
            retryable=True,
            cause=cause,
        )


# ============================================================
# 召回层异常
# ============================================================

class RecallError(DKIError):
    """召回层基础异常"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RECALL_ERROR",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, retryable, cause)


class BM25InitError(RecallError):
    """
    BM25/jieba 初始化失败 (降级到字符级分词)
    
    场景: jieba 词典加载失败、rank_bm25 不可用
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="RECALL_BM25_INIT",
            retryable=False,
            cause=cause,
        )


class VectorSearchError(RecallError):
    """
    向量检索失败
    
    场景: FAISS 索引损坏、Embedding 服务不可达
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(
            message,
            error_code="RECALL_VECTOR",
            retryable=True,
            cause=cause,
        )


# ============================================================
# 限流异常
# ============================================================

class RateLimitError(DKIError):
    """
    限流异常
    
    场景: 用户请求频率超限、并发数超限
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        user_id: str = "",
        retry_after: float = 0,
    ):
        super().__init__(
            message,
            error_code="RATE_LIMIT",
            retryable=True,
        )
        self.user_id = user_id
        self.retry_after = retry_after


class CircuitBreakerOpenError(DKIError):
    """
    熔断器开启异常
    
    场景: 适配器连续失败次数超过阈值, 熔断器开启
    """
    
    def __init__(
        self,
        message: str = "Circuit breaker is open",
        recovery_timeout: float = 0,
    ):
        super().__init__(
            message,
            error_code="CIRCUIT_BREAKER_OPEN",
            retryable=True,
        )
        self.recovery_timeout = recovery_timeout
