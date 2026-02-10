"""
Attention Profiler
注意力计算性能分析器

用于监控和调试 FlashAttention 性能

Author: AGI Demo Project
Version: 1.0.0
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from dki.attention.config import ProfilingConfig


@dataclass
class ProfileRecord:
    """性能记录"""
    operation: str
    latency_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    input_shape: str
    output_shape: str
    backend: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionProfiler:
    """
    注意力计算性能分析器
    
    功能:
    1. 记录每次注意力计算的延迟
    2. 监控 GPU 内存使用
    3. 生成性能报告
    4. 支持导出到文件
    
    Example:
        profiler = AttentionProfiler()
        
        with profiler.profile("kv_injection"):
            result = optimizer.inject(...)
        
        # 获取报告
        report = profiler.get_report()
    """
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        """
        初始化分析器
        
        Args:
            config: 性能监控配置
        """
        self.config = config or ProfilingConfig()
        self._records: List[ProfileRecord] = []
        self._enabled = self.config.enabled
        
        logger.info(f"AttentionProfiler initialized (enabled={self._enabled})")
    
    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """设置启用状态"""
        self._enabled = value
    
    @contextmanager
    def profile(
        self,
        operation: str,
        backend: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        性能分析上下文管理器
        
        Args:
            operation: 操作名称
            backend: 后端类型
            metadata: 额外元数据
            
        Example:
            with profiler.profile("kv_injection", backend="fa3"):
                result = optimizer.inject(...)
        """
        if not self._enabled:
            yield
            return
        
        # 记录开始状态
        start_time = time.perf_counter()
        start_memory = self._get_memory_stats() if self.config.log_memory else {}
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.perf_counter()
            end_memory = self._get_memory_stats() if self.config.log_memory else {}
            
            # 创建记录
            record = ProfileRecord(
                operation=operation,
                latency_ms=(end_time - start_time) * 1000,
                memory_allocated_mb=end_memory.get("allocated_mb", 0),
                memory_reserved_mb=end_memory.get("reserved_mb", 0),
                input_shape="",
                output_shape="",
                backend=backend,
                metadata=metadata or {},
            )
            
            self._records.append(record)
            
            # 日志输出
            if self.config.log_latency:
                logger.debug(
                    f"[Profiler] {operation}: {record.latency_ms:.2f}ms "
                    f"(memory: {record.memory_allocated_mb:.1f}MB)"
                )
    
    def record(
        self,
        operation: str,
        latency_ms: float,
        input_tensor: Optional[torch.Tensor] = None,
        output_tensor: Optional[torch.Tensor] = None,
        backend: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        手动记录性能数据
        
        Args:
            operation: 操作名称
            latency_ms: 延迟 (毫秒)
            input_tensor: 输入张量 (用于记录形状)
            output_tensor: 输出张量
            backend: 后端类型
            metadata: 额外元数据
        """
        if not self._enabled:
            return
        
        memory_stats = self._get_memory_stats() if self.config.log_memory else {}
        
        record = ProfileRecord(
            operation=operation,
            latency_ms=latency_ms,
            memory_allocated_mb=memory_stats.get("allocated_mb", 0),
            memory_reserved_mb=memory_stats.get("reserved_mb", 0),
            input_shape=str(input_tensor.shape) if input_tensor is not None else "",
            output_shape=str(output_tensor.shape) if output_tensor is not None else "",
            backend=backend,
            metadata=metadata or {},
        )
        
        self._records.append(record)
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """获取 GPU 内存统计"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
    
    def get_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        Returns:
            包含统计信息的字典
        """
        if not self._records:
            return {"message": "No records available"}
        
        # 按操作分组
        operations = {}
        for record in self._records:
            op = record.operation
            if op not in operations:
                operations[op] = {
                    "count": 0,
                    "total_latency_ms": 0,
                    "min_latency_ms": float("inf"),
                    "max_latency_ms": 0,
                    "latencies": [],
                }
            
            operations[op]["count"] += 1
            operations[op]["total_latency_ms"] += record.latency_ms
            operations[op]["min_latency_ms"] = min(
                operations[op]["min_latency_ms"], record.latency_ms
            )
            operations[op]["max_latency_ms"] = max(
                operations[op]["max_latency_ms"], record.latency_ms
            )
            operations[op]["latencies"].append(record.latency_ms)
        
        # 计算统计
        for op, stats in operations.items():
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["count"]
            
            # 计算 P50, P90, P99
            latencies = sorted(stats["latencies"])
            n = len(latencies)
            stats["p50_latency_ms"] = latencies[int(n * 0.5)]
            stats["p90_latency_ms"] = latencies[int(n * 0.9)]
            stats["p99_latency_ms"] = latencies[min(int(n * 0.99), n - 1)]
            
            # 移除原始数据
            del stats["latencies"]
        
        # 后端统计
        backend_stats = {}
        for record in self._records:
            backend = record.backend
            if backend not in backend_stats:
                backend_stats[backend] = 0
            backend_stats[backend] += 1
        
        # 内存统计
        memory_stats = {}
        if self.config.log_memory:
            memory_records = [r for r in self._records if r.memory_allocated_mb > 0]
            if memory_records:
                memory_stats = {
                    "avg_allocated_mb": sum(r.memory_allocated_mb for r in memory_records) / len(memory_records),
                    "max_allocated_mb": max(r.memory_allocated_mb for r in memory_records),
                    "avg_reserved_mb": sum(r.memory_reserved_mb for r in memory_records) / len(memory_records),
                }
        
        return {
            "total_records": len(self._records),
            "time_range": {
                "start": self._records[0].timestamp.isoformat(),
                "end": self._records[-1].timestamp.isoformat(),
            },
            "operations": operations,
            "backend_usage": backend_stats,
            "memory": memory_stats,
        }
    
    def get_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        return [
            {
                "operation": r.operation,
                "latency_ms": r.latency_ms,
                "memory_allocated_mb": r.memory_allocated_mb,
                "memory_reserved_mb": r.memory_reserved_mb,
                "input_shape": r.input_shape,
                "output_shape": r.output_shape,
                "backend": r.backend,
                "timestamp": r.timestamp.isoformat(),
                "metadata": r.metadata,
            }
            for r in self._records
        ]
    
    def export_to_file(self, filepath: Optional[str] = None):
        """
        导出记录到文件
        
        Args:
            filepath: 文件路径 (默认使用配置中的路径)
        """
        import json
        
        filepath = filepath or self.config.log_path
        if not filepath:
            logger.warning("No filepath specified for export")
            return
        
        data = {
            "report": self.get_report(),
            "records": self.get_records(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Profiler data exported to {filepath}")
    
    def clear(self):
        """清除所有记录"""
        self._records.clear()
    
    def __len__(self) -> int:
        """返回记录数量"""
        return len(self._records)
