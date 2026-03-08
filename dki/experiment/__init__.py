"""Experiment tools and utilities for DKI system.

v7.0: 实验系统使用 DKIPlugin 替代 DKISystem
- SQLiteDataAdapter: IUserDataAdapter 的 SQLite 实现，供 DKIPlugin 使用
- ExperimentRunner: 已移除 DKISystem 依赖
"""

from dki.experiment.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    InjectionInfo,
    InjectionInfoViewer,
)
from dki.experiment.metrics import MetricsCalculator
from dki.experiment.data_generator import ExperimentDataGenerator
from dki.experiment.sqlite_adapter import SQLiteDataAdapter

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "InjectionInfo",
    "InjectionInfoViewer",
    "MetricsCalculator",
    "ExperimentDataGenerator",
    "SQLiteDataAdapter",
]
