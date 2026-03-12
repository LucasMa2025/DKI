"""Experiment tools and utilities for DKI system.

v9.1 重构: 独立持久化层
- 使用 dki.experiment.store (从 demo.store 复制, 仅 SQLite)
- 使用独立的 dki.db 数据库 (不与 demo.db 共享)
- 使用 dki.integration.create_plugin 标准集成 DKIPlugin
- 通过 ConfigDrivenAdapter 映射 demo_* 表

架构对比:
  旧: ExperimentRunner → SQLiteDataAdapter → DKIPlugin (hack 注入)
  新: ExperimentRunner → dki.experiment.store.SQLiteChatStore (独立 dki.db)
                       → create_plugin(adapter_config) → DKIPlugin
      (与 demo 应用架构一致, 但使用独立数据库)
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
from dki.experiment.dki_bridge import build_experiment_adapter_config

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "InjectionInfo",
    "InjectionInfoViewer",
    "MetricsCalculator",
    "ExperimentDataGenerator",
    "build_experiment_adapter_config",
]
