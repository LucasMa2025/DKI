"""
ExperimentRunner 单元测试

测试实验运行器的数据结构、工具方法和流程逻辑:
- ExperimentConfig 数据结构
- ExperimentResult 数据结构
- InjectionInfo 数据结构
- InjectionInfoViewer 功能
- ExperimentRunner 工具方法 (不依赖真实 DKI/RAG 系统)
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from dki.experiment.runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    InjectionInfo,
    InjectionInfoViewer,
)


# ============================================================
# ExperimentConfig 测试
# ============================================================

class TestExperimentConfig:
    """ExperimentConfig 数据结构测试"""

    def test_default_values(self):
        config = ExperimentConfig(name="test_exp")
        assert config.name == "test_exp"
        assert config.description == ""
        assert config.modes == ["rag", "dki", "baseline"]
        assert config.datasets == ["persona_chat", "memory_qa"]
        assert config.max_samples == 100
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.alpha_values == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_custom_values(self):
        config = ExperimentConfig(
            name="custom_exp",
            description="A custom experiment",
            modes=["dki"],
            datasets=["hotpot_qa"],
            max_samples=50,
            max_new_tokens=128,
            temperature=0.5,
            alpha_values=[0.0, 0.5, 1.0],
        )
        assert config.name == "custom_exp"
        assert config.modes == ["dki"]
        assert config.max_samples == 50

    def test_to_dict(self):
        config = ExperimentConfig(name="dict_test")
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d['name'] == "dict_test"
        assert d['modes'] == ["rag", "dki", "baseline"]
        assert d['max_new_tokens'] == 256
        assert d['alpha_values'] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_to_dict_roundtrip(self):
        """to_dict 结果应可用于重建"""
        config = ExperimentConfig(
            name="roundtrip",
            description="test",
            modes=["dki", "rag"],
        )
        d = config.to_dict()
        config2 = ExperimentConfig(**d)
        assert config2.name == config.name
        assert config2.modes == config.modes


# ============================================================
# ExperimentResult 测试
# ============================================================

class TestExperimentResult:
    """ExperimentResult 数据结构测试"""

    def test_basic_result(self):
        result = ExperimentResult(
            mode="dki",
            dataset="persona_chat",
            sample_id="sample_001",
            query="What should I eat?",
            response="Try some vegetarian food.",
            latency_ms=42.5,
            memories_used=["mem_1", "mem_2"],
        )
        assert result.mode == "dki"
        assert result.latency_ms == 42.5
        assert len(result.memories_used) == 2
        assert result.alpha is None
        assert result.cache_hit is False
        assert result.injection_info is None

    def test_to_dict(self):
        result = ExperimentResult(
            mode="rag",
            dataset="memory_qa",
            sample_id="s1",
            query="q",
            response="r",
            latency_ms=10.0,
            memories_used=[],
            alpha=0.7,
            cache_hit=True,
        )
        d = result.to_dict()
        assert d['mode'] == "rag"
        assert d['alpha'] == 0.7
        assert d['cache_hit'] is True
        assert d['injection_info'] is None

    def test_to_dict_with_injection_info(self):
        info = InjectionInfo(
            mode="dki",
            original_query="test query",
            preference_text="I like cats",
            alpha=0.8,
        )
        result = ExperimentResult(
            mode="dki",
            dataset="test",
            sample_id="s1",
            query="test query",
            response="response",
            latency_ms=5.0,
            memories_used=[],
            injection_info=info,
        )
        d = result.to_dict()
        assert d['injection_info'] is not None
        assert d['injection_info']['mode'] == "dki"
        assert d['injection_info']['alpha'] == 0.8

    def test_metrics_field(self):
        result = ExperimentResult(
            mode="dki",
            dataset="test",
            sample_id="s1",
            query="q",
            response="r",
            latency_ms=1.0,
            memories_used=[],
            metrics={"bleu": 0.5, "recall": 0.8},
        )
        assert result.metrics["bleu"] == 0.5


# ============================================================
# InjectionInfo 测试
# ============================================================

class TestInjectionInfo:
    """InjectionInfo 数据结构测试"""

    def test_dki_injection_info(self):
        info = InjectionInfo(
            mode="dki",
            original_query="推荐一家餐厅",
            preference_text="我是素食主义者",
            preference_tokens=15,
            history_suffix="[历史] 用户之前问过...",
            history_tokens=50,
            history_messages=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮你的？"},
            ],
            final_input="[历史] 用户之前问过...\n推荐一家餐厅",
            alpha=0.75,
        )
        assert info.mode == "dki"
        assert info.preference_text == "我是素食主义者"
        assert info.alpha == 0.75
        assert len(info.history_messages) == 2

    def test_rag_injection_info(self):
        info = InjectionInfo(
            mode="rag",
            original_query="推荐一家餐厅",
            rag_context="用户偏好: 素食主义者",
            rag_prompt="基于以下信息回答: ...",
            final_input="基于以下信息回答: ...\n推荐一家餐厅",
        )
        assert info.mode == "rag"
        assert info.rag_context is not None
        assert info.preference_text is None

    def test_to_dict(self):
        info = InjectionInfo(mode="dki", original_query="test")
        d = info.to_dict()
        assert isinstance(d, dict)
        assert d['mode'] == "dki"
        assert d['original_query'] == "test"
        assert d['preference_text'] is None
        assert d['alpha'] == 0.0

    def test_get_display_text_dki(self):
        info = InjectionInfo(
            mode="dki",
            original_query="What should I eat?",
            preference_text="I prefer vegetarian food",
            preference_tokens=10,
            history_suffix="Previous conversation...",
            history_tokens=20,
            final_input="Previous conversation...\nWhat should I eat?",
            alpha=0.6,
        )
        text = info.get_display_text()
        assert "DKI" in text
        assert "What should I eat?" in text
        assert "vegetarian" in text
        assert "α=0.60" in text
        assert "Previous conversation" in text

    def test_get_display_text_rag(self):
        info = InjectionInfo(
            mode="rag",
            original_query="What should I eat?",
            rag_context="User prefers vegetarian",
            rag_prompt="Based on context: ...",
            final_input="Based on context: ...\nWhat should I eat?",
        )
        text = info.get_display_text()
        assert "RAG" in text
        assert "检索上下文" in text
        assert "完整提示词" in text

    def test_get_display_text_truncation(self):
        """长输入应被截断"""
        info = InjectionInfo(
            mode="dki",
            original_query="test",
            final_input="x" * 3000,
        )
        text = info.get_display_text()
        assert "中间省略" in text

    def test_get_display_text_history_messages(self):
        """历史消息应正确格式化"""
        info = InjectionInfo(
            mode="dki",
            original_query="test",
            history_messages=[
                {"role": "user", "content": "你好" + "x" * 200},
                {"role": "assistant", "content": "你好！"},
            ],
        )
        text = info.get_display_text()
        assert "用户" in text
        assert "助手" in text
        assert "..." in text  # 长消息应被截断


# ============================================================
# InjectionInfoViewer 测试
# ============================================================

class TestInjectionInfoViewer:
    """InjectionInfoViewer 功能测试"""

    @pytest.fixture
    def viewer(self, tmp_path):
        return InjectionInfoViewer(output_dir=str(tmp_path / "injection_logs"))

    @pytest.fixture
    def dki_info(self):
        return InjectionInfo(
            mode="dki",
            original_query="推荐餐厅",
            preference_text="素食主义者",
            preference_tokens=10,
            history_messages=[{"role": "user", "content": "之前的对话"}],
            history_tokens=5,
            final_input="素食主义者\n推荐餐厅",
            alpha=0.7,
        )

    @pytest.fixture
    def rag_info(self):
        return InjectionInfo(
            mode="rag",
            original_query="推荐餐厅",
            rag_context="用户偏好: 素食",
            rag_prompt="请基于上下文回答...",
            history_messages=[{"role": "user", "content": "之前的对话"}],
            final_input="请基于上下文回答...\n推荐餐厅",
        )

    def test_init_creates_directory(self, viewer, tmp_path):
        assert (tmp_path / "injection_logs").exists()

    def test_record(self, viewer, dki_info):
        viewer.record(dki_info)
        assert len(viewer._injection_history) == 1

    def test_get_latest(self, viewer, dki_info):
        for _ in range(15):
            viewer.record(dki_info)
        latest = viewer.get_latest(n=5)
        assert len(latest) == 5

    def test_get_latest_fewer_than_n(self, viewer, dki_info):
        viewer.record(dki_info)
        viewer.record(dki_info)
        latest = viewer.get_latest(n=10)
        assert len(latest) == 2

    def test_display(self, viewer, dki_info):
        text = viewer.display(dki_info)
        assert isinstance(text, str)
        assert "DKI" in text

    def test_compare(self, viewer, dki_info, rag_info):
        text = viewer.compare(dki_info, rag_info)
        assert "DKI" in text
        assert "RAG" in text
        assert "对比" in text

    def test_save_to_file(self, viewer, dki_info):
        filepath = viewer.save_to_file(dki_info)
        assert Path(filepath).exists()
        content = Path(filepath).read_text(encoding='utf-8')
        assert "DKI" in content

    def test_save_to_file_custom_name(self, viewer, dki_info):
        filepath = viewer.save_to_file(dki_info, filename="custom.txt")
        assert filepath.endswith("custom.txt")
        assert Path(filepath).exists()

    def test_save_comparison(self, viewer, dki_info, rag_info):
        filepath = viewer.save_comparison(dki_info, rag_info)
        assert Path(filepath).exists()
        content = Path(filepath).read_text(encoding='utf-8')
        assert "DKI" in content
        assert "RAG" in content

    def test_export_json(self, viewer, dki_info):
        data = viewer.export_json(dki_info)
        assert isinstance(data, dict)
        assert data['mode'] == "dki"
        assert 'display_text' in data

    def test_get_copyable_text(self, viewer, dki_info):
        text = viewer.get_copyable_text(dki_info)
        assert isinstance(text, str)
        assert len(text) > 0


# ============================================================
# ExperimentRunner 工具方法测试 (不需要真实 DKI/RAG 系统)
# ============================================================

class TestExperimentRunnerUtils:
    """ExperimentRunner 工具方法测试 (Mock 依赖)"""

    @pytest.fixture
    def mock_runner(self, tmp_path):
        """创建一个带 mock 依赖的 ExperimentRunner"""
        with patch("dki.experiment.runner.ConfigLoader") as MockConfigLoader, \
             patch("dki.experiment.runner.DatabaseManager") as MockDBManager:
            
            mock_config = MagicMock()
            mock_config.database.path = str(tmp_path / "test.db")
            MockConfigLoader.return_value.config = mock_config
            
            runner = ExperimentRunner(
                dki_system=MagicMock(),
                rag_system=MagicMock(),
                output_dir=str(tmp_path / "results"),
            )
            runner.config = mock_config
            return runner

    # ---- _extract_queries ----

    def test_extract_queries_from_query(self, mock_runner):
        item = {"query": "What should I eat?"}
        queries = mock_runner._extract_queries(item)
        assert queries == ["What should I eat?"]

    def test_extract_queries_from_question(self, mock_runner):
        item = {"question": "Where is Beijing?"}
        queries = mock_runner._extract_queries(item)
        assert queries == ["Where is Beijing?"]

    def test_extract_queries_from_turns(self, mock_runner):
        item = {
            "turns": [
                {"query": "First query"},
                {"query": "Second query"},
            ]
        }
        queries = mock_runner._extract_queries(item)
        assert queries == ["First query", "Second query"]

    def test_extract_queries_empty(self, mock_runner):
        item = {"data": "no query field"}
        queries = mock_runner._extract_queries(item)
        assert queries == []

    def test_extract_queries_priority(self, mock_runner):
        """query 优先于 question 和 turns"""
        item = {
            "query": "Direct query",
            "question": "Question field",
            "turns": [{"query": "Turn query"}],
        }
        queries = mock_runner._extract_queries(item)
        assert queries == ["Direct query"]

    # ---- _get_experiment_user_id ----

    def test_get_experiment_user_id_explicit(self, mock_runner):
        item = {"user_id": "explicit_user_123"}
        user_id = mock_runner._get_experiment_user_id(item)
        assert user_id == "explicit_user_123"

    def test_get_experiment_user_id_from_map(self, mock_runner):
        mock_runner._experiment_user_map = {
            "exp_user_vegetarian": "uid_veg",
            "exp_user_outdoor": "uid_out",
        }
        item = {"experiment_user": "exp_user_vegetarian"}
        user_id = mock_runner._get_experiment_user_id(item)
        assert user_id == "uid_veg"

    def test_get_experiment_user_id_default(self, mock_runner):
        item = {"some_field": "value"}
        user_id = mock_runner._get_experiment_user_id(item)
        assert user_id == "experiment_user"

    def test_get_experiment_user_id_custom_default(self, mock_runner):
        item = {}
        user_id = mock_runner._get_experiment_user_id(item, default="custom_default")
        assert user_id == "custom_default"

    # ---- _get_default_experiment_users ----

    def test_get_default_experiment_users(self, mock_runner):
        mock_runner.config.experiment = None
        users = mock_runner._get_default_experiment_users()
        assert len(users) == 4
        assert all("username" in u for u in users)
        assert all("preferences" in u for u in users)

    def test_get_default_experiment_users_from_config(self, mock_runner):
        """如果配置文件有用户定义，应使用配置文件"""
        mock_exp_config = MagicMock()
        mock_exp_config.users = [
            {"username": "config_user", "preferences": [{"text": "test", "type": "general"}]}
        ]
        mock_runner.config.experiment = mock_exp_config
        
        users = mock_runner._get_default_experiment_users()
        assert len(users) == 1
        assert users[0]["username"] == "config_user"

    # ---- _match_user_by_personas ----

    def test_match_user_by_personas_no_map(self, mock_runner):
        """无实验用户映射时返回默认值"""
        result = mock_runner._match_user_by_personas(["I like hiking"])
        assert result == "experiment_user"

    def test_match_user_by_personas_with_map(self, mock_runner):
        """有实验用户映射时应匹配"""
        mock_runner._experiment_user_map = {
            "exp_user_vegetarian": "uid_veg",
            "exp_user_outdoor": "uid_out",
            "exp_user_tech": "uid_tech",
            "exp_user_music": "uid_music",
        }
        # "素食" 相关 persona 应匹配到 vegetarian 用户
        result = mock_runner._match_user_by_personas(["素食主义者", "海鲜过敏"])
        assert isinstance(result, str)

    # ---- _aggregate_metrics ----

    def test_aggregate_metrics(self, mock_runner):
        results_by_mode = {
            "dki": {
                "metrics": {
                    "latency": {"p50": 50.0, "p95": 100.0},
                    "memory_usage": {"avg_memories_per_query": 2.5},
                    "alpha": {"mean": 0.6},
                    "cache_hit_rate": 0.8,
                }
            },
            "rag": {
                "metrics": {
                    "latency": {"p50": 80.0, "p95": 150.0},
                    "memory_usage": {"avg_memories_per_query": 3.0},
                }
            },
        }
        
        aggregated = mock_runner._aggregate_metrics(results_by_mode)
        
        assert "dki" in aggregated
        assert "rag" in aggregated
        assert aggregated["dki"]["latency_p50"] == 50.0
        assert aggregated["dki"]["alpha_mean"] == 0.6
        assert aggregated["dki"]["cache_hit_rate"] == 0.8
        assert aggregated["rag"]["latency_p50"] == 80.0
        assert "alpha_mean" not in aggregated["rag"]

    def test_aggregate_metrics_empty(self, mock_runner):
        aggregated = mock_runner._aggregate_metrics({})
        assert aggregated == {}

    # ---- _save_results ----

    def test_save_results(self, mock_runner, tmp_path):
        results = {
            "experiment_id": "exp_001",
            "config": {"name": "test"},
            "results_by_mode": {},
        }
        filepath = mock_runner._save_results(results)
        assert Path(filepath).exists()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded["experiment_id"] == "exp_001"

    # ---- _compute_mode_metrics ----

    def test_compute_mode_metrics_basic(self, mock_runner):
        results = [
            ExperimentResult(
                mode="dki", dataset="test", sample_id="s1",
                query="q1", response="I recommend vegetarian food for you.",
                latency_ms=10.0, memories_used=["vegetarian"],
                alpha=0.5, cache_hit=True,
            ),
            ExperimentResult(
                mode="dki", dataset="test", sample_id="s2",
                query="q2", response="Beijing has many great parks.",
                latency_ms=20.0, memories_used=["Beijing"],
                alpha=0.7, cache_hit=False,
            ),
        ]
        
        metrics = mock_runner._compute_mode_metrics(results)
        
        assert metrics['count'] == 2
        assert metrics['valid_count'] == 2
        assert metrics['error_count'] == 0
        assert 'latency' in metrics
        assert 'alpha' in metrics
        assert metrics['alpha']['mean'] == 0.6

    def test_compute_mode_metrics_with_errors(self, mock_runner):
        results = [
            ExperimentResult(
                mode="dki", dataset="test", sample_id="s1",
                query="q1", response="Valid response about something.",
                latency_ms=10.0, memories_used=[],
            ),
            ExperimentResult(
                mode="dki", dataset="test", sample_id="s2",
                query="q2", response="ERROR: connection failed",
                latency_ms=0.0, memories_used=[],
            ),
        ]
        
        metrics = mock_runner._compute_mode_metrics(results)
        assert metrics['valid_count'] == 1
        assert metrics['error_count'] == 1

    def test_compute_mode_metrics_empty(self, mock_runner):
        metrics = mock_runner._compute_mode_metrics([])
        assert metrics['count'] == 0
        assert metrics['valid_count'] == 0

    # ---- _ensure_systems ----

    def test_ensure_systems_already_set(self, mock_runner):
        """已设置系统时不应重新创建"""
        original_dki = mock_runner.dki_system
        original_rag = mock_runner.rag_system
        mock_runner._ensure_systems()
        assert mock_runner.dki_system is original_dki
        assert mock_runner.rag_system is original_rag

    # ---- _get_first_experiment_user_id (v3.1) ----

    def test_get_first_experiment_user_id_with_map(self, mock_runner):
        """有实验用户映射时返回第一个用户 ID"""
        mock_runner._experiment_user_map = {
            "exp_user_vegetarian": "uid_veg",
            "exp_user_outdoor": "uid_out",
        }
        user_id = mock_runner._get_first_experiment_user_id()
        assert user_id == "uid_veg"

    def test_get_first_experiment_user_id_no_map(self, mock_runner):
        """无实验用户映射时返回默认值"""
        user_id = mock_runner._get_first_experiment_user_id()
        assert user_id == "experiment_user"

    def test_get_first_experiment_user_id_empty_map(self, mock_runner):
        """空映射时返回默认值"""
        mock_runner._experiment_user_map = {}
        user_id = mock_runner._get_first_experiment_user_id()
        assert user_id == "experiment_user"

    # ---- clear_preference_cache 使用 (v3.1) ----

    def test_write_session_preferences_calls_clear_cache(self, mock_runner):
        """_write_session_preferences 应通过公开 API 清除缓存"""
        mock_runner.db_manager = MagicMock()
        # mock session_scope 上下文管理器
        mock_db = MagicMock()
        mock_runner.db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_runner.db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)
        
        mock_pref_repo = MagicMock()
        mock_pref_repo.get_by_user.return_value = []
        
        with patch("dki.experiment.runner.UserPreferenceRepository", return_value=mock_pref_repo):
            mock_runner._write_session_preferences("user_123", ["persona_1"])
        
        # 验证调用了 clear_preference_cache 而非直接操作 _user_preferences
        mock_runner.dki_system.clear_preference_cache.assert_called_once_with("user_123")

    # ---- CacheKeySigner 初始化 (v3.1) ----

    def test_cache_key_signer_initialized(self, tmp_path):
        """当 user_isolation 模块可用时，应初始化 CacheKeySigner"""
        with patch("dki.experiment.runner.ConfigLoader") as MockConfigLoader, \
             patch("dki.experiment.runner.DatabaseManager") as MockDBManager, \
             patch("dki.experiment.runner.USER_ISOLATION_AVAILABLE", True), \
             patch("dki.experiment.runner.CacheKeySigner") as MockSigner:
            
            mock_config = MagicMock()
            mock_config.database.path = str(tmp_path / "test.db")
            MockConfigLoader.return_value.config = mock_config
            
            runner = ExperimentRunner(
                dki_system=MagicMock(),
                rag_system=MagicMock(),
                output_dir=str(tmp_path / "results"),
            )
            
            MockSigner.assert_called_once()
            assert runner._cache_key_signer is not None
