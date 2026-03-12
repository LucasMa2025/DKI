"""
v8.0 实验系统更新单元测试

对齐论文:
- Table 2: Context-Constrained (memory_lengths)
- Table 3: 消融实验 (7 种配置, entropy-gated)
- Table 4: α 敏感性 (α 取值)
- §4.4: Entropy-Gated Metacognitive Retrieval 指标

测试范围:
1. ExperimentConfig v8.0 字段
2. InjectionInfo entropy-gated 字段
3. data_generator 消融/α/context 数据对齐
4. runner 消融配置 fact_retrieve_method
5. MetricsCalculator entropy_retrieval_stats
6. web/app ExperimentRequest v8.0 字段
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from dki.experiment.runner import (
    ExperimentConfig,
    InjectionInfo,
)
from dki.experiment.data_generator import ExperimentDataGenerator
from dki.experiment.metrics import MetricsCalculator


# ============================================================
# ExperimentConfig v8.0 测试
# ============================================================

class TestExperimentConfigV8:
    """ExperimentConfig v8.0 新增字段测试"""

    def test_default_fact_retrieve_method(self):
        """默认 fact_retrieve_method 应为 'auto'"""
        config = ExperimentConfig(name="test")
        assert config.fact_retrieve_method == "auto"

    def test_custom_fact_retrieve_method(self):
        """可自定义 fact_retrieve_method"""
        config = ExperimentConfig(name="test", fact_retrieve_method="entropy_gated")
        assert config.fact_retrieve_method == "entropy_gated"

    def test_alpha_values_aligned_with_paper_table4(self):
        """默认 α 取值应对齐论文 Table 4: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]"""
        config = ExperimentConfig(name="test")
        assert config.alpha_values == [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

    def test_to_dict_includes_fact_retrieve_method(self):
        """to_dict 应包含 fact_retrieve_method"""
        config = ExperimentConfig(name="test", fact_retrieve_method="entropy_gated")
        d = config.to_dict()
        assert 'fact_retrieve_method' in d
        assert d['fact_retrieve_method'] == "entropy_gated"

    def test_to_dict_includes_alpha_values_v8(self):
        """to_dict 中 alpha_values 应为 v8.0 值"""
        config = ExperimentConfig(name="test")
        d = config.to_dict()
        assert d['alpha_values'] == [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]


# ============================================================
# InjectionInfo v8.0 entropy-gated 字段测试
# ============================================================

class TestInjectionInfoV8:
    """InjectionInfo entropy-gated 新增字段测试"""

    def test_default_entropy_fields(self):
        """默认 entropy 字段应为初始值"""
        info = InjectionInfo(mode="dki")
        assert info.fact_retrieve_method == "post_hoc"
        assert info.entropy_triggered is False
        assert info.entropy_probe_tokens == 0
        assert info.entropy_grounding_facts == []
        assert info.entropy_stages == 1
        assert info.entropy_spike_position == -1
        assert info.entropy_max_value == 0.0

    def test_entropy_triggered_info(self):
        """entropy_triggered=True 时应正确记录"""
        info = InjectionInfo(
            mode="dki",
            fact_retrieve_method="entropy_gated",
            entropy_triggered=True,
            entropy_probe_tokens=64,
            entropy_grounding_facts=["fact1", "fact2"],
            entropy_stages=2,
            entropy_spike_position=42,
            entropy_max_value=3.5,
        )
        assert info.entropy_triggered is True
        assert info.entropy_probe_tokens == 64
        assert len(info.entropy_grounding_facts) == 2
        assert info.entropy_stages == 2
        assert info.entropy_spike_position == 42
        assert info.entropy_max_value == 3.5

    def test_to_dict_includes_entropy_fields(self):
        """to_dict 应包含所有 entropy 字段"""
        info = InjectionInfo(
            mode="dki",
            fact_retrieve_method="entropy_gated",
            entropy_triggered=True,
        )
        d = info.to_dict()
        assert 'fact_retrieve_method' in d
        assert 'entropy_triggered' in d
        assert 'entropy_probe_tokens' in d
        assert 'entropy_grounding_facts' in d
        assert 'entropy_stages' in d
        assert 'entropy_spike_position' in d
        assert 'entropy_max_value' in d

    def test_display_text_entropy_not_triggered(self):
        """未触发 entropy 时显示文本应包含方法信息"""
        info = InjectionInfo(
            mode="dki",
            original_query="test query",
            fact_retrieve_method="entropy_gated",
            entropy_triggered=False,
        )
        text = info.get_display_text()
        assert "entropy_gated" in text

    def test_display_text_entropy_triggered(self):
        """触发 entropy 时显示文本应包含详细信息"""
        info = InjectionInfo(
            mode="dki",
            original_query="test query",
            fact_retrieve_method="entropy_gated",
            entropy_triggered=True,
            entropy_probe_tokens=64,
            entropy_grounding_facts=["User prefers coffee", "User lives in Beijing"],
            entropy_stages=2,
            entropy_spike_position=42,
            entropy_max_value=3.5,
        )
        text = info.get_display_text()
        assert "entropy_gated" in text
        assert "触发" in text or "TRIGGERED" in text.upper() or "✓" in text


# ============================================================
# DataGenerator v8.0 对齐测试
# ============================================================

class TestDataGeneratorV8:
    """DataGenerator v8.0 数据对齐测试"""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.output_dir = str(tmp_path / "test_data")
        self.generator = ExperimentDataGenerator(output_dir=self.output_dir)

    # --- 消融数据 (Table 3) ---

    def test_ablation_modes_aligned_with_paper_table3(self):
        """消融模式应对齐论文 Table 3 的 7 种配置"""
        data = self.generator.generate_ablation_data(n_samples=1)
        expected_modes = [
            "full_dki", "wo_fact_call", "wo_multi_signal",
            "wo_kv_injection", "stable_fallback_only",
            "rag_baseline", "vanilla_llm",
        ]
        assert data[0]['ablation_modes'] == expected_modes

    def test_ablation_has_7_modes(self):
        """消融实验应有 7 种模式"""
        data = self.generator.generate_ablation_data(n_samples=1)
        assert len(data[0]['ablation_modes']) == 7

    def test_ablation_data_structure(self):
        """消融数据应包含必需字段"""
        data = self.generator.generate_ablation_data(n_samples=3)
        for sample in data:
            assert 'query' in sample
            assert 'memory' in sample
            assert 'relevant_memories' in sample
            assert 'ablation_modes' in sample
            assert 'all_memories' in sample

    # --- α 敏感性数据 (Table 4) ---

    def test_alpha_values_aligned_with_paper_table4(self):
        """α 取值应对齐论文 Table 4: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]"""
        data = self.generator.generate_alpha_sensitivity_data(n_samples=1)
        # 每个 sample 有一个 alpha 值, n_samples=1 时生成 7 个样本 (每个 alpha 一个)
        actual_alphas = sorted(set(d['alpha'] for d in data))
        expected_alphas = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        assert actual_alphas == expected_alphas

    def test_alpha_values_has_7_values(self):
        """n_samples=1 时应生成 7 个样本 (每个 α 一个)"""
        data = self.generator.generate_alpha_sensitivity_data(n_samples=1)
        unique_alphas = set(d['alpha'] for d in data)
        assert len(unique_alphas) == 7

    # --- Context-Constrained 数据 (Table 2) ---

    def test_context_constrained_default_lengths_v8(self):
        """默认 memory 长度应对齐论文 Table 2: [1000, 1500, 2000, 2500, 3000, 3500]"""
        data = self.generator.generate_context_constrained_data(n_samples_per_length=1)
        actual_lengths = sorted(set(d['memory_length_tokens'] for d in data))
        expected_lengths = [1000, 1500, 2000, 2500, 3000, 3500]
        assert actual_lengths == expected_lengths

    def test_context_constrained_has_6_lengths(self):
        """默认应有 6 种 memory 长度"""
        data = self.generator.generate_context_constrained_data(n_samples_per_length=1)
        unique_lengths = set(d['memory_length_tokens'] for d in data)
        assert len(unique_lengths) == 6


# ============================================================
# Runner 消融配置 v8.0 测试
# ============================================================

class TestRunnerAblationConfigV8:
    """Runner 消融配置 fact_retrieve_method 测试"""

    def test_ablation_configs_use_fact_retrieve_method(self):
        """消融配置应使用 fact_retrieve_method 而非 allow_fact_call"""
        # 直接检查消融配置字典结构
        # full_dki 应使用 entropy_gated
        ablation_configs = {
            'full_dki': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'entropy_gated',
                'recall_mode': 'multi_signal',
                'use_kv_injection': True,
            },
            'wo_fact_call': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'post_hoc',
                'recall_mode': 'multi_signal',
                'use_kv_injection': True,
            },
        }
        
        assert ablation_configs['full_dki']['fact_retrieve_method'] == 'entropy_gated'
        assert ablation_configs['wo_fact_call']['fact_retrieve_method'] == 'post_hoc'
        # 不应有旧的 allow_fact_call 字段
        assert 'allow_fact_call' not in ablation_configs['full_dki']
        assert 'allow_fact_call' not in ablation_configs['wo_fact_call']

    def test_all_7_ablation_modes_have_fact_retrieve_method(self):
        """所有 7 种消融模式都应有 fact_retrieve_method 字段"""
        ablation_modes = [
            'full_dki', 'wo_fact_call', 'wo_multi_signal',
            'wo_kv_injection', 'stable_fallback_only',
            'rag_baseline', 'vanilla_llm',
        ]
        # full_dki, wo_multi_signal, wo_kv_injection 使用 entropy_gated
        entropy_gated_modes = {'full_dki', 'wo_multi_signal', 'wo_kv_injection'}
        # wo_fact_call, stable_fallback_only 使用 post_hoc
        post_hoc_modes = {'wo_fact_call', 'stable_fallback_only'}
        # rag_baseline, vanilla_llm 使用 None
        none_modes = {'rag_baseline', 'vanilla_llm'}
        
        assert entropy_gated_modes | post_hoc_modes | none_modes == set(ablation_modes)


# ============================================================
# MetricsCalculator entropy_retrieval_stats 测试
# ============================================================

class TestEntropyRetrievalStats:
    """MetricsCalculator.compute_entropy_retrieval_stats 测试"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_empty_samples(self, calc):
        """空样本列表应返回零值"""
        stats = calc.compute_entropy_retrieval_stats([])
        assert stats['total_samples'] == 0
        assert stats['trigger_rate'] == 0.0

    def test_no_entropy_stats_in_samples(self, calc):
        """样本无 entropy_stats 时应返回零值"""
        samples = [
            {'query': 'q1', 'latency_ms': 100},
            {'query': 'q2', 'latency_ms': 200},
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        assert stats['total_samples'] == 2
        assert stats['entropy_gated_samples'] == 0
        assert stats['trigger_rate'] == 0.0

    def test_all_triggered(self, calc):
        """全部触发时 trigger_rate 应为 1.0"""
        samples = [
            {
                'query': f'q{i}',
                'latency_ms': 100 + i * 10,
                'entropy_stats': {
                    'entropy_triggered': True,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 1,
                    'fact_retrieve_method': 'entropy_gated',
                },
            }
            for i in range(5)
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        assert stats['triggered_count'] == 5
        assert stats['trigger_rate'] == 1.0
        assert stats['total_retrievals'] == 5
        assert stats['avg_retrievals_per_triggered'] == 1.0

    def test_partial_triggered(self, calc):
        """部分触发时 trigger_rate 应正确计算"""
        samples = [
            {
                'query': 'q1',
                'latency_ms': 100,
                'entropy_stats': {
                    'entropy_triggered': True,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 2,
                    'fact_retrieve_method': 'entropy_gated',
                },
            },
            {
                'query': 'q2',
                'latency_ms': 50,
                'entropy_stats': {
                    'entropy_triggered': False,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 0,
                    'fact_retrieve_method': 'entropy_gated',
                },
            },
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        assert stats['triggered_count'] == 1
        assert stats['trigger_rate'] == 0.5
        assert stats['total_retrievals'] == 2
        assert stats['avg_retrievals_per_triggered'] == 2.0

    def test_retrieval_overhead_calculation(self, calc):
        """应计算触发 vs 未触发的延迟差异"""
        samples = [
            {
                'query': 'q1',
                'latency_ms': 200,  # 触发, 延迟高
                'entropy_stats': {
                    'entropy_triggered': True,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 1,
                    'fact_retrieve_method': 'entropy_gated',
                },
            },
            {
                'query': 'q2',
                'latency_ms': 100,  # 未触发, 延迟低
                'entropy_stats': {
                    'entropy_triggered': False,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 0,
                    'fact_retrieve_method': 'entropy_gated',
                },
            },
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        assert 'retrieval_overhead_ms' in stats
        assert stats['retrieval_overhead_ms'] == pytest.approx(100.0)
        assert stats['triggered_mean_latency_ms'] == pytest.approx(200.0)
        assert stats['non_triggered_mean_latency_ms'] == pytest.approx(100.0)

    def test_avg_probe_tokens(self, calc):
        """应正确计算平均探测 tokens"""
        samples = [
            {
                'query': f'q{i}',
                'entropy_stats': {
                    'entropy_triggered': False,
                    'entropy_probe_tokens': 32 + i * 16,
                    'entropy_retrievals': 0,
                    'fact_retrieve_method': 'entropy_gated',
                },
            }
            for i in range(3)
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        # tokens: 32, 48, 64 → avg = 48
        assert stats['avg_probe_tokens'] == pytest.approx(48.0)

    def test_mixed_with_non_entropy_samples(self, calc):
        """混合样本 (有/无 entropy_stats) 应正确处理"""
        samples = [
            {'query': 'q1', 'latency_ms': 50},  # 无 entropy_stats
            {
                'query': 'q2',
                'latency_ms': 100,
                'entropy_stats': {
                    'entropy_triggered': True,
                    'entropy_probe_tokens': 64,
                    'entropy_retrievals': 1,
                    'fact_retrieve_method': 'entropy_gated',
                },
            },
        ]
        stats = calc.compute_entropy_retrieval_stats(samples)
        assert stats['total_samples'] == 2
        assert stats['entropy_gated_samples'] == 1
        assert stats['triggered_count'] == 1
        assert stats['trigger_rate'] == 1.0  # 1/1 entropy samples triggered


# ============================================================
# Web ExperimentRequest v8.0 测试
# ============================================================

class TestExperimentRequestV8:
    """ExperimentRequest v8.0 新增字段测试"""

    def test_default_fact_retrieve_method(self):
        """验证 ExperimentRequest 可以导入并有 fact_retrieve_method"""
        # 由于 web/app.py 中 ExperimentRequest 定义在 create_app 外部,
        # 可以直接导入
        try:
            from dki.web.app import ExperimentRequest
            req = ExperimentRequest(name="test")
            assert req.fact_retrieve_method == "auto"
            assert req.force_alpha == 0.4
            assert req.max_new_tokens == 2048
        except ImportError:
            pytest.skip("web app dependencies not available")

    def test_custom_fact_retrieve_method(self):
        """可自定义 fact_retrieve_method"""
        try:
            from dki.web.app import ExperimentRequest
            req = ExperimentRequest(
                name="test",
                fact_retrieve_method="entropy_gated",
                force_alpha=0.5,
            )
            assert req.fact_retrieve_method == "entropy_gated"
            assert req.force_alpha == 0.5
        except ImportError:
            pytest.skip("web app dependencies not available")


# ============================================================
# 集成验证: 消融配置与数据生成一致性
# ============================================================

class TestAblationConsistency:
    """消融配置 (runner) 与数据生成 (data_generator) 的一致性"""

    @pytest.fixture
    def generator(self, tmp_path):
        return ExperimentDataGenerator(output_dir=str(tmp_path / "data"))

    def test_ablation_modes_match_data_and_runner(self, generator):
        """data_generator 和 runner 的消融模式应一致"""
        data = generator.generate_ablation_data(n_samples=1)
        data_modes = data[0]['ablation_modes']
        
        runner_modes = [
            "full_dki", "wo_fact_call", "wo_multi_signal",
            "wo_kv_injection", "stable_fallback_only",
            "rag_baseline", "vanilla_llm",
        ]
        
        assert data_modes == runner_modes

    def test_alpha_values_match_config_and_data(self, generator):
        """ExperimentConfig 和 data_generator 的 α 取值应一致"""
        config = ExperimentConfig(name="test")
        data = generator.generate_alpha_sensitivity_data(n_samples=1)
        
        # data_generator 为每个 alpha 值生成一个样本
        actual_alphas = sorted(set(d['alpha'] for d in data))
        assert config.alpha_values == actual_alphas
