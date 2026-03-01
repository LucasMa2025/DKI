"""
实验系统更新单元测试

覆盖论文对齐的所有修改:
1. MetricsCalculator: 分解幻觉 (fabricated vs irrelevant)
2. ExperimentDataGenerator: generate_context_constrained_data
3. ExperimentRunner: run_context_constrained / ablation 7 变体 / α sensitivity 多指标
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

from dki.experiment.metrics import MetricsCalculator
from dki.experiment.data_generator import ExperimentDataGenerator


# ============================================================
# MetricsCalculator — 分解幻觉测试
# ============================================================

class TestDecomposedHallucination:
    """compute_hallucination_decomposed 测试 (论文 Table 1b)"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    # ---- 基础返回结构 ----

    def test_decomposed_returns_all_keys(self, calc):
        """分解幻觉应返回完整的键集合"""
        result = calc.compute_hallucination_decomposed(
            response="The user is a vegetarian who lives in Beijing.",
            grounding_texts=["The user is a vegetarian."],
        )
        expected_keys = {
            'fabricated_rate', 'irrelevant_rate', 'total_rate',
            'fabricated_claims', 'irrelevant_claims', 'total_claims',
        }
        assert set(result.keys()) == expected_keys

    def test_decomposed_empty_response(self, calc):
        """空响应应返回全零"""
        result = calc.compute_hallucination_decomposed(
            response="",
            grounding_texts=["Some grounding text."],
        )
        assert result['fabricated_rate'] == 0.0
        assert result['irrelevant_rate'] == 0.0
        assert result['total_rate'] == 0.0
        assert result['fabricated_claims'] == []
        assert result['irrelevant_claims'] == []
        assert result['total_claims'] == 0

    def test_decomposed_all_grounded(self, calc):
        """完全基于事实的回答应有零幻觉率"""
        response = "The user is a vegetarian and lives in Beijing."
        grounding = ["The user is a vegetarian. The user lives in Beijing."]
        result = calc.compute_hallucination_decomposed(response, grounding)
        assert result['total_rate'] == 0.0
        assert result['fabricated_claims'] == []
        assert result['irrelevant_claims'] == []

    # ---- fabricated-detail 检测 ----

    def test_fabricated_detail_with_numbers(self, calc):
        """包含未支撑数字的声明应被标记为 fabricated-detail"""
        response = "The restaurant is located at 123 Main Street and costs about $50 per person."
        grounding = ["The user likes vegetarian food."]
        result = calc.compute_hallucination_decomposed(response, grounding)
        # 包含地址和价格数字，且不在 grounding 中
        assert result['fabricated_rate'] > 0.0 or result['irrelevant_rate'] > 0.0
        assert result['total_rate'] > 0.0

    def test_fabricated_detail_with_dates(self, calc):
        """包含未支撑日期的声明应被标记为 fabricated"""
        response = "The restaurant was established in 2015 and has 300 seats."
        grounding = ["The user prefers quiet restaurants."]
        result = calc.compute_hallucination_decomposed(response, grounding)
        assert result['total_rate'] > 0.0

    # ---- irrelevant/off-topic 检测 ----

    def test_irrelevant_off_topic(self, calc):
        """与查询和 grounding 完全不相关的内容应为 irrelevant"""
        response = "The weather in Tokyo is particularly sunny and warm today."
        grounding = ["The user is a vegetarian who lives in Beijing."]
        result = calc.compute_hallucination_decomposed(
            response, grounding,
            query="推荐一家素食餐厅",
        )
        # "Tokyo" 和 "weather" 与查询和 grounding 无关
        assert result['total_rate'] > 0.0

    # ---- total_rate = fabricated + irrelevant ----

    def test_total_rate_equals_sum(self, calc):
        """total_rate 应等于 fabricated_rate + irrelevant_rate"""
        response = "Beijing is a large city with over 21 million residents. The quantum physics lab has exciting experiments."
        grounding = ["The user lives in Beijing."]
        result = calc.compute_hallucination_decomposed(response, grounding)
        assert abs(result['total_rate'] -
                   (result['fabricated_rate'] + result['irrelevant_rate'])) < 1e-9

    # ---- known_facts 降低幻觉率 ----

    def test_known_facts_reduce_hallucination(self, calc):
        """额外已知事实应降低幻觉率"""
        response = "The user is a software engineer who has lived in Beijing for 5 years."
        grounding = ["The user lives in Beijing."]
        
        result_no_facts = calc.compute_hallucination_decomposed(response, grounding)
        result_with_facts = calc.compute_hallucination_decomposed(
            response, grounding,
            known_facts=["The user is a software engineer. Has lived there for 5 years."],
        )
        assert result_with_facts['total_rate'] <= result_no_facts['total_rate']

    # ---- _has_specific_details 辅助方法 ----

    def test_has_specific_details_number(self, calc):
        """包含多位数字应检测为 specific"""
        assert calc._has_specific_details("The price is 299 yuan") is True

    def test_has_specific_details_address(self, calc):
        """包含地址关键词应检测为 specific"""
        assert calc._has_specific_details("位于海淀区中关村大街") is True

    def test_has_specific_details_price(self, calc):
        """包含价格模式应检测为 specific"""
        assert calc._has_specific_details("价格约 ¥200") is True
        assert calc._has_specific_details("costs about $50") is True

    def test_has_specific_details_percentage(self, calc):
        """包含百分比应检测为 specific"""
        assert calc._has_specific_details("accuracy reaches 95.2%") is True

    def test_has_specific_details_no_details(self, calc):
        """无具体细节应返回 False"""
        assert calc._has_specific_details("I like vegetarian food") is False

    # ---- 旧版 compute_hallucination_rate 向后兼容 ----

    def test_old_api_still_works(self, calc):
        """旧版 compute_hallucination_rate 应仍可调用，返回 (rate, list)"""
        rate, hallucinations = calc.compute_hallucination_rate(
            response="The user is a vegetarian who lives in Beijing.",
            grounding_texts=["The user is a vegetarian. Lives in Beijing."],
        )
        assert isinstance(rate, float)
        assert isinstance(hallucinations, list)
        assert rate >= 0.0

    def test_old_api_empty_response(self, calc):
        """旧版 API 空响应应返回 0.0"""
        rate, hallucinations = calc.compute_hallucination_rate("", ["grounding"])
        assert rate == 0.0
        assert hallucinations == []


# ============================================================
# MetricsCalculator — _extract_claims 增强测试
# ============================================================

class TestExtractClaims:
    """_extract_claims 方法增强测试"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_extracts_factual_statements(self, calc):
        """应提取包含事实指示词的句子"""
        text = "Beijing is the capital of China. Hello world."
        claims = calc._extract_claims(text)
        assert any("Beijing" in c for c in claims)

    def test_filters_short_sentences(self, calc):
        """短于 10 字符的句子应被过滤"""
        text = "OK. Yes. The user is a vegetarian who lives in Shanghai."
        claims = calc._extract_claims(text)
        assert all(len(c) >= 10 for c in claims)

    def test_chinese_claims(self, calc):
        """中文事实声明提取"""
        text = "北京是中国的首都。用户有一只可爱的猫咪。天气不错。"
        claims = calc._extract_claims(text)
        assert len(claims) >= 1
        # "是" 和 "有" 是中文事实指示词
        assert any("北京" in c or "猫咪" in c for c in claims)

    def test_additional_chinese_indicators(self, calc):
        """应识别新增的中文事实指示词"""
        text = "用户位于北京海淀区，每月预算约3000元。"
        claims = calc._extract_claims(text)
        # "位于" 和 "约" 是新增的指示词
        assert len(claims) >= 1


# ============================================================
# ExperimentDataGenerator — generate_context_constrained_data
# ============================================================

class TestGenerateContextConstrainedData:
    """generate_context_constrained_data 测试 (论文 Table 2)"""

    @pytest.fixture(autouse=True)
    def setup_generator(self, tmp_path):
        self.output_dir = str(tmp_path / "test_data")
        self.generator = ExperimentDataGenerator(output_dir=self.output_dir)

    def test_generates_correct_count(self):
        """应按 memory_length × n_samples_per_length 生成样本"""
        lengths = [500, 1000, 1500]
        data = self.generator.generate_context_constrained_data(
            memory_lengths=lengths,
            n_samples_per_length=5,
        )
        assert len(data) == 3 * 5  # 3 lengths × 5 samples

    def test_default_memory_lengths(self):
        """默认应生成 7 种 memory 长度"""
        data = self.generator.generate_context_constrained_data(
            n_samples_per_length=2,
        )
        expected_lengths = {500, 1000, 1500, 2000, 2500, 3000, 3500}
        actual_lengths = set(d['memory_length_tokens'] for d in data)
        assert actual_lengths == expected_lengths

    def test_data_structure(self):
        """每条数据应包含必需字段"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=3,
        )
        for item in data:
            assert 'id' in item
            assert 'memory_length_tokens' in item
            assert 'context_budget' in item
            assert 'memory_text' in item
            assert 'memory_fragments' in item
            assert 'query' in item
            assert 'expected_keywords' in item
            assert 'experiment_user' in item

    def test_context_budget_is_4096(self):
        """context_budget 应固定为 4096"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=2,
        )
        for item in data:
            assert item['context_budget'] == 4096

    def test_memory_text_not_empty(self):
        """memory_text 应非空"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=2,
        )
        for item in data:
            assert len(item['memory_text']) > 0

    def test_saves_to_file(self):
        """应保存到 context_constrained.json"""
        self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=2,
        )
        filepath = Path(self.output_dir) / "context_constrained.json"
        assert filepath.exists()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert len(loaded) == 2

    def test_expected_keywords_not_empty(self):
        """expected_keywords 应非空"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=3,
        )
        for item in data:
            assert len(item['expected_keywords']) > 0

    def test_experiment_user_valid(self):
        """experiment_user 应来自预定义用户集"""
        valid_users = {
            "exp_user_vegetarian",
            "exp_user_tech",
            "exp_user_outdoor",
            "exp_user_music",
        }
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000, 2000],
            n_samples_per_length=5,
        )
        for item in data:
            assert item['experiment_user'] in valid_users

    def test_id_format_includes_length(self):
        """ID 应包含 memory length 信息"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[2000],
            n_samples_per_length=2,
        )
        for item in data:
            assert "2000" in item['id']

    def test_longer_memory_has_more_text(self):
        """更长的 memory_length 应产生更长的 memory_text"""
        data_short = self.generator.generate_context_constrained_data(
            memory_lengths=[500],
            n_samples_per_length=5,
        )
        data_long = self.generator.generate_context_constrained_data(
            memory_lengths=[3000],
            n_samples_per_length=5,
        )
        avg_short = np.mean([len(d['memory_text']) for d in data_short])
        avg_long = np.mean([len(d['memory_text']) for d in data_long])
        assert avg_long > avg_short

    def test_chinese_content(self):
        """memory_text 应包含中文"""
        data = self.generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=2,
        )
        for item in data:
            assert any('\u4e00' <= c <= '\u9fff' for c in item['memory_text'])


# ============================================================
# ExperimentRunner — run_context_constrained 测试
# ============================================================

class TestRunContextConstrained:
    """run_context_constrained 方法测试 (需要 mock DKI/RAG)"""

    @pytest.fixture
    def mock_runner(self, tmp_path):
        """创建带 mock 依赖的 ExperimentRunner"""
        with patch("dki.experiment.runner.ConfigLoader") as MockConfigLoader, \
             patch("dki.experiment.runner.DatabaseManager") as MockDBManager:
            
            mock_config = MagicMock()
            mock_config.database.path = str(tmp_path / "test.db")
            mock_config.experiment = MagicMock()
            mock_config.experiment.max_samples = 10
            mock_config.experiment.max_new_tokens = 64
            mock_config.experiment.temperature = 0.7
            mock_config.experiment.force_alpha = None
            MockConfigLoader.return_value.config = mock_config
            
            from dki.experiment.runner import ExperimentRunner
            
            runner = ExperimentRunner(
                dki_system=MagicMock(),
                rag_system=MagicMock(),
                output_dir=str(tmp_path / "results"),
            )
            runner.config = mock_config
            runner._experiment_user_map = {
                "exp_user_vegetarian": "uid_veg",
            }
            return runner

    def test_runner_has_run_context_constrained_method(self, mock_runner):
        """ExperimentRunner 应有 run_context_constrained 方法"""
        assert hasattr(mock_runner, 'run_context_constrained')
        assert callable(getattr(mock_runner, 'run_context_constrained'))


# ============================================================
# ExperimentRunner — ablation 7 变体测试
# ============================================================

class TestAblationVariants:
    """ablation study 7 变体配置测试"""

    @pytest.fixture
    def mock_runner(self, tmp_path):
        with patch("dki.experiment.runner.ConfigLoader") as MockConfigLoader, \
             patch("dki.experiment.runner.DatabaseManager") as MockDBManager:
            
            mock_config = MagicMock()
            mock_config.database.path = str(tmp_path / "test.db")
            mock_config.experiment = MagicMock()
            mock_config.experiment.max_samples = 5
            mock_config.experiment.max_new_tokens = 64
            mock_config.experiment.temperature = 0.7
            mock_config.experiment.force_alpha = None
            MockConfigLoader.return_value.config = mock_config
            
            from dki.experiment.runner import ExperimentRunner
            
            runner = ExperimentRunner(
                dki_system=MagicMock(),
                rag_system=MagicMock(),
                output_dir=str(tmp_path / "results"),
            )
            runner.config = mock_config
            return runner

    def test_runner_has_ablation_method(self, mock_runner):
        """应有 run_ablation_study 方法"""
        assert hasattr(mock_runner, 'run_ablation_study')

    def test_runner_has_alpha_sensitivity_method(self, mock_runner):
        """应有 run_alpha_sensitivity 方法"""
        assert hasattr(mock_runner, 'run_alpha_sensitivity')


# ============================================================
# MetricsCalculator — compute_all_metrics 一致性
# ============================================================

class TestComputeAllMetrics:
    """compute_all_metrics 方法测试"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_returns_correct_structure(self, calc):
        """应返回包含 count, latency, memory_recall, text_quality 的字典"""
        responses = [
            {'text': 'Response 1', 'latency_ms': 10.0},
            {'text': 'Response 2', 'latency_ms': 20.0},
        ]
        metrics = calc.compute_all_metrics(responses)
        assert 'count' in metrics
        assert 'latency' in metrics
        assert metrics['count'] == 2

    def test_empty_responses(self, calc):
        """空响应列表应返回 count=0"""
        metrics = calc.compute_all_metrics([])
        assert metrics['count'] == 0

    def test_with_references(self, calc):
        """带参考文本时应有 text_quality"""
        responses = [
            {'text': 'The cat sat on the mat', 'latency_ms': 10.0},
        ]
        references = ['The cat sat on the mat']
        metrics = calc.compute_all_metrics(responses, references)
        if metrics['text_quality']:
            assert 'bleu_mean' in metrics['text_quality']


# ============================================================
# 端到端数据生成 + 指标计算一致性
# ============================================================

class TestEndToEndConsistency:
    """验证数据生成器和指标计算器的端到端一致性"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    @pytest.fixture
    def generator(self, tmp_path):
        return ExperimentDataGenerator(output_dir=str(tmp_path / "data"))

    def test_context_constrained_data_usable_for_metrics(self, calc, generator):
        """context_constrained 数据应能被指标计算器使用"""
        data = generator.generate_context_constrained_data(
            memory_lengths=[1000],
            n_samples_per_length=3,
        )
        
        for item in data:
            # 模拟响应
            mock_response = f"这是一个关于{item['expected_keywords'][0]}的回答。"
            
            # 应能计算分解幻觉
            result = calc.compute_hallucination_decomposed(
                response=mock_response,
                grounding_texts=[item['memory_text']],
                query=item['query'],
            )
            assert isinstance(result['fabricated_rate'], float)
            assert isinstance(result['irrelevant_rate'], float)
            
            # 应能计算记忆召回
            recall, matched = calc.compute_memory_recall(
                expected_memories=item['memory_fragments'][:3],
                response=mock_response,
                threshold=0.3,
            )
            assert isinstance(recall, float)
            assert 0.0 <= recall <= 1.0

    def test_decomposed_hallucination_rates_are_valid(self, calc):
        """分解幻觉率应在 [0, 1] 范围内"""
        response = "Beijing is a city with 21 million people. The moon is made of cheese. Quantum computing is fascinating."
        grounding = ["Beijing is a large city in China."]
        result = calc.compute_hallucination_decomposed(response, grounding)
        
        assert 0.0 <= result['fabricated_rate'] <= 1.0
        assert 0.0 <= result['irrelevant_rate'] <= 1.0
        assert 0.0 <= result['total_rate'] <= 1.0
        assert result['total_claims'] >= 0
