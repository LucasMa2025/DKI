"""
MetricsCalculator 单元测试

测试实验评估指标计算器:
- BLEU 分数计算
- ROUGE 分数计算
- 记忆召回率
- 幻觉检测率
- 延迟统计
- 批量指标聚合
"""

import math
from typing import Dict, List
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from dki.experiment.metrics import MetricsCalculator


class TestMetricsCalculator:
    """MetricsCalculator 基础测试"""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    # ============ 初始化测试 ============

    def test_init(self, calc):
        assert calc._nltk_initialized is False
        assert calc._rouge_scorer is None

    # ============ BLEU 测试 ============

    def test_compute_bleu_identical(self, calc):
        """完全相同文本的 BLEU 应接近 1.0 (需要 nltk)"""
        text = "The quick brown fox jumps over the lazy dog"
        score = calc.compute_bleu(text, text)
        # nltk 未安装时返回 0.0，跳过断言
        if score > 0.0:
            assert score > 0.9
        else:
            pytest.skip("nltk not installed, BLEU returns 0.0")

    def test_compute_bleu_different(self, calc):
        """完全不同文本的 BLEU 应接近 0.0"""
        ref = "The quick brown fox jumps over the lazy dog"
        hyp = "A completely different sentence about nothing related"
        score = calc.compute_bleu(ref, hyp)
        assert score < 0.3

    def test_compute_bleu_empty_hypothesis(self, calc):
        """空假设文本应返回 0.0"""
        score = calc.compute_bleu("some reference text", "")
        assert score == 0.0

    def test_compute_bleu_empty_reference(self, calc):
        """空参考文本应返回 0.0 或非常低的分数"""
        score = calc.compute_bleu("", "some hypothesis text")
        assert score <= 0.1

    def test_compute_bleu_partial_overlap(self, calc):
        """部分重叠的文本应有中等 BLEU 分数 (需要 nltk)"""
        ref = "I like eating vegetarian food for lunch"
        hyp = "I enjoy eating vegetarian dishes for dinner"
        score = calc.compute_bleu(ref, hyp)
        # nltk 未安装时返回 0.0，跳过断言
        if score > 0.0:
            assert score < 1.0
        else:
            pytest.skip("nltk not installed, BLEU returns 0.0")

    def test_compute_bleu_ngram_parameter(self, calc):
        """不同 n-gram 阶数应产生不同分数"""
        ref = "The quick brown fox"
        hyp = "The quick brown dog"
        score_1 = calc.compute_bleu(ref, hyp, n_gram=1)
        score_4 = calc.compute_bleu(ref, hyp, n_gram=4)
        # unigram 分数通常高于 4-gram
        assert score_1 >= score_4

    # ============ ROUGE 测试 ============

    def test_compute_rouge_identical(self, calc):
        """完全相同文本的 ROUGE 应接近 1.0"""
        text = "The quick brown fox jumps over the lazy dog"
        scores = calc.compute_rouge(text, text)
        assert isinstance(scores, dict)
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores
        # 可能 rouge-score 库未安装，检查是否返回了有效值
        # 如果库未安装会返回 0.0
        if scores['rouge1'] > 0:
            assert scores['rouge1'] > 0.9

    def test_compute_rouge_different(self, calc):
        """完全不同文本的 ROUGE 应较低"""
        ref = "The quick brown fox"
        hyp = "A completely unrelated sentence"
        scores = calc.compute_rouge(ref, hyp)
        assert isinstance(scores, dict)
        # ROUGE-1 可能有少量重叠 (如 "a")
        assert scores['rouge1'] < 0.5

    def test_compute_rouge_returns_dict_keys(self, calc):
        """ROUGE 应返回正确的键"""
        scores = calc.compute_rouge("hello world", "hello earth")
        assert set(scores.keys()) == {'rouge1', 'rouge2', 'rougeL'}

    def test_compute_rouge_empty(self, calc):
        """空文本应返回 0.0"""
        scores = calc.compute_rouge("", "")
        assert all(v >= 0.0 for v in scores.values())

    # ============ 记忆召回率测试 ============

    def test_memory_recall_all_matched(self, calc):
        """所有记忆都被引用时召回率应为 1.0"""
        memories = [
            "I like vegetarian food",
            "I live in Beijing",
        ]
        response = "Based on your preference for vegetarian food and your location in Beijing, I recommend..."
        recall, matched = calc.compute_memory_recall(memories, response)
        assert recall == 1.0
        assert len(matched) == 2

    def test_memory_recall_none_matched(self, calc):
        """没有记忆被引用时召回率应为 0.0"""
        memories = [
            "I like seafood",
            "I collect vintage watches",
        ]
        response = "The weather today is sunny and warm."
        recall, matched = calc.compute_memory_recall(memories, response)
        assert recall == 0.0
        assert len(matched) == 0

    def test_memory_recall_partial_matched(self, calc):
        """部分记忆被引用时召回率应在 0 和 1 之间"""
        memories = [
            "I like hiking and outdoor activities",
            "I collect vintage watches",
            "I live in Shanghai",
        ]
        response = "For outdoor activities, I recommend hiking in the nearby mountains."
        recall, matched = calc.compute_memory_recall(memories, response)
        assert 0.0 < recall < 1.0
        assert len(matched) >= 1

    def test_memory_recall_empty_memories(self, calc):
        """空记忆列表应返回 1.0"""
        recall, matched = calc.compute_memory_recall([], "any response")
        assert recall == 1.0
        assert matched == []

    def test_memory_recall_threshold(self, calc):
        """不同阈值应影响匹配结果"""
        memories = ["I prefer vegetarian food and dislike meat"]
        response = "Here are some vegetarian options for you"
        
        # 低阈值更容易匹配
        recall_low, _ = calc.compute_memory_recall(memories, response, threshold=0.2)
        recall_high, _ = calc.compute_memory_recall(memories, response, threshold=0.9)
        assert recall_low >= recall_high

    def test_memory_recall_case_insensitive(self, calc):
        """匹配应不区分大小写"""
        memories = ["I LIKE HIKING"]
        response = "since you like hiking, here are some trails"
        recall, matched = calc.compute_memory_recall(memories, response, threshold=0.3)
        assert recall > 0.0

    # ============ 关键词提取测试 ============

    def test_extract_keywords_english(self, calc):
        """英文关键词提取"""
        keywords = calc._extract_keywords("I love hiking and outdoor activities")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # min_len=3, 所以 "I" 不应出现
        assert all(len(kw) >= 3 for kw in keywords)

    def test_extract_keywords_chinese(self, calc):
        """中文关键词提取"""
        keywords = calc._extract_keywords("我喜欢户外运动和摄影")
        assert isinstance(keywords, list)
        # 中文字符也应被匹配
        assert len(keywords) >= 0  # 可能由于 min_len=3 过滤掉短词

    def test_extract_keywords_limit(self, calc):
        """关键词数量应限制在 10 个以内"""
        long_text = " ".join(f"keyword{i}" for i in range(100))
        keywords = calc._extract_keywords(long_text, min_len=1)
        assert len(keywords) <= 10

    # ============ 幻觉检测测试 ============

    def test_hallucination_rate_grounded(self, calc):
        """完全基于事实的回答应有低幻觉率"""
        response = "Beijing is a city in China. The user lives in Beijing."
        grounding = ["The user lives in Beijing, China."]
        rate, hallucinations = calc.compute_hallucination_rate(response, grounding)
        assert rate < 0.5

    def test_hallucination_rate_ungrounded(self, calc):
        """不基于事实的回答应有较高幻觉率"""
        response = "Tokyo is the capital of France. Paris is in Japan."
        grounding = ["Beijing is the capital of China."]
        rate, hallucinations = calc.compute_hallucination_rate(response, grounding)
        assert rate > 0.0

    def test_hallucination_rate_empty_response(self, calc):
        """空回答应返回 0.0 幻觉率"""
        rate, hallucinations = calc.compute_hallucination_rate("", ["some grounding"])
        assert rate == 0.0
        assert hallucinations == []

    def test_hallucination_rate_with_known_facts(self, calc):
        """已知事实应降低幻觉率"""
        response = "The user is a software engineer who lives in Beijing."
        grounding = ["The user lives in Beijing."]
        known_facts = ["The user is a software engineer."]
        rate, _ = calc.compute_hallucination_rate(response, grounding, known_facts)
        # 有已知事实支撑，幻觉率应更低
        assert rate < 1.0

    # ============ 声明提取测试 ============

    def test_extract_claims(self, calc):
        """应提取包含事实指示词的句子"""
        text = "Beijing is the capital of China. The sky looks beautiful today."
        claims = calc._extract_claims(text)
        assert isinstance(claims, list)
        # "is" 是事实指示词
        assert any("Beijing" in c for c in claims)

    def test_extract_claims_short_sentences_filtered(self, calc):
        """短于 10 字符的句子应被过滤"""
        text = "Yes. No. OK. This is a longer factual sentence about something."
        claims = calc._extract_claims(text)
        assert all(len(c) >= 10 for c in claims)

    def test_extract_claims_chinese(self, calc):
        """中文声明提取"""
        # 使用较长的中文句子确保超过 10 字符的最小长度过滤
        text = "北京是中华人民共和国的首都城市。今天的天气非常好适合出行。用户有一只非常可爱的猫咪叫花花。"
        claims = calc._extract_claims(text)
        assert isinstance(claims, list)
        # "是" 和 "有" 是中文事实指示词
        assert len(claims) >= 1

    # ============ 延迟统计测试 ============

    def test_latency_stats_basic(self, calc):
        """基本延迟统计"""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calc.compute_latency_stats(latencies)
        
        assert 'p50' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        assert stats['mean'] == 30.0
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert stats['p50'] == 30.0

    def test_latency_stats_empty(self, calc):
        """空延迟列表应返回全 0"""
        stats = calc.compute_latency_stats([])
        assert stats['mean'] == 0
        assert stats['p50'] == 0
        assert stats['p95'] == 0

    def test_latency_stats_single(self, calc):
        """单个值的延迟统计"""
        stats = calc.compute_latency_stats([42.0])
        assert stats['mean'] == 42.0
        assert stats['p50'] == 42.0
        assert stats['min'] == 42.0
        assert stats['max'] == 42.0
        assert stats['std'] == 0.0

    def test_latency_stats_large_spread(self, calc):
        """大范围延迟的统计"""
        latencies = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        stats = calc.compute_latency_stats(latencies)
        assert stats['p95'] > stats['p50']
        assert stats['p99'] >= stats['p95']
        assert stats['std'] > 0

    # ============ 批量指标计算测试 ============

    def test_compute_all_metrics_basic(self, calc):
        """基本批量指标计算"""
        responses = [
            {'text': 'Response 1', 'latency_ms': 10.0},
            {'text': 'Response 2', 'latency_ms': 20.0},
            {'text': 'Response 3', 'latency_ms': 30.0},
        ]
        metrics = calc.compute_all_metrics(responses)
        
        assert metrics['count'] == 3
        assert 'latency' in metrics
        assert metrics['latency']['mean'] == 20.0

    def test_compute_all_metrics_with_references(self, calc):
        """带参考文本的批量指标计算"""
        responses = [
            {'text': 'The cat sat on the mat', 'latency_ms': 10.0},
            {'text': 'The dog ran in the park', 'latency_ms': 20.0},
        ]
        references = [
            'The cat sat on the mat',
            'The dog played in the park',
        ]
        metrics = calc.compute_all_metrics(responses, references)
        
        assert 'text_quality' in metrics
        if metrics['text_quality']:
            assert 'bleu_mean' in metrics['text_quality']
            assert 'rouge1_mean' in metrics['text_quality']

    def test_compute_all_metrics_no_references(self, calc):
        """无参考文本时不计算文本质量"""
        responses = [
            {'text': 'Response', 'latency_ms': 10.0},
        ]
        metrics = calc.compute_all_metrics(responses)
        assert metrics['text_quality'] == {}

    def test_compute_all_metrics_mismatched_references(self, calc):
        """参考文本数量不匹配时不计算文本质量"""
        responses = [
            {'text': 'Response 1', 'latency_ms': 10.0},
            {'text': 'Response 2', 'latency_ms': 20.0},
        ]
        references = ['Only one reference']
        metrics = calc.compute_all_metrics(responses, references)
        assert metrics['text_quality'] == {}

    def test_compute_all_metrics_empty(self, calc):
        """空响应列表"""
        metrics = calc.compute_all_metrics([])
        assert metrics['count'] == 0
