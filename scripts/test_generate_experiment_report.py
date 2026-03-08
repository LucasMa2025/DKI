#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKI 实验报告生成器 - 单元测试
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 确保可以导入被测模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_experiment_report import (
    load_json,
    analyze_datasets,
    analyze_persona_chat_experiment,
    analyze_longmemeval_experiment,
    compare_with_paper_predictions,
    generate_report,
)


class TestLoadJson(unittest.TestCase):
    """测试 JSON 加载函数"""
    
    def test_load_valid_json(self):
        """测试加载有效的 JSON 文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
            f.flush()
            result = load_json(f.name)
        self.assertEqual(result, {"key": "value"})
        os.unlink(f.name)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        result = load_json("/nonexistent/path/file.json")
        self.assertIsNone(result)
    
    def test_load_invalid_json(self):
        """测试加载无效的 JSON 文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not valid json {{{")
            f.flush()
            result = load_json(f.name)
        self.assertIsNone(result)
        os.unlink(f.name)


class TestAnalyzeDatasets(unittest.TestCase):
    """测试数据集分析函数"""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)
    
    def _write_json(self, name, data):
        path = Path(self.tmpdir) / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path
    
    def test_ablation_dataset(self):
        """测试消融数据集分析"""
        data = [
            {"query": "q1", "ablation_modes": ["full_dki", "no_gating"]},
            {"query": "q2", "ablation_modes": ["full_dki", "no_memory"]},
        ]
        path = self._write_json("ablation", data)
        stats = analyze_datasets({"ablation": path})
        
        self.assertIn("ablation", stats)
        self.assertEqual(stats["ablation"]["count"], 2)
        self.assertEqual(stats["ablation"]["unique_queries"], 2)
        self.assertIn("full_dki", stats["ablation"]["ablation_modes"])
    
    def test_alpha_sensitivity_dataset(self):
        """测试 Alpha 敏感性数据集分析"""
        data = [
            {"query": "q1", "alpha": 0.0},
            {"query": "q1", "alpha": 0.5},
            {"query": "q2", "alpha": 1.0},
        ]
        path = self._write_json("alpha_sensitivity", data)
        stats = analyze_datasets({"alpha_sensitivity": path})
        
        self.assertIn("alpha_sensitivity", stats)
        self.assertEqual(stats["alpha_sensitivity"]["count"], 3)
        self.assertEqual(stats["alpha_sensitivity"]["alpha_values"], [0.0, 0.5, 1.0])
    
    def test_context_constrained_dataset(self):
        """测试上下文受限数据集分析"""
        data = [
            {"context_budget": 4096, "memory_length_tokens": 2000},
            {"context_budget": 4096, "memory_length_tokens": 3000},
        ]
        path = self._write_json("context_constrained", data)
        stats = analyze_datasets({"context_constrained": path})
        
        self.assertIn("context_constrained", stats)
        self.assertEqual(stats["context_constrained"]["context_budgets"], [4096])
        self.assertEqual(stats["context_constrained"]["memory_lengths"], [2000, 3000])
    
    def test_persona_chat_dataset(self):
        """测试 PersonaChat 数据集分析"""
        data = [
            {"turns": [{"query": "q1"}, {"query": "q2"}]},
            {"turns": [{"query": "q3"}]},
        ]
        path = self._write_json("persona_chat", data)
        stats = analyze_datasets({"persona_chat": path})
        
        self.assertIn("persona_chat", stats)
        self.assertEqual(stats["persona_chat"]["total_sessions"], 2)
        self.assertEqual(stats["persona_chat"]["total_turns"], 3)
        self.assertEqual(stats["persona_chat"]["avg_turns_per_session"], 1.5)
    
    def test_memory_qa_dataset(self):
        """测试 Memory QA 数据集分析"""
        data = [
            {"expected_memory_use": True},
            {"expected_memory_use": False},
            {"expected_memory_use": True},
        ]
        path = self._write_json("memory_qa", data)
        stats = analyze_datasets({"memory_qa": path})
        
        self.assertIn("memory_qa", stats)
        self.assertEqual(stats["memory_qa"]["expected_memory_use_true"], 2)
    
    def test_nonexistent_file(self):
        """测试不存在的文件"""
        stats = analyze_datasets({"missing": Path("/nonexistent/file.json")})
        self.assertEqual(stats, {})


class TestAnalyzePersonaChatExperiment(unittest.TestCase):
    """测试 PersonaChat 实验结果分析"""
    
    def test_none_input(self):
        """测试 None 输入"""
        result = analyze_persona_chat_experiment(None)
        self.assertIsNone(result)
    
    def test_basic_analysis(self):
        """测试基本分析"""
        data = {
            "experiment_id": "test_exp",
            "config": {"datasets": ["persona_chat"]},
            "started_at": "2026-01-01T00:00:00",
            "completed_at": "2026-01-01T01:00:00",
            "aggregated_metrics": {
                "dki": {"latency_p50": 100, "latency_p95": 200, "avg_memories": 5, "cache_hit_rate": 0.7},
            },
            "results_by_mode": {
                "dki": {
                    "samples": [
                        {
                            "latency_ms": 100,
                            "response": "Hello world",
                            "alpha": 0.4,
                            "cache_hit": False,
                            "memories_used": ["m1", "m2"],
                            "injection_info": {"preference_tokens": 30, "history_tokens": 200},
                        },
                        {
                            "latency_ms": 80,
                            "response": "Hi there",
                            "alpha": 0.4,
                            "cache_hit": True,
                            "memories_used": ["m1"],
                            "injection_info": {"preference_tokens": 30, "history_tokens": 150},
                        },
                    ]
                },
                "baseline": {
                    "samples": [
                        {"latency_ms": 50, "response": "OK"},
                    ]
                },
            },
        }
        
        result = analyze_persona_chat_experiment(data)
        
        self.assertEqual(result["experiment_id"], "test_exp")
        self.assertIn("dki", result["mode_analysis"])
        self.assertIn("baseline", result["mode_analysis"])
        
        dki = result["mode_analysis"]["dki"]
        self.assertEqual(dki["total_samples"], 2)
        self.assertEqual(dki["latency_stats"]["mean"], 90.0)
        self.assertIn("dki_specific", dki)
        self.assertEqual(dki["dki_specific"]["alpha_mean"], 0.4)
        self.assertEqual(dki["dki_specific"]["cache_hit_rate"], 0.5)
    
    def test_first_vs_subsequent_latency(self):
        """测试首轮 vs 后续轮延迟分析"""
        data = {
            "experiment_id": "test",
            "config": {},
            "started_at": "",
            "completed_at": "",
            "aggregated_metrics": {},
            "results_by_mode": {
                "dki": {
                    "samples": [
                        {"latency_ms": 100, "response": "a" * 100, "alpha": 0.4, "cache_hit": False,
                         "memories_used": [], "injection_info": {"preference_tokens": 0, "history_tokens": 0}},
                        {"latency_ms": 50, "response": "b" * 100, "alpha": 0.4, "cache_hit": True,
                         "memories_used": [], "injection_info": {"preference_tokens": 0, "history_tokens": 0}},
                    ]
                }
            },
        }
        
        result = analyze_persona_chat_experiment(data)
        dki_spec = result["mode_analysis"]["dki"]["dki_specific"]
        
        self.assertEqual(dki_spec["first_turn_latency_mean"], 100.0)
        self.assertEqual(dki_spec["subsequent_turn_latency_mean"], 50.0)
        # 归一化: 100/100=1.0 vs 50/100=0.5, reduction = (1-0.5/1.0)*100 = 50%
        self.assertEqual(dki_spec["normalized_latency_reduction_pct"], 50.0)


class TestAnalyzeLongmemevalExperiment(unittest.TestCase):
    """测试 LongMemEval 实验结果分析"""
    
    def test_none_input(self):
        """测试 None 输入"""
        result = analyze_longmemeval_experiment(None)
        self.assertIsNone(result)
    
    def test_basic_analysis(self):
        """测试基本分析"""
        data = {
            "benchmark": "longmemeval",
            "config": {"max_samples": 10},
            "started_at": "2026-01-01T00:00:00",
            "summary": {
                "longmemeval_multi_turn": {
                    "dki": {"keyword_recall": 0.7, "answer_match": 0.7},
                    "rag": {"keyword_recall": 0.9, "answer_match": 0.9},
                }
            },
            "results_by_dataset": {
                "longmemeval_multi_turn": {
                    "dki": {
                        "metrics": {"keyword_recall": 0.7},
                        "samples": [
                            {"answer_match": 1.0, "keyword_recall": 1.0, "question_type": "factual"},
                            {"answer_match": 0.0, "keyword_recall": 0.0, "question_type": "factual"},
                            {"answer_match": 0.5, "keyword_recall": 0.5, "question_type": "temporal"},
                        ],
                    },
                    "rag": {
                        "metrics": {"keyword_recall": 0.9},
                        "samples": [
                            {"answer_match": 1.0, "keyword_recall": 1.0, "question_type": "factual"},
                        ],
                    },
                }
            },
        }
        
        result = analyze_longmemeval_experiment(data)
        
        self.assertEqual(result["benchmark"], "longmemeval")
        self.assertIn("longmemeval_multi_turn", result["detailed"])
        
        dki_detail = result["detailed"]["longmemeval_multi_turn"]["dki"]
        self.assertEqual(dki_detail["sample_count"], 3)
        self.assertEqual(dki_detail["accuracy_breakdown"]["correct"], 1)
        self.assertEqual(dki_detail["accuracy_breakdown"]["partial"], 1)
        self.assertEqual(dki_detail["accuracy_breakdown"]["wrong"], 1)
        
        # 按问题类型分析
        self.assertIn("factual", dki_detail["by_question_type"])
        self.assertEqual(dki_detail["by_question_type"]["factual"]["count"], 2)


class TestComparePaperPredictions(unittest.TestCase):
    """测试论文预测对比数据"""
    
    def test_predictions_structure(self):
        """测试预测数据结构完整性"""
        predictions = compare_with_paper_predictions()
        
        self.assertIn("table1_core", predictions)
        self.assertIn("table1b_hallucination", predictions)
        self.assertIn("table2_context_constrained", predictions)
        self.assertIn("table3_ablation", predictions)
        self.assertIn("table4_alpha_sensitivity", predictions)
    
    def test_table1_core_values(self):
        """测试表1核心值"""
        predictions = compare_with_paper_predictions()
        core = predictions["table1_core"]
        
        self.assertEqual(core["rag_memory_recall"], 87.3)
        self.assertEqual(core["dki_memory_recall"], 86.2)
        self.assertEqual(core["dki_cache_hit_rate"], 69.7)
    
    def test_alpha_sensitivity_range(self):
        """测试 Alpha 敏感性值范围"""
        predictions = compare_with_paper_predictions()
        alpha = predictions["table4_alpha_sensitivity"]
        
        self.assertEqual(alpha["optimal_range"], [0.4, 0.7])
        self.assertIn(0.0, alpha["results"])
        self.assertIn(1.0, alpha["results"])


class TestGenerateReport(unittest.TestCase):
    """测试报告生成函数"""
    
    def test_report_generation_with_minimal_data(self):
        """测试最小数据下的报告生成"""
        dataset_stats = {
            "persona_chat": {"count": 10, "total_sessions": 10, "total_turns": 50, "avg_turns_per_session": 5.0},
        }
        persona_analysis = None
        longmemeval_analysis = None
        paper_predictions = compare_with_paper_predictions()
        
        report = generate_report(dataset_stats, persona_analysis, longmemeval_analysis, paper_predictions)
        
        self.assertIsInstance(report, str)
        self.assertIn("DKI (Dynamic KV Injection) 实验报告", report)
        self.assertIn("实验概述", report)
        self.assertIn("实验数据集", report)
    
    def test_report_contains_all_sections(self):
        """测试报告包含所有主要章节"""
        dataset_stats = {"persona_chat": {"count": 10, "total_sessions": 10, "total_turns": 50, "avg_turns_per_session": 5.0}}
        paper_predictions = compare_with_paper_predictions()
        
        report = generate_report(dataset_stats, None, None, paper_predictions)
        
        expected_sections = [
            "1. 实验概述",
            "2. 实验数据集",
            "3. PersonaChat 实验结果",
            "4. LongMemEval 多轮记忆评估结果",
            "5. 与论文理论预测的对比分析",
            "6. 实验结果评价与讨论",
            "7. 结论与建议",
        ]
        for section in expected_sections:
            self.assertIn(section, report, f"报告缺少章节: {section}")
    
    def test_report_with_full_data(self):
        """测试完整数据下的报告生成"""
        dataset_stats = {
            "ablation": {"count": 50, "ablation_modes": ["full_dki", "no_gating"], "unique_queries": 5, "queries": ["q1", "q2"]},
            "alpha_sensitivity": {"count": 300, "alpha_values": [0.0, 0.5, 1.0], "unique_queries": 3},
            "context_constrained": {"count": 210, "context_budgets": [4096], "memory_lengths": [2000, 3000]},
            "persona_chat": {"count": 100, "total_sessions": 100, "total_turns": 500, "avg_turns_per_session": 5.0},
        }
        
        persona_analysis = {
            "experiment_id": "test",
            "config": {"datasets": ["persona_chat"], "max_samples": 10, "temperature": 0.7, "force_alpha": 0.4},
            "started_at": "2026-01-01",
            "completed_at": "2026-01-01",
            "aggregated_metrics": {
                "dki": {"latency_p50": 100, "latency_p95": 200, "avg_memories": 5, "cache_hit_rate": 0.7},
                "rag": {"latency_p50": 80, "latency_p95": 150, "avg_memories": 3},
                "baseline": {"latency_p50": 50, "latency_p95": 100, "avg_memories": 0},
            },
            "mode_analysis": {
                "dki": {
                    "total_samples": 50,
                    "latency_stats": {"mean": 100, "median": 90, "stdev": 20, "min": 50, "max": 200},
                    "response_length_stats": {"mean": 2500, "median": 2400, "min": 500, "max": 4000},
                    "dki_specific": {
                        "alpha_mean": 0.4,
                        "cache_hit_rate": 0.72,
                        "avg_memories_used": 5,
                        "avg_preference_tokens": 35,
                        "avg_history_tokens": 2000,
                        "first_turn_latency_mean": 120,
                        "subsequent_turn_latency_mean": 90,
                        "latency_reduction_raw_pct": 25.0,
                        "first_turn_ms_per_char": 8.5,
                        "subsequent_ms_per_char": 7.2,
                        "normalized_latency_reduction_pct": 15.3,
                        "first_turn_avg_resp_len": 1500,
                        "subsequent_avg_resp_len": 2800,
                    },
                },
                "rag": {
                    "total_samples": 50,
                    "latency_stats": {"mean": 80, "median": 75, "stdev": 15, "min": 40, "max": 150},
                    "response_length_stats": {"mean": 2600, "median": 2500, "min": 600, "max": 4500},
                    "rag_specific": {"avg_memories_used": 3},
                },
                "baseline": {
                    "total_samples": 50,
                    "latency_stats": {"mean": 50, "median": 45, "stdev": 10, "min": 20, "max": 100},
                    "response_length_stats": {"mean": 2000, "median": 1900, "min": 300, "max": 3500},
                },
            },
        }
        
        longmemeval_analysis = {
            "benchmark": "longmemeval",
            "config": {"max_samples": 10, "force_alpha": 0.4},
            "started_at": "2026-01-01",
            "summary": {
                "longmemeval_multi_turn": {
                    "dki": {"keyword_recall": 0.7, "answer_match": 0.7, "rouge_l": 0.1, "latency_p50": 1700, "valid_samples": 10},
                    "rag": {"keyword_recall": 0.95, "answer_match": 0.92, "rouge_l": 0.19, "latency_p50": 1700, "valid_samples": 10},
                    "baseline": {"keyword_recall": 0.15, "answer_match": 0.2, "rouge_l": 0.01, "latency_p50": 3400, "valid_samples": 10},
                }
            },
            "detailed": {
                "longmemeval_multi_turn": {
                    "dki": {"metrics": {}, "sample_count": 10, "accuracy_breakdown": {"correct": 6, "partial": 2, "wrong": 2, "accuracy_rate": 60.0}},
                    "rag": {"metrics": {}, "sample_count": 10, "accuracy_breakdown": {"correct": 8, "partial": 2, "wrong": 0, "accuracy_rate": 80.0}},
                    "baseline": {"metrics": {}, "sample_count": 10, "accuracy_breakdown": {"correct": 1, "partial": 3, "wrong": 6, "accuracy_rate": 10.0}},
                }
            },
        }
        
        paper_predictions = compare_with_paper_predictions()
        report = generate_report(dataset_stats, persona_analysis, longmemeval_analysis, paper_predictions)
        
        # 验证报告包含关键数据
        self.assertIn("72.0%", report)  # Cache 命中率
        self.assertIn("归一化延迟", report)
        self.assertIn("LongMemEval", report)
        self.assertIn("消融实验", report)
        self.assertIn("Alpha 敏感性", report)
        self.assertIn("结论与建议", report)
        
        # 验证报告长度合理
        self.assertGreater(len(report), 5000)


class TestReportMarkdownFormat(unittest.TestCase):
    """测试报告 Markdown 格式正确性"""
    
    def test_table_format(self):
        """测试表格格式"""
        dataset_stats = {"persona_chat": {"count": 10, "total_sessions": 10, "total_turns": 50, "avg_turns_per_session": 5.0}}
        paper_predictions = compare_with_paper_predictions()
        report = generate_report(dataset_stats, None, None, paper_predictions)
        
        # 检查表格分隔符
        lines = report.split("\n")
        table_separators = [l for l in lines if l.startswith("| ---")]
        self.assertGreater(len(table_separators), 0, "报告中应包含表格")
        
        # 每个表格分隔符前应有表头
        for i, line in enumerate(lines):
            if line.startswith("| ---"):
                self.assertTrue(i > 0 and lines[i-1].startswith("|"), 
                              f"表格分隔符前应有表头 (行 {i})")
    
    def test_heading_hierarchy(self):
        """测试标题层级"""
        dataset_stats = {"persona_chat": {"count": 10, "total_sessions": 10, "total_turns": 50, "avg_turns_per_session": 5.0}}
        paper_predictions = compare_with_paper_predictions()
        report = generate_report(dataset_stats, None, None, paper_predictions)
        
        lines = report.split("\n")
        h1_count = sum(1 for l in lines if l.startswith("# ") and not l.startswith("## "))
        h3_count = sum(1 for l in lines if l.startswith("### "))
        
        self.assertGreater(h1_count, 0, "报告应包含一级标题")


if __name__ == "__main__":
    unittest.main(verbosity=2)
