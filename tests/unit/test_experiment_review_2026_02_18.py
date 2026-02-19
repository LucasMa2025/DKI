"""
单元测试: 2026-02-18 实验系统与演示系统审查修复验证

审查发现的问题:
1. /api/experiment/run 未传递共享的 dki_system/rag_system 给 ExperimentRunner
2. _match_user_by_personas 逐字符匹配而非按词匹配
3. data_generator.py 有重复的 generate_all 方法 (死代码已移除)

测试策略:
- 使用 Mock 对象，不需要 GPU 或真实数据库
- 验证每个修复点的正确行为
"""

import os
import sys
import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# BUG 1: /api/experiment/run 应传递共享系统实例
# ============================================================================

class TestExperimentRunEndpoint:
    """验证 /api/experiment/run 端点使用共享的 dki_system 和 rag_system"""

    def test_run_endpoint_passes_shared_systems(self):
        """
        /api/experiment/run 应将 get_dki_system() 和 get_rag_system()
        的返回值传递给 ExperimentRunner, 而非创建新实例。
        
        验证方法: 检查 app.py 源码中 ExperimentRunner 的构造调用。
        """
        import inspect
        from dki.web.app import create_app
        
        # 获取 create_app 的源代码
        source = inspect.getsource(create_app)
        
        # 验证 /api/experiment/run 端点中 ExperimentRunner 使用了共享系统
        # 查找 run_experiment 函数体中的 ExperimentRunner 构造
        # 它应该包含 dki_system= 和 rag_system= 参数
        assert "dki = get_dki_system()" in source, \
            "/api/experiment/run 应调用 get_dki_system()"
        assert "rag = get_rag_system()" in source, \
            "/api/experiment/run 应调用 get_rag_system()"
        
        # 验证 ExperimentRunner 构造时传递了系统实例
        # 在 run_experiment 函数中应该有 ExperimentRunner(dki_system=dki, rag_system=rag)
        assert "ExperimentRunner(" in source
        
        # 确保不再有无参数的 ExperimentRunner() 调用 (排除 run-persona-chat 之外)
        # 简单检查: run_experiment 函数中的 ExperimentRunner 应带参数
        lines = source.split('\n')
        in_run_experiment = False
        found_runner_with_args = False
        for line in lines:
            if 'async def run_experiment' in line:
                in_run_experiment = True
            elif in_run_experiment and 'async def ' in line:
                break
            elif in_run_experiment and 'ExperimentRunner(' in line:
                # 应该不是空参数
                if 'ExperimentRunner()' not in line:
                    found_runner_with_args = True
        
        assert found_runner_with_args, \
            "run_experiment 中的 ExperimentRunner 应传递 dki_system 和 rag_system"

    def test_persona_chat_endpoint_also_passes_shared_systems(self):
        """验证 /api/experiment/run-persona-chat 也使用共享系统"""
        import inspect
        from dki.web.app import create_app
        
        source = inspect.getsource(create_app)
        
        # run-persona-chat 端点应该也使用 get_dki_system() 和 get_rag_system()
        lines = source.split('\n')
        in_persona_chat = False
        uses_get_dki = False
        uses_get_rag = False
        for line in lines:
            if 'async def run_persona_chat_experiment' in line:
                in_persona_chat = True
            elif in_persona_chat and 'async def ' in line:
                break
            elif in_persona_chat:
                if 'get_dki_system()' in line:
                    uses_get_dki = True
                if 'get_rag_system()' in line:
                    uses_get_rag = True
        
        assert uses_get_dki, "run-persona-chat 应使用 get_dki_system()"
        assert uses_get_rag, "run-persona-chat 应使用 get_rag_system()"


# ============================================================================
# BUG 2: _match_user_by_personas 应按词匹配而非逐字符
# ============================================================================

class TestMatchUserByPersonas:
    """验证 _match_user_by_personas 按词匹配"""

    def setup_method(self):
        """创建一个 mock ExperimentRunner 用于测试"""
        with patch('dki.experiment.runner.ConfigLoader'), \
             patch('dki.experiment.runner.DatabaseManager'):
            self.runner = self._create_runner()
    
    def _create_runner(self):
        from dki.experiment.runner import ExperimentRunner
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner.config = MagicMock()
        runner.config.experiment = None
        runner.db_manager = MagicMock()
        runner._experiment_user_map = {
            "exp_user_vegetarian": "user_001",
            "exp_user_outdoor": "user_002",
            "exp_user_tech": "user_003",
        }
        return runner

    def test_word_matching_not_char_matching(self):
        """
        修复前: 逐字符匹配 'for word in pref_text' 导致每个字符都被匹配
        修复后: 按词分割后匹配, 中文词和英文词都正确处理
        """
        from dki.experiment.runner import ExperimentRunner
        
        # 使用包含明确素食关键词的 personas
        personas = ["我是素食主义者，不吃肉类", "我对海鲜过敏"]
        
        result = self.runner._match_user_by_personas(personas)
        
        # 应该匹配到 vegetarian 用户 (因为偏好中包含素食相关词)
        assert result == "user_001", \
            f"应匹配到 vegetarian 用户, 但得到 {result}"

    def test_no_false_positive_char_match(self):
        """
        验证不会因为单字符匹配而产生误匹配。
        
        修复前: '我' 这个字符在所有偏好和 personas 中都出现,
        导致所有用户得分相同, 无法区分。
        修复后: 按词匹配, 只有有意义的词才计分。
        """
        # 技术相关的 personas
        personas = ["我是一名数据科学家", "我擅长Python和机器学习"]
        
        result = self.runner._match_user_by_personas(personas)
        
        # 应该匹配到 tech 用户
        assert result == "user_003", \
            f"应匹配到 tech 用户, 但得到 {result}"

    def test_empty_personas(self):
        """空 personas 列表应返回第一个实验用户"""
        result = self.runner._match_user_by_personas([])
        # 空 personas 文本不会匹配任何关键词, 应返回第一个用户
        first_user_id = list(self.runner._experiment_user_map.values())[0]
        assert result == first_user_id

    def test_no_experiment_user_map(self):
        """无实验用户映射时应返回默认用户"""
        self.runner._experiment_user_map = {}
        result = self.runner._match_user_by_personas(["素食"])
        assert result == "experiment_user"  # 默认值

    def test_word_extraction_regex(self):
        """验证词提取正则表达式正确工作"""
        text = "我是素食主义者，不吃任何肉类和seafood"
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
        
        # 应提取中文词和英文词
        assert "我是素食主义者" in words or any("素食" in w for w in words)
        assert "seafood" in words
        # 不应包含标点符号
        assert "，" not in words


# ============================================================================
# BUG 3: data_generator.py 重复 generate_all 方法 (死代码移除)
# ============================================================================

class TestDataGeneratorGenerateAll:
    """验证 generate_all 方法唯一且支持所有参数"""

    def test_generate_all_accepts_extended_params(self):
        """generate_all 应接受 include_chinese, include_advanced, include_long_sessions"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import inspect
        
        sig = inspect.signature(ExperimentDataGenerator.generate_all)
        params = list(sig.parameters.keys())
        
        assert 'include_chinese' in params, \
            "generate_all 应接受 include_chinese 参数"
        assert 'include_advanced' in params, \
            "generate_all 应接受 include_advanced 参数"
        assert 'include_long_sessions' in params, \
            "generate_all 应接受 include_long_sessions 参数"

    def test_no_duplicate_generate_all(self):
        """验证 generate_all 方法没有重复定义"""
        import inspect
        from dki.experiment.data_generator import ExperimentDataGenerator
        
        # 获取源代码
        source = inspect.getsource(ExperimentDataGenerator)
        
        # 计算 'def generate_all' 出现的次数
        count = source.count('def generate_all')
        assert count == 1, \
            f"generate_all 应只定义一次, 但找到 {count} 次"

    def test_generate_all_returns_extended_datasets(self):
        """generate_all(include_chinese=True, ...) 应返回扩展数据集"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            result = gen.generate_all(
                persona_sessions=2,
                hotpot_samples=2,
                memory_qa_samples=2,
                include_chinese=True,
                include_advanced=True,
                include_long_sessions=True,
            )
            
            # 基础数据集
            assert 'persona_chat' in result
            assert 'hotpot_qa' in result
            assert 'memory_qa' in result
            
            # 扩展数据集
            assert 'cn_persona_chat' in result, "应包含中文 PersonaChat"
            assert 'multi_turn_coherence' in result, "应包含多轮连贯性数据"
            assert 'ablation' in result, "应包含消融实验数据"
            assert 'long_session_persona_chat' in result, "应包含长会话数据"

    def test_generate_all_without_extended(self):
        """generate_all(include_chinese=False, ...) 应只返回基础数据集"""
        from dki.experiment.data_generator import ExperimentDataGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ExperimentDataGenerator(tmpdir)
            result = gen.generate_all(
                persona_sessions=2,
                hotpot_samples=2,
                memory_qa_samples=2,
                include_chinese=False,
                include_advanced=False,
                include_long_sessions=False,
            )
            
            # 只有基础数据集
            assert 'persona_chat' in result
            assert 'hotpot_qa' in result
            assert 'memory_qa' in result
            
            # 不应有扩展数据集
            assert 'cn_persona_chat' not in result
            assert 'long_session_persona_chat' not in result


# ============================================================================
# 实验系统使用 DKISystem 组件的验证
# ============================================================================

class TestExperimentUseDKISystem:
    """验证实验系统确实使用 DKISystem 组件"""

    def test_runner_stores_dki_system(self):
        """ExperimentRunner 应存储传入的 dki_system"""
        from dki.experiment.runner import ExperimentRunner
        
        mock_dki = MagicMock()
        mock_rag = MagicMock()
        
        with patch('dki.experiment.runner.ConfigLoader'), \
             patch('dki.experiment.runner.DatabaseManager'):
            runner = ExperimentRunner(dki_system=mock_dki, rag_system=mock_rag)
        
        assert runner.dki_system is mock_dki
        assert runner.rag_system is mock_rag

    def test_runner_lazy_creates_systems(self):
        """ExperimentRunner._ensure_systems 应在需要时创建系统"""
        from dki.experiment.runner import ExperimentRunner
        
        with patch('dki.experiment.runner.ConfigLoader'), \
             patch('dki.experiment.runner.DatabaseManager'), \
             patch('dki.experiment.runner.DKISystem') as MockDKI, \
             patch('dki.experiment.runner.RAGSystem') as MockRAG:
            
            runner = ExperimentRunner()
            assert runner.dki_system is None
            assert runner.rag_system is None
            
            runner._ensure_systems()
            
            assert runner.dki_system is not None
            assert runner.rag_system is not None
            MockDKI.assert_called_once()
            MockRAG.assert_called_once()

    def test_runner_calls_dki_chat(self):
        """实验运行时应调用 DKISystem.chat()"""
        from dki.experiment.runner import ExperimentRunner, ExperimentConfig
        
        mock_dki = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_response.latency_ms = 100.0
        mock_response.memories_used = []
        mock_response.cache_hit = False
        mock_response.gating_decision.alpha = 0.5
        mock_response.gating_decision.should_inject = True
        mock_response.metadata = {'hybrid_injection': {}}
        mock_dki.chat.return_value = mock_response
        
        mock_rag = MagicMock()
        
        with patch('dki.experiment.runner.ConfigLoader'), \
             patch('dki.experiment.runner.DatabaseManager'):
            runner = ExperimentRunner(dki_system=mock_dki, rag_system=mock_rag)
        
        config = ExperimentConfig(name="test", max_new_tokens=64)
        
        item = {
            'id': 'test_001',
            'query': 'hello',
            '_dataset': 'test',
        }
        
        result = runner._run_single_query(
            mode='dki',
            query='hello',
            session_id='test_session',
            item=item,
            config=config,
        )
        
        mock_dki.chat.assert_called_once()
        assert result.mode == 'dki'
        assert result.response == "test response"

    def test_runner_calls_rag_chat(self):
        """实验运行时应调用 RAGSystem.chat()"""
        from dki.experiment.runner import ExperimentRunner, ExperimentConfig
        
        mock_dki = MagicMock()
        mock_rag = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "rag response"
        mock_response.latency_ms = 80.0
        mock_response.memories_used = []
        mock_response.prompt_info = None
        mock_rag.chat.return_value = mock_response
        
        with patch('dki.experiment.runner.ConfigLoader'), \
             patch('dki.experiment.runner.DatabaseManager'):
            runner = ExperimentRunner(dki_system=mock_dki, rag_system=mock_rag)
        
        config = ExperimentConfig(name="test", max_new_tokens=64)
        
        item = {
            'id': 'test_001',
            'query': 'hello',
            '_dataset': 'test',
        }
        
        result = runner._run_single_query(
            mode='rag',
            query='hello',
            session_id='test_session',
            item=item,
            config=config,
        )
        
        mock_rag.chat.assert_called_once()
        assert result.mode == 'rag'
        assert result.response == "rag response"


# ============================================================================
# Metrics 计算正确性
# ============================================================================

class TestMetricsCalculator:
    """验证 MetricsCalculator 的基本正确性"""

    def test_latency_stats_empty(self):
        """空延迟列表应返回全零"""
        from dki.experiment.metrics import MetricsCalculator
        mc = MetricsCalculator()
        stats = mc.compute_latency_stats([])
        assert stats['p50'] == 0
        assert stats['mean'] == 0

    def test_latency_stats_normal(self):
        """正常延迟列表应返回合理统计"""
        from dki.experiment.metrics import MetricsCalculator
        mc = MetricsCalculator()
        latencies = [100, 200, 300, 400, 500]
        stats = mc.compute_latency_stats(latencies)
        assert stats['mean'] == 300.0
        assert stats['min'] == 100.0
        assert stats['max'] == 500.0
        assert stats['p50'] == 300.0

    def test_memory_recall_empty(self):
        """空记忆列表应返回 1.0"""
        from dki.experiment.metrics import MetricsCalculator
        mc = MetricsCalculator()
        recall, matched = mc.compute_memory_recall([], "some response")
        assert recall == 1.0
        assert matched == []

    def test_memory_recall_with_match(self):
        """有匹配记忆时应返回正确召回率"""
        from dki.experiment.metrics import MetricsCalculator
        mc = MetricsCalculator()
        memories = ["User likes vegetarian food"]
        response = "Based on your preference for vegetarian food, I recommend..."
        recall, matched = mc.compute_memory_recall(memories, response, threshold=0.3)
        assert recall > 0.0

    def test_hallucination_rate_grounded(self):
        """完全有依据的回复应有低幻觉率"""
        from dki.experiment.metrics import MetricsCalculator
        mc = MetricsCalculator()
        response = "Beijing is the capital of China."
        grounding = ["Beijing is the capital of China."]
        rate, hallucinations = mc.compute_hallucination_rate(response, grounding)
        assert rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
