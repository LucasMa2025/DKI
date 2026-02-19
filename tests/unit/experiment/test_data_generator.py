"""
ExperimentDataGenerator 单元测试

测试实验数据生成器:
- PersonaChat 数据生成 (英文/中文)
- HotpotQA 数据生成
- MemoryQA 数据生成
- α 敏感度数据生成
- 多轮连贯性数据生成
- 消融实验数据生成
- 实验用户分配逻辑
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from dki.experiment.data_generator import ExperimentDataGenerator


class TestExperimentDataGenerator:
    """ExperimentDataGenerator 测试"""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        """使用临时目录避免污染真实文件系统"""
        self.output_dir = str(tmp_path / "test_data")
        self.generator = ExperimentDataGenerator(output_dir=self.output_dir)

    # ============ 初始化测试 ============

    def test_init_creates_directory(self):
        assert Path(self.output_dir).exists()

    def test_init_custom_path(self, tmp_path):
        custom = str(tmp_path / "custom_data")
        gen = ExperimentDataGenerator(output_dir=custom)
        assert Path(custom).exists()

    # ============ PersonaChat 测试 ============

    def test_generate_persona_chat_basic(self):
        """基本 PersonaChat 数据生成"""
        data = self.generator.generate_persona_chat(n_sessions=5, n_turns_per_session=3)
        
        assert len(data) == 5
        assert all('session_id' in d for d in data)
        assert all('personas' in d for d in data)
        assert all('turns' in d for d in data)
        assert all('experiment_user' in d for d in data)

    def test_persona_chat_session_structure(self):
        """PersonaChat 会话结构验证"""
        data = self.generator.generate_persona_chat(
            n_sessions=4, n_turns_per_session=2, n_personas_per_session=2
        )
        
        for session in data:
            assert len(session['personas']) == 2
            assert len(session['turns']) == 2
            
            for turn in session['turns']:
                assert 'turn_id' in turn
                assert 'query' in turn
                assert 'expected_keywords' in turn
                assert 'relevant_memories' in turn

    def test_persona_chat_experiment_user_assignment(self):
        """实验用户应轮流分配"""
        data = self.generator.generate_persona_chat(n_sessions=8)
        
        users = [d['experiment_user'] for d in data]
        unique_users = set(users)
        
        # 应有 4 个不同的实验用户
        assert len(unique_users) == 4
        
        # 前 4 个应各不相同
        assert len(set(users[:4])) == 4

    def test_persona_chat_saves_file(self):
        """应保存到 JSON 文件"""
        self.generator.generate_persona_chat(n_sessions=2)
        
        filepath = Path(self.output_dir) / 'persona_chat.json'
        assert filepath.exists()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert len(loaded) == 2

    def test_persona_chat_metadata(self):
        """元数据应正确"""
        data = self.generator.generate_persona_chat(n_sessions=1)
        
        meta = data[0]['metadata']
        assert meta['dataset'] == 'persona_chat'
        assert 'generated_at' in meta
        assert 'experiment_user' in meta

    # ============ 中文 PersonaChat 测试 ============

    def test_generate_chinese_persona_chat(self):
        """中文 PersonaChat 数据生成"""
        data = self.generator.generate_chinese_persona_chat(n_sessions=4)
        
        assert len(data) == 4
        for session in data:
            assert session['metadata']['dataset'] == 'cn_persona_chat'
            assert session['metadata']['language'] == 'zh'
            assert 'experiment_user' in session
            # 验证中文内容
            assert any('\u4e00' <= c <= '\u9fff' for p in session['personas'] for c in p)

    def test_chinese_persona_chat_user_assignment(self):
        """中文 PersonaChat 也应轮流分配实验用户"""
        data = self.generator.generate_chinese_persona_chat(n_sessions=8)
        users = set(d['experiment_user'] for d in data)
        assert len(users) == 4

    # ============ HotpotQA 测试 ============

    def test_generate_hotpot_qa_basic(self):
        """基本 HotpotQA 数据生成"""
        data = self.generator.generate_hotpot_qa(n_samples=10)
        
        assert len(data) == 10
        for sample in data:
            assert 'id' in sample
            assert 'question' in sample
            assert 'supporting_facts' in sample
            assert 'expected_answer' in sample
            assert 'entity_values' in sample

    def test_hotpot_qa_structure(self):
        """HotpotQA 数据结构验证"""
        data = self.generator.generate_hotpot_qa(n_samples=5)
        
        for sample in data:
            assert sample['id'].startswith('hotpot_')
            assert len(sample['supporting_facts']) == 2
            assert sample['expected_answer'] != ""
            assert sample['metadata']['dataset'] == 'hotpot_qa'

    def test_hotpot_qa_saves_file(self):
        """应保存到 JSON 文件"""
        self.generator.generate_hotpot_qa(n_samples=3)
        
        filepath = Path(self.output_dir) / 'hotpot_qa.json'
        assert filepath.exists()

    # ============ MemoryQA 测试 ============

    def test_generate_memory_qa_basic(self):
        """基本 MemoryQA 数据生成"""
        data = self.generator.generate_memory_qa(n_samples=10)
        
        assert len(data) == 10
        for sample in data:
            assert 'id' in sample
            assert 'memory' in sample
            assert 'query' in sample
            assert 'expected_memory_use' in sample
            assert 'filled_vars' in sample

    def test_memory_qa_structure(self):
        """MemoryQA 数据结构验证"""
        data = self.generator.generate_memory_qa(n_samples=5)
        
        for sample in data:
            assert sample['id'].startswith('memqa_')
            assert sample['expected_memory_use'] is True
            assert sample['metadata']['dataset'] == 'memory_qa'
            # 模板变量应已填充
            assert '{' not in sample['memory']

    def test_memory_qa_saves_file(self):
        """应保存到 JSON 文件"""
        self.generator.generate_memory_qa(n_samples=3)
        
        filepath = Path(self.output_dir) / 'memory_qa.json'
        assert filepath.exists()

    # ============ α 敏感度数据测试 ============

    def test_generate_alpha_sensitivity(self):
        """α 敏感度数据生成"""
        data = self.generator.generate_alpha_sensitivity_data(n_samples=5)
        
        # 5 samples × 6 alpha values = 30
        assert len(data) == 30
        
        for sample in data:
            assert 'id' in sample
            assert 'memory' in sample
            assert 'query' in sample
            assert 'alpha' in sample
            assert 0.0 <= sample['alpha'] <= 1.0

    def test_alpha_sensitivity_all_values(self):
        """应包含所有 α 值"""
        data = self.generator.generate_alpha_sensitivity_data(n_samples=1)
        
        alphas = sorted(set(d['alpha'] for d in data))
        expected = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert alphas == expected

    def test_alpha_sensitivity_saves_file(self):
        """应保存到 JSON 文件"""
        self.generator.generate_alpha_sensitivity_data(n_samples=2)
        
        filepath = Path(self.output_dir) / 'alpha_sensitivity.json'
        assert filepath.exists()

    # ============ 多轮连贯性数据测试 ============

    def test_generate_multi_turn_coherence(self):
        """多轮连贯性数据生成"""
        data = self.generator.generate_multi_turn_coherence(n_sessions=6)
        
        assert len(data) == 6
        for session in data:
            assert 'session_id' in session
            assert 'personas' in session
            assert 'turns' in session
            assert session['metadata']['dataset'] == 'multi_turn_coherence'

    def test_multi_turn_coherence_structure(self):
        """多轮连贯性数据结构验证"""
        data = self.generator.generate_multi_turn_coherence(n_sessions=3)
        
        for session in data:
            assert len(session['personas']) >= 2
            assert len(session['turns']) >= 3
            
            for turn in session['turns']:
                assert 'query' in turn
                assert 'tests_memory' in turn
                if turn['tests_memory']:
                    assert 'expected_recall' in turn
                    assert len(turn['expected_recall']) > 0

    def test_multi_turn_coherence_scenarios_cycle(self):
        """场景应循环使用"""
        data = self.generator.generate_multi_turn_coherence(n_sessions=6)
        # 3 个场景循环
        scenario_indices = [d['metadata']['scenario_idx'] for d in data]
        assert scenario_indices == [0, 1, 2, 0, 1, 2]

    # ============ 消融实验数据测试 ============

    def test_generate_ablation_data(self):
        """消融实验数据生成"""
        data = self.generator.generate_ablation_data(n_samples=10)
        
        assert len(data) == 10
        for sample in data:
            assert 'id' in sample
            assert 'memory' in sample
            assert 'query' in sample
            assert 'relevant_memories' in sample
            assert 'ablation_modes' in sample
            assert 'all_memories' in sample

    def test_ablation_data_modes(self):
        """消融实验应包含所有模式"""
        data = self.generator.generate_ablation_data(n_samples=1)
        
        expected_modes = [
            "full_dki", "no_gating", "no_history",
            "no_preference_kv", "rag_baseline", "no_memory",
        ]
        assert data[0]['ablation_modes'] == expected_modes

    def test_ablation_data_relevant_memories(self):
        """相关记忆应来自全部记忆列表"""
        data = self.generator.generate_ablation_data(n_samples=5)
        
        for sample in data:
            for mem in sample['relevant_memories']:
                assert mem in sample['all_memories']

    # ============ generate_all 测试 ============

    def test_generate_all_basic(self):
        """generate_all 应生成所有数据集"""
        result = self.generator.generate_all(
            persona_sessions=2,
            hotpot_samples=2,
            memory_qa_samples=2,
            include_chinese=True,
            include_advanced=True,
        )
        
        assert 'persona_chat' in result
        assert 'hotpot_qa' in result
        assert 'memory_qa' in result
        assert 'cn_persona_chat' in result
        assert 'multi_turn_coherence' in result
        assert 'ablation' in result

    def test_generate_all_without_chinese(self):
        """不包含中文数据"""
        result = self.generator.generate_all(
            persona_sessions=2,
            hotpot_samples=2,
            memory_qa_samples=2,
            include_chinese=False,
            include_advanced=False,
        )
        
        assert 'persona_chat' in result
        assert 'cn_persona_chat' not in result
        assert 'multi_turn_coherence' not in result

    # ============ 边界条件测试 ============

    def test_zero_samples(self):
        """零样本应返回空列表"""
        data = self.generator.generate_persona_chat(n_sessions=0)
        assert data == []

    def test_single_sample(self):
        """单样本生成"""
        data = self.generator.generate_memory_qa(n_samples=1)
        assert len(data) == 1

    def test_large_batch(self):
        """大批量生成不应崩溃"""
        data = self.generator.generate_persona_chat(n_sessions=200)
        assert len(data) == 200

    def test_persona_more_than_available(self):
        """请求的 persona 数量超过可用数量时应正常处理"""
        data = self.generator.generate_persona_chat(
            n_sessions=1,
            n_personas_per_session=20,  # 超过可用的 15 个
        )
        assert len(data) == 1
        # 应返回所有可用的 personas
        assert len(data[0]['personas']) <= 20

    # ============ 长会话 PersonaChat 测试 ============

    def test_generate_long_session_basic(self):
        """基本长会话 PersonaChat 数据生成"""
        data = self.generator.generate_long_session_persona_chat(
            n_sessions=4,
            n_turns_per_session=5,
        )
        
        assert len(data) == 4
        for session in data:
            assert 'session_id' in session
            assert 'personas' in session
            assert 'turns' in session
            assert 'experiment_user' in session
            assert session.get('session_type') == 'long'

    def test_long_session_structure(self):
        """长会话数据结构验证"""
        data = self.generator.generate_long_session_persona_chat(
            n_sessions=2,
            n_turns_per_session=3,
        )
        
        for session in data:
            assert len(session['personas']) >= 3  # 长会话场景有丰富的 personas
            assert len(session['turns']) <= 3  # 不超过请求的轮次
            
            for turn in session['turns']:
                assert 'turn_id' in turn
                assert 'query' in turn
                assert 'expected_keywords' in turn
                assert 'relevant_memories' in turn
                assert 'expected_length_range' in turn
                # 每轮查询应较长 (长会话特征)
                assert len(turn['query']) > 50

    def test_long_session_metadata(self):
        """长会话元数据应正确"""
        data = self.generator.generate_long_session_persona_chat(n_sessions=1)
        
        meta = data[0]['metadata']
        assert meta['dataset'] == 'long_session_persona_chat'
        assert meta['session_type'] == 'long'
        assert meta['language'] == 'zh'
        assert 'min_turn_length' in meta
        assert 'max_turn_length' in meta
        assert 'experiment_user' in meta
        assert 'generated_at' in meta

    def test_long_session_experiment_user_cycling(self):
        """长会话应循环分配实验用户 (4 个场景)"""
        data = self.generator.generate_long_session_persona_chat(n_sessions=8)
        
        users = [d['experiment_user'] for d in data]
        unique_users = set(users)
        
        # 有 4 个场景模板
        assert len(unique_users) == 4

    def test_long_session_saves_file(self):
        """长会话应保存到 JSON 文件"""
        self.generator.generate_long_session_persona_chat(n_sessions=2)
        
        filepath = Path(self.output_dir) / 'long_session_persona_chat.json'
        assert filepath.exists()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert len(loaded) == 2

    def test_long_session_turn_limit(self):
        """n_turns_per_session 应限制每个会话的轮次"""
        data = self.generator.generate_long_session_persona_chat(
            n_sessions=1,
            n_turns_per_session=3,
        )
        
        # 场景模板有 8-10 轮, 但应被限制为 3 轮
        assert len(data[0]['turns']) == 3

    def test_long_session_relevant_memories(self):
        """每轮的 relevant_memories 应基于 expected_keywords 从 personas 中匹配"""
        data = self.generator.generate_long_session_persona_chat(n_sessions=1)
        
        session = data[0]
        personas_text = " ".join(session['personas']).lower()
        
        for turn in session['turns']:
            for mem in turn['relevant_memories']:
                # 相关记忆应来自 personas
                assert mem in session['personas']

    def test_long_session_chinese_content(self):
        """长会话应包含中文内容"""
        data = self.generator.generate_long_session_persona_chat(n_sessions=1)
        
        session = data[0]
        # personas 应包含中文
        assert any('\u4e00' <= c <= '\u9fff' for p in session['personas'] for c in p)
        # turns 的 query 应包含中文
        assert any(
            '\u4e00' <= c <= '\u9fff'
            for t in session['turns']
            for c in t['query']
        )

    def test_long_session_zero_sessions(self):
        """零会话应返回空列表"""
        data = self.generator.generate_long_session_persona_chat(n_sessions=0)
        assert data == []

    # ============ generate_all 更新测试 ============

    def test_generate_all_with_long_sessions(self):
        """generate_all 应包含长会话数据集"""
        result = self.generator.generate_all(
            persona_sessions=2,
            hotpot_samples=2,
            memory_qa_samples=2,
            include_chinese=True,
            include_advanced=True,
            include_long_sessions=True,
        )
        
        assert 'persona_chat' in result
        assert 'hotpot_qa' in result
        assert 'memory_qa' in result
        assert 'cn_persona_chat' in result
        assert 'multi_turn_coherence' in result
        assert 'ablation' in result
        assert 'long_session_persona_chat' in result

    def test_generate_all_without_long_sessions(self):
        """generate_all 可以排除长会话数据集"""
        result = self.generator.generate_all(
            persona_sessions=2,
            hotpot_samples=2,
            memory_qa_samples=2,
            include_chinese=False,
            include_advanced=False,
            include_long_sessions=False,
        )
        
        assert 'persona_chat' in result
        assert 'long_session_persona_chat' not in result
        assert 'cn_persona_chat' not in result
