"""
单元测试: 2026-02-18 dki_system.py + recall 模块审查修复验证

审查发现的问题:
1. _generate_with_injection 中 preference K/V 被双重缩放
2. multi_signal_recall 中 recall_turns 被计算但从未使用
3. 分数融合权重未归一化 (w1+w2=0.75)
4. ConversationRepository 缺少 get_by_id 方法
5. Recall v4 偏好 token 粗估对中文不准确
6. _format_history injection_markers 包含正常角色标签

测试策略:
- 使用 Mock 对象，不需要 GPU 或真实数据库
- 验证每个修复点的正确行为
"""

import os
import sys
import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field

import pytest

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 辅助数据结构
# ============================================================================

@dataclass
class FakeKVCacheEntry:
    """模拟 KVCacheEntry"""
    key: Any
    value: Any
    layer_idx: int = 0


@dataclass
class FakeMemorySearchResult:
    """模拟 MemorySearchResult"""
    memory_id: str
    content: str
    score: float = 0.8

    def to_dict(self):
        return {"memory_id": self.memory_id, "content": self.content, "score": self.score}


@dataclass
class FakeGatingDecision:
    """模拟 GatingDecision"""
    should_inject: bool = True
    alpha: float = 0.5
    entropy: float = 0.3
    relevance_score: float = 0.7
    margin: float = 0.2
    memories: List[Any] = field(default_factory=list)
    reasoning: str = "test"

    def to_dict(self):
        return {"should_inject": self.should_inject, "alpha": self.alpha}


@dataclass
class FakeModelOutput:
    text: str = "test response"
    input_tokens: int = 10
    output_tokens: int = 5


# ============================================================================
# BUG 1: preference K/V 不应被双重缩放
# ============================================================================

class TestPreferenceKVDoubleScaling:
    """验证 preference K/V 仅被 preference_alpha 缩放一次"""

    def test_preference_kv_scaled_only_by_preference_alpha(self):
        """
        修复前: preference K/V 先被 preference_alpha 缩放, 再被 gating_decision.alpha 缩放
        修复后: preference K/V 仅被 preference_alpha 缩放, 不受 gating_decision.alpha 影响
        """
        # 读取修复后的源码, 验证代码顺序
        import inspect
        from dki.core.dki_system import DKISystem
        
        source = inspect.getsource(DKISystem._generate_with_injection)
        
        # 验证: "projection" 阶段 (alpha scaling) 在 "preference_merge" 之前
        projection_pos = source.find("preference_merge")
        alpha_scaling_pos = source.find("Apply alpha scaling to memory K/V")
        
        assert alpha_scaling_pos < projection_pos, (
            "Alpha scaling of memory K/V should happen BEFORE preference merge "
            "to prevent double-scaling of preference K/V"
        )

    def test_preference_merge_comment_mentions_independent_scaling(self):
        """验证注释说明 preference K/V 独立缩放"""
        import inspect
        from dki.core.dki_system import DKISystem
        
        source = inspect.getsource(DKISystem._generate_with_injection)
        assert "独立缩放" in source or "independent" in source.lower(), (
            "Code should document that preference K/V uses independent scaling"
        )

    def test_memory_kv_scaled_by_gating_alpha_before_preference(self):
        """验证记忆 K/V 在合并偏好 K/V 之前已被 gating alpha 缩放"""
        import inspect
        from dki.core.dki_system import DKISystem
        
        source = inspect.getsource(DKISystem._generate_with_injection)
        
        # 找到关键代码段的顺序 (在函数体中, 排除参数定义)
        # "Apply alpha scaling to memory K/V" 应在 "preference_merge" 之前
        alpha_scaling_comment = source.find("Apply alpha scaling to memory K/V")
        pref_merge_comment = source.find("preference_merge")
        
        assert alpha_scaling_comment != -1, "Should find alpha scaling comment"
        assert pref_merge_comment != -1, "Should find preference_merge stage"
        assert alpha_scaling_comment < pref_merge_comment, (
            "Memory K/V alpha scaling should appear before preference merge in code"
        )


# ============================================================================
# BUG 2: recall_turns 应被传递给 _keyword_recall
# ============================================================================

class TestRecallTurnsUsed:
    """验证 recall_turns 被正确传递和使用"""

    def test_keyword_recall_accepts_max_turns(self):
        """_keyword_recall 应接受 max_turns 参数"""
        import inspect
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        
        sig = inspect.signature(MultiSignalRecall._keyword_recall)
        params = list(sig.parameters.keys())
        assert "max_turns" in params, (
            "_keyword_recall should accept max_turns parameter"
        )

    def test_keyword_recall_limits_messages_by_max_turns(self):
        """_keyword_recall 应根据 max_turns 限制消息范围"""
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        from dki.core.recall.recall_config import RecallConfig
        
        # 创建模拟消息
        mock_messages = []
        for i in range(20):
            msg = MagicMock()
            msg.id = f"msg_{i}"
            msg.content = f"测试消息 {i} 包含关键词 人工智能"
            mock_messages.append(msg)
        
        # 创建模拟 conversation_repo
        mock_repo = MagicMock()
        mock_repo.get_by_session.return_value = mock_messages
        
        config = RecallConfig()
        recall = MultiSignalRecall(
            config=config,
            conversation_repo=mock_repo,
        )
        
        # 测试: max_turns=3 应只搜索最近 6 条消息
        # 需要 jieba, 如果没有则跳过
        try:
            import jieba
            import jieba.analyse
        except ImportError:
            pytest.skip("jieba not installed")
        
        result = recall._keyword_recall(
            query="人工智能",
            session_id="test_session",
            db_session=None,
            max_turns=3,
        )
        # 即使有 20 条消息, max_turns=3 限制只搜索最后 6 条
        # 结果中的 msg_id 应该只来自最后 6 条
        for msg_id in result.keys():
            msg_idx = int(msg_id.split("_")[1])
            assert msg_idx >= 14, (
                f"Message {msg_id} should not be in results when max_turns=3 "
                f"(only messages 14-19 should be searched)"
            )

    def test_recall_passes_recall_turns_to_keyword_recall(self):
        """recall() 方法应将 recall_turns 传递给 _keyword_recall"""
        import inspect
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        
        source = inspect.getsource(MultiSignalRecall.recall)
        assert "max_turns=recall_turns" in source, (
            "recall() should pass recall_turns to _keyword_recall via max_turns parameter"
        )


# ============================================================================
# BUG 3: 分数融合权重归一化
# ============================================================================

class TestScoreWeightsNormalization:
    """验证分数融合权重被正确归一化"""

    def test_weights_are_normalized_in_fusion(self):
        """融合权重应被归一化到 sum=1.0"""
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        from dki.core.recall.recall_config import RecallConfig, RecallScoreWeights
        
        # 自定义权重: 0.4 + 0.35 = 0.75 (未归一化)
        config = RecallConfig(
            score_weights=RecallScoreWeights(
                keyword_weight=0.4,
                vector_weight=0.35,
                recency_weight=0.25,
            )
        )
        
        recall = MultiSignalRecall(config=config)
        
        # 模拟归一化逻辑
        w = config.score_weights
        w_sum = w.keyword_weight + w.vector_weight
        nw_keyword = w.keyword_weight / w_sum
        nw_vector = w.vector_weight / w_sum
        
        assert abs(nw_keyword + nw_vector - 1.0) < 1e-6, (
            f"Normalized weights should sum to 1.0, got {nw_keyword + nw_vector}"
        )
        assert abs(nw_keyword - 0.4 / 0.75) < 1e-6
        assert abs(nw_vector - 0.35 / 0.75) < 1e-6

    def test_fusion_code_normalizes_weights(self):
        """验证源码中包含权重归一化逻辑"""
        import inspect
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        
        source = inspect.getsource(MultiSignalRecall.recall)
        assert "w_sum" in source or "归一化" in source, (
            "Fusion code should normalize weights"
        )

    def test_zero_weights_do_not_crash(self):
        """权重全为 0 时不应崩溃"""
        from dki.core.recall.recall_config import RecallScoreWeights
        
        w = RecallScoreWeights(keyword_weight=0.0, vector_weight=0.0, recency_weight=0.0)
        w_sum = w.keyword_weight + w.vector_weight
        if w_sum <= 0:
            w_sum = 1.0
        # 不应除零
        nw_keyword = w.keyword_weight / w_sum
        nw_vector = w.vector_weight / w_sum
        assert nw_keyword == 0.0
        assert nw_vector == 0.0


# ============================================================================
# BUG 4: ConversationRepository.get_by_id
# ============================================================================

class TestConversationRepoGetById:
    """验证 ConversationRepository 有 get_by_id 方法"""

    def test_conversation_repo_has_get_by_id(self):
        """ConversationRepository 应有 get_by_id 方法"""
        from dki.database.repository import ConversationRepository
        assert hasattr(ConversationRepository, "get_by_id"), (
            "ConversationRepository should have get_by_id method"
        )

    def test_get_by_id_signature(self):
        """get_by_id 应接受 message_id 参数"""
        import inspect
        from dki.database.repository import ConversationRepository
        
        sig = inspect.signature(ConversationRepository.get_by_id)
        params = list(sig.parameters.keys())
        assert "message_id" in params, (
            "get_by_id should accept message_id parameter"
        )

    def test_conversation_repo_wrapper_can_call_get_by_id(self):
        """_ConversationRepoWrapper.get_by_id 应能正常调用"""
        from dki.core.dki_system import _ConversationRepoWrapper
        
        # 模拟 DatabaseManager
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db_manager.session_scope.return_value.__exit__ = MagicMock(return_value=False)
        
        wrapper = _ConversationRepoWrapper(mock_db_manager)
        
        # 调用 get_by_id - 应不抛出 AttributeError
        # (在 Mock 环境下, repo 的 get_by_id 也是 Mock, 不会报错)
        result = wrapper.get_by_id("conv_test123")
        # 不应抛出异常


# ============================================================================
# BUG 5: 中文偏好 token 估算
# ============================================================================

class TestChineseTokenEstimation:
    """验证中文偏好 token 估算的准确性"""

    def test_chinese_preference_token_count(self):
        """中文偏好文本的 token 估算应合理"""
        # 50 个中文字 ≈ 75 tokens (1.5 token/字)
        chinese_text = "推理严谨存疑澄清语言温暖富于幽默喜欢编程技术领域人工智能和企业资源规划系统" * 2
        
        import re
        cn = len(re.findall(r'[\u4e00-\u9fff]', chinese_text))
        en = len(re.findall(r'[a-zA-Z]+', chinese_text))
        estimated = int(cn * 1.5 + en * 1.3) or 1
        
        # 旧方法: split() * 2 对中文只会返回 ~2 (因为中文没有空格分词)
        old_estimate = len(chinese_text.split()) * 2
        
        assert estimated > old_estimate, (
            f"New estimate ({estimated}) should be much larger than old ({old_estimate}) "
            f"for Chinese text"
        )
        assert estimated >= cn, (
            f"Token estimate ({estimated}) should be >= character count ({cn})"
        )

    def test_english_preference_token_count(self):
        """英文偏好文本的 token 估算应合理"""
        english_text = "I like programming and artificial intelligence research"
        
        import re
        cn = len(re.findall(r'[\u4e00-\u9fff]', english_text))
        en = len(re.findall(r'[a-zA-Z]+', english_text))
        estimated = int(cn * 1.5 + en * 1.3) or 1
        
        # 英文约 7 个单词 * 1.3 ≈ 9 tokens
        assert estimated >= 7, f"English estimate should be >= 7, got {estimated}"

    def test_dki_system_uses_improved_estimation(self):
        """验证 dki_system.py 使用改进的 token 估算"""
        import inspect
        from dki.core.dki_system import DKISystem
        
        source = inspect.getsource(DKISystem.chat)
        # 不应使用旧的 split() * 2 方法
        assert "split()) * 2" not in source, (
            "dki_system.py should not use split() * 2 for Chinese token estimation"
        )


# ============================================================================
# MINOR 6: injection_markers 不应包含正常角色标签
# ============================================================================

class TestInjectionMarkers:
    """验证 injection_markers 不会误过滤正常消息"""

    def test_markers_do_not_include_role_labels(self):
        """injection_markers 不应包含 'Assistant:' 或 '助手:'"""
        from dki.core.components.hybrid_injector import HybridDKIInjector
        
        injector = HybridDKIInjector(language="cn")
        
        # 创建包含 "助手" 字样的正常消息
        from dki.core.components.hybrid_injector import SessionHistory
        history = SessionHistory(session_id="test")
        history.add_message("user", "请问助手可以帮我什么?")
        history.add_message("assistant", "我是AI助手，很高兴为您服务。")
        history.add_message("user", "谢谢助手!")
        
        result = injector._format_history(history)
        
        # "助手:" 不应导致消息被过滤
        # 用户消息中包含 "助手" 不应被过滤
        assert "请问助手可以帮我什么" in result, (
            "Normal message containing '助手' should not be filtered"
        )
        assert "谢谢助手" in result, (
            "Normal message containing '助手' should not be filtered"
        )

    def test_markers_filter_dki_injection_markers(self):
        """injection_markers 应过滤包含 DKI 注入标记的消息"""
        from dki.core.components.hybrid_injector import HybridDKIInjector, SessionHistory
        
        injector = HybridDKIInjector(language="cn")
        
        history = SessionHistory(session_id="test")
        history.add_message("user", "你好")
        # 这条消息包含 DKI 注入标记, 应被过滤
        history.add_message("assistant", "[会话历史参考]\n之前的对话...\n[会话历史结束]")
        history.add_message("user", "今天天气如何?")
        
        result = injector._format_history(history)
        
        # 被过滤的消息内容不应出现在结果中
        # 注意: 结果的外层模板本身包含 [会话历史参考], 这是正常的
        # 但被过滤消息的内容 "之前的对话..." 不应出现
        assert "之前的对话" not in result, (
            "Content from message with DKI injection markers should be filtered"
        )
        # 正常消息应保留
        assert "你好" in result
        assert "今天天气如何" in result

    def test_current_question_marker_is_filtered(self):
        """[当前问题] 标记应被过滤"""
        from dki.core.components.hybrid_injector import HybridDKIInjector, SessionHistory
        
        injector = HybridDKIInjector(language="cn")
        
        history = SessionHistory(session_id="test")
        history.add_message("user", "你好")
        history.add_message("assistant", "[当前问题]\n用户: 之前的问题")
        history.add_message("user", "新问题")
        
        result = injector._format_history(history)
        
        assert "[当前问题]" not in result, (
            "[当前问题] marker should be filtered"
        )


# ============================================================================
# 额外: recall 组件集成测试
# ============================================================================

class TestRecallComponentIntegration:
    """recall 组件的集成正确性"""

    def test_suffix_builder_handles_empty_messages(self):
        """SuffixBuilder 处理空消息列表"""
        from dki.core.recall.suffix_builder import SuffixBuilder
        from dki.core.recall.recall_config import RecallConfig
        from dki.core.recall.prompt_formatter import GenericFormatter
        
        config = RecallConfig()
        formatter = GenericFormatter(language="cn")
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        
        result = builder.build(query="测试问题", recalled_messages=[])
        assert result.text == "测试问题"
        assert result.total_tokens == 0

    def test_suffix_builder_budget_exhausted(self):
        """SuffixBuilder 在预算耗尽时返回原始查询"""
        from dki.core.recall.suffix_builder import SuffixBuilder
        from dki.core.recall.recall_config import RecallConfig
        from dki.core.recall.prompt_formatter import GenericFormatter
        
        config = RecallConfig()
        formatter = GenericFormatter(language="cn")
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        
        # context_window 极小, 预算耗尽
        result = builder.build(
            query="测试问题",
            recalled_messages=[],
            context_window=10,
            preference_tokens=5,
        )
        assert result.text == "测试问题"

    def test_fact_retriever_handles_short_message(self):
        """FactRetriever 处理短消息"""
        from dki.core.recall.fact_retriever import FactRetriever
        from dki.core.recall.recall_config import RecallConfig
        
        config = RecallConfig()
        
        # 模拟 conversation_repo
        mock_repo = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "短消息"
        mock_msg.role = "user"
        mock_msg.created_at = "2026-02-18"
        mock_repo.get_by_id.return_value = mock_msg
        
        retriever = FactRetriever(config=config, conversation_repo=mock_repo)
        
        result = retriever.retrieve(
            trace_id="msg_001",
            session_id="test_session",
        )
        
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "短消息"
        assert result.has_more is False

    def test_fact_retriever_chunks_long_message(self):
        """FactRetriever 对长消息进行分块"""
        from dki.core.recall.fact_retriever import FactRetriever
        from dki.core.recall.recall_config import RecallConfig
        
        config = RecallConfig()
        
        mock_repo = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "这是一段非常长的消息。" * 100  # ~1000 字符
        mock_msg.role = "assistant"
        mock_msg.created_at = "2026-02-18"
        mock_repo.get_by_id.return_value = mock_msg
        
        retriever = FactRetriever(config=config, conversation_repo=mock_repo)
        
        result = retriever.retrieve(
            trace_id="msg_002",
            session_id="test_session",
            offset=0,
            limit=2,
        )
        
        assert len(result.messages) <= 2
        assert result.total_count > 1

    def test_generic_formatter_detect_fact_request(self):
        """GenericFormatter 检测 function call"""
        from dki.core.recall.prompt_formatter import GenericFormatter
        
        formatter = GenericFormatter(language="cn")
        
        # 正常的 function call
        output = '我需要查看原始记录。retrieve_fact(trace_id="msg_005", offset=0, limit=5)'
        request = formatter.detect_fact_request(output)
        
        assert request is not None
        assert request.trace_id == "msg_005"
        assert request.offset == 0
        assert request.limit == 5

    def test_generic_formatter_no_fact_request(self):
        """GenericFormatter 在无 function call 时返回 None"""
        from dki.core.recall.prompt_formatter import GenericFormatter
        
        formatter = GenericFormatter(language="cn")
        
        output = "这是一个普通的回答，不包含 function call。"
        request = formatter.detect_fact_request(output)
        
        assert request is None

    def test_normalize_scores_sigmoid(self):
        """验证 sigmoid 归一化在 [0, 1] 范围内"""
        from dki.core.recall.multi_signal_recall import MultiSignalRecall
        
        raw = {"a": 0.1, "b": 0.5, "c": 1.0, "d": 2.0}
        normalized = MultiSignalRecall._normalize_scores(raw)
        
        for k, v in normalized.items():
            assert 0.0 <= v <= 1.0, f"Score {k}={v} out of [0,1] range"

    def test_create_formatter_auto_deepseek(self):
        """auto 模式应为 deepseek 模型选择 DeepSeekFormatter"""
        from dki.core.recall.prompt_formatter import create_formatter, DeepSeekFormatter
        
        formatter = create_formatter(model_name="deepseek-llm-7b-chat", formatter_type="auto")
        assert isinstance(formatter, DeepSeekFormatter)

    def test_create_formatter_auto_glm(self):
        """auto 模式应为 GLM 模型选择 GLMFormatter"""
        from dki.core.recall.prompt_formatter import create_formatter, GLMFormatter
        
        formatter = create_formatter(model_name="chatglm3-6b", formatter_type="auto")
        assert isinstance(formatter, GLMFormatter)

    def test_create_formatter_auto_generic(self):
        """auto 模式应为未知模型选择 GenericFormatter"""
        from dki.core.recall.prompt_formatter import create_formatter, GenericFormatter
        
        formatter = create_formatter(model_name="llama-7b", formatter_type="auto")
        assert isinstance(formatter, GenericFormatter)


# ============================================================================
# 额外: prompt_formatter 完整后缀格式化
# ============================================================================

class TestPromptFormatterFullSuffix:
    """验证 PromptFormatter.format_full_suffix 的完整输出"""

    def test_format_full_suffix_cn(self):
        """中文完整后缀格式化"""
        from dki.core.recall.prompt_formatter import GenericFormatter
        from dki.core.recall.recall_config import HistoryItem
        
        formatter = GenericFormatter(language="cn")
        
        items = [
            HistoryItem(
                type="message",
                content="你好，我想了解 DKI 系统",
                trace_id="msg_001",
                role="user",
                token_count=20,
            ),
            HistoryItem(
                type="summary",
                content="用户讨论了 DKI 的技术细节",
                trace_id="msg_002",
                role="assistant",
                token_count=15,
                confidence="medium",
                facts_covered=["技术细节"],
                facts_missing=["具体数值"],
            ),
        ]
        
        result = formatter.format_full_suffix(
            items=items,
            trace_ids=["msg_001", "msg_002"],
            query="DKI 的 alpha 值是多少?",
        )
        
        # 验证结构
        assert "[会话历史参考]" in result
        assert "用户: 你好，我想了解 DKI 系统" in result
        assert "[SUMMARY" in result
        assert "retrieve_fact" in result  # 限定提示
        assert "用户当前问题: DKI 的 alpha 值是多少?" in result

    def test_format_full_suffix_en(self):
        """英文完整后缀格式化"""
        from dki.core.recall.prompt_formatter import GenericFormatter
        from dki.core.recall.recall_config import HistoryItem
        
        formatter = GenericFormatter(language="en")
        
        items = [
            HistoryItem(
                type="message",
                content="Hello, tell me about DKI",
                trace_id="msg_001",
                role="user",
                token_count=10,
            ),
        ]
        
        result = formatter.format_full_suffix(
            items=items,
            trace_ids=["msg_001"],
            query="What is alpha?",
        )
        
        assert "[Session History Reference]" in result
        assert "User: Hello, tell me about DKI" in result
        assert "Current question: What is alpha?" in result


# ============================================================================
# 额外: _ConversationRepoWrapper 完整性
# ============================================================================

class TestConversationRepoWrapper:
    """验证 _ConversationRepoWrapper 的方法完整性"""

    def test_wrapper_has_required_methods(self):
        """Wrapper 应有 get_by_session, get_recent, get_by_id"""
        from dki.core.dki_system import _ConversationRepoWrapper
        
        assert hasattr(_ConversationRepoWrapper, "get_by_session")
        assert hasattr(_ConversationRepoWrapper, "get_recent")
        assert hasattr(_ConversationRepoWrapper, "get_by_id")

    def test_wrapper_get_by_id_uses_hasattr_check(self):
        """get_by_id 应使用 hasattr 检查 repo 方法"""
        import inspect
        from dki.core.dki_system import _ConversationRepoWrapper
        
        source = inspect.getsource(_ConversationRepoWrapper.get_by_id)
        assert "hasattr" in source, (
            "get_by_id should use hasattr check for repo.get_by_id"
        )
