"""
单元测试: BM25 中文召回改进 (v7.1)

验证 BM25 分词改进后能正确召回中文历史消息。

修复内容:
1. _bm25_score: 优先使用 jieba 分词 (替代简单单字+bigram)
2. _bm25_score: 添加中文停用词过滤 (避免高频词稀释权重)
3. _search_with_bm25_only: 过滤 score=0 的消息 (避免返回无关结果)
4. _search_with_bm25_only: BM25 无结果时回退到最近消息

测试场景来源: DKI/docs/UI/关键词.md
- 查询: "你好, 你还记得我给你纠正了哪本书的作者吗"
- 历史: 关于《挪威的森林》作者纠正的对话
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.adapters.base import ChatMessage


# ============================================================================
# 测试数据 (来自 关键词.md)
# ============================================================================

def _make_msg(content: str, role: str = "user", msg_id: str = "0") -> ChatMessage:
    """创建测试用 ChatMessage"""
    return ChatMessage(
        message_id=msg_id,
        session_id="test_session",
        user_id="user01",
        role=role,
        content=content,
        timestamp=datetime.now(),
    )


# 历史消息 (模拟数据库中存储的消息)
HISTORY_MESSAGES = [
    _make_msg(
        "谢谢,你还记得呢, 你读过挪威的森林吗,你喜欢这本书吗",
        role="user", msg_id="msg_001"
    ),
    _make_msg(
        "噢，不好意思，Lucas！我并没有读过《挪威的森林》，我只是个语言模型，"
        "我知道一些基本信息，但没有个人阅读体验。关于这本书，我知道它是一部经典的日本小说，"
        "由宫泽贤治创作，讲述了一个年轻女孩的成长故事。",
        role="assistant", msg_id="msg_002"
    ),
    _make_msg(
        "你是不是搞错了,挪威的森林事村上春树的作品,你知道这个作家吗",
        role="user", msg_id="msg_003"
    ),
    _make_msg(
        "OH NO! 我真是太糟糕了！你是对的，我搞错了！挪威的森林是村上春树的作品，"
        "一个非常经典的日本小说。村上春树是日本最知名的作家之一。",
        role="assistant", msg_id="msg_004"
    ),
    # 无关消息 (噪音)
    _make_msg("今天天气真好", role="user", msg_id="msg_noise_01"),
    _make_msg("是的，阳光明媚", role="assistant", msg_id="msg_noise_02"),
    _make_msg("帮我推荐一家餐厅", role="user", msg_id="msg_noise_03"),
    _make_msg("推荐川菜馆子，口味不错", role="assistant", msg_id="msg_noise_04"),
]

# 核心测试查询
RECALL_QUERY = "你好, 你还记得我给你纠正了哪本书的作者吗"


# ============================================================================
# 测试 BM25 分词改进
# ============================================================================

class TestBM25Tokenizer:
    """测试 BM25 分词器改进"""
    
    def _get_adapter_class(self):
        """获取 ConfigDrivenAdapter 类 (仅需要 _bm25_score 方法)"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        return ConfigDrivenAdapter
    
    def _create_mock_adapter(self):
        """创建 mock adapter 用于测试 _bm25_score"""
        cls = self._get_adapter_class()
        adapter = object.__new__(cls)
        return adapter
    
    def test_bm25_score_returns_positive_for_relevant_messages(self):
        """核心测试: BM25 应该为相关消息返回正分数"""
        adapter = self._create_mock_adapter()
        
        results = adapter._bm25_score(RECALL_QUERY, HISTORY_MESSAGES)
        
        # 找到关键消息的分数
        score_map = {msg.message_id: score for msg, score in results}
        
        # 包含 "纠正" 相关内容的消息应有正分数
        # msg_003: "你是不是搞错了,挪威的森林事村上春树的作品" — 纠正场景
        # msg_004: "你是对的，我搞错了" — 承认被纠正
        # msg_001/002: 关于书和作者的对话
        assert score_map.get("msg_003", 0) > 0, \
            f"msg_003 (纠正消息) 应有正分数, 实际: {score_map.get('msg_003', 0)}"
        assert score_map.get("msg_001", 0) > 0, \
            f"msg_001 (书相关) 应有正分数, 实际: {score_map.get('msg_001', 0)}"
    
    def test_bm25_relevant_messages_rank_higher_than_noise(self):
        """相关消息的 BM25 分数应高于无关消息"""
        adapter = self._create_mock_adapter()
        
        results = adapter._bm25_score(RECALL_QUERY, HISTORY_MESSAGES)
        score_map = {msg.message_id: score for msg, score in results}
        
        # 相关消息的最高分
        relevant_ids = ["msg_001", "msg_002", "msg_003", "msg_004"]
        relevant_max = max(score_map.get(mid, 0) for mid in relevant_ids)
        
        # 噪音消息的最高分
        noise_ids = ["msg_noise_01", "msg_noise_02", "msg_noise_03", "msg_noise_04"]
        noise_max = max(score_map.get(mid, 0) for mid in noise_ids)
        
        assert relevant_max > noise_max, \
            f"相关消息最高分 ({relevant_max:.3f}) 应高于噪音最高分 ({noise_max:.3f})"
    
    def test_bm25_stopwords_filtered(self):
        """停用词应被过滤, 不参与评分"""
        adapter = self._create_mock_adapter()
        
        # 查询全是停用词
        stopword_query = "你的了吗"
        results = adapter._bm25_score(stopword_query, HISTORY_MESSAGES)
        
        # 如果停用词被正确过滤, 所有分数应为 0 或很低
        max_score = max(score for _, score in results)
        # 注: 如果 jieba 不可用, 单字分词可能仍然有少量匹配
        # 但分数应该远低于正常查询
        assert max_score < 1.0, \
            f"全停用词查询的最高分应很低, 实际: {max_score:.3f}"
    
    def test_bm25_specific_keywords_match(self):
        """测试具体关键词: "纠正" + "作者" + "书" 应命中相关消息"""
        adapter = self._create_mock_adapter()
        
        # 精确查询
        precise_query = "纠正 作者 书"
        results = adapter._bm25_score(precise_query, HISTORY_MESSAGES)
        score_map = {msg.message_id: score for msg, score in results}
        
        # msg_003 包含 "作" + 书相关内容, 应有正分数
        # msg_001 包含 "书"
        has_positive = any(score > 0 for _, score in results)
        assert has_positive, "精确关键词查询应至少有一条消息得分 > 0"
    
    def test_bm25_empty_query(self):
        """空查询应返回全零分"""
        adapter = self._create_mock_adapter()
        
        results = adapter._bm25_score("", HISTORY_MESSAGES)
        assert all(score == 0.0 for _, score in results)
    
    def test_bm25_empty_messages(self):
        """空消息列表应返回空结果"""
        adapter = self._create_mock_adapter()
        
        results = adapter._bm25_score(RECALL_QUERY, [])
        assert results == []


# ============================================================================
# 测试 _search_with_bm25_only 的 score>0 过滤
# ============================================================================

class TestBM25SearchFiltering:
    """测试 BM25 检索结果过滤"""
    
    @pytest.mark.asyncio
    async def test_bm25_search_filters_zero_score(self):
        """score=0 的消息不应被返回"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter,
            ConfigDrivenAdapterConfig,
        )
        
        adapter = object.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        
        # Mock _get_user_messages
        adapter._get_user_messages = AsyncMock(return_value=HISTORY_MESSAGES)
        
        results = await adapter._search_with_bm25_only(
            user_id="user01",
            query=RECALL_QUERY,
            limit=5,
            session_id=None,
        )
        
        # 应返回有正分数的消息
        assert len(results) > 0, "应返回至少一条消息"
        
        # 验证返回的消息确实与查询相关
        relevant_ids = {"msg_001", "msg_002", "msg_003", "msg_004"}
        returned_ids = {msg.message_id for msg in results}
        
        # 至少应命中一条相关消息
        overlap = relevant_ids & returned_ids
        assert len(overlap) > 0, \
            f"应至少命中一条相关消息, 返回的 IDs: {returned_ids}"
    
    @pytest.mark.asyncio
    async def test_bm25_search_fallback_on_no_match(self):
        """当 BM25 完全无匹配时, 应回退到最近消息"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter,
            ConfigDrivenAdapterConfig,
        )
        
        adapter = object.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        
        # 使用不可能匹配任何消息的查询
        impossible_query = "xyzzyx quantum entanglement photon"
        adapter._get_user_messages = AsyncMock(return_value=HISTORY_MESSAGES)
        
        results = await adapter._search_with_bm25_only(
            user_id="user01",
            query=impossible_query,
            limit=5,
            session_id=None,
        )
        
        # 应回退到最近的消息 (最多 5 条)
        assert len(results) > 0, "无匹配时应回退到最近消息"
        assert len(results) <= 5, "回退时最多返回 5 条"


# ============================================================================
# 测试停用词表覆盖
# ============================================================================

class TestStopwords:
    """测试停用词表"""
    
    def test_stopwords_is_frozenset(self):
        """停用词表应为 frozenset (不可变, 性能优)"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        assert isinstance(ConfigDrivenAdapter._CN_STOPWORDS, frozenset)
    
    def test_common_stopwords_present(self):
        """常见停用词应在表中"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        sw = ConfigDrivenAdapter._CN_STOPWORDS
        
        for word in ['的', '了', '你', '我', '吗', '是', '在']:
            assert word in sw, f"常见停用词 '{word}' 应在停用词表中"
    
    def test_content_words_not_in_stopwords(self):
        """有信息量的词不应在停用词表中"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        sw = ConfigDrivenAdapter._CN_STOPWORDS
        
        content_words = ['纠正', '作者', '书', '挪威', '森林', '村上', '春树', '推荐', '餐厅']
        for word in content_words:
            assert word not in sw, f"内容词 '{word}' 不应在停用词表中"


# ============================================================================
# 端到端场景测试 (模拟 关键词.md 场景)
# ============================================================================

class TestKeywordDocScenario:
    """
    端到端场景: 关键词.md 中的实际场景
    
    用户先讨论了《挪威的森林》作者, 纠正了 AI 的错误,
    后续查询 "你好, 你还记得我给你纠正了哪本书的作者吗"
    应该能召回相关历史。
    """
    
    def test_scenario_bm25_recalls_correction_context(self):
        """BM25 应能召回 "纠正作者" 相关的历史消息"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        
        adapter = object.__new__(ConfigDrivenAdapter)
        results = adapter._bm25_score(RECALL_QUERY, HISTORY_MESSAGES)
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤正分数
        positive_results = [(msg, score) for msg, score in results if score > 0]
        
        assert len(positive_results) >= 2, \
            f"应至少召回 2 条正分数消息, 实际: {len(positive_results)}"
        
        # 检查 top-3 中是否包含关键消息
        top_3_ids = {msg.message_id for msg, _ in positive_results[:3]}
        
        # msg_003 ("你是不是搞错了") 或 msg_001 ("你读过挪威的森林吗") 应在 top-3
        key_msgs = {"msg_001", "msg_003"}
        assert top_3_ids & key_msgs, \
            f"Top-3 应包含纠正相关消息, 实际 top-3: {top_3_ids}"
    
    def test_scenario_noise_messages_excluded_from_top(self):
        """噪音消息不应出现在 top 结果中"""
        from dki.adapters.config_driven_adapter import ConfigDrivenAdapter
        
        adapter = object.__new__(ConfigDrivenAdapter)
        results = adapter._bm25_score(RECALL_QUERY, HISTORY_MESSAGES)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤正分数
        positive_results = [(msg, score) for msg, score in results if score > 0]
        
        if positive_results:
            # 噪音消息不应在正分数结果中
            noise_ids = {"msg_noise_01", "msg_noise_02", "msg_noise_03", "msg_noise_04"}
            positive_ids = {msg.message_id for msg, _ in positive_results}
            
            noise_in_positive = noise_ids & positive_ids
            # 允许少量噪音 (如 "好" 可能出现在多处), 但不应占主导
            assert len(noise_in_positive) <= 1, \
                f"正分数结果中噪音消息不应超过 1 条, 实际: {noise_in_positive}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
