"""
v7.2: 近轮对话获取与合并逻辑的单元测试

测试 DKIPlugin._merge_recent_and_recalled 方法:
- 近轮优先, BM25 补充
- message_id 去重
- 空列表边界情况
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from dki.adapters.base import ChatMessage, IUserDataAdapter, UserPreference


# ============ Fixtures ============

def _make_msg(msg_id: str, content: str, role: str = "user",
              session_id: str = "s1", user_id: str = "u1",
              minutes_ago: int = 0) -> ChatMessage:
    """创建测试用 ChatMessage"""
    return ChatMessage(
        message_id=msg_id,
        session_id=session_id,
        user_id=user_id,
        role=role,
        content=content,
        timestamp=datetime.now() - timedelta(minutes=minutes_ago),
    )


class TestMergeRecentAndRecalled:
    """测试 _merge_recent_and_recalled 合并逻辑"""
    
    def _create_plugin_with_merge(self):
        """创建一个最小化的 DKIPlugin 实例用于测试 merge 方法"""
        from dki.core.dki_plugin import DKIPlugin
        
        # Mock 模型
        model = MagicMock()
        model.model_name = "test-model"
        model.tokenizer = None
        
        adapter = MagicMock(spec=IUserDataAdapter)
        
        # 构建足够真实的 config mock, 避免 MagicMock 比较问题
        mock_cfg = MagicMock()
        mock_cfg.dki = MagicMock()
        mock_cfg.dki.recall = {}
        mock_cfg.dki.hybrid_injection = MagicMock()
        mock_cfg.dki.hybrid_injection.language = "cn"
        mock_cfg.dki.hybrid_injection.preference = MagicMock()
        mock_cfg.dki.hybrid_injection.preference.alpha = 0.4
        
        # model config 需要返回真实值避免类型比较错误
        mock_model_cfg = MagicMock()
        mock_model_cfg.default_engine = "test"
        mock_model_cfg.engines = {}  # 空字典, 不触发 engine 配置
        mock_cfg.model = mock_model_cfg
        
        with patch('dki.core.dki_plugin.ConfigLoader') as mock_config:
            mock_config.return_value.config = mock_cfg
            
            plugin = DKIPlugin(
                model_adapter=model,
                user_data_adapter=adapter,
                config=mock_cfg,
            )
        
        return plugin
    
    def test_recent_only(self):
        """近轮消息无 BM25 召回时, 返回近轮消息"""
        plugin = self._create_plugin_with_merge()
        
        recent = [
            _make_msg("m1", "你好", minutes_ago=5),
            _make_msg("m2", "你好呀", role="assistant", minutes_ago=4),
        ]
        recalled = []
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        assert len(merged) == 2
        assert merged[0].message_id == "m1"
        assert merged[1].message_id == "m2"
    
    def test_recalled_only(self):
        """无近轮消息时, 返回 BM25 召回结果"""
        plugin = self._create_plugin_with_merge()
        
        recent = []
        recalled = [
            _make_msg("m5", "之前聊过 ERP", minutes_ago=100),
            _make_msg("m6", "SAP 和金蝶对比", minutes_ago=90),
        ]
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        assert len(merged) == 2
        assert merged[0].message_id == "m5"
        assert merged[1].message_id == "m6"
    
    def test_dedup_by_message_id(self):
        """相同 message_id 的消息应去重, 近轮优先"""
        plugin = self._create_plugin_with_merge()
        
        # m2 同时出现在近轮和 BM25 召回中
        recent = [
            _make_msg("m1", "你好", minutes_ago=5),
            _make_msg("m2", "讨论 ERP", minutes_ago=4),
        ]
        recalled = [
            _make_msg("m2", "讨论 ERP", minutes_ago=4),  # 重复
            _make_msg("m3", "SAP 对比", minutes_ago=100),
        ]
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        assert len(merged) == 3  # m1, m2, m3 (m2 去重)
        ids = [m.message_id for m in merged]
        assert ids == ["m1", "m2", "m3"]
    
    def test_recent_first_then_bm25(self):
        """近轮消息在前, BM25 补充在后"""
        plugin = self._create_plugin_with_merge()
        
        recent = [
            _make_msg("m10", "最近的消息1", minutes_ago=2),
            _make_msg("m11", "最近的消息2", role="assistant", minutes_ago=1),
        ]
        recalled = [
            _make_msg("m1", "很久以前的消息", minutes_ago=1000),
            _make_msg("m5", "中间的消息", minutes_ago=500),
        ]
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        assert len(merged) == 4
        # 近轮在前
        assert merged[0].message_id == "m10"
        assert merged[1].message_id == "m11"
        # BM25 在后
        assert merged[2].message_id == "m1"
        assert merged[3].message_id == "m5"
    
    def test_cross_session_recent(self):
        """近轮消息可以跨 session"""
        plugin = self._create_plugin_with_merge()
        
        recent = [
            _make_msg("m1", "上个session的消息", session_id="s1", minutes_ago=30),
            _make_msg("m2", "上个session的回复", session_id="s1", role="assistant", minutes_ago=29),
            _make_msg("m3", "当前session的消息", session_id="s2", minutes_ago=2),
            _make_msg("m4", "当前session的回复", session_id="s2", role="assistant", minutes_ago=1),
        ]
        recalled = [
            _make_msg("m100", "更早的相关消息", session_id="s0", minutes_ago=1000),
        ]
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        assert len(merged) == 5
        # 包含两个 session 的近轮消息
        session_ids = [m.session_id for m in merged[:4]]
        assert "s1" in session_ids
        assert "s2" in session_ids
    
    def test_both_empty(self):
        """两个列表都为空时返回空"""
        plugin = self._create_plugin_with_merge()
        
        merged = plugin._merge_recent_and_recalled([], [])
        assert merged == []
    
    def test_large_overlap(self):
        """大量重叠时正确去重"""
        plugin = self._create_plugin_with_merge()
        
        # 10 条近轮消息
        recent = [_make_msg(f"m{i}", f"msg-{i}", minutes_ago=10-i) for i in range(10)]
        # BM25 召回了其中 5 条 + 5 条新的
        recalled = (
            [_make_msg(f"m{i}", f"msg-{i}", minutes_ago=10-i) for i in range(5)]
            + [_make_msg(f"m{i}", f"bm25-{i}", minutes_ago=100+i) for i in range(10, 15)]
        )
        
        merged = plugin._merge_recent_and_recalled(recent, recalled)
        
        # 10 近轮 + 5 新 BM25 = 15
        assert len(merged) == 15
        # 前 10 条是近轮
        for i in range(10):
            assert merged[i].message_id == f"m{i}"


class TestGetMaxRecentTurns:
    """测试 _get_max_recent_turns 配置读取"""
    
    def test_default_value(self):
        """无配置时返回默认值 5"""
        from dki.core.dki_plugin import DKIPlugin
        
        model = MagicMock()
        model.model_name = "test"
        model.tokenizer = None
        adapter = MagicMock(spec=IUserDataAdapter)
        
        mock_cfg = MagicMock()
        mock_cfg.dki = MagicMock()
        mock_cfg.dki.recall = {}
        mock_cfg.dki.hybrid_injection = MagicMock()
        mock_cfg.dki.hybrid_injection.language = "cn"
        mock_cfg.dki.hybrid_injection.preference = MagicMock()
        mock_cfg.dki.hybrid_injection.preference.alpha = 0.4
        mock_model_cfg = MagicMock()
        mock_model_cfg.default_engine = "test"
        mock_model_cfg.engines = {}
        mock_cfg.model = mock_model_cfg
        
        with patch('dki.core.dki_plugin.ConfigLoader') as mock_config:
            mock_config.return_value.config = mock_cfg
            
            plugin = DKIPlugin(
                model_adapter=model,
                user_data_adapter=adapter,
                config=mock_cfg,
            )
        
        result = plugin._get_max_recent_turns()
        assert result == 5


class TestAdapterGetRecentMessages:
    """测试适配器的 get_recent_messages 方法"""
    
    @pytest.mark.asyncio
    async def test_base_adapter_default_returns_empty(self):
        """基类默认实现返回空列表"""
        from dki.adapters.base import IUserDataAdapter
        
        # 创建一个最小的具体实现
        class MinimalAdapter(IUserDataAdapter):
            async def connect(self): pass
            async def disconnect(self): pass
            async def get_user_profile(self, user_id, **kwargs): return None
            async def get_user_preferences(self, user_id, **kwargs): return []
            async def get_session_history(self, session_id, **kwargs): return []
            async def search_relevant_history(self, user_id, query, **kwargs): return []
            async def health_check(self): return True
        
        adapter = MinimalAdapter()
        result = await adapter.get_recent_messages("user1", limit=10)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_example_adapter_recent_messages(self):
        """ExampleAdapter 的 get_recent_messages 实现"""
        from dki.adapters.example_adapter import ExampleAdapter
        
        adapter = ExampleAdapter()
        await adapter.connect()
        
        # 添加跨 session 的消息
        now = datetime.now()
        for i, sid in enumerate(["s1", "s1", "s2", "s2"]):
            role = "user" if i % 2 == 0 else "assistant"
            adapter.add_message(
                session_id=sid,
                user_id="u1",
                role=role,
                content=f"msg-{i}",
            )
        
        # 获取近轮消息
        recent = await adapter.get_recent_messages("u1", limit=10)
        
        # 应包含两个 session 的消息
        assert len(recent) == 4
        # 按时间正序
        for i in range(1, len(recent)):
            assert recent[i].timestamp >= recent[i-1].timestamp
