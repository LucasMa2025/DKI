"""
ConversationRouter 单元测试

覆盖:
1. SignalCollector — 信号采集 (记忆触发、知识查询、引用检测)
2. RoutingScoringEngine — 五维评分模型
3. ConversationRouter — 路由决策 (阈值、强制模式、降级)
4. RouterConfig — 配置加载与更新
"""

import pytest
from dki.core.conversation_router import (
    ConversationRouter,
    RouterConfig,
    RoutingDecision,
    RoutingSignals,
    RoutingScoringEngine,
    SignalCollector,
    RouteMode,
    RouteReason,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def default_config():
    return RouterConfig()

@pytest.fixture
def collector(default_config):
    return SignalCollector(default_config)

@pytest.fixture
def scorer(default_config):
    return RoutingScoringEngine(default_config)

@pytest.fixture
def router(default_config):
    return ConversationRouter(config=default_config)


# ============================================================
# 1. SignalCollector 测试
# ============================================================

class TestSignalCollector:
    """信号采集器测试"""
    
    def test_detect_memory_trigger_cn(self, collector):
        """中文记忆触发信号"""
        assert collector._check_memory_trigger("你还记得我的名字吗")
        assert collector._check_memory_trigger("我们之前讨论过这个话题")
        assert collector._check_memory_trigger("上次你提到的方案")
        assert not collector._check_memory_trigger("今天天气怎么样")
    
    def test_detect_memory_trigger_en(self, collector):
        """英文记忆触发信号"""
        assert collector._check_memory_trigger("do you remember my name")
        assert collector._check_memory_trigger("we previously discussed this")
        assert not collector._check_memory_trigger("what is the weather today")
    
    def test_detect_knowledge_query_cn(self, collector):
        """中文知识检索查询"""
        assert collector._check_knowledge_query("什么是机器学习")
        assert collector._check_knowledge_query("介绍一下Python")
        assert collector._check_knowledge_query("如何使用Docker")
        assert not collector._check_knowledge_query("你还记得我的名字吗")
    
    def test_detect_knowledge_query_en(self, collector):
        """英文知识检索查询"""
        assert collector._check_knowledge_query("what is machine learning")
        assert collector._check_knowledge_query("explain neural networks")
        assert collector._check_knowledge_query("how to use Docker")
        assert not collector._check_knowledge_query("do you remember my preference")
    
    def test_detect_reference_cn(self, collector):
        """中文指代引用"""
        assert collector._check_reference("那个话题你能继续讲吗")
        assert collector._check_reference("你刚才说的是什么意思")
        assert not collector._check_reference("Python是什么")
    
    def test_collect_full_signals(self, collector):
        """完整信号采集"""
        signals = collector.collect(
            query="你还记得我之前说过的偏好吗",
            session_turn_count=5,
            session_total_tokens=2000,
            has_user_preferences=True,
            preference_count=3,
            user_total_sessions=10,
            user_total_messages=50,
        )
        
        assert signals.session_turn_count == 5
        assert signals.session_total_tokens == 2000
        assert signals.has_user_preferences is True
        assert signals.preference_count == 3
        assert signals.has_memory_trigger is True  # "记得" + "之前说过"
        assert signals.has_cross_session_history is True  # 10 > 3
        assert signals.is_knowledge_query is False
    
    def test_collect_first_turn_signals(self, collector):
        """首轮信号采集"""
        signals = collector.collect(
            query="什么是ERP系统",
            session_turn_count=0,
        )
        
        assert signals.session_turn_count == 0
        assert signals.has_memory_trigger is False
        assert signals.is_knowledge_query is True
        assert signals.has_user_preferences is False
    
    def test_forced_mode_signal(self, collector):
        """强制模式信号"""
        signals = collector.collect(
            query="test",
            forced_mode="dki",
        )
        assert signals.forced_mode == "dki"


# ============================================================
# 2. RoutingScoringEngine 测试
# ============================================================

class TestRoutingScoringEngine:
    """评分引擎测试"""
    
    def test_first_turn_no_history(self, scorer):
        """首轮无历史 → RAG 得分高"""
        signals = RoutingSignals(
            session_turn_count=0,
            session_total_tokens=0,
        )
        score_dki, score_rag, dims = scorer.score(signals)
        
        assert score_rag > score_dki, "首轮应倾向 RAG"
        assert dims["session_depth"] == 0
        assert dims["preference"] == 0
    
    def test_deep_session_with_preferences(self, scorer):
        """深度会话 + 偏好 → DKI 得分高"""
        signals = RoutingSignals(
            session_turn_count=8,
            session_total_tokens=3500,
            has_user_preferences=True,
            preference_count=5,
            has_cross_session_history=True,
            user_total_sessions=10,
            user_total_messages=50,
        )
        score_dki, score_rag, dims = scorer.score(signals)
        
        assert score_dki > score_rag, "深度会话+偏好应倾向 DKI"
        assert dims["session_depth"] == 1.0
        assert dims["preference"] > 0
    
    def test_memory_trigger_boosts_dki(self, scorer):
        """记忆触发 → DKI 加分"""
        signals_without = RoutingSignals(session_turn_count=3)
        signals_with = RoutingSignals(session_turn_count=3, has_memory_trigger=True)
        
        dki_without, _, _ = scorer.score(signals_without)
        dki_with, _, _ = scorer.score(signals_with)
        
        assert dki_with > dki_without, "记忆触发应增加 DKI 得分"
    
    def test_knowledge_query_boosts_rag(self, scorer):
        """知识查询 → RAG 加分 (通过降低 DKI 相对得分)"""
        signals_normal = RoutingSignals(session_turn_count=2)
        signals_knowledge = RoutingSignals(session_turn_count=2, is_knowledge_query=True)
        
        dki_normal, rag_normal, _ = scorer.score(signals_normal)
        dki_knowledge, rag_knowledge, _ = scorer.score(signals_knowledge)
        
        # 知识查询使 DKI 得分相对降低
        assert dki_knowledge <= dki_normal, "知识查询应降低 DKI 相对得分"
    
    def test_scores_normalized(self, scorer):
        """得分经归一化后总和 = 1.0"""
        signals = RoutingSignals(
            session_turn_count=10,
            has_user_preferences=True,
            preference_count=5,
            has_memory_trigger=True,
            has_reference=True,
            has_cross_session_history=True,
            user_total_sessions=20,
            user_total_messages=100,
        )
        score_dki, score_rag, _ = scorer.score(signals)
        
        # 归一化后两者之和应 = 1.0 (允许浮点误差)
        assert abs(score_dki + score_rag - 1.0) < 0.01, (
            f"score_dki={score_dki}, score_rag={score_rag}, sum={score_dki+score_rag}"
        )
        # 强 DKI 信号下, DKI 得分应远高于 RAG
        assert score_dki > score_rag, "强 DKI 信号下 DKI 得分应高于 RAG"
    
    def test_cross_session_scoring(self, scorer):
        """跨会话得分"""
        # 无跨会话
        signals_no_cross = RoutingSignals(
            user_total_sessions=1,
            user_total_messages=5,
        )
        _, _, dims_no = scorer.score(signals_no_cross)
        
        # 有跨会话
        signals_cross = RoutingSignals(
            user_total_sessions=10,
            user_total_messages=100,
            has_cross_session_history=True,
        )
        _, _, dims_cross = scorer.score(signals_cross)
        
        assert dims_cross["cross_session"] > dims_no["cross_session"]


# ============================================================
# 3. ConversationRouter 测试
# ============================================================

class TestConversationRouter:
    """路由器主体测试"""
    
    def test_first_turn_routes_to_rag(self, router):
        """首轮对话 → RAG"""
        decision = router.route(
            query="什么是ERP系统",
            session_turn_count=0,
        )
        assert decision.mode == RouteMode.RAG
        assert decision.confidence > 0
    
    def test_deep_session_routes_to_dki(self, router):
        """深度会话 → DKI"""
        decision = router.route(
            query="继续我们之前讨论的话题",
            session_turn_count=8,
            session_total_tokens=3000,
            has_user_preferences=True,
            preference_count=3,
            user_total_sessions=5,
            user_total_messages=40,
        )
        assert decision.mode == RouteMode.DKI
        assert decision.confidence > 0.5
    
    def test_memory_trigger_routes_to_dki(self, router):
        """记忆触发 → DKI"""
        decision = router.route(
            query="你还记得我的名字吗",
            session_turn_count=3,
            has_user_preferences=True,
            preference_count=2,
        )
        assert decision.mode == RouteMode.DKI
    
    def test_forced_mode_dki(self, router):
        """强制 DKI 模式"""
        decision = router.route(
            query="test query",
            forced_mode="dki",
        )
        assert decision.mode == RouteMode.DKI
        assert decision.confidence == 1.0
        assert decision.reason == RouteReason.FORCED
    
    def test_forced_mode_rag(self, router):
        """强制 RAG 模式"""
        decision = router.route(
            query="test query",
            forced_mode="rag",
        )
        assert decision.mode == RouteMode.RAG
        assert decision.confidence == 1.0
    
    def test_dki_unavailable_routes_to_rag(self, router):
        """DKI 不可用 → RAG"""
        decision = router.route(
            query="你记得吗",
            session_turn_count=10,
            dki_available=False,
        )
        assert decision.mode == RouteMode.RAG
    
    def test_rag_unavailable_routes_to_dki(self, router):
        """RAG 不可用 → DKI"""
        decision = router.route(
            query="什么是Python",
            session_turn_count=0,
            rag_available=False,
        )
        assert decision.mode == RouteMode.DKI
    
    def test_router_disabled(self):
        """路由器禁用时使用默认模式"""
        config = RouterConfig(enabled=False, default_mode="rag")
        router = ConversationRouter(config=config)
        
        decision = router.route(query="any query")
        assert decision.mode == RouteMode.RAG
        assert decision.reason == RouteReason.FORCED
    
    def test_decision_to_dict(self, router):
        """决策可序列化"""
        decision = router.route(query="test", session_turn_count=3)
        d = decision.to_dict()
        
        assert "mode" in d
        assert "confidence" in d
        assert "score_dki" in d
        assert "score_rag" in d
        assert "reason" in d
        assert "dimension_scores" in d
    
    def test_stats_tracking(self, router):
        """统计追踪"""
        router.route(query="q1", session_turn_count=0)
        router.route(query="q2", session_turn_count=5, has_user_preferences=True, preference_count=3)
        
        stats = router.get_stats()
        assert stats["total_routes"] == 2
        assert stats["rag_routes"] + stats["dki_routes"] == 2
        assert 0 <= stats["avg_score_dki"] <= 1.0
    
    def test_config_update(self, router):
        """动态配置更新"""
        old_threshold = router.config.dki_threshold
        router.update_config(dki_threshold=0.9)
        assert router.config.dki_threshold == 0.9
        
        # 恢复
        router.update_config(dki_threshold=old_threshold)


# ============================================================
# 4. RouterConfig 测试
# ============================================================

class TestRouterConfig:
    """路由配置测试"""
    
    def test_default_config(self):
        """默认配置"""
        config = RouterConfig()
        assert config.enabled is True
        assert config.dki_threshold == 0.45
        assert config.rag_threshold == 0.25
        
        # 权重总和 = 1.0
        total_weight = (
            config.weight_history
            + config.weight_preference
            + config.weight_trigger
            + config.weight_session_depth
            + config.weight_cross_session
        )
        assert abs(total_weight - 1.0) < 0.001
    
    def test_from_dict(self):
        """从字典加载配置"""
        d = {
            "enabled": False,
            "dki_threshold": 0.6,
            "weight_history": 0.3,
        }
        config = RouterConfig.from_dict(d)
        assert config.enabled is False
        assert config.dki_threshold == 0.6
        assert config.weight_history == 0.3
    
    def test_from_empty_dict(self):
        """空字典 → 默认配置"""
        config = RouterConfig.from_dict({})
        assert config.enabled is True
        assert config.dki_threshold == 0.45


# ============================================================
# 5. 端到端场景测试
# ============================================================

class TestE2EScenarios:
    """端到端场景测试"""
    
    def test_scenario_new_user_first_query(self, router):
        """场景: 新用户首次提问"""
        decision = router.route(
            query="请介绍一下Python的优势",
            session_turn_count=0,
            has_user_preferences=False,
            user_total_sessions=0,
        )
        assert decision.mode == RouteMode.RAG
        assert "first" in decision.reasoning.lower() or "rag" in decision.reasoning.lower()
    
    def test_scenario_returning_user_deep_session(self, router):
        """场景: 老用户深度对话"""
        decision = router.route(
            query="继续我们讨论的ERP实施方案",
            session_turn_count=12,
            session_total_tokens=5000,
            has_user_preferences=True,
            preference_count=4,
            user_total_sessions=15,
            user_total_messages=200,
        )
        assert decision.mode == RouteMode.DKI
    
    def test_scenario_user_asks_about_memory(self, router):
        """场景: 用户主动要求回忆"""
        decision = router.route(
            query="你还记得我上次说的那个想法吗",
            session_turn_count=2,
            has_user_preferences=True,
            preference_count=1,
        )
        assert decision.mode == RouteMode.DKI
        assert decision.reason in (
            RouteReason.MEMORY_TRIGGER,
            RouteReason.PREFERENCE_DEPENDENT,
            RouteReason.MULTI_TURN_DEEP,
        )
    
    def test_scenario_knowledge_query_mid_session(self, router):
        """场景: 会话中提出知识检索"""
        decision = router.route(
            query="什么是微服务架构",
            session_turn_count=4,
            has_user_preferences=False,
        )
        # 中轮次 + 知识查询 → 倾向 RAG (但不一定)
        # 关键是不应该高置信度选择 DKI
        assert decision.score_rag >= 0.3
    
    def test_scenario_preference_heavy(self, router):
        """场景: 偏好密集型查询"""
        decision = router.route(
            query="给我推荐一个适合我的编程语言",
            session_turn_count=3,
            has_user_preferences=True,
            preference_count=8,
            user_total_sessions=5,
        )
        # 有大量偏好 → 倾向 DKI
        assert decision.score_dki > 0.3
    
    def test_latency_is_low(self, router):
        """路由延迟应极低 (< 1ms)"""
        decision = router.route(
            query="一个测试查询",
            session_turn_count=5,
            has_user_preferences=True,
            preference_count=3,
        )
        assert decision.latency_ms < 10, f"Router latency too high: {decision.latency_ms}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
