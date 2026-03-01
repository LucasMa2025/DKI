"""
Reference Resolver 单元测试

测试指代解析器的各种解析场景
"""

import pytest
from datetime import datetime
from dki.core.components.reference_resolver import (
    ReferenceResolver,
    ReferenceResolverConfig,
    ReferenceType,
    ReferenceScope,
    ResolvedReference,
    Message,
)


class TestReferenceResolver:
    """Reference Resolver 测试类"""
    
    @pytest.fixture
    def resolver(self):
        """创建默认配置的解析器"""
        return ReferenceResolver()
    
    @pytest.fixture
    def resolver_cn(self):
        """创建中文解析器"""
        return ReferenceResolver(language="cn")
    
    @pytest.fixture
    def resolver_en(self):
        """创建英文解析器"""
        return ReferenceResolver(language="en")
    
    @pytest.fixture
    def sample_history(self):
        """创建示例历史消息"""
        return [
            Message(role="user", content="你好，我想了解一下Python"),
            Message(role="assistant", content="好的，Python是一种流行的编程语言"),
            Message(role="user", content="它有什么特点？"),
            Message(role="assistant", content="Python简洁易读，有丰富的库"),
            Message(role="user", content="我想学习机器学习"),
            Message(role="assistant", content="我建议从scikit-learn开始"),
        ]
    
    # ============ 时间指代测试 ============
    
    def test_temporal_cn_just_now(self, resolver_cn):
        """测试中文时间指代 - 刚刚"""
        result = resolver_cn.resolve("刚刚你说的是什么意思？")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
        assert result.matched_keyword == "刚刚"
    
    def test_temporal_cn_just_now_alt(self, resolver_cn):
        """测试中文时间指代 - 刚才"""
        result = resolver_cn.resolve("刚才提到的方案")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
        assert result.matched_keyword == "刚才"
    
    def test_temporal_cn_recently(self, resolver_cn):
        """测试中文时间指代 - 最近"""
        result = resolver_cn.resolve("最近我们聊了很多")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.CURRENT_SESSION
        assert result.matched_keyword == "最近"
    
    def test_temporal_cn_last_time(self, resolver_cn):
        """测试中文时间指代 - 上次"""
        result = resolver_cn.resolve("上次你推荐的餐厅")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_5_10_TURNS
        assert result.matched_keyword == "上次"
    
    def test_temporal_en_just_now(self, resolver_en):
        """测试英文时间指代 - just now"""
        result = resolver_en.resolve("What you said just now")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
    
    def test_temporal_en_recently(self, resolver_en):
        """测试英文时间指代 - recently"""
        result = resolver_en.resolve("We talked about this recently")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.CURRENT_SESSION
    
    def test_temporal_en_last_time(self, resolver_en):
        """测试英文时间指代 - last time"""
        result = resolver_en.resolve("The restaurant you recommended last time")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_5_10_TURNS
    
    # ============ 实体指代测试 ============
    
    def test_referential_cn_that_thing(self, resolver_cn):
        """测试中文实体指代 - 那件事"""
        result = resolver_cn.resolve("那件事后来怎么样了？")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.LAST_SHARED_TOPIC
        assert result.matched_keyword == "那件事"
    
    def test_referential_cn_that_issue(self, resolver_cn):
        """测试中文实体指代 - 那个问题"""
        result = resolver_cn.resolve("那个问题解决了吗？")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.LAST_SHARED_TOPIC
        assert result.matched_keyword == "那个问题"
    
    def test_referential_cn_that_topic(self, resolver_cn):
        """测试中文实体指代 - 那个话题"""
        result = resolver_cn.resolve("那个话题我们还没聊完")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.LAST_SHARED_TOPIC
        assert result.matched_keyword == "那个话题"
    
    def test_referential_en_that_thing(self, resolver_en):
        """测试英文实体指代 - that thing"""
        result = resolver_en.resolve("What happened with that thing?")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.LAST_SHARED_TOPIC
    
    def test_referential_en_that_issue(self, resolver_en):
        """测试英文实体指代 - that issue"""
        result = resolver_en.resolve("Is that issue resolved?")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.LAST_SHARED_TOPIC
    
    # ============ 立场指代测试 ============
    
    def test_stance_cn_you_said_before(self, resolver_cn):
        """测试中文立场指代 - 之前你说的"""
        result = resolver_cn.resolve("之前你说的那个观点")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
        assert result.matched_keyword == "之前你说的"
    
    def test_stance_cn_you_mentioned(self, resolver_cn):
        """测试中文立场指代 - 你之前提到"""
        result = resolver_cn.resolve("你之前提到的建议")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_stance_en_you_said_earlier(self, resolver_en):
        """测试英文立场指代 - you said earlier"""
        result = resolver_en.resolve("What you said earlier about this")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_stance_en_you_mentioned(self, resolver_en):
        """测试英文立场指代 - you mentioned"""
        result = resolver_en.resolve("The suggestion you mentioned")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    # ============ v6.3 首次出现指代测试 (first_occurrence_session) ============
    
    def test_first_occurrence_cn_first_time(self, resolver_cn):
        """测试中文首次指代 - 第一次"""
        result = resolver_cn.resolve("第一次聊这个话题是什么时候？")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert result.matched_keyword == "第一次"
    
    def test_first_occurrence_cn_first(self, resolver_cn):
        """测试中文首次指代 - 首次"""
        result = resolver_cn.resolve("首次提到这个概念是什么时候？")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert result.matched_keyword == "首次"
    
    def test_first_occurrence_cn_start(self, resolver_cn):
        """测试中文首次指代 - 开始"""
        result = resolver_cn.resolve("开始的时候你说了什么？")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert result.matched_keyword == "开始"
    
    def test_first_occurrence_cn_how_started(self, resolver_cn):
        """测试中文主题起源指代 - 怎么聊起 → FIRST_TOPIC_GENESIS"""
        result = resolver_cn.resolve("我们怎么聊起这个话题的？")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.matched_keyword == "怎么聊起"
    
    def test_first_occurrence_cn_how_mentioned(self, resolver_cn):
        """测试中文主题起源指代 - 怎么说起 → FIRST_TOPIC_GENESIS"""
        result = resolver_cn.resolve("怎么说起这件事的？")
        assert result.reference_type == ReferenceType.REFERENTIAL
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.matched_keyword == "怎么说起"
    
    def test_first_occurrence_cn_last(self, resolver_cn):
        """测试中文 '最后' 映射到 LAST_1_3_TURNS"""
        result = resolver_cn.resolve("最后你说了什么？")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
        assert result.matched_keyword == "最后"
    
    def test_first_occurrence_en_first_time(self, resolver_en):
        """测试英文首次指代 - first time"""
        result = resolver_en.resolve("When was the first time we discussed this?")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
    
    def test_first_occurrence_en_at_the_beginning(self, resolver_en):
        """测试英文首次指代 - at the beginning"""
        result = resolver_en.resolve("What did you say at the beginning?")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
    
    def test_first_occurrence_en_when_we_started(self, resolver_en):
        """测试英文首次指代 - when we started"""
        result = resolver_en.resolve("What were we talking about when we started?")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
    
    def test_first_occurrence_en_at_last(self, resolver_en):
        """测试英文 'at last' 映射到 LAST_1_3_TURNS"""
        result = resolver_en.resolve("What did you say at last?")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
    
    def test_first_occurrence_en_in_the_end(self, resolver_en):
        """测试英文 'in the end' 映射到 LAST_1_3_TURNS"""
        result = resolver_en.resolve("What was your conclusion in the end?")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_1_3_TURNS
    
    def test_first_occurrence_recall_turns(self, resolver_cn):
        """测试首次出现的召回轮数 = session_max_turns"""
        result = resolver_cn.resolve("第一次聊这个")
        assert result.recall_turns == resolver_cn.config.session_max_turns
    
    def test_first_occurrence_with_history(self, resolver_cn, sample_history):
        """测试首次出现指代带历史解析 — 应返回会话开头内容"""
        result = resolver_cn.resolve("第一次聊的是什么？", history=sample_history)
        
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert result.resolved_content is not None
        # 首次出现应从开头取，所以 source_turns 应从 0 开始
        assert 0 in result.source_turns
        # 内容应包含第一条消息
        assert "Python" in result.resolved_content
    
    def test_first_occurrence_with_long_history(self, resolver_cn):
        """测试首次出现指代 — 长历史只取前 N 条"""
        # 创建 20 条消息的历史
        long_history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            long_history.append(Message(role=role, content=f"消息 {i}"))
        
        result = resolver_cn.resolve("第一次说的是什么？", history=long_history)
        
        assert result.resolved_content is not None
        # 默认 last_few_turns=3, 所以取前 3*2=6 条
        assert "消息 0" in result.resolved_content
        assert "消息 5" in result.resolved_content
        # 不应包含后面的消息
        assert "消息 10" not in result.resolved_content
    
    def test_first_occurrence_empty_history(self, resolver_cn):
        """测试首次出现指代 — 空历史"""
        result = resolver_cn.resolve("第一次聊的是什么？", history=[])
        
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert result.resolved_content is None
        assert result.source_turns == []
    
    # ============ 无指代测试 ============
    
    def test_no_reference_normal_question(self, resolver):
        """测试普通问题无指代"""
        result = resolver.resolve("今天天气怎么样？")
        assert result.reference_type == ReferenceType.NONE
    
    def test_no_reference_empty_query(self, resolver):
        """测试空查询无指代"""
        result = resolver.resolve("")
        assert result.reference_type == ReferenceType.NONE
    
    def test_no_reference_whitespace(self, resolver):
        """测试空白查询无指代"""
        result = resolver.resolve("   ")
        assert result.reference_type == ReferenceType.NONE
    
    # ============ 召回轮数测试 ============
    
    def test_recall_turns_just_now(self, resolver_cn):
        """测试 '刚刚' 的召回轮数"""
        result = resolver_cn.resolve("刚刚说的")
        assert result.recall_turns == resolver_cn.config.last_few_turns
    
    def test_recall_turns_recently(self, resolver_cn):
        """测试 '最近' 的召回轮数"""
        result = resolver_cn.resolve("最近聊的")
        assert result.recall_turns == resolver_cn.config.session_max_turns
    
    def test_recall_turns_last_time(self, resolver_cn):
        """测试 '上次' 的召回轮数"""
        result = resolver_cn.resolve("上次说的")
        assert result.recall_turns == resolver_cn.config.recent_turns
    
    # ============ 带历史的解析测试 ============
    
    def test_resolve_with_history(self, resolver_cn, sample_history):
        """测试带历史的解析"""
        result = resolver_cn.resolve("刚刚你说的", history=sample_history)
        
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.resolved_content is not None
        assert len(result.source_turns) > 0
    
    def test_resolve_with_history_content_format(self, resolver_cn, sample_history):
        """测试解析内容格式"""
        result = resolver_cn.resolve("刚刚你说的", history=sample_history)
        
        # 内容应该包含用户和助手的对话
        if result.resolved_content:
            assert "用户" in result.resolved_content or "助手" in result.resolved_content
    
    # ============ 配置更新测试 ============
    
    def test_update_config_just_now_turns(self, resolver):
        """测试更新 '刚刚' 召回轮数"""
        resolver.update_config(just_now_turns=5)
        assert resolver.config.last_few_turns == 5
    
    def test_update_config_recently_turns(self, resolver):
        """测试更新 '最近' 召回轮数"""
        resolver.update_config(recently_turns=20)
        assert resolver.config.recent_turns == 20
    
    def test_update_config_session_max_turns(self, resolver):
        """测试更新会话最大轮数"""
        resolver.update_config(session_max_turns=100)
        assert resolver.config.session_max_turns == 100
    
    def test_update_config_legacy_params(self, resolver):
        """测试兼容旧参数名"""
        resolver.update_config(last_few_turns=7, recent_turns=15)
        assert resolver.config.last_few_turns == 7
        assert resolver.config.recent_turns == 15
    
    def test_update_config_new_params_priority(self, resolver):
        """测试新参数优先级"""
        resolver.update_config(
            just_now_turns=5,
            last_few_turns=10,  # 旧参数应被忽略
        )
        assert resolver.config.last_few_turns == 5
    
    # ============ 添加映射测试 ============
    
    def test_add_mapping_cn(self, resolver):
        """测试添加中文映射"""
        resolver.add_mapping(
            keyword="前天",
            scope="last_5_10_turns",
            ref_type="temporal",
            language="cn"
        )
        
        result = resolver.resolve("前天你说的")
        assert result.reference_type == ReferenceType.TEMPORAL
        assert result.scope == ReferenceScope.LAST_5_10_TURNS
    
    def test_add_mapping_en(self, resolver):
        """测试添加英文映射"""
        resolver.add_mapping(
            keyword="yesterday",
            scope="last_5_10_turns",
            ref_type="temporal",
            language="en"
        )
        
        result = resolver.resolve("What you said yesterday")
        assert result.reference_type == ReferenceType.TEMPORAL
    
    # ============ 统计信息测试 ============
    
    def test_get_stats(self, resolver):
        """测试获取统计信息"""
        stats = resolver.get_stats()
        
        assert 'language' in stats
        assert 'last_few_turns' in stats
        assert 'recent_turns' in stats
        assert 'session_max_turns' in stats
        assert 'mappings_cn_count' in stats
        assert 'mappings_en_count' in stats
    
    # ============ 自动语言检测测试 ============
    
    def test_auto_language_detection_cn(self, resolver):
        """测试自动语言检测 - 中文"""
        result = resolver.resolve("刚刚你说的")
        assert result.metadata.get('language') == 'cn'
    
    def test_auto_language_detection_en(self, resolver):
        """测试自动语言检测 - 英文"""
        result = resolver.resolve("What you said just now")
        assert result.metadata.get('language') == 'en'


class TestReferenceResolverConfig:
    """Reference Resolver 配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ReferenceResolverConfig()
        
        assert config.last_few_turns == 3
        assert config.recent_turns == 10
        assert config.session_max_turns == 50
        assert len(config.reference_mappings_cn) > 0
        assert len(config.reference_mappings_en) > 0
    
    def test_default_config_has_first_occurrence_mappings(self):
        """测试默认配置包含 first_occurrence_session 映射"""
        config = ReferenceResolverConfig()
        
        # 中文映射 — 无主题限定 → first_occurrence_session
        assert "第一次" in config.reference_mappings_cn
        assert config.reference_mappings_cn["第一次"]["scope"] == "first_occurrence_session"
        assert "首次" in config.reference_mappings_cn
        assert config.reference_mappings_cn["首次"]["scope"] == "first_occurrence_session"
        assert "开始" in config.reference_mappings_cn
        assert config.reference_mappings_cn["开始"]["scope"] == "first_occurrence_session"
        
        # 中文映射 — 带主题暗示 → first_topic_genesis
        assert "怎么聊起" in config.reference_mappings_cn
        assert config.reference_mappings_cn["怎么聊起"]["scope"] == "first_topic_genesis"
        assert "怎么说起" in config.reference_mappings_cn
        assert config.reference_mappings_cn["怎么说起"]["scope"] == "first_topic_genesis"
        assert "最早提到" in config.reference_mappings_cn
        assert config.reference_mappings_cn["最早提到"]["scope"] == "first_topic_genesis"
        
        # 英文映射 — 无主题限定 → first_occurrence_session
        assert "first time" in config.reference_mappings_en
        assert config.reference_mappings_en["first time"]["scope"] == "first_occurrence_session"
        assert "at the beginning" in config.reference_mappings_en
        assert config.reference_mappings_en["at the beginning"]["scope"] == "first_occurrence_session"
        assert "when we started" in config.reference_mappings_en
        assert config.reference_mappings_en["when we started"]["scope"] == "first_occurrence_session"
        
        # 英文映射 — 带主题暗示 → first_topic_genesis
        assert "how did we start talking about" in config.reference_mappings_en
        assert config.reference_mappings_en["how did we start talking about"]["scope"] == "first_topic_genesis"
        assert "when we talked about" in config.reference_mappings_en
        assert config.reference_mappings_en["when we talked about"]["scope"] == "first_topic_genesis"
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            'last_few_turns': 5,
            'recent_turns': 15,
            'session_max_turns': 100,
            'use_llm_fallback': True,
        }
        
        config = ReferenceResolverConfig.from_dict(data)
        
        assert config.last_few_turns == 5
        assert config.recent_turns == 15
        assert config.session_max_turns == 100
        assert config.use_llm_fallback is True
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = ReferenceResolverConfig()
        data = config.to_dict()
        
        assert 'last_few_turns' in data
        assert 'recent_turns' in data
        assert 'session_max_turns' in data
        assert 'reference_mappings_cn' in data
        assert 'reference_mappings_en' in data


class TestResolvedReference:
    """ResolvedReference 数据类测试"""
    
    def test_recall_turns_property(self):
        """测试 recall_turns 属性"""
        ref = ResolvedReference(
            reference_type=ReferenceType.TEMPORAL,
            scope=ReferenceScope.LAST_1_3_TURNS,
            metadata={'recall_turns': 5}
        )
        
        assert ref.recall_turns == 5
    
    def test_recall_turns_none(self):
        """测试 recall_turns 为 None"""
        ref = ResolvedReference(
            reference_type=ReferenceType.NONE,
            scope=ReferenceScope.CUSTOM,
        )
        
        assert ref.recall_turns is None


class TestReferenceScope:
    """v6.3: ReferenceScope 枚举测试"""
    
    def test_first_occurrence_session_enum_exists(self):
        """测试 FIRST_OCCURRENCE_SESSION 枚举值存在"""
        scope = ReferenceScope.FIRST_OCCURRENCE_SESSION
        assert scope.value == "first_occurrence_session"
    
    def test_first_occurrence_session_from_string(self):
        """测试从字符串创建 FIRST_OCCURRENCE_SESSION"""
        scope = ReferenceScope("first_occurrence_session")
        assert scope == ReferenceScope.FIRST_OCCURRENCE_SESSION
    
    def test_all_scopes_present(self):
        """测试所有 scope 枚举值都存在"""
        expected_scopes = {
            "last_1_3_turns",
            "last_5_10_turns",
            "current_session",
            "last_shared_topic",
            "assistant_last_stance",
            "first_occurrence_session",
            "first_topic_genesis",
            "custom",
        }
        actual_scopes = {s.value for s in ReferenceScope}
        assert expected_scopes == actual_scopes


class TestGetRecallTurnsForScope:
    """v6.3: _get_recall_turns_for_scope 方法测试"""
    
    def test_first_occurrence_returns_session_max(self):
        """测试 FIRST_OCCURRENCE_SESSION 返回 session_max_turns"""
        resolver = ReferenceResolver()
        turns = resolver._get_recall_turns_for_scope(ReferenceScope.FIRST_OCCURRENCE_SESSION)
        assert turns == resolver.config.session_max_turns
    
    def test_first_occurrence_custom_config(self):
        """测试自定义配置下 FIRST_OCCURRENCE_SESSION 的召回轮数"""
        config = ReferenceResolverConfig(session_max_turns=100)
        resolver = ReferenceResolver(config=config)
        turns = resolver._get_recall_turns_for_scope(ReferenceScope.FIRST_OCCURRENCE_SESSION)
        assert turns == 100
    
    def test_all_scopes_return_positive_int(self):
        """测试所有 scope 都返回正整数"""
        resolver = ReferenceResolver()
        for scope in ReferenceScope:
            if scope == ReferenceScope.CUSTOM:
                continue  # CUSTOM 走默认分支
            turns = resolver._get_recall_turns_for_scope(scope)
            assert isinstance(turns, int)
            assert turns > 0, f"Scope {scope} returned non-positive turns: {turns}"


class TestFindFirstOccurrence:
    """v6.3: _find_first_occurrence 方法测试"""
    
    def test_empty_history(self):
        """测试空历史"""
        resolver = ReferenceResolver()
        content, turns = resolver._find_first_occurrence([])
        assert content is None
        assert turns == []
    
    def test_short_history(self):
        """测试短历史 (少于 N*2 条)"""
        resolver = ReferenceResolver()  # last_few_turns=3, 所以取前 6 条
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        content, turns = resolver._find_first_occurrence(history)
        assert content is not None
        assert "你好" in content
        assert turns == [0, 1]
    
    def test_long_history_takes_first_n(self):
        """测试长历史只取前 N 条"""
        config = ReferenceResolverConfig(last_few_turns=2)
        resolver = ReferenceResolver(config=config)
        
        history = [Message(role="user" if i % 2 == 0 else "assistant", content=f"msg_{i}") for i in range(20)]
        content, turns = resolver._find_first_occurrence(history)
        
        # last_few_turns=2, 所以取前 2*2=4 条
        assert "msg_0" in content
        assert "msg_3" in content
        assert "msg_4" not in content
        assert turns == [0, 1, 2, 3]
    
    def test_returns_formatted_content(self):
        """测试返回格式化内容"""
        resolver = ReferenceResolver()
        history = [
            Message(role="user", content="第一条消息"),
            Message(role="assistant", content="第一条回复"),
        ]
        content, turns = resolver._find_first_occurrence(history)
        assert "用户: 第一条消息" in content
        assert "助手: 第一条回复" in content


class TestTopicGenesis:
    """v6.3: FIRST_TOPIC_GENESIS 主题起源定位测试"""
    
    @pytest.fixture
    def resolver_cn(self):
        return ReferenceResolver(language="cn")
    
    @pytest.fixture
    def resolver_en(self):
        return ReferenceResolver(language="en")
    
    @pytest.fixture
    def mozart_dki_history(self):
        """模拟建议中的经典场景: 莫扎特 → KV → DKI"""
        return [
            Message(role="user", content="你知道莫扎特的作品号吗？"),
            Message(role="assistant", content="莫扎特的作品号使用 KV 编号系统，由路德维希·冯·克歇尔编纂。"),
            Message(role="user", content="KV 编号有什么特点？"),
            Message(role="assistant", content="KV 编号按照创作时间排序，从 KV 1 到 KV 626。"),
            Message(role="user", content="说到 KV，我突然想起了 DKI 系统，它也用到了 K/V Cache"),
            Message(role="assistant", content="是的，DKI 是 Dynamic K/V Injection 的缩写，它利用 K/V Cache 来实现记忆注入。"),
            Message(role="user", content="DKI 怎么实现记忆召回的？"),
            Message(role="assistant", content="DKI 使用多信号融合召回，包括关键词、BM25、向量相似度等。"),
        ]
    
    # ============ 关键词映射测试 ============
    
    def test_genesis_cn_how_started(self, resolver_cn):
        """'怎么聊起' 应映射到 FIRST_TOPIC_GENESIS"""
        result = resolver_cn.resolve("我们怎么聊起 DKI 的？")
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.reference_type == ReferenceType.REFERENTIAL
    
    def test_genesis_cn_earliest_mention(self, resolver_cn):
        """'最早提到' 应映射到 FIRST_TOPIC_GENESIS"""
        result = resolver_cn.resolve("最早提到 DKI 是什么时候？")
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
    
    def test_genesis_en_how_did_we_start(self, resolver_en):
        """'how did we start talking about' 应映射到 FIRST_TOPIC_GENESIS"""
        result = resolver_en.resolve("how did we start talking about DKI?")
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
    
    def test_genesis_en_when_first_discuss(self, resolver_en):
        """'when did we first discuss' 应映射到 FIRST_TOPIC_GENESIS"""
        result = resolver_en.resolve("when did we first discuss DKI?")
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
    
    # ============ Level 1: 命名+定义 (强) ============
    
    def test_genesis_level1_naming_definition(self, resolver_cn):
        """Level 1: 命中命名+定义模式 → confidence=1.0"""
        # 构造一个 Level 1 直接命中的场景 (首次出现即为命名定义)
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
            Message(role="user", content="DKI 是 Dynamic K/V Injection，你知道吗？"),
            Message(role="assistant", content="是的，DKI 是一种记忆注入技术。"),
        ]
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.resolved_content is not None
        assert "DKI" in result.resolved_content
        assert result.metadata.get("genesis_level") == "naming_definition"
        assert result.confidence == 1.0
    
    def test_genesis_mozart_dki_hits_convergence(self, resolver_cn, mozart_dki_history):
        """经典场景: 莫扎特→DKI, 首次命中应为 Level 2 (概念收敛)
        
        因为 turn 4 '说到 KV，我突然想起了 DKI' 包含收敛标记 '想起了',
        这是 DKI 的真正起源点 (用户从类比引入), 而非 turn 5 的命名定义。
        """
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=mozart_dki_history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.resolved_content is not None
        assert "DKI" in result.resolved_content
        # 应命中 Level 2: 概念收敛 (turn 4 包含 "想起了" + "DKI")
        assert result.metadata.get("genesis_level") == "concept_convergence"
        assert result.confidence == 0.75
        assert result.metadata.get("genesis_turn_index") == 4
    
    # ============ Level 2: 概念收敛 (中) ============
    
    def test_genesis_level2_concept_convergence(self, resolver_cn):
        """Level 2: 命中收敛标记 '想起了' + 主题实体"""
        history = [
            Message(role="user", content="今天天气真好"),
            Message(role="assistant", content="是啊，适合出去走走"),
            Message(role="user", content="说到走路，我想起了健身计划"),
            Message(role="assistant", content="健身计划是个好主意"),
        ]
        result = resolver_cn.resolve(
            "我们怎么聊起健身计划的？",
            history=history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.metadata.get("genesis_level") == "concept_convergence"
        assert result.confidence == 0.75
        assert "健身计划" in result.resolved_content
    
    # ============ Level 3: 角色切换 (弱) ============
    
    def test_genesis_level3_role_shift(self, resolver_cn):
        """Level 3: 主题出现后紧跟深入讨论标记"""
        history = [
            Message(role="user", content="随便聊聊吧"),
            Message(role="assistant", content="好的，你想聊什么？"),
            Message(role="user", content="听说过 Rust 吗"),
            Message(role="user", content="Rust 怎么实现内存安全的？"),
            Message(role="assistant", content="Rust 通过所有权系统实现内存安全"),
        ]
        result = resolver_cn.resolve(
            "我们怎么聊起 Rust 的？",
            history=history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        # 应该命中 Level 3 (角色切换: "怎么实现" 出现在 Rust 之后)
        assert result.metadata.get("genesis_level") == "role_shift"
        assert result.confidence == 0.5
    
    # ============ 降级测试 ============
    
    def test_genesis_fallback_keyword_only(self, resolver_cn):
        """主题实体存在但三层都未命中 → keyword_only (confidence=0.4)"""
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好"),
            Message(role="user", content="Python 很有趣"),
            Message(role="assistant", content="确实"),
        ]
        result = resolver_cn.resolve(
            "我们怎么聊起 Python 的？",
            history=history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.metadata.get("genesis_level") == "keyword_only"
        assert result.confidence == 0.4
    
    def test_genesis_fallback_no_entity_in_history(self, resolver_cn):
        """主题实体在历史中不存在 → 降级到 first_occurrence"""
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好"),
        ]
        result = resolver_cn.resolve(
            "我们怎么聊起量子计算的？",
            history=history,
        )
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.metadata.get("genesis_level") == "fallback"
        assert result.confidence == 0.3
    
    def test_genesis_no_history(self, resolver_cn):
        """无历史 → resolved_content 为 None"""
        result = resolver_cn.resolve("我们怎么聊起 DKI 的？")
        assert result.scope == ReferenceScope.FIRST_TOPIC_GENESIS
        assert result.resolved_content is None
    
    # ============ 上下文窗口测试 ============
    
    def test_genesis_context_window(self, resolver_cn, mozart_dki_history):
        """Genesis 命中后应包含前后 context_window 条消息"""
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=mozart_dki_history,
        )
        # genesis_context_window=1, 命中点前后各 1 条
        genesis_idx = result.metadata.get("genesis_turn_index")
        assert genesis_idx is not None
        # source_turns 应包含 [genesis_idx-1, genesis_idx, genesis_idx+1]
        assert len(result.source_turns) <= 3
        assert genesis_idx in result.source_turns
    
    # ============ 主题提取测试 ============
    
    def test_extract_topic_cn(self):
        """测试中文主题提取"""
        resolver = ReferenceResolver(language="cn")
        topic = resolver._extract_topic_from_query("我们怎么聊起 DKI 的？")
        assert topic is not None
        assert "DKI" in topic
    
    def test_extract_topic_cn_chinese_entity(self):
        """测试中文实体提取"""
        resolver = ReferenceResolver(language="cn")
        topic = resolver._extract_topic_from_query("最早提到机器学习是什么时候？")
        assert topic is not None
        assert "机器学习" in topic
    
    def test_extract_topic_en(self):
        """测试英文主题提取"""
        resolver = ReferenceResolver(language="en")
        topic = resolver._extract_topic_from_query("how did we start talking about DKI?")
        assert topic is not None
        assert "DKI" in topic
    
    def test_extract_topic_empty(self):
        """无法提取主题 → 返回 None"""
        resolver = ReferenceResolver(language="cn")
        # 纯指代词, 无实体
        topic = resolver._extract_topic_from_query("怎么聊起的？")
        # 可能返回 None 或空 — 取决于停用词覆盖
        # 关键是不应崩溃
        assert topic is None or isinstance(topic, str)
    
    # ============ 召回轮数测试 ============
    
    def test_genesis_recall_turns(self):
        """FIRST_TOPIC_GENESIS 召回轮数 = session_max_turns"""
        resolver = ReferenceResolver()
        turns = resolver._get_recall_turns_for_scope(ReferenceScope.FIRST_TOPIC_GENESIS)
        assert turns == resolver.config.session_max_turns
    
    # ============ 权重/置信度传递测试 ============
    
    def test_genesis_confidence_propagated_to_result(self, resolver_cn, mozart_dki_history):
        """Genesis confidence 应传递到 ResolvedReference.confidence"""
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=mozart_dki_history,
        )
        # 莫扎特→DKI 场景命中 Level 2 (概念收敛) → confidence=0.75
        assert result.confidence == 0.75
        assert result.metadata.get("genesis_confidence") == 0.75
    
    def test_genesis_metadata_contains_topic_entity(self, resolver_cn, mozart_dki_history):
        """Genesis metadata 应包含 topic_entity"""
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=mozart_dki_history,
        )
        assert result.metadata.get("topic_entity") is not None
        assert "DKI" in result.metadata["topic_entity"]
    
    # ============ 经典场景: 莫扎特 → DKI ============
    
    def test_mozart_dki_scenario_does_not_return_mozart(self, resolver_cn, mozart_dki_history):
        """经典场景: 问 DKI 起源时, 不应返回莫扎特相关内容作为起点"""
        result = resolver_cn.resolve(
            "我们怎么聊起 DKI 的？",
            history=mozart_dki_history,
        )
        # 不应该返回 index 0/1 (莫扎特相关)
        assert 0 not in result.source_turns or result.metadata.get("genesis_level") != "fallback"
        # 应该指向 DKI 首次被当作讨论对象的那一轮
        genesis_idx = result.metadata.get("genesis_turn_index")
        assert genesis_idx is not None
        assert genesis_idx >= 4  # DKI 在 index 4 或 5 首次出现


class TestCorrectionNegation:
    """v6.4: 纠正/否定指代测试 — '不是的,你再想想' 等场景"""
    
    @pytest.fixture
    def resolver_cn(self):
        return ReferenceResolver(language="cn")
    
    @pytest.fixture
    def resolver_en(self):
        return ReferenceResolver(language="en")
    
    @pytest.fixture
    def correction_history(self):
        """模拟纠正场景的历史"""
        return [
            Message(role="user", content="莫扎特出生在哪一年？"),
            Message(role="assistant", content="莫扎特出生于 1756 年，出生地是奥地利的萨尔茨堡。"),
            Message(role="user", content="他最著名的作品是什么？"),
            Message(role="assistant", content="莫扎特最著名的作品包括《费加罗的婚礼》、《唐·乔瓦尼》和《魔笛》。"),
        ]
    
    # ============ 中文纠正关键词映射测试 ============
    
    def test_correction_cn_bu_shi_de(self, resolver_cn):
        """'不是的' 应映射到 ASSISTANT_LAST_STANCE / stance"""
        result = resolver_cn.resolve("不是的,你再想想")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_cn_bu_dui(self, resolver_cn):
        """'不对' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_cn.resolve("不对，应该是另一个答案")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_cn_ni_shuo_cuo_le(self, resolver_cn):
        """'你说错了' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_cn.resolve("你说错了，不是这样的")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_cn_zai_xiang_xiang(self, resolver_cn):
        """'再想想' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_cn.resolve("再想想，答案不对")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_cn_cuo_le(self, resolver_cn):
        """'错了' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_cn.resolve("错了，再试一次")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_cn_bu_shi_zhe_yang(self, resolver_cn):
        """'不是这样' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_cn.resolve("不是这样的，你理解有误")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    # ============ 英文纠正关键词映射测试 ============
    
    def test_correction_en_thats_wrong(self, resolver_en):
        """'that's wrong' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_en.resolve("that's wrong, try again")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_en_think_again(self, resolver_en):
        """'think again' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_en.resolve("think again, that's not the answer")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_en_youre_wrong(self, resolver_en):
        """'you're wrong' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_en.resolve("you're wrong about that")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_en_not_correct(self, resolver_en):
        """'not correct' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_en.resolve("that's not correct")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    def test_correction_en_reconsider(self, resolver_en):
        """'reconsider' 应映射到 ASSISTANT_LAST_STANCE"""
        result = resolver_en.resolve("please reconsider your answer")
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
    
    # ============ 纠正场景带历史测试 ============
    
    def test_correction_with_history_returns_last_assistant(self, resolver_cn, correction_history):
        """纠正场景应返回助手最近一条回复"""
        result = resolver_cn.resolve("不是的,你再想想", history=correction_history)
        
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
        assert result.resolved_content is not None
        # 应包含助手最近的回复内容
        assert "费加罗" in result.resolved_content or "魔笛" in result.resolved_content
        # 应包含前一条用户消息作为上下文
        assert "最著名" in result.resolved_content
        # metadata 应标记为纠正
        assert result.metadata.get("is_correction") is True
    
    def test_correction_with_history_source_turns(self, resolver_cn, correction_history):
        """纠正场景的 source_turns 应指向助手最近回复及其上下文"""
        result = resolver_cn.resolve("错了", history=correction_history)
        
        # 最后一条 assistant 消息是 index 3, 前一条 user 是 index 2
        assert 3 in result.source_turns
        assert 2 in result.source_turns
    
    def test_correction_empty_history(self, resolver_cn):
        """纠正场景空历史 → resolved_content 为 None"""
        result = resolver_cn.resolve("不是的,你再想想", history=[])
        
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
        assert result.resolved_content is None
    
    def test_correction_no_assistant_in_history(self, resolver_cn):
        """纠正场景但历史中无助手消息 → resolved_content 为 None"""
        history = [
            Message(role="user", content="你好"),
            Message(role="user", content="再说一次"),
        ]
        result = resolver_cn.resolve("不是的,你再想想", history=history)
        
        assert result.reference_type == ReferenceType.STANCE
        assert result.resolved_content is None
    
    # ============ 非纠正的立场查询不受影响 ============
    
    def test_stance_query_not_affected(self, resolver_cn, correction_history):
        """传统立场查询 ('之前你说的') 不应被标记为纠正"""
        result = resolver_cn.resolve("之前你说的那个观点", history=correction_history)
        
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
        # 不应有 is_correction 标记
        assert result.metadata.get("is_correction") is not True
    
    # ============ _is_correction_query 方法测试 ============
    
    def test_is_correction_query_cn(self, resolver_cn):
        """测试中文纠正检测"""
        assert resolver_cn._is_correction_query("不是的,你再想想") is True
        assert resolver_cn._is_correction_query("错了") is True
        assert resolver_cn._is_correction_query("你说错了") is True
        assert resolver_cn._is_correction_query("今天天气怎么样") is False
    
    def test_is_correction_query_en(self, resolver_en):
        """测试英文纠正检测"""
        assert resolver_en._is_correction_query("that's wrong") is True
        assert resolver_en._is_correction_query("think again") is True
        assert resolver_en._is_correction_query("reconsider your answer") is True
        assert resolver_en._is_correction_query("how is the weather") is False
    
    # ============ 经典场景: "不是的,你再想想" ============
    
    def test_classic_correction_scenario(self, resolver_cn):
        """经典场景: 用户问问题 → 助手回答 → 用户说'不是的,你再想想'"""
        history = [
            Message(role="user", content="1+1等于几？"),
            Message(role="assistant", content="1+1等于3。"),
        ]
        result = resolver_cn.resolve("不是的,你再想想", history=history)
        
        assert result.reference_type == ReferenceType.STANCE
        assert result.scope == ReferenceScope.ASSISTANT_LAST_STANCE
        assert result.resolved_content is not None
        assert "1+1等于3" in result.resolved_content
        assert result.metadata.get("is_correction") is True
    
    def test_recall_turns_for_correction(self, resolver_cn):
        """纠正场景的召回轮数应为 recent_turns"""
        result = resolver_cn.resolve("不是的,你再想想")
        assert result.recall_turns == resolver_cn.config.recent_turns


class TestGenesisConfig:
    """v6.3: Genesis 配置测试"""
    
    def test_default_genesis_config(self):
        """测试默认 Genesis 配置"""
        config = ReferenceResolverConfig()
        assert config.genesis_context_window == 1
        assert "naming_definition" in config.genesis_confidence_levels
        assert "concept_convergence" in config.genesis_confidence_levels
        assert "role_shift" in config.genesis_confidence_levels
        assert len(config.genesis_naming_patterns) > 0
        assert len(config.genesis_convergence_markers) > 0
        assert len(config.genesis_role_shift_markers) > 0
    
    def test_genesis_config_from_dict(self):
        """测试从字典加载 Genesis 配置"""
        data = {
            'genesis_context_window': 2,
            'genesis_confidence_levels': {
                'naming_definition': 0.9,
                'concept_convergence': 0.6,
                'role_shift': 0.3,
            },
        }
        config = ReferenceResolverConfig.from_dict(data)
        assert config.genesis_context_window == 2
        assert config.genesis_confidence_levels['naming_definition'] == 0.9
    
    def test_genesis_config_to_dict(self):
        """测试 Genesis 配置序列化"""
        config = ReferenceResolverConfig()
        d = config.to_dict()
        assert 'genesis_context_window' in d
        assert 'genesis_confidence_levels' in d
        assert 'genesis_naming_patterns' in d
        assert 'genesis_convergence_markers' in d
        assert 'genesis_role_shift_markers' in d
    
    def test_custom_genesis_context_window(self):
        """测试自定义 context_window 影响返回范围"""
        config = ReferenceResolverConfig(genesis_context_window=2)
        resolver = ReferenceResolver(config=config, language="cn")
        
        history = [
            Message(role="user", content="消息0"),
            Message(role="assistant", content="消息1"),
            Message(role="user", content="消息2"),
            Message(role="assistant", content="消息3"),
            Message(role="user", content="说到消息2，我想起了 TestTopic"),
            Message(role="assistant", content="TestTopic 是一个测试主题"),
            Message(role="user", content="消息6"),
            Message(role="assistant", content="消息7"),
        ]
        result = resolver.resolve(
            "我们怎么聊起 TestTopic 的？",
            history=history,
        )
        # context_window=2, 命中点前后各 2 条
        genesis_idx = result.metadata.get("genesis_turn_index")
        if genesis_idx is not None:
            expected_start = max(0, genesis_idx - 2)
            expected_end = min(len(history), genesis_idx + 3)
            assert len(result.source_turns) == expected_end - expected_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
