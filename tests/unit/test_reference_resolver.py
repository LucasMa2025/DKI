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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
