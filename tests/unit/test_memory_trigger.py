"""
Memory Trigger 单元测试

测试记忆触发器的各种触发场景
"""

import pytest
from dki.core.components.memory_trigger import (
    MemoryTrigger,
    MemoryTriggerConfig,
    TriggerType,
    TriggerResult,
)


class TestMemoryTrigger:
    """Memory Trigger 测试类"""
    
    @pytest.fixture
    def trigger(self):
        """创建默认配置的触发器"""
        return MemoryTrigger()
    
    @pytest.fixture
    def trigger_cn(self):
        """创建中文触发器"""
        return MemoryTrigger(language="cn")
    
    @pytest.fixture
    def trigger_en(self):
        """创建英文触发器"""
        return MemoryTrigger(language="en")
    
    # ============ 元认知表达测试 ============
    
    def test_meta_cognitive_cn_just_discussed(self, trigger_cn):
        """测试中文元认知表达 - 刚刚讨论"""
        result = trigger_cn.detect("我们刚刚讨论的那个方案怎么样？")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    def test_meta_cognitive_cn_before_you_said(self, trigger_cn):
        """测试中文元认知表达 - 之前你说"""
        result = trigger_cn.detect("之前你说过这个问题有解决方案")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    def test_meta_cognitive_cn_that_thing(self, trigger_cn):
        """测试中文元认知表达 - 那件事"""
        result = trigger_cn.detect("那件事后来怎么样了？")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    def test_meta_cognitive_cn_do_you_remember(self, trigger_cn):
        """测试中文元认知表达 - 你还记得吗"""
        result = trigger_cn.detect("你还记得我们上次聊的内容吗？")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    def test_meta_cognitive_en_just_discussed(self, trigger_en):
        """测试英文元认知表达 - just discussed"""
        result = trigger_en.detect("What we just discussed sounds good")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    def test_meta_cognitive_en_you_mentioned(self, trigger_en):
        """测试英文元认知表达 - you mentioned"""
        result = trigger_en.detect("Earlier you mentioned a solution")
        assert result.triggered
        assert result.trigger_type == TriggerType.META_COGNITIVE
    
    # ============ 状态变化测试 ============
    
    def test_state_change_cn_changed_mind(self, trigger_cn):
        """测试中文状态变化 - 改变想法"""
        result = trigger_cn.detect("我现在改变想法了，不想要那个方案")
        assert result.triggered
        assert result.trigger_type == TriggerType.STATE_CHANGE
    
    def test_state_change_cn_add_something(self, trigger_cn):
        """测试中文状态变化 - 补充一点"""
        result = trigger_cn.detect("补充一点，我还需要考虑预算问题")
        assert result.triggered
        assert result.trigger_type == TriggerType.STATE_CHANGE
    
    def test_state_change_cn_my_preference(self, trigger_cn):
        """测试中文状态变化 - 我的偏好是"""
        result = trigger_cn.detect("我的偏好是使用 Python 而不是 Java")
        assert result.triggered
        assert result.trigger_type == TriggerType.STATE_CHANGE
    
    def test_state_change_en_changed_mind(self, trigger_en):
        """测试英文状态变化 - changed my mind"""
        result = trigger_en.detect("I've changed my mind about that")
        assert result.triggered
        assert result.trigger_type == TriggerType.STATE_CHANGE
    
    def test_state_change_en_let_me_add(self, trigger_en):
        """测试英文状态变化 - let me add"""
        result = trigger_en.detect("Let me add one more thing")
        assert result.triggered
        assert result.trigger_type == TriggerType.STATE_CHANGE
    
    # ============ 长期价值信号测试 ============
    
    def test_long_term_value_cn_please_remember(self, trigger_cn):
        """测试中文长期价值 - 请记住"""
        result = trigger_cn.detect("请记住我对海鲜过敏")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_long_term_value_cn_i_like(self, trigger_cn):
        """测试中文长期价值 - 我喜欢"""
        result = trigger_cn.detect("我喜欢简洁的代码风格")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_long_term_value_cn_my_plan(self, trigger_cn):
        """测试中文长期价值 - 我的计划是"""
        result = trigger_cn.detect("我的计划是下个月完成这个项目")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_long_term_value_cn_vegetarian(self, trigger_cn):
        """测试中文长期价值 - 素食"""
        result = trigger_cn.detect("我是素食主义者")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_long_term_value_en_please_remember(self, trigger_en):
        """测试英文长期价值 - please remember"""
        result = trigger_en.detect("Please remember I'm allergic to seafood")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_long_term_value_en_i_prefer(self, trigger_en):
        """测试英文长期价值 - I prefer"""
        result = trigger_en.detect("I prefer clean code style")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    # ============ 回顾请求测试 ============
    
    def test_recall_request_cn_what_discussed(self, trigger_cn):
        """测试中文回顾请求 - 我们聊了什么"""
        result = trigger_cn.detect("最近我们聊了什么？")
        assert result.triggered
        assert result.trigger_type == TriggerType.RECALL_REQUEST
    
    def test_recall_request_cn_summarize(self, trigger_cn):
        """测试中文回顾请求 - 总结一下"""
        result = trigger_cn.detect("总结一下我们的对话")
        assert result.triggered
        assert result.trigger_type == TriggerType.RECALL_REQUEST
    
    def test_recall_request_en_what_discussed(self, trigger_en):
        """测试英文回顾请求 - what did we discuss"""
        result = trigger_en.detect("What did we discuss recently?")
        assert result.triggered
        assert result.trigger_type == TriggerType.RECALL_REQUEST
    
    # ============ 观点查询测试 ============
    
    def test_opinion_query_cn_new_thoughts(self, trigger_cn):
        """测试中文观点查询 - 有新看法吗"""
        result = trigger_cn.detect("你对这件事有新的看法吗？")
        assert result.triggered
        assert result.trigger_type == TriggerType.OPINION_QUERY
    
    def test_opinion_query_cn_how_do_you_think(self, trigger_cn):
        """测试中文观点查询 - 你现在怎么看"""
        result = trigger_cn.detect("你现在怎么看这个问题？")
        assert result.triggered
        assert result.trigger_type == TriggerType.OPINION_QUERY
    
    def test_opinion_query_en_new_thoughts(self, trigger_en):
        """测试英文观点查询 - any new thoughts"""
        result = trigger_en.detect("Do you have any new thoughts on this?")
        assert result.triggered
        assert result.trigger_type == TriggerType.OPINION_QUERY
    
    # ============ 无触发测试 ============
    
    def test_no_trigger_normal_question(self, trigger):
        """测试普通问题不触发"""
        result = trigger.detect("今天天气怎么样？")
        assert not result.triggered
        assert result.trigger_type == TriggerType.NONE
    
    def test_no_trigger_empty_message(self, trigger):
        """测试空消息不触发"""
        result = trigger.detect("")
        assert not result.triggered
        assert result.trigger_type == TriggerType.NONE
    
    def test_no_trigger_whitespace(self, trigger):
        """测试空白消息不触发"""
        result = trigger.detect("   ")
        assert not result.triggered
        assert result.trigger_type == TriggerType.NONE
    
    # ============ 自动语言检测测试 ============
    
    def test_auto_language_detection_cn(self, trigger):
        """测试自动语言检测 - 中文"""
        result = trigger.detect("我们刚刚讨论的方案")
        assert result.triggered
        assert result.metadata.get('language') == 'cn'
    
    def test_auto_language_detection_en(self, trigger):
        """测试自动语言检测 - 英文"""
        result = trigger.detect("What we just discussed")
        assert result.triggered
        assert result.metadata.get('language') == 'en'
    
    # ============ 配置更新测试 ============
    
    def test_update_config_add_pattern(self, trigger):
        """测试动态添加模式"""
        # 添加自定义模式
        trigger.add_pattern(
            TriggerType.LONG_TERM_VALUE,
            r"我的生日是",
            "cn"
        )
        
        result = trigger.detect("我的生日是5月1日")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    def test_update_config_via_method(self, trigger):
        """测试通过 update_config 方法更新"""
        trigger.update_config(
            custom_patterns=[
                {
                    "trigger_type": "long_term_value",
                    "pattern": r"我的电话是",
                    "language": "cn"
                }
            ]
        )
        
        result = trigger.detect("我的电话是12345678")
        assert result.triggered
        assert result.trigger_type == TriggerType.LONG_TERM_VALUE
    
    # ============ 统计信息测试 ============
    
    def test_get_stats(self, trigger):
        """测试获取统计信息"""
        stats = trigger.get_stats()
        
        assert 'language' in stats
        assert 'use_classifier' in stats
        assert 'pattern_counts' in stats
        assert TriggerType.META_COGNITIVE.value in stats['pattern_counts']
    
    # ============ 主题提取测试 ============
    
    def test_topic_extraction(self, trigger_cn):
        """测试主题提取"""
        result = trigger_cn.detect("我们刚刚讨论的Python项目怎么样？")
        assert result.triggered
        assert result.extracted_topic is not None
        # 主题应该包含 "Python项目" 相关内容


class TestMemoryTriggerConfig:
    """Memory Trigger 配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MemoryTriggerConfig()
        
        assert len(config.meta_cognitive_patterns_cn) > 0
        assert len(config.meta_cognitive_patterns_en) > 0
        assert config.use_classifier is False
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            'meta_cognitive_patterns_cn': [r"自定义模式"],
            'use_classifier': True,
            'classifier_threshold': 0.8,
        }
        
        config = MemoryTriggerConfig.from_dict(data)
        
        assert config.meta_cognitive_patterns_cn == [r"自定义模式"]
        assert config.use_classifier is True
        assert config.classifier_threshold == 0.8
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = MemoryTriggerConfig()
        data = config.to_dict()
        
        assert 'meta_cognitive_patterns_cn' in data
        assert 'use_classifier' in data
        assert 'classifier_threshold' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
