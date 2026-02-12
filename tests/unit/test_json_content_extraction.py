"""
Unit tests for JSON content extraction in ConfigDrivenAdapter

测试 content_json_key 配置项的功能:
- 从 JSON 字符串中提取实际内容
- 支持嵌套 key
- 支持数组索引
- 解析失败时安全回退到原始内容
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from dki.adapters.config_driven_adapter import (
    ConfigDrivenAdapter,
    ConfigDrivenAdapterConfig,
    TableMapping,
    DatabaseConfig,
)


class TestJsonContentExtraction:
    """测试 JSON 内容提取功能"""
    
    @pytest.fixture
    def adapter(self):
        """创建测试用适配器"""
        config = ConfigDrivenAdapterConfig()
        adapter = ConfigDrivenAdapter(config)
        return adapter
    
    def test_no_json_key_returns_raw_content(self, adapter):
        """没有配置 json_key 时返回原始内容"""
        raw = '{"text": "Hello", "model": "gpt-4"}'
        result = adapter._extract_json_content(raw, None)
        assert result == raw
    
    def test_simple_json_key(self, adapter):
        """简单的 JSON key 提取"""
        raw = '{"text": "Hello World", "model": "gpt-4"}'
        result = adapter._extract_json_content(raw, "text")
        assert result == "Hello World"
    
    def test_nested_json_key(self, adapter):
        """嵌套的 JSON key 提取"""
        raw = '{"data": {"text": "Nested Content", "meta": {}}}'
        result = adapter._extract_json_content(raw, "data.text")
        assert result == "Nested Content"
    
    def test_deeply_nested_json_key(self, adapter):
        """深层嵌套的 JSON key 提取"""
        raw = '{"response": {"data": {"content": {"text": "Deep Content"}}}}'
        result = adapter._extract_json_content(raw, "response.data.content.text")
        assert result == "Deep Content"
    
    def test_array_index_json_key(self, adapter):
        """数组索引的 JSON key 提取"""
        raw = '{"choices": [{"text": "First Choice"}, {"text": "Second Choice"}]}'
        result = adapter._extract_json_content(raw, "choices.0.text")
        assert result == "First Choice"
        
        result = adapter._extract_json_content(raw, "choices.1.text")
        assert result == "Second Choice"
    
    def test_invalid_json_returns_raw(self, adapter):
        """无效 JSON 返回原始内容"""
        raw = "This is not JSON"
        result = adapter._extract_json_content(raw, "text")
        assert result == raw
    
    def test_missing_key_returns_raw(self, adapter):
        """缺失的 key 返回原始内容"""
        raw = '{"other": "value"}'
        result = adapter._extract_json_content(raw, "text")
        assert result == raw
    
    def test_missing_nested_key_returns_raw(self, adapter):
        """缺失的嵌套 key 返回原始内容"""
        raw = '{"data": {"other": "value"}}'
        result = adapter._extract_json_content(raw, "data.text")
        assert result == raw
    
    def test_empty_content_returns_empty(self, adapter):
        """空内容返回空"""
        result = adapter._extract_json_content("", "text")
        assert result == ""
        
        result = adapter._extract_json_content(None, "text")
        assert result is None
    
    def test_non_string_value_converted_to_string(self, adapter):
        """非字符串值转换为字符串"""
        raw = '{"count": 42}'
        result = adapter._extract_json_content(raw, "count")
        assert result == "42"
        
        raw = '{"active": true}'
        result = adapter._extract_json_content(raw, "active")
        assert result == "True"
    
    def test_array_out_of_bounds_returns_raw(self, adapter):
        """数组越界返回原始内容"""
        raw = '{"choices": [{"text": "Only One"}]}'
        result = adapter._extract_json_content(raw, "choices.5.text")
        assert result == raw
    
    def test_real_world_openai_response(self, adapter):
        """真实世界的 OpenAI 响应格式"""
        raw = '''{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "推荐您去川菜馆，那里的麻婆豆腐很正宗。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }'''
        result = adapter._extract_json_content(raw, "choices.0.message.content")
        assert result == "推荐您去川菜馆，那里的麻婆豆腐很正宗。"
    
    def test_real_world_simple_response(self, adapter):
        """简单的响应格式"""
        raw = '{"text": "这是AI的回复", "model": "gpt-4", "tokens": 50}'
        result = adapter._extract_json_content(raw, "text")
        assert result == "这是AI的回复"


class TestTableMappingWithJsonKey:
    """测试 TableMapping 的 content_json_key 配置"""
    
    def test_table_mapping_default_no_json_key(self):
        """默认没有 json_key"""
        mapping = TableMapping(
            table="messages",
            fields={"content": "content"},
        )
        assert mapping.content_json_key is None
    
    def test_table_mapping_with_json_key(self):
        """配置 json_key"""
        mapping = TableMapping(
            table="messages",
            fields={"content": "content"},
            content_json_key="text",
        )
        assert mapping.content_json_key == "text"


class TestConfigFromDict:
    """测试从字典加载配置"""
    
    def test_simplified_format_with_json_key(self):
        """简化格式配置 json_key"""
        data = {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "user",
            "password": "pass",
            "messages_content_json_key": "text",
            "preferences_content_json_key": "content",
        }
        config = ConfigDrivenAdapterConfig.from_dict(data)
        
        assert config.messages.content_json_key == "text"
        assert config.preferences.content_json_key == "content"
    
    def test_full_format_with_json_key(self):
        """完整格式配置 json_key"""
        data = {
            "database": {
                "type": "postgresql",
                "host": "localhost",
            },
            "messages": {
                "table": "chat_messages",
                "fields": {"content": "content"},
                "content_json_key": "choices.0.message.content",
            },
            "preferences": {
                "table": "user_prefs",
                "fields": {"preference_text": "data"},
                "content_json_key": "value",
            },
        }
        config = ConfigDrivenAdapterConfig.from_dict(data)
        
        assert config.messages.content_json_key == "choices.0.message.content"
        assert config.preferences.content_json_key == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
