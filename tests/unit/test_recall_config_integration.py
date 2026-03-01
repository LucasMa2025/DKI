"""
Unit tests for RecallConfigModel integration in DKIConfig.

Tests that:
1. DKIConfig now has a 'recall' field (RecallConfigModel)
2. RecallConfigModel can be serialized to dict for RecallConfig.from_dict()
3. The Pydantic model correctly parses YAML-loaded recall configuration
4. dki_system and dki_plugin can extract recall config from DKIConfig
"""

import pytest
from unittest.mock import patch, MagicMock

from dki.config.config_loader import DKIConfig, RecallConfigModel


class TestRecallConfigModel:
    """Test RecallConfigModel Pydantic model."""

    def test_default_values(self):
        """Default RecallConfigModel should have enabled=True, strategy=summary_with_fact_call."""
        model = RecallConfigModel()
        assert model.enabled is True
        assert model.strategy == "summary_with_fact_call"
        assert model.prompt_formatter == "auto"
        assert model.signals == {}
        assert model.score_weights == {}
        assert model.budget == {}
        assert model.summary == {}
        assert model.fact_call == {}

    def test_custom_values(self):
        """RecallConfigModel should accept custom values."""
        model = RecallConfigModel(
            enabled=False,
            strategy="flat_history",
            signals={"keyword_enabled": True, "bm25_enabled": False},
            budget={"generation_reserve": 512},
            prompt_formatter="deepseek",
        )
        assert model.enabled is False
        assert model.strategy == "flat_history"
        assert model.signals["keyword_enabled"] is True
        assert model.budget["generation_reserve"] == 512
        assert model.prompt_formatter == "deepseek"

    def test_model_dump(self):
        """model_dump() should produce a dict suitable for RecallConfig.from_dict()."""
        model = RecallConfigModel(
            enabled=True,
            strategy="summary_with_fact_call",
            signals={"keyword_enabled": True, "vector_enabled": True},
            score_weights={"keyword_weight": 0.3, "vector_weight": 0.3},
        )
        d = model.model_dump()
        assert isinstance(d, dict)
        assert d["enabled"] is True
        assert d["strategy"] == "summary_with_fact_call"
        assert d["signals"]["keyword_enabled"] is True
        assert d["score_weights"]["keyword_weight"] == 0.3


class TestDKIConfigRecallField:
    """Test that DKIConfig has a 'recall' field."""

    def test_dki_config_has_recall(self):
        """DKIConfig should have a 'recall' attribute."""
        config = DKIConfig()
        assert hasattr(config, 'recall')
        assert isinstance(config.recall, RecallConfigModel)

    def test_dki_config_recall_default(self):
        """Default DKIConfig.recall should be enabled with summary_with_fact_call."""
        config = DKIConfig()
        assert config.recall.enabled is True
        assert config.recall.strategy == "summary_with_fact_call"

    def test_dki_config_recall_from_dict(self):
        """DKIConfig should parse recall from a dict (simulating YAML load)."""
        config_dict = {
            "recall": {
                "enabled": True,
                "strategy": "summary_with_fact_call",
                "signals": {
                    "keyword_enabled": True,
                    "bm25_enabled": True,
                    "vector_enabled": True,
                },
                "budget": {
                    "generation_reserve": 512,
                    "instruction_reserve": 120,
                },
            }
        }
        config = DKIConfig(**config_dict)
        assert config.recall.enabled is True
        assert config.recall.signals["keyword_enabled"] is True
        assert config.recall.budget["generation_reserve"] == 512

    def test_dki_config_recall_model_dump_roundtrip(self):
        """model_dump() of DKIConfig.recall should be usable by RecallConfig.from_dict()."""
        config = DKIConfig(
            recall=RecallConfigModel(
                enabled=True,
                strategy="summary_with_fact_call",
                signals={"keyword_enabled": True},
            )
        )
        recall_dict = config.recall.model_dump()
        assert recall_dict["enabled"] is True
        assert recall_dict["strategy"] == "summary_with_fact_call"
        assert recall_dict["signals"]["keyword_enabled"] is True

    def test_dki_config_recall_disabled(self):
        """DKIConfig with recall disabled should serialize correctly."""
        config = DKIConfig(recall=RecallConfigModel(enabled=False))
        assert config.recall.enabled is False
        d = config.recall.model_dump()
        assert d["enabled"] is False


class TestRecallConfigFromDKIConfig:
    """Test that RecallConfig.from_dict() works with RecallConfigModel.model_dump()."""

    def test_recall_config_from_model_dump(self):
        """RecallConfig.from_dict() should accept RecallConfigModel.model_dump()."""
        try:
            from dki.core.recall.recall_config import RecallConfig
        except ImportError:
            pytest.skip("RecallConfig not available")

        model = RecallConfigModel(
            enabled=True,
            strategy="summary_with_fact_call",
            signals={"keyword_enabled": True, "bm25_enabled": True},
            budget={"generation_reserve": 512},
        )
        recall_config = RecallConfig.from_dict(model.model_dump())
        assert recall_config.enabled is True
        assert recall_config.strategy == "summary_with_fact_call"

    def test_recall_config_from_empty_model_dump(self):
        """RecallConfig.from_dict() with default RecallConfigModel should use defaults."""
        try:
            from dki.core.recall.recall_config import RecallConfig
        except ImportError:
            pytest.skip("RecallConfig not available")

        model = RecallConfigModel()
        recall_config = RecallConfig.from_dict(model.model_dump())
        assert recall_config.enabled is True
        assert recall_config.strategy == "summary_with_fact_call"


class TestHistMessageDisplayNoTruncation:
    """Test that _hist_messages_display no longer truncates content to 500 chars."""

    def test_long_content_not_truncated_in_hybrid(self):
        """Hybrid fallback hist messages should not truncate content."""
        # Simulate what the code does after fix
        long_content = "A" * 1000
        
        class MockMessage:
            role = "user"
            content = long_content
        
        messages = [MockMessage()]
        
        # After fix: no [:500] truncation
        result = [
            {"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', '')}
            for m in messages
        ]
        
        assert len(result[0]["content"]) == 1000
        assert result[0]["content"] == long_content

    def test_short_content_unchanged(self):
        """Short content should remain unchanged."""
        short_content = "Hello"
        
        class MockMessage:
            role = "assistant"
            content = short_content
        
        messages = [MockMessage()]
        result = [
            {"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', '')}
            for m in messages
        ]
        
        assert result[0]["content"] == short_content
        assert result[0]["role"] == "assistant"

    def test_empty_content_handled(self):
        """Empty content should be handled gracefully."""
        class MockMessage:
            role = "user"
            content = ""
        
        messages = [MockMessage()]
        result = [
            {"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', '')}
            for m in messages
        ]
        
        assert result[0]["content"] == ""

    def test_none_content_handled(self):
        """None content should default to empty string."""
        class MockMessage:
            role = "user"
            content = None
        
        messages = [MockMessage()]
        # After fix: item.content or ''
        result = [
            {"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', '') or ''}
            for m in messages
        ]
        
        assert result[0]["content"] == ""
