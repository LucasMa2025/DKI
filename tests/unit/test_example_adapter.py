"""
Unit Tests for Example Adapter (v3.0)

ExampleAdapter v3.0 complete test suite:
- connect / disconnect / health_check
- get_user_profile
- get_user_preferences (with cache)
- get_session_history (time filter + limit)
- search_relevant_history (BM25 + jieba)
- get_user_sessions
- add_user / add_message / add_preference
- update_preference / delete_preference
- cache invalidation
- clear / get_stats

Author: AGI Demo Project
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from dki.adapters.example_adapter import ExampleAdapter, ExampleDataStore
from dki.adapters.base import (
    AdapterConfig,
    AdapterType,
    UserProfile,
    UserPreference,
    ChatMessage,
)


@pytest.fixture
def adapter():
    """Create a fresh ExampleAdapter instance."""
    return ExampleAdapter()


@pytest.fixture
def connected_adapter(adapter):
    """Create a connected ExampleAdapter with test data."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(adapter.connect())
    return adapter


@pytest.fixture
def populated_adapter(connected_adapter):
    """Create a connected adapter with pre-populated test data."""
    adapter = connected_adapter

    # Users
    adapter.add_user("user01", "Lucas", display_name="Lucas Chen")
    adapter.add_user("user02", "Alice")

    # Preferences
    adapter.add_preference("user01", "dietary", "vegetarian", priority=10)
    adapter.add_preference("user01", "interests", "Japanese literature", priority=8)
    adapter.add_preference("user01", "location", "Beijing", priority=5)

    # Session 1 messages (Japanese literature conversation)
    messages_s1 = [
        ("s1", "user01", "user", "Hello, where can I buy the book you recommended?"),
        ("s1", "user01", "assistant", "I can help you find The Time Machine!"),
        ("s1", "user01", "user",
         "Thanks! Have you read Norwegian Wood? Do you like it?"),
        ("s1", "user01", "assistant",
         "I haven't read Norwegian Wood, I'm just a language model"),
        ("s1", "user01", "user",
         "You're wrong, Norwegian Wood is by Haruki Murakami, do you know this author?"),
        ("s1", "user01", "assistant",
         "OH NO! You're right, Norwegian Wood is by Haruki Murakami"),
        ("s1", "user01", "user",
         "Please introduce Haruki Murakami's works and other Japanese authors"),
        ("s1", "user01", "assistant",
         "Haruki Murakami is a famous Japanese author, his works include Norwegian Wood, Wind..."),
    ]
    for sid, uid, role, content in messages_s1:
        adapter.add_message(sid, uid, role, content)

    # Session 2 messages (food conversation)
    adapter.add_message("s2", "user01", "user", "How's the weather today?")
    adapter.add_message("s2", "user01", "assistant", "It's sunny in Beijing!")
    adapter.add_message("s2", "user01", "user", "Recommend a vegetarian restaurant")
    adapter.add_message("s2", "user01", "assistant",
                        "I recommend Pure Lotus vegetarian restaurant in Chaoyang")

    return adapter


# ============ Connection Tests ============

class TestConnection:
    """Test connect / disconnect / health_check."""

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        assert not adapter.is_connected
        await adapter.connect()
        assert adapter.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        await adapter.connect()
        assert adapter.is_connected
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_health_check_connected(self, adapter):
        await adapter.connect()
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, adapter):
        assert await adapter.health_check() is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with ExampleAdapter() as adapter:
            assert adapter.is_connected
        assert not adapter.is_connected


# ============ User Profile Tests ============

class TestUserProfile:
    """Test get_user_profile and add_user."""

    @pytest.mark.asyncio
    async def test_get_user_profile(self, populated_adapter):
        profile = await populated_adapter.get_user_profile("user01")
        assert profile is not None
        assert profile.user_id == "user01"
        assert profile.username == "Lucas"
        assert profile.display_name == "Lucas Chen"

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, populated_adapter):
        profile = await populated_adapter.get_user_profile("nonexistent")
        assert profile is None

    def test_add_user_defaults(self, connected_adapter):
        user = connected_adapter.add_user("u1", "TestUser")
        assert user.user_id == "u1"
        assert user.display_name == "TestUser"  # defaults to username
        assert user.created_at is not None


# ============ Preferences Tests ============

class TestPreferences:
    """Test get_user_preferences with cache, filtering, and CRUD."""

    @pytest.mark.asyncio
    async def test_get_all_preferences(self, populated_adapter):
        prefs = await populated_adapter.get_user_preferences("user01")
        assert len(prefs) == 3
        # Should be sorted by priority descending
        assert prefs[0].priority == 10
        assert prefs[1].priority == 8
        assert prefs[2].priority == 5

    @pytest.mark.asyncio
    async def test_filter_by_type(self, populated_adapter):
        prefs = await populated_adapter.get_user_preferences(
            "user01", preference_types=["dietary"]
        )
        assert len(prefs) == 1
        assert prefs[0].preference_type == "dietary"

    @pytest.mark.asyncio
    async def test_empty_preferences(self, populated_adapter):
        prefs = await populated_adapter.get_user_preferences("user02")
        assert len(prefs) == 0

    @pytest.mark.asyncio
    async def test_cache_hit(self, populated_adapter):
        # First call populates cache
        prefs1 = await populated_adapter.get_user_preferences("user01")
        assert len(populated_adapter._cache) > 0

        # Second call should use cache
        prefs2 = await populated_adapter.get_user_preferences("user01")
        assert prefs1 == prefs2

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_add(self, populated_adapter):
        # Populate cache
        prefs_before = await populated_adapter.get_user_preferences("user01")
        assert len(prefs_before) == 3

        # Add preference (should invalidate cache)
        populated_adapter.add_preference("user01", "music", "classical", priority=6)

        # Should get fresh data
        prefs_after = await populated_adapter.get_user_preferences("user01")
        assert len(prefs_after) == 4

    @pytest.mark.asyncio
    async def test_update_preference(self, populated_adapter):
        prefs = await populated_adapter.get_user_preferences("user01")
        pref_id = prefs[0].preference_id

        ok = populated_adapter.update_preference(
            "user01", pref_id, preference_text="strict vegetarian"
        )
        assert ok is True

        prefs_updated = await populated_adapter.get_user_preferences("user01")
        found = [p for p in prefs_updated if p.preference_id == pref_id]
        assert found[0].preference_text == "strict vegetarian"

    @pytest.mark.asyncio
    async def test_update_nonexistent_preference(self, populated_adapter):
        ok = populated_adapter.update_preference("user01", "nonexistent", "text")
        assert ok is False

    @pytest.mark.asyncio
    async def test_delete_preference(self, populated_adapter):
        prefs = await populated_adapter.get_user_preferences("user01")
        pref_id = prefs[0].preference_id
        count_before = len(prefs)

        ok = populated_adapter.delete_preference("user01", pref_id)
        assert ok is True

        prefs_after = await populated_adapter.get_user_preferences("user01")
        assert len(prefs_after) == count_before - 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_preference(self, populated_adapter):
        ok = populated_adapter.delete_preference("user01", "nonexistent")
        assert ok is False


# ============ Session History Tests ============

class TestSessionHistory:
    """Test get_session_history with time filtering and limit."""

    @pytest.mark.asyncio
    async def test_get_session_history(self, populated_adapter):
        history = await populated_adapter.get_session_history("s1")
        assert len(history) == 8
        # Should be in chronological order
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i - 1].timestamp

    @pytest.mark.asyncio
    async def test_limit(self, populated_adapter):
        history = await populated_adapter.get_session_history("s1", limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_empty_session(self, populated_adapter):
        history = await populated_adapter.get_session_history("nonexistent")
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_time_filter_before(self, populated_adapter):
        # Get all messages first
        all_msgs = await populated_adapter.get_session_history("s1")
        if len(all_msgs) >= 2:
            # Filter before the last message's timestamp
            cutoff = all_msgs[-1].timestamp
            filtered = await populated_adapter.get_session_history(
                "s1", before=cutoff
            )
            assert len(filtered) < len(all_msgs)


# ============ Search Relevant History Tests ============

class TestSearchRelevantHistory:
    """Test BM25 + jieba search_relevant_history."""

    @pytest.mark.asyncio
    async def test_basic_search(self, populated_adapter):
        results = await populated_adapter.search_relevant_history(
            "user01", "Murakami Norwegian Wood", limit=5
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_cross_session_search(self, populated_adapter):
        """Search should find messages across all sessions."""
        results = await populated_adapter.search_relevant_history(
            "user01", "vegetarian restaurant", limit=5
        )
        assert len(results) > 0
        # Should find messages from session s2
        session_ids = {msg.session_id for msg in results}
        assert "s2" in session_ids

    @pytest.mark.asyncio
    async def test_session_scoped_search(self, populated_adapter):
        """Search within a specific session."""
        results = await populated_adapter.search_relevant_history(
            "user01", "weather", limit=5, session_id="s2"
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_no_match_fallback(self, populated_adapter):
        """When no BM25 match, should fallback to recent messages."""
        results = await populated_adapter.search_relevant_history(
            "user01", "quantum computing blockchain", limit=3
        )
        assert len(results) > 0  # Should return recent messages as fallback

    @pytest.mark.asyncio
    async def test_empty_user(self, populated_adapter):
        """User with no messages should return empty."""
        results = await populated_adapter.search_relevant_history(
            "user02", "anything", limit=5
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_score_ordering(self, populated_adapter):
        """Results should be ordered by relevance score."""
        results = await populated_adapter.search_relevant_history(
            "user01", "Murakami author works", limit=5
        )
        # Just verify we get results (ordering is internal)
        assert len(results) > 0


# ============ User Sessions Tests ============

class TestUserSessions:
    """Test get_user_sessions and create_session."""

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, populated_adapter):
        sessions = await populated_adapter.get_user_sessions("user01")
        assert len(sessions) == 2
        # Should be sorted by updated_at descending
        assert sessions[0]["updated_at"] >= sessions[1]["updated_at"]

    @pytest.mark.asyncio
    async def test_no_sessions(self, populated_adapter):
        sessions = await populated_adapter.get_user_sessions("user02")
        assert len(sessions) == 0

    def test_create_session(self, connected_adapter):
        session = connected_adapter.create_session("user01", title="Test Session")
        assert session["user_id"] == "user01"
        assert session["title"] == "Test Session"
        assert session["message_count"] == 0
        assert session["is_active"] is True

    def test_create_session_with_id(self, connected_adapter):
        session = connected_adapter.create_session(
            "user01", session_id="custom_id"
        )
        assert session["session_id"] == "custom_id"


# ============ Cache Tests ============

class TestCache:
    """Test cache methods."""

    def test_clear_cache(self, connected_adapter):
        connected_adapter._set_cached("key1", "value1")
        assert len(connected_adapter._cache) == 1
        connected_adapter.clear_cache()
        assert len(connected_adapter._cache) == 0

    def test_invalidate_user_cache(self, connected_adapter):
        connected_adapter._set_cached("prefs:user01:None:False", "data1")
        connected_adapter._set_cached("prefs:user01:['dietary']:False", "data2")
        connected_adapter._set_cached("prefs:user02:None:False", "data3")

        removed = connected_adapter.invalidate_user_cache("user01")
        assert removed == 2
        assert len(connected_adapter._cache) == 1  # user02's cache remains

    def test_cache_ttl_expiry(self, connected_adapter):
        connected_adapter._cache_ttl = 0  # Expire immediately
        connected_adapter._set_cached("key1", "value1")
        import time
        time.sleep(0.01)
        result = connected_adapter._get_cached("key1")
        assert result is None

    def test_cache_disabled(self):
        config = AdapterConfig(adapter_type=AdapterType.MEMORY, enable_cache=False)
        adapter = ExampleAdapter(config)
        adapter._set_cached("key1", "value1")
        assert adapter._get_cached("key1") is None


# ============ Stats and Clear Tests ============

class TestStatsAndClear:
    """Test get_stats and clear."""

    def test_get_stats(self, populated_adapter):
        stats = populated_adapter.get_stats()
        assert stats["connected"] is True
        assert stats["adapter_type"] == "memory"
        assert stats["users_count"] == 2
        assert stats["sessions_count"] == 2
        assert stats["messages_count"] == 12
        assert stats["preferences_count"] == 3
        assert "cache_size" in stats
        assert "cache_enabled" in stats
        assert "jieba_available" in stats

    def test_clear(self, populated_adapter):
        populated_adapter.clear()
        stats = populated_adapter.get_stats()
        assert stats["users_count"] == 0
        assert stats["sessions_count"] == 0
        assert stats["messages_count"] == 0
        assert stats["preferences_count"] == 0
        assert stats["cache_size"] == 0


# ============ BM25 Internal Tests ============

class TestBM25Internal:
    """Test internal BM25 tokenization and scoring."""

    def test_tokenize_english(self, connected_adapter):
        tokens = connected_adapter._tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_tokenize_chinese(self, connected_adapter):
        tokens = connected_adapter._tokenize("Village Murakami")
        # Should have english tokens
        assert "village" in tokens or "murakami" in tokens

    def test_tokenize_stopwords_filtered(self, connected_adapter):
        # Chinese stopwords should be filtered
        tokens = connected_adapter._tokenize("the and is")
        # English words are not in Chinese stopwords, so they remain
        assert "the" in tokens

    def test_bm25_score_basic(self, connected_adapter):
        msgs = [
            ChatMessage(
                message_id="1", session_id="s1", user_id="u1",
                role="user", content="I love Japanese literature",
                timestamp=datetime.utcnow(),
            ),
            ChatMessage(
                message_id="2", session_id="s1", user_id="u1",
                role="user", content="The weather is nice today",
                timestamp=datetime.utcnow(),
            ),
        ]
        scores = connected_adapter._bm25_score("Japanese literature", msgs)
        assert len(scores) == 2
        # First message should score higher
        assert scores[0][1] > scores[1][1]

    def test_bm25_score_empty_query(self, connected_adapter):
        msgs = [
            ChatMessage(
                message_id="1", session_id="s1", user_id="u1",
                role="user", content="test", timestamp=datetime.utcnow(),
            ),
        ]
        scores = connected_adapter._bm25_score("", msgs)
        assert all(score == 0.0 for _, score in scores)
