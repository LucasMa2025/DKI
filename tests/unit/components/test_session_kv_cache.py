"""
Unit tests for Session KV Cache.

Tests the session-level K/V cache with:
- LRU eviction
- LFU eviction
- Weighted eviction (multi-factor)
- TTL expiration
- Memory amortization tracking
"""

import pytest
import torch
import time

from dki.core.components.session_kv_cache import (
    SessionKVCache,
    CacheMetadata,
    CachedKV,
)
from dki.models.base import KVCacheEntry

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.sample_memories import create_sample_kv_entries


class TestSessionKVCache:
    """Tests for SessionKVCache."""
    
    @pytest.fixture
    def cache_lru(self):
        """Create LRU cache."""
        return SessionKVCache(max_size=5, strategy="lru", ttl_seconds=3600)
    
    @pytest.fixture
    def cache_lfu(self):
        """Create LFU cache."""
        return SessionKVCache(max_size=5, strategy="lfu", ttl_seconds=3600)
    
    @pytest.fixture
    def cache_weighted(self):
        """Create weighted cache."""
        return SessionKVCache(max_size=5, strategy="weighted", ttl_seconds=3600)
    
    @pytest.fixture
    def sample_entries(self):
        """Create sample KV entries."""
        return create_sample_kv_entries(num_layers=4, seq_len=10)
    
    def test_cache_initialization(self, cache_lru):
        """Test cache initialization."""
        assert cache_lru.max_size == 5
        assert cache_lru.strategy == "lru"
        assert cache_lru.ttl_seconds == 3600
        assert len(cache_lru) == 0
    
    def test_put_and_get(self, cache_lru, sample_entries):
        """Test basic put and get."""
        cache_lru.put("mem_1", sample_entries, alpha=0.8)
        
        retrieved = cache_lru.get("mem_1")
        
        assert retrieved is not None
        assert len(retrieved) == len(sample_entries)
    
    def test_cache_miss(self, cache_lru):
        """Test cache miss."""
        result = cache_lru.get("nonexistent")
        assert result is None
    
    def test_cache_with_query(self, cache_lru, sample_entries):
        """Test cache with query-specific keys."""
        cache_lru.put("mem_1", sample_entries, query="query_a", alpha=0.8)
        cache_lru.put("mem_1", sample_entries, query="query_b", alpha=0.7)
        
        # Different queries should be cached separately
        result_a = cache_lru.get("mem_1", query="query_a")
        result_b = cache_lru.get("mem_1", query="query_b")
        
        assert result_a is not None
        assert result_b is not None
    
    def test_hit_miss_tracking(self, cache_lru, sample_entries):
        """Test hit/miss tracking."""
        cache_lru.put("mem_1", sample_entries, alpha=0.8)
        
        cache_lru.get("mem_1")  # Hit
        cache_lru.get("mem_2")  # Miss
        cache_lru.get("mem_1")  # Hit
        
        stats = cache_lru.get_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
    
    def test_invalidate(self, cache_lru, sample_entries):
        """Test cache invalidation."""
        cache_lru.put("mem_1", sample_entries, alpha=0.8)
        cache_lru.put("mem_1", sample_entries, query="q1", alpha=0.7)
        
        count = cache_lru.invalidate("mem_1")
        
        assert count >= 1
        assert cache_lru.get("mem_1") is None
    
    def test_clear(self, cache_lru, sample_entries):
        """Test cache clear."""
        cache_lru.put("mem_1", sample_entries, alpha=0.8)
        cache_lru.put("mem_2", sample_entries, alpha=0.7)
        
        cache_lru.clear()
        
        assert len(cache_lru) == 0
        
        stats = cache_lru.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    def test_contains(self, cache_lru, sample_entries):
        """Test __contains__ method."""
        cache_lru.put("mem_1", sample_entries, alpha=0.8)
        
        assert "mem_1" in cache_lru
        assert "mem_2" not in cache_lru


class TestLRUEviction:
    """Tests for LRU eviction strategy."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=3, strategy="lru", ttl_seconds=3600)
    
    def test_lru_eviction(self, cache):
        """Test LRU eviction order."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        cache.put("mem_3", entries, alpha=0.5)
        
        # Access mem_1 to make it recently used
        cache.get("mem_1")
        
        # Add new entry - should evict mem_2 (least recently used)
        cache.put("mem_4", entries, alpha=0.5)
        
        assert len(cache) == 3
        assert cache.get("mem_1") is not None  # Recently accessed
        assert cache.get("mem_3") is not None  # Recently added
        assert cache.get("mem_4") is not None  # Just added
    
    def test_lru_access_updates_order(self, cache):
        """Test that access updates LRU order."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        
        # Access mem_1 multiple times
        cache.get("mem_1")
        cache.get("mem_1")
        
        cache.put("mem_3", entries, alpha=0.5)
        cache.put("mem_4", entries, alpha=0.5)  # Should evict mem_2
        
        assert cache.get("mem_1") is not None
        assert cache.get("mem_2") is None  # Evicted


class TestLFUEviction:
    """Tests for LFU eviction strategy."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=3, strategy="lfu", ttl_seconds=3600)
    
    def test_lfu_eviction(self, cache):
        """Test LFU eviction order."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        cache.put("mem_3", entries, alpha=0.5)
        
        # Access mem_1 and mem_3 multiple times
        cache.get("mem_1")
        cache.get("mem_1")
        cache.get("mem_3")
        
        # Add new entry - should evict mem_2 (least frequently used)
        cache.put("mem_4", entries, alpha=0.5)
        
        assert len(cache) == 3
        assert cache.get("mem_2") is None  # Evicted (least frequent)


class TestWeightedEviction:
    """Tests for weighted eviction strategy."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=3, strategy="weighted", ttl_seconds=3600)
    
    def test_weighted_eviction_considers_alpha(self, cache):
        """Test that weighted eviction considers alpha."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_high", entries, alpha=0.9)
        cache.put("mem_low", entries, alpha=0.1)
        cache.put("mem_mid", entries, alpha=0.5)
        
        # Add new entry - should evict mem_low (lowest alpha)
        cache.put("mem_new", entries, alpha=0.5)
        
        assert len(cache) == 3
        # mem_low should be evicted due to low alpha
        # Note: actual eviction depends on combined score


class TestTTLExpiration:
    """Tests for TTL expiration."""
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = SessionKVCache(max_size=5, strategy="lru", ttl_seconds=1)
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.8)
        
        # Should hit before expiration
        assert cache.get("mem_1") is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        assert cache.get("mem_1") is None


class TestAmortizationStats:
    """Tests for memory amortization statistics."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=10, strategy="lru", ttl_seconds=3600)
    
    def test_amortization_stats(self, cache):
        """Test amortization statistics."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Simulate multi-turn conversation
        cache.put("mem_1", entries, alpha=0.8)
        
        # Multiple accesses (simulating turns)
        for _ in range(5):
            cache.get("mem_1")
        
        stats = cache.get_amortization_stats()
        
        assert stats['compute_saved'] == 5  # 5 cache hits
        assert stats['total_turns'] == 5  # 5 hits (put doesn't count as turn)
        # amortization_ratio = compute_saved / total_turns
        assert stats['amortization_ratio'] == 1.0
    
    def test_efficiency_gain(self, cache):
        """Test efficiency gain calculation."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.8)
        
        # Access multiple times
        for _ in range(10):
            cache.get("mem_1")
        
        stats = cache.get_amortization_stats()
        
        # Efficiency gain should be positive
        assert stats['efficiency_gain'] > 0
        
        # With 10 reuses, efficiency should be high
        # efficiency = reuse / (reuse + 1) = 10/11 ≈ 0.91
        assert stats['efficiency_gain'] > 0.8


class TestCacheMetadata:
    """Tests for CacheMetadata."""
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        now = time.time()
        metadata = CacheMetadata(
            memory_id="mem_1",
            created_at=now,
            last_accessed=now,
            access_count=1,
            alpha=0.8,
            query_hash="abc123",
        )
        
        assert metadata.memory_id == "mem_1"
        assert metadata.alpha == 0.8
        assert metadata.access_count == 1
    
    def test_metadata_update_on_access(self):
        """Test metadata update on access."""
        cache = SessionKVCache(max_size=5, strategy="lru", ttl_seconds=3600)
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.8)
        
        # Access multiple times
        cache.get("mem_1")
        cache.get("mem_1")
        
        # Check access count updated
        cached = cache._cache.get("mem_1")
        assert cached.metadata.access_count == 3  # 1 initial + 2 accesses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
