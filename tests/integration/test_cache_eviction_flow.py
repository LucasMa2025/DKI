"""
Integration tests for cache eviction flow.

Tests the complete cache lifecycle:
- Cache population
- Eviction triggers
- Tier promotion/demotion
- Cache invalidation
"""

import pytest
import torch
import tempfile
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixtures.sample_memories import create_sample_kv_entries

from dki.core.components.session_kv_cache import SessionKVCache
from dki.core.components.tiered_kv_cache import TieredKVCache, CacheTier


class TestSessionCacheEvictionFlow:
    """Tests for session cache eviction flow."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=5, strategy="weighted", ttl_seconds=3600)
    
    def test_eviction_on_capacity(self, cache):
        """Test eviction when cache reaches capacity."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill cache
        for i in range(5):
            cache.put(f"mem_{i}", entries, alpha=0.5)
        
        assert len(cache) == 5
        
        # Add one more - should trigger eviction
        cache.put("mem_new", entries, alpha=0.5)
        
        assert len(cache) == 5  # Still at capacity
    
    def test_weighted_eviction_prefers_low_alpha(self, cache):
        """Test that weighted eviction prefers low alpha entries."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add entries with different alphas
        cache.put("mem_high", entries, alpha=0.9)
        cache.put("mem_low", entries, alpha=0.1)
        cache.put("mem_mid_1", entries, alpha=0.5)
        cache.put("mem_mid_2", entries, alpha=0.5)
        cache.put("mem_mid_3", entries, alpha=0.5)
        
        # Add one more
        cache.put("mem_new", entries, alpha=0.5)
        
        # High alpha entry should still be present
        assert cache.get("mem_high") is not None
    
    def test_lru_eviction_order(self):
        """Test LRU eviction order."""
        cache = SessionKVCache(max_size=3, strategy="lru", ttl_seconds=3600)
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        cache.put("mem_3", entries, alpha=0.5)
        
        # Access mem_1 to make it recently used
        cache.get("mem_1")
        
        # Add new entry
        cache.put("mem_4", entries, alpha=0.5)
        
        # mem_2 should be evicted (least recently used)
        assert cache.get("mem_1") is not None
        assert cache.get("mem_3") is not None
        assert cache.get("mem_4") is not None


class TestTieredCacheEvictionFlow:
    """Tests for tiered cache eviction flow."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=2,
            l2_max_size=3,
            l3_path=str(tmp_path / "kv_cache"),
            enable_l3=True,
            enable_l4=True,
        )
    
    def test_l1_to_l2_demotion(self, cache):
        """Test demotion from L1 to L2."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill L1
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        
        assert len(cache._l1_cache) == 2
        
        # Add one more - should demote to L2
        cache.put("mem_3", entries, alpha=0.5)
        
        assert len(cache._l1_cache) == 2
        assert len(cache._l2_cache) >= 1
    
    def test_l2_to_l3_demotion(self, cache):
        """Test demotion from L2 to L3."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill L1 and L2
        for i in range(6):
            cache.put(f"mem_{i}", entries, alpha=0.5)
        
        # Check L3 files
        l3_files = list(cache.l3_path.glob("*.kv"))
        assert len(l3_files) >= 1
    
    def test_promotion_on_access(self, cache):
        """Test promotion on cache access."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill cache to force demotion
        for i in range(4):
            cache.put(f"mem_{i}", entries, alpha=0.5)
        
        # Some entries should be in L2
        initial_promotions = cache._stats['promotions']
        
        # Access an L2 entry
        if cache._l2_cache:
            l2_key = list(cache._l2_cache.keys())[0]
            memory_id = l2_key.split(":")[0]
            cache.get(memory_id)
            
            # Should have promoted
            assert cache._stats['promotions'] > initial_promotions


class TestCacheInvalidationFlow:
    """Tests for cache invalidation flow."""
    
    @pytest.fixture
    def session_cache(self):
        return SessionKVCache(max_size=10, strategy="lru", ttl_seconds=3600)
    
    @pytest.fixture
    def tiered_cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=5,
            l2_max_size=10,
            l3_path=str(tmp_path / "kv_cache"),
        )
    
    def test_session_cache_invalidation(self, session_cache):
        """Test session cache invalidation."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add entries with same memory_id but different queries
        session_cache.put("mem_1", entries, query="query_a", alpha=0.5)
        session_cache.put("mem_1", entries, query="query_b", alpha=0.5)
        session_cache.put("mem_2", entries, alpha=0.5)
        
        # Invalidate mem_1
        count = session_cache.invalidate("mem_1")
        
        assert count >= 1
        assert session_cache.get("mem_1", query="query_a") is None
        assert session_cache.get("mem_1", query="query_b") is None
        assert session_cache.get("mem_2") is not None
    
    def test_tiered_cache_invalidation(self, tiered_cache):
        """Test tiered cache invalidation across tiers."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add entries
        tiered_cache.put("mem_1", entries, alpha=0.5)
        tiered_cache.put("mem_2", entries, alpha=0.5)
        
        # Invalidate
        count = tiered_cache.invalidate("mem_1")
        
        assert count >= 1
        
        result, _ = tiered_cache.get("mem_1")
        assert result is None


class TestCacheTTLFlow:
    """Tests for cache TTL expiration flow."""
    
    def test_session_cache_ttl(self):
        """Test session cache TTL expiration."""
        cache = SessionKVCache(max_size=5, strategy="lru", ttl_seconds=1)
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        
        # Should hit before expiration
        assert cache.get("mem_1") is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        assert cache.get("mem_1") is None
    
    def test_tiered_cache_ttl(self, tmp_path):
        """Test tiered cache TTL expiration."""
        cache = TieredKVCache(
            l1_max_size=5,
            l2_max_size=10,
            l3_path=str(tmp_path / "kv_cache"),
            ttl_seconds=1,
        )
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        
        # Should hit before expiration
        result, _ = cache.get("mem_1")
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        result, _ = cache.get("mem_1")
        assert result is None


class TestCacheStatisticsFlow:
    """Tests for cache statistics tracking."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=5, strategy="lru", ttl_seconds=3600)
    
    def test_hit_miss_tracking(self, cache):
        """Test hit/miss statistics tracking."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        
        # Hits
        cache.get("mem_1")
        cache.get("mem_1")
        
        # Misses
        cache.get("mem_2")
        cache.get("mem_3")
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.5
    
    def test_amortization_tracking(self, cache):
        """Test amortization statistics tracking."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        
        # Simulate multi-turn reuse
        for _ in range(10):
            cache.get("mem_1")
        
        stats = cache.get_amortization_stats()
        
        assert stats['compute_saved'] == 10
        assert stats['amortization_ratio'] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
