"""
Behavior tests for injection isolation.

Tests security and isolation properties:
- Session isolation
- Memory isolation
- Cache isolation
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixtures.sample_memories import create_sample_kv_entries

from dki.core.components.session_kv_cache import SessionKVCache
from dki.core.components.tiered_kv_cache import TieredKVCache


class TestSessionIsolation:
    """Tests for session isolation properties."""
    
    def test_separate_session_caches(self):
        """
        Test that different sessions have separate caches.
        
        Property: Session A cannot see Session B's cache
        """
        cache_a = SessionKVCache(max_size=10, strategy="lru")
        cache_b = SessionKVCache(max_size=10, strategy="lru")
        
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add to session A
        cache_a.put("mem_1", entries, alpha=0.8)
        
        # Session B should not see it
        assert cache_b.get("mem_1") is None
        assert cache_a.get("mem_1") is not None
    
    def test_session_cache_independence(self):
        """
        Test that session caches are independent.
        
        Property: Operations on one cache don't affect another
        """
        cache_a = SessionKVCache(max_size=5, strategy="lru")
        cache_b = SessionKVCache(max_size=5, strategy="lru")
        
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill cache A
        for i in range(5):
            cache_a.put(f"mem_{i}", entries, alpha=0.5)
        
        # Cache B should be empty
        assert len(cache_b) == 0
        
        # Clear cache A
        cache_a.clear()
        
        # Add to cache B
        cache_b.put("mem_0", entries, alpha=0.5)
        
        # Cache B should have entry
        assert cache_b.get("mem_0") is not None


class TestMemoryIsolation:
    """Tests for memory isolation properties."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=10, strategy="lru")
    
    def test_memory_id_isolation(self, cache):
        """
        Test that different memory IDs are isolated.
        
        Property: Memory A's K/V is separate from Memory B's K/V
        """
        entries_a = create_sample_kv_entries(num_layers=2, seq_len=5, seed=1)
        entries_b = create_sample_kv_entries(num_layers=2, seq_len=5, seed=2)
        
        cache.put("mem_a", entries_a, alpha=0.8)
        cache.put("mem_b", entries_b, alpha=0.7)
        
        retrieved_a = cache.get("mem_a")
        retrieved_b = cache.get("mem_b")
        
        # Should be different
        assert not torch.allclose(
            retrieved_a[0].key,
            retrieved_b[0].key,
        )
    
    def test_memory_invalidation_isolation(self, cache):
        """
        Test that invalidating one memory doesn't affect others.
        
        Property: Invalidate(A) doesn't affect B
        """
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_a", entries, alpha=0.8)
        cache.put("mem_b", entries, alpha=0.7)
        
        # Invalidate mem_a
        cache.invalidate("mem_a")
        
        # mem_b should still exist
        assert cache.get("mem_a") is None
        assert cache.get("mem_b") is not None


class TestQueryIsolation:
    """Tests for query-specific cache isolation."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=10, strategy="lru")
    
    def test_query_specific_caching(self, cache):
        """
        Test that different queries have separate cache entries.
        
        Property: Same memory, different queries → Different cache entries
        """
        entries_q1 = create_sample_kv_entries(num_layers=2, seq_len=5, seed=1)
        entries_q2 = create_sample_kv_entries(num_layers=2, seq_len=5, seed=2)
        
        cache.put("mem_1", entries_q1, query="query_a", alpha=0.8)
        cache.put("mem_1", entries_q2, query="query_b", alpha=0.7)
        
        retrieved_a = cache.get("mem_1", query="query_a")
        retrieved_b = cache.get("mem_1", query="query_b")
        
        # Both should exist
        assert retrieved_a is not None
        assert retrieved_b is not None
        
        # Should be different
        assert not torch.allclose(
            retrieved_a[0].key,
            retrieved_b[0].key,
        )
    
    def test_query_invalidation_isolation(self, cache):
        """
        Test that invalidating memory removes all query variants.
        
        Property: Invalidate(mem) removes all query-specific entries
        """
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, query="query_a", alpha=0.8)
        cache.put("mem_1", entries, query="query_b", alpha=0.7)
        cache.put("mem_2", entries, alpha=0.6)
        
        # Invalidate mem_1
        count = cache.invalidate("mem_1")
        
        # Both query variants should be gone
        assert cache.get("mem_1", query="query_a") is None
        assert cache.get("mem_1", query="query_b") is None
        
        # mem_2 should still exist
        assert cache.get("mem_2") is not None


class TestTieredCacheIsolation:
    """Tests for tiered cache isolation."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=3,
            l2_max_size=5,
            l3_path=str(tmp_path / "kv_cache"),
        )
    
    def test_tier_isolation(self, cache):
        """
        Test that entries in different tiers are isolated.
        
        Property: L1 entry is separate from L2 entry
        """
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill to force tier distribution
        for i in range(5):
            cache.put(f"mem_{i}", entries, alpha=0.5)
        
        # Entries should be distributed across tiers
        assert len(cache._l1_cache) > 0
        assert len(cache._l2_cache) > 0
        
        # Each entry should be in only one tier
        l1_keys = set(cache._l1_cache.keys())
        l2_keys = set(cache._l2_cache.keys())
        
        assert l1_keys.isdisjoint(l2_keys)
    
    def test_invalidation_across_tiers(self, cache):
        """
        Test that invalidation works across all tiers.
        
        Property: Invalidate removes from all tiers
        """
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add entry
        cache.put("mem_1", entries, alpha=0.5)
        
        # Invalidate
        count = cache.invalidate("mem_1")
        
        # Should be removed from all tiers
        result, _ = cache.get("mem_1")
        assert result is None


class TestAlphaIsolation:
    """Tests for alpha value isolation."""
    
    @pytest.fixture
    def cache(self):
        return SessionKVCache(max_size=10, strategy="weighted")
    
    def test_alpha_stored_per_entry(self, cache):
        """
        Test that alpha is stored per entry.
        
        Property: Each entry has its own alpha
        """
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.9)
        cache.put("mem_2", entries, alpha=0.3)
        
        # Access both
        cache.get("mem_1")
        cache.get("mem_2")
        
        # Check metadata
        meta_1 = cache._cache["mem_1"].metadata
        meta_2 = cache._cache["mem_2"].metadata
        
        assert meta_1.alpha == 0.9
        assert meta_2.alpha == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
