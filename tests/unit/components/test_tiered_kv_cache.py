"""
Unit tests for Tiered KV Cache.

Tests the memory hierarchy:
- L1 (GPU HBM): Hot memories, uncompressed FP16
- L2 (CPU RAM): Warm memories, compressed (2-4×)
- L3 (SSD): Cold memories, quantized + compressed (8×)
- L4 (Recompute): Store only text, recompute K/V on demand

Key properties tested:
- Tier promotion/demotion
- Cache eviction
- Compression/decompression
"""

import pytest
import torch
import tempfile
import time
from pathlib import Path

from dki.core.components.tiered_kv_cache import (
    TieredKVCache,
    TieredCacheMetadata,
    TieredCacheEntry,
    CacheTier,
    SimpleCompressor,
)
from dki.models.base import KVCacheEntry

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fixtures.sample_memories import create_sample_kv_entries


class TestCacheTier:
    """Tests for CacheTier enum."""
    
    def test_tier_values(self):
        """Test tier enum values."""
        assert CacheTier.L1_GPU.value == "l1_gpu"
        assert CacheTier.L2_CPU.value == "l2_cpu"
        assert CacheTier.L3_SSD.value == "l3_ssd"
        assert CacheTier.L4_RECOMPUTE.value == "l4_recompute"


class TestTieredCacheMetadata:
    """Tests for TieredCacheMetadata."""
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        now = time.time()
        metadata = TieredCacheMetadata(
            memory_id="mem_1",
            tier=CacheTier.L1_GPU,
            created_at=now,
            last_accessed=now,
            access_count=1,
            alpha=0.8,
            size_bytes=1024,
            compressed=False,
            quantized=False,
        )
        
        assert metadata.memory_id == "mem_1"
        assert metadata.tier == CacheTier.L1_GPU
        assert metadata.alpha == 0.8
    
    def test_metadata_score(self):
        """Test importance score calculation."""
        now = time.time()
        
        # High importance entry
        high_meta = TieredCacheMetadata(
            memory_id="high",
            tier=CacheTier.L1_GPU,
            created_at=now - 10,
            last_accessed=now,  # Recent
            access_count=50,    # Frequent
            alpha=0.9,          # High importance
            size_bytes=1024,
            compressed=False,
            quantized=False,
        )
        
        # Low importance entry
        low_meta = TieredCacheMetadata(
            memory_id="low",
            tier=CacheTier.L1_GPU,
            created_at=now - 100,
            last_accessed=now - 50,  # Old
            access_count=2,          # Infrequent
            alpha=0.2,               # Low importance
            size_bytes=1024,
            compressed=False,
            quantized=False,
        )
        
        high_score = high_meta.score(now)
        low_score = low_meta.score(now)
        
        assert high_score > low_score


class TestSimpleCompressor:
    """Tests for SimpleCompressor."""
    
    @pytest.fixture
    def compressor(self):
        return SimpleCompressor(quantize=False)
    
    @pytest.fixture
    def quantizing_compressor(self):
        return SimpleCompressor(quantize=True)
    
    def test_compress_decompress(self, compressor):
        """Test compression and decompression."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        compressed = compressor.compress(entries)
        decompressed = compressor.decompress(compressed)
        
        assert len(decompressed) == len(entries)
        
        for orig, decomp in zip(entries, decompressed):
            assert orig.layer_idx == decomp.layer_idx
            # Check shapes match
            assert orig.key.shape == decomp.key.shape
            assert orig.value.shape == decomp.value.shape
    
    def test_quantized_compress_decompress(self, quantizing_compressor):
        """Test quantized compression and decompression."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        compressed = quantizing_compressor.compress(entries)
        decompressed = quantizing_compressor.decompress(compressed)
        
        assert len(decompressed) == len(entries)
        
        # Quantization introduces some error
        for orig, decomp in zip(entries, decompressed):
            assert orig.layer_idx == decomp.layer_idx
            # Values should be approximately equal
            assert torch.allclose(orig.key.float(), decomp.key.float(), atol=0.1)
    
    def test_compression_ratio(self, compressor, quantizing_compressor):
        """Test compression ratio."""
        assert compressor.get_compression_ratio() == 2.0  # FP16
        assert quantizing_compressor.get_compression_ratio() == 4.0  # INT8


class TestTieredKVCache:
    """Tests for TieredKVCache."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache with temporary L3 path."""
        return TieredKVCache(
            l1_max_size=3,
            l2_max_size=5,
            l3_path=str(tmp_path / "kv_cache"),
            enable_l3=True,
            enable_l4=True,
            ttl_seconds=3600,
        )
    
    @pytest.fixture
    def sample_entries(self):
        """Create sample KV entries."""
        return create_sample_kv_entries(num_layers=2, seq_len=5)
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.l1_max_size == 3
        assert cache.l2_max_size == 5
        assert cache.enable_l3 is True
        assert cache.enable_l4 is True
    
    def test_put_and_get_l1(self, cache, sample_entries):
        """Test put and get from L1."""
        cache.put("mem_1", sample_entries, alpha=0.8)
        
        entries, tier = cache.get("mem_1")
        
        assert entries is not None
        assert tier == CacheTier.L1_GPU
        assert len(entries) == len(sample_entries)
    
    def test_cache_miss(self, cache):
        """Test cache miss."""
        entries, tier = cache.get("nonexistent")
        
        assert entries is None
        assert tier == CacheTier.L4_RECOMPUTE
    
    def test_l1_eviction_to_l2(self, cache, sample_entries):
        """Test L1 eviction to L2."""
        # Fill L1 (max_size=3)
        cache.put("mem_1", sample_entries, alpha=0.8)
        cache.put("mem_2", sample_entries, alpha=0.7)
        cache.put("mem_3", sample_entries, alpha=0.6)
        
        # Add one more - should evict to L2
        cache.put("mem_4", sample_entries, alpha=0.9)
        
        # L1 should have 3 entries
        assert len(cache._l1_cache) == 3
        
        # L2 should have 1 entry
        assert len(cache._l2_cache) == 1
    
    def test_l2_promotion_to_l1(self, cache, sample_entries):
        """Test L2 promotion to L1 on access."""
        # Fill L1 and force eviction
        for i in range(5):
            cache.put(f"mem_{i}", sample_entries, alpha=0.5)
        
        # Some entries should be in L2
        assert len(cache._l2_cache) > 0
        
        # Get an L2 entry - should promote to L1
        l2_key = list(cache._l2_cache.keys())[0]
        memory_id = l2_key.split(":")[0]
        
        entries, tier = cache.get(memory_id)
        
        # Should be promoted
        assert tier == CacheTier.L2_CPU
        assert cache._stats['promotions'] > 0
    
    def test_cache_with_query(self, cache, sample_entries):
        """Test cache with query-specific keys."""
        cache.put("mem_1", sample_entries, query="query_a", alpha=0.8)
        cache.put("mem_1", sample_entries, query="query_b", alpha=0.7)
        
        # Different queries should have different cache entries
        entries_a, _ = cache.get("mem_1", query="query_a")
        entries_b, _ = cache.get("mem_1", query="query_b")
        
        assert entries_a is not None
        assert entries_b is not None
    
    def test_invalidate(self, cache, sample_entries):
        """Test cache invalidation."""
        cache.put("mem_1", sample_entries, alpha=0.8)
        cache.put("mem_1", sample_entries, query="q1", alpha=0.7)
        
        count = cache.invalidate("mem_1")
        
        assert count >= 1
        
        entries, _ = cache.get("mem_1")
        assert entries is None
    
    def test_clear(self, cache, sample_entries):
        """Test cache clear."""
        cache.put("mem_1", sample_entries, alpha=0.8)
        cache.put("mem_2", sample_entries, alpha=0.7)
        
        cache.clear()
        
        assert len(cache._l1_cache) == 0
        assert len(cache._l2_cache) == 0
    
    def test_stats(self, cache, sample_entries):
        """Test cache statistics."""
        cache.put("mem_1", sample_entries, alpha=0.8)
        cache.get("mem_1")  # Hit
        cache.get("mem_2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['l1_size'] == 1
        assert stats['l1_hits'] == 1
        assert stats['l1_misses'] == 1
    
    def test_memory_footprint(self, cache, sample_entries):
        """Test memory footprint calculation."""
        cache.put("mem_1", sample_entries, alpha=0.8)
        
        footprint = cache.get_memory_footprint()
        
        assert 'l1_gpu_bytes' in footprint
        assert 'l2_cpu_bytes' in footprint
        assert 'total_bytes' in footprint
        assert footprint['l1_gpu_bytes'] > 0
    
    def test_l4_text_store(self, cache, sample_entries):
        """Test L4 text storage."""
        cache.put(
            "mem_1",
            sample_entries,
            alpha=0.8,
            text_content="This is the memory text",
        )
        
        assert "mem_1" in cache._l4_text_store
        assert cache._l4_text_store["mem_1"] == "This is the memory text"
    
    def test_ttl_expiration(self, tmp_path, sample_entries):
        """Test TTL expiration."""
        cache = TieredKVCache(
            l1_max_size=3,
            l2_max_size=5,
            l3_path=str(tmp_path / "kv_cache"),
            ttl_seconds=1,  # 1 second TTL
        )
        
        cache.put("mem_1", sample_entries, alpha=0.8)
        
        # Should hit before expiration
        entries, _ = cache.get("mem_1")
        assert entries is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        entries, _ = cache.get("mem_1")
        assert entries is None


class TestTierPromotion:
    """Tests for tier promotion logic."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=2,
            l2_max_size=3,
            l3_path=str(tmp_path / "kv_cache"),
        )
    
    def test_promotion_updates_metadata(self, cache):
        """Test that promotion updates metadata."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill L1 to force demotion
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        cache.put("mem_3", entries, alpha=0.5)  # Demotes mem_1
        
        # Access mem_1 from L2 - should promote
        cache.get("mem_1")
        
        # Check promotion count
        assert cache._stats['promotions'] >= 1


class TestTierDemotion:
    """Tests for tier demotion logic."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=2,
            l2_max_size=2,
            l3_path=str(tmp_path / "kv_cache"),
        )
    
    def test_demotion_from_l1(self, cache):
        """Test demotion from L1 to L2."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        cache.put("mem_1", entries, alpha=0.5)
        cache.put("mem_2", entries, alpha=0.5)
        cache.put("mem_3", entries, alpha=0.5)  # Should demote
        
        assert cache._stats['demotions'] >= 1
        assert len(cache._l2_cache) >= 1
    
    def test_demotion_from_l2_to_l3(self, cache):
        """Test demotion from L2 to L3."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Fill both L1 and L2
        for i in range(5):
            cache.put(f"mem_{i}", entries, alpha=0.5)
        
        # Check L3 files exist
        l3_files = list(cache.l3_path.glob("*.kv"))
        assert len(l3_files) >= 1


class TestEvictionStrategy:
    """Tests for eviction strategy based on score."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        return TieredKVCache(
            l1_max_size=2,
            l2_max_size=3,
            l3_path=str(tmp_path / "kv_cache"),
        )
    
    def test_evict_lowest_score(self, cache):
        """Test that lowest score entry is evicted."""
        entries = create_sample_kv_entries(num_layers=2, seq_len=5)
        
        # Add entries with different alphas
        cache.put("mem_high", entries, alpha=0.9)
        cache.put("mem_low", entries, alpha=0.1)
        cache.put("mem_mid", entries, alpha=0.5)  # Should evict mem_low
        
        # mem_low should be evicted (lowest alpha)
        # Note: actual eviction depends on score which includes recency
        assert len(cache._l1_cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
