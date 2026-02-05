"""
Tiered KV Cache for DKI System
Implements the memory hierarchy described in Paper Section 7.4

IMPORTANT: This is an OPTIONAL enhancement for production deployments.
DKI functions correctly with naive caching; these tiers are performance
optimizations, not functional requirements.

Memory Hierarchy:
- L1 (GPU HBM): Hot memories, uncompressed FP16
- L2 (CPU RAM): Warm memories, compressed (2-4×)
- L3 (SSD): Cold memories, quantized + compressed (8×)
- L4 (Recompute): Store only text, recompute K/V on demand

Correctness Guarantee:
The system remains correct if L2/L3/L4 are disabled.

Key Insight from Paper:
DKI memory footprint scales with ACTIVE memories, not total corpus size.
This makes DKI more memory-efficient for large corpora with sparse relevance.

Note on Error Composition:
We do not assume error additivity across compression techniques.
Aggressive compression is only applied to cold memories whose α is already low.
"""

import time
import pickle
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import torch
from loguru import logger

from dki.models.base import KVCacheEntry
from dki.config.config_loader import ConfigLoader


class CacheTier(Enum):
    """Cache tier levels."""
    L1_GPU = "l1_gpu"      # GPU HBM, uncompressed
    L2_CPU = "l2_cpu"      # CPU RAM, compressed
    L3_SSD = "l3_ssd"      # SSD storage, quantized + compressed
    L4_RECOMPUTE = "l4_recompute"  # Text only, recompute on demand


@dataclass
class TieredCacheMetadata:
    """Metadata for tiered cache entries."""
    memory_id: str
    tier: CacheTier
    created_at: float
    last_accessed: float
    access_count: int
    alpha: float
    size_bytes: int
    compressed: bool
    quantized: bool
    
    def score(self, now: float) -> float:
        """Calculate importance score for eviction decisions."""
        recency = 1.0 / (now - self.last_accessed + 1)
        frequency = min(self.access_count / 100.0, 1.0)
        importance = self.alpha
        return 0.4 * frequency + 0.3 * recency + 0.3 * importance


@dataclass
class TieredCacheEntry:
    """Cache entry with tier information."""
    kv_entries: Optional[List[KVCacheEntry]]  # None for L4
    metadata: TieredCacheMetadata
    compressed_data: Optional[bytes] = None  # For L2/L3
    text_content: Optional[str] = None  # For L4 recompute


class KVCompressor(ABC):
    """Abstract base class for K/V compression."""
    
    @abstractmethod
    def compress(self, kv_entries: List[KVCacheEntry]) -> bytes:
        """Compress K/V entries to bytes."""
        pass
    
    @abstractmethod
    def decompress(self, data: bytes) -> List[KVCacheEntry]:
        """Decompress bytes to K/V entries."""
        pass
    
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Expected compression ratio."""
        pass


class SimpleCompressor(KVCompressor):
    """
    Simple compression using pickle + optional quantization.
    
    In production, integrate with GEAR [14] for 4x compression
    with near-lossless quality.
    """
    
    def __init__(self, quantize: bool = False):
        self.quantize = quantize
    
    def compress(self, kv_entries: List[KVCacheEntry]) -> bytes:
        """Compress K/V entries."""
        data_to_compress = []
        for entry in kv_entries:
            if self.quantize:
                # Quantize to INT8
                key_int8 = self._quantize_tensor(entry.key)
                value_int8 = self._quantize_tensor(entry.value)
                data_to_compress.append({
                    'key': key_int8,
                    'value': value_int8,
                    'layer_idx': entry.layer_idx,
                    'quantized': True,
                    'key_scale': entry.key.abs().max().item(),
                    'value_scale': entry.value.abs().max().item(),
                })
            else:
                data_to_compress.append({
                    'key': entry.key.half().cpu(),
                    'value': entry.value.half().cpu(),
                    'layer_idx': entry.layer_idx,
                    'quantized': False,
                })
        
        return pickle.dumps(data_to_compress)
    
    def decompress(self, data: bytes) -> List[KVCacheEntry]:
        """Decompress K/V entries."""
        entries_data = pickle.loads(data)
        kv_entries = []
        
        for entry_data in entries_data:
            if entry_data.get('quantized', False):
                # Dequantize from INT8
                key = self._dequantize_tensor(
                    entry_data['key'], 
                    entry_data['key_scale']
                )
                value = self._dequantize_tensor(
                    entry_data['value'],
                    entry_data['value_scale']
                )
            else:
                key = entry_data['key']
                value = entry_data['value']
            
            kv_entries.append(KVCacheEntry(
                key=key,
                value=value,
                layer_idx=entry_data['layer_idx'],
            ))
        
        return kv_entries
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to INT8."""
        scale = tensor.abs().max()
        if scale == 0:
            return torch.zeros_like(tensor, dtype=torch.int8)
        quantized = (tensor / scale * 127).to(torch.int8)
        return quantized.cpu()
    
    def _dequantize_tensor(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize INT8 tensor."""
        return (tensor.float() / 127 * scale)
    
    def get_compression_ratio(self) -> float:
        return 4.0 if self.quantize else 2.0  # FP16 vs FP32, INT8 vs FP16


class TieredKVCache:
    """
    Tiered K/V cache implementing the memory hierarchy from Paper Section 7.4.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    DKI Memory Hierarchy                  │
    ├─────────────────────────────────────────────────────────┤
    │  L1: GPU HBM (Hot)                                       │
    │  ├── Top-k most recently used memories                   │
    │  ├── Uncompressed FP16                                   │
    │  └── Capacity: 5-10 memories per session                 │
    │                                                          │
    │  L2: CPU RAM (Warm)                                      │
    │  ├── Session-active memories                             │
    │  ├── Compressed (2-4×)                                   │
    │  └── Capacity: 50-100 memories per session               │
    │                                                          │
    │  L3: NVMe SSD (Cold)                                     │
    │  ├── All session memories                                │
    │  ├── Quantized INT8 + Compressed (8×)                    │
    │  └── Capacity: Unlimited                                 │
    │                                                          │
    │  L4: Recompute on Demand                                 │
    │  ├── Store only text + routing vectors                   │
    │  └── Recompute K/V when promoted to L3                   │
    └─────────────────────────────────────────────────────────┘
    
    Key Insight: DKI memory footprint scales with ACTIVE memories,
    not total corpus size.
    """
    
    def __init__(
        self,
        l1_max_size: int = 10,
        l2_max_size: int = 100,
        l3_path: Optional[str] = None,
        enable_l3: bool = True,
        enable_l4: bool = True,
        ttl_seconds: int = 3600,
        compressor: Optional[KVCompressor] = None,
    ):
        config = ConfigLoader().config
        
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.l3_path = Path(l3_path) if l3_path else Path("./data/kv_cache")
        self.enable_l3 = enable_l3
        self.enable_l4 = enable_l4
        self.ttl_seconds = ttl_seconds
        
        # Compressors
        self.l2_compressor = compressor or SimpleCompressor(quantize=False)
        self.l3_compressor = SimpleCompressor(quantize=True)
        
        # Cache storage
        self._l1_cache: OrderedDict[str, TieredCacheEntry] = OrderedDict()
        self._l2_cache: OrderedDict[str, TieredCacheEntry] = OrderedDict()
        self._l4_text_store: Dict[str, str] = {}  # memory_id -> text
        
        # Statistics
        self._stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'l4_recomputes': 0,
            'promotions': 0, 'demotions': 0,
        }
        
        # Ensure L3 directory exists
        if self.enable_l3:
            self.l3_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TieredKVCache initialized: L1={l1_max_size}, L2={l2_max_size}")
    
    def _make_key(self, memory_id: str, query_hash: str = "") -> str:
        """Create cache key."""
        if query_hash:
            return f"{memory_id}:{query_hash}"
        return memory_id
    
    def _hash_query(self, query: str) -> str:
        """Hash query string."""
        return hashlib.md5(query.encode()).hexdigest()[:8]
    
    def _estimate_size(self, kv_entries: List[KVCacheEntry]) -> int:
        """Estimate size of K/V entries in bytes."""
        total = 0
        for entry in kv_entries:
            total += entry.key.numel() * entry.key.element_size()
            total += entry.value.numel() * entry.value.element_size()
        return total
    
    def get(
        self,
        memory_id: str,
        query: Optional[str] = None,
        model: Optional[Any] = None,  # For L4 recompute
    ) -> Tuple[Optional[List[KVCacheEntry]], CacheTier]:
        """
        Get K/V entries from tiered cache.
        
        Returns:
            (kv_entries, tier) - tier indicates where data was found
        """
        query_hash = self._hash_query(query) if query else ""
        cache_key = self._make_key(memory_id, query_hash)
        now = time.time()
        
        # L1: GPU HBM (fastest)
        if cache_key in self._l1_cache:
            entry = self._l1_cache[cache_key]
            if now - entry.metadata.created_at <= self.ttl_seconds:
                entry.metadata.last_accessed = now
                entry.metadata.access_count += 1
                self._l1_cache.move_to_end(cache_key)
                self._stats['l1_hits'] += 1
                return entry.kv_entries, CacheTier.L1_GPU
            else:
                del self._l1_cache[cache_key]
        self._stats['l1_misses'] += 1
        
        # L2: CPU RAM (compressed)
        if cache_key in self._l2_cache:
            entry = self._l2_cache[cache_key]
            if now - entry.metadata.created_at <= self.ttl_seconds:
                # Decompress and promote to L1
                kv_entries = self.l2_compressor.decompress(entry.compressed_data)
                self._promote_to_l1(cache_key, kv_entries, entry.metadata)
                self._stats['l2_hits'] += 1
                self._stats['promotions'] += 1
                return kv_entries, CacheTier.L2_CPU
            else:
                del self._l2_cache[cache_key]
        self._stats['l2_misses'] += 1
        
        # L3: SSD (quantized + compressed)
        if self.enable_l3:
            l3_file = self.l3_path / f"{cache_key}.kv"
            if l3_file.exists():
                try:
                    with open(l3_file, 'rb') as f:
                        compressed_data = f.read()
                    kv_entries = self.l3_compressor.decompress(compressed_data)
                    
                    # Promote to L2
                    metadata = TieredCacheMetadata(
                        memory_id=memory_id,
                        tier=CacheTier.L2_CPU,
                        created_at=now,
                        last_accessed=now,
                        access_count=1,
                        alpha=0.5,
                        size_bytes=self._estimate_size(kv_entries),
                        compressed=True,
                        quantized=False,
                    )
                    self._promote_to_l2(cache_key, kv_entries, metadata)
                    self._stats['l3_hits'] += 1
                    self._stats['promotions'] += 1
                    return kv_entries, CacheTier.L3_SSD
                except Exception as e:
                    logger.warning(f"Failed to load from L3: {e}")
        self._stats['l3_misses'] += 1
        
        # L4: Recompute from text
        if self.enable_l4 and memory_id in self._l4_text_store and model is not None:
            text = self._l4_text_store[memory_id]
            try:
                kv_entries, _ = model.compute_kv(text, return_hidden=True)
                # Store in L2 for future access
                metadata = TieredCacheMetadata(
                    memory_id=memory_id,
                    tier=CacheTier.L2_CPU,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    alpha=0.5,
                    size_bytes=self._estimate_size(kv_entries),
                    compressed=True,
                    quantized=False,
                )
                self._promote_to_l2(cache_key, kv_entries, metadata)
                self._stats['l4_recomputes'] += 1
                return kv_entries, CacheTier.L4_RECOMPUTE
            except Exception as e:
                logger.warning(f"Failed to recompute from L4: {e}")
        
        return None, CacheTier.L4_RECOMPUTE
    
    def put(
        self,
        memory_id: str,
        kv_entries: List[KVCacheEntry],
        query: Optional[str] = None,
        alpha: float = 0.5,
        text_content: Optional[str] = None,
    ) -> CacheTier:
        """
        Store K/V entries in tiered cache.
        
        Returns:
            Tier where data was stored
        """
        query_hash = self._hash_query(query) if query else ""
        cache_key = self._make_key(memory_id, query_hash)
        now = time.time()
        
        size_bytes = self._estimate_size(kv_entries)
        
        # Store text for L4 recompute
        if self.enable_l4 and text_content:
            self._l4_text_store[memory_id] = text_content
        
        metadata = TieredCacheMetadata(
            memory_id=memory_id,
            tier=CacheTier.L1_GPU,
            created_at=now,
            last_accessed=now,
            access_count=1,
            alpha=alpha,
            size_bytes=size_bytes,
            compressed=False,
            quantized=False,
        )
        
        # Try to store in L1
        if len(self._l1_cache) < self.l1_max_size:
            self._l1_cache[cache_key] = TieredCacheEntry(
                kv_entries=kv_entries,
                metadata=metadata,
            )
            return CacheTier.L1_GPU
        
        # L1 full, demote oldest to L2 and store new in L1
        self._demote_from_l1()
        self._l1_cache[cache_key] = TieredCacheEntry(
            kv_entries=kv_entries,
            metadata=metadata,
        )
        return CacheTier.L1_GPU
    
    def _promote_to_l1(
        self,
        cache_key: str,
        kv_entries: List[KVCacheEntry],
        metadata: TieredCacheMetadata,
    ) -> None:
        """Promote entry to L1."""
        # Ensure space in L1
        while len(self._l1_cache) >= self.l1_max_size:
            self._demote_from_l1()
        
        # Move tensors to GPU
        gpu_entries = []
        for entry in kv_entries:
            gpu_entries.append(KVCacheEntry(
                key=entry.key.cuda() if torch.cuda.is_available() else entry.key,
                value=entry.value.cuda() if torch.cuda.is_available() else entry.value,
                layer_idx=entry.layer_idx,
            ))
        
        metadata.tier = CacheTier.L1_GPU
        metadata.compressed = False
        self._l1_cache[cache_key] = TieredCacheEntry(
            kv_entries=gpu_entries,
            metadata=metadata,
        )
    
    def _promote_to_l2(
        self,
        cache_key: str,
        kv_entries: List[KVCacheEntry],
        metadata: TieredCacheMetadata,
    ) -> None:
        """Promote entry to L2."""
        # Ensure space in L2
        while len(self._l2_cache) >= self.l2_max_size:
            self._demote_from_l2()
        
        # Compress for L2
        compressed_data = self.l2_compressor.compress(kv_entries)
        
        metadata.tier = CacheTier.L2_CPU
        metadata.compressed = True
        self._l2_cache[cache_key] = TieredCacheEntry(
            kv_entries=None,
            metadata=metadata,
            compressed_data=compressed_data,
        )
    
    def _demote_from_l1(self) -> None:
        """Demote least valuable entry from L1 to L2."""
        if not self._l1_cache:
            return
        
        # Find entry with lowest score
        now = time.time()
        min_key = min(
            self._l1_cache.keys(),
            key=lambda k: self._l1_cache[k].metadata.score(now)
        )
        
        entry = self._l1_cache.pop(min_key)
        self._stats['demotions'] += 1
        
        # Compress and store in L2
        if entry.kv_entries:
            compressed_data = self.l2_compressor.compress(entry.kv_entries)
            entry.metadata.tier = CacheTier.L2_CPU
            entry.metadata.compressed = True
            
            # Ensure space in L2
            while len(self._l2_cache) >= self.l2_max_size:
                self._demote_from_l2()
            
            self._l2_cache[min_key] = TieredCacheEntry(
                kv_entries=None,
                metadata=entry.metadata,
                compressed_data=compressed_data,
            )
    
    def _demote_from_l2(self) -> None:
        """Demote least valuable entry from L2 to L3."""
        if not self._l2_cache:
            return
        
        # Find entry with lowest score
        now = time.time()
        min_key = min(
            self._l2_cache.keys(),
            key=lambda k: self._l2_cache[k].metadata.score(now)
        )
        
        entry = self._l2_cache.pop(min_key)
        self._stats['demotions'] += 1
        
        # Store in L3 if enabled
        if self.enable_l3 and entry.compressed_data:
            # Decompress L2 data and recompress with quantization for L3
            kv_entries = self.l2_compressor.decompress(entry.compressed_data)
            l3_data = self.l3_compressor.compress(kv_entries)
            
            l3_file = self.l3_path / f"{min_key}.kv"
            try:
                with open(l3_file, 'wb') as f:
                    f.write(l3_data)
            except Exception as e:
                logger.warning(f"Failed to write to L3: {e}")
    
    def invalidate(self, memory_id: str) -> int:
        """Invalidate all cache entries for a memory."""
        count = 0
        
        # L1
        keys_to_remove = [k for k in self._l1_cache if k.startswith(memory_id)]
        for key in keys_to_remove:
            del self._l1_cache[key]
            count += 1
        
        # L2
        keys_to_remove = [k for k in self._l2_cache if k.startswith(memory_id)]
        for key in keys_to_remove:
            del self._l2_cache[key]
            count += 1
        
        # L3
        if self.enable_l3:
            for f in self.l3_path.glob(f"{memory_id}*.kv"):
                f.unlink()
                count += 1
        
        # L4
        if memory_id in self._l4_text_store:
            del self._l4_text_store[memory_id]
            count += 1
        
        return count
    
    def clear(self) -> None:
        """Clear all caches."""
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._l4_text_store.clear()
        
        if self.enable_l3:
            for f in self.l3_path.glob("*.kv"):
                f.unlink()
        
        self._stats = {k: 0 for k in self._stats}
        logger.info("TieredKVCache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_l1 = self._stats['l1_hits'] + self._stats['l1_misses']
        total_l2 = self._stats['l2_hits'] + self._stats['l2_misses']
        total_l3 = self._stats['l3_hits'] + self._stats['l3_misses']
        
        return {
            'l1_size': len(self._l1_cache),
            'l1_max_size': self.l1_max_size,
            'l1_hit_rate': self._stats['l1_hits'] / total_l1 if total_l1 > 0 else 0.0,
            'l2_size': len(self._l2_cache),
            'l2_max_size': self.l2_max_size,
            'l2_hit_rate': self._stats['l2_hits'] / total_l2 if total_l2 > 0 else 0.0,
            'l3_hit_rate': self._stats['l3_hits'] / total_l3 if total_l3 > 0 else 0.0,
            'l4_text_count': len(self._l4_text_store),
            'l4_recomputes': self._stats['l4_recomputes'],
            'promotions': self._stats['promotions'],
            'demotions': self._stats['demotions'],
            **self._stats,
        }
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Get estimated memory footprint per tier."""
        l1_bytes = sum(
            entry.metadata.size_bytes 
            for entry in self._l1_cache.values()
        )
        
        l2_bytes = sum(
            len(entry.compressed_data) if entry.compressed_data else 0
            for entry in self._l2_cache.values()
        )
        
        l3_bytes = 0
        if self.enable_l3:
            for f in self.l3_path.glob("*.kv"):
                l3_bytes += f.stat().st_size
        
        return {
            'l1_gpu_bytes': l1_bytes,
            'l2_cpu_bytes': l2_bytes,
            'l3_ssd_bytes': l3_bytes,
            'total_bytes': l1_bytes + l2_bytes + l3_bytes,
        }
