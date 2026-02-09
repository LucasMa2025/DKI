"""
Statistics Routes
FastAPI routes for system statistics

Author: AGI Demo Project
Version: 1.0.0
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from loguru import logger


# Track startup time
_startup_time = time.time()

# Mock stats for demo
_dki_stats = {
    "total_requests": 0,
    "l1_hits": 0,
    "l2_hits": 0,
    "l3_computes": 0,
    "avg_alpha": 0.0,
    "injection_rate": 0.0,
}


class DKIStatsResponse(BaseModel):
    total_requests: int
    l1_hits: int
    l2_hits: int
    l3_computes: int
    avg_alpha: float
    injection_rate: float


class CacheStatsResponse(BaseModel):
    l1_size: int
    l1_max_size: int
    l1_hit_rate: float
    l2_hit_rate: float


class AdapterStatsResponse(BaseModel):
    type: str
    connected: bool


class SystemStatsResponse(BaseModel):
    dki_stats: DKIStatsResponse
    cache_stats: CacheStatsResponse
    adapter_stats: AdapterStatsResponse
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str


def create_stats_router() -> APIRouter:
    """Create statistics router."""
    
    router = APIRouter(prefix="/api", tags=["Statistics"])
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.
        """
        return HealthResponse(
            status="healthy",
            version="1.0.0",
        )
    
    @router.get("/stats", response_model=SystemStatsResponse)
    async def get_system_stats():
        """
        Get comprehensive system statistics.
        """
        uptime = time.time() - _startup_time
        
        # Calculate hit rates
        total = _dki_stats["l1_hits"] + _dki_stats["l2_hits"] + _dki_stats["l3_computes"]
        l1_hit_rate = _dki_stats["l1_hits"] / total if total > 0 else 0
        l2_hit_rate = _dki_stats["l2_hits"] / total if total > 0 else 0
        
        return SystemStatsResponse(
            dki_stats=DKIStatsResponse(
                total_requests=_dki_stats["total_requests"],
                l1_hits=_dki_stats["l1_hits"],
                l2_hits=_dki_stats["l2_hits"],
                l3_computes=_dki_stats["l3_computes"],
                avg_alpha=_dki_stats["avg_alpha"],
                injection_rate=_dki_stats["injection_rate"],
            ),
            cache_stats=CacheStatsResponse(
                l1_size=456,  # Mock data
                l1_max_size=1000,
                l1_hit_rate=l1_hit_rate,
                l2_hit_rate=l2_hit_rate,
            ),
            adapter_stats=AdapterStatsResponse(
                type="memory",  # Demo adapter
                connected=True,
            ),
            uptime_seconds=uptime,
        )
    
    @router.get("/stats/dki", response_model=DKIStatsResponse)
    async def get_dki_stats():
        """
        Get DKI-specific statistics.
        """
        return DKIStatsResponse(
            total_requests=_dki_stats["total_requests"],
            l1_hits=_dki_stats["l1_hits"],
            l2_hits=_dki_stats["l2_hits"],
            l3_computes=_dki_stats["l3_computes"],
            avg_alpha=_dki_stats["avg_alpha"],
            injection_rate=_dki_stats["injection_rate"],
        )
    
    @router.get("/stats/cache", response_model=CacheStatsResponse)
    async def get_cache_stats():
        """
        Get cache statistics.
        """
        total = _dki_stats["l1_hits"] + _dki_stats["l2_hits"] + _dki_stats["l3_computes"]
        l1_hit_rate = _dki_stats["l1_hits"] / total if total > 0 else 0
        l2_hit_rate = _dki_stats["l2_hits"] / total if total > 0 else 0
        
        return CacheStatsResponse(
            l1_size=456,
            l1_max_size=1000,
            l1_hit_rate=l1_hit_rate,
            l2_hit_rate=l2_hit_rate,
        )
    
    return router


def record_dki_request(
    cache_tier: str = "L3",
    alpha: float = 0.0,
    injected: bool = False,
):
    """
    Record a DKI request for statistics.
    """
    _dki_stats["total_requests"] += 1
    
    if cache_tier == "L1":
        _dki_stats["l1_hits"] += 1
    elif cache_tier == "L2":
        _dki_stats["l2_hits"] += 1
    else:
        _dki_stats["l3_computes"] += 1
    
    # Update running average of alpha
    n = _dki_stats["total_requests"]
    _dki_stats["avg_alpha"] = (
        (_dki_stats["avg_alpha"] * (n - 1) + alpha) / n
    )
    
    # Update injection rate
    if injected:
        _dki_stats["injection_rate"] = (
            (_dki_stats["injection_rate"] * (n - 1) + 1) / n
        )
    else:
        _dki_stats["injection_rate"] = (
            (_dki_stats["injection_rate"] * (n - 1)) / n
        )
