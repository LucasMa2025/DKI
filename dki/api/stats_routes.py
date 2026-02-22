"""
Statistics Routes
FastAPI routes for system statistics

Author: AGI Demo Project
Version: 1.1.0 - Fixed camelCase response for frontend compatibility
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger


# Track startup time
_startup_time = time.time()

# Real stats tracking (updated by record_dki_request)
_dki_stats = {
    "total_requests": 0,
    "l1_hits": 0,
    "l2_hits": 0,
    "l3_computes": 0,
    "avg_alpha": 0.0,
    "injection_rate": 0.0,
}


def create_stats_router() -> APIRouter:
    """Create statistics router."""
    
    router = APIRouter(prefix="/api", tags=["Statistics"])
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
        }
    
    @router.get("/stats")
    async def get_system_stats():
        """
        Get comprehensive system statistics.
        Returns camelCase keys to match frontend TypeScript types.
        """
        uptime = time.time() - _startup_time
        
        # Calculate hit rates
        total = _dki_stats["l1_hits"] + _dki_stats["l2_hits"] + _dki_stats["l3_computes"]
        l1_hit_rate = _dki_stats["l1_hits"] / total if total > 0 else 0
        l2_hit_rate = _dki_stats["l2_hits"] / total if total > 0 else 0
        
        # Return camelCase keys matching frontend SystemStats type
        return {
            "dkiStats": {
                "totalRequests": _dki_stats["total_requests"],
                "l1Hits": _dki_stats["l1_hits"],
                "l2Hits": _dki_stats["l2_hits"],
                "l3Computes": _dki_stats["l3_computes"],
                "avgAlpha": _dki_stats["avg_alpha"],
                "injectionRate": _dki_stats["injection_rate"],
            },
            "cacheStats": {
                "l1Size": total,  # Use actual total as L1 size approximation
                "l1MaxSize": 1000,
                "l1HitRate": l1_hit_rate,
                "l2HitRate": l2_hit_rate,
            },
            "adapterStats": {
                "type": "sqlite",  # Demo adapter type
                "connected": True,
            },
            "uptimeSeconds": uptime,
        }
    
    @router.get("/stats/dki")
    async def get_dki_stats():
        """Get DKI-specific statistics."""
        return {
            "totalRequests": _dki_stats["total_requests"],
            "l1Hits": _dki_stats["l1_hits"],
            "l2Hits": _dki_stats["l2_hits"],
            "l3Computes": _dki_stats["l3_computes"],
            "avgAlpha": _dki_stats["avg_alpha"],
            "injectionRate": _dki_stats["injection_rate"],
        }
    
    @router.get("/stats/cache")
    async def get_cache_stats():
        """Get cache statistics."""
        total = _dki_stats["l1_hits"] + _dki_stats["l2_hits"] + _dki_stats["l3_computes"]
        l1_hit_rate = _dki_stats["l1_hits"] / total if total > 0 else 0
        l2_hit_rate = _dki_stats["l2_hits"] / total if total > 0 else 0
        
        return {
            "l1Size": total,
            "l1MaxSize": 1000,
            "l1HitRate": l1_hit_rate,
            "l2HitRate": l2_hit_rate,
        }
    
    return router


def record_dki_request(
    cache_tier: str = "L3",
    alpha: float = 0.0,
    injected: bool = False,
):
    """
    Record a DKI request for statistics.
    Called from dki_routes.py and app.py chat handler.
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


def get_stats_snapshot() -> dict:
    """Get a snapshot of current stats (for use by other modules)."""
    return dict(_dki_stats)
