"""DKI core components."""

from dki.core.components.memory_influence_scaling import MemoryInfluenceScaling
from dki.core.components.query_conditioned_projection import QueryConditionedProjection
from dki.core.components.dual_factor_gating import DualFactorGating
from dki.core.components.session_kv_cache import SessionKVCache
from dki.core.components.position_remapper import PositionRemapper
from dki.core.components.hybrid_injector import (
    HybridDKIInjector,
    HybridInjectionConfig,
    HybridInjectionResult,
    UserPreference,
    SessionHistory,
    SessionMessage,
    create_hybrid_injector,
)
from dki.core.components.memory_trigger import (
    MemoryTrigger,
    MemoryTriggerConfig,
    TriggerType,
    TriggerResult,
    create_memory_trigger,
)
from dki.core.components.reference_resolver import (
    ReferenceResolver,
    ReferenceResolverConfig,
    ReferenceType,
    ReferenceScope,
    ResolvedReference,
    Message,
    create_reference_resolver,
)

__all__ = [
    # Core components
    "MemoryInfluenceScaling",
    "QueryConditionedProjection",
    "DualFactorGating",
    "SessionKVCache",
    "PositionRemapper",
    # Hybrid injector
    "HybridDKIInjector",
    "HybridInjectionConfig",
    "HybridInjectionResult",
    "UserPreference",
    "SessionHistory",
    "SessionMessage",
    "create_hybrid_injector",
    # Memory trigger
    "MemoryTrigger",
    "MemoryTriggerConfig",
    "TriggerType",
    "TriggerResult",
    "create_memory_trigger",
    # Reference resolver
    "ReferenceResolver",
    "ReferenceResolverConfig",
    "ReferenceType",
    "ReferenceScope",
    "ResolvedReference",
    "Message",
    "create_reference_resolver",
]
