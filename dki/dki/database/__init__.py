"""Database module for DKI system."""

from dki.database.models import (
    Base,
    Session,
    Memory,
    Conversation,
    KVCache,
    Experiment,
    ExperimentResult,
    AuditLog,
    ModelRegistry,
    DemoUser,
    UserPreference,
)
from dki.database.repository import (
    SessionRepository,
    MemoryRepository,
    ConversationRepository,
    ExperimentRepository,
    AuditLogRepository,
    DemoUserRepository,
    UserPreferenceRepository,
)
from dki.database.connection import DatabaseManager

__all__ = [
    "Base",
    "Session", 
    "Memory",
    "Conversation",
    "KVCache",
    "Experiment",
    "ExperimentResult",
    "AuditLog",
    "ModelRegistry",
    "DemoUser",
    "UserPreference",
    "SessionRepository",
    "MemoryRepository",
    "ConversationRepository",
    "ExperimentRepository",
    "AuditLogRepository",
    "DemoUserRepository",
    "UserPreferenceRepository",
    "DatabaseManager",
]
