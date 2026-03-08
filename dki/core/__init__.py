"""
Core modules for DKI system.

v4.0: 新增 ConversationRouter (动态路由)
"""

from dki.core.dki_system import DKISystem
from dki.core.rag_system import RAGSystem
from dki.core.memory_router import MemoryRouter
from dki.core.embedding_service import EmbeddingService
from dki.core.plugin_interface import (
    DKIPlugin,
    DKIPluginInterface,
    DKIPluginConfig,
    DKIMiddleware,
)
from dki.core.plugin_manager import DKIPluginManager
from dki.core.conversation_router import ConversationRouter, RouterConfig

__all__ = [
    "DKISystem",
    "RAGSystem",
    "MemoryRouter",
    "EmbeddingService",
    "DKIPlugin",
    "DKIPluginInterface",
    "DKIPluginConfig",
    "DKIMiddleware",
    "DKIPluginManager",
    "ConversationRouter",
    "RouterConfig",
]
