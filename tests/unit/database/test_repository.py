"""
Unit tests for Repository pattern implementation.

Tests CRUD operations for:
- Session
- Memory
- Conversation
- Experiment
- AuditLog
"""

import pytest
import numpy as np
from datetime import datetime

from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository,
    MemoryRepository,
    ConversationRepository,
    ExperimentRepository,
    AuditLogRepository,
)


class TestSessionRepository:
    """Tests for SessionRepository."""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database."""
        DatabaseManager.reset_instance()
        manager = DatabaseManager(db_path=":memory:")
        yield manager
        DatabaseManager.reset_instance()
    
    def test_create_session(self, db_manager):
        """Test creating a session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            session = repo.create(
                session_id="test_session",
                user_id="user_001",
            )
            
            assert session.id == "test_session"
            assert session.user_id == "user_001"
    
    def test_create_session_auto_id(self, db_manager):
        """Test creating session with auto-generated ID."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            session = repo.create(user_id="user_001")
            
            assert session.id is not None
            assert session.id.startswith("sess_")
    
    def test_get_session(self, db_manager):
        """Test getting session by ID."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            repo.create(session_id="test_session", user_id="user_001")
            
            session = repo.get("test_session")
            
            assert session is not None
            assert session.id == "test_session"
    
    def test_get_nonexistent_session(self, db_manager):
        """Test getting non-existent session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            session = repo.get("nonexistent")
            
            assert session is None
    
    def test_get_or_create_existing(self, db_manager):
        """Test get_or_create with existing session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            repo.create(session_id="test_session", user_id="user_001")
            
            session, created = repo.get_or_create("test_session")
            
            assert session.id == "test_session"
            assert created is False
    
    def test_get_or_create_new(self, db_manager):
        """Test get_or_create with new session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            session, created = repo.get_or_create("new_session")
            
            assert session.id == "new_session"
            assert created is True
    
    def test_list_by_user(self, db_manager):
        """Test listing sessions by user."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            repo.create(session_id="sess_1", user_id="user_001")
            repo.create(session_id="sess_2", user_id="user_001")
            repo.create(session_id="sess_3", user_id="user_002")
            
            sessions = repo.list_by_user("user_001")
            
            assert len(sessions) == 2
    
    def test_update_session(self, db_manager):
        """Test updating session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            repo.create(session_id="test_session", user_id="user_001")
            
            updated = repo.update("test_session", user_id="user_002")
            
            assert updated.user_id == "user_002"
    
    def test_delete_session(self, db_manager):
        """Test soft deleting session."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            
            repo.create(session_id="test_session", user_id="user_001")
            
            result = repo.delete("test_session")
            
            assert result is True
            
            # Session should be inactive
            session = repo.get("test_session")
            assert session.is_active is False


class TestMemoryRepository:
    """Tests for MemoryRepository."""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database."""
        DatabaseManager.reset_instance()
        manager = DatabaseManager(db_path=":memory:")
        yield manager
        DatabaseManager.reset_instance()
    
    @pytest.fixture
    def session_id(self, db_manager):
        """Create a session for testing."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            repo.create(session_id="test_session", user_id="user_001")
        return "test_session"
    
    def test_create_memory(self, db_manager, session_id):
        """Test creating a memory."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            memory = repo.create(
                session_id=session_id,
                content="Test memory content",
            )
            
            assert memory.id is not None
            assert memory.content == "Test memory content"
    
    def test_create_memory_with_embedding(self, db_manager, session_id):
        """Test creating memory with embedding."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            embedding = np.random.randn(384).astype(np.float32)
            
            memory = repo.create(
                session_id=session_id,
                content="Test content",
                embedding=embedding,
            )
            
            assert memory.embedding is not None
    
    def test_create_memory_with_metadata(self, db_manager, session_id):
        """Test creating memory with metadata."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            memory = repo.create(
                session_id=session_id,
                content="Test content",
                metadata={"category": "test", "importance": "high"},
            )
            
            assert memory.metadata['category'] == "test"
    
    def test_get_memory(self, db_manager, session_id):
        """Test getting memory by ID."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            created = repo.create(
                session_id=session_id,
                content="Test content",
            )
            
            memory = repo.get(created.id)
            
            assert memory is not None
            assert memory.content == "Test content"
    
    def test_get_by_session(self, db_manager, session_id):
        """Test getting memories by session."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            repo.create(session_id=session_id, content="Memory 1")
            repo.create(session_id=session_id, content="Memory 2")
            repo.create(session_id=session_id, content="Memory 3")
            
            memories = repo.get_by_session(session_id)
            
            assert len(memories) == 3
    
    def test_get_embedding(self, db_manager, session_id):
        """Test getting memory embedding."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            embedding = np.random.randn(384).astype(np.float32)
            
            memory = repo.create(
                session_id=session_id,
                content="Test content",
                embedding=embedding,
            )
            
            retrieved = repo.get_embedding(memory.id)
            
            assert retrieved is not None
            assert len(retrieved) == 384
    
    def test_update_embedding(self, db_manager, session_id):
        """Test updating memory embedding."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            memory = repo.create(
                session_id=session_id,
                content="Test content",
            )
            
            new_embedding = np.random.randn(384).astype(np.float32)
            
            result = repo.update_embedding(memory.id, new_embedding)
            
            assert result is True
    
    def test_search_by_content(self, db_manager, session_id):
        """Test searching memories by content."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            repo.create(session_id=session_id, content="I like pizza")
            repo.create(session_id=session_id, content="I like pasta")
            repo.create(session_id=session_id, content="The weather is nice")
            
            results = repo.search_by_content(session_id, "like")
            
            assert len(results) == 2
    
    def test_delete_memory(self, db_manager, session_id):
        """Test soft deleting memory."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            memory = repo.create(
                session_id=session_id,
                content="Test content",
            )
            
            result = repo.delete(memory.id)
            
            assert result is True
    
    def test_bulk_create(self, db_manager, session_id):
        """Test bulk creating memories."""
        with db_manager.session_scope() as db:
            repo = MemoryRepository(db)
            
            memories_data = [
                {"content": f"Memory {i}"}
                for i in range(5)
            ]
            
            memories = repo.bulk_create(session_id, memories_data)
            
            assert len(memories) == 5


class TestConversationRepository:
    """Tests for ConversationRepository."""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database."""
        DatabaseManager.reset_instance()
        manager = DatabaseManager(db_path=":memory:")
        yield manager
        DatabaseManager.reset_instance()
    
    @pytest.fixture
    def session_id(self, db_manager):
        """Create a session for testing."""
        with db_manager.session_scope() as db:
            repo = SessionRepository(db)
            repo.create(session_id="test_session", user_id="user_001")
        return "test_session"
    
    def test_create_conversation(self, db_manager, session_id):
        """Test creating conversation entry."""
        with db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            
            conv = repo.create(
                session_id=session_id,
                role="user",
                content="Hello!",
            )
            
            assert conv.id is not None
            assert conv.role == "user"
            assert conv.content == "Hello!"
    
    def test_create_with_injection_info(self, db_manager, session_id):
        """Test creating conversation with injection info."""
        with db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            
            conv = repo.create(
                session_id=session_id,
                role="assistant",
                content="Response",
                injection_mode="dki",
                injection_alpha=0.8,
                memory_ids=["mem_1", "mem_2"],
                latency_ms=150.0,
            )
            
            assert conv.injection_mode == "dki"
            assert conv.injection_alpha == 0.8
    
    def test_get_by_session(self, db_manager, session_id):
        """Test getting conversation history."""
        with db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            
            repo.create(session_id=session_id, role="user", content="Hi")
            repo.create(session_id=session_id, role="assistant", content="Hello!")
            repo.create(session_id=session_id, role="user", content="How are you?")
            
            history = repo.get_by_session(session_id)
            
            assert len(history) == 3
    
    def test_get_recent(self, db_manager, session_id):
        """Test getting recent conversation turns."""
        with db_manager.session_scope() as db:
            repo = ConversationRepository(db)
            
            for i in range(10):
                repo.create(
                    session_id=session_id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                )
            
            recent = repo.get_recent(session_id, n_turns=3)
            
            # n_turns * 2 = 6 messages
            assert len(recent) <= 6


class TestAuditLogRepository:
    """Tests for AuditLogRepository."""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database."""
        DatabaseManager.reset_instance()
        manager = DatabaseManager(db_path=":memory:")
        yield manager
        DatabaseManager.reset_instance()
    
    def test_log_action(self, db_manager):
        """Test logging an action."""
        with db_manager.session_scope() as db:
            repo = AuditLogRepository(db)
            
            log = repo.log(
                action="dki_generate",
                session_id="test_session",
                memory_ids=["mem_1", "mem_2"],
                alpha=0.8,
                mode="dki",
            )
            
            assert log.action == "dki_generate"
            assert log.alpha == 0.8
    
    def test_log_with_metadata(self, db_manager):
        """Test logging with metadata."""
        with db_manager.session_scope() as db:
            repo = AuditLogRepository(db)
            
            log = repo.log(
                action="test_action",
                metadata={"key": "value", "count": 42},
            )
            
            assert log.metadata['key'] == "value"
    
    def test_get_by_session(self, db_manager):
        """Test getting logs by session."""
        with db_manager.session_scope() as db:
            repo = AuditLogRepository(db)
            
            repo.log(action="action_1", session_id="sess_1")
            repo.log(action="action_2", session_id="sess_1")
            repo.log(action="action_3", session_id="sess_2")
            
            logs = repo.get_by_session("sess_1")
            
            assert len(logs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
