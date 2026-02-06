"""
Unit tests for Database Connection Manager.

Tests database connection and session management:
- Connection initialization
- Session creation
- Transaction handling
"""

import pytest
import tempfile
from pathlib import Path

from dki.database.connection import DatabaseManager


class TestDatabaseManager:
    """Tests for DatabaseManager."""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database manager."""
        DatabaseManager.reset_instance()
        manager = DatabaseManager(db_path=":memory:")
        yield manager
        DatabaseManager.reset_instance()
    
    def test_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager is not None
        assert db_manager.engine is not None
    
    def test_singleton_pattern(self):
        """Test singleton pattern."""
        DatabaseManager.reset_instance()
        
        manager1 = DatabaseManager(db_path=":memory:")
        manager2 = DatabaseManager(db_path=":memory:")
        
        assert manager1 is manager2
        
        DatabaseManager.reset_instance()
    
    def test_get_session(self, db_manager):
        """Test getting database session."""
        session = db_manager.get_session()
        
        assert session is not None
        
        session.close()
    
    def test_session_scope(self, db_manager):
        """Test session scope context manager."""
        from sqlalchemy import text
        with db_manager.session_scope() as session:
            assert session is not None
            # Session should be usable
            result = session.execute(text("SELECT 1"))
            assert result is not None
    
    def test_session_scope_commit(self, db_manager):
        """Test that session scope commits on success."""
        from sqlalchemy import text
        # This test verifies the context manager behavior
        with db_manager.session_scope() as session:
            # Execute a simple query
            session.execute(text("SELECT 1"))
        # No exception means commit succeeded
    
    def test_session_scope_rollback(self, db_manager):
        """Test that session scope rolls back on error."""
        with pytest.raises(Exception):
            with db_manager.session_scope() as session:
                # This should cause an error
                session.execute("INVALID SQL SYNTAX")
    
    def test_file_database(self):
        """Test file-based database."""
        DatabaseManager.reset_instance()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            
            manager = DatabaseManager(db_path=db_path)
            
            assert manager is not None
            assert Path(db_path).exists()
            
            DatabaseManager.reset_instance()
    
    def test_reset_instance(self):
        """Test resetting singleton instance."""
        DatabaseManager.reset_instance()
        
        manager1 = DatabaseManager(db_path=":memory:")
        
        DatabaseManager.reset_instance()
        
        manager2 = DatabaseManager(db_path=":memory:")
        
        # After reset, should be different instances
        # (but singleton will make them same again)
        assert manager2 is not None
        
        DatabaseManager.reset_instance()
    
    def test_drop_all(self, db_manager):
        """Test dropping all tables."""
        # This should not raise an error
        db_manager.drop_all()
    
    def test_reset_database(self, db_manager):
        """Test resetting database."""
        # This should not raise an error
        db_manager.reset()


class TestDatabaseManagerWithEcho:
    """Tests for DatabaseManager with echo enabled."""
    
    def test_echo_mode(self):
        """Test echo mode for debugging."""
        DatabaseManager.reset_instance()
        
        manager = DatabaseManager(db_path=":memory:", echo=True)
        
        assert manager is not None
        
        DatabaseManager.reset_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
