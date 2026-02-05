-- DKI Database Schema
-- SQLite Database Initialization Script

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============ Sessions Table ============
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON string
    is_active INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- ============ Memories Table ============
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    content TEXT NOT NULL,
    embedding BLOB,  -- Serialized numpy array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON string
    is_active INTEGER DEFAULT 1,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

-- ============ Conversations Table ============
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    injection_mode TEXT,  -- 'rag', 'dki', 'none'
    injection_alpha REAL,
    memory_ids TEXT,  -- JSON array of memory IDs
    latency_ms REAL,
    metadata TEXT,  -- JSON string
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);

-- ============ KV Cache Table (for persistent caching) ============
CREATE TABLE IF NOT EXISTS kv_cache (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    layer_idx INTEGER NOT NULL,
    key_cache BLOB,  -- Serialized tensor
    value_cache BLOB,  -- Serialized tensor
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    metadata TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(memory_id, model_name, layer_idx)
);

CREATE INDEX IF NOT EXISTS idx_kv_cache_memory_id ON kv_cache(memory_id);
CREATE INDEX IF NOT EXISTS idx_kv_cache_last_accessed ON kv_cache(last_accessed);

-- ============ Experiments Table ============
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config TEXT NOT NULL,  -- JSON string
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);

-- ============ Experiment Results Table ============
CREATE TABLE IF NOT EXISTS experiment_results (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    mode TEXT NOT NULL,  -- 'rag', 'dki', 'baseline'
    dataset TEXT NOT NULL,
    metrics TEXT NOT NULL,  -- JSON string with all metrics
    sample_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_mode ON experiment_results(mode);

-- ============ Audit Log Table ============
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    action TEXT NOT NULL,
    memory_ids TEXT,  -- JSON array
    alpha REAL,
    mode TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_session_id ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- ============ Model Registry Table ============
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    engine TEXT NOT NULL,  -- vllm, llama, deepseek, glm
    model_name TEXT NOT NULL,
    config TEXT,  -- JSON string
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_registry_engine ON model_registry(engine);

-- ============ Sample Data for Testing ============
-- Insert default session
INSERT OR IGNORE INTO sessions (id, user_id, metadata) 
VALUES ('default', 'test_user', '{"purpose": "testing"}');

-- Insert sample memories
INSERT OR IGNORE INTO memories (id, session_id, content, metadata) 
VALUES 
    ('mem_001', 'default', 'User prefers vegetarian food and is allergic to seafood.', '{"type": "preference"}'),
    ('mem_002', 'default', 'User lives in Beijing and works as a software engineer.', '{"type": "profile"}'),
    ('mem_003', 'default', 'User enjoys hiking and photography on weekends.', '{"type": "hobby"}');

-- ============ Views ============
-- Session summary view
CREATE VIEW IF NOT EXISTS v_session_summary AS
SELECT 
    s.id AS session_id,
    s.user_id,
    s.created_at,
    COUNT(DISTINCT m.id) AS memory_count,
    COUNT(DISTINCT c.id) AS conversation_count,
    MAX(c.created_at) AS last_activity
FROM sessions s
LEFT JOIN memories m ON s.id = m.session_id AND m.is_active = 1
LEFT JOIN conversations c ON s.id = c.session_id
GROUP BY s.id;

-- Experiment summary view
CREATE VIEW IF NOT EXISTS v_experiment_summary AS
SELECT 
    e.id AS experiment_id,
    e.name,
    e.status,
    e.created_at,
    COUNT(r.id) AS result_count,
    GROUP_CONCAT(DISTINCT r.mode) AS modes_tested
FROM experiments e
LEFT JOIN experiment_results r ON e.id = r.experiment_id
GROUP BY e.id;
