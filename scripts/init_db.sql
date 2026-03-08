-- DKI Database Schema (SQLite)
-- SQLite Database Initialization Script
-- Updated: 2026-03-08 (P0/P1/P2 优化)

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
    engine TEXT NOT NULL,  -- vllm, llama, deepseek, glm, sglang
    model_name TEXT NOT NULL,
    config TEXT,  -- JSON string
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_registry_engine ON model_registry(engine);

-- ============ Demo Users Table (P0/P1: 增加密码和邮箱索引) ============
CREATE TABLE IF NOT EXISTS demo_users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT,
    email TEXT,
    avatar TEXT,
    password_hash TEXT,  -- SHA-256 hash, NULL = demo mode (任意密码登录)
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login_at DATETIME,
    metadata TEXT DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_demo_users_username ON demo_users(username);
CREATE INDEX IF NOT EXISTS idx_demo_users_email ON demo_users(email);

-- ============ Demo Sessions Table ============
CREATE TABLE IF NOT EXISTS demo_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT 'New Chat',
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES demo_users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_demo_sessions_user_id ON demo_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_demo_sessions_updated_at ON demo_sessions(updated_at);

-- ============ Demo Messages Table ============
CREATE TABLE IF NOT EXISTS demo_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    embedding BLOB,  -- 预留 pgvector 支持 (SQLite 下为 NULL)
    metadata TEXT DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES demo_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_demo_messages_session_id ON demo_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_demo_messages_user_id ON demo_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_demo_messages_created_at ON demo_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_demo_messages_user_created ON demo_messages(user_id, created_at);

-- ============ User Preferences Table ============
CREATE TABLE IF NOT EXISTS user_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_text TEXT NOT NULL,
    preference_type TEXT DEFAULT 'general',  -- general, style, technical, format, domain, other
    priority INTEGER DEFAULT 5,              -- 0-10, higher = more important
    category TEXT,                           -- Optional category for grouping
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- ============ Demo Preferences Table (Demo App 专用) ============
CREATE TABLE IF NOT EXISTS demo_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_text TEXT NOT NULL,
    preference_type TEXT DEFAULT 'general',
    priority INTEGER DEFAULT 5,
    category TEXT,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES demo_users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_demo_preferences_user_id ON demo_preferences(user_id);

-- ============ Function Call Logs Table (v3.2 function call 日志) ============
CREATE TABLE IF NOT EXISTS function_call_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT,
    turn_id TEXT,                       -- 关联的对话轮次 (conversation.id)
    request_id TEXT,                    -- 关联的请求 ID
    round_index INTEGER DEFAULT 0,     -- 在同一轮推理中的 fact call 轮次 (0-based)
    function_name TEXT NOT NULL,        -- 函数名 (如 retrieve_fact)
    arguments TEXT NOT NULL DEFAULT '{}', -- 函数参数 (JSON: trace_id, offset, limit 等)
    response_text TEXT,                 -- 函数返回的文本
    response_tokens INTEGER DEFAULT 0,  -- 返回文本的 token 估算
    status TEXT DEFAULT 'success',      -- success, error, timeout, budget_exceeded
    error_message TEXT,                 -- 错误信息 (如有)
    prompt_before TEXT,                 -- 调用前的完整 prompt (可选, 大文本)
    prompt_after TEXT,                  -- 调用后的完整 prompt (可选, 大文本)
    model_output_before TEXT,           -- 触发 function call 的模型输出
    latency_ms REAL DEFAULT 0,         -- 函数调用耗时 (ms)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'          -- 其他元数据 (JSON)
);

CREATE INDEX IF NOT EXISTS idx_function_call_logs_session_id ON function_call_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_user_id ON function_call_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_request_id ON function_call_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_created_at ON function_call_logs(created_at);

-- ============ Cache Audit Log Table (v3.1 用户隔离) ============
CREATE TABLE IF NOT EXISTS cache_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,           -- get, put, delete, invalidate
    cache_key TEXT,
    cache_tier TEXT,                -- memory, redis, compute
    success INTEGER DEFAULT 1,
    denied_reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cache_audit_logs_user_id ON cache_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_cache_audit_logs_created_at ON cache_audit_logs(created_at);

-- ============ DKI Injection Logs Table (P1-1: 可观测性持久化) ============
CREATE TABLE IF NOT EXISTS dki_injection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    query TEXT,
    injection_strategy TEXT,        -- recall_v4, stable, none_fallback, error_fallback
    injection_enabled INTEGER DEFAULT 0,
    alpha REAL DEFAULT 0.0,
    preference_tokens INTEGER DEFAULT 0,
    history_tokens INTEGER DEFAULT 0,
    query_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0,
    adapter_latency_ms REAL DEFAULT 0,
    injection_latency_ms REAL DEFAULT 0,
    inference_latency_ms REAL DEFAULT 0,
    preference_cache_hit INTEGER DEFAULT 0,
    preference_cache_tier TEXT,
    retrieval_mode TEXT,
    memory_triggered INTEGER DEFAULT 0,
    trigger_type TEXT,
    reference_resolved INTEGER DEFAULT 0,
    reference_type TEXT,
    error_code TEXT,                -- P0-1: 结构化异常错误码
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_request_id ON dki_injection_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_user_id ON dki_injection_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_created_at ON dki_injection_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_strategy ON dki_injection_logs(injection_strategy);

-- ============ Rate Limit Events Table (P1-2: 限流事件记录) ============
CREATE TABLE IF NOT EXISTS rate_limit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,       -- rate_limited, circuit_open, circuit_half_open, circuit_closed
    detail TEXT,                    -- 限流/熔断详情 (JSON)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_events_user_id ON rate_limit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_events_created_at ON rate_limit_events(created_at);

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

-- Insert demo users for experiment system (含密码哈希: 密码均为 "demo123")
INSERT OR IGNORE INTO demo_users (id, username, display_name, email, password_hash, is_active)
VALUES 
    ('user_test', 'test_user', 'Test User', 'test@example.com', 
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', 1),
    ('user_alice', 'alice', 'Alice', 'alice@example.com',
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', 1),
    ('user_bob', 'bob', 'Bob', 'bob@example.com',
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', 1);

-- Insert sample user preferences for experiment system
INSERT OR IGNORE INTO user_preferences (id, user_id, preference_text, preference_type, priority)
VALUES 
    ('pref_001', 'user_test', '我是素食主义者，对海鲜过敏。', 'general', 10),
    ('pref_002', 'user_test', '我住在北京，是一名软件工程师。', 'general', 8),
    ('pref_003', 'user_test', '我周末喜欢徒步和摄影。', 'general', 6),
    ('pref_004', 'user_alice', '请用简洁专业的语言回答问题。', 'style', 10),
    ('pref_005', 'user_alice', '我对机器学习和深度学习特别感兴趣。', 'domain', 8),
    ('pref_006', 'user_bob', '请用通俗易懂的语言解释技术概念。', 'style', 10),
    ('pref_007', 'user_bob', '我是前端开发工程师，主要使用 React。', 'domain', 8);

-- Insert sample demo preferences
INSERT OR IGNORE INTO demo_preferences (id, user_id, preference_text, preference_type, priority)
VALUES 
    ('dpref_001', 'user_test', '我是素食主义者，对海鲜过敏。', 'general', 10),
    ('dpref_002', 'user_test', '我住在北京，是一名软件工程师。', 'general', 8),
    ('dpref_003', 'user_test', '我周末喜欢徒步和摄影。', 'general', 6),
    ('dpref_004', 'user_alice', '请用简洁专业的语言回答问题。', 'style', 10),
    ('dpref_005', 'user_alice', '我对机器学习和深度学习特别感兴趣。', 'domain', 8),
    ('dpref_006', 'user_bob', '请用通俗易懂的语言解释技术概念。', 'style', 10),
    ('dpref_007', 'user_bob', '我是前端开发工程师，主要使用 React。', 'domain', 8);

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

-- DKI Injection statistics view (P1-1)
CREATE VIEW IF NOT EXISTS v_dki_injection_stats AS
SELECT
    DATE(created_at) AS date,
    injection_strategy,
    COUNT(*) AS request_count,
    AVG(latency_ms) AS avg_latency_ms,
    AVG(alpha) AS avg_alpha,
    SUM(CASE WHEN injection_enabled = 1 THEN 1 ELSE 0 END) AS injection_count,
    SUM(CASE WHEN preference_cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hit_count,
    SUM(CASE WHEN error_code IS NOT NULL THEN 1 ELSE 0 END) AS error_count
FROM dki_injection_logs
GROUP BY DATE(created_at), injection_strategy;
