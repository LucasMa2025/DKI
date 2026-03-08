-- ============================================================================
-- DKI Database Schema (PostgreSQL + pgvector)
-- PostgreSQL Database Initialization Script
-- Updated: 2026-03-08 (P0/P1/P2 优化)
--
-- 使用方法:
--   1. 创建数据库:
--      CREATE DATABASE dkidemo OWNER postgres;
--   2. 安装 pgvector 扩展 (可选, 用于向量检索):
--      CREATE EXTENSION IF NOT EXISTS vector;
--   3. 执行此脚本:
--      psql -U postgres -d dkidemo -f scripts/init_db_postgresql.sql
--
-- 与 SQLite 版本的区别:
--   - 使用 SERIAL/BIGSERIAL 替代 AUTOINCREMENT
--   - 使用 BOOLEAN 替代 INTEGER (0/1)
--   - 使用 TIMESTAMPTZ 替代 DATETIME
--   - 使用 JSONB 替代 TEXT (JSON string)
--   - 使用 vector(dim) 替代 BLOB (embedding)
--   - 支持 pgvector 向量索引 (HNSW/IVFFlat)
--   - 支持 GIN 索引加速 JSONB 查询
-- ============================================================================

-- 安装 pgvector 扩展 (如果可用)
CREATE EXTENSION IF NOT EXISTS vector;

-- 安装 pg_trgm 扩展 (用于模糊搜索)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============ Sessions Table ============
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- ============ Memories Table ============
CREATE TABLE IF NOT EXISTS memories (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) REFERENCES sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),  -- pgvector: 768 维 (all-MiniLM-L6-v2 等)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
-- pgvector HNSW 索引 (cosine 距离)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ============ Conversations Table ============
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    injection_mode VARCHAR(16),
    injection_alpha REAL,
    memory_ids JSONB DEFAULT '[]',
    latency_ms REAL,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);

-- ============ KV Cache Table ============
CREATE TABLE IF NOT EXISTS kv_cache (
    id VARCHAR(64) PRIMARY KEY,
    memory_id VARCHAR(64) NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_name VARCHAR(256) NOT NULL,
    layer_idx INTEGER NOT NULL,
    key_cache BYTEA,
    value_cache BYTEA,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    UNIQUE(memory_id, model_name, layer_idx)
);

CREATE INDEX IF NOT EXISTS idx_kv_cache_memory_id ON kv_cache(memory_id);
CREATE INDEX IF NOT EXISTS idx_kv_cache_last_accessed ON kv_cache(last_accessed);

-- ============ Experiments Table ============
CREATE TABLE IF NOT EXISTS experiments (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    config JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(16) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);

-- ============ Experiment Results Table ============
CREATE TABLE IF NOT EXISTS experiment_results (
    id VARCHAR(64) PRIMARY KEY,
    experiment_id VARCHAR(64) NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    mode VARCHAR(16) NOT NULL,
    dataset VARCHAR(256) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',
    sample_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_mode ON experiment_results(mode);

-- ============ Audit Log Table ============
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    action VARCHAR(64) NOT NULL,
    memory_ids JSONB DEFAULT '[]',
    alpha REAL,
    mode VARCHAR(16),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_session_id ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- ============ Model Registry Table ============
CREATE TABLE IF NOT EXISTS model_registry (
    id VARCHAR(64) PRIMARY KEY,
    engine VARCHAR(32) NOT NULL,
    model_name VARCHAR(512) NOT NULL,
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_model_registry_engine ON model_registry(engine);

-- ============ Demo Users Table (P0/P1: 增加密码和邮箱索引) ============
CREATE TABLE IF NOT EXISTS demo_users (
    id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    display_name VARCHAR(128),
    email VARCHAR(128),
    avatar VARCHAR(256),
    password_hash VARCHAR(128),  -- SHA-256 hash, NULL = demo mode
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_demo_users_username ON demo_users(username);
CREATE INDEX IF NOT EXISTS idx_demo_users_email ON demo_users(email);

-- ============ Demo Sessions Table ============
CREATE TABLE IF NOT EXISTS demo_sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES demo_users(id) ON DELETE CASCADE,
    title VARCHAR(256) DEFAULT 'New Chat',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_demo_sessions_user_id ON demo_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_demo_sessions_updated_at ON demo_sessions(updated_at);

-- ============ Demo Messages Table (支持 pgvector) ============
CREATE TABLE IF NOT EXISTS demo_messages (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES demo_sessions(id) ON DELETE CASCADE,
    user_id VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    embedding_vector vector(768),  -- pgvector: 用于语义检索
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_messages_session_id ON demo_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_demo_messages_user_id ON demo_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_demo_messages_created_at ON demo_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_demo_messages_user_created ON demo_messages(user_id, created_at);
-- pgvector HNSW 索引 (cosine 距离, 用于 BM25+Embedding 混合检索)
CREATE INDEX IF NOT EXISTS idx_demo_messages_embedding ON demo_messages
    USING hnsw (embedding_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- pg_trgm 索引 (用于 BM25-like 全文搜索)
CREATE INDEX IF NOT EXISTS idx_demo_messages_content_trgm ON demo_messages
    USING gin (content gin_trgm_ops);

-- ============ User Preferences Table ============
CREATE TABLE IF NOT EXISTS user_preferences (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    preference_text TEXT NOT NULL,
    preference_type VARCHAR(32) DEFAULT 'general',
    priority INTEGER DEFAULT 5,
    category VARCHAR(64),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- ============ Demo Preferences Table ============
CREATE TABLE IF NOT EXISTS demo_preferences (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES demo_users(id) ON DELETE CASCADE,
    preference_text TEXT NOT NULL,
    preference_type VARCHAR(32) DEFAULT 'general',
    priority INTEGER DEFAULT 5,
    category VARCHAR(64),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_demo_preferences_user_id ON demo_preferences(user_id);

-- ============ Function Call Logs Table ============
CREATE TABLE IF NOT EXISTS function_call_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    turn_id VARCHAR(64),
    request_id VARCHAR(64),
    round_index INTEGER DEFAULT 0,
    function_name VARCHAR(128) NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}',
    response_text TEXT,
    response_tokens INTEGER DEFAULT 0,
    status VARCHAR(32) DEFAULT 'success',
    error_message TEXT,
    prompt_before TEXT,
    prompt_after TEXT,
    model_output_before TEXT,
    latency_ms REAL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_function_call_logs_session_id ON function_call_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_user_id ON function_call_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_request_id ON function_call_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_function_call_logs_created_at ON function_call_logs(created_at);

-- ============ Cache Audit Log Table ============
CREATE TABLE IF NOT EXISTS cache_audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    action VARCHAR(32) NOT NULL,
    cache_key VARCHAR(256),
    cache_tier VARCHAR(32),
    success BOOLEAN DEFAULT TRUE,
    denied_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cache_audit_logs_user_id ON cache_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_cache_audit_logs_created_at ON cache_audit_logs(created_at);

-- ============ DKI Injection Logs Table (P1-1: 可观测性持久化) ============
CREATE TABLE IF NOT EXISTS dki_injection_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    session_id VARCHAR(64),
    query TEXT,
    injection_strategy VARCHAR(64),
    injection_enabled BOOLEAN DEFAULT FALSE,
    alpha REAL DEFAULT 0.0,
    preference_tokens INTEGER DEFAULT 0,
    history_tokens INTEGER DEFAULT 0,
    query_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0,
    adapter_latency_ms REAL DEFAULT 0,
    injection_latency_ms REAL DEFAULT 0,
    inference_latency_ms REAL DEFAULT 0,
    preference_cache_hit BOOLEAN DEFAULT FALSE,
    preference_cache_tier VARCHAR(32),
    retrieval_mode VARCHAR(32),
    memory_triggered BOOLEAN DEFAULT FALSE,
    trigger_type VARCHAR(64),
    reference_resolved BOOLEAN DEFAULT FALSE,
    reference_type VARCHAR(64),
    error_code VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_request_id ON dki_injection_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_user_id ON dki_injection_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_created_at ON dki_injection_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_strategy ON dki_injection_logs(injection_strategy);
-- BRIN 索引: 适合时序数据, 比 B-tree 更节省空间
CREATE INDEX IF NOT EXISTS idx_dki_injection_logs_created_brin ON dki_injection_logs
    USING brin (created_at) WITH (pages_per_range = 128);

-- ============ Rate Limit Events Table (P1-2) ============
CREATE TABLE IF NOT EXISTS rate_limit_events (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    detail JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_events_user_id ON rate_limit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_events_created_at ON rate_limit_events(created_at);

-- ============ Sample Data ============
INSERT INTO sessions (id, user_id, metadata)
VALUES ('default', 'test_user', '{"purpose": "testing"}')
ON CONFLICT (id) DO NOTHING;

INSERT INTO memories (id, session_id, content, metadata)
VALUES 
    ('mem_001', 'default', 'User prefers vegetarian food and is allergic to seafood.', '{"type": "preference"}'),
    ('mem_002', 'default', 'User lives in Beijing and works as a software engineer.', '{"type": "profile"}'),
    ('mem_003', 'default', 'User enjoys hiking and photography on weekends.', '{"type": "hobby"}')
ON CONFLICT (id) DO NOTHING;

-- 密码哈希: SHA-256("demo123") = a665a459...
INSERT INTO demo_users (id, username, display_name, email, password_hash, is_active)
VALUES 
    ('user_test', 'test_user', 'Test User', 'test@example.com',
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', TRUE),
    ('user_alice', 'alice', 'Alice', 'alice@example.com',
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', TRUE),
    ('user_bob', 'bob', 'Bob', 'bob@example.com',
     'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', TRUE)
ON CONFLICT (id) DO NOTHING;

INSERT INTO user_preferences (id, user_id, preference_text, preference_type, priority)
VALUES 
    ('pref_001', 'user_test', '我是素食主义者，对海鲜过敏。', 'general', 10),
    ('pref_002', 'user_test', '我住在北京，是一名软件工程师。', 'general', 8),
    ('pref_003', 'user_test', '我周末喜欢徒步和摄影。', 'general', 6),
    ('pref_004', 'user_alice', '请用简洁专业的语言回答问题。', 'style', 10),
    ('pref_005', 'user_alice', '我对机器学习和深度学习特别感兴趣。', 'domain', 8),
    ('pref_006', 'user_bob', '请用通俗易懂的语言解释技术概念。', 'style', 10),
    ('pref_007', 'user_bob', '我是前端开发工程师，主要使用 React。', 'domain', 8)
ON CONFLICT (id) DO NOTHING;

INSERT INTO demo_preferences (id, user_id, preference_text, preference_type, priority)
VALUES 
    ('dpref_001', 'user_test', '我是素食主义者，对海鲜过敏。', 'general', 10),
    ('dpref_002', 'user_test', '我住在北京，是一名软件工程师。', 'general', 8),
    ('dpref_003', 'user_test', '我周末喜欢徒步和摄影。', 'general', 6),
    ('dpref_004', 'user_alice', '请用简洁专业的语言回答问题。', 'style', 10),
    ('dpref_005', 'user_alice', '我对机器学习和深度学习特别感兴趣。', 'domain', 8),
    ('dpref_006', 'user_bob', '请用通俗易懂的语言解释技术概念。', 'style', 10),
    ('dpref_007', 'user_bob', '我是前端开发工程师，主要使用 React。', 'domain', 8)
ON CONFLICT (id) DO NOTHING;

-- ============ Views ============
CREATE OR REPLACE VIEW v_session_summary AS
SELECT 
    s.id AS session_id,
    s.user_id,
    s.created_at,
    COUNT(DISTINCT m.id) AS memory_count,
    COUNT(DISTINCT c.id) AS conversation_count,
    MAX(c.created_at) AS last_activity
FROM sessions s
LEFT JOIN memories m ON s.id = m.session_id AND m.is_active = TRUE
LEFT JOIN conversations c ON s.id = c.session_id
GROUP BY s.id;

CREATE OR REPLACE VIEW v_experiment_summary AS
SELECT 
    e.id AS experiment_id,
    e.name,
    e.status,
    e.created_at,
    COUNT(r.id) AS result_count,
    STRING_AGG(DISTINCT r.mode, ',') AS modes_tested
FROM experiments e
LEFT JOIN experiment_results r ON e.id = r.experiment_id
GROUP BY e.id;

-- DKI Injection statistics view (P1-1)
CREATE OR REPLACE VIEW v_dki_injection_stats AS
SELECT
    DATE(created_at) AS date,
    injection_strategy,
    COUNT(*) AS request_count,
    AVG(latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
    AVG(alpha) AS avg_alpha,
    COUNT(*) FILTER (WHERE injection_enabled = TRUE) AS injection_count,
    COUNT(*) FILTER (WHERE preference_cache_hit = TRUE) AS cache_hit_count,
    COUNT(*) FILTER (WHERE error_code IS NOT NULL) AS error_count
FROM dki_injection_logs
GROUP BY DATE(created_at), injection_strategy;

-- ============ 自动更新 updated_at 触发器 ============
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为需要 updated_at 的表创建触发器
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN SELECT unnest(ARRAY[
        'sessions', 'memories', 'demo_sessions',
        'user_preferences', 'demo_preferences'
    ]) LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trigger_update_%s_updated_at ON %I; '
            'CREATE TRIGGER trigger_update_%s_updated_at '
            'BEFORE UPDATE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();',
            tbl, tbl, tbl, tbl
        );
    END LOOP;
END;
$$;

-- ============ 分区建议 (大规模部署) ============
-- 对于 dki_injection_logs 和 function_call_logs 等高写入量表,
-- 建议使用 PostgreSQL 分区表按月分区:
--
-- CREATE TABLE dki_injection_logs_partitioned (
--     LIKE dki_injection_logs INCLUDING ALL
-- ) PARTITION BY RANGE (created_at);
--
-- CREATE TABLE dki_injection_logs_2026_03
--     PARTITION OF dki_injection_logs_partitioned
--     FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
--
-- 配合 pg_partman 扩展可自动管理分区创建和过期数据清理。

-- ============ 连接池建议 ============
-- 生产环境建议使用 PgBouncer 作为连接池:
--   pool_mode = transaction
--   max_client_conn = 200
--   default_pool_size = 20
--   reserve_pool_size = 5
--
-- SQLAlchemy 配置:
--   pool_size=20, max_overflow=30, pool_timeout=10,
--   pool_pre_ping=True, pool_recycle=1800
