-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- TABLES
-- ============================================================================

-- Users table (placeholder for future auth integration)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Documents table - Knowledge base for context files
-- session_id = NULL: Global knowledge base (persistent)
-- session_id = UUID: Ephemeral upload for specific session (auto-cleanup after 24h)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID DEFAULT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    raw_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    session_id UUID DEFAULT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

-- Form states table
CREATE TABLE IF NOT EXISTS form_states (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_session_id TEXT,
    pdf_filename TEXT,
    pdf_storage_path TEXT,
    original_pdf_storage_path TEXT,
    applied_edits JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours'  -- Auto-expire after 24h
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_session_null ON documents((session_id IS NULL));

-- Document chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON document_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session_null ON document_chunks((session_id IS NULL));

-- Vector similarity search index (HNSW for speed)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Form states indexes
CREATE INDEX IF NOT EXISTS idx_form_states_user_id ON form_states(user_id);
CREATE INDEX IF NOT EXISTS idx_form_states_expires_at ON form_states(expires_at);
CREATE INDEX IF NOT EXISTS idx_form_states_created_at ON form_states(created_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE form_states ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own record"
    ON users FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own record"
    ON users FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Users can view global KB and own documents"
    ON documents FOR SELECT
    USING (
        session_id IS NULL
        OR
        auth.uid() = user_id
    );

CREATE POLICY "Users can insert own documents"
    ON documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own documents"
    ON documents FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own documents"
    ON documents FOR DELETE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can view global KB and own chunks"
    ON document_chunks FOR SELECT
    USING (
        session_id IS NULL
        OR
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = document_chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert own document chunks"
    ON document_chunks FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = document_chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update own document chunks"
    ON document_chunks FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = document_chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete own document chunks"
    ON document_chunks FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM documents
            WHERE documents.id = document_chunks.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can view own form states"
    ON form_states FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own form states"
    ON form_states FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own form states"
    ON form_states FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own form states"
    ON form_states FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Vector similarity search function with session filtering for ephemeral indexing
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5,
    filter_user_id uuid DEFAULT NULL,
    filter_session_id uuid DEFAULT NULL,
    search_global_kb boolean DEFAULT false
)
RETURNS TABLE (
    id uuid,
    document_id uuid,
    chunk_text text,
    chunk_index integer,
    similarity float,
    filename text,
    metadata jsonb,
    session_id uuid,
    source_type text
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_text,
        dc.chunk_index,
        1 - (dc.embedding <=> query_embedding) AS similarity,
        d.filename,
        dc.metadata,
        dc.session_id,
        CASE 
            WHEN dc.session_id IS NULL THEN 'knowledge_base'::text
            ELSE 'ephemeral'::text
        END AS source_type
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE 
        (filter_user_id IS NULL OR d.user_id = filter_user_id)
        AND (
            (filter_session_id IS NOT NULL AND dc.session_id = filter_session_id)
            OR
            (search_global_kb = true AND dc.session_id IS NULL)
        )
        AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Cleanup expired form states function
CREATE OR REPLACE FUNCTION cleanup_expired_form_states()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    DELETE FROM form_states
    WHERE expires_at IS NOT NULL
    AND expires_at < NOW();
END;
$$;

-- Cleanup ephemeral documents older than 24 hours
-- Note: document_chunks will automatically cascade delete via FK constraint
CREATE OR REPLACE FUNCTION cleanup_ephemeral_documents()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    DELETE FROM documents
    WHERE session_id IS NOT NULL
    AND created_at < NOW() - INTERVAL '24 hours';
END;
$$;

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_form_states_updated_at BEFORE UPDATE ON form_states
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- DEFAULT TEST USER (for development)
-- ============================================================================

-- Insert a default test user for development
-- In production, this should be removed and users created via auth
INSERT INTO users (id, email, created_at)
VALUES ('00000000-0000-0000-0000-000000000001', 'test@example.com', NOW())
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- AUTOMATED CLEANUP WITH PG_CRON
-- ============================================================================

-- Enable pg_cron extension (available on Supabase free tier)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule cleanup jobs to run every hour

-- 1. Cleanup ephemeral documents older than 24 hours
SELECT cron.schedule(
    'cleanup-ephemeral-documents',
    '0 * * * *',  -- Every hour at minute 0
    $$SELECT cleanup_ephemeral_documents()$$
);

-- 2. Cleanup expired form states
SELECT cron.schedule(
    'cleanup-expired-form-states',
    '0 * * * *',  -- Every hour at minute 0
    $$SELECT cleanup_expired_form_states()$$
);
