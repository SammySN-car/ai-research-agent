-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Documents Table ──────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_size   INTEGER,
    chunk_count INTEGER DEFAULT 0,
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Chunks Table (with vector embeddings) ────────────
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(384),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Queries Table (logging) ──────────────────────────
CREATE TABLE IF NOT EXISTS queries (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question    TEXT NOT NULL,
    answer      TEXT,
    sources     JSONB DEFAULT '[]'::jsonb,
    tools_used  JSONB DEFAULT '[]'::jsonb,
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Vector Similarity Search Function ────────────────
-- Called via Supabase RPC: supabase.rpc("match_documents", {...})
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_count     INT   DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id         UUID,
    content    TEXT,
    metadata   JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.content,
        chunks.metadata,
        1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM chunks
    WHERE 1 - (chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ── Index for faster vector search ───────────────────
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);