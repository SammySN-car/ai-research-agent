"""Database CRUD operations."""

from app.db.client import get_supabase_client
from typing import Any, cast

def create_document(
    filename: str, file_type: str, file_size: int, chunk_count: int
) -> dict:
    """Insert a new document record and return it."""
    client = get_supabase_client()
    data = {
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "chunk_count": chunk_count,
    }
    response = client.table("documents").insert(data).execute()
    return cast(dict[str, Any], response.data[0])

def create_chunks(document_id: int, chunk_records: list[dict]) -> None:
    """Insert chunk records for a document."""
    client = get_supabase_client()
    rows = [
        {
            "document_id": document_id,
            "chunk_index": record["chunk_index"],
            "content": record["content"],
            "embedding": record["embedding"],
            "metadata": record.get("metadata", {}),
        }
        for record in chunk_records
    ]
    client.table("chunks").insert(rows).execute()

def search_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[dict]:
    """Search for similar chunks using the match_documents RPC function."""
    client = get_supabase_client()
    response = client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "threshold": threshold,
        },
    ).execute()
    return cast(list[dict], response.data)

def list_documents() -> list[dict]:
    """List all documents in the database."""
    client = get_supabase_client()
    response = client.table("documents").select("*").execute()
    return cast(list[dict], response.data)

def delete_document(doc_id: str) -> None:
    """Delete a document and its chunks (cascading)."""
    client = get_supabase_client()
    client.table("documents").delete().eq("id", doc_id).execute()

def save_query(
    question: str, answer: str, sources: list[dict], tools_used: list[str], latency_ms: int,
) -> dict:
    """Log a query and its response."""
    client = get_supabase_client()
    result = client.table("queries").insert({
        "question": question,
        "answer": answer,
        "sources": sources,
        "tools_used": tools_used,
        "latency_ms": latency_ms,
    }).execute()
    return cast(dict[str, Any], result.data[0])