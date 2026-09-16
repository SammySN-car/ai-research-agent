"""RAG retrieval — vector search and context formatting."""

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.db.operations import search_chunks
from app.rag.embeddings import get_embedding_engine


def retrieve_context(
        query: str, 
        top_k: int = TOP_K, 
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
    """Embed the query and search for similar document chunks."""
    embedding_engine = get_embedding_engine()
    query_embedding = embedding_engine.embed_text(query)
    return search_chunks(query_embedding, top_k, threshold)


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""

    if not results:
        return "No relevant documents found."

    context_parts = []
    for i,chunk in enumerate(results,1):
        similarity = chunk.get("similarity", 0)
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_parts.append(
            f"[Source {i} — {source}] (similarity: {similarity:.2f})\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def retrieve_and_format(
    query: str, top_k: int = TOP_K
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and return formatted context + raw results."""
    results = retrieve_context(query, top_k=top_k)
    context = format_context(results)
    return context, results
