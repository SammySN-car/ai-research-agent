"""Component tests for the AI Research Agent."""

import pytest


def test_embedding_dimensions():
    """Verify embedding output has correct dimensions."""
    from app.rag.embeddings import get_embedding_engine

    engine = get_embedding_engine()
    embedding = engine.embed_text("test sentence")
    assert len(embedding) == 384, f"Expected 384, got {len(embedding)}"


def test_embedding_similarity():
    """Verify similar texts produce similar embeddings."""
    from app.rag.embeddings import get_embedding_engine

    engine = get_embedding_engine()
    e1 = engine.embed_text("machine learning algorithms")
    e2 = engine.embed_text("artificial intelligence models")
    e3 = engine.embed_text("chocolate cake recipe")

    sim_related = engine.similarity(e1, e2)
    sim_unrelated = engine.similarity(e1, e3)

    assert sim_related > sim_unrelated, (
        f"Related similarity ({sim_related:.3f}) should be > "
        f"unrelated similarity ({sim_unrelated:.3f})"
    )


def test_text_chunking():
    """Verify text splitting produces correct chunks."""
    from app.rag.ingestion import chunk_text

    text = "Hello world. " * 200  # ~2600 characters
    chunks = chunk_text(text)

    assert len(chunks) > 1, "Should produce multiple chunks"
    for chunk in chunks:
        assert len(chunk) <= 550, f"Chunk too large: {len(chunk)} chars"


def test_calculate_tool():
    """Verify the calculate tool returns correct results."""
    from app.agent.tools import calculate

    assert calculate.invoke({"expression": "2 + 2"}) == "4"
    assert calculate.invoke({"expression": "10 * 5 + 3"}) == "53"
    assert "error" in calculate.invoke({"expression": "invalid"}).lower()
