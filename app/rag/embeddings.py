"""Embedding engine using sentence-transformers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL


class EmbeddingEngine:
    """Generates text embeddings using a local sentence-transformer model."""

    def __init__(self):
        """Initialize the embedding engine."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        self.dim = EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized vector."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed a batch of text strings into normalized vectors."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return embeddings.tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two normalized vectors (dot product)."""
        return float(np.dot(a, b))

# ── Singleton pattern ─────────────────────────────────
_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Return a cached EmbeddingEngine instance (loads model once)."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
