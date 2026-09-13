"""Central configuration — all settings in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load environment variables from .env in the project root
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys & LLM Endpoint (OpenRouter) ───────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── Embedding Settings (Local sentence-transformers) ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Chunking Settings ─────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval Settings ─────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ── LLM Settings ──────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))