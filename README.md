# 🔬 AI Research Agent

An autonomous, stateful research assistant built with **LangGraph**, **LangChain**, **Supabase (pgvector)**, and **OpenRouter**.

The agent ingests research documents (PDF, TXT, MD), computes dense vector embeddings locally using `BAAI/bge-small-en-v1.5` (384 dimensions, zero API cost, unmetered), stores and indexes them in PostgreSQL with pgvector, and executes a multi-step LangGraph workflow to retrieve, reason, call tools, and synthesize cited research answers.

---

## ✨ Features

- 🧠 **Autonomous Agent Loop**: Cyclic multi-step reasoning, query decomposition, and tool execution powered by LangGraph.
- ⚡ **Local & Unmetered Embeddings**: `BAAI/bge-small-en-v1.5` generates normalized 384-dimensional dense vectors with automatic CUDA GPU acceleration and CPU fallback.
- 🗄️ **Production Vector Store**: Supabase PostgreSQL with `pgvector`, IVFFlat cosine distance indexing (`vector_cosine_ops`), and optimized stored procedure (`match_documents`).
- 🌐 **Universal LLM Gateway**: OpenRouter integration supporting `openai/gpt-4o-mini`, Claude 3.5 Sonnet, DeepSeek, or any OpenAI-compatible model.
- 📄 **Smart Ingestion Pipeline**: Recursive character chunking with configurable overlap to preserve context across boundaries.
- 🚀 **Modern Python Architecture**: Built with Python 3.11+ using native namespace packages (clean tree with zero redundant `__init__.py` files).
- 🔌 **FastAPI REST Endpoints**: Production-ready API for uploading research papers, asking queries, and managing ingested documents.

---

## 🚀 Architecture

```text
                  ┌─────────────────────────────────┐
                  │    User Request / Question      │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │     FastAPI REST Endpoint       │
                  │   (/api/ask, /api/upload, ...)  │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Agent State                           │
│              { messages, context, tools_used, iteration }              │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │         Retrieve Node         │   │          Agent Node           │
    │  • Local BGE-small embedding  │   │  • OpenRouter LLM Reasoning   │
    │  • Supabase pgvector cosine   │   │  • Decides whether to answer  │
    │    search via match_documents │   │    or call external tools     │
    └───────────────────────────────┘   └───────────────┬───────────────┘
                                                        │
                                                        ▼
                                        ┌───────────────────────────────┐
                                        │          Tool Node            │
                                        │  • search_docs                │
                                        │  • summarize_text             │
                                        │  • calculate                  │
                                        └───────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful multi-actor graph with conditional looping and checkpoints |
| **LLM Abstraction** | [LangChain](https://github.com/langchain-ai/langchain) | LCEL primitives, message schemas, and tool calling interface |
| **LLM Gateway** | [OpenRouter](https://openrouter.ai/) | Single API key access to top open and closed LLMs (`openai/gpt-4o-mini`) |
| **Embedding Engine** | [Sentence Transformers](https://www.sbert.net/) | `BAAI/bge-small-en-v1.5` (384-dim, MTEB rank leader, runs locally & free) |
| **Vector Database** | [Supabase](https://supabase.com/) | Managed PostgreSQL with `pgvector` extension and IVFFlat index |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Async REST API with automatic OpenAPI Swagger documentation |

---

## 📁 Project Structure

```text
ai-research-agent/
├── app/
│   ├── config.py             # Central environment variables & hyperparameter configs
│   ├── agent/
│   │   ├── graph.py          # LangGraph StateGraph, nodes, and conditional edges
│   │   └── tools.py          # Agent tools (document search, summarization, calculation)
│   ├── api/
│   │   └── main.py           # FastAPI application and endpoint routers
│   ├── db/
│   │   ├── client.py         # Authenticated Supabase client singleton
│   │   ├── operations.py     # Document/chunk CRUD and vector search RPC queries
│   │   └── schema.sql        # Database tables, pgvector extension, IVFFlat index, and RPC
│   └── rag/
│       ├── chain.py          # LangChain LCEL RAG synthesis chain
│       ├── embeddings.py     # Local SentenceTransformer embedding engine (singleton)
│       ├── ingestion.py      # File loaders, recursive character text chunking
│       └── retrieval.py      # Retriever interface and vector similarity search
├── tests/                    # Integration and unit tests
├── uploads/                  # Temporary document staging directory
├── .env.example              # Template for environment credentials
├── .gitignore                # Git ignore rules
├── REFERENCE.md              # Deep-dive theoretical and technical guide
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/SammySN-car/ai-research-agent.git
cd ai-research-agent
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\venv\Scripts\activate.bat

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Acceleration (Optional)**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheel:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🗄️ Database Setup (Supabase pgvector)

1. Create a project at [supabase.com](https://supabase.com).
2. Go to the **SQL Editor** in your Supabase dashboard.
3. Paste and run the entire contents of [`app/db/schema.sql`](app/db/schema.sql).
4. This script automatically:
   - Enables the `vector` extension.
   - Creates the `documents`, `chunks`, and `queries` tables with foreign keys and cascade deletes.
   - Builds an **IVFFlat** cosine index on `chunks(embedding vector_cosine_ops)`.
   - Creates the `match_documents(query_embedding, match_count, match_threshold)` RPC function.

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Configure your variables in `.env`:

```dotenv
# ── OpenRouter LLM ─────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
TEMPERATURE=0.1

# ── Supabase pgvector ─────────────────────────────────
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-key

# ── Local Embeddings (BAAI/bge-small-en-v1.5) ─────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# ── RAG Hyperparameters ───────────────────────────────
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

---

## 🧪 Testing Components

You can test individual milestones and components directly:

### Test Supabase Database Connection & Vector RPC
```bash
python -c "from app.db.client import get_supabase_client; print('Connected:', get_supabase_client() is not None)"
```

### Test Local Embedding Engine
```bash
python -c "from app.rag.embeddings import get_embedding_engine; engine = get_embedding_engine(); vec = engine.embed_text('test'); print('Model loaded! Vector dim:', len(vec), '| Device:', engine.device)"
```

---

## 🚀 Running the API Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Alternative ReDoc: `http://localhost:8000/redoc`

---

## 🗺️ Project Milestones & Progress

- [x] **Milestone 1: Project Architecture & Environment Setup**
  - Clean directory structure, `.env` management, and central configuration in `app/config.py`.
- [x] **Milestone 2: Supabase pgvector Database Layer**
  - Tables, IVFFlat index, `match_documents` RPC, and CRUD operations in `app/db/`.
- [x] **Milestone 3: Local Embedding Engine**
  - `BAAI/bge-small-en-v1.5` singleton with dynamic CUDA/CPU fallback and normalized cosine vectors in `app/rag/embeddings.py`.
- [x] **Milestone 4: Document Ingestion Pipeline**
  - PDF/TXT/MD parsing, recursive chunking, and batch vector storage in `app/rag/ingestion.py`.
- [ ] **Milestone 5: Vector Search & Hybrid Retrieval**
  - Semantic similarity search and contextual retrieval in `app/rag/retrieval.py`.
- [ ] **Milestone 6: RAG LCEL Synthesis Chain**
  - Prompt templates, context injection, and cited answering in `app/rag/chain.py`.
- [ ] **Milestone 7: Agent Tools**
  - Tool wrappers for document retrieval, summarization, and calculations in `app/agent/tools.py`.
- [ ] **Milestone 8: LangGraph Workflow**
  - Cyclic state graph with conditional edges, dynamic query routing, and reflection in `app/agent/graph.py`.
- [ ] **Milestone 9: FastAPI Endpoints & UI Integration**
  - File upload, query endpoint, streaming responses, and chat history.

---

## 📚 Deep-Dive Documentation

For comprehensive theoretical foundations, architectural rationale, and LangChain/LangGraph design patterns, consult [`REFERENCE.md`](REFERENCE.md).

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
