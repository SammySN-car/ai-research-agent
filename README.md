# AI Research Agent

An autonomous, stateful research assistant built with **LangGraph**, **LangChain**, **Supabase (pgvector)**, and **OpenRouter**.

The agent processes uploaded research documents (PDF, TXT, MD), computes dense vector embeddings locally using `sentence-transformers`, stores them in PostgreSQL with pgvector, and uses a multi-step LangGraph agent workflow to reason, invoke tools, and synthesize cited answers.

---

## 🚀 Architecture

```text
User Request / Query
       │
       ▼
[FastAPI REST API]
       │
       ▼
[LangGraph Agent Workflow]
  ├── State: Messages, Context, Tools Used
  ├── Retrieve Node (pgvector cosine search via Supabase RPC)
  ├── Agent Node (Reasoning via OpenRouter)
  └── Tool Node (search_docs, summarize_text, calculate)
```

---

## 🛠️ Tech Stack

- **Agent Framework**: LangGraph (stateful graph with conditional loops)
- **LLM Abstraction**: LangChain (LCEL, tool calling, structured outputs)
- **LLM Gateway**: OpenRouter (`openai/gpt-4o-mini` or any compatible model)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions, runs locally & free)
- **Vector Database**: Supabase (PostgreSQL + pgvector extension)
- **API**: FastAPI + Uvicorn

---

## 📁 Project Structure

```text
ai-research-agent/
├── app/
│   ├── config.py         # Central configuration & hyperparameters
│   ├── db/               # Supabase client & CRUD operations
│   ├── rag/              # Document parsing, chunking, embeddings & retrieval
│   ├── agent/            # LangGraph workflow, nodes, edges & tools
│   └── api/              # FastAPI endpoints (/upload, /ask, /documents)
├── tests/                # Component & integration test suite
├── uploads/              # Local storage for uploaded documents
├── .env.example          # Environment variables template
├── requirements.txt      # Project dependencies
└── REFERENCE.md          # Comprehensive theory and implementation guide
```

---

## ⚙️ Quickstart

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/SammySN-car/ai-research-agent.git
cd ai-research-agent

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
TEMPERATURE=0.1

SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 3. Run the Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at `http://localhost:8000/docs`.
