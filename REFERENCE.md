# AI Research Agent — Complete Reference Guide

> Everything you need to understand before building. Theory → Reference Code → Project Milestones.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Theory: LLMs & Generative AI](#2-theory-llms-and-generative-ai)
3. [Theory: Embeddings & Vector Search](#3-theory-embeddings-and-vector-search)
4. [Theory: RAG (Retrieval-Augmented Generation)](#4-theory-rag-retrieval-augmented-generation)
5. [Theory: LangChain](#5-theory-langchain)
   - [What is LangChain?](#what-is-langchain)
   - [Package Ecosystem](#package-ecosystem)
   - [Core Concept 1: Messages](#core-concept-1-messages)
   - [Core Concept 2: Chat Models](#core-concept-2-chat-models)
   - [Core Concept 3: Prompt Templates](#core-concept-3-prompt-templates)
   - [Core Concept 4: Output Parsers](#core-concept-4-output-parsers)
   - [Core Concept 5: Tools](#core-concept-5-tools)
   - [Core Concept 6: LCEL (LangChain Expression Language) Deep Dive](#core-concept-6-langchain-expression-language-lcel---deep-dive)
   - [Core Concept 7: Callbacks and Debugging](#core-concept-7-callbacks-and-debugging)
   - [Core Concept 8: Short-term Memory (Chat History)](#core-concept-8-short-term-memory-chat-history)
   - [Core Concept 9: Retrieval](#core-concept-9-retrieval)
   - [Core Concept 10: Event Streaming](#core-concept-10-event-streaming)
   - [Core Concept 11: Human-in-the-Loop](#core-concept-11-human-in-the-loop)
   - [Core Concept 12: Guardrails](#core-concept-12-guardrails)
   - [Core Concept 13: Multi-agent Patterns (Overview)](#core-concept-13-multi-agent-patterns-overview)
   - [Core Concept 14: Testing LangChain Applications](#core-concept-14-testing-langchain-applications)
   - [Core Concept 15: Advanced Tool Calling & Schema Control](#core-concept-15-advanced-tool-calling--schema-control)
   - [Core Concept 16: RunnableConfig & Runtime Execution](#core-concept-16-runnableconfig--runtime-execution)
   - [Core Concept 17: Model Context Protocol (MCP)](#core-concept-17-model-context-protocol-mcp)
   - [Core Concept 18: Advanced Retrieval Strategies & Hybrid Search](#core-concept-18-advanced-retrieval-strategies--hybrid-search)
   - [Core Concept 19: RAG Evaluation & Benchmarking (RAGAS)](#core-concept-19-rag-evaluation--benchmarking-ragas)
   - [Common LangChain Gotchas](#common-langchain-gotchas)
6. [Theory: LangGraph](#6-theory-langgraph)
   - [What is LangGraph?](#what-is-langgraph)
   - [Why LangGraph over Plain LangChain?](#why-langgraph-over-plain-langchain)
   - [Mental Model: Graphs](#mental-model-graphs)
   - [Core Concept 1: State (Deep Dive)](#core-concept-1-state-deep-dive)
   - [Core Concept 2: Nodes](#core-concept-2-nodes)
   - [Core Concept 3: Edges](#core-concept-3-edges)
   - [Core Concept 4: Building and Compiling the Graph](#core-concept-4-building-and-compiling-the-graph)
   - [Core Concept 5: Prebuilt Components](#core-concept-5-prebuilt-components)
   - [Core Concept 6: Checkpointing and Persistence](#core-concept-6-checkpointing-and-persistence)
   - [Core Concept 7: Recursion Limit](#core-concept-7-recursion-limit)
   - [Core Concept 8: Streaming from Graphs](#core-concept-8-streaming-from-graphs)
   - [Core Concept 9: Graph API vs Functional API](#core-concept-9-graph-api-vs-functional-api)
   - [Core Concept 10: Workflows vs Agents](#core-concept-10-workflows-vs-agents)
   - [Core Concept 11: Interrupts and Human-in-the-Loop (Deep Dive)](#core-concept-11-interrupts-and-human-in-the-loop-deep-dive)
   - [Core Concept 12: Subgraphs](#core-concept-12-subgraphs)
   - [Core Concept 13: Fault Tolerance](#core-concept-13-fault-tolerance)
   - [Core Concept 14: Streaming Modes (Deep Dive)](#core-concept-14-streaming-modes-deep-dive)
   - [Core Concept 15: Stores (Cross-thread Memory)](#core-concept-15-stores-cross-thread-memory)
   - [Core Concept 16: Time Travel](#core-concept-16-time-travel)
   - [Core Concept 17: Common Errors & Fixes](#core-concept-17-common-errors--fixes)
   - [Core Concept 18: Application Structure](#core-concept-18-application-structure)
   - [Core Concept 19: Testing LangGraph Applications](#core-concept-19-testing-langgraph-applications)
   - [Core Concept 20: Parallel Node Execution (Fan-Out & Fan-In / Map-Reduce)](#core-concept-20-parallel-node-execution-fan-out--fan-in--map-reduce)
   - [Core Concept 21: Multi-Agent Architectures (Deep Dive)](#core-concept-21-multi-agent-architectures-deep-dive)
   - [Core Concept 22: State Inspection & Dynamic Editing (`get_state` & `update_state`)](#core-concept-22-state-inspection--dynamic-editing-get_state--update_state)
   - [Core Concept 23: Complete Memory Model: Short-Term vs Long-Term vs Semantic](#core-concept-23-complete-memory-model-short-term-vs-long-term-vs-semantic)
   - [Our Agent's Graph (Full Picture)](#our-agents-graph-full-picture)
   - [Common LangGraph Gotchas](#common-langgraph-gotchas)
7. [Theory: Supabase & PostgreSQL](#7-theory-supabase-and-postgresql)
8. [Reference Code: Core Building Blocks](#8-reference-code-core-building-blocks)
   - [8.1 Sentence Transformers (Embeddings)](#81-sentence-transformers-embeddings)
   - [8.2 Text Splitting (Chunking)](#82-text-splitting-chunking)
   - [8.3 PDF Parsing](#83-pdf-parsing)
   - [8.4 Database Operations (Supabase)](#84-database-operations-supabase)
   - [8.5 Chat Model via OpenRouter (LLM)](#85-chat-model-via-openrouter-llm)
9. [Reference Code: LangChain Patterns](#9-reference-code-langchain-patterns)
   - [9.1 Basic Chain (LCEL)](#91-basic-chain-lcel)
   - [9.2 RAG Chain with Context Injection](#92-rag-chain-with-context-injection)
   - [9.3 Tool Calling (Manual Execution)](#93-tool-calling-manual-execution)
   - [9.4 Agent with Tools (Prebuilt ReAct)](#94-agent-with-tools-prebuilt-react)
   - [9.5 Structured Output (Pydantic)](#95-structured-output-pydantic)
   - [9.6 Multi-Step Chain (Compose Chains)](#96-multi-step-chain-compose-chains)
   - [9.7 Streaming Output](#97-streaming-output)
   - [9.8 Error Handling with Fallbacks](#98-error-handling-with-fallbacks)
   - [9.9 Message Trimming (Context Engineering)](#99-message-trimming-context-engineering)
   - [9.10 Event Streaming (Real-time UI)](#910-event-streaming-real-time-ui)
   - [9.11 Human-in-the-Loop (Interrupt Pattern)](#911-human-in-the-loop-interrupt-pattern)
   - [9.12 Dynamic Model & Parameter Switching with `configurable_fields`](#912-dynamic-model--parameter-switching-with-configurable_fields)
   - [9.13 Hybrid Retrieval (BM25 + Dense Vector Search)](#913-hybrid-retrieval-bm25--dense-vector-search)
10. [Reference Code: LangGraph Patterns](#10-reference-code-langgraph-patterns)
   - [10.1 Simple Linear Graph](#101-simple-linear-graph)
   - [10.2 Conditional Routing](#102-conditional-routing)
   - [10.3 Tool-Calling Agent Loop (Manual — Core Pattern)](#103-tool-calling-agent-loop-manual--core-pattern)
   - [10.4 Prebuilt ReAct Agent](#104-prebuilt-react-agent)
   - [10.5 Using Prebuilt ToolNode and tools_condition](#105-using-prebuilt-toolnode-and-tools_condition)
   - [10.6 Graph with Checkpointing (Conversation Memory)](#106-graph-with-checkpointing-conversation-memory)
   - [10.7 Streaming from a Graph](#107-streaming-from-a-graph)
   - [10.8 Subgraph Composition](#108-subgraph-composition)
   - [10.9 Fault-Tolerant Node with Retry](#109-fault-tolerant-node-with-retry)
   - [10.10 Stream Modes](#1010-stream-modes)
   - [10.11 Time Travel (State Forking)](#1011-time-travel-state-forking)
   - [10.12 Parallel Node Execution (Fan-Out & Fan-In / Map-Reduce)](#1012-parallel-node-execution-fan-out--fan-in--map-reduce)
   - [10.13 Multi-Agent Supervisor Pattern](#1013-multi-agent-supervisor-pattern)
   - [10.14 State Inspection & Dynamic Editing (`update_state`)](#1014-state-inspection--dynamic-editing-update_state)
11. [Milestone 1: Project Setup & Config](#milestone-1-project-setup-and-config)
12. [Milestone 2: Database Schema](#milestone-2-database-schema)
13. [Milestone 3: Embedding Engine](#milestone-3-embedding-engine)
14. [Milestone 4: Document Ingestion Pipeline](#milestone-4-document-ingestion-pipeline)
15. [Milestone 5: RAG Retrieval](#milestone-5-rag-retrieval)
16. [Milestone 6: LangChain Retrieval Chain](#milestone-6-langchain-retrieval-chain)
17. [Milestone 7: Tool Definitions](#milestone-7-tool-definitions)
18. [Milestone 8: LangGraph Agent](#milestone-8-langgraph-agent)
19. [Milestone 9: FastAPI Endpoints](#milestone-9-fastapi-endpoints)
20. [Milestone 10: Testing & Polish](#milestone-10-testing-and-polish)
21. [Interview Talking Points](#interview-talking-points)

---

# 1. Project Architecture

## What We're Building

An AI Research Agent that:

1. Accepts uploaded documents (PDF, TXT, MD)
2. Chunks them, embeds them, stores them in Supabase (PostgreSQL + pgvector)
3. When a user asks a question:
   - Searches for relevant chunks via vector similarity
   - Decides which tools to invoke (search, calculate, summarize)
   - Synthesizes an answer with citations
4. Stores query history for reference and analytics

## High-Level Flow

```text
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│            LangGraph Agent              │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  │ search  │  │ calcul- │  │ summa- │  │
│  │ _docs   │  │  ate    │  │  rize  │  │
│  └────┬────┘  └────┬────┘  └───┬────┘  │
│       │            │           │        │
│       ▼            ▼           ▼        │
│  ┌─────────────────────────────────┐    │
│  │       State Management          │    │
│  │   (messages, context, tools)    │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│          Supabase (PostgreSQL)          │
│  ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │documents │ │  chunks  │ │queries │  │
│  │          │ │+ vectors │ │        │  │
│  └──────────┘ └──────────┘ └────────┘  │
└─────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Agent Framework | LangGraph | Stateful multi-step workflows with loops and conditional routing |
| LLM Abstraction | LangChain | Chains, tools, structured output, prompt templates |
| Embeddings | sentence-transformers | Local, fast, 384-dim vectors — no API calls needed |
| Database | Supabase (PostgreSQL + pgvector) | Hosted, vector search built-in, generous free tier |
| API | FastAPI | Async-native, automatic OpenAPI docs, type-safe |
| LLM Provider | OpenRouter (`openai/gpt-4o-mini`) | Unified API gateway, access to 100+ models, OpenAI-compatible endpoint, reliable tool calling |

---

# 2. Theory: LLMs and Generative AI

## What is an LLM?

A **Large Language Model (LLM)** is a deep neural network — typically based on the **Transformer architecture** — trained on massive text corpora to predict the next token in a sequence. Through this seemingly simple objective (next-token prediction), the model learns grammar, facts, reasoning patterns, and even coding ability.

Key properties of modern LLMs:

- **Autoregressive generation**: The model generates text one token at a time, feeding each generated token back as input for the next prediction.
- **In-context learning**: LLMs can perform tasks they were never explicitly trained on by following instructions and examples provided in the prompt — no fine-tuning required.
- **Emergent abilities**: At sufficient scale (billions of parameters), LLMs exhibit capabilities like chain-of-thought reasoning, code generation, and multi-step problem solving that smaller models lack.

## Key Concepts

### Tokens

Text is split into **tokens** — subword units, not whole words. Tokenization uses algorithms like Byte-Pair Encoding (BPE) that balance vocabulary size with coverage.

- `"embedding"` → `["embed", "ding"]` (2 tokens)
- `"ChatGPT"` → `["Chat", "G", "PT"]` (3 tokens)
- A rough rule of thumb: **1 token ≈ 4 characters** in English.

Every model has a **context window** — the maximum number of tokens it can process in a single request (input + output combined):

| Model | Context Window | Approximate Pages |
|---|---|---|
| `gpt-4o-mini` | 128K tokens | ~200 pages |
| `gpt-4o` | 128K tokens | ~200 pages |
| `claude-3.5-sonnet` | 200K tokens | ~300 pages |

**Why this matters for our project**: When we build RAG, we stuff retrieved document chunks into the prompt. We must ensure `system prompt + context chunks + user question + expected answer` fits within the context window.

### Temperature

Temperature controls the **randomness** of the model's token selection:

- **Temperature = 0**: Deterministic — always picks the highest-probability token. Best for factual Q&A.
- **Temperature = 0.5**: Moderate creativity. Good for general-purpose tasks.
- **Temperature = 1.0**: High creativity. Good for brainstorming, creative writing.
- **Temperature > 1.0**: Increasingly chaotic and incoherent output.

**Our project uses `temperature=0.1`** — near-deterministic for consistent, factual research answers, with a tiny bit of variation to avoid repetitive phrasing.

### Function Calling / Tool Use

Modern LLMs support **structured tool calling**: you describe available tools (name, description, parameters) in the system message, and the model responds with a structured JSON object indicating which tool to call and with what arguments.

The flow works like this:

1. You send the LLM a prompt along with tool definitions.
2. The LLM decides **whether** a tool is needed based on the question.
3. If yes, the LLM returns a `tool_call` object (not free text) specifying the tool name and arguments.
4. Your code executes the tool and sends the result back to the LLM.
5. The LLM incorporates the tool result to produce its final answer.

This is the foundation of our agent — the LLM dynamically decides when to search documents, perform calculations, or summarize text.

### Prompt Engineering Basics

The quality of LLM output depends heavily on prompt design:

- **System prompt**: Sets the persona, rules, and context. Persistent across the conversation.
- **Few-shot examples**: Providing input/output examples in the prompt dramatically improves accuracy on structured tasks.
- **Chain-of-thought**: Asking the model to "think step by step" improves reasoning on complex questions.

In our project, we use a system prompt that includes retrieved document context and instructions to cite sources.

---

# 3. Theory: Embeddings and Vector Search

## What is an Embedding?

An **embedding** is a dense, fixed-length vector (array of floats) that represents a piece of text in a continuous mathematical space. Unlike keyword-based approaches (which match exact words), embeddings capture **semantic meaning** — texts with similar meanings land near each other in the vector space, even if they use completely different words.

```text
"machine learning" → [0.12, -0.45, 0.78, ..., 0.33]   (384 dimensions)
"artificial intelligence" → [0.14, -0.41, 0.80, ..., 0.31]  ← nearby!
"chocolate cake recipe" → [-0.67, 0.22, -0.11, ..., 0.89]  ← far away
```

## Why Embeddings?

| Feature | Keyword Search (TF-IDF, BM25) | Embedding Search (Vector) |
|---|---|---|
| Matches synonyms | ❌ | ✅ |
| Handles paraphrasing | ❌ | ✅ |
| Language-agnostic | ❌ | ✅ (multilingual models) |
| Requires exact words | ✅ | ❌ |
| Speed at scale | ✅ Very fast | ✅ Fast with indexes (IVFFlat, HNSW) |
| Interpretability | ✅ Easy to debug | ❌ Black box |

For our research agent, embedding search is essential — users will ask questions in natural language that won't match document text word-for-word.

## Cosine Similarity

The standard metric for comparing embeddings is **cosine similarity** — it measures the angle between two vectors, ignoring magnitude:

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}$$

- **1.0** = identical direction (semantically identical)
- **0.0** = orthogonal (unrelated)
- **-1.0** = opposite (rare with normalized embeddings)

When embeddings are **L2-normalized** (unit length), cosine similarity simplifies to a dot product:

$$\text{cosine\_similarity} = \mathbf{A} \cdot \mathbf{B} \quad \text{(when } \|\mathbf{A}\| = \|\mathbf{B}\| = 1\text{)}$$

Our embedding model outputs normalized vectors, so we use dot product for fast comparison.

**pgvector operators**:

| Operator | Metric | Use |
|---|---|---|
| `<=>` | Cosine distance | `1 - cosine_similarity` (lower = more similar) |
| `<->` | L2 (Euclidean) distance | Straight-line distance |
| `<#>` | Inner product (negative) | For normalized vectors |

## Our Embedding Model

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max Sequence Length | 256 tokens (~200 words) |
| Speed | ~14,000 sentences/sec on GPU, fast on CPU |
| Size | ~80 MB |
| Normalization | L2-normalized output |

**Why this model?** It's the best balance of quality, speed, and size for a portfolio project. It runs locally (no API costs), loads in seconds, and produces high-quality embeddings for English text.

---

# 4. Theory: RAG (Retrieval-Augmented Generation)

## The Problem

LLMs have fundamental limitations that RAG addresses:

1. **Knowledge cutoff**: Training data has a fixed date. The model doesn't know about events after that date.
2. **No private data access**: The model has never seen your company's internal documents, research papers, or proprietary data.
3. **Hallucination**: When the model doesn't know an answer, it may confidently fabricate plausible-sounding but incorrect information.
4. **No citations**: Without grounding, the model can't tell you *where* it got its information.

## The Solution: RAG

**Retrieval-Augmented Generation** solves these problems by grounding the LLM's responses in actual retrieved documents:

```text
┌─────────────────────────────────────────────────────┐
│                    RAG Pipeline                     │
│                                                     │
│  1. RETRIEVE    2. AUGMENT       3. GENERATE        │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐  │
│  │  Embed   │   │ Inject     │   │ LLM answers  │  │
│  │  query,  │──▶│ retrieved  │──▶│ based on     │  │
│  │  search  │   │ chunks     │   │ context      │  │
│  │  vectors │   │ into       │   │ + cites      │  │
│  │          │   │ prompt     │   │ sources      │  │
│  └──────────┘   └────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Step 1 — RETRIEVE**: Embed the user's question with the same model used for documents. Search the vector store for the most similar chunks (top-K nearest neighbors).

**Step 2 — AUGMENT**: Insert the retrieved chunks into the LLM's system prompt as context. The prompt template explicitly instructs the model to answer based on this context.

**Step 3 — GENERATE**: The LLM reads the context + question and generates a grounded answer. Because the answer is derived from real documents, it can include citations.

## Chunking Strategy

Documents must be split into smaller pieces before embedding, for two reasons:

1. **Context window limits**: An entire PDF won't fit in a single prompt.
2. **Embedding precision**: Embedding a whole document produces a blurred, averaged vector. Embedding small chunks produces precise, topic-specific vectors that match targeted queries better.

### Our Chunking Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 500 characters | Small enough for precise retrieval, large enough for coherent context |
| Overlap | 50 characters | Prevents information loss at chunk boundaries |
| Method | Recursive character splitting | Respects paragraph and sentence boundaries |

### How Recursive Character Splitting Works

The `RecursiveCharacterTextSplitter` tries separators in order of priority:

1. `"\n\n"` — Split on paragraph breaks (best)
2. `"\n"` — Split on line breaks
3. `". "` — Split on sentence boundaries
4. `" "` — Split on word boundaries (last resort)

It recursively tries the highest-priority separator first. If a resulting chunk is still too large, it falls back to the next separator. This preserves document structure as much as possible.

### Overlap Explained

```text
Chunk 1: "...the model uses attention mechanisms to weigh the importance of"
Chunk 2: "to weigh the importance of different input tokens when generating"
              ▲─── 50-char overlap ───▲
```

Without overlap, a sentence split exactly at a chunk boundary would lose its meaning in both chunks. The 50-character overlap ensures continuity.

---

# 5. Theory: LangChain

## What is LangChain?

**LangChain** is a Python framework for building applications powered by language models. It provides standardized abstractions for the most common LLM operations — prompts, model calls, output parsing, tool use, and retrieval — so you don't have to write boilerplate code for each.

Think of LangChain as the **"requests" library for LLMs** — you *could* make raw HTTP calls to the OpenAI API, but LangChain gives you a cleaner, composable interface with built-in support for streaming, async, retries, and composability.

## Package Ecosystem

LangChain is split into multiple packages. Understanding which package to import from is critical:

| Package | What It Contains | Install |
|---|---|---|
| `langchain-core` | Base abstractions: Runnables, messages, prompts, output parsers, tools | Installed automatically |
| `langchain` | Higher-level chains, agents, retrieval strategies | `pip install langchain` |
| `langchain-openai` | OpenAI + Azure OpenAI chat models and embeddings | `pip install langchain-openai` |
| `langchain-community` | Third-party integrations (Supabase, Wikipedia, etc.) | `pip install langchain-community` |
| `langchain-text-splitters` | Document splitting / chunking utilities | `pip install langchain-text-splitters` |
| `langgraph` | Stateful agent workflows (graphs) | `pip install langgraph` |

**Rule of thumb**: Import from the most specific package. Use `langchain_core` for base types, `langchain_openai` for OpenAI models, `langchain_community` for integrations.

## Core Concept 1: Messages

LangChain uses a **message-based** interface for chat models. Every interaction is a list of typed messages:

```python
from langchain_core.messages import (
    SystemMessage,    # Sets the AI's persona / rules
    HumanMessage,     # User input
    AIMessage,        # Model response
    ToolMessage,      # Result of a tool call (sent back to the model)
)

messages = [
    SystemMessage(content="You are a research assistant."),
    HumanMessage(content="What is RAG?"),
]
```

**Message flow in a tool-calling conversation**:

```text
1. SystemMessage    → "You are a research assistant..."
2. HumanMessage     → "Search for papers about transformers"
3. AIMessage        → (contains tool_calls: [{name: "search_docs", args: {...}}])
4. ToolMessage      → "Found 3 papers: ..." (tool_call_id must match step 3)
5. AIMessage        → "Based on the search results, here are the papers..."
```

The `AIMessage` in step 3 doesn't contain text — it contains structured `tool_calls`. Your code executes the tool, wraps the result in a `ToolMessage` (with the matching `tool_call_id`), and sends it back. The model then generates a final text response.

**Shorthand tuple syntax** (used in prompts and simple invocations):

```python
# These are equivalent:
messages = [("system", "You are helpful."), ("human", "Hi!")]
messages = [SystemMessage(content="You are helpful."), HumanMessage(content="Hi!")]
```

## Core Concept 2: Chat Models

LangChain wraps LLM providers behind a **unified interface**. Every chat model implements standard methods (`invoke`, `stream`, `batch`, `ainvoke`, `astream`, `bind_tools`, `with_structured_output`), making switching models or API providers a one-line configuration change.

### Connecting to OpenRouter (Multi-Model Gateway)

[OpenRouter](https://openrouter.ai) exposes an **OpenAI-compatible API endpoint** (`https://openrouter.ai/api/v1`) that gives access to hundreds of models (GPT-4o, Claude 3.5 Sonnet, Llama 3.3, Gemini 1.5, DeepSeek) through a single API key and unified billing.

Because it adheres to the OpenAI API specification, you connect to OpenRouter in LangChain using standard `ChatOpenAI` from `langchain_openai` without installing any extra provider packages:

```python
import os
from langchain_openai import ChatOpenAI

# Initialize ChatOpenAI pointing to OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",                     # OpenRouter model slug (provider/model)
    temperature=0.1,                                # Determinism (0.0 = strict, 1.0 = creative)
    api_key=os.getenv("OPENROUTER_API_KEY"),        # Your OpenRouter API key
    base_url="https://openrouter.ai/api/v1",        # OpenRouter gateway URL
    default_headers={
        "HTTP-Referer": "http://localhost:8000",   # Optional: your app URL for OpenRouter rankings
        "X-Title": "AI Research Agent",            # Optional: your app title on OpenRouter dashboard
    },
)

# Direct invocation — returns an AIMessage
response = llm.invoke("What is RAG?")
print(response.content)        # "RAG stands for..."
print(type(response))          # <class 'langchain_core.messages.AIMessage'>

# With a list of messages
response = llm.invoke([
    ("system", "Answer in one sentence."),
    ("human", "What is machine learning?"),
])
print(response.content)
```

> [!NOTE]
> **Why this project needs only ONE API key**:
> In our research agent, document embeddings are computed **100% locally** using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions) on your machine. You do **not** need an OpenAI embedding key or any vector API billing. Your OpenRouter key handles all reasoning, tool decisions, and answer synthesis!

### Key `ChatOpenAI` Parameters (with OpenRouter)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | Required | OpenRouter model slug, e.g. `"openai/gpt-4o-mini"`, `"anthropic/claude-3.5-sonnet"` |
| `api_key` | `str` | `OPENAI_API_KEY` | Your OpenRouter key (`sk-or-v1-...`), read from `OPENROUTER_API_KEY` |
| `base_url` | `str` | OpenAI default | Point to `"https://openrouter.ai/api/v1"` for OpenRouter |
| `temperature` | `float` | `0.7` | Sampling temperature: `0` = deterministic, `1` = creative |
| `max_tokens` | `int` | `None` | Maximum tokens allowed in the completion response |
| `default_headers` | `dict` | `None` | Custom headers (`HTTP-Referer`, `X-Title`) for OpenRouter analytics |
| `streaming` | `bool` | `False` | Enable token-by-token streaming |
| `max_retries` | `int` | `2` | Maximum retry attempts upon transient network errors |

**Model with tools bound** (used in our agent):

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search documents."""
    return "results"

# bind_tools() tells the model about available tools
llm_with_tools = llm.bind_tools([search])
response = llm_with_tools.invoke("Search for info about Python")

# If the model wants to call a tool:
print(response.tool_calls)  # [{"name": "search", "args": {"query": "Python"}, "id": "call_abc123"}]
print(response.content)     # "" (empty when making tool calls)
```

## Core Concept 3: Prompt Templates

Prompt templates separate the **structure** of a prompt from its **data**. This makes prompts reusable, testable, and composable in LCEL chains.

### ChatPromptTemplate (Primary — used in our project)

```python
from langchain_core.prompts import ChatPromptTemplate

# From tuples — the most common pattern
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# .invoke() fills in variables and returns a list of messages
messages = prompt.invoke({
    "context": "RAG = Retrieval-Augmented Generation...",
    "question": "What is RAG?",
})
# Returns: [SystemMessage(content="You are a research assistant..."),
#           HumanMessage(content="What is RAG?")]
```

### MessagesPlaceholder (For dynamic message history)

When building agents, you need to inject a variable-length list of messages (chat history, tool calls) into the template:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Injects N messages here
    ("human", "{question}"),
])

# Usage with chat history
from langchain_core.messages import HumanMessage, AIMessage

messages = prompt.invoke({
    "chat_history": [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
    ],
    "question": "What are its main uses?",
})
```

### PromptTemplate (For plain text — less common)

```python
from langchain_core.prompts import PromptTemplate

# For non-chat models or sub-prompts
template = PromptTemplate.from_template(
    "Summarize this text in {num_sentences} sentences:\n{text}"
)
result = template.invoke({"num_sentences": 3, "text": "Long document..."})
```

## Core Concept 4: Output Parsers

Output parsers transform the raw `AIMessage` from the LLM into the format you need.

### StrOutputParser (Most common — used in our project)

Extracts the `.content` string from an `AIMessage`:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# In a chain: prompt | llm | parser
# llm returns AIMessage(content="RAG stands for...")
# parser returns "RAG stands for..." (just the string)
```

### JsonOutputParser

Parses LLM output as JSON (when you prompt the model to return JSON):

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Answer(BaseModel):
    text: str = Field(description="The answer")
    confidence: float = Field(description="Confidence 0-1")

parser = JsonOutputParser(pydantic_object=Answer)

# Include format instructions in your prompt
print(parser.get_format_instructions())
# "The output should be formatted as a JSON instance that conforms to..."
```

### Structured Output (Best approach — used in our project)

Instead of parsing text output, use `.with_structured_output()` to get the model to return a Pydantic object directly via function calling:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class ResearchAnswer(BaseModel):
    """A research answer with sources."""
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    sources: list[str] = Field(description="Source references")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm = llm.with_structured_output(ResearchAnswer)

# Returns a ResearchAnswer instance (not text, not dict — a Pydantic object)
result = structured_llm.invoke("What is retrieval-augmented generation?")
print(result.answer)      # "RAG is a technique..."
print(result.confidence)  # 0.95
print(result.sources)     # ["Lewis et al. 2020", ...]
```

**Why `.with_structured_output()` is better than JSON parsing**: It uses the model's native function-calling capability (not text parsing), so it's more reliable and never produces malformed JSON.

## Core Concept 5: Tools

Tools are **functions the LLM can decide to call**. The LLM reads the tool's name, description, and parameter schema to decide when and how to invoke it.

### Defining Tools with the `@tool` Decorator

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search uploaded documents for content relevant to the query.

    Use this tool when the user asks about information that might be
    in the uploaded research documents.
    """
    # The docstring is critical — the LLM reads it to decide when to use this tool
    return f"Search results for: {query}"

# What the LLM sees:
print(search_docs.name)          # "search_docs"
print(search_docs.description)   # "Search uploaded documents for..."
print(search_docs.args_schema.model_json_schema())
# {"properties": {"query": {"type": "string"}}, "required": ["query"]}
```

### Tool Schema (What the LLM Receives)

When you call `llm.bind_tools([search_docs, calculate])`, LangChain sends the model a JSON schema for each tool:

```json
{
  "type": "function",
  "function": {
    "name": "search_docs",
    "description": "Search uploaded documents for content relevant to the query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": ""}
      },
      "required": ["query"]
    }
  }
}
```

**Docstring best practices**:
- First line: What the tool does (concise)
- Remaining lines: When to use it and examples
- The LLM uses this to decide which tool to call — vague descriptions = poor tool selection

### Tool Execution Flow

```python
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Model decides to call a tool
response = llm_with_tools.invoke([HumanMessage(content="Search for RAG papers")])

# 2. Check if the model wants to call tools
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc['name']}, Args: {tc['args']}, ID: {tc['id']}")
        # Tool: search_docs, Args: {"query": "RAG papers"}, ID: call_abc123

        # 3. Execute the tool
        result = search_docs.invoke(tc["args"])

        # 4. Wrap result in ToolMessage (id MUST match)
        tool_msg = ToolMessage(content=result, tool_call_id=tc["id"])

        # 5. Send everything back to the model for final answer
        final = llm_with_tools.invoke([
            HumanMessage(content="Search for RAG papers"),
            response,       # The AIMessage with tool_calls
            tool_msg,       # The tool result
        ])
        print(final.content)  # "Here are some papers about RAG..."
```

## Core Concept 6: LangChain Expression Language (LCEL) — Deep Dive

LCEL is the **composition system** that connects LangChain components using the pipe operator (`|`). Understanding LCEL is essential — it's how you build every chain and pipeline in this project.

### The Runnable Protocol

Every LangChain component implements the **Runnable** interface:

```python
class Runnable:
    def invoke(self, input)        # Run synchronously
    def ainvoke(self, input)       # Run asynchronously
    def stream(self, input)        # Stream output tokens
    def astream(self, input)       # Async stream
    def batch(self, inputs)        # Run on multiple inputs
```

Anything that implements `Runnable` can be piped with `|`. This includes prompts, models, parsers, tools, and custom functions.

### Basic Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
parser = StrOutputParser()

# The pipe operator creates a RunnableSequence
chain = prompt | llm | parser

# Equivalent to: parser.invoke(llm.invoke(prompt.invoke({"question": "..."})))
result = chain.invoke({"question": "What is Python?"})
```

### RunnablePassthrough (Pass input through unchanged)

Used when you need to forward the original input alongside transformed data:

```python
from langchain_core.runnables import RunnablePassthrough

# Pass the question through while also retrieving context
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
result = chain.invoke("What is machine learning?")
# "What is machine learning?" is passed as both the retriever input AND {question}
```

### RunnableParallel (Run multiple chains simultaneously)

Creates a dict where each value is a Runnable that processes the same input:

```python
from langchain_core.runnables import RunnableParallel

# Both branches receive the same input and run in parallel
parallel = RunnableParallel(
    context=retriever,                    # Retrieves documents
    question=RunnablePassthrough(),       # Passes input through
)

result = parallel.invoke("What is RAG?")
# result = {"context": [Document(...)], "question": "What is RAG?"}
```

### RunnableLambda (Wrap any function as a Runnable)

Converts a plain Python function into a Runnable so it can be used in LCEL chains:

```python
from langchain_core.runnables import RunnableLambda

def format_docs(docs: list) -> str:
    """Convert retrieved documents to a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Now usable in a chain
chain = retriever | RunnableLambda(format_docs) | prompt | llm | parser
```

### Complete RAG Chain Example (LCEL)

Putting it all together — this is the pattern used in our Milestone 6:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_context(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context:\n{context}"),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_context),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is attention in transformers?")
```

### Streaming

Every LCEL chain supports streaming out of the box:

```python
# Stream tokens as they're generated
for chunk in chain.stream({"question": "Explain RAG"}):
    print(chunk, end="", flush=True)

# Async streaming (for FastAPI)
async for chunk in chain.astream({"question": "Explain RAG"}):
    print(chunk, end="", flush=True)
```

### Async Support

Every `.invoke()` has an async counterpart:

```python
# Synchronous
result = chain.invoke({"question": "What is RAG?"})

# Asynchronous (use in FastAPI async endpoints)
result = await chain.ainvoke({"question": "What is RAG?"})
```

### Batch Processing

Process multiple inputs efficiently:

```python
questions = [
    {"question": "What is RAG?"},
    {"question": "What is LangChain?"},
    {"question": "What is an embedding?"},
]

# Processes all 3 in parallel
results = chain.batch(questions)
```

## Core Concept 7: Callbacks and Debugging

LangChain provides a **callback system** for logging, debugging, and tracing:

```python
from langchain_core.callbacks import StdOutCallbackHandler

# See every step of the chain execution
result = chain.invoke(
    {"question": "What is RAG?"},
    config={"callbacks": [StdOutCallbackHandler()]},
)
```

**LangSmith** (optional) provides a web UI for tracing:

```python
# Set environment variables for automatic tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-langsmith-key
```

## Core Concept 8: Short-term Memory (Chat History)

Managing conversation history is a critical component of any chat-based AI system. As the conversation grows, you will eventually hit context window limits or accumulate unnecessary token costs. LangChain provides native utilities to manage chat history without complex custom logic.

### Trimming Messages
The `trim_messages()` utility lets you truncate your chat history based on specific criteria. The most common approach is limiting the history by token count or by a maximum number of messages.

When trimming, you can use two strategies:
- `strategy="last"`: Keeps the most recent messages (most common).
- `strategy="first"`: Keeps the earliest messages.

### Filtering Messages
The `filter_messages()` utility allows you to remove specific types of messages from the history, for example, retaining only `HumanMessage` and `AIMessage` objects while discarding intermediate `SystemMessage` or `ToolMessage` instances if they are no longer relevant to the LLM's context.

### Why Memory Management Matters
Effective memory management directly impacts:
1. **Context Window Limits**: Every model has a maximum token limit. Passing a massive history will cause the API call to fail.
2. **Cost**: API pricing is usually based on input tokens. Sending the entire history for every turn gets expensive fast.
3. **Relevance**: Older messages might distract the model or provide stale context.

```python
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, trim_messages, filter_messages
)

messages = [
    SystemMessage(content="You are a research assistant."),
    HumanMessage(content="What is ML?"),
    AIMessage(content="Machine learning is..."),
    HumanMessage(content="What about deep learning?"),
    AIMessage(content="Deep learning is a subset..."),
    HumanMessage(content="Tell me about transformers."),
]

# Trim to last N tokens (keeps most recent messages)
trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",
    token_counter=len,  # Simple char-based; use tiktoken for real token counting
    allow_partial=False,
    include_system=True,  # Always keep the system message
)

# Filter by type
filtered = filter_messages(messages, include_types=[HumanMessage, AIMessage])
```

## Core Concept 9: Retrieval

Retrieval is how you provide your LLM with external knowledge (commonly known as RAG: Retrieval-Augmented Generation). The full retrieval pipeline in LangChain involves: **Load → Split → Embed → Store → Retrieve**.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. Load: extract text and metadata from files
# In our project: parse_file() in app/rag/ingestion.py handles PDF, TXT, MD
loader = TextLoader("document.txt")  # or PyPDFLoader("paper.pdf")
raw_docs = loader.load()

# 2. Split: break into overlapping chunks to fit LLM context window
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(raw_docs)
print(f"Created {len(chunks)} chunks from {len(raw_docs)} document(s)")

# 3. Embed & Store: convert text to vectors and store in vector database
# In our project: Local sentence-transformers ('all-MiniLM-L6-v2', 384 dimensions)
# requires NO external API keys and runs completely free/offline:
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# (If using direct OpenAI embeddings: OpenAIEmbeddings(model="text-embedding-3-small"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# 4. Retrieve: query vector store for semantic matches
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # Top 3 most relevant chunks
)
retrieved_docs = retriever.invoke("What is transformer self-attention?")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"[{i}] {doc.page_content[:80]}... (Metadata: {doc.metadata})")
```

## Core Concept 10: Event Streaming

When building real-time UIs, waiting for a final, monolithic response from an agent can lead to a poor user experience. LangChain provides the `astream_events` (v2) API to stream granular events as a chain or agent executes.

### Types of Events
The event stream emits dictionaries that describe what is happening at each step of the pipeline. Common event types include:
- `on_chat_model_start`, `on_chat_model_stream`, `on_chat_model_end`
- `on_tool_start`, `on_tool_end`
- `on_chain_start`, `on_chain_end`

### Why This Matters
By filtering and handling these events, you can update your UI immediately. You can show intermediate tool calls, stream tokens directly as they are generated by the LLM, or display loading indicators for specific sub-components of your chain.

```python
# Stream events from a chain
async for event in chain.astream_events(
    {"question": "What is RAG?"},
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            print(content, end="", flush=True)
    elif kind == "on_tool_start":
        print(f"\nTool called: {event['name']}")
    elif kind == "on_tool_end":
        print(f"Tool result: {event['data'].get('output', '')}")
```

## Core Concept 11: Human-in-the-Loop

In production systems, you often cannot allow autonomous agents to execute irreversible actions (database deletions, sending emails, processing payments) without human oversight.

### Gated Tool Execution Pattern

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@tool
def delete_database_record(record_id: str) -> str:
    """Permanently delete a record from the database."""
    return f"Record {record_id} successfully deleted."

def human_approval_gate(action_summary: str) -> bool:
    """Pause execution and request explicit human confirmation."""
    print(f"\n[APPROVAL REQUIRED]: {action_summary}")
    user_choice = input("Approve action? (yes/no): ").strip().lower()
    return user_choice in ("yes", "y")

def safe_tool_executor(tool_name: str, args: dict):
    """Execute sensitive tools only after passing human gate."""
    if tool_name == "delete_database_record":
        if not human_approval_gate(f"Delete record id='{args.get('record_id')}'"):
            return "Execution aborted: Action rejected by operator."
    return delete_database_record.invoke(args)

# Test gated execution
status = safe_tool_executor("delete_database_record", {"record_id": "rec_9921"})
print(status)
```

## Core Concept 12: Guardrails

Guardrails validate inputs before they reach the LLM and verify outputs before returning them to users, preventing prompt injections, toxic content, and malformed structures.

```python
import re
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── 1. Input Guardrail: Block Disallowed Patterns / Injections ─
FORBIDDEN_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"reveal secret key",
    r"drop database",
]

def input_guardrail(user_query: str) -> str:
    """Validate and sanitize user input before passing to LLM."""
    query_lower = user_query.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query_lower):
            raise ValueError(f"Security violation: Query contains disallowed phrase '{pattern}'")
    if len(user_query.strip()) < 3:
        raise ValueError("Query too short to process.")
    return user_query.strip()

# ── 2. Output Guardrail: Strict Schema & Value Validation ───────
class ValidatedResponse(BaseModel):
    answer: str = Field(description="Factual answer grounded in context.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0.")

    @field_validator("answer")
    def validate_answer(cls, v: str) -> str:
        if "I don't know" in v and len(v) > 200:
            raise ValueError("Uncertain responses must be concise.")
        return v

guardrailed_model = llm.with_structured_output(ValidatedResponse)

def safe_rag_pipeline(raw_query: str) -> ValidatedResponse:
    clean_query = input_guardrail(raw_query)  # Rejects malicious input
    return guardrailed_model.invoke(f"Answer factually: {clean_query}")

# Safe invocation
result = safe_rag_pipeline("What is backpropagation in neural networks?")
print(f"Answer: {result.answer[:80]}... | Confidence: {result.confidence}")
```

## Core Concept 13: Multi-agent Patterns (Overview)

When single monolithic prompts become too complex, multi-agent systems break the task across specialized agents.

### Router Pattern in LCEL

```python
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Classification Model
class RouteDecision(BaseModel):
    destination: Literal["math_specialist", "code_expert", "general"] = Field(
        description="Select the specialized sub-agent for the query."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user query into: 'math_specialist', 'code_expert', or 'general'."),
    ("human", "{query}"),
])
router_chain = router_prompt | llm.with_structured_output(RouteDecision)

# 2. Worker Chains
math_worker = ChatPromptTemplate.from_template("You are a Math Specialist. Solve:\n{query}") | llm | StrOutputParser()
code_worker = ChatPromptTemplate.from_template("You are a Python Expert. Code:\n{query}") | llm | StrOutputParser()
general_worker = ChatPromptTemplate.from_template("You are a Helpful Assistant. Answer:\n{query}") | llm | StrOutputParser()

# 3. Dynamic Dispatcher
def dispatch_query(inputs: dict) -> str:
    query = inputs["query"]
    decision = router_chain.invoke({"query": query})
    if decision.destination == "math_specialist":
        return math_worker.invoke({"query": query})
    elif decision.destination == "code_expert":
        return code_worker.invoke({"query": query})
    return general_worker.invoke({"query": query})

print(dispatch_query({"query": "Solve: integrate x^2 dx from 0 to 3."}))
print(dispatch_query({"query": "Write a Python function to check for palindromes."}))
```

## Core Concept 14: Testing LangChain Applications

Testing LLM applications is notoriously difficult because LLMs are non-deterministic. However, robust LangChain testing relies on mocking and isolation.

### Testing Strategies
- **Unit Testing**: You should test your chains by isolating them from external network calls. Use tools like `unittest.mock` to intercept LLM calls and return predefined responses (like an `AIMessage`). This ensures your parsing, prompt formatting, and routing logic work as expected.
- **Integration Testing**: These tests actually hit the LLM provider API. They are useful for checking prompt quality and ensuring the model behaves as expected with real data, but they should be run less frequently due to cost and latency.
- **Deterministic Mocks**: LangChain provides utilities like `FakeListLLM` to supply a predetermined list of responses for testing complex chains predictably.

```python
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

def test_chain_with_mock():
    """Test a chain without making real API calls."""
    mock_response = AIMessage(content="Mocked answer")
    with patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response):
        # Assuming `my_chain` is defined in your application
        result = my_chain.invoke({"question": "test"})
        assert "Mocked answer" in result
```

## Core Concept 15: Advanced Tool Calling & Schema Control

While the `@tool` decorator handles basic cases, production agents often require strict schema validation, forced tool usage, runtime dependency injection, and graceful error handling.

### 1. Explicit Pydantic Schemas (`args_schema`)
Define multi-parameter tools with validated types, field descriptions, default values, and constraint checks:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results (1-20)")
    category: str | None = Field(default=None, description="Optional category filter")

@tool("advanced_search", args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 5, category: str | None = None) -> str:
    """Search documents with category filtering and pagination."""
    return f"Found {max_results} results for '{query}' in category '{category}'"
```

### 2. Forcing Tool Selection (`tool_choice`)
Control whether and how the LLM calls tools using `tool_choice`:

```python
llm = ChatOpenAI(model="gpt-4o-mini")

# 1. "auto" (default): Model decides whether to answer with text or call tools
llm_auto = llm.bind_tools([advanced_search], tool_choice="auto")

# 2. "required": Model MUST call at least one tool (cannot respond with text)
llm_required = llm.bind_tools([advanced_search], tool_choice="required")

# 3. "none": Model is forbidden from calling tools
llm_none = llm.bind_tools([advanced_search], tool_choice="none")

# 4. Force a SPECIFIC tool by name
llm_forced = llm.bind_tools([advanced_search], tool_choice="advanced_search")
```

### 3. Tool Error Handling (`handle_tool_error`)
Prevent tool crashes from terminating the entire agent workflow:

```python
def custom_error_handler(error: Exception) -> str:
    """Format error message sent back to the LLM so it can self-correct."""
    return f"Tool execution failed with error: {error}. Please check arguments and retry."

@tool(handle_tool_error=custom_error_handler)
def execute_sql(query: str) -> str:
    """Execute a database query."""
    if "DROP" in query.upper():
        raise ValueError("Destructive queries are prohibited.")
    return "Query result..."
```

### 4. Runtime Dependency Injection (`InjectedToolArg`)
Inject state or sensitive credentials into tools at runtime without exposing them to the LLM's prompt:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def fetch_user_profile(
    user_id: str,
    api_token: Annotated[str, InjectedToolArg],  # Hidden from LLM schema!
) -> str:
    """Fetch user profile details. The LLM only sees `user_id`."""
    return f"Profile for {user_id} using internal token"
```

## Core Concept 16: RunnableConfig & Runtime Execution

`RunnableConfig` is the configuration dictionary passed to any `.invoke()`, `.ainvoke()`, `.stream()`, or `.batch()` call. It carries runtime metadata, tracing tags, callbacks, and configuration settings across the entire chain or graph.

### Key Fields in `RunnableConfig`

| Field | Type | Purpose |
|---|---|---|
| `configurable` | `dict` | Runtime parameters (thread ID, model name, temperature, user ID) |
| `callbacks` | `list` | Custom event handlers or tracing loggers |
| `tags` | `list[str]` | Labels for filtering traces in LangSmith (e.g. `["production", "rag"]`) |
| `metadata` | `dict` | Arbitrary tracking info (e.g. `{"user_id": "u123", "session_id": "s456"}`) |
| `recursion_limit` | `int` | Maximum execution steps before halting (default: 25) |
| `max_concurrency` | `int` | Maximum parallel threads for `.batch()` |

### Dynamic Model Switching with `configurable_fields`
Switch LLM models, temperatures, or prompts dynamically at request time without changing chain definitions:

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).configurable_fields(
    model_name=ConfigurableField(
        id="llm_model",
        name="LLM Model",
        description="The OpenAI model to use",
    ),
    temperature=ConfigurableField(
        id="llm_temp",
        name="Temperature",
        description="Sampling temperature",
    ),
)

# Standard invocation uses defaults (gpt-4o-mini, temp 0.1)
res1 = model.invoke("Hello")

# Override parameters dynamically per request via config
res2 = model.invoke(
    "Solve complex logic puzzle",
    config={"configurable": {"llm_model": "gpt-4o", "llm_temp": 0.0}},
)
```

## Core Concept 17: Model Context Protocol (MCP)

**Model Context Protocol (MCP)** is an open standard introduced by Anthropic that standardizes how language models connect to external data sources, developer tools, and enterprise environments.

### Why MCP Matters
Before MCP, every AI framework implemented custom tool wrappers for every API. MCP creates a universal protocol:
```text
┌─────────────────┐       MCP Protocol        ┌───────────────────────┐
│ LangChain Agent │ ◄───────────────────────► │   MCP Server (Tools)  │
│  (MCP Client)   │    JSON-RPC 2.0 (stdio/   │ (Files, GitHub, DB,   │
│                 │          SSE)             │  Slack, Custom APIs)  │
└─────────────────┘                           └───────────────────────┘
```

### Using MCP Tools in LangChain (`langchain-mcp-adapters`)

```python
# Connecting to an MCP server via stdio transport
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def run_mcp_agent():
    async with MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
                "transport": "stdio",
            }
        }
    ) as client:
        # Automatically converts MCP tools to LangChain BaseTool instances
        tools = client.get_tools()
        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(llm, tools)
        response = await agent.ainvoke({"messages": [("user", "List files in directory")]})
        print(response["messages"][-1].content)
```

## Core Concept 18: Advanced Retrieval Strategies & Hybrid Search

Naive vector search matches keywords poorly when technical acronyms, exact IDs, or rare names are queried. Production RAG pipelines use multi-stage retrieval:

### 1. Hybrid Search (Ensemble Retriever)
Combines **Dense Vector Search** (semantic similarity) with **Sparse BM25 Search** (exact keyword matching) using **Reciprocal Rank Fusion (RRF)**:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 1. Keyword search (exact terms, codes, part numbers)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 2. Vector search (conceptual meaning, synonyms)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Ensemble combining both with weighted scores (50% vector, 50% BM25)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("PostgreSQL error 42P01")
```

### 2. Contextual Compression & Reranking
Vector search retrieves top-20 candidates cheaply; a cross-encoder reranks the top-5 accurately:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Reranker model scores (query, document) pairs directly
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever,
)
```

## Core Concept 19: RAG Evaluation & Benchmarking (RAGAS)

Evaluating RAG systems requires automated scoring across both retrieval and generation stages without relying solely on manual inspection. The industry standard is **RAGAS** (Retrieval Augmented Generation Assessment):

```text
               ┌───────────────────────┐
               │    User Question      │
               └──────────┬────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐
│ Context Recall   │             │ Answer Relevance │
│ (Did we retrieve │             │ (Did we answer   │
│  all facts?)     │             │  the question?)  │
└──────────────────┘             └──────────────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐
│ Context Precision│             │   Faithfulness   │
│ (Are top chunks  │             │ (Is answer free  │
│  most relevant?) │             │  of hallucination│
└──────────────────┘             └──────────────────┘
```

| Metric | Target | Formula / Meaning |
|---|---|---|
| **Faithfulness** | Generation | $\frac{\text{Claims grounded in context}}{\text{Total claims in answer}}$. Measures hallucination. |
| **Answer Relevance** | Generation | Semantic similarity between question and generated answer. Penalizes incomplete answers. |
| **Context Precision** | Retrieval | Measures if relevant chunks appear at rank 1, 2 rather than 5, 6 (Mean Reciprocal Rank). |
| **Context Recall** | Retrieval | $\frac{\text{Ground-truth facts found in context}}{\text{Total ground-truth facts}}$. Measures coverage. |

### Programmatic Evaluation Example (LLM-as-a-Judge)

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class FaithfulnessScore(BaseModel):
    reasoning: str = Field(description="Step-by-step verification of whether claims in the answer are supported by the context.")
    is_faithful: bool = Field(description="True if all claims are supported by context without hallucination.")
    score: float = Field(ge=0.0, le=1.0, description="1.0 if fully grounded, 0.0 if fabricated.")

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator assessing RAG faithfulness. Check if the Answer is strictly grounded in the Context."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}\n\nEvaluate."),
])

judge_chain = judge_prompt | eval_llm.with_structured_output(FaithfulnessScore)

# Evaluate sample answer
test_context = "Transformers use multi-head self-attention mechanisms, first introduced in the paper 'Attention Is All You Need' (2017)."
test_answer = "Transformers rely on multi-head attention and were introduced in 2017 by Vaswani et al."

assessment = judge_chain.invoke({"context": test_context, "answer": test_answer})
print(f"Faithful: {assessment.is_faithful} | Score: {assessment.score}")
print(f"Reasoning: {assessment.reasoning}")
```

## Common LangChain Gotchas

1. **Import paths matter**: Use `langchain_openai` (not `langchain.chat_models`). The old paths are deprecated.
2. **`AIMessage` vs string**: `llm.invoke()` returns an `AIMessage` object, not a string. Use `.content` to get the text, or pipe through `StrOutputParser()`.
3. **Template variables**: Every `{variable}` in your prompt template must be provided in `.invoke()`. Missing variables raise an error.
4. **Tool docstrings**: The LLM reads tool docstrings to decide which tool to use. Poor descriptions = poor tool selection. Write them as instructions for the AI.
5. **`tool_call_id` matching**: When returning `ToolMessage`, the `tool_call_id` must match the `id` from the corresponding `tool_call`. Mismatches cause errors.
6. **OpenRouter `base_url` & model slug**: When routing through OpenRouter, always provide `base_url="https://openrouter.ai/api/v1"` and prefix models with provider tags (e.g. `"openai/gpt-4o-mini"`). Omitting the `base_url` makes LangChain default to `api.openai.com` directly, causing an invalid key error with OpenRouter keys (`sk-or-v1-...`).

---

# 6. Theory: LangGraph

## What is LangGraph?

**LangGraph** is a library (built on top of LangChain) for creating **stateful, multi-step agent workflows** as directed graphs. While LangChain chains are linear pipelines (`A → B → C`), LangGraph supports **loops, branches, and conditional routing** — essential for agents that need to decide, act, observe, and repeat.

Think of it like this: LangChain LCEL is a **conveyor belt** (data flows in one direction). LangGraph is a **flowchart** — data can loop back, branch, and merge based on runtime decisions.

## Why LangGraph over Plain LangChain?

| Feature | LangChain (LCEL) | LangGraph |
|---|---|---|
| Linear pipelines | ✅ | ✅ |
| Conditional branching | ❌ | ✅ |
| Loops (agent decides to retry) | ❌ | ✅ |
| Persistent state across steps | ❌ | ✅ |
| Multi-step tool use | Limited | ✅ |
| Human-in-the-loop | ❌ | ✅ |
| Checkpointing / time-travel | ❌ | ✅ |
| Subgraphs (nested workflows) | ❌ | ✅ |

Our research agent needs LangGraph because:
1. The agent retrieves context, then **decides** whether to use a tool or answer directly.
2. After using a tool, the agent **loops back** to decide if more tools are needed.
3. State (messages, context, tools used) must **persist** across the entire workflow.
4. We need a **conditional exit** — the loop must stop when the LLM decides no more tools are needed.

## Mental Model: Graphs

A LangGraph workflow is a **directed graph** with three components:

```text
┌───────────────────────────────────────────────────────┐
│                   StateGraph                          │
│                                                       │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│   │  Node A │───▶│  Node B │───▶│  Node C │          │
│   │(function)│    │(function)│    │(function)│          │
│   └─────────┘    └────┬────┘    └─────────┘          │
│                       │                               │
│                  conditional                          │
│                   edge (router)                       │
│                       │                               │
│                  ┌────▼────┐                          │
│                  │  Node D │                          │
│                  │(function)│─────── loop back ──────▶│
│                  └─────────┘           to Node B      │
│                                                       │
│   State = TypedDict flowing through all nodes         │
└───────────────────────────────────────────────────────┘
```

- **Nodes** = Python functions that process and update state
- **Edges** = Connections between nodes (static or conditional)
- **State** = A shared TypedDict that every node reads from and writes to

## Core Concept 1: State (Deep Dive)

State is the **single most important concept** in LangGraph. It's a `TypedDict` that flows through every node. Each node receives the full state, does work, and returns a **partial update** that gets merged back.

### Basic State

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
```

When a node returns `{"context": "some text"}`, LangGraph merges it into the state — `question` and `answer` remain unchanged, only `context` is updated.

### State with Reducers (Critical for Messages)

The problem: when two different nodes both return `{"messages": [...]}`, the default behavior is to **replace** the value. But for chat messages, we want to **append** (accumulate the conversation).

LangGraph solves this with **reducers** — functions that define how a field is updated:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # add_messages APPENDS new messages instead of replacing
    messages: Annotated[list, add_messages]
    
    # These use default behavior (replace on update)
    context: str
    tools_used: list[str]
```

### How `add_messages` Works

```python
# Initial state
state = {"messages": [HumanMessage(content="Hi")]}

# Node returns new messages
node_output = {"messages": [AIMessage(content="Hello!")]}

# With add_messages reducer: APPENDS
# Result: messages = [HumanMessage("Hi"), AIMessage("Hello!")]

# Without reducer (default): REPLACES
# Result: messages = [AIMessage("Hello!")]  ← HumanMessage lost!
```

**This is why `Annotated[list, add_messages]` is critical for our agent** — without it, tool results would overwrite the original question, and the conversation history would be lost.

### Custom Reducers

You can define custom reducers for any field:

```python
import operator
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]    # Built-in: append messages
    tools_used: Annotated[list, operator.add]  # Concatenate lists
    step_count: Annotated[int, operator.add]   # Sum integers
    context: str                                # Default: replace
```

With `operator.add` on `tools_used`, returning `{"tools_used": ["search"]}` will **append** `"search"` to the existing list instead of replacing it.

### State Field Access in Nodes

```python
def my_node(state: AgentState) -> dict:
    # READ from state
    messages = state["messages"]
    context = state.get("context", "")  # .get() for optional fields
    
    # DO work
    result = process(messages, context)
    
    # WRITE partial update (only the fields you changed)
    return {"context": result, "tools_used": ["search"]}
    # Other fields (messages, etc.) remain unchanged
```

## Core Concept 2: Nodes

Nodes are **Python functions** that receive the current state and return state updates. They are the "workers" of your graph.

### Node Rules

1. **Input**: Always receives the full `state: AgentState` dictionary
2. **Output**: Returns a `dict` with **only the fields that changed** (partial update)
3. **Side effects**: Nodes can call APIs, databases, LLMs — anything
4. **Pure functions preferred**: Given the same state, a node should produce the same output (helps with debugging and checkpointing)

### Node Examples

```python
# Simple node — transforms state
def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks for the question."""
    question = state["messages"][-1].content
    context = search_vector_store(question)
    return {"context": context}

# LLM node — calls the model
def agent_node(state: AgentState) -> dict:
    """Call the LLM with context and tools."""
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)
    system = SystemMessage(content=f"Context:\n{state['context']}")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}  # add_messages will APPEND this

# Tool execution node
def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        tool = next(t for t in TOOLS if t.name == tc["name"])
        result = tool.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}  # Appended to message history
```

### Adding Nodes to the Graph

```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)

# Each node gets a string name and a function
graph.add_node("retrieve", retrieve_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
```

## Core Concept 3: Edges

Edges define the **flow** between nodes. There are three types:

### Static Edges (Always follow this path)

```python
# After "retrieve", ALWAYS go to "agent"
graph.add_edge("retrieve", "agent")

# After "tools", ALWAYS go back to "agent" (loop)
graph.add_edge("tools", "agent")
```

### Entry Point (Where the graph starts)

```python
# The first node to execute
graph.set_entry_point("retrieve")

# Alternative: use START constant
from langgraph.graph import START
graph.add_edge(START, "retrieve")  # Equivalent
```

### Conditional Edges (Runtime routing)

The most powerful feature — a function inspects the state and decides which node to visit next:

```python
from typing import Literal

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route based on whether the LLM wants to call tools."""
    last_message = state["messages"][-1]
    
    # If the AI message contains tool_calls → go to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise → end the graph
    return "__end__"

# Add the conditional edge from "agent" node
graph.add_conditional_edges("agent", should_continue)
```

**Key points about conditional edges**:
- The routing function receives the full state
- It returns a **string** — the name of the next node, or `"__end__"` to finish
- Use `Literal` type hints to declare all possible return values
- `"__end__"` is the special string that terminates the graph (equivalent to `END`)

### `END` vs `"__end__"`

```python
from langgraph.graph import END

# These are equivalent:
graph.add_edge("final_node", END)
# The router returning "__end__" also ends the graph

# In conditional edges, return the string "__end__"
def router(state) -> Literal["next_node", "__end__"]:
    return "__end__"  # Ends the graph
```

## Core Concept 4: Building and Compiling the Graph

### Complete Graph Construction

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# 1. Add all nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# 2. Set entry point
graph.set_entry_point("retrieve")

# 3. Add edges
graph.add_edge("retrieve", "agent")         # retrieve → agent (always)
graph.add_conditional_edges("agent", should_continue)  # agent → tools OR __end__
graph.add_edge("tools", "agent")            # tools → agent (loop back)

# 4. Compile into a runnable
app = graph.compile()
```

### Compiling

`graph.compile()` converts the graph definition into a **Runnable** — the same interface as LangChain chains. This means you can call:

```python
# Invoke the entire graph
result = app.invoke({
    "messages": [HumanMessage(content="What is RAG?")],
    "context": "",
    "tools_used": [],
})

# Result is the final state after the graph completes
print(result["messages"][-1].content)  # The agent's final answer
```

### Visualizing the Graph

LangGraph can generate a visual diagram of your graph:

```python
# Print as ASCII
print(app.get_graph().draw_ascii())

# Generate Mermaid diagram (paste into mermaid.live to visualize)
print(app.get_graph().draw_mermaid())

# Save as PNG (requires graphviz installed)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

## Core Concept 5: Prebuilt Components

LangGraph provides several prebuilt components so you don't have to build everything from scratch:

### `create_react_agent` (Quickest way to build an agent)

For simple use cases, LangGraph provides a prebuilt **ReAct agent** (Reasoning + Acting):

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculate(expr: str) -> str:
    """Calculate a math expression."""
    return str(eval(expr, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Creates a complete agent with tool-calling loop
agent = create_react_agent(model=llm, tools=[search, calculate])

# The agent automatically:
# 1. Sends the question to the LLM
# 2. If LLM returns tool_calls → executes tools
# 3. Sends tool results back to LLM
# 4. Repeats until LLM responds with text (no tool calls)
result = agent.invoke({"messages": [("human", "What is 25 * 4?")]})
print(result["messages"][-1].content)
```

**When to use `create_react_agent`**: Simple agents where you just need an LLM + tools loop.
**When to build a custom graph**: When you need custom nodes (like our `retrieve_node`), custom state fields, or non-standard routing logic.

### `ToolNode` (Prebuilt tool executor)

Instead of writing your own tool execution node, use the built-in one:

```python
from langgraph.prebuilt import ToolNode

# Automatically executes any tool_calls in the last AI message
tool_node = ToolNode(tools=[search, calculate])

# Use in your graph
graph.add_node("tools", tool_node)
```

### `tools_condition` (Prebuilt routing function)

Instead of writing your own `should_continue`, use the built-in one:

```python
from langgraph.prebuilt import tools_condition

# Automatically routes to "tools" if tool_calls exist, else "__end__"
graph.add_conditional_edges("agent", tools_condition)
```

### Our Project: Custom vs Prebuilt

In our project (Milestone 8), we build a **custom graph** instead of using `create_react_agent` because:
1. We need a `retrieve_node` that runs **before** the agent (to fetch document context)
2. We need custom state fields (`context`, `tools_used`) beyond just `messages`
3. We want to understand how the internals work (it's a learning project!)

But the `create_react_agent` approach is shown in §10.3 as a reference pattern.

## Core Concept 6: Checkpointing and Persistence

LangGraph can save graph state at every step, enabling:
- **Conversation memory** across API requests
- **Time-travel debugging** (replay from any step)
- **Error recovery** (resume from the last successful step)

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpoint (for development)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Each invocation with a thread_id maintains separate conversation state
config = {"configurable": {"thread_id": "user-123"}}

# First message
result1 = app.invoke(
    {"messages": [HumanMessage(content="What is RAG?")]},
    config=config,
)

# Second message — the agent remembers the first!
result2 = app.invoke(
    {"messages": [HumanMessage(content="Can you elaborate on that?")]},
    config=config,
)
# The agent has access to both messages in state["messages"]
```

**Production checkpointers**: For deployment, use `langgraph-checkpoint-postgres` or `langgraph-checkpoint-sqlite` instead of `MemorySaver`.

## Core Concept 7: Recursion Limit

Since LangGraph supports loops, there's a risk of infinite loops (e.g., the LLM keeps calling tools forever). The **recursion limit** is your safety net:

```python
# Default recursion limit is 25 steps
result = app.invoke(
    {"messages": [HumanMessage(content="...")]},
    config={"recursion_limit": 10},  # Max 10 node executions
)
```

If the graph exceeds the limit, it raises `GraphRecursionError`. In our agent, the loop is: `agent → tools → agent → tools → ...` — each iteration counts as 2 steps, so a limit of 10 allows 5 tool-use rounds.

## Core Concept 8: Streaming from Graphs

LangGraph supports streaming at two levels:

### Stream graph events (node-by-node)

```python
# Stream each node's output as it completes
for event in app.stream({"messages": [HumanMessage(content="What is RAG?")]}):
    for node_name, node_output in event.items():
        print(f"--- {node_name} ---")
        print(node_output)
```

### Stream LLM tokens within nodes

```python
# Stream individual tokens from the LLM
async for event in app.astream_events(
    {"messages": [HumanMessage(content="What is RAG?")]},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

## Core Concept 9: Graph API vs Functional API

LangGraph provides two main ways to define applications: the **Graph API** and the **Functional API**. The choice depends on the complexity of your workflow and your preferred programming style.

- **Graph API (`StateGraph`)**: This approach uses explicit nodes and edges to build a visual graph structure. It is highly structured and best for complex, multi-step workflows where understanding the state machine flow is critical.
- **Functional API (`@entrypoint` / `@task`)**: A more Pythonic, decorator-based approach. It feels like writing standard Python functions and is best for simpler workflows or when you don't need explicit graph visualization.

### Comparison

| Feature | Graph API (`StateGraph`) | Functional API (`@entrypoint`) |
|---|---|---|
| **Style** | State machine, explicit nodes/edges | Decorator-based, standard functions |
| **Complexity** | High (handles complex cycles/routing well) | Low-to-Medium (simpler flows) |
| **Visualization** | Excellent (clear graph structure) | Minimal (function call tree) |
| **State Management** | Centralized `State` with reducers | Passed via standard function arguments/returns |

Our project uses the **Graph API** (starting in Milestone 8), as building autonomous agents often requires complex routing, cycles, and explicit state management that the Graph API excels at handling.

```python
# Functional API example
from langgraph.func import entrypoint, task

@task
def analyze(text: str) -> str:
    """Analyze the input text."""
    return f"Analysis of: {text}"

@task  
def summarize(analysis: str) -> str:
    """Summarize the analysis."""
    return f"Summary: {analysis}"

@entrypoint()
def workflow(text: str) -> str:
    """Define the main workflow."""
    # `.result()` waits for the task to finish
    analysis = analyze(text).result()
    summary = summarize(analysis).result()
    return summary

# Execute the workflow
result = workflow.invoke("Some research text")
print(result)
```

## Core Concept 10: Workflows vs Agents

When building AI applications, you generally operate on a spectrum between strict workflows and autonomous agents.

- **Workflows**: Deterministic, predefined execution paths. The nodes and edges are fixed at design time. You know exactly what steps will happen and in what order. This is highly reliable but inflexible.
- **Agents**: Autonomous and dynamic. The LLM decides the execution path at runtime, typically by choosing which tool to call next. The system uses a generic router that inspects the LLM's output to determine the next node.

Most real-world applications exist somewhere in the middle as **Hybrid Systems**. Our project is a hybrid: we have a deterministic initial retrieve step to gather context, followed by an autonomous agent loop where the LLM decides how to use its tools to accomplish the task.

### Comparing Implementations

```python
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, add_messages

# ── Pattern A: Deterministic Workflow (Static Sequential Edges) 
class WorkflowState(TypedDict):
    query: str
    cleaned_query: str
    report: str

def clean_input(state: WorkflowState) -> dict:
    return {"cleaned_query": state["query"].strip().lower()}

def generate_report(state: WorkflowState) -> dict:
    return {"report": f"Report for {state['cleaned_query']}"}

workflow = StateGraph(WorkflowState)
workflow.add_node("clean", clean_input)
workflow.add_node("generate", generate_report)
workflow.add_edge(START, "clean")
workflow.add_edge("clean", "generate")  # Static, fixed edge
workflow.add_edge("generate", END)
app_workflow = workflow.compile()

# ── Pattern B: Autonomous Agent (Cyclic Routing Loop) ───────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Final answer.")]}

def router(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

agent = StateGraph(AgentState)
agent.add_node("llm", call_model)
agent.add_edge(START, "llm")
agent.add_conditional_edges("llm", router)  # Dynamic edge
app_agent = agent.compile()
```

## Core Concept 11: Interrupts and Human-in-the-Loop (Deep Dive)

A powerful feature of LangGraph is the ability to pause execution, wait for human input, and resume right where it left off. This is essential for sensitive actions that require approval or for collaborative workflows.

To use interrupts, you **must use a checkpointer** (like `MemorySaver`). LangGraph needs to save the exact state of the graph so it can be restored later.

- `interrupt()` function: Call this inside a node to pause execution and surface data to the user.
- `Command(resume=value)`: When the human responds, use `Command` to provide the input back to the graph and resume execution.
- `interrupt_before` and `interrupt_after`: You can also configure interrupts when calling `.compile()` to pause execution before or after specific nodes run.

```python
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage

def human_review_node(state: AgentState) -> dict:
    """Pause for human review before executing sensitive actions."""
    last_msg = state["messages"][-1]
    
    # Pause execution and send data to the human
    human_input = interrupt({
        "question": "Should I execute this action?",
        "proposed_action": last_msg.content,
    })
    
    # This code runs after the human responds and the graph resumes
    if human_input == "approve":
        return {"messages": [AIMessage(content="Action approved and executed.")]}
    return {"messages": [AIMessage(content="Action rejected by human.")]}

# To resume the graph:
# app.invoke(Command(resume="approve"), config=config)
```

## Core Concept 12: Subgraphs

As your LangGraph application grows, putting all logic into a single graph becomes unmanageable. **Subgraphs** allow you to nest compiled graphs within nodes of a parent graph.

### Use Cases
- **Reusable Workflows**: Create a single RAG subgraph and use it across multiple different agent parent graphs.
- **Separation of Concerns**: Isolate complex logic into its own domain.
- **Different State Schemas**: A subgraph can have a completely different state schema than the parent graph. LangGraph handles mapping the inputs and outputs automatically.

A `compiled_subgraph` can be passed directly as the function to `add_node()`.

```python
from langgraph.graph import StateGraph

# 1. Define a sub-workflow
subgraph = StateGraph(SubState)
subgraph.add_node("step1", step1_fn)
subgraph.add_node("step2", step2_fn)
subgraph.set_entry_point("step1")
subgraph.add_edge("step1", "step2")
compiled_sub = subgraph.compile()

# 2. Use it as a node in the parent graph
parent = StateGraph(ParentState)
parent.add_node("main", main_node)
# Use the compiled subgraph directly as a node
parent.add_node("sub_workflow", compiled_sub)  
parent.add_edge("main", "sub_workflow")
```

## Core Concept 13: Fault Tolerance

When building agents that interact with unreliable external APIs, network failures are inevitable. LangGraph provides built-in fault tolerance via the `RetryPolicy`.

You can configure a node to automatically retry upon failure by passing a `RetryPolicy` from `langgraph.pregel`. You can control the maximum number of attempts, the backoff factor (how long to wait between retries), and even specific retry conditions (e.g., only retry on HTTP 500 errors).

```python
from langgraph.pregel import RetryPolicy

# Define the policy: Max 3 attempts, doubling wait time each try
retry_policy = RetryPolicy(
    max_attempts=3, 
    backoff_factor=2.0
)

# Add a node with the retry policy attached
graph.add_node(
    "call_api",
    api_node,
    retry=retry_policy,
)
```

## Core Concept 14: Streaming Modes (Deep Dive)

LangGraph provides flexible streaming capabilities for building responsive UIs. You can choose exactly what data you want to stream back to the client using different `stream_mode` arguments:

- `stream_mode="values"`: Streams the full state dictionary after every node finishes execution. Useful for simple debugging.
- `stream_mode="updates"`: Streams only the exact state dictionary returned by the node (the "update"). This is the most efficient and standard mode.
- `stream_mode="messages"`: Streams raw LLM tokens as `(message_chunk, metadata)` tuples. This is essential for building real-time ChatGPT-like typing interfaces.
- `stream_mode="custom"`: Allows you to stream completely custom data payloads from inside a node using `StreamWriter`.

You can also pass a list of modes (e.g., `["updates", "messages"]`) to stream multiple types of data simultaneously.

```python
# Stream only state updates (most efficient for tracking node progress)
for event in app.stream(inputs, stream_mode="updates"):
    print(event)

# Stream LLM tokens for real-time UI typing effect
for msg_chunk, metadata in app.stream(inputs, stream_mode="messages"):
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)

# Custom streaming with StreamWriter
from langgraph.types import StreamWriter

def my_node(state: State, writer: StreamWriter) -> dict:
    # Stream custom data back to the client immediately
    writer({"status": "processing..."})
    result = do_work(state)
    writer({"status": "done"})
    return {"result": result}
```

## Core Concept 15: Stores (Cross-thread Memory)

While Checkpointers save state for a specific thread (conversation), **Stores** provide long-term, cross-thread memory. Stores allow an agent to remember facts, user preferences, or past interactions across completely different conversations.

- `InMemoryStore` is the basic implementation for testing.
- Data is organized into namespaces (e.g., by user ID or topic) to keep it structured.
- Common operations include `put()` (save), `get()` (retrieve), and `search()` (find related items).

```python
from langgraph.store.memory import InMemoryStore

# Initialize the store
store = InMemoryStore()

# Save data into a specific namespace
store.put(
    ("users", "alice"),     # Namespace tuple
    "preferences",          # Key
    {"language": "Python", "level": "intermediate"} # Value
)

# Retrieve data later, even in a different conversation thread
item = store.get(("users", "alice"), "preferences")
print(item.value)  # {"language": "Python", "level": "intermediate"}

# Attach the store to the compiled graph alongside the checkpointer
# app = graph.compile(checkpointer=memory, store=store)
```

## Core Concept 16: Time Travel

Because LangGraph uses checkpointers to save the entire state at every step, it unlocks powerful "Time Travel" capabilities. You can step back to any previous state in the execution history.

### Use Cases
- **Debugging**: Inspect exactly what the state looked like before a failure occurred.
- **Forking/A-B Testing**: Go back to a previous checkpoint, modify the state slightly, and resume execution to explore an alternative path without re-running the earlier steps.

```python
# List all checkpoints for a specific conversation thread
for state in app.get_state_history(config):
    print(f"Step: {state.metadata.get('step')}, State: {state.values}")

# Replay or fork execution from a specific checkpoint
# Simply pass the checkpoint_id in the config when invoking
old_config = {
    "configurable": {
        "thread_id": "user-1", 
        "checkpoint_id": "abc123"
    }
}
# Execution resumes from that exact state
result = app.invoke(None, config=old_config)
```

## Core Concept 17: Common Errors & Fixes

When developing LangGraph applications, you will encounter these 4 common errors. Here is the exact code showing how to trigger and resolve each:

### 1. `GRAPH_RECURSION_LIMIT`
Raised when a cyclic graph loops beyond the configured step limit (default: 25).
```python
# Fix A: If intentional long workflow, increase recursion limit
app.invoke(inputs, config={"recursion_limit": 50})

# Fix B: If unintended infinite loop, inspect your conditional edge router:
def safe_router(state: AgentState) -> Literal["tools", "__end__"]:
    # Guard against looping forever by checking loop count or empty tool calls
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls and len(state.get("tools_used", [])) < 5:
        return "tools"
    return "__end__"  # Ensure termination
```

### 2. `INVALID_CHAT_HISTORY`
Raised when the sequence of messages violates provider requirements (e.g., OpenAI requires a `ToolMessage` to follow an `AIMessage` with `tool_calls`).
```python
from langchain_core.messages import AIMessage, ToolMessage

# Correct message sequence:
ai_msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "python"}, "id": "call_123"}])
# MUST match tool_call_id!
tool_msg = ToolMessage(content="Python 3.12 release notes", tool_call_id="call_123")
valid_history = [ai_msg, tool_msg]
```

### 3. `INVALID_GRAPH_NODE_RETURN_VALUE`
Raised when a node returns a primitive (string, list, None) instead of a dictionary partial update.
```python
# ❌ ERROR: Nodes cannot return raw strings
# def bad_node(state: AgentState):
#     return "Here is retrieved context"

# ✅ FIX: Always return a dict matching state keys
def good_node(state: AgentState) -> dict:
    return {"context": "Here is retrieved context"}
```

### 4. `MISSING_CHECKPOINTER`
Raised when using `interrupt()` or conversational memory without compiling the graph with a checkpointer.
```python
from langgraph.checkpoint.memory import MemorySaver

# ❌ Raises error if interrupt() is invoked:
# app = graph.compile()

# ✅ FIX: Compile with checkpointer
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

## Core Concept 18: Application Structure

For production LangGraph applications, modular file separation ensures scalability and testability:

```python
# ── File 1: app/agent/state.py ─────────────────────────────────
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    tools_used: list[str]

# ── File 2: app/agent/nodes.py ─────────────────────────────────
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

def retrieve_node(state: AgentState) -> dict:
    return {"context": "Retrieved document text..."}

def agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")
    return {"messages": [llm.invoke(state["messages"])]}

# ── File 3: app/agent/graph.py ─────────────────────────────────
from langgraph.graph import StateGraph, START, END

def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_edge("agent", END)
    return graph.compile()
```

## Core Concept 19: Testing LangGraph Applications

Testing graph-based applications requires a layered approach, from individual functions to the compiled workflow.

- **Unit Testing Nodes**: Because nodes are standard Python functions that take and return dictionaries, they are extremely easy to unit test in isolation.
- **Testing the Graph**: You can invoke the compiled graph with known inputs and assert that the final output matches expectations.
- **Mocking LLMs**: Use libraries like `unittest.mock` to intercept LLM calls and return deterministic responses, ensuring tests run quickly and reliably.
- **Structure Verification**: Assert that the graph structure itself (the nodes and edges) contains exactly what you expect.

```python
from langchain_core.messages import HumanMessage

def test_retrieve_node():
    """Test the retrieve node in isolation."""
    # Mock input state
    state = {
        "messages": [HumanMessage(content="What is RAG?")], 
        "context": "", 
        "tools_used": []
    }
    
    # Run just the node function
    result = retrieve_node(state)
    
    # Verify the partial state update
    assert "context" in result
    assert len(result["context"]) > 0

def test_graph_structure():
    """Verify the graph has the expected nodes."""
    agent = create_agent()
    graph = agent.get_graph()
    
    node_names = [n.name for n in graph.nodes]
    assert "retrieve" in node_names
    assert "agent" in node_names
    assert "use_tool" in node_names
```

## Core Concept 20: Parallel Node Execution (Fan-Out & Fan-In / Map-Reduce)

In sequential chains, operations run one after another. In LangGraph, nodes can run **concurrently in parallel**, cutting overall latency dramatically when gathering data from multiple sources or evaluating multiple hypotheses.

### 1. Fan-Out (Parallel Dispatch)
A node branches into multiple downstream nodes simultaneously:

```text
               ┌──────────────┐
               │    Start     │
               └──────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  ┌───────────┐ ┌───────────┐ ┌───────────┐
  │ SearchWeb │ │ SearchDB  │ │ FetchDocs │ (Concurrent execution)
  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
               ┌──────────────┐
               │  Synthesize  │ (Fan-in / Barrier Synchronization)
               └──────────────┘
```

### 2. State Reducers for Parallel Writes
When multiple nodes execute concurrently, they both write to the state at the same time. **You must use reducers** (like `operator.add`) so updates append instead of conflicting:

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]  # Reducer concatenates results

def search_web(state: ResearchState) -> dict:
    return {"findings": [f"Web finding on {state['topic']}"]}

def search_db(state: ResearchState) -> dict:
    return {"findings": [f"DB finding on {state['topic']}"]}

def synthesize(state: ResearchState) -> dict:
    # Receives combined findings from both web and db
    all_findings = " | ".join(state["findings"])
    return {"findings": [f"Final Report: {all_findings}"]}

graph = StateGraph(ResearchState)
graph.add_node("web", search_web)
graph.add_node("db", search_db)
graph.add_node("synth", synthesize)

# Fan-out: entry point triggers both web and db in parallel
graph.set_entry_point("web")
graph.add_edge("web", "synth")

# Adding multiple edges from START triggers parallel branches
from langgraph.graph import START
graph.add_edge(START, "web")
graph.add_edge(START, "db")

# Fan-in: both must finish before synthesize runs
graph.add_edge(["web", "db"], "synth")
graph.add_edge("synth", END)
```

## Core Concept 21: Multi-Agent Architectures (Deep Dive)

When building complex agentic systems, dividing labor across specialized agents outperforms a single massive monolithic prompt. LangGraph natively supports the 3 canonical multi-agent architectures:

### 1. Supervisor / Router Pattern
A central supervisor LLM reviews the conversation and selects which worker agent to call next:

```text
                ┌──────────────────┐
                │ Supervisor Agent │ ◄──────┐
                └────────┬─────────┘        │
          Routes to:     │                  │
      ┌──────────────────┼────────────────┐ │ Reports back
      ▼                  ▼                ▼ │
┌───────────┐      ┌───────────┐    ┌───────────┐
│ WebWorker │      │ SQLWorker │    │ MathWorker│
└───────────┘      └───────────┘    └───────────┘
```

### 2. Network / Swarm Pattern (Peer Handoffs)
There is no central coordinator. Agents talk directly to each other and explicitly hand off control using `Command(goto="next_agent", update={...})`:

```python
from langgraph.types import Command

def triage_agent(state: AgentState) -> Command[Literal["billing_agent", "tech_support"]]:
    """Triage customer and hand off directly to specialized agent."""
    query = state["messages"][-1].content
    if "invoice" in query.lower() or "pay" in query.lower():
        return Command(
            goto="billing_agent",
            update={"messages": [("ai", "Transferring you to Billing Specialist...")]}
        )
    return Command(
        goto="tech_support",
        update={"messages": [("ai", "Transferring to Technical Support...")]}
    )
```

### 3. Hierarchical Teams
Each sub-system is encapsulated as an autonomous **subgraph** with its own internal state machine, exposing only clean high-level results to the top-level parent supervisor.

| Architecture | Best When | Complexity | Coordination |
|---|---|---|---|
| **Supervisor** | Tasks have clear delegation steps and need centralized tracking | Medium | Centralized |
| **Network (Swarm)** | Conversational flows with specialized domain handoffs | Low-Medium | Decentralized |
| **Hierarchical** | Large enterprise apps with isolated security/domain boundaries | High | Multi-level |

## Core Concept 22: State Inspection & Dynamic Editing (`get_state` & `update_state`)

LangGraph persistence is not a black box. You can inspect the live state of any thread, read historical checkpoints, and programmatically alter state during execution.

### Inspecting State
```python
config = {"configurable": {"thread_id": "session-42"}}

# Get the most recent state snapshot
snapshot = app.get_state(config)
print("Current values:", snapshot.values)
print("Next nodes scheduled to run:", snapshot.next)
print("Checkpoint metadata:", snapshot.metadata)
```

### Programmatic State Injection with `update_state`
Inject human corrections, override previous agent errors, or reset variables without restarting the workflow:

```python
# 1. Update state directly on a thread
app.update_state(
    config,
    {"context": "Corrected factual context provided by human reviewer."},
)

# 2. Update state pretending a specific node produced it
# (Useful for re-routing or testing recovery paths)
app.update_state(
    config,
    {"tools_used": ["manual_override"]},
    as_node="agent",
)

# Resume execution with modified state
result = app.invoke(None, config=config)
```

## Core Concept 23: Complete Memory Model: Short-Term vs Long-Term vs Semantic

| Memory Dimension | Scope | Storage Backend in LangGraph | Primary Purpose | Example |
|---|---|---|---|---|
| **Short-Term Memory** | Single Thread / Session | `BaseCheckpointSaver` (`MemorySaver`, `PostgresSaver`, `SqliteSaver`) | Preserving multi-turn chat history, intermediate tool calls, and step snapshots | "What did I ask you two messages ago?" |
| **Long-Term Memory** | Cross-Thread / Across Users & Sessions | `BaseStore` (`InMemoryStore`, Redis, Database stores) | Retaining user profiles, preferences, past project facts across completely new conversations | "Remember that I prefer Python and use PostgreSQL." |
| **Semantic / Procedural Memory** | Global Knowledge Base | Vector Stores + RAG (`pgvector`, Chroma, Pinecone) | Grounding the agent in external enterprise manuals, research papers, or documentation | "Search research papers for transformer architecture details." |

## Our Agent's Graph (Full Picture)

Here is how our AI research agent's stateful graph operates in Milestone 8:

```text
User sends question
       │
       ▼
┌─────────────┐
│  retrieve   │  Embed question → search pgvector DB → format top-K chunks
└──────┬──────┘
       │ (always advances)
       ▼
┌─────────────┐
│    agent    │  LLM receives: System prompt + retrieved context + messages + tool schemas
└──────┬──────┘
       │
   ┌───▼───┐
   │router │  Inspect last AIMessage: Does it contain tool_calls?
   └───┬───┘
       │
    ┌──┴──────────────┐
    │                 │
    ▼ YES             ▼ NO
┌────────┐      ┌──────────┐
│ tools  │      │ __end__  │  → Return final synthesized answer with citations
└───┬────┘      └──────────┘
    │
    │ (loops back with ToolMessage results)
    ▼
┌─────────────┐
│    agent    │  LLM inspects tool output → decides: need more tools? or ready to answer?
└──────┬──────┘
       │
   ┌───▼───┐
   │router │  ... loop continues until model produces final text (no tool calls)
   └───────┘
```

## Common LangGraph Gotchas

1. **`"__end__"` not `END` in routers**: Conditional edge functions return the **string** `"__end__"`, not the `END` constant. The `END` constant is only used with `graph.add_edge()`.
2. **Missing `add_messages` reducer**: Without `Annotated[list, add_messages]`, each node's `{"messages": [...]}` return **replaces** the entire message list instead of appending. This silently breaks everything.
3. **`ToolMessage` needs `tool_call_id`**: Every `ToolMessage` must have a `tool_call_id` that matches the `id` from the corresponding `tool_call` in the `AIMessage`. Mismatches cause cryptic errors.
4. **Conflicting edges**: Don't add both `graph.add_edge("agent", END)` and `graph.add_conditional_edges("agent", router)` on the same node — the conditional edge handles routing to `__end__` already.
5. **State mutation**: Never mutate state directly (`state["messages"].append(msg)`). Always return a new dict from your node. LangGraph's reducer system handles the merge.
6. **Recursion limit**: The default is 25 steps. For agents that use many tools, you may need to increase it (`config={"recursion_limit": 50}`). If hit unexpectedly, debug routing logic for infinite loops.
7. **Node return type**: Nodes must return a `dict` (partial state update), not the full state or a raw string. Returning the full state object will cause unexpected overwrites.

---

# 7. Theory: Supabase and PostgreSQL

## What is Supabase?

**Supabase** is an open-source Backend-as-a-Service built on **PostgreSQL**. It provides a hosted PostgreSQL database with a REST API, authentication, storage, and real-time subscriptions — all without managing infrastructure.

## Why Supabase for Our Project?

| Feature | Benefit |
|---|---|
| **pgvector extension** | Native vector storage and similarity search directly in PostgreSQL — no separate vector DB needed |
| **Hosted PostgreSQL** | No local database setup; connect via URL + API key |
| **Python client** | `supabase-py` provides a clean, typed API for CRUD operations |
| **Free tier** | 500 MB database, 1 GB file storage — more than enough for development |
| **SQL Editor** | Run raw SQL in the browser for schema setup and debugging |

## Key Concepts

### Tables

Standard PostgreSQL tables store our structured data:

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Vectors with pgvector

The `pgvector` extension adds a `VECTOR(n)` column type for storing embeddings:

```sql
-- Enable the extension (run once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add an embedding column to any table
ALTER TABLE chunks ADD COLUMN embedding VECTOR(384);
```

### Similarity Search

pgvector supports similarity search using distance operators:

```sql
-- Find the 5 most similar chunks to a query embedding
-- <=> is cosine distance (lower = more similar)
SELECT id, content, 1 - (embedding <=> '[0.12, -0.45, ...]') AS similarity
FROM chunks
ORDER BY embedding <=> '[0.12, -0.45, ...]'
LIMIT 5;
```

### RPC Functions

For complex queries, we create a **PostgreSQL function** and call it via Supabase RPC. This keeps the vector search logic on the database side where it's fastest:

```sql
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_count INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
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
```

---

# 8. Reference Code: Core Building Blocks

## 8.1 Sentence Transformers (Embeddings)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single text
embedding = model.encode("machine learning", normalize_embeddings=True)
print(f"Shape: {embedding.shape}")  # (384,)

# Batch encoding
embeddings = model.encode(
    ["machine learning", "deep learning", "chocolate cake"],
    normalize_embeddings=True,
)
print(f"Batch shape: {embeddings.shape}")  # (3, 384)
```

## 8.2 Text Splitting (Chunking)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

text = "Your long document text here..."
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")
```

## 8.3 PDF Parsing

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)
print(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
```

## 8.4 Database Operations (Supabase)

```python
from supabase import create_client, Client

url = "https://your-project.supabase.co"
key = "your-anon-key"
supabase: Client = create_client(url, key)

# Insert a document
result = supabase.table("documents").insert({
    "filename": "paper.pdf",
    "file_type": "pdf",
    "file_size": 102400,
}).execute()

# Query documents
docs = supabase.table("documents").select("*").execute()

# Call the vector search RPC function
matches = supabase.rpc("match_documents", {
    "query_embedding": [0.12, -0.45, ...],  # 384-dim vector
    "match_count": 5,
    "match_threshold": 0.3,
}).execute()
```

## 8.5 Chat Model via OpenRouter (LLM)

```python
import os
from langchain_openai import ChatOpenAI

# Initialize ChatOpenAI pointing to OpenRouter
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
    temperature=0.1,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Research Agent",
    },
)

# Test invocation
response = llm.invoke("Summarize why pgvector is effective for RAG in two sentences.")
print(response.content)
```

---

# 9. Reference Code: LangChain Patterns

> Runnable code snippets covering every LangChain pattern used in this project. Copy, modify, and learn.

> [!TIP]
> **OpenRouter Compatibility**: All patterns below use `ChatOpenAI`. When using OpenRouter, simply supply `api_key=os.getenv("OPENROUTER_API_KEY")` and `base_url="https://openrouter.ai/api/v1"` (or import your initialized `llm` from `app.config`). Because OpenRouter is 100% OpenAI-API compatible, LCEL pipes, streaming, tool binding, and structured output work seamlessly out of the box.

## 9.1 Basic Chain (LCEL)

The simplest possible chain: prompt → LLM → string output.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant."),
    ("human", "{question}"),
])

parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"question": "What is Python?"})
print(result)  # Plain string, not AIMessage
```

## 9.2 RAG Chain with Context Injection

The pattern used in Milestone 6 — retrieve context, inject it into the prompt, and answer:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Simulate a retriever (replace with real vector search)
def fake_retriever(query: str) -> str:
    return f"[Context about '{query}' from documents...]"

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on this context:\n{context}\n\n"
               "If the context doesn't contain the answer, say so."),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

rag_chain = (
    {
        "context": RunnableLambda(fake_retriever),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is attention in transformers?")
print(answer)
```

## 9.3 Tool Calling (Manual Execution)

How tools work under the hood — the LLM decides to call tools, you execute them, send results back:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def search_database(query: str) -> str:
    """Search the research database for relevant papers."""
    return f"Found 3 papers about '{query}': Paper A, Paper B, Paper C"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([search_database, calculate])

# Step 1: Send the question
messages = [HumanMessage(content="How many papers are about transformers?")]
response = llm_with_tools.invoke(messages)

# Step 2: Execute any tool calls
if response.tool_calls:
    messages.append(response)  # Add AIMessage with tool_calls
    for tc in response.tool_calls:
        # Find and execute the tool
        tool_map = {"search_database": search_database, "calculate": calculate}
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Step 3: Send results back for final answer
    final = llm_with_tools.invoke(messages)
    print(final.content)
else:
    print(response.content)
```

## 9.4 Agent with Tools (Prebuilt ReAct)

The easiest way to build a tool-calling agent — LangGraph's `create_react_agent` handles the loop:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(model=llm, tools=[calculate])

result = agent.invoke({"messages": [("human", "What is 25 * 17 + 3?")]})
print(result["messages"][-1].content)
```

## 9.5 Structured Output (Pydantic)

Get the LLM to return typed Python objects instead of free text:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class ResearchAnswer(BaseModel):
    """A research answer with sources."""
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="List of source references")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm = llm.with_structured_output(ResearchAnswer)

result = structured_llm.invoke("What is retrieval-augmented generation?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.sources}")
```

## 9.6 Multi-Step Chain (Compose Chains)

Chain multiple LCEL chains together — each step builds on the previous:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Step 1: Generate an outline
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a brief outline for a research summary."),
    ("human", "Topic: {topic}"),
])
outline_chain = outline_prompt | llm | StrOutputParser()

# Step 2: Write the summary from the outline
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise research summary from this outline:\n{outline}"),
    ("human", "Make it 2-3 paragraphs."),
])
summary_chain = summary_prompt | llm | StrOutputParser()

# Compose: pipe outline into summary
from langchain_core.runnables import RunnableLambda

full_chain = (
    outline_chain
    | RunnableLambda(lambda outline: {"outline": outline})
    | summary_chain
)

result = full_chain.invoke({"topic": "Vector databases for RAG"})
print(result)
```

## 9.7 Streaming Output

Stream tokens as they're generated — essential for responsive UIs:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# Synchronous streaming
for chunk in chain.stream({"question": "Explain how RAG works"}):
    print(chunk, end="", flush=True)
print()  # Newline at end
```

## 9.8 Error Handling with Fallbacks

Use `.with_fallbacks()` to gracefully handle API errors:

```python
from langchain_openai import ChatOpenAI

# Primary model
primary = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Fallback model (different provider or model)
fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=3)

# If primary fails, automatically retry with fallback
robust_llm = primary.with_fallbacks([fallback])

# Use in a chain just like a normal LLM
chain = prompt | robust_llm | StrOutputParser()
```

## 9.9 Message Trimming (Context Engineering)

Keep conversation history within the model's context window:

```python
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, trim_messages,
)

messages = [
    SystemMessage(content="You are a research assistant."),
    HumanMessage(content="What is ML?"),
    AIMessage(content="Machine learning is a branch of AI..."),
    HumanMessage(content="What about deep learning?"),
    AIMessage(content="Deep learning uses neural networks..."),
    HumanMessage(content="Explain transformers."),
    AIMessage(content="Transformers use self-attention..."),
    HumanMessage(content="What is RAG?"),
]

# Keep only recent messages within token budget, but always keep system message
trimmed = trim_messages(
    messages,
    max_tokens=200,
    strategy="last",           # Keep the most recent messages
    token_counter=len,         # Use len() for chars; use tiktoken for real tokens
    allow_partial=False,       # Don't split messages mid-content
    include_system=True,       # Always keep the SystemMessage at the front
)

# Use in a chain — insert trimming before the prompt
from langchain_core.runnables import RunnableLambda

def trim_history(messages):
    return trim_messages(messages, max_tokens=4000, strategy="last",
                         token_counter=len, include_system=True)

chain = RunnableLambda(trim_history) | prompt | llm | StrOutputParser()
```

## 9.10 Event Streaming (Real-time UI)

Stream granular events from any chain or agent for real-time UIs:

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

async def stream_events():
    async for event in chain.astream_events(
        {"question": "What is RAG?"},
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_chain_start":
            print(f"\n[Chain started: {event.get('name', 'unknown')}]")
        elif kind == "on_chain_end":
            print(f"\n[Chain finished]")

asyncio.run(stream_events())
```

## 9.11 Human-in-the-Loop (Interrupt Pattern)

Pause a LangGraph agent for human approval before executing sensitive actions:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: State) -> dict:
    # Agent proposes an action
    return {"messages": [AIMessage(content="I want to delete the database.")]}

def approval_node(state: State) -> dict:
    # Pause for human input
    response = interrupt("Do you approve this action? (yes/no)")
    if response == "yes":
        return {"messages": [AIMessage(content="Action approved. Proceeding.")]}
    return {"messages": [AIMessage(content="Action cancelled by user.")]}

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("approval", approval_node)
graph.set_entry_point("agent")
graph.add_edge("agent", "approval")

app = graph.compile(checkpointer=MemorySaver())

# First invoke — pauses at interrupt
config = {"configurable": {"thread_id": "1"}}
result = app.invoke({"messages": [HumanMessage(content="Clean up")]}, config=config)

# Resume with human input
result = app.invoke(Command(resume="yes"), config=config)
print(result["messages"][-1].content)
```

## 9.12 Dynamic Model & Parameter Switching with `configurable_fields`

Switch between different LLMs or adjust temperatures dynamically on a per-request basis without rebuilding chains:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical advisor."),
    ("human", "{query}"),
])

# Base model with configurable fields
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).configurable_fields(
    model_name=ConfigurableField(
        id="llm_model",
        name="Model Name",
        description="Override model (e.g. gpt-4o, gpt-4o-mini)",
    ),
    temperature=ConfigurableField(
        id="llm_temperature",
        name="Temperature",
        description="Sampling temperature",
    ),
)

chain = prompt | llm | StrOutputParser()

# Standard run (uses gpt-4o-mini, temp 0.7)
res_default = chain.invoke({"query": "Explain quantum computing briefly."})

# Dynamically route difficult queries to gpt-4o with zero temperature
res_complex = chain.invoke(
    {"query": "Derive the mathematical formula for backpropagation."},
    config={
        "configurable": {
            "llm_model": "gpt-4o",
            "llm_temperature": 0.0,
        }
    },
)
```

## 9.13 Hybrid Retrieval (BM25 + Dense Vector Search)

Combine keyword-based search (BM25) for acronyms/exact codes with semantic vector search for concepts using reciprocal rank fusion:

```python
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import SupabaseVectorStore

sample_docs = [
    Document(page_content="Error code 0x80070005 indicates access denied in Windows authentication."),
    Document(page_content="Permission denied errors occur when the caller lacks read/write privileges."),
    Document(page_content="PostgreSQL authentication methods include md5, scram-sha-256, and trust."),
]

# 1. Sparse keyword retriever (excels at exact codes like '0x80070005')
bm25_retriever = BM25Retriever.from_documents(sample_docs)
bm25_retriever.k = 2

# 2. Dense vector retriever (excels at conceptual meaning like 'login failure')
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Combine with 60% semantic, 40% keyword weighting
# ensemble = EnsembleRetriever(
#     retrievers=[vector_retriever, bm25_retriever],
#     weights=[0.6, 0.4],
# )
# results = ensemble.invoke("Why do I get access denied 0x80070005?")
```

---

# 10. Reference Code: LangGraph Patterns

> Runnable graph examples from simple to complex. These map directly to the patterns used in Milestone 8.

## 10.1 Simple Linear Graph

The most basic graph — one node, no branching:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    output: str

def process(state: State) -> dict:
    return {"output": state["input"].upper()}

graph = StateGraph(State)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"input": "hello world", "output": ""})
print(result["output"])  # "HELLO WORLD"
```

## 10.2 Conditional Routing

Route to different nodes based on runtime state:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    needs_tool: bool
    result: str

def classify(state: State) -> dict:
    return {"needs_tool": "calculate" in state["query"].lower()}

def use_tool(state: State) -> dict:
    return {"result": "Tool result: 42"}

def direct_answer(state: State) -> dict:
    return {"result": "Direct answer: Hello!"}

def router(state: State) -> Literal["tool", "direct"]:
    if state["needs_tool"]:
        return "tool"
    return "direct"

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("tool", use_tool)
graph.add_node("direct", direct_answer)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", router)
graph.add_edge("tool", END)
graph.add_edge("direct", END)

app = graph.compile()

# Test both paths
print(app.invoke({"query": "calculate 2+2", "needs_tool": False, "result": ""}))
print(app.invoke({"query": "say hello", "needs_tool": False, "result": ""}))
```

## 10.3 Tool-Calling Agent Loop (Manual — Core Pattern)

This is the **exact pattern** used in our project's Milestone 8. The agent loops: LLM → check for tool calls → execute tools → LLM again:

```python
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages


# ── State ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # CRITICAL: append, don't replace


# ── Tools ──────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22°C, partly cloudy"

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"Population of {city}: ~14 million"

TOOLS = [get_weather, get_population]


# ── Nodes ──────────────────────────────────────────────
def call_model(state: AgentState) -> dict:
    """Call the LLM with tools available."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
    messages = [SystemMessage(content="You are helpful.")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        tool_fn = next(t for t in TOOLS if t.name == tc["name"])
        result = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


# ── Routing ────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# ── Build Graph ────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # Loop back

app = graph.compile()

# ── Run ────────────────────────────────────────────────
result = app.invoke({
    "messages": [HumanMessage(content="What's the weather and population of Tokyo?")]
})
print(result["messages"][-1].content)
```

## 10.4 Prebuilt ReAct Agent

The quickest way — `create_react_agent` builds the entire loop for you:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22°C, partly cloudy"

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"Population of {city}: ~14 million"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(model=llm, tools=[get_weather, get_population])

result = agent.invoke({"messages": [("human", "What's the weather and population of Tokyo?")]})
print(result["messages"][-1].content)
```

## 10.5 Using Prebuilt ToolNode and tools_condition

Replace custom tool execution and routing with LangGraph's built-ins:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

TOOLS = [search]

def agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools=TOOLS))  # Prebuilt!
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)  # Prebuilt!
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Search for RAG papers")]})
print(result["messages"][-1].content)
```

## 10.6 Graph with Checkpointing (Conversation Memory)

Add memory so the agent remembers previous messages across invocations:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")

# Compile with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Thread ID keeps conversations separate
config = {"configurable": {"thread_id": "user-42"}}

# First message
r1 = app.invoke({"messages": [HumanMessage(content="My name is Alice")]}, config=config)
print(r1["messages"][-1].content)

# Second message — agent remembers!
r2 = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config=config)
print(r2["messages"][-1].content)  # "Your name is Alice"
```

## 10.7 Streaming from a Graph

Stream node outputs as the graph executes:

```python
from langchain_core.messages import HumanMessage

# Stream node-by-node updates
for event in app.stream({"messages": [HumanMessage(content="Hello!")]}):
    for node_name, output in event.items():
        print(f"\n--- Node: {node_name} ---")
        if "messages" in output:
            for msg in output["messages"]:
                print(f"  [{msg.__class__.__name__}] {msg.content[:100]}")
```

## 10.8 Subgraph Composition

Use a compiled graph as a node inside a parent graph:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# ── Child subgraph ─────────────────────────────────────
class SubState(TypedDict):
    data: str

def step_a(state: SubState) -> dict:
    return {"data": state["data"] + " → processed by step_a"}

def step_b(state: SubState) -> dict:
    return {"data": state["data"] + " → processed by step_b"}

sub = StateGraph(SubState)
sub.add_node("a", step_a)
sub.add_node("b", step_b)
sub.set_entry_point("a")
sub.add_edge("a", "b")
sub.add_edge("b", END)
compiled_sub = sub.compile()

# ── Parent graph ───────────────────────────────────────
class ParentState(TypedDict):
    data: str

def prepare(state: ParentState) -> dict:
    return {"data": "input:" + state["data"]}

parent = StateGraph(ParentState)
parent.add_node("prepare", prepare)
parent.add_node("sub_workflow", compiled_sub)  # Subgraph as a node!
parent.set_entry_point("prepare")
parent.add_edge("prepare", "sub_workflow")
parent.add_edge("sub_workflow", END)

app = parent.compile()
result = app.invoke({"data": "hello"})
print(result["data"])
# "input:hello → processed by step_a → processed by step_b"
```

## 10.9 Fault-Tolerant Node with Retry

Add retry policies to nodes that call external APIs:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.pregel import RetryPolicy

class State(TypedDict):
    messages: Annotated[list, add_messages]

def unreliable_api_node(state: State) -> dict:
    """Simulates a node that might fail (e.g., external API call)."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("API temporarily unavailable")
    return {"messages": [("ai", "API call succeeded!")]}

graph = StateGraph(State)
graph.add_node(
    "api_call",
    unreliable_api_node,
    retry=RetryPolicy(
        max_attempts=3,        # Try up to 3 times
        backoff_factor=2.0,    # Exponential backoff: 1s, 2s, 4s
    ),
)
graph.set_entry_point("api_call")
graph.add_edge("api_call", END)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Call API")]})
```

## 10.10 Stream Modes

Compare different stream modes for different use cases:

```python
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="What is RAG?")]}

# MODE 1: "values" — full state after each node
for state_snapshot in app.stream(inputs, stream_mode="values"):
    if "messages" in state_snapshot:
        print(f"Messages count: {len(state_snapshot['messages'])}")

# MODE 2: "updates" — only changes from each node (most efficient)
for update in app.stream(inputs, stream_mode="updates"):
    for node_name, changes in update.items():
        print(f"Node '{node_name}' changed: {list(changes.keys())}")

# MODE 3: "messages" — stream LLM tokens for real-time display
for msg_chunk, metadata in app.stream(inputs, stream_mode="messages"):
    if hasattr(msg_chunk, "content") and msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)

# Multiple modes at once
for mode, data in app.stream(inputs, stream_mode=["updates", "messages"]):
    print(f"[{mode}] {type(data)}")
```

## 10.11 Time Travel (State Forking)

Replay from any checkpoint or fork the conversation:

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Assume `app` is compiled with a checkpointer
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user-1"}}

# Run a conversation
app.invoke({"messages": [HumanMessage(content="What is Python?")]}, config=config)
app.invoke({"messages": [HumanMessage(content="What about Java?")]}, config=config)

# View state history (all checkpoints)
for state in app.get_state_history(config):
    checkpoint_id = state.config["configurable"]["checkpoint_id"]
    msg_count = len(state.values.get("messages", []))
    print(f"Checkpoint: {checkpoint_id}, Messages: {msg_count}")

# Fork from an earlier checkpoint (time travel)
# Get the first checkpoint's ID from history
history = list(app.get_state_history(config))
old_checkpoint = history[-1].config  # Earliest state

# Resume from that point with a different question
result = app.invoke(
    {"messages": [HumanMessage(content="What about Rust instead?")]},
    config=old_checkpoint,
)
print(result["messages"][-1].content)
```

## 10.12 Parallel Node Execution (Fan-Out & Fan-In / Map-Reduce)

Run multiple data collection tasks concurrently and synthesize the aggregated results:

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    query: str
    academic_papers: Annotated[list[str], operator.add]
    web_articles: Annotated[list[str], operator.add]
    final_summary: str

def fetch_academic_papers(state: ResearchState) -> dict:
    """Simulate fetching from arXiv/Semantic Scholar."""
    return {"academic_papers": [f"Paper: Attention Is All You Need (for '{state['query']}')"]}

def fetch_web_articles(state: ResearchState) -> dict:
    """Simulate web search."""
    return {"web_articles": [f"Blog: Understanding Transformers (for '{state['query']}')"]}

def synthesize_report(state: ResearchState) -> dict:
    """Fan-in node: combines outputs once all parallel branches finish."""
    sources = state["academic_papers"] + state["web_articles"]
    return {"final_summary": f"Research Summary on '{state['query']}':\n" + "\n".join(f"- {s}" for s in sources)}

graph = StateGraph(ResearchState)
graph.add_node("academic", fetch_academic_papers)
graph.add_node("web", fetch_web_articles)
graph.add_node("synthesize", synthesize_report)

# Fan-out: Both run simultaneously from START
graph.add_edge(START, "academic")
graph.add_edge(START, "web")

# Fan-in: Barrier synchronization — synthesize waits for both to finish
graph.add_edge(["academic", "web"], "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()
output = app.invoke({
    "query": "transformer models",
    "academic_papers": [],
    "web_articles": [],
    "final_summary": "",
})
print(output["final_summary"])
```

## 10.13 Multi-Agent Supervisor Pattern

A central supervisor LLM decides which specialized worker agent to call next, routes queries, and produces a combined answer:

```python
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field

class RouterDecision(BaseModel):
    next_worker: Literal["coder", "researcher", "FINISH"] = Field(
        description="Select the specialized worker to invoke next, or 'FINISH' if complete."
    )

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_node(state: MultiAgentState) -> dict:
    """Supervisor decides which worker to route to next."""
    structured_llm = llm.with_structured_output(RouterDecision)
    system_prompt = SystemMessage(content=(
        "You are a team supervisor managing two workers: 'coder' and 'researcher'. "
        "Given the conversation, decide who should act next. If the user's request is "
        "satisfied, return 'FINISH'."
    ))
    decision = structured_llm.invoke([system_prompt] + state["messages"])
    return {"next_step": decision.next_worker}

def coder_node(state: MultiAgentState) -> dict:
    """Specialized Python coding agent."""
    res = llm.invoke([SystemMessage(content="You are a Python expert.")] + state["messages"])
    return {"messages": [res]}

def researcher_node(state: MultiAgentState) -> dict:
    """Specialized research/literature agent."""
    res = llm.invoke([SystemMessage(content="You are an academic researcher.")] + state["messages"])
    return {"messages": [res]}

def route_next(state: MultiAgentState) -> Literal["coder", "researcher", "__end__"]:
    step = state["next_step"]
    if step == "FINISH":
        return "__end__"
    return step

graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("coder", coder_node)
graph.add_node("researcher", researcher_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_next)
graph.add_edge("coder", "supervisor")       # Report back to supervisor
graph.add_edge("researcher", "supervisor")  # Report back to supervisor

app = graph.compile()
```

## 10.14 State Inspection & Dynamic Editing (`update_state`)

Inspect execution state, override erroneous responses, or inject human feedback into a live thread:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def bot_node(state: ChatState) -> dict:
    return {"messages": [AIMessage(content="I am confident the Earth is flat.")]}

graph = StateGraph(ChatState)
graph.add_node("bot", bot_node)
graph.add_edge(START, "bot")
graph.add_edge("bot", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-101"}}

# 1. Run the agent
app.invoke({"messages": [HumanMessage(content="Tell me about the Earth.")]}, config=config)

# 2. Inspect state
state_snapshot = app.get_state(config)
print("Before correction:", state_snapshot.values["messages"][-1].content)

# 3. Intervene & correct state dynamically as if the bot corrected itself
app.update_state(
    config,
    {"messages": [AIMessage(content="Correction: The Earth is an oblate spheroid.")]},
    as_node="bot",
)

# 4. Verify updated state
corrected_snapshot = app.get_state(config)
print("After correction:", corrected_snapshot.values["messages"][-1].content)
```

---

# Milestone 1: Project Setup and Config

## What We're Doing

Setting up the project structure, virtual environment, dependencies, environment variables, and the central configuration module that every other module imports from.

## Files to Create

- `.env` — API keys and secrets (git-ignored)
- `app/__init__.py` — Makes `app` a Python package
- `app/config.py` — Central configuration
- `app/db/__init__.py` — Database subpackage
- `app/rag/__init__.py` — RAG subpackage
- `app/agent/__init__.py` — Agent subpackage
- `app/api/__init__.py` — API subpackage
- `requirements.txt` — Dependencies

## Reference Code

### `.env`

```dotenv
# OpenRouter LLM Configuration
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
TEMPERATURE=0.1

# Supabase Configuration (Vector Database)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key-here
```

### `app/config.py`

```python
"""Central configuration — all settings in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── API Keys & LLM Endpoint (OpenRouter) ───────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── Embedding Settings (Local — runs offline/free) ────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Chunking Settings ─────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retrieval Settings ─────────────────────────────────
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# ── LLM Settings ──────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
```

### `requirements.txt`

```text
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langgraph>=0.2.0
sentence-transformers>=3.0.0
supabase>=2.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
python-dotenv>=1.0.0
pypdf>=4.0.0
numpy>=1.26.0
pydantic>=2.0.0
python-multipart>=0.0.9
```

## Checklist

- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file with placeholder keys
- [ ] `config.py` loads environment variables
- [ ] Upload directory auto-created
- [ ] All `__init__.py` files in place

---

# Milestone 2: Database Schema

## What We're Doing

Creating Supabase tables, enabling the pgvector extension, and creating the vector search SQL function. Run these SQL statements in the **Supabase SQL Editor**.

## Schema Design

| Table | Purpose | Key Columns |
|---|---|---|
| `documents` | Tracks uploaded files | `id`, `filename`, `file_type`, `file_size`, `chunk_count` |
| `chunks` | Stores text chunks + embeddings | `id`, `document_id` (FK), `chunk_index`, `content`, `embedding` |
| `queries` | Logs user queries + answers | `id`, `question`, `answer`, `sources`, `tools_used`, `latency_ms` |

## Reference Code

### `app/db/schema.sql`

```sql
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
```

### `app/db/client.py`

```python
"""Supabase client singleton."""

from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_KEY

_client: Client | None = None


def get_supabase_client() -> Client:
    """Return a cached Supabase client instance."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
```

### `app/db/operations.py`

```python
"""Database CRUD operations."""

from app.db.client import get_supabase_client


def create_document(
    filename: str, file_type: str, file_size: int, chunk_count: int
) -> dict:
    """Insert a new document record and return it."""
    client = get_supabase_client()
    result = client.table("documents").insert({
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "chunk_count": chunk_count,
    }).execute()
    return result.data[0]


def create_chunks(document_id: str, chunk_records: list[dict]) -> None:
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
    result = client.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "match_threshold": threshold,
    }).execute()
    return result.data


def list_documents() -> list[dict]:
    """Return all documents ordered by creation date."""
    client = get_supabase_client()
    result = (
        client.table("documents")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def delete_document(doc_id: str) -> None:
    """Delete a document and its chunks (cascading)."""
    client = get_supabase_client()
    client.table("documents").delete().eq("id", doc_id).execute()


def save_query(
    question: str,
    answer: str,
    sources: list[dict],
    tools_used: list[str],
    latency_ms: int,
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
    return result.data[0]
```

## Checklist

- [ ] pgvector extension enabled
- [ ] `documents` table created
- [ ] `chunks` table created with `VECTOR(384)` column
- [ ] `queries` table created
- [ ] `match_documents` function created
- [ ] IVFFlat index created on embeddings
- [ ] `client.py` connects to Supabase
- [ ] `operations.py` CRUD functions work

---

# Milestone 3: Embedding Engine

## What We're Doing

Building the embedding module using `sentence-transformers`. This module is responsible for converting text into 384-dimensional vectors and computing similarity scores.

## Reference Code

### `app/rag/embeddings.py`

```python
"""Embedding engine using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL


class EmbeddingEngine:
    """Generates text embeddings using a local sentence-transformer model."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized vector."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed a batch of texts into normalized vectors."""
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
```

## Checklist

- [ ] `EmbeddingEngine` class created
- [ ] `embed_text()` returns a 384-element list for a single string
- [ ] `embed_batch()` handles multiple texts efficiently
- [ ] `similarity()` computes dot product between normalized vectors
- [ ] Model loads successfully on first call (singleton pattern)
- [ ] Output dimension is verified as 384

---

# Milestone 4: Document Ingestion Pipeline

## What We're Doing

Building the end-to-end ingestion pipeline: **Parse → Chunk → Embed → Store**.

```text
Upload File → Parse (PDF/TXT/MD) → Chunk (500 chars) → Embed (384-dim) → Store in Supabase
```

## Reference Code

### `app/rag/ingestion.py`

```python
"""Document ingestion pipeline: parse → chunk → embed → store."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.db.operations import create_chunks, create_document
from app.rag.embeddings import get_embedding_engine


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_file(file_path: str) -> str:
    """Parse a file based on its extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def ingest_document(file_path: str) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → store."""
    path = Path(file_path)

    # 1. Parse
    text = parse_file(file_path)

    # 2. Chunk
    chunks = chunk_text(text)

    # 3. Embed
    engine = get_embedding_engine()
    embeddings = engine.embed_batch(chunks)

    # 4. Store document record
    doc = create_document(
        filename=path.name,
        file_type=path.suffix.lstrip("."),
        file_size=path.stat().st_size,
        chunk_count=len(chunks),
    )

    # 5. Store chunks with embeddings
    chunk_records = [
        {
            "chunk_index": i,
            "content": content,
            "embedding": embedding,
            "metadata": {"source": path.name},
        }
        for i, (content, embedding) in enumerate(zip(chunks, embeddings))
    ]
    create_chunks(doc["id"], chunk_records)

    return doc
```

## Checklist

- [ ] PDF parsing extracts text correctly
- [ ] TXT and MD files read with proper encoding
- [ ] Unsupported file types raise `ValueError`
- [ ] Text chunking respects size and overlap settings
- [ ] Embedding generation works for all chunks
- [ ] Document record stored in `documents` table
- [ ] Chunk records stored in `chunks` table with embeddings

---

# Milestone 5: RAG Retrieval

## What We're Doing

Building the vector search retrieval system that converts a user query into an embedding, searches for similar chunks, and formats the results as context for the LLM.

## Reference Code

### `app/rag/retrieval.py`

```python
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
    engine = get_embedding_engine()
    query_embedding = engine.embed_text(query)
    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        threshold=threshold,
    )
    return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    if not results:
        return "No relevant documents found."

    context_parts = []
    for i, chunk in enumerate(results, 1):
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
```

## Checklist

- [ ] Query is embedded with the same model used for documents
- [ ] Vector search returns ranked results via `match_documents` RPC
- [ ] Context formatted with source labels and similarity scores
- [ ] Empty results handled gracefully
- [ ] `retrieve_and_format()` returns both formatted context and raw data

---

# Milestone 6: LangChain Retrieval Chain

## What We're Doing

Building a LangChain chain that combines retrieved document context with the LLM to produce grounded answers. This is a standalone RAG chain (without the full agent) — useful for simple question-answering.

## Reference Code

### `app/rag/chain.py`

```python
"""LangChain RAG chain — retrieval + LLM answer generation."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
)
from app.rag.retrieval import retrieve_and_format

RAG_SYSTEM_PROMPT = """\
You are a research assistant. Answer the user's question based ONLY on the \
provided context. If the context doesn't contain enough information, say so \
honestly. Always cite your sources using [Source N] notation.

Context:
{context}"""


def create_rag_chain():
    """Create a LangChain RAG chain (prompt → LLM → string output)."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Research Agent",
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def answer_with_rag(question: str) -> dict:
    """Answer a question using the RAG pipeline."""
    context, sources = retrieve_and_format(question)
    chain = create_rag_chain()
    answer = chain.invoke({"context": context, "question": question})
    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }
```

## Checklist

- [ ] RAG chain created with `prompt | llm | parser` (LCEL)
- [ ] System prompt instructs the LLM to answer from context only
- [ ] Answer generated from retrieved context
- [ ] Sources tracked and returned alongside the answer
- [ ] Handles edge cases (no relevant context found)

---

# Milestone 7: Tool Definitions

## What We're Doing

Creating tools that the LangGraph agent can invoke. Each tool has a descriptive docstring (the LLM reads these to decide which tool to use) and a typed signature.

## Reference Code

### `app/agent/tools.py`

```python
"""Agent tools — functions the LLM can invoke during reasoning."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
)
from app.db.operations import list_documents
from app.rag.retrieval import format_context, retrieve_context


@tool
def search_docs(query: str) -> str:
    """Search uploaded documents for content relevant to the query.

    Use this tool when the user asks a question that might be answered
    by the uploaded research documents.
    """
    results = retrieve_context(query, top_k=5)
    if not results:
        return "No relevant documents found."
    return format_context(results)


@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into a concise overview.

    Use this tool when the retrieved context is too long or when
    the user explicitly asks for a summary.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Research Agent",
        },
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text concisely. "
                   "Preserve key facts and findings."),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool for any arithmetic or numerical computation.
    Examples: '25 * 4 + 10', '1024 / 8', 'round(3.14159, 2)'.
    """
    allowed_builtins = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_builtins)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def list_all_documents() -> str:
    """List all uploaded documents with their chunk counts.

    Use this tool when the user wants to know what documents
    are available in the knowledge base.
    """
    docs = list_documents()
    if not docs:
        return "No documents uploaded yet."
    lines = [
        f"- {d['filename']} ({d['chunk_count']} chunks, {d['file_type']})"
        for d in docs
    ]
    return "\n".join(lines)


# All tools the agent can use
TOOLS = [search_docs, summarize_text, calculate, list_all_documents]
```

## Checklist

- [ ] `search_docs` retrieves and formats document chunks
- [ ] `summarize_text` uses LLM to condense long text
- [ ] `calculate` safely evaluates math expressions (restricted `eval`)
- [ ] `list_all_documents` returns formatted document listing
- [ ] All tools have descriptive docstrings for the LLM
- [ ] `TOOLS` list exported for use by the agent

---

# Milestone 8: LangGraph Agent

## What We're Doing

Building the stateful agent workflow with LangGraph. The agent follows a **retrieve → reason → (optionally use tools in a loop) → answer** pattern.

## Architecture

```text
START → retrieve → agent → [should_continue] → use_tool → agent → ... → END
                     ▲                            │
                     └────────────────────────────┘
                          (loop until done)
```

## Reference Code

### `app/agent/graph.py`

```python
"""LangGraph agent — stateful, multi-step research workflow."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph

from app.agent.tools import TOOLS
from app.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
)
from app.rag.retrieval import retrieve_and_format


# ── State Definition ──────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    tools_used: list[str]


# ── Node Functions ────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks for the user's question."""
    last_message = state["messages"][-1]
    question = last_message.content
    context, _sources = retrieve_and_format(question)
    return {"context": context}


def agent_node(state: AgentState) -> dict:
    """Call the LLM with tools bound, passing retrieved context."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Research Agent",
        },
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    context = state.get("context", "No context available.")
    system_msg = SystemMessage(content=(
        "You are a research assistant. Answer questions using the provided "
        "context and tools. Always cite sources when using document context.\n\n"
        f"Retrieved Context:\n{context}"
    ))

    messages = [system_msg] + list(state["messages"])
    response = llm_with_tools.invoke(messages)

    # Track tool usage
    tools_used = list(state.get("tools_used", []))
    if response.tool_calls:
        for tc in response.tool_calls:
            tools_used.append(tc["name"])

    return {"messages": [response], "tools_used": tools_used}


def tool_node(state: AgentState) -> dict:
    """Execute any tool calls from the last LLM response."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        # Find the matching tool by name
        tool_fn = next(t for t in TOOLS if t.name == tool_call["name"])
        result = tool_fn.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


# ── Routing Function ─────────────────────────────────
def should_continue(state: AgentState) -> Literal["use_tool", "__end__"]:
    """Decide whether to execute tools or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "__end__"


# ── Graph Assembly ────────────────────────────────────
def create_agent(checkpointer: MemorySaver | None = None) -> CompiledStateGraph:
    """Build and compile the LangGraph agent with optional checkpoint memory."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("use_tool", tool_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("use_tool", "agent")

    return graph.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────
_memory = MemorySaver()
_agent: CompiledStateGraph | None = None


def get_agent() -> CompiledStateGraph:
    """Return a cached compiled agent with active thread checkpointer."""
    global _agent
    if _agent is None:
        _agent = create_agent(checkpointer=_memory)
    return _agent
```

## Checklist

- [ ] `AgentState` uses `Annotated[list, add_messages]` for proper message accumulation
- [ ] `retrieve_node` embeds the question and fetches relevant chunks
- [ ] `agent_node` calls the LLM with tools bound and context injected
- [ ] `tool_node` executes tool calls and returns `ToolMessage` results
- [ ] `should_continue` routes to `use_tool` or `__end__`
- [ ] Agent loops correctly: `agent → tool → agent → ... → end`
- [ ] `get_agent()` singleton avoids rebuilding the graph per request

---

# Milestone 9: FastAPI Endpoints

## What We're Doing

Building the REST API with FastAPI. The API exposes endpoints for uploading documents, asking questions, listing documents, and deleting documents.

## Reference Code

### `app/api/main.py`

```python
"""FastAPI REST API for the AI Research Agent."""

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.graph import get_agent
from app.config import UPLOAD_DIR
from app.db.operations import delete_document, list_documents, save_query
from app.rag.ingestion import ingest_document

app = FastAPI(
    title="AI Research Agent",
    description="Upload documents and ask research questions powered by RAG + LangGraph.",
    version="1.0.0",
)


# ── Request / Response Models ─────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The research question to answer")
    thread_id: str | None = Field(default=None, description="Optional conversation thread ID for memory")


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    tools_used: list[str]
    latency_ms: int


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    chunk_count: int


# ── Endpoints ─────────────────────────────────────────
@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "AI Research Agent API", "status": "running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, TXT, or MD) for ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed = {".pdf", ".txt", ".md"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    try:
        doc = ingest_document(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        message="Document uploaded and ingested successfully",
        document_id=doc["id"],
        filename=doc["filename"],
        chunk_count=doc["chunk_count"],
    )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a research question. The agent retrieves context, uses tools, and answers."""
    agent = get_agent()
    start = time.time()
    thread_id = request.thread_id or "default"
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({
        "messages": [HumanMessage(content=request.question)],
        "context": "",
        "tools_used": [],
    }, config=config)

    latency_ms = int((time.time() - start) * 1000)
    answer = result["messages"][-1].content
    tools_used = result.get("tools_used", [])

    # Log the query
    save_query(
        question=request.question,
        answer=answer,
        sources=[],
        tools_used=tools_used,
        latency_ms=latency_ms,
    )

    return AnswerResponse(
        answer=answer,
        sources=[],
        tools_used=tools_used,
        latency_ms=latency_ms,
    )


@app.get("/documents")
def get_documents():
    """List all uploaded documents."""
    return {"documents": list_documents()}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    """Delete a document and all its chunks."""
    try:
        delete_document(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"message": "Document deleted successfully"}
```

### Running the Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are auto-generated at `http://localhost:8000/docs` (Swagger UI).

## Checklist

- [ ] `POST /upload` accepts file uploads and ingests them
- [ ] `POST /ask` invokes the agent and returns structured responses
- [ ] `GET /documents` lists all documents
- [ ] `DELETE /documents/{doc_id}` removes a document and its chunks
- [ ] `GET /` health check returns status
- [ ] Input validation (file types, empty questions)
- [ ] Error handling with proper HTTP status codes
- [ ] Query logging to the `queries` table

---

# Milestone 10: Testing and Polish

## What We're Doing

Testing every layer of the application, fixing bugs, adding polish, and ensuring the project is portfolio-ready.

## Test Script

### `tests/test_components.py`

```python
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


def test_parse_txt(tmp_path):
    """Verify TXT file parsing."""
    from app.rag.ingestion import parse_file

    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, world!", encoding="utf-8")
    result = parse_file(str(test_file))
    assert result == "Hello, world!"


def test_parse_unsupported(tmp_path):
    """Verify unsupported file types raise ValueError."""
    from app.rag.ingestion import parse_file

    test_file = tmp_path / "test.xlsx"
    test_file.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(str(test_file))


def test_calculate_tool():
    """Verify the calculate tool returns correct results."""
    from app.agent.tools import calculate

    assert calculate.invoke({"expression": "2 + 2"}) == "4"
    assert calculate.invoke({"expression": "10 * 5 + 3"}) == "53"
    assert "error" in calculate.invoke({"expression": "invalid"}).lower()
```

### Running Tests

```bash
pytest tests/ -v --tb=short
```

## Polish Checklist

- [ ] All functions have docstrings
- [ ] Type hints on all function signatures
- [ ] No bare `print()` statements — use `logging` module if needed
- [ ] Edge cases handled:
  - Empty documents
  - Invalid file types
  - Missing API keys (fail fast with clear error)
  - Empty search results
  - LLM rate limits / API errors
- [ ] Consistent error handling with `HTTPException` in API layer
- [ ] `.gitignore` includes: `venv/`, `.env`, `uploads/`, `__pycache__/`

## Final Project Structure

```text
ai-research-agent/
├── .env                  # API keys (git-ignored)
├── .gitignore
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── config.py         # Central configuration
│   ├── db/
│   │   ├── __init__.py
│   │   ├── client.py     # Supabase client singleton
│   │   ├── operations.py # CRUD operations
│   │   └── schema.sql    # Database schema + functions
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py # Embedding engine
│   │   ├── ingestion.py  # Parse → chunk → embed → store
│   │   ├── retrieval.py  # Vector search + context formatting
│   │   └── chain.py      # LangChain RAG chain
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py      # Agent tool definitions
│   │   └── graph.py      # LangGraph agent workflow
│   └── api/
│       ├── __init__.py
│       └── main.py       # FastAPI endpoints
├── tests/
│   └── test_components.py
├── uploads/              # Uploaded files (git-ignored)
└── venv/                 # Virtual environment (git-ignored)
```

---

# Interview Talking Points

1. **Architecture**: "I built a RAG pipeline with a LangGraph agent that retrieves document context, reasons about which tools to use, and synthesizes answers with citations."
2. **Why LangGraph**: "LangChain chains are linear — my agent needs loops (tool → decide → tool again) and conditional routing, which LangGraph handles natively."
3. **RAG Design**: "I chunk documents at 500 chars with 50-char overlap using recursive splitting, embed with a local sentence-transformer model, and search with pgvector cosine similarity."
4. **Tool Calling**: "The LLM decides when to invoke tools based on the question — it reads tool docstrings and makes structured function calls, not free text."
5. **Database**: "Supabase gives me hosted PostgreSQL with the pgvector extension — vector search runs as an RPC function directly in the database for performance."
6. **State Management**: "LangGraph's `TypedDict` state with `add_messages` annotation ensures messages accumulate correctly as the agent loops through retrieve → reason → tool → reason."
7. **Challenges**: "Getting the agent loop right — conditional edges must route to `__end__` (not `END`) in LangGraph, and tool results need proper `ToolMessage` objects with matching `tool_call_id`."
8. **Improvements**: "I'd add a reranking step (cross-encoder) after vector search, hybrid search (BM25 + vector), evaluation metrics (RAGAS), streaming responses, and a React frontend."
