"""Agent tools — functions the LLM can invoke during reasoning."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

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
        api_key=(
            SecretStr(OPENROUTER_API_KEY)
            if OPENROUTER_API_KEY is not None
            else None
        ),
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
