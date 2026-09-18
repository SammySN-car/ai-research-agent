"""LangChain RAG chain — retrieval + LLM answer generation."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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
    """Create a RAG chain that retrieves context and generates an answer."""
    if OPENROUTER_API_KEY is None:
        raise ValueError("OPENROUTER_API_KEY must be configured")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=SecretStr(OPENROUTER_API_KEY),
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


def rag_chain(question: str) -> dict:
    """Answer a question using the RAG pipeline."""
    context, sources = retrieve_and_format(question)
    chain = create_rag_chain()
    answer = chain.invoke({"context": context, "question": question})
    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }
