"""FastAPI REST API for the AI Research Agent."""

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

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

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


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
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",)

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

    start = time.time()
    agent = get_agent()
    thread_id = request.thread_id or "default"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

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
