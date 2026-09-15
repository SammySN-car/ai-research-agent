"""Document ingestion pipeline: parse → chunk → embed → store."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.db.operations import create_chunks, create_document
from app.rag.embeddings import get_embedding_engine


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(file_path)
    pages = [page.extract_text() for page in pdf_reader.pages]
    return "\n".join(pages)


def parse_file(file_path: str) -> str:
    """Parse a file based on its extension."""
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension == ".pdf":
        return parse_pdf(file_path)
    elif extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return text_splitter.split_text(text)


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
