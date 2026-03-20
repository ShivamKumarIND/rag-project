"""FastAPI entry point for the Document Q&A application."""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_all_llms, get_default_llm_id
from llm_factory import get_llm_instance
from rag_pipeline import (
    build_vector_store,
    generate_answer,
    load_and_chunk_pdf,
    retrieve_context,
)
from vector_store import delete_store, get_store, save_store

app = FastAPI(title="DocTalk", description="Personal Document Q&A with RAG")

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ── Pydantic models ──────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    session_id: str
    llm_id: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    message: str
    session_id: str
    chunk_count: int
    filename: str


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/llms")
def list_llms():
    """Return available LLMs and the default selection."""
    return {
        "llms": get_all_llms(),
        "default": get_default_llm_id(),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload a PDF, chunk it, and build a vector store for the session."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save PDF to uploads/
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Load, chunk, embed
        chunks = load_and_chunk_pdf(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        store = build_vector_store(chunks)
        save_store(session_id, store)

        return UploadResponse(
            message="PDF uploaded and indexed",
            session_id=session_id,
            chunk_count=len(chunks),
            filename=file.filename,
        )
    finally:
        # Clean up uploaded file
        if file_path.exists():
            os.remove(file_path)


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Answer a question using the session's vector store and chosen LLM."""
    store = get_store(req.session_id)
    if store is None:
        raise HTTPException(status_code=404, detail="No document uploaded for this session. Please upload a PDF first.")

    # Retrieve relevant context
    context_docs = retrieve_context(store, req.question, k=4)

    # Get LLM and generate answer
    try:
        llm = get_llm_instance(req.llm_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM: {e}")

    answer = generate_answer(llm, req.question, context_docs)

    # Build sources list
    sources = []
    for doc in context_docs:
        sources.append({
            "page": doc.metadata.get("page", "N/A"),
            "snippet": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
        })

    return AskResponse(answer=answer, sources=sources)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Delete the vector store for a given session."""
    deleted = delete_store(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session cleared"}


# Mount frontend static files (must be LAST so API routes take priority)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
