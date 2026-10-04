"""
FastAPI Chat API for the RAG Chatbot.
Exposes endpoints for querying the Agentic AI eBook knowledge base.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.rag_pipeline import query_rag
from src.core.config import settings

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="A RAG-based chatbot that answers questions strictly from the Agentic AI eBook using LangGraph + Pinecone.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request."""
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask about the Agentic AI eBook.")


class ChunkInfo(BaseModel):
    """Information about a retrieved context chunk."""
    text: str
    page_number: int | str
    source: str
    score: float
    id: str


class ChatResponse(BaseModel):
    """Chat response with answer, context, and confidence."""
    question: str
    answer: str
    retrieved_chunks: list[ChunkInfo]
    relevant_chunks: list[ChunkInfo]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score based on retrieval similarity.")
    is_grounded: bool = Field(..., description="Whether the answer is verified as grounded in the source context.")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Agentic AI RAG Chatbot",
        "model": settings.OPENAI_MODEL,
        "index": settings.PINECONE_INDEX_NAME,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question about the Agentic AI eBook.

    The pipeline:
    1. Retrieves relevant chunks from Pinecone
    2. Grades chunks for relevance
    3. Generates a grounded answer using the LLM
    4. Verifies the answer is not hallucinated

    Returns the answer, source context, and confidence score.
    """
    try:
        result = query_rag(request.question)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@app.get("/health")
def health():
    """Detailed health check including config validation."""
    errors = settings.validate()
    return {
        "healthy": len(errors) == 0,
        "errors": errors,
        "config": {
            "model": settings.OPENAI_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "index": settings.PINECONE_INDEX_NAME,
            "top_k": settings.TOP_K,
        },
    }
