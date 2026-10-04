"""
Configuration module for the RAG Chatbot.
Loads environment variables and provides centralized settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))

    # Embedding dimension for text-embedding-3-small
    EMBEDDING_DIMENSION: int = 1536

    # PDF path
    PDF_PATH: str = os.getenv("PDF_PATH", os.path.join("data", "Ebook-Agentic-AI.pdf"))

    @classmethod
    def validate(cls) -> list[str]:
        """Validate that required settings are configured. Returns list of errors."""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")
        return errors


settings = Settings()
