"""
PDF Ingestion Pipeline
- Extracts text from PDF using PyMuPDF (fitz)
- Chunks text using LangChain's RecursiveCharacterTextSplitter
- Generates embeddings via OpenAI
- Upserts vectors into Pinecone
"""

import os
import sys
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from config import settings


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF using PyMuPDF.

    Returns:
        List of dicts with keys: 'text', 'page_number', 'source'
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        # Skip pages with very little text (likely images/covers)
        if text and len(text.strip()) > 50:
            pages.append({
                "text": text.strip(),
                "page_number": page_num + 1,
                "source": os.path.basename(pdf_path),
            })

    doc.close()
    print(f"✅ Extracted text from {len(pages)} pages")
    return pages


def chunk_documents(
    pages: list[dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """
    Split extracted page texts into smaller, overlapping chunks.

    Returns:
        List of dicts with keys: 'text', 'page_number', 'source', 'chunk_index'
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    global_idx = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split_text in splits:
            chunks.append({
                "id": f"chunk-{global_idx}",
                "text": split_text,
                "page_number": page["page_number"],
                "source": page["source"],
                "chunk_index": global_idx,
            })
            global_idx += 1

    print(f"✅ Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Create Pinecone index if it does not exist."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"📦 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Index '{index_name}' created")
    else:
        print(f"ℹ️  Index '{index_name}' already exists")


def upsert_to_pinecone(chunks: list[dict], batch_size: int = 100) -> int:
    """
    Generate embeddings and upsert chunks into Pinecone.

    Returns:
        Number of vectors upserted.
    """
    # Initialize clients
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    # Create index
    create_pinecone_index(pc, settings.PINECONE_INDEX_NAME, settings.EMBEDDING_DIMENSION)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    # Process in batches
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings
        print(f"  ⏳ Embedding batch {i // batch_size + 1}...")
        vectors = embeddings_model.embed_documents(texts)

        # Prepare upsert data
        upsert_data = []
        for chunk, vector in zip(batch, vectors):
            upsert_data.append({
                "id": chunk["id"],
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                },
            })

        # Upsert to Pinecone
        index.upsert(vectors=upsert_data)
        total_upserted += len(upsert_data)
        print(f"  ✅ Upserted {total_upserted}/{len(chunks)} vectors")

    print(f"\n🎉 Ingestion complete! {total_upserted} vectors stored in Pinecone.")
    return total_upserted


def run_ingestion(pdf_path: str = None) -> int:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → Pinecone.

    Returns:
        Number of vectors upserted.
    """
    pdf_path = pdf_path or settings.PDF_PATH

    # Validate config
    errors = settings.validate()
    if errors:
        print("❌ Configuration errors:")
        for err in errors:
            print(f"   - {err}")
        print("\nPlease set the required environment variables in your .env file.")
        sys.exit(1)

    print("=" * 60)
    print("📄 RAG Chatbot — PDF Ingestion Pipeline")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Index: {settings.PINECONE_INDEX_NAME}")
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Chunk size: {settings.CHUNK_SIZE} | Overlap: {settings.CHUNK_OVERLAP}")
    print("=" * 60)

    # Step 1: Extract text
    print("\n📖 Step 1: Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk documents
    print("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(pages)

    # Step 3: Generate embeddings & store in Pinecone
    print("\n🧠 Step 3: Generating embeddings & storing in Pinecone...")
    count = upsert_to_pinecone(chunks)

    return count


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(pdf)
