# 🤖 Agentic AI — RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions strictly from the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf), built with **LangGraph**, **Pinecone**, **OpenAI**, and **Streamlit**.

---

## 📐 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         User Interface                             │
│              Streamlit Chat UI  /  FastAPI REST API                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                    LangGraph RAG Pipeline                           │
│                                                                    │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐ │
│   │ Retrieve │───▶│Grade Relevance│───▶│ Generate │───▶│Halluc.  │ │
│   │          │    │              │    │          │    │Check    │ │
│   └──────────┘    └──────────────┘    └──────────┘    └─────────┘ │
│        │                                    │                      │
│        ▼                                    ▼                      │
│   Pinecone                          OpenAI GPT-4o-mini             │
│   Vector Search                     Grounded Generation            │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Ingestion Pipeline                            │
│                                                                    │
│   PDF ──▶ PyMuPDF ──▶ Text Chunks ──▶ OpenAI Embeddings ──▶ Pinecone │
│           Extract      (1000 chars,     text-embedding-3-small       │
│                         200 overlap)                                │
└────────────────────────────────────────────────────────────────────┘
```

### Pipeline Nodes (LangGraph)

| Node | Purpose |
|------|---------|
| **Retrieve** | Embeds the user query and performs top-K vector search against Pinecone |
| **Grade Relevance** | Uses the LLM to filter out chunks that aren't relevant to the question |
| **Generate** | Produces a grounded answer using only the relevant context chunks |
| **Hallucination Check** | Verifies the generated answer is fully supported by the source context |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://app.pinecone.io/) (free tier works)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example and fill in your API keys
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-actual-key
PINECONE_API_KEY=your-actual-pinecone-key
PINECONE_INDEX_NAME=agentic-ai-ebook
```

### 5. Download the PDF

Place the [Agentic AI eBook PDF](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) in the `data/` folder:

```bash
mkdir data
# Download manually or:
curl -o data/Ebook-Agentic-AI.pdf https://konverge.ai/pdf/Ebook-Agentic-AI.pdf
```

### 6. Run Ingestion (One-Time)

```bash
python ingest.py
```

This extracts text from the PDF, chunks it, generates embeddings, and stores everything in Pinecone. You only need to run this once.

### 7. Launch the Chatbot

**Option A — Streamlit UI (Recommended):**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Option B — FastAPI REST API:**

```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

**Option C — Command Line:**

```bash
python rag_pipeline.py "What is Agentic AI?"
```

---

## 📡 API Usage

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```

**Response:**
```json
{
  "question": "What is Agentic AI?",
  "answer": "Based on the eBook, Agentic AI refers to...",
  "retrieved_chunks": [...],
  "relevant_chunks": [...],
  "confidence": 0.8523,
  "is_grounded": true
}
```

### `GET /health`
Returns service health and configuration status.

---

## 💡 Sample Queries

| # | Question | Expected Coverage |
|---|----------|-------------------|
| 1 | **What is Agentic AI?** | Core definition and overview |
| 2 | **What are the key components of an AI agent?** | Architecture breakdown |
| 3 | **How does Agentic AI differ from traditional AI?** | Comparison and distinctions |
| 4 | **What industries can benefit from Agentic AI?** | Use cases and applications |
| 5 | **What are the challenges in deploying AI agents?** | Implementation hurdles |
| 6 | **What role do LLMs play in Agentic AI?** | LLM integration and importance |

---

## 📁 Project Structure

```
agentic-ai-rag-chatbot/
├── config.py            # Centralized configuration & env loading
├── ingest.py            # PDF → chunks → embeddings → Pinecone
├── rag_pipeline.py      # LangGraph RAG pipeline (4 nodes)
├── api.py               # FastAPI REST API
├── app.py               # Streamlit Chat UI
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
├── data/
│   └── Ebook-Agentic-AI.pdf  # Source PDF (not committed)
└── README.md            # This file
```

---

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `PINECONE_API_KEY` | — | Pinecone API key (required) |
| `PINECONE_INDEX_NAME` | `agentic-ai-ebook` | Pinecone index name |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SCORE_THRESHOLD` | `0.3` | Minimum similarity score |

---

## 🛡️ How Grounding Works

The chatbot ensures answers are strictly based on the PDF through a multi-layer approach:

1. **Retrieval-Only Context** — The LLM only sees text chunks from the PDF, never its training data.
2. **Relevance Grading** — An LLM judge filters out chunks that aren't relevant to the question.
3. **Strict System Prompt** — The generation prompt explicitly instructs the LLM to only use provided context.
4. **Hallucination Detection** — A separate LLM call verifies the answer is fully grounded in the sources.
5. **Transparency** — Every response includes the source chunks and confidence scores for verification.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
