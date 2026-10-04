"""
RAG Pipeline using LangGraph
- Defines a stateful graph with Retrieve → Grade → Generate → Hallucination Check nodes
- Ensures answers are strictly grounded in the retrieved PDF context
"""

from typing import TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

from config import settings


# ──────────────────────────────────────────────
# State Definition
# ──────────────────────────────────────────────

class RAGState(TypedDict):
    """State that flows through the RAG graph."""
    question: str
    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    answer: str
    confidence: float
    is_grounded: bool


# ──────────────────────────────────────────────
# Shared Resources (initialized once)
# ──────────────────────────────────────────────

_llm = None
_embeddings = None
_pinecone_index = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    return _llm


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    return _embeddings


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pinecone_index


# ──────────────────────────────────────────────
# Node 1: Retrieve
# ──────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve top-K relevant chunks from Pinecone using the user's question.
    """
    question = state["question"]
    embeddings = get_embeddings()
    index = get_pinecone_index()

    # Embed the question
    query_vector = embeddings.embed_query(question)

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=settings.TOP_K,
        include_metadata=True,
    )

    # Extract chunks with scores
    chunks = []
    for match in results.get("matches", []):
        chunks.append({
            "text": match["metadata"]["text"],
            "page_number": match["metadata"].get("page_number", "N/A"),
            "source": match["metadata"].get("source", "Unknown"),
            "score": round(match["score"], 4),
            "id": match["id"],
        })

    return {"retrieved_chunks": chunks}


# ──────────────────────────────────────────────
# Node 2: Grade Relevance
# ──────────────────────────────────────────────

GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Given a user question and a document chunk,
determine if the chunk contains information relevant to answering the question.
Reply with ONLY 'yes' or 'no'."""),
    ("human", "Question: {question}\n\nDocument chunk:\n{chunk}\n\nIs this chunk relevant?"),
])


def grade_relevance_node(state: RAGState) -> dict:
    """
    Grade each retrieved chunk for relevance to the question.
    Filter out irrelevant chunks.
    """
    question = state["question"]
    retrieved = state["retrieved_chunks"]
    llm = get_llm()
    chain = GRADING_PROMPT | llm | StrOutputParser()

    relevant = []
    for chunk in retrieved:
        # Only grade if the similarity score is above threshold
        if chunk["score"] >= settings.SCORE_THRESHOLD:
            result = chain.invoke({
                "question": question,
                "chunk": chunk["text"],
            })
            if result.strip().lower() == "yes":
                relevant.append(chunk)

    return {"relevant_chunks": relevant}


# ──────────────────────────────────────────────
# Node 3: Generate Answer
# ──────────────────────────────────────────────

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant that answers questions STRICTLY based on the
provided context from the "Agentic AI" eBook. Follow these rules:

1. ONLY use information from the provided context chunks to answer.
2. If the context does not contain enough information to answer, say:
   "I don't have enough information in the eBook to answer this question."
3. Be precise, clear, and cite the relevant page numbers when possible.
4. Do NOT make up or infer information beyond what's in the context.
5. Structure your answer clearly with paragraphs or bullet points when appropriate."""),
    ("human", """Context from the Agentic AI eBook:
---
{context}
---

Question: {question}

Answer (based strictly on the context above):"""),
])


def generate_node(state: RAGState) -> dict:
    """
    Generate an answer using the LLM, grounded in the relevant chunks.
    """
    relevant = state.get("relevant_chunks", [])
    question = state["question"]
    llm = get_llm()

    if not relevant:
        return {
            "answer": "I don't have enough relevant information in the Agentic AI eBook to answer this question. Please try rephrasing or asking about a topic covered in the eBook.",
            "confidence": 0.0,
            "is_grounded": True,
        }

    # Build context string
    context_parts = []
    for i, chunk in enumerate(relevant, 1):
        context_parts.append(
            f"[Chunk {i} | Page {chunk['page_number']} | Score: {chunk['score']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    # Generate
    chain = GENERATION_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Compute confidence as average of relevance scores
    avg_score = sum(c["score"] for c in relevant) / len(relevant)
    confidence = round(min(avg_score, 1.0), 4)

    return {
        "answer": answer,
        "confidence": confidence,
    }


# ──────────────────────────────────────────────
# Node 4: Hallucination Check
# ──────────────────────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hallucination detector. Given a set of source context chunks and
a generated answer, determine if the answer is FULLY grounded in the context.

Reply with ONLY 'grounded' or 'not_grounded'."""),
    ("human", """Context:
{context}

Generated Answer:
{answer}

Is the answer fully grounded in the context?"""),
])


def hallucination_check_node(state: RAGState) -> dict:
    """
    Verify that the generated answer is grounded in the retrieved context.
    """
    relevant = state.get("relevant_chunks", [])
    answer = state.get("answer", "")

    if not relevant or not answer:
        return {"is_grounded": False}

    llm = get_llm()
    context = "\n\n".join(c["text"] for c in relevant)

    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})

    is_grounded = result.strip().lower() == "grounded"

    # If not grounded, add a disclaimer
    if not is_grounded:
        answer = state["answer"] + "\n\n⚠️ *Note: This answer may contain information not fully supported by the eBook. Please verify with the source document.*"
        return {"is_grounded": False, "answer": answer}

    return {"is_grounded": True}


# ──────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Build and compile the LangGraph RAG pipeline.

    Graph Flow:
        retrieve → grade_relevance → generate → hallucination_check → END
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    # Define edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_edge("grade_relevance", "generate")
    graph.add_edge("generate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Main Query Function
# ──────────────────────────────────────────────

def query_rag(question: str) -> dict:
    """
    Run a question through the RAG pipeline.

    Args:
        question: The user's question about the Agentic AI eBook.

    Returns:
        Dict with keys: answer, retrieved_chunks, relevant_chunks, confidence, is_grounded
    """
    graph = build_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "is_grounded": False,
    }

    result = graph.invoke(initial_state)

    return {
        "question": question,
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "relevant_chunks": result["relevant_chunks"],
        "confidence": result["confidence"],
        "is_grounded": result["is_grounded"],
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is Agentic AI?"
    print(f"\n🔍 Question: {q}\n")
    result = query_rag(q)
    print(f"📝 Answer:\n{result['answer']}\n")
    print(f"📊 Confidence: {result['confidence']}")
    print(f"✅ Grounded: {result['is_grounded']}")
    print(f"📄 Relevant chunks: {len(result['relevant_chunks'])}")
