"""
Streamlit Chat UI for the Agentic AI RAG Chatbot.
A polished, modern chat interface with context display.
"""

import streamlit as st
import time
from rag_pipeline import query_rag
from config import settings


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic AI — RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0;
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    .bot-message {
        background: #f8f9fb;
        color: #1a1a2e;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        border: 1px solid #e8eaf0;
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .confidence-high {
        background: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
    }

    /* Context chunk card */
    .chunk-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s ease;
    }
    .chunk-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .chunk-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8f9fb;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AI — RAG Chatbot</h1>
    <p>Ask anything about the Agentic AI eBook. Answers are strictly grounded in the source document.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**Model:** `{settings.OPENAI_MODEL}`")
    st.markdown(f"**Embeddings:** `{settings.EMBEDDING_MODEL}`")
    st.markdown(f"**Vector DB:** Pinecone (`{settings.PINECONE_INDEX_NAME}`)")
    st.markdown(f"**Top-K:** `{settings.TOP_K}`")

    st.divider()

    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is Agentic AI?",
        "What are the key components of an AI agent?",
        "How does Agentic AI differ from traditional AI?",
        "What industries can benefit from Agentic AI?",
        "What are the challenges in deploying AI agents?",
        "What role do LLMs play in Agentic AI?",
    ]

    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()

    st.markdown("### 📊 Pipeline Info")
    st.markdown("""
    ```
    ┌─────────────┐
    │   Retrieve   │  Vector search (Pinecone)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Grade Chunks │  LLM relevance filter
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Generate   │  Grounded answer
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Halluc.     │  Verification check
    │  Check       │
    └──────┬──────┘
           │
        [Answer]
    ```
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ──────────────────────────────────────────────
# Chat History
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

            # Show metadata if available
            if "metadata" in msg:
                meta = msg["metadata"]

                # Confidence badge
                conf = meta.get("confidence", 0)
                if conf >= 0.7:
                    badge_class = "confidence-high"
                    label = "High"
                elif conf >= 0.4:
                    badge_class = "confidence-medium"
                    label = "Medium"
                else:
                    badge_class = "confidence-low"
                    label = "Low"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="confidence-badge {badge_class}">
                        📊 Confidence: {conf:.1%} ({label})
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    grounded_icon = "✅" if meta.get("is_grounded") else "⚠️"
                    st.markdown(f"{grounded_icon} **Grounded:** {'Yes' if meta.get('is_grounded') else 'No'}")
                with col3:
                    st.markdown(f"📄 **Chunks used:** {meta.get('chunk_count', 0)}")

                # Show context chunks in expander
                if meta.get("relevant_chunks"):
                    with st.expander(f"📚 View {len(meta['relevant_chunks'])} source chunks", expanded=False):
                        for i, chunk in enumerate(meta["relevant_chunks"], 1):
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-meta">
                                    <span>📄 Page {chunk['page_number']}</span>
                                    <span>📊 Score: {chunk['score']:.4f}</span>
                                    <span>🆔 {chunk['id']}</span>
                                </div>
                                <div style="font-size: 0.85rem; color: #374151;">
                                    {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Chat Input
# ──────────────────────────────────────────────

# Handle prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask a question about the Agentic AI eBook...")

if prefill:
    question = prefill

if question:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching knowledge base & generating answer..."):
            start_time = time.time()

            try:
                result = query_rag(question)
                elapsed = time.time() - start_time

                # Display answer
                st.markdown(result["answer"])

                # Metadata display
                conf = result["confidence"]
                if conf >= 0.7:
                    badge_class = "confidence-high"
                    label = "High"
                elif conf >= 0.4:
                    badge_class = "confidence-medium"
                    label = "Medium"
                else:
                    badge_class = "confidence-low"
                    label = "Low"

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="confidence-badge {badge_class}">
                        📊 Confidence: {conf:.1%} ({label})
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    grounded_icon = "✅" if result["is_grounded"] else "⚠️"
                    st.markdown(f"{grounded_icon} **Grounded:** {'Yes' if result['is_grounded'] else 'No'}")
                with col3:
                    st.markdown(f"📄 **Chunks:** {len(result['relevant_chunks'])}")
                with col4:
                    st.markdown(f"⏱️ **Time:** {elapsed:.1f}s")

                # Context chunks
                if result["relevant_chunks"]:
                    with st.expander(f"📚 View {len(result['relevant_chunks'])} source chunks", expanded=False):
                        for i, chunk in enumerate(result["relevant_chunks"], 1):
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-meta">
                                    <span>📄 Page {chunk['page_number']}</span>
                                    <span>📊 Score: {chunk['score']:.4f}</span>
                                    <span>🆔 {chunk['id']}</span>
                                </div>
                                <div style="font-size: 0.85rem; color: #374151;">
                                    {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Store in history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "confidence": result["confidence"],
                        "is_grounded": result["is_grounded"],
                        "chunk_count": len(result["relevant_chunks"]),
                        "relevant_chunks": result["relevant_chunks"],
                        "elapsed": elapsed,
                    },
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                })
