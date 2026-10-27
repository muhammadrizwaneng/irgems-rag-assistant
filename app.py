"""
app.py
------
Streamlit chat UI for the IRGEMS RAG Product Assistant.

Flow:
1. Load persisted ChromaDB + MiniLM embeddings
2. Retrieve relevant chunks for each user question
3. Ask a local Ollama LLM to answer using only that context
4. Show the reply + source snippets
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irgems_knowledge"

# Prefer llama3.2:3b; allow override via env var
DEFAULT_OLLAMA_MODEL = os.getenv("IRGEMS_OLLAMA_MODEL", "llama3.2:3b")
FALLBACK_OLLAMA_MODEL = "qwen2.5:3b"

TOP_K = 4
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are the official IRGEMS jewelry store assistant for https://irgems.pk.

Your job is to help customers with:
- product details (rings, bracelet, materials, finishes)
- prices (quote PKR amounts from context)
- care / cleaning / storage advice
- shipping, COD, returns, and store FAQs

Strict rules:
1. Answer ONLY using the retrieved CONTEXT below. If the context is insufficient, say you are not sure and suggest contacting support (WhatsApp +92 3000795296 or email mrizwansar@gmail.com).
2. Never invent products, prices, or policies.
3. IRGEMS sells individual / single pieces — NOT wholesale.
4. Be warm, clear, and concise. Use short paragraphs or bullet points when helpful.
5. Mention prices in PKR when available.
6. If multiple matching products appear, briefly compare them.
"""


# ---------------------------------------------------------------------------
# Cached resources (loaded once per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Connecting to knowledge base…")
def get_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run prepare_data.py then ingest.py first."
        )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


@st.cache_resource(show_spinner="Connecting to Ollama…")
def get_llm(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def retrieve_context(query: str, k: int = TOP_K) -> list[Document]:
    """Return the top-k most relevant chunks for a question."""
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


def format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        name = doc.metadata.get("name", "")
        header = f"[Source {i} | {source}"
        if name:
            header += f" | {name}"
        header += "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"CUSTOMER QUESTION:\n{question}\n\n"
        "Answer the customer helpfully using only the CONTEXT."
    )


def generate_answer(question: str, model_name: str) -> tuple[str, list[Document]]:
    """Retrieve → prompt → generate."""
    docs = retrieve_context(question)
    context = format_context(docs) if docs else "No relevant documents found."

    llm = get_llm(model_name)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, context)),
    ]
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, docs


def source_label(doc: Document) -> str:
    """Friendly label for the sources expander."""
    doc_type = doc.metadata.get("doc_type", "")
    source = doc.metadata.get("source", "unknown")
    name = doc.metadata.get("name", "")

    if doc_type == "product":
        price = doc.metadata.get("sale_price") or doc.metadata.get("regular_price")
        price_bit = f" — PKR {price}" if price else ""
        return f"Product: {name}{price_bit}"
    if source.endswith(".txt"):
        return f"Store info: {source}"
    return str(source)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 820px; padding-top: 1.5rem; }
        .irgems-hero {
            background: linear-gradient(135deg, #1a1a1a 0%, #3d2c1e 55%, #6b4f2a 100%);
            color: #f7f1e8;
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .irgems-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.75rem;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .irgems-hero p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.98rem;
        }
        .stChatMessage { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.title("IRGEMS Settings")
    st.sidebar.caption("Local RAG assistant for irgems.pk")

    model = st.sidebar.selectbox(
        "Ollama model",
        options=[DEFAULT_OLLAMA_MODEL, FALLBACK_OLLAMA_MODEL, "llama3.2", "qwen2.5:3b"],
        index=0,
        help="Pull a model first, e.g. `ollama pull llama3.2:3b`",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Quick tips**
- Ask about prices, materials, or care
- Ask about COD / shipping / returns
- Single pieces only — no wholesale

**Support**
- WhatsApp: +92 3000795296
- Email: mrizwansar@gmail.com
- Site: [irgems.pk](https://irgems.pk)
        """
    )

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return model


def main() -> None:
    st.set_page_config(
        page_title="IRGEMS Assistant",
        page_icon="💎",
        layout="centered",
    )
    inject_styles()

    model_name = render_sidebar()

    st.markdown(
        """
        <div class="irgems-hero">
            <h1>IRGEMS Assistant</h1>
            <p>Ask about our rings, bracelet, prices, materials, care tips, shipping &amp; returns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verify vector store early so the user sees a clear message
    try:
        get_vectorstore()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Setup steps:\n"
            "1. `python prepare_data.py`\n"
            "2. `python ingest.py`\n"
            "3. `ollama pull llama3.2:3b`\n"
            "4. `streamlit run app.py`"
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm the IRGEMS shopping assistant. "
                    "Ask me anything about our jewelry, prices, care instructions, "
                    "or store policies."
                ),
                "sources": [],
            }
        ]

    # Render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources") or []
            if sources and message["role"] == "assistant":
                with st.expander("Sources used"):
                    for doc in sources:
                        st.markdown(f"**{source_label(doc)}**")
                        preview = doc.page_content.strip()
                        if len(preview) > 350:
                            preview = preview[:350].rstrip() + "…"
                        st.caption(preview)

    # Chat input
    question = st.chat_input("Ask about a product, price, care tip, or policy…")
    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching IRGEMS knowledge base…"):
            try:
                answer, docs = generate_answer(question, model_name)
            except Exception as exc:  # noqa: BLE001 — surface Ollama / network errors
                answer = (
                    "Sorry — I couldn't reach the local LLM. "
                    "Make sure Ollama is running and the selected model is pulled.\n\n"
                    f"Details: `{exc}`"
                )
                docs = []

        st.markdown(answer)
        if docs:
            with st.expander("Sources used"):
                for doc in docs:
                    st.markdown(f"**{source_label(doc)}**")
                    preview = doc.page_content.strip()
                    if len(preview) > 350:
                        preview = preview[:350].rstrip() + "…"
                    st.caption(preview)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": docs}
    )


if __name__ == "__main__":
    main()
