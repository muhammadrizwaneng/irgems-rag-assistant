"""
ingest.py
---------
Build / refresh the ChromaDB vector store for the IRGEMS RAG assistant.

Pipeline:
1. Load cleaned products (JSON) + store text files
2. Split into overlapping chunks
3. Embed with sentence-transformers (all-MiniLM-L6-v2)
4. Persist to chroma_db/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

CLEANED_PRODUCTS = DATA_DIR / "cleaned_products.json"
TEXT_FILES = [
    ("about", DATA_DIR / "about_irgems.txt"),
    ("care", DATA_DIR / "care_guide.txt"),
    ("faq", DATA_DIR / "faq.txt"),
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irgems_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_product_documents(path: Path) -> list[Document]:
    """Turn each cleaned product into a LangChain Document."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_data.py first."
        )

    with path.open(encoding="utf-8") as handle:
        products = json.load(handle)

    documents: list[Document] = []
    for index, product in enumerate(products):
        metadata = {
            "source": "products",
            "doc_type": "product",
            "name": product.get("name", ""),
            "sku": product.get("sku", ""),
            "category": product.get("category", ""),
            "sale_price": product.get("sale_price") or "",
            "regular_price": product.get("regular_price") or "",
            "product_index": index,
        }
        documents.append(
            Document(page_content=product["text"], metadata=metadata)
        )

    print(f"Loaded {len(documents)} product document(s)")
    return documents


def load_text_documents() -> list[Document]:
    """Load about / care / FAQ knowledge base files."""
    documents: list[Document] = []

    for doc_type, path in TEXT_FILES:
        if not path.exists():
            print(f"WARNING: missing {path.name} — skipping")
            continue

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            print(f"WARNING: {path.name} is empty — skipping")
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "doc_type": doc_type,
                    "name": path.stem,
                },
            )
        )
        print(f"Loaded text file: {path.name} ({len(content)} chars)")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk documents for better retrieval.

    Products are usually short enough to stay as one chunk;
    longer policy / FAQ files get split with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Keep a stable chunk id in metadata for debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def build_embeddings() -> HuggingFaceEmbeddings:
    """Create the local MiniLM embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def persist_vectorstore(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> None:
    """Wipe any previous DB and write a fresh Chroma collection."""
    if CHROMA_DIR.exists():
        print(f"Removing old vector store at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing ChromaDB → {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()  # noqa: SLF001 — useful sanity check
    print(f"Ingestion complete. Collection '{COLLECTION_NAME}' has {count} vector(s).")


def main() -> None:
    print("=" * 60)
    print("IRGEMS — Knowledge Base Ingestion")
    print("=" * 60)

    product_docs = load_product_documents(CLEANED_PRODUCTS)
    text_docs = load_text_documents()
    all_docs = product_docs + text_docs

    if not all_docs:
        raise RuntimeError("No documents found to ingest.")

    chunks = split_documents(all_docs)
    embeddings = build_embeddings()
    persist_vectorstore(chunks, embeddings)

    print("Done. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
