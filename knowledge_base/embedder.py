"""
knowledge_base/embedder.py

Handles two things:
  1. Converting LegalChunk text → dense vectors (embedding)
  2. Storing those vectors in Pinecone with metadata

Why intfloat/multilingual-e5-large?
  - Handles Arabic and English in the SAME vector space
  - 1024 dimensions — good accuracy vs speed tradeoff
  - Open source, runs locally (no API cost per embedding)
  - Top performer on Arabic semantic similarity benchmarks

Pinecone namespace strategy:
  - "egyptian-tax-law"  → VAT law, income tax, unified procedures
  - "company-policies"  → internal rules uploaded by the company
"""
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from .schemas import LegalChunk

load_dotenv()

# ── Lazy-loaded globals (same pattern as Phase 1 OCR reader) ──────────────
_embedding_model = None
_pinecone_index = None


def _get_embedding_model():
    """Load the multilingual embedding model once, cache it."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (first run downloads ~560MB)...")
        _embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        print("Embedding model ready.")
    return _embedding_model


def _get_pinecone_index():
    """Connect to Pinecone and return the index object."""
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone, ServerlessSpec

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "financial-auditor")

        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment. "
                "Add it to your .env file."
            )

        pc = Pinecone(api_key=api_key)

        # Create the index if it doesn't exist yet
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            print(f"Creating Pinecone index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=1024,          # must match multilingual-e5-large output
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until index is ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Connected to existing Pinecone index '{index_name}'.")

        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string into a 1024-dimensional vector.

    Note: multilingual-e5 works best with a prefix:
      - "query: ..."   for search queries
      - "passage: ..." for documents being stored

    We always use "passage: " here because this is for storage.
    At retrieval time (retriever.py), we use "query: ".
    """
    model = _get_embedding_model()
    vector = model.encode(f"passage: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_chunks(chunks: List[LegalChunk]) -> List[tuple]:
    """
    Embed a list of LegalChunks in batch (much faster than one-by-one).
    Returns list of (chunk_id, vector, metadata) tuples ready for Pinecone upsert.
    """
    if not chunks:
        return []

    model = _get_embedding_model()

    # Batch encode all texts at once
    texts = [f"passage: {chunk.text}" for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    return [
        (chunk.chunk_id, vector.tolist(), chunk.to_pinecone_metadata())
        for chunk, vector in zip(chunks, vectors)
    ]


def upsert_chunks(
    chunks: List[LegalChunk],
    namespace: Optional[str] = None,
    batch_size: int = 100,
) -> int:
    """
    Embed chunks and upsert them into Pinecone.

    Args:
        chunks:     List of LegalChunk objects to store
        namespace:  Override namespace (default: use chunk.namespace)
        batch_size: Pinecone upsert batch size (max 100 per request)

    Returns:
        Number of chunks successfully upserted
    """
    if not chunks:
        print("No chunks to upsert.")
        return 0

    index = _get_pinecone_index()
    embedded = embed_chunks(chunks)

    total_upserted = 0

    # Group by namespace
    by_namespace: dict[str, list] = {}
    for chunk, (chunk_id, vector, metadata) in zip(chunks, embedded):
        ns = namespace or chunk.namespace
        by_namespace.setdefault(ns, []).append((chunk_id, vector, metadata))

    for ns, records in by_namespace.items():
        print(f"Upserting {len(records)} chunks to namespace '{ns}'...")

        # Pinecone upsert in batches
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            index.upsert(
                vectors=[
                    {"id": r[0], "values": r[1], "metadata": r[2]}
                    for r in batch
                ],
                namespace=ns,
            )
            total_upserted += len(batch)
            print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"Done. Total chunks in Pinecone: {total_upserted}")
    return total_upserted


def get_index_stats() -> dict:
    """Returns statistics about the Pinecone index (chunk counts per namespace)."""
    index = _get_pinecone_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "namespaces": {
            ns: info.vector_count
            for ns, info in stats.namespaces.items()
        },
    }


def delete_namespace(namespace: str) -> None:
    """
    Deletes all vectors in a namespace.
    Useful for re-loading updated law documents.
    """
    index = _get_pinecone_index()
    index.delete(delete_all=True, namespace=namespace)
    print(f"Deleted all vectors in namespace '{namespace}'")
