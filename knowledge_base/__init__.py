"""
knowledge_base/
The RAG knowledge base for the AI Financial Auditor.

Public API:
    load_seed_data()              → List[LegalChunk]  (chunks + builds BM25)
    retrieve(query, namespace)    → List[RetrievalResult]
    upsert_chunks(chunks)         → int
    evaluate_retrieval(cases)     → dict
    LegalChunk                    → schema model
    RetrievalResult               → schema model
"""
from .loader import load_seed_data, load_pdf, load_policy_document
from .retriever import retrieve, retrieve_by_article, evaluate_retrieval, build_bm25_index
from .embedder import upsert_chunks, get_index_stats
from .schemas import LegalChunk, RetrievalResult

__all__ = [
    "load_seed_data",
    "load_pdf",
    "load_policy_document",
    "retrieve",
    "retrieve_by_article",
    "evaluate_retrieval",
    "build_bm25_index",
    "upsert_chunks",
    "get_index_stats",
    "LegalChunk",
    "RetrievalResult",
]
