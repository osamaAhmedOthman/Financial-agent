"""
knowledge_base/retriever.py

Hybrid retrieval combining:
  1. Vector search  — finds semantically similar articles
  2. BM25 search    — finds exact keyword matches (e.g. "المادة 15")

Why hybrid?
  Pure vector search misses exact article references.
  If an accountant writes "المادة 15 فقرة أ", vector search might return
  semantically similar articles but miss article 15 exactly.
  BM25 catches that. The fusion gives the best of both.

Fusion strategy: Reciprocal Rank Fusion (RRF)
  - Score = 1 / (rank_in_vector_results + k) + 1 / (rank_in_bm25_results + k)
  - k=60 is the standard constant that prevents top ranks from dominating too much
  - Simple, parameter-free, works well in practice
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from .schemas import LegalChunk, RetrievalResult

load_dotenv()

# BM25 index is built at load time from the chunks currently in memory.
# In production, this would be persisted to disk. For Phase 2 it's rebuilt
# each session from the seed data.
_bm25_index = None
_bm25_chunks: List[LegalChunk] = []


def build_bm25_index(chunks: List[LegalChunk]) -> None:
    """
    Build an in-memory BM25 index from a list of LegalChunks.
    Call this after loading seed data or after every Pinecone upsert.

    BM25 tokenizes by whitespace and works on raw text — no embedding needed.
    """
    global _bm25_index, _bm25_chunks
    from rank_bm25 import BM25Okapi

    _bm25_chunks = chunks
    tokenized_corpus = [chunk.text.split() for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built with {len(chunks)} documents.")


def _vector_search(
    query: str,
    namespace: str,
    top_k: int,
) -> List[tuple[LegalChunk, float]]:
    """
    Query Pinecone with a dense vector and return (chunk, score) pairs.
    Uses "query: " prefix as required by multilingual-e5.
    """
    from .embedder import _get_pinecone_index, _get_embedding_model

    model = _get_embedding_model()
    query_vector = model.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    index = _get_pinecone_index()
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    chunks_with_scores = []
    for match in results.matches:
        if match.metadata:
            chunk = LegalChunk.from_pinecone_metadata(match.metadata)
            chunks_with_scores.append((chunk, match.score))

    return chunks_with_scores


def _bm25_search(
    query: str,
    top_k: int,
) -> List[tuple[LegalChunk, float]]:
    """
    BM25 keyword search over the in-memory corpus.
    Returns (chunk, normalized_score) pairs.
    """
    if _bm25_index is None or not _bm25_chunks:
        return []

    query_tokens = query.split()
    scores = _bm25_index.get_scores(query_tokens)

    # Pair each chunk with its BM25 score and sort descending
    scored = sorted(
        zip(_bm25_chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Normalize scores to 0-1 range
    max_score = scored[0][1] if scored and scored[0][1] > 0 else 1.0
    return [(chunk, score / max_score) for chunk, score in scored if score > 0]


def _reciprocal_rank_fusion(
    vector_results: List[tuple[LegalChunk, float]],
    bm25_results: List[tuple[LegalChunk, float]],
    k: int = 60,
) -> List[tuple[LegalChunk, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) for each list where the document appears.
    Documents appearing in both lists get a bonus from both.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, LegalChunk] = {}

    for rank, (chunk, _) in enumerate(vector_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # Normalize fused scores to 0-1
    max_score = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0
    return [
        (chunk_map[cid], rrf_scores[cid] / max_score)
        for cid in sorted_ids
    ]


def retrieve(
    query: str,
    namespace: str = "egyptian-tax-law",
    top_k: int = 5,
    use_hybrid: bool = True,
) -> List[RetrievalResult]:
    """
    Main retrieval function. Called by the LangGraph agent in Phase 3.

    Args:
        query:       The search query (Arabic or English)
        namespace:   Which Pinecone namespace to search
        top_k:       Number of results to return
        use_hybrid:  If True, combine vector + BM25. If False, vector only.

    Returns:
        List[RetrievalResult] sorted by relevance (most relevant first)

    Example:
        results = retrieve("ما هو معدل ضريبة القيمة المضافة على الفواتير؟")
        for r in results:
            print(r.chunk.article_number, r.score)
    """
    print(f"Retrieving for: '{query[:60]}...' in namespace '{namespace}'")

    # Always do vector search
    vector_results = _vector_search(query, namespace=namespace, top_k=top_k * 2)

    if not use_hybrid or _bm25_index is None:
        # Vector-only mode
        return [
            RetrievalResult(chunk=chunk, score=score, source="vector")
            for chunk, score in vector_results[:top_k]
        ]

    # Hybrid: combine vector + BM25
    bm25_results = _bm25_search(query, top_k=top_k * 2)
    fused = _reciprocal_rank_fusion(vector_results, bm25_results)

    return [
        RetrievalResult(
            chunk=chunk,
            score=round(score, 4),
            source="hybrid",
        )
        for chunk, score in fused[:top_k]
    ]


def retrieve_by_article(
    article_number: str,
    law_code: Optional[str] = None,
    namespace: str = "egyptian-tax-law",
) -> Optional[RetrievalResult]:
    """
    Fetch a specific article by number and optionally by law code.
    Used when the agent knows EXACTLY which article it needs.

    Example:
        result = retrieve_by_article("المادة 15", law_code="VAT_67_2016")
    """
    from .embedder import _get_pinecone_index

    index = _get_pinecone_index()

    # Build a metadata filter for Pinecone
    filter_dict: dict = {"article_number": {"$eq": article_number}}
    if law_code:
        filter_dict["law_code"] = {"$eq": law_code}

    # Use the article number itself as the query vector
    from .embedder import _get_embedding_model
    model = _get_embedding_model()
    query_vector = model.encode(
        f"query: {article_number}", normalize_embeddings=True
    ).tolist()

    results = index.query(
        vector=query_vector,
        top_k=1,
        namespace=namespace,
        include_metadata=True,
        filter=filter_dict,
    )

    if results.matches:
        chunk = LegalChunk.from_pinecone_metadata(results.matches[0].metadata)
        return RetrievalResult(chunk=chunk, score=results.matches[0].score, source="direct")

    return None


def evaluate_retrieval(
    test_cases: List[dict],
    namespace: str = "egyptian-tax-law",
    top_k: int = 5,
) -> dict:
    """
    Task 2.6 — Measures retrieval quality against known correct answers.

    Args:
        test_cases: List of {"query": str, "expected_article": str} dicts
        namespace:  Which namespace to query
        top_k:      How many results to check for the expected article

    Returns:
        Dict with recall@k, precision, and per-case results

    Example:
        test_cases = [
            {"query": "ما معدل الضريبة على الفواتير", "expected_article": "المادة 6"},
            {"query": "متى تصدر الفاتورة الضريبية", "expected_article": "المادة 15"},
        ]
        metrics = evaluate_retrieval(test_cases)
        print(metrics["recall_at_5"])  # should be > 0.80
    """
    hits = 0
    results_log = []

    for case in test_cases:
        query = case["query"]
        expected = case["expected_article"]

        retrieved = retrieve(query, namespace=namespace, top_k=top_k)
        retrieved_articles = [r.chunk.article_number for r in retrieved]

        hit = expected in retrieved_articles
        hits += int(hit)

        results_log.append({
            "query": query,
            "expected": expected,
            "retrieved": retrieved_articles,
            "hit": hit,
            "top_result": retrieved_articles[0] if retrieved_articles else None,
            "top_score": retrieved[0].score if retrieved else 0.0,
        })

        status = "✅" if hit else "❌"
        print(f"{status} '{query[:40]}' → expected {expected}, got {retrieved_articles[:3]}")

    recall = hits / len(test_cases) if test_cases else 0.0

    print(f"\nRecall@{top_k}: {recall:.0%} ({hits}/{len(test_cases)})")
    return {
        "recall_at_k": recall,
        "k": top_k,
        "total_cases": len(test_cases),
        "hits": hits,
        "cases": results_log,
    }
