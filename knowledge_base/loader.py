"""
knowledge_base/loader.py

The master script for Phase 2.
Orchestrates: chunk → embed → upsert → build BM25 index.

Run this ONCE to populate Pinecone with Egyptian tax law.
After that, retriever.py handles all queries.

Usage:
    # Load from seed text (when you don't have the actual PDFs yet):
    python -m knowledge_base.loader --mode seed

    # Load from a real PDF:
    python -m knowledge_base.loader --mode pdf --path knowledge_base/raw_docs/vat_law.pdf

    # Check what's in the index:
    python -m knowledge_base.loader --mode stats
"""
import argparse
import os
from typing import List
from .chunker import (
    chunk_legal_pdf,
    chunk_legal_text,
    VAT_LAW_SEED,
    INCOME_TAX_SEED,
    UNIFIED_TAX_PROCEDURES_SEED,
)
from .embedder import upsert_chunks, get_index_stats
from .retriever import build_bm25_index
from .schemas import LegalChunk


def load_seed_data() -> List[LegalChunk]:
    """
    Loads the hardcoded seed articles (the most important VAT/tax articles).
    Use this when you don't have the actual PDF files yet.
    Returns all chunks AND builds the BM25 index in memory.
    """
    print("=" * 55)
    print("Loading Egyptian Tax Law seed data...")
    print("=" * 55)

    all_chunks: List[LegalChunk] = []

    # ── VAT Law 67/2016 ────────────────────────────────────────────────────
    vat_chunks = chunk_legal_text(
        text=VAT_LAW_SEED,
        law_name="قانون القيمة المضافة رقم 67 لسنة 2016",
        law_code="VAT_67_2016",
        source_file="vat_law_67_2016_seed.txt",
        namespace="egyptian-tax-law",
        year=2016,
        tags=["vat", "invoice", "tax-rate", "registration", "exemption"],
    )
    print(f"VAT Law: {len(vat_chunks)} chunks")
    all_chunks.extend(vat_chunks)

    # ── Income Tax Law 91/2005 ─────────────────────────────────────────────
    income_chunks = chunk_legal_text(
        text=INCOME_TAX_SEED,
        law_name="قانون الضريبة على الدخل رقم 91 لسنة 2005",
        law_code="INCOME_TAX_91_2005",
        source_file="income_tax_91_2005_seed.txt",
        namespace="egyptian-tax-law",
        year=2005,
        tags=["income-tax", "tax-brackets", "deductions", "expenses"],
    )
    print(f"Income Tax Law: {len(income_chunks)} chunks")
    all_chunks.extend(income_chunks)

    # ── Unified Tax Procedures 206/2020 ────────────────────────────────────
    procedures_chunks = chunk_legal_text(
        text=UNIFIED_TAX_PROCEDURES_SEED,
        law_name="قانون الإجراءات الضريبية الموحد رقم 206 لسنة 2020",
        law_code="UNIFIED_PROCEDURES_206_2020",
        source_file="unified_procedures_206_2020_seed.txt",
        namespace="egyptian-tax-law",
        year=2020,
        tags=["registration", "tax-id", "filing", "audit", "penalties"],
    )
    print(f"Unified Procedures Law: {len(procedures_chunks)} chunks")
    all_chunks.extend(procedures_chunks)

    print(f"\nTotal: {len(all_chunks)} chunks ready for embedding")

    # Build BM25 index in memory for hybrid search
    build_bm25_index(all_chunks)

    return all_chunks


def load_pdf(
    pdf_path: str,
    law_name: str,
    law_code: str,
    namespace: str = "egyptian-tax-law",
    year: int = None,
    tags: list = None,
) -> List[LegalChunk]:
    """Load a single PDF and add it to the knowledge base."""
    chunks = chunk_legal_pdf(
        pdf_path=pdf_path,
        law_name=law_name,
        law_code=law_code,
        namespace=namespace,
        year=year,
        tags=tags or [],
    )
    if chunks:
        upsert_chunks(chunks)
        build_bm25_index(chunks)
    return chunks


def load_policy_document(
    pdf_path: str,
    policy_name: str,
    company_name: str,
) -> List[LegalChunk]:
    """
    Load a company-specific policy document into the 'company-policies' namespace.
    Used for internal audit rules that supplement Egyptian tax law.
    """
    return load_pdf(
        pdf_path=pdf_path,
        law_name=policy_name,
        law_code=f"POLICY_{company_name.upper().replace(' ', '_')}",
        namespace="company-policies",
        tags=["company-policy", company_name.lower()],
    )


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Egyptian tax law into Pinecone")
    parser.add_argument(
        "--mode",
        choices=["seed", "pdf", "stats", "test"],
        default="seed",
        help="seed=load hardcoded articles, pdf=load a PDF file, stats=show index stats, test=run retrieval test"
    )
    parser.add_argument("--path", help="Path to PDF (required for --mode pdf)")
    parser.add_argument("--law-name", help="Law name (required for --mode pdf)")
    parser.add_argument("--law-code", help="Law code (required for --mode pdf)")
    parser.add_argument("--no-upsert", action="store_true", help="Chunk only, skip Pinecone upload")
    args = parser.parse_args()

    if args.mode == "seed":
        chunks = load_seed_data()
        if not args.no_upsert:
            upsert_chunks(chunks)
            print("\nSeed data loaded into Pinecone successfully.")
        else:
            print(f"\nChunked {len(chunks)} articles (skipped Pinecone upload).")
            for c in chunks:
                print(f"  {c.law_code} | {c.article_number} | {len(c.text)} chars")

    elif args.mode == "pdf":
        if not args.path:
            print("ERROR: --path required for pdf mode")
        else:
            chunks = load_pdf(
                pdf_path=args.path,
                law_name=args.law_name or os.path.basename(args.path),
                law_code=args.law_code or "UNKNOWN",
            )
            print(f"Loaded {len(chunks)} chunks from {args.path}")

    elif args.mode == "stats":
        stats = get_index_stats()
        print(f"\nPinecone index stats:")
        print(f"  Total vectors: {stats['total_vectors']}")
        for ns, count in stats["namespaces"].items():
            print(f"  Namespace '{ns}': {count} vectors")

    elif args.mode == "test":
        # Quick retrieval test using the 2 uploaded PDFs as policy documents
        from .retriever import retrieve
        chunks = load_seed_data()
        upsert_chunks(chunks)

        test_queries = [
            "ما هو معدل ضريبة القيمة المضافة على الفواتير؟",
            "ما هي البيانات الإلزامية في الفاتورة الضريبية؟",
            "ما هي عقوبة الفاتورة المخالفة؟",
            "ما هي الخدمات المعفاة من الضريبة؟",
            "كيف يتم حساب الضريبة على الدخل؟",
        ]
        print("\n" + "=" * 55)
        print("Retrieval test results:")
        print("=" * 55)
        for q in test_queries:
            results = retrieve(q, top_k=3)
            print(f"\nQuery: {q}")
            for r in results:
                print(f"  [{r.score:.3f}] {r.chunk.law_code} | {r.chunk.article_number}")
