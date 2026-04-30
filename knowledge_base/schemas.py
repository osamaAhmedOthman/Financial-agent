"""
knowledge_base/schemas.py

Pydantic models for the RAG knowledge base.
Every legal article stored in Pinecone maps to one LegalChunk.
"""
from pydantic import BaseModel, Field
from typing import Optional


class LegalChunk(BaseModel):
    """
    Represents a single retrievable unit in the knowledge base.
    One chunk = one legal article, one policy section, or one company rule.

    This is what gets embedded and stored in Pinecone.
    At retrieval time, we return these objects with a similarity score.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    chunk_id: str = Field(
        description="Unique ID for this chunk. Format: '{law_code}_article_{n}'"
    )
    law_name: str = Field(
        description="Human-readable law name, e.g. 'قانون القيمة المضافة 67/2016'"
    )
    law_code: str = Field(
        description="Short machine-readable code, e.g. 'VAT_67_2016'"
    )
    source_file: str = Field(
        description="Original filename this chunk was extracted from"
    )

    # ── Structure ──────────────────────────────────────────────────────────
    chapter: Optional[str] = Field(
        default=None,
        description="Chapter heading, e.g. 'الفصل الأول — تعريفات'"
    )
    article_number: Optional[str] = Field(
        default=None,
        description="Article identifier, e.g. 'المادة 15' or 'Article 15'"
    )

    # ── Content ────────────────────────────────────────────────────────────
    text: str = Field(
        description="Full text of this article or section"
    )
    text_en: Optional[str] = Field(
        default=None,
        description="English translation or summary (optional, for hybrid queries)"
    )

    # ── Metadata for citation ──────────────────────────────────────────────
    namespace: str = Field(
        default="egyptian-tax-law",
        description="Pinecone namespace: 'egyptian-tax-law' or 'company-policies'"
    )
    year: Optional[int] = Field(
        default=None,
        description="Year the law was enacted"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags, e.g. ['vat', 'invoice', 'exemption']"
    )

    def to_pinecone_metadata(self) -> dict:
        """
        Returns a flat dict suitable for Pinecone metadata storage.
        Pinecone metadata cannot contain nested objects.

        FIX: Pinecone enforces a hard 40KB limit per vector's metadata.
        Storing full article text in large PDFs blows this limit instantly.
        We truncate text to 800 chars for metadata storage — enough to show
        a readable preview in the UI and for keyword filtering.
        The FULL text lives in the LegalChunk object in memory (BM25 index)
        and is returned when the agent needs it via the retriever.
        """
        MAX_TEXT_CHARS = 800  # safe ceiling well under the 40KB limit

        return {
            "chunk_id":       self.chunk_id,
            "law_name":       self.law_name,
            "law_code":       self.law_code,
            "source_file":    self.source_file,
            "chapter":        self.chapter or "",
            "article_number": self.article_number or "",
            "text":           self.text[:MAX_TEXT_CHARS],
            "namespace":      self.namespace,
            "year":           self.year or 0,
            "tags":           ",".join(self.tags),
        }

    @classmethod
    def from_pinecone_metadata(cls, metadata: dict) -> "LegalChunk":
        """Reconstructs a LegalChunk from Pinecone metadata dict."""
        return cls(
            chunk_id=metadata["chunk_id"],
            law_name=metadata["law_name"],
            law_code=metadata["law_code"],
            source_file=metadata["source_file"],
            chapter=metadata.get("chapter") or None,
            article_number=metadata.get("article_number") or None,
            text=metadata["text"],
            namespace=metadata.get("namespace", "egyptian-tax-law"),
            year=metadata.get("year") or None,
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
        )


class RetrievalResult(BaseModel):
    """A LegalChunk returned from a search, with its relevance score."""
    chunk: LegalChunk
    score: float = Field(description="Relevance score 0.0–1.0 (higher = more relevant)")
    source: str = Field(description="'vector', 'bm25', or 'hybrid'")
