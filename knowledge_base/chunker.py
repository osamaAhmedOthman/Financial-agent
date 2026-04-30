"""
knowledge_base/chunker.py

Splits Egyptian legal PDF documents into article-level chunks.

Why article-level (not token-level)?
  Egyptian tax law articles are self-contained legal units.
  "المادة 15 — ..." should NEVER be split across two chunks.
  A chunk that ends mid-article is legally meaningless.

Strategy:
  1. Extract raw text from the PDF (reusing Phase 1 extractor)
  2. Detect chapter headings  (الفصل / Chapter)
  3. Split on article boundaries (المادة / Article)
  4. Tag each chunk with its chapter, article number, and source law
  5. Return List[LegalChunk] ready for embedding
"""
import re
import os
import uuid
from typing import List, Optional
from .schemas import LegalChunk


# ── Regex patterns for Arabic and English legal structure ─────────────────

# Matches: المادة (15), المادة 15, المادة(15), Article 15, Article (15)
ARTICLE_PATTERN = re.compile(
    r'((?:المادة|مادة)\s*[\(\(]?\s*\d+\s*[\)\)]?'
    r'|Article\s*[\(\(]?\s*\d+\s*[\)\)]?)',
    re.UNICODE
)

# Matches: الفصل الأول, الفصل الثاني, Chapter 1, CHAPTER ONE
CHAPTER_PATTERN = re.compile(
    r'((?:الفصل|الباب)\s+[\u0600-\u06FFa-zA-Z\s]+(?:\n|$)'
    r'|Chapter\s+\w+(?:\n|$)'
    r'|CHAPTER\s+\w+(?:\n|$))',
    re.UNICODE
)

# Extract just the article number digit from a match like "المادة (15)"
ARTICLE_NUMBER_EXTRACT = re.compile(r'\d+')


# ── Seed texts used by the tests and the loader ──────────────────────────

VAT_LAW_SEED = """
الفصل الأول
المادة 3: يجب على كل ممول التسجيل في ضريبة القيمة المضافة خلال المدة القانونية وتحديث بياناته بدقة.

المادة 6: يكون سعر ضريبة القيمة المضافة 14% على السلع والخدمات الخاضعة للضريبة، ما لم ينص القانون على غير ذلك.

المادة 15: يجب أن تتضمن فاتورة ضريبية البيانات الإلزامية ومنها اسم البائع، الرقم الضريبي، تاريخ الإصدار، وقيمة الضريبة. وتُعد الفاتورة الضريبية صحيحة فقط إذا اكتملت هذه البيانات.

المادة 23: تعفى بعض الخدمات والسلع المحددة من الضريبة وفقاً للضوابط الواردة بالقانون.
""".strip()


INCOME_TAX_SEED = """
Chapter One
Article 1: Income tax is imposed on net annual profits according to the provisions of the law.

Article 2: Deductions and allowable expenses must be supported by valid documentation.

Article 5: Taxpayers shall keep accounting records and submit returns within the prescribed deadlines.
""".strip()


UNIFIED_TAX_PROCEDURES_SEED = """
الفصل الأول
المادة 3: يجب أن يكون الرقم الضريبي متكوّناً من تسعة أرقام صحيحة ويُستخدم في جميع المكاتبات الضريبية.

المادة 15: تلتزم الممولين بتقديم الإقرار في المواعيد المحددة وسداد المستحقات الضريبية.

المادة 42: تُطبق الجزاءات على المخالفات الإجرائية وفقاً لأحكام القانون.
""".strip()


def _extract_article_number(header: str) -> str:
    """Pulls the number out of 'المادة (15)' → 'المادة 15'"""
    match = ARTICLE_NUMBER_EXTRACT.search(header)
    if match:
        return f"المادة {match.group()}" if "مادة" in header else f"Article {match.group()}"
    return header.strip()


def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    Falls back to PyMuPDF for scanned documents.
    (Reuses the same logic as Phase 1 but kept local to avoid circular imports)
    """
    import pdfplumber

    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    if not full_text.strip():
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                full_text += page.get_text() + "\n"
        except Exception as e:
            print(f"PyMuPDF fallback failed: {e}")

    return full_text.strip()


def chunk_legal_pdf(
    pdf_path: str,
    law_name: str,
    law_code: str,
    namespace: str = "egyptian-tax-law",
    year: Optional[int] = None,
    tags: Optional[list] = None,
) -> List[LegalChunk]:
    """
    Splits a legal PDF into chunks for embedding.

    Strategy (two-pass):
      Pass 1 — Article-level split on Arabic/English article markers.
               Used for actual law texts (المادة 15, Article 15, etc.)
      Pass 2 — Paragraph-level split (fallback).
               Used when the PDF has no article markers at all —
               e.g. policy briefs, investment guides, research reports.
               Each paragraph of 100+ characters becomes one chunk.

    This means ANY PDF can be loaded, not just formal law documents.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    print(f"Chunking: {filename}")
    raw_text = _extract_text_from_pdf(pdf_path)

    if not raw_text:
        print("  WARNING: No text extracted from PDF")
        return []

    # Pass 1: try article-level split
    chunks = _split_into_articles(
        raw_text=raw_text,
        law_name=law_name,
        law_code=law_code,
        source_file=filename,
        namespace=namespace,
        year=year,
        tags=tags or [],
    )

    # Pass 2: fall back to paragraph chunking if no articles found
    if not chunks:
        print(f"  No article markers found — switching to paragraph chunking.")
        chunks = _split_into_paragraphs(
            raw_text=raw_text,
            law_name=law_name,
            law_code=law_code,
            source_file=filename,
            namespace=namespace,
            year=year,
            tags=tags or [],
        )

    print(f"  Found {len(chunks)} chunks")
    return chunks


def _split_into_paragraphs(
    raw_text: str,
    law_name: str,
    law_code: str,
    source_file: str,
    namespace: str,
    year: Optional[int],
    tags: list,
    min_chars: int = 120,
    max_chars: int = 800,
) -> List[LegalChunk]:
    """
    Fallback chunker for documents with no article markers.
    Splits on blank lines, then groups short paragraphs together
    until they reach max_chars to avoid tiny useless chunks.

    min_chars: ignore paragraphs shorter than this (page numbers, headers)
    max_chars: soft ceiling per chunk (matches Pinecone metadata safe limit)
    """
    # Split on one or more blank lines
    raw_paragraphs = re.split(r'\n\s*\n', raw_text)

    chunks: List[LegalChunk] = []
    buffer = ""
    chunk_index = 0

    for para in raw_paragraphs:
        para = para.strip()

        # Skip noise: very short lines, page numbers, lone digits
        if len(para) < min_chars or re.fullmatch(r'[\d\s\.\-]+', para):
            continue

        # If adding this paragraph would exceed max_chars, flush the buffer first
        if buffer and len(buffer) + len(para) > max_chars:
            chunk_index += 1
            chunks.append(LegalChunk(
                chunk_id=f"{law_code}_para_{chunk_index:03d}",
                law_name=law_name,
                law_code=law_code,
                source_file=source_file,
                chapter=None,
                article_number=None,
                text=buffer.strip(),
                namespace=namespace,
                year=year,
                tags=tags,
            ))
            buffer = para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    # Flush any remaining text
    if buffer and len(buffer) >= min_chars:
        chunk_index += 1
        chunks.append(LegalChunk(
            chunk_id=f"{law_code}_para_{chunk_index:03d}",
            law_name=law_name,
            law_code=law_code,
            source_file=source_file,
            chapter=None,
            article_number=None,
            text=buffer.strip(),
            namespace=namespace,
            year=year,
            tags=tags,
        ))

    return chunks


def chunk_legal_text(
    text: str,
    law_name: str,
    law_code: str,
    source_file: str = "manual_input",
    namespace: str = "egyptian-tax-law",
    year: Optional[int] = None,
    tags: Optional[list] = None,
) -> List[LegalChunk]:
    """
    Same as chunk_legal_pdf but accepts raw text directly.
    Used when you paste law text manually or from a web scrape.
    """
    return _split_into_articles(
        raw_text=text,
        law_name=law_name,
        law_code=law_code,
        source_file=source_file,
        namespace=namespace,
        year=year,
        tags=tags or [],
    )


def _split_into_articles(
    raw_text: str,
    law_name: str,
    law_code: str,
    source_file: str,
    namespace: str,
    year: Optional[int],
    tags: list,
) -> List[LegalChunk]:
    """
    Core splitting logic. Handles both Arabic and English legal documents.

    Algorithm:
      1. Track current chapter heading as we scan
      2. Split text on article boundary markers
      3. Each segment between two article markers = one chunk
      4. Skip segments that are too short (headers, blank pages)
    """
    chunks: List[LegalChunk] = []
    current_chapter = "غير محدد"  # "unspecified"

    # Split on article markers, keeping the delimiters
    parts = ARTICLE_PATTERN.split(raw_text)
    # parts = [pre_text, "المادة 1", article_1_text, "المادة 2", article_2_text, ...]

    i = 0
    article_index = 0

    while i < len(parts):
        part = parts[i].strip()

        # Check if any chapter headings appear in this segment
        chapter_matches = CHAPTER_PATTERN.findall(part)
        if chapter_matches:
            current_chapter = chapter_matches[-1].strip()

        # Check if the NEXT part is an article header
        if i + 1 < len(parts) and ARTICLE_PATTERN.match(parts[i + 1].strip()):
            article_header = parts[i + 1].strip()
            article_body = parts[i + 2].strip() if i + 2 < len(parts) else ""

            # Skip empty or very short articles (likely OCR noise or page numbers)
            full_article_text = f"{article_header}\n{article_body}"
            if len(article_body) < 20:
                i += 2
                continue

            article_index += 1
            chunk_id = f"{law_code}_article_{article_index:03d}"

            chunk = LegalChunk(
                chunk_id=chunk_id,
                law_name=law_name,
                law_code=law_code,
                source_file=source_file,
                chapter=current_chapter,
                article_number=_extract_article_number(article_header),
                text=full_article_text,
                namespace=namespace,
                year=year,
                tags=tags,
            )
            chunks.append(chunk)
            i += 2  # skip past the header we just consumed
        else:
            i += 1

    return chunks


