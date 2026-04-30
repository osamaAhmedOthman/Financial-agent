"""
ingestion/
The document ingestion pipeline for the AI Financial Auditor.

Public API:
  process_document_pipeline(file_path) → ExtractedDocument
  ExtractionError                       → raised on unreadable files

Internal modules (import directly if needed):
  extractors  — universal_extractor, extract_text_from_pdf, etc.
  classifiers — classify_document
  validators  — refine_extraction_with_rules, validate_financial_math
  schemas     — ExtractedDocument, LineItem
"""

from .processors import process_document_pipeline, ExtractionError
from .schemas import ExtractedDocument, LineItem

__all__ = [
    "process_document_pipeline",
    "ExtractionError",
    "ExtractedDocument",
    "LineItem",
]
