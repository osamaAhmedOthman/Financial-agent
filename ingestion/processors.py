"""
ingestion/processors.py

Master pipeline for document ingestion.

Steps:
  1. Extract raw text  (OCR / pdfplumber / pandas)
  2. Classify doc type  (keyword classifier)
  3. Extract financial fields  (regex → LLM fallback)  <- NEW
  4. Build ExtractedDocument schema
  5. Refine and validate with business rules
"""
import time
import os
from .extractors import universal_extractor
from .classifiers import classify_document
from .llm_extractor import extract_financial_fields
from .validators import refine_extraction_with_rules
from .schemas import ExtractedDocument


class ExtractionError(Exception):
    """Raised when a file produces no text at all."""
    pass


def process_document_pipeline(file_path: str) -> ExtractedDocument:
    """
    Master pipeline. Returns a fully populated ExtractedDocument.

    
    extract_financial_fields() fills these from the OCR text using
    regex (fast, works on clean PDFs) with LLM fallback (works on
    garbled OCR output from scanned invoices).
    """
    start_time = time.time()
    filename = os.path.basename(file_path)
    print(f"\n{'─'*50}")
    print(f"Processing: {filename}")
    print(f"{'─'*50}")

    # Step 1: Extract raw text
    raw_text = universal_extractor(file_path)

    if not raw_text.strip():
        raise ExtractionError(
            f"Could not extract any text from '{filename}'. "
            "File may be corrupted, password-protected, or unsupported."
        )
    print(f"Extracted {len(raw_text)} characters.")

    # Step 2: Classify document type
    doc_type, classification_confidence = classify_document(raw_text)
    print(f"Classified as: '{doc_type}' (confidence: {classification_confidence:.0%})")

    # Step 3: Extract financial fields (the missing step in the old version)
    fields = extract_financial_fields(raw_text)

    has_amounts = fields["total_amount"] > 0 or fields["subtotal"] > 0
    extraction_confidence = (
        classification_confidence * 0.5 + (0.85 if has_amounts else 0.0) * 0.5
    )

    # Step 4: Build the schema
    extracted_data = ExtractedDocument(
        doc_type=doc_type,
        raw_text=raw_text,
        cleaned_raw_text=fields.get("cleaned_raw_text"),
        subtotal=fields["subtotal"],
        total_tax=fields["total_tax"],
        total_amount=fields["total_amount"],
        vendor_name=fields.get("vendor_name"),
        tax_id=fields.get("tax_id"),
        date=fields.get("date"),
        currency=fields.get("currency", "EGP"),
        extraction_confidence=extraction_confidence,
        doc_category=fields.get("doc_category", "invoice"),
    )

    # Step 5: Validate
    final_doc = refine_extraction_with_rules(extracted_data)

    elapsed = time.time() - start_time
    status = "Valid" if final_doc.validation_status else "Needs review"

    print(f"\nResult      : {status}")
    print(f"Doc type    : {final_doc.doc_type}")
    print(f"Subtotal    : {final_doc.subtotal:,.2f} {final_doc.currency}")
    print(f"Tax         : {final_doc.total_tax:,.2f} {final_doc.currency}")
    print(f"Total       : {final_doc.total_amount:,.2f} {final_doc.currency}")
    print(f"Confidence  : {final_doc.extraction_confidence:.0%}")
    print(f"Time taken  : {elapsed:.2f}s")
    print(f"{'─'*50}\n")

    return final_doc
