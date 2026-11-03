"""Unit tests for evidence-based extraction confidence (no network, no keys)."""
from ingestion.confidence import compute_extraction_confidence

GOOD = {
    "subtotal": 10000.0, "total_tax": 1400.0, "total_amount": 11400.0,
    "tax_id": "123456789", "vendor_name": "Nile Office Supplies LLC",
    "date": "2026-10-15",
}


def test_clean_invoice_passes_review_cutoffs():
    # Even with a weak keyword classifier (25%), a clean extraction must clear
    # both the validator cutoff (0.70) and the report cutoff (0.60).
    assert compute_extraction_confidence(GOOD, 0.25, "invoice") >= 0.85


def test_failed_extraction_scores_low():
    empty = {"subtotal": 0.0, "total_tax": 0.0, "total_amount": 0.0,
             "tax_id": None, "vendor_name": None, "date": None}
    assert compute_extraction_confidence(empty, 0.25, "invoice") < 0.30


def test_math_mismatch_lowers_confidence():
    bad = dict(GOOD, total_amount=15000.0)
    assert (compute_extraction_confidence(bad, 0.25, "invoice")
            < compute_extraction_confidence(GOOD, 0.25, "invoice") - 0.25)


def test_missing_tax_id_stays_above_review_cutoff():
    no_id = dict(GOOD, tax_id=None)
    assert compute_extraction_confidence(no_id, 0.25, "invoice") >= 0.70


def test_financial_statement_keeps_old_formula():
    fs = {"subtotal": 5000000.0, "total_tax": 100000.0, "total_amount": 400000.0}
    assert compute_extraction_confidence(fs, 0.4, "financial_statement") == round(0.4 * 0.5 + 0.85 * 0.5, 3)
