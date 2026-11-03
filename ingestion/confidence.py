"""
ingestion/confidence.py

Evidence-based extraction confidence.

The old formula was  classifier_score * 0.5 + (0.85 if any amount found) * 0.5.
That measured the keyword classifier, not the extraction: a clean English invoice
scored 55% (below the review cutoff) just because the classifier matched few keywords.

For invoices, confidence now comes from facts about the extracted fields:

    amounts found (subtotal and total)        0.35  (0.15 if only one of them)
    subtotal + tax matches total              0.30
    tax id is a valid 9-digit number          0.10
    vendor name found                         0.05
    date found                                0.05
    keyword classifier (scaled)               0.15

Financial statements keep the old formula, because their fields do not follow the
subtotal + tax = total rule and the new checks would not apply.
"""
import re


def _is_math_consistent(subtotal: float, tax: float, total: float) -> bool:
    if total <= 0:
        return False
    tolerance = max(1.0, 0.005 * total)
    return abs((subtotal + tax) - total) <= tolerance


def compute_extraction_confidence(
    fields: dict,
    classification_confidence: float,
    doc_category: str = "invoice",
) -> float:
    subtotal = float(fields.get("subtotal") or 0.0)
    tax = float(fields.get("total_tax") or 0.0)
    total = float(fields.get("total_amount") or 0.0)

    if doc_category == "financial_statement":
        has_amounts = total > 0 or subtotal > 0
        return round(
            classification_confidence * 0.5 + (0.85 if has_amounts else 0.0) * 0.5, 3
        )

    score = 0.0

    if total > 0 and subtotal > 0:
        score += 0.35
    elif total > 0 or subtotal > 0:
        score += 0.15

    if _is_math_consistent(subtotal, tax, total):
        score += 0.30

    if re.fullmatch(r"\d{9}", str(fields.get("tax_id") or "")):
        score += 0.10

    if fields.get("vendor_name"):
        score += 0.05

    if fields.get("date"):
        score += 0.05

    score += 0.15 * max(0.0, min(float(classification_confidence), 1.0))

    return round(min(score, 1.0), 3)
