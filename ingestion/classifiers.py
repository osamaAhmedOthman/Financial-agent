import re
from typing import Literal, Tuple


# Returns (doc_type, confidence_score) instead of just doc_type

def classify_document(
    text: str
) -> Tuple[Literal["invoice", "balance_sheet", "tax_return", "contract", "unknown"], float]:
    """
    Identifies the type of financial document based on keyword pattern matching.
    Returns a tuple of (document_type, confidence_score).

    Confidence is calculated as:
      - matched_keywords / total_keywords_in_category
    This prevents a single lucky keyword from getting full confidence.
    """

    clean_text = " ".join(text.lower().split())
    header_text = clean_text[:500]  # titles usually appear in the first 500 chars

    # Each category has weighted keywords.
    # Strong keywords (more specific) get weight 2, generic ones get weight 1.
    document_patterns = {
        "invoice": {
            "فاتورة ضريبية": 2, "فاتورة": 2, "tax invoice": 2,
            "invoice": 1, "bill to": 1, "amount due": 1,
            "purchase order": 1, "vendor": 1, "itemized": 1,
        },
        "balance_sheet": {
            "ميزانية العمومية": 2, "المركز المالي": 2, "balance sheet": 2,
            "financial position": 2, "current assets": 1, "liabilities": 1,
            "equity": 1, "assets": 1,
        },
        "tax_return": {
            "إقرار ضريبي": 2, "مصلحة الضرائب": 2, "tax return": 2,
            "tax declaration": 2, "vat return": 2, "income tax": 1,
        },
        "contract": {
            "عقد اتفاقية": 2, "طرف أول": 2, "طرف ثاني": 2,
            "contract": 1, "agreement": 1, "terms and conditions": 1,
            "hereby": 1, "signature": 1,
        },
    }

    scores: dict[str, float] = {}

    for doc_type, keywords in document_patterns.items():
        total_weight = sum(keywords.values())
        matched_weight = 0

        for keyword, weight in keywords.items():
            # Priority: header match → full weight. Body match → half weight.
            if keyword in header_text:
                matched_weight += weight
            elif keyword in clean_text:
                matched_weight += weight * 0.5

        if matched_weight > 0:
            scores[doc_type] = matched_weight / total_weight

    if not scores:
        return "unknown", 0.0

    best_type = max(scores, key=scores.get)
    best_score = round(min(scores[best_type], 1.0), 2)  # cap at 1.0

    return best_type, best_score


# ── Unit test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "This is a TAX INVOICE for professional services rendered. Amount due: 5000 EGP.",
        "Balance Sheet — Financial Position as of December 31, 2024. Current Assets...",
        "إقرار ضريبي — مصلحة الضرائب المصرية — السنة الضريبية 2024",
        "Some random document with no financial keywords.",
    ]
    for s in samples:
        doc_type, confidence = classify_document(s)
        print(f"Type: {doc_type:<15} | Confidence: {confidence:.0%} | Text: {s[:50]}...")
