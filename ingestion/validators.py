import re
from .schemas import ExtractedDocument


def validate_financial_math(doc: ExtractedDocument) -> bool:
    """
    Performs a mathematical consistency check on extracted financial data.
    Checks: sum(line_items) == subtotal, and subtotal + tax == total_amount.

    """
    ROUNDING_TOLERANCE = 0.10  # EGP — accounts for rounding in scanned docs

    # FIX: Don't pass validation if there's nothing to validate
    if not doc.line_items and doc.subtotal == 0.0:
        print("⚠️  Validation skipped — no line items and zero subtotal.")
        return False

    # Check 1: Do line items sum to the stated subtotal?
    if doc.line_items:
        calculated_subtotal = round(sum(item.total_price for item in doc.line_items), 2)
        subtotal_diff = abs(calculated_subtotal - doc.subtotal)

        if subtotal_diff > ROUNDING_TOLERANCE:
            print(
                f"❌ Subtotal mismatch: line items sum to {calculated_subtotal:.2f}, "
                f"but document states {doc.subtotal:.2f} (diff: {subtotal_diff:.2f})"
            )
            return False

    # Check 2: Does subtotal + tax equal the grand total?
    expected_total = round(doc.subtotal + doc.total_tax, 2)
    total_diff = abs(expected_total - doc.total_amount)

    if total_diff > ROUNDING_TOLERANCE:
        print(
            f"❌ Total mismatch: subtotal ({doc.subtotal:.2f}) + tax ({doc.total_tax:.2f}) "
            f"= {expected_total:.2f}, but document states {doc.total_amount:.2f} "
            f"(diff: {total_diff:.2f})"
        )
        return False

    print("✅ Math validation passed.")
    return True


def validate_egyptian_tax_id(tax_id: str) -> bool:
    """
    NEW: Validates that a Tax ID matches the Egyptian Unified Tax Registration format.
    Egyptian tax IDs are exactly 9 digits, no letters.
    Example valid: "123456789"
    """
    if not tax_id:
        return False
    digits_only = re.sub(r"\D", "", tax_id)
    return len(digits_only) == 9


def validate_vat_rate(rate: float) -> bool:
    """
    NEW: Checks that a VAT rate is a legally recognized Egyptian rate.
    Standard rate: 14%. Some goods have 0% or 5%.
    """
    VALID_EGYPTIAN_VAT_RATES = {0.0, 5.0, 14.0}
    return rate in VALID_EGYPTIAN_VAT_RATES


def refine_extraction_with_rules(doc: ExtractedDocument) -> ExtractedDocument:
    """
    Applies business-logic cleanup rules to the extracted document.
    Runs BEFORE the LangGraph agent in Phase 3 so the agent gets clean input.
    """

    # Rule 1: Strip non-digits from Tax ID and validate format
    if doc.tax_id:
        cleaned_id = re.sub(r"\D", "", doc.tax_id)
        if validate_egyptian_tax_id(cleaned_id):
            doc.tax_id = cleaned_id
        else:
            print(
                f"⚠️  Tax ID '{doc.tax_id}' doesn't match Egyptian 9-digit format. "
                f"Keeping raw value for human review."
            )
            # Don't discard it — flag for human review instead
            doc.requires_human_review = True

    # Rule 2: Validate VAT rates on all line items
    for item in doc.line_items:
        if item.tax_rate is not None and not validate_vat_rate(item.tax_rate):
            print(f"⚠️  Unusual VAT rate {item.tax_rate}% on '{item.description}' — flagging.")
            doc.requires_human_review = True

    # Rule 3: Flag low-confidence extractions for human review
    if doc.extraction_confidence < 0.70:
        print(f"⚠️  Low extraction confidence ({doc.extraction_confidence:.0%}) — flagging for review.")
        doc.requires_human_review = True

    # Rule 4: Run math validation and set status
    doc.validation_status = validate_financial_math(doc)

    if not doc.validation_status:
        doc.requires_human_review = True

    return doc
