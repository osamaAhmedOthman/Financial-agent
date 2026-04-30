"""
agent/auditor.py

The single public entry point for the agent.
Everything outside the agent/ folder (API, UI, tests) uses only this.

Usage:
    from agent.auditor import run_audit
    from ingestion.processors import process_document_pipeline

    extracted = process_document_pipeline("invoice.pdf")
    report = run_audit(extracted)
    print(report.summary_line())
"""
import os
from dotenv import load_dotenv
from .schemas import AuditReport, AuditState, Violation
from .graph import build_audit_graph

load_dotenv()

# Compile once at import time — reused for every audit call
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        print("Compiling LangGraph audit graph...")
        _graph = build_audit_graph()
        print("Graph ready.")
    return _graph


def _to_float(value) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def _sum_line_items(extracted_doc) -> float | None:
    """Return line items sum when available, otherwise None."""
    line_items = getattr(extracted_doc, "line_items", None) or []
    if not line_items:
        return None

    total = 0.0
    for item in line_items:
        if hasattr(item, "total_price"):
            total += _to_float(getattr(item, "total_price", 0.0))
        elif isinstance(item, dict):
            total += _to_float(item.get("total_price", 0.0))
    return round(total, 2)


def _enforce_strict_math_cross_check(extracted_doc, report: AuditReport) -> AuditReport:
    """
    Enforce deterministic accounting equations.
    If any strict check fails, add a critical violation and force score to 0.
    """
    subtotal = _to_float(getattr(extracted_doc, "subtotal", 0.0))
    total_tax = _to_float(getattr(extracted_doc, "total_tax", 0.0))
    total_amount = _to_float(getattr(extracted_doc, "total_amount", 0.0))
    line_items_sum = _sum_line_items(extracted_doc)

    tolerance = 0.01
    discrepancy_reasons = []

    if line_items_sum is not None and abs(line_items_sum - subtotal) > tolerance:
        discrepancy_reasons.append(
            f"Sum(Line Items) mismatch: {line_items_sum:.2f} != Base Amount {subtotal:.2f}"
        )

    expected_grand_total = round(subtotal + total_tax, 2)
    if abs(expected_grand_total - total_amount) > tolerance:
        discrepancy_reasons.append(
            f"Base Amount + Total VAT mismatch: {expected_grand_total:.2f} != Grand Total {total_amount:.2f}"
        )

    if not discrepancy_reasons:
        return report

    discrepancy_message = " | ".join(discrepancy_reasons)
    violation = Violation(
        severity="critical",
        field="Mathematical Integrity",
        found_value=discrepancy_message,
        expected_value="Sum(Line Items) == Base Amount AND Base Amount + Total VAT == Grand Total",
        explanation=(
            "Critical Mathematical Discrepancy detected. The agent cannot auto-correct totals "
            "when deterministic arithmetic checks fail."
        ),
        legal_reference="Internal Financial Integrity Rule — Deterministic Cross-Check",
        law_code="MATH_INTEGRITY",
    )

    recommendations = list(report.recommendations)
    rec = "Review source amounts and recalculate totals manually due to critical mathematical discrepancy."
    if rec not in recommendations:
        recommendations.append(rec)

    legal_summary = (report.legal_summary or "")
    forced_note = "Critical Mathematical Discrepancy: deterministic cross-check failed."
    if forced_note not in legal_summary:
        legal_summary = (f"{legal_summary} {forced_note}").strip()

    return report.model_copy(update={
        "violations": [*report.violations, violation],
        "compliance_score": 0.0,
        "compliance_level": "non_compliant",
        "requires_human_review": True,
        "recommendations": recommendations,
        "legal_summary": legal_summary,
    })


def run_audit(extracted_doc, verbose: bool = True) -> AuditReport:
    """
    Run the full auditing pipeline on an ExtractedDocument.

    Args:
        extracted_doc: An ExtractedDocument from ingestion.processors
        verbose:       If True, print node-by-node progress

    Returns:
        AuditReport with violations, score, and recommendations

    Example:
        from ingestion.processors import process_document_pipeline
        from agent.auditor import run_audit

        doc = process_document_pipeline("invoice.pdf")
        report = run_audit(doc)
        print(report.compliance_score)   # e.g. 71.0
        print(report.violations)         # list of Violation objects
    """
    if verbose:
        print("\n" + "═" * 52)
        print(f"Starting audit: {extracted_doc.doc_type.upper()}")
        print("═" * 52)

    # Build the initial state from the ExtractedDocument
    initial_state = {
        "document_type":         extracted_doc.doc_type,
        "raw_text":              getattr(extracted_doc, "cleaned_raw_text", None) or extracted_doc.raw_text,
        "subtotal":              extracted_doc.subtotal,
        "total_tax":             extracted_doc.total_tax,
        "total_amount":          extracted_doc.total_amount,
        "currency":              extracted_doc.currency,
        "vendor_name":           extracted_doc.vendor_name,
        "tax_id":                extracted_doc.tax_id,
        "extraction_confidence": extracted_doc.extraction_confidence,
        # doc_category routes the agent to the correct audit rules
        "doc_category": getattr(extracted_doc, "doc_category", "invoice"),
        # Filled by nodes:
        "legal_queries":    [],
        "retrieved_laws":   [],
        "web_context":      None,
        "web_search_used":  False,
        "violations":       [],
        "recommendations":  [],
        "compliance_score": 0.0,
        "agent_confidence": 0.0,
        "legal_summary":    "",
        "laws_consulted":   [],
        "report":           None,
        "requires_human_review": False,
    }

    graph = _get_graph()
    final_state = graph.invoke(initial_state)

    # Extract and validate the final report
    report_dict = final_state.get("report")
    if not report_dict:
        raise RuntimeError("Agent completed but produced no report. Check node logs.")

    report = AuditReport(**report_dict)
    report = _enforce_strict_math_cross_check(extracted_doc, report)

    if verbose:
        print("\n" + "─" * 52)
        print(f"AUDIT COMPLETE")
        print(f"  Report ID  : {report.report_id}")
        print(f"  Score      : {report.compliance_score:.1f}/100")
        print(f"  Level      : {report.compliance_level}")
        print(f"  Violations : {len(report.violations)} ({report.critical_count} critical)")
        print(f"  Confidence : {report.agent_confidence:.0%}")
        print(f"  HITL flag  : {report.requires_human_review}")
        print("─" * 52 + "\n")

    return report


def run_audit_from_text(
    raw_text: str,
    doc_type: str = "invoice",
    subtotal: float = 0.0,
    total_tax: float = 0.0,
    total_amount: float = 0.0,
    currency: str = "EGP",
    tax_id: str = None,
) -> AuditReport:
    """
    Run an audit from raw text directly, without going through the ingestion pipeline.
    Useful for testing the agent with synthetic documents.
    """
    from ingestion.schemas import ExtractedDocument

    doc = ExtractedDocument(
        doc_type=doc_type,
        raw_text=raw_text,
        subtotal=subtotal,
        total_tax=total_tax,
        total_amount=total_amount,
        currency=currency,
        tax_id=tax_id,
        extraction_confidence=0.95,
    )
    return run_audit(doc)
