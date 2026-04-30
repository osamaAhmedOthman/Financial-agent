"""
agent/graph.py

The LangGraph agent — the reasoning brain of the auditor.

Graph structure:
  query_builder → legal_retriever → [web_search?] → auditor → report_generator
                                         ↑ conditional edge

Each node is a plain Python function that:
  1. Reads from the shared state dict
  2. Does its work (LLM call, tool call, computation)
  3. Returns a dict of ONLY the fields it changed
"""
import json
import os
import uuid
import operator
from typing import Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from .schemas import AuditState, AuditReport, Violation
from .tools import search_legal_knowledge, search_web, needs_web_search
from .prompts import (QUERY_BUILDER_PROMPT, WEB_CONTEXT_TEMPLATE,
                      AUDITOR_PROMPT_INVOICE, AUDITOR_PROMPT_FINANCIAL_STATEMENT,
                      _INVOICE_CATEGORY_INSTRUCTION, _FS_CATEGORY_INSTRUCTION)

load_dotenv()


# ── Proper TypedDict state for LangGraph ──────────────────────────────────

class GraphState(TypedDict, total=False):
    # Input fields
    document_type:          str
    raw_text:               str
    subtotal:               float
    total_tax:              float
    total_amount:           float
    currency:               str
    vendor_name:            str
    tax_id:                 str
    extraction_confidence:  float
    doc_category:           str   # NEW: "invoice" | "financial_statement" | "unknown"
    # Accumulated by nodes
    legal_queries:          list
    retrieved_laws:         list
    violations:             list
    recommendations:        list
    laws_consulted:         list
    # Scalar outputs
    web_context:            str
    web_search_used:        bool
    compliance_score:       float
    agent_confidence:       float
    legal_summary:          str
    report:                 dict
    requires_human_review:  bool


# ── LLM client (lazy-loaded, same pattern as Phase 1 & 2) ─────────────────

_llm_client = None

def _get_llm():
    """Returns a Groq client cached for the session."""
    global _llm_client
    if _llm_client is None:
        from groq import Groq
        _llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _llm_client


def _call_llm(prompt: str, as_json: bool = False, temperature: float = 0.1) -> str:
    """
    Single LLM call via Groq API.
    Uses llama-3.3-70b-versatile — strong enough for financial reasoning.
    Temperature 0.1 for deterministic, consistent audit results.
    """
    client = _get_llm()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
        response_format={"type": "json_object"} if as_json else None,
    )
    return response.choices[0].message.content.strip()


# ── Node 1: Query builder ──────────────────────────────────────────────────

def query_builder_node(state: dict) -> dict:
    """Generates legal search queries adapted to document type."""
    print("  [Node 1] Building legal queries...")

    doc_category = state.get("doc_category", "invoice")
    cat_instruction = (_FS_CATEGORY_INSTRUCTION
                       if doc_category == "financial_statement"
                       else _INVOICE_CATEGORY_INSTRUCTION)

    prompt = QUERY_BUILDER_PROMPT.format(
        document_type=state.get("document_type", "invoice"),
        doc_category=doc_category,
        subtotal=state.get("subtotal", 0.0),
        currency=state.get("currency", "EGP"),
        total_tax=state.get("total_tax", 0.0),
        total_amount=state.get("total_amount", 0.0),
        tax_id=state.get("tax_id") or "غير موجود",
        category_instruction=cat_instruction,
    )

    raw_output = _call_llm(prompt, temperature=0.2)
    queries = [
        line.strip()
        for line in raw_output.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ][:4]

    baseline = (
        "ما هي متطلبات الإفصاح الضريبي في القوائم المالية وفق قانون 91/2005؟"
        if doc_category == "financial_statement"
        else "ما هو المعدل القانوني لضريبة القيمة المضافة على الفواتير؟"
    )
    if baseline not in queries:
        queries.insert(0, baseline)

    print(f"  [Node 1] Generated {len(queries)} queries")
    return {"legal_queries": queries}


# ── Node 2: Legal retriever ────────────────────────────────────────────────

def legal_retriever_node(state: dict) -> dict:
    """
    Runs all legal queries against Pinecone and collects the results.
    """
    queries = state.get("legal_queries", [])
    print(f"  [Node 2] Retrieving laws for {len(queries)} queries...")

    seen_ids = set()
    all_results = []

    for query in queries:
        results = search_legal_knowledge(query=query, top_k=3)
        for r in results:
            chunk_id = f"{r['law_code']}_{r['article_number']}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_results.append(r)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    laws_consulted = list({r["law_code"] for r in all_results})

    print(f"  [Node 2] Retrieved {len(all_results)} unique legal chunks")
    return {
        "retrieved_laws": all_results,
        "laws_consulted": laws_consulted,
    }


# ── Conditional edge: does this need web search? ───────────────────────────

def should_search_web(state: dict) -> Literal["web_search", "auditor"]:
    """
    FIX: Reads currency directly from the state dict.
    Previously reconstructed AuditState which defaulted currency to "EGP",
    losing the actual currency value passed in the initial state.
    """
    currency = state.get("currency", "EGP")
    raw_text = state.get("raw_text", "")
    doc_type = state.get("document_type", "")

    if needs_web_search(doc_type, currency, raw_text):
        print(f"  [Edge] Routing to web search (currency={currency})")
        return "web_search"
    print("  [Edge] Routing directly to auditor")
    return "auditor"


# ── Node 3: Web search (optional) ─────────────────────────────────────────

def web_search_node(state: dict) -> dict:
    """
    Fetches live data when Pinecone doesn't have it (exchange rates, etc.)
    """
    print("  [Node 3] Running web search...")
    currency = state.get("currency", "EGP")

    query = f"سعر الصرف {currency} مقابل الجنيه المصري اليوم"
    if currency.upper() == "USD":
        query = "USD EGP exchange rate today Egypt Central Bank"

    web_result = search_web(query=query)
    print(f"  [Node 3] Web search complete ({len(web_result)} chars)")

    return {
        "web_context": web_result,
        "web_search_used": True,
    }


# ── Node 4: Auditor ────────────────────────────────────────────────────────

def auditor_node(state: dict) -> dict:
    """Core reasoning node — selects prompt based on document category."""
    print("  [Node 4] Running audit reasoning...")

    retrieved_laws = state.get("retrieved_laws", [])
    law_blocks = []
    for law in retrieved_laws[:6]:
        block = (
            f"القانون: {law['law_name']}\n"
            f"المادة: {law['article_number']}\n"
            f"النص: {law['text'][:400]}\n"
            f"الصلة: {law['score']:.2f}"
        )
        law_blocks.append(block)
    legal_context = "\n\n---\n\n".join(law_blocks) if law_blocks else "لا توجد مواد قانونية متاحة"

    web_context = state.get("web_context")
    web_section = ""
    if web_context:
        web_section = WEB_CONTEXT_TEMPLATE.format(web_context=web_context[:600])

    subtotal     = state.get("subtotal", 0.0)
    total_tax    = state.get("total_tax", 0.0)
    total_amount = state.get("total_amount", 0.0)
    doc_category = state.get("doc_category", "invoice")

    # ── Correct VAT reasoning for mixed-rate invoices ──────────────────────
    # Problem: expected_vat = subtotal × 0.14 is WRONG when some items are
    # exempt (0% VAT). Example: subtotal=16500, but 1500 is exempt, so
    # taxable_base=15000 and correct VAT=2100, NOT 16500×0.14=2310.
    #
    # Correct approach: back-calculate the taxable base from declared tax.
    # If declared_tax/0.14 ≤ subtotal → some items are exempt → valid.
    # Only flag if declared_tax > subtotal×0.14 (overcharged) or
    # declared_tax/0.14 > subtotal (impossible, tax exceeds base).

    if total_tax > 0:
        # The taxable portion of the subtotal (back-calculated from declared tax)
        implied_taxable_base = round(total_tax / 0.14, 2)
        # The exempt (0% VAT) portion
        exempt_amount = round(subtotal - implied_taxable_base, 2)
        # Maximum possible VAT if everything were taxable
        max_possible_vat = round(subtotal * 0.14, 2)
    else:
        implied_taxable_base = 0.0
        exempt_amount = subtotal
        max_possible_vat = round(subtotal * 0.14, 2)

    # Determine if the VAT is mathematically valid
    vat_is_valid = (
        total_tax > 0
        and implied_taxable_base <= subtotal + 0.5  # allow small rounding
        and implied_taxable_base >= 0
    )

    # Select the right prompt based on document category
    if doc_category == "financial_statement":
        prompt = AUDITOR_PROMPT_FINANCIAL_STATEMENT.format(
            subtotal=subtotal,
            currency=state.get("currency", "EGP"),
            total_tax=total_tax,
            total_amount=total_amount,
            raw_text_preview=state.get("raw_text", "")[:300],
            legal_context=legal_context,
        )
    else:
        prompt = AUDITOR_PROMPT_INVOICE.format(
            document_type=state.get("document_type", "invoice"),
            subtotal=subtotal,
            currency=state.get("currency", "EGP"),
            total_tax=total_tax,
            total_amount=total_amount,
            tax_id=state.get("tax_id") or "غير موجود",
            raw_text_preview=state.get("raw_text", "")[:300],
            legal_context=legal_context,
            web_context_section=web_section,
            # Pass the correct values to the prompt
            implied_taxable_base=implied_taxable_base,
            exempt_amount=max(exempt_amount, 0),
            max_possible_vat=max_possible_vat,
            vat_is_valid=vat_is_valid,
        )

    raw_json = _call_llm(prompt, as_json=True)

    try:
        result = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"  [Node 4] JSON parse failed: {e}. Using safe defaults.")
        result = {
            "violations": [],
            "recommendations": ["تعذر تحليل الوثيقة — يرجى المراجعة اليدوية"],
            "compliance_score": 0.0,
            "agent_confidence": 0.0,
            "legal_summary": "فشل التحليل التلقائي",
            "laws_consulted": [],
        }

    print(f"  [Node 4] Score: {result.get('compliance_score', 0):.0f}/100 | "
          f"Violations: {len(result.get('violations', []))}")

    return {
        "violations":       result.get("violations", []),
        "recommendations":  result.get("recommendations", []),
        "compliance_score": float(result.get("compliance_score", 0.0)),
        "agent_confidence": float(result.get("agent_confidence", 0.0)),
        "legal_summary":    result.get("legal_summary", ""),
        "laws_consulted":   result.get("laws_consulted", state.get("laws_consulted", [])),
    }


# ── Node 5: Report generator ───────────────────────────────────────────────

def report_generator_node(state: dict) -> dict:
    """
    Converts raw audit results into a validated AuditReport Pydantic object.
    """
    print("  [Node 5] Building final report...")

    violations = []
    for v in state.get("violations", []):
        try:
            violations.append(Violation(**v))
        except Exception as e:
            print(f"  [Node 5] Skipping malformed violation: {e}")

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    violations.sort(key=lambda v: severity_order.get(v.severity, 9))

    score = state.get("compliance_score", 0.0)
    critical_count = sum(1 for v in violations if v.severity == "critical")

    # Compliance level tiers — based on BOTH score AND critical violations
    if score >= 90 and critical_count == 0:
        level = "compliant"
    elif score >= 70 and critical_count == 0:
        level = "minor_issues"
    elif score >= 50:
        level = "major_issues"
    else:
        level = "non_compliant"

    agent_confidence = state.get("agent_confidence", 0.0)
    extraction_conf  = state.get("extraction_confidence", 1.0)

    # HITL: only flag for genuinely uncertain or severely non-compliant
    requires_review = (
        agent_confidence < 0.70
        or extraction_conf < 0.60
        or critical_count >= 2
        or score < 40
    )

    report = AuditReport(
        report_id=str(uuid.uuid4())[:8].upper(),
        document_type=state.get("document_type", ""),
        compliance_score=round(score, 1),
        compliance_level=level,
        violations=violations,
        recommendations=state.get("recommendations", []),
        laws_consulted=state.get("laws_consulted", []),
        legal_summary=state.get("legal_summary", ""),
        agent_confidence=round(agent_confidence, 3),
        requires_human_review=requires_review,
        web_search_used=state.get("web_search_used", False),
        raw_document_text=state.get("raw_text", "")[:500] or None,
    )

    print(f"  [Node 5] Report {report.report_id}: {report.summary_line()}")

    return {
        "report": report.model_dump(),
        "requires_human_review": requires_review,
    }


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_audit_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Call this once at startup and reuse the compiled graph.

    Returns a compiled LangGraph graph ready for .invoke()
    """
    from langgraph.graph import StateGraph, END

    # FIX: Use GraphState (TypedDict) instead of plain dict.
    # LangGraph with TypedDict properly MERGES partial node outputs
    # instead of replacing the whole state — this is what caused subtotal=0.0.
    workflow = StateGraph(GraphState)

    # Register nodes directly — they receive and return dicts
    workflow.add_node("query_builder",    query_builder_node)
    workflow.add_node("legal_retriever",  legal_retriever_node)
    workflow.add_node("web_search",       web_search_node)
    workflow.add_node("auditor",          auditor_node)
    workflow.add_node("report_generator", report_generator_node)

    workflow.set_entry_point("query_builder")

    workflow.add_edge("query_builder",    "legal_retriever")
    workflow.add_edge("web_search",       "auditor")
    workflow.add_edge("auditor",          "report_generator")
    workflow.add_edge("report_generator", END)

    workflow.add_conditional_edges(
        "legal_retriever",
        should_search_web,
        {
            "web_search": "web_search",
            "auditor":    "auditor",
        }
    )

    return workflow.compile()
