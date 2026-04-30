"""
agent/schemas.py

Output schemas for the LangGraph auditing agent.
These are what the agent PRODUCES — separate from the ingestion
schemas (what we read) and knowledge base schemas (what we store).
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
from datetime import datetime, timezone


class Violation(BaseModel):
    """A single compliance violation found in the document."""

    severity: Literal["critical", "warning", "info"] = Field(
        description="critical=financial penalty likely, warning=requires correction, info=advisory"
    )
    field: str = Field(
        description="Which field or value the violation is about, e.g. 'VAT rate', 'Tax ID'"
    )
    found_value: str = Field(
        description="What the document actually says"
    )
    expected_value: str = Field(
        description="What the law requires"
    )
    explanation: str = Field(
        description="Plain-language explanation of why this is a violation"
    )
    legal_reference: str = Field(
        description="Exact article that was violated, e.g. 'المادة 6 — قانون 67/2016'"
    )
    law_code: str = Field(
        description="Machine-readable law code, e.g. 'VAT_67_2016'"
    )


class AuditReport(BaseModel):
    """
    The final structured output of the AI Financial Auditor.
    One report per document processed.

    This is the contract between:
      - agent/     (produces it)
      - api/       (serves it)
      - ui/        (displays it)
    """

    # ── Identity ───────────────────────────────────────────────────────────
    report_id: str = Field(description="Unique ID for this audit report")
    document_type: str = Field(description="Invoice, balance sheet, tax return, etc.")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of report generation"
    )

    # ── Compliance score ───────────────────────────────────────────────────
    compliance_score: float = Field(
        ge=0.0, le=100.0,
        description="Overall compliance score: 100 = fully compliant, 0 = severely non-compliant"
    )
    compliance_level: Literal["compliant", "minor_issues", "major_issues", "non_compliant"] = Field(
        description="Human-readable compliance tier"
    )

    # ── Findings ───────────────────────────────────────────────────────────
    violations: List[Violation] = Field(
        default_factory=list,
        description="All violations found, ordered by severity (critical first)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable steps to bring the document into compliance"
    )

    # ── Legal context ──────────────────────────────────────────────────────
    laws_consulted: List[str] = Field(
        default_factory=list,
        description="List of law codes that were retrieved and checked"
    )
    legal_summary: str = Field(
        description="One-paragraph plain-language summary of the audit findings"
    )

    # ── Quality signals ────────────────────────────────────────────────────
    agent_confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the agent is in its findings (0.0–1.0)"
    )
    requires_human_review: bool = Field(
        default=False,
        description="True if confidence is low or critical violations were found"
    )
    web_search_used: bool = Field(
        default=False,
        description="True if the agent had to search the web for live data"
    )

    # ── Raw context (for debugging and LangSmith tracing) ─────────────────
    raw_document_text: Optional[str] = Field(
        default=None,
        description="Truncated raw text that was audited (first 500 chars)"
    )

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def summary_line(self) -> str:
        return (
            f"Score: {self.compliance_score:.0f}/100 | "
            f"Critical: {self.critical_count} | "
            f"Warnings: {self.warning_count} | "
            f"{'⚠ Human review needed' if self.requires_human_review else 'Auto-approved'}"
        )


class AuditState(BaseModel):
    """
    The shared state object passed between all LangGraph nodes.
    Each node reads from it and writes back to it.

    Think of this as the agent's working memory for one audit run.
    """

    # Input (set at the start, never modified)
    document_type: str = ""
    raw_text: str = ""
    subtotal: float = 0.0
    total_tax: float = 0.0
    total_amount: float = 0.0
    currency: str = "EGP"
    vendor_name: Optional[str] = None
    tax_id: Optional[str] = None
    extraction_confidence: float = 0.0

    # Built up by the query builder node
    legal_queries: List[str] = Field(default_factory=list)

    # Filled by the legal retriever node
    retrieved_laws: List[dict] = Field(default_factory=list)

    # Filled by the web search node (optional)
    web_context: Optional[str] = None
    web_search_used: bool = False

    # Filled by the auditor node
    violations: List[dict] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    compliance_score: float = 0.0
    agent_confidence: float = 0.0
    legal_summary: str = ""
    laws_consulted: List[str] = Field(default_factory=list)

    # Final output
    report: Optional[dict] = None
    requires_human_review: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
