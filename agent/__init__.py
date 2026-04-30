"""
agent/
The LangGraph auditing agent.

Public API (everything else imports only from here):
    run_audit(extracted_doc)   → AuditReport
    run_audit_from_text(...)   → AuditReport
    AuditReport                → the output schema
    Violation                  → individual violation model
"""
from .auditor import run_audit, run_audit_from_text
from .schemas import AuditReport, Violation, AuditState

__all__ = [
    "run_audit",
    "run_audit_from_text",
    "AuditReport",
    "Violation",
    "AuditState",
]
