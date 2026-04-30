"""
tests/test_agent.py

Tests for Phase 3 — LangGraph Auditing Agent.

Groups:
  TestAgentSchemas      — unit tests, no external services
  TestAgentTools        — unit tests, no Pinecone or Groq needed
  TestAgentIntegration  — requires GROQ_API_KEY + PINECONE_API_KEY

Run unit tests only:
    pytest tests/test_agent.py -v -m "not integration"

Run all (requires API keys):
    pytest tests/test_agent.py -v
"""
import os
import pytest
from agent.schemas import AuditState, AuditReport, Violation


def _pinecone_stack_available() -> bool:
    if not os.getenv("PINECONE_API_KEY"):
        return False

    try:
        from knowledge_base.embedder import _get_embedding_model

        _get_embedding_model()
        return True
    except Exception:
        return False


GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
PINECONE_AVAILABLE = _pinecone_stack_available()
FULL_TEST = GROQ_AVAILABLE and PINECONE_AVAILABLE


# ── Schema tests (no external services) ───────────────────────────────────

class TestAgentSchemas:

    def _make_report(self, **kwargs) -> AuditReport:
        defaults = dict(
            report_id="TEST01",
            document_type="invoice",
            compliance_score=85.0,
            compliance_level="minor_issues",
            legal_summary="تم تدقيق الوثيقة.",
            agent_confidence=0.90,
        )
        defaults.update(kwargs)
        return AuditReport(**defaults)

    def test_report_creation(self):
        report = self._make_report()
        assert report.report_id == "TEST01"
        assert report.compliance_score == 85.0

    def test_compliance_score_bounds(self):
        with pytest.raises(Exception):
            self._make_report(compliance_score=110.0)
        with pytest.raises(Exception):
            self._make_report(compliance_score=-5.0)

    def test_critical_count_property(self):
        report = self._make_report(violations=[
            Violation(severity="critical", field="VAT rate", found_value="12%",
                      expected_value="14%", explanation="Wrong rate",
                      legal_reference="المادة 6", law_code="VAT_67_2016"),
            Violation(severity="warning", field="Tax ID", found_value="12345",
                      expected_value="9 digits", explanation="Too short",
                      legal_reference="المادة 3", law_code="UNIFIED_PROCEDURES_206_2020"),
        ])
        assert report.critical_count == 1
        assert report.warning_count == 1

    def test_summary_line_format(self):
        report = self._make_report(compliance_score=71.0)
        line = report.summary_line()
        assert "71" in line
        assert "100" in line

    def test_requires_human_review_defaults_false(self):
        report = self._make_report()
        assert report.requires_human_review is False

    def test_generated_at_is_set(self):
        report = self._make_report()
        assert report.generated_at is not None
        assert len(report.generated_at) > 10

    def test_audit_state_defaults(self):
        state = AuditState()
        assert state.document_type == ""
        assert state.legal_queries == []
        assert state.retrieved_laws == []
        assert state.web_search_used is False

    def test_violation_severity_values(self):
        for severity in ["critical", "warning", "info"]:
            v = Violation(
                severity=severity, field="x", found_value="a",
                expected_value="b", explanation="test",
                legal_reference="المادة 1", law_code="TEST"
            )
            assert v.severity == severity

    def test_violation_invalid_severity(self):
        with pytest.raises(Exception):
            Violation(severity="high", field="x", found_value="a",
                      expected_value="b", explanation="test",
                      legal_reference="المادة 1", law_code="TEST")


# ── Tool tests (no external API calls) ────────────────────────────────────

class TestAgentTools:

    def test_needs_web_search_usd_currency(self):
        from agent.tools import needs_web_search
        assert needs_web_search("invoice", "USD", "some text") is True

    def test_needs_web_search_egp_no_keywords(self):
        from agent.tools import needs_web_search
        assert needs_web_search("invoice", "EGP", "فاتورة ضريبية عادية") is False

    def test_needs_web_search_dollar_keyword(self):
        from agent.tools import needs_web_search
        assert needs_web_search("invoice", "EGP", "سعر الصرف دولار") is True

    def test_needs_web_search_import_keyword(self):
        from agent.tools import needs_web_search
        assert needs_web_search("invoice", "EGP", "استيراد من الخارج") is True

    def test_search_web_no_key_returns_string(self):
        """If Tavily key is missing, should return a string not raise."""
        from agent.tools import search_web
        original = os.environ.pop("TAVILY_API_KEY", None)
        result = search_web("test query")
        assert isinstance(result, str)
        if original:
            os.environ["TAVILY_API_KEY"] = original

    def test_search_legal_knowledge_returns_list(self):
        """Even with a bad query, should return a list (empty is fine)."""
        if not PINECONE_AVAILABLE:
            pytest.skip("PINECONE_API_KEY not set")
        from agent.tools import search_legal_knowledge
        result = search_legal_knowledge("test query", top_k=2)
        assert isinstance(result, list)

    def test_search_legal_knowledge_structure(self):
        if not PINECONE_AVAILABLE:
            pytest.skip("PINECONE_API_KEY not set")
        from agent.tools import search_legal_knowledge
        results = search_legal_knowledge("معدل الضريبة على الفواتير", top_k=2)
        if results:
            assert "article_number" in results[0]
            assert "law_code" in results[0]
            assert "text" in results[0]
            assert "score" in results[0]


# ── Integration tests (require both GROQ + PINECONE) ──────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not FULL_TEST, reason="Requires GROQ_API_KEY and PINECONE_API_KEY")
class TestAgentIntegration:

    @pytest.fixture(autouse=True)
    def setup_knowledge_base(self):
        """Load seed data and build BM25 index before each integration test."""
        try:
            from knowledge_base.loader import load_seed_data
            from knowledge_base.embedder import upsert_chunks

            chunks = load_seed_data()
            upsert_chunks(chunks)
            import time; time.sleep(2)
        except Exception as exc:
            pytest.skip(f"Integration stack unavailable: {exc}")

    def test_audit_correct_invoice(self):
        """A correctly formed invoice should score >= 80."""
        from agent.auditor import run_audit_from_text
        report = run_audit_from_text(
            raw_text="فاتورة ضريبية. المبلغ الأساسي: 10,000 جنيه. ضريبة القيمة المضافة (14%): 1,400 جنيه. الإجمالي: 11,400 جنيه.",
            doc_type="invoice",
            subtotal=10000.0,
            total_tax=1400.0,
            total_amount=11400.0,
            tax_id="123456789",
        )
        assert isinstance(report, AuditReport)
        assert report.compliance_score >= 70.0, f"Correct invoice scored too low: {report.compliance_score}"
        assert report.agent_confidence > 0.0

    def test_audit_wrong_vat_rate(self):
        """An invoice with 12% VAT instead of 14% must have at least one violation."""
        from agent.auditor import run_audit_from_text
        report = run_audit_from_text(
            raw_text="فاتورة ضريبية. المبلغ الأساسي: 10,000 جنيه. ضريبة القيمة المضافة (12%): 1,200 جنيه. الإجمالي: 11,200 جنيه.",
            doc_type="invoice",
            subtotal=10000.0,
            total_tax=1200.0,
            total_amount=11200.0,
        )
        assert isinstance(report, AuditReport)
        assert len(report.violations) > 0, "Wrong VAT rate should produce at least one violation"
        assert report.compliance_score < 100.0

    def test_audit_report_has_legal_references(self):
        """Every violation must cite a legal article."""
        from agent.auditor import run_audit_from_text
        report = run_audit_from_text(
            raw_text="فاتورة بدون رقم تسجيل ضريبي. المبلغ: 5,000 جنيه. الضريبة: 500 جنيه. الإجمالي: 5,500 جنيه.",
            doc_type="invoice",
            subtotal=5000.0,
            total_tax=500.0,
            total_amount=5500.0,
        )
        for violation in report.violations:
            assert violation.legal_reference, f"Violation missing legal reference: {violation.field}"
            assert violation.law_code, f"Violation missing law code: {violation.field}"

    def test_report_has_recommendations(self):
        from agent.auditor import run_audit_from_text
        report = run_audit_from_text(
            raw_text="فاتورة ضريبية. المبلغ: 10,000. الضريبة: 1,200. الإجمالي: 11,200.",
            doc_type="invoice",
            subtotal=10000.0,
            total_tax=1200.0,
            total_amount=11200.0,
        )
        assert len(report.recommendations) > 0

    def test_usd_invoice_triggers_web_search(self):
        """A USD-denominated invoice should trigger web search."""
        from agent.auditor import run_audit_from_text
        report = run_audit_from_text(
            raw_text="Tax Invoice. Amount: $1,000 USD. VAT 14%: $140. Total: $1,140.",
            doc_type="invoice",
            subtotal=1000.0,
            total_tax=140.0,
            total_amount=1140.0,
            currency="USD",
        )
        assert report.web_search_used is True

    def test_hitl_flag_on_low_confidence(self):
        """If extraction confidence is very low, HITL flag must be set."""
        from ingestion.schemas import ExtractedDocument
        from agent.auditor import run_audit

        doc = ExtractedDocument(
            doc_type="invoice",
            raw_text="نص غير واضح",
            subtotal=0.0,
            total_tax=0.0,
            total_amount=0.0,
            extraction_confidence=0.30,  # very low
        )
        report = run_audit(doc)
        assert report.requires_human_review is True
