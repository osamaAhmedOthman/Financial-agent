"""
tests/test_ingestion.py

Run with:  pytest tests/test_ingestion.py -v
"""
import pytest
from ingestion.classifiers import classify_document
from ingestion.validators import validate_financial_math, validate_egyptian_tax_id, validate_vat_rate
from ingestion.schemas import ExtractedDocument, LineItem


# ── Classifier tests ───────────────────────────────────────────────────────

class TestClassifier:

    def test_english_invoice(self):
        text = "TAX INVOICE — Amount Due: 5000 EGP. Bill to: Acme Corp."
        doc_type, confidence = classify_document(text)
        assert doc_type == "invoice"
        assert confidence > 0.3

    def test_arabic_invoice(self):
        text = "فاتورة ضريبية — الإجمالي: 10,000 جنيه"
        doc_type, confidence = classify_document(text)
        assert doc_type == "invoice"
        assert confidence > 0.3

    def test_balance_sheet(self):
        text = "Balance Sheet — Financial Position as of Dec 31, 2024. Current Assets: 500,000"
        doc_type, confidence = classify_document(text)
        assert doc_type == "balance_sheet"

    def test_arabic_tax_return(self):
        text = "إقرار ضريبي — مصلحة الضرائب المصرية — السنة الضريبية 2024"
        doc_type, confidence = classify_document(text)
        assert doc_type == "tax_return"
        assert confidence > 0.3

    def test_unknown_document(self):
        text = "This document has no financial keywords at all."
        doc_type, confidence = classify_document(text)
        assert doc_type == "unknown"
        assert confidence == 0.0

    def test_confidence_is_between_0_and_1(self):
        text = "invoice bill amount due purchase order vendor tax invoice"
        _, confidence = classify_document(text)
        assert 0.0 <= confidence <= 1.0


# ── Validator tests ────────────────────────────────────────────────────────

class TestValidators:

    def _make_doc(self, **kwargs) -> ExtractedDocument:
        defaults = dict(
            doc_type="invoice",
            raw_text="sample text",
            subtotal=1000.0,
            total_tax=140.0,
            total_amount=1140.0,
            extraction_confidence=0.9,
            line_items=[
                LineItem(description="Service A", quantity=1, unit_price=1000.0, total_price=1000.0)
            ]
        )
        defaults.update(kwargs)
        return ExtractedDocument(**defaults)

    def test_math_passes_on_correct_numbers(self):
        doc = self._make_doc()
        assert validate_financial_math(doc) is True

    def test_math_fails_on_wrong_total(self):
        doc = self._make_doc(total_amount=9999.0)
        assert validate_financial_math(doc) is False

    def test_math_fails_with_no_line_items_and_zero_subtotal(self):
        doc = self._make_doc(line_items=[], subtotal=0.0, total_tax=0.0, total_amount=0.0)
        assert validate_financial_math(doc) is False

    def test_egyptian_tax_id_valid(self):
        assert validate_egyptian_tax_id("123456789") is True
        assert validate_egyptian_tax_id("12-345-6789") is True  # with dashes — cleaned

    def test_egyptian_tax_id_invalid(self):
        assert validate_egyptian_tax_id("12345") is False        # too short
        assert validate_egyptian_tax_id("1234567890") is False   # too long
        assert validate_egyptian_tax_id("") is False

    def test_vat_rate_valid(self):
        assert validate_vat_rate(14.0) is True
        assert validate_vat_rate(0.0) is True
        assert validate_vat_rate(5.0) is True

    def test_vat_rate_invalid(self):
        assert validate_vat_rate(12.0) is False
        assert validate_vat_rate(20.0) is False


# ── Schema tests ───────────────────────────────────────────────────────────

class TestSchemas:

    def test_unknown_doc_type_accepted(self):
        doc = ExtractedDocument(doc_type="unknown", raw_text="x")
        assert doc.doc_type == "unknown"

    def test_optional_fields_default_to_none(self):
        doc = ExtractedDocument(doc_type="invoice", raw_text="x")
        assert doc.vendor_name is None
        assert doc.tax_id is None
        assert doc.date is None

    def test_processed_at_is_set_automatically(self):
        doc = ExtractedDocument(doc_type="invoice", raw_text="x")
        assert doc.processed_at is not None
        assert len(doc.processed_at) > 0

    def test_negative_line_item_price_raises(self):
        with pytest.raises(Exception):
            LineItem(description="Bad item", unit_price=100.0, total_price=-50.0)
