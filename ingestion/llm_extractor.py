"""
ingestion/llm_extractor.py

Two-pass financial field extraction with document-type awareness.


  1. Prompt now targets FOOTER totals, not line-item amounts (WaelSoft fix)
  2. Document category detection routes balance sheets to different rules
  3. Sanity checks catch LLM errors like tax=total confusion
  4. Arabic-Indic digit normalisation before all matching
"""
import os
import json
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalise_arabic_numbers(text: str) -> str:
    return text.translate(_AR_DIGITS).translate(_FA_DIGITS)


def clean_financial_string(value) -> str:
    """
    Normalize OCR-financial numeric strings before float conversion.
    - Removes currency markers like EGP / L.E / ج.م
    - Treats ',' strictly as thousands separator
    - Keeps '.' as decimal separator
    """
    if value is None:
        return ""

    s = normalise_arabic_numbers(str(value))
    s = re.sub(r"(?i)\b(?:egp|l\.?e\.?)\b", "", s)
    s = re.sub(r"ج\s*\.?\s*م", "", s)
    s = re.sub(r"جنيه(?:\s*مصري)?", "", s, flags=re.IGNORECASE)

    # Thousands separators are commas in this pipeline; decimal separator is dot.
    s = s.replace(",", "")
    s = s.replace(" ", "")

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else ""


# ── Document category detection ───────────────────────────────────────────

_FS_SIGNALS = [
    "قائمة المركز المالي", "قائمة الأرباح", "قائمة الدخل الشامل",
    "consolidated financial", "balance sheet", "income statement",
    "statement of financial position", "المجمعة المختصرة",
    "حقوق الملكية", "الأصول غير المتداولة", "الالتزامات",
    "ربح الفترة", "مجمل الربح", "صافي الأصول", "اإليرادات",
]
_INV_SIGNALS = [
    "فاتورة ضريبية", "tax invoice", "فاتورة", "invoice number",
    "رقم الفاتورة", "amount due", "المبلغ المستحق",
    "total vat", "grand total", "vat amount", "vat number",
]


def detect_document_category(raw_text: str) -> str:
    text_lower = normalise_arabic_numbers(raw_text.lower())
    fs_score  = sum(1 for s in _FS_SIGNALS  if s in text_lower)
    inv_score = sum(1 for s in _INV_SIGNALS if s in text_lower)
    if fs_score > inv_score and fs_score >= 2:
        return "financial_statement"
    elif inv_score > 0:
        return "invoice"
    return "unknown"


# ── Regex extraction (Pass 1) ─────────────────────────────────────────────

_GRAND_TOTAL_RE = [
    r"total amount due", r"grand total", r"amount due",
    r"إجمالي المبلغ المستحق", r"الإجمالي الكلي", r"المبلغ المستحق",
    r"total\s*\(incl", r"المبلغ الإجمالي",
]
_VAT_RE = [
    r"total vat", r"total tax", r"vat amount",
    r"مجموع ضريبة القيمة المضافة", r"إجمالي الضريبة",
    r"total vat\s*\(14", r"ضريبة القيمة المضافة",
]
_SUBTOTAL_RE = [
    r"total excluding vat", r"subtotal", r"net amount",
    r"total\s*\(excl", r"الإجمالي قبل الضريبة",
    r"إجمالي بدون ضريبة", r"total taxable amount excluding",
]


def _find_amount(labels, text):
    for label in labels:
        m = re.search(rf"(?:{label})[:\s\|]*([0-9][0-9,\s]*\.?[0-9]*)", text, re.IGNORECASE)
        if m:
            try:
                cleaned = clean_financial_string(m.group(1))
                v = float(cleaned) if cleaned else 0.0
                if v > 0:
                    return round(v, 2)
            except ValueError:
                pass
    return None


def _regex_extract(text: str) -> dict:
    norm = normalise_arabic_numbers(text.lower())
    r = _empty_fields()
    s = _find_amount(_SUBTOTAL_RE, norm)
    t = _find_amount(_VAT_RE, norm)
    a = _find_amount(_GRAND_TOTAL_RE, norm)
    if s: r["subtotal"]      = s
    if t: r["total_tax"]     = t
    if a: r["total_amount"]  = a
    m = re.search(r"(?<!\d)(\d{9})(?!\d)", norm)
    if m: r["tax_id"] = m.group(1)
    return r


# ── LLM prompts ───────────────────────────────────────────────────────────

_INVOICE_PROMPT = """
You are a financial data extraction system for Egyptian tax invoices.

Extracted text from invoice (may contain OCR errors):
---
{raw_text}
---

IMPORTANT: Look ONLY at the FOOTER/SUMMARY rows at the bottom of the invoice.
IGNORE individual line item amounts in the table body.

Extract these 3 numbers from the FOOTER:
1. subtotal: Total BEFORE tax (labeled: Subtotal, Total Excl VAT, الاجمالي بدون ضريبة)
2. total_tax: Tax amount ONLY (labeled: Total VAT, مجموع ضريبة القيمة المضافة)
3. total_amount: The FINAL grand total (labeled: Grand Total, Amount Due, المبلغ المستحق)

Rules:
- total_amount MUST be larger than total_tax and larger than subtotal
- total_amount = subtotal + total_tax (approximately)
- Number formatting rule is strict: ',' is a thousands separator and '.' is the decimal separator.
- Examples: '18,500.00' -> 18500.00, '2,100' -> 2100.00, 'EGP 16,500.00' -> 16500.00
- tax_id must be exactly 9 digits (the seller's tax registration number)
- If Arabic numerals like ١٢٣ appear, convert to 123
- Also produce cleaned_raw_text: a cleaned version of the OCR text with the same line structure,
    correcting obvious OCR gibberish in Arabic/English accounting terms.
- Keep section order and line breaks as much as possible.

Return ONLY valid JSON:
{{
  "subtotal": <number>,
  "total_tax": <number>,
  "total_amount": <number>,
  "tax_id": "<9 digits or null>",
  "vendor_name": "<seller name or null>",
  "date": "<YYYY-MM-DD or null>",
    "currency": "EGP",
    "cleaned_raw_text": "<cleaned text preserving structure>"
}}
"""

_FS_PROMPT = """
You are a financial data extraction system for Egyptian financial statements.

Extracted text from financial statement:
---
{raw_text}
---

This is a FINANCIAL STATEMENT (balance sheet / income statement), NOT an invoice.

Extract:
1. total_revenue: Total revenues for the most recent period
2. net_profit: Net profit after tax for the most recent period
3. total_assets: Total assets (from balance sheet)
4. income_tax: Income tax expense
5. company_name: Company name
6. period_end: Period end date
7. cleaned_raw_text: cleaned OCR text with preserved sections and line breaks

Return ONLY valid JSON:
{{
  "total_revenue": <number or 0>,
  "net_profit": <number or 0>,
  "total_assets": <number or 0>,
  "income_tax": <number or 0>,
  "company_name": "<name or null>",
  "period_end": "<YYYY-MM-DD or null>",
    "document_category": "financial_statement",
    "cleaned_raw_text": "<cleaned text preserving structure>"
}}
"""


# ── LLM call ─────────────────────────────────────────────────────────────

def llm_extract_fields(raw_text: str, doc_category: str = "invoice") -> dict:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return _empty_fields()

    snippet = normalise_arabic_numbers(raw_text)[:3500]
    prompt = (_FS_PROMPT if doc_category == "financial_statement"
              else _INVOICE_PROMPT).format(raw_text=snippet)

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())

        if doc_category == "financial_statement":
            return _map_fs(result)

        cleaned = _fix_invoice_result(result)
        print(f"  [LLM] sub={cleaned['subtotal']}, "
              f"tax={cleaned['total_tax']}, total={cleaned['total_amount']}")
        return cleaned

    except Exception as e:
        print(f"  [LLM Extractor] Failed: {e}")
        return _empty_fields()


def _fix_invoice_result(r: dict) -> dict:
    """Sanity-check and auto-correct common LLM mistakes."""
    sub   = _safe_float(r.get("subtotal"))
    tax   = _safe_float(r.get("total_tax"))
    total = _safe_float(r.get("total_amount"))

    # Fix 1: total < tax → LLM swapped them
    if total > 0 and tax > 0 and total < tax:
        total, tax = tax, total

    # Fix 2: total == tax → LLM duplicated the tax value as total
    if total > 0 and tax > 0 and abs(total - tax) < 0.01:
        if sub > 0:
            total = round(sub + tax, 2)
        else:
            total = 0.0  # can't fix, flag for human review

    # Fix 3: subtotal missing but total and tax are present → infer it
    if sub == 0.0 and total > 0 and tax > 0:
        inferred = round(total - tax, 2)
        if inferred > 0:
            sub = inferred

    return {
        "subtotal":    sub,
        "total_tax":   tax,
        "total_amount": total,
        "tax_id":      _clean_tax_id(r.get("tax_id")),
        "vendor_name": _clean_str(r.get("vendor_name")),
        "date":        _clean_str(r.get("date")),
        "currency":    r.get("currency", "EGP"),
        "cleaned_raw_text": _clean_str(r.get("cleaned_raw_text")),
    }


def _map_fs(r: dict) -> dict:
    """Map financial statement LLM output to ExtractedDocument fields."""
    return {
        "subtotal":    _safe_float(r.get("total_revenue")),
        "total_tax":   _safe_float(r.get("income_tax")),
        "total_amount": _safe_float(r.get("net_profit")),
        "tax_id":      None,
        "vendor_name": _clean_str(r.get("company_name")),
        "date":        _clean_str(r.get("period_end")),
        "currency":    "EGP",
        "cleaned_raw_text": _clean_str(r.get("cleaned_raw_text")),
    }


# ── Main entry point ──────────────────────────────────────────────────────

def extract_financial_fields(raw_text: str) -> dict:
    """
    Main entry point. Detects document category then runs extraction.
    Returns dict with all fields plus 'doc_category' key.
    """
    if not raw_text or not raw_text.strip():
        r = _empty_fields()
        r["doc_category"] = "unknown"
        return r

    doc_category = detect_document_category(raw_text)
    print(f"  [Extractor] Category: {doc_category}")

    # Pass 1: regex (invoices only, needs clean text)
    if doc_category != "financial_statement":
        regex_r = _regex_extract(raw_text)
        if regex_r["total_amount"] > 0 or regex_r["subtotal"] > 0:
            print(f"  [Extractor] Regex OK — "
                  f"total={regex_r['total_amount']}, sub={regex_r['subtotal']}")
            regex_r["doc_category"] = doc_category
            return regex_r

    # Pass 2: LLM
    print(f"  [Extractor] Calling LLM...")
    result = llm_extract_fields(raw_text, doc_category=doc_category)
    result["doc_category"] = doc_category
    return result


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    if val is None: return 0.0
    try:
        cleaned = clean_financial_string(val)
        return round(float(cleaned), 2) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def _clean_tax_id(val) -> Optional[str]:
    if not val: return None
    d = re.sub(r"\D", "", str(val))
    return d if len(d) == 9 else None


def _clean_str(val) -> Optional[str]:
    if not val or str(val).strip() in ("null", "None", ""): return None
    return str(val).strip()


def _empty_fields() -> dict:
    return {
        "subtotal": 0.0, "total_tax": 0.0, "total_amount": 0.0,
        "tax_id": None, "vendor_name": None, "date": None,
        "currency": "EGP", "doc_category": "unknown", "cleaned_raw_text": None,
    }
