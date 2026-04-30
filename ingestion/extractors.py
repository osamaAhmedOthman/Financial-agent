import os
import re
import pdfplumber
import pandas as pd
from difflib import get_close_matches
from typing import Optional


# Importing this module previously crashed if EasyOCR wasn't installed.
_ocr_reader = None

_EN_ACCOUNTING_TERMS = [
    "invoice", "software", "hardware", "subtotal", "total", "amount", "amount due",
    "tax", "vat", "tax invoice", "grand total", "currency", "balance", "statement",
    "financial", "revenue", "assets", "liabilities", "net profit", "income tax",
]

_COMMON_OCR_REPLACEMENTS = {
    "ilaeusguuare": "Software",
    "1nvoice": "invoice",
    "invo1ce": "invoice",
    "subtota1": "subtotal",
    "tota1": "total",
    "va1": "vat",
    "ta x": "tax",
    "ضريبه": "ضريبة",
    "القيمه": "القيمة",
    "الضريبيه": "الضريبية",
}

def _get_ocr_reader():
    """Returns a cached EasyOCR reader. Downloads models on first call."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("Initializing EasyOCR (first-time model download may take a moment)...")
        _ocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _ocr_reader


def _preprocess_image_for_ocr(image_array):
    """Improve OCR quality with denoising + adaptive thresholding."""
    import cv2

    if image_array is None:
        return image_array

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned


def clean_extracted_text(raw_text: str) -> str:
    """Post-process OCR text while preserving line structure."""
    if not raw_text:
        return ""

    def _fix_token(token: str) -> str:
        base = re.sub(r"[^A-Za-z]", "", token or "")
        if len(base) < 5:
            return token

        lower_base = base.lower()
        if lower_base in _COMMON_OCR_REPLACEMENTS:
            candidate = _COMMON_OCR_REPLACEMENTS[lower_base]
            return token.replace(base, candidate)

        matches = get_close_matches(lower_base, _EN_ACCOUNTING_TERMS, n=1, cutoff=0.84)
        if matches:
            replacement = matches[0]
            if base.istitle():
                replacement = replacement.title()
            elif base.isupper():
                replacement = replacement.upper()
            return token.replace(base, replacement)
        return token

    text = raw_text
    for bad, good in _COMMON_OCR_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text, flags=re.IGNORECASE)

    lines = []
    for line in text.splitlines():
        tokens = re.split(r"(\s+)", line)
        fixed_tokens = [_fix_token(t) if not t.isspace() else t for t in tokens]
        fixed_line = "".join(fixed_tokens)
        fixed_line = re.sub(r"[ \t]{2,}", " ", fixed_line).strip()
        lines.append(fixed_line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF.
    Strategy:
      1. Try direct text extraction with pdfplumber (fast, perfect quality).
      2. If the PDF is scanned (no selectable text), fall back to EasyOCR.
    """
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"pdfplumber failed on {file_path}: {e}")

    # FIX 2: If no text found, treat as scanned PDF and run OCR page by page
    if not full_text.strip():
        print(f"No selectable text found in {file_path} — running OCR fallback...")
        full_text = _ocr_scanned_pdf(file_path)

    return clean_extracted_text(full_text.strip())


def _ocr_scanned_pdf(file_path: str) -> str:
    """
    Converts each PDF page to an image and runs EasyOCR on it.
    Used as fallback when pdfplumber finds no text.
    """
    try:
        import fitz  # PyMuPDF — add to requirements.txt: pymupdf
        import numpy as np

        reader = _get_ocr_reader()
        doc = fitz.open(file_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page at 2x resolution for better OCR accuracy
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img_array = img_array[:, :, :3]
            preprocessed = _preprocess_image_for_ocr(img_array)
            results = reader.readtext(preprocessed, detail=0)
            all_text.append(" ".join(results))

        return clean_extracted_text("\n".join(all_text))

    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")
        return ""
    except Exception as e:
        print(f"OCR fallback failed: {e}")
        return ""


def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from image files (JPG, PNG) using EasyOCR.
    """
    try:
        import cv2

        reader = _get_ocr_reader()
        image = cv2.imread(file_path)
        image_for_ocr = _preprocess_image_for_ocr(image) if image is not None else file_path

        # detail=1 gives us bounding boxes so we can sort by position
        results = reader.readtext(image_for_ocr, detail=1)

        if not results:
            return ""

        # Sort by vertical position (top to bottom), then horizontal (right to left for Arabic)
        results.sort(key=lambda r: (r[0][0][1], -r[0][0][0]))

        texts = [item[1] for item in results]
        return clean_extracted_text(" ".join(texts))

    except Exception as e:
        print(f"Image OCR failed on {file_path}: {e}")
        return ""


def extract_text_from_excel(file_path: str) -> str:
    """
    Reads an Excel file and converts each sheet to a Markdown table.
    LLMs understand Markdown tables far better than raw CSV strings.
    NEW: Added column type inference to preserve numeric formatting.
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        all_sheets_text = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Drop entirely empty rows and columns
            df = df.dropna(how="all").dropna(axis=1, how="all")

            if df.empty:
                continue

            # NEW: Round floats to 2 decimal places for cleaner LLM input
            df = df.round(2)

            markdown_table = df.to_markdown(index=False)
            all_sheets_text.append(f"### Sheet: {sheet_name}\n{markdown_table}")

        return clean_extracted_text("\n\n".join(all_sheets_text)) if all_sheets_text else ""

    except Exception as e:
        print(f"Excel extraction failed on {file_path}: {e}")
        return ""


def universal_extractor(file_path: str) -> str:
    """
    Main entry point for the ingestion pipeline.
    Routes files to the correct extractor based on extension.
    Returns extracted text, or empty string on failure (never raises).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        return extract_text_from_image(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_from_excel(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""


# ── Unit test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = universal_extractor(sys.argv[1])
        print(f"Extracted {len(result)} characters:")
        print(result[:500])
    else:
        print("Usage: python extractors.py path/to/file.pdf")
