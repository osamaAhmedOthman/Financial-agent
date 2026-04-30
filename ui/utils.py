"""
ui/utils.py

Shared utilities for UI components.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from html import escape


def doc_type_ar(doc_type: str) -> str:
    """Translate document type to Arabic."""
    mapping = {
        "invoice": "فاتورة",
        "balance_sheet": "ميزانية",
        "tax_return": "إقرار ضريبي",
        "contract": "عقد",
        "unknown": "غير محدد",
    }
    return mapping.get(doc_type, doc_type)


def score_display(score: float, critical: int) -> tuple[str, str]:
    """Get color and label for compliance score display."""
    if score >= 90 and critical == 0:
        return "#0f9d58", "ممتثل"
    if score >= 70 and critical == 0:
        return "#f4b400", "مخالفات بسيطة"
    if score >= 50:
        return "#ff6d00", "مخالفات جوهرية"
    return "#db4437", "غير ممتثل"


def _severity_meta(sev: str) -> tuple[str, str]:
    key = (sev or "info").lower()
    if key == "critical":
        return "critical", "حرج"
    if key == "warning":
        return "warning", "تحذير"
    return "compliant", "ممتثل"


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


def _shape_arabic_text(value: str) -> str:
    """Shape Arabic text and reorder it for correct RTL rendering in PDFs."""
    if value is None:
      return ""

    text = str(value)
    if not _ARABIC_CHAR_RE.search(text):
      return text

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        # Fallback: return original text if shaping libraries are unavailable.
        return text


def _resolve_pdf_font_path() -> str | None:
    """Return first existing Arabic-capable TTF font path."""
    env_font = os.getenv("PDF_ARABIC_FONT_PATH")
    candidates = [
        env_font,
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "Amiri-Regular.ttf"),
        r"C:\Windows\Fonts\amiri-regular.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def generate_pdf_report(report: dict) -> bytes | None:
    """Generate a modern RTL PDF report from an HTML/CSS template."""
    try:
        from xhtml2pdf import pisa

        font_path = _resolve_pdf_font_path()
        if not font_path:
            return None

        score = float(report.get("compliance_score", 0.0))
        report_id = escape(str(report.get("report_id", "N/A")))
        generated_at = report.get("generated_at")

        if generated_at:
            try:
                generated_label = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                generated_label = str(generated_at)
        else:
            generated_label = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        if score >= 90:
            score_class = "compliant"
            score_label = "ملتزم"
        elif score >= 70:
            score_class = "warning"
            score_label = "تحذيرات"
        else:
            score_class = "critical"
            score_label = "مخالفات جوهرية"

        title_ar = _shape_arabic_text("تقرير التدقيق المالي")
        report_number_ar = _shape_arabic_text("رقم التقرير")
        status_ar = _shape_arabic_text("الحالة العامة")
        violations_table_title_ar = _shape_arabic_text("جدول المخالفات")
        no_violations_ar = _shape_arabic_text("لا توجد مخالفات. التقرير متوافق.")
        generated_by_ar = _shape_arabic_text("تم الإنشاء بواسطة Financial Audit AI")
        score_label = _shape_arabic_text(score_label)

        rows = []
        for v in report.get("violations", []):
            sev_class, sev_label = _severity_meta(v.get("severity", "info"))
            rows.append(
                """
                <tr>
                    <td><span class=\"sev {sev_class}\">{sev_label}</span></td>
                    <td>{field}</td>
                    <td>{found}</td>
                    <td>{expected}</td>
                    <td>{legal}</td>
                </tr>
                """.format(
                    sev_class=sev_class,
                    sev_label=escape(_shape_arabic_text(sev_label)),
                    field=escape(_shape_arabic_text(str(v.get("field", "-")))),
                    found=escape(_shape_arabic_text(str(v.get("found_value", "-")))),
                    expected=escape(_shape_arabic_text(str(v.get("expected_value", "-")))),
                    legal=escape(_shape_arabic_text(str(v.get("legal_reference", "-")))),
                )
            )

        table_body = "\n".join(rows) if rows else (
            f"<tr><td colspan='5' class='empty'>{escape(no_violations_ar)}</td></tr>"
        )

        html = f"""
        <html>
          <head>
            <meta charset="utf-8" />
            <style>
              @page {{
                size: A4;
                margin: 22mm 14mm 24mm 14mm;
              }}
              @font-face {{
                font-family: "Amiri";
                src: url("{font_path}");
              }}
              body {{
                font-family: "Amiri";
                direction: rtl;
                unicode-bidi: embed;
                color: #0f172a;
                font-size: 11pt;
                line-height: 1.5;
              }}
              .pdf-container {{
                direction: rtl;
                text-align: right;
              }}
              .header {{
                border: 1px solid #dbe4f0;
                border-radius: 10px;
                background: linear-gradient(120deg, #f5f7fb, #eef7ff);
                padding: 12px 14px;
                margin-bottom: 10px;
              }}
              .header table {{
                width: 100%;
                border-collapse: collapse;
              }}
              .logo {{
                width: 120px;
                height: 44px;
                border: 1px dashed #94a3b8;
                border-radius: 8px;
                text-align: center;
                vertical-align: middle;
                font-size: 9pt;
                color: #475569;
              }}
              .title {{
                font-size: 18pt;
                font-weight: 700;
                color: #0b3b66;
                margin-bottom: 4px;
              }}
              .meta {{
                color: #334155;
                font-size: 10pt;
              }}
              .summary {{
                border: 1px solid #dbe4f0;
                border-radius: 10px;
                background: #f8fafc;
                padding: 10px 12px;
                margin: 12px 0;
              }}
              .summary .score {{
                font-size: 24pt;
                font-weight: 700;
                margin-bottom: 2px;
              }}
              .summary .label {{
                font-size: 10pt;
                color: #334155;
              }}
              .score.critical {{ color: #b91c1c; }}
              .score.warning {{ color: #a16207; }}
              .score.compliant {{ color: #166534; }}
              h3 {{
                color: #0b3b66;
                margin: 10px 0 6px 0;
                font-size: 13pt;
              }}
              table.violations {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                word-wrap: break-word;
              }}
              table.violations th {{
                background: #0f3d68;
                color: #ffffff;
                padding: 7px 6px;
                border: 1px solid #dbe4f0;
                font-size: 10pt;
              }}
              table.violations td {{
                border: 1px solid #dbe4f0;
                padding: 6px;
                vertical-align: top;
                white-space: pre-wrap;
                font-size: 9.5pt;
              }}
              .sev {{
                display: inline-block;
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 9pt;
                font-weight: 700;
              }}
              .sev.critical {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
              .sev.warning {{ background: #fef9c3; color: #854d0e; border: 1px solid #fde68a; }}
              .sev.compliant {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
              .empty {{
                text-align: center;
                color: #166534;
                background: #f0fdf4;
              }}
              .footer {{
                position: fixed;
                bottom: -8mm;
                left: 0;
                right: 0;
                border-top: 1px solid #dbe4f0;
                color: #475569;
                font-size: 9pt;
                padding-top: 5px;
              }}
              .footer .left {{ float: left; direction: ltr; }}
              .footer .right {{ float: right; }}
            </style>
          </head>
          <body>
            <div class="pdf-container">
            <div class="header">
              <table>
                <tr>
                  <td style="width:78%;">
                    <div class="title">{escape(title_ar)}</div>
                    <div class="meta">{escape(report_number_ar)}: {report_id}</div>
                  </td>
                  <td style="width:22%; text-align:left;">
                    <div class="logo">LOGO</div>
                  </td>
                </tr>
              </table>
            </div>

            <div class="summary">
              <div>Compliance Score</div>
              <div class="score {score_class}">{score:.0f}/100</div>
              <div class="label">{escape(status_ar)}: {escape(score_label)}</div>
            </div>

            <h3>{escape(violations_table_title_ar)}</h3>
            <table class="violations">
              <thead>
                <tr>
                  <th style="width:12%;">{escape(_shape_arabic_text("الحالة"))}</th>
                  <th style="width:18%;">{escape(_shape_arabic_text("الحقل"))}</th>
                  <th style="width:20%;">{escape(_shape_arabic_text("القيمة الحالية"))}</th>
                  <th style="width:20%;">{escape(_shape_arabic_text("القيمة المتوقعة"))}</th>
                  <th style="width:30%;">{escape(_shape_arabic_text("المرجع القانوني"))}</th>
                </tr>
              </thead>
              <tbody>
                {table_body}
              </tbody>
            </table>

            <div class="footer">
              <span class="right">{escape(generated_by_ar)}</span>
              <span class="left">{escape(generated_label)}</span>
            </div>
            </div>
          </body>
        </html>
        """

        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(src=html, dest=pdf_buffer, encoding="utf-8")
        if pisa_status.err:
            return None
        return pdf_buffer.getvalue()

    except ImportError:
        return None
    except Exception:
        return None
