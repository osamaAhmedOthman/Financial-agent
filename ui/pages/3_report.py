"""
ui/pages/3_report.py

Report viewer — renders the AuditReport as a visual, structured page.
Includes compliance score gauge, violations table, legal citations,
recommendations checklist, and PDF export.
"""
import streamlit as st
import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import doc_type_ar, score_display, generate_pdf_report

st.set_page_config(page_title="التقرير", page_icon="📊", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("يرجى تسجيل الدخول أولاً")
    st.stop()

if not st.session_state.get("audit_report"):
    st.warning("لا يوجد تقرير بعد. يرجى تشغيل التدقيق أولاً.")
    st.stop()

report = st.session_state.audit_report
violations = report.get("violations", [])
recommendations = report.get("recommendations", [])
score = report.get("compliance_score", 0)
level = report.get("compliance_level", "unknown")
critical_count = sum(1 for v in violations if v.get("severity") == "critical")
warning_count  = sum(1 for v in violations if v.get("severity") == "warning")

# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 2])
with col_title:
    st.markdown(f"# 📊 تقرير التدقيق المالي")
    st.markdown(f"رقم التقرير: **{report.get('report_id', 'N/A')}** · "
                f"نوع المستند: **{doc_type_ar(report.get('document_type', ''))}**")
with col_meta:
    generated = report.get("generated_at", "")
    if generated:
        try:
            dt = datetime.fromisoformat(generated)
            st.markdown(f"📅 {dt.strftime('%Y-%m-%d %H:%M')} UTC")
        except:
            pass
    if report.get("web_search_used"):
        st.markdown("🌐 *تم استخدام البحث الإلكتروني*")
    if report.get("requires_human_review"):
        st.error("⚠ هذا التقرير يحتاج مراجعة بشرية")

st.divider()

# ── Compliance score ───────────────────────────────────────────────────────
st.markdown("## نقاط الامتثال")

col_score, col_breakdown, col_conf = st.columns([2, 3, 2])

with col_score:
    score_color, score_label = score_display(score, critical_count)
    st.markdown(f"""
    <div style='text-align:center; padding:24px; background:white;
                border-radius:12px; border:2px solid {score_color};'>
        <div style='font-size:52px; font-weight:700; color:{score_color};'>{score:.0f}</div>
        <div style='font-size:14px; color:#64748b;'>من 100</div>
        <div style='font-size:16px; font-weight:600; color:{score_color}; margin-top:8px;'>
            {score_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_breakdown:
    st.markdown("**تفصيل المخالفات**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("مخالفات حرجة", critical_count,
                  delta=None, delta_color="off")
    with c2:
        st.metric("تحذيرات", warning_count)
    with c3:
        info_count = len(violations) - critical_count - warning_count
        st.metric("ملاحظات", info_count)

    # Score bar
    bar_html = f"""
    <div style='margin-top:12px;'>
        <div style='display:flex; height:10px; border-radius:5px; overflow:hidden;
                    background:#e2e8f0;'>
            <div style='width:{score}%; background:{score_color}; transition:width 1s;'></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

with col_conf:
    conf = report.get("agent_confidence", 0)
    st.markdown("**ثقة الذكاء الاصطناعي**")
    st.markdown(f"""
    <div style='text-align:center; padding:20px; background:#f8fafc;
                border-radius:10px; border:1px solid #e2e8f0;'>
        <div style='font-size:32px; font-weight:700; color:#1e40af;'>{conf:.0%}</div>
        <div style='font-size:12px; color:#64748b;'>دقة التحليل</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Legal summary ──────────────────────────────────────────────────────────
legal_summary = report.get("legal_summary", "")
if legal_summary:
    st.markdown("## ملخص التدقيق")
    st.markdown(f"""
    <div class='audit-card' style='direction:rtl; text-align:right; border-left:4px solid #3b82f6;'>
        {legal_summary}
    </div>
    """, unsafe_allow_html=True)

# ── Violations ─────────────────────────────────────────────────────────────
st.markdown("## المخالفات المكتشفة")

if not violations:
    st.success("🎉 لا توجد مخالفات! المستند متوافق مع القانون المصري.")
else:
    for i, v in enumerate(violations, 1):
        sev = v.get("severity", "info")
        badge_class = f"badge-{sev}"
        badge_text  = {"critical": "حرجة", "warning": "تحذير", "info": "ملاحظة"}.get(sev, sev)
        border_color = {"critical": "#dc2626", "warning": "#f59e0b", "info": "#3b82f6"}.get(sev, "#94a3b8")

        with st.expander(
            f"{'🔴' if sev=='critical' else '🟡' if sev=='warning' else '🔵'} "
            f"مخالفة {i}: {v.get('field', '')}",
            expanded=(sev == "critical"),
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**الخطورة:** <span class='{badge_class}'>{badge_text}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**الحقل:** `{v.get('field', '')}`")
                st.markdown(f"**القيمة الموجودة:** `{v.get('found_value', '')}`")
                st.markdown(f"**القيمة المطلوبة:** `{v.get('expected_value', '')}`")
            with col_b:
                st.markdown("**الشرح:**")
                st.markdown(f"> {v.get('explanation', '')}")
                st.markdown(f"**المرجع القانوني:** `{v.get('legal_reference', '')}`")

st.divider()

# ── Recommendations ────────────────────────────────────────────────────────
if recommendations:
    st.markdown("## التوصيات والإجراءات المطلوبة")
    for i, rec in enumerate(recommendations, 1):
        col_check, col_text = st.columns([1, 12])
        with col_check:
            st.checkbox("", key=f"rec_{i}")
        with col_text:
            st.markdown(rec)

st.divider()

# ── Laws consulted ─────────────────────────────────────────────────────────
laws = report.get("laws_consulted", [])
if laws:
    st.markdown("## القوانين التي تم الرجوع إليها")
    law_names = {
        "VAT_67_2016":                  "قانون القيمة المضافة رقم 67 لسنة 2016",
        "INCOME_TAX_91_2005":           "قانون الضريبة على الدخل رقم 91 لسنة 2005",
        "UNIFIED_PROCEDURES_206_2020":  "قانون الإجراءات الضريبية الموحد رقم 206 لسنة 2020",
    }
    cols = st.columns(min(len(laws), 3))
    for i, law in enumerate(laws):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div style='background:#f0f9ff; border:1px solid #bae6fd;
                        padding:10px 14px; border-radius:8px; font-size:13px;'>
                📖 {law_names.get(law, law)}
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Export ─────────────────────────────────────────────────────────────────
st.markdown("## تصدير التقرير")

col_json, col_pdf = st.columns(2)

with col_json:
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        label="⬇ تحميل JSON",
        data=report_json,
        file_name=f"audit_report_{report.get('report_id', 'export')}.json",
        mime="application/json",
        use_container_width=True,
    )

with col_pdf:
    pdf_bytes = generate_pdf_report(report)
    if pdf_bytes:
        st.download_button(
            label="⬇ تحميل PDF",
            data=pdf_bytes,
            file_name=f"audit_report_{report.get('report_id', 'export')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info(
            "لتفعيل تصدير PDF ثبّت الحزم التالية:\n"
            "`pip install xhtml2pdf`\n"
            "ثم عرّف خط TTF عربي عبر المتغير `PDF_ARABIC_FONT_PATH` "
            "(مثال: `C:/Windows/Fonts/arial.ttf` أو مسار `Amiri-Regular.ttf`)."
        )
