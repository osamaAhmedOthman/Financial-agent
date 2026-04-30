"""
ui/pages/4_history.py

History page — shows all audit sessions from the current session.
In Phase 6 this will pull from a real database.
"""
import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import doc_type_ar

st.set_page_config(page_title="السجل", page_icon="📜", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("يرجى تسجيل الدخول أولاً")
    st.stop()

st.markdown("# 📜 سجل عمليات التدقيق")

history = st.session_state.get("audit_history", [])

if not history:
    st.info("لا توجد عمليات تدقيق سابقة في هذه الجلسة.")
    st.stop()

st.markdown(f"إجمالي العمليات: **{len(history)}**")
st.divider()

for i, entry in enumerate(reversed(history), 1):
    report = entry.get("report", {})
    score = entry.get("score", 0)
    violations = report.get("violations", [])
    critical = sum(1 for v in violations if v.get("severity") == "critical")

    score_color = (
        "#0f9d58" if score >= 90 and critical == 0 else
        "#f4b400" if score >= 70 else
        "#ff6d00" if score >= 50 else
        "#db4437"
    )

    with st.expander(
        f"تقرير #{entry.get('report_id', 'N/A')} — "
        f"نوع: {doc_type_ar(entry.get('doc_type', ''))} — "
        f"النقاط: {score:.0f}/100",
        expanded=(i == 1),
    ):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**رقم التقرير:** `{entry.get('report_id', 'N/A')}`")
        with col2:
            st.markdown(
                f"**النقاط:** <span style='color:{score_color}; font-weight:700;'>{score:.0f}/100</span>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(f"**المخالفات الحرجة:** {critical}")
        with col4:
            st.markdown(f"**إجمالي المخالفات:** {len(violations)}")

        if st.button(f"عرض التقرير الكامل", key=f"view_{i}"):
            st.session_state.audit_report = report
            st.switch_page("pages/3_report.py")
