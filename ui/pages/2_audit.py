"""
ui/pages/2_audit.py

Audit page — triggers the LangGraph agent on the extracted document.
Shows live node-by-node progress as the agent reasons.
"""
import streamlit as st
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="التدقيق", page_icon="🤖", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("يرجى تسجيل الدخول أولاً")
    st.stop()

st.markdown("# 🤖 تدقيق المستند")


def _doc_type_ar(doc_type: str) -> str:
    return {
        "invoice": "فاتورة",
        "balance_sheet": "ميزانية",
        "tax_return": "إقرار ضريبي",
        "contract": "عقد",
    }.get(doc_type, doc_type)

# ── Guard: need extracted doc ──────────────────────────────────────────────
if not st.session_state.get("extracted_doc"):
    st.warning("لم يتم رفع أي مستند بعد. يرجى الانتقال إلى صفحة **رفع مستند** أولاً.")
    st.stop()

doc = st.session_state.extracted_doc

# ── Document summary before audit ─────────────────────────────────────────
st.markdown("### المستند المراد تدقيقه")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("النوع", _doc_type_ar(doc.doc_type))
with col2:
    st.metric("المبلغ الكلي", f"{doc.total_amount:,.2f} {doc.currency}")
with col3:
    st.metric("الضريبة المذكورة", f"{doc.total_tax:,.2f} {doc.currency}")

# This avoids false warnings when some items are VAT-exempt (0%)
if doc.doc_type == "invoice" and doc.subtotal > 0 and doc.total_tax > 0:
    implied_taxable = doc.total_tax / 0.14
    max_possible_vat = doc.subtotal * 0.14
    # Only warn if the declared tax EXCEEDS what's possible
    # (i.e., they're overcharging VAT on exempt items)
    if doc.total_tax > max_possible_vat + 0.5:
        st.warning(
            f"⚠ الضريبة المذكورة ({doc.total_tax:,.2f}) "
            f"تتجاوز الحد الأقصى الممكن ({max_possible_vat:,.2f}) "
            f"— قد تكون هناك أصناف معفاة محسوبة بضريبة"
        )
    elif doc.total_tax <= max_possible_vat + 0.5:
        exempt = round(doc.subtotal - implied_taxable, 2)
        if exempt > 0.5:
            st.info(
                f"ℹ الفاتورة تحتوي على أصناف معفاة من الضريبة "
                f"({exempt:,.2f} {doc.currency} بمعدل 0%) — "
                f"الحساب صحيح"
            )

st.divider()

# ── Pre-warm knowledge base (once per session) ────────────────────────────
# The BM25 index and embedding model load slowly on first use.
# We cache them in session_state so subsequent audits are fast.
if "kb_warmed" not in st.session_state:
    with st.spinner("تحضير قاعدة المعرفة القانونية (مرة واحدة فقط)..."):
        try:
            from knowledge_base.loader import load_seed_data
            from knowledge_base.retriever import build_bm25_index
            chunks = load_seed_data()
            build_bm25_index(chunks)
            st.session_state.kb_warmed = True
        except Exception as e:
            st.warning(f"تعذر تحميل قاعدة المعرفة: {e}")
            st.session_state.kb_warmed = False

# ── Audit trigger ──────────────────────────────────────────────────────────
col_btn, col_info = st.columns([2, 3])

with col_btn:
    run_audit = st.button(
        "🚀 بدء التدقيق الذكي",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("audit_running", False),
    )

with col_info:
    st.markdown("""
    <div style='background:#f0f9ff; border-left:4px solid #0ea5e9;
                padding:12px 16px; border-radius:0 8px 8px 0;'>
        <strong>ماذا سيحدث؟</strong><br>
        <small>
        1. صياغة أسئلة قانونية من بيانات المستند<br>
        2. البحث في قاعدة قوانين الضرائب المصرية<br>
        3. تحليل المخالفات وإصدار التقرير
        </small>
    </div>
    """, unsafe_allow_html=True)

# ── Run the agent ──────────────────────────────────────────────────────────
if run_audit:
    st.session_state.audit_running = True

    st.markdown("### تقدم عملية التدقيق")

    # Node progress placeholders
    steps = [
        ("query_builder",    "بناء الأسئلة القانونية",        "🔎"),
        ("legal_retriever",  "البحث في القوانين",             "📚"),
        ("web_search",       "البحث على الإنترنت (اختياري)",  "🌐"),
        ("auditor",          "التحليل والمقارنة القانونية",   "⚖"),
        ("report_generator", "توليد التقرير النهائي",         "📊"),
    ]

    step_placeholders = []
    for icon, label, _ in steps:
        ph = st.empty()
        ph.markdown(f"⬜ {_ } {label}")
        step_placeholders.append(ph)

    overall_bar = st.progress(0, text="جارٍ التدقيق...")

    try:
        from agent.auditor import run_audit as agent_run_audit

        # Simulate step-by-step display while agent runs
        # We run the agent in one call but update UI to show progress
        step_placeholders[0].markdown("🔄 🔎 بناء الأسئلة القانونية...")
        overall_bar.progress(10)

        import threading
        result_container = {}
        error_container = {}

        def _run():
            try:
                result_container["report"] = agent_run_audit(doc, verbose=True)
            except Exception as e:
                error_container["error"] = e

        thread = threading.Thread(target=_run)
        thread.start()

        # Animate steps while waiting
        step_delays = [3, 5, 2, 8, 2]  # approximate seconds per step
        cumulative = 0
        total_est = sum(step_delays)

        for i, (_, label, icon) in enumerate(steps):
            elapsed = 0
            while thread.is_alive() and elapsed < step_delays[i]:
                time.sleep(0.5)
                elapsed += 0.5
                cumulative += 0.5
                pct = min(int((cumulative / total_est) * 90) + 5, 90)
                overall_bar.progress(pct)

            if thread.is_alive() or i < len(steps) - 1:
                step_placeholders[i].markdown(f"✅ {icon} {label}")
                if i + 1 < len(steps):
                    step_placeholders[i + 1].markdown(f"🔄 {steps[i+1][2]} {steps[i+1][1]}...")

        thread.join(timeout=120)

        if "error" in error_container:
            raise error_container["error"]

        report = result_container.get("report")
        if not report:
            raise RuntimeError("لم يُنتج الوكيل أي تقرير")

        # Mark all done
        for i, (_, label, icon) in enumerate(steps):
            step_placeholders[i].markdown(f"✅ {icon} {label}")

        overall_bar.progress(100, text="اكتمل التدقيق ✓")

        # Save to session + history
        report_dict = report.model_dump()
        st.session_state.audit_report = report_dict
        st.session_state.audit_history.append({
            "doc_type":  doc.doc_type,
            "score":     report_dict.get("compliance_score", 0),
            "report_id": report_dict.get("report_id", ""),
            "report":    report_dict,
        })

        st.success(f"✅ تم التدقيق! رقم التقرير: **{report_dict.get('report_id')}**")
        st.info("انتقل إلى صفحة **التقرير** لعرض النتائج التفصيلية.")

    except Exception as e:
        overall_bar.progress(0)
        st.error(f"فشل التدقيق: {e}")
        st.exception(e)
    finally:
        st.session_state.audit_running = False

# ── Show previous result if exists ────────────────────────────────────────
elif st.session_state.get("audit_report"):
    report = st.session_state.audit_report
    score = report.get("compliance_score", 0)
    st.success(f"آخر تدقيق: تقرير **{report.get('report_id')}** — نقاط الامتثال: **{score:.0f}/100**")
    st.info("انتقل إلى صفحة **التقرير** لعرض التفاصيل، أو اضغط التدقيق مرة أخرى لإعادة التحليل.")
