"""
ui/pages/1_upload.py

Upload page — accepts PDF, Excel, and images.
Runs the Phase 1 ingestion pipeline and previews extracted data.
"""
import streamlit as st
import sys, os, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import doc_type_ar

st.set_page_config(page_title="رفع مستند", page_icon="📄", layout="wide")

# Auth guard
if not st.session_state.get("authenticated"):
    st.error("يرجى تسجيل الدخول أولاً")
    st.stop()

st.markdown("# 📄 رفع مستند مالي")
st.markdown("ارفع فاتورة، ميزانية، أو إقراراً ضريبياً للتدقيق.")

# ── File uploader ──────────────────────────────────────────────────────────
col_upload, col_info = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "اختر ملفاً",
        type=["pdf", "xlsx", "xls", "jpg", "jpeg", "png"],
        help="الصيغ المدعومة: PDF، Excel، JPG، PNG",
    )

with col_info:
    st.markdown("""
    <div class='audit-card'>
        <h4>📋 ما يمكنك رفعه</h4>
        <ul>
            <li>فواتير ضريبية (PDF أو صورة)</li>
            <li>ميزانيات عمومية (Excel)</li>
            <li>إقرارات ضريبية (PDF)</li>
            <li>عقود تجارية (PDF)</li>
        </ul>
        <p style='color:#64748b; font-size:13px;'>
            الحد الأقصى للحجم: 10 MB
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Process uploaded file ──────────────────────────────────────────────────
if uploaded_file is not None:
    st.divider()
    st.markdown(f"**الملف:** `{uploaded_file.name}` — {uploaded_file.size / 1024:.1f} KB")

    process_btn = st.button("🔍 استخراج البيانات", type="primary", use_container_width=True)

    if process_btn:
        with st.spinner("جارٍ قراءة المستند..."):
            # Save to temp file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # استيراد محرك المعالجة
                from ingestion.processors import process_document_pipeline
                
                # استيراد الخطأ أو تعريفه محلياً لضمان عدم توقف البرنامج
                try:
                    from ingestion.processors import ExtractionError
                except ImportError:
                    class ExtractionError(Exception): pass

                progress = st.progress(0, text="استخراج النص...")
                time.sleep(0.3)
                progress.progress(30, text="تصنيف المستند...")

                # تنفيذ عملية المعالجة
                extracted = process_document_pipeline(tmp_path)

                progress.progress(70, text="التحقق من البيانات...")
                time.sleep(0.2)
                progress.progress(100, text="اكتمل الاستخراج ✓")

                st.session_state.extracted_doc = extracted
                st.session_state.audit_report = None  # إعادة ضبط تقرير التدقيق السابق

                st.success("تم استخراج البيانات بنجاح!")

            except Exception as e:
                # فحص نوع الخطأ بطريقة مرنة (Dynamic Check)
                # هذا يحل مشكلة عدم تطابق الكلاسات في بيئة Kaggle
                error_name = type(e).__name__
                
                if error_name == "ExtractionError":
                    st.error(f"⚠️ تعذر قراءة الملف: {e}")
                else:
                    st.error(f"❌ خطأ غير متوقع: {e}")
                    # لسهولة الإصلاح، سنطبع الخطأ في الـ Terminal الخاص بكاجل
                    import traceback
                    print(f"DEBUG: Error type: {error_name}")
                    print(traceback.format_exc())
                
                st.stop()
            finally:
                # التأكد من مسح الملف المؤقت دائماً
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ── Preview extracted data ─────────────────────────────────────────────────
if st.session_state.get("extracted_doc"):
    doc = st.session_state.extracted_doc
    st.divider()
    st.markdown("## نتيجة الاستخراج")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("نوع المستند", doc_type_ar(doc.doc_type))
    with col2:
        st.metric("المبلغ الأساسي", f"{doc.subtotal:,.2f} {doc.currency}")
    with col3:
        st.metric("إجمالي الضريبة", f"{doc.total_tax:,.2f} {doc.currency}")
    with col4:
        confidence_pct = f"{doc.extraction_confidence:.0%}"
        st.metric("دقة الاستخراج", confidence_pct)

    # Validation status
    if doc.validation_status:
        st.success("✅ الأرقام متسقة رياضياً (المجموع + الضريبة = الإجمالي)")
    else:
        st.warning("⚠ تناقض في الأرقام — قد يكون الاستخراج غير مكتمل")

    if doc.requires_human_review:
        st.warning("🔍 هذا المستند يحتاج مراجعة بشرية بسبب انخفاض الدقة أو وجود بيانات غير مكتملة")

    # Details
    with st.expander("تفاصيل المستند المستخرجة"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**البيانات الأساسية**")
            st.json({
                "نوع المستند":    doc.doc_type,
                "اسم الجهة":      doc.vendor_name or "غير محدد",
                "الرقم الضريبي":  doc.tax_id or "غير موجود",
                "العملة":         doc.currency,
                "التاريخ":        doc.date or "غير محدد",
            })
        with col_b:
            st.markdown("**الأرقام المالية**")
            st.json({
                "المبلغ الأساسي":     doc.subtotal,
                "إجمالي الضريبة":     doc.total_tax,
                "الإجمالي الكلي":     doc.total_amount,
                "صحة الأرقام":       doc.validation_status,
                "دقة الاستخراج":     doc.extraction_confidence,
            })

    # Line items if present
    if doc.line_items:
        st.markdown("**بنود الفاتورة**")
        import pandas as pd
        items_data = [
            {
                "البيان":      item.description,
                "الكمية":      item.quantity,
                "سعر الوحدة":  item.unit_price,
                "الإجمالي":    item.total_price,
                "الضريبة %":   item.tax_rate,
            }
            for item in doc.line_items
        ]
        st.dataframe(pd.DataFrame(items_data), use_container_width=True)

    # Raw text preview
    cleaned_preview = getattr(doc, "cleaned_raw_text", None) or ""
    with st.expander("النص المنقح المستخرج"):
        if cleaned_preview:
            st.text(cleaned_preview[:1500] + ("..." if len(cleaned_preview) > 1500 else ""))
        else:
            st.info("لا يوجد نص منقح حالياً. سيتم عرض النص الخام.")
            st.text(doc.raw_text[:1000] + ("..." if len(doc.raw_text) > 1000 else ""))

    with st.expander("النص الخام المستخرج"):
        st.text(doc.raw_text[:1000] + ("..." if len(doc.raw_text) > 1000 else ""))

    st.divider()
    st.info("✅ المستند جاهز للتدقيق — انتقل إلى صفحة **التدقيق** من القائمة العلوية")
