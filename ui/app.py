"""
ui/app.py

Main entry point for the AI Financial Auditor Streamlit UI.
Run with: streamlit run ui/app.py

Pages:
  1_upload.py    — Upload financial documents
  2_audit.py     — Run the LangGraph audit agent
  3_report.py    — View the structured audit report
  4_history.py   — Past audit sessions
"""
import streamlit as st
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="المدقق المالي الذكي",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Arabic-friendly font */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2744 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Main background */
    .stApp { background: #f7f9fc; }

    /* Cards */
    .audit-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Compliance score colors */
    .score-compliant     { color: #0f9d58; font-weight: 700; }
    .score-minor         { color: #f4b400; font-weight: 700; }
    .score-major         { color: #ff6d00; font-weight: 700; }
    .score-noncompliant  { color: #db4437; font-weight: 700; }

    /* Violation severity badges */
    .badge-critical { background:#fde8e8; color:#c53030; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .badge-warning  { background:#fef3c7; color:#92400e; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .badge-info     { background:#e0f2fe; color:#075985; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }

    /* Hide Streamlit default menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }

    /* Arabic text alignment */
    .arabic { direction: rtl; text-align: right; }

    /* Progress steps */
    .step-done    { color: #0f9d58; }
    .step-active  { color: #1a73e8; font-weight: 600; }
    .step-pending { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "audit_report" not in st.session_state:
    st.session_state.audit_report = None
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 المدقق المالي الذكي")
    st.markdown("*AI Financial Auditor*")
    st.divider()

    # Auth check
    if not st.session_state.authenticated:
        st.markdown("### تسجيل الدخول")
        with st.form("login_form"):
            username = st.text_input("اسم المستخدم", placeholder="admin")
            password = st.text_input("كلمة المرور", type="password", placeholder="••••••")
            login_btn = st.form_submit_button("دخول", use_container_width=True)

            if login_btn:
                # Simple auth — replace with JWT in Phase 6
                USERS = {"admin": "admin123", "auditor": "audit456"}
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("بيانات خاطئة")
    else:
        st.success(f"مرحباً، {st.session_state.username}")

        # Pipeline status
        st.markdown("### حالة المهمة")
        doc_done = st.session_state.extracted_doc is not None
        rep_done = st.session_state.audit_report is not None

        st.markdown(f"""
        {'✅' if doc_done else '○'} **رفع المستند**
        {'✅' if rep_done else ('🔄 جاهز' if doc_done else '○')} **التدقيق**
        {'✅' if rep_done else '○'} **التقرير**
        """)

        st.divider()

        # Quick stats
        if rep_done:
            report = st.session_state.audit_report
            score = report.get("compliance_score", 0)
            color = (
                "score-compliant" if score >= 90 else
                "score-minor" if score >= 70 else
                "score-major" if score >= 50 else
                "score-noncompliant"
            )
            st.markdown(f"**آخر نتيجة:** <span class='{color}'>{score:.0f}/100</span>",
                        unsafe_allow_html=True)
            violations = report.get("violations", [])
            critical = sum(1 for v in violations if v.get("severity") == "critical")
            if critical:
                st.error(f"⚠ {critical} مخالفة حرجة")

        st.divider()

        if st.button("تسجيل الخروج", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.extracted_doc = None
            st.session_state.audit_report = None
            st.rerun()

# ── Main content ───────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:60px 0;'>
            <h1 style='font-size:48px;'>📊</h1>
            <h2>المدقق المالي الذكي</h2>
            <p style='color:#64748b; font-size:16px;'>
                نظام ذكي لتدقيق الفواتير والمستندات المالية<br>
                وفقاً لقوانين الضرائب المصرية
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("## مرحباً بك في نظام التدقيق المالي الذكي")
    st.markdown("استخدم القائمة الجانبية للتنقل بين الصفحات، أو ابدأ بـ **رفع مستند** من القائمة أعلاه.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='audit-card' style='text-align:center;'>
            <h2>📄</h2>
            <h4>ارفع مستندك</h4>
            <p style='color:#64748b;'>PDF · Excel · صورة</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='audit-card' style='text-align:center;'>
            <h2>🤖</h2>
            <h4>تدقيق ذكي</h4>
            <p style='color:#64748b;'>مراجعة قانون 67/2016</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='audit-card' style='text-align:center;'>
            <h2>📊</h2>
            <h4>تقرير مفصل</h4>
            <p style='color:#64748b;'>مخالفات · توصيات · نقاط</p>
        </div>
        """, unsafe_allow_html=True)
