"""
QMed Quantum — Consumer Application
Early Cancer Detection powered by Quantum-Classical AI Ensemble

Streamlit consumer app. Backend pipeline, trained models, OCR, and validator
are identical to the SIH 2026 prototype — only the UI is rebuilt for end-users.

Run:  streamlit run app.py
"""
import os, sys, json, io, time, hashlib
import numpy as np
import pandas as pd
import streamlit as st

# ─── path setup (keep backend intact) ───────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "src"),
          os.path.join(HERE, "src", "ocr"),
          os.path.join(HERE, "src", "validator"),
          os.path.join(HERE, "src", "multicancer"),
          os.path.join(HERE, "src", "ensemble"),
          os.path.join(HERE, "src", "selflearn")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import analyze
from ocr_pipeline import extract_fields
from unified_score import _load as load_bundle
from feedback import log_feedback

# ─── page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="QMed Quantum · AI Cancer Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CANCERS = ["breast", "lung", "prostate", "colon", "cervical"]
CANCER_ICONS = {"breast": "🎗️", "lung": "🫁", "prostate": "🩺", "colon": "🔬", "cervical": "🌸"}
CANCER_COLORS = {"breast": "#FF6B9D", "lung": "#64B5F6", "prostate": "#81C784", "colon": "#FFB74D", "cervical": "#CE93D8"}

# ─── premium CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ── */
:root {
    --bg-primary: #06080f;
    --bg-secondary: #0c1018;
    --bg-card: rgba(14, 20, 35, 0.65);
    --bg-glass: rgba(16, 24, 44, 0.55);
    --border: rgba(56, 82, 130, 0.25);
    --border-glow: rgba(100, 149, 237, 0.2);
    --text-primary: #e8edf5;
    --text-secondary: #8896b3;
    --text-muted: #5a6a88;
    --accent-blue: #4f8ef7;
    --accent-cyan: #22d3ee;
    --accent-purple: #a78bfa;
    --accent-green: #34d399;
    --accent-red: #f87171;
    --accent-amber: #fbbf24;
    --radius: 16px;
    --radius-sm: 10px;
    --radius-xs: 6px;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}
h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.02em;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(175deg, #060a14 0%, #0a0f1f 30%, #080c18 70%, #060a14 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79, 142, 247, 0.25); border-radius: 3px; }

/* ── Nav bar ── */
.nav-container {
    display: flex; align-items: center; gap: 6px;
    padding: 6px; background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 14px; backdrop-filter: blur(20px);
    margin-bottom: 28px;
}
.nav-btn {
    padding: 10px 20px; border-radius: 10px;
    font-size: 14px; font-weight: 500;
    color: var(--text-secondary); cursor: pointer;
    transition: all 0.25s ease; border: none;
    background: transparent; text-decoration: none;
}
.nav-btn:hover { color: var(--text-primary); background: rgba(79, 142, 247, 0.08); }
.nav-btn.active {
    background: linear-gradient(135deg, rgba(79, 142, 247, 0.15), rgba(34, 211, 238, 0.08));
    color: var(--accent-blue); border: 1px solid rgba(79, 142, 247, 0.25);
    font-weight: 600; box-shadow: 0 0 20px rgba(79, 142, 247, 0.08);
}

/* ── Logo ── */
.logo-area {
    display: flex; align-items: center; gap: 14px;
    padding: 0 14px; margin-right: auto;
}
.logo-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #4f8ef7, #22d3ee);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 4px 16px rgba(79, 142, 247, 0.3);
}
.logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.01em;
}
.logo-sub {
    font-size: 11px; color: var(--text-muted);
    font-weight: 400; letter-spacing: 0.02em;
}

/* ── Section hero ── */
.section-hero {
    padding: 40px 0 20px 0;
}
.section-hero h1 {
    font-size: 38px; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #e8edf5 0%, #8bb3f7 60%, #22d3ee 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.15;
}
.section-hero p {
    color: var(--text-secondary); font-size: 16px; margin-top: 8px;
    line-height: 1.5; max-width: 660px;
}

/* ── Glass card ── */
.glass {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px; backdrop-filter: blur(16px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover {
    border-color: var(--border-glow);
    box-shadow: 0 4px 30px rgba(79, 142, 247, 0.06);
}

/* ── KPI metric card ── */
.kpi-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px; backdrop-filter: blur(16px);
    text-align: center;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px; font-weight: 700; color: var(--text-primary);
    line-height: 1.1;
}
.kpi-label {
    font-size: 11.5px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-top: 6px;
}

/* ── Risk bands ── */
.risk-high { border-left: 4px solid var(--accent-red); }
.risk-high .kpi-value { color: var(--accent-red); }
.risk-moderate { border-left: 4px solid var(--accent-amber); }
.risk-moderate .kpi-value { color: var(--accent-amber); }
.risk-low { border-left: 4px solid var(--accent-green); }
.risk-low .kpi-value { color: var(--accent-green); }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed rgba(79, 142, 247, 0.2);
    border-radius: var(--radius);
    padding: 44px 24px; text-align: center;
    background: rgba(79, 142, 247, 0.03);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: rgba(79, 142, 247, 0.35);
    background: rgba(79, 142, 247, 0.06);
}
.upload-icon { font-size: 42px; margin-bottom: 12px; }
.upload-title { font-size: 17px; font-weight: 600; color: var(--text-primary); }
.upload-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* ── Recommendation banner ── */
.rec-banner {
    border-radius: var(--radius);
    padding: 20px 24px;
    backdrop-filter: blur(16px);
}
.rec-high {
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.08), rgba(248, 113, 113, 0.03));
    border: 1px solid rgba(248, 113, 113, 0.2);
}
.rec-moderate {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.03));
    border: 1px solid rgba(251, 191, 36, 0.2);
}
.rec-low {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.08), rgba(52, 211, 153, 0.03));
    border: 1px solid rgba(52, 211, 153, 0.2);
}

/* ── Model bar ── */
.model-bar-wrap {
    margin: 8px 0;
}
.model-bar-label {
    display: flex; justify-content: space-between;
    font-size: 13px; margin-bottom: 4px;
}
.model-bar-name { color: var(--text-secondary); font-weight: 500; }
.model-bar-val { color: var(--text-primary); font-weight: 600; font-family: 'Space Grotesk'; }
.model-bar-track {
    height: 8px; border-radius: 4px;
    background: rgba(255,255,255,0.04);
    overflow: hidden;
}
.model-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s ease;
}
.bar-quantum { background: linear-gradient(90deg, #a78bfa, #7c3aed); }
.bar-classical { background: linear-gradient(90deg, #4f8ef7, #22d3ee); }

/* ── Pill/chip ── */
.chip {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 11.5px; font-weight: 600; margin-right: 6px;
    margin-bottom: 4px;
}
.chip-quantum { background: rgba(167, 139, 250, 0.12); color: #c4b5fd; border: 1px solid rgba(167, 139, 250, 0.25); }
.chip-classical { background: rgba(79, 142, 247, 0.12); color: #93bbfd; border: 1px solid rgba(79, 142, 247, 0.25); }
.chip-cancer { background: rgba(34, 211, 238, 0.08); color: #67e8f9; border: 1px solid rgba(34, 211, 238, 0.2); }
.chip-ok { background: rgba(52, 211, 153, 0.1); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.2); }
.chip-warn { background: rgba(251, 191, 36, 0.1); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.2); }

/* ── Tabs (streamlit) ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-glass); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 10px 22px;
    color: var(--text-secondary); font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(79, 142, 247, 0.12) !important;
    border-color: rgba(79, 142, 247, 0.3) !important;
    color: var(--accent-blue) !important;
}

/* ── Misc overrides ── */
.stButton > button {
    border-radius: var(--radius-sm); font-weight: 600;
    border: 1px solid rgba(79, 142, 247, 0.3);
    background: linear-gradient(135deg, rgba(79, 142, 247, 0.12), rgba(34, 211, 238, 0.06));
    color: var(--accent-blue); transition: all 0.25s ease;
    padding: 10px 28px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(79, 142, 247, 0.22), rgba(34, 211, 238, 0.12));
    border-color: rgba(79, 142, 247, 0.5);
    box-shadow: 0 4px 20px rgba(79, 142, 247, 0.15);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f8ef7, #22d3ee);
    color: #fff; border: none;
    box-shadow: 0 4px 20px rgba(79, 142, 247, 0.25);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(79, 142, 247, 0.35);
    transform: translateY(-1px);
}
div[data-testid="stFileUploader"] {
    border: none; background: transparent;
}
div[data-testid="stTextArea"] textarea {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text-primary);
}
div[data-testid="stSelectbox"] > div { border-radius: var(--radius-sm); }
.stRadio > div { gap: 8px; }
.stExpander { border: 1px solid var(--border); border-radius: var(--radius-sm); }

/* ── Metric card (Streamlit native) ── */
div[data-testid="stMetric"] {
    background: var(--bg-glass); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 16px;
    backdrop-filter: blur(12px);
}

/* ── Table / dataframe ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); overflow: hidden; }

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 0.4; }

/* ── Disclaimer ── */
.disclaimer {
    font-size: 12px; color: var(--text-muted);
    padding: 16px 20px; border-radius: var(--radius-sm);
    background: rgba(248, 113, 113, 0.04);
    border: 1px solid rgba(248, 113, 113, 0.1);
    margin-top: 20px; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Navigation ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "analyze"

def nav_to(p):
    st.session_state.page = p


# Render nav bar
cols = st.columns([3.5, 1, 1, 1])
with cols[0]:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-icon">🧬</div>
        <div>
            <div class="logo-text">QMed Quantum</div>
            <div class="logo-sub">AI Cancer Screening</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    if st.button("🔍  Analyze Report", key="nav_analyze", use_container_width=True):
        nav_to("analyze")
with cols[2]:
    if st.button("📊  Performance", key="nav_perf", use_container_width=True):
        nav_to("performance")
with cols[3]:
    if st.button("⚙️  Settings", key="nav_settings", use_container_width=True):
        nav_to("settings")

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

page = st.session_state.page


# ═════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE REPORT
# ═════════════════════════════════════════════════════════════════════
if page == "analyze":
    st.markdown("""
    <div class="section-hero">
        <h1>Analyze Your Medical Report</h1>
        <p>Upload a clinical report, pathology findings, or lab results — our 
        AI ensemble of 6 models including hybrid quantum classifiers will 
        analyze it in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input section ──
    tab_upload, tab_paste, tab_demo = st.tabs(["📤 Upload File", "📝 Paste Text", "📁 Try Demo"])

    src = None
    src_kind = "auto"
    src_label = ""

    with tab_upload:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📎</div>
            <div class="upload-title">Drag & drop your medical report</div>
            <div class="upload-sub">PDF · TXT · PNG · JPG · TIFF — plain text, digital PDF, or scanned image</div>
        </div>
        """, unsafe_allow_html=True)
        f = st.file_uploader("Upload report",
                             type=["txt","pdf","png","jpg","jpeg","tif","tiff"],
                             label_visibility="collapsed")
        if f is not None:
            src = f.read()
            ext = os.path.splitext(f.name)[1].lower()
            src_kind = ("text" if ext == ".txt" else
                        "pdf"  if ext == ".pdf" else "image")
            src_label = f.name
            st.markdown(f"""
            <div class="glass" style="margin-top:12px; padding:14px 20px">
                <span class="chip chip-ok">✓ Loaded</span>
                <span style="color: var(--text-primary); font-weight:500">{f.name}</span>
                <span style="color: var(--text-muted); font-size:13px; margin-left:8px">
                    {len(src):,} bytes · {src_kind}
                </span>
            </div>
            """, unsafe_allow_html=True)
            if src_kind == "image":
                st.image(src, caption="Uploaded scan preview", width=460)

    with tab_paste:
        txt = st.text_area("Paste your report text below", height=200,
                           placeholder="Example: PSA 8.4 ng/mL, Free/Total 0.11, DRE suspicious, Gleason 7, Hemoglobin 13.1 g/dL ...")
        if st.button("Analyze pasted text", type="primary", key="btn_paste"):
            if txt.strip():
                src, src_kind = txt, "text"
                src_label = "pasted text"

    with tab_demo:
        st.markdown("""
        <div class="glass" style="padding: 16px 20px; margin-bottom: 12px">
            <span style="color: var(--text-secondary); font-size: 14px">
                Don't have a report? Try one of our sample clinical reports below.
            </span>
        </div>
        """, unsafe_allow_html=True)
        samples = {
            "🎗️ Breast cancer report":    ("data/samples/breast_report.txt",   "text"),
            "🫁 Lung cancer report":      ("data/samples/lung_report.txt",     "text"),
            "🩺 Prostate cancer report":  ("data/samples/prostate_report.txt", "text"),
            "🔬 Colon cancer report":     ("data/samples/colon_report.txt",    "text"),
            "🌸 Cervical cancer report":  ("data/samples/cervical_report.txt", "text"),
        }
        pick = st.selectbox("Select a sample report", list(samples.keys()), label_visibility="collapsed")
        if st.button("Analyze sample", type="primary", key="btn_sample"):
            rel, kind = samples[pick]
            src = os.path.join(HERE, rel)
            src_kind = kind
            src_label = pick

    # ── Cancer type override ──
    override = st.selectbox("Cancer type",
                            ["Auto-detect (AI router)", *[f"{CANCER_ICONS[c]} {c.capitalize()}" for c in CANCERS]],
                            help="Leave on Auto-detect to let the AI router determine the cancer type from your report.")
    if override == "Auto-detect (AI router)":
        override_cancer = None
    else:
        override_cancer = override.split(" ", 1)[1].lower()

    # ── Run analysis ──
    if src is not None:
        with st.spinner("Analyzing report — running OCR → validation → cancer routing → 6 AI models…"):
            t0 = time.time()
            r = analyze(src, src_kind, override_cancer=override_cancer)
            elapsed = time.time() - t0

        # ── REJECTED ──
        if r["status"] == "rejected":
            st.markdown(f"""
            <div class="glass" style="border-left: 4px solid var(--accent-red); margin-top: 20px">
                <h3 style="margin:0; color: var(--accent-red)">❌ Document Not Recognized</h3>
                <p style="color: var(--text-secondary); margin-top: 8px; font-size: 14px">
                    {r['reason']}
                </p>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px">
                    💡 {r['hint']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🔍 Validator details"):
                st.json(r["validation"])
            st.stop()

        # ── SUCCESS ──
        s = r["summary"]
        p = r["prediction"]

        # timing chip
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin: 16px 0">
            <span class="chip chip-ok">✓ Complete</span>
            <span style="color: var(--text-muted); font-size: 13px">
                Analyzed in {elapsed:.2f}s · Source: {src_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ──
        risk_cls = {"HIGH RISK": "risk-high", "MODERATE RISK": "risk-moderate", "LOW RISK": "risk-low"}.get(s["risk_band"], "")
        risk_color = {"HIGH RISK": "var(--accent-red)", "MODERATE RISK": "var(--accent-amber)", "LOW RISK": "var(--accent-green)"}.get(s["risk_band"], "var(--text-primary)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card {risk_cls}">
                <div class="kpi-value" style="color:{risk_color}">{s['final_score']:.1%}</div>
                <div class="kpi-label">Risk Score</div>
                <div style="margin-top:8px; font-weight:600; font-size:13px; color:{risk_color}">{s['risk_band']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            stage_txt = "N/A" if s["predicted_stage"] is None else f"Stage {s['predicted_stage']}"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{stage_txt}</div>
                <div class="kpi-label">Predicted Stage</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            early_color = "var(--accent-red)" if s["early_risk_score"] >= 0.6 else ("var(--accent-amber)" if s["early_risk_score"] >= 0.3 else "var(--accent-green)")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{early_color}">{s['early_risk_score']:.1%}</div>
                <div class="kpi-label">Early Risk (Pre-cancer)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            cancer_icon = CANCER_ICONS.get(s["cancer"], "🔬")
            cancer_color = CANCER_COLORS.get(s["cancer"], "#67e8f9")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size:28px; color:{cancer_color}">{cancer_icon} {s['cancer'].upper()}</div>
                <div class="kpi-label">Detected Cancer Type</div>
            </div>""", unsafe_allow_html=True)

        # ── Recommendation banner ──
        rec_cls = {"HIGH RISK": "rec-high", "MODERATE RISK": "rec-moderate", "LOW RISK": "rec-low"}.get(s["risk_band"], "")
        st.markdown(f"""
        <div class="rec-banner {rec_cls}" style="margin-top: 20px">
            <div style="display:flex; align-items:center; gap: 10px; margin-bottom: 8px">
                <span style="font-size: 22px">{'🚨' if 'HIGH' in s['risk_band'] else ('⚠️' if 'MODERATE' in s['risk_band'] else '✅')}</span>
                <span style="font-size: 16px; font-weight: 600; color: {risk_color}">{p['recommended_action']}</span>
            </div>
            <div style="color: var(--text-secondary); font-size: 13px">
                Analysis context: {p['positive_label']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # ── Model results ──
        col_models, col_info = st.columns([3, 2])

        with col_models:
            st.markdown("""
            <div style="margin-bottom: 16px">
                <h3 style="margin:0; font-size: 20px">Model Predictions</h3>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px">
                    Each of 6 models independently scored your report
                </p>
            </div>
            """, unsafe_allow_html=True)

            pm = p["per_model"]
            sorted_models = sorted(pm.items(), key=lambda x: -x[1])

            for model_name, prob in sorted_models:
                is_quantum = any(q in model_name for q in ["Quantum", "VQC", "Variational"])
                bar_cls = "bar-quantum" if is_quantum else "bar-classical"
                chip_cls = "chip-quantum" if is_quantum else "chip-classical"
                chip_label = "Quantum" if is_quantum else "Classical"
                w = max(prob * 100, 1)

                st.markdown(f"""
                <div class="model-bar-wrap">
                    <div class="model-bar-label">
                        <span class="model-bar-name">
                            <span class="chip {chip_cls}">{chip_label}</span>
                            {model_name}
                        </span>
                        <span class="model-bar-val">{prob:.1%}</span>
                    </div>
                    <div class="model-bar-track">
                        <div class="model-bar-fill {bar_cls}" style="width:{w}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            # Router info
            st.markdown("""
            <div style="margin-bottom: 16px">
                <h3 style="margin:0; font-size: 20px">Analysis Details</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass" style="margin-bottom: 12px">
                <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.6px; color:var(--text-muted); margin-bottom:10px">Cancer Type Routing</div>
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <span style="font-weight:600; color:var(--text-primary)">
                        {cancer_icon} {r['router']['chosen'].capitalize()}
                    </span>
                    <span class="chip {'chip-warn' if r['router']['override_used'] else 'chip-ok'}">
                        {'User override' if r['router']['override_used'] else 'AI detected'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quality metrics
            ocr_conf = r['ocr']['confidence']
            med_score = r['validation']['score']
            field_comp = r['fields']['completeness_key']

            st.markdown(f"""
            <div class="glass" style="margin-bottom: 12px">
                <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.6px; color:var(--text-muted); margin-bottom:12px">Document Quality</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px">
                    <span style="color:var(--text-secondary); font-size:13px">OCR Confidence</span>
                    <span style="font-weight:600; font-size:13px; color:{'var(--accent-green)' if ocr_conf >= 0.8 else 'var(--accent-amber)'}">{ocr_conf:.0%}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px">
                    <span style="color:var(--text-secondary); font-size:13px">Medical Validity</span>
                    <span style="font-weight:600; font-size:13px; color:{'var(--accent-green)' if med_score >= 0.3 else 'var(--accent-amber)'}">{med_score:.0%}</span>
                </div>
                <div style="display:flex; justify-content:space-between">
                    <span style="color:var(--text-secondary); font-size:13px">Field Completeness</span>
                    <span style="font-weight:600; font-size:13px; color:{'var(--accent-green)' if field_comp >= 0.7 else 'var(--accent-amber)'}">{field_comp:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if r["fields"]["missing_key"]:
                missing_list = ", ".join(r["fields"]["missing_key"][:6])
                st.markdown(f"""
                <div class="glass" style="border-left: 3px solid var(--accent-amber); padding: 14px 18px">
                    <div style="color: var(--accent-amber); font-weight: 600; font-size: 13px; margin-bottom: 4px">
                        ⚠️ Missing critical fields
                    </div>
                    <div style="color: var(--text-muted); font-size: 12px">{missing_list}</div>
                </div>
                """, unsafe_allow_html=True)

            for w in r["ocr"]["warnings"]:
                st.warning(w)

        # ── Extracted data (collapsible) ──
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        with st.expander("🔬 Extracted fields from report"):
            st.json(r["fields"]["extracted"])

        with st.expander("📄 Raw extracted text"):
            st.code(r["ocr"]["text"][:1500], language="text")

        with st.expander("📋 Full analysis JSON"):
            st.json(r)

        # ── Clinician feedback ──
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        with st.expander("🔁 Submit clinical feedback"):
            st.markdown("""
            <div style="color: var(--text-secondary); font-size: 13px; margin-bottom: 12px">
                If you're a clinician, you can submit the confirmed diagnosis to help improve model accuracy.
            </div>
            """, unsafe_allow_html=True)
            fb_label = st.radio("Confirmed diagnosis",
                                ["Not yet", "Benign / Negative", "Malignant / Positive"],
                                horizontal=True, key="fb_label")
            fb_notes = st.text_input("Notes (optional)", key="fb_notes")
            if fb_label != "Not yet" and st.button("Submit feedback", key="fb_btn"):
                true_label = 1 if "Malignant" in fb_label else 0
                b = load_bundle(s["cancer"])
                fv = [float(r["fields"]["extracted"].get(f, 0.0)) for f in b["feature_names"]]
                rec = log_feedback(s["cancer"], fv, true_label, s["final_score"], fb_notes)
                st.success(f"✓ Feedback logged (#{rec['vec_hash']}). Thank you.")

        # ── Disclaimer ──
        st.markdown("""
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer</strong> — QMed Quantum is a research prototype and is 
            <strong>not a certified medical device</strong>. Results should be reviewed by a qualified 
            healthcare professional. Never make medical decisions based solely on AI predictions. 
            Always consult your doctor.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ═════════════════════════════════════════════════════════════════════
elif page == "performance":
    st.markdown("""
    <div class="section-hero">
        <h1>Model Performance</h1>
        <p>Benchmark results across all 5 cancer types, tested on held-out patients. 
        Our quantum-classical ensemble achieves &gt;96% accuracy on most cancers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load results
    try:
        with open(os.path.join(HERE, "results", "multicancer_results.json")) as fp:
            R = json.load(fp)
    except FileNotFoundError:
        st.error("Benchmark results not found. Please run model training first.")
        st.stop()

    # Summary KPIs
    all_accs = []
    all_aucs = []
    for cancer, models in R.items():
        for m, mt in models.items():
            if m != "Ensemble Consensus":
                all_accs.append(mt.get("accuracy", 0))
                all_aucs.append(mt.get("roc_auc", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{np.mean(all_accs):.1%}</div>
        <div class="kpi-label">Avg Accuracy</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{np.mean(all_aucs):.3f}</div>
        <div class="kpi-label">Avg ROC-AUC</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">5</div>
        <div class="kpi-label">Cancer Types</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">30</div>
        <div class="kpi-label">Trained Models</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Per-cancer benchmarks
    for cancer in CANCERS:
        if cancer not in R:
            continue
        models = R[cancer]
        icon = CANCER_ICONS.get(cancer, "🔬")
        color = CANCER_COLORS.get(cancer, "#67e8f9")

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin: 24px 0 12px 0">
            <span style="font-size:24px">{icon}</span>
            <h3 style="margin:0; font-size:20px; color:{color}">{cancer.capitalize()}</h3>
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for m, mt in models.items():
            if m == "Ensemble Consensus":
                continue
            is_q = "Quantum" in m or "Variational" in m
            rows.append({
                "Model": m,
                "Type": "⚛️ Quantum" if is_q else "📐 Classical",
                "Accuracy": f"{mt.get('accuracy', 0):.1%}",
                "Sensitivity": f"{mt.get('sensitivity', 0):.1%}",
                "Specificity": f"{mt.get('specificity', 0):.1%}",
                "F1 Score": f"{mt.get('f1', 0):.3f}",
                "ROC-AUC": f"{mt.get('roc_auc', 0):.4f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Ensemble row
        ens = models.get("Ensemble Consensus")
        if ens:
            st.markdown(f"""
            <div class="glass" style="padding: 12px 18px; margin-bottom: 8px; border-left: 3px solid {color}">
                <span style="font-weight:600; color:var(--text-primary); font-size:13px">
                    Ensemble Consensus:
                </span>
                <span style="color:var(--text-secondary); font-size:13px; margin-left:8px">
                    Accuracy {ens.get('accuracy', 0):.1%} ·
                    Sensitivity {ens.get('sensitivity', 0):.1%} ·
                    ROC-AUC {ens.get('roc_auc', 0):.4f}
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top: 32px">
        📊 <strong>Note</strong> — Breast cancer benchmarks use real UCI Wisconsin data (569 patients). 
        Other cancer types use literature-grounded synthetic datasets (800 patients each). 
        All results are on 22% held-out test sets.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════════
elif page == "settings":
    st.markdown("""
    <div class="section-hero">
        <h1>Settings & System Info</h1>
        <p>Model status, system information, and administration tools.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    st.markdown("""
    <div style="margin-bottom: 16px">
        <h3 style="margin:0; font-size: 20px">Model Status</h3>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for i, c in enumerate(CANCERS):
        with cols[i]:
            p_path = os.path.join(HERE, "models", "multicancer", f"{c}.joblib")
            exists = os.path.exists(p_path)
            icon = CANCER_ICONS[c]
            color = CANCER_COLORS[c]
            status = "✅ Ready" if exists else "❌ Missing"
            st.markdown(f"""
            <div class="glass" style="text-align:center; padding: 20px">
                <div style="font-size: 28px; margin-bottom: 8px">{icon}</div>
                <div style="font-weight: 600; color: {color}; font-size: 14px">{c.capitalize()}</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    # Router status
    router_ok = os.path.exists(os.path.join(HERE, "models", "router", "router.joblib"))
    st.markdown(f"""
    <div class="glass" style="margin-top: 16px; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center">
        <span style="font-weight: 500; color: var(--text-secondary)">Cancer Type Router</span>
        <span class="chip {'chip-ok' if router_ok else 'chip-warn'}">{'✅ Active' if router_ok else '⚠️ Missing'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    # Feedback / Self-learning
    st.markdown("""
    <div style="margin-bottom: 16px">
        <h3 style="margin:0; font-size: 20px">Clinical Feedback Log</h3>
        <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px">
            Clinician feedback is logged for safe incremental model updates.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fb_dir = os.path.join(HERE, "feedback")
    fb_files = [f for f in os.listdir(fb_dir) if f.endswith(".jsonl")] if os.path.isdir(fb_dir) else []

    if fb_files:
        fb_rows = []
        for f in fb_files:
            fp = os.path.join(fb_dir, f)
            with open(fp) as file:
                lines = file.readlines()
            cancer_name = f.replace("_feedback.jsonl", "").capitalize()
            fb_rows.append({"Cancer": cancer_name, "Records": len(lines), "File": f})
        df = pd.DataFrame(fb_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No feedback records yet. Submit clinical feedback after analyzing reports.")

    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    # About
    st.markdown("""
    <div style="margin-bottom: 16px">
        <h3 style="margin:0; font-size: 20px">About</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 16px">
            <div class="logo-icon" style="width: 48px; height: 48px; font-size: 24px; border-radius: 12px">🧬</div>
            <div>
                <div style="font-family: 'Space Grotesk'; font-size: 22px; font-weight: 700">QMed Quantum</div>
                <div style="color: var(--text-muted); font-size: 13px">Hybrid Quantum-Classical AI · Early Cancer Detection</div>
            </div>
        </div>
        <div style="color: var(--text-secondary); font-size: 14px; line-height: 1.8">
            <strong>Pipeline:</strong> Upload → OCR (Tesseract + medical vocab) → Validation → Cancer-type routing (TF-IDF) → 
            Field extraction (regex + fuzzy match) → Per-cancer ensemble (4 classical + 2 quantum models) → 
            Stage classification → Early risk scoring → Unified risk band<br><br>
            <span class="chip chip-classical">Logistic Regression</span>
            <span class="chip chip-classical">Random Forest</span>
            <span class="chip chip-classical">Gradient Boosting</span>
            <span class="chip chip-classical">SVM (RBF)</span>
            <span class="chip chip-quantum">Quantum Kernel SVM</span>
            <span class="chip chip-quantum">Variational Quantum Classifier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Disclaimer</strong> — QMed Quantum is a research prototype developed for SIH 2026. 
        It is <strong>not a certified medical device</strong>. Breast cancer data uses real UCI Wisconsin dataset 
        (569 patients). Other cancers use literature-grounded synthetic data. Always consult a qualified 
        healthcare professional before making medical decisions.
    </div>
    """, unsafe_allow_html=True)
