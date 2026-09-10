"""
QMed-Quantum — Flagship Streamlit application
Team Synergy 6 · SIH 2026 · Problem Statement SIH26139 (Egreen Quanta)
Run:  streamlit run streamlit_app.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from inference import ART, RESULTS, REF_MEANS, REF_STDS, predict_all, risk_band, CIRCUIT_ASCII
import report_parser as rp
from prs import polygenic_risk_score, SNP_PANEL
from explainer import explain_prediction, explain_benchmarks

FEATURE_NAMES = ART["feature_names"]
THRESHOLDS = ART["thresholds"]
BEST = RESULTS["best_model"]

st.set_page_config(page_title="QMed-Quantum · Synergy 6", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- flagship theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif}
h1,h2,h3,.hero-title{font-family:'Space Grotesk',sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 80% -10%,#16233f 0%,#0b1220 55%)}
[data-testid="stSidebar"]{background:#0d1628;border-right:1px solid #1e2c47}
.hero{padding:34px 36px;border-radius:20px;margin-bottom:22px;
 background:linear-gradient(120deg,#101c34 0%,#13244a 45%,#1b1550 100%);
 border:1px solid #2a3f66;position:relative;overflow:hidden}
.hero:after{content:'Ψ';position:absolute;right:26px;top:-38px;font-size:170px;
 color:#ffffff0d;font-family:'Space Grotesk'}
.hero-title{font-size:34px;font-weight:700;color:#f2f6ff;letter-spacing:.5px}
.hero-sub{color:#9db2d6;font-size:14.5px;margin-top:6px;max-width:760px}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:11.5px;font-weight:600;margin-right:6px}
.pill-q{background:#2a1b4d;color:#c9a7ff;border:1px solid #6b46b8}
.pill-c{background:#12314f;color:#6fc7ff;border:1px solid #1f5c8f}
.pill-g{background:#123527;color:#5fe3a8;border:1px solid #1f7a54}
.gcard{background:#111c31;border:1px solid #22345a;border-radius:16px;padding:18px 20px;margin-bottom:14px}
.kpi-num{font-size:30px;font-weight:700;font-family:'Space Grotesk'}
.kpi-lab{color:#8fa1c0;font-size:12px;text-transform:uppercase;letter-spacing:.8px}
.card-ok{border-left:5px solid #42d99a}.card-bad{border-left:5px solid #ff6b6b}
.card-warn{border-left:5px solid #ffb84c}.card-info{border-left:5px solid #4cc3ff}
.explain-card{background:#101a2e;border:1px solid #22345a;border-radius:14px;padding:16px 20px;margin-bottom:12px}
.team-chip{display:inline-block;background:#111c31;border:1px solid #22345a;border-radius:12px;
 padding:10px 16px;margin:4px;font-size:13px}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:#111c31;border:1px solid #22345a;border-radius:10px;padding:8px 18px}
.stTabs [aria-selected="true"]{background:#1b3a66!important;border-color:#4cc3ff!important}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🧬 QMed-Quantum")
    st.caption("Hybrid Quantum ML Platform for Early Disease Detection")
    st.divider()
    page = st.radio("Navigate", ["🏠 Overview", "📄 Report Analyzer", "🔢 Manual Input",
                                 "📊 Benchmarks", "🔍 Explainability",
                                 "⚛️ Quantum Lab", "👥 About · Synergy 6"])
    st.divider()
    st.markdown(f"**Best model:** {BEST}")
    st.markdown(f"**Test ROC-AUC:** `{RESULTS['models'][BEST]['roc_auc']}`")
    st.markdown(f"**Python:** 3.12 · PennyLane 0.45")
    st.caption("SIH 2026 · SIH26139 · Egreen Quanta")

# ---------------------------------------------------------------- helpers
def hero():
    st.markdown(f"""
<div class="hero">
 <div class="hero-title">QMed-Quantum</div>
 <div class="hero-sub">A hybrid <b>quantum-classical machine learning</b> platform that reads real
 clinical reports, extracts cytology &amp; genetic biomarkers, computes polygenic risk, and delivers
 <b>early disease detection</b> with explainable, plain-language results.</div>
 <div style="margin-top:14px">
  <span class="pill pill-q">⚛️ Hybrid Quantum Models</span>
  <span class="pill pill-c">📄 AI Report Intake</span>
  <span class="pill pill-g">🧬 Polygenic Risk Scoring</span>
  <span class="pill pill-c">🗣️ Plain-Language AI Explainer</span>
 </div></div>""", unsafe_allow_html=True)

def verdict_ui(probs, prefix=""):
    best_p = probs[BEST]
    cols = st.columns(len(probs))
    for c, (name, p) in zip(cols, probs.items()):
        band, tone = risk_band(p)
        is_q = "Quantum" in name
        c.markdown(f"""<div class="gcard card-{tone}">
 <div class="kpi-lab">{name} {'<span class="pill pill-q">QUANTUM</span>' if is_q else '<span class="pill pill-c">CLASSICAL</span>'}</div>
 <div class="kpi-num" style="color:{'#ff6b6b' if tone=='bad' else ('#ffb84c' if tone=='warn' else '#42d99a')}">{p*100:.1f}%</div>
 <div class="kpi-lab">disease probability · {band}</div>
 <div style="font-size:12px;color:#8fa1c0;margin-top:6px">Early-detection verdict (th={THRESHOLDS.get(name,.5):.2f}):
 <b>{'MALIGNANT' if p >= THRESHOLDS.get(name,.5) else 'BENIGN'}</b></div></div>""",
        unsafe_allow_html=True)
    return best_p

# ================================================================ OVERVIEW
if page == "🏠 Overview":
    hero()
    bm = RESULTS["models"][BEST]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="gcard"><div class="kpi-lab">Best model accuracy</div><div class="kpi-num" style="color:#4cc3ff">{bm["accuracy"]*100:.1f}%</div><div class="kpi-lab">{BEST}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="gcard"><div class="kpi-lab">ROC-AUC</div><div class="kpi-num" style="color:#c9a7ff">{bm["roc_auc"]}</div><div class="kpi-lab">test set n={RESULTS["dataset"]["test_size"]}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="gcard"><div class="kpi-lab">Early-detection sensitivity</div><div class="kpi-num" style="color:#42d99a">{bm["early_detection_mode"]["sensitivity_recall"]*100:.1f}%</div><div class="kpi-lab">threshold-tuned</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="gcard"><div class="kpi-lab">Models compared</div><div class="kpi-num">5</div><div class="kpi-lab">3 classical · 2 quantum</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔄 How the platform works")
    st.markdown("""<div class="gcard">
<b>1 · Upload</b> a lab / genetic / DNA report (PDF, TXT, CSV) &nbsp;→&nbsp;
<b>2 · Parse</b> clinical cytology features, biomarkers (BRCA1/2, HER2, ER/PR, Ki-67) &amp; SNP genotypes &nbsp;→&nbsp;
<b>3 · Predict</b> with 5 models incl. hybrid quantum kernel SVM &amp; variational quantum classifier &nbsp;→&nbsp;
<b>4 · Score</b> polygenic risk from your DNA panel &nbsp;→&nbsp;
<b>5 · Explain</b> everything in plain language, with statistical evidence.
</div>""", unsafe_allow_html=True)

    st.markdown("### 🏗️ System architecture")
    st.code("""Clinical report / DNA export (PDF·TXT·CSV)
            │
     Report Intake Engine ── clinical features (30) ── biomarkers ── SNP genotypes
            │                      │                    │               │
            │            ┌─────────┴────────┐           │        PRS engine
            │            │ Classical branch │           │     (10-SNP GWAS panel)
            │            │ LR · RF · SVM    │           │               │
            │            ├──────────────────┤           │               │
            │            │ Quantum branch   │           │               │
            │            │ PCA→4 qubits     │           │               │
            │            │ QSVC · VQC       │           │               │
            │            └─────────┬────────┘           │               │
            └──────────────────────┴────────────────────┴───────────────┘
                                   ▼
            Benchmarking · threshold tuning · explainability · AI explainer
                                   ▼
                    Streamlit flagship UI (this app)""", language="text")
    st.info(explain_benchmarks(RESULTS))

# ================================================================ REPORT ANALYZER
elif page == "📄 Report Analyzer":
    hero()
    st.markdown("### Upload a medical report")
    up = st.file_uploader("Lab report, genetic test, or DNA genotype export",
                          type=["pdf", "txt", "csv"])
    col_demo, _ = st.columns([1, 3])
    demo = col_demo.button("🎲 Use bundled sample report instead")
    text = None
    if up:
        text = rp.extract_text(up.name, up.read())
        st.success(f"Parsed **{up.name}** ({len(text):,} characters extracted).")
    elif demo:
        sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "sample_report.txt")
        if os.path.exists(sp):
            text = open(sp).read()
            st.success("Loaded bundled sample: FNA cytology + biomarkers + DNA panel.")
        else:
            st.warning("samples/sample_report.txt not found.")
    if text:
        with st.expander("📃 Raw extracted text"):
            st.text(text[:4000])
        feats = rp.parse_clinical_features(text)
        markers = rp.parse_genetic_markers(text)
        snps = rp.parse_snps(text)
        st.markdown(f"### Extraction results — **{len(feats)}/30** cytology features · "
                    f"**{len(markers)}** biomarkers · **{len(snps)}** SNP genotypes")
        vec, vreport = rp.assemble_vector(feats, REF_MEANS)
        imputed = sum(1 for r in vreport if r["confidence"] == 0.0)
        df = pd.DataFrame(vreport)
        st.dataframe(df.style.map(
            lambda v: "color:#ffb84c" if v == "imputed (population mean)" else "color:#42d99a",
            subset=["source"]), use_container_width=True, height=280)

        prs = polygenic_risk_score(snps) if snps else None
        if prs:
            st.markdown(f"**🧬 Polygenic Risk Score:** relative risk "
                        f"**{prs['relative_risk']}×** population average — *{prs['band']}* "
                        f"({prs['snps_matched']}/{prs['snps_total']} panel SNPs matched)")
            st.dataframe(pd.DataFrame(prs["details"]), use_container_width=True)

        st.markdown("### ⚛️ Hybrid quantum diagnosis")
        with st.spinner("Running classical + quantum inference …"):
            probs = predict_all(vec)
        verdict_ui(probs)

        st.markdown("### 🗣️ AI explanation in plain words")
        cards = explain_prediction(probs, BEST, THRESHOLDS, vec, FEATURE_NAMES,
                                   REF_MEANS, REF_STDS, prs=prs, markers=markers,
                                   imputed_count=imputed)
        for cd in cards:
            st.markdown(f'<div class="explain-card card-{cd["tone"]}"><b>{cd["title"]}</b>'
                        f'<div style="margin-top:8px;color:#c6d4ee">{cd["body"]}</div></div>',
                        unsafe_allow_html=True)
        st.session_state["last_report"] = {"vector": vec.tolist(), "probs": probs}

# ================================================================ MANUAL INPUT
elif page == "🔢 Manual Input":
    hero()
    st.markdown("### Enter the 30 FNA cytology measurements directly")
    mode = st.radio("Input mode", ["Paste comma-separated values", "Load a real test patient",
                                   "Edit each field"], horizontal=True)
    vec = None
    if mode == "Paste comma-separated values":
        raw = st.text_area("30 comma-separated numbers",
                           placeholder="17.99,10.38,122.8,1001,0.1184, …", height=90)
        if st.button("⚛️ Run diagnosis", type="primary"):
            vals = [v for v in raw.replace("\n", ",").split(",") if v.strip()]
            try:
                vals = [float(v) for v in vals]
                if len(vals) != 30:
                    st.error(f"Need exactly 30 values, got {len(vals)}.")
                else:
                    vec = np.array(vals)
            except ValueError:
                st.error("Non-numeric value detected.")
    elif mode == "Load a real test patient":
        X, y = ART["X_test"], ART["y_test"]
        i = st.slider("Test patient index", 0, len(X) - 1, 0)
        st.caption(f"Ground truth (hidden from models): **{'MALIGNANT' if y[i]==1 else 'BENIGN'}**")
        st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES, "value": X[i]}), height=250)
        if st.button("⚛️ Run diagnosis", type="primary"):
            vec = np.array(X[i], dtype=float)
    else:
        cols = st.columns(3)
        vals = []
        for i, f in enumerate(FEATURE_NAMES):
            vals.append(cols[i % 3].number_input(f, value=float(REF_MEANS[i]), format="%.5g"))
        if st.button("⚛️ Run diagnosis", type="primary"):
            vec = np.array(vals)
    if vec is not None:
        with st.spinner("Running quantum inference …"):
            probs = predict_all(vec)
        verdict_ui(probs)
        st.markdown("### 🗣️ AI explanation")
        for cd in explain_prediction(probs, BEST, THRESHOLDS, vec, FEATURE_NAMES,
                                     REF_MEANS, REF_STDS, imputed_count=0):
            st.markdown(f'<div class="explain-card card-{cd["tone"]}"><b>{cd["title"]}</b>'
                        f'<div style="margin-top:8px;color:#c6d4ee">{cd["body"]}</div></div>',
                        unsafe_allow_html=True)

# ================================================================ BENCHMARKS
elif page == "📊 Benchmarks":
    hero()
    static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static")
    rows = []
    for n, m in RESULTS["models"].items():
        rows.append({"Model": n, "Type": "⚛️ quantum" if m["type"] == "hybrid_quantum" else "classical",
                     "Accuracy": m["accuracy"], "Sensitivity": m["sensitivity_recall"],
                     "Specificity": m["specificity"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                     "Train (s)": m["train_time_s"],
                     "Early-detection sensitivity": m["early_detection_mode"]["sensitivity_recall"],
                     "Tuned threshold": m["early_detection_mode"]["threshold"]})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.highlight_max(subset=["Accuracy", "Sensitivity", "F1", "ROC-AUC",
                                                "Early-detection sensitivity"], color="#1b3a66"),
                 use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(os.path.join(static, "roc_curves.png"), caption="ROC curves — quantum vs classical")
    c2.image(os.path.join(static, "benchmark.png"), caption="Metric benchmark")
    st.image(os.path.join(static, "confusion_matrices.png"), caption="Confusion matrices")
    st.bar_chart(df.set_index("Model")[["ROC-AUC", "Accuracy"]])
    st.info(explain_benchmarks(RESULTS))

# ================================================================ EXPLAINABILITY
elif page == "🔍 Explainability":
    hero()
    e = RESULTS["explainability"]
    st.markdown(f"**Method:** {e['method']}")
    static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static")
    c1, c2 = st.columns(2)
    c1.image(os.path.join(static, "feature_importance.png"),
             caption="Top-10 clinical feature importances (Random Forest)")
    qdf = pd.DataFrame(e["top_quantum_components"])
    c2.markdown("#### Quantum component importance (VQC)")
    c2.dataframe(qdf, use_container_width=True)
    c2.bar_chart(qdf.set_index("component")["importance"])
    st.markdown("#### Top clinical drivers")
    st.dataframe(pd.DataFrame(e["top_classical_features"]), use_container_width=True)

# ================================================================ QUANTUM LAB
elif page == "⚛️ Quantum Lab":
    hero()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hybrid quantum-classical circuit")
        st.code(CIRCUIT_ASCII, language="text")
        st.markdown("""<div class="gcard"><b>Why quantum?</b><br><span style="color:#9db2d6">
Quantum feature maps embed each patient into an exponentially large Hilbert space where
non-linear disease patterns become linearly separable — the fidelity kernel then measures
similarity no classical RBF kernel can express. Our benchmark shows this empirically: the
quantum kernel achieved the highest accuracy, sensitivity and AUC of all five models.</span></div>""",
                    unsafe_allow_html=True)
    with c2:
        static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static")
        st.image(os.path.join(static, "vqc_loss.png"), caption="VQC training convergence")
        evr = RESULTS["dataset"]["pca_explained_variance"]
        st.markdown("#### Quantum feature compression (PCA → 4 qubits)")
        st.bar_chart(pd.DataFrame({"explained variance": evr},
                                  index=[f"PC{i+1}" for i in range(len(evr))]))
        st.caption(f"Total variance captured: {sum(evr)*100:.1f}% of the 30 original features")

# ================================================================ ABOUT
elif page == "👥 About · Synergy 6":
    hero()
    st.markdown("### 👥 Team Synergy 6")
    st.markdown("""<div class="gcard">
<span class="team-chip">🧑‍💻 Member 1 — Quantum ML &amp; model training <i>(add name)</i></span>
<span class="team-chip">🧑‍💻 Member 2 — Backend &amp; report intake engine <i>(add name)</i></span>
<span class="team-chip">🧑‍💻 Member 3 — Frontend &amp; UX <i>(add name)</i></span>
<span class="team-chip">🧑‍💻 Member 4 — Data pipeline &amp; PRS <i>(add name)</i></span>
<span class="team-chip">🧑‍💻 Member 5 — Explainability &amp; docs <i>(add name)</i></span>
<span class="team-chip">🧑‍💻 Member 6 — Research, pitch &amp; QA <i>(add name)</i></span>
</div>""", unsafe_allow_html=True)
    st.markdown("### 🎯 Problem statement")
    st.markdown(f"**SIH26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection** · "
                f"Organization: **Egreen Quanta** · Theme: **MedTech/BioTech/HealthTech** · Category: Software")
    st.markdown("### 🧰 Tech stack")
    st.table(pd.DataFrame({
        "Layer": ["Quantum", "ML", "Report intake", "Risk scoring", "Explainability", "UI", "Language"],
        "Technology": ["PennyLane 0.45 (default.qubit simulator, NISQ-portable)",
                       "scikit-learn 1.9 (LR · RF · SVM-RBF · QSVC kernel)",
                       "Deterministic NLP parser (regex + alias maps + fuzzy match), pypdf for PDFs",
                       "10-SNP GWAS polygenic panel with per-allele odds ratios",
                       "Permutation importance + threshold tuning + grounded AI explainer",
                       "Streamlit flagship UI (this app) + Flask REST API (legacy, web/app.py)",
                       "Python 3.12.3"]}))
    st.markdown("### 📂 Repository")
    st.code("""qmed-quantum/
├── streamlit_app.py        # flagship UI (this app)
├── src/
│   ├── train.py            # trains all 5 models + benchmarks + explainability
│   ├── inference.py        # shared classical + quantum inference engine
│   ├── report_parser.py    # report → features / biomarkers / SNPs
│   ├── prs.py              # polygenic risk score engine
│   └── explainer.py        # plain-language AI explainer
├── models/artifacts.pkl    # trained models, PCA, scalers, quantum states, VQC weights
├── results/results.json    # full benchmark + thresholds + explainability
├── samples/                # sample reports for the demo
├── web/                    # legacy Flask REST API + dashboard + plots
├── tests/                  # smoke tests
└── docs/                   # architecture, API, demo script, judge Q&A""", language="text")
    st.warning("⚠️ Research prototype for SIH 2026 — not a certified medical device. "
               "Always consult a qualified oncologist.")
