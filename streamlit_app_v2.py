"""
QMed-Quantum v2 — Multi-cancer Streamlit app (new pages, original preserved).

Adds these pages on top of the original:
  🧬 Multi-Cancer Analyzer — upload txt/pdf/image -> OCR -> validate ->
                              route -> ensemble score -> stage -> early risk
  📷 OCR Playground        — inspect the OCR pipeline field-by-field
  🛡️ Doc Validator         — check whether a doc is a valid medical report
  🔁 Self-Learning         — log clinician feedback + trigger safe incremental update
  📊 Multi-Cancer Metrics  — per-cancer benchmarks (5 models × 5 cancers)

Original `streamlit_app.py` still works — this is a separate entry point.
Run:  streamlit run streamlit_app_v2.py
"""
import os, sys, json, io
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src", "ocr"))
sys.path.insert(0, os.path.join(HERE, "src", "validator"))
sys.path.insert(0, os.path.join(HERE, "src", "multicancer"))
sys.path.insert(0, os.path.join(HERE, "src", "ensemble"))
sys.path.insert(0, os.path.join(HERE, "src", "selflearn"))

from pipeline import analyze
from ocr_pipeline import extract, extract_fields, flag_missing
from medical_validator import validate as validate_text
from unified_score import score_from_fields, _load as load_bundle
from feedback import log_feedback, apply_incremental_update

st.set_page_config(page_title="QMed-Quantum v2 · Multi-Cancer",
                   page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 80% -10%,#16233f 0%,#0b1220 55%)}
[data-testid="stSidebar"]{background:#0d1628;border-right:1px solid #1e2c47}
.hero{padding:26px 30px;border-radius:18px;margin-bottom:18px;
 background:linear-gradient(120deg,#101c34 0%,#13244a 45%,#1b1550 100%);
 border:1px solid #2a3f66}
.card{background:#111c31;border:1px solid #22345a;border-radius:14px;padding:16px 20px;margin-bottom:12px}
.card-hi{border-left:5px solid #ff6b6b}.card-mo{border-left:5px solid #ffb84c}.card-lo{border-left:5px solid #42d99a}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:11.5px;font-weight:600;margin-right:6px}
.pill-q{background:#2a1b4d;color:#c9a7ff;border:1px solid #6b46b8}
.pill-c{background:#12314f;color:#6fc7ff;border:1px solid #1f5c8f}
.pill-g{background:#123527;color:#5fe3a8;border:1px solid #1f7a54}
.kpi{font-size:28px;font-weight:700;font-family:'Space Grotesk'}
.small{color:#8fa1c0;font-size:12px;text-transform:uppercase;letter-spacing:.6px}
</style>
""", unsafe_allow_html=True)

CANCERS = ["breast", "lung", "prostate", "colon", "cervical"]

with st.sidebar:
    st.markdown("## 🧬 QMed-Quantum v2")
    st.caption("Multi-cancer · Quantum-Classical Ensemble · Team Synergy 6")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🧬 Multi-Cancer Analyzer",
        "📷 OCR Playground",
        "🛡️ Doc Validator",
        "🔁 Self-Learning",
        "📊 Multi-Cancer Metrics",
    ])
    st.divider()
    st.caption("Original single-cancer app still available:\n`streamlit run streamlit_app.py`")

# ------------------------------------------------------------- HERO
def hero(title, sub):
    st.markdown(f"""
<div class="hero">
 <h1 style="margin:0;color:#f2f6ff">{title}</h1>
 <div style="color:#9db2d6;margin-top:6px">{sub}</div>
 <div style="margin-top:10px">
   <span class="pill pill-c">Classical ML</span>
   <span class="pill pill-q">Quantum ML</span>
   <span class="pill pill-g">OCR · Validator · Self-Learn</span>
 </div>
</div>
""", unsafe_allow_html=True)


def band_class(band):
    return {"HIGH RISK":"card-hi","MODERATE RISK":"card-mo","LOW RISK":"card-lo"}.get(band,"card")


# ------------------------------------------------------------- OVERVIEW
if page == "🏠 Overview":
    hero("QMed-Quantum v2 · Multi-Cancer Early-Detection Platform",
         "5 cancer domains · 6 models per domain (4 classical + 2 quantum) · stage prediction · "
         "early-risk (pre-cancer) score · medical OCR · doc validator · safe self-learning.")

    c1,c2,c3,c4 = st.columns(4)
    for col, (num, lab) in zip([c1,c2,c3,c4], [
        ("5","cancer types"),
        ("30+","trained models"),
        ("6","models per cancer"),
        ("Stages 0-IV","stage predictor"),
    ]):
        with col:
            st.markdown(f'<div class="card"><div class="kpi">{num}</div><div class="small">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("### Pipeline")
    st.code("""upload (txt/pdf/image)
   │
   ▼   OCR (Tesseract + medical vocab correction, per-word confidence)
   │
   ▼   Medical-doc validator  ── rejects non-medical documents
   │
   ▼   Cancer-type router  (TF-IDF + LogReg)
   │
   ▼   Field extraction (regex + fuzzy label match)  ── flags missing critical fields
   │
   ▼   Per-cancer ensemble
        ├─ Classical: LogReg, RandomForest, GradientBoost, SVM(RBF)
        ├─ Quantum:   Quantum Kernel SVM  +  Variational Quantum Classifier
        ├─ Stage classifier (multiclass RF)
        └─ Early-risk classifier (pre-cancer GBM)
   │
   ▼   Weighted final score + risk band + recommended action
""", language="text")

    st.markdown("### Data note")
    st.info("MVP uses **UCI Breast Cancer Wisconsin (real, 569 patients)** + literature-grounded "
            "synthetic datasets (800 per cancer) for the other four domains. This is a research "
            "prototype for SIH 2026 — not a certified medical device.")


# ------------------------------------------------------------- ANALYZER
elif page == "🧬 Multi-Cancer Analyzer":
    hero("Multi-Cancer Analyzer",
         "Upload a real report (TXT / PDF / scanned image) or paste text. The pipeline validates, "
         "routes, OCR-extracts, and runs the full ensemble.")

    tab_up, tab_paste, tab_sample = st.tabs(["📤 Upload", "📝 Paste text", "📁 Try a sample"])
    src = None; src_kind = "auto"

    with tab_up:
        f = st.file_uploader("Report file", type=["txt","pdf","png","jpg","jpeg","tif","tiff"])
        if f is not None:
            src = f.read()
            ext = os.path.splitext(f.name)[1].lower()
            src_kind = ("text" if ext in (".txt",) else
                        "pdf"  if ext == ".pdf" else "image")
            st.caption(f"Loaded `{f.name}` ({len(src)} bytes, kind={src_kind})")

    with tab_paste:
        txt = st.text_area("Paste report text", height=200,
                           placeholder="PSA 8.4 ng/mL, Free/Total 0.11, DRE suspicious, Gleason 7 ...")
        if st.button("Analyze pasted text"):
            src, src_kind = txt, "text"

    with tab_sample:
        samp = st.selectbox("Sample report", [
            "data/samples/prostate_report.txt",
            "data/samples/lung_report.txt",
            "data/samples/colon_report.txt",
            "data/samples/cervical_report.txt",
            "data/samples/breast_report.txt",
            "data/samples/prostate_report_scan.png  (scanned image → OCR)",
            "data/samples/irrelevant_random.txt  (should be REJECTED)",
        ])
        if st.button("Analyze sample"):
            path = os.path.join(HERE, samp.split()[0])
            src = path
            src_kind = "auto"

    override = st.selectbox("Override cancer type (optional)", ["auto (route)", *CANCERS])
    override_cancer = None if override == "auto (route)" else override

    if src is not None and (isinstance(src, (bytes, str)) and (len(src) if isinstance(src,bytes) else True)):
        with st.spinner("Analyzing…"):
            r = analyze(src, src_kind, override_cancer=override_cancer)

        # ------- rejection path
        if r["status"] == "rejected":
            st.error(f"❌ Document rejected at validation stage")
            st.markdown(f'<div class="card card-hi"><b>Reason:</b> {r["reason"]}<br/>'
                        f'<b>Hint:</b> {r["hint"]}</div>', unsafe_allow_html=True)
            with st.expander("Validation details"):
                st.json(r["validation"])
            st.stop()

        s = r["summary"]
        p = r["prediction"]

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="card {band_class(s["risk_band"])}"><div class="kpi">{s["final_score"]:.3f}</div><div class="small">unified score</div><div style="margin-top:6px;font-weight:600">{s["risk_band"]}</div></div>', unsafe_allow_html=True)
        stage_txt = "—" if s["predicted_stage"] is None else f"Stage {s['predicted_stage']}"
        c2.markdown(f'<div class="card"><div class="kpi">{stage_txt}</div><div class="small">predicted stage</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><div class="kpi">{s["early_risk_score"]:.2f}</div><div class="small">early-risk (pre-cancer)</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="card"><div class="kpi">{s["cancer"].upper()}</div><div class="small">routed cancer domain</div></div>', unsafe_allow_html=True)

        st.markdown("### Per-model probabilities")
        pm = p["per_model"]
        dfm = pd.DataFrame([{"Model": k, "Probability": v} for k,v in pm.items()])
        st.dataframe(dfm, use_container_width=True, hide_index=True)
        st.bar_chart(dfm.set_index("Model"))

        st.markdown("### Cancer-type router distribution")
        dfr = pd.DataFrame(r["router"]["distribution"])
        st.dataframe(dfr, use_container_width=True, hide_index=True)

        st.markdown("### OCR / ingest")
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("OCR confidence", f"{r['ocr']['confidence']:.3f}")
        oc2.metric("Medical validator score", f"{r['validation']['score']:.3f}")
        oc3.metric("Field completeness (key)", f"{r['fields']['completeness_key']*100:.0f} %")
        if r["fields"]["missing_key"]:
            st.warning("⚠️ Missing critical fields: " + ", ".join(r["fields"]["missing_key"]))
        if r["ocr"]["warnings"]:
            for w in r["ocr"]["warnings"]:
                st.warning(w)

        with st.expander("Extracted fields"):
            st.json(r["fields"]["extracted"])
        with st.expander("Extracted text (first 800 chars)"):
            st.code(r["ocr"]["text"][:800])
        with st.expander("Full JSON report"):
            st.json(r)

        # feedback panel
        st.markdown("### 🔁 Clinician feedback (safe self-learning)")
        fb_label = st.radio("Confirmed diagnosis (true label)", ["not yet","benign / negative","malignant / positive"], horizontal=True)
        fb_notes = st.text_input("Notes (optional)")
        if fb_label != "not yet" and st.button("Submit feedback"):
            true_label = 1 if "malignant" in fb_label else 0
            b = load_bundle(s["cancer"])
            fv = [float(r["fields"]["extracted"].get(f, 0.0)) for f in b["feature_names"]]
            rec = log_feedback(s["cancer"], fv, true_label, s["final_score"], fb_notes)
            st.success(f"Feedback logged (hash {rec['vec_hash']}). See Self-Learning page to trigger update.")


# ------------------------------------------------------------- OCR PLAYGROUND
elif page == "📷 OCR Playground":
    hero("OCR Playground", "Inspect the medical OCR pipeline end-to-end.")
    f = st.file_uploader("Upload PDF or scan (png/jpg)", type=["pdf","png","jpg","jpeg","tif","tiff","txt"])
    if f is not None:
        data = f.read()
        ext = os.path.splitext(f.name)[1].lower()
        kind = ("text" if ext == ".txt" else "pdf" if ext == ".pdf" else "image")
        with st.spinner("Running OCR…"):
            r = extract(data, kind)
        c1,c2,c3 = st.columns(3)
        c1.metric("Confidence", f"{r['confidence']:.3f}")
        c2.metric("Pages", len(r["pages"]))
        c3.metric("Chars", len(r["text"]))
        if r["warnings"]:
            for w in r["warnings"]: st.warning(w)
        st.markdown("**Pipeline:** " + " → ".join(r["pipeline"]))
        st.markdown("### Extracted text")
        st.text_area("", value=r["text"], height=280)
        st.markdown("### Auto-parsed fields")
        fields = extract_fields(r["text"])
        st.json(fields)


# ------------------------------------------------------------- VALIDATOR
elif page == "🛡️ Doc Validator":
    hero("Medical Document Validator",
         "Rejects random / non-medical uploads before they reach the models.")
    txt = st.text_area("Paste any text", height=220,
                       placeholder="Try pasting a random paragraph — it will be rejected.")
    if st.button("Validate"):
        r = validate_text(txt)
        if r["is_medical"]:
            st.success(f"✅ Accepted as medical report — score {r['score']:.3f}")
        else:
            st.error(f"❌ Rejected — score {r['score']:.3f}")
        st.json(r)


# ------------------------------------------------------------- SELF-LEARNING
elif page == "🔁 Self-Learning":
    hero("Safe Self-Learning",
         "Feedback log + safe incremental update. Full retrain is still a manual admin action "
         "(run `python3 src/multicancer/train_all.py`).")
    fb_dir = os.path.join(HERE, "feedback")
    files = [f for f in os.listdir(fb_dir) if f.endswith(".jsonl")] if os.path.isdir(fb_dir) else []
    st.markdown(f"**Feedback files:** {len(files)}")
    for f in files:
        p = os.path.join(fb_dir, f)
        with open(p) as fp:
            lines = fp.readlines()
        st.markdown(f"- `{f}` — {len(lines)} record(s)")
    cancer = st.selectbox("Cancer bundle", CANCERS)
    mode = st.selectbox("Update mode", ["safe (log only)","semi (partial_fit on pending batch)"])
    if st.button("Apply update"):
        m = "safe" if mode.startswith("safe") else "semi"
        res = apply_incremental_update(cancer, mode=m)
        st.json(res)


# ------------------------------------------------------------- METRICS
elif page == "📊 Multi-Cancer Metrics":
    hero("Multi-Cancer Benchmarks",
         "Per-cancer test-set metrics: accuracy, sensitivity, specificity, F1, ROC-AUC.")
    try:
        with open(os.path.join(HERE, "results", "multicancer_results.json")) as fp:
            R = json.load(fp)
    except FileNotFoundError:
        st.warning("multicancer_results.json missing — retrain first."); st.stop()
    for cancer, models in R.items():
        st.markdown(f"### {cancer.upper()}")
        rows = []
        for m, mt in models.items():
            rows.append({"Model": m, **{k:mt.get(k) for k in
                        ["accuracy","sensitivity","specificity","f1","roc_auc","threshold","train_seconds"]}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
