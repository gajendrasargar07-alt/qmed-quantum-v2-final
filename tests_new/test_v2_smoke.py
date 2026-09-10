"""
QMed-Quantum v2 — smoke tests for the new multi-cancer / OCR / validator /
self-learning / unified-scoring modules. Preserves original tests/ untouched.

Run:  python3 tests_new/test_v2_smoke.py
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
for p in [os.path.join(BASE, "src"), os.path.join(BASE, "src", "ocr"),
          os.path.join(BASE, "src", "validator"), os.path.join(BASE, "src", "multicancer"),
          os.path.join(BASE, "src", "ensemble"), os.path.join(BASE, "src", "selflearn")]:
    sys.path.insert(0, p)

PASS = FAIL = 0

def check(name, cond, note=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  [{tag}] {name}" + (f"  — {note}" if note else ""))

# ------------------------------------------------------ 1. datasets load
print("\n1) datasets load")
from datasets import load_all
ds = load_all()
check("5 cancer types", set(ds.keys()) == {"breast","lung","prostate","colon","cervical"})
for k, d in ds.items():
    check(f"{k} shapes", d["X"].shape[0] == len(d["y"]) == len(d["stage"]))

# ------------------------------------------------------ 2. bundles exist
print("\n2) trained bundles")
import joblib
for c in ds.keys():
    p = os.path.join(BASE, "models", "multicancer", f"{c}.joblib")
    check(f"bundle {c} exists", os.path.exists(p))
    b = joblib.load(p)
    check(f"{c} has 4 classical models", len(b["classical_models"]) == 4)
    check(f"{c} has qsvc + vqc_weights", b["qsvc"] is not None and b["vqc_weights"] is not None)
    check(f"{c} has stage_clf", b["stage_clf"] is not None)
    check(f"{c} has early_risk_clf", b["early_risk_clf"] is not None)
    check(f"{c} has sgd_incremental (self-learn)", b["sgd_incremental"] is not None)

# ------------------------------------------------------ 3. router
print("\n3) router")
router = joblib.load(os.path.join(BASE, "models", "router", "router.joblib"))
for text, expected in [
    ("PSA 9.4 ng/mL Gleason 7 prostate biopsy DRE suspicious", "prostate"),
    ("Chest CT solitary pulmonary nodule 22 mm CYFRA 21-1 elevated pack years 40", "lung"),
    ("HPV positive pap smear CIN 2 Schiller test", "cervical"),
    ("CEA CA 19-9 elevated colonoscopy tubular adenoma FIT positive", "colon"),
    ("BRCA1 HER2 3+ mean radius mean texture ductal carcinoma", "breast"),
]:
    pred = router["pipeline"].predict([text])[0]
    check(f"route '{text[:30]}...' -> {expected}", pred == expected, f"got={pred}")

# ------------------------------------------------------ 4. validator
print("\n4) medical validator")
from medical_validator import validate
ok = validate("PSA 8.4 ng/mL, Gleason 7, DRE suspicious, biopsy prostate adenocarcinoma, hemoglobin 13 g/dL")
bad = validate("Hi mom, we should have pizza tonight. The dog is happy. See you at 7 pm.")
check("real medical report accepted", ok["is_medical"] is True)
check("random text rejected",         bad["is_medical"] is False)

# ------------------------------------------------------ 5. OCR field extraction
print("\n5) OCR field extraction")
from ocr_pipeline import extract, extract_fields
demo = ("PSA: 12.5 ng/mL. Free/Total PSA ratio 0.08. DRE: suspicious. "
        "Gleason score 8. Hemoglobin 12.4 g/dL. ALP: 195 U/L.")
f = extract_fields(demo)
for k in ("psa","free_psa_ratio","gleason","hemoglobin","alp","dre_suspicious"):
    check(f"field {k} extracted", k in f, f"got fields={list(f.keys())}")

# ------------------------------------------------------ 6. OCR on scanned image
print("\n6) OCR on scanned PNG")
scan = os.path.join(BASE, "data", "samples", "prostate_report_scan.png")
if os.path.exists(scan):
    r = extract(scan, "image")
    check("OCR confidence > 0.7", r["confidence"] > 0.7, f"conf={r['confidence']}")
    check("psa in OCR text", "psa" in r["text"].lower())
else:
    check("scanned sample exists", False, "run data/samples/make_scanned_sample.py")

# ------------------------------------------------------ 7. unified score
print("\n7) unified scoring")
from unified_score import score
b = joblib.load(os.path.join(BASE, "models", "multicancer", "prostate.joblib"))
x = b["X_test"][0].tolist()
r = score("prostate", x)
check("final_score in [0,1]", 0.0 <= r["final_score"] <= 1.0)
check("risk_band present",    r["risk_band"] in ("HIGH RISK","MODERATE RISK","LOW RISK"))
check("stage predicted",      r["predicted_stage"] in (0,1,2,3,4) or r["predicted_stage"] is None)
check("6 per-model scores",   len(r["per_model"]) == 6)

# ------------------------------------------------------ 8. full pipeline
print("\n8) full pipeline on all samples")
from pipeline import analyze
for sample, expected_cancer in [
    ("prostate_report.txt", "prostate"),
    ("lung_report.txt",     "lung"),
    ("colon_report.txt",    "colon"),
    ("cervical_report.txt", "cervical"),
    ("breast_report.txt",   "breast"),
]:
    path = os.path.join(BASE, "data", "samples", sample)
    r = analyze(path, "text")
    check(f"pipeline({sample}) status=ok", r["status"] == "ok")
    check(f"pipeline({sample}) -> {expected_cancer}", r["summary"]["cancer"] == expected_cancer,
          f"got={r['summary']['cancer']}")

# 8b — rejection
r_bad = analyze(os.path.join(BASE, "data", "samples", "irrelevant_random.txt"), "text")
check("irrelevant doc rejected", r_bad["status"] == "rejected")

# 8c — scanned image
r_scan = analyze(scan, "image")
check("scanned image pipeline ok", r_scan["status"] == "ok")
check("scanned image -> prostate", r_scan["summary"]["cancer"] == "prostate")

# ------------------------------------------------------ 9. self-learning
print("\n9) self-learning (safe mode)")
from feedback import log_feedback, apply_incremental_update
rec = log_feedback("breast", [1.0]*30, 1, 0.88, "smoke-test feedback")
check("feedback logged", "vec_hash" in rec)
res = apply_incremental_update("breast", mode="safe")
check("safe update returns without touching model", res["updated"] is False)

# ------------------------------------------------------ 10. original preserved
print("\n10) original project preserved")
for path in ["models/artifacts.pkl", "src/train.py", "src/inference.py",
             "web/app.py", "streamlit_app.py", "README.md"]:
    check(f"original {path} present", os.path.exists(os.path.join(BASE, path)))

print(f"\n=== v2 SMOKE RESULT: {PASS} PASS / {FAIL} FAIL ===")
sys.exit(0 if FAIL == 0 else 1)
