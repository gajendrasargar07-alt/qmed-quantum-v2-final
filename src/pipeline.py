"""
QMed-Quantum v2 — Top-level orchestrator.

Pipeline:
  raw upload (txt/pdf/image)
    -> OCR ingest (ocr_pipeline.extract)
    -> Medical validator (reject non-medical docs)
    -> Cancer-type router (or use user-selected cancer)
    -> Field extraction + missing-field flagging
    -> Per-cancer unified scoring (classical + quantum ensemble)
    -> Return a single structured report
"""
import os, joblib
import numpy as np

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
for p in [HERE, os.path.join(HERE, "ocr"), os.path.join(HERE, "validator"),
          os.path.join(HERE, "multicancer"), os.path.join(HERE, "ensemble"),
          os.path.join(HERE, "selflearn")]:
    if p not in sys.path: sys.path.insert(0, p)

from ocr_pipeline import extract, extract_fields, flag_missing
from medical_validator import validate
from unified_score import score_from_fields, _load

# alias map: OCR-canonical field name -> per-cancer feature name (or list)
FIELD_ALIAS = {
    "gleason":       ["gleason_proxy", "gleason"],
    "ca19_9":        ["ca19_9"],
    "cyfra21_1":     ["cyfra21_1"],
    "free_psa_ratio":["free_psa_ratio"],
    "nodule_mm":     ["nodule_mm"],
    "fev1_pct":      ["fev1_pct"],
    "pack_years":    ["pack_years"],
    "prostate_volume":["prostate_volume"],
    "dre_suspicious":["dre_suspicious"],
    "hpv_positive": ["hpv_positive"],
    "dx_cin":       ["dx_cin"],
    "schiller":     ["schiller"],
    "citology":     ["citology"],
    "fit_test":     ["fit_test"],
    "spiculation":  ["spiculation"],
}

def _apply_aliases(fields: dict, target_feature_names: list) -> dict:
    out = dict(fields)
    for src, dsts in FIELD_ALIAS.items():
        if src in out:
            for d in dsts:
                if d in target_feature_names and d not in out:
                    out[d] = out[src]
    return out

BASE = os.path.dirname(HERE)
ROUTER_PATH = os.path.join(BASE, "models", "router", "router.joblib")

def _load_router():
    return joblib.load(ROUTER_PATH)

def analyze(source, source_kind: str = "auto",
            override_cancer: str = None) -> dict:
    """
    Full end-to-end analysis.
    source: path, bytes, or raw text
    source_kind: 'auto' | 'text' | 'pdf' | 'image'
    override_cancer: force a specific cancer type ('breast'/'lung'/...) or None
    """
    # 1. OCR / text ingest
    ocr = extract(source, source_kind)
    text = ocr["text"]

    # 2. Validate
    v = validate(text)
    if not v["is_medical"]:
        return {
            "status":   "rejected",
            "stage":    "validation",
            "ocr":      ocr,
            "validation": v,
            "reason":   v["reason"],
            "hint":     ("Please upload a valid medical / pathology / lab report. "
                         "Random documents, chat screenshots, or non-clinical PDFs are not accepted."),
        }

    # 3. Route to cancer type
    router = _load_router()
    router_probs = router["pipeline"].predict_proba([text])[0]
    router_classes = router["pipeline"].classes_
    router_top = list(sorted(zip(router_classes, router_probs), key=lambda x: -x[1]))
    router_choice = router_top[0][0]

    # user override / validator hint fallback
    cancer = override_cancer or v.get("cancer_hint") or router_choice

    # 4. Field extraction + missing flag
    fields_raw = extract_fields(text)
    bundle = _load(cancer)
    required = bundle["feature_names"]
    key_fields = bundle["key_fields"]
    fields = _apply_aliases(fields_raw, required)

    missing_all = flag_missing(fields, required)
    missing_key = flag_missing(fields, key_fields)

    # 5. Score
    score_out = score_from_fields(cancer, fields)

    # 6. Compile
    report = {
        "status":     "ok",
        "ocr":        ocr,
        "validation": v,
        "router":     {
            "chosen": cancer,
            "predicted": router_choice,
            "override_used": bool(override_cancer),
            "distribution": [{"cancer": c, "prob": round(float(p), 4)} for c, p in router_top],
        },
        "fields":     {
            "extracted": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                          for k, v in fields.items()},
            "required":  required,
            "key_fields": key_fields,
            "completeness_all": missing_all["completeness"],
            "completeness_key": missing_key["completeness"],
            "missing_all": missing_all["missing"],
            "missing_key": missing_key["missing"],
            "flag_missing_critical": bool(missing_key["missing"]),
        },
        "prediction": score_out,
    }
    # attach a top-level convenience summary
    report["summary"] = {
        "cancer": cancer,
        "final_score": score_out["final_score"],
        "risk_band":   score_out["risk_band"],
        "predicted_stage": score_out["predicted_stage"],
        "early_risk_score": score_out["early_risk_score"],
        "ocr_confidence":  ocr["confidence"],
        "medical_score":   v["score"],
        "critical_fields_missing": report["fields"]["missing_key"],
    }
    return report


if __name__ == "__main__":
    demo = (
        "Pathology report. Patient age 58 year-old male. "
        "PSA: 12.5 ng/mL. Free/Total PSA ratio 0.08. DRE: suspicious nodule. "
        "Gleason score 8. Prostate volume 62 mL. Testosterone 380 ng/dL. "
        "ALP: 195 U/L. Hemoglobin 12.4 g/dL. Creatinine 1.1 mg/dL."
    )
    r = analyze(demo, "text")
    import json
    print(json.dumps(r["summary"], indent=2))
    print("router chose ->", r["router"]["chosen"])
    print("per-model:", r["prediction"]["per_model"])
