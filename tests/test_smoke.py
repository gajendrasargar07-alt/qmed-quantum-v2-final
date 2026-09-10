"""QMed-Quantum smoke tests — run: python3 tests/test_smoke.py"""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import report_parser as rp
from prs import polygenic_risk_score
from inference import ART, RESULTS, predict_all, REF_MEANS

def test_parser():
    text = open(os.path.join(BASE, "samples", "sample_report.txt")).read()
    feats = rp.parse_clinical_features(text)
    assert len(feats) == 30, f"expected 30 features, got {len(feats)}: {sorted(feats)}"
    assert abs(feats["mean radius"]["value"] - 17.99) < 1e-6
    markers = rp.parse_genetic_markers(text)
    assert any(m["marker"] == "BRCA1" for m in markers)
    snps = rp.parse_snps(text)
    assert snps.get("rs2981582") == "AG", snps
    print(f"  parser OK: {len(feats)} features, {len(markers)} markers, {len(snps)} SNPs")

def test_prs():
    prs = polygenic_risk_score({"rs2981582": "AG", "rs3803662": "TT", "rs1045485": "CC"})
    assert prs and prs["snps_matched"] == 3 and prs["relative_risk"] > 0
    print(f"  PRS OK: relative risk {prs['relative_risk']} ({prs['band']})")

def test_inference():
    X, y = ART["X_test"], ART["y_test"]
    probs = predict_all(X[0])
    assert len(probs) == 5 and all(0 <= p <= 1 for p in probs.values())
    assert RESULTS["best_model"] in probs
    print(f"  inference OK: {len(probs)} models, best={RESULTS['best_model']}")

def test_artifacts():
    assert os.path.exists(os.path.join(BASE, "models", "artifacts.pkl"))
    assert os.path.exists(os.path.join(BASE, "results", "results.json"))
    assert len(REF_MEANS) == 30
    print("  artifacts OK")

if __name__ == "__main__":
    print("QMed-Quantum smoke tests")
    test_artifacts(); test_parser(); test_prs(); test_inference()
    print("ALL TESTS PASSED ✅")
