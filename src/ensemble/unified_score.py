"""
Unified scoring engine.

Given a feature vector for a specific cancer type, run:
  - all classical models  (LogReg, RF, GBM, SVM)
  - Quantum Kernel SVM
  - Variational Quantum Classifier
  - stage classifier
  - early-risk classifier

Produce a single, calibrated final score (0..1) + risk band + stage + per-model
probabilities. Weights favor higher-AUC models; VQCs (usually the weakest in
this MVP) get a small weight to reflect their contribution without dragging
the final score down.

Bands:
  >= 0.75  HIGH RISK
  >= 0.40  MODERATE RISK
  <  0.40  LOW RISK
"""
import os, numpy as np, joblib
import pennylane as qml
from pennylane import numpy as pnp

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MC_DIR = os.path.join(BASE, "models", "multicancer")

# per-model weights (sum to 1 after normalization)
DEFAULT_WEIGHTS = {
    "Logistic Regression":         0.15,
    "Random Forest":               0.20,
    "Gradient Boosting":           0.20,
    "Classical SVM (RBF)":         0.15,
    "Quantum Kernel SVM":          0.20,
    "Variational Quantum Classifier": 0.10,
}

_CACHE = {}

def _load(cancer):
    if cancer in _CACHE: return _CACHE[cancer]
    path = os.path.join(MC_DIR, f"{cancer}.joblib")
    b = joblib.load(path)
    n = int(b["n_qubits"])
    dev = qml.device("default.qubit", wires=n)

    def embedding(x):
        qml.AngleEmbedding(x, wires=range(n), rotation="Y")
        for i in range(n-1): qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n-1, 0])

    @qml.qnode(dev)
    def state_circuit(x):
        embedding(x); return qml.state()

    @qml.qnode(dev, interface="autograd")
    def vqc_circuit(weights, x):
        embedding(x); qml.StronglyEntanglingLayers(weights, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    b["_state_circuit"] = state_circuit
    b["_vqc_circuit"]   = vqc_circuit
    _CACHE[cancer] = b
    return b

def _q_prep(bundle, X):
    Xs = bundle["q_scaler"].transform(X)
    Z  = bundle["q_pca"].transform(Xs)
    lo, hi = bundle["q_lo"], bundle["q_hi"]
    Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return np.clip(Zn, 0, 1) * np.pi

def score(cancer: str, features: list, weights: dict = None) -> dict:
    b = _load(cancer)
    x = np.atleast_2d(np.asarray(features, dtype=np.float32))
    xs = b["scaler"].transform(x)
    per = {}
    for name, clf in b["classical_models"].items():
        per[name] = float(clf.predict_proba(xs)[0, 1])

    xq = _q_prep(b, x)
    state = b["_state_circuit"](pnp.array(xq[0], requires_grad=False))
    K = np.abs(b["q_states_train"] @ state.conj()) ** 2
    per["Quantum Kernel SVM"] = float(b["qsvc"].predict_proba(K.reshape(1, -1))[0, 1])
    W = pnp.array(b["vqc_weights"], requires_grad=False)
    v = float(b["_vqc_circuit"](W, pnp.array(xq[0], requires_grad=False)))
    per["Variational Quantum Classifier"] = (v + 1.0) / 2.0

    # stage
    stage_pred = None; stage_conf = None
    if b.get("stage_clf") is not None:
        sp = b["stage_clf"].predict(xs)[0]
        try:
            sconf = float(np.max(b["stage_clf"].predict_proba(xs)[0]))
        except Exception:
            sconf = None
        stage_pred = int(sp); stage_conf = sconf

    # early-risk (pre-cancer)
    early = float(b["early_risk_clf"].predict_proba(xs)[0, 1])

    # unified score = weighted mean
    w = (weights or DEFAULT_WEIGHTS).copy()
    tot = sum(w.get(k, 0) for k in per)
    if tot <= 0: tot = 1
    final = sum(per[k] * w.get(k, 0) for k in per) / tot

    if final >= 0.75:
        band, action = "HIGH RISK", "Immediate clinical follow-up recommended"
    elif final >= 0.40:
        band, action = "MODERATE RISK", "Additional screening / repeat labs advised"
    else:
        band, action = "LOW RISK", "Routine monitoring"

    return {
        "cancer":           cancer,
        "per_model":        {k: round(v, 4) for k, v in per.items()},
        "final_score":      round(float(final), 4),
        "risk_band":        band,
        "recommended_action": action,
        "predicted_stage":  stage_pred,
        "stage_confidence": None if stage_conf is None else round(stage_conf, 4),
        "early_risk_score": round(early, 4),
        "positive_label":   b["positive_label"],
        "feature_names":    b["feature_names"],
        "weights_used":     {k: round(w.get(k, 0), 3) for k in per},
    }


def score_from_fields(cancer: str, fields: dict) -> dict:
    """Convenience: fill feature vector from OCR fields, use 0 for missing."""
    b = _load(cancer)
    x = [float(fields.get(f, 0.0)) for f in b["feature_names"]]
    return score(cancer, x)


if __name__ == "__main__":
    # test on one real patient from prostate test set
    b = _load("prostate")
    x = b["X_test"][0].tolist()
    r = score("prostate", x)
    for k, v in r.items():
        if k != "feature_names":
            print(f"{k}: {v}")
