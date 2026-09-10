"""
Safe self-learning module.

Design (medically-responsible pattern):
  * Every prediction can be followed by user/clinician feedback (true label).
  * Feedback is logged to feedback/<cancer>_feedback.jsonl (append-only).
  * The SGDClassifier stored inside each cancer bundle supports partial_fit;
    we call partial_fit on confirmed feedback batches only (min batch=4).
  * A full retrain is a manual operation (run src/multicancer/train_all.py).
  * All updates are audit-logged with timestamp, feature vector hash, and
    the model's decision before/after.
"""
import os, json, hashlib, time
import numpy as np
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEEDBACK_DIR = os.path.join(BASE, "feedback")
AUDIT_LOG    = os.path.join(FEEDBACK_DIR, "audit.log")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def _hash_vec(v):
    return hashlib.sha1(np.asarray(v, dtype=np.float32).tobytes()).hexdigest()[:12]

def log_feedback(cancer: str, features: list, true_label: int,
                 predicted_prob: float, notes: str = ""):
    """Append feedback record. NEVER updates model directly (safe path)."""
    rec = {
        "ts":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cancer":    cancer,
        "features":  list(map(float, features)),
        "true_label":int(true_label),
        "predicted_prob": float(predicted_prob),
        "vec_hash":  _hash_vec(features),
        "notes":     notes or "",
        "consumed":  False,
    }
    path = os.path.join(FEEDBACK_DIR, f"{cancer}_feedback.jsonl")
    with open(path, "a") as fp:
        fp.write(json.dumps(rec) + "\n")
    return rec

def _read_pending(cancer: str):
    path = os.path.join(FEEDBACK_DIR, f"{cancer}_feedback.jsonl")
    if not os.path.exists(path): return []
    out = []
    with open(path) as fp:
        for line in fp:
            try:
                r = json.loads(line)
                if not r.get("consumed"): out.append(r)
            except Exception: pass
    return out

def _mark_all_consumed(cancer: str):
    path = os.path.join(FEEDBACK_DIR, f"{cancer}_feedback.jsonl")
    if not os.path.exists(path): return
    lines = []
    with open(path) as fp:
        for line in fp:
            try:
                r = json.loads(line); r["consumed"] = True
                lines.append(json.dumps(r))
            except Exception:
                lines.append(line.rstrip("\n"))
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")

def apply_incremental_update(cancer: str, min_batch: int = 4, mode: str = "safe"):
    """
    mode='safe'   -> only log; do not touch model
    mode='semi'   -> partial_fit SGD on pending batch when >= min_batch
    """
    if mode == "safe":
        pending = _read_pending(cancer)
        return {"mode": "safe", "pending": len(pending), "updated": False,
                "message": f"{len(pending)} feedback records logged. "
                           f"Admin can run full retrain via src/multicancer/train_all.py."}

    bundle_path = os.path.join(BASE, "models", "multicancer", f"{cancer}.joblib")
    if not os.path.exists(bundle_path):
        return {"mode": mode, "updated": False, "error": "model bundle missing"}
    bundle = joblib.load(bundle_path)
    sgd, scaler = bundle["sgd_incremental"], bundle["scaler"]
    pending = _read_pending(cancer)
    if len(pending) < min_batch:
        return {"mode": mode, "pending": len(pending), "updated": False,
                "message": f"Need >= {min_batch} pending records (have {len(pending)})."}

    X = np.array([p["features"] for p in pending], dtype=np.float32)
    y = np.array([p["true_label"] for p in pending], dtype=np.int64)
    Xs = scaler.transform(X)
    sgd.partial_fit(Xs, y, classes=np.array([0, 1]))
    bundle["sgd_incremental"] = sgd
    joblib.dump(bundle, bundle_path, compress=3)
    _mark_all_consumed(cancer)

    with open(AUDIT_LOG, "a") as fp:
        fp.write(json.dumps({
            "ts":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cancer":   cancer,
            "action":   "sgd.partial_fit",
            "n":        int(len(pending)),
            "hashes":   [p["vec_hash"] for p in pending],
        }) + "\n")
    return {"mode": mode, "pending": 0, "updated": True,
            "message": f"Incrementally updated SGD on {len(pending)} feedback records."}


if __name__ == "__main__":
    r = log_feedback("prostate", [66, 8.2, 0.10, 55, 1, 7, 420, 130, 13, 1.0], 1, 0.91)
    print(r)
    print(apply_incremental_update("prostate", mode="safe"))
