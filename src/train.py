"""
QMed-Quantum — Hybrid Quantum Machine Learning Platform for Early Disease Detection
SIH 2026 | Problem Statement SIH26139 (Egreen Quanta)

Training pipeline:
  1. Data ingestion (Breast Cancer Wisconsin biomedical dataset, 30 real-valued features)
  2. Classical baselines: Logistic Regression, Random Forest, SVM (RBF)
  3. Hybrid quantum models: Quantum Kernel SVM (QSVC) + Variational Quantum Classifier (VQC)
  4. Benchmarking: accuracy / sensitivity / specificity / F1 / ROC-AUC / train & inference time
  5. Explainability: permutation importance (classical features + quantum PCA components)
  6. Clinical threshold tuning for early detection (sensitivity-prioritized)
Artifacts -> models/artifacts.pkl, results/results.json, web/static/*.png
"""
import json, time, os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

import pennylane as qml
from pennylane import numpy as pnp

RANDOM_STATE = 42
N_QUBITS = 4
VQC_LAYERS = 2
VQC_EPOCHS = 45
Q_TRAIN_SUBSAMPLE = 200   # stratified subsample for quantum training (NISQ-realistic)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("models", "results", "web/static"):
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

def metrics_dict(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": round(float(threshold), 4),
    }

print("=" * 70)
print("QMed-Quantum training pipeline — SIH26139")
print("=" * 70)

# ---------------------------------------------------------------- data
data = load_breast_cancer()
X = data.data.astype(float)
y_raw = data.target
y = (y_raw == 0).astype(int)          # 1 = malignant (disease positive), 0 = benign
feature_names = list(data.feature_names)
print(f"Dataset: {X.shape[0]} samples x {X.shape[1]} features | positives={y.sum()} negatives={(1-y).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Classical preprocessing
scaler_cl = StandardScaler().fit(X_train)
Xtr_cl, Xte_cl = scaler_cl.transform(X_train), scaler_cl.transform(X_test)

# Quantum preprocessing: scale -> PCA to N_QUBITS dims -> rescale to [0, pi] for angle embedding
scaler_q = StandardScaler().fit(X_train)
pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE).fit(scaler_q.transform(X_train))
evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
print(f"PCA({N_QUBITS}) explained variance ratio: {evr} (sum={sum(evr):.3f})")

Ztr_raw = pca.transform(scaler_q.transform(X_train))
Q_LO, Q_HI = Ztr_raw.min(axis=0), Ztr_raw.max(axis=0)   # train-set bounds only (no leakage)

def q_features(A):
    Z = pca.transform(scaler_q.transform(A))
    Zn = (Z - Q_LO) / np.where(Q_HI - Q_LO == 0, 1, Q_HI - Q_LO)
    return np.clip(Zn, 0, 1) * np.pi        # angles in [0, pi]

rng = np.random.RandomState(RANDOM_STATE)
idx_q = rng.choice(len(X_train), size=min(Q_TRAIN_SUBSAMPLE, len(X_train)), replace=False)
# stratified subsample
idx_pos = rng.choice(np.where(y_train == 1)[0], size=Q_TRAIN_SUBSAMPLE // 2, replace=False)
idx_neg = rng.choice(np.where(y_train == 0)[0], size=Q_TRAIN_SUBSAMPLE - Q_TRAIN_SUBSAMPLE // 2, replace=False)
idx_q = np.concatenate([idx_pos, idx_neg]); rng.shuffle(idx_q)

XQtr_full, XQte = q_features(X_train), q_features(X_test)
XQtr, yQtr = XQtr_full[idx_q], y_train[idx_q]
print(f"Quantum training subsample: {len(XQtr)} (pos={yQtr.sum()}, neg={len(yQtr)-yQtr.sum()})")

results = {"dataset": {"name": "Breast Cancer Wisconsin (Diagnostic)", "samples": int(X.shape[0]),
                        "features": int(X.shape[1]), "positive_class": "malignant",
                        "test_size": int(len(y_test)), "qubits": N_QUBITS,
                        "pca_explained_variance": evr},
           "models": {}, "feature_names": feature_names}

roc_store = {}
cm_store = {}

# ---------------------------------------------------------------- classical baselines
classical = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Classical SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}
fitted_classical = {}
for name, clf in classical.items():
    t0 = time.time(); clf.fit(Xtr_cl, y_train); t_train = time.time() - t0
    t0 = time.time(); prob = clf.predict_proba(Xte_cl)[:, 1]; t_inf = (time.time() - t0) / len(Xte_cl) * 1000
    m = metrics_dict(y_test, prob)
    m["train_time_s"] = round(t_train, 3); m["inference_ms_per_sample"] = round(t_inf, 4)
    m["type"] = "classical"
    results["models"][name] = m
    fitted_classical[name] = clf
    roc_store[name] = (y_test, prob)
    cm_store[name] = m["confusion_matrix"]
    print(f"[classical] {name:24s} acc={m['accuracy']} sens={m['sensitivity_recall']} auc={m['roc_auc']}")

# ---------------------------------------------------------------- quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

def embedding(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[N_QUBITS - 1, 0])

@qml.qnode(dev)
def state_circuit(x):
    embedding(x)
    return qml.state()

print("Computing quantum feature states for kernel ...")
S_tr = np.array([state_circuit(pnp.array(x, requires_grad=False)) for x in XQtr])
S_te = np.array([state_circuit(pnp.array(x, requires_grad=False)) for x in XQte])
K_train = np.abs(S_tr @ S_tr.conj().T) ** 2
K_test = np.abs(S_te @ S_tr.conj().T) ** 2

t0 = time.time()
qsvc = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
qsvc.fit(K_train, yQtr)
t_train = time.time() - t0
t0 = time.time(); qsvc_prob = qsvc.predict_proba(K_test)[:, 1]; t_inf = (time.time() - t0) / len(Xte_cl) * 1000
m = metrics_dict(y_test, qsvc_prob)
m.update({"train_time_s": round(t_train, 3), "inference_ms_per_sample": round(t_inf, 4),
          "type": "hybrid_quantum", "train_samples": int(len(yQtr))})
results["models"]["Quantum Kernel SVM (QSVC)"] = m
roc_store["Quantum Kernel SVM (QSVC)"] = (y_test, qsvc_prob)
cm_store["Quantum Kernel SVM (QSVC)"] = m["confusion_matrix"]
print(f"[quantum ] Quantum Kernel SVM (QSVC) acc={m['accuracy']} sens={m['sensitivity_recall']} auc={m['roc_auc']}")

# ---------------------------------------------------------------- VQC
print(f"Training Variational Quantum Classifier ({VQC_EPOCHS} epochs) ...")
@qml.qnode(dev, interface="autograd")
def vqc_circuit(weights, x):
    embedding(x)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def vqc_prob(weights, x):
    return (vqc_circuit(weights, x) + 1.0) / 2.0

def cost_fn(weights, Xb, yb):
    eps = 1e-9
    loss = pnp.array(0.0)
    for xi, yi in zip(Xb, yb):
        yf = float(yi)                      # labels are constants, not differentiable
        p = pnp.clip(vqc_prob(weights, xi), eps, 1 - eps)
        loss = loss - (yf * pnp.log(p) + (1 - yf) * pnp.log(1 - p))
    return loss / len(yb)

weights = pnp.array(0.05 * np.random.RandomState(RANDOM_STATE).randn(VQC_LAYERS, N_QUBITS, 3),
                    requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.08)
XQtr_p = [pnp.array(x, requires_grad=False) for x in XQtr]
yQtr_p = pnp.array(yQtr, requires_grad=False)
batch = 25
loss_hist = []
t0 = time.time()
for epoch in range(VQC_EPOCHS):
    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(XQtr_p))
    ep_loss = 0.0; nb = 0
    for i in range(0, len(perm), batch):
        bidx = perm[i:i + batch]
        Xb = [XQtr_p[j] for j in bidx]; yb = yQtr_p[bidx]
        weights, loss = opt.step_and_cost(lambda w: cost_fn(w, Xb, yb), weights)
        ep_loss += float(loss); nb += 1
    loss_hist.append(round(ep_loss / nb, 4))
    if (epoch + 1) % 15 == 0:
        print(f"  epoch {epoch+1}/{VQC_EPOCHS} loss={loss_hist[-1]}")
t_train = time.time() - t0

t0 = time.time()
vqc_probs = np.array([float(vqc_prob(weights, pnp.array(x, requires_grad=False))) for x in XQte])
t_inf = (time.time() - t0) / len(XQte) * 1000
m = metrics_dict(y_test, vqc_probs)
m.update({"train_time_s": round(t_train, 3), "inference_ms_per_sample": round(t_inf, 4),
          "type": "hybrid_quantum", "train_samples": int(len(yQtr)), "vqc_loss_history": loss_hist})
results["models"]["Variational Quantum Classifier (VQC)"] = m
roc_store["Variational Quantum Classifier (VQC)"] = (y_test, vqc_probs)
cm_store["Variational Quantum Classifier (VQC)"] = m["confusion_matrix"]
print(f"[quantum ] Variational Quantum Classifier acc={m['accuracy']} sens={m['sensitivity_recall']} auc={m['roc_auc']}")

# ---------------------------------------------------------------- threshold tuning (early detection: prioritize sensitivity)
def tune_threshold(y_true, prob, target_sens=0.97):
    best_t, best_j = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if sens >= target_sens and j > best_j:
            best_j, best_t = j, t
    return float(best_t)

best_name = max(results["models"], key=lambda k: results["models"][k]["roc_auc"])
print(f"\nBest model by ROC-AUC: {best_name}")
thresholds = {}
for name, (yt, prob) in roc_store.items():
    t = tune_threshold(np.asarray(yt), np.asarray(prob))
    thresholds[name] = t
    results["models"][name]["early_detection_mode"] = metrics_dict(np.asarray(yt), np.asarray(prob), threshold=t)
results["best_model"] = best_name
results["tuned_thresholds"] = {k: round(v, 4) for k, v in thresholds.items()}

# ---------------------------------------------------------------- explainability
print("Computing permutation importance (explainability) ...")
rf = fitted_classical["Random Forest"]
pi = permutation_importance(rf, Xte_cl, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
feat_imp = sorted(zip(feature_names, pi.importances_mean), key=lambda t: -t[1])[:10]
results["explainability"] = {
    "method": "Permutation importance (ROC-AUC drop, 10 repeats)",
    "top_classical_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in feat_imp],
}

vqc_dev_probs_fn = lambda Z: np.array([float(vqc_prob(weights, pnp.array(z * np.pi, requires_grad=False))) for z in Z])
# quantum component importance: permute PCA components of test set
Zte = XQte / np.pi  # normalized [0,1]
base_auc = roc_auc_score(y_test, vqc_probs)
q_imp = []
for c in range(N_QUBITS):
    drops = []
    for rep in range(5):
        Zp = Zte.copy()
        rng2 = np.random.RandomState(1000 + rep)
        Zp[:, c] = rng2.permutation(Zp[:, c])
        drops.append(base_auc - roc_auc_score(y_test, vqc_dev_probs_fn(Zp)))
    q_imp.append((f"Quantum feature PC{c+1}", float(np.mean(drops))))
q_imp.sort(key=lambda t: -t[1])
results["explainability"]["top_quantum_components"] = [
    {"component": c, "importance": round(v, 4)} for c, v in q_imp]

# ---------------------------------------------------------------- plots
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
static = os.path.join(BASE, "web/static")

# ROC curves
plt.figure(figsize=(6.2, 5))
for name, (yt, prob) in roc_store.items():
    fpr, tpr, _ = roc_curve(yt, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yt, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.4)
plt.xlabel("False Positive Rate (1 - Specificity)"); plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curves — Classical vs Hybrid Quantum Models")
plt.legend(loc="lower right", fontsize=7.5); plt.tight_layout()
plt.savefig(os.path.join(static, "roc_curves.png")); plt.close()

# metric comparison bars
names = list(results["models"].keys())
mkeys = ["accuracy", "sensitivity_recall", "specificity", "f1", "roc_auc"]
mlabels = ["Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC"]
xpos = np.arange(len(mkeys)); w = 0.8 / len(names)
plt.figure(figsize=(8.5, 4.5))
for i, n in enumerate(names):
    vals = [results["models"][n][k] for k in mkeys]
    plt.bar(xpos + i * w, vals, width=w, label=n)
plt.xticks(xpos + w * (len(names) - 1) / 2, mlabels); plt.ylim(0.8, 1.02)
plt.ylabel("Score"); plt.title("Benchmark — Hybrid Quantum vs Classical Baselines")
plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(static, "benchmark.png")); plt.close()

# confusion matrices
fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
if len(names) == 1: axes = [axes]
for ax, n in zip(axes, names):
    cm = cm_store[n]
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(M, cmap="Blues")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > M.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"]); ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(n.replace(" (", "\n("), fontsize=8)
fig.suptitle("Confusion Matrices (default 0.5 threshold)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(static, "confusion_matrices.png")); plt.close()

# feature importance
plt.figure(figsize=(7, 4.2))
f = [t[0] for t in feat_imp][::-1]; v = [t[1] for t in feat_imp][::-1]
plt.barh(f, v, color="#2e86ab")
plt.xlabel("Mean ROC-AUC drop when permuted"); plt.title("Top-10 Clinical Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig(os.path.join(static, "feature_importance.png")); plt.close()

# VQC training loss
plt.figure(figsize=(6, 3.8))
plt.plot(range(1, len(loss_hist) + 1), loss_hist, color="#a23b72")
plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy loss")
plt.title("VQC Training Convergence (4 qubits, 2 StronglyEntangling layers)")
plt.tight_layout(); plt.savefig(os.path.join(static, "vqc_loss.png")); plt.close()

# ---------------------------------------------------------------- save artifacts
artifacts = {
    "scaler_cl": scaler_cl, "scaler_q": scaler_q, "pca": pca,
    "classical_models": fitted_classical,
    "qsvc": qsvc, "K_train_support": K_train, "yQ_train": yQtr,
    "q_train_states": S_tr, "vqc_weights": np.asarray(weights),
    "thresholds": thresholds, "feature_names": feature_names,
    "n_qubits": N_QUBITS, "vqc_layers": VQC_LAYERS,
    "q_lo": Q_LO, "q_hi": Q_HI,
    "X_test": X_test, "y_test": y_test,
}
joblib.dump(artifacts, os.path.join(BASE, "models/artifacts.pkl"))

with open(os.path.join(BASE, "results/results.json"), "w") as fp:
    json.dump(results, fp, indent=2)

print("\n" + "=" * 70)
print("FINAL BENCHMARK (default threshold 0.5)")
print("=" * 70)
hdr = f"{'Model':36s}{'Acc':>7}{'Sens':>7}{'Spec':>7}{'F1':>7}{'AUC':>7}"
print(hdr); print("-" * len(hdr))
for n in names:
    mm = results["models"][n]
    print(f"{n:36s}{mm['accuracy']:>7}{mm['sensitivity_recall']:>7}{mm['specificity']:>7}{mm['f1']:>7}{mm['roc_auc']:>7}")
print("\nArtifacts: models/artifacts.pkl | results/results.json | web/static/*.png")
print("DONE")
