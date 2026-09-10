"""
QMed-Quantum — inference web application.
Flask backend: loads trained classical + hybrid quantum artifacts and serves
benchmarks, live predictions, quantum circuit info and explainability.

Run:  python3 web/app.py   ->  http://localhost:5000
"""
import os, json, time
import numpy as np
import joblib
from flask import Flask, render_template, jsonify, request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder="templates", static_folder="static")

ART = joblib.load(os.path.join(BASE, "models", "artifacts.pkl"))
with open(os.path.join(BASE, "results", "results.json")) as fp:
    RESULTS = json.load(fp)

N_QUBITS = ART["n_qubits"]

# PennyLane device is only needed if we re-run quantum inference
import pennylane as qml
from pennylane import numpy as pnp

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

@qml.qnode(dev, interface="autograd")
def vqc_circuit(weights, x):
    embedding(x)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

WEIGHTS = pnp.array(ART["vqc_weights"], requires_grad=False)
Q_STATES_TRAIN = ART["q_train_states"]

def q_features(A):
    Z = ART["pca"].transform(ART["scaler_q"].transform(np.atleast_2d(A)))
    # reuse train-time normalization bounds approximated via stored pca ranges
    lo, hi = ART["q_lo"], ART["q_hi"]
    Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return np.clip(Zn, 0, 1) * np.pi

def predict_all(x30):
    """x30: length-30 raw feature vector -> per-model disease probabilities."""
    x = np.atleast_2d(np.asarray(x30, dtype=float))
    xc = ART["scaler_cl"].transform(x)
    xq = q_features(x)
    out = {}
    for name, clf in ART["classical_models"].items():
        out[name] = float(clf.predict_proba(xc)[0, 1])
    s = state_circuit(pnp.array(xq[0], requires_grad=False))
    k = np.abs(Q_STATES_TRAIN @ s.conj()) ** 2
    out["Quantum Kernel SVM (QSVC)"] = float(ART["qsvc"].predict_proba(k.reshape(1, -1))[0, 1])
    out["Variational Quantum Classifier (VQC)"] = float(
        (vqc_circuit(WEIGHTS, pnp.array(xq[0], requires_grad=False)) + 1.0) / 2.0)
    return out

def risk_band(p):
    if p >= 0.75: return "HIGH RISK — immediate clinical follow-up recommended"
    if p >= 0.40: return "MODERATE RISK — additional screening advised"
    return "LOW RISK — routine monitoring"

@app.route("/")
def index():
    return render_template("index.html", results=RESULTS)

@app.route("/api/results")
def api_results():
    return jsonify(RESULTS)

@app.route("/api/sample", methods=["GET"])
def api_sample():
    """Return a random real test patient (features + true label)."""
    X, y = ART["X_test"], ART["y_test"]
    i = int(np.random.randint(0, len(X)))
    return jsonify({"index": i, "features": X[i].tolist(),
                    "true_label": int(y[i]),
                    "feature_names": ART["feature_names"]})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    t0 = time.time()
    feats = request.get_json(force=True)["features"]
    probs = predict_all(feats)
    resp = {"elapsed_ms": round((time.time() - t0) * 1000, 1), "models": {}}
    for name, p in probs.items():
        th = ART["thresholds"].get(name, 0.5)
        resp["models"][name] = {
            "disease_probability": round(p, 4),
            "prediction_default": "MALIGNANT (disease)" if p >= 0.5 else "BENIGN",
            "prediction_early_detection": "MALIGNANT (disease)" if p >= th else "BENIGN",
            "tuned_threshold": round(float(th), 4),
            "risk_band": risk_band(p),
        }
    best = RESULTS["best_model"]
    resp["consensus_probability"] = round(float(np.mean(list(probs.values()))), 4)
    resp["best_model"] = best
    resp["best_model_probability"] = round(probs[best], 4)
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
