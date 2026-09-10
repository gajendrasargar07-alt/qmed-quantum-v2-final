"""
QMed-Quantum · Shared inference engine (Synergy 6)
Loads trained artifacts once and exposes predict_all() for any 30-dim vector.
Used by both the Streamlit app and any external integration.
"""
import os, json
import numpy as np
import joblib
import pennylane as qml
from pennylane import numpy as pnp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ART = joblib.load(os.path.join(BASE, "models", "artifacts.pkl"))
with open(os.path.join(BASE, "results", "results.json")) as fp:
    RESULTS = json.load(fp)

N_QUBITS = ART["n_qubits"]
REF_MEANS = ART["scaler_cl"].mean_          # population means from training data
REF_STDS = ART["scaler_cl"].scale_          # population stds from training data

dev = qml.device("default.qubit", wires=N_QUBITS)

def _embedding(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[N_QUBITS - 1, 0])

@qml.qnode(dev)
def _state_circuit(x):
    _embedding(x)
    return qml.state()

@qml.qnode(dev, interface="autograd")
def _vqc_circuit(weights, x):
    _embedding(x)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

_WEIGHTS = pnp.array(ART["vqc_weights"], requires_grad=False)
_Q_STATES = ART["q_train_states"]

def q_features(A):
    Z = ART["pca"].transform(ART["scaler_q"].transform(np.atleast_2d(A)))
    lo, hi = ART["q_lo"], ART["q_hi"]
    Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return np.clip(Zn, 0, 1) * np.pi

def predict_all(x30):
    """30-dim raw vector -> {model_name: disease_probability}."""
    x = np.atleast_2d(np.asarray(x30, dtype=float))
    xc = ART["scaler_cl"].transform(x)
    xq = q_features(x)
    out = {n: float(c.predict_proba(xc)[0, 1]) for n, c in ART["classical_models"].items()}
    s = _state_circuit(pnp.array(xq[0], requires_grad=False))
    k = np.abs(_Q_STATES @ s.conj()) ** 2
    out["Quantum Kernel SVM (QSVC)"] = float(ART["qsvc"].predict_proba(k.reshape(1, -1))[0, 1])
    out["Variational Quantum Classifier (VQC)"] = float(
        (_vqc_circuit(_WEIGHTS, pnp.array(xq[0], requires_grad=False)) + 1.0) / 2.0)
    return out

def risk_band(p):
    if p >= 0.75: return ("HIGH RISK", "bad")
    if p >= 0.40: return ("MODERATE RISK", "warn")
    return ("LOW RISK", "ok")

CIRCUIT_ASCII = r"""
0: ──RY(x0)──╭●────────────────────────╭StronglyEntanglingLayers──┤ ⟨Z⟩
1: ──RY(x1)──╰X──╭●────────────────────┤  (2 layers, 24 params)   ──┤
2: ──RY(x2)──────╰X──╭●────────────────┤                          ──┤
3: ──RY(x3)──────────╰X──╭●────────────┤                          ──┤
                         ╰── (ring CNOT entangler)

  QSVC kernel :  K(x,x') = |⟨φ(x')|φ(x)⟩|²     (state fidelity)
  VQC readout :  p(disease) = (⟨Z₀⟩ + 1) / 2
  Backend     :  PennyLane default.qubit statevector simulator
                 (swap one line → IBM Quantum / Amazon Braket)
"""
