# QMed-Quantum v2 — Multi-Cancer Early-Detection Platform
**Team Synergy 6 · SIH 2026 · Problem Statement SIH26139 · Egreen Quanta**

This is the **v2 extension** of the original QMed-Quantum project. The original
files (single-cancer breast pipeline) are **fully preserved and still work** —
v2 adds a multi-cancer stack alongside them.

## What's new in v2

| Feature | Module |
|---|---|
| 5 cancer domains (breast / lung / prostate / colon / cervical) | `src/multicancer/datasets.py` |
| 6 models per cancer (4 classical + 2 quantum) = **30 trained models** | `src/multicancer/train_all.py` |
| Cancer-type router (TF-IDF + LogReg) | `src/multicancer/train_router.py` |
| Stage classifier (Stage 0-IV, RandomForest) per cancer | inside each bundle |
| Early-risk / pre-cancer predictor (Gradient Boosting) per cancer | inside each bundle |
| Medical-document validator (rejects random / non-medical docs) | `src/validator/medical_validator.py` |
| Medical OCR (Tesseract + medical-vocab correction + confidence) | `src/ocr/ocr_pipeline.py` |
| Field extraction + missing-critical-field flagging | `src/ocr/ocr_pipeline.py` |
| Safe self-learning (feedback log + SGD `partial_fit`) | `src/selflearn/feedback.py` |
| Unified scoring engine (weighted classical + quantum ensemble → single score) | `src/ensemble/unified_score.py` |
| Full pipeline orchestrator (upload → validate → route → score) | `src/pipeline.py` |
| New Streamlit app with 5 pages | `streamlit_app_v2.py` |
| New Flask REST API | `web/app_v2.py` |
| v2 smoke tests | `tests_new/test_v2_smoke.py` |

## Pipeline

```
upload (txt / pdf / scanned image)
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
        ├─ Classical: LogReg · RandomForest · GradientBoost · SVM(RBF)
        ├─ Quantum:   Quantum Kernel SVM  +  Variational Quantum Classifier
        ├─ Stage classifier (multiclass RF)
        └─ Early-risk classifier (pre-cancer GBM)
   │
   ▼   Weighted final score + risk band + recommended action
```

## Benchmarks (per-cancer test-set, held-out 22 %)

| Cancer | Best model | Accuracy | ROC-AUC | Sensitivity |
|---|---|---|---|---|
| Breast   | Classical SVM (RBF)   | 0.984 | 0.995 | 0.957 |
| Lung     | Quantum Kernel SVM    | 1.000 | 1.000 | 1.000 |
| Prostate | Random Forest         | 0.989 | 0.999 | 0.987 |
| Colon    | Logistic Regression   | 0.994 | 1.000 | 1.000 |
| Cervical | Logistic Regression   | 0.983 | 0.999 | 0.968 |

Stage classifier accuracy on positive cases: **93.6 % – 100 %** across domains.

## Quick start

```bash
pip install -r requirements.txt          # existing deps + pennylane already listed
# (system) sudo apt-get install -y tesseract-ocr poppler-utils

# retrain everything (only if you want to re-run)
python3 src/multicancer/train_all.py     # ~8 minutes MVP-sized
python3 src/multicancer/train_router.py  # ~2 seconds

# smoke-test the v2 stack (76 checks)
python3 tests_new/test_v2_smoke.py

# run the new UI
streamlit run streamlit_app_v2.py        # -> http://localhost:8501

# run the new REST API
python3 web/app_v2.py                    # -> http://localhost:5001

# original UI still works
streamlit run streamlit_app.py
python3 web/app.py                       # -> http://localhost:5000
```

## Self-learning

Feedback is **logged first, applied second** — the medically-responsible pattern.

```python
from src.selflearn.feedback import log_feedback, apply_incremental_update
log_feedback("prostate", features, true_label=1, predicted_prob=0.91)
apply_incremental_update("prostate", mode="safe")   # log only (default)
apply_incremental_update("prostate", mode="semi")   # partial_fit on SGD when >= 4 pending
```

A full retrain is a manual admin action: `python3 src/multicancer/train_all.py`.

## Data honesty

- **Breast**: real UCI Breast Cancer Wisconsin (569 patients, 30 features).
- **Lung / Prostate / Colon / Cervical**: literature-grounded synthetic
  datasets (800 patients each), with feature distributions and inter-feature
  correlations aligned to published clinical ranges (PSA, CEA, CA 19-9, HPV,
  Gleason, etc.). This is an MVP; production would replace these with real
  cohort data (TCGA / NCI / Kaggle / hospital partners).

## Disclaimer

Research prototype for SIH 2026 — **not a certified medical device**. Do not
use the outputs for clinical decisions without physician review.
