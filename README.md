# QMed-Quantum — Hybrid Quantum ML Platform for Early Disease Detection
**Team Synergy 6 · SIH 2026 · Problem Statement SIH26139 · Egreen Quanta · MedTech/BioTech/HealthTech**
Python 3.12.3 · PennyLane 0.45.1 · scikit-learn 1.9.0 · Streamlit

## What it does
Upload a real clinical report (FNA cytology, biomarker panel, DNA genotype export — PDF/TXT/CSV) →
the AI Report Intake Engine extracts the 30 cytology features, biomarkers (BRCA1/2, HER2, ER/PR,
Ki-67, CA 15-3, TP53…) and SNP genotypes → 5 models (3 classical + 2 hybrid quantum) predict →
a 10-SNP GWAS polygenic risk score is computed → a grounded AI explainer breaks everything down
in plain language.

## Flagship features
1. **AI report intake** — deterministic NLP parser (alias maps + fuzzy matching) for lab/genetic reports
2. **Polygenic Risk Scoring** — 10 published breast-cancer GWAS SNPs with per-allele odds ratios
3. **Hybrid quantum diagnosis** — Quantum Kernel SVM (fidelity kernel) + Variational Quantum Classifier
4. **Early-detection mode** — sensitivity-prioritized threshold tuning (≥0.97 sensitivity floor)
5. **Explainability** — permutation importance on clinical features AND quantum components
6. **Plain-language AI explainer** — every sentence grounded in actual numbers (optional LLM polish)
7. **Flagship Streamlit UI** — 7 pages, dark glass theme, Synergy 6 branding

## Benchmark (114-patient held-out test set)
| Model | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | .9649 | .9286 | .9861 | .9512 | .9960 |
| Random Forest | .9649 | .9048 | 1.000 | .9500 | .9942 |
| Classical SVM (RBF) | .9649 | .9524 | .9722 | .9524 | .9947 |
| **Quantum Kernel SVM** | **.9825** | **.9762** | .9861 | **.9762** | **.9974** |
| Variational Quantum Classifier | .8246 | .5238 | 1.000 | .6875 | .9643 |

## Quick start
```bash
pip install -r requirements.txt
python3 tests/test_smoke.py       # verify everything
streamlit run streamlit_app.py    # flagship UI  -> http://localhost:8501
python3 web/app.py                # legacy Flask REST API -> http://localhost:5000
python3 src/train.py              # retrain all models from scratch
```
See `docs/DEMO_SCRIPT.md` (8-minute judge walkthrough + Q&A) and `docs/ARCHITECTURE.md`.
**Disclaimer:** research prototype for SIH 2026 — not a certified medical device.
