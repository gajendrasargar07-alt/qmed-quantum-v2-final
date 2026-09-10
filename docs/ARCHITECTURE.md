# QMed-Quantum — Technical Architecture (Team Synergy 6 · SIH26139)

## Pipeline
1. **Intake** — `src/report_parser.py`: PDF/TXT/CSV → text (pypdf) → deterministic NLP
   (alias maps + fuzzy match) → 30 cytology features, biomarker panel, SNP genotypes.
   Missing features imputed with training population means and visibly flagged.
2. **Risk layer** — `src/prs.py`: 10-SNP GWAS panel; per-allele odds ratios multiplied
   against population expectation → relative-risk multiplier + banding.
3. **Classical branch** — StandardScaler → Logistic Regression / Random Forest / SVM-RBF.
4. **Quantum branch** — StandardScaler → PCA(4) → [0,π] angle embedding + ring-CNOT entangler.
   - **QSVC**: fidelity kernel K(x,x′)=|⟨φ(x′)|φ(x)⟩|² → precomputed-kernel SVM.
   - **VQC**: 2-layer StronglyEntanglingLayers (24 params), p(disease)=(⟨Z₀⟩+1)/2, Adam + BCE.
5. **Benchmarking** — accuracy / sensitivity / specificity / F1 / ROC-AUC / train & inference time.
6. **Early detection** — Youden-optimal threshold search with sensitivity floor 0.97 per model.
7. **Explainability** — permutation importance on clinical features and quantum PCA components.
8. **Explanation** — `src/explainer.py`: grounded natural-language cards (no hallucinated numbers);
   optional LLM polish when `OPENAI_API_KEY` is set.
9. **UI** — `streamlit_app.py` (flagship) + legacy Flask REST API (`web/app.py`).

## Scalability notes
- Quantum training is subsample-limited (kernel is O(n²)); switch to `default.tensor` or
  shot-based backends for larger cohorts; batch kernel computation via PennyLane `broadcast`.
- Inference engine loads artifacts once (module-level singletons) — safe for Streamlit reruns
  and multi-worker Flask (`gunicorn -w 4`).
- Swap dataset by replacing `load_breast_cancer()` in `src/train.py`; pipeline is dataset-agnostic.

## Hardware portability
Circuits: 4 qubits, depth ≈ 12, hardware-efficient ansatz — within NISQ budgets.
`qml.device("qiskit.remote", wires=4, backend="ibm_brisbane")` requires no circuit changes.
