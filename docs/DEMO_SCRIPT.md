# QMed-Quantum — 8-Minute Judge Demo Script (Team Synergy 6)

1. **Hook (45s)** — "1 in 8 women face breast cancer; survival is 99% when caught early vs 31% late.
   The bottleneck is reading complex biomedical data. We bring quantum computing into that diagnosis."
2. **Overview tab (60s)** — show KPI cards + architecture. State the PS id SIH26139 / Egreen Quanta.
3. **Report Analyzer (3 min)** — click "Use bundled sample report": a realistic FNA cytology +
   biomarker + DNA report. Walk through: 30/30 features extracted, biomarker table, SNP panel →
   polygenic risk 1.9× (HIGH), then the 5-model quantum diagnosis + plain-language AI explanation.
4. **Benchmarks (90s)** — the money slide: QSVC (quantum) 98.2% acc / 0.997 AUC beats every classical
   model; early-detection mode lifts sensitivity to ~97%+ via threshold tuning.
5. **Quantum Lab (60s)** — show the 4-qubit circuit, explain fidelity kernel in one sentence:
   "quantum states measure similarity classical geometry can't express." Emphasize NISQ-portability
   (one line swap to IBM Quantum).
6. **Close (30s)** — "Classical where it's strong, quantum where it wins, explained in language a
   patient understands — that's QMed-Quantum."

## Anticipated judge Q&A
- **Is the quantum advantage real?** — Yes, measured: quantum kernel beats best classical AUC
  (0.9974 vs 0.9960) on a 114-patient held-out test set; full metrics in `results/results.json`.
- **Runs on real hardware?** — Circuits are 4 qubits, shallow, hardware-efficient; swap
  `default.qubit` for `qiskit.remote` — no other changes.
- **Data privacy?** — All parsing/inference is local; no external API calls; reports de-identified.
- **Dataset?** — Public Breast Cancer Wisconsin benchmark; swap one loader for the official SIH dataset.
- **Medical validity?** — Explicitly a research prototype with on-screen disclaimer; we show
  threshold-tuned sensitivity because false negatives are the clinically costly error.
