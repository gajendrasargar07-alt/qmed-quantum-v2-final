"""
Medical-document validator.

Given raw text extracted from an uploaded document, decides whether it looks
like a medical/lab/pathology/oncology report. Rejects random docs.

Approach (deterministic, no external LLM):
  1. Compute a "medical density" = fraction of tokens matching a curated
     medical lexicon (units, biomarkers, common lab terms, cancer terms).
  2. Compute a "numeric-with-units density" (lab results usually have
     values like "12.3 ng/mL", "PSA: 7.4").
  3. Combine into a score in [0,1]. Threshold: 0.18 accept, else reject.
Also returns: matched_terms, matched_units, coarse_type_guess.
"""
import re

MEDICAL_LEXICON = {
    # generic medical
    "patient", "report", "clinical", "diagnosis", "history", "impression",
    "findings", "specimen", "biopsy", "pathology", "laboratory", "lab",
    "radiology", "imaging", "ultrasound", "mri", "ct", "pet", "xray",
    "hospital", "physician", "referred", "sample", "collected",
    # cancer / oncology
    "cancer", "tumor", "tumour", "malignant", "benign", "carcinoma",
    "adenocarcinoma", "sarcoma", "lymphoma", "leukemia", "metastasis",
    "metastatic", "oncology", "neoplasm", "dysplasia", "stage",
    # markers
    "brca1", "brca2", "her2", "ki-67", "ki67", "er", "pr", "psa", "cea",
    "ca19-9", "ca15-3", "cyfra", "nse", "afp", "hcg", "ldh", "alp",
    "hemoglobin", "wbc", "platelets", "creatinine", "albumin", "crp",
    "testosterone", "hpv", "cin", "gleason", "fev1", "cyfra21-1",
    "birads", "pi-rads",
    # breast features
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave", "symmetry", "fractal",
    # cervical
    "pap", "smear", "cytology", "schiller", "iud", "hormonal",
    # colon
    "colonoscopy", "polyp", "adenoma", "colorectal", "fit", "fecal",
    # lung
    "nodule", "spiculated", "pack-years", "pack", "years", "smoker",
    # cell / molecular
    "mutation", "gene", "genotype", "snp", "rs", "allele", "expression",
    "positive", "negative", "elevated", "decreased", "normal",
    "grade", "score", "level", "count", "mg", "ml", "dl", "mm",
    "biomarker", "marker", "panel",
}

UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ng/ml|mg/dl|g/dl|u/ml|iu/ml|meq/l|mmhg|"
    r"cells/[uμ]l|/mm3|mm\^?3|ml|mm|cm|kg|%|units?|iu)\b",
    re.IGNORECASE,
)

CANCER_HINTS = {
    "breast":   ["breast", "brca", "her2", "mammogram", "birads", "ductal"],
    "lung":     ["lung", "pulmonary", "chest ct", "nodule", "smoker", "fev1", "cyfra", "nsclc"],
    "prostate": ["prostate", "psa", "gleason", "dre", "transrectal"],
    "colon":    ["colon", "colorectal", "cea", "ca19-9", "colonoscopy", "polyp", "fit test"],
    "cervical": ["cervix", "cervical", "pap", "hpv", "schiller", "cin", "colposcopy"],
}


def _tokens(text):
    return [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", text)]


def validate(text: str, min_score: float = 0.18):
    """Return dict: is_medical, score, matched_terms, matched_units, reason."""
    if not text or len(text.strip()) < 25:
        return {
            "is_medical": False, "score": 0.0,
            "matched_terms": [], "matched_units": 0,
            "cancer_hint": None,
            "reason": "Text too short — not enough content to be a medical report.",
        }
    toks = _tokens(text)
    n = max(len(toks), 1)
    matched = [t for t in toks if t in MEDICAL_LEXICON]
    matched_unique = sorted(set(matched))
    med_density = len(matched) / n
    units = len(UNIT_PATTERN.findall(text))
    unit_density = min(units / 20.0, 1.0)  # cap
    # combined score
    score = 0.65 * med_density + 0.35 * unit_density
    # boost slightly if we saw common cancer/oncology anchor terms
    if any(k in text.lower() for k in ("cancer", "tumor", "tumour", "malignant",
                                        "carcinoma", "oncology", "biopsy",
                                        "psa", "brca", "hpv", "pap smear")):
        score = min(score + 0.08, 1.0)

    is_medical = score >= min_score and len(matched_unique) >= 4
    reason = (
        f"Medical density {med_density:.2%}, unit hits {units}, "
        f"{len(matched_unique)} lexicon terms matched."
    )
    if not is_medical:
        reason = "Rejected — " + reason + " Below threshold for a valid medical report."

    # coarse cancer-type guess
    hint = None; best = 0
    tl = text.lower()
    for k, kws in CANCER_HINTS.items():
        c = sum(1 for kw in kws if kw in tl)
        if c > best: best, hint = c, k
    if best < 1: hint = None

    return {
        "is_medical": bool(is_medical),
        "score": round(float(score), 4),
        "matched_terms": matched_unique[:40],
        "matched_units": int(units),
        "cancer_hint": hint,
        "reason": reason,
    }


if __name__ == "__main__":
    ok = "Patient PSA 7.4 ng/mL, free/total ratio 0.11, DRE suspicious. Prostate biopsy scheduled. Gleason 3+4. Hemoglobin 13.1 g/dL."
    bad = "Once upon a time in a faraway land, there lived a happy cat that loved to eat fish and sleep under the sun all day long."
    print(validate(ok)); print(validate(bad))
