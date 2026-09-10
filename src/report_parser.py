"""
QMed-Quantum · Report Intake Engine (Synergy 6)
Parses clinical lab reports, genetic test reports and raw DNA genotype exports
(23andMe-style) into model-ready inputs.

Three extraction layers:
  1. Clinical feature extraction  -> the 30 FNA cytology features the ML models consume
  2. Genetic marker extraction    -> BRCA1/2, HER2, ER/PR, Ki-67, CA 15-3, CEA ...
  3. SNP genotype extraction      -> rsID genotypes for polygenic risk scoring (prs.py)

No external API required — deterministic NLP (regex + alias maps + fuzzy matching).
"""
import re, io
import numpy as np

# ----------------------------------------------------------------------------- 30 model features
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error",
    "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension",
]

# alias -> canonical feature (all lowercase keys)
ALIAS = {}
def _alias(canon, *names):
    for n in names + (canon, canon.replace(" ", "_")):
        ALIAS[n.lower()] = canon

_alias("mean radius", "radius mean", "mean radius (mm)", "radius_mean", "avg radius")
_alias("mean texture", "texture mean", "mean texture", "texture_mean")
_alias("mean perimeter", "perimeter mean", "mean perimeter (mm)", "perimeter_mean")
_alias("mean area", "area mean", "mean area (mm2)", "mean area (mm²)", "area_mean")
_alias("mean smoothness", "smoothness mean", "smoothness_mean")
_alias("mean compactness", "compactness mean", "compactness_mean")
_alias("mean concavity", "concavity mean", "concavity_mean")
_alias("mean concave points", "concave points mean", "mean concave point",
       "concave_points_mean", "concave points_mean")
_alias("mean symmetry", "symmetry mean", "symmetry_mean")
_alias("mean fractal dimension", "fractal dimension mean", "fractal_dimension_mean")
for base, se in [("radius", "radius error"), ("texture", "texture error"),
                 ("perimeter", "perimeter error"), ("area", "area error"),
                 ("smoothness", "smoothness error"), ("compactness", "compactness error"),
                 ("concavity", "concavity error"), ("concave points", "concave points error"),
                 ("symmetry", "symmetry error"), ("fractal dimension", "fractal dimension error")]:
    _alias(se, f"{base} se", f"{base} std error", f"se {base}", f"{base}_error",
           f"{base} standard error")
for base in ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
             "concavity", "concave points", "symmetry", "fractal dimension"]:
    _alias(f"worst {base}", f"{base} worst", f"largest {base}", f"max {base}",
           f"worst_{base}", f"{base}_worst")

GENETIC_MARKERS = {
    "BRCA1": r"\bBRCA\s*-?\s*1\b", "BRCA2": r"\bBRCA\s*-?\s*2\b",
    "HER2": r"\bHER\s*-?\s*2\b|\bERBB2\b", "ER (Estrogen Receptor)": r"\bestrogen receptor\b|\bER\s*(status|positive|negative|:)",
    "PR (Progesterone Receptor)": r"\bprogesterone receptor\b|\bPR\s*(status|positive|negative|:)",
    "Ki-67": r"\bKi\s*-?\s*67\b", "CA 15-3": r"\bCA\s*15[\-\s]?3\b", "CA 27.29": r"\bCA\s*27[.\-]?29\b",
    "CEA": r"\bCEA\b|carcinoembryonic", "TP53": r"\bTP53\b|\bp53\b", "PALB2": r"\bPALB2\b",
    "CHEK2": r"\bCHEK2\b", "ATM": r"\bATM\b",
}
POS_NEG = re.compile(r"(positive|negative|detected|not detected|pathogenic|benign|variant of uncertain significance|vus)", re.I)

NUM = r"(-?\d+(?:\.\d+)?)"

def extract_text(filename: str, raw: bytes) -> str:
    """txt / csv / pdf -> plain text."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            return f"[PDF extraction failed: {e}]"
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

def _fuzzy_feature(token: str):
    t = token.strip().lower().rstrip(":")
    if t in ALIAS:
        return ALIAS[t], 1.0
    # contains-match fallback
    for k in sorted(ALIAS, key=len, reverse=True):
        if t and (t in k or k in t) and len(t) > 5:
            return ALIAS[k], 0.8
    return None, 0.0

def parse_clinical_features(text: str):
    """Find 'name : value' / 'name = value' pairs and map to the 30 model features."""
    found = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(rf"^[\"']?([A-Za-z][A-Za-z0-9 _()\-\./²]{{2,45}}?)[\"']?\s*[:=,\t]\s*{NUM}", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        canon, conf = _fuzzy_feature(name)
        if canon and canon not in found:
            found[canon] = {"value": val, "matched_as": name.strip(), "confidence": conf}
    # also try CSV header row + one data row (exported lab CSVs)
    if len(found) < 5:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            header = [h.strip() for h in re.split(r"[,;\t]", lines[0])]
            for row in lines[1:6]:
                cells = [c.strip() for c in re.split(r"[,;\t]", row)]
                if len(cells) != len(header):
                    continue
                for h, c in zip(header, cells):
                    try:
                        val = float(c)
                    except ValueError:
                        continue
                    canon, conf = _fuzzy_feature(h)
                    if canon and canon not in found:
                        found[canon] = {"value": val, "matched_as": h, "confidence": conf * 0.9}
    return found

def parse_genetic_markers(text: str):
    out = []
    for label, pat in GENETIC_MARKERS.items():
        for m in re.finditer(pat, text, re.I):
            window = text[max(0, m.start() - 60): m.end() + 80]
            status = POS_NEG.search(window)
            out.append({"marker": label,
                        "context": " ".join(window.split())[:140],
                        "status": status.group(1).capitalize() if status else "Mentioned"})
            break
    return out

def parse_snps(text: str):
    """23andMe-style: 'rsID  chrom  pos  genotype'. Also catches 'rs1234: AG'."""
    snps = {}
    for m in re.finditer(r"\b(rs\d{3,10})\b[\s,:;]+(?:[\dXYMT]{1,2}\s+\d+\s+)?([ACGTDI-]{1,2})\b", text):
        g = m.group(2).upper().replace("-", "")
        if g and g not in ("D", "I"):
            snps[m.group(1)] = g
    return snps

def assemble_vector(found: dict, reference_means: np.ndarray):
    """Build the 30-dim vector; missing features imputed with population means (flagged)."""
    vec, report = [], []
    for i, f in enumerate(FEATURES):
        if f in found:
            vec.append(found[f]["value"])
            report.append({"feature": f, "value": found[f]["value"], "source": "report",
                           "confidence": found[f]["confidence"], "matched_as": found[f]["matched_as"]})
        else:
            vec.append(float(reference_means[i]))
            report.append({"feature": f, "value": float(reference_means[i]),
                           "source": "imputed (population mean)", "confidence": 0.0, "matched_as": "-"})
    return np.array(vec, dtype=float), report
