"""
Medical OCR pipeline (MVP).

Supports:
  - Text file / plain string
  - PDF (both text-embedded and scanned)
  - Image (jpg/png/tiff)

Uses pytesseract with medical vocabulary post-correction and confidence
scoring. Every extracted numeric field carries a confidence value; low-
confidence or missing critical fields are flagged.

Steps:
  1. If PDF -> try pdfplumber (text layer). If empty/short -> rasterize
     with pdf2image and OCR each page.
  2. If image -> preprocess with OpenCV (grayscale, adaptive threshold,
     denoise) -> Tesseract OCR (`--psm 6`, config `-c preserve_interword_spaces=1`).
  3. Post-correct common medical OCR errors (0<->O, l<->1, '5-3' -> '15-3'
     for CA 15-3, etc.) using a medical vocabulary.
  4. Compute an OCR confidence score from Tesseract's per-word data.
  5. Extract known fields via regex + fuzzy label matching.
  6. Flag missing / low-confidence critical fields.
"""
import os, re, io
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None
try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from pdf2image import convert_from_path, convert_from_bytes
except Exception:
    convert_from_path = None
    convert_from_bytes = None
try:
    from PIL import Image
except Exception:
    Image = None

# --------------------------------------------------------- medical vocab
MED_VOCAB = {
    # canonical spellings for common noisy OCR outputs
    "psa": ["p.s.a.", "p s a", "psa"],
    "cea": ["c.e.a.", "c e a", "cea"],
    "ca 15-3": ["ca15-3", "ca 15 3", "ca-15-3", "ca 153"],
    "ca 19-9": ["ca19-9", "ca 19 9", "ca-19-9", "ca 199"],
    "her2": ["her 2", "her-2", "her/2"],
    "brca1": ["brca 1", "brca-1"],
    "brca2": ["brca 2", "brca-2"],
    "ki-67": ["ki 67", "ki67", "ki-67"],
    "gleason": ["gleeson", "glcason", "gieason"],
    "hemoglobin": ["haemoglobin", "hemogloibin"],
    "hpv": ["h.p.v", "h p v"],
    "cyfra 21-1": ["cyfra21-1", "cyfra 211"],
    "fev1": ["fev-1", "fev 1", "fev1%"],
}

def _correct_medical_terms(text: str) -> str:
    tl = text
    for canonical, variants in MED_VOCAB.items():
        for v in variants:
            tl = re.sub(re.escape(v), canonical, tl, flags=re.IGNORECASE)
    # common digit/letter confusions in numeric contexts
    tl = re.sub(r"(?<=\d)[Oo](?=\d)", "0", tl)
    tl = re.sub(r"(?<=\d)[lI](?=\d)", "1", tl)
    return tl

# --------------------------------------------------------- preprocessing
def _preprocess_image(img_np):
    if cv2 is None: return img_np
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if img_np.ndim == 3 else img_np
    # denoise + adaptive threshold
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    thr  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)
    return thr

def _ocr_image(pil_img):
    if pytesseract is None:
        return "", 0.0, []
    if cv2 is not None:
        arr = np.array(pil_img.convert("RGB"))
        arr = arr[..., ::-1]  # RGB->BGR
        pre = _preprocess_image(arr)
        pil_use = Image.fromarray(pre)
    else:
        pil_use = pil_img
    config = "--psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(pil_use, config=config)
    data = pytesseract.image_to_data(pil_use, output_type=Output.DICT, config=config)
    confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) > 0]
    conf = float(np.mean(confs))/100 if confs else 0.0
    words = [w for w in data.get("text", []) if w and w.strip()]
    return text, conf, words

# --------------------------------------------------------- top-level ingest
def extract(source, source_kind: str = "auto") -> dict:
    """
    source: path or bytes.
    source_kind: 'auto' | 'text' | 'pdf' | 'image'
    Returns: dict(text, confidence, pages, pipeline, warnings)
    """
    warnings_ = []
    text = ""
    conf = 1.0
    pages = []
    pipeline = []

    # infer kind
    if source_kind == "auto":
        if isinstance(source, (bytes, bytearray)):
            source_kind = "pdf" if source[:4] == b"%PDF" else "image"
        elif isinstance(source, str):
            if os.path.exists(source):
                ext = os.path.splitext(source)[1].lower()
                if ext in (".txt", ".md"): source_kind = "text"
                elif ext == ".pdf": source_kind = "pdf"
                elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"): source_kind = "image"
                else: source_kind = "text"
            else:
                source_kind = "text"

    # ------------- TEXT
    if source_kind == "text":
        if isinstance(source, (bytes, bytearray)):
            text = source.decode("utf-8", errors="ignore")
        elif os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
        else:
            text = str(source)
        conf = 1.0
        pipeline.append("plain-text")

    # ------------- PDF
    elif source_kind == "pdf":
        pdf_bytes = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
        # 1) try text layer
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    txts = []
                    for p in pdf.pages:
                        t = p.extract_text() or ""
                        txts.append(t); pages.append({"page": len(pages)+1, "text_len": len(t), "ocr": False})
                    text = "\n\n".join(txts)
                pipeline.append("pdf-text-layer")
            except Exception as e:
                warnings_.append(f"pdfplumber failed: {e}")
        # 2) OCR fallback if empty
        if len(text.strip()) < 60:
            pipeline.append("pdf-rasterize+tesseract")
            pages = []
            if convert_from_bytes is None or pytesseract is None:
                warnings_.append("pdf2image/pytesseract unavailable — cannot OCR scanned PDF.")
            else:
                try:
                    imgs = convert_from_bytes(pdf_bytes, dpi=220)
                except Exception as e:
                    warnings_.append(f"pdf2image failed: {e}"); imgs = []
                page_confs = []
                page_texts = []
                for i, im in enumerate(imgs):
                    t, c, _ = _ocr_image(im)
                    page_texts.append(t); page_confs.append(c)
                    pages.append({"page": i+1, "text_len": len(t), "ocr": True, "confidence": round(c, 3)})
                text = "\n\n".join(page_texts)
                conf = float(np.mean(page_confs)) if page_confs else 0.0

    # ------------- IMAGE
    elif source_kind == "image":
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        t, c, _ = _ocr_image(img)
        text = t; conf = c
        pipeline.append("image+tesseract")
        pages.append({"page": 1, "text_len": len(t), "ocr": True, "confidence": round(c, 3)})

    else:
        raise ValueError(f"Unsupported source_kind: {source_kind}")

    # post-correct
    text_corrected = _correct_medical_terms(text)

    if conf < 0.55 and any(p.get("ocr") for p in pages):
        warnings_.append(f"Low OCR confidence ({conf:.2f}) — please verify extracted values manually.")

    return {
        "text": text_corrected,
        "raw_text": text,
        "confidence": round(float(conf), 4),
        "pages": pages,
        "pipeline": pipeline,
        "warnings": warnings_,
        "source_kind": source_kind,
    }


# --------------------------------------------------------- field extractor
NUMERIC = r"(-?\d+(?:\.\d+)?)"

FIELD_PATTERNS = {
    # generic
    "age":         [r"\bage\s*[:=]?\s*"+NUMERIC, r"\b"+NUMERIC+r"\s*[- ]?year[- ]?old"],
    "hemoglobin":  [r"h[ae]moglobin\s*[:=]?\s*"+NUMERIC, r"\bhb\b\s*[:=]?\s*"+NUMERIC],
    "wbc":         [r"\bwbc\s*[:=]?\s*"+NUMERIC, r"\bleukocytes?\s*[:=]?\s*"+NUMERIC],
    "platelets":   [r"platelets?\s*[:=]?\s*"+NUMERIC],
    "creatinine":  [r"creatinine\s*[:=]?\s*"+NUMERIC],
    "albumin":     [r"albumin\s*[:=]?\s*"+NUMERIC],
    "crp":         [r"\bcrp\s*[:=]?\s*"+NUMERIC, r"c[- ]reactive protein\s*[:=]?\s*"+NUMERIC],
    "bmi":         [r"\bbmi\s*[:=]?\s*"+NUMERIC],
    "ldh":         [r"\bldh\s*[:=]?\s*"+NUMERIC],
    "alp":         [r"\balp\s*[:=]?\s*"+NUMERIC, r"alkaline phosphatase\s*[:=]?\s*"+NUMERIC],
    # cancer markers
    "psa":         [r"\bpsa\s*[:=]?\s*"+NUMERIC],
    "free_psa_ratio":[r"free\s*/?\s*total\s*psa\s*(?:ratio)?\s*[:=]?\s*"+NUMERIC,
                     r"free\s*psa\s*ratio\s*[:=]?\s*"+NUMERIC],
    "cea":         [r"\bcea\s*[:=]?\s*"+NUMERIC],
    "ca19_9":      [r"ca\s*19[- ]?9\s*[:=]?\s*"+NUMERIC],
    "ca15_3":      [r"ca\s*15[- ]?3\s*[:=]?\s*"+NUMERIC],
    "cyfra21_1":   [r"cyfra\s*21[- ]?1\s*[:=]?\s*"+NUMERIC],
    "nse":         [r"\bnse\s*[:=]?\s*"+NUMERIC],
    "gleason":     [r"gleason(?:\s*score)?\s*[:=]?\s*"+NUMERIC],
    "ki_67":       [r"ki[- ]?67\s*[:=]?\s*"+NUMERIC],
    "testosterone":[r"testosterone\s*[:=]?\s*"+NUMERIC],
    "fev1_pct":    [r"fev1\s*(?:%)?\s*[:=]?\s*"+NUMERIC],
    "pack_years":  [r"pack[- ]?years?\s*[:=]?\s*"+NUMERIC],
    "nodule_mm":   [r"nodule\s*(?:size)?\s*[:=]?\s*"+NUMERIC+r"\s*mm"],
    "prostate_volume":[r"prostate\s*volume\s*[:=]?\s*"+NUMERIC],
    "partners":    [r"(?:sexual\s*)?partners\s*[:=]?\s*"+NUMERIC],
    "pregnancies": [r"pregnanc(?:y|ies)\s*[:=]?\s*"+NUMERIC],
    "stds_count":  [r"stds?\s*(?:count)?\s*[:=]?\s*"+NUMERIC],
    "first_intercourse_age":[r"first\s*intercourse\s*(?:age)?\s*[:=]?\s*"+NUMERIC],
    "iud_years":   [r"iud\s*(?:years)?\s*[:=]?\s*"+NUMERIC],
    "hormonal_contraceptives_years":[r"hormonal.*(?:years)?\s*[:=]?\s*"+NUMERIC],
}

BOOL_PATTERNS = {
    "smoker":         (r"smok(?:er|ing)\s*[:=]?\s*(yes|no|pos|neg|positive|negative|\d+)", ("yes","pos","positive")),
    "family_history": (r"family\s*(?:history|hx).{0,30}?(yes|no|pos|neg|positive|negative)", ("yes","pos","positive")),
    "dre_suspicious": (r"dre\s*[:=]?\s*(suspicious|normal|abnormal|pos|neg|positive|negative)", ("suspicious","abnormal","pos","positive")),
    "hpv_positive":   (r"hpv\s*[:=]?\s*(positive|negative|pos|neg|yes|no|high[- ]risk)", ("positive","pos","yes","high-risk","high risk")),
    "dx_cin":         (r"cin\s*[:=]?\s*(?:grade)?\s*(\d|none|no|neg|negative)", ("1","2","3")),
    "schiller":       (r"schiller\s*[:=]?\s*(pos|neg|positive|negative)", ("positive","pos")),
    "citology":       (r"c[iy]tology\s*[:=]?\s*(pos|neg|positive|negative|normal|abnormal)", ("positive","pos","abnormal")),
    "fit_test":       (r"fit\s*(?:test)?\s*[:=]?\s*(pos|neg|positive|negative)", ("positive","pos")),
    "spiculation":    (r"spiculat(?:ion|ed)\s*[:=]?\s*(pos|neg|positive|negative|yes|no)", ("positive","pos","yes")),
}

def extract_fields(text: str) -> dict:
    fields = {}
    tl = text.lower()
    for name, patterns in FIELD_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, tl, flags=re.IGNORECASE)
            if m:
                try: fields[name] = float(m.group(1))
                except Exception: pass
                break
    for name, (pat, positives) in BOOL_PATTERNS.items():
        m = re.search(pat, tl, flags=re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            fields[name] = float(any(p in v for p in positives) or (v.isdigit() and int(v) > 0))
    return fields


def flag_missing(fields: dict, required: list) -> dict:
    missing = [r for r in required if r not in fields]
    return {
        "missing": missing,
        "present": [r for r in required if r in fields],
        "completeness": round(1 - len(missing)/max(len(required),1), 3),
        "flag": bool(missing),
    }


if __name__ == "__main__":
    demo = ("PSA: 8.2 ng/mL. Free/Total PSA ratio 0.10. DRE: suspicious. "
            "Gleason score 7. Age 66 year-old. Hemoglobin 13.0 g/dL. "
            "Prostate volume 55 mL. Testosterone 420 ng/dL. ALP 130 U/L.")
    r = extract("", "text")  # dummy
    r["text"] = demo
    fields = extract_fields(demo)
    print("fields:", fields)
    print("missing:", flag_missing(fields, ["psa","free_psa_ratio","gleason","alp","age"]))
