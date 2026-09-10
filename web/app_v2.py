"""
QMed-Quantum v2 — REST API for the multi-cancer pipeline.
Original web/app.py is untouched. Run: python3 web/app_v2.py -> :5001
Endpoints:
  GET  /api/v2/cancers                        list trained cancer bundles
  GET  /api/v2/metrics                        multi-cancer benchmark JSON
  POST /api/v2/analyze     {text, cancer?}    full pipeline on pasted text
  POST /api/v2/analyze_file (multipart)       upload txt/pdf/image
  POST /api/v2/feedback    {cancer,features,true_label,predicted_prob,notes}
  POST /api/v2/update      {cancer,mode}      trigger safe/semi update
"""
import os, sys, json
from flask import Flask, request, jsonify

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [os.path.join(BASE, "src"), os.path.join(BASE, "src", "ocr"),
          os.path.join(BASE, "src", "validator"), os.path.join(BASE, "src", "multicancer"),
          os.path.join(BASE, "src", "ensemble"), os.path.join(BASE, "src", "selflearn")]:
    sys.path.insert(0, p)

from pipeline import analyze
from feedback import log_feedback, apply_incremental_update

app = Flask(__name__)

def _list_cancers():
    d = os.path.join(BASE, "models", "multicancer")
    return sorted([f.replace(".joblib","") for f in os.listdir(d) if f.endswith(".joblib")])

@app.route("/api/v2/cancers")
def cancers():
    return jsonify({"cancers": _list_cancers()})

@app.route("/api/v2/metrics")
def metrics():
    p = os.path.join(BASE, "results", "multicancer_results.json")
    if not os.path.exists(p): return jsonify({"error":"not trained yet"}), 404
    with open(p) as fp: return jsonify(json.load(fp))

@app.route("/api/v2/analyze", methods=["POST"])
def analyze_text():
    body = request.get_json(force=True) or {}
    text = body.get("text","")
    cancer = body.get("cancer")
    r = analyze(text, "text", override_cancer=cancer)
    return jsonify(r)

@app.route("/api/v2/analyze_file", methods=["POST"])
def analyze_file():
    f = request.files.get("file")
    if not f: return jsonify({"error":"no file"}), 400
    data = f.read()
    ext = os.path.splitext(f.filename)[1].lower()
    kind = ("text" if ext == ".txt" else "pdf" if ext == ".pdf" else "image")
    cancer = request.form.get("cancer") or None
    r = analyze(data, kind, override_cancer=cancer)
    return jsonify(r)

@app.route("/api/v2/feedback", methods=["POST"])
def feedback():
    b = request.get_json(force=True) or {}
    rec = log_feedback(b["cancer"], b["features"], int(b["true_label"]),
                       float(b.get("predicted_prob", 0)), b.get("notes",""))
    return jsonify({"ok": True, "record": rec})

@app.route("/api/v2/update", methods=["POST"])
def update():
    b = request.get_json(force=True) or {}
    return jsonify(apply_incremental_update(b["cancer"], mode=b.get("mode","safe")))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
