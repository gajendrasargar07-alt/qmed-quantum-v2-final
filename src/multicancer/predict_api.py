"""Thin convenience API for UIs — keeps original streamlit_app.py untouched."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pipeline import analyze
from unified_score import score, score_from_fields, _load as load_bundle
from ocr_pipeline import extract, extract_fields
from medical_validator import validate as validate_text
from selflearn.feedback import log_feedback, apply_incremental_update

def list_cancers():
    return sorted([f.replace(".joblib","") for f in os.listdir(
        os.path.join(os.path.dirname(HERE), "..", "models", "multicancer"))
        if f.endswith(".joblib")])
