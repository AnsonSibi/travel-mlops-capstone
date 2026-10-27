"""Inference helpers for the gender model (kept free of training deps like MLflow)."""
import numpy as np
import pandas as pd

UNCERTAIN_THRESHOLD = 0.6


def first_name(names) -> pd.Series:
    """'Roy Braun' -> '^roy$' (anchors let n-grams capture prefixes/suffixes)."""
    names = pd.Series(names).astype(str)
    return "^" + names.str.strip().str.split().str[0].str.lower() + "$"


def predict_gender(model, names, threshold: float = UNCERTAIN_THRESHOLD) -> list[dict]:
    """Predict with a confidence threshold - shared by the API and Streamlit."""
    proba = model.predict_proba(pd.Series(names))
    classes = model.classes_
    out = []
    for name, p in zip(names, proba):
        best = int(np.argmax(p))
        label = classes[best] if p[best] >= threshold else "uncertain"
        out.append({"name": name, "gender": label, "confidence": round(float(p[best]), 4),
                    "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)}})
    return out
