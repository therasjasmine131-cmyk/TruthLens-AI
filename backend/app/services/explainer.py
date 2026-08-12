"""Explainable AI.

Produces a per-prediction explanation from the *actual* model internals:

* **Linear / probabilistic models** (LogisticRegression, LinearSVC,
  MultinomialNB): coefficients are multiplied by the document's TF-IDF value for
  each present feature, giving the per-feature contribution to the decision.
* **Tree ensembles** (Random Forest): the model has no per-prediction
  coefficients, so we rank features by ``document TF-IDF weight * global
  feature_importance``. This is clearly labelled as a proxy.
"""

from __future__ import annotations

import numpy as np

from ..ml.model_manager import model_manager

POSITIVE = "positive"
NEGATIVE = "negative"
LOW = "low"


def _coef_contributions(text: str, top_n: int) -> list[dict]:
    bundle = model_manager.bundle
    clf = bundle.model.named_steps["clf"]
    coef = clf.coef_
    if hasattr(coef, "toarray"):
        coef = coef.toarray()
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 2:
        coef = coef[0]

    # coefficient sign convention: positive -> REAL, negative -> FAKE
    names = bundle.feature_names
    vec = bundle.vectorizer.transform([text])
    doc_weights = vec.toarray()[0]

    contribs = []
    for idx in np.argsort(np.abs(coef))[::-1]:
        w = float(doc_weights[idx])
        c = float(coef[idx])
        if w <= 0:
            continue
        contribs.append(
            {
                "term": names[idx],
                "weight": round(c, 5),
                "contribution": round(c * w, 5),
            }
        )
        if len(contribs) >= top_n:
            break
    return contribs


def _importance_proxy(text: str, top_n: int) -> list[dict]:
    bundle = model_manager.bundle
    names = bundle.feature_names
    importances = np.asarray(bundle.feature_importances or [], dtype=float)
    if importances.size == 0:
        return []
    vec = bundle.vectorizer.transform([text])
    doc_weights = vec.toarray()[0]
    contribs = []
    for idx in np.argsort(importances * doc_weights)[::-1]:
        if doc_weights[idx] <= 0:
            continue
        contribs.append(
            {
                "term": names[idx],
                "weight": round(float(importances[idx]), 6),
                "contribution": round(float(importances[idx] * doc_weights[idx]), 6),
            }
        )
        if len(contribs) >= top_n:
            break
    return contribs


def explain_prediction(text: str, prediction: str, top_n: int = 8) -> dict:
    """Return features that influenced the prediction, with direction labels."""
    if not model_manager.ready:
        return {"method": None, "features": [], "note": "No model available."}

    bundle = model_manager.bundle
    explainability = bundle.metadata.get("explainability", "feature_importance")
    clf_name = bundle.model.named_steps["clf"].__class__.__name__

    if explainability == "coefficients":
        raw = _coef_contributions(text, top_n)
        method = "model-coefficients"
        note = (
            "Coefficients from the linear decision boundary, scaled by each "
            "term's TF-IDF weight in this article."
        )
        for item in raw:
            if item["contribution"] >= 0.005:
                item["influence"] = "positive"
            elif item["contribution"] <= -0.005:
                item["influence"] = "negative"
            else:
                item["influence"] = "low"
        features = raw
    else:
        raw = _importance_proxy(text, top_n)
        method = "tfidf-x-feature-importance"
        note = (
            "Tree ensembles have no per-prediction coefficients, so features are "
            "ranked by their TF-IDF weight in this article multiplied by the "
            "model's global feature importance (proxy)."
        )
        features = []
        for rank, item in enumerate(raw):
            item["influence"] = "positive" if rank < 3 else "low"
            features.append(item)

    return {
        "method": method,
        "model_class": clf_name,
        "features": features,
        "note": note,
        "direction_label": (
            "Positive influence pushes toward REAL; negative pushes toward FAKE."
            if method == "model-coefficients"
            else "Higher-ranked terms carry more weight in the decision."
        ),
    }
