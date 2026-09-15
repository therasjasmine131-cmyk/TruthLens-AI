"""Model manager.

Loads the trained artifacts (``best_model.joblib``, ``best_vectorizer.joblib``,
``model_metadata.json``) from disk once and serves predictions. The loader is
lazy so the API can boot (and report health) even when artifacts are missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from ..config import Config


@dataclass
class ModelBundle:
    model: object
    vectorizer: object
    metadata: dict
    feature_names: list[str] = field(default_factory=list)
    feature_importances: list[float] = field(default_factory=list)
    load_error: str | None = None
    _loaded: bool = False


class ModelManager:
    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir or Config.ML_ARTIFACTS_DIR)
        self.bundle: ModelBundle | None = None

    # ---- loading ---------------------------------------------------------
    def load(self) -> ModelBundle | None:
        """Load (or reload) the artifacts. Returns the bundle or None."""
        model_path = self.artifacts_dir / "best_model.joblib"
        vectorizer_path = self.artifacts_dir / "best_vectorizer.joblib"
        metadata_path = self.artifacts_dir / "model_metadata.json"

        if not all(p.exists() for p in (model_path, vectorizer_path, metadata_path)):
            self.bundle = None
            return None

        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = list(vectorizer.get_feature_names_out())
            clf = model.named_steps["clf"]
            importances = (
                list(np.asarray(clf.feature_importances_, dtype=float))
                if hasattr(clf, "feature_importances_")
                else []
            )
            self.bundle = ModelBundle(
                model=model,
                vectorizer=vectorizer,
                metadata=metadata,
                feature_names=feature_names,
                feature_importances=importances,
                _loaded=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.bundle = ModelBundle(model=None, vectorizer=None, metadata={}, load_error=str(exc))
        return self.bundle

    @property
    def ready(self) -> bool:
        if self.bundle is None:
            return False
        return self.bundle._loaded

    def metadata(self) -> dict:
        if not self.ready:
            return {}
        return self.bundle.metadata

    # ---- prediction ------------------------------------------------------
    def predict_proba(self, text: str):
        """Return (p_real, p_fake) or raise when model is unavailable."""
        if not self.ready:
            raise RuntimeError("ML model is not available. Train the model first.")
        model = self.bundle.model
        vectorizer = self.bundle.vectorizer
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            # Full pipeline: vectorizer is embedded, classifier is the last step.
            clf = model.named_steps["clf"]
            vec = model[:-1]
        else:
            # Bare classifier trained on a separately-saved vectorizer.
            clf = model
            vec = vectorizer
        proba = clf.predict_proba(vec.transform([text]))[0]
        labels = self.bundle.metadata.get("class_labels", ["FAKE", "REAL"])
        p_real = float(proba[labels.index("REAL")])
        p_fake = float(proba[labels.index("FAKE")])
        return p_real, p_fake

    def top_tfidf_terms(self, text: str, top_n: int = 10) -> list[dict]:
        """TF-IDF-weighted keywords for one document, computed from the model's
        own vectorizer (not hard-coded)."""
        if not self.ready:
            return []
        vec = self.bundle.vectorizer.transform([text])
        scores = vec.toarray()[0]
        order = np.argsort(scores)[::-1][:top_n]
        terms = []
        names = self.bundle.feature_names
        for idx in order:
            if scores[idx] <= 0:
                continue
            terms.append({"term": names[idx], "score": round(float(scores[idx]), 4)})
        return terms

    def status(self) -> dict:
        if self.ready:
            return {"status": "online", "model": self.bundle.metadata.get("best_model")}
        if self.bundle and self.bundle.load_error:
            return {"status": "error", "error": self.bundle.load_error}
        return {"status": "offline", "error": "artifacts not found"}


model_manager = ModelManager()
