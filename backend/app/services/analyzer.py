"""Orchestrates a full analysis: validation -> stats -> ML -> explanation -> save."""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..extensions import db
from ..ml.model_manager import model_manager
from ..models import Prediction
from ..utils.errors import ValidationError
from ..utils.validators import validate_inputs
from .explainer import explain_prediction
from .probabilities import confidence_level, three_way_prediction
from .settings_service import get as get_setting
from .text_stats import compute_text_stats


def _model_info() -> dict:
    meta = model_manager.metadata()
    metrics = meta.get("metrics", {})
    return {
        "name": meta.get("best_model", "Unknown"),
        "model_class": meta.get("model_class"),
        "vectorizer": meta.get("vectorizer"),
        "dataset_source": meta.get("dataset_source"),
        "train_samples": meta.get("train_samples"),
        "test_samples": meta.get("test_samples"),
        "n_features": meta.get("n_features"),
        "training_date": meta.get("training_date"),
        "explainability": meta.get("explainability"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
        },
    }


def analyze(headline: str | None, article: str | None, *, save: bool = True) -> dict:
    """Run the full analysis pipeline and return a JSON-safe result."""
    validate_inputs(headline, article)

    headline = (headline or "").strip()
    article = (article or "").strip()
    combined = (headline + " " + article).strip()

    levels = get_setting("confidence_levels", Config.CONFIDENCE_LEVELS)
    article_stats = compute_text_stats(headline, article)
    keywords = model_manager.top_tfidf_terms(combined, top_n=10)

    p_real, p_fake = model_manager.predict_proba(combined)
    result = three_way_prediction(p_real, p_fake)
    explanation = explain_prediction(combined, result["prediction"], top_n=8)

    model_info = _model_info()
    prediction = result["prediction"]

    payload = {
        "prediction": prediction,
        "confidence": result["confidence"],
        "confidence_level": confidence_level(result["confidence"], levels),
        "confidence_bands": levels,
        "probabilities": result["probabilities"],
        "model_raw": result["model_raw"],
        "model": model_info["name"],
        "model_info": model_info,
        "keywords": keywords,
        "article_stats": article_stats,
        "explanation": explanation,
        "disclaimer": (
            "TruthLens AI provides machine-learning-based estimates from patterns "
            "learned from its training data. A prediction is not proof that an "
            "article is true or false. Always verify important claims using "
            "reliable sources."
        ),
        "saved": False,
    }

    if save:
        record = Prediction(
            headline=headline,
            article_text=article,
            prediction=prediction,
            real_probability=result["probabilities"]["real"],
            fake_probability=result["probabilities"]["fake"],
            uncertain_probability=result["probabilities"]["uncertain"],
            confidence=result["confidence"],
            model_name=model_info["name"],
            word_count=article_stats["word_count"],
            character_count=article_stats["character_count"],
            sentence_count=article_stats["sentence_count"],
            top_keywords=[k["term"] for k in keywords],
            analysis_metadata={
                "article_stats": article_stats,
                "keywords": keywords,
                "probabilities": result["probabilities"],
                "model_raw": result["model_raw"],
                "explanation": explanation,
                "model_info": model_info,
                "confidence_level": payload["confidence_level"],
            },
        )
        db.session.add(record)
        db.session.commit()
        payload["saved"] = True
        payload["history_id"] = record.id

    return payload


def analyze_headline_only(headline: str) -> dict:
    """Analyze using just a headline (article text is empty)."""
    return analyze(headline, None)


def build_report_data(record: Prediction) -> dict:
    """Assemble a full printable report for a stored prediction."""
    meta: dict[str, Any] = record.analysis_metadata or {}
    return {
        "history_id": record.id,
        "headline": record.headline,
        "article_text": record.article_text,
        "prediction": record.prediction,
        "confidence": record.confidence,
        "confidence_level": meta.get("confidence_level", confidence_level(record.confidence)),
        "probabilities": meta.get("probabilities") or {
            "real": record.real_probability,
            "fake": record.fake_probability,
            "uncertain": record.uncertain_probability,
        },
        "model_raw": meta.get("model_raw") or {},
        "article_stats": meta.get("article_stats") or {
            "word_count": record.word_count,
            "character_count": record.character_count,
            "sentence_count": record.sentence_count,
        },
        "keywords": meta.get("keywords") or [{"term": t} for t in (record.top_keywords or [])],
        "explanation": meta.get("explanation") or {},
        "model_info": meta.get("model_info") or {"name": record.model_name},
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "disclaimer": (
            "TruthLens AI provides machine-learning-based estimates from patterns "
            "learned from its training data. A prediction is not proof that an "
            "article is true or false. Always verify important claims using "
            "reliable sources."
        ),
    }
