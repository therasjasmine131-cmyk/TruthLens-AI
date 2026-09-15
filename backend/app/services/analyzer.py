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
from .verification_service import run_verification

HEADLINE_ONLY_CAVEAT = (
    "Headline-only analysis: with only a headline to work from, the model has "
    "very little text and cannot verify real-world events. Treat this as a rough "
    "guess, not a verdict. Verify with the full article and reliable sources."
)

_HEADLINE_ONLY_CAP = "Moderate Confidence"
_HIGH_LEVELS = ("Very High Confidence", "High Confidence")


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


def analyze(headline: str | None, article: str | None, *, save: bool = True,
            include_debug: bool = False) -> dict:
    """Run the full analysis pipeline and return a JSON-safe result."""
    validate_inputs(headline, article)

    headline = (headline or "").strip()
    article = (article or "").strip()
    headline_only = not article
    combined = (headline + " " + article).strip()

    levels = get_setting("confidence_levels", Config.CONFIDENCE_LEVELS)
    article_stats = compute_text_stats(headline, article)

    ml_ready = model_manager.ready
    if ml_ready:
        p_real, p_fake = model_manager.predict_proba(combined)
        result = three_way_prediction(p_real, p_fake)
        keywords = model_manager.top_tfidf_terms(combined, top_n=10)
    else:
        result = {
            "prediction": None,
            "confidence": 0.0,
            "decided": False,
            "uncertain_threshold": None,
            "probabilities": {"real": None, "fake": None, "uncertain": None},
            "model_raw": None,
        }
        keywords = []

    explanation = explain_prediction(
        combined, result["prediction"] or "UNVERIFIED", top_n=8
    ) if ml_ready and result["prediction"] else {}

    model_info = _model_info() if ml_ready else {"name": "Unavailable", "metrics": {}}
    prediction = result["prediction"]

    level = confidence_level(result["confidence"], levels) if ml_ready else "Low Confidence"
    if headline_only and level in _HIGH_LEVELS:
        level = _HEADLINE_ONLY_CAP

    verification = run_verification(headline, article, include_debug=include_debug)
    overall_verdict = None
    no_evidence_overall = None
    if verification:
        overall_verdict = verification["overall"]["verdict"]
        no_evidence_overall = verification["overall"].get("confidence_label", "") == "No live evidence retrieved"

    payload = {
        "prediction": prediction,
        "confidence": result["confidence"],
        "confidence_level": level,
        "confidence_bands": levels,
        "decided": result["decided"],
        "uncertain_threshold": result["uncertain_threshold"],
        "headline_only": headline_only,
        "caveat": HEADLINE_ONLY_CAVEAT if headline_only else None,
        "probabilities": result["probabilities"],
        "model_raw": result["model_raw"],
        "model": model_info["name"],
        "model_info": model_info,
        "keywords": keywords,
        "article_stats": article_stats,
        "explanation": explanation,
        "disclaimer": (
            "This is an ML-based prediction, not proof of factual truth. "
            "TruthLens AI reflects statistical patterns learned from a training "
            "dataset and can be wrong. Always verify important claims using "
            "reliable sources."
        ),
        "saved": False,
        "verification": verification,
        "verdict": overall_verdict,
        "verdict_inference_only": no_evidence_overall,
    }

    if save and prediction is not None:
        try:
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
                    "verification": verification,
                },
            )
            db.session.add(record)
            db.session.commit()
            payload["saved"] = True
            payload["history_id"] = record.id
        except Exception as exc:  # ephemeral/read-only storage must not break analyze
            db.session.rollback()
            payload["saved"] = False
            payload["save_error"] = str(exc)

    return payload


def analyze_headline_only(headline: str, *, include_debug: bool = False) -> dict:
    """Analyze using just a headline (article text is empty)."""
    return analyze(headline, None, include_debug=include_debug)


def build_report_data(record: Prediction) -> dict:
    """Assemble a full printable report for a stored prediction."""
    meta: dict[str, Any] = record.analysis_metadata or {}
    headline_only = not (record.article_text or "").strip()
    stored_level = meta.get("confidence_level", confidence_level(record.confidence))
    if headline_only and stored_level in _HIGH_LEVELS:
        stored_level = _HEADLINE_ONLY_CAP
    return {
        "history_id": record.id,
        "headline": record.headline,
        "article_text": record.article_text,
        "prediction": record.prediction,
        "confidence": record.confidence,
        "confidence_level": stored_level,
        "headline_only": headline_only,
        "caveat": HEADLINE_ONLY_CAVEAT if headline_only else None,
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
            "This is an ML-based prediction, not proof of factual truth. "
            "TruthLens AI reflects statistical patterns learned from a training "
            "dataset and can be wrong. Always verify important claims using "
            "reliable sources."
        ),
    }
