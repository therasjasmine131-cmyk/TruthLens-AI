"""Tests for analysis behaviour: headline-only honesty cap and live check."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Config
from app.ml.model_manager import model_manager
from app.services.analyzer import analyze

FAKE_PROBA = (0.05, 0.95)  # p_real, p_fake -> FAKE with confidence ~0.855

_METADATA = {
    "best_model": "Random Forest",
    "model_class": "RandomForestClassifier",
    "vectorizer": "TfidfVectorizer (5,000 features, unigrams)",
    "dataset_source": "ISOT Fake News (full)",
    "train_samples": 26938,
    "test_samples": 8980,
    "n_features": 5000,
    "metrics": {"accuracy": 0.99},
}


def _patch_model(monkeypatch):
    monkeypatch.setattr(
        model_manager, "bundle", SimpleNamespace(_loaded=True)
    )
    monkeypatch.setattr(model_manager, "predict_proba", lambda text: FAKE_PROBA)
    monkeypatch.setattr(model_manager, "top_tfidf_terms", lambda text, top_n=10: [])
    monkeypatch.setattr(model_manager, "metadata", lambda: dict(_METADATA))
    monkeypatch.setattr(
        "app.services.analyzer.explain_prediction",
        lambda *a, **k: {"method": None, "features": [], "note": ""},
    )
    monkeypatch.setattr(
        "app.services.analyzer.get_setting",
        lambda *a, **k: dict(Config.CONFIDENCE_LEVELS),
    )


def test_headline_only_caps_high_confidence(monkeypatch):
    _patch_model(monkeypatch)
    result = analyze("Tamil Nadu inks MoUs worth Rs 67,000 crore", None, save=False)
    assert result["headline_only"] is True
    assert result["caveat"]
    assert result["prediction"] == "FAKE"
    assert result["confidence_level"] not in {"High Confidence", "Very High Confidence"}
    assert result["confidence_level"] == "Moderate Confidence"


def test_full_article_keeps_high_confidence(monkeypatch):
    _patch_model(monkeypatch)
    article = (
        "The government signed ninety seven memorandums of understanding. "
        "The agreements were exchanged at the investment conclave in Chennai. "
        "The projects are expected to create more than one hundred thousand jobs."
    )
    result = analyze("Tamil Nadu inks MoUs worth Rs 67,000 crore", article, save=False)
    assert result["headline_only"] is False
    assert result["caveat"] is None
    assert result["confidence_level"] == "High Confidence"


def test_analyze_headline_route_includes_live_check(client, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        "app.routes.analyze.live_news_check",
        lambda *a, **k: {
            "available": True,
            "source": "gemini",
            "label": "REAL",
            "confidence": 0.9,
            "reasoning": "This matches real reporting on the Vettri conclave.",
        },
    )
    monkeypatch.setattr(Config, "GEMINI_API_KEY", "test-key")
    resp = client.post(
        "/api/analyze/headline",
        json={"headline": "Tamil Nadu inks MoUs worth Rs 67,000 crore"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["headline_only"] is True
    assert body["confidence_level"] == "Moderate Confidence"
    assert body["live_check"]["label"] == "REAL"
    assert body["live_check"]["source"] == "gemini"
    assert body["history_id"] is not None


def test_analyze_headline_route_without_key_omits_live_check(client, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(Config, "GEMINI_API_KEY", "")
    resp = client.post("/api/analyze/headline", json={"headline": "A short headline here"})
    assert resp.status_code == 200
    assert resp.get_json()["live_check"] is None
