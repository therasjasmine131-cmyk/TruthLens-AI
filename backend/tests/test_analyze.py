"""Tests for the honesty of the REAL / FAKE / UNCERTAIN probability scheme.

We verify that we never manufacture an uncertainty percentage and that the
probability distribution always reflects the raw binary model output (so
REAL + FAKE = 100% and UNCERTAIN is an abstain decision, not a probability).
"""

from __future__ import annotations

from app.services.probabilities import (  # noqa: E402
    LABEL_FAKE,
    LABEL_REAL,
    LABEL_UNCERTAIN,
    UNCERTAIN_THRESHOLD,
    three_way_prediction,
)

from types import SimpleNamespace  # noqa: E402

from app.config import Config  # noqa: E402
from app.ml.model_manager import model_manager  # noqa: E402
from app.services.analyzer import analyze  # noqa: E402


def test_high_confidence_predicts_fake_raw_probabilities():
    result = three_way_prediction(0.2, 0.8)
    assert result["prediction"] == LABEL_FAKE
    assert result["decided"] is True
    assert result["confidence"] == 0.8
    # Raw binary probabilities are preserved and sum to 1 (100%).
    assert abs(result["probabilities"]["real"] + result["probabilities"]["fake"] - 1.0) < 1e-6
    assert result["probabilities"]["uncertain"] == 0.0
    assert result["probabilities"]["real"] == 0.2
    assert result["probabilities"]["fake"] == 0.8


def test_high_confidence_predicts_real_raw_probabilities():
    result = three_way_prediction(0.92, 0.08)
    assert result["prediction"] == LABEL_REAL
    assert result["decided"] is True
    assert result["confidence"] == 0.92
    assert abs(result["probabilities"]["real"] + result["probabilities"]["fake"] - 1.0) < 1e-6
    assert result["probabilities"]["uncertain"] == 0.0
    assert result["probabilities"]["real"] == 0.92


def test_low_confidence_abstains_as_uncertain_without_manufacturing():
    # A genuine toss-up: the model is only ~60% sure. We abstain but we must
    # NOT create a fake third probability; real + fake still sum to 100%.
    result = three_way_prediction(0.6, 0.4)
    assert result["prediction"] == LABEL_UNCERTAIN
    assert result["decided"] is False
    assert result["confidence"] == 0.6
    assert result["confidence"] < UNCERTAIN_THRESHOLD
    assert abs(result["probabilities"]["real"] + result["probabilities"]["fake"] - 1.0) < 1e-6
    assert result["probabilities"]["uncertain"] == 0.0
    # No invented residual: real + fake + uncertain still equals exactly 1.0.
    total = (
        result["probabilities"]["real"]
        + result["probabilities"]["fake"]
        + result["probabilities"]["uncertain"]
    )
    assert abs(total - 1.0) < 1e-6


def test_unshifted_inputs_sum_to_one():
    # p_real + p_fake are normalized; even skewed inputs must yield 100%.
    result = three_way_prediction(10.0, 1.0)  # out of [0,1] -> clamped+normalized
    assert abs(result["probabilities"]["real"] + result["probabilities"]["fake"] - 1.0) < 1e-6
    assert result["probabilities"]["uncertain"] == 0.0


FAKE_PROBA = (0.2, 0.8)  # p_real, p_fake -> FAKE with confidence ~0.80 (High band)

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
