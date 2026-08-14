"""Tests for the /api/detect-ai-text endpoint."""

from __future__ import annotations


def test_detect_ai_text_returns_heuristic(client, monkeypatch):
    # Force the offline heuristic path (no Gemini / network in tests).
    monkeypatch.setattr(
        "ml.ai_text_detector.detector.AiTextDetector._gemini_score", lambda self, text: None
    )
    resp = client.post(
        "/api/detect-ai-text",
        json={
            "text": (
                "Artificial intelligence is transforming the modern world in significant ways. "
                "It is important to note that this technology plays a crucial role in many sectors. "
                "Furthermore, in conclusion, we must embrace these changes and adapt to the future."
            )
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["backend"] == "heuristic"
    assert 0.0 <= body["ai_generated_score"] <= 1.0
    assert body["label"] in {"Likely AI", "Likely Human", "Uncertain"}
    assert "signals" in body


def test_detect_ai_text_rejects_short_text(client):
    resp = client.post("/api/detect-ai-text", json={"text": "too short"})
    assert resp.status_code == 400
    assert resp.get_json()["status"] == "error"


def test_detect_ai_text_rejects_missing_text(client):
    resp = client.post("/api/detect-ai-text", json={})
    assert resp.status_code == 400


def test_detect_ai_text_rejects_oversized_text(client, monkeypatch):
    monkeypatch.setattr(
        "ml.ai_text_detector.detector.AiTextDetector._gemini_score", lambda self, text: None
    )
    resp = client.post("/api/detect-ai-text", json={"text": "x" * 50000})
    assert resp.status_code == 400
