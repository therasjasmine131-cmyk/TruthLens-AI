"""Tests for the AI-text detector package."""

from __future__ import annotations

from ml.ai_text_detector import detect_ai_generated
from ml.ai_text_detector.detector import _score_to_label
from ml.ai_text_detector.features import burstiness_score, marker_density, repetition_score
from ml.ai_text_detector.gemini import GeminiClassifier, _extract_score
from ml.ai_text_detector.lm import BigramLM, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, WORLD 123!") == ["hello", "world", "123"]


def test_lm_predictability_prefers_common_text():
    data = {
        "unigrams": {"the": 100, "a": 60, "cat": 40, "dog": 30, "sat": 25, "on": 50, "mat": 20, "zebra": 1},
        "bigrams": {"the cat": 30, "cat sat": 20, "sat on": 15, "on the": 25, "the mat": 5},
        "total_tokens": 326,
        "reference_log_prob": -4.0,
        "reference_sd": 0.6,
    }
    lm = BigramLM(data)
    assert lm.available
    common = lm.predictability_component("the cat sat on the mat and the dog sat")
    rare = lm.predictability_component("the zebra sat on the mat and the zebra sat")
    assert common > rare


def test_bigram_lm_missing_data_unavailable():
    assert not BigramLM({}).available


def test_burstiness_score_ranges():
    assert 0.0 <= burstiness_score("One. Two. Three.") <= 1.0
    assert burstiness_score("") in (0.0, 0.5)


def test_repetition_score_repeated_text_high():
    repeated = "the cat sat the cat sat the cat sat the cat sat"
    assert repetition_score(repeated) > repetition_score("a quick brown fox jumped over the lazy dog fence gate")


def test_marker_density_picks_up_ai_phrases():
    assert marker_density("It is important to note that. Furthermore, in conclusion.") > 0.5


def test_extract_score_parses_json_and_bare():
    assert _extract_score('{"score": 0.85}') == 0.85
    assert _extract_score('json {"score": 0}') == 0.0
    assert _extract_score("nothing here") is None


def test_gemini_classifier_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not GeminiClassifier(api_key="").available
    assert GeminiClassifier(api_key="abc").available


def test_detect_empty_text_returns_error():
    result = detect_ai_generated("   ")
    assert result.get("status") == "error"


def test_detect_heuristic_backend_without_gemini(monkeypatch):
    monkeypatch.setattr("ml.ai_text_detector.detector.AiTextDetector._gemini_score", lambda self, text: None)
    result = detect_ai_generated(
        "Artificial intelligence is transforming the modern world. "
        "It is important to note that this technology plays a crucial role. "
        "Furthermore, in conclusion, we must embrace these changes and adapt."
    )
    assert "ai_generated_score" in result
    assert result["backend"] == "heuristic"
    assert 0.0 <= result["ai_generated_score"] <= 1.0


def test_score_to_label_bands():
    assert _score_to_label(0.8) == "Likely AI"
    assert _score_to_label(0.1) == "Likely Human"
    assert _score_to_label(0.5) == "Uncertain"
