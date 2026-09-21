"""Tests for POST /api/verify.

Covers the landscape mapping (REAL->TRUE / FALSE->FALSE), input validation,
language modes, the guarantee that Gemini wins when available (FINAL engine
first, live check second), and the free-unlimited fallback chain the product
now guarantees:

  Gemini  ->  Ollama (judges the SAME gathered evidence)  ->  rule-engine
  (forced REAL/FAKE, confidence capped low)

Never a 502 just because an AI provider is down, and NEVER silently
UNVERIFIED: the endpoint always returns TRUE or FALSE.
"""

from __future__ import annotations

import os

import pytest

VALID_VERDICTS = {"TRUE", "FALSE", "UNVERIFIED"}


def _verification(verdict="UNVERIFIED", *, evidence=(), sources=(),
                  language=("english", "English"),
                  gemini_validation=None) -> dict:
    code, label = language
    return {
        "status": "completed",
        "language": {"code": code, "label": label},
        "pipeline": {"sources_used": list(sources)},
        "claims": [
            {"text": "A test claim about the Tamil Nadu school holiday.",
             "verdict": verdict, "evidence": list(evidence)},
        ] if verdict != "UNVERIFIED" else [],
        "overall": {"verdict": verdict, "confidence": 0.7,
                    "explanation": "reasoning from test", "mixed": False},
        "gemini_validation": gemini_validation,
        "evidence_matrix": [
            {**e, "claim": "A test claim"} for e in evidence
        ],
        "notes": {},
    }


@pytest.fixture()
def verify_client(client, monkeypatch):
    def _make(live=None, verdict="UNVERIFIED", evidence=(), sources=(),
              language=("english", "English"), gemini_validation=None,
              ollama=None, cloud=None, openai=None, verify_text=None):
        import app.routes.verify as verify_mod

        monkeypatch.setattr(verify_mod, "live_news_check", lambda *a, **k: live)
        monkeypatch.setattr(verify_mod, "run_verification", lambda *a, **k: _verification(
            verdict, evidence=evidence, sources=sources, language=language,
            gemini_validation=gemini_validation))
        monkeypatch.setattr(verify_mod, "ollama_judge", lambda *a, **k: ollama)
        monkeypatch.setattr(verify_mod, "cloud_ai_judge",
                            lambda *a, **k: cloud if cloud is not None else openai)
        if verify_text is not None:
            monkeypatch.setattr(verify_mod, "verify_text", verify_text)
        return client

    return _make


def _post(client, **payload):
    return client.post("/api/verify", json=payload)


def test_maps_real_to_true(verify_client):
    client = verify_client(live={"label": "REAL", "confidence": 0.9,
                                 "reasoning": "matches reporting"})
    resp = _post(client, headline="The Earth revolves around the Sun.",
                 article="Astronomers confirm heliocentrism.")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["final_verdict"] == "TRUE"
    assert body["confidence"] == 0.9
    assert body["verdict_source"] == "live-check"


def test_maps_false_to_false(verify_client):
    client = verify_client(live={"label": "FAKE", "confidence": 0.8,
                                 "reasoning": "contradicts facts"})
    resp = _post(client, headline="Vaccines cause autism.")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["final_verdict"] == "FALSE"
    assert body["verdict_source"] == "live-check"


def test_openai_fallback_judges_evidence(verify_client):
    """Gemini unavailable but a cloud AI (OpenAI) IS reachable: it judges the
    SAME gathered evidence and the endpoint returns its verdict."""
    client = verify_client(
        live={"label": "UNVERIFIED", "confidence": 0.4, "reasoning": "too recent"},
        verdict="UNVERIFIED",
        evidence=[{"title": "Widget Corp unveils fusion", "relation": "SUPPORT",
                   "source_name": "example.in", "url": "https://a.example"}],
        cloud={"source": "openai", "label": "REAL", "confidence": 0.66,
               "reasoning": "reporting corroborates the claim"},
        ollama={"label": "FAKE", "confidence": 0.9, "reasoning": "should not be used"},
    )
    resp = _post(client, headline="Widget Corp unveiled a fusion reactor.")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["final_verdict"] == "TRUE"
    assert body["verdict_source"] == "openai"
    assert body["confidence"] == 0.66
    assert body["reasoning"] == "reporting corroborates the claim"
    assert body["fallback_note"]
    assert "OpenAI" in body["fallback_note"]


def test_groq_fallback_labels_source(verify_client):
    """A Groq verdict is reported as the deciding source."""
    client = verify_client(
        live=None, verdict="UNVERIFIED",
        cloud={"source": "groq", "label": "FAKE", "confidence": 0.7,
               "reasoning": "evidence contradicts the claim"},
    )
    resp = _post(client, headline="Widget Corp invented warp drive.")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["final_verdict"] == "FALSE"
    assert body["verdict_source"] == "groq"
    assert "Groq" in body["fallback_note"]


def test_ollama_fallback_judges_evidence(verify_client):
    """Gemini unavailable but Ollama IS reachable: Ollama judges the SAME
    gathered evidence and the endpoint returns its verdict."""
    client = verify_client(
        live={"label": "UNVERIFIED", "confidence": 0.4, "reasoning": "too recent"},
        verdict="UNVERIFIED",
        evidence=[{"title": "Widget Corp unveils fusion", "relation": "SUPPORT",
                   "source_name": "example.in", "url": "https://a.example"}],
        ollama={"label": "FAKE", "confidence": 0.72,
                "reasoning": "evidence contradicts the claim"},
    )
    resp = _post(client, headline="Widget Corp invented warp drive.")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["final_verdict"] == "FALSE"
    assert body["verdict_source"] == "ollama"
    assert body["confidence"] == 0.72
    assert body["reasoning"] == "evidence contradicts the claim"
    assert body["fallback_note"]


def test_no_ai_provider_returns_error(verify_client):
    """Neither Gemini, OpenAI nor Ollama available: the API returns an explicit
    ai_unavailable error instead of fabricating a verdict."""
    client = verify_client(live=None, verdict="UNVERIFIED", ollama=None, openai=None)
    resp = _post(client, headline="Something nobody can confirm yet.")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "ai_unavailable"
    assert "unavailable" in body["error"].lower()


def test_no_ai_provider_error_even_with_false_evidence(verify_client):
    """Even when the evidence rules lean FALSE, no AI available means an error -
    the API never presents an evidence-only guess as the AI verdict."""
    client = verify_client(
        live=None, verdict="FALSE", sources=("factcheck", "newsapi"),
        ollama=None, openai=None,
        evidence=[{"title": "Fabrication exposed", "relation": "CONTRADICT",
                   "source_name": "factcheck.org", "url": "https://f.example"}],
    )
    resp = _post(client, headline="A clearly fabricated report.")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "ai_unavailable"


def test_gemini_final_validation_resolves_when_no_live(verify_client):
    """At the LAST step Gemini validates the result and explains WHY."""
    client = verify_client(
        live=None,
        verdict="UNVERIFIED",
        gemini_validation={
            "available": True, "source": "gemini", "model": "m",
            "label": "REAL", "confidence": 0.91, "agrees": True,
            "reasoning": "Trusted outlets and officials confirm the report.",
        },
    )
    resp = _post(client, headline="Elections were announced on Friday.",
                 article="Election commission confirms the announcement.")
    body = resp.get_json()
    assert body["final_verdict"] == "TRUE"
    assert body["confidence"] == 0.91
    assert body["reasoning"] == "Trusted outlets and officials confirm the report."
    assert body["gemini_validation"]["label"] == "REAL"
    assert body["gemini_validation"]["reasoning"]


def test_final_engine_is_the_decision_maker(verify_client):
    """Gemini's FINAL engine (REAL/FAKE) always wins over the live check -
    the local models only ever give a suggestion."""
    client = verify_client(
        live={"label": "FAKE", "confidence": 0.95, "reasoning": "contradicts facts"},
        verdict="REAL",
        gemini_validation={
            "available": True, "source": "gemini", "model": "m",
            "label": "REAL", "confidence": 0.8, "agrees": True,
            "reasoning": "looks credible",
        },
    )
    resp = _post(client, headline="Vaccines cause autism.")
    body = resp.get_json()
    assert body["final_verdict"] == "TRUE"
    assert body["confidence"] == 0.8
    assert body["reasoning"] == "looks credible"
    assert body["gemini_validation"]["label"] == "REAL"


def test_live_check_decides_when_engine_unavailable(verify_client):
    """When the FINAL engine did not run, Gemini's live check is used."""
    client = verify_client(
        live={"label": "FAKE", "confidence": 0.95, "reasoning": "contradicts facts"},
        verdict="REAL",
        gemini_validation=None,
    )
    resp = _post(client, headline="Vaccines cause autism.")
    assert resp.get_json()["final_verdict"] == "FALSE"


def test_missing_gemini_never_invents_sources(verify_client):
    """The no-fabrication guarantee: when no AI provider produced a verdict the
    API reports ai_unavailable rather than inventing sources or a verdict."""
    client = verify_client(live=None, verdict="UNVERIFIED", ollama=None, openai=None)
    resp = _post(client, headline="No known reporting about this claim.",
                 article="This is a completely invented claim text.")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "ai_unavailable"
    assert "final_verdict" not in body


def test_invalid_language_rejected():
    from app import create_app
    from app.config import TestConfig

    with create_app(TestConfig).test_client() as client:
        resp = _post(client, headline="Hello", language="klingon")
        assert resp.status_code == 400


def test_missing_text_rejected(verify_client):
    client = verify_client()
    resp = _post(client, article="   ")
    assert resp.status_code == 400


def test_text_too_long_rejected(verify_client):
    client = verify_client()
    resp = _post(client, article="x" * 40000)
    assert resp.status_code == 400


def test_english_language_mode_honoured(verify_client, monkeypatch):
    captured = {}

    def _verify_text(text, headline=None, language_mode="auto", include_debug=False):
        captured["mode"] = language_mode
        return _verification("UNVERIFIED")

    client = verify_client(
        live={"label": "REAL", "confidence": 0.9, "reasoning": "matches reporting"},
        verdict="UNVERIFIED",
        language=("english", "English"),
        verify_text=_verify_text,
    )
    resp = _post(client, headline="Hello world", language="english")
    assert resp.status_code == 200
    assert captured["mode"] == "english"
    assert resp.get_json()["language_mode"] == "english"


def test_full_pipeline_offline_no_key():
    """End-to-end fully offline: with no AI provider reachable the endpoint
    returns an explicit ai_unavailable error - it never fabricates a verdict."""
    from app import create_app
    from app.config import TestConfig
    import app.routes.verify as verify_mod

    monkeypatch_os = pytest.MonkeyPatch()
    monkeypatch_os.setenv("TRUTHLENS_LIVE_EVIDENCE", "0")
    monkeypatch_os.setenv("GEMINI_API_KEY", "")
    monkeypatch_os.setenv("OPENAI_API_KEY", "")
    monkeypatch_os.setenv("BAZAARLINK_API_KEY", "")
    try:
        with create_app(TestConfig).test_client() as client:
            monkeypatch_os.setattr(verify_mod, "ollama_judge", lambda *a, **k: None)
            monkeypatch_os.setattr(verify_mod, "cloud_ai_judge", lambda *a, **k: None)
            resp = _post(client, headline="The Earth revolves around the Sun.",
                         article="Astronomers say the planet orbits the star.")
            assert resp.status_code == 503
            body = resp.get_json()
            assert body["status"] == "ai_unavailable"
            assert "unavailable" in body["error"].lower()
    finally:
        monkeypatch_os.undo()
        for name in ("TRUTHLENS_LIVE_EVIDENCE", "GEMINI_API_KEY",
                     "OPENAI_API_KEY", "BAZAARLINK_API_KEY"):
            os.environ.pop(name, None)