"""Tests for POST /api/verify.

Covers the landscape mapping (REAL->TRUE / FALSE->FALSE / UNVERIFIED->UNVERIFIED),
input validation, language modes, and the *no invented sources* guarantee: when
nothing credible supports or contradicts a claim the endpoint must answer
UNVERIFIED and must not fabricate evidence.
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
def verify_client(client):
    def _make(live=None, verdict="UNVERIFIED", evidence=(), sources=(),
              language=("english", "English"), gemini_validation=None):
        import app.routes.verify as verify_mod

        verify_mod.live_news_check = lambda *a, **k: live
        verify_mod.run_verification = lambda *a, **k: _verification(
            verdict, evidence=evidence, sources=sources, language=language,
            gemini_validation=gemini_validation)
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


def test_maps_false_to_false(verify_client):
    client = verify_client(live={"label": "FAKE", "confidence": 0.8,
                                 "reasoning": "contradicts facts"})
    resp = _post(client, headline="Vaccines cause autism.")
    assert resp.status_code == 200
    assert resp.get_json()["final_verdict"] == "FALSE"


def test_maps_unverified_to_unverified(verify_client):
    client = verify_client(live={"label": "UNVERIFIED", "confidence": 0.4,
                                 "reasoning": "too recent"})
    resp = _post(client, headline="A brand new invention on Mars.")
    assert resp.status_code == 200
    assert resp.get_json()["final_verdict"] == "UNVERIFIED"


def test_no_live_and_no_evidence_stays_unverified(verify_client):
    client = verify_client(live=None, verdict="UNVERIFIED")
    resp = _post(client, headline="Something nobody can confirm yet.")
    body = resp.get_json()
    assert body["final_verdict"] == "UNVERIFIED"
    assert body["evidence_matrix"] == []
    assert body["sources_used"] == []


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


def test_no_invented_sources_never_fabricated(verify_client):
    """The guarantee: UNVERIFIED + empty evidence when no source exists."""
    client = verify_client(live=None, verdict="UNVERIFIED")
    resp = _post(client, headline="No known reporting about this claim.",
                 article="This is a completely invented claim text with no basis.")
    body = resp.get_json()
    assert body["final_verdict"] == "UNVERIFIED"
    assert "http" not in str(body["evidence_matrix"])
    assert body["sources_used"] in ([], None)


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


def test_english_language_mode_honoured(verify_client):
    client = verify_client(live=None, verdict="UNVERIFIED",
                           language=("english", "English"))

    import app.routes.verify as verify_mod
    captured = {}

    def _verify_text(text, headline=None, language_mode="auto", include_debug=False):
        captured["mode"] = language_mode
        return _verification("UNVERIFIED")

    verify_mod.verify_text = _verify_text
    resp = _post(client, headline="Hello world", language="english")
    assert resp.status_code == 200
    assert captured["mode"] == "english"
    assert resp.get_json()["language_mode"] == "english"


def test_full_pipeline_offline_no_key():
    """End-to-end offline: no Gemini, knowledge-base-only, still returns 200."""
    from app import create_app
    from app.config import TestConfig
    monkeypatch_os = pytest.MonkeyPatch()
    monkeypatch_os.setenv("TRUTHLENS_LIVE_EVIDENCE", "0")
    monkeypatch_os.setenv("GEMINI_API_KEY", "")
    try:
        with create_app(TestConfig).test_client() as client:
            resp = _post(client, headline="The Earth revolves around the Sun.",
                         article="Astronomers say the planet orbits the star.")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["final_verdict"] in VALID_VERDICTS
            assert body["gemini_validation"] is None
            assert isinstance(body["claims_analyzed"], int)
            assert isinstance(body["evidence_items"], int)
    finally:
        monkeypatch_os.undo()
        for name in ("TRUTHLENS_LIVE_EVIDENCE",):
            os.environ.pop(name, None)