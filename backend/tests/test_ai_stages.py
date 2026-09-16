"""Tests for the multi-stage AI pipeline (analysis #1, reviewer #2, decision engine).

AI stages are mocked - no network calls. The decision engine is tested with
realistic evidence fixtures covering the priority order:

    strong independent evidence > AI agreement > ML (lowest authority).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRUTHLENS_LIVE_EVIDENCE", "0")

import pytest  # noqa: E402

from app.services.verification import ai_stage  # noqa: E402
from app.services.verification.scoring import final_decision_for_claim  # noqa: E402


def _claim(text: str = "The Earth revolves around the Sun") -> dict:
    return {"text": text, "opinion": False, "prediction": False, "type": "general-factual"}


def _ev(relation: str, significance: float = 0.6, domain: str = "a.org",
        tier: str = "primary", score: float = 0.8) -> dict:
    return {
        "source_name": domain.rsplit(".", 1)[0].title(),
        "domain": domain,
        "title": "Evidence title",
        "snippet": "Evidence snippet text",
        "text": "Evidence snippet text",
        "url": f"https://{domain}/x",
        "date": None,
        "retrieved_from": "knowledge-base",
        "language": "en",
        "relation": relation,
        "relevance_score": 0.85,
        "relation_score": 0.9,
        "reasons": [],
        "numerical": [],
        "source_score": score,
        "source_tier": tier,
        "source_reasons": [],
        "temporal": {"status": "fresh"},
        "significance": significance,
    }


def _cross(domains: list[str]) -> dict:
    return {
        "distinct_support_domains": 0,
        "distinct_contradict_domains": 0,
        "independent_support": len(domains) >= 2,
        "independent_contradiction": False,
    }


def _ai1(decision: str, confidence: float = 0.9) -> dict:
    return {"available": True, "source": "gemini", "model": "m",
            "decision": decision, "confidence": confidence, "reasoning": "reason"}


def _ai2(verdict: str, confidence: float = 0.85, agrees: bool = True) -> dict:
    return {"available": True, "source": "gemini", "model": "m",
            "verdict": verdict, "confidence": confidence,
            "agrees_with_first": agrees, "problems": [], "reasoning": "reason"}


# --- decision engine ---------------------------------------------------------

def test_strong_independent_evidence_beats_ml_and_conflicting_ai():
    evs = [_ev("SUPPORTS", domain="a.org"), _ev("SUPPORTS", domain="b.org")]
    final = final_decision_for_claim(
        _claim(), {"prediction": "FALSE", "confidence": 0.92}, evs,
        _cross(["a.org", "b.org"]),
        _ai1("CONTRADICT"), _ai2("CONTRADICT", agrees=False),
    )
    assert final["verdict"] == "REAL"
    assert final["authority"] == "evidence"
    assert final["conflicts"]
    assert final["independent_sources"] is True


def test_ml_real_ignored_when_strong_evidence_contradicts():
    evs = [_ev("CONTRADICTS"), _ev("CONTRADICTS", domain="b.org")]
    final = final_decision_for_claim(
        _claim(), {"prediction": "REAL", "confidence": 0.9}, evs,
        _cross(["a.org", "b.org"]), None, None,
    )
    assert final["verdict"] == "FALSE"
    assert final["authority"] == "evidence"


def test_evidence_and_ai_agree():
    evs = [_ev("SUPPORTS"), _ev("SUPPORTS", domain="b.org")]
    final = final_decision_for_claim(
        _claim(), None, evs, _cross(["a.org", "b.org"]),
        _ai1("SUPPORT"), _ai2("SUPPORT"),
    )
    assert final["verdict"] == "REAL"
    assert final["authority"] == "evidence+ai"
    assert final["confidence"] > 0.8


def test_ai_decides_when_no_evidence_and_reviewer_agrees():
    final = final_decision_for_claim(
        _claim(), None, [], _cross([]), _ai1("SUPPORT", 0.9), _ai2("SUPPORT", 0.85),
    )
    assert final["verdict"] == "REAL"
    assert final["authority"] == "ai"


def test_ai_disagreement_leaves_claim_unverified():
    final = final_decision_for_claim(
        _claim(), None, [], _cross([]), _ai1("SUPPORT"), _ai2("CONTRADICT", agrees=False),
    )
    assert final["verdict"] == "UNVERIFIED"
    assert final["conflicts"]


def test_ai_contradicts_drives_false_when_ai_agree():
    final = final_decision_for_claim(
        _claim(), None, [], _cross([]), _ai1("CONTRADICT", 0.9), _ai2("CONTRADICT", 0.8),
    )
    assert final["verdict"] == "FALSE"
    assert final["authority"] == "ai"


def test_weak_evidence_conflicting_with_ai_stays_unverified():
    evs = [_ev("SUPPORTS", significance=0.32)]
    final = final_decision_for_claim(
        _claim(), None, evs, _cross(["a.org"]), _ai1("CONTRADICT", 0.9), _ai2("CONTRADICT"),
    )
    assert final["verdict"] == "UNVERIFIED"
    assert final["authority"] == "none"
    assert final["conflicts"]


def test_no_ai_offline_behavior_is_preserved():
    evs = [_ev("SUPPORTS", significance=0.6)]
    final = final_decision_for_claim(_claim(), None, evs, _cross(["a.org"]), None, None)
    assert final["verdict"] == "REAL"
    assert final["authority"] == "evidence"


def test_ml_alone_never_decides():
    final = final_decision_for_claim(
        _claim(), {"prediction": "REAL", "confidence": 0.99}, [], _cross([]), None, None,
    )
    assert final["verdict"] == "UNVERIFIED"


def test_no_evidence_no_ai_is_unverified():
    final = final_decision_for_claim(_claim(), None, [], _cross([]), None, None)
    assert final["verdict"] == "UNVERIFIED"


# --- ai_stage parsing / record building ------------------------------------

def test_parse_strict_json_object():
    text = '{"decision": "SUPPORT", "confidence": 0.9, "reasoning": "yes"}'
    obj = ai_stage._parse_json_object(text)
    assert obj == {"decision": "SUPPORT", "confidence": 0.9, "reasoning": "yes"}


def test_parse_json_regex_fallback():
    text = ('Sure! Here is the answer: {"decision":"CONTRADICT",'
            '"confidence":0.7,"reasoning":"evidence says otherwise"}')
    obj = ai_stage._parse_json_object(text)
    assert obj["decision"] == "CONTRADICT"
    assert obj["confidence"] == pytest.approx(0.7)


def test_analyze_claim_ai_builds_record(monkeypatch):
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda system, user, temperature=0.1: {
        "decision": "support", "confidence": 0.87, "reasoning": "Evidence backs it.",
    })
    result = ai_stage.analyze_claim_ai("some claim", [_ev("SUPPORTS")], "english")
    assert result["available"] is True
    assert result["decision"] == "SUPPORT"
    assert result["confidence"] == pytest.approx(0.87, abs=0.001)
    assert result["source"] == "gemini"


def test_analyze_claim_ai_invalid_decision_is_none(monkeypatch):
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda system, user, temperature=0.1: {
        "decision": "MAYBE", "confidence": 0.99, "reasoning": "x",
    })
    assert ai_stage.analyze_claim_ai("claim", [], "english") is None


def test_analyze_claim_ai_accepts_plural_verdicts(monkeypatch):
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda system, user, temperature=0.1: {
        "decision": "SUPPORTS", "confidence": 0.9, "reasoning": "Evidence supports it.",
    })
    result = ai_stage.analyze_claim_ai("claim", [_ev("SUPPORTS")], "english")
    assert result is not None
    assert result["decision"] == "SUPPORT"


def test_review_claim_ai_builds_record(monkeypatch):
    monkeypatch.setattr(ai_stage, "_call_bazaarlink", lambda system, user, temperature=0.1: {
        "verdict": "support", "confidence": 0.8, "agrees_with_first": True,
        "problems": ["Evidence is old"], "reasoning": "ok",
    })
    ai1 = _ai1("SUPPORT")
    result = ai_stage.review_claim_ai("claim", [_ev("SUPPORTS")], ai1, "english")
    assert result["available"] is True
    assert result["verdict"] == "SUPPORT"
    assert result["source"] == "bazaarlink"
    assert result["agrees_with_first"] is True
    assert result["problems"] == ["Evidence is old"]


def test_build_evidence_context_includes_relations():
    ctx = ai_stage.build_evidence_context(
        [_ev("SUPPORTS", domain="a.org"), _ev("CONTRADICTS", domain="b.org")]
    )
    assert "[SUPPORTS]" in ctx
    assert "[CONTRADICTS]" in ctx
    assert "a.org" in ctx


# --- verifier integration (AI mocked) ---------------------------------------

def test_verify_text_records_ai_stages(monkeypatch):
    import app.services.verification.verifier as ver

    monkeypatch.setattr(ver.ai_stage, "available", lambda: True)
    monkeypatch.setattr(ver.ai_stage, "analyze_claim_ai",
                        lambda claim, evidence, lang: _ai1("SUPPORT", 0.9))
    monkeypatch.setattr(ver.ai_stage, "review_claim_ai",
                        lambda claim, evidence, ai1, lang: _ai2("SUPPORT", 0.85))

    report = ver.verify_text("The Earth revolves around the Sun.")
    claim = report["claims"][0]
    assert claim["ai_analysis_1"]["decision"] == "SUPPORT"
    assert claim["ai_review"]["verdict"] == "SUPPORT"
    assert claim["final_verdict"] == "REAL"
    assert claim["final_authority"] in {"evidence+ai", "evidence", "ai"}
    assert claim["ml_prediction"] is None or claim["ml_prediction"] in {"REAL", "FALSE"}
    assert report["pipeline"]["ai_used"] is True
    assert report["stages"] is None


def test_verify_text_debug_stages_filled(monkeypatch):
    import app.services.verification.verifier as ver

    monkeypatch.setattr(ver.ai_stage, "available", lambda: True)
    monkeypatch.setattr(ver.ai_stage, "analyze_claim_ai",
                        lambda claim, evidence, lang: _ai1("SUPPORT", 0.9))
    monkeypatch.setattr(ver.ai_stage, "review_claim_ai",
                        lambda claim, evidence, ai1, lang: _ai2("SUPPORT", 0.85))

    report = ver.verify_text("The Earth revolves around the Sun.", include_debug=True)
    assert report["stages"] is not None
    stages = report["claims"][0]["stages"]
    assert set(stages) == {"NN_RESULT", "EVIDENCE_RESULT", "AI_RESULT_1",
                           "AI_REVIEW_RESULT", "FINAL_RESULT"}
    assert stages["FINAL_RESULT"]["verdict"] == "REAL"


def test_verify_text_degrades_without_ai(monkeypatch):
    import app.services.verification.verifier as ver

    monkeypatch.setattr(ver.ai_stage, "available", lambda: False)
    report = ver.verify_text("The Earth revolves around the Sun.")
    assert report["pipeline"]["ai_used"] is False
    assert report["overall"]["verdict"] == "REAL"
    assert all(c["ai_analysis_1"] is None for c in report["claims"])