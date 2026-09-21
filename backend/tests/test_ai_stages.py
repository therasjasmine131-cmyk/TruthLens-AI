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


def test_analyze_claim_ai_injects_full_article_context(monkeypatch):
    captured = {}
    def _fake(system, user, temperature=0.1):
        captured["user"] = user
        return {"decision": "CONTRADICT", "confidence": 0.8, "reasoning": "x"}
    monkeypatch.setattr(ai_stage, "_call_gemini", _fake)
    article = "HEADLINE: Fires near the coast\nARTICLE: A large blaze broke out overnight."
    ai_stage.analyze_claim_ai("A large blaze broke out", [_ev("CONTRADICTS")],
                              "english", article=article)
    assert "Article context" in captured["user"]
    assert "HEADLINE: Fires near the coast" in captured["user"]
    assert "Claim to verify" in captured["user"]


def test_review_claim_ai_injects_full_article_context(monkeypatch):
    captured = {}
    def _fake(system, user, temperature=0.1):
        captured["user"] = user
        return {"verdict": "CONTRADICT", "confidence": 0.8,
                "agrees_with_first": False, "problems": [], "reasoning": "x"}
    monkeypatch.setattr(ai_stage, "_call_bazaarlink", _fake)
    ai1 = _ai1("SUPPORT")
    article = "HEADLINE: Fires near the coast\nARTICLE: A large blaze broke out overnight."
    ai_stage.review_claim_ai("A large blaze broke out", [_ev("CONTRADICTS")],
                             ai1, "english", article=article)
    assert "Article context" in captured["user"]
    assert "HEADLINE: Fires near the coast" in captured["user"]
    assert "Claim under review" in captured["user"]


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


# --- FINAL verification engine (Gemini) -------------------------------------

def _overall(verdict="REAL", confidence=0.9):
    return {"verdict": verdict, "confidence": confidence,
            "explanation": "Evidence supports it."}


def _engine_inputs():
    return dict(
        headline="Government announces new tax rebate",
        article="The finance ministry announced the new rebate on Monday.",
        language="english",
        initial_verdict="FAKE",
        initial_confidence=0.83,
        initial_reasoning="Local BiGRU stylistic signal.",
        search_results=[{
            "title": "Ministry announces tax rebate",
            "url": "https://example.gov/new-rebate",
            "source": "Government Portal",
            "source_tier": "official",
            "date": "2026-09-20",
            "relation": "SUPPORTS",
            "snippet": "The finance ministry announced the rebate.",
        }],
        provisional=_overall("REAL"),
    )


def test_final_engine_builds_record(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: {
        "verdict": "REAL", "confidence": 95,
        "reasoning": "Official announcement confirms the rebate.",
        "sources_checked": [{
            "title": "Ministry announces tax rebate",
            "url": "https://example.gov/new-rebate",
            "source_type": "official",
            "published_date": "2026-09-20",
            "supports_claim": True,
        }],
        "key_claims_verified": [{
            "claim": "New tax rebate announced on Monday",
            "status": "SUPPORTED",
            "evidence_strength": "HIGH",
        }],
    })
    result = ai_stage.final_verdict_engine(**_engine_inputs())
    assert result is not None
    assert result["available"] is True
    assert result["source"] == "gemini"
    assert result["verdict"] == "REAL"
    assert result["confidence"] == 95
    assert result["initial_model_was_correct"] is False
    assert result["sources_checked"][0]["url"] == "https://example.gov/new-rebate"
    assert result["key_claims_verified"][0]["status"] == "SUPPORTED"


def test_final_engine_never_returns_unverified(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: {
        "verdict": "UNVERIFIED", "confidence": 30, "reasoning": "cannot tell",
    })
    assert ai_stage.final_verdict_engine(**_engine_inputs()) is None


def test_final_engine_rejects_invalid_verdict(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: {
        "verdict": "MAYBE", "confidence": 50, "reasoning": "x",
    })
    assert ai_stage.final_verdict_engine(**_engine_inputs()) is None


def test_final_engine_sanitizes_fabricated_sources(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: {
        "verdict": "FAKE", "confidence": 88, "reasoning": "contradicted",
        "sources_checked": [
            {"title": "Real source", "url": "https://example.gov/new-rebate",
             "supports_claim": False},
            {"title": "Invented source", "url": "https://totally-made-up.example/x",
             "supports_claim": True},
        ],
    })
    result = ai_stage.final_verdict_engine(**_engine_inputs())
    urls = [s["url"] for s in result["sources_checked"]]
    assert urls == ["https://example.gov/new-rebate"]
    assert result["verdict"] == "FALSE"


def test_final_engine_infers_correctness_when_missing(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: {
        "verdict": "REAL", "confidence": 90, "reasoning": "matches official filing",
    })
    inputs = _engine_inputs()
    inputs["initial_verdict"] = "REAL"
    result = ai_stage.final_verdict_engine(**inputs)
    assert result["initial_model_was_correct"] is True


def test_final_engine_formats_search_results_never_empty():
    text = ai_stage.format_search_results([])
    assert "no search results" in text
    text = ai_stage.format_search_results([{
        "title": "T", "url": "https://x.example/1", "source": "S",
        "source_tier": "news", "date": "2026-01-01", "relation": "SUPPORTS",
        "snippet": "hi",
    }])
    assert "https://x.example/1" in text
    assert "[SUPPORTS]" in text


def test_final_engine_unavailable_is_none(monkeypatch):
    ai_stage._CACHE.clear()
    monkeypatch.setattr(ai_stage, "_call_gemini", lambda s, u, temperature=0.1: None)
    assert ai_stage.final_verdict_engine(**_engine_inputs()) is None


# --- verifier integration (AI mocked) ---------------------------------------

def test_verify_text_records_ai_stages(monkeypatch):
    import app.services.verification.verifier as ver

    monkeypatch.setattr(ver.ai_stage, "available", lambda: True)
    monkeypatch.setattr(ver.ai_stage, "analyze_claim_ai",
                        lambda claim, evidence, lang, article=None: _ai1("SUPPORT", 0.9))
    monkeypatch.setattr(ver.ai_stage, "review_claim_ai",
                        lambda claim, evidence, ai1, lang, article=None: _ai2("SUPPORT", 0.85))

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
                        lambda claim, evidence, lang, article=None: _ai1("SUPPORT", 0.9))
    monkeypatch.setattr(ver.ai_stage, "review_claim_ai",
                        lambda claim, evidence, ai1, lang, article=None: _ai2("SUPPORT", 0.85))

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


def test_verify_text_adds_gemini_final_validation(monkeypatch):
    """With GEMINI_API_KEY set, the last pipeline stage is Gemini validation."""
    import app.services.verification.verifier as ver

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(ver.ai_stage, "available", lambda: True)
    monkeypatch.setattr(ver.ai_stage, "analyze_claim_ai",
                        lambda claim, evidence, lang, article=None: _ai1("SUPPORT", 0.9))
    monkeypatch.setattr(ver.ai_stage, "review_claim_ai",
                        lambda claim, evidence, ai1, lang, article=None: _ai2("SUPPORT", 0.85))
    monkeypatch.setattr(ver.ai_stage, "final_verdict_engine",
                        lambda headline, article, language, initial_verdict,
                               initial_confidence, initial_reasoning,
                               search_results, provisional: {
                            "available": True, "source": "gemini", "model": "m",
                            "verdict": "REAL", "confidence": 95,
                            "initial_model_verdict": "REAL",
                            "initial_model_confidence": 0.83,
                            "initial_model_was_correct": False,
                            "reasoning": "Independent reporting confirms the claim.",
                            "sources_checked": [],
                            "key_claims_verified": [],
                        })

    report = ver.verify_text("The Earth revolves around the Sun.", include_debug=True)
    assert report["gemini_validation"] is not None
    assert report["gemini_validation"]["label"] == "REAL"
    assert report["gemini_validation"]["confidence_score"] == 95
    assert report["pipeline"]["gemini_final_validated"] is True
    assert "GEMINI FINAL VALIDATION" in report["stages"]["PIPELINE"]


def test_verify_text_skips_gemini_validation_without_key(monkeypatch):
    import app.services.verification.verifier as ver

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(ver.ai_stage, "available", lambda: True)
    monkeypatch.setattr(ver.ai_stage, "final_verdict_engine",
                        lambda *a, **k: {"label": "REAL"})
    report = ver.verify_text("The Earth revolves around the Sun.")
    assert report["gemini_validation"] is None
    assert report["pipeline"]["gemini_final_validated"] is False