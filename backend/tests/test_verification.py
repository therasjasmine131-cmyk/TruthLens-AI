"""Tests for the evidence-based verification pipeline (REAL/FALSE/UNVERIFIED).

These tests exercise claim decomposition, knowledge-base retrieval,
opinion/prediction handling, Tamil/Tanglish detection and the overall verdict
combination - without requiring any live network sources (the knowledge base
is local and deterministic).
"""

from __future__ import annotations

import pytest

from app.services.verification.claims import decompose_claims
from app.services.verification.knowledge import lookup_knowledge
from app.services.verification.language import detect_language
from app.services.verification.verifier import verify_text


# --- decomposition -----------------------------------------------------------
def test_decompose_splits_conjoined_claims():
    claims = decompose_claims(
        "Earth revolves around the Sun and the Moon is a satellite of Earth."
    )
    texts = [c["text"] for c in claims]
    assert len(claims) >= 2
    assert any("Earth revolves around the Sun" in t for t in texts)
    assert any("Moon is a satellite of Earth" in t for t in texts)


def test_decompose_marks_opinion():
    claims = decompose_claims("I think this politician is the best.")
    assert claims and claims[0]["opinion"] is True


def test_decompose_marks_prediction():
    claims = decompose_claims("India will win the next World Cup.")
    assert claims and claims[0]["prediction"] is True


def test_decompose_drops_attribution_keeps_claim():
    # "The government announced that X" must keep the X as the verifiable claim.
    claims = decompose_claims(
        "The government announced that every college student will receive 50000."
    )
    assert claims
    assert all("announced" not in c["text"].lower() for c in claims)


# --- language ----------------------------------------------------------------
def test_detect_english():
    assert detect_language("The economy is growing steadily.")["code"] == "english"


def test_detect_tamil():
    assert detect_language("இந்தியா 28 மாநிலங்களை கொண்டுள்ளது.")["code"] == "tamil"


def test_detect_tanglish():
    assert detect_language("TN la school 15 days close ah?")["code"] == "tanglish"


# --- knowledge base ----------------------------------------------------------
def test_kb_contradicts_scam_pattern():
    hits = lookup_knowledge("every college student will receive 50000 every month")
    assert hits and hits[0]["relation"] == "CONTRADICTS"


def test_kb_supports_classic_fact():
    hits = lookup_knowledge("The Earth revolves around the Sun")
    assert hits and hits[0]["relation"] == "SUPPORTS"


def test_kb_does_not_match_negated_flip():
    # Order-sensitive matching: this paraphrases an anti-fact and must not win.
    hits = lookup_knowledge("The Sun revolves around the Earth")
    assert not hits or hits[0]["relation"] != "SUPPORTS"


# --- end-to-end verdicts (local-only) ---------------------------------------
def test_real_claim_verdict():
    report = verify_text("The Earth revolves around the Sun.")
    assert report["overall"]["verdict"] == "REAL"


def test_false_claim_verdict():
    report = verify_text("Every college student will receive 50000 every month.")
    assert report["overall"]["verdict"] == "FALSE"


def test_known_false_scam_warning_text():
    report = verify_text(
        "The government announced a new scheme giving every college student "
        "50000 rupees every month for the rest of the year."
    )
    assert report["overall"]["verdict"] == "FALSE"


def test_opinion_is_unverified():
    report = verify_text("I think this politician is the best.")
    assert report["overall"]["verdict"] == "UNVERIFIED"


def test_prediction_is_unverified():
    report = verify_text("India will win the next World Cup.")
    assert report["overall"]["verdict"] == "UNVERIFIED"


def test_mixed_content_is_marked_mixed():
    report = verify_text(
        "Earth revolves around the Sun. The government announced that every "
        "college student will receive 50000 every month."
    )
    assert report["overall"]["mixed"] is True
    verdicts = {c["verdict"] for c in report["claims"]}
    assert verdicts == {"REAL", "FALSE"}


def test_tamil_text_still_verified():
    report = verify_text("Earth revolves around the sun, indha claim correct ah illa?")
    assert report["language"]["code"] == "tanglish" or report["language"]["code"] == "tamil"
    assert report["overall"]["verdict"] in {"REAL", "UNVERIFIED"}


# --- analytic service --------------------------------------------------------
def test_run_verification_includes_ml_article(app):
    from app.services.verification_service import run_verification

    report = run_verification("India has 28 states", None)
    assert report is not None
    assert report["overall"]["verdict"] == "REAL"
    assert "ml_article" in report


def test_run_verification_none_without_text(app):
    from app.services.verification_service import run_verification

    assert run_verification(None, None) is None