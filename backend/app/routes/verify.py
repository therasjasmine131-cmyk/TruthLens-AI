"""Verify endpoint for the Ghost-Buster-style evidence pipeline.

Decision chain (highest quality first, each step fails gracefully to the next):

1. ``Gemini`` - the FINAL engine first, then the live web check. Best quality
   (Google Search grounding when the key allows it, otherwise the free live
   web search results). The Stage-1 BiGRU signal is a suggestion only.
2. ``Ollama`` - if Gemini is unavailable/quota-limited, a local Ollama model
   judges the SAME evidence the pipeline already gathered (no search of its
   own). Free and unlimited wherever Ollama is running.
3. ``rule-engine`` - last resort: the evidence-derived overall verdict forced
   to REAL/FAKE with confidence capped low. Never a 502 just because an AI
   provider is down.

``final_verdict`` is always TRUE or FALSE.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from ..services.live_check import live_news_check
from ..services.ollama_check import ollama_judge
from ..services.openai_check import openai_judge
from ..services.verification_service import run_verification
from ..services.verification.language import VALID_MODES
from ..services.verification.verifier import verify_text
from ..services.verification.scoring import (
    VERDICT_FALSE as FALSE_LABEL,
    VERDICT_REAL as REAL_LABEL,
    VERDICT_UNVERIFIED as UNVERIFIED_LABEL,
)

logger = logging.getLogger("truthlens.verify")

bp = Blueprint("verify", __name__, url_prefix="/api/verify")

_VERDICT_MAP = {
    REAL_LABEL: "TRUE",
    FALSE_LABEL: "FALSE",
    "FAKE": "FALSE",  # live check / Ollama use FAKE; evidence scoring uses FALSE
}

MAX_TEXT_CHARS = 30000


def _gemini_basis(gemini_validation: dict | None, live: dict | None) -> tuple[str | None, bool]:
    """Resolve THE Gemini verdict: FINAL engine first, live check second.

    Returns ``(verdict_label, from_engine)`` or ``(None, False)`` when neither
    Gemini stage produced a REAL/FAKE verdict.
    """
    if gemini_validation and gemini_validation.get("label"):
        return gemini_validation["label"], True
    if live and live.get("label") and live["label"] in ("REAL", "FAKE"):
        return live["label"], False
    return None, False


def _evidence_digest(verification: dict | None, max_chars: int = 6000) -> str:
    """Compact text digest of the evidence already gathered (for the Ollama judge)."""
    parts: list[str] = []
    for claim in (verification or {}).get("claims", []) or []:
        claim_text = str(claim.get("text") or "")[:240]
        parts.append(f"CLAIM: {claim_text or '(no text)'}")
        evidence = (claim.get("evidence") or [])[:5]
        if not evidence:
            continue
        for e in evidence:
            relation = str(e.get("relation") or e.get("relation_hint") or "NEUTRAL")
            source = str(e.get("source_name") or e.get("domain") or "unknown")
            title = str(e.get("title") or "")[:120]
            snippet = str(e.get("snippet") or e.get("text") or e.get("reason") or "")[:220]
            url = str(e.get("url") or "")
            parts.append(
                f"- [{relation}] {title or '(untitled)'} ({source})"
                f"{url and ' | ' + url or ''}\n"
                f"  snippet: {snippet}"
            )
    joined = "\n".join(parts)
    if not joined.strip():
        return "(no evidence retrieved)"
    return joined[:max_chars]


def _rule_engine_basis(verification: dict | None) -> tuple[str, float, str, str]:
    """Last-resort evidence-rule verdict, forced REAL/FAKE, confidence capped low.

    Returns ``(label, confidence_0_1, reasoning, note)``.
    """
    overall = (verification or {}).get("overall") or {}
    ml = (verification or {}).get("ml_article") or {}
    overall_verdict = str(overall.get("verdict") or "").upper()
    if overall_verdict in ("REAL", "FALSE", "FAKE"):
        label = "REAL" if overall_verdict == "REAL" else "FALSE"
        conf = max(0.0, min(0.55, float(overall.get("confidence") or 0.0)))
    else:
        prediction = str(ml.get("prediction") or "").upper()
        if prediction in ("REAL", "FALSE"):
            label = prediction
            conf = 0.35
        else:
            label = "REAL"
            conf = 0.30
    reasoning = str(overall.get("explanation") or (
        f"Evidence rules could not reach a strong verdict; "
        f"labelled {label} with low confidence."))
    note = (
        "Rule-engine last resort: Gemini, OpenAI and Ollama were unavailable, "
        "so the evidence-derived result was used with confidence capped low."
    )
    return label, round(conf, 2), reasoning, note


@bp.post("")
def verify():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    article = data.get("article") or data.get("text")
    language = str(data.get("language", "auto")).strip().lower()
    include_debug = (request.args.get("debug") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    if language not in VALID_MODES:
        return jsonify({"error": f"Invalid language mode '{language}'."}), 400

    text = " ".join(filter(None, [headline, article])).strip()
    if not text:
        return jsonify({"error": "A headline or article is required.", "status": "error"}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"Text too long (max {MAX_TEXT_CHARS} chars)."}), 400

    if language in (None, "", "auto"):
        verification = run_verification(headline, article, include_debug=include_debug)
    else:
        verification = verify_text(article, headline=headline,
                                   language_mode=language,
                                   include_debug=include_debug)

    live = live_news_check(headline, article)
    gemini_validation = (verification or {}).get("gemini_validation")

    verdict, from_engine = _gemini_basis(gemini_validation, live)
    source = None
    fallback_note = None

    if verdict is None:
        # Tier 2: OpenAI (ChatGPT) judges the SAME gathered evidence (no search).
        oai = openai_judge(
            headline, article,
            _evidence_digest(verification),
            language="english" if language == "auto" else language,
        )
        if oai:
            verdict = oai.get("label")
            source = "openai"
            confidence_value = oai.get("confidence")
            reasoning_value = oai.get("reasoning")
            fallback_note = (
                "Gemini was unavailable (quota/error), so OpenAI (ChatGPT) judged "
                "the already-gathered evidence. AI source: OpenAI."
            )
            logger.info("[API] OpenAI fallback verdict: %s confidence=%s",
                        verdict, confidence_value)

    if verdict is None:
        # Tier 3: local Ollama judges the SAME gathered evidence (no search).
        ollama = ollama_judge(
            headline, article,
            _evidence_digest(verification),
            language="english" if language == "auto" else language,
        )
        if ollama:
            verdict = ollama.get("label")
            source = "ollama"
            confidence_value = ollama.get("confidence")
            reasoning_value = ollama.get("reasoning")
            fallback_note = (
                "Gemini and OpenAI were unavailable, so a local Ollama model "
                "judged the already-gathered evidence. AI source: Ollama."
            )
            logger.info("[API] Ollama fallback verdict: %s confidence=%s",
                        verdict, confidence_value)

    if verdict is None:
        # Tier 4: rule engine, forced REAL/FAKE, confidence capped low.
        verdict, confidence_value, reasoning_value, fallback_note = \
            _rule_engine_basis(verification)
        source = "rule-engine"
        logger.warning("[API] Rule-engine fallback verdict: %s confidence=%s",
                       verdict, confidence_value)

    if source is None:
        if from_engine:
            source = "final-engine"
            confidence_value = gemini_validation.get("confidence")
            reasoning_value = gemini_validation.get("reasoning")
        else:
            source = "live-check"
            confidence_value = live.get("confidence")
            reasoning_value = live.get("reasoning")

    logger.info(
        "[API] Final verdict: %s  confidence=%.3f  source=%s",
        _VERDICT_MAP.get(verdict, "UNKNOWN"), confidence_value or 0.0, source,
    )

    basis_texts = {
        "final-engine": (
            "Gemini decided TRUE or FALSE using live web evidence and its own "
            "reasoning. The local trained network's signal is only a suggestion "
            "shown next to the verdict - it never decides."
        ),
        "live-check": (
            "Gemini's live news check decided TRUE or FALSE. The local trained "
            "network's signal is only a suggestion shown next to the verdict."
        ),
        "ollama": (
            "A local Ollama model judged the gathered evidence because Gemini "
            "and OpenAI were unavailable. The local trained network's signal is "
            "only a suggestion shown next to the verdict."
        ),
        "openai": (
            "OpenAI (ChatGPT) decided TRUE or FALSE from the gathered evidence "
            "because Gemini was unavailable. The local trained network's signal "
            "is only a suggestion shown next to the verdict."
        ),
        "rule-engine": (
            "Rule-engine result used because Gemini, OpenAI and Ollama were all "
            "unavailable; confidence is capped low."
        ),
    }

    return jsonify(
        {
            "status": "completed",
            "final_verdict": _VERDICT_MAP.get(verdict, "UNKNOWN"),
            "confidence": confidence_value or 0.0,
            "reasoning": reasoning_value or "",
            "verdict_source": source,
            "fallback_note": fallback_note,
            "language_mode": language,
            "language_detected": (verification or {}).get("language", {}).get("label"),
            "claims_analyzed": len((verification or {}).get("claims", [])),
            "evidence_items": sum(
                len(c.get("evidence", []))
                for c in (verification or {}).get("claims", [])
            ),
            "sources_used": (verification or {}).get("pipeline", {}).get("sources_used"),
            "evidence_matrix": (verification or {}).get("evidence_matrix", []),
            "claims": (verification or {}).get("claims", []),
            "live_check": live,
            "gemini_validation": gemini_validation,
            "verification": verification,
            "stages": (verification or {}).get("stages") if include_debug else None,
            "notes": {
                "verdict_basis": basis_texts.get(source, basis_texts["final-engine"])
            },
        }
    )