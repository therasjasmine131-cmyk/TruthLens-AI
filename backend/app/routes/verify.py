"""Verify endpoint for the Ghost-Buster-style evidence pipeline.

``final_verdict`` is TRUE or FALSE, decided ONLY by Gemini (the FINAL engine
first, then the live check). The BiGRU neural network is a first-stage
suggestion that Gemini receives but can override. If Gemini cannot produce a
verdict this is returned as an API error - never silently converted to FAKE
and never downgraded to UNVERIFIED.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from ..services.live_check import live_news_check
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
    "FAKE": "FALSE",  # live_check (Gemini) uses FAKE; evidence scoring uses FALSE
}


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

MAX_TEXT_CHARS = 30000


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
    if verdict is None:
        # Gemini (the only decision-maker) produced no REAL/FAKE verdict.
        # Missing verification is an API error - it must not become a guess.
        logger.error(
            "[API] Gemini produced no verdict - engine_available=%s live_available=%s -> 502",
            bool(gemini_validation), bool(live),
        )
        return jsonify({
            "status": "error",
            "error": "Gemini verification is temporarily unavailable and could "
                     "not produce a REAL or FAKE verdict.",
            "gemini_validation": gemini_validation,
            "live_check": live,
            "verification": verification,
        }), 502

    if from_engine:
        basis = "final-engine"
        confidence_value = gemini_validation.get("confidence")
        reasoning_value = gemini_validation.get("reasoning")
    else:
        basis = "live-check"
        confidence_value = live.get("confidence")
        reasoning_value = live.get("reasoning")
    logger.info(
        "[API] Final verdict: %s  confidence=%.3f  basis=%s",
        _VERDICT_MAP.get(verdict, "UNKNOWN"), confidence_value or 0.0, basis,
    )

    return jsonify(
        {
            "status": "completed",
            "final_verdict": _VERDICT_MAP.get(verdict, "UNKNOWN"),
            "confidence": confidence_value or 0.0,
            "reasoning": reasoning_value or "",
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
                "verdict_basis": (
                    "Gemini decides TRUE or FALSE using live data and its own "
                    "knowledge. The local trained network's signal is only a "
                    "suggestion shown next to the verdict - it never decides."
                )
            },
        }
    )