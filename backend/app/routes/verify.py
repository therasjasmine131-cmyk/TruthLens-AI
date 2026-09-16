"""Verify endpoint: evidence-led multi-stage fact verification.

Returns a ``final_verdict`` of TRUE / FALSE / UNVERIFIED, computed the same way
as the in-page analysis but exposed as a first-class API:

* Language mode: auto / english / tamil / tanglish.
* Gemini knowledge check (when GEMINI_API_KEY is set) feeds the evidence grid.
* The BiGRU network is only a secondary stylistic signal - the final verdict is
  evidence-led and defaults to UNVERIFIED when no evidence backs a claim
  (never invents sources).
"""

from __future__ import annotations

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

bp = Blueprint("verify", __name__, url_prefix="/api/verify")

_VERDICT_MAP = {
    REAL_LABEL: "TRUE",
    FALSE_LABEL: "FALSE",
    "FAKE": "FALSE",  # live_check (Gemini) uses FAKE; evidence scoring uses FALSE
    UNVERIFIED_LABEL: "UNVERIFIED",
}

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

    verdict = UNVERIFIED_LABEL
    if live and live.get("label"):
        verdict = live["label"]
    elif verification and verification.get("overall"):
        verdict = verification["overall"].get("verdict") or UNVERIFIED_LABEL

    return jsonify(
        {
            "status": "completed",
            "final_verdict": _VERDICT_MAP.get(verdict, "UNVERIFIED"),
            "confidence": (
                (live.get("confidence") if live and live.get("label") else None)
                or (
                    verification.get("overall", {}).get("confidence")
                    if verification and verification.get("overall")
                    else None
                )
                or 0.0
            ),
            "reasoning": (
                (live.get("reasoning") if live and live.get("label") else None)
                or (
                    verification.get("overall", {}).get("explanation")
                    if verification and verification.get("overall")
                    else ""
                )
                or ""
            ),
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
            "verification": verification,
            "stages": (verification or {}).get("stages") if include_debug else None,
            "notes": {
                "verdict_basis": (
                    "Final verdict is evidence-led. The biometric-style signal "
                    "from the local network never fabricates sources; when "
                    "nothing credible confirms or contradicts a claim, the "
                    "verdict stays UNVERIFIED."
                )
            },
        }
    )