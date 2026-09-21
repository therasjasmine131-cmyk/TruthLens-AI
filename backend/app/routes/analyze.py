"""Analysis endpoints: full article + headline, or headline only."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..config import Config
from ..services.analyzer import analyze, analyze_headline_only
from ..services.live_check import live_news_check

bp = Blueprint("analyze", __name__, url_prefix="/api")


def _live_check(headline, article):
    if not Config.GEMINI_API_KEY:
        return None
    return live_news_check(headline, article)


def _ai_verdict(live_check: dict | None, verification: dict | None) -> dict | None:
    """Build the article-level AI verdict: the FINAL engine decision comes
    first, then Gemini's live check, then the evidence-driven overall verdict.
    The AI verdict is the one shown as the final answer."""

    if verification:
        gemini_validation = verification.get("gemini_validation")
        if gemini_validation and gemini_validation.get("label"):
            return {
                "verdict": gemini_validation["label"],
                "confidence": gemini_validation.get("confidence", 0.5),
                "reasoning": gemini_validation.get("reasoning", ""),
                "source": "gemini",
            }
    if live_check and live_check.get("label"):
        return {
            "verdict": live_check["label"],
            "confidence": live_check.get("confidence", 0.5),
            "reasoning": live_check.get("reasoning", ""),
            "source": "gemini",
        }
    if verification and verification.get("overall"):
        overall = verification["overall"]
        verdict = overall.get("verdict")
        if verdict in ("REAL", "FALSE", "UNVERIFIED"):
            return {
                "verdict": "FAKE" if verdict == "FALSE" else verdict,
                "confidence": overall.get("confidence", 0.5),
                "reasoning": overall.get("explanation", ""),
                "source": "evidence+ai",
            }
    return None


@bp.post("/analyze")
def analyze_article():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    article = data.get("article")
    save = bool(data.get("save", True))
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze(headline, article, save=save, include_debug=debug)
    live_check = _live_check(headline, article)
    result["live_check"] = live_check
    result["ai_verdict"] = _ai_verdict(live_check, result.get("verification"))
    return jsonify(result)


@bp.post("/analyze/headline")
def analyze_headline():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    if not headline or not headline.strip():
        return jsonify({"error": "A headline is required.", "status": "error"}), 400
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze_headline_only(headline, include_debug=debug)
    live_check = _live_check(headline, None)
    result["live_check"] = live_check
    result["ai_verdict"] = _ai_verdict(live_check, result.get("verification"))
    return jsonify(result)
