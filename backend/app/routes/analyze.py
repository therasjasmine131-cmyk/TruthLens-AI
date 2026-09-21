"""Analysis endpoints: full article + headline, or headline only."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..config import Config
from ..services.analyzer import analyze, analyze_headline_only
from ..services.live_check import live_news_check
from ..services.llm_check import cloud_ai_judge
from ..services.ollama_check import ollama_judge
from ..services.verification.scoring import _confidence_label
from .verify import _evidence_digest

bp = Blueprint("analyze", __name__, url_prefix="/api")

AI_UNAVAILABLE_MESSAGE = (
    "AI verification is temporarily unavailable: every AI provider "
    "(Gemini, Groq, OpenRouter, ChatGPT and the local model) is "
    "rate-limited, out of quota or unreachable. Please try again in a few "
    "minutes."
)


def _live_check(headline, article):
    if not Config.GEMINI_API_KEY:
        return None
    return live_news_check(headline, article)


def _cloud_verdict(body: dict, headline, article) -> dict | None:
    """Cloud AI (Groq/OpenRouter/ChatGPT) fallback verdict over the evidence."""
    cloud = cloud_ai_judge(headline, article, _evidence_digest(body.get("verification")))
    if not cloud or not cloud.get("label"):
        return None
    return {
        "verdict": cloud["label"],
        "confidence": cloud.get("confidence", 0.5),
        "reasoning": cloud.get("reasoning", ""),
        "source": cloud.get("source", "openai"),
    }


def _ollama_verdict(body: dict, headline, article) -> dict | None:
    """Local Ollama fallback verdict over the gathered evidence."""
    ollama = ollama_judge(headline, article, _evidence_digest(body.get("verification")))
    if not ollama or not ollama.get("label"):
        return None
    return {
        "verdict": ollama["label"],
        "confidence": ollama.get("confidence", 0.5),
        "reasoning": ollama.get("reasoning", ""),
        "source": "ollama",
    }


def _engine_ai_verdict(verification: dict | None) -> dict | None:
    """The FINAL engine (Gemini) verdict when it exists."""
    if not verification:
        return None
    gemini_validation = verification.get("gemini_validation")
    if gemini_validation and gemini_validation.get("label"):
        return {
            "verdict": gemini_validation["label"],
            "confidence": gemini_validation.get("confidence", 0.5),
            "reasoning": gemini_validation.get("reasoning", ""),
            "source": "gemini",
        }
    return None


def _apply_ai_final(body: dict, ai_verdict: dict | None) -> None:
    """Normalize every verdict surface to the AI verdict (live-check path)."""
    verdict = (ai_verdict or {}).get("verdict")
    if verdict not in ("REAL", "FAKE", "FALSE"):
        return
    label = "REAL" if verdict == "REAL" else "FALSE"
    body["verdict"] = label
    body["verdict_inference_only"] = False
    verification = body.get("verification")
    if not verification:
        return
    confidence = ai_verdict.get("confidence", 0.5)
    reasoning = ai_verdict.get("reasoning", "") or ""
    authority = {
        "gemini": "gemini (AI)",
        "groq": "groq (AI)",
        "openrouter": "openrouter (AI)",
        "openai": "openai (AI)",
        "ollama": "ollama (AI)",
        "evidence+ai": "evidence+ai",
    }.get(ai_verdict.get("source"), ai_verdict.get("source", "ai"))
    overall = verification.get("overall")
    if overall:
        overall["verdict"] = label
        overall["confidence"] = confidence
        overall["confidence_label"] = _confidence_label(confidence)
        overall["explanation"] = reasoning or overall.get("explanation", "")
        overall["mixed"] = False
        counts = overall.get("counts") or {}
        counts.update({
            "total_claims": len(verification.get("claims", []) or []),
            "real": len([c for c in (verification.get("claims", []) or []) if c.get("verdict") == "REAL"]),
            "false": len([c for c in (verification.get("claims", []) or []) if c.get("verdict") == "FALSE"]),
            "unverified": 0,
        })
        overall["counts"] = counts
    for claim in verification.get("claims", []) or []:
        claim["verdict"] = label
        claim["final_verdict"] = label
        claim["final_authority"] = authority
        claim["final_confidence"] = confidence
    for item in verification.get("evidence_matrix", []) or []:
        item["claim_verdict"] = label


def _resolve_ai(result: dict, headline, article) -> tuple[dict | None, dict | None]:
    """Resolve THE AI verdict, trying every provider in order.

    Order: Gemini FINAL engine -> Gemini live check -> Groq -> OpenRouter ->
    OpenAI (ChatGPT) -> local Ollama. Returns ``(ai_verdict, live_check)`` where
    ``ai_verdict`` is ``None`` when no AI provider could decide (the caller then
    returns an error).
    """
    ai_verdict = _engine_ai_verdict(result.get("verification"))
    if ai_verdict:
        return ai_verdict, None

    live_check = _live_check(headline, article)
    if live_check and live_check.get("label") in ("REAL", "FAKE"):
        return {
            "verdict": live_check["label"],
            "confidence": live_check.get("confidence", 0.5),
            "reasoning": live_check.get("reasoning", ""),
            "source": "gemini",
        }, live_check

    ai_verdict = _cloud_verdict(result, headline, article)
    if ai_verdict:
        return ai_verdict, live_check

    ai_verdict = _ollama_verdict(result, headline, article)
    return ai_verdict, live_check


@bp.post("/analyze")
def analyze_article():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    article = data.get("article")
    save = bool(data.get("save", True))
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze(headline, article, save=save, include_debug=debug)

    ai_verdict, live_check = _resolve_ai(result, headline, article)
    if ai_verdict is None:
        return jsonify({"error": AI_UNAVAILABLE_MESSAGE,
                        "status": "ai_unavailable"}), 503
    _apply_ai_final(result, ai_verdict)
    result["live_check"] = live_check
    result["ai_verdict"] = ai_verdict
    return jsonify(result)


@bp.post("/analyze/headline")
def analyze_headline():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    if not headline or not headline.strip():
        return jsonify({"error": "A headline is required.", "status": "error"}), 400
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze_headline_only(headline, include_debug=debug)

    ai_verdict, live_check = _resolve_ai(result, headline, None)
    if ai_verdict is None:
        return jsonify({"error": AI_UNAVAILABLE_MESSAGE,
                        "status": "ai_unavailable"}), 503
    _apply_ai_final(result, ai_verdict)
    result["live_check"] = live_check
    result["ai_verdict"] = ai_verdict
    return jsonify(result)
