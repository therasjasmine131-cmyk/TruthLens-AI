"""Analysis endpoints: full article + headline, or headline only."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..config import Config
from ..services.analyzer import analyze, analyze_headline_only
from ..services.live_check import live_news_check
from ..services.openai_check import openai_judge
from .verify import _evidence_digest

bp = Blueprint("analyze", __name__, url_prefix="/api")


def _live_check(headline, article):
    if not Config.GEMINI_API_KEY:
        return None
    return live_news_check(headline, article)


def _openai_verdict(body: dict, headline, article) -> dict | None:
    """OpenAI (ChatGPT) fallback verdict over the gathered evidence."""
    oai = openai_judge(headline, article, _evidence_digest(body.get("verification")))
    if not oai or not oai.get("label"):
        return None
    return {
        "verdict": oai["label"],
        "confidence": oai.get("confidence", 0.5),
        "reasoning": oai.get("reasoning", ""),
        "source": "openai",
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
    if verdict not in ("REAL", "FAKE"):
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
        "openai": "openai (AI)",
        "evidence+ai": "evidence+ai",
        "rule-engine": "rule-engine",
    }.get(ai_verdict.get("source"), ai_verdict.get("source", "ai"))
    overall = verification.get("overall")
    if overall:
        overall["verdict"] = label
        overall["confidence"] = confidence
        overall["explanation"] = reasoning or overall.get("explanation", "")
        overall["mixed"] = False
    for claim in verification.get("claims", []) or []:
        claim["verdict"] = label
        claim["final_verdict"] = label
        claim["final_authority"] = authority
        claim["final_confidence"] = confidence
    for item in verification.get("evidence_matrix", []) or []:
        item["claim_verdict"] = label


def _force_rule_engine(body: dict) -> dict | None:
    """Last-resort definitive verdict when no AI signal decided at all.

    Mirrors /api/verify behaviour: forced REAL/FAKE, confidence capped low,
    so the Analyze page never presents UNVERIFIED as the answer.
    """
    verification = body.get("verification") or {}
    overall = verification.get("overall") or {}
    prediction = str(body.get("prediction") or "").upper()
    overall_verdict = str(overall.get("verdict") or "").upper()
    if overall_verdict in ("REAL", "FALSE", "FAKE"):
        label = "REAL" if overall_verdict == "REAL" else "FALSE"
        conf = max(0.0, min(0.55, float(overall.get("confidence") or 0.0)))
        reasoning = overall.get("explanation") or ""
    elif prediction in ("REAL", "FALSE", "FAKE"):
        label = "FALSE" if prediction == "FAKE" else prediction
        conf = 0.35
        reasoning = (
            "No AI verdict was reachable; the neural network was the only "
            "usable signal, applied with low confidence."
        )
    else:
        label = "REAL"
        conf = 0.30
        reasoning = "No decisive signal was available; labelled with minimal confidence."
    verdict = {"verdict": label, "confidence": round(conf, 2), "source": "rule-engine"}
    _apply_ai_final(body, verdict)
    return verdict


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

    ai_verdict = _engine_ai_verdict(result.get("verification"))
    live_check = None
    if ai_verdict is None:
        # Engine gave no verdict - Gemini's live check can still decide.
        live_check = _live_check(headline, article)
        ai_verdict = _ai_verdict(live_check, result.get("verification"))
        if ai_verdict is None or ai_verdict.get("verdict") not in ("REAL", "FAKE"):
            ai_verdict = _openai_verdict(result, headline, article)
        if ai_verdict is None or ai_verdict.get("verdict") not in ("REAL", "FAKE"):
            ai_verdict = _force_rule_engine(result)
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

    ai_verdict = _engine_ai_verdict(result.get("verification"))
    live_check = None
    if ai_verdict is None:
        live_check = _live_check(headline, None)
        ai_verdict = _ai_verdict(live_check, result.get("verification"))
        if ai_verdict is None or ai_verdict.get("verdict") not in ("REAL", "FAKE"):
            ai_verdict = _openai_verdict(result, headline, None)
        if ai_verdict is None or ai_verdict.get("verdict") not in ("REAL", "FAKE"):
            ai_verdict = _force_rule_engine(result)
        _apply_ai_final(result, ai_verdict)

    result["live_check"] = live_check
    result["ai_verdict"] = ai_verdict
    return jsonify(result)
