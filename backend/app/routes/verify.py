"""Verify endpoint for the Ghost-Buster-style evidence pipeline.

Decision chain (highest quality first, each step fails gracefully to the next):

1. ``Gemini`` - the FINAL engine first, then the live web check. Best quality
   (Google Search grounding when the key allows it, otherwise the free live
   web search results). The Stage-1 BiGRU signal is a suggestion only.
2. ``Groq`` / ``OpenRouter`` / ``OpenAI`` - free/cloud AI judges that see the
   SAME evidence the pipeline already gathered (no search of their own). Tried
   in that order; the first REAL/FAKE verdict wins.
3. ``Ollama`` - a local model judging the same evidence. Free and unlimited
   wherever Ollama is running.
4. If NO AI provider can decide, the endpoint returns an ``ai_unavailable``
   error - it never fabricates a verdict.

``final_verdict`` is always TRUE or FALSE.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from ..services.live_check import live_news_check
from ..services.llm_check import cloud_ai_judge
from ..services.verification.scoring import _confidence_label
from ..services.ollama_check import ollama_judge
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

AI_UNAVAILABLE_MESSAGE = (
    "AI verification is temporarily unavailable: every AI provider "
    "(Gemini, Groq, OpenRouter, ChatGPT and the local model) is "
    "rate-limited, out of quota or unreachable. Please try again in a few "
    "minutes."
)

_CLOUD_LABELS = {
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "openai": "OpenAI (ChatGPT)",
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
    """Deprecated: retained only for callers/tests that still import it.

    The API no longer fabricates a verdict when no AI is available - it returns
    an ``ai_unavailable`` error instead. This helper still produces a
    definitive low-confidence label for internal/offline use.
    """
    overall = (verification or {}).get("overall") or {}
    overall_verdict = str(overall.get("verdict") or "").upper()
    if overall_verdict in ("REAL", "FALSE", "FAKE"):
        label = "REAL" if overall_verdict == "REAL" else "FALSE"
        conf = max(0.0, min(0.55, float(overall.get("confidence") or 0.0)))
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
        # Tier 2: a cloud AI (Groq -> OpenRouter -> ChatGPT) judges the SAME
        # gathered evidence (no search of its own).
        cloud = cloud_ai_judge(
            headline, article,
            _evidence_digest(verification),
            language="english" if language == "auto" else language,
        )
        if cloud:
            verdict = cloud.get("label")
            source = cloud.get("source")
            confidence_value = cloud.get("confidence")
            reasoning_value = cloud.get("reasoning")
            cloud_label = _CLOUD_LABELS.get(source, source or "a cloud AI")
            fallback_note = (
                "Gemini was unavailable (quota/error), so "
                f"{cloud_label} judged the already-gathered evidence. "
                f"AI source: {cloud_label}."
            )
            logger.info("[API] %s fallback verdict: %s confidence=%s",
                        cloud_label, verdict, confidence_value)

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
                "Gemini and the cloud AI providers (Groq, OpenRouter, ChatGPT) "
                "were unavailable, so a local Ollama model judged the "
                "already-gathered evidence. AI source: Ollama."
            )
            logger.info("[API] Ollama fallback verdict: %s confidence=%s",
                        verdict, confidence_value)

    if verdict is None:
        # No AI provider could decide: return an error instead of inventing one.
        logger.warning("[API] No AI provider available - returning ai_unavailable")
        return jsonify({
            "status": "ai_unavailable",
            "error": AI_UNAVAILABLE_MESSAGE,
        }), 503

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
            "and the cloud AI providers were unavailable. The local trained "
            "network's signal is only a suggestion shown next to the verdict."
        ),
        "groq": (
            "Groq decided TRUE or FALSE from the gathered evidence because "
            "Gemini was unavailable. The local trained network's signal is only "
            "a suggestion shown next to the verdict."
        ),
        "openrouter": (
            "OpenRouter decided TRUE or FALSE from the gathered evidence "
            "because Gemini was unavailable. The local trained network's signal "
            "is only a suggestion shown next to the verdict."
        ),
        "openai": (
            "OpenAI (ChatGPT) decided TRUE or FALSE from the gathered evidence "
            "because Gemini was unavailable. The local trained network's signal "
            "is only a suggestion shown next to the verdict."
        ),
    }

    verification = verification or {}

    # Keep the embedded verification.overall and per-claim surfaces consistent
    # with the AI final verdict (confidence label/counts were built pre-AI).
    _final_label = _VERDICT_MAP.get(verdict, "UNKNOWN")
    _final_conf = confidence_value or 0.0
    _overall = verification.get("overall")
    if _overall:
        _overall["verdict"] = _final_label
        _overall["confidence"] = _final_conf
        _overall["confidence_label"] = _confidence_label(_final_conf)
        _overall["explanation"] = reasoning_value or _overall.get("explanation", "")
        _overall["mixed"] = False
    for _c in verification.get("claims", []) or []:
        if not from_engine:
            _c["verdict"] = _final_label
        _c["final_verdict"] = _final_label
        _c["final_confidence"] = _final_conf
        if source:
            _c["final_authority"] = _CLOUD_LABELS.get(source, source) + " (AI)"
    if _overall:
        _claims = verification.get("claims", []) or []
        _counts = _overall.get("counts") or {}
        _counts.update({
            "total_claims": len(_claims),
            "real": len([c for c in _claims if c.get("verdict") == "REAL"]),
            "false": len([c for c in _claims if c.get("verdict") == "FALSE"]),
            "unverified": 0,
        })
        _overall["counts"] = _counts
    for _item in verification.get("evidence_matrix", []) or []:
        _item["claim_verdict"] = _final_label

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