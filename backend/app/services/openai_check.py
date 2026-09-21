"""OpenAI (ChatGPT) judge - cloud AI fallback when Gemini is unavailable.

Used as the AI tier between Gemini and the local Ollama judge. Like Ollama it
judges the SAME evidence the pipeline already gathered (no web search of its
own) and returns a REAL/FAKE verdict. Degrades to ``None`` whenever the API key
is missing, the request fails, or the model refuses a REAL/FAKE answer, so the
caller can fall through to Ollama, then the rule-engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger("truthlens.openai_check")

DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 30
MAX_INPUT_CHARS = 7000

_SYSTEM_PROMPT = (
    "You are a news claim judge for TruthLens AI. You are given a headline, an "
    "article, and a list of evidence ALREADY gathered from the web. Judge the "
    "claims using ONLY that evidence - you must not rely on your prior "
    "knowledge or training cutoff.\n"
    "Rules:\n"
    "- REAL: the evidence supports the claim, or is thin but does not "
    "contradict it (absence of evidence is not proof of fake).\n"
    "- FAKE: the evidence directly contradicts the claim or shows it is "
    "fabricated.\n"
    "- Never output any third value.\n"
    "- confidence (0.0-1.0) reflects how strong the evidence is.\n"
    "- reasoning: 1-2 short sentences citing the evidence."
)

_USER_TEMPLATE = (
    "HEADLINE:\n{headline}\n\n"
    "ARTICLE:\n{article}\n\n"
    "LANGUAGE:\n{language}\n\n"
    "EVIDENCE ALREADY GATHERED (judge ONLY this):\n"
    "{evidence}\n\n"
    'Respond with STRICT JSON only: '
    '{{"label": "REAL or FAKE", "confidence": 0.0, "reasoning": "..."}}'
)

_LABEL_RE = re.compile(r'"label"\s*:\s*"(REAL|FAKE)"', re.IGNORECASE)
_CONF_RE = re.compile(r'"confidence"\s*:\s*(0?\.\d+|\d\.\d+|1|0)\b')
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _user(headline: str, article: str, evidence: str,
          language: str = "english") -> str:
    article = (article or "").strip()
    return _USER_TEMPLATE.format(
        headline=(headline or "").strip()[:2000],
        article=article[:2500],
        language=language or "english",
        evidence=(evidence or "")[:MAX_INPUT_CHARS],
    )


def _call(model: str, system: str, user: str) -> str | None:
    url = os.environ.get("OPENAI_URL", DEFAULT_URL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 https only
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) and content.strip() else None
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, OSError, ValueError, KeyError) as exc:
        logger.info("[OPENAI] unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def _parse(reply: str) -> dict | None:
    label_match = _LABEL_RE.search(reply or "")
    if not label_match:
        return None
    conf_match = _CONF_RE.search(reply)
    try:
        confidence = max(0.0, min(1.0, float(conf_match.group(1))))
    except (AttributeError, TypeError, ValueError):
        confidence = 0.5
    reason_match = _REASON_RE.search(reply)
    return {
        "label": label_match.group(1).upper(),
        "confidence": round(confidence, 3),
        "reasoning": (reason_match.group(1) if reason_match else "")[:500],
    }


def openai_judge(headline: str | None, article: str | None, evidence: str,
                 language: str = "english") -> dict | None:
    """Return an OpenAI verdict dict, or None when OpenAI is unavailable."""
    if not available():
        return None
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    user = _user(headline or "", article or "", evidence, language)
    reply = _call(model, _SYSTEM_PROMPT, user)
    parsed = _parse(reply) if reply else None
    if not parsed:
        return None
    logger.info(
        "[OPENAI] verdict=%s confidence=%s model=%s",
        parsed["label"], parsed["confidence"], model,
    )
    return {"source": "openai", "model": model, **parsed}
