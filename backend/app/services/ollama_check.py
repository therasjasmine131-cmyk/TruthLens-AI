"""Local Ollama judge - free, unlimited fallback when Gemini is unavailable.

Ollama runs a local LLM (e.g. ``llama3.2``) and judges the SAME evidence the
evidence pipeline already gathered. It performs NO web search of its own: it
only reasons over the provided evidence. It degrades to ``None`` whenever
Ollama is not reachable (e.g. serverless hosts without a local model, or the
user has not installed/running it), so the caller can move to the rule-engine
last resort.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger("truthlens.ollama_check")

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SECONDS = 30
MAX_INPUT_CHARS = 7000

_SYSTEM_PROMPT = (
    "You are a news claim judge for TruthLens AI. You are given a headline, an "
    "article, and a list of evidence ALREADY gathered from the web. Judge the "
    "claims using ONLY that evidence - you have no web access and must not rely "
    "on your prior knowledge or training cutoff.\n"
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


def _system() -> str:
    return _SYSTEM_PROMPT


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
    url = f"{os.environ.get('OLLAMA_URL', DEFAULT_URL)}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 (loopback only)
            data = json.loads(resp.read().decode("utf-8"))
        content = data.get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, dict):
            return json.dumps(content)
        return None
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, OSError, ValueError) as exc:
        logger.info("[OLLAMA] unavailable (%s: %s)", type(exc).__name__, exc)
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


def ollama_judge(headline: str | None, article: str | None, evidence: str,
                 language: str = "english") -> dict | None:
    """Return an Ollama verdict dict, or None when Ollama is unavailable."""
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL).strip()
    if not model:
        return None
    user = _user(headline or "", article or "", evidence, language)
    reply = _call(model, _system(), user)
    parsed = _parse(reply) if reply else None
    if not parsed:
        return None
    logger.info(
        "[OLLAMA] verdict=%s confidence=%s model=%s",
        parsed["label"], parsed["confidence"], model,
    )
    return {"source": "ollama", "model": model, **parsed}