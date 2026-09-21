"""OpenAI-compatible cloud AI judge (Groq, OpenRouter, ChatGPT).

Groq, OpenRouter and OpenAI all expose the same ``/chat/completions`` schema, so
one client handles them all. Providers are tried in order and the first
REAL/FAKE verdict wins. Every call degrades to ``None`` on a missing key, a
network/quota error, or an unparseable answer, so the caller can fall through to
the next provider, then the local Ollama judge, then the "AI unavailable" error.

Configure each provider entirely from the environment::

    GROQ_API_KEY / GROQ_URL / GROQ_MODEL
    OPENROUTER_API_KEY / OPENROUTER_URL / OPENROUTER_MODEL
    OPENAI_API_KEY / OPENAI_URL / OPENAI_MODEL

Like Ollama, these judges only see the evidence the pipeline already gathered -
they never search the web themselves.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger("truthlens.llm_check")

TIMEOUT_SECONDS = 30
MAX_INPUT_CHARS = 7000

# Groq sits behind Cloudflare and rejects requests without a browser-like UA.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

PROVIDERS = (
    {
        "name": "groq",
        "key_env": "GROQ_API_KEY",
        "url_env": "GROQ_URL",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model_env": "GROQ_MODEL",
        "model": "openai/gpt-oss-20b",
        "headers": {},
    },
    {
        "name": "openrouter",
        "key_env": "OPENROUTER_API_KEY",
        "url_env": "OPENROUTER_URL",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model_env": "OPENROUTER_MODEL",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "headers": {
            "HTTP-Referer": "https://truthlens-ai-prod.vercel.app",
            "X-Title": "TruthLens AI",
        },
    },
    {
        "name": "openai",
        "key_env": "OPENAI_API_KEY",
        "url_env": "OPENAI_URL",
        "url": "https://api.openai.com/v1/chat/completions",
        "model_env": "OPENAI_MODEL",
        "model": "gpt-4o-mini",
        "headers": {},
    },
)

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


def _provider_config(provider: dict) -> tuple[str, str, str, dict]:
    key = os.environ.get(provider["key_env"], "").strip()
    url = os.environ.get(provider["url_env"], provider["url"]).strip() or provider["url"]
    model = os.environ.get(provider["model_env"], provider["model"]).strip() or provider["model"]
    return key, url, model, provider["headers"]


def available(provider: dict) -> bool:
    return bool(os.environ.get(provider["key_env"], "").strip())


def available_providers() -> list[str]:
    return [p["name"] for p in PROVIDERS if available(p)]


def _user(headline: str, article: str, evidence: str,
          language: str = "english") -> str:
    article = (article or "").strip()
    return _USER_TEMPLATE.format(
        headline=(headline or "").strip()[:2000],
        article=article[:2500],
        language=language or "english",
        evidence=(evidence or "")[:MAX_INPUT_CHARS],
    )


def _call(provider: dict, key: str, url: str, model: str,
          headers: dict, system: str, user: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
    }
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": _USER_AGENT,
        **headers,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=request_headers, method="POST",
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
        logger.info("[%s] unavailable (%s: %s)",
                    provider["name"].upper(), type(exc).__name__, exc)
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


def _judge_with(provider: dict, headline: str | None, article: str | None,
                evidence: str, language: str = "english") -> dict | None:
    key, url, model, headers = _provider_config(provider)
    if not key:
        return None
    user = _user(headline or "", article or "", evidence, language)
    reply = _call(provider, key, url, model, headers, _SYSTEM_PROMPT, user)
    parsed = _parse(reply) if reply else None
    if not parsed:
        return None
    logger.info("[%s] verdict=%s confidence=%s model=%s",
                provider["name"].upper(), parsed["label"], parsed["confidence"], model)
    return {"source": provider["name"], "model": model, **parsed}


def cloud_ai_judge(headline: str | None, article: str | None, evidence: str,
                   language: str = "english") -> dict | None:
    """Try each configured cloud provider in order; return the first verdict.

    Returns ``{"source", "model", "label", "confidence", "reasoning"}`` or
    ``None`` when no provider is configured or none could decide.
    """
    for provider in PROVIDERS:
        if not available(provider):
            continue
        verdict = _judge_with(provider, headline, article, evidence, language)
        if verdict:
            return verdict
    return None


def openai_judge(headline: str | None, article: str | None, evidence: str,
                 language: str = "english") -> dict | None:
    """Backward-compatible single-provider (OpenAI) entry point."""
    provider = next(p for p in PROVIDERS if p["name"] == "openai")
    return _judge_with(provider, headline, article, evidence, language)
