"""Gemini-based live knowledge check for news claims.

The ML classifier judges text *style* patterns learned from a 2016-17 US
political dataset. It cannot verify whether an event actually happened, so
recent or non-US news (for example Indian business headlines) can be
misfiled as FAKE. This service asks Gemini to verify the claim against its
knowledge of the real world.

It degrades gracefully: returns ``None`` whenever Gemini is unavailable, so
the rest of the analysis pipeline is unaffected.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_INPUT_CHARS = 6000
TIMEOUT_SECONDS = 25
MAX_RETRIES = 2

_SYSTEM_PROMPT = (
    "You are a careful news verification assistant. You check whether a news "
    "claim is true, false, or cannot be verified, using your knowledge of "
    "real-world reporting. Be accurate and honest: if an event is too recent, "
    "too local, or too niche for you to know, answer UNVERIFIED instead of "
    "guessing. Do not treat an opinion column as false merely because it is "
    "opinionated. Real news can come from anywhere in the world and may "
    "mention any language, currency, numbers, or company names - none of that "
    "makes a claim false."
)

_USER_TEMPLATE = (
    "Verify this news claim.\n\n"
    "HEADLINE:\n{headline}\n"
    "{article_part}"
    "Respond with STRICT JSON only, no markdown, exactly:\n"
    '{{"label": "REAL or FAKE or UNVERIFIED", "confidence": 0.0, "reasoning": "one or two short sentences"}}\n'
    "Rules:\n"
    "- label REAL: the claim matches real-world reporting or is a plausible "
    "summary of a known event.\n"
    "- label FAKE: the claim contradicts established facts or comes from a "
    "known disinformation source.\n"
    "- label UNVERIFIED: you genuinely do not know - too recent, local, or "
    "niche to confirm.\n"
    "- confidence: 0.0 to 1.0, how sure you are about your label.\n"
    "- reasoning: one or two short sentences explaining your decision."
)

_LABEL_RE = re.compile(r'"label"\s*:\s*"(REAL|FAKE|UNVERIFIED)"', re.IGNORECASE)
_CONF_RE = re.compile(r'"confidence"\s*:\s*(0?\.\d+|\d\.\d+|1|0)\b')
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def live_news_check(headline: str | None, article: str | None) -> dict | None:
    """Return a Gemini verdict dict, or None on any failure/unavailability."""
    headline = (headline or "").strip()
    article = (article or "").strip()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    claim = (headline + " " + article).strip()
    if not claim:
        return None

    article_part = f"ARTICLE:\n{article}\n\n" if article else ""
    prompt = _USER_TEMPLATE.format(headline=headline[:2000], article_part=article_part)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    body = json.dumps(payload).encode("utf-8")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"

    headers = [{"x-goog-api-key": api_key}, {"Authorization": f"Bearer {api_key}"}]
    for attempt in range(MAX_RETRIES + 1):
        transient = False
        for auth in headers:
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={**auth, "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 (https only)
                    data = json.loads(resp.read().decode("utf-8"))
                reply = _text_from_response(data)
                if not reply:
                    continue
                parsed = _parse_verdict(reply)
                if parsed is not None:
                    return parsed
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403):
                    continue
                if exc.code in (429, 500, 503):
                    transient = True
                    break
                return None
            except Exception:  # noqa: BLE001 - network/timeout/parse: degrade
                return None
        if not transient:
            return None
        time.sleep(0.5 * (2 ** attempt))
    return None


def _parse_verdict(reply: str) -> dict | None:
    label_match = _LABEL_RE.search(reply)
    if not label_match:
        return None
    conf_match = _CONF_RE.search(reply)
    reason_match = _REASON_RE.search(reply)
    label = label_match.group(1).upper()
    confidence = max(0.0, min(1.0, float(conf_match.group(1)))) if conf_match else 0.5
    reasoning = reason_match.group(1) if reason_match else ""
    return {
        "available": True,
        "source": "gemini",
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning[:500],
    }


def _text_from_response(data: dict) -> str | None:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts) or None
    except (KeyError, IndexError, TypeError):
        return None
