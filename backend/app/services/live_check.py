"""Gemini-based live knowledge check for news claims.

The neural network judges text *style* patterns learned from a 2016-17 US
political dataset. It cannot verify whether an event actually happened, so
recent or non-US news (for example Indian business headlines) can be
misfiled as FAKE. This service asks Gemini to verify the claim against its
knowledge of the real world.

It degrades gracefully: returns ``None`` whenever Gemini is unavailable, so
the rest of the analysis pipeline is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date

logger = logging.getLogger("truthlens.live_check")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_INPUT_CHARS = 6000
TIMEOUT_SECONDS = 25
MAX_RETRIES = 2

_SYSTEM_PROMPT = (
    "You are a careful news verification assistant. You check whether a news "
    "claim is true or false using Google Search (you have it enabled) and your "
    "own knowledge. Today is {today}.\n"
    "Rules:\n"
    "- Always search the CURRENT facts. An old article cannot reject a current "
    "claim (e.g. a 2025 article saying a person is not Chief Minister must not "
    "reject a 2026 claim that they are). Search for the current status.\n"
    "- For current office holders, elections, political claims, government "
    "schemes, weather alerts, recent company announcements and breaking news "
    "you MUST rely on fresh, live search results, not only memory.\n"
    "- Search official/government sources, press releases, reputable news and "
    "fact-check organizations; also actively search for contradicting "
    "evidence.\n"
    "- Cite only real pages returned by search; never invent URLs.\n"
    "- If you cannot verify because the event is new or has no coverage yet, "
    "say FAKE is NOT the default - search official or primary sources first.\n"
    "- Real news can come from anywhere in the world and may mention any "
    "language, currency, numbers, or company names - none of that makes a "
    "claim false. Do not treat an opinion column as false merely because it is "
    "opinionated."
)

_USER_TEMPLATE = (
    "Verify this news claim.\n\n"
    "HEADLINE:\n{headline}\n"
    "{article_part}"
    "Respond with STRICT JSON only, no markdown, exactly:\n"
    '{{"label": "REAL or FAKE or UNVERIFIED", "confidence": 0.0, "reasoning": "one or two short sentences"}}\n'
    "Rules:\n"
    "- label REAL: current web evidence (or overwhelming knowledge) supports the claim.\n"
    "- label FAKE: current web evidence contradicts established facts or it is "
    "a known disinformation claim.\n"
    "- label UNVERIFIED: only when you genuinely cannot decide after searching.\n"
    "- confidence: 0.0 to 1.0, how sure you are about your label.\n"
    "- reasoning: one or two short sentences explaining your decision."
)

_LABEL_RE = re.compile(r'"label"\s*:\s*"(REAL|FAKE|UNVERIFIED)"', re.IGNORECASE)
_CONF_RE = re.compile(r'"confidence"\s*:\s*(0?\.\d+|\d\.\d+|1|0)\b')
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)

_GROUNDING_TOOL = [{
    "google_search_retrieval": {
        "dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.5},
    },
}]


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
    prompt = _USER_TEMPLATE.format(
        headline=headline[:2000], article_part=article_part,
    )
    system = _SYSTEM_PROMPT.format(today=date.today().isoformat())
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        "tools": _GROUNDING_TOOL,
    }
    body = json.dumps(payload).encode("utf-8")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"

    headers = [{"x-goog-api-key": api_key}, {"Authorization": f"Bearer {api_key}"}]
    for attempt in range(MAX_RETRIES + 1):
        transient = False
        for auth in headers:
            for grounded in (True, False):
                try:
                    if not grounded:
                        payload.pop("tools", None)
                        body = json.dumps(payload).encode("utf-8")
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
                    if parsed is None:
                        continue
                    chunks, queries = _grounding_from_response(data)
                    parsed["source_types_checked"] = sorted(
                        {_source_type_for_url(c["url"]) for c in chunks})
                    parsed["sources"] = chunks
                    parsed["web_search_queries"] = queries
                    logger.info(
                        "[GEMINI] live-check label=%s confidence=%s grounded_sources=%d queries=%d",
                        parsed.get("label"), parsed.get("confidence"),
                        len(chunks), len(queries),
                    )
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
            if transient:
                break
        if not transient:
            return None
        time.sleep(0.5 * (2 ** attempt))
    return None


def _grounding_from_response(data: dict) -> tuple[list[dict], list[str]]:
    chunks: list[dict] = []
    queries: list[str] = []
    try:
        meta = data["candidates"][0].get("groundingMetadata") or {}
        for chunk in meta.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            uri = str(web.get("uri") or "").strip()
            if uri:
                chunks.append({"title": str(web.get("title") or "")[:200], "url": uri})
        queries = [str(q) for q in (meta.get("webSearchQueries") or [])][:8]
    except (KeyError, IndexError, TypeError):
        pass
    return chunks, queries


def _source_type_for_url(url: str) -> str:
    if re.search(r"\.gov", url, re.I):
        return "official"
    if re.search(r"\.(in|com|org|edu|net)/", url, re.I) and re.search(
        r"factcheck|fact-check|snopes|altnews|boomlive|pib", url, re.I):
        return "fact-check"
    return "news"


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
