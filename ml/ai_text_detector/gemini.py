"""Optional Gemini-based AI-text detection backend.

Calls the Google Gemini API over HTTPS using only the standard library, so the
detector runs without ``google-generativeai`` or heavy dependencies. The key
is read from the environment (``GEMINI_API_KEY``) at call time - never
hard-coded or logged.

Two auth styles are attempted in order: the ``x-goog-api-key`` header for
Google-issued API keys (``AIza...``) and ``Authorization: Bearer`` for
OAuth-style credentials (``AQ...``).

If Gemini is unavailable the detector silently falls back to the offline
heuristic backend (see ``detector.py``).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .lm import tokenize

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_INPUT_CHARS = 2000
TIMEOUT_SECONDS = 25
MAX_RETRIES = 2

_SYSTEM_PROMPT = (
    "You are an expert AI-text detector. You evaluate whether a text was "
    "written by a large language model or by a human. Consider signals such "
    "as uniformly polished phrasing, over-used transition words, lack of "
    "specific detail, repetitive structure, and absence of idiosyncrasy. "
    "Be conservative: mark text as AI only when there is real evidence, and "
    "factor in that high-quality human writing can also be fluent."
)

# Output must be strict JSON with a single "score" field.
_USER_TEMPLATE = (
    "Classify the following text as AI-generated or human-written.\n"
    "Respond with STRICT JSON only, no markdown, exactly:\n"
    '{{"score": 0.0}}\n'
    "where score is your probability (0.0 to 1.0) that the text was generated "
    "by an AI. 1.0 = definitely AI-generated, 0.0 = definitely human.\n\n"
    "TEXT:\n{text}"
)

_SCORE_RE = re.compile(r'"score"\s*:\s*(0?\.\d+|\d\.\d+|1|0)"?\s*[,}\]]')


def _extract_score(reply: str) -> float | None:
    m = _SCORE_RE.search(reply)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


class GeminiClassifier:
    """Thin REST client for Gemini text classification."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = GEMINI_BASE_URL,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.base_url = base_url.rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def classify(self, text: str) -> float | None:
        """Return an AI-likelihood in 0..1, or None on any failure."""
        if not self.available:
            return None
        if len(tokenize(text)) < 5:
            return None
        prompt = _USER_TEMPLATE.format(text=text[:MAX_INPUT_CHARS])
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/models/{self.model}:generateContent"

        auth_styles = (self._auth_headers(), self._auth_headers(bearer=True))
        for attempt in range(MAX_RETRIES + 1):
            transient = False
            for headers in auth_styles:
                try:
                    req = urllib.request.Request(
                        url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 (https only)
                        data = json.loads(resp.read().decode("utf-8"))
                    reply = self._text_from_response(data)
                    return _extract_score(reply) if reply else None
                except urllib.error.HTTPError as exc:
                    if exc.code in (400, 401, 403):
                        # Auth/format problem: try the alternate style, then give up.
                        continue
                    if exc.code in (429, 500, 503):
                        # Transient quota/overload: back off and retry the request.
                        transient = True
                        break
                    return None
                except Exception:  # noqa: BLE001 - network/timeout/parse: degrade
                    return None
            if not transient:
                return None
            time.sleep(0.5 * (2 ** attempt))
        return None

    def _auth_headers(self, bearer: bool = False) -> dict:
        if bearer:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {"x-goog-api-key": self.api_key}

    @staticmethod
    def _text_from_response(data: dict) -> str | None:
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts) or None
        except (KeyError, IndexError, TypeError):
            return None
