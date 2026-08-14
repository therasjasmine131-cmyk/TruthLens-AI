"""NewsAPI client for the trending-news analyzer.

A thin standard-library wrapper around the NewsAPI v2 top-headlines endpoint.
The API key is read from config (``NEWSAPI_KEY``); requests fail fast with a
clear message when the key is missing or the upstream returns an error.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from ..config import Config
from ..utils.errors import ServiceUnavailableError

_TIMEOUT_SECONDS = 15
_CONTENT_TRUNC_RE = re.compile(r"\s*\[[+\d]+\s+chars\]\s*$", re.IGNORECASE)
_DEFAULT_COUNTRY = "us"


def _clean(value: str | None, limit: int = 500) -> str:
    value = (value or "").strip()
    value = _CONTENT_TRUNC_RE.sub("", value)
    return value[:limit]


def fetch_top_headlines(country: str = _DEFAULT_COUNTRY, page_size: int = 12) -> list[dict]:
    """Return the top headlines for ``country`` as cleaned article dicts."""
    api_key = Config.NEWSAPI_KEY
    if not api_key:
        raise ServiceUnavailableError(
            "NEWSAPI_KEY is not configured. Set it in your .env to enable trending news.",
            status_code=503,
        )

    params = {
        "country": country,
        "pageSize": str(min(max(page_size, 1), 100)),
        "apiKey": api_key,
    }
    url = f"{Config.NEWSAPI_BASE_URL}/top-headlines?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310 (https only)
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServiceUnavailableError(
            f"NewsAPI request failed (HTTP {exc.code}).", status_code=502
        ) from exc
    except Exception as exc:  # noqa: BLE001 - network timeouts etc.
        raise ServiceUnavailableError(
            "Could not reach NewsAPI; check your connection and try again.", status_code=502
        ) from exc

    if data.get("status") != "ok":
        message = (data.get("message") or "unknown NewsAPI error").strip()
        raise ServiceUnavailableError(f"NewsAPI error: {message}", status_code=502)

    articles = []
    for i, a in enumerate(data.get("articles", [])):
        if not (a.get("title") or a.get("description")):
            continue
        articles.append(
            {
                "id": i + 1,
                "source": (a.get("source") or {}).get("name") or "Unknown",
                "author": _clean(a.get("author"), limit=120),
                "headline": _clean(a.get("title"), limit=Config.MAX_HEADLINE_LENGTH),
                "description": _clean(a.get("description"), limit=600),
                "article": _clean(a.get("content"), limit=Config.MAX_ARTICLE_LENGTH),
                "url": a.get("url") or "",
                "image_url": a.get("urlToImage") or "",
                "published_at": a.get("publishedAt") or "",
            }
        )
    return articles
