"""Evidence retrieval for atomic claims.

Priority sources (all configurable; every one fails gracefully):

1. Built-in knowledge base         (offline, authoritative references)
2. Free live web news              (keyless Google News RSS / DuckDuckGo)
3. Wikipedia / Tamil Wikipedia     (keyless, full-page extracts)
4. Google Fact Check Tools API     (needs FACT_CHECK_API_KEY)
5. NewsAPI                         (needs NEWSAPI_KEY; snippets only)
6. GDELT                           (keyless, free, ~15-min latency news)
7. Gemini                          (needs GEMINI_API_KEY; LLM reference)

If live sources are unavailable the function returns whatever it found; an
empty list tells the verdict engine that evidence was insufficient.
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import cache
from .knowledge import lookup_knowledge
from .sources import _domain_from_url
from .textutil import char_similarity, combined_similarity

TIMEOUT_SECONDS = 8
MAX_RETRIES = 1
MAX_RESULTS_PER_SOURCE = 4

#: Set TRUTHLENS_LIVE_EVIDENCE=0 to force offline mode (knowledge base only).
#: Handy for tests, demos and offline CI; the verdict engine treats missing
#: sources identically (absent evidence -> UNVERIFIED) either way.
LIVE_EVIDENCE_ENABLED = os.environ.get("TRUTHLENS_LIVE_EVIDENCE", "1") not in {
    "0", "false", "no", "off",
}

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWSAPI = "https://newsapi.org/v2/everything"

_GEMINI_SYSTEM = (
    "You are a careful news-verification assistant. Judge whether a claim is "
    "SUPPORTED, CONTRADICTED, or NOT ADDRESSED by publicly known facts and "
    "press reporting. Never speculate; if you do not know, answer NOT ADDRESSED."
)
_GEMINI_USER = (
    "Claim: {claim}\n"
    'Respond with STRICT JSON only: {{"relation": "SUPPORT|CONTRADICT|NOT_ADDRESSED", '
    '"confidence": 0.0, "note": "one short sentence"}}'
)


def _http_json(url: str, headers=None, data=None, timeout=TIMEOUT_SECONDS):
    """Fetch JSON with retry/backoff. Returns parsed dict or None."""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data or None, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            last_err = exc
            time.sleep(0.3 * (attempt + 1))
    return None


# ---- retriever implementations ----------------------------------------------

def _wikipedia_search(query: str, language: str = "en", limit: int = 4) -> list[dict]:
    base = "en" if language != "tamil" else "ta"
    if language == "tanglish":
        base = "en"
    url = (f"https://{base}.wikipedia.org/w/api.php?action=query&list=search"
           f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit={limit}")
    data = _http_json(url)
    items = []
    if not data:
        return items
    for hit in (data.get("query", {}).get("search") or [])[:limit]:
        title = hit.get("title", "")
        snippet = html.unescape(re.sub(r"<[^>]+>", "", hit.get("snippet", "")))
        items.append({
            "source_name": f"Wikipedia ({'Tamil' if base == 'ta' else 'English'})",
            "domain": f"{base}.wikipedia.org",
            "title": title,
            "snippet": snippet,
            "text": "",
            "url": f"https://{base}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "date": None,
            "retrieved_from": "wikipedia",
            "language": base,
            "has_full_text": True,
        })
    return items


def _wikipedia_extract(title: str, language: str = "en") -> str:
    base = "ta" if language == "tamil" else "en"
    url = (f"https://{base}.wikipedia.org/w/api.php?action=query&prop=extracts"
           f"&exintro&explaintext&titles={urllib.parse.quote(title)}&redirects=1&format=json")
    data = _http_json(url)
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {}) or {}
    for page in pages.values():
        extract = page.get("extract") or ""
        if extract:
            return extract[:2500]
    return ""


def _factcheck(query: str) -> list[dict]:
    key = os.environ.get("FACT_CHECK_API_KEY", "")
    if not key:
        return []
    url = f"{FACTCHECK_API}?query={urllib.parse.quote(query)}&key={urllib.parse.quote(key)}&pageSize=4"
    data = _http_json(url)
    items = []
    if not data:
        return items
    for claim in (data.get("claims") or [])[:4]:
        reviews = claim.get("claimReview") or []
        if not reviews:
            continue
        review = reviews[0]
        publisher = review.get("publisher", {}) or {}
        url_review = review.get("url", "")
        items.append({
            "source_name": publisher.get("name") or "Google Fact Check",
            "domain": _domain_from_url(url_review),
            "title": review.get("title") or (claim.get("text", "") or "")[:120],
            "snippet": (claim.get("text") or "")[:300],
            "text": (claim.get("text") or "")[:1500],
            "url": url_review,
            "date": review.get("reviewDate"),
            "retrieved_from": "factcheck",
            "language": "en",
            "has_full_text": True,
        })
    return items


def _newsapi(query: str) -> list[dict]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    url = (f"{NEWSAPI}?q={urllib.parse.quote(query)}&language=en"
           f"&sortBy=publishedAt&pageSize=4&apiKey={urllib.parse.quote(key)}")
    data = _http_json(url)
    items = []
    if not data or data.get("status") != "ok":
        return items
    for article in (data.get("articles") or [])[:4]:
        source = article.get("source", {}) or {}
        url_article = article.get("url", "")
        items.append({
            "source_name": source.get("name") or url_article,
            "domain": _domain_from_url(url_article),
            "title": article.get("title", ""),
            "snippet": article.get("description", ""),
            "text": article.get("description", ""),
            "url": url_article,
            "date": (article.get("publishedAt") or "")[:10] or None,
            "retrieved_from": "newsapi",
            "language": "en",
            "has_full_text": False,
        })
    return items


def _gdelt(query: str) -> list[dict]:
    url = ("https://api.gdeltproject.org/api/v2/doc/doc"
           f"?query={urllib.parse.quote(query)}&mode=artlist&maxrecords=15"
           f"&format=json&sort=hybridrel")
    data = _http_json(url)
    items = []
    if not data:
        return items
    for article in (data.get("articles") or [])[:4]:
        url_article = article.get("url", "")
        if not url_article:
            continue
        seen = article.get("seendate") or ""
        date = None
        if len(seen) >= 8:
            date = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}"
        items.append({
            "source_name": article.get("domain") or _domain_from_url(url_article),
            "domain": article.get("domain") or _domain_from_url(url_article),
            "title": article.get("title", ""),
            "snippet": article.get("title", "")[:300],
            "text": article.get("title", ""),
            "url": url_article,
            "date": date,
            "retrieved_from": "gdelt",
            "language": article.get("language", "en")[:2] or "en",
            "has_full_text": False,
        })
    return items


def _live_news(query: str) -> list[dict]:
    """Keyless live news search: Google News RSS -> DuckDuckGo fallback.

    The same free search used to ground Gemini, wired directly into the
    evidence pipeline so real reporting is shown even when Gemini is down.
    """
    from .web_search import search_web

    results = search_web(query or "", limit=5)
    items: list[dict] = []
    for r in results:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        if not title or not url or not url.startswith("http"):
            continue
        snippet = (r.get("snippet") or "").strip() or title
        items.append({
            "source_name": r.get("source_name") or r.get("domain") or _domain_from_url(url),
            "domain": r.get("domain") or _domain_from_url(url),
            "title": title[:200],
            "snippet": snippet[:300],
            "text": snippet[:800],
            "url": url,
            "date": r.get("published_date"),
            "retrieved_from": "web-search",
            "language": "en",
            "relation_hint": None,
            "has_full_text": False,
        })
    return items


def _gemini_reference(claim: str) -> list[dict]:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return []
    payload = {
        "contents": [{"parts": [{"text": _GEMINI_USER.format(claim=claim[:1200])}]}],
        "systemInstruction": {"parts": [{"text": _GEMINI_SYSTEM}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    data = _http_json(url, headers=headers, data=json.dumps(payload).encode("utf-8"), timeout=20)
    if not data:
        return []
    try:
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError, TypeError):
        return []
    relation_re = re.search(r'"relation"\s*:\s*"(SUPPORT|CONTRADICT|NOT_ADDRESSED)"', text)
    note_re = re.search(r'"note"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return [{
        "source_name": "Gemini (AI reference)",
        "domain": "generativelanguage.googleapis.com",
        "title": "AI knowledge reference",
        "snippet": note_re.group(1) if note_re else text[:200],
        "text": text,
        "url": "",
        "date": None,
        "retrieved_from": "gemini",
        "language": "en",
        "relation_hint": relation_re.group(1) if relation_re else None,
        "has_full_text": True,
    }]


# ---- orchestration -----------------------------------------------------------

def _dedup(items: list[dict], max_items: int = 8) -> list[dict]:
    out: list[dict] = []
    seen_urls: set[str] = set()
    for item in items:
        url = item.get("url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        dup = False
        for existing in out:
            if combined_similarity(item.get("title") or "", existing.get("title") or "") > 0.88:
                dup = True
                break
        if not dup:
            out.append(item)
        if len(out) >= max_items:
            break
    return out


def build_query(claim: str, language_code: str) -> str:
    from .language import build_search_query
    return build_search_query(claim, language_code)[:200]


def retrieve_evidence_claim(claim: dict, query: str, language_code: str) -> list[dict]:
    """Retrieve candidate evidence items for one atomic claim."""
    claim_text = claim.get("text", "")
    cached = cache.get("claim", "evidence", query, language_code)
    if cached is not None:
        return cached

    items: list[dict] = []

    # 1. offline knowledge base (instant, authoritative for classics)
    for entry in lookup_knowledge(claim_text):
        items.append({
            "source_name": entry["source"],
            "domain": entry["url"] and _domain_from_url(entry["url"]),
            "title": entry["statement"][:120],
            "snippet": entry["statement"],
            "text": entry["statement"],
            "pattern": entry.get("pattern", ""),
            "url": entry["url"],
            "date": entry["date"],
            "retrieved_from": "knowledge-base",
            "language": "en",
            "relation_hint": entry["relation"],
            "has_full_text": True,
        })

    # 2. live sources in parallel (each internally cached); skipped in
    # offline mode (TRUTHLENS_LIVE_EVIDENCE=0).
    def _run_retriever(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - retrieval must never crash analysis
            return []

    if LIVE_EVIDENCE_ENABLED:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            wiki_future = pool.submit(_run_retriever, lambda: _wikipedia_search(query, language_code))
            web_future = pool.submit(_run_retriever, lambda: _live_news(query or claim_text))
            factcheck_future = pool.submit(_run_retriever, lambda: _factcheck(query))
            news_future = pool.submit(_run_retriever, lambda: _newsapi(query))
            gdelt_future = pool.submit(_run_retriever, lambda: _gdelt(query))
            gemini_future = pool.submit(_run_retriever, lambda: _gemini_reference(claim_text))

            items.extend(wiki_future.result())
            items.extend(web_future.result())
            items.extend(news_future.result())
            items.extend(factcheck_future.result())
            items.extend(gdelt_future.result())
            items.extend(gemini_future.result())

        # fetch Wikipedia full text extracts (gives the relevance scorer real content)
        for item in items:
            if item.get("retrieved_from") == "wikipedia" and not item.get("text"):
                extract = _wikipedia_extract(item.get("title", ""), item.get("language", "en"))
                if extract:
                    item["text"] = extract

    items = _dedup(items, max_items=8)
    cache.set("claim", items, "evidence", query, language_code)
    return items