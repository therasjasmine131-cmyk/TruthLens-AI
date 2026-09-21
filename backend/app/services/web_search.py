"""Free, keyless live web search used to ground Gemini.

The Gemini Google Search grounding tool requires a billed project, so this
module fetches REAL current web results without any API key:

* Google News RSS (``news.google.com/rss/search``) for news-shaped queries.
* DuckDuckGo HTML (``html.duckduckgo.com/html/``) as a general fallback.

Only stdlib is used. Every failure degrades to an empty list so callers keep
working offline.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 8
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def build_query(headline: str, article: str = "", max_len: int = 160) -> str:
    """A short, search-engine friendly query from the headline/article."""
    text = (headline or "").strip() or (article or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_len].strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    except ValueError:
        return ""


def _parse_google_news_rss(xml_text: str, limit: int) -> list[dict]:
    """Parse a Google News RSS feed into real result dicts."""
    results: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("google news rss parse failed: %s", exc)
        return results
    for item in root.iter("item"):
        title = _clean(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        source_el = item.find("source")
        source_name = _clean(source_el.text) if source_el is not None else ""
        pub = _clean(item.findtext("pubDate") or "")
        results.append({
            "title": title[:200],
            "url": link,
            "snippet": title[:240],
            "source_name": source_name or _domain(link),
            "domain": _domain(link),
            "published_date": pub or None,
            "source_type": "news",
            "type": "news",
            "relation": "NEUTRAL",
        })
        if len(results) >= limit:
            break
    return results


def _parse_ddg_html(page: str, limit: int) -> list[dict]:
    """Parse DuckDuckGo HTML results into real result dicts."""
    results: list[dict] = []
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    for match in pattern.finditer(page):
        url = html.unescape(match.group("url"))
        if url.startswith("//duckduckgo.com/l/"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            url = (qs.get("uddg") or [url])[0]
        title = _clean(re.sub(r"<[^>]+>", "", match.group("title")))
        snippet = _clean(re.sub(r"<[^>]+>", "", match.group("snippet")))
        if not title or not url.startswith("http"):
            continue
        results.append({
            "title": title[:200],
            "url": url,
            "snippet": snippet[:240],
            "source_name": _domain(url),
            "domain": _domain(url),
            "published_date": None,
            "source_type": "web",
            "type": "web",
            "relation": "NEUTRAL",
        })
        if len(results) >= limit:
            break
    return results


def _fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310 (https only)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.debug("web search fetch failed for %s: %s", url, exc)
        return None


def search_web(query: str, limit: int = 6) -> list[dict]:
    """Return real current web results for ``query`` (never raises)."""
    query = (query or "").strip()
    if not query:
        return []
    encoded = urllib.parse.quote_plus(query)
    news_url = (
        "https://news.google.com/rss/search?q="
        f"{encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    page = _fetch(news_url)
    if page:
        results = _parse_google_news_rss(page, limit)
        if results:
            logger.info("[SEARCH] google-news results=%d q=%s", len(results), query[:80])
            return results
    ddg_url = f"https://html.duckduckgo.com/html/?q={encoded}"
    page = _fetch(ddg_url)
    if page:
        results = _parse_ddg_html(page, limit)
        if results:
            logger.info("[SEARCH] duckduckgo results=%d q=%s", len(results), query[:80])
            return results
    logger.info("[SEARCH] no results q=%s", query[:80])
    return []