"""Tests for the free, keyless web search used to ground Gemini."""

from __future__ import annotations

from app.services import web_search


_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>IMD issues orange alert for six Kerala districts - The Hindu</title>
    <link>https://news.google.com/rss/articles/ABC123</link>
    <pubDate>Sun, 21 Sep 2026 06:00:00 GMT</pubDate>
    <source url="https://www.thehindu.com">The Hindu</source>
  </item>
  <item>
    <title>Heavy rain continues in Kerala</title>
    <link>https://news.google.com/rss/articles/DEF456</link>
    <pubDate>Sun, 21 Sep 2026 05:00:00 GMT</pubDate>
    <source url="https://www.onmanorama.com">Onmanorama</source>
  </item>
</channel></rss>
"""

_DDG = """
<div class="result">
  <a class="result__a" href="https://example.gov.in/scheme">Official scheme page</a>
  <a class="result__snippet">No such free laptop scheme exists.</a>
</div>
"""


def test_parse_google_news_rss_extracts_real_items():
    items = web_search._parse_google_news_rss(_RSS, limit=5)
    assert len(items) == 2
    assert items[0]["title"].startswith("IMD issues orange alert")
    assert items[0]["url"] == "https://news.google.com/rss/articles/ABC123"
    assert items[0]["source_name"] == "The Hindu"
    assert items[0]["source_type"] == "news"
    assert items[0]["published_date"].startswith("Sun, 21 Sep 2026")


def test_parse_google_news_rss_respects_limit():
    assert len(web_search._parse_google_news_rss(_RSS, limit=1)) == 1


def test_parse_google_news_rss_bad_xml_returns_empty():
    assert web_search._parse_google_news_rss("<not-xml", limit=5) == []


def test_parse_ddg_html_extracts_title_and_url():
    items = web_search._parse_ddg_html(_DDG, limit=5)
    assert len(items) == 1
    assert items[0]["url"] == "https://example.gov.in/scheme"
    assert items[0]["title"] == "Official scheme page"
    assert items[0]["source_type"] == "web"


def test_build_query_prefers_headline():
    assert web_search.build_query("  Big   news  ", "body text") == "Big news"
    assert web_search.build_query("", "only body") == "only body"


def test_search_web_uses_news_then_falls_back(monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return _RSS if "news.google.com" in url else _DDG

    monkeypatch.setattr(web_search, "_fetch", fake_fetch)
    items = web_search.search_web("kerala rain", limit=5)
    # Google News (2) + DuckDuckGo (1) are fetched in parallel and merged.
    assert len(items) == 3
    assert any("news.google.com" in c for c in calls)


def test_search_web_parses_bing_news_rss(monkeypatch):
    bing = """<?xml version="1.0"?>
    <rss version="2.0" xmlns:News="https://www.bing.com/news/search">
      <channel><item>
        <title>CM to launch one-gram gold ring scheme</title>
        <link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url=https%3a%2f%2ftoi.example%2fstory&amp;c=1</link>
        <description>Babies born in government hospitals get rings.</description>
        <pubDate>Fri, 18 Sep 2026 19:37:00 GMT</pubDate>
        <News:Source>Times of India</News:Source>
      </item></channel></rss>
    """
    monkeypatch.setattr(web_search, "_fetch", lambda url: bing if "bing.com/news" in url else "")
    items = web_search.search_web("gold ring scheme", limit=5)
    assert len(items) == 1
    assert items[0]["url"] == "https://toi.example/story"
    assert items[0]["source_name"] == "Times of India"


def test_search_web_falls_back_to_ddg_when_news_empty(monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return "" if "news.google.com" in url else _DDG

    monkeypatch.setattr(web_search, "_fetch", fake_fetch)
    items = web_search.search_web("laptop scheme", limit=5)
    assert len(items) == 1
    assert any("duckduckgo" in c for c in calls)


def test_search_web_empty_query_returns_empty():
    assert web_search.search_web("   ") == []


def test_live_news_retriever_resolves_import_and_returns_items(monkeypatch):
    """Regression: _live_news used a broken relative import (.web_search)
    that silently returned [] on every call, starving the evidence pipeline."""
    from app.services.verification import evidence

    fake_items = [
        {"title": "CM to launch gold ring scheme", "url": "https://example.com/1",
         "source_name": "Example", "domain": "example.com"}
    ]
    monkeypatch.setattr(web_search, "search_web", lambda q, limit=5: fake_items)
    out = evidence._live_news("gold ring scheme")
    assert len(out) == 1
    assert out[0]["retrieved_from"] == "web-search"
    assert out[0]["relation_hint"] is None
    assert out[0]["has_full_text"] is False


def test_search_web_never_raises_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(web_search, "_fetch", lambda url: None)
    assert web_search.search_web("anything") == []