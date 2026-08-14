"""Tests for the /api/news/trending endpoint."""

from __future__ import annotations

from app.routes import news as news_module
from app.utils.errors import ServiceUnavailableError

SAMPLE_ARTICLES = [
    {
        "id": 1,
        "source": "Test Wire",
        "author": "Jane Doe",
        "headline": "City approves new transit plan",
        "description": "The council voted late Tuesday.",
        "article": "The council voted late Tuesday to approve the plan. [+120 chars]",
        "url": "https://example.com/story/1",
        "image_url": "",
        "published_at": "2026-08-14T10:00:00Z",
    }
]


def test_trending_returns_articles(client, monkeypatch):
    monkeypatch.setattr(news_module, "fetch_top_headlines", lambda country="us", page_size=12: SAMPLE_ARTICLES)
    resp = client.get("/api/news/trending?country=gb")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["country"] == "gb"
    assert body["total"] == 1
    assert body["items"][0]["headline"] == "City approves new transit plan"


def test_trending_rejects_bad_country(client):
    resp = client.get("/api/news/trending?country=zz")
    assert resp.status_code == 400


def test_trending_handles_unconfigured_key(client, monkeypatch):
    def raise_unconfigured(country="us", page_size=12):
        raise ServiceUnavailableError(
            "NEWSAPI_KEY is not configured. Set it in your .env to enable trending news.",
            status_code=503,
        )

    monkeypatch.setattr(news_module, "fetch_top_headlines", raise_unconfigured)
    resp = client.get("/api/news/trending")
    assert resp.status_code == 503
    assert "NEWSAPI_KEY" in resp.get_json()["error"]
