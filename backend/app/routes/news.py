"""Trending-news endpoint (NewsAPI)."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.news_service import fetch_top_headlines

bp = Blueprint("news", __name__, url_prefix="/api/news")

VALID_COUNTRIES = {
    "us", "gb", "ca", "au", "de", "fr", "in", "jp", "br", "mx", "za", "ng", "ru",
}


@bp.get("/trending")
def trending():
    country = (request.args.get("country") or "us").strip().lower()
    if country not in VALID_COUNTRIES:
        return jsonify(
            {
                "error": f"Unsupported country '{country}'. Choose one of: "
                f"{', '.join(sorted(VALID_COUNTRIES))}.",
                "status": "error",
            }
        ), 400
    try:
        page_size = int(request.args.get("pageSize", "12"))
    except ValueError:
        page_size = 12
    articles = fetch_top_headlines(country=country, page_size=page_size)
    return jsonify({"country": country, "total": len(articles), "items": articles})
