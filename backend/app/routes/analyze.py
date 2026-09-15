"""Analysis endpoints: full article + headline, or headline only."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..config import Config
from ..services.analyzer import analyze, analyze_headline_only
from ..services.live_check import live_news_check
from ..utils.errors import ServiceUnavailableError
from ..ml.model_manager import model_manager

bp = Blueprint("analyze", __name__, url_prefix="/api")


def _ensure_model():
    if not model_manager.ready:
        raise ServiceUnavailableError(
            "ML model is not available. Train the model first (python ml/train.py).",
            status_code=503,
        )


def _live_check(headline, article):
    if not Config.GEMINI_API_KEY:
        return None
    return live_news_check(headline, article)


@bp.post("/analyze")
def analyze_article():
    _ensure_model()
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    article = data.get("article")
    save = bool(data.get("save", True))
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze(headline, article, save=save, include_debug=debug)
    result["live_check"] = _live_check(headline, article)
    return jsonify(result)


@bp.post("/analyze/headline")
def analyze_headline():
    _ensure_model()
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    if not headline or not headline.strip():
        return jsonify({"error": "A headline is required.", "status": "error"}), 400
    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes", "on"}
    result = analyze_headline_only(headline, include_debug=debug)
    result["live_check"] = _live_check(headline, None)
    return jsonify(result)
