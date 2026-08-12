"""Analysis endpoints: full article + headline, or headline only."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.analyzer import analyze, analyze_headline_only
from ..utils.errors import ServiceUnavailableError
from ..ml.model_manager import model_manager

bp = Blueprint("analyze", __name__, url_prefix="/api")


def _ensure_model():
    if not model_manager.ready:
        raise ServiceUnavailableError(
            "ML model is not available. Train the model first (python ml/train.py).",
            status_code=503,
        )


@bp.post("/analyze")
def analyze_article():
    _ensure_model()
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    article = data.get("article")
    save = bool(data.get("save", True))
    return jsonify(analyze(headline, article, save=save))


@bp.post("/analyze/headline")
def analyze_headline():
    _ensure_model()
    data = request.get_json(silent=True) or {}
    headline = data.get("headline")
    if not headline or not headline.strip():
        return jsonify({"error": "A headline is required.", "status": "error"}), 400
    return jsonify(analyze_headline_only(headline))
