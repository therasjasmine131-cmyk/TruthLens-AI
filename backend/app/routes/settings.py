"""Settings endpoints: read and persist application settings."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import settings_service

bp = Blueprint("settings", __name__, url_prefix="/api/settings")


@bp.get("")
def get_settings():
    return jsonify(settings_service.get_all())


@bp.put("")
def update_settings():
    data = request.get_json(silent=True) or {}
    allowed = {"theme", "max_article_length", "max_headline_length", "confidence_levels"}
    payload = {k: v for k, v in data.items() if k in allowed}
    return jsonify(settings_service.set_many(payload))
