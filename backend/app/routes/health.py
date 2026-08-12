"""Health endpoint: reports backend, database, model and vectorizer status."""

from __future__ import annotations

from flask import Blueprint, jsonify
from sqlalchemy import text

from ..extensions import db
from ..ml.model_manager import model_manager

bp = Blueprint("health", __name__, url_prefix="/api/health")


def _database_status() -> dict:
    try:
        db.session.execute(text("SELECT 1"))
        return {"status": "connected", "healthy": True}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "healthy": False, "error": str(exc)}


@bp.get("")
def health():
    model_status = model_manager.status()
    vectorizer_ready = bool(model_manager.ready)

    return jsonify(
        {
            "status": "ok",
            "backend": {"status": "healthy"},
            "database": _database_status(),
            "model": model_status,
            "vectorizer": {
                "status": "ready" if vectorizer_ready else "unavailable",
                "ready": vectorizer_ready,
            },
            "model_name": model_manager.metadata().get("best_model"),
        }
    )
