"""Batch analysis: process many rows in chunks, never crashing on a bad row."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..ml.model_manager import model_manager
from ..services.analyzer import analyze
from ..utils.errors import ServiceUnavailableError
from ..utils.validators import validate_csv_row

bp = Blueprint("batch", __name__, url_prefix="/api/batch")


@bp.post("/analyze")
def batch_analyze():
    """Expects {"rows": [{"headline":..., "article":...}, ...]}.

    Each row is analyzed independently; malformed rows are reported, not fatal.
    Results are NOT persisted to history (batch runs are treated as bulk work).
    """
    if not model_manager.ready:
        raise ServiceUnavailableError(
            "ML model is not available. Train the model first.", status_code=503
        )
    data = request.get_json(silent=True) or {}
    rows = data.get("rows")
    if not isinstance(rows, list):
        return jsonify({"error": "Expected a JSON array under 'rows'.", "status": "error"}), 400
    if not rows:
        return jsonify({"error": "No rows to analyze.", "status": "error"}), 400
    if len(rows) > 2000:
        return jsonify({"error": "Batch too large (max 2000 rows).", "status": "error"}), 400

    results = []
    errors = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            results.append({"index": index, "error": "row is not an object"})
            errors += 1
            continue
        problem = validate_csv_row(row)
        if problem:
            results.append({"index": index, "error": problem})
            errors += 1
            continue
        try:
            out = analyze(row.get("headline"), row.get("article"), save=False)
            results.append(
                {
                    "index": index,
                    "headline": (row.get("headline") or "").strip()[:200],
                    "prediction": out["prediction"],
                    "confidence": out["confidence"],
                    "probabilities": out["probabilities"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"index": index, "error": f"analysis failed: {exc}"})
            errors += 1

    real = sum(1 for r in results if r.get("prediction") == "REAL")
    fake = sum(1 for r in results if r.get("prediction") == "FAKE")
    uncertain = sum(1 for r in results if r.get("prediction") == "UNCERTAIN")
    return jsonify(
        {
            "total": len(rows),
            "completed": len(results) - errors,
            "errors": errors,
            "real": real,
            "fake": fake,
            "uncertain": uncertain,
            "results": results,
        }
    )
