"""Prediction history: list, search, filter, paginate, get one, delete, clear."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request
from sqlalchemy import desc, or_

from ..extensions import db
from ..models import Prediction, VALID_LABELS
from ..services.analyzer import build_report_data, analyze
from ..utils.errors import NotFoundError, ValidationError

bp = Blueprint("history", __name__, url_prefix="/api/history")


def _apply_filters(query, args):
    search = (args.get("search") or "").strip().lower()
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(Prediction.headline.ilike(like), Prediction.article_text.ilike(like))
        )

    label = (args.get("prediction") or "").strip().upper()
    if label:
        if label not in VALID_LABELS:
            raise ValidationError(f"Unknown prediction filter: {label}")
        query = query.filter(Prediction.prediction == label)

    min_conf = args.get("min_confidence", type=float)
    max_conf = args.get("max_confidence", type=float)
    if min_conf is not None:
        query = query.filter(Prediction.confidence >= min_conf)
    if max_conf is not None:
        query = query.filter(Prediction.confidence <= max_conf)

    date_from = args.get("date_from")
    date_to = args.get("date_to")
    if date_from:
        try:
            query = query.filter(Prediction.created_at >= datetime.fromisoformat(date_from))
        except ValueError as exc:
            raise ValidationError(f"Invalid date_from: {exc}") from exc
    if date_to:
        try:
            query = query.filter(Prediction.created_at <= datetime.fromisoformat(date_to))
        except ValueError as exc:
            raise ValidationError(f"Invalid date_to: {exc}") from exc

    return query


def _sort_column(key: str):
    mapping = {
        "created_at": Prediction.created_at,
        "headline": Prediction.headline,
        "prediction": Prediction.prediction,
        "confidence": Prediction.confidence,
        "word_count": Prediction.word_count,
        "model_name": Prediction.model_name,
    }
    return mapping.get(key, Prediction.created_at)


@bp.get("")
def list_history():
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(1, request.args.get("per_page", 12, type=int)))
    sort_by = request.args.get("sort_by", "created_at")
    order = request.args.get("order", "desc")

    query = _apply_filters(Prediction.query, request.args)
    column = _sort_column(sort_by)
    query = query.order_by(desc(column) if order == "desc" else column)

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify(
        {
            "items": [p.to_dict() for p in pagination.items],
            "total": pagination.total,
            "page": pagination.page,
            "per_page": pagination.per_page,
            "pages": pagination.pages,
            "has_prev": pagination.has_prev,
            "has_next": pagination.has_next,
        }
    )


@bp.get("/<int:history_id>")
def get_history_item(history_id: int):
    record = db.session.get(Prediction, history_id)
    if record is None:
        raise NotFoundError("Prediction not found.", status_code=404)
    return jsonify(record.to_full_dict())


@bp.delete("/<int:history_id>")
def delete_history_item(history_id: int):
    record = db.session.get(Prediction, history_id)
    if record is None:
        raise NotFoundError("Prediction not found.", status_code=404)
    db.session.delete(record)
    db.session.commit()
    return jsonify({"status": "ok", "deleted": history_id})


@bp.post("/<int:history_id>/reanalyze")
def reanalyze(history_id: int):
    record = db.session.get(Prediction, history_id)
    if record is None:
        raise NotFoundError("Prediction not found.", status_code=404)
    result = analyze(record.headline, record.article_text, save=True)
    return jsonify(result)


@bp.delete("/clear")
def clear_history():
    """Delete all history records (confirmation is handled client-side)."""
    count = db.session.query(Prediction).delete()
    db.session.commit()
    return jsonify({"status": "ok", "deleted": count})


@bp.get("/export")
def export_history():
    records = Prediction.query.order_by(desc(Prediction.created_at)).all()
    return jsonify({"items": [p.to_full_dict() for p in records]})


@bp.get("/report/<int:history_id>")
def report_data(history_id: int):
    record = db.session.get(Prediction, history_id)
    if record is None:
        raise NotFoundError("Prediction not found.", status_code=404)
    return jsonify(build_report_data(record))
