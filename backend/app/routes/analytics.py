"""Analytics: aggregate statistics computed from the real prediction history."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request
from sqlalchemy import func

from ..extensions import db
from ..models import Prediction, VALID_LABELS

bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


def _bucketize(timestamp, bucket: str):
    if bucket == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    return timestamp.replace(minute=0, second=0, microsecond=0)


@bp.get("")
def analytics():
    bucket = request.args.get("bucket", "day")
    if bucket not in ("hour", "day"):
        bucket = "day"

    total = db.session.query(func.count(Prediction.id)).scalar() or 0
    label_counts = dict(
        db.session.query(Prediction.prediction, func.count(Prediction.id))
        .group_by(Prediction.prediction)
        .all()
    )
    avg_confidence = (
        db.session.query(func.avg(Prediction.confidence)).scalar() or 0.0
    )
    avg_confidence = round(float(avg_confidence), 4)

    model_counts = dict(
        db.session.query(Prediction.model_name, func.count(Prediction.id))
        .group_by(Prediction.model_name)
        .all()
    )

    # Confidence histogram with fixed bins.
    bins = [0, 0.5, 0.75, 0.9, 1.0001]
    labels_bin = ["0-49%", "50-74%", "75-89%", "90-100%"]
    confidence_hist = [0] * 4
    for (conf,) in db.session.query(Prediction.confidence).all():
        for i, upper in enumerate(bins[1:]):
            if conf < upper:
                confidence_hist[i] += 1
                break

    # Trend over time.
    days = request.args.get("days", 14, type=int)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    trend_rows = (
        db.session.query(Prediction.created_at, Prediction.prediction)
        .filter(Prediction.created_at >= since)
        .all()
    )
    buckets: dict[datetime, Counter] = defaultdict(Counter)
    for ts, label in trend_rows:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        buckets[_bucketize(ts, bucket)][label] += 1

    trend = []
    cursor = _bucketize(since, bucket)
    step = timedelta(hours=1 if bucket == "hour" else 24)
    while cursor <= datetime.now(timezone.utc):
        c = buckets.get(cursor, Counter())
        trend.append(
            {
                "bucket": cursor.isoformat(),
                "real": c.get("REAL", 0),
                "fake": c.get("FAKE", 0),
                "uncertain": c.get("UNCERTAIN", 0),
                "total": sum(c.values()),
            }
        )
        cursor += step

    return jsonify(
        {
            "total_analyses": total,
            "label_counts": {label: label_counts.get(label, 0) for label in VALID_LABELS},
            "average_confidence": avg_confidence,
            "model_counts": model_counts,
            "confidence_histogram": [
                {"bucket": b, "count": confidence_hist[i]} for i, b in enumerate(labels_bin)
            ],
            "trend": trend,
            "bucket": bucket,
            "days": days,
        }
    )
