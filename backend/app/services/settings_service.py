"""Persisted application settings (key/value on the AppSetting table)."""

from __future__ import annotations

from typing import Any

from ..extensions import db
from ..models import AppSetting

DEFAULTS: dict[str, Any] = {
    "theme": "system",
    "max_article_length": 12000,
    "max_headline_length": 500,
    "confidence_levels": {
        "very_high_min": 0.90,
        "high_min": 0.75,
        "moderate_min": 0.50,
    },
}


def get_all() -> dict:
    stored: dict = {}
    for row in AppSetting.query.all():
        stored[row.key] = row.value
    merged = dict(DEFAULTS)
    merged.update(stored)
    return merged


def get(key: str, default: Any = None) -> Any:
    row = AppSetting.query.get(key)
    if row is not None:
        return row.value
    return DEFAULTS.get(key, default)


def set_many(values: dict) -> dict:
    for key, value in values.items():
        row = AppSetting.query.get(key)
        if row is None:
            row = AppSetting(key=key, value=value)
            db.session.add(row)
        else:
            row.value = value
    db.session.commit()
    return get_all()
