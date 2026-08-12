"""Database models.

`Prediction` stores one completed analysis. Structured analysis data is kept in
the `analysis_metadata` JSON column; the scalar columns exist for fast
filtering/sorting in the history table. No sensitive personal information is
stored.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text

from ..extensions import db

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"
LABEL_UNCERTAIN = "UNCERTAIN"
VALID_LABELS = {LABEL_REAL, LABEL_FAKE, LABEL_UNCERTAIN}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(Integer, primary_key=True)
    headline = db.Column(String(600), default="", nullable=False)
    article_text = db.Column(Text, default="", nullable=False)
    prediction = db.Column(String(16), nullable=False, index=True)
    real_probability = db.Column(Float, nullable=False)
    fake_probability = db.Column(Float, nullable=False)
    uncertain_probability = db.Column(Float, nullable=False)
    confidence = db.Column(Float, nullable=False, index=True)
    model_name = db.Column(String(80), nullable=False, index=True)
    word_count = db.Column(Integer, default=0, nullable=False)
    character_count = db.Column(Integer, default=0, nullable=False)
    sentence_count = db.Column(Integer, default=0, nullable=False)
    top_keywords = db.Column(JSON, default=list, nullable=False)
    analysis_metadata = db.Column(JSON, default=dict, nullable=False)
    created_at = db.Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    def to_dict(self) -> dict:
        """Public, JSON-safe representation used by the REST API."""
        return {
            "id": self.id,
            "headline": self.headline,
            "prediction": self.prediction,
            "real_probability": self.real_probability,
            "fake_probability": self.fake_probability,
            "uncertain_probability": self.uncertain_probability,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "sentence_count": self.sentence_count,
            "top_keywords": self.top_keywords or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_full_dict(self) -> dict:
        data = self.to_dict()
        data["article_text"] = self.article_text
        data["analysis_metadata"] = self.analysis_metadata or {}
        return data


class AppSetting(db.Model):
    """Key/value application settings (persisted, e.g. confidence thresholds)."""

    __tablename__ = "app_settings"

    key = db.Column(String(80), primary_key=True)
    value = db.Column(JSON, nullable=False)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
