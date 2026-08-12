"""Application configuration.

All paths resolve relative to the repository root so the backend works no
matter the working directory. Environment variables override the defaults
(see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
INSTANCE_DIR = BACKEND_DIR / "instance"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "truthlens-dev-secret-change-me")
    DEBUG = _env_bool("FLASK_DEBUG", False)
    TESTING = False

    DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{INSTANCE_DIR / 'truthlens.db'}")
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CORS_ORIGINS = [
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
        if o.strip()
    ]

    ML_ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", str(REPO_ROOT / "ml" / "artifacts")))
    DATASET_RAW_DIR = Path(os.environ.get("DATASET_RAW_DIR", str(REPO_ROOT / "ml" / "data" / "raw")))

    MAX_ARTICLE_LENGTH = int(os.environ.get("MAX_ARTICLE_LENGTH", "12000"))
    MAX_HEADLINE_LENGTH = int(os.environ.get("MAX_HEADLINE_LENGTH", "500"))

    # Confidence-level bands (fractions, applied to the model confidence).
    CONFIDENCE_LEVELS = {
        "very_high_min": float(os.environ.get("CONFIDENCE_VERY_HIGH_MIN", "0.90")),
        "high_min": float(os.environ.get("CONFIDENCE_HIGH_MIN", "0.75")),
        "moderate_min": float(os.environ.get("CONFIDENCE_MODERATE_MIN", "0.50")),
    }

    JSON_SORT_KEYS = False


class TestConfig(Config):
    TESTING = True
    DEBUG = True
    SECRET_KEY = "test-secret"
    DATABASE_URL = "sqlite:///:memory:"
    INSTANCE_PATH = str(INSTANCE_DIR)
