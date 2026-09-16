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


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Existing environment variables win over .env values. Only simple
    ``KEY=VALUE`` lines and ``#`` comments are supported; values may be
    wrapped in single or double quotes.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(REPO_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _repo_path(name: str, default: Path) -> Path:
    """Resolve a configurable path against the repo root so the app works from
    any working directory (e.g. Railway runs from ``backend/``)."""
    raw = os.environ.get(name)
    p = Path(raw) if raw else default
    return p if p.is_absolute() else (REPO_ROOT / p)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "truthlens-dev-secret-change-me")
    DEBUG = _env_bool("FLASK_DEBUG", False)
    TESTING = False

    # Vercel serverless provides no writable disk: use an in-memory DB there
    # (history lives per-instance; the analyze endpoint stays fully functional).
    _is_serverless = os.environ.get("VERCEL") == "1"
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "sqlite:///:memory:" if _is_serverless
        else f"sqlite:///{INSTANCE_DIR / 'truthlens.db'}",
    )
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CORS_ORIGINS = [
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]

    # Neural-network artifacts (weights.npz, vocab.json, config.json, ...) in
    # models/fake_news_neural_network/. The runtime is pure NumPy/Python.
    NN_MODEL_DIR = _repo_path("NN_MODEL_DIR", REPO_ROOT / "models" / "fake_news_neural_network")
    DATASET_RAW_DIR = _repo_path("DATASET_RAW_DIR", REPO_ROOT / "ml" / "data" / "raw")

    MAX_ARTICLE_LENGTH = int(os.environ.get("MAX_ARTICLE_LENGTH", "12000"))
    MAX_HEADLINE_LENGTH = int(os.environ.get("MAX_HEADLINE_LENGTH", "500"))

    # External API keys (loaded from .env; never commit the real values).
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
    NEWSAPI_BASE_URL = os.environ.get("NEWSAPI_BASE_URL", "https://newsapi.org/v2")
    FACT_CHECK_API_KEY = os.environ.get("FACT_CHECK_API_KEY", "")

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
