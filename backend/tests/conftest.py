"""Shared fixtures / path setup for backend tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app import create_app  # noqa: E402
from app.config import TestConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _no_cloud_keys(monkeypatch):
    """Keep tests offline: never let a real .env key trigger live AI calls."""
    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "BAZAARLINK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Deterministic, network-free evidence (knowledge base only) for unit tests.
    monkeypatch.setenv("TRUTHLENS_LIVE_EVIDENCE", "0")


@pytest.fixture()
def app():
    return create_app(TestConfig)


@pytest.fixture()
def client(app):
    return app.test_client()
