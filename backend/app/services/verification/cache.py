"""Small TTL cache for search results, URL content and completed verifications.

Reduces external API usage (NewsAPI / Gemini / Google Fact Check have rate
limits) and speeds up repeated claims. Persists to JSON files under
``<repo>/.cache/`` so restarts do not clear the warm cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

REPO_CACHE_DIR = Path(__file__).resolve().parents[4] / ".cache"

_lock = threading.Lock()
_cache: dict[str, dict] = {}

_default_ttl = {
    "search": 3600 * 6,          # 6 hours
    "url": 3600 * 24,            # 24 hours
    "claim": 3600 * 12,          # 12 hours
    "wikipedia": 3600 * 24 * 7,  # 1 week
    "factcheck": 3600 * 6,       # 6 hours
    "newsapi": 3600,             # 1 hour
    "gemini": 3600 * 6,          # 6 hours
}


def _key(kind: str, *parts: str) -> str:
    payload = "|".join([str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{kind}:{digest}"


def _load() -> None:
    if _cache or not REPO_CACHE_DIR.exists():
        return
    try:
        for blob in REPO_CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(blob.read_text(encoding="utf-8"))
                for key, value in data.items():
                    if value.get("expires", 0) > time.time():
                        _cache[key] = value
            except (OSError, ValueError, KeyError):
                continue
    except OSError:
        pass


def _flush() -> None:
    try:
        REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            bucket: dict[str, dict] = {}
            for key, value in _cache.items():
                bucket[key] = value
        out = {k: v for k, v in bucket.items() if v.get("expires", 0) > time.time()}
        (REPO_CACHE_DIR / "cache.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def get(kind: str, *parts: str):
    _load()
    key = _key(kind, *parts)
    with _lock:
        entry = _cache.get(key)
    if entry and entry.get("expires", 0) > time.time():
        return entry.get("value")
    return None


def set(kind: str, value, *parts: str, ttl: int | None = None) -> None:
    _load()
    key = _key(kind, *parts)
    expires = time.time() + (ttl if ttl is not None else _default_ttl.get(kind, 3600))
    with _lock:
        _cache[key] = {"expires": expires, "value": value}
    if len(_cache) % 25 == 0:
        _flush()


def clear() -> None:
    with _lock:
        _cache.clear()
    try:
        if REPO_CACHE_DIR.exists():
            for blob in REPO_CACHE_DIR.glob("*.json"):
                blob.unlink()
    except OSError:
        pass