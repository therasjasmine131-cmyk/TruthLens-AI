"""Dataset explorer: statistics computed from the actual dataset files."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from flask import Blueprint, jsonify, request

from ..config import Config
from ..utils.errors import ServiceUnavailableError

bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")


@lru_cache(maxsize=1)
def _load_isot_stats() -> pd.DataFrame:
    true_csv = Config.DATASET_RAW_DIR / "True.csv"
    fake_csv = Config.DATASET_RAW_DIR / "Fake.csv"
    if true_csv.exists() and fake_csv.exists():
        true_df = pd.read_csv(true_csv)
        fake_df = pd.read_csv(fake_csv)
        true_df["label"] = "REAL"
        fake_df["label"] = "FAKE"
        return pd.concat([true_df, fake_df], ignore_index=True)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _load_sample_stats() -> pd.DataFrame:
    sample = Config.DATASET_RAW_DIR.parent / "sample.csv"
    if sample.exists():
        return pd.read_csv(sample)
    return pd.DataFrame()


def _compute_stats(df: pd.DataFrame, source: str) -> dict:
    df = df.dropna(subset=["text"])
    total = len(df)
    real = int((df["label"] == "REAL").sum()) if "label" in df else 0
    fake = int((df["label"] == "FAKE").sum()) if "label" in df else 0
    text = df["text"].fillna("").astype(str)
    missing = int(df["text"].isna().sum()) + int(df["headline"].isna().sum()) if "headline" in df else int(df["text"].isna().sum())
    duplicates = int(df.duplicated(subset=["text"]).sum())
    avg_len = round(float(text.str.len().mean()), 1) if total else 0.0

    length_bins = [0, 500, 1000, 2000, 4000, 10000, float("inf")]
    length_labels = ["<500", "500-1k", "1k-2k", "2k-4k", "4k-10k", "10k+"]
    hist = [0] * len(length_labels)
    for value in text.str.len():
        for i, upper in enumerate(length_bins[1:]):
            if value < upper:
                hist[i] += 1
                break

    return {
        "source": source,
        "total_records": total,
        "real_count": real,
        "fake_count": fake,
        "class_balance": {"real": real, "fake": fake},
        "missing_values": missing,
        "duplicate_count": duplicates,
        "average_article_length": avg_len,
        "length_histogram": [
            {"bucket": b, "count": hist[i]} for i, b in enumerate(length_labels)
        ],
    }


@bp.get("/stats")
def dataset_stats():
    raw = _load_isot_stats()
    if raw.empty:
        sample = _load_sample_stats()
        if not sample.empty:
            df = sample
            df["label"] = df["label"].astype(str).str.upper()
            return jsonify({"stats": _compute_stats(df, "bundled sample"), "using_raw": False})
        raise ServiceUnavailableError(
            "Dataset not found. Place True.csv and Fake.csv in ml/data/raw/ to load the full ISOT dataset.",
            status_code=503,
        )

    df = raw
    df["label"] = df["label"].astype(str).str.upper()
    return jsonify({"stats": _compute_stats(df, "ISOT Fake News (full)"), "using_raw": True})


@bp.get("/samples")
def dataset_samples():
    """A small, safe preview of dataset rows (no external content injection)."""
    df = _load_sample_stats()
    if df.empty:
        return jsonify({"items": []})
    limit = request.args.get("limit", 10, type=int)
    limit = max(1, min(50, limit))
    items = []
    for _, row in df.head(limit).iterrows():
        items.append(
            {
                "headline": str(row.get("headline", ""))[:160],
                "text_preview": str(row.get("text", ""))[:300],
                "label": str(row.get("label", "")).upper(),
                "subject": str(row.get("subject", "")),
            }
        )
    return jsonify({"items": items})
