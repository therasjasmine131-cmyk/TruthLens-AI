"""Sample articles for the "Try Demo" flow.

These are real rows drawn from the training dataset and are clearly labelled as
educational dataset samples - never presented as current or genuine news.
"""

from __future__ import annotations

import random

import pandas as pd
from flask import Blueprint, jsonify

from ..config import Config
from ..utils.errors import ServiceUnavailableError

bp = Blueprint("samples", __name__, url_prefix="/api/samples")

DEMO_COUNT = 6


def _load_samples() -> pd.DataFrame:
    sample = Config.DATASET_RAW_DIR.parent / "sample.csv"
    if sample.exists():
        return pd.read_csv(sample)
    return pd.DataFrame()


@bp.get("")
def samples():
    df = _load_samples()
    if df.empty:
        raise ServiceUnavailableError("No sample data available.", status_code=503)

    rng = random.Random(42)
    picked = []
    for label in ("REAL", "FAKE"):
        group = df[df["label"].astype(str).str.upper() == label]
        if group.empty:
            continue
        for _, row in group.sample(min(DEMO_COUNT // 2, len(group)), random_state=42).iterrows():
            picked.append(
                {
                    "id": len(picked) + 1,
                    "label": "REAL" if label == "REAL" else "FAKE",
                    "headline": str(row.get("headline", "")),
                    "article": str(row.get("text", ""))[:3000],
                    "kind": "dataset sample",
                    "disclaimer": "Educational demonstration only - dataset sample, not a real news source.",
                }
            )
    return jsonify({"items": picked})
