"""Dataset loading for TruthLens AI.

Sources (see ``ml/data/README.md`` for license details):

* Primary: the ISOT Fake News dataset (``True.csv`` / ``Fake.csv``). Place the
  two CSV files in ``ml/data/raw/``. The files are too large to commit, so the
  loader falls back to the bundled ``sample.csv`` when they are absent.
* Bundled fallback: ``ml/data/sample.csv`` - a stratified random sample of the
  ISOT dataset committed to the repository so the project trains out-of-the-box.

Example::

    df = load_dataset()
    X_train, X_test, y_train, y_test = stratified_split(df)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
SAMPLE_CSV = DATA_DIR / "sample.csv"
DEFAULT_RAW_DIR = DATA_DIR / "raw"

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"


def _load_isot(raw_dir: Path) -> pd.DataFrame:
    true_csv = raw_dir / "True.csv"
    fake_csv = raw_dir / "Fake.csv"
    if not (true_csv.exists() and fake_csv.exists()):
        raise FileNotFoundError(f"ISOT CSVs not found in {raw_dir}")
    true_df = pd.read_csv(true_csv, encoding="utf-8")
    fake_df = pd.read_csv(fake_csv, encoding="utf-8")
    true_df = true_df.rename(columns={"title": "headline"})
    fake_df = fake_df.rename(columns={"title": "headline"})
    for df, label in ((true_df, LABEL_REAL), (fake_df, LABEL_FAKE)):
        df["label"] = label
    df = pd.concat([true_df, fake_df], ignore_index=True)
    return df[["headline", "text", "label", "subject", "date"]]


def load_dataset(raw_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the best available dataset: full ISOT first, bundled sample fallback."""
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    try:
        df = _load_isot(raw_dir)
        df.attrs["source"] = "ISOT Fake News (full)"
    except FileNotFoundError:
        if not SAMPLE_CSV.exists():
            raise FileNotFoundError(
                "No dataset found. Put True.csv and Fake.csv in ml/data/raw/ "
                "or run ml/train.py with the bundled sample."
            )
        df = pd.read_csv(SAMPLE_CSV)
        df.attrs["source"] = f"Bundled sample of ISOT ({len(df)} rows)"
    df["text"] = df["text"].fillna("").astype(str)
    df["headline"] = df["headline"].fillna("").astype(str)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].str.strip().str.upper()
    return df


def stratified_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split into train/test while preserving class proportions (NumPy only)."""
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    index = np.zeros(len(df), dtype=bool)
    for label in df["label"].unique():
        idxs = np.flatnonzero((df["label"] == label).to_numpy())
        rng.shuffle(idxs)
        n_test = int(round(len(idxs) * test_size))
        index[idxs[:n_test]] = True
    return df[~index].reset_index(drop=True), df[index].reset_index(drop=True)


_CANON_RE = None


def _canonical_key(text: str) -> str:
    """Lower-cased alphanumeric fingerprint used to group near-identical rows."""
    global _CANON_RE
    if _CANON_RE is None:
        _CANON_RE = re.compile(r"[^a-z0-9]+")
    return _CANON_RE.sub(" ", text.lower()).strip()


def _stratified_group_split(groups: pd.DataFrame, size: float, seed: int):
    """Split *groups* (already label-tagged) by class while preserving balance."""
    rng = np.random.default_rng(seed)
    groups = groups.copy().reset_index(drop=True)
    index = np.zeros(len(groups), dtype=bool)
    for label in groups["label"].unique():
        idxs = np.flatnonzero((groups["label"] == label).to_numpy())
        rng.shuffle(idxs)
        n = int(round(len(idxs) * size))
        index[idxs[:n]] = True
    return groups[~index].reset_index(drop=True), groups[index].reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    val_size: float = 0.25,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Leakage-safe train/validation/test split.

    Rows that share a canonicalised headline+text fingerprint are grouped and
    the *groups* are split (never a single group across folds). This stops
    republished/duplicated articles from leaking between training and testing.
    Preserves class proportions using the majority label of each group.
    """
    df = df.copy()
    df["_gkey"] = (df["headline"] + " " + df["text"]).map(_canonical_key)
    group_label = (
        df.groupby("_gkey")["label"].agg(lambda s: s.mode().iloc[0]).rename("label")
    )
    gdf = group_label.reset_index()

    g_train_all, g_test = _stratified_group_split(gdf, test_size, seed)
    g_train, g_val = _stratified_group_split(g_train_all, val_size / (1 - test_size), seed)

    parts = {}
    for name, g in (("train", g_train), ("val", g_val), ("test", g_test)):
        part = df[df["_gkey"].isin(g["_gkey"].values)]
        parts[name] = part.drop(columns=["_gkey"]).reset_index(drop=True)

    return parts["train"], parts["val"], parts["test"]


def build_samples(sample_size: int = 600, seed: int = 42) -> None:
    """Rebuild ``ml/data/sample.csv`` from the full ISOT CSVs (if present)."""
    df = _load_isot(DEFAULT_RAW_DIR)
    samples = []
    for label, group in df.groupby("label"):
        samples.append(group.sample(min(sample_size, len(group)), random_state=seed))
    sample = pd.concat(samples, ignore_index=True).reset_index(drop=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {len(sample)} rows to {SAMPLE_CSV}")