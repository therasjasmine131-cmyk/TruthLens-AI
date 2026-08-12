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
from pathlib import Path

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
    """Split into train/test while preserving class proportions."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )


def build_samples(sample_size: int = 600, seed: int = 42) -> None:
    """Rebuild ``ml/data/sample.csv`` from the full ISOT CSVs (if present)."""
    df = _load_isot(DEFAULT_RAW_DIR)
    samples = []
    for label, group in df.groupby("label"):
        samples.append(group.sample(min(sample_size, len(group)), random_state=seed))
    sample = pd.concat(samples, ignore_index=True).reset_index(drop=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {len(sample)} rows to {SAMPLE_CSV}")
