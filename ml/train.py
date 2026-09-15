"""TruthLens AI - reproducible training pipeline.

Usage:
    python ml/train.py                     # uses ISOT (if raw CSVs present) else bundled sample
    python ml/train.py --test-size 0.2 --seed 42
    python ml/train.py --sample            # force the bundled sample

The pipeline performs a leakage-safe train/validation/test split (grouping
canonical near-duplicates so republished articles never appear in two folds),
fits the TF-IDF vectorisers on the training folds only, trains five models
including a word+character n-gram ensemble and selects the best one by
validation F1. The winning pipeline is saved under ``ml/artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ml.dataset import LABEL_FAKE, LABEL_REAL, load_dataset, group_split
from ml.preprocess import clean_for_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent / "data"

SEED = 42
CLASSES = [LABEL_FAKE, LABEL_REAL]

FULL_TFIDF_PARAMS = dict(
    max_features=30000,
    min_df=3,
    max_df=0.80,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
)

WORD_CHAR_TFIDF_W = dict(
    max_features=20000,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
)
WORD_CHAR_TFIDF_C = dict(
    max_features=20000,
    min_df=3,
    max_df=0.90,
    ngram_range=(2, 5),
    analyzer="char_wb",
    sublinear_tf=True,
    strip_accents="unicode",
)

RF_TFIDF_PARAMS = dict(max_features=5000, min_df=3, max_df=0.85, ngram_range=(1, 1), sublinear_tf=True)


def _text_column(df: pd.DataFrame) -> pd.Series:
    return (df["headline"] + " " + df["text"]).str.strip().map(clean_for_features)


def _calibrated_linear_svc() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, random_state=SEED, class_weight="balanced"), cv=3
    )


def _evaluate(pipeline: Pipeline, X, y_true) -> dict:
    y_pred = pipeline.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "recall": float(recall_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "f1": float(f1_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "support": int(len(y_true)),
    }
    try:
        proba = pipeline.predict_proba(X)
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1], pos_label=LABEL_REAL)
        idx = np.linspace(0, len(fpr) - 1, 80).round().astype(int)
        metrics["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    except Exception as exc:  # noqa: BLE001 - report but continue
        metrics["roc_auc"] = None
        metrics["roc_curve"] = None
        metrics["roc_error"] = str(exc)
    return metrics


def _leakage_report(X_train, X_val, X_test) -> None:
    """Cross-fold similarity check: report max cosine to nearest train row."""
    from sklearn.preprocessing import normalize as _normalize

    corpus = list(X_train) + list(X_val) + list(X_test)
    vec = TfidfVectorizer(max_features=20000, min_df=1, ngram_range=(1, 1), sublinear_tf=True)
    mat = vec.fit_transform(corpus)
    mat = _normalize(mat, axis=1)
    n_train = len(X_train)
    train_block = mat[:n_train]
    other = mat[n_train:]
    sims = (other @ train_block.T).toarray()
    if sims.shape[0] == 0:
        print("[leak-check] no validation/test rows, skipping")
        return
    max_sim = float(sims.max())
    ratio_near = float((sims.max(axis=1) > 0.95).mean())
    print(f"[leak-check] max train/other cosine={max_sim:.3f} near-dup(>0.95)={ratio_near:.3f}")


def train(args) -> None:
    t0 = time.time()
    df = load_dataset()
    df = df.drop_duplicates(subset=["headline", "text"])
    source = df.attrs.get("source", "unknown")

    df_train, df_val, df_test = group_split(df, val_size=0.25, test_size=args.test_size, seed=args.seed)

    X_train = _text_column(df_train).tolist()
    y_train = df_train["label"].values
    X_val = _text_column(df_val).tolist()
    y_val = df_val["label"].values
    X_test = _text_column(df_test).tolist()
    y_test = df_test["label"].values

    print(f"[dataset] {source}")
    print(f"[split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    results: dict = {}
    artifacts: dict = {}
    vectorizers: dict = {}

    def fit_pipeline(name, pipe, Xtr, Xv, ytr, yv, vec) -> None:
        pipe.fit(Xtr, ytr)
        val_f1 = float(f1_score(yv, pipe.predict(Xv), pos_label=LABEL_REAL))
        test_metrics = _evaluate(pipe, X_test, y_test)
        test_metrics["val_f1"] = val_f1
        results[name] = test_metrics
        artifacts[name] = pipe
        vectorizers[name] = vec
        print(f"[train] {name}: val_f1={val_f1:.4f} test_f1={test_metrics['f1']:.4f}")

    # ---- 1/2) word TF-IDF + logistic regression / calibrated SVM ----
    tfidf_full = TfidfVectorizer(**FULL_TFIDF_PARAMS)
    Xtr_full = tfidf_full.fit_transform(X_train)
    fit_pipeline(
        "Logistic Regression (word 1-2g)",
        Pipeline([("tfidf", tfidf_full), ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight="balanced"))]),
        X_train, X_val, y_train, y_val, tfidf_full,
    )
    fit_pipeline(
        "Linear SVM (word 1-2g)",
        Pipeline([("tfidf", tfidf_full), ("clf", _calibrated_linear_svc())]),
        X_train, X_val, y_train, y_val, tfidf_full,
    )
    fit_pipeline(
        "Naive Bayes (word 1-2g)",
        Pipeline([("tfidf", tfidf_full), ("clf", MultinomialNB(alpha=1.0))]),
        X_train, X_val, y_train, y_val, tfidf_full,
    )

    # ---- 3) word + character n-gram ensemble + logistic regression ----
    features = FeatureUnion([
        ("word", TfidfVectorizer(**WORD_CHAR_TFIDF_W)),
        ("char", TfidfVectorizer(**WORD_CHAR_TFIDF_C)),
    ])
    fit_pipeline(
        "LogReg word+char (ensemble)",
        Pipeline([("features", features), ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight="balanced"))]),
        X_train, X_val, y_train, y_val, features,
    )

    # ---- 4) Random Forest on a reduced word vectorizer ----
    tfidf_rf = TfidfVectorizer(**RF_TFIDF_PARAMS)
    fit_pipeline(
        "Random Forest (word 1g)",
        Pipeline([("tfidf", tfidf_rf), ("clf", RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED, class_weight="balanced"))]),
        X_train, X_val, y_train, y_val, tfidf_rf,
    )

    _leakage_report(X_train, X_val, X_test)

    # ---- Best model selection (by validation F1) ----
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best_model = artifacts[best_name]
    best_vec = vectorizers[best_name]
    clf = best_model.named_steps["clf"]

    n_features = len(best_vec.vocabulary_) if hasattr(best_vec, "vocabulary_") else best_vec.transform([""]).shape[1]
    explainability = "coefficients" if clf.__class__.__name__ in (
        "LogisticRegression",
        "LinearSVC",
        "CalibratedClassifierCV",
        "MultinomialNB",
    ) else "feature_importance"

    metadata = {
        "best_model": best_name,
        "explainability": explainability,
        "model_class": clf.__class__.__name__,
        "vectorizer": best_vec.__class__.__name__,
        "dataset_source": source,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "n_features": int(n_features),
        "class_labels": CLASSES,
        "classes_": CLASSES,
        "training_date": pd.Timestamp.now().isoformat(),
        "random_seed": args.seed,
        "split": "grouped-canonical (leakage-safe)",
        "metrics": results[best_name],
        "all_model_metrics": results,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib", compress=4)
    joblib.dump(best_vec, ARTIFACTS_DIR / "best_vectorizer.joblib", compress=4)
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with open(ARTIFACTS_DIR / "model_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[best] {best_name} (val_f1={results[best_name]['val_f1']:.4f})")
    print(f"[features] {n_features}")
    print(f"[artifacts] {ARTIFACTS_DIR}")
    print(f"[done] {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TruthLens AI classifiers")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()