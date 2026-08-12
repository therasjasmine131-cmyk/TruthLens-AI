"""TruthLens AI - reproducible training pipeline.

Usage:
    python ml/train.py                     # uses ISOT (if raw CSVs present) else bundled sample
    python ml/train.py --test-size 0.2 --seed 42
    python ml/train.py --sample            # force the bundled sample

The pipeline performs a stratified train/validation/test split, fits TF-IDF on
the training folds only (no leakage), trains four classifiers, selects the best
one by validation F1 and saves reproducible artifacts under ``ml/artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.dataset import LABEL_FAKE, LABEL_REAL, load_dataset, stratified_split
from ml.preprocess import clean_for_stats

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

SEED = 42
CLASSES = [LABEL_FAKE, LABEL_REAL]

# Models that use the full TF-IDF vectorizer.
FULL_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, C=1.0, random_state=SEED, class_weight="balanced"
    ),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, random_state=SEED, class_weight="balanced"),
        cv=3,
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
}

# Random Forest is trained on a reduced vectorizer so the artifact stays small.
RF_MODEL = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED, class_weight="balanced"
)
RF_TFIDF_PARAMS = dict(max_features=5000, min_df=3, max_df=0.85, ngram_range=(1, 1), sublinear_tf=True)

FULL_TFIDF_PARAMS = dict(
    max_features=30000,
    min_df=3,
    max_df=0.80,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
)


def _text_column(df: pd.DataFrame) -> pd.Series:
    return (df["headline"] + " " + df["text"]).str.strip().map(clean_for_stats)


def _evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "recall": float(recall_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "f1": float(f1_score(y_true, y_pred, pos_label=LABEL_REAL)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "support": int(len(y_true)),
    }
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            proba = model.decision_function(X)
            proba = np.column_stack([-proba, proba])
        proba_real = proba[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba_real))
        fpr, tpr, _ = roc_curve(y_true, proba_real, pos_label=LABEL_REAL)
        idx = np.linspace(0, len(fpr) - 1, 80).round().astype(int)
        metrics["roc_curve"] = {
            "fpr": fpr[idx].tolist(),
            "tpr": tpr[idx].tolist(),
        }
    except Exception as exc:  # noqa: BLE001 - report but continue
        metrics["roc_auc"] = None
        metrics["roc_curve"] = None
        metrics["roc_error"] = str(exc)
    return metrics


def train(args) -> None:
    t0 = time.time()
    df = load_dataset()
    source = df.attrs.get("source", "unknown")

    df_full, df_test = stratified_split(df, test_size=args.test_size, seed=args.seed)
    df_train, df_val = stratified_split(df_full, test_size=0.25, seed=args.seed)

    X_train = _text_column(df_train)
    y_train = df_train["label"].values
    X_val = _text_column(df_val)
    y_val = df_val["label"].values
    X_test = _text_column(df_test)
    y_test = df_test["label"].values

    print(f"[dataset] {source}")
    print(f"[split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    results: dict = {}
    artifacts: dict = {}

    # ---- Full-feature models ----
    tfidf_full = TfidfVectorizer(**FULL_TFIDF_PARAMS)
    X_train_full = tfidf_full.fit_transform(X_train)

    for name, estimator in FULL_MODELS.items():
        model = Pipeline([("tfidf", tfidf_full), ("clf", estimator)])
        model.fit(X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), pos_label=LABEL_REAL)
        test_metrics = _evaluate(model, X_test, y_test)
        test_metrics["val_f1"] = float(val_f1)
        results[name] = test_metrics
        artifacts[name] = model
        print(f"[train] {name}: val_f1={val_f1:.4f} test_f1={test_metrics['f1']:.4f}")

    # ---- Random Forest (reduced vectorizer) ----
    tfidf_rf = TfidfVectorizer(**RF_TFIDF_PARAMS)
    X_train_rf = tfidf_rf.fit_transform(X_train)
    rf_model = Pipeline([("tfidf", tfidf_rf), ("clf", RF_MODEL)])
    rf_model.fit(X_train, y_train)
    val_f1 = f1_score(y_val, rf_model.predict(X_val), pos_label=LABEL_REAL)
    test_metrics = _evaluate(rf_model, X_test, y_test)
    test_metrics["val_f1"] = float(val_f1)
    results["Random Forest"] = test_metrics
    artifacts["Random Forest"] = rf_model
    print(f"[train] Random Forest: val_f1={val_f1:.4f} test_f1={test_metrics['f1']:.4f}")

    # ---- Best model selection (by validation F1) ----
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best_model = artifacts[best_name]

    # Which vectorizer does the best model use?
    if best_model.named_steps.get("tfidf").max_features == 5000:
        best_tfidf = tfidf_rf
        vectorizer_name = "TfidfVectorizer (5,000 features, unigrams)"
    else:
        best_tfidf = tfidf_full
        vectorizer_name = "TfidfVectorizer (30,000 features, 1-2 grams)"

    n_features = len(best_tfidf.vocabulary_)
    explainability = "coefficients" if best_model.named_steps["clf"].__class__.__name__ in (
        "LogisticRegression",
        "LinearSVC",
        "CalibratedClassifierCV",
        "MultinomialNB",
    ) else "feature_importance"

    metadata = {
        "best_model": best_name,
        "explainability": explainability,
        "model_class": best_model.named_steps["clf"].__class__.__name__,
        "vectorizer": vectorizer_name,
        "dataset_source": source,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "n_features": n_features,
        "class_labels": CLASSES,
        "classes_": CLASSES,
        "training_date": pd.Timestamp.now().isoformat(),
        "random_seed": args.seed,
        "metrics": results[best_name],
        "all_model_metrics": results,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib", compress=4)
    joblib.dump(best_tfidf, ARTIFACTS_DIR / "best_vectorizer.joblib", compress=4)
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
