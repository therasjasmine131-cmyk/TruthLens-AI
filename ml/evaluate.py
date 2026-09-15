"""Evaluate a trained TruthLens AI model against the held-out test set.

Loads the artifacts produced by ``ml/train.py`` and prints a detailed report
(metrics, confusion matrix, ROC-AUC, per-class breakdown). Optional
``--full-report`` writes ``ml/artifacts/evaluation_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from ml.dataset import LABEL_REAL, load_dataset, group_split
from ml.preprocess import clean_for_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def evaluate(full_report: bool = False, test_size: float = 0.2) -> dict:
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())

    df = load_dataset()
    _, _, df_test = group_split(df, val_size=0.25, test_size=test_size, seed=metadata["random_seed"])
    X_test = (df_test["headline"] + " " + df_test["text"]).map(clean_for_features)
    y_test = df_test["label"].values

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=metadata["class_labels"]).tolist()
    try:
        auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    except Exception:
        auc = None

    print(f"Model: {metadata['best_model']}")
    print(f"Dataset: {metadata['dataset_source']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    for label in metadata["class_labels"]:
        row = report[label]
        print(f"  {label:6s} precision={row['precision']:.4f} recall={row['recall']:.4f} "
              f"f1={row['f1-score']:.4f} support={row['support']}")
    print(f"ROC-AUC (REAL positive): {auc:.4f}")
    print("Confusion matrix [row=actual, col=predicted]:")
    print(cm)

    if full_report:
        out = {
            "model": metadata["best_model"],
            "accuracy": report["accuracy"],
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
        }
        (ARTIFACTS_DIR / "evaluation_report.json").write_text(json.dumps(out, indent=2))
        print(f"Wrote {ARTIFACTS_DIR / 'evaluation_report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best model")
    parser.add_argument("--full-report", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    evaluate(full_report=args.full_report, test_size=args.test_size)


if __name__ == "__main__":
    main()
