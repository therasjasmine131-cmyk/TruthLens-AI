"""Model performance: real, measured metrics from training artifacts."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify

from ..config import Config
from ..ml.model_manager import model_manager

bp = Blueprint("model_perf", __name__, url_prefix="/api/model-performance")

COMPARISON_FILE = Config.ML_ARTIFACTS_DIR / "model_comparison.json"
METADATA_FILE = Config.ML_ARTIFACTS_DIR / "model_metadata.json"


@bp.get("")
def model_performance():
    """Return the measured comparison data. Empty result when untrained."""
    if not COMPARISON_FILE.exists() or not METADATA_FILE.exists():
        return jsonify(
            {
                "trained": False,
                "message": "Train the model to view performance metrics.",
                "models": [],
                "best_model": None,
            }
        )

    comparison = json.loads(COMPARISON_FILE.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    models = []
    for name, metrics in comparison.items():
        models.append(
            {
                "name": name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "val_f1": metrics.get("val_f1"),
                "support": metrics.get("support"),
                "confusion_matrix": metrics.get("confusion_matrix"),
                "roc_curve": metrics.get("roc_curve"),
            }
        )

    best_metrics = metadata.get("metrics", {})
    return jsonify(
        {
            "trained": True,
            "best_model": metadata.get("best_model"),
            "best_metrics": {
                "accuracy": best_metrics.get("accuracy"),
                "precision": best_metrics.get("precision"),
                "recall": best_metrics.get("recall"),
                "f1": best_metrics.get("f1"),
                "roc_auc": best_metrics.get("roc_auc"),
            },
            "best_confusion_matrix": best_metrics.get("confusion_matrix"),
            "models": models,
            "dataset_source": metadata.get("dataset_source"),
            "train_samples": metadata.get("train_samples"),
            "test_samples": metadata.get("test_samples"),
            "n_features": metadata.get("n_features"),
            "training_date": metadata.get("training_date"),
            "class_labels": metadata.get("class_labels"),
        }
    )
