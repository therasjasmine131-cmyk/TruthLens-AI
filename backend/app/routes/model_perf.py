"""Model performance: real, measured metrics from the trained neural network."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify

from ..config import Config
from ..ml.model_manager import model_manager

bp = Blueprint("model_perf", __name__, url_prefix="/api/model-performance")

METRICS_FILE = Config.NN_MODEL_DIR / "metrics.json"
METADATA_FILE = Config.NN_MODEL_DIR / "model_metadata.json"


@bp.get("")
def model_performance():
    """Return the measured metrics for the trained network. Empty when untrained."""
    if not METRICS_FILE.exists() or not METADATA_FILE.exists():
        return jsonify(
            {
                "trained": False,
                "message": "Train the model to view performance metrics.",
                "models": [],
                "best_model": None,
            }
        )

    metrics_doc = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    test = metrics_doc.get("test", {})
    val = metrics_doc.get("val", {})

    model = metadata.get("best_model", "Neural Network (BiGRU)")
    models = [
        {
            "name": model,
            "architecture": metadata.get("architecture"),
            "accuracy": test.get("accuracy"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "roc_auc": test.get("roc_auc"),
            "val_accuracy": val.get("accuracy"),
            "val_f1": val.get("f1"),
            "support": test.get("support"),
            "confusion_matrix": test.get("confusion_matrix"),
        }
    ]

    return jsonify(
        {
            "trained": True,
            "best_model": model,
            "best_metrics": {
                "accuracy": test.get("accuracy"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "roc_auc": test.get("roc_auc"),
            },
            "best_confusion_matrix": test.get("confusion_matrix"),
            "models": models,
            "dataset_source": metadata.get("dataset_source"),
            "train_samples": metadata.get("train_samples"),
            "test_samples": metadata.get("test_samples"),
            "n_features": metadata.get("vocab_size"),
            "forward_pass": "NumPy BiGRU (no torch/runtime deps)",
            "training_date": metadata.get("training_date"),
            "class_labels": metadata.get("class_labels"),
        }
    )