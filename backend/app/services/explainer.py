"""Explainable AI (neural network edition).

The BiGRU classifier has no linear coefficients. To explain a single prediction
we use the model's own internals: every token changes the hidden state of the
forward GRU (and of the backward GRU) by some amount, and tokens that move the
state the most are the ones the network leaned on. Words are ranked by that
hidden-state change, weighted so long, low-information words do not dominate.

The result describes *what the model looked at*, not evidence that the news is
true or false.
"""

from __future__ import annotations

from ..ml.model_manager import model_manager

POSITIVE = "positive"
NEGATIVE = "negative"
LOW = "low"


def explain_prediction(text: str, prediction: str, top_n: int = 8) -> dict:
    """Return the tokens that most influenced the NN prediction."""
    if not model_manager.ready:
        return {"method": None, "features": [], "note": "No model available."}

    try:
        raw = model_manager.top_keywords(text, top_n=top_n)
    except Exception:  # noqa: BLE001 - explanation is best-effort
        raw = []

    features = []
    for rank, item in enumerate(raw):
        features.append(
            {
                "term": item["term"],
                "weight": round(item["score"], 6),
                "contribution": round(item["score"], 6),
                "influence": POSITIVE if rank < 3 else LOW,
            }
        )

    return {
        "method": "nn-token-influence",
        "model_class": "BiGRU (Embedding->BiGRU->Dense)",
        "features": features,
        "note": (
            "Model-associated features: words ranked by how much they moved the "
            "neural network's hidden state during the forward/backward pass. This "
            "shows which tokens the network leaned on for its REAL/FAKE call, "
            "not evidence that the news itself is true or false."
        ),
        "direction_label": (
            "These are model-associated features: tokens the neural network's "
            "training linked to its REAL/FAKE decision. They reflect statistical "
            "patterns from the training data - not proof the news is true or "
            "false."
        ),
    }