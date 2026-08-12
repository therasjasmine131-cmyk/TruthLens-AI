"""Conversion of the binary model output into the REAL / FAKE / UNCERTAIN scheme.

A binary classifier yields ``p_real`` and ``p_fake`` (summing to 1). The model's
decision margin is ``margin = |p_real - p_fake|`` and the mass ``1 - margin``
represents how much the model is *undecided*. That mass is assigned to the
UNCERTAIN class:

    real'      = p_real * margin
    fake'      = p_fake * margin
    uncertain' = 1 - margin

These always sum to 1 and are all derived from the actual model probabilities.
UNCERTAIN becomes the winning class when the model's top probability is below
~0.78 (i.e. the margin is small), which is a principled "abstain when unsure"
behaviour.
"""

from __future__ import annotations

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"
LABEL_UNCERTAIN = "UNCERTAIN"


def confidence_level(confidence: float, levels: dict | None = None) -> str:
    """Map a confidence fraction to a human label using the configured bands."""
    levels = levels or {}
    very_high = float(levels.get("very_high_min", 0.90))
    high = float(levels.get("high_min", 0.75))
    moderate = float(levels.get("moderate_min", 0.50))

    if confidence >= very_high:
        return "Very High Confidence"
    if confidence >= high:
        return "High Confidence"
    if confidence >= moderate:
        return "Moderate Confidence"
    return "Low Confidence"


def three_way_prediction(p_real: float, p_fake: float) -> dict:
    """Compute prediction, confidence and the three-way probability split."""
    p_real = max(0.0, min(1.0, float(p_real)))
    p_fake = max(0.0, min(1.0, float(p_fake)))
    total = p_real + p_fake
    if total <= 0:
        p_real, p_fake = 0.5, 0.5
    else:
        p_real, p_fake = p_real / total, p_fake / total

    margin = abs(p_real - p_fake)
    uncertain = round(1 - margin, 6)
    real = round(p_real * margin, 6)
    fake = round(p_fake * margin, 6)

    if uncertain >= real and uncertain >= fake:
        prediction = LABEL_UNCERTAIN
        confidence = uncertain
    elif real >= fake:
        prediction = LABEL_REAL
        confidence = real
    else:
        prediction = LABEL_FAKE
        confidence = fake

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "real": real,
            "fake": fake,
            "uncertain": uncertain,
        },
        "model_raw": {"p_real": round(p_real, 6), "p_fake": round(p_fake, 6)},
    }
