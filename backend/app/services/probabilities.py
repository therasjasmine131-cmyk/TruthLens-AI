"""Conversion of the binary model output into REAL / FAKE / UNCERTAIN labels.

The trained classifier is **binary**: it outputs ``p_real`` and ``p_fake`` which
always sum to 1. UNCERTAIN is **not** a third class with its own probability --
there is no third class in the model. Instead, UNCERTAIN is an *abstain*
decision we make when the model's top-class probability is too low to trust
(i.e. the model is "undecided").

Honesty rules enforced here:
* ``probabilities.real`` and ``probabilities.fake`` are the raw model
  probabilities (they sum to 1, i.e. 100%).
* ``probabilities.uncertain`` is always ``0.0`` -- we never manufacture a
  phoney uncertainty percentage by subtracting anything from the raw
  probabilities.
* ``confidence`` is the raw top-class probability (``max(p_real, p_fake)``).
* When ``confidence`` drops below :data:`UNCERTAIN_THRESHOLD` we abstain and
  label the result UNCERTAIN.
"""

from __future__ import annotations

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"
LABEL_UNCERTAIN = "UNCERTAIN"

#: Abstain threshold: if the model's top-class probability is below this, we
#: do not trust either class and return UNCERTAIN (an abstain, not a class).
UNCERTAIN_THRESHOLD = 0.78


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
    """Compute an honest prediction/confidence and the raw probability split.

    Returns a dict with:
      * ``prediction``   - REAL, FAKE or UNCERTAIN (abstain)
      * ``confidence``   - the raw top-class probability
      * ``decided``      - True if a REAL/FAKE call was made, False if abstained
      * ``uncertain_threshold`` - the abstain threshold used
      * ``probabilities`` - real + fake = 1 (100%), uncertain = 0.0
      * ``model_raw``    - the raw binary probabilities
    """
    p_real = max(0.0, min(1.0, float(p_real)))
    p_fake = max(0.0, min(1.0, float(p_fake)))
    total = p_real + p_fake
    if total <= 0:
        p_real, p_fake = 0.5, 0.5
    else:
        p_real, p_fake = p_real / total, p_fake / total

    confidence = max(p_real, p_fake)

    if confidence < UNCERTAIN_THRESHOLD:
        prediction = LABEL_UNCERTAIN
        decided = False
    elif p_real >= p_fake:
        prediction = LABEL_REAL
        decided = True
    else:
        prediction = LABEL_FAKE
        decided = True

    return {
        "prediction": prediction,
        "confidence": confidence,
        "decided": decided,
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
        "probabilities": {
            "real": round(p_real, 6),
            "fake": round(p_fake, 6),
            "uncertain": 0.0,
        },
        "model_raw": {"p_real": round(p_real, 6), "p_fake": round(p_fake, 6)},
    }

