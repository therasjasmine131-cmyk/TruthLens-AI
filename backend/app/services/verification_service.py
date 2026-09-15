"""Integration service: merges the evidence-based verification pipeline
(REAL / FALSE / UNVERIFIED) with the existing ML prediction into one API-safe
dictionary so the frontend can render a proper evidence matrix."""

from __future__ import annotations

import logging

from ..ml.model_manager import model_manager
from .verification.verifier import verify_text

logger = logging.getLogger("truthlens.verification_service")


def _article_ml(headline: str | None, article: str | None) -> dict | None:
    """Article-level ML prediction (stylistic signal, not a verdict)."""
    combined = " ".join(filter(None, [headline, article])).strip()
    if not combined or not model_manager.ready:
        return None
    try:
        p_real, p_fake = model_manager.predict_proba(combined)
        return {
            "prediction": "REAL" if p_real >= p_fake else "FALSE",
            "prediction_classic": "REAL" if p_real >= p_fake else "FAKE",
            "confidence": round(max(p_real, p_fake), 3),
            "probabilities": {"real": round(p_real, 3), "fake": round(p_fake, 3)},
        }
    except Exception:  # noqa: BLE001 - ML must never break verification
        logger.exception("article-level ML signal failed")
        return None


def run_verification(headline: str | None, article: str | None) -> dict | None:
    """Run the evidence pipeline. Returns None only if there is no text.

    ``verify_text`` already joins headline+text itself, so pass article as the
    body to avoid duplicating the headline.
    """
    body = article if (article or "").strip() else None
    if not (headline or "").strip() and body is None:
        return None
    try:
        report = verify_text(body, headline=headline)
    except Exception:  # noqa: BLE001 - evidence failures degrade, not crash
        logger.exception("verification pipeline failed")
        report = {
            "status": "failed",
            "language": None,
            "pipeline": None,
            "claims": [],
            "overall": {
                "verdict": "UNVERIFIED",
                "confidence": 0.4,
                "confidence_label": "Verification unavailable",
                "explanation": "The verification pipeline could not be completed.",
                "mixed": False,
            },
            "evidence_matrix": [],
            "notes": {},
        }
    if report is not None:
        report["ml_article"] = _article_ml(headline, article)
    return report