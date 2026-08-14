"""AI-generated text detector.

Backends (best available wins, weakest always works):

1. **Gemini backend (optional).** Calls the Google Gemini API with the key in
   ``GEMINI_API_KEY`` and blends its verdict with the heuristic for stability.
   Requires network access; unavailable keys degrade silently.
2. **Transformer backend (optional).** Loads ``roberta-base-openai-detector``
   when ``transformers``/``torch`` are installed and ``TRANSFORMER_AI_DETECTOR=1``.
3. **Heuristic baseline (default).** Zero external dependencies. Combines
   burstiness, phrase repetition, AI-typical phrasing density, structural
   uniformity and a bigram perplexity proxy trained on the human-written news
   corpus (see ``build_lm.py``).

:func:`detect_ai_generated` never raises. If an optional backend fails to load
it silently falls back to the heuristic and marks the result with a ``warning``
field.

Limitations: AI-text detection is probabilistic, model-dependent and not
100% reliable. Short or highly stylised text may be mislabelled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from . import features
from .gemini import GeminiClassifier
from .lm import BigramLM

LABEL_LIKELY_AI = "Likely AI"
LABEL_LIKELY_HUMAN = "Likely Human"
LABEL_UNCERTAIN = "Uncertain"

LABEL_AI_MIN = 0.60
LABEL_HUMAN_MAX = 0.40

# Heuristic signal weights (sums to 1.0). Configurable via env for tuning.
WEIGHTS = {
    "burstiness": float(os.environ.get("AI_DETECT_W_BURSTINESS", "0.30")),
    "perplexity": float(os.environ.get("AI_DETECT_W_PERPLEXITY", "0.25")),
    "repetition": float(os.environ.get("AI_DETECT_W_REPETITION", "0.20")),
    "markers": float(os.environ.get("AI_DETECT_W_MARKERS", "0.15")),
    "structure": float(os.environ.get("AI_DETECT_W_STRUCTURE", "0.10")),
}

MIN_CHARS = 20


@dataclass
class _TransformerState:
    pipe: Any | None = None
    error: str | None = None
    tried: bool = False


class AiTextDetector:
    """Lazy-loading singleton with heuristic + optional Gemini/RoBERTa backends."""

    def __init__(self) -> None:
        self._lm: BigramLM | None = None
        self._transformer = _TransformerState()
        self._gemini: GeminiClassifier | None = None

    # ---- lazy backends ---------------------------------------------------
    @property
    def lm(self) -> BigramLM:
        if self._lm is None:
            self._lm = BigramLM.load()
        return self._lm

    @property
    def gemini(self) -> GeminiClassifier | None:
        if self._gemini is None and os.environ.get("GEMINI_API_KEY", "").strip():
            self._gemini = GeminiClassifier()
        return self._gemini

    def _transformer_enabled(self) -> bool:
        return os.environ.get("TRANSFORMER_AI_DETECTOR", "").strip().lower() in {"1", "true", "yes", "on"}

    def _load_transformer(self) -> Any | None:
        """Load the HF pipeline once; returns None on any failure."""
        state = self._transformer
        if state.tried:
            return state.pipe
        state.tried = True
        if not self._transformer_enabled():
            state.error = "TRANSFORMER_AI_DETECTOR is not enabled"
            return None
        try:
            from transformers import pipeline  # type: ignore

            model = os.environ.get("HF_AI_DETECTOR_MODEL", "roberta-base-openai-detector")
            state.pipe = pipeline("text-classification", model=model, truncation=True)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            state.error = str(exc)
            state.pipe = None
        return state.pipe

    # ---- heuristic -------------------------------------------------------
    def _perplexity_component(self, text: str) -> float:
        return self.lm.predictability_component(text)

    def heuristic_score(self, text: str) -> dict:
        burst = 1.0 - features.burstiness_score(text)  # 1 = uniform = AI-like
        perp = self._perplexity_component(text)
        rep = features.repetition_score(text)
        markers = features.marker_density(text)
        struct = features.structure_uniformity(text)
        w = WEIGHTS
        score = (
            w["burstiness"] * burst
            + w["perplexity"] * perp
            + w["repetition"] * rep
            + w["markers"] * markers
            + w["structure"] * struct
        )
        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "signals": {
                "burstiness_uniformity": round(burst, 4),
                "perplexity_predictability": round(perp, 4),
                "phrase_repetition": round(rep, 4),
                "ai_marker_density": round(markers, 4),
                "structure_uniformity": round(struct, 4),
            },
        }

    def _transformer_score(self, text: str) -> float | None:
        """0..1 AI-likelihood from the RoBERTa detector, or None."""
        pipe = self._load_transformer()
        if pipe is None:
            return None
        try:
            out = pipe(text[:2000], truncation=True)[0]
            label = out.get("label", "")
            score = float(out.get("score", 0.5))
            # roberta-base-openai-detector emits "LABEL_1" (fake/AI) or "LABEL_0" (real).
            if "LABEL_1" in label or label.lower() in {"fake", "ai", "ai-generated"}:
                return score
            if "LABEL_0" in label or label.lower() in {"real", "human"}:
                return 1.0 - score
            return score
        except Exception:  # noqa: BLE001
            return None

    # ---- public API ------------------------------------------------------
    def detect(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "Text is required.", "status": "error"}

        heuristic = self.heuristic_score(text)
        result: dict[str, Any] = {
            "ai_generated_score": heuristic["score"],
            "label": _score_to_label(heuristic["score"]),
            "backend": "heuristic",
            "signals": heuristic["signals"],
            "input_chars": len(text),
        }

        # Gemini is the preferred optional backend; the transformer (if enabled)
        # is only consulted when Gemini is unconfigured.
        gemini_client = self.gemini
        gemini_score = self._gemini_score(text) if gemini_client is not None else None
        if gemini_score is not None:
            blended = 0.7 * gemini_score + 0.3 * heuristic["score"]
            result["ai_generated_score"] = round(max(0.0, min(1.0, blended)), 4)
            result["label"] = _score_to_label(result["ai_generated_score"])
            result["backend"] = "gemini"
            result["gemini_score"] = round(gemini_score, 4)
            return result
        if gemini_client is not None:
            result["warning"] = "Gemini detector failed; using heuristic fallback."
            return result

        transformer_score = self._transformer_score(text)
        if transformer_score is not None:
            # Blend the transformer signal with the heuristic for stability.
            blended = 0.7 * transformer_score + 0.3 * heuristic["score"]
            result["ai_generated_score"] = round(max(0.0, min(1.0, blended)), 4)
            result["label"] = _score_to_label(result["ai_generated_score"])
            result["backend"] = "roberta"
            result["transformer_score"] = round(transformer_score, 4)
        elif self._transformer.tried:
            result["warning"] = (
                "Transformer detector unavailable; using heuristic fallback. "
                f"({self._transformer.error or 'not enabled'})"
            )
        return result

    def _gemini_score(self, text: str) -> float | None:
        """0..1 AI-likelihood from Gemini, or None when unavailable/failed."""
        client = self.gemini
        if client is None or not client.available:
            return None
        return client.classify(text)


def _score_to_label(score: float) -> str:
    if score >= LABEL_AI_MIN:
        return LABEL_LIKELY_AI
    if score <= LABEL_HUMAN_MAX:
        return LABEL_LIKELY_HUMAN
    return LABEL_UNCERTAIN


ai_text_detector = AiTextDetector()


def get_detector() -> AiTextDetector:
    return ai_text_detector


def detect_ai_generated(text: str) -> dict:
    """Detect whether ``text`` was likely AI-generated.

    Returns ``{"ai_generated_score": 0..1, "label": ..., "backend": ...,
    "signals": {...}}``. Never raises; on invalid input returns an ``error``
    dict with ``status: "error"``.
    """
    return ai_text_detector.detect(text)
