"""AI-generated text detection for TruthLens AI.

Exposes :func:`detect_ai_generated` (and a module-level singleton detector)
used by the backend ``/api/detect-ai-text`` endpoint. The detector never
raises: model-load failures degrade to the heuristic baseline with a
``warning`` field.
"""

from .detector import AiTextDetector, detect_ai_generated, get_detector

__all__ = ["AiTextDetector", "detect_ai_generated", "get_detector"]
