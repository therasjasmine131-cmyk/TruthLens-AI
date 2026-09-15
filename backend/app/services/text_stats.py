"""Text statistics computed from the exact submitted text.

All values below are derived from the *combined* input (headline + article) --
exactly the same text that is sent to the ML model. Using one source string for
both statistics and the model prevents the "0 chars vs 24 words" mismatch where
the on-page counter and the result statistics were looking at different text.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_SENT_RE = re.compile(r"[.!?]+")
_CAPS_RE = re.compile(r"\b[A-Z][A-Za-z]*\b")


def compute_text_stats(headline: str | None, article: str | None) -> dict:
    """Compute stats from the combined (headline + article) submitted text."""
    full = ((headline or "") + " " + (article or "")).strip()

    words = [w for w in _WS_RE.split(full) if w]
    sentences = [s for s in _SENT_RE.split(full) if s.strip()]
    unique = set(w.lower() for w in words)

    chars = len(full)
    return {
        "word_count": len(words),
        "character_count": chars,
        "sentence_count": len(sentences),
        "average_sentence_length": round(len(words) / len(sentences), 2) if sentences else 0.0,
        "unique_words": len(unique),
        "vocabulary_richness": round(len(unique) / len(words), 4) if words else 0.0,
        "capitalized_words": len(_CAPS_RE.findall(full)),
        "exclamation_marks": full.count("!"),
        "question_marks": full.count("?"),
    }

