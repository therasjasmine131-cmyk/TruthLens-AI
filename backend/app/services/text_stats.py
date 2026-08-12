"""Text statistics computed from the actual submitted article."""

from __future__ import annotations

import re

from ml.preprocess import clean_for_stats

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_CAPS_RE = re.compile(r"\b[A-Z][A-Za-z]*\b")


def compute_text_stats(headline: str | None, article: str) -> dict:
    """All values below are derived from the real input text."""
    text = clean_for_stats(article)
    full = clean_for_stats((headline or "") + " " + text)
    words = full.split()
    sentences = [s for s in _SENT_RE.split(full) if s.strip()] or ([full] if full.strip() else [])
    chars = len(article)
    unique = set(w.lower() for w in words)
    vocab_richness = len(unique) / len(words) if words else 0.0
    avg_sentence_len = len(words) / len(sentences) if sentences else 0.0

    return {
        "word_count": len(words),
        "character_count": chars,
        "sentence_count": len(sentences),
        "average_sentence_length": round(avg_sentence_len, 2),
        "unique_words": len(unique),
        "vocabulary_richness": round(vocab_richness, 4),
        "capitalized_words": len(_CAPS_RE.findall(text)),
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
    }
