"""Text preprocessing helpers for the TruthLens AI pipeline.

The neural network tokenizes *words* (not TF-IDF n-grams), so preprocessing
only needs to produce a clean, lower-cased, punctuation-free string. There is
no dependency on scikit-learn or any external NLP model; a small built-in
stop-word list is used by the legacy ``tokenize()`` helper only.
"""

from __future__ import annotations

import re

#: Built-in stop words used by :func:`tokenize` (word-frequency keywords).
STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "are", "was", "were", "will", "would", "could", "should", "can", "not",
    "but", "all", "she", "he", "they", "them", "their", "there", "here",
    "been", "being", "which", "who", "whom", "whose", "than", "then", "when",
    "where", "what", "why", "how", "into", "onto", "over", "under", "between",
    "among", "through", "during", "about", "after", "before", "because",
    "while", "may", "might", "must", "do", "does", "did", "doing", "at", "an",
    "as", "if", "in", "of", "on", "or", "so", "to", "by", "doe", "else",
    "each", "few", "more", "most", "some", "such", "same", "too", "very",
    "just", "also", "new", "one", "two", "reuters", "said", "says",
    "according",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def clean_for_features(text: str | None) -> str:
    """Aggressive cleaning used before tokenization.

    Lower-cases, strips accents and punctuation, removes HTML/URLs and keeps
    only word characters - the same normalisation applied during NN training.
    """
    if not text:
        return ""
    from unicodedata import normalize

    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).lower()
    return text.strip()


def clean_text(text: str | None) -> str:
    """Return a lower-cased, punctuation-free version of *text*."""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip().lower()


def clean_for_stats(text: str | None) -> str:
    """Light cleaning that keeps sentence punctuation for statistics."""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str | None) -> list[str]:
    """Split cleaned text into words, dropping stop-words and short tokens."""
    words = clean_text(text).split()
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def join_text(headline: str | None, article: str | None) -> str:
    """Combine headline and article the same way the training data was built."""
    parts = [p for p in (headline, article) if p and p.strip()]
    return " ".join(parts).strip()