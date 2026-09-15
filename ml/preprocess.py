"""Text preprocessing helpers for the TruthLens AI pipeline.

The pipeline deliberately uses only standard-library `re` and scikit-learn's
bundled English stop-word list so that training and inference run without any
external NLTK / spaCy model downloads. NLTK can be swapped in for fancier
lemmatization without changing the rest of the architecture.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#: Tokens considered placeholders and dropped during feature extraction.
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "reuters",
    "said",
    "says",
    "also",
    "new",
    "would",
    "could",
    "according",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def clean_for_features(text: str | None) -> str:
    """Aggressive cleaning used for TF-IDF features.

    Lower-cases, strips accents and punctuation, expands common short forms and
    keeps only word characters - string n-grams then see consistent text.
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
    """Return a lower-cased, punctuation-free version of *text*.

    Newlines are preserved as spaces so that sentence splitting still works on
    the original text (see :func:`clean_for_stats`).
    """
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
