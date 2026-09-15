"""Shared text helpers for the verification pipeline.

Everything here is standard-library only (regex + difflib), so the whole
analysis pipeline runs offline without NLTK/spaCy model downloads.
"""

from __future__ import annotations

import difflib
import re

_WS_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[^a-z0-9\u0B80-\u0BFF]+")
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Abbreviations that must not trigger sentence splits.
_ABBR_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|approx|"
    r"No|Inc|Ltd|Jr|Govt|Dept|U\.S|U\.K|A\.M|P\.M|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.(?=\s)"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\u0B80-\u0BFF\"'“„])")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "as", "that", "this", "these", "those", "it", "its", "not",
    "no", "so", "if", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "than", "too", "very", "just", "does",
    "doing", "done", "has", "have", "had", "having", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "among", "let", "while", "you", "your", "we", "our", "they", "their",
    "he", "she", "him", "her", "his", "i", "me", "my", "us", "them", "what",
    "who", "whom", "which", "whose",
}


def clean_text(text: str | None) -> str:
    """Lower-case, alphanumeric-only (keeps Tamil script) normalization."""
    if not text:
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str | None) -> list[str]:
    """Return meaningful tokens of *text* (drops stop words and single chars)."""
    tokens = clean_text(text).split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def token_set(text: str | None) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str | None, b: str | None) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_similarity(a: str | None, b: str | None) -> float:
    """Character-level ratio, robust to word-order changes and Tanglish spelling."""
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def combined_similarity(a: str | None, b: str | None) -> float:
    ra, rb = char_similarity(a, b), jaccard(a, b)
    return 0.5 * ra + 0.5 * rb


def split_sentences(text: str | None) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    if not text:
        return []
    text = _HTML_RE.sub(" ", text or "")
    text = _WS_RE.sub(" ", text).strip()
    text = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return [p.replace("<DOT>", ".") for p in parts]


def word_count(text: str | None) -> int:
    return len(tokenize(text))