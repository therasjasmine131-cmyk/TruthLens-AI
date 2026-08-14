"""Stylometric features used by the AI-text heuristic detector.

Each function returns a float in 0..1. Values near 1 indicate a signal that
is typical of AI-generated text; values near 0 are typical of human writing.
"""

from __future__ import annotations

import re
from statistics import mean, stdev

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Phrases that LLM writing (and low-effort AI copy) over-uses.
AI_PHRASES = [
    "it is important to note",
    "in today's world",
    "in the realm of",
    "plays a crucial role",
    "plays a vital role",
    "when it comes to",
    "as we all know",
    "it can be seen",
    "it is worth noting",
    "to sum up",
    "in conclusion",
    "in summary",
    "in order to",
    "due to the fact that",
    "a wide range of",
    "it is essential to",
    "in addition to",
    "on the other hand",
    "as a result",
    "furthermore",
    "moreover",
    "additionally",
    "consequently",
    "therefore",
    "overall",
    "lastly",
    "firstly",
    "secondly",
    "thirdly",
    "it is widely known",
    "in the field of",
    "there is no doubt that",
]


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if not parts and text.strip():
        return [text.strip()]
    return parts


def burstiness_score(text: str) -> float:
    """Coefficient-of-variation of sentence word counts, rescaled to 0..1.

    Human prose typically has highly variable sentence lengths; LLM output
    tends to be much more uniform. 1.0 = very human-like variation."""
    lens = [len(words(s)) for s in sentences(text)]
    lens = [n for n in lens if n > 0]
    if len(lens) < 3:
        return 0.5
    m = mean(lens)
    if m <= 0:
        return 0.5
    cv = stdev(lens) / m if len(lens) > 1 else 0.0
    return max(0.0, min(1.0, cv / 0.9))


def repetition_score(text: str) -> float:
    """Fraction of overlapping word-trigrams that are duplicates (0..1).

    AI text reuses phrasing more than human writing."""
    toks = words(text)
    if len(toks) < 6:
        return 0.5
    trigrams = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def marker_density(text: str) -> float:
    """AI-typical phrase density per 100 words, clipped to 0..1."""
    low = text.lower()
    count = sum(low.count(p) for p in AI_PHRASES)
    n_words = max(1, len(words(text)))
    return min(1.0, (count / n_words) * 100 / 4.0)


def structure_uniformity(text: str) -> float:
    """Detect bullet/list-like uniform structures common in AI summaries."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.5
    bulletish = sum(
        1 for ln in lines if re.match(r"^\s*(?:[*\-•]|\d+[.)])\s+", ln)
    )
    ratio = bulletish / len(lines)
    return max(0.0, min(1.0, 0.3 + ratio * 0.7))
