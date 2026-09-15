"""Negation detection.

Handles statements like *"the government did NOT announce X"* versus
*"the government announced X"* so keyword overlap is not mistaken for support.
"""

from __future__ import annotations

import re

NEGATION_PATTERNS = [
    r"\bdid\s+not\b", r"\bdoes\s+not\b", r"\bdo\s+not\b", r"\bhas\s+not\b",
    r"\bhave\s+not\b", r"\bhad\s+not\b", r"\bcannot\b", r"\bcan't\b",
    r"\bwon't\b", r"\bwouldn't\b", r"\bisn't\b", r"\baren't\b", r"\basn't\b",
    r"\bain't\b", r"\bdidn't\b", r"\bdoesn't\b", r"\bdon't\b", r"\bhasn't\b",
    r"\bhaven't\b", r"\bnever\b", r"\bno\b", r"\bnothing\b", r"\bnobody\b",
    r"\bneither\b", r"\bnor\b", r"\bdenied\b", r"\bdenies\b", r"\brejected\b",
    r"\brefuted\b", r"\brefutes\b", r"\bdebunked\b", r"\bwithout\b",
    r"\bno longer\b", r"\bfalsely claimed\b", r"\bnot true\b", r"\bfalse\b",
    r"\nnot\b",
]

_NEG_RE = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]

# Words that look like negation but indicate the sentence is reporting that a
# *rumour/claim* is false ("the claim that X is false" -> X is false, but the
# surrounding context is a refutation). The presence of these is logged so the
# relevance layer can check whether evidence *also* negates the claim.
REFUTATION_MARKERS = re.compile(
    r"\b(denied|denies|rejected|refuted|debunked|false rumor|fake news|"
    r"misinformation|no evidence|there is no evidence)\b",
    re.IGNORECASE,
)


def detect_negation(text: str | None) -> dict:
    """Return negation info for *text*.

    ``{"negated": bool, "patterns": [...]}`` — ``negated`` means the text
    asserts a negative (e.g. "the government did NOT announce X").
    """
    text = (text or "").strip()
    matches = []
    for pattern, regex in zip(NEGATION_PATTERNS, _NEG_RE):
        if regex.search(text):
            matches.append(pattern.strip())
    return {"negated": len(matches) > 0, "patterns": matches[:6]}


def strip_negation_effect(text: str | None) -> str:
    """Return a version of the text with negators removed (for similarity)."""
    if not text:
        return text
    cleaned = re.sub(r"\bnot\b|\bnever\b|\bno\b|\bnothing\b", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(did|does|do|has|have|had|can|will|would|is|are|was|were)\s+not\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(didn't|doesn't|don't|hasn't|haven't|can't|cannot|won't|wouldn't|isn't|aren't)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())