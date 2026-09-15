"""Language detection and normalization for English / Tamil / Tanglish.

* Tamil: text containing Tamil Unicode script (U+0B80-U+0BFF).
* Tanglish: Latin script that mixes Tamil function words/markers with English
  content words (e.g. ``"school la 15 days close ah?"``).
* English: anything else recognizably Latin.

For search we build an *English-oriented query* from Tanglish by dropping Tamil
function words and romanized markers while keeping the English content words.
This makes ``"tamilnadu schools 15 days leave"`` and ``"TN la school 15 days
close ah?"`` converge on the same claim query.
"""

from __future__ import annotations

import re

TAMIL_RANGE_RE = re.compile(r"[\u0B80-\u0BFF]")

# Romanized Tamil function words / spoken markers common in Tanglish.
TANGLISH_MARKERS = {
    "ah", "aa", "aho", "ahp", "keh", "ha", "da", "di", "ma", "pa", "rey",
    "illa", "illai", "ilaa", "illaa", "konjam", "konja","nalla", "therila",
    "theriyala", "theriyum", "mudiyum", "mudiyadhu", "mudiyathu",
    "enga", "engal", "unga", "ungal", "enakku", "ena", "enna", "eppadi",
    "eppo", "engae", "yenga", "yepdi", "pannalam", "pannam", "pannunga",
    "aachu", "aagiduchu", "vachu", "vandhu", "vandha", "poi", "pottu",
    "irukku", "iruka", "illadha", "oda", "kooda", "kuda", "pakkam",
    "thaan", "daan", "tane", "than", "pora", "varalaam", "vedukalam",
    "class", "school",
}

# Tamil function words (romanized) that should be dropped when building queries.
TANGLISH_FUNCTION_WORDS = {
    "ah", "aa", "ha", "da", "di", "la", "le", "laa", "neh", "na", "de",
    "illa", "illai", "ilaa", "konjam", "konja", "nalla", "therila",
    "theriyala", "mudiyum", "mudiyadhu", "enakku", "ena", "enna", "eppadi",
    "eppo", "engae", "yenga", "pannalam", "aachu", "vachu", "vandhu",
    "pottu", "irukku", "iruka", "oda", "kooda", "kuda", "pakkam", "thaan",
    "daan", "tane", "than", "ge", "ae", "u",
}

# Common Tanglish -> English spelling variants for matching robustness.
TANGLISH_VARIANTS = {
    "tamilnadu": "tamil nadu",
    "tn": "tamil nadu",
    "chennai": "chennai",
    "govt": "government",
    "govt.": "government",
    "clg": "college",
    "scl": "school",
    "schl": "school",
    "leave": "holiday",
    "holidays": "holiday",
    "close": "closed",
    "closed": "closed",
    "colleges": "college",
    "schools": "school",
    "sol": "school",
}

MODE_AUTO = "auto"
MODE_ENGLISH = "english"
MODE_TAMIL = "tamil"
MODE_TANGLISH = "tanglish"
VALID_MODES = {MODE_AUTO, MODE_ENGLISH, MODE_TAMIL, MODE_TANGLISH}


def detect_language(text: str | None, forced: str | None = None) -> dict:
    """Detect the language of *text*.

    Returns ``{"code", "label"}`` where code is one of english/tamil/tanglish.
    """
    text = (text or "").strip()
    code = MODE_ENGLISH
    if not text:
        code = MODE_AUTO
    elif forced in VALID_MODES and forced != MODE_AUTO:
        code = forced
    elif TAMIL_RANGE_RE.search(text):
        code = MODE_TAMIL
    else:
        words = set(re.findall(r"[a-z]+", text.lower()))
        if len(TANGLISH_MARKERS & words) >= 1 or _has_tanglish_suffix(text):
            code = MODE_TANGLISH
    label = {
        MODE_ENGLISH: "English",
        MODE_TAMIL: "Tamil",
        MODE_TANGLISH: "Tanglish",
        MODE_AUTO: "Auto-detect",
    }[code]
    return {"code": code, "label": label}


def _has_tanglish_suffix(text: str) -> bool:
    """Detect spoken-style markers like ``school la``, ``close ah``."""
    return bool(
        re.search(
            r"\b(school|college|class|holiday|leave|news|today|election)\s+"
            r"(la|le|ah|aa|anna|unga|oda|pannu|pannunga|irukku|vandhu)\b",
            text.lower(),
        )
    )


def build_search_query(text: str | None, language_code: str = MODE_ENGLISH) -> str:
    """Build an English-ish search query from the raw claim text.

    * Tamil: keep the original Tamil text as the query (Wikipedia etc. support it).
    * Tanglish: drop Tamil function words and normalize common variants, then
      search English content words.
    * English: pass through (lightly cleaned).
    """
    text = (text or "").strip()
    if not text:
        return ""
    if language_code == MODE_TAMIL:
        return text

    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    normalized = []
    for w in words:
        if w in TANGLISH_FUNCTION_WORDS:
            continue
        if w.isalpha() or w.isdigit():
            normalized.append(TANGLISH_VARIANTS.get(w, w))
        else:
            normalized.append(w)
    return " ".join(normalized).strip() or text


def normalize_for_matching(text: str | None, language_code: str = MODE_ENGLISH) -> str:
    """Normalize text so Tanglish and English variants collapse together."""
    text = build_search_query(text, language_code)
    from .textutil import clean_text
    return clean_text(text)