"""Lightweight entity extraction (people / organizations / locations / currencies).

Uses capitalization patterns plus a known-entities lexicon so the whole thing
runs offline. The extracted entities are used to build better search queries
and to check entity overlap between a claim and its evidence.
"""

from __future__ import annotations

import re

KNOWN_ORGANIZATIONS = {
    "rbi", "isro", "nasa", "who", "un", "united nations", "unesco", "unicef",
    "world bank", "imf", "cbi", "ed", "sebi", "ncdrc", "apec", "niti aayog",
    "gst council", "delhi government", "tamil nadu government", "kerala government",
    "karnataka government", "central government", "state government", "government of india",
    "government of tamil nadu", "pm", "pmo", "cabinet", "supreme court", "high court",
    "election commission", "ec", "customs", "trai", "airtel", "jio", "bcci", "icc",
    "fifa", "icmr", "covid", "cisf", "crpf", "iit", "iisc", "aiims",
}

KNOWN_LOCATIONS = {
    "india", "tamil nadu", "chennai", "new delhi", "delhi", "mumbai", "bengaluru",
    "hyderabad", "kolkata", "pune", "kochi", "kerala", "karnataka", "maharashtra",
    "andhra pradesh", "telangana", "gujarat", "rajasthan", "uttar pradesh",
    "punjab", "haryana", "bihar", "odisha", "west bengal", "assam", "goa",
    "chandigarh", "pondicherry", "jammu", "kashmir", "sri lanka", "pakistan",
    "bangladesh", "china", "united states", "usa", "us", "uk", "europe", "venice",
    "pacific", "atlantic", "indian ocean", "arctic", "antarctica", "asia",
}

_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_SKIP_CAPS = {
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "and", "or", "with",
    "by", "from", "after", "before", "during", "over", "under", "news", "world",
    "india", "school", "schools", "college", "colleges", "government", "state",
    "city", "today", "yesterday", "tomorrow", "week", "month", "year", "new",
}

_CAPS_SEQUENCE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_MONTH_DAY_RE = re.compile(
    r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
    r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{2,4}|\s+\d{2,4})?\b",
    re.I,
)
_MONTH_YEAR_RE = re.compile(
    r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
    r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
    r"\s+\d{4}\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_CURRENCY_RE = re.compile(
    r"\b(?:Rs\.?|₹|\$|£|€)?\s?\d[\d,]*(?:\.\d+)?\s?(?:crore|lakh|million|billion|"
    r"rupees|rps|dollars|usd|inr|rs)?\b",
    re.I,
)


def _is_common_caps(word: str) -> bool:
    return word.lower() in _SKIP_CAPS or word.lower().rstrip("s,.") in _SKIP_CAPS


def extract_entities(text: str | None) -> dict:
    """Extract people/org/location/date mentions. Returns dict of lists."""
    text = (text or "").strip()
    result: dict = {"people": [], "organizations": [], "locations": [],
                    "dates": [], "years": [], "currencies": []}
    if not text:
        return result

    lowered = text.lower()
    for org in KNOWN_ORGANIZATIONS:
        if re.search(rf"\b{re.escape(org)}\b", lowered):
            result["organizations"].append(org.title() if " " not in org else org.title())
    for loc in KNOWN_LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", lowered):
            result["locations"].append(loc.title())

    result["locations"] = _dedup_order(result["locations"])
    result["organizations"] = _dedup_order(result["organizations"])

    # "Ministry of X", "Department of X"
    for m in re.finditer(
        r"\b(?:ministry|department|government of)\s+of?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        text,
    ):
        term = re.sub(r"\s+", " ", m.group(0))
        if term.lower().strip() not in {o.lower() for o in result["organizations"]}:
            result["organizations"].append(term)

    # Capitalized sequences that look like proper nouns (people/places).
    for m in _CAPS_SEQUENCE_RE.finditer(text):
        phrase = m.group(1)
        head = phrase.split()[0].lower()
        if _is_common_caps(head) or head in _MONTHS or head in _DAYS:
            continue
        # degree phrases like "Prime Minister Narendra Modi" -> keep, classify org-ish
        if re.search(r"\b(?:minister|ministerial|president|commissioner|"
                     r"director|secretary|governor|chief|chairman|professor|dr)\b", phrase, re.I):
            result["people"].append(phrase)
        elif phrase.lower() not in {p.lower() for p in result["organizations"] + result["locations"]}:
            result["people"].append(phrase)

    for m in _MONTH_DAY_RE.finditer(text):
        result["dates"].append(m.group(0).strip())
    for m in _MONTH_YEAR_RE.finditer(text):
        result["dates"].append(m.group(0).strip())
    for m in _YEAR_RE.finditer(text):
        result["years"].append(m.group(0))
    for m in _CURRENCY_RE.finditer(text):
        token = re.sub(r"\s+", " ", m.group(0)).strip()
        if re.search(r"\d", token):
            result["currencies"].append(token[:40])

    result["people"] = _dedup_order(result["people"])[:12]
    result["organizations"] = _dedup_order(result["organizations"])[:12]
    result["locations"] = _dedup_order(result["locations"])[:10]
    result["dates"] = _dedup_order(result["dates"])[:8]
    result["years"] = _dedup_order(result["years"])[:8]
    result["currencies"] = _dedup_order(result["currencies"])[:8]
    return result


def _dedup_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = re.sub(r"[^a-z0-9]+", "", item.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out