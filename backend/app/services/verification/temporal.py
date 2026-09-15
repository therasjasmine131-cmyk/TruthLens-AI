"""Temporal expression extraction and temporal-consistency checking.

Detects absolute dates, years, and relative terms ("today", "yesterday",
"this week", "recently", ...) plus tense markers ("will", "announced"), then
classifies a claim's temporal reference so evidence publication dates can be
checked for freshness (a 2022 article is not evidence that something happened
in 2026).
"""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MON_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|" \
          r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

DATE_RE = re.compile(
    rf"\b({_MON_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?"
    r"|\b(\d{{1,2}})[-/](\d{{1,2}})[-/](\d{{2,4}})"
    rf"|\b({_MON_RE})\s+(\d{{4}})\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

RELATIVE_TERMS = {
    "today": "today", "yesterday": "yesterday", "tomorrow": "tomorrow",
    "now": "current", "currently": "current", "this week": "current",
    "this month": "current", "this year": "current", "this quarter": "current",
    "recently": "recent", "latest": "recent", "recent": "recent",
    "last week": "past", "last month": "past", "last year": "past",
    "past week": "past", "past month": "past", "past year": "past",
    "last night": "past", "last evening": "past", "this morning": "current",
    "tonight": "current", "in the coming": "future", "upcoming": "future",
    "next week": "future", "next month": "future", "next year": "future",
    "soon": "future", "shortly": "future",
}

_REFERENCE_REGEXES = [
    (re.compile(r"\b(today)\b", re.I), "today"),
    (re.compile(r"\b(yesterday)\b", re.I), "yesterday"),
    (re.compile(r"\b(tomorrow)\b", re.I), "tomorrow"),
    (re.compile(r"\b(this\s+week)\b", re.I), "this week"),
    (re.compile(r"\b(this\s+month)\b", re.I), "this month"),
    (re.compile(r"\b(this\s+year)\b", re.I), "this year"),
    (re.compile(r"\b(recently|latest|recent)\b", re.I), "recent"),
    (re.compile(r"\b(last\s+week|last\s+month|last\s+year)\b", re.I), "recent"),
]

TENSE_CURRENT = re.compile(r"\b(announced|reported|confirmed|as of|according to|"
                           r"as per|has been|have been|was|were)\b", re.I)
TENSE_FUTURE = re.compile(r"\b(will|tomorrow|next\s+week|next\s+month|next\s+year|"
                          r"upcoming|by\s+2030|by\s+2029|by\s+2028)\b", re.I)
TENSE_PAST = re.compile(r"\b(was|were|had|has happened|happened|occurred|took place|"
                        r"yesterday|last\s+week|last\s+month|last\s+year)\b", re.I)


def extract_temporal(text: str | None) -> dict:
    """Return ``{"dates", "years", "relative", "references"}``."""
    text = (text or "").strip()
    dates: list[str] = []
    years: list[str] = []
    relative: set[str] = set()

    for m in DATE_RE.finditer(text):
        token = m.group(0).strip()
        if token and token not in dates:
            dates.append(token)
    for m in YEAR_RE.finditer(text):
        if m.group(0) not in years:
            years.append(m.group(0))
    lowered = " " + text.lower() + " "
    for rx, key in _REFERENCE_REGEXES:
        if rx.search(lowered):
            relative.add(key)
    return {
        "dates": dates[:8],
        "years": years[:8],
        "relative": sorted(relative),
        "references": _temporal_reference(dates, years, relative, text),
    }


def _temporal_reference(dates, years, relative, text) -> str:
    """Classify the claim as historical / current / future / timeless."""
    text_lower = text.lower()
    if any(t in relative for t in ("tomorrow",)) or TENSE_FUTURE.search(text_lower):
        return "future"
    if any(t in relative for t in ("yesterday", "recent")) or TENSE_PAST.search(text_lower):
        return "recent-past"
    if any(t in relative for t in ("today", "this week", "this month", "this year", "current")):
        return "current"
    # Explicit past years/dates push toward historical; current-year mentions
    # with announcement verbs stay "current".
    try:
        latest_year = max(int(y) for y in years) if years else None
    except ValueError:
        latest_year = None
    if latest_year is not None and latest_year < date.today().year - 1:
        return "historical"
    if dates or TENSE_CURRENT.search(text_lower):
        return "recent-past"
    return "timeless"


def classify_reference(text: str | None) -> str:
    return extract_temporal(text)["references"]


def parse_date(token: str) -> date | None:
    """Best-effort ``date`` parse for common textual date formats."""
    token = token.strip().strip(",")
    nums = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", token)
    if nums:
        try:
            d, m, y = int(nums.group(1)), int(nums.group(2)), int(nums.group(3))
            y += 2000 if y < 100 else 0
            return date(y, m, d)
        except ValueError:
            pass
    m = re.fullmatch(rf"({_MON_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?", token, re.I)
    mm = re.fullmatch(rf"({_MON_RE})\s+(\d{{4}})", token, re.I)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        day, year = int(m.group(2)), int(m.group(3) or 0) or _current_year()
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    if mm:
        month = _MONTHS.get(mm.group(1).lower())
        if month:
            try:
                return date(int(mm.group(2)), month, 1)
            except ValueError:
                return None
    return None


def _current_year() -> int:
    return date.today().year


def check_temporal_match(claim_time_ref: str, evidence_date: str | None,
                         days_stale: int = 730) -> dict:
    """Compare the claim's time reference with an evidence publication date.

    Returns ``{"status", "reason"}`` where status is
    ``fresh | stale | future-aware | unknown | not_applicable``.
    """
    if not evidence_date:
        return {"status": "unknown", "reason": "Evidence has no publication date."}
    ev_date = parse_date(evidence_date)
    if ev_date is None:
        return {"status": "unknown", "reason": f"Could not parse date: {evidence_date}"}
    today = date.today()
    age_days = (today - ev_date).days

    if claim_time_ref == "current":
        if age_days < 0:
            return {"status": "future-aware", "reason": "Evidence dated in the future; treat cautiously."}
        return {"status": "fresh" if age_days <= days_stale else "stale",
                "reason": f"Evidence is {age_days} days old."}
    if claim_time_ref == "recent-past":
        return {"status": "fresh" if age_days <= days_stale else "stale",
                "reason": f"Evidence is {age_days} days old."}
    if claim_time_ref == "future":
        return {"status": "unknown", "reason": "Claim concerns the future; publication alone is not proof."}
    if claim_time_ref in ("timeless", "historical"):
        return {"status": "not_applicable", "reason": "Timeless/historical claim; publication age is less critical."}
    return {"status": "not_applicable", "reason": "No clear temporal signal."}


def freshness_bonus(status: str) -> float:
    return {
        "fresh": 0.15,
        "not_applicable": 0.05,
        "unknown": 0.0,
        "future-aware": -0.05,
        "stale": -0.25,
    }[status]