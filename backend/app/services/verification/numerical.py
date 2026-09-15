"""Numerical claim extraction and comparison.

Handles percentages, money, counts, durations (days/years/hours), temperatures,
and statistics. Numbers present in a claim are treated as *key assertions*:
an evidence source that gives a materially different figure for the same
quantity supports a CONTRADICTS relation.
"""

from __future__ import annotations

import re

_UNIT_MAP = {
    "days": "days", "day": "days", "weeks": "weeks", "week": "weeks",
    "months": "months", "month": "months", "years": "years", "year": "years",
    "hours": "hours", "hour": "hours", "minutes": "minutes", "minute": "minutes",
    "percent": "percent", "percentage": "percent", "%": "percent",
    "people": "people", "students": "people", "crore": "crore", "lakh": "lakh",
    "million": "million", "billion": "billion", "rupees": "currency",
    "rs": "currency", "inr": "currency", "dollars": "currency", "usd": "currency",
    "degrees": "temperature", "degree": "temperature", "celsius": "temperature",
    "km": "length", "kilometers": "length", "kmph": "speed", "km/h": "speed",
    "kg": "mass", "kilograms": "mass", "litres": "volume", "liters": "volume",
}

_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|₹|rs\.?|usd|\$|kmph|km/h|km|kg|"
    r"crore|lakh|million|billion|days?|weeks?|months?|years?|hours?|minorities?|"
    r"minutes?|people|students|degrees?|celsius|litres?|liters?|percent|percentage)"
    r"|(?<!\w)(\d+(?:[.,]\d+)?)(?![\w.])",
    re.I,
)
_NUM_ONLY_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w.])(?= *[^a-zA-Z]|$)")
_CURRENCY_PREFIX_RE = re.compile(r"(?<!\w)([₹$€£]\s?\d[\d,]*(?:\.\d+)?)")

_LARGE_MULTIPLIERS = {"lakh": 100000, "crore": 10000000, "million": 1000000,
                      "billion": 1000000000}


def _to_float(token: str) -> float:
    token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return float("nan")


def extract_numbers(text: str | None) -> list[dict]:
    """Extract numbers with their units and scaled values.

    Each item: ``{"raw", "value", "scaled", "unit"}``. ``scaled`` normalizes
    lakh/crore/million/billion to plain numbers so comparisons are fair.
    """
    text = (text or "").replace(",", "")
    found: list[dict] = []
    seen: set[tuple] = set()

    for m in _NUMBER_TOKEN_RE.finditer(text):
        value_tok, unit = m.group(1), m.group(2)
        bare = m.group(3)
        raw = m.group(0)
        if value_tok is None and bare is None:
            continue
        if value_tok:
            value = _to_float(value_tok)
            unit_key = _UNIT_MAP.get(unit.lower())
            scaled = value * _LARGE_MULTIPLIERS.get(unit.lower(), 1) \
                if unit.lower() in _LARGE_MULTIPLIERS else value
        else:
            value = _to_float(bare)
            unit_key = None
            scaled = value
        key = (round(scaled, 2), unit_key)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "raw": raw,
            "value": round(value, 4),
            "scaled": round(scaled, 5),
            "unit": unit_key,
        })

    # Currency-prefixed numbers (₹50,000 / $10,000)
    for m in _CURRENCY_PREFIX_RE.finditer(text):
        token = re.sub(r"[\s₹$€£,]", "", m.group(0))
        try:
            value = float(token)
        except ValueError:
            continue
        key = ("money", round(value, 2))
        if key in seen:
            continue
        seen.add(key)
        found.append({"raw": m.group(0), "value": round(value, 2),
                      "scaled": value, "unit": "currency"})

    return found[:16]


def key_number(claim_numbers: list[dict]) -> dict | None:
    """Pick the most salient number in a claim (largest unit/value)."""
    if not claim_numbers:
        return None
    scorable = [n for n in claim_numbers if n["unit"] in (
        "days", "weeks", "months", "years", "hours", "percent", "currency",
        "people", "crore", "lakh", "million", "billion", "temperature",
    )]
    if not scorable:
        return None
    return max(scorable, key=lambda n: abs(n["scaled"]))


def compare_claim_numbers(claim_numbers: list[dict],
                          evidence_numbers: list[dict]) -> dict:
    """Compare claim numbers against evidence numbers.

    Returns ``{"consistent", "mismatch", "checked", "details"}``.
    """
    if not claim_numbers:
        return {"consistent": True, "mismatch": False, "checked": 0, "details": []}
    if not evidence_numbers:
        return {"consistent": False, "mismatch": False, "checked": 0, "details": []}

    checked, mismatches = 0, []
    for cn in claim_numbers:
        if cn["unit"] is None or cn["unit"] not in _UNIT_MAP.values():
            continue
        candidates = [en for en in evidence_numbers if en["unit"] == cn["unit"]]
        if not candidates:
            continue
        checked += 1
        best = min(candidates, key=lambda en: abs(en["scaled"] - cn["scaled"]))
        diff = abs(best["scaled"] - cn["scaled"])
        max_abs = max(abs(best["scaled"]), abs(cn["scaled"]), 1.0)
        ratio = diff / max_abs
        match = ratio < 0.02 or diff == 0.0
        mismatches.append({
            "claim": cn["raw"], "evidence": best["raw"],
            "diff": round(diff, 3), "ratio": round(ratio, 3), "match": match,
        })

    return {
        "consistent": not any(not d["match"] for d in mismatches),
        "mismatch": any(not d["match"] for d in mismatches),
        "checked": checked,
        "details": mismatches[:6],
    }