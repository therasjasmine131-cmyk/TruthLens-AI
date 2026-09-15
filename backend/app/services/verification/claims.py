"""Atomic claim decomposition and claim-type classification.

Breaks an article into independently-verifiable atomic claims:

    "Schools in Tamil Nadu will remain closed for 15 days because the
     government announced a new holiday."

becomes:

    1. Schools in Tamil Nadu will remain closed for 15 days.
    2. The government announced a new holiday (reason for the closure).

Each claim is classified (government-announcement, weather, health, ...) and
flagged for subjectivity (opinion), futurity (prediction) and negation so the
evidence layer knows *what* to check and *how* to interpret matches.
"""

from __future__ import annotations

import re

from .negation import detect_negation, strip_negation_effect
from .textutil import split_sentences

# ---- connector strategies ---------------------------------------------------

REASON_CONNECTORS = re.compile(
    r"\s(?:because|due\s+to|owing\s+to|as\s+a\s+result\s+of|on\s+account\s+of|"
    r"given\s+that|in\s+light\s+of|as\s+per)\s", re.I)
SINCE_AS_REASON = re.compile(r"\s(?:since|as)\s", re.I)
CONTRAST_CONNECTORS = re.compile(
    r"\s(?:but|however|although|though|whereas|while|yet|despite|"
    r"nevertheless|instead)\s", re.I)
ADDITIVE_CONNECTORS = re.compile(r"\s(?:and|as\s+well\s+as|moreover|"
                                 r"in\s+addition|furthermore|also)\s", re.I)
RELATIVE_CONNECTORS = re.compile(r"[,]?\s(?:which|who|that)\s", re.I)

_VERB_MARKERS = re.compile(
    r"\b(?:will|would|shall|can|could|should|is|are|was|were|has|have|had|"
    r"has\s+been|have\s+been|am|be|announced|decided|said|says|reported|"
    r"confirmed|declared|issued|ordered|introduced|approved|passed|denied|"
    r"comes|rolled\s+out|to\s+be)\b",
    re.I,
)

# ---- opinion markers --------------------------------------------------------

_OPINION_PHRASES = re.compile(
    r"\b(i\s+(think|believe|feel|guess|personally\s+believe|reckon|would\s+say|"
    r"am\s+convinced)|in\s+my\s+(opinion|view)|my\s+opinion|according\s+to\s+me|"
    r"i\s+fee|i\s+suppose|i\s+think\s+it\s+is|i\s+am\s+sure)\b",
    re.I,
)
_OPINION_ADJECTIVES = {
    "best", "worst", "great", "terrible", "amazing", "disgusting", "beautiful",
    "wonderful", "awful", "fantastic", "brilliant", "horrible", "lovely",
    "ugly", "incredible", "excellent", "poor",
}
_OPINION_SUBJECTS = {"this", "that", "it", "he", "she", "they", "the idea",
                     "the plan", "the movie", "the book", "the decision"}

# ---- prediction markers -----------------------------------------------------

_PREDICTION_MODALS = re.compile(
    r"\b(will|shall|would|can|could|might|may|is\s+expected\s+to|is\s+set\s+to|"
    r"plans\s+to|is\s+going\s+to|likely\s+to|expected\s+to|is\s+predicted\s+to|"
    r"is\s+projected\s+to|will\s+see|will\s+lead\s+to|will\s+be\s+able\s+to)\b",
    re.I,
)
_FUTURE_REFERENCE = re.compile(
    r"\b(tomorrow|next\s+week|next\s+month|next\s+year|upcoming|in\s+the\s+future|"
    r"in\s+future|soon|by\s+20(?:2\d|3\d)|this\s+century|in\s+the\s+coming)\b",
    re.I,
)
_ATTRIBUTION = re.compile(
    r"\b(announced|declared|confirmed|according\s+to|reported|said|says|"
    r"announced\s+that|has\s+announced|have\s+announced|decided\s+to|agreed\s+to|"
    r"approved|ordered|issued|introduced|passed)\b",
    re.I,
)
# Speculative verbs that indicate a prediction rather than a reported fact.
_SPECULATIVE_VERBS = re.compile(
    r"\b(win|wins|beat|become|achieve|reach|improve|rise|falls?|grow|develop|"
    r"top|break|produce|make|create|find|discover|lead|dominate|surpass|"
    r"defeat|qualify|claim\s+the|secure)\b",
    re.I,
)

# ---- claim-type keyword lexicons ---------------------------------------------

_CLAIM_TYPE_LEXICON: dict[str, set[str]] = {
    "government-announcement": {
        "government", "govt", "ministry", "minister", "chief minister", "pm",
        "prime minister", "cabinet", "announced", "announcement", "scheme",
        "policy", "order", "notification", "department", "official", "banned",
        "launched", "launch", "parliament", "assembly", "bill", "act",
    },
    "politics": {
        "politics", "political", "politician", "party", "leader", "candidate",
        "campaign", "union", "protest", "rally", "delhi", "loksabha",
    },
    "election": {
        "election", "vote", "voting", "voter", "poll", "polling", "result",
        "seat", "mandate", "ballot", "elected", "win", "won", "winner",
    },
    "weather": {
        "weather", "rain", "rainfall", "flood", "storm", "cyclone", "heat",
        "temperature", "drought", "monsoon", "humidity", "hurricane", "climate",
    },
    "education": {
        "school", "schools", "college", "colleges", "university", "students",
        "student", "exam", "exams", "admission", "syllabus", "holiday",
        "education", "teacher", "board", "semester", "classes", "fees",
    },
    "health": {
        "health", "hospital", "doctor", "disease", "virus", "vaccine",
        "medical", "medicine", "patients", "covid", "death", "deaths", "cure",
        "treatment", "bacteria", "infection", "eyesight",
    },
    "finance": {
        "rbi", "finance", "bank", "banks", "interest", "rate", "inflation",
        "loan", "rupee", "money", "crore", "lakh", "budget", "tax", "gst",
        "stock", "market", "deposit", "account", "salary", "income", "price",
    },
    "business": {
        "business", "company", "companies", "industry", "corporate", "startup",
        "ceo", "revenue", "profit", "trademark", "ipo", "merger", "firm",
    },
    "sports": {
        "sports", "cricket", "football", "match", "team", "player", "world cup",
        "tournament", "league", "medal", "olympics", "championship", "coach",
    },
    "technology": {
        "technology", "tech", "app", "software", "smartphone", "phone",
        "battery", "electric", "gadget", "internet", "satellite", "robot",
        "ai", "computer", "device", "charge", "charging", "device",
    },
    "crime": {
        "crime", "police", "arrest", "arrested", "murder", "theft", "fraud",
        "court", "judge", "robbery", "attack", "scam", "illegal", "accused",
    },
    "public-figure": {
        "chief minister", "minister", "pm", "prime minister", "actor",
        "actress", "star", "celebrity", "politician", "mp", "mla", "singer",
    },
    "science": {
        "science", "scientist", "scientists", "research", "study", "planet",
        "earth", "sun", "moon", "gravity", "ocean", "space", "isro", "nasa",
        "physics", "biology", "chemistry", "molecule", "cell", "energy",
    },
    "law": {
        "law", "act", "rule", "rules", "legal", "court", "constitution",
        "amendment", "regulation", "bill", "order", "right", "rights",
    },
    "statistics": {
        "percent", "percentage", "rate", "count", "figure", "data", "average",
        "rank", "ranking", "statistics", "number", "total", "increase",
        "decrease", "million", "billion", "crore", "lakh",
    },
    "numerical": {
        "days", "months", "years", "week", "hours", "percent", "percentage",
        "rupees", "rs", "₹", "$", "crore", "lakh", "million", "billion",
        "degrees", "km", "kg", "rate", "price",
    },
    "dates": {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december", "today",
        "yesterday", "tomorrow", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "week", "month", "year", "date",
        "deadline",
    },
    "locations": {
        "india", "tamil nadu", "chennai", "delhi", "mumbai", "bengaluru",
        "kerala", "karnataka", "pune", "hyderabad", "state", "district",
        "village", "city", "country", "pacific", "ocean",
    },
}

_TYPE_PRIORITY = [
    "government-announcement", "health", "weather", "finance", "election",
    "education", "science", "crime", "sports", "business", "technology",
    "public-figure", "law", "locations", "dates", "numerical", "politics",
]

_DEFAULT_TYPE = "general-factual"


def classify_claim_type(text: str) -> str:
    """Return the most specific claim category for *text*."""
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9₹$%]+|[.+]+", lowered.lower()))
    scores: dict[str, int] = {}
    for cat, kws in _CLAIM_TYPE_LEXICON.items():
        score = 0
        for kw in kws:
            if kw in {"₹", "$", "%", "."}:
                if kw in lowered:
                    score += 1
            elif " " in kw:
                if kw in lowered:
                    score += 3
            elif kw in words:
                score += 1
        if score:
            scores[cat] = score

    # numerical / locations / dates are often *modifiers* of a deeper topic -
    # only report them as the category when nothing more meaningful matched.
    for weak in ("numerical", "locations", "dates"):
        if weak in scores and len(scores) > 1:
            scores.pop(weak, None)
    if not scores:
        return _DEFAULT_TYPE

    best = max(scores.items(), key=lambda kv: (kv[1], -_TYPE_PRIORITY.index(kv[0]) if kv[0] in _TYPE_PRIORITY else 99))
    return best[0]


def is_opinion(text: str) -> bool:
    """Heuristic check for subjective statements."""
    if _OPINION_PHRASES.search(text):
        return True
    lowered = text.lower().strip()
    tokens = re.findall(r"[a-z]+", lowered)
    subject_word = tokens[0] if tokens else ""
    return subject_word in _OPINION_SUBJECTS and bool({t for t in tokens} & _OPINION_ADJECTIVES)


def is_prediction(text: str) -> bool:
    """A future claim with no attribution is a prediction (not yet verifiable)."""
    if _ATTRIBUTION.search(text):
        return False
    return bool(_PREDICTION_MODALS.search(text)) or bool(_FUTURE_REFERENCE.search(text))


def _subject_of(sentence: str) -> str:
    """Heuristic noun-phrase subject extraction (words before first verb)."""
    m = _VERB_MARKERS.search(sentence)
    if not m:
        return sentence.strip()
    subject = sentence[: m.start()].strip()
    subject = re.sub(r"^(the|a|an|this|that|these|those|my|our|his|her|its)\s+", "", subject)
    subject = re.sub(r"\s+(because|due|owing|and|but|however)$", "", subject)
    return subject.strip()


def _split_clauses(sentence: str) -> list[str]:
    """Split one sentence into atomic claim clauses (keeps reason clauses)."""
    sentence = sentence.strip()
    if not sentence:
        return []
    if is_opinion(sentence) or is_prediction(sentence):
        return [sentence]

    parts = [sentence]
    # First split at reason/contrast/additive connectors; the first piece keeps
    # the sentence subject, later pieces get the subject re-attached.
    for connector in (
        REASON_CONNECTORS, SINCE_AS_REASON, CONTRAST_CONNECTORS,
        ADDITIVE_CONNECTORS, RELATIVE_CONNECTORS,
    ):
        new_parts: list[str] = []
        for part in parts:
            found = connector.search(part)
            if found and found.start() > 8:
                head, tail = part[: found.start()], part[found.end():]
                tail = re.sub(r"^(which|who|that)", " ", tail).strip()
                subject = _subject_of(head)
                has_own_subject = bool(
                    re.search(r"^(?:the|a|an|this|that|these|those|my|our|his|her|its|it|"
                              r"he|she|they|we|you|all|every|each|any|no|government|india|"
                              r"tamil\s+nadu|there)\b",
                              tail, re.I)
                )
                if len(token_words(tail)) < 4:
                    # tiny reason phrase ("because of heavy rainfall") is not a
                    # separate claim - keep it attached to the main claim.
                    new_parts.append(part)
                    continue
                if tail and not has_own_subject:
                    tail = f"{subject} {tail}".strip()
                new_parts.append(head.strip())
                if tail.strip().strip("., "):
                    new_parts.append(tail.strip())
            else:
                new_parts.append(part)
        parts = new_parts

    result = []
    for part in parts:
        part = re.sub(r"[\s,;:]+$", "", part).strip()
        part = re.sub(r"^[\s,;:]+", "", part).strip()
        part = part.rstrip(".")
        part = part.rstrip(" ")
        if len(token_words(part)) >= 4:
            result.append(part + ".")
    return result


def token_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9]+", text) if w.lower() not in {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    }]


def decompose_claims(text: str | None) -> list[dict]:
    """Decompose *text* into atomic claims with metadata.

    Each returned claim::

        {"text", "type", "opinion", "prediction", "negation",
         "negation_patterns", "index"}
    """
    raw_claims: list[str] = []
    for sentence in split_sentences(text or ""):
        sentence_reported = bool(_ATTRIBUTION.search(sentence))
        for clause in _split_clauses(sentence):
            if clause:
                raw_claims.append((clause, sentence_reported))

    # Deduplicate near-identical claims.
    unique: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for claim, reported in raw_claims:
        key = re.sub(r"[^a-z0-9 ]+", "", claim.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((claim, reported))

    claims: list[dict] = []
    for i, (claim, reported) in enumerate(unique):
        negation = detect_negation(claim)
        claims.append({
            "index": i + 1,
            "text": claim,
            "type": classify_claim_type(claim),
            "opinion": is_opinion(claim),
            # A future-tense claim that is *attributed to an authority* is a
            # reported fact, not a speculative prediction.
            "prediction": is_prediction(claim) and not reported,
            "negation": negation["negated"],
            "negation_patterns": negation["patterns"],
        })
    return claims