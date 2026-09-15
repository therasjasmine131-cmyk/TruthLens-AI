"""Evidence relevance scoring and SUPPORTS / CONTRADICTS / NEUTRAL labelling.

Avoids bare keyword matching: combines token Jaccard, character similarity,
entity overlap and number agreement, and is negation-aware so that
*"the government did NOT announce X"* does not count as support for
*"the government announced X"*.
"""

from __future__ import annotations

from .negation import detect_negation, strip_negation_effect
from .numerical import compare_claim_numbers, extract_numbers
from .textutil import clean_text, combined_similarity, jaccard, tokenize

RELATIONS = ("SUPPORTS", "CONTRADICTS", "NEUTRAL")

# Strength of the contradictory trigger when claim == inverse of evidence.
NEGATION_SIMILARITY_THRESHOLD = 0.55
RELEVANCE_MIN = 0.22


def relation_strength(relation: str) -> float:
    return {"SUPPORTS": 1.0, "CONTRADICTS": -1.0, "NEUTRAL": 0.0}[relation]


def _agree_on_numbers(claim: str, evidence: str) -> dict:
    cn = extract_numbers(claim)
    en = extract_numbers(evidence)
    return compare_claim_numbers(cn, en)


def classify_evidence(claim: dict, evidence: dict) -> dict:
    """Classify one evidence item against one atomic claim.

    Returns ``{"relation", "relevance", "score", "reasons", "numerical"}``.
    """
    claim_text = claim.get("text", "")
    evidence_text = " ".join(filter(None, [evidence.get("title"), evidence.get("text")]))
    if not evidence_text.strip():
        evidence_text = evidence.get("snippet", "")

    reasons: list[str] = []

    # 1) relevance = content overlap (token + chars + entities).
    sim = combined_similarity(evidence_text, claim_text)
    entity_boost = _entity_overlap(claim_text, evidence_text)
    relevance = min(1.0, sim + entity_boost * 0.15)
    # Knowledge-base items carry the exact matched pattern - very precise.
    if evidence.get("pattern"):
        pattern_sim = combined_similarity(claim_text, evidence["pattern"])
        relevance = max(relevance, min(1.0, pattern_sim * 0.95))

    # 2) explicit relation hints from the retrieval layer (knowledge base / Gemini).
    hint = evidence.get("relation_hint")
    if hint:
        hint = {"SUPPORT": "SUPPORTS", "CONTRADICT": "CONTRADICTS",
                "NOT_ADDRESSED": "NEUTRAL"}.get(hint, hint)

    if relevance < RELEVANCE_MIN:
        return {
            "relation": "NEUTRAL",
            "relevance": round(relevance, 3),
            "score": 0.0,
            "reasons": ["Evidence shares too little content with the claim."],
            "numerical": _agree_on_numbers(claim_text, evidence_text),
        }

    # 3) Negation-aware contradiction.
    claim_neg = detect_negation(claim_text)
    ev_neg = detect_negation(evidence_text)
    claim_core = strip_negation_effect(claim_text)
    evidence_core = strip_negation_effect(evidence_text)
    core_sim = combined_similarity(evidence_core, claim_core)

    negation_conflict = (claim_neg["negated"] != ev_neg["negated"]) and core_sim >= NEGATION_SIMILARITY_THRESHOLD
    if negation_conflict:
        return {
            "relation": "CONTRADICTS",
            "relevance": round(relevance, 3),
            "score": round(min(1.0, core_sim), 3),
            "reasons": ["Claim and evidence have opposite polarity on the same assertion."],
            "numerical": _agree_on_numbers(claim_text, evidence_text),
        }

    # 4) Numerical comparison (only when the claim asserts a specific number).
    numeric = _agree_on_numbers(claim_text, evidence_text)
    if numeric["checked"] and numeric["mismatch"]:
        return {
            "relation": "CONTRADICTS",
            "relevance": round(relevance, 3),
            "score": round(min(1.0, core_sim + 0.15), 3),
            "reasons": [f"Claim number {n['claim']} differs from evidence {n['evidence']}."
                        for n in numeric["details"][:2]],
            "numerical": numeric,
        }

    # 5) Decide SUPPORTS vs NEUTRAL from similarity.
    if hint in ("SUPPORTS",):
        relation = "SUPPORTS"
        reasons.append("Source directly supports the claim (reference lookup).")
    elif hint == "CONTRADICTS":
        relation = "CONTRADICTS"
        reasons.append("Source directly refutes the claim (reference lookup).")
    elif core_sim >= 0.6:
        relation = "SUPPORTS"
        reasons.append("Evidence content closely matches the claim.")
    elif core_sim >= 0.35:
        related = jaccard(evidence_core, claim_core)
        if related >= 0.12:
            relation = "SUPPORTS"
            reasons.append("Evidence discusses the same subject consistently.")
        else:
            relation = "NEUTRAL"
            reasons.append("Evidence covers the same topic but not specifically this claim.")
    else:
        relation = "NEUTRAL"
        reasons.append("Evidence does not specifically address this claim.")

    return {
        "relation": relation,
        "relevance": round(relevance, 3),
        "score": round(core_sim, 3),
        "reasons": reasons[:3],
        "numerical": numeric,
    }


def _entity_overlap(claim: str, evidence: str) -> float:
    from .entities import extract_entities
    ce = extract_entities(claim)
    ee = extract_entities(evidence)
    names = lambda d: {n.lower() for n in d.get("organizations", []) + d.get("locations", []) + d.get("people", [])}
    cn, en = names(ce), names(ee)
    if not cn or not en:
        return 0.0
    return len(cn & en) / max(len(cn), 1)


def significance_of(relation: str, relevance: float, source_score: float) -> float:
    """Combine relation, relevance and source quality into one weighted signal."""
    if relation == "NEUTRAL":
        return 0.0
    magnitude = relation_strength(relation)
    relevance_w = relevance
    source_w = max(0.0, min(1.0, (source_score - 4.0) / 6.0))
    return magnitude * (0.6 * relevance_w + 0.4 * source_w)