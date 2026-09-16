"""Final verdict engine: REAL / FALSE / UNVERIFIED with calibrated confidence.

The neural network classifier is only one input. The *primary* signal is
evidence: strong credible support -> REAL, strong credible contradiction ->
FALSE, otherwise UNVERIFIED. The network is a small agree/disagree modifier
only and can never override strong contradictory evidence.
"""

from __future__ import annotations

import statistics

from .relevance import RELATIONS

VERDICT_REAL = "REAL"
VERDICT_FALSE = "FALSE"
VERDICT_UNVERIFIED = "UNVERIFIED"

#: Total *significance* mass (support or contradiction) needed for a confident
#: verdict. Roughly one strong, highly-relevant, credible source.
STRONG_THRESHOLD = 0.85
WEAK_THRESHOLD = 0.30


def _balanced_sigmoid(x: float) -> float:
    return x / (1.0 + abs(x))


def confidence_from_strength(support: float, contradict: float,
                             verdict: str, ml_conf: float | None,
                             contradictory_mass: float | None = None) -> float:
    """Map evidence strength + ML agreement onto a calibrated 0..1 confidence."""
    if verdict == VERDICT_REAL:
        conf = 0.52 + 0.46 * min(1.0, support / STRONG_THRESHOLD)
    elif verdict == VERDICT_FALSE:
        conf = 0.52 + 0.46 * min(1.0, abs(contradict) / STRONG_THRESHOLD)
    else:  # UNVERIFIED
        conflict = 0.5 if contradictory_mass is not None else 0.0
        conf = 0.35 + 0.18 * conflict + 0.08 * _balanced_sigmoid(support - contradict)
    # ML is a tie-breaker only.
    if ml_conf is not None:
        agrees = (verdict == VERDICT_REAL and ml_conf >= 0.9) or \
            (verdict == VERDICT_FALSE and ml_conf >= 0.9)
        conf += 0.03 if agrees else 0.0
    return max(0.05, min(0.98, conf))


def verdict_for_claim(claim: dict, ml: dict, classified_evidence: list[dict]) -> dict:
    """Compute the verdict and confidence for a single atomic claim.

    ``classified_evidence`` is a list of dicts already containing the evidence
    item plus its relation/relevance/source-score/temporal fields.
    """
    text = claim.get("text", "")
    support = sum(e["significance"] for e in classified_evidence if e["relation"] == "SUPPORTS")
    contradict = sum(e["significance"] for e in classified_evidence if e["relation"] == "CONTRADICTS")

    ml_conf = (ml or {}).get("confidence")

    # --- Special forms ------------------------------------------------------
    if claim.get("opinion"):
        return {
            "verdict": VERDICT_UNVERIFIED,
            "confidence": 0.5,
            "confidence_label": "Subjective statement",
            "reason": ("This statement is subjective (an opinion or value judgement) and "
                       "is not the kind of claim that can be verified as fact."),
            "signal": {"support": 0.0, "contradict": 0.0,
                       "supporting_counts": 0, "contradicting_counts": 0},
        }

    if claim.get("prediction") and not classified_evidence:
        return {
            "verdict": VERDICT_UNVERIFIED,
            "confidence": 0.5,
            "confidence_label": "Prediction",
            "reason": ("This is a prediction about the future. Predictions cannot be "
                       "verified as fact until the event occurs or is officially decided."),
            "signal": {"support": 0.0, "contradict": 0.0,
                       "supporting_counts": 0, "contradicting_counts": 0},
        }

    n_support = sum(1 for e in classified_evidence if e["relation"] == "SUPPORTS")
    n_contradict = sum(1 for e in classified_evidence if e["relation"] == "CONTRADICTS")

    # --- No meaningful evidence ---------------------------------------------
    if not classified_evidence:
        return {
            "verdict": VERDICT_UNVERIFIED,
            "confidence": round(confidence_from_strength(0.0, 0.0, VERDICT_UNVERIFIED, ml_conf), 2),
            "confidence_label": "No live evidence retrieved",
            "reason": ("No supporting or contradicting evidence could be retrieved from "
                       "the available sources. Absence of evidence is not proof the claim "
                       "is false - it simply cannot be verified right now."),
            "signal": {"support": 0.0, "contradict": 0.0,
                       "supporting_counts": 0, "contradicting_counts": 0},
        }

    only_neutral = not n_support and not n_contradict
    if only_neutral:
        return {
            "verdict": VERDICT_UNVERIFIED,
            "confidence": round(confidence_from_strength(0.0, 0.0, VERDICT_UNVERIFIED, ml_conf), 2),
            "confidence_label": "Evidence not specific",
            "reason": ("Retrieved sources discuss related topics but none directly supports "
                       "or contradicts this specific claim."),
            "signal": {"support": support, "contradict": contradict,
                       "supporting_counts": 0, "contradicting_counts": 0},
        }

    # --- Evidence-driven decision -------------------------------------------
    quality_support = sum(max(e["source_score"], 0.0) for e in classified_evidence if e["relation"] == "SUPPORTS")
    quality_contra = sum(max(e["source_score"], 0.0) for e in classified_evidence if e["relation"] == "CONTRADICTS")

    verdict = VERDICT_UNVERIFIED
    reason = ""

    if contradict >= STRONG_THRESHOLD and abs(contradict) > support * 1.15:
        verdict = VERDICT_FALSE
        reason = _false_reason(claim, classified_evidence)
    elif (support >= STRONG_THRESHOLD and contradict == 0.0) or (
            support >= STRONG_THRESHOLD and support > abs(contradict) * 1.5 and n_contradict == 0):
        verdict = VERDICT_REAL
        reason = _real_reason(claim, classified_evidence)
    elif support > 0 and contradict > 0:
        if support > abs(contradict) * 2.0:
            verdict = VERDICT_REAL
            reason = _real_reason(claim, classified_evidence)
        elif abs(contradict) > support * 2.0:
            verdict = VERDICT_FALSE
            reason = _false_reason(claim, classified_evidence)
        else:
            reason = ("Evidence is conflicting: some sources support the claim while "
                      "others contradict it. Too uncertain for a firm verdict.")
    elif support >= WEAK_THRESHOLD:
        verdict = VERDICT_REAL
        reason = _real_reason(claim, classified_evidence)
    elif abs(contradict) >= WEAK_THRESHOLD:
        verdict = VERDICT_FALSE
        reason = _false_reason(claim, classified_evidence)
    else:
        reason = ("Available evidence is too weak or vague to confirm or refute "
                  "this claim.")

    conf = confidence_from_strength(support, contradict, verdict, ml_conf,
                                    contradictory_mass=min(support, abs(contradict))
                                    if verdict == VERDICT_UNVERIFIED else None)

    label = _confidence_label(conf)
    if verdict == VERDICT_UNVERIFIED and not (n_support and n_contradict):
        label = "Evidence insufficient"

    return {
        "verdict": verdict,
        "confidence": round(conf, 2),
        "confidence_label": label,
        "reason": reason,
        "signal": {
            "support": round(support, 3),
            "contradict": round(contradict, 3),
            "supporting_counts": n_support,
            "contradicting_counts": n_contradict,
            "support_source_quality": round(quality_support, 2),
            "contradict_source_quality": round(quality_contra, 2),
        },
    }


AI_VERDICT_MAP = {
    "SUPPORT": VERDICT_REAL,
    "CONTRADICT": VERDICT_FALSE,
    "INSUFFICIENT": VERDICT_UNVERIFIED,
}


def final_decision_for_claim(
    claim: dict,
    ml: dict | None,
    classified_evidence: list[dict],
    cross_source: dict | None,
    ai1: dict | None,
    ai2: dict | None,
) -> dict:
    """Multi-stage final decision for one atomic claim.

    Priority order (ML is the *lowest* authority - it never decides alone):

    1. strong direct credible evidence (with source independence)
    2. AI evidence analysis #1 (+ adversarial review #2 when they agree)
    3. ML prediction is examined for conflict but never overrides strong facts

    Degrades cleanly: when AI stages are unavailable the result is exactly the
    evidence-driven verdict (``verdict_for_claim``), preserving offline use.
    """
    base = verdict_for_claim(claim, ml, classified_evidence)
    verdict = base["verdict"]
    confidence = base["confidence"]
    label = base["confidence_label"]
    reason = base["reason"]
    authority = "none"
    conflicts: list[str] = []

    signal = base.get("signal", {})
    support = signal.get("support", 0.0)
    contradict = signal.get("contradict", 0.0)
    n_support = signal.get("supporting_counts", 0)
    n_contradict = signal.get("contradicting_counts", 0)
    strong = support >= STRONG_THRESHOLD or abs(contradict) >= STRONG_THRESHOLD
    cross_source = cross_source or {}
    independent = bool(
        cross_source.get("independent_support") or cross_source.get("independent_contradiction")
    )

    ai_decision = (ai1 or {}).get("decision") if (ai1 or {}).get("available") else None
    ai_review = (ai2 or {}).get("verdict") if (ai2 or {}).get("available") else None
    ai_verdict = AI_VERDICT_MAP.get(ai_decision) if ai_decision else None
    ai_review_verdict = AI_VERDICT_MAP.get(ai_review) if ai_review else None
    ai_agree = bool(ai_decision and ai_review and ai_decision == ai_review)

    ml_pred = (ml or {}).get("prediction")

    if ml_pred in (VERDICT_REAL, VERDICT_FALSE) and verdict in (VERDICT_REAL, VERDICT_FALSE) \
            and ml_pred != verdict:
        conflicts.append(
            f"Neural network leans {ml_pred} while evidence/AI suggest {verdict} - "
            "disagreement flagged for investigation."
        )
    if ai_verdict and verdict in (VERDICT_REAL, VERDICT_FALSE) and ai_verdict != verdict:
        conflicts.append(
            f"The AI analysis leaned {ai_verdict} but the confirmed verdict is {verdict}."
        )
    if ai_decision and ai_review and ai_decision != ai_review:
        conflicts.append(
            "The AI reviewer disagreed with AI analysis #1; evidence was used to resolve it."
        )

    def _ai_confidence() -> float:
        confs = [(ai1 or {}).get("confidence", 0.5)]
        if ai_agree:
            confs.append((ai2 or {}).get("confidence", 0.5))
        return min(0.95, 0.55 + 0.38 * (sum(confs) / len(confs)))

    if verdict in (VERDICT_REAL, VERDICT_FALSE):
        if strong:
            authority = "evidence"
            if independent:
                confidence = min(0.98, confidence + 0.02)
            if ai_verdict == verdict:
                authority = "evidence+ai"
                confidence = min(0.98, 0.5 * confidence + 0.5 * _ai_confidence())
        elif ai_verdict is None:
            authority = "evidence"
        elif ai_verdict == verdict:
            authority = "evidence+ai"
            confidence = min(0.95, 0.55 * confidence + 0.45 * _ai_confidence())
        elif ai_verdict != verdict:
            verdict = VERDICT_UNVERIFIED
            confidence = 0.45
            label = "Evidence vs AI conflict"
            authority = "none"
            conflicts.append(
                "Evidence here is weak and the AI analysis disagrees with it; "
                "too uncertain for a firm verdict."
            )
            reason = ("The retrieved evidence is not strong enough to override the "
                      "conflicting AI analysis, so the claim stays unverified pending "
                      "better sources.")
    elif ai_decision:
        if ai_verdict == VERDICT_UNVERIFIED:
            authority = "none"
        elif ai_review is None or ai_agree:
            verdict = ai_verdict
            authority = "ai"
            confidence = _ai_confidence()
            label = _confidence_label(confidence)
            reason = _ai_reason(claim, ai_verdict, ai1, ai2, ai_agree)
        else:
            authority = "none"
            conflicts.append(
                "AI analysis #1 and the reviewer disagreed; the claim stays unverified."
            )
        if verdict == VERDICT_UNVERIFIED and not (n_support or n_contradict) \
                and label == "No live evidence retrieved":
            label = "No evidence retrieved"
    else:
        authority = "none"

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "confidence_label": label,
        "reason": reason,
        "signal": signal,
        "authority": authority,
        "base_verdict": base["verdict"],
        "strong_evidence": strong,
        "independent_sources": independent,
        "conflicts": conflicts,
        "support": round(support, 3),
        "contradict": round(contradict, 3),
    }


def _ai_reason(claim: dict, ai_verdict: str, ai1: dict | None,
               ai2: dict | None, ai_agree: bool) -> str:
    conf_txt = f"{round(((ai1 or {}).get('confidence', 0.5)) * 100)}%"
    if ai_verdict == VERDICT_REAL:
        msg = f"AI analysis of the retrieved evidence supports the claim ({conf_txt} confident)."
    elif ai_verdict == VERDICT_FALSE:
        msg = f"AI analysis of the retrieved evidence contradicts the claim ({conf_txt} confident)."
    else:
        msg = "AI analysis found the retrieved evidence insufficient to decide."
    if ai_agree and ai_verdict != VERDICT_UNVERIFIED:
        msg += " An adversarial reviewer agrees."
    return msg


def _real_reason(claim: dict, classified_evidence: list[dict]) -> str:
    evs = [e for e in classified_evidence if e["relation"] == "SUPPORTS"]
    evs.sort(key=lambda e: e["significance"], reverse=True)
    names = {e.get("source_name") or e.get("domain") for e in evs[:2] if e.get("source_name") or e.get("domain")}
    names = " and ".join(sorted(names)[:2]) or "credible sources"
    return (f"Credible sources support the claim. Notably {names} publish material "
            f"consistent with \"{claim['text'][:140]}\".")


def _false_reason(claim: dict, classified_evidence: list[dict]) -> str:
    evs = [e for e in classified_evidence if e["relation"] == "CONTRADICTS"]
    evs.sort(key=lambda e: e["significance"], reverse=True)
    names = {e.get("source_name") or e.get("domain") for e in evs[:2] if e.get("source_name") or e.get("domain")}
    names = " and ".join(sorted(names)[:2]) or "credible sources"
    return (f"Credible sources directly contradict the claim. {names} report facts "
            f"inconsistent with \"{claim['text'][:140]}\".")


def _confidence_label(conf: float) -> str:
    if conf >= 0.85:
        return "High confidence"
    if conf >= 0.65:
        return "Moderate confidence"
    if conf >= 0.45:
        return "Low confidence"
    return "Very low confidence"


def combine_overall(claim_results: list[dict]) -> dict:
    """Combine per-claim verdicts into an overall REAL/FALSE/UNVERIFIED verdict."""
    if not claim_results:
        return {"verdict": VERDICT_UNVERIFIED, "confidence": 0.4,
                "explanation": "No claim content could be isolated for verification.",
                "mixed": False}

    real = [c for c in claim_results if c["verdict"] == VERDICT_REAL]
    false = [c for c in claim_results if c["verdict"] == VERDICT_FALSE]
    unverified = [c for c in claim_results if c["verdict"] == VERDICT_UNVERIFIED]

    verifiable = real + false
    mixed = bool(real and false)
    vote = 0

    if real and not false:
        verdict = VERDICT_REAL
        vote = len(real)
        conf = statistics.mean(c["confidence"] for c in real)
    elif false and not real:
        verdict = VERDICT_FALSE
        vote = len(false)
        conf = statistics.mean(c["confidence"] for c in false)
    elif mixed:
        real_mass = sum(c["confidence"] for c in real)
        false_mass = sum(c["confidence"] for c in false)
        if len(real) > len(false) or (len(real) == len(false) and real_mass > false_mass):
            verdict = VERDICT_REAL
            vote = len(real)
        elif len(false) > len(real) or (len(false) == len(real) and false_mass > real_mass):
            verdict = VERDICT_FALSE
            vote = len(false)
        else:
            verdict = VERDICT_UNVERIFIED
        conf = statistics.mean(
            [c["confidence"] for c in verifiable] or [0.5]
        ) if verifiable else 0.5
    else:
        # Nothing verifiable at all.
        conf = statistics.mean([c["confidence"] for c in unverified]) if unverified else 0.5
        verdict = VERDICT_UNVERIFIED

    explanation_parts = []
    if mixed:
        explanation_parts.append(
            f"The content mixes verified and refuted claims "
            f"({len(real)} supported, {len(false)} contradicted, "
            f"{len(unverified)} unverifiable)."
        )
    if verdict == VERDICT_REAL:
        explanation_parts.append(
            f"The supported claims outweigh the others "
            f"({vote} verified claim(s) against {len(verifiable) - vote} refuted)."
        )
    elif verdict == VERDICT_FALSE:
        explanation_parts.append(
            f"Credible sources contradict the key claims "
            f"({vote} refuted claim(s))."
        )
    else:
        explanation_parts.append(
            "There is not enough credible evidence to confirm or refute the "
            "claims; some may be opinions or predictions."
        )

    return {
        "verdict": verdict,
        "confidence": round(min(0.98, conf), 2),
        "confidence_label": _confidence_label(min(0.98, conf)),
        "mixed": mixed,
        "counts": {"real": len(real), "false": len(false),
                   "unverified": len(unverified), "total_claims": len(claim_results)},
        "explanation": " ".join(explanation_parts),
    }