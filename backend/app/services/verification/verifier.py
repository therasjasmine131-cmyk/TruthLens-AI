"""End-to-end claim verification orchestrator.

Multi-stage pipeline (each stage logged; ML is the LOWEST authority):

    user input -> preprocessing -> ML classifier -> claim decomposition ->
    atomic claim extraction -> evidence retrieval -> AI analysis #1 ->
    evidence-based reasoning -> AI critic / review #2 -> consistency check ->
    final decision engine -> overall verdict

Every stage degrades gracefully: if the AI or evidence services are
unavailable the pipeline downgrades confidence and falls back to
evidence-driven verdicts (and UNVERIFIED where no evidence exists) exactly as
an offline build would.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from datetime import datetime, timezone

from . import ai_stage
from . import evidence as evidence_service
from . import temporal as temporal_service
from .claims import decompose_claims
from .language import detect_language
from .numerical import extract_numbers
from .relevance import classify_evidence, significance_of
from .scoring import combine_overall, final_decision_for_claim, verdict_for_claim
from .sources import score_source

logger = logging.getLogger("truthlens.verifier")

NOW_ISO = datetime.now(timezone.utc)

MAX_AI_CLAIMS = int(os.environ.get("AI_CLAIMS_LIMIT", "4"))


def _ml_signal(claim_text: str):
    """Query the local classifier for one claim (best-effort)."""
    try:
        from ...ml.model_manager import model_manager
        if not model_manager.ready:
            return None
        p_real, p_fake = model_manager.predict_proba(claim_text)
        prediction = "REAL" if p_real >= p_fake else "FALSE"
        return {
            "prediction": prediction,
            "confidence": round(max(p_real, p_fake), 3),
            "probabilities": {"real": round(p_real, 3), "fake": round(p_fake, 3)},
        }
    except Exception:  # noqa: BLE001 - ML failure must not break verification
        logger.exception("ML signal failed for a claim")
        return None


def _evidence_signal(claim: dict, ml: dict | None) -> dict:
    """Evidence retrieval, classification and evidence-side verdict for a claim."""
    claim_text = claim["text"]
    query = evidence_service.build_query(claim_text, claim["_language"])
    temporal_ref = temporal_service.classify_reference(claim_text)

    raw_items = evidence_service.retrieve_evidence_claim(claim, query, claim["_language"])
    classified: list[dict] = []
    for item in raw_items:
        relation = classify_evidence(claim, item)
        if relation["relation"] == "NEUTRAL" and relation["relevance"] < 0.18:
            continue  # not worth listing or scoring
        source = score_source(
            item.get("source_name"), item.get("url"),
            published_date=item.get("date"),
            retrieved_from=item.get("retrieved_from"),
            title=item.get("title"),
        )
        relevance = relation["relevance"]
        if item.get("retrieved_from") == "knowledge-base" and item.get("relation_hint"):
            relevance = max(relevance, 0.75)
        temporal = temporal_service.check_temporal_match(temporal_ref, item.get("date"))
        significance = significance_of(relation["relation"], relevance, float(source["score"]))
        if relation["relation"] != "NEUTRAL" and temporal["status"] == "stale":
            significance *= 0.5  # de-prioritize outdated evidence
        classified.append({
            **item,
            "relation": relation["relation"],
            "relevance_score": relevance,
            "relation_score": relation["score"],
            "reasons": relation["reasons"],
            "numerical": relation["numerical"],
            "source_score": source["score"],
            "source_tier": source["tier"],
            "source_reasons": source["reasons"],
            "temporal": temporal,
            "significance": round(significance, 4),
        })

    support_domains = {e["domain"] for e in classified if e["relation"] == "SUPPORTS" if e.get("domain")}
    contra_domains = {e["domain"] for e in classified if e["relation"] == "CONTRADICTS" if e.get("domain")}
    cross_source = {
        "distinct_support_domains": len(support_domains),
        "distinct_contradict_domains": len(contra_domains),
        "independent_support": len(support_domains) >= 2,
        "independent_contradiction": len(contra_domains) >= 2,
    }

    base = verdict_for_claim(claim, ml, classified)
    counts = base.get("signal", {})
    return {
        "classified": classified,
        "cross_source": cross_source,
        "base": base,
        "temporal_reference": temporal_ref,
        "numbers": extract_numbers(claim_text)[:6],
        "supporting_counts": counts.get("supporting_counts", 0),
        "contradicting_counts": counts.get("contradicting_counts", 0),
    }


def _source_credibility(classified: list[dict]) -> dict:
    tiers: dict[str, int] = {}
    domains: set[str] = set()
    qualities: list[float] = []
    for e in classified:
        tier = e.get("source_tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
        if e.get("domain"):
            domains.add(e["domain"])
        score = e.get("source_score")
        if isinstance(score, (int, float)):
            qualities.append(float(score))
    return {
        "items": len(classified),
        "distinct_domains": len(domains),
        "avg_source_quality": round(sum(qualities) / len(qualities), 2) if qualities else 0.0,
        "tiers": tiers,
        "independent": len(domains) >= 2,
    }


def _finalize_claim(claim: dict, evidence_result: dict, ai1: dict | None,
                    ai2: dict | None, include_debug: bool) -> dict:
    """Merge evidence + AI stages into the final per-claim record."""
    ml = claim.get("_ml")
    classified = evidence_result["classified"]
    cross_source = evidence_result["cross_source"]
    final = final_decision_for_claim(
        claim, ml, classified, cross_source, ai1, ai2,
    )

    supporting = [e for e in classified if e["relation"] == "SUPPORTS"]
    contradicting = [e for e in classified if e["relation"] == "CONTRADICTS"]

    stages = {
        "ML_RESULT": {
            "prediction": (ml or {}).get("prediction"),
            "confidence": (ml or {}).get("confidence"),
        },
        "EVIDENCE_RESULT": {
            "verdict": evidence_result["base"]["verdict"],
            "confidence": evidence_result["base"]["confidence"],
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "independent_sources": cross_source.get("independent_support")
            or cross_source.get("independent_contradiction"),
        },
        "AI_RESULT_1": ai1,
        "AI_REVIEW_RESULT": ai2,
        "FINAL_RESULT": {
            "verdict": final["verdict"],
            "confidence": final["confidence"],
            "authority": final["authority"],
        },
    }

    record = {
        "index": claim["index"],
        "text": claim["text"],
        "original_claim": claim["text"],
        "type": claim.get("type", "general-factual"),
        "opinion": claim.get("opinion", False),
        "prediction": claim.get("prediction", False),
        "negation": claim.get("negation", False),
        "negation_patterns": claim.get("negation_patterns", []),
        "temporal_reference": evidence_result["temporal_reference"],
        "numbers": evidence_result["numbers"],
        "ml": ml,
        "ml_prediction": (ml or {}).get("prediction"),
        "ml_confidence": (ml or {}).get("confidence"),
        "evidence": classified,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "source_credibility": _source_credibility(classified),
        "cross_source": cross_source,
        "supporting_counts": len(supporting),
        "contradicting_counts": len(contradicting),
        "ai_analysis_1": ai1,
        "ai_review": ai2,
        "conflicts": final["conflicts"],
        "verdict": final["verdict"],
        "final_verdict": final["verdict"],
        "confidence": final["confidence"],
        "final_confidence": final["confidence"],
        "confidence_label": final["confidence_label"],
        "final_authority": final["authority"],
        "reason": final["reason"],
        "stages": stages if include_debug else None,
    }
    return record


def _run_ai_stages(claims: list[dict], evidence_results: list[dict],
                   language: dict) -> tuple[dict, dict]:
    """Run AI analysis #1 + review #2 for a bounded set of claims, in parallel."""
    ai1_by_idx: dict[int, dict] = {}
    ai2_by_idx: dict[int, dict] = {}
    if not ai_stage.available():
        return ai1_by_idx, ai2_by_idx

    limit = min(MAX_AI_CLAIMS, len(claims))
    lang_code = language.get("code", "english")

    def _job(idx: int):
        claim = claims[idx]
        classified = evidence_results[idx]["classified"]
        ai1 = ai_stage.analyze_claim_ai(claim["text"], classified, lang_code)
        ai2 = ai_stage.review_claim_ai(claim["text"], classified, ai1, lang_code) if ai1 else None
        return idx, ai1, ai2

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_job, i) for i in range(limit)]
            for future in futures:
                idx, ai1, ai2 = future.result()
                ai1_by_idx[idx] = ai1
                ai2_by_idx[idx] = ai2
    except Exception:  # noqa: BLE001 - AI stages must never crash verification
        logger.exception("AI reasoning stages degraded")
    return ai1_by_idx, ai2_by_idx


def verify_text(text: str | None, headline: str | None = None,
                language_mode: str = "auto", include_debug: bool = False) -> dict:
    """Run the full multi-stage verification pipeline."""
    full_text = " ".join(filter(None, [headline, text])).strip() or (headline or "")
    lang = detect_language(full_text, language_mode)

    claims = decompose_claims(full_text)
    for claim in claims:
        claim["_language"] = lang["code"]

    evidence_results = []
    for claim in claims:
        ml = _ml_signal(claim["text"])
        claim["_ml"] = ml
        evidence_results.append(_evidence_signal(claim, ml))

    ai1_by_idx, ai2_by_idx = _run_ai_stages(claims, evidence_results, lang)

    claim_results = []
    for i, claim in enumerate(claims):
        claim_results.append(_finalize_claim(
            claim, evidence_results[i],
            ai1_by_idx.get(i), ai2_by_idx.get(i), include_debug,
        ))

    overall = combine_overall(claim_results)

    evidence_matrix = []
    for cr in claim_results:
        for e in cr["evidence"]:
            evidence_matrix.append({
                "claim": cr["text"],
                "claim_verdict": cr["verdict"],
                "evidence_title": e.get("title") or (e.get("snippet") or "")[:120],
                "source": e.get("source_name"),
                "domain": e.get("domain"),
                "url": e.get("url"),
                "date": e.get("date"),
                "relation": e["relation"],
                "relevance": e["relevance_score"],
                "source_quality": e["source_score"],
                "source_tier": e["source_tier"],
                "source_reasons": e.get("source_reasons", []),
                "retrieved_from": e.get("retrieved_from"),
            })

    retrieved_sources = sorted({
        e.get("retrieved_from", "unknown")
        for cr in claim_results for e in cr["evidence"]
    })
    live_used = bool({"wikipedia", "newsapi", "factcheck", "gemini"} & set(retrieved_sources))

    ai_count = sum(1 for k, v in ai1_by_idx.items() if v)
    review_count = sum(1 for k, v in ai2_by_idx.items() if v)

    pipeline = {
        "language_detected": lang["label"],
        "claims_extracted": len(claims),
        "evidence_items": sum(len(cr["evidence"]) for cr in claim_results),
        "sources_used": retrieved_sources,
        "live_evidence_used": live_used,
        "model_used_as": "secondary-signal",
        "ai_used": ai_count > 0,
        "ai_claims_analyzed": ai_count,
        "ai_reviews_completed": review_count,
    }

    stages = {
        "PIPELINE": [
            "USER INPUT", "PREPROCESSING", "ML CLASSIFIER",
            "CLAIM EXTRACTION", "EVIDENCE RETRIEVAL", "AI ANALYSIS #1",
            "AI REVIEW #2", "FINAL DECISION",
        ],
        "claims_analyzed": len(claim_results),
        "ai_available": ai_stage.available() if ai_count else False,
        "ai_claims": ai_count,
        "agreement": {
            "authorities": {},
        },
    }
    for cr in claim_results:
        auth = cr["final_authority"] or "none"
        stages["agreement"]["authorities"][auth] = \
            stages["agreement"]["authorities"].get(auth, 0) + 1
    if ai_count:
        agreements = sum(1 for k in ai1_by_idx if ai1_by_idx[k]
                         and ai2_by_idx.get(k)
                         and ai1_by_idx[k].get("decision") == ai2_by_idx[k].get("verdict"))
        stages["agreement"]["ai_first_vs_reviewer_agree"] = agreements
        stages["agreement"]["ai_first_vs_reviewer_total"] = review_count

    return {
        "status": "completed",
        "language": lang,
        "timestamp": NOW_ISO.isoformat(),
        "pipeline": pipeline,
        "claims": claim_results,
        "overall": overall,
        "evidence_matrix": evidence_matrix,
        "stages": stages if include_debug else None,
        "notes": {
            "model_note": (
                "The local ML model is used as a secondary stylistic signal only. "
                "Verdicts are driven by retrieved evidence and source credibility."
            ),
            "no_evidence_note": (
                "No live evidence sources were configured or reachable. Results for "
                "claims without a knowledge-base match are based on the local model "
                "and should not be treated as confirmed."
            ) if not live_used else None,
            "live_evidence_note": (
                "Live sources used for evidence: "
                + ", ".join(sorted(retrieved_sources)) + "."
            ) if live_used else None,
            "ai_note": (
                "Free-tier AI analysis (#1) with an adversarial reviewer (#2) ran on "
                f"{ai_count} claim(s). AI verdicts never override strong, independent, "
                "credible evidence."
            ) if ai_count else (
                "No AI analysis ran: GEMINI_API_KEY is not configured. Set it to "
                "enable AI analysis #1 and the AI critic (#2)."
            ),
        },
    }


def verify_with_ml_override(text: str | None, headline: str | None = None,
                            language_mode: str = "auto",
                            ml_result: dict | None = None) -> dict:
    """Run the pipeline and attach the article-level ML prediction."""
    report = verify_text(text, headline, language_mode)
    if ml_result:
        report["ml_article"] = ml_result
    return report