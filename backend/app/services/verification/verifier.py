"""End-to-end claim verification orchestrator.

Pipeline: language detection -> text cleaning -> claim decomposition ->
claim-type classification -> ML signal -> evidence retrieval ->
source trust scoring -> relevance/relation analysis -> temporal check ->
cross-source check -> verdict + calibrated confidence + explanation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import evidence as evidence_service
from . import temporal as temporal_service
from .claims import classify_claim_type, decompose_claims
from .language import detect_language
from .numerical import extract_numbers
from .relevance import classify_evidence, significance_of
from .scoring import combine_overall, verdict_for_claim
from .sources import score_source

logger = logging.getLogger("truthlens.verifier")

NOW_ISO = datetime.now(timezone.utc)


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


def _classify_one(claim: dict, ml: dict | None) -> dict:
    """Full evidence check for one atomic claim."""
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
        # Knowledge-base entries are curated refutations/supports -> boost relevance.
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

    # --- cross-source independence --------------------------------------------
    support_domains = {e["domain"] for e in classified if e["relation"] == "SUPPORTS" if e.get("domain")}
    contra_domains = {e["domain"] for e in classified if e["relation"] == "CONTRADICTS" if e.get("domain")}

    verdict = verdict_for_claim(claim, ml, classified)
    counts = verdict.get("signal", {})
    return {
        **claim,
        "temporal_reference": temporal_ref,
        "numbers": extract_numbers(claim_text)[:6],
        "ml": ml,
        "verdict": verdict["verdict"],
        "confidence": verdict["confidence"],
        "confidence_label": verdict["confidence_label"],
        "reason": verdict["reason"],
        "evidence": classified,
        "cross_source": {
            "distinct_support_domains": len(support_domains),
            "distinct_contradict_domains": len(contra_domains),
            "independent_support": len(support_domains) >= 2,
            "independent_contradiction": len(contra_domains) >= 2,
        },
        "supporting_counts": counts.get("supporting_counts", 0),
        "contradicting_counts": counts.get("contradicting_counts", 0),
    }


def verify_text(text: str | None, headline: str | None = None,
                language_mode: str = "auto") -> dict:
    """Run the full verification pipeline and return a JSON-safe report."""
    full_text = " ".join(filter(None, [headline, text])).strip() or (headline or "")
    lang = detect_language(full_text, language_mode)

    claims = decompose_claims(full_text)
    for claim in claims:
        claim["_language"] = lang["code"]

    claim_results = []
    for claim in claims:
        ml = _ml_signal(claim["text"])
        result = _classify_one(claim, ml)
        claim_results.append(result)

    overall = combine_overall(claim_results)

    # --- evidence matrix (flat table for the UI) ------------------------------
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

    pipeline = {
        "language_detected": lang["label"],
        "claims_extracted": len(claims),
        "evidence_items": sum(len(cr["evidence"]) for cr in claim_results),
        "sources_used": retrieved_sources,
        "live_evidence_used": live_used,
        "model_used_as": "secondary-signal",
    }

    return {
        "status": "completed",
        "language": lang,
        "timestamp": NOW_ISO.isoformat(),
        "pipeline": pipeline,
        "claims": claim_results,
        "overall": overall,
        "evidence_matrix": evidence_matrix,
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