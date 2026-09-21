"""AI reasoning stages for the multi-stage verification pipeline.

Two independent AI providers sit between evidence retrieval and the final
decision engine. Using different providers ensures genuine cross-checking:

* ``analyze_claim_ai`` - AI analysis #1 (Gemini): given the claim and the
  retrieved evidence, decide SUPPORT / CONTRADICT / INSUFFICIENT using ONLY
  the evidence.
* ``review_claim_ai`` - AI analysis #2 (BazaarLink/DeepSeek): an adversarial
  reviewer that actively looks for errors in AI #1 (mis-stated claim,
  off-topic evidence, staleness, hallucination, independence, credibility...).

Both degrade to ``None`` whenever their respective provider is unavailable,
so the rest of the pipeline behaves exactly as before (evidence-driven
verdicts, ML tie-breaker).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date

from ..web_search import build_query, search_web

logger = logging.getLogger("truthlens.ai_stage")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

BAZAARLINK_BASE_URL = "https://api.bazaarlink.ai/v1"
BAZAARLINK_MODEL = os.environ.get("BAZAARLINK_MODEL", "deepseek/deepseek-v4-flash-0731")
BAZAARLINK_FREE_FALLBACK = os.environ.get("BAZAARLINK_FREE_FALLBACK", "auto:free")

TIMEOUT_SECONDS = 12
GROUNDED_TIMEOUT_SECONDS = 15
MAX_RETRIES = 1
MAX_EVIDENCE_IN_CONTEXT = 5
MAX_EVIDENCE_CHARS = 320

_CACHE: dict[str, dict] = {}

_SYSTEM_ANALYZE = (
    "You are a skeptical fact-check analyst. Given one atomic claim and the "
    "retrieved evidence for it, decide whether the evidence SUPPORTS the claim, "
    "CONTRADICTS it, or is INSUFFICIENT to decide. Base your decision ONLY on "
    "the provided evidence. If the evidence does not directly address the "
    "exact claim, or is too weak, answer INSUFFICIENT. Never use outside "
    "knowledge to invent support or contradiction. Never call a claim FALSE "
    "merely because no evidence was found."
)

_USER_ANALYZE = (
    "Article context (language: {language}):\n{article_context}\n\n"
    "Claim to verify:\n{claim}\n\n"
    "Retrieved evidence:\n{evidence}\n\n"
    'Respond with STRICT JSON only, no markdown:\n'
    '{{"decision": "SUPPORT or CONTRADICT or INSUFFICIENT", '
    '"confidence": 0.0, "reasoning": "one or two short sentences"}}\n'
    "- SUPPORT: strong, relevant evidence directly backs the claim.\n"
    "- CONTRADICT: strong, relevant evidence directly contradicts it.\n"
    "- INSUFFICIENT: evidence missing, off-topic, weak, or conflicting.\n"
    "- confidence: 0.0 to 1.0 how sure you are about your decision.\n"
    "- reasoning: explain which evidence you used and why."
)

_SYSTEM_REVIEW = (
    "You are an adversarial fact-check reviewer. A colleague (AI #1) has "
    "analyzed a claim against retrieved evidence. You must NOT trust their "
    "conclusion. Actively hunt for errors:\n"
    "1. Was the original claim understood correctly?\n"
    "2. Is the evidence actually about the same claim?\n"
    "3. Is the evidence recent enough?\n"
    "4. Are dates consistent?\n"
    "5. Are numbers consistent?\n"
    "6. Is there any contradiction?\n"
    "7. Are the supporting/contradicting sources independent?\n"
    "8. Are the sources credible?\n"
    "9. Did AI #1 rely on an unsupported assumption?\n"
    "10. Did AI #1 hallucinate any fact not present in the evidence?\n"
    "11. Does the evidence really prove the claim?\n"
    "12. Would the honest answer instead be INSUFFICIENT/UNVERIFIED?\n"
    "Then give YOUR OWN verdict about the claim, based on the same evidence."
)

_USER_REVIEW = (
    "Article context (language: {language}):\n{article_context}\n\n"
    "Claim under review:\n{claim}\n\n"
    "Retrieved evidence:\n{evidence}\n\n"
    "AI #1 analysis:\n{ai1}\n\n"
    'Respond with STRICT JSON only, no markdown:\n'
    '{{"verdict": "SUPPORT or CONTRADICT or INSUFFICIENT", '
    '"confidence": 0.0, "agrees_with_first": true or false, '
    '"problems": ["short problem", "..."], "reasoning": "one short sentence"}}\n'
    "- verdict: your independent judgment of the claim on the evidence.\n"
    "- agrees_with_first: whether you agree with AI #1's decision.\n"
    "- problems: concrete flaws you found in AI #1's reasoning (empty is fine).\n"
    "- reasoning: one short sentence summarising your verdict.\n"
    "- List real problems only; do not invent issues to disagree."
)

_SYSTEM_FINAL_ENGINE = (
    "You are the claim-verification engine for TruthLens AI. You verify "
    "factual claims using Google Search grounding - live web evidence, not "
    "memorized knowledge.\n\n"
    "CRITICAL RULE:\n"
    "Your training data has a cutoff and WILL be wrong about anything that "
    "happened after it - current office-holders, party leaders, alliance names, "
    "election results, recent events. You do NOT know current facts from memory "
    "alone. Before stating any fact about a role, title, office-holder, party, "
    "or event status, you MUST search for it and check it against what you "
    "find. If your prior belief contradicts the search results, the search "
    "results win - always. Today is {today}.\n\n"
    "PROCESS:\n"
    "1. Read the headline and/or article given to you.\n"
    "2. Extract the individual factual claims (ignore opinions/predictions "
    "unless stated as fact).\n"
    "3. For each claim, generate multiple search query variants (not just the "
    "exact wording given) and search using Google Search grounding.\n"
    "4. Identify which sources are original reporting vs. syndicated copies of "
    "the same story. Treat a syndication chain as ONE piece of evidence, not "
    "several.\n"
    "5. Classify each source: official primary source / established news "
    "outlet / fact-check organization / other credible / "
    "low-quality-or-unknown.\n"
    "6. Compare the claimed event's timing against what your search results "
    "show - is this being reported as current, or is it old news being reused?\n"
    "7. Actively search for contradicting evidence, not just confirming "
    "evidence.\n"
    "8. If no evidence exists yet, that means evidence is LOW - it does NOT "
    "automatically mean the claim is false. Search again with alternative "
    "queries, official sources, and organization/party websites before "
    "concluding evidence is weak.\n"
    "9. If you notice yourself disagreeing with search results because \"that's "
    "not what I know,\" search again, and defer to consistent, credible search "
    "results over your own memory.\n"
    "10. For Tamil Nadu / Indian regional political claims specifically, also "
    "try queries in Tamil and with common English transliterations, and check "
    "party/official sources in addition to English-language national media.\n\n"
    "OUTPUT RULES:\n"
    "- verdict must be exactly \"REAL\" or \"FAKE\" - never any third value.\n"
    "- REAL: the core claim is supported by credible evidence - a primary "
    "source, or multiple genuinely independent reports.\n"
    "- FAKE: the core claim is directly contradicted by credible search "
    "evidence, or is demonstrably fabricated.\n"
    "- If evidence is thin or mixed, still choose the better-supported verdict, "
    "but set confidence LOW and explain why. Never invent evidence to appear "
    "more certain than the evidence supports.\n"
    "- confidence reflects the strength of evidence found via search, not how "
    "sure you \"feel.\" Bands: 90-100 = primary or multiple independent "
    "credible sources; 75-89 = good support; 50-74 = limited or mixed; 25-49 = "
    "weak; 0-24 = very little.\n\n"
    "Respond ONLY with a single JSON object, no markdown, no text outside the "
    "JSON, matching the exact structure requested by the user (verdict, "
    "confidence, evidence_strength, reasoning, sources_checked, "
    "key_claims_verified, limitations). Cite only REAL sources you actually "
    "checked - never invent titles, URLs, quotes, or dates."
).format(today=date.today().isoformat())

_USER_FINAL_ENGINE = (
    "INPUT DATA:\n"
    "HEADLINE: {headline}\n"
    "ARTICLE BODY: {article}\n"
    "LANGUAGE: {language}\n"
    "STAGE_1_VERDICT: {initial_verdict}\n"
    "STAGE_1_CONFIDENCE: {initial_confidence}%\n\n"
    "LIVE SEARCH RESULTS (retrieved now via Google News / web search, plus "
    "knowledge-bound evidence; use ONLY these, or pages YOUR OWN Google Search "
    "returns - never invent any titles, URLs, dates, quotes, or sources):\n"
    "{search_results}\n\n"
    "EXTRA PIPELINE CONTEXT (informational only, not authoritative):\n"
    "{provisional}\n\n"
    "OUTPUT FORMAT:\n"
    'Respond ONLY with a single JSON object, no markdown, no text outside the '
    'JSON:\n'
    '{{\n'
    '  "verdict": "REAL" | "FAKE",\n'
    '  "confidence": <integer 0-100>,\n'
    '  "evidence_strength": "HIGH" | "MEDIUM" | "LOW" | "VERY_LOW",\n'
    '  "reasoning": "<2-4 sentences citing what search found>",\n'
    '  "sources_checked": [\n'
    '    {{"title": "string", "url": "string", "source_type": "string", '
    '"published_date": "string", "supports_claim": true, '
    '"is_duplicate_of": ""}}\n'
    '  ],\n'
    '  "key_claims_verified": [\n'
    '    {{"claim": "string", "status": "SUPPORTED", "evidence_strength": "string"}}\n'
    '  ],\n'
    '  "limitations": ["string"]\n'
    '}}\n'
    "- verdict: exactly \"REAL\" or \"FAKE\"; never any third value.\n"
    "- REAL: supported by a primary source, or multiple genuinely independent "
    "reports. FAKE: directly contradicted by credible search evidence, or "
    "demonstrably fabricated.\n"
    "- If evidence is thin or mixed, choose the better-supported verdict but "
    "set confidence LOW and explain; never invent evidence.\n"
    "- confidence = strength of evidence found via search (see system bands), "
    "not how sure you feel.\n"
    "- sources_checked: only real sources you actually used; mark syndicated "
    "copies with is_duplicate_of (empty string for an original report).\n"
    "- key_claims_verified: each extracted factual claim and how it checked "
    "out, with evidence strength.\n"
    "- limitations: genuine gaps (e.g. few sources, old coverage, no official "
    "confirmation yet). Empty list when none."
)

_DECISION_RE = re.compile(r'"decision"\s*:\s*"(SUPPORT[^",]*|CONTRADICT[^",]*|INSUFFICIENT[^",]*|UNVERIFIED)"', re.I)
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(SUPPORT[^",]*|CONTRADICT[^",]*|INSUFFICIENT[^",]*|UNVERIFIED)"', re.I)
_CONF_RE = re.compile(r'"confidence"\s*:\s*(0?\.\d+|\d\.\d+|1|0)\b')
_AGREES_RE = re.compile(r'"agrees_with_first"\s*:\s*(true|false)', re.I)
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_PROBLEM_RE = re.compile(r'"\s*((?:[^"\\]|\\.){3,}?)\s*"\s*[,}\]]', re.DOTALL)

_CANONICAL = {
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT",
    "CONTRADICT": "CONTRADICT", "CONTRADICTS": "CONTRADICT",
    "INSUFFICIENT": "INSUFFICIENT", "UNVERIFIED": "INSUFFICIENT",
}

_FINAL_LABELS = {
    "REAL": "REAL", "TRUE": "REAL", "SUPPORT": "REAL", "SUPPORTS": "REAL",
    "FAKE": "FALSE", "FALSE": "FALSE", "CONTRADICT": "FALSE",
    "CONTRADICTS": "FALSE",
    "UNVERIFIED": "UNVERIFIED", "INSUFFICIENT": "UNVERIFIED",
    "CANNOT VERIFY": "UNVERIFIED", "UNCERTAIN": "UNVERIFIED",
}


def _normalize_verdict(value: object) -> str | None:
    """Map an LLM verdict string to SUPPORT / CONTRADICT / INSUFFICIENT."""
    canonical = _CANONICAL.get(str(value or "").strip().upper())
    return canonical if canonical else None


def available() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("BAZAARLINK_API_KEY", "").strip()
    )


def model_name() -> str:
    parts = []
    if os.environ.get("GEMINI_API_KEY", "").strip():
        parts.append(f"gemini:{GEMINI_MODEL}")
    if os.environ.get("BAZAARLINK_API_KEY", "").strip():
        parts.append(f"bazaarlink:{BAZAARLINK_MODEL}")
    return " + ".join(parts) or "none"


def _cache_key(*parts: str) -> str:
    raw = "\x1f".join(parts)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


def build_evidence_context(evidence: list[dict]) -> str:
    """Compact, read-only summary of the classified evidence for an AI stage."""
    rows = []
    for e in evidence[:MAX_EVIDENCE_IN_CONTEXT]:
        relation = e.get("relation", "NEUTRAL")
        source = e.get("source_name") or e.get("domain") or "unknown"
        snippet = (e.get("snippet") or e.get("text") or e.get("title") or "")[:MAX_EVIDENCE_CHARS]
        quality = e.get("source_score")
        tier = e.get("source_tier", "unknown")
        domain = e.get("domain", "")
        quality_txt = f" (quality {quality:.2f}, tier {tier})" if isinstance(quality, (int, float)) else ""
        rows.append(f"- [{relation}] {source}{quality_txt}\n  domain: {domain}\n  {snippet}")
    return "\n\n".join(rows) if rows else "(no evidence retrieved)"


# ---------------------------------------------------------------------------
# Gemini call (AI #1)
# ---------------------------------------------------------------------------

def _grounding_from_gemini(data: dict) -> tuple[list[dict], list[str]]:
    """Pull real Google Search grounding chunks/queries from a response."""
    chunks: list[dict] = []
    queries: list[str] = []
    try:
        meta = data["candidates"][0].get("groundingMetadata") or {}
        for chunk in meta.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            uri = str(web.get("uri") or "").strip()
            if uri:
                chunks.append({"title": str(web.get("title") or "")[:200], "url": uri})
        queries = [str(q) for q in (meta.get("webSearchQueries") or [])][:8]
    except (KeyError, IndexError, TypeError) as exc:  # noqa: B841
        logger.debug("no grounding metadata: %s", exc)
    return chunks, queries


_GROUNDING_TOOL = [{
    "google_search_retrieval": {
        "dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.5},
    },
}]


def _call_gemini_grounded(system: str, user: str,
                          temperature: float = 0.1) -> tuple[dict | None,
                                                            list[dict], list[str]]:
    """Call Gemini WITH real Google Search grounding.

    Returns ``(parsed_object, grounding_chunks, web_search_queries)``. If the
    API rejects the grounding tool, retries WITHOUT it so the engine still
    works. ``parsed_object`` is ``None`` when Gemini is unavailable.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None, [], []
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers_list = [{"x-goog-api-key": key}, {"Authorization": f"Bearer {key}"}]
    for attempt in range(MAX_RETRIES + 1):
        transient = False
        for auth in headers_list:
            for payload in _grounded_payloads(system, user, temperature):
                grounded = "tools" in payload
                try:
                    req = urllib.request.Request(
                        url, data=json.dumps(payload).encode("utf-8"),
                        headers={**auth, "Content-Type": "application/json"},
                        method="POST",
                    )
                    timeout = GROUNDED_TIMEOUT_SECONDS if grounded else TIMEOUT_SECONDS
                    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 https only
                        data = json.loads(resp.read().decode("utf-8"))
                    text = _text_from_gemini(data)
                    if not text:
                        continue
                    obj = _parse_json_object(text)
                    if not obj:
                        continue
                    chunks, queries = _grounding_from_gemini(data)
                    logger.info(
                        "[GEMINI] final-engine parsed verdict=%s  grounded=%s  chunks=%d  queries=%d",
                        obj.get("verdict"), grounded, len(chunks), len(queries),
                    )
                    return obj, chunks, queries
                except urllib.error.HTTPError as exc:
                    if grounded and exc.code in (400, 429, 500, 503):
                        continue
                    if exc.code in (400, 401, 403):
                        break
                    if exc.code in (429, 500, 503):
                        transient = True
                        break
                    return None, [], []
                except Exception:  # noqa: BLE001 - network/timeout: try non-grounded
                    if grounded:
                        continue
                    return None, [], []
            if transient:
                break
        if not transient:
            return None, [], []
        time.sleep(0.5 * (2 ** attempt))
    return None, [], []


def _grounded_payloads(system: str, user: str, temperature: float) -> list[dict]:
    """Two payload attempts: with real Google Search grounding, then without."""
    base = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    return [
        {**base, "tools": _GROUNDING_TOOL},
        base,
    ]


def _call_gemini(system: str, user: str, temperature: float = 0.1) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    headers_list = [{"x-goog-api-key": key}, {"Authorization": f"Bearer {key}"}]
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        transient = False
        for auth in headers_list:
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={**auth, "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 https only
                    data = json.loads(resp.read().decode("utf-8"))
                text = _text_from_gemini(data)
                if not text:
                    continue
                return _parse_json_object(text)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403):
                    continue
                if exc.code in (429, 500, 503):
                    transient = True
                    break
                return None
            except Exception:  # noqa: BLE001 - network/timeout: degrade gracefully
                return None
        if not transient:
            return None
        time.sleep(0.5 * (2 ** attempt))
    return None


def _text_from_gemini(data: dict) -> str | None:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text.strip() or None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# BazaarLink call (AI #2) — OpenAI-compatible /v1/chat/completions
# ---------------------------------------------------------------------------

def _call_bazaarlink(system: str, user: str, temperature: float = 0.1) -> dict | None:
    key = os.environ.get("BAZAARLINK_API_KEY", "")
    if not key:
        return None
    body = json.dumps({
        "model": BAZAARLINK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    url = f"{BAZAARLINK_BASE_URL}/chat/completions"
    for attempt in range(MAX_RETRIES + 2):
        transient = False
        if attempt > 0:
            body = json.dumps({
                "model": BAZAARLINK_FREE_FALLBACK,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            text = _text_from_bazaarlink(data)
            if not text:
                return None
            return _parse_json_object(text)
        except urllib.error.HTTPError as exc:
            if exc.code in (402, 429, 500, 503):
                transient = True
            else:
                return None
        except Exception:  # noqa: BLE001
            return None
        if not transient:
            return None
        time.sleep(0.5 * (2 ** attempt))
    return None


def _text_from_bazaarlink(data: dict) -> str | None:
    try:
        return data["choices"][0]["message"]["content"].strip() or None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_model_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    except ValueError:
        pass
    out: dict = {}
    decision = _DECISION_RE.search(text) or _VERDICT_RE.search(text)
    conf = _CONF_RE.search(text)
    agree = _AGREES_RE.search(text)
    reason = _REASON_RE.search(text)
    if decision:
        out["decision"] = decision.group(1).upper()
    if conf:
        out["confidence"] = max(0.0, min(1.0, float(conf.group(1))))
    if agree:
        out["agrees_with_first"] = agree.group(1).lower() == "true"
    if reason:
        out["reasoning"] = reason.group(1)[:400]
    return out


def _parse_json_object(text: str) -> dict | None:
    try:
        start = text.find("{")
        end = text.rfind("}")
        obj = json.loads(text[start:end + 1]) if (start != -1 and end > start) else {}
        if isinstance(obj, dict) and obj:
            return obj
    except ValueError:
        pass
    fallback = _extract_model_json(text)
    return fallback or None


def _problems_from(text: str) -> list[str]:
    problems = _PROBLEM_RE.findall(text)
    return [p.strip().replace("\\n", " ")[:160] for p in problems][:5]


# ---------------------------------------------------------------------------
# Public API — AI #1 (Gemini) + AI #2 (BazaarLink)
# ---------------------------------------------------------------------------

def analyze_claim_ai(claim: str, evidence: list[dict], language: str,
                     article: str | None = None) -> dict | None:
    """AI analysis #1 (Gemini) for one atomic claim, grounded on the full
    article context (HEADLINE + ARTICLE + LANGUAGE). Returns a dict or None."""
    ctx = build_evidence_context(evidence)
    key = _cache_key("ai1", claim, ctx, article or "")
    if key in _CACHE:
        return _CACHE[key]
    result = _run_analyze(claim, ctx, language, article)
    _CACHE[key] = result
    return result


def _run_analyze(claim: str, ctx: str, language: str, article: str | None) -> dict | None:
    user = _USER_ANALYZE.format(
        language=language or "english",
        article_context=(article or "").strip()[:2500] or "(only the claim below was provided)",
        claim=claim[:1200],
        evidence=ctx or "(no evidence retrieved)",
    )
    obj = _call_gemini(_SYSTEM_ANALYZE, user)
    if not obj:
        return None
    decision = _normalize_verdict(obj.get("decision") or obj.get("verdict"))
    if not decision:
        return None
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "available": True,
        "source": "gemini",
        "model": GEMINI_MODEL,
        "decision": decision,
        "confidence": round(confidence, 3),
        "reasoning": str(obj.get("reasoning", ""))[:400],
    }


def review_claim_ai(claim: str, evidence: list[dict], ai1: dict,
                    language: str, article: str | None = None) -> dict | None:
    """AI analysis #2 (BazaarLink): adversarial review of AI #1, grounded on
    the full article context."""
    ctx = build_evidence_context(evidence)
    key = _cache_key("ai2", claim, ctx, article or "",
                     json.dumps(ai1, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    result = _run_review(claim, ctx, ai1, language, article)
    _CACHE[key] = result
    return result


def _run_review(claim: str, ctx: str, ai1: dict, language: str,
                article: str | None) -> dict | None:
    ai1_txt = (
        f"Decision: {ai1.get('decision', 'n/a')}, "
        f"confidence {ai1.get('confidence', 'n/a')}. "
        f"Reasoning: {ai1.get('reasoning', '')}"
    )
    user = _USER_REVIEW.format(
        language=language or "english",
        article_context=(article or "").strip()[:2500] or "(only the claim below was provided)",
        claim=claim[:1200],
        evidence=ctx or "(no evidence retrieved)",
        ai1=ai1_txt[:800],
    )
    obj = _call_bazaarlink(_SYSTEM_REVIEW, user)
    if not obj:
        return None
    verdict = _normalize_verdict(obj.get("verdict") or obj.get("decision"))
    if not verdict:
        return None
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    agrees = bool(obj.get("agrees_with_first", False))
    if "agrees_with_first" not in obj:
        agrees = (verdict == ai1.get("decision"))
    problems = [str(p) for p in (obj.get("problems") or [])][:5]
    if not problems and obj.get("reasoning"):
        problems = _problems_from(obj.get("reasoning", ""))
    return {
        "available": True,
        "source": "bazaarlink",
        "model": BAZAARLINK_MODEL,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "agrees_with_first": agrees,
        "problems": problems,
        "reasoning": str(obj.get("reasoning", ""))[:400],
    }


# ---------------------------------------------------------------------------
# FINAL verification engine (Gemini) - the last stage
# ---------------------------------------------------------------------------

def _normalize_final_label(value: object) -> str | None:
    """Map a label string to REAL / FALSE / UNVERIFIED."""
    return _FINAL_LABELS.get(str(value or "").strip().upper())


def _tier_to_source_type(tier: str | None) -> str:
    normalized = {
        "primary": "official", "authoritative": "official",
        "authoritative-reference": "official",
        "official": "official", "government": "official",
        "news": "news", "secondary": "news",
        "factcheck": "fact-check", "fact-check": "fact-check",
    }.get(str(tier or "").strip().lower())
    return normalized or "reference"


_SOURCE_FIELDS = ("title", "url", "source_type", "published_date", "supports_claim")
_CLAIM_FIELDS = ("claim", "status", "evidence_strength")


def format_search_results(items: list[dict]) -> str:
    """Compact, real source list for the FINAL engine prompt (never invented)."""
    rows = []
    for e in items[:16]:
        relation = e.get("relation", "NEUTRAL")
        source = e.get("source") or e.get("source_name") or e.get("domain") or "unknown"
        url = e.get("url") or ""
        date = e.get("date") or e.get("published_date") or ""
        snippet = (e.get("snippet") or e.get("text") or e.get("title") or "")[:240]
        rows.append(
            f"- [{relation}] \"{str(e.get('title') or '')[:160]}\" "
            f"({source}, type: {_tier_to_source_type(e.get('type') or e.get('source_tier'))}"
            f"{', ' + str(date) if date else ''})\n"
            f"  url: {url}\n  snippet: {snippet}"
        )
    return "\n\n".join(rows) if rows else "(no search results were retrieved)"


def _clean_sources_checked(value: object, allowed_urls: set[str],
                           allowed_sources: list[dict]) -> list[dict]:
    """Keep only sources that were ACTUALLY retrieved - never fabricate URLs."""
    out: list[dict] = []
    for raw in (value or [])[:8]:
        if not isinstance(raw, dict):
            continue
        item: dict = {}
        url = str(raw.get("url") or "").strip()
        if url and (url not in allowed_urls):
            continue
        item["url"] = url or None
        item["title"] = str(raw.get("title") or "")[:200]
        item["source_type"] = str(raw.get("source_type")
                                  or raw.get("type") or "reference")[:40]
        for src in allowed_sources:
            if url and url == src.get("url"):
                item.setdefault("published_date",
                                str(src.get("date") or src.get("published_date") or "") or None)
                break
        if "published_date" not in item:
            item["published_date"] = (str(raw.get("published_date") or "")
                                      if raw.get("published_date") else None)
        item["supports_claim"] = bool(raw.get("supports_claim", True))
        dup = str(raw.get("is_duplicate_of") or "").strip()
        if dup:
            item["is_duplicate_of"] = dup[:200]
        if item.get("url") or item.get("title"):
            out.append(item)
    return out


def _clean_key_claims(value: object) -> list[dict]:
    status_map = {
        "SUPPORTED": "SUPPORTED", "SUPPORT": "SUPPORTED", "SUPPORTS": "SUPPORTED",
        "CONFIRMED": "SUPPORTED", "TRUE": "SUPPORTED", "REAL": "SUPPORTED",
        "CONTRADICTED": "CONTRADICTED", "REFUTED": "CONTRADICTED",
        "CONTRADICTS": "CONTRADICTED", "CONTRADICT": "CONTRADICTED",
        "DISPROVEN": "CONTRADICTED", "FALSE": "CONTRADICTED", "FAKE": "CONTRADICTED",
    }
    out: list[dict] = []
    for raw in (value or [])[:8]:
        if not isinstance(raw, dict):
            continue
        status = status_map.get(str(raw.get("status") or "").strip().upper(), "UNVERIFIED")
        strength = str(raw.get("evidence_strength") or "").upper()[:8]
        if strength not in {"HIGH", "MEDIUM", "LOW"}:
            strength = "—"
        out.append({
            "claim": str(raw.get("claim") or "")[:240],
            "status": status,
            "evidence_strength": strength,
        })
    return out


def final_verdict_engine(headline: str, article: str, language: str,
                         initial_verdict: str, initial_confidence: float,
                         initial_reasoning: str,
                         search_results: list[dict] | None = None,
                         provisional: dict | None = None) -> dict | None:
    """Gemini independently verifies the news and is the FINAL decision-maker.

    Never blindly trusts the initial model. Grounds on the article, the live
    search results (real sources only), and its own reasoning, then returns the
    per-spec JSON verdict (REAL/FAKE, 0-100 confidence, sources_checked,
    key_claims_verified, initial_model_was_correct). Returns ``None`` when the
    provider is unavailable or refuses a REAL/FAKE verdict (so the evidence-led
    pipeline result stands).
    """
    allowed_urls = {
        str(e.get("url") or "").strip() for e in (search_results or []) if e.get("url")
    }
    key = _cache_key(
        "final_engine", headline[:1500], article[:3500], language,
        initial_verdict, str(initial_confidence), date.today().isoformat(),
        json.dumps(search_results or [], sort_keys=True, default=str)[:4000],
    )
    if key in _CACHE:
        return _CACHE[key]
    result = _run_final_engine(
        headline, article, language, initial_verdict, initial_confidence,
        initial_reasoning, search_results or [], provisional,
        allowed_urls, key,
    )
    if result is not None:
        _CACHE[key] = result
    return result


def _run_final_engine(headline: str, article: str, language: str,
                      initial_verdict: str, initial_confidence: float,
                      initial_reasoning: str, search_results: list[dict],
                      provisional: dict | None, allowed_urls: set[str],
                      cache_key: str) -> dict | None:
    provisional_txt = (
        f"verdict={provisional.get('verdict', 'n/a')}, "
        f"confidence={provisional.get('confidence', 0):.2f}, "
        f"counts={provisional.get('counts', {})}, "
        f"explanation={str(provisional.get('explanation') or '')[:400]}"
    ) if provisional else "(not available)"
    # Free, keyless live web search: guarantees real current sources even when
    # the paid Google Search grounding tool is quota-limited.
    web_results: list[dict] = []
    if not any(str(e.get("url") or "").strip() for e in search_results):
        web_results = search_web(build_query(headline, article), limit=6)
    combined_results = list(search_results) + web_results
    allowed_urls |= {str(r.get("url") or "").strip() for r in web_results if r.get("url")}
    user = _USER_FINAL_ENGINE.format(
        headline=(headline or "").strip()[:2000],
        article=(article or "").strip()[:4000],
        language=language or "english",
        today=date.today().isoformat(),
        initial_verdict=str(initial_verdict or "n/a").upper(),
        initial_confidence=f"{initial_confidence:.0f}" if initial_confidence else "n/a",
        initial_reasoning=str(initial_reasoning or "")[:400],
        search_results=format_search_results(combined_results),
        provisional=provisional_txt,
    )
    obj, chunks, queries = _call_gemini_grounded(_SYSTEM_FINAL_ENGINE, user,
                                                 temperature=0.1)
    if not obj:
        return None
    grounding_urls = {c["url"] for c in chunks}
    allowed_urls |= grounding_urls
    label = _normalize_final_label(obj.get("verdict") or obj.get("label")
                                   or obj.get("decision"))
    # The engine is REAL/FAKE only: a refusal maps to "no verdict".
    if label not in ("REAL", "FALSE"):
        return None
    try:
        conf = max(0.0, min(100.0, float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    initial_norm = _normalize_final_label(initial_verdict)
    if obj.get("initial_model_was_correct") is not None:
        was_correct = bool(obj["initial_model_was_correct"])
    else:
        was_correct = bool(initial_norm) and (label == initial_norm)
    sources = _clean_sources_checked(
        obj.get("sources_checked"), allowed_urls, combined_results)
    # Surface real sources the model omitted: Google grounding first, then the
    # free live web search results.
    if not sources:
        if chunks:
            sources = [
                {"title": c["title"], "url": c["url"], "source_type": "Google Search",
                 "published_date": None, "supports_claim": True}
                for c in chunks[:8]
            ]
        else:
            sources = [
                {"title": r["title"], "url": r["url"], "source_type": r.get("source_type") or "web",
                 "published_date": r.get("published_date"), "supports_claim": True}
                for r in web_results[:8]
            ]
    strength = str(obj.get("evidence_strength") or "").strip().upper()[:8]
    if strength not in {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}:
        strength = ""
    limitations = [str(x).strip()[:200] for x in (obj.get("limitations") or [])
                   if isinstance(x, str) and x.strip()][:10]
    logger.info(
        "[GEMINI] Parsed verdict=%s confidence=%d sources=%d queries=%d web=%d strength=%s",
        label, round(conf), len(sources), len(queries), len(web_results),
        strength or "-",
    )
    return {
        "available": True,
        "source": "gemini",
        "model": GEMINI_MODEL,
        "verdict": label,            # "REAL" or "FALSE" (final convention)
        "confidence": round(conf),   # 0-100 (evidence strength)
        "initial_model_verdict": _normalize_final_label(initial_verdict) or "n/a",
        "initial_model_confidence": round(initial_confidence, 3),
        "initial_model_was_correct": was_correct,
        "reasoning": str(obj.get("reasoning", ""))[:800],
        "evidence_strength": strength,
        "limitations": limitations,
        "sources_checked": sources,
        "key_claims_verified": _clean_key_claims(obj.get("key_claims_verified")),
    }
