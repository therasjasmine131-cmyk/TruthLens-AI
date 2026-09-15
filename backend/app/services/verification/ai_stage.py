"""AI reasoning stages for the multi-stage verification pipeline.

Two independent free-tier Gemini stages (no SDK, plain HTTPS) sit between
evidence retrieval and the final decision engine:

* ``analyze_claim_ai`` - AI analysis #1: given the claim and the retrieved
  evidence, decide SUPPORT / CONTRADICT / INSUFFICIENT using ONLY the evidence.
* ``review_claim_ai`` - AI analysis #2: an adversarial reviewer that actively
  looks for errors in AI #1 (mis-stated claim, off-topic evidence, staleness,
  hallucination, independence, credibility...).

Both degrade to ``None`` whenever Gemini is unavailable, so the rest of the
pipeline behaves exactly as before (evidence-driven verdicts, ML tie-breaker).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
TIMEOUT_SECONDS = 18
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
    "Claim (language: {language}):\n{claim}\n\n"
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
    "Claim (language: {language}):\n{claim}\n\n"
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

_DECISION_RE = re.compile(r'"decision"\s*:\s*"(SUPPORT|CONTRADICT|INSUFFICIENT)"', re.I)
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(SUPPORT|CONTRADICT|INSUFFICIENT)"', re.I)
_CONF_RE = re.compile(r'"confidence"\s*:\s*(0?\.\d+|\d\.\d+|1|0)\b')
_AGREES_RE = re.compile(r'"agrees_with_first"\s*:\s*(true|false)', re.I)
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_PROBLEM_RE = re.compile(r'"\s*((?:[^"\\]|\\.){3,}?)\s*"\s*[,}\]]', re.DOTALL)


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


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


def _call(system: str, user: str, temperature: float = 0.1):
    """One Gemini completion. Returns the parsed JSON dict or None."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    url = f"{GEMINI_BASE_URL}/models/{model_name()}:generateContent"
    headers = [{"x-goog-api-key": key}, {"Authorization": f"Bearer {key}"}]
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        transient = False
        for auth in headers:
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={**auth, "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 https only
                    data = json.loads(resp.read().decode("utf-8"))
                text = _text_from_response(data)
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


def _text_from_response(data: dict) -> str | None:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text.strip() or None
    except (KeyError, IndexError, TypeError):
        return None


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


def analyze_claim_ai(claim: str, evidence: list[dict], language: str) -> dict | None:
    """AI analysis #1 for one atomic claim. Returns a dict or None."""
    ctx = build_evidence_context(evidence)
    key = _cache_key("ai1", claim, ctx)
    if key in _CACHE:
        return _CACHE[key]
    result = _run_analyze(claim, ctx, language)
    _CACHE[key] = result
    return result


def _run_analyze(claim: str, ctx: str, language: str) -> dict | None:
    user = _USER_ANALYZE.format(
        language=language or "english",
        claim=claim[:1200],
        evidence=ctx or "(no evidence retrieved)",
    )
    obj = _call(_SYSTEM_ANALYZE, user)
    if not obj:
        return None
    decision = str(obj.get("decision") or obj.get("verdict") or "").upper()
    if decision not in {"SUPPORT", "CONTRADICT", "INSUFFICIENT"}:
        return None
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "available": True,
        "source": "gemini",
        "model": model_name(),
        "decision": decision,
        "confidence": round(confidence, 3),
        "reasoning": str(obj.get("reasoning", ""))[:400],
    }


def review_claim_ai(claim: str, evidence: list[dict], ai1: dict,
                    language: str) -> dict | None:
    """AI analysis #2: adversarial review of AI #1. Returns a dict or None."""
    ctx = build_evidence_context(evidence)
    key = _cache_key("ai2", claim, ctx,
                     json.dumps(ai1, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    result = _run_review(claim, ctx, ai1, language)
    _CACHE[key] = result
    return result


def _run_review(claim: str, ctx: str, ai1: dict, language: str) -> dict | None:
    ai1_txt = (
        f"Decision: {ai1.get('decision', 'n/a')}, "
        f"confidence {ai1.get('confidence', 'n/a')}. "
        f"Reasoning: {ai1.get('reasoning', '')}"
    )
    user = _USER_REVIEW.format(
        language=language or "english",
        claim=claim[:1200],
        evidence=ctx or "(no evidence retrieved)",
        ai1=ai1_txt[:800],
    )
    obj = _call(_SYSTEM_REVIEW, user)
    if not obj:
        return None
    verdict = str(obj.get("verdict") or obj.get("decision") or "").upper()
    if verdict not in {"SUPPORT", "CONTRADICT", "INSUFFICIENT"}:
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
        "source": "gemini",
        "model": model_name(),
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "agrees_with_first": agrees,
        "problems": problems,
        "reasoning": str(obj.get("reasoning", ""))[:400],
    }