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
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

BAZAARLINK_BASE_URL = "https://api.bazaarlink.ai/v1"
BAZAARLINK_MODEL = os.environ.get("BAZAARLINK_MODEL", "deepseek/deepseek-v4-flash-0731")
BAZAARLINK_FREE_FALLBACK = os.environ.get("BAZAARLINK_FREE_FALLBACK", "auto:free")

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

_SYSTEM_VALIDATE = (
    "You are the FINAL VALIDATOR of a multi-stage fact-check pipeline. A "
    "pipeline (neural-network signal + evidence retrieval + AI analysis #1 + "
    "adversarial AI review #2) has already produced a PROVISIONAL verdict for "
    "the submitted news text. Do not blindly trust it - independently judge the "
    "text yourself and give YOUR OWN final verdict: REAL (true news), FAKE "
    "(false/misleading), or UNVERIFIED (cannot be confirmed).\n"
    "Critically, you MUST explain WHY: if REAL, name the specific facts or "
    "reporting that make it credible; if FAKE, point out exactly what is wrong "
    "or misleading and why; if UNVERIFIED, say what is missing. Base this on "
    "your knowledge of the world and on the evidence summary below. Never "
    "invent specific sources or URLs that were not provided."
)

_USER_VALIDATE = (
    "Article (language: {language}):\n{article}\n\n"
    "Pipeline result (provisional):\n{overall}\n\n"
    "Adds up to a verdict of \"{verdict}\" at {confidence}% confidence.\n\n"
    "Claim-by-claim summary:\n{claims}\n\n"
    'Respond with STRICT JSON only, no markdown:\n'
    '{{"label": "REAL or FAKE or UNVERIFIED", "confidence": 0.0, '
    '"agrees": true or false, "reasoning": "explain WHY it is REAL, WHY it '
    'is FAKE, or why it cannot be verified - a few clear sentences"}}\n'
    "- label: your independent final validation of the whole article.\n"
    "- confidence: 0.0 to 1.0 how sure you are about your label.\n"
    "- agrees: whether you agree with the pipeline's provisional verdict.\n"
    "- reasoning: the 'why' - concrete reasons a reader will understand.\n"
    "- Never mention pipeline stage names; speak as a fact-checker to a reader."
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

def analyze_claim_ai(claim: str, evidence: list[dict], language: str) -> dict | None:
    """AI analysis #1 (Gemini) for one atomic claim. Returns a dict or None."""
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
                    language: str) -> dict | None:
    """AI analysis #2 (BazaarLink): adversarial review of AI #1."""
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
# Final validation (Gemini) - the last stage
# ---------------------------------------------------------------------------

def _normalize_final_label(value: object) -> str | None:
    """Map a label string to the verdict convention REAL / FALSE / UNVERIFIED."""
    return _FINAL_LABELS.get(str(value or "").strip().upper())


def final_validation(article: str, overall: dict, claims: list[dict],
                     language: str = "english") -> dict | None:
    """Gemini independently validates the pipeline's final result and says WHY.

    This is the LAST stage of the pipeline: after the AI test result (NN +
    evidence + AI analysis #1 + adversarial review #2) is produced, Gemini
    reviews the whole article, gives its own Real/Fake/Unverified label and a
    plain-language explanation of WHY. Returns ``None`` when Gemini is not
    configured or unreachable (the pipeline result is then unchanged).
    """
    provisional = (overall or {}).get("verdict") or "UNVERIFIED"
    confidence = (overall or {}).get("confidence") or 0.0
    explanation = (overall or {}).get("explanation") or ""
    rows = []
    for claim in (claims or [])[:6]:
        rows.append(
            f"- \"{str(claim.get('text'))[:200]}\" -> {claim.get('verdict')}"
            f" (authority: {claim.get('final_authority', claim.get('authority', 'n/a'))}, "
            f"confidence {claim.get('confidence', 0):.2f})"
        )
    overall_txt = (
        f"verdict={provisional}, confidence={confidence:.2f}, "
        f"explanation={str(explanation)[:600]}"
    )
    key = _cache_key("final_validate", article[:2000], overall_txt)
    if key in _CACHE:
        return _CACHE[key]
    result = _run_final_validation(article, overall_txt, provisional,
                                   confidence, rows, language)
    _CACHE[key] = result
    return result


def _run_final_validation(article: str, overall_txt: str, provisional: str,
                          confidence: float, rows: list[str],
                          language: str) -> dict | None:
    user = _USER_VALIDATE.format(
        language=language or "english",
        article=article[:3500] or "(no text)",
        overall=overall_txt,
        verdict=provisional,
        confidence=confidence,
        claims="\n".join(rows) if rows else "(no claims extracted)",
    )
    obj = _call_gemini(_SYSTEM_VALIDATE, user, temperature=0.1)
    if not obj:
        return None
    label = _normalize_final_label(obj.get("label") or obj.get("verdict")
                                   or obj.get("decision"))
    if not label:
        return None
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    agrees = bool(obj.get("agrees", obj.get("agrees_with_first", False)))
    if "agrees" not in obj and "agrees_with_first" not in obj:
        agrees = (label == _normalize_final_label(provisional))
    return {
        "available": True,
        "source": "gemini",
        "model": GEMINI_MODEL,
        "label": label,
        "confidence": round(conf, 3),
        "agrees": agrees,
        "reasoning": str(obj.get("reasoning", ""))[:600],
    }
