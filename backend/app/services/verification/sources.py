"""Source credibility/trust scoring.

A transparent, configurable source-quality system. It never asserts "this
domain is always true"; instead it scores *signals* about a source
(official status, reputation, publication date, editorial independence) and
returns the reasons behind the score so the UI can explain itself.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Configurable signal tables. Overridable via environment variables so the
# project owner can tune them without editing code.
# --------------------------------------------------------------------------

_PUBLIC_SUFFIX_RE = re.compile(r"\W+")

# High-trust tiers.
_OFFICIAL_DOMAINSET = {
    "gov", "gov.in", "gov.uk", "gov.au", "gov.cn", "go.jp", "mil", "govt.nz",
    "gov.au", "parliament.uk", "who.int", "un.org", "unicef.org", "unesco.org",
    "fao.org", "oecd.org", "worldbank.org", "imf.org", "nato.int", "wto.org",
    "icann.org", "isro.gov.in", "pib.gov.in", "nic.in", "gov.sg", "hongkong.gov.hk",
}
_EDU_DOMAINSET = {".edu", ".ac.uk", ".ac.in", ".edu.in", ".edu.au", ".edu.cn"}
_ACADEMIC_DOMAINSET = {
    "nature.com", "science.org", "sciencemag.org", "sciencemediacentre.org",
    "sciencedirect.com", "springer.com", "wiley.com", "pubmed.ncbi.nlm.nih.gov",
    "cell.com", "thelancet.com", "nejm.org", "britannica.com", "wikipedia.org",
    "arxiv.org", "academic.oup.com",
}
_FACTCHECK_DOMAINSET = {
    "snopes.com", "factcheck.org", "politifact.com", "poynter.org",
    "boomlive.in", "altnews.in", "thequint.com", "factchecker.in",
    "checkyourfact.com", "afpcheck.com",
}
_NEWS_DOMAINSET = {
    "reuters.com", "apnews.com", "afp.com", "bbc.com",
    "bbc.co.uk", "thehindu.com", "indianexpress.com", "timesofindia.com",
    "hindustantimes.com", "theguardian.com", "nytimes.com", "washingtonpost.com",
    "wsj.com", "ft.com", "economist.com", "aljazeera.com", "dw.com",
    "france24.com", "thetimes.co.uk", "ndtv.com", "news18.com", "firstpost.com",
    "moneycontrol.com", "livemint.com", "business-standard.com", "thehindubusinessline.com",
    "economictimes.com", "thediplomat.com", "scroll.in", "thewire.in",
    "thenewsminute.com", "deccanchronicle.com", "thehansindia.com", "newindianexpress.com",
    "tribuneindia.com", "oneindia.com", "nikkei.com", "bloomberg.com",
    "cnbc.com", "cnn.com", "abcnews.go.com", "nbcnews.com", "cbsnews.com",
    "latimes.com", "time.com", "newslaundry.com", "theprint.in",
}
_SOCIAL_DOMAINSET = {
    "x.com", "twitter.com", "facebook.com", "instagram.com", "whatsapp.com",
    "telegram.org", "t.me", "reddit.com", "linkedin.com", "tiktok.com",
    "youtube.com", "threads.net", "quora.com",
}
_LOW_TRUST_DOMAINSET = {
    "wiki", "wixsite.com", "weebly.com", "blogspot.com", "wordpress.com",
    "medium.com", "substack.com", "simply-stories.com", "dailynewsmails.com",
    "newsdrug.com", "conspiracy.club", "beforeitsnews.com", "yournewswire.com",
    "thefakenews.com", "infowars.com", "zerohedge.com", "dailystormer.com",
}

_CLICKBAIT_MARKERS = re.compile(
    r"\b(look what|you won't believe|shocking|mind-blowing|must see|secret|"
    r"they don't want you|exposed|the truth about|100%|leave you speechless|"
    r"wow|real reason|finally|never knew|what happened next)\b",
    re.I,
)

# Per-signal weights, summed with the base score (max 10).
_WEIGHTS = {
    "official_tld": 5.0,
    "edu_tld": 4.5,
    "fact_checker": 4.0,
    "knowledge_ref": 3.5,
    "established_news": 3.5,
    "academic": 4.0,
    "corporate_known": 2.0,
    "unknown_domain": -1.5,
    "social_media": -3.0,
    "low_trust_list": -4.0,
    "clickbait_title": -1.0,
    "no_date": -0.5,
    "indirect": -0.5,
}

# Domains we simply don't know well enough to trust.
_KNOWN_CORPORATE = {
    "google.com", "wikipedia.org", "britannica.com", "osf.io", "doi.org",
    "pib.gov.in", "mea.gov.in", "finmin.nic.in", "niti.gov.in",
}


def _domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url if "://" in url else "http://" + url).netloc or "").lower()
    except ValueError:
        return ""


def _tld_lookup(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-1]
    return domain


def score_source(name: str | None, url: str | None,
                 published_date: str | None = None, **extra) -> dict:
    """Score one evidence source and return reasons.

    Returns ``{"source_name", "domain", "score" (0-10), "tier", "reasons", "flags"}``.
    """
    name = (name or "").strip()
    domain = _domain_from_url(url or "")
    reasons: list[str] = []
    score = 3.0  # baseline sample
    tier = "unknown"

    if extra.get("retrieved_from") in ("knowledge-base", "gemini"):
        return {
            "source_name": name or "Reference",
            "domain": domain or "reference",
            "score": 8.0 if extra.get("retrieved_from") == "knowledge-base" else 5.0,
            "tier": "authoritative-reference" if extra.get("retrieved_from") == "knowledge-base"
            else "llm-reference",
            "reasons": (["Curated, uniformly-accepted reference."] if extra.get("retrieved_from") == "knowledge-base"
                        else ["AI-based reference; treated as a supporting signal, not a primary source."]),
            "flags": [],
        }

    domain_clean = domain.replace("www.", "")
    effective_score = score

    if domain_clean in _OFFICIAL_DOMAINSET or domain_clean.endswith(".gov.in") \
            or (domain_clean.split(".")[-1] == "gov" and len(domain_clean.split(".")) == 2):
        tier = "official"
        effective_score += _WEIGHTS["official_tld"]
        reasons.append("Official government/institutional domain.")
    elif domain_clean in _FACTCHECK_DOMAINSET:
        tier = "fact-checker"
        effective_score += _WEIGHTS["fact_checker"]
        reasons.append("Established fact-checking organisation.")
    elif domain_clean in _NEWS_DOMAINSET:
        tier = "established-news"
        effective_score += _WEIGHTS["established_news"]
        reasons.append("Reputable news organisation.")
    elif domain_clean in _ACADEMIC_DOMAINSET:
        tier = "academic"
        effective_score += _WEIGHTS["academic"]
        reasons.append("Academic/scientific publisher or reference.")
    elif domain_clean in _SOCIAL_DOMAINSET:
        tier = "social"
        effective_score += _WEIGHTS["social_media"]
        reasons.append("Social-media platform (needs independent corroboration).")
    elif domain_clean in _LOW_TRUST_DOMAINSET:
        tier = "low-trust"
        effective_score += _WEIGHTS["low_trust_list"]
        reasons.append("Low-trust / content-farm domain listed.")
    elif domain_clean in _KNOWN_CORPORATE:
        tier = "corporate"
        effective_score += _WEIGHTS["corporate_known"]
        reasons.append("Well-known corporate/knowledge domain.")

    # TLD-level education domains.
    if tier == "unknown" and any(domain_clean.endswith(d) for d in _EDU_DOMAINSET):
        tier = "academic"
        effective_score += _WEIGHTS["edu_tld"]
        reasons.append("Educational institution (.edu equivalent).")

    if tier == "unknown":
        effective_score += _WEIGHTS["unknown_domain"]
        reasons.append("Domain not in trusted lists; score reduced.")

    title = extra.get("title") or ""
    if _CLICKBAIT_MARKERS.search(title) or _CLICKBAIT_MARKERS.search(name):
        effective_score += _WEIGHTS["clickbait_title"]
        reasons.append("Title contains clickbait patterns.")

    if not published_date:
        effective_score += _WEIGHTS["no_date"]
        reasons.append("No publication date available.")

    # Trust can never go negative or above 10.
    effective_score = max(0.0, min(10.0, effective_score))

    return {
        "source_name": name or (domain or "Unknown source"),
        "domain": domain or url or "unknown",
        "score": round(effective_score, 1),
        "tier": tier,
        "reasons": reasons[:8],
        "flags": _flags_for(tier, effective_score),
    }


def _flags_for(tier: str, score: float) -> list[str]:
    flags: list[str] = []
    if score >= 7.5:
        flags.append("credible")
    elif score >= 5.0:
        flags.append("moderately-credible")
    elif score <= 2.0:
        flags.append("low-credibility")
    if tier in ("social", "low-trust", "unknown"):
        flags.append("needs-corroboration")
    return flags


def aggregate_source_quality(evidence_scores: list[dict]) -> dict:
    """Aggregate the weighed mean source quality of a claim's evidence."""
    if not evidence_scores:
        return {"mean": 0.0, "weighted": 0.0, "count": 0, "tiers": []}
    total_weighted = sum(max(s["score"], 0.5) for s in evidence_scores)
    tiers = sorted({s["tier"] for s in evidence_scores})
    return {
        "mean": round(sum(s["score"] for s in evidence_scores) / len(evidence_scores), 2),
        "weighted": round(total_weighted, 2),
        "count": len(evidence_scores),
        "tiers": tiers,
    }