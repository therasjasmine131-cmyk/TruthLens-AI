"""End-to-end evaluation of the evidence-based verification system.

Builds a small, labelled set of claims (verifiable locally via the knowledge
base + opinion/prediction handling), runs each through
``verify_text`` and reports the agreement rate of the REAL / FALSE /
UNVERIFIED verdicts against expected labels.

Usage:
    python ml/evaluate_claims.py
    python ml/evaluate_claims.py --json   # print a machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.verification.verifier import verify_text  # noqa: E402

#: (text, expected verdict, note)
#: "REALmean" asserts the overall verdict must be exactly REAL/FALSE or at least
#: the marked claim must appear; we simply require the overall verdict to equal
#: `expected` for these hand-picked local-only cases.
CASES = [
    # Classic verified-science facts (knowledge base -> SUPPORTS)
    ("The Earth revolves around the Sun.", "REAL", "science"),
    ("The Pacific Ocean is the largest ocean on Earth.", "REAL", "geography"),
    ("India has 28 states.", "REAL", "geography"),
    ("The human body has 206 bones in adults.", "REAL", "health"),
    # Known scam / misinformation patterns (knowledge base -> CONTRADICTS)
    ("Every college student will receive 50000 rupees every month.", "FALSE", "scam"),
    (
        "The government announced that every college student will receive "
        "50000 every month under a new scheme.",
        "FALSE",
        "scam-with-attribution",
    ),
    ("Every Indian citizen will receive 10000 rupees from a new scheme.", "FALSE", "generic-scheme"),
    # Opinions / predictions must never be confirmed as facts
    ("I think this politician is the best.", "UNVERIFIED", "opinion"),
    ("India will win the next World Cup.", "UNVERIFIED", "prediction"),
    ("This year's monsoon will be above average.", "UNVERIFIED", "prediction-2"),
    # Unverifiable local-only content stays honest
    ("RBI cuts the repo rate by 100 basis points this week.", "UNVERIFIED", "unverifiable"),
    ("TN la school 15 days close ah?", "UNVERIFIED", "tanglish-unknown"),
]


def evaluate() -> dict:
    rows = []
    correct = 0
    for text, expected, note in CASES:
        report = verify_text(text)
        actual = report["overall"]["verdict"]
        ok = actual == expected
        correct += int(ok)
        rows.append(
            {
                "claim": text[:90],
                "expected": expected,
                "actual": actual,
                "ok": ok,
                "note": note,
                "claims_extracted": len(report["claims"]),
                "evidence_items": report["pipeline"]["evidence_items"],
            }
        )
        marker = "OK " if ok else "X"
        print(
            f"[{marker}] expected={expected:10s} actual={actual:10s} "
            f"ev={rows[-1]['evidence_items']:2d}  ({note})  {text[:70]}"
        )

    accuracy = correct / max(1, len(rows))
    print(f"\nEnd-to-end verdict agreement: {correct}/{len(rows)} = {accuracy:.2%}")
    return {
        "cases": rows,
        "total": len(rows),
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the verification system")
    parser.add_argument("--json", action="store_true", help="print JSON summary")
    args = parser.parse_args()
    result = evaluate()
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()