"""Builds ``lm_data.json``: a compact bigram language model from the
human-written (REAL) half of the news corpus, plus calibration statistics.

The model powers the perplexity proxy inside the AI-text detector. Building
it from the real-news corpus keeps the detector fully offline.

Usage:
    python ml/ai_text_detector/build_lm.py
    python ml/ai_text_detector/build_lm.py --min-count 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset import load_dataset  # noqa: E402
from lm import LM_DATA_PATH, BigramLM, tokenize  # noqa: E402

TOP_UNIGRAMS = 20000
TOP_BIGRAMS = 120000


def build(min_count: int) -> dict:
    df = load_dataset()
    real = df[df["label"] == "REAL"]
    if len(real) == 0:
        raise SystemExit("No REAL-labelled documents found in the dataset.")

    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    total_tokens = 0
    for doc in real["text"]:
        toks = tokenize(doc)
        if not toks:
            continue
        total_tokens += len(toks)
        unigrams.update(toks)
        for prev, word in zip(toks, toks[1:]):
            bigrams[f"{prev} {word}"] += 1

    top_uni = {w: c for w, c in unigrams.most_common(TOP_UNIGRAMS) if c >= min_count}
    top_bi = {g: c for g, c in bigrams.most_common(TOP_BIGRAMS) if c >= min_count}

    # Calibrate the predictability component on the real corpus.
    lm = BigramLM(
        {
            "unigrams": top_uni,
            "bigrams": top_bi,
            "total_tokens": total_tokens,
        }
    )
    alps = [
        lp
        for doc in real["text"]
        if (lp := lm.avg_log_prob(doc)) is not None
    ]
    ref_mean = statistics.mean(alps) if alps else -7.5
    ref_sd = statistics.stdev(alps) if len(alps) > 1 else 1.2

    return {
        "unigrams": top_uni,
        "bigrams": top_bi,
        "total_tokens": total_tokens,
        "docs": int(len(real)),
        "reference_log_prob": round(ref_mean, 6),
        "reference_sd": round(ref_sd, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AI-text detector bigram LM.")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum n-gram count to keep (default 2).")
    args = parser.parse_args()

    data = build(args.min_count)
    LM_DATA_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(
        f"Wrote {LM_DATA_PATH} "
        f"({len(data['unigrams']):,} unigrams, {len(data['bigrams']):,} bigrams, "
        f"{data['total_tokens']:,} tokens from {data['docs']:,} real docs; "
        f"ref log-prob {data['reference_log_prob']:.3f} ± {data['reference_sd']:.3f})"
    )


if __name__ == "__main__":
    main()
