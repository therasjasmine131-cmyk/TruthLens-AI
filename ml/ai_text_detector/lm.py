"""Compact smoothed word-bigram language model used as a perplexity proxy.

LLM-generated text is typically *more predictable* (lower perplexity) than
human-written text. We estimate predictability with a small bigram model
trained on the human-written (REAL) news corpus (see ``build_lm.py``) so the
AI-text detector has zero heavyweight dependencies.

This is deliberately an approximation, not a scientific language model: it
uses only the most frequent unigrams/bigrams, smooths with a Laplace-style
mass, and knows nothing about syntax or semantics.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

LM_DATA_PATH = Path(__file__).resolve().parent / "lm_data.json"

_WORD_RE = re.compile(r"[a-z0-9]+")
OOV_FLOOR = 1e-7  # floor probability for out-of-vocabulary tokens
MIN_TOKENS = 8  # below this, log-probability estimates are unstable


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens (news text is largely ASCII)."""
    return _WORD_RE.findall(text.lower())


class BigramLM:
    """Laplace-smoothed bigram model over a fixed vocabulary."""

    def __init__(self, data: dict | None = None) -> None:
        data = data or {}
        self.unigrams: dict[str, int] = data.get("unigrams", {})
        self.bigrams: dict[str, int] = data.get("bigrams", {})
        self.total_tokens = int(data.get("total_tokens", 0))
        self.vocab_size = len(self.unigrams)
        self.vocab = set(self.unigrams)
        # Calibration reference computed over the training corpus.
        self.ref_mean = float(data.get("reference_log_prob", -7.5))
        self.ref_sd = float(data.get("reference_sd", 1.2))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BigramLM":
        path = Path(path) if path else LM_DATA_PATH
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return cls(data)

    @property
    def available(self) -> bool:
        return self.total_tokens > 0 and self.vocab_size > 0

    def _uni_prob(self, word: str) -> float:
        if word in self.vocab:
            return (self.unigrams[word] + 1.0) / (self.total_tokens + self.vocab_size)
        return OOV_FLOOR

    def _bi_prob(self, prev: str, word: str) -> float:
        key = f"{prev} {word}"
        if key in self.bigrams:
            return (self.bigrams[key] + 1.0) / (self.unigrams.get(prev, 0) + self.vocab_size)
        if prev in self.vocab:
            # Unseen continuation: interpolate toward the unigram.
            return self._uni_prob(word) * 0.5
        return self._uni_prob(word)

    def avg_log_prob(self, text: str) -> float | None:
        """Mean per-token log-probability; higher = more predictable = more
        AI-like under this proxy. Returns ``None`` when unusable."""
        if not self.available:
            return None
        tokens = tokenize(text)
        if len(tokens) < MIN_TOKENS:
            return None
        total = math.log(self._uni_prob(tokens[0]))
        for prev, word in zip(tokens, tokens[1:]):
            total += math.log(max(self._bi_prob(prev, word), OOV_FLOOR))
        return total / len(tokens)

    def perplexity(self, text: str) -> float | None:
        alp = self.avg_log_prob(text)
        return math.exp(-alp) if alp is not None else None

    def predictability_component(self, text: str) -> float:
        """Map the log-probability to a 0..1 'predictability' score centred on
        the human-writing reference: 0.5 = typical human, >0.5 = more
        predictable (AI-like), <0.5 = less predictable (human-like)."""
        alp = self.avg_log_prob(text)
        if alp is None:
            return 0.5
        shift = (alp - self.ref_mean) / max(self.ref_sd, 1e-6)
        return max(0.0, min(1.0, 0.5 + 0.5 * max(-1.0, min(1.0, shift / 1.5))))
