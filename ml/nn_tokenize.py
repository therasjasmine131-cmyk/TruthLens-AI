"""Word tokenizer + vocabulary for the neural network classifier.

Both the PyTorch trainer (``ml/train.py``) and the pure-NumPy runtime
(``ml/nn_forward.py`` / ``backend/app/ml/model_manager.py``) share this module
so training and inference tokenize identically. No external NLP libraries.

Special tokens: ``<PAD>`` (id 0), ``<UNK>`` (id 1).
"""

from __future__ import annotations

import json
from pathlib import Path

from .preprocess import clean_for_features

PAD = "<PAD>"
UNK = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def tokenize(text: str | None) -> list[str]:
    """Return the lower-cased, punctuation-free word list for a document."""
    if not text:
        return []
    return clean_for_features(text).split()


def build_vocab(texts: list[str], max_words: int = 30000, min_count: int = 2) -> "Vocab":
    """Build a frequency-ranked vocabulary over the corpus.

    Only words appearing at least ``min_count`` times are kept (capped at
    ``max_words``), so the embedding table stays compact.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    words = [w for w, c in counter.items() if c >= min_count]
    words.sort(key=lambda w: (-counter[w], w))
    words = words[: max(0, max_words)]

    word_to_id = {PAD: PAD_ID, UNK: UNK_ID}
    word_to_id.update({w: i + 2 for i, w in enumerate(words)})
    return Vocab(word_to_id)


def load_vocab(path: str | Path) -> "Vocab":
    """Load a vocab saved by :meth:`Vocab.save`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Vocab(data)


class Vocab:
    """Word <-> integer id mapping with JSON persistence."""

    def __init__(self, word_to_id: dict[str, int]) -> None:
        self.word_to_id = dict(word_to_id)

    def __len__(self) -> int:
        return len(self.word_to_id)

    def size(self) -> int:
        return len(self.word_to_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.word_to_id, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def id_of(self, word: str) -> int:
        return self.word_to_id.get(word, UNK_ID)

    def ids_of(self, text: str | None, max_len: int) -> list[int]:
        """Encode *text* into integer ids, truncated/padded to ``max_len``."""
        ids = [self.id_of(w) for w in tokenize(text)]
        if len(ids) < max_len:
            ids += [PAD_ID] * (max_len - len(ids))
        return ids[:max_len]