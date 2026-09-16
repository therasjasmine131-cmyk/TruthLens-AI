"""Model manager (pure-NumPy neural network).

Loads the trained neural-network artifacts from ``models/fake_news_neural_network``
(``weights.npz``, ``vocab.json``, ``config.json``, ``model_metadata.json``) once
and serves predictions with a pure-NumPy BiGRU forward pass - no PyTorch, no
scikit-learn. The loader is lazy so the API can boot (and report health) even
when artifacts are missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config


@dataclass
class ModelBundle:
    model: object
    vocab: object
    config: dict
    metadata: dict
    load_error: str | None = None
    _loaded: bool = False


class ModelManager:
    def __init__(self, nn_dir: str | Path | None = None) -> None:
        self.nn_dir = Path(nn_dir or Config.NN_MODEL_DIR)
        self.bundle: ModelBundle | None = None

    # ---- loading ---------------------------------------------------------
    def load(self) -> ModelBundle | None:
        """Load (or reload) the NumPy artifacts. Returns the bundle or None."""
        weights_path = self.nn_dir / "weights.npz"
        vocab_path = self.nn_dir / "vocab.json"
        config_path = self.nn_dir / "config.json"
        metadata_path = self.nn_dir / "model_metadata.json"

        if not all(p.exists() for p in (weights_path, vocab_path, config_path, metadata_path)):
            self.bundle = None
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            weights = np.load(weights_path)

            from ml.nn_forward import NumpyBiGRU
            from ml.nn_tokenize import Vocab

            model = NumpyBiGRU(dict(weights), config)
            vocab = Vocab(json.loads(vocab_path.read_text(encoding="utf-8")))
            self.bundle = ModelBundle(
                model=model,
                vocab=vocab,
                config=config,
                metadata=metadata,
                _loaded=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.bundle = ModelBundle(
                model=None, vocab=None, config={}, metadata={}, load_error=str(exc)
            )
        return self.bundle

    @property
    def ready(self) -> bool:
        if self.bundle is None:
            return False
        return self.bundle._loaded

    def metadata(self) -> dict:
        if not self.ready:
            return {}
        return self.bundle.metadata

    def max_len(self) -> int:
        return int(self.bundle.config.get("max_len", 400)) if self.ready else 400

    # ---- prediction ------------------------------------------------------
    def predict_proba(self, text: str):
        """Return (p_real, p_fake) or raise when model is unavailable."""
        if not self.ready:
            raise RuntimeError("ML model is not available. Train the model first.")
        ids = self.bundle.vocab.ids_of(text, self.max_len())
        return self.bundle.model.proba(ids)

    def top_keywords(self, text: str, top_n: int = 10) -> list[dict]:
        """Neural keywords: per-token hidden-state influence for *text*.

        Ranks words by how much the BiGRU hidden state changes when that token
        is processed (forward + backward directions) - the model's own notion
        of which terms moved its REAL/FAKE signal.
        """
        if not self.ready:
            return []
        return self.bundle.model.keywords(text, self.bundle.vocab, self.max_len(), top_n)

    def status(self) -> dict:
        if self.ready:
            return {"status": "online", "model": self.bundle.metadata.get("best_model")}
        if self.bundle and self.bundle.load_error:
            return {"status": "error", "error": self.bundle.load_error}
        return {"status": "offline", "error": "artifacts not found"}


model_manager = ModelManager()