"""Pure-NumPy neural network runtime.

This is the *same* architecture that ``ml/train.py`` trains in PyTorch
(Embedding -> BiGRU -> Dense -> ReLU -> Output). Weights are exported to
``weights.npz`` and the forward pass is re-implemented with NumPy so the
deployed backend does not need PyTorch - the API stays small enough for a
serverless function and is fast to cold-start.

Parity: ``ml/train.py`` runs a torch-vs-numpy output comparison after training
and fails loudly if the two disagree beyond a tiny tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyBiGRU:
    """Embedding -> bidirectional GRU -> dense -> softmax, in pure NumPy.

    GRU gate order matches PyTorch (r, z, n). Class order is
    ``["FAKE", "REAL"]`` so ``proba[:, 1]`` is ``p_real``.
    """

    def __init__(self, weights: dict[str, np.ndarray], config: dict) -> None:
        self.config = dict(config)
        self.embedding = np.asarray(weights["embedding"], dtype=np.float32)
        g = _GruParams.from_weights(weights, "gru_f")
        self._gru_f = g
        self._gru_b = _GruParams.from_weights(weights, "gru_b")
        self.dense_w = np.asarray(weights["dense_w"], dtype=np.float32)
        self.dense_b = np.asarray(weights["dense_b"], dtype=np.float32)
        self.out_w = np.asarray(weights["out_w"], dtype=np.float32)
        self.out_b = np.asarray(weights["out_b"], dtype=np.float32)

    @staticmethod
    def _gru_step(x: np.ndarray, h: np.ndarray, g: "_GruParams") -> np.ndarray:
        hdim = g.w_hh.shape[1]
        xg = x @ g.w_ih.T + g.b_ih           # (3h,) input part
        hg = h @ g.w_hh.T + g.b_hh           # (3h,) hidden part
        r = 1.0 / (1.0 + np.exp(-(xg[:hdim] + hg[:hdim])))
        z = 1.0 / (1.0 + np.exp(-(xg[hdim : 2 * hdim] + hg[hdim : 2 * hdim])))
        n = np.tanh(xg[2 * hdim :] + r * hg[2 * hdim :])
        return (1.0 - z) * n + z * h

    def _forward_one(self, ids) -> np.ndarray:
        """Run the BiGRU over a single padded id sequence; return logits.

        Only real tokens (before trailing padding) are processed - running the
        GRU over the fixed padding vector hundreds of times would wash the
        final hidden state toward a padding-only fixed point.
        """
        ids_np = np.asarray(ids)
        nz = np.flatnonzero(ids_np)
        last = int(nz.max()) + 1 if nz.size else 0
        seq = self.embedding[ids_np[:last]]    # (last, E)
        hdim = self._gru_f.w_hh.shape[1]
        h_f = np.zeros(hdim, dtype=np.float32)
        h_b = np.zeros(hdim, dtype=np.float32)
        for t in range(0, last):
            h_f = self._gru_step(seq[t], h_f, self._gru_f)
        for t in range(last - 1, -1, -1):
            h_b = self._gru_step(seq[t], h_b, self._gru_b)
        vec = np.concatenate([h_f, h_b])     # (2h,)
        vec = self.dense_w @ vec + self.dense_b
        vec = np.maximum(vec, 0.0)           # ReLU
        return self.out_w @ vec + self.out_b

    def proba(self, ids) -> tuple[float, float]:
        """Return (p_real, p_fake) via softmax over the two logits."""
        logits = self._forward_one(ids)
        mx = logits.max()
        e = np.exp(logits - mx)              # stable
        p = e / e.sum()
        return float(p[1]), float(p[0])

    def _token_influence(self, ids) -> np.ndarray:
        """Per-token influence = |delta hidden| in fwd + bwd directions."""
        ids_np = np.asarray(ids)
        nz = np.flatnonzero(ids_np)
        last = int(nz.max()) + 1 if nz.size else 0
        seq = self.embedding[ids_np[:last]]    # (last, E)
        hdim = self._gru_f.w_hh.shape[1]
        t = seq.shape[0]

        fwd_jump = np.zeros(t, dtype=np.float32)
        h = np.zeros(hdim, dtype=np.float32)
        for i in range(t):
            h_new = self._gru_step(seq[i], h, self._gru_f)
            fwd_jump[i] = float(np.linalg.norm(h_new - h))
            h = h_new

        bwd_jump = np.zeros(t, dtype=np.float32)
        h = np.zeros(hdim, dtype=np.float32)
        for i in range(t - 1, -1, -1):
            h_new = self._gru_step(seq[i], h, self._gru_b)
            bwd_jump[i] = float(np.linalg.norm(h_new - h))
            h = h_new

        return fwd_jump + bwd_jump

    def keywords(self, text: str | None, vocab: "object", max_len: int,
                 top_n: int = 10) -> list[dict]:
        """Top vocabulary-weighted keywords for *text* (token influence)."""
        from .nn_tokenize import tokenize

        from .preprocess import STOP_WORDS

        words = tokenize(text)
        if not words:
            return []
        ids = vocab.ids_of(text, max_len)
        influence = self._token_influence(ids)

        scores: dict[str, float] = {}
        for i, word in enumerate(words[:max_len]):
            if len(word) < 3 or word in STOP_WORDS:
                continue
            scores[word] = scores.get(word, 0.0) + float(influence[i])

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [
            {"term": word, "score": round(score, 4)}
            for word, score in ranked[:top_n]
        ]


class _GruParams:
    __slots__ = ("w_ih", "w_hh", "b_ih", "b_hh")

    def __init__(self, w_ih, w_hh, b_ih, b_hh) -> None:
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    @classmethod
    def from_weights(cls, weights: dict, prefix: str) -> "_GruParams":
        return cls(
            np.asarray(weights[f"{prefix}_ih"], dtype=np.float32),
            np.asarray(weights[f"{prefix}_hh"], dtype=np.float32),
            np.asarray(weights[f"{prefix}_b_ih"], dtype=np.float32),
            np.asarray(weights[f"{prefix}_b_hh"], dtype=np.float32),
        )


def build_weights_from_torch(state: dict) -> dict[str, np.ndarray]:
    """Convert a trained torch module's state dict into NumPy arrays.

    Expects the exact parameter names produced by ``ml.train.BiGRUNet``.
    """
    out: dict[str, np.ndarray] = {"embedding": np.asarray(
        state["embedding.weight"].detach().cpu().numpy(), dtype=np.float32)}

    for prefix, target in (("gru_f", "gru_f"), ("gru_b", "gru_b")):
        out[f"{target}_ih"] = np.asarray(
            state[f"{prefix}.weight_ih"].detach().cpu().numpy(), dtype=np.float32)
        out[f"{target}_hh"] = np.asarray(
            state[f"{prefix}.weight_hh"].detach().cpu().numpy(), dtype=np.float32)
        out[f"{target}_b_ih"] = np.asarray(
            state[f"{prefix}.bias_ih"].detach().cpu().numpy(), dtype=np.float32)
        out[f"{target}_b_hh"] = np.asarray(
            state[f"{prefix}.bias_hh"].detach().cpu().numpy(), dtype=np.float32)

    out["dense_w"] = np.asarray(state["dense.weight"].detach().cpu().numpy(), dtype=np.float32)
    out["dense_b"] = np.asarray(state["dense.bias"].detach().cpu().numpy(), dtype=np.float32)
    out["out_w"] = np.asarray(state["out.weight"].detach().cpu().numpy(), dtype=np.float32)
    out["out_b"] = np.asarray(state["out.bias"].detach().cpu().numpy(), dtype=np.float32)
    return out


def load_model(model_dir: str | Path) -> NumpyBiGRU:
    """Load config + weights from ``models/fake_news_neural_network/``."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weights = np.load(model_dir / "weights.npz")
    return NumpyBiGRU(dict(weights), config)


def load_vocab(path: str | Path):
    from .nn_tokenize import Vocab

    return Vocab(json.loads(Path(path).read_text(encoding="utf-8")))