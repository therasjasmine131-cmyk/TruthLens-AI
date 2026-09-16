"""TruthLens AI - neural network training pipeline.

Reproducible PyTorch training of the BiGRU classifier. The trained weights are
exported to ``models/fake_news_neural_network/`` together with ``vocab.json``,
``config.json`` and ``metrics.json`` so the API can serve predictions with pure
NumPy (no torch required in production).

Usage::

    python ml/train.py                        # full ISOT, all defaults
    python ml/train.py --max-len 360 --epochs 12 --batch-size 128
    python ml/train.py --limit 5000           # fast smoke run on a slice
    python ml/train.py --sample               # force the bundled sample

PyTorch is a *training-time* dependency only (``pip install torch``); it is
NOT required to run the backend or the NumPy forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import LABEL_FAKE, LABEL_REAL, SAMPLE_CSV, group_split, load_dataset  # noqa: E402
from ml.nn_forward import build_weights_from_torch  # noqa: E402
from ml.nn_tokenize import Vocab, build_vocab  # noqa: E402

try:  # torch is a dev dependency used only by this training script
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - only hit when torch is missing
    raise SystemExit(
        "PyTorch is required to train the model:  pip install torch"
    ) from None

MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "fake_news_neural_network"
SEED = 42
CLASSES = [LABEL_FAKE, LABEL_REAL]  # index 0 = FAKE, index 1 = REAL


class BiGRUNet(nn.Module):
    """Embedding -> bidirectional GRU -> dense -> ReLU -> 2-class output."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 dense_dim: int, num_classes: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru_f = nn.GRUCell(embed_dim, hidden_dim)
        self.gru_b = nn.GRUCell(embed_dim, hidden_dim)
        self.dense = nn.Linear(hidden_dim * 2, dense_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(dense_dim, num_classes)

    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)                # (B, T, E) with trailing padding
        # only process real tokens; running the GRU over hundreds of padding
        # steps collapses the final hidden state to a padding fixed point
        tmax = int(x.ne(0).sum(dim=1).max().clamp_min(1).item())
        h = torch.zeros(x.size(0), self.gru_f.hidden_size, device=x.device)
        for t in range(tmax):
            h = self.gru_f(emb[:, t], h)
        hb = torch.zeros(x.size(0), self.gru_b.hidden_size, device=x.device)
        for t in range(tmax - 1, -1, -1):
            hb = self.gru_b(emb[:, t], hb)
        v = torch.cat([h, hb], dim=1)
        v = torch.relu(self.dense(v))
        v = self.drop(v)
        return self.out(v)


def _torch_model(vocab_size, embed_dim, hidden_dim, dense_dim, dropout) -> BiGRUNet:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return BiGRUNet(vocab_size, embed_dim, hidden_dim, dense_dim, dropout=dropout)





def _evaluate(model: BiGRUNet, ids: np.ndarray, labels: np.ndarray, batch_size: int = 256) -> dict:
    """Honest metrics on held-out data (no leakage, real predictions)."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(ids), batch_size):
            batch = torch.from_numpy(ids[i:i + batch_size]).long()
            logits = model(batch)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(p)
            all_y.append(labels[i:i + batch_size])
    prob = np.concatenate(all_probs)
    y = np.concatenate(all_y)

    pred = prob.argmax(axis=1)
    p_real = prob[:, 1]
    y_real = (y == LABEL_REAL).astype(int)

    # ---- metrics (REAL is the positive class, matching the app) ----
    tp = int(((pred == 1) & (y_real == 1)).sum())
    fp = int(((pred == 1) & (y_real == 0)).sum())
    fn = int(((pred == 0) & (y_real == 1)).sum())
    tn = int(((pred == 0) & (y_real == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / max(1, len(y))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # ROC-AUC via rank statistic (REAL positive)
    order = np.argsort(p_real)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(p_real))
    n_pos = int(y_real.sum())
    n_neg = len(y_real) - n_pos
    if n_pos and n_neg:
        auc = (ranks[y_real == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = 0.5

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
        "support": int(len(y)),
        "p_real_mean": round(float(p_real.mean()), 4),
    }


class _NpBatcher:
    """Feeds padded, constant-length ids to the torch model during training."""

    def __init__(self, ids: np.ndarray, labels: np.ndarray, batch_size: int) -> None:
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.order = np.random.default_rng(SEED).permutation(len(ids)).copy()

    def __len__(self) -> int:
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __iter__(self):
        for start in range(0, len(self.order), self.batch_size):
            idx = self.order[start:start + self.batch_size]
            yield torch.from_numpy(self.ids[idx]).long(), torch.from_numpy(self.labels[idx])


def train(args) -> None:
    t0 = time.time()

    if args.sample:
        df = pd.read_csv(SAMPLE_CSV)
        df.attrs["source"] = f"Bundled sample ({len(df)} rows)"
        source = df.attrs["source"]
    else:
        df = load_dataset()
        source = df.attrs.get("source", "unknown")
        print(f"[dataset] {source}")
        df = df.drop_duplicates(subset=["headline", "text"])

    df_train, df_val, df_test = group_split(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    if args.limit:
        df_train = df_train.head(args.limit)
    print(f"[dataset] {source}  ({len(df)} rows, {len(df_train)} train / "
          f"{len(df_val)} val / {len(df_test)} test)")

    train_texts = (df_train["headline"] + " " + df_train["text"]).tolist()
    vocab = build_vocab(train_texts, max_words=args.max_words, min_count=args.min_count)

    def _encode(sub):  # (n, max_len) int32 padded arrays
        arr = np.zeros((len(sub), args.max_len), dtype=np.int64)
        for i, txt in enumerate((sub["headline"] + " " + sub["text"]).tolist()):
            ids = vocab.ids_of(txt, args.max_len)
            arr[i] = ids
        return arr.copy()

    ids_train = _encode(df_train)
    ids_val = _encode(df_val)
    ids_test = _encode(df_test)
    label_train = df_train["label"].map({LABEL_REAL: 1, LABEL_FAKE: 0}).to_numpy(dtype=np.int64).copy()
    label_val = df_val["label"].map({LABEL_REAL: 1, LABEL_FAKE: 0}).to_numpy(dtype=np.int64).copy()
    label_test = df_test["label"].map({LABEL_REAL: 1, LABEL_FAKE: 0}).to_numpy(dtype=np.int64).copy()

    model = _torch_model(vocab.size(), args.embed_dim, args.hidden_dim,
                         args.dense_dim, args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[device] {device}   [max_len] {args.max_len}   [vocab] {vocab.size()}")

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for bx, by in _NpBatcher(ids_train, label_train, args.batch_size):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batch += 1

        val_loss, val_acc = _valid(model, ids_val, label_val, device, args.batch_size)
        print(f"[epoch {epoch:02d}] loss={ep_loss / n_batch:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early stop] no improvement for {patience} epochs")
                break

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, ids_test, label_test, args.batch_size)
    val_metrics = _evaluate(model, ids_val, label_val, args.batch_size)
    print(f"\n[test] accuracy={test_metrics['accuracy']:.4f} precision={test_metrics['precision']:.4f} "
          f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} "
          f"roc_auc={test_metrics['roc_auc']:.4f}")
    print(f"[test confusion matrix] {test_metrics['confusion_matrix']}")

    # ---- export -----------------------------------------------------
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    weights = build_weights_from_torch(model.state_dict())
    np.savez(
        MODEL_ROOT / "weights.npz",
        embedding=weights["embedding"],
        gru_f_ih=weights["gru_f_ih"], gru_f_hh=weights["gru_f_hh"],
        gru_f_b_ih=weights["gru_f_b_ih"], gru_f_b_hh=weights["gru_f_b_hh"],
        gru_b_ih=weights["gru_b_ih"], gru_b_hh=weights["gru_b_hh"],
        gru_b_b_ih=weights["gru_b_b_ih"], gru_b_b_hh=weights["gru_b_b_hh"],
        dense_w=weights["dense_w"], dense_b=weights["dense_b"],
        out_w=weights["out_w"], out_b=weights["out_b"],
    )
    vocab.save(MODEL_ROOT / "vocab.json")
    config = {
        "architecture": "Embedding -> BiGRU -> Dense -> ReLU -> Output",
        "class_labels": CLASSES,
        "vocab_size": vocab.size(),
        "max_len": args.max_len,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "dense_dim": args.dense_dim,
        "dropout": args.dropout,
        "device": str(device),
    }
    (MODEL_ROOT / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    metadata = {
        "best_model": "Neural Network (BiGRU)",
        "model_class": "BiGRUNet",
        "vectorizer": f"Word tokenizer (vocab {vocab.size()}, max_len {args.max_len})",
        "architecture": config["architecture"],
        "dataset_source": source,
        "train_samples": int(len(ids_train)),
        "validation_samples": int(len(ids_val)),
        "test_samples": int(len(ids_test)),
        "n_features": vocab.size(),
        "vocab_size": vocab.size(),
        "training_date": str(pd_now()),
        "random_seed": args.seed,
        "split": "grouped-canonical (leakage-safe)",
        "metrics": test_metrics,
        "val_metrics": val_metrics,
    }
    (MODEL_ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_ROOT / "metrics.json").write_text(
        json.dumps({"test": test_metrics, "val": val_metrics,
                    "config": config, "metadata": metadata}, indent=2),
        encoding="utf-8")

    _parity_check(model, vocab, args.max_len)
    print(f"\n[artifacts] {MODEL_ROOT}")
    print(f"[done] {time.time() - t0:.1f}s")


def _valid(model, ids, labels, device, batch_size):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            bx = torch.from_numpy(ids[start:start + batch_size]).long().to(device)
            by = torch.from_numpy(labels[start:start + batch_size]).to(device)
            logits = model(bx)
            total_loss += float(nn.functional.cross_entropy(logits, by).item())
            correct += int((logits.argmax(1) == by).sum().item())
    return total_loss / max(1, (len(ids) + batch_size - 1) // batch_size), \
        correct / max(1, len(ids))


def _parity_check(model: BiGRUNet, vocab: Vocab, max_len: int) -> None:
    """Verify the NumPy forward matches PyTorch on random inputs (max diff < 1e-4)."""
    from ml.nn_forward import load_model as _load

    np_model = _load(MODEL_ROOT)
    model.eval()
    worst = 0.0
    with torch.inference_mode():
        for _ in range(12):
            ids = torch.randint(0, vocab.size(), (max_len,)).tolist()
            logits = model(
                torch.from_numpy(np.asarray(ids)[None]).long()
            ).softmax(dim=1).cpu().numpy()[0]
            p_real, p_fake = np_model.proba(ids)
            diff = max(abs(float(logits[1]) - p_real),
                       abs(float(logits[0]) - p_fake))
            worst = max(worst, diff)
    assert worst < 1e-4, f"numpy/torch parity check failed (max diff {worst})"
    print(f"[parity] torch vs numpy max prob diff = {worst:.2e}  OK")


def pd_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TruthLens BiGRU classifier")
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--max-words", type=int, default=30000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap training rows (smoke runs)")
    parser.add_argument("--sample", action="store_true",
                        help="train on the bundled sample instead of ISOT")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()