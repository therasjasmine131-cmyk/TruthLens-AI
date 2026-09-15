"""Stress-test: run ``count`` additional real + fake news items (stratified)
through both the ML classifier and the evidence-based verification engine and
report how often the tool gives a *correct* answer.

Defines success on three levels:

* **ML classifier**        - the trained model's REAL/FAKE call vs. the label.
* **Evidence verdict**     - the verification verdict REAL/FALSE/UNVERIFIED; only
  items that actually produced a REAL or FALSE verdict count as "covered".
* **Hybrid answer**        - the verdict when confident, otherwise the classifier
  call as a fallback (this is the end-to-end "answer" the UI would show).

Runs in offline mode (knowledge base only - no live API calls) so it is fast
and reproducible: ``TRUTHLENS_LIVE_EVIDENCE=0``.

Usage::

    python ml/evaluate_holdout.py                # 200 items (100 real + 100 fake)
    python ml/evaluate_holdout.py --count 400 --seed 7 --engine hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TRUTHLENS_LIVE_EVIDENCE"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd  # noqa: E402

from ml.dataset import LABEL_FAKE, LABEL_REAL, load_dataset, group_split, _canonical_key  # noqa: E402
from backend.app.ml.model_manager import model_manager  # noqa: E402
from backend.app.services.verification.verifier import verify_text  # noqa: E402

ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
MAX_TEXT = 3000


def _classifier_answer(text: str) -> dict | None:
    if not model_manager.ready:
        return None
    p_real, p_fake = model_manager.predict_proba(text)
    return {
        "prediction": LABEL_REAL if p_real >= p_fake else LABEL_FAKE,
        "confidence": round(max(p_real, p_fake), 3),
    }


def build_rows(df_real: pd.DataFrame, df_fake: pd.DataFrame, count: int, seed: int):
    half = max(1, count // 2)
    real = df_real.sample(min(half, len(df_real)), random_state=seed)
    fake = df_fake.sample(min(half, len(df_fake)), random_state=seed)
    return pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate(count: int, seed: int, engine: str) -> dict:
    if engine in {"ml", "hybrid"}:
        model_manager.load()
        if not model_manager.ready:
            raise SystemExit(
                "ML artifacts not found - train the model first (python ml/train.py)."
            )

    df = load_dataset()
    df = df.drop_duplicates(subset=["headline", "text"])
    df = df[df["headline"].str.strip() != ""]

    # Exclude every row whose canonical group the trained model has seen
    # (same split/seed used by ml/train.py) so all N items are genuinely unseen.
    _tr, _va, _te = group_split(df, val_size=0.25, test_size=0.2, seed=42)
    train_keys = set(
        _canonical_key(h + " " + t)
        for h, t in zip(_tr["headline"], _tr["text"])
    )
    keys = (df["headline"] + " " + df["text"]).map(_canonical_key)
    df = df[~keys.isin(train_keys)]

    df_real = df[df["label"] == LABEL_REAL]
    df_fake = df[df["label"] == LABEL_FAKE]
    rows = build_rows(df_real, df_fake, count, seed)
    print(f"[data] {len(rows)} rows ({int((rows['label'] == LABEL_REAL).sum())} REAL / "
          f"{int((rows['label'] == LABEL_FAKE).sum())} FAKE) seed={seed} "
          f"(trained-on rows excluded, {len(df)} unseen rows available)")

    t0 = time.time()
    results = []
    for idx, row in rows.iterrows():
        title = str(row["headline"])
        body = str(row["text"])[:MAX_TEXT]
        combined = (title + " " + body)[:8000]
        label = row["label"]

        ml = _classifier_answer(combined)
        rep = verify_text(body, headline=title)
        verdict = rep["overall"]["verdict"]
        confident = verdict in (LABEL_REAL, LABEL_FAKE)
        hybrid = verdict if confident else (ml["prediction"] if ml else verdict)

        results.append({
            "index": idx,
            "label": label,
            "headline": title[:120],
            "ml_prediction": ml["prediction"] if ml else None,
            "ml_confidence": ml["confidence"] if ml else None,
            "verdict": verdict,
            "verdict_confidence": rep["overall"]["confidence"],
            "hybrid": hybrid,
            "claims_extracted": rep["pipeline"]["claims_extracted"],
            "evidence_items": rep["pipeline"]["evidence_items"],
            "outcome": "covered" if confident else "unverified",
            "ml_ok": bool(ml and ml["prediction"] == label),
            "hybrid_ok": bool(hybrid == label),
        })
    elapsed = time.time() - t0

    rdf = pd.DataFrame(results)
    rdf["ml_ok"] = rdf["ml_prediction"] == rdf["label"]
    rdf["hybrid_ok"] = rdf["hybrid"] == rdf["label"]
    rdf["verdict_wrong"] = ((rdf["verdict"] == LABEL_FAKE) & (rdf["label"] == LABEL_REAL)) | \
                           ((rdf["verdict"] == LABEL_REAL) & (rdf["label"] == LABEL_FAKE))

    print(f"\n[done] {len(results)} articles in {elapsed:.1f}s "
          f"({elapsed / max(1, len(results)):.2f}s/article)")

    # ---------- Metrics ----------
    def _block(key: str, heading: str):
        if key == "ml":
            y_pred = rdf["ml_prediction"].values
        elif key == "hybrid":
            y_pred = rdf["hybrid"].values
        else:
            return None
        if y_pred is None or pd.isna(y_pred).any():
            return None
        y_true = rdf["label"].values
        n = len(y_true)
        p = precision_score(y_true, y_pred, pos_label=LABEL_REAL, zero_division=0)
        r = recall_score(y_true, y_pred, pos_label=LABEL_REAL, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=LABEL_REAL, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE]).tolist()
        print(f"\n--- {heading} ---")
        print(f"  accuracy      : {accuracy_score(y_true, y_pred):.4f} ({int((y_pred == y_true).sum())}/{n})")
        print(f"  precision REAL: {p:.4f}   recall REAL: {r:.4f}   F1: {f1:.4f}")
        print(f"  confusion [row=label, col=pred] (REAL/FAKE): {cm}")
        return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
                "recall": float(r), "f1": float(f1), "confusion_matrix": cm}

    metrics = {}
    if engine in {"ml", "hybrid"}:
        metrics["ml"] = _block("ml", "ML classifier (REAL vs FAKE)")
    if engine in {"verification", "hybrid"}:
        covered = rdf[rdf["outcome"] == "covered"]
        print(f"\n--- evidence verdict coverage ---")
        print(f"  confident verdicts  : {len(covered)}/{len(rdf)} "
              f"({len(covered) / len(rdf):.1%})  (UNVERIFIED = no evidence found)")
        print(f"  verdict dist        : {rdf['verdict'].value_counts().to_dict()}")
        print(f"  verdict contradicts label (dangerous): {int(rdf['verdict_wrong'].sum())}")
        wrong = rdf[rdf["verdict_wrong"]]
        if len(wrong):
            print("\n  --- dangerous cases (verdict contradicts the label) ---")
            for _, w in wrong.head(10).iterrows():
                print(f"    label={w['label']:5s} verdict={w['verdict']:5s} | {w['headline']}")
        if len(covered):
            ok = int((covered["verdict"] == covered["label"]).sum())
            print(f"  accuracy on covered: {ok}/{len(covered)} = {ok / len(covered):.3f}")
        metrics["verification"] = {
            "coverage": len(covered), "covered_ok": int((covered["verdict"] == covered["label"]).sum()),
            "verdict_distribution": rdf["verdict"].value_counts().to_dict(),
            "dangerous_wrong": int(rdf["verdict_wrong"].sum()),
        }
    if engine == "hybrid":
        metrics["hybrid"] = _block("hybrid", "hybrid answer (verdict, else classifier)")

    report = {
        "config": {"count": len(rdf), "seed": seed, "engine": engine,
                   "live_evidence": False, "max_text_chars": MAX_TEXT},
        "metrics": metrics,
        "sample_rows": results,
        "runtime_seconds": round(elapsed, 1),
    }
    out = ARTIFACTS_DIR / "holdout_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[report] {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on N hold-out labelled articles")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--engine", choices=["ml", "verification", "hybrid"], default="hybrid")
    args = parser.parse_args()
    evaluate(args.count, args.seed, args.engine)


if __name__ == "__main__":
    main()