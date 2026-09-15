# TruthLens AI

Hybrid fact-checking tool that combines a trained ML classifier with an
evidence-based verification engine. For every submitted article it returns a
classifier prediction (`REAL` / `FAKE`) **and** an explainable evidence verdict
(`REAL` / `FALSE` / `UNVERIFIED`) backed by retrieved sources, claim
decomposition, and a curated knowledge base.

Demo data, verification pipeline, and trained model are all local — no external
calls are required for the evidence engine's knowledge-base mode.

## How it works

```
article text (+ headline)
        │
        ├── ML CLASSIFIER ────────────────► REAL / FAKE / UNCERTAIN (+probabilities)
        │     Linear SVM on word 1-2 grams
        │
        └── VERIFICATION ENGINE ──────────► evidence verdict + full report
              1. decompose article into atomic claims (opinion / prediction
                 handling, Tamil & Tanglish language detection)
              2. retrieve evidence  (knowledge base + optional live sources)
              3. relevance scoring + source tiering + cross-source checks
              4. per-claim verdict  (SUPPORTS / CONTRADICTS / NEUTRAL)
              5. overall verdict      (REAL / FALSE / UNVERIFIED)
```

Verdicts are only asserted when evidence actually supports them — otherwise the
engine honestly returns `UNVERIFIED` (no fabricated "correct-looking" answers).
`FALSE` (evidence verdict) is the same class as `FAKE` (classifier prediction).

## Repository layout

```
backend/                Flask API (analyze, batch, history, settings, ...)
  app/ml/               model loading wrapper (model_manager)
  app/services/         analyzer, explainer, verification_service
  app/services/verification/  evidence engine modules
  tests/                pytest suite
  static/               built frontend (copied from frontend/dist)
frontend/               React + Vite + Tailwind UI
ml/                     training + evaluation pipeline (train, dataset, ...)
  artifacts/            trained model + evaluation reports
  data/                 1,200-row sample dataset (ISOT-derived)
DEPLOYMENT.md, .env.example   deployment notes & configuration
```

## Quick start

### 1. Train the model (one-time)

```powershell
pip install -r requirements.txt
python ml/train.py            # trains 5 models, keeps the best Linear SVM
```

### 2. Run the backend

```powershell
Copy-Item .env.example .env   # then edit values (leave API keys blank for offline mode)
python backend/run.py         # http://localhost:5000
```

Health check: `GET http://localhost:5000/api/health`

### 3. Frontend (development)

```powershell
cd frontend
npm install
npm run dev                   # http://localhost:5173 (VITE_API_URL empty = same origin)
```

For production, build and serve from the backend:

```powershell
cd frontend
npm run build                 # outputs frontend/dist
Copy-Item -Recurse dist\* ..\backend\static\
```

## Evaluation

```powershell
python ml/evaluate.py --full-report          # train/test metrics of the best model
python ml/evaluate_claims.py                 # evidence engine on curated knowledge-base claims
python ml/evaluate_holdout.py --count 200 --seed 7
                                            # holdout audit: 200+ genuinely unseen articles,
                                            # reports ML accuracy, verdict coverage, and the
                                            # hybrid (verdict-else-classifier) accuracy
```

Latest supervised results (on the bundled 1,200-row sample):

| Check | Result |
|---|---|
| ML classifier (test split) | accuracy 0.971, F1 0.971, AUC 0.998 |
| Evidence engine (curated claims) | 12/12 = 100% |
| 200-article holdout audit (unseen) | ML/hybrid accuracy 0.97, 0 dangerous wrong verdicts |

## Tests

```powershell
python -m pytest backend/tests -q     # backend (34 tests)
cd frontend; npm test                  # frontend (25 tests)
```

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `ML_ARTIFACTS_DIR` | trained-model location (default `ml/artifacts`) |
| `DATASET_RAW_DIR` | full ISOT CSVs for retraining (`ml/data/raw/`) |
| `TRUTHLENS_LIVE_EVIDENCE` | set `0` to force knowledge-base-only mode |
| `FACT_CHECK_API_KEY` / `NEWSAPI_KEY` / `GEMINI_API_KEY` | optional live evidence + AI-text backends; when unset, the app gracefully falls back to offline heuristics |

See `.env.example` and `DEPLOYMENT.md` for full details.

## Links

- Repository: https://github.com/therasjasmine131-cmyk/TruthLens-AI
- Deployment: see `DEPLOYMENT.md` (Railway backend + Vercel frontend)