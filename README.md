# TruthLens AI

Hybrid fact-checking tool that combines a trained neural network (Embedding →
BiGRU) with an evidence-based verification engine. For every submitted article
it returns a classifier prediction (`REAL` / `FAKE`) **and** an explainable
evidence verdict (`REAL` / `FALSE` / `UNVERIFIED`) backed by retrieved sources,
claim decomposition, and a curated knowledge base.

Demo data, verification pipeline, and trained model are all local — no external
calls are required for the evidence engine's knowledge-base mode.

## How it works

```
article text (+ headline)
        │
        ├── NN CLASSIFIER ─────────────────► REAL / FAKE / UNCERTAIN (+probabilities)
        │     Embedding → BiGRU (trained with PyTorch, runs on pure NumPy)
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
  nn_tokenize.py, nn_forward.py   vocabulary + pure-NumPy BiGRU inference
  data/                 1,200-row sample dataset (ISOT-derived)
models/                 exported runtime model
  fake_news_neural_network/  weights.npz, vocab.json, config.json, metrics.json
DEPLOYMENT.md, .env.example   deployment notes & configuration
```

## Quick start

### 1. Train the model (one-time)

```powershell
pip install -r requirements-train.txt   # includes PyTorch (CPU)
python ml/train.py                      # trains/fine-tunes the BiGRU and exports models/fake_news_neural_network/
```

The command-line defaults target the full ISOT dataset (`ml/data/raw/`). Use
`python ml/train.py --sample --epochs 2` for a quick smoke run on the bundled
1,200-row sample.

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
python ml/train.py                     # prints train/val/test metrics + NumPy-vs-torch parity check
python ml/evaluate_claims.py           # evidence engine on curated knowledge-base claims
```

Latest results (on the full ISOT training run — see `models/fake_news_neural_network/metrics.json` for the deployed model):

| Check | Result |
|---|---|
| Neural network (test split) | see metrics.json (model_manager /api/model-perf reports live) |
| Evidence engine (curated claims) | 12/12 = 100% |

## Tests

```powershell
python -m pytest backend/tests -q     # backend (64 tests)
cd frontend; npm test                  # frontend (26 tests)
```

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `NN_MODEL_DIR` | exported model location (default `models/fake_news_neural_network`) |
| `DATASET_RAW_DIR` | full ISOT CSVs for retraining (`ml/data/raw/`) |
| `TRUTHLENS_LIVE_EVIDENCE` | set `0` to force knowledge-base-only mode |
| `FACT_CHECK_API_KEY` / `NEWSAPI_KEY` / `GEMINI_API_KEY` | optional live evidence + AI-text backends; when unset, the app gracefully falls back to offline heuristics |

See `.env.example` and `DEPLOYMENT.md` for full details.

## Links

- Live app (Vercel serverless): https://truthlens-ai-prod.vercel.app
- Repository: https://github.com/therasjasmine131-cmyk/TruthLens-AI
- Deployment: see `DEPLOYMENT.md`. Hosted as a single Vercel Python service
  (entrypoint `api/index.py`, Flask serves both the API and the built SPA);
  the DB is in-memory (ephemeral) on the serverless tier.