# TruthLens AI — Fake News Detection System

AI-first fake-news detection. A trained **Neural Network (Embedding → BiGRU)** reads the article and predicts `REAL` / `FAKE`, while an **evidence-based verification engine** decomposes it into atomic claims, retrieves real news sources, and produces an explainable evidence verdict (`REAL` / `FALSE` / `UNVERIFIED`) — finished off by a **free cloud-AI judge** (Gemini → Groq → OpenRouter → ChatGPT → Ollama) so the app **never breaks and never fabricates a verdict** when a provider is offline.

## How it works

```
article text (+ headline)
        │
        ├── NN CLASSIFIER ───────────────► REAL / FAKE / UNCERTAIN (+ probabilities)
        │     Embedding → BiGRU (PyTorch-trained, runs on pure NumPy)
        │
        └── VERIFICATION ENGINE ────────► evidence verdict + full report
              1. decompose article into atomic claims (opinion / prediction
                 handling, Tamil & Tanglish language detection)
              2. retrieve evidence  (knowledge base + live web search:
                 Google News RSS / Bing News / DuckDuckGo — keyless)
              3. relevance scoring + source tiering + cross-source checks
              4. per-claim verdict  (SUPPORTS / CONTRADICTS / NEUTRAL)
              5. AI judge passes the final verdict chain:
                 Gemini → Groq → OpenRouter → ChatGPT → Ollama
              6. overall verdict  (REAL / FALSE / UNVERIFIED)
```

- **Never fabricates:** if no provider can decide, the API honestly returns
  `ai_unavailable` with the evidence verdict intact — no invented "correct-looking"
  answers.
- **Live evidence:** the evidence engine pulls real headlines from free
  news RSS (Google News, Bing News, DuckDuckGo) so the AI judge is grounded in
  actual reporting — even when every paid API key is missing.
- `FALSE` (evidence verdict) is the same class as `FAKE` (classifier prediction).

## Features

- **Neural classifier** — Embedding → BiGRU trained on the ISOT dataset; exported
  to run purely on NumPy (no PyTorch required at runtime)
- **Claim decomposition** — conjoined claims, opinion/prediction marking,
  attribution stripping, negation and numerical handling, temporal reasoning
- **Evidence engine** — knowledge base + keyless live news search, relevance
  scoring, source tiering, cross-source agreement checks
- **AI final verdict** — Gemini → Groq → OpenRouter → ChatGPT → Ollama chain,
  with a healthy confidence label (`High / Medium / Low`) and per-claim recounts
- **Explainability** — token influence, evidence matrix, per-claim verdicts, and
  a generated report per article
- **Extra detectors** — AI-text detection, Trending News feed, batch analysis,
  model performance dashboard (confusion matrix, ROC, precision/recall/F1)
- **Clean responsive UI** — React 18 + Vite + TypeScript + Tailwind CSS,
  interactive charts via Recharts

## Tech Stack

- **Backend:** Python 3.11, Flask 3.x, NumPy (BiGRU runtime), optional PyTorch
  (training only)
- **ML:** Embedding + BiGRU, ISOT-derived dataset (`ml/data/`), pure-NumPy
  forward pass (`ml/nn_forward.py`)
- **Frontend:** React 18, Vite 5, TypeScript, Tailwind CSS, Recharts
- **Testing:** pytest (backend), Vitest + Testing Library (frontend)
- **Deploy:** Vercel serverless (single Python service, `api/index.py`)

## Getting Started

### 1. Train the model (one-time)

```powershell
pip install -r requirements-train.txt   # includes PyTorch (CPU)
python ml/train.py                      # trains the BiGRU and exports models/
```

Use `python ml/train.py --sample --epochs 2` for a quick smoke run on the bundled
1,200-row sample.

### 2. Run the backend

```powershell
Copy-Item .env.example .env   # then edit values (API keys optional — offline mode works)
python backend/run.py         # http://localhost:5000
```

Health check: `GET http://localhost:5000/api/health`

### 3. Frontend (development)

```powershell
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

## Command return shape (clean AI contract on `POST /api/analyze/headline`)

```json
{
  "overall": {
    "verdict": "REAL",
    "confidence": 0.95,
    "confidence_label": "High confidence",
    "counts": { "real": 1, "false": 0, "unverified": 0, "total_claims": 1 },
    "sources": ["web-search"],
    "verdict_reason": "evidence + AI-chain support the claim"
  },
  "claims": [{ "text": "...", "verdict": "REAL", "evidence": [...] }],
  "evidence": [...],
  "stats": { "processing_time_ms": 2300, "ai_provider": "groq" },
  "success": true
}
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/analyze` | Full article analysis (NN + evidence + AI verdict) |
| `POST /api/analyze/headline` | Headline-only analysis (same engine, `"headline"` field) |
| `POST /api/verify` | Evidence verification endpoint (final verdict chain) |
| `POST /api/batch/analyze` | Batch analysis of multiple articles |
| `POST /api/detect-ai-text` | AI-text / AI-generated-content detection |
| `GET /api/news/trending` | Trending news feed |
| `GET /api/history` · `GET /api/history/export` | Saved analyses + export |
| `GET /api/history/report/<id>` | Generated per-article PDF-style report |
| `GET /api/model-performance` | Confusion matrix, ROC, precision/recall/F1 |
| `GET /api/analytics` | Aggregate statistics |
| `GET /api/dataset/stats` · `GET /api/dataset/samples` | Dataset explorer |
| `GET /api/settings` · `PUT /api/settings` | Runtime configuration |
| `GET /api/health` | Health + capability probe |

## Serverless (Vercel)

`api/index.py` + `vercel.json` deploy the whole app (API + built SPA) as a single
serverless Python function:

```powershell
npx vercel --prod
```

Set the optional provider keys in the Vercel project's Environment Variables
(`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`) — the
app works fine without them (offline evidence + `ai_unavailable` verdict).
Note: the Hobby-tier function timeout is capped at 60 s (`maxDuration`).

## Project Structure

```
├── api/index.py                # Vercel serverless entry (Flask app)
├── backend/                    # Flask API
│   ├── app/
│   │   ├── routes/             # analyze, verify, batch, history, news, ...
│   │   ├── services/           # analyzer, llm_check, web_search, verifier
│   │   │   └── verification/   # evidence engine modules (claims, scoring, ...)
│   │   ├── ml/model_manager.py # NumPy inference wrapper
│   │   └── config.py
│   ├── run.py                  # local dev server
│   └── tests/                  # pytest suite (hermetic, offline)
├── frontend/                   # React + Vite + Tailwind + TS UI
├── ml/                         # training pipeline (train, dataset, nn_forward)
├── models/                     # exported BiGRU runtime (weights.npz, vocab.json)
├── PROJECT_ABSTRACT.txt
├── .env.example
├── requirements.txt            # runtime deps (NumPy-only inference)
└── vercel.json
```

## Testing

```powershell
python -m pytest backend/tests -q   # 93 tests, hermetic (no network)
cd frontend; npm test               # 29 tests (Vitest + Testing Library)
```

## Sample Text to Try

> "Scientists have confirmed that the first McDonald's restaurant on Mars opened
> its doors to astronauts this week, serving zero-gravity cheeseburgers."

## License

College mini project — free to use for educational purposes.