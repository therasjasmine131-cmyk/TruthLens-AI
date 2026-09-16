# TruthLens AI — Free Deployment Guide

TruthLens deploys as a **single Vercel Python service** (free tier, no credit
card): the Flask app serves both the API and the built React SPA from one
serverless function (`api/index.py`), so the frontend calls `/api` on the same
origin.

| Service | Role                         | Platform          | Stack                            |
| ------- | ---------------------------- | ----------------- | -------------------------------- |
| App     | Flask API + SPA + NN inference | Vercel (hobby)  | Python / Flask / NumPy (no scikit-learn, no torch at runtime) |

The trained model in `models/fake_news_neural_network/` is committed to the
repo and only uses **NumPy at runtime** — the backend boots with the trained
network and needs no retraining, no torch, no sklearn on the server.

---

## 0. One-time: push this repo to GitHub

```bash
git init
git add -A
git commit -m "Initial commit"
# create an empty repo on github.com, then:
git remote add origin git@github.com:<you>/truthlens-ai.git
git push -u origin main
```

> **Security:** `.env` is never committed. Copy `.env.example` to `.env`
> locally to test with real API keys.

---

## 1. Local verification (before pushing)

```bash
# Backend tests (from repo root)
python -m pip install -r backend/requirements.txt
python -m pytest backend/tests -q

# Frontend production build
cd frontend && npm install && npm run build && cd ..

# Serve the built SPA from the backend (commit-required layout)
Copy-Item -Recurse frontend\dist\* backend\static\
```

---

## 2. Deploy to Vercel (single service)

1. Install the Vercel CLI: `npm i -g vercel` and log in (`vercel login`).
2. From the repo root, create the production deployment:
   ```bash
   vercel --prod --yes
   ```
   Vercel reads `vercel.json` (service web → `api/index.py`). The Python
   function auto-detects `requirements.txt` at the repo root (which pins
   Flask/NumPy only — no scikit-learn, no torch, keeping the free tier happy).
3. Set production environment variables in the Vercel dashboard
   (Project → Settings → Environment Variables):
   ```env
   SECRET_KEY=<long-random-string>
   # Optional live backends (offline mode works without them):
   GEMINI_API_KEY=<...>
   NEWSAPI_KEY=<...>
   FACT_CHECK_API_KEY=<...>
   ```
4. Your public site is `https://<your-app>.vercel.app` (this project uses
   `https://truthlens-ai-prod.vercel.app`).

Expected result: `GET https://<your-app>.vercel.app/api/health` returns `ok`
with `neural_network.ready: true`, and `/api/verify` responds.

---

## 3. Verify the deployment

- **Health:** `curl https://<your-app>.vercel.app/api/health`
- **Verify a claim:** `POST /api/verify` with `{ "headline": "...", "article": "...", "language": "auto" }`
- Open the site and run an analysis. Watch the Vercel function logs for the
  first boot (NumPy model load is fast and memory-light).

---

## 4. Updating the model

1. Train/extend the network: `python ml/train.py` (needs
   `requirements-train.txt` incl. PyTorch; only run locally).
2. This rewrites `models/fake_news_neural_network/`.
3. Rebuild the SPA, `Copy-Item` `frontend\dist\*` into `backend\static\`,
   commit, then `vercel --prod --yes`.