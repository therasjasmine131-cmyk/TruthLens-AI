# TruthLens AI — Free Deployment Guide

TruthLens deploys as **two free services**:

| Service | Role                         | Platform | Stack                         |
| ------- | ---------------------------- | -------- | ----------------------------- |
| Backend | Flask API + ML model inference | Railway | Python / Gunicorn / Nixpacks  |
| Frontend| React SPA                     | Vercel   | Vite / React / TypeScript     |

Both platforms have a free/hobby tier (no credit card required). The ML
artifacts in `ml/artifacts/` are committed to the repo, so the backend boots
with the trained model — **no retraining on the server**.

---

## 0. One-time: push this repo to GitHub

Railway and Vercel both deploy from a GitHub repository, so the first step is
to make this folder a repo and push it:

```bash
git init
git add -A
git commit -m "Initial commit"
# create an empty repo on github.com, then:
git remote add origin git@github.com:<you>/truthlens-ai.git
git push -u origin main
```

> **Security:** `.gitignore` excludes `.env`, `ml/data/raw/`, SQLite DBs and
> generated build artifacts. **Never commit a real `.env`.**

---

## 1. Backend → Railway

The config is in `railway.json` (root directory `backend`, Nixpacks build,
Gunicorn start command, `/api/health` healthcheck).

1. In the Railway dashboard: **New Project → Deploy from GitHub repo** → pick
   this repo. Railway reads `railway.json` automatically.
2. Open the service → **Variables** and set:
   ```env
   SECRET_KEY=<long-random-string>          # e.g. `openssl rand -hex 32`
   CORS_ORIGINS=https://<your-app>.vercel.app
   # PORT is injected by Railway; do NOT set it.
   # Optional: DATABASE_URL=sqlite:///truthlens.db (default) or a Railway Postgres URL.
   # Optional: GEMINI_API_KEY / NEWSAPI_KEY for the extra analyzers.
   ```
3. Deploy → visit the **Settings → Networking → Generate Domain** to get your
   public URL, then **copy it** — you'll need it for the frontend.

Expected result: `GET https://<backend>.up.railway.app/api/health` returns
`ok` and the app responds on `/api`.

---

## 2. Frontend → Vercel

The config is in `frontend/vercel.json` (SPA rewrite). Vercel auto-detects the
Vite build (`npm run build`, output `frontend/dist`).

1. [vercel.com](https://vercel.com) → **Add New → Project** → import the same
   GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add the environment variable (pointing at your Railway service):
   ```env
   VITE_API_URL=https://<backend>.up.railway.app
   ```
   > If `VITE_API_URL` is left empty the SPA calls `/api` on the same origin —
   > only workable if the backend serves the built frontend from
   > `backend/static/`.
4. Deploy. Your public site is `https://<your-app>.vercel.app`.

---

## 3. Point the frontend at the backend

`frontend/src/api/client.ts` already reads `VITE_API_URL` (with a dev fallback),
so no code change is required once the variable is set. Update the `CORS_ORIGINS`
on Railway to match the final Vercel URL if it differs from what you set in step 1.

---

## 4. Verify the deployment

- **Backend:** `curl https://<backend>.up.railway.app/api/health`
- **Frontend:** open `https://<your-app>.vercel.app` and run an analysis.
- Watch the deploy logs in the Railway/Vercel dashboards for the first boot
  (the model loads at startup; it can take ~30–60s).

---

## Local verification (before pushing)

```bash
# Backend tests (from repo root)
python -m pip install -r requirements.txt
python -m pytest backend/tests

# Frontend production build
cd frontend && npm install && npm run build && cd ..
```
