# Deep · portfolio RAG

The backend for **Deep**, the assistant on my portfolio. It answers questions from my portfolio site, resume, case studies, articles and every public GitHub repo's markdown, and it picks up new repos automatically.

```
GitHub Action (daily + manual)                     Netlify (portfolio)
  list public repos → .md files                        Deep widget
  + index.html sections + PDFs                            │ POST /api/ask
  → chunk (heading-aware, ~450 tokens)                    ▼
  → embed only new/changed chunks          Vercel (FastAPI, api/index.py)
    (gemini-embedding-001, 768-d)            embed question (Gemini)
  → upsert / delete stale ──────────►  Supabase pgvector: match_chunks()
                                             hybrid = vector + full-text (RRF)
                                           answer: Gemini → fallback Groq
                                           → { answer, sources[] }
```

| Piece | Service (free tier) |
|---|---|
| Vector DB + hybrid search | Supabase Postgres + pgvector |
| API | Vercel Python function (FastAPI) |
| Embeddings | Gemini `gemini-embedding-001` @ 768 dims |
| LLM | Gemini `gemini-3.5-flash-lite` → fallback Groq `openai/gpt-oss-120b` |
| Ingestion | GitHub Actions cron |

---

## Setup (≈20 minutes)

### 1. Supabase
1. Create a project at supabase.com (pick the Mumbai region `ap-south-1` for lowest latency from India).
2. **SQL Editor → New query** → paste `supabase/schema.sql` → **Run**.
3. **Project Settings → API Keys**: copy the **Project URL**, the **publishable** key (`sb_publishable_…`) and the **secret** key (`sb_secret_…`). If you only see the legacy keys, use `anon` and `service_role` instead; both work.

### 2. Groq
Create a key at console.groq.com → API Keys.

### 3. Push this folder to a new GitHub repo
For example `portfolio-rag` (public or private, either is fine).

### 4. GitHub Action secrets
Repo → **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `GEMINI_API_KEY` | your AI Studio key |
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_SECRET_KEY` | `sb_secret_…` (or `service_role`) |

Optional **Variables** (same page, *Variables* tab): `PORTFOLIO_REPO` (default `Portfolio`, the repo holding `index.html` and the PDFs) and `SKIP_REPOS` (comma-separated repo names to exclude).

Then go to **Actions → Ingest portfolio knowledge → Run workflow**. The first run embeds everything; it takes a few minutes because it paces itself under the free-tier limit. The log ends with `✓ done`. You can check rows in Supabase **Table Editor → chunks**.

### 5. Vercel
1. **Add New → Project** → import the `portfolio-rag` repo. Framework preset: **Other**. Leave the build settings empty.
2. **Environment Variables**:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | AI Studio key |
| `GROQ_API_KEY` | `gsk_…` |
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_PUBLISHABLE_KEY` | `sb_publishable_…` (or `anon`), **never the secret key** |
| `ALLOWED_ORIGINS` | `https://YOUR-SITE.netlify.app` (add your custom domain too, comma-separated) |

3. Deploy, then open `https://<project>.vercel.app/api/health`. Every value should be `true`.
4. Test it:
   ```bash
   curl -X POST https://<project>.vercel.app/api/ask \
     -H "Content-Type: application/json" -H "Origin: https://YOUR-SITE.netlify.app" \
     -d '{"question":"What did he do at CyCrew?"}'
   ```

### 6. Portfolio (Netlify)
Replace `index.html` in the Portfolio repo with the patched one, and set:
```js
var DEEP_API = "https://<project>.vercel.app/api/ask";
```
Push, and Netlify redeploys. Changes in the widget:
- calls the API, with a typing indicator and conversation memory for follow-ups
- if the API is down, slow (>25 s) or rate-limited, it falls back to the old keyword answers (`DEEP_FACTS`), so the widget never breaks
- updated greeting and fallback text (you may also want to remove the "In development" badge)

---

## Day-to-day

- **New repo or updated README:** nothing to do. The daily run (≈01:47 IST) picks it up. For an instant update, use **Actions → Run workflow**.
- **Hide a repo from Deep:** add the topic `no-rag` to it on GitHub, or list it in `SKIP_REPOS`. Its chunks are deleted on the next run.
- **Private repos are never indexed**, because the Action only lists public repos.
- **Changing the embedding model or dimension:** truncate the table (`truncate public.chunks;`), update `vector(768)` in the schema and `EMBED_DIM` in both places, then re-run. Changing the LLM is just an environment variable.
- **Supabase free projects pause after 7 days without activity.** The daily Action queries the DB, which counts as activity.

## Local testing

```bash
pip install -r requirements.txt -r ingest/requirements.txt uvicorn

# see exactly what gets indexed, without calling any API
python ingest/ingest.py --dry-run --no-github --local-portfolio ../Portfolio --dump chunks.json

# run the API locally (export the Vercel env vars first; add http://localhost:5500 to ALLOWED_ORIGINS)
uvicorn api.index:app --reload --port 8787
```

## Tuning knobs (env vars)

| Var | Default | Where |
|---|---|---|
| `MATCH_COUNT` | 6 | API: chunks sent to the LLM |
| `RATE_LIMIT_PER_MIN` | 8 | API: per-IP, per instance |
| `GEMINI_MODEL` / `GROQ_MODEL` | `gemini-3.5-flash-lite` / `openai/gpt-oss-120b` | API |
| `EMBED_BATCH`, `EMBED_PAUSE_S` | 20, 21 s | Ingest: pacing for the free embedding tier |
| `MAX_EMBED_PER_RUN` | 900 | Ingest: stays under ~1,000 embeds/day; the rest continue the next day |
