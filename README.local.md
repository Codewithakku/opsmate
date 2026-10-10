# OpsMate — Frontend + Backend + Ollama + PostgreSQL

Runs with a local Ollama daemon and Docker PostgreSQL.

## Stack

- Frontend (`frontend/`) — Next.js UI on port **3001**. Its `/api/search` and `/api/assist`
  routes are thin proxies to the backend (no model SDK in the browser app).
- Backend (`backend/`) — Express API on port 5000. **Ollama is the only model provider.**
- Ollama — local LLM server at `http://localhost:11434`, model `OLLAMA_MODEL` (default `llama3.2:3b`).
- PostgreSQL — Docker container; stores the persistent answer cache in `query_cache`
  and every answered query in `queries`.

There is no OpenAI / Anthropic / Gemini / Supabase / Upstash code path any more.

## Request flow (fastest first)

1. **Order fast-path** — a query containing an order id (`ORD-1001`) is answered from the
   `orders` table with a template. No LLM call at all (~ms).
2. **L1 memory cache** — per-process LRU + TTL (~1 ms).
3. **L2 PostgreSQL cache** — `query_cache` table, exact normalized-query match (~5-20 ms).
4. **Ollama** — one streamed chat completion; the answer is written back to `query_cache`
   (L2) and `queries` (audit), so the next identical query is a cache hit.

Single-flight coalescing means two identical requests arriving together share one model call.

### Stop button

Pressing **■ Stop** aborts the browser fetch; the Next proxy drops the connection, the backend's
`res.on("close")` handler aborts the Ollama request, and a partially generated answer is **never
written to `query_cache`**. Server logs show `[Assist] client disconnected … cancelling generation`.

## Run modes

### Local dev (npm)

```bash
# prerequisites: `ollama serve` + `ollama pull llama3.2:3b`, and Postgres on :5432
docker compose up -d postgres

cd backend  && npm install && npm run dev   # port 5000
cd frontend && npm install && npm run dev   # http://localhost:3001
```

The browser only calls the frontend origin. In Docker the frontend calls the backend at
`http://backend:5000`; for local npm dev set `BACKEND_URL=http://localhost:5000` (default).

### Docker (single port)

```bash
docker compose up --build -d
```

Everything is served from **http://localhost:3001** through Caddy:

- `/` → Next.js frontend
- `/api/search`, `/api/assist`, `/api/cache/*` → Next.js route handlers (which proxy to the backend)
- `/api/health`, `/api/classify`, `/api/test-order`, `/api/backend-search` → Express backend
- `/ws` → backend WebSocket

Do not run local dev and the Docker stack at the same time — both use host port 3001.
Note: the backend container needs to reach Ollama on the host (`OLLAMA_HOST=http://host.docker.internal:11434`).

## Database

PostgreSQL runs inside Docker (`docker compose up -d postgres`).

- Database: `opsmate`, user: `opsmate`
- `query_cache` — the answer cache. Created automatically at backend startup:
  `query_norm` (PK), `query`, `answer`, `model`, `prompt_version`, `hits`, `latency_ms`,
  `created_at`, `expires_at`.
- `queries` — one row per distinct query that the LLM answered (`query`, `answer`, `model`, `cached`,
  `latency_ms`). Unique on `query` (upserted), so the same question never accumulates rows. Cache
  hits are **not** written here. Backend startup de-duplicates existing rows and ensures the index.

Cache rules:

- Generic questions: `CACHE_TTL_SEC` (default 24 h).
- Questions containing a number or an e-mail (live/personal data): `CACHE_LIVE_TTL_SEC` (default 60 s).
- `prompt_version` is derived from the system prompt, so editing the prompt invalidates old rows.
- Cache **hits never write** to `query_cache`. Only a genuine MISS with a successful answer
  performs the upsert, so concurrent identical requests share a single write (the `hits`
  column is legacy; per-tier hit counts live in memory under `/api/cache/stats`).
- Conversation history (last 3 completed turns) is sent to the model on a MISS; it is never
  part of the cache key and never written to the cache.
- Latency: the system prompt is kept compact (~550 tokens instead of ~2,900) because CPU prompt
  evaluation is the dominant cost. A measured miss went from ~182 s to ~61 s at 513 prompt tokens.
- If Postgres is down, the cache fails open: requests still work, `dbErrors` in `/api/cache/stats` grows.

## Environment (`backend/.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | — | PostgreSQL connection string (`@postgres:` is rewritten to `@localhost:` for local dev) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model name |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps the model resident so requests don't pay the load cost |
| `OLLAMA_NUM_CTX` | `4096` | Context window (smaller = faster prompt eval) |
| `OLLAMA_TIMEOUT_MS` | `600000` | Hard ceiling per model call |
| `OLLAMA_FIRST_TOKEN_TIMEOUT_MS` | `180000` | Aborts a wedged call before the first token |
| `AI_MAX_TOKENS` | `80` | Output cap — the biggest latency lever on CPU-only Ollama |
| `CACHE_ENABLED` | `true` | Master switch for both cache tiers |
| `CACHE_L1_TTL_SEC` / `CACHE_TTL_SEC` / `CACHE_LIVE_TTL_SEC` | `300` / `86400` / `60` | Cache lifetimes |
| `CACHE_DB_TIMEOUT_MS` | `8000` | Postgres cache I/O ceiling |
| `ORDER_FAST_PATH` | `true` | Answer status questions from the DB without the LLM |

Retired keys (`ANTHROPIC_*`, `GEMINI_*`, `OPENAI_*`, `SUPABASE_*`, `UPSTASH_*`) are commented out
in `.env`, `backend/.env` and `frontend/.env.local`.

## Verify it works

```bash
curl http://localhost:5000/health          # provider=ollama, dbReady=true, cache stats
curl http://localhost:5000/api/cache/stats

# order status → DB fast-path, no LLM, then cached
curl -X POST http://localhost:5000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"where is order ORD-1001"}'      # 1st: X-Cache: MISS
curl -X POST http://localhost:5000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"where is order ORD-1001"}'      # 2nd: X-Cache: L1_MEMORY

# generic question → Ollama once, then served from Postgres
curl -X POST http://localhost:5000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"how long does a refund take to show up"}'

# prove the DB tier: clear L1 only, keep the rows
curl -X POST http://localhost:5000/api/cache/flush -H 'Content-Type: application/json' \
  -d '{"memory":true,"database":false}'
curl -X POST http://localhost:5000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"how long does a refund take to show up"}'   # X-Cache: L2_POSTGRES

# clear everything and start over
curl -X POST http://localhost:5000/api/cache/flush -H 'Content-Type: application/json' -d '{}'
```

Watch `X-Cache` (`MISS` → `L2_POSTGRES` → `L1_MEMORY`) and `Server-Timing` on each response.

## Troubleshooting

- `503 OLLAMA_UNAVAILABLE` — run `ollama serve`; check `OLLAMA_HOST`.
- `503 MODEL_NOT_FOUND` — run `ollama pull llama3.2:3b`.
- Very slow first answer — this machine runs Ollama on **CPU only** (`ollama ps` shows
  `size_vram: 0`), roughly 0.3-2 tokens/s. Reduce `AI_MAX_TOKENS`, keep `OLLAMA_KEEP_ALIVE`
  high, and let the cache serve repeats (a measured example: 97.7 s miss → 15 ms `L2_POSTGRES` hit).
- `dbReady: false` — Postgres is not reachable; `docker compose up -d postgres`.
- `EADDRINUSE` on 5000/3001 — another instance is still running.
