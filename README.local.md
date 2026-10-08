# OpsMate — Frontend + Backend + Gemini + PostgreSQL

Runs with Docker PostgreSQL and npm.

## Stack

- Frontend (`frontend/`) — Next.js search UI on port **3001**
- Backend (`backend/`) — Express API on port 5000
- PostgreSQL — Docker container, stores each query/answer in `queries`
- Gemini — `GEMINI_MODEL` (default `gemini-3.6-flash`)

## Run modes

### Local dev (npm)

- Frontend: `cd frontend && npm run dev` → **http://localhost:3001**
- Backend: `cd backend && npm run dev` → port 5000
- Browser calls only the frontend origin; Next.js rewrites `/_backend/*` to the
  Express API (see `frontend/next.config.ts`), so ports never clash.

### Docker (single port)

```bash
docker compose up --build -d
```

Everything is served from **http://localhost:3001** through Caddy:

- `/` → Next.js frontend
- `/api/*` → Express backend (search, classify, test-order, health)
- `/ws` → backend WebSocket

Only port 3001 is published; frontend/backend run on their own internal ports.

## Database

PostgreSQL runs inside Docker.

Database:
- Database: `opsmate`
- User: `opsmate`
- Table: `queries`

## 1. Configure env

Root `.env`, `backend/.env`, and `frontend/.env.local` are already configured.

Backend uses:

`DATABASE_URL`

## 2. Create PostgreSQL table

Start the PostgreSQL container:

```bash
docker compose up -d postgres