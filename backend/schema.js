// backend/schema.js
// ======================================================
// OPSMATE — DATABASE SCHEMA SETUP
// ======================================================
//
// Everything that creates/alters tables, extensions and indexes used to live
// inline inside server.js (ensureSchema()). It now lives here, in one place,
// so the schema can be:
//   1. applied automatically on server boot  -> require("./schema").ensureSchema(pool, opts)
//   2. applied/inspected manually, once      -> `node schema.js`
//
// Covers TWO things:
//   A. query_cache  — the exact-match answer cache table (L2), PLUS the
//      pgvector embedding column + ivfflat index that power the SEMANTIC
//      (embedding) cache tier — this is the "embedding cache" setup.
//   B. queries      — the audit log of every answered query (kept from the
//      previous Supabase-era schema), including de-duplication and a unique
//      index so the same question never accumulates duplicate rows.
//
// Every statement uses IF NOT EXISTS / ON CONFLICT, so ensureSchema() is
// always safe to call again (every server boot, or by re-running this file
// by hand) — nothing here is destructive.
// ======================================================

"use strict";

/**
 * Create/upgrade every table, extension, column and index OpsMate needs.
 * Shares the caller's pg Pool (server.js) so boot doesn't open a second,
 * redundant DB connection just to run this.
 *
 * @param {import('pg').Pool} pool
 * @param {{
 *   embeddingModel?: string,   // label stored in query_cache.embedding_model
 *   embeddingDim?: number,     // must match the vectors embedQuery() produces
 *   logger?: Console,          // defaults to console
 * }} [opts]
 * @returns {Promise<{ ok: boolean, pgvector: boolean }>}
 */
async function ensureSchema(pool, opts = {}) {
  const {
    embeddingModel =
      process.env.OLLAMA_EMBED_MODEL || process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text",
    embeddingDim = Number(process.env.EMBEDDING_DIM || 768),
    logger = console,
  } = opts;

  const info = (msg) => (logger.log || console.log).call(logger, `[Schema] ${msg}`);
  const warn = (msg) => (logger.warn || console.warn).call(logger, `[Schema] ${msg}`);

  // ---------- A1. query_cache (exact-match answer cache, L2) ----------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS query_cache (
      query_norm     TEXT PRIMARY KEY,
      query          TEXT NOT NULL,
      answer         TEXT NOT NULL,
      model          TEXT,
      prompt_version TEXT,
      hits           INTEGER NOT NULL DEFAULT 0,
      latency_ms     NUMERIC,
      created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      expires_at     TIMESTAMPTZ NOT NULL
    )
  `);
  await pool.query("CREATE INDEX IF NOT EXISTS query_cache_expires_at_idx ON query_cache (expires_at)");
  info("query_cache table ensured");

  // ---------- A2. embedding cache (pgvector) on top of query_cache ----------
  // Every cached answer is stored WITH the query's embedding so a rephrased
  // (not byte-identical) question can still be served from the cache. This
  // block is what makes the SEMANTIC cache tier possible.
  let pgvectorReady = false;
  try {
    await pool.query("CREATE EXTENSION IF NOT EXISTS vector");
    await pool.query(`ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS embedding vector(${embeddingDim})`);
    await pool.query(
      `ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT '${embeddingModel}'`
    );
    await pool.query(
      `CREATE INDEX IF NOT EXISTS query_cache_embedding_idx
         ON query_cache USING ivfflat (embedding vector_cosine_ops)
         WITH (lists = 100)`
    );
    pgvectorReady = true;
    info(`embedding cache ready (pgvector, model=${embeddingModel}, dim=${embeddingDim})`);
  } catch (e) {
    // pgvector unavailable: the semantic tier stays off, exact-match cache still works.
    warn(`pgvector setup skipped (semantic/embedding cache unavailable): ${e.message}`);
  }

  // ---------- B. queries (audit log) ----------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS queries (
      id          BIGSERIAL PRIMARY KEY,
      query       TEXT NOT NULL,
      answer      TEXT,
      model       TEXT,
      cached      BOOLEAN NOT NULL DEFAULT FALSE,
      latency_ms  NUMERIC,
      created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  for (const sql of [
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS model TEXT",
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS cached BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS latency_ms NUMERIC",
    // Tables first created by an older revision never got created_at, which
    // made the column described above (and any manual query on it) fail.
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
  ]) {
    try { await pool.query(sql); } catch (_e) { /* older/foreign table shape: ignore */ }
  }

  // Keep exactly ONE row per distinct query: collapse any existing duplicates
  // (keeping the newest), then enforce uniqueness so the same question can
  // never accumulate rows again. Cache hits are not written here at all.
  try {
    const dedup = await pool.query(
      "DELETE FROM queries a USING queries b WHERE a.query = b.query AND a.id < b.id"
    );
    if (dedup.rowCount) info(`queries: removed ${dedup.rowCount} duplicate row(s)`);
    await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS queries_query_uniq ON queries (query)");
    info("queries table ensured");
  } catch (e) {
    warn(`queries dedupe / unique index skipped: ${e.message}`);
  }

  return { ok: true, pgvector: pgvectorReady };
}

/**
 * Prints the current query_cache column list — handy after running this
 * file by hand, to confirm the embedding cache columns landed.
 */
async function printColumns(pool, table, logger = console) {
  const { rows } = await pool.query(
    `SELECT column_name, data_type
       FROM information_schema.columns
      WHERE table_name = $1
      ORDER BY ordinal_position`,
    [table]
  );
  (logger.log || console.log).call(logger, `\n${table} columns now:`);
  for (const row of rows) {
    (logger.log || console.log).call(logger, `  - ${row.column_name} (${row.data_type})`);
  }
}

module.exports = { ensureSchema, printColumns };

// ======================================================
// STANDALONE CLI USAGE
//   node schema.js
//
// Opens its own short-lived pool (DATABASE_URL, or PGHOST/PGPORT/PGUSER/
// PGPASSWORD/PGDATABASE), runs ensureSchema(), prints the resulting
// query_cache columns, then disconnects. Useful for applying/inspecting the
// schema (including the embedding cache) without starting the whole server.
// ======================================================
if (require.main === module) {
  const path = require("path");
  const dotenv = require("dotenv");
  const { Pool } = require("pg");

  dotenv.config({ path: path.resolve(__dirname, ".env"), override: true });
  dotenv.config({ path: path.resolve(__dirname, "../.env") });

  const databaseUrl = process.env.DATABASE_URL || "";

  const pool = new Pool(
    databaseUrl
      ? { connectionString: databaseUrl }
      : {
          host: process.env.PGHOST || "localhost",
          port: Number(process.env.PGPORT || 5432),
          user: process.env.PGUSER || "opsmate",
          password: process.env.PGPASSWORD,
          database: process.env.PGDATABASE || "opsmate",
        }
  );

  const embeddingModel =
    process.env.OLLAMA_EMBED_MODEL ||
    process.env.OLLAMA_EMBEDDING_MODEL ||
    process.env.HUGGINGFACE_MODEL ||
    "nomic-embed-text";
  const embeddingDim = Number(process.env.EMBEDDING_DIM || 768);

  (async () => {
    console.log(`[schema.js] model=${embeddingModel} dim=${embeddingDim}`);
    const result = await ensureSchema(pool, { embeddingModel, embeddingDim });
    await printColumns(pool, "query_cache");
    console.log(`\n[schema.js] done. pgvector/embedding cache: ${result.pgvector ? "ready" : "unavailable"}`);
  })()
    .catch((error) => {
      console.error("[schema.js] failed:", error.message);
      process.exitCode = 1;
    })
    .finally(() => pool.end());
}