// setup-semantic-cache.js
//
// Run once to add semantic-cache support (pgvector) on top of the
// existing query_cache table. Safe to re-run — every step uses
// IF NOT EXISTS, so running it twice does nothing destructive.
//
// Usage:
//   node setup-semantic-cache.js
//
// Reads DB connection the same way backend/index.js does: either
// DATABASE_URL, or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE.
// Also reads HUGGINGFACE_MODEL to label the embedding_model default.

const { Pool } = require("pg");
require("dotenv").config();

const pool = new Pool(
  process.env.DATABASE_URL
    ? { connectionString: process.env.DATABASE_URL }
    : {
        host: process.env.PGHOST || "localhost",
        port: Number(process.env.PGPORT || 5432),
        user: process.env.PGUSER || "opsmate",
        password: process.env.PGPASSWORD,
        database: process.env.PGDATABASE || "opsmate",
      }
);

const EMBEDDING_MODEL =
  process.env.HUGGINGFACE_MODEL ||
  "sentence-transformers/all-MiniLM-L6-v2";

// all-MiniLM-L6-v2 = 384 dims. If you switch HF models, update this
// (and note: existing rows embedded with the old model/dimension
// will no longer be comparable — see embedding_model column).
const EMBEDDING_DIM = Number(process.env.EMBEDDING_DIM || 384);

async function run(label, sql) {
  process.stdout.write(`- ${label} ... `);
  await pool.query(sql);
  console.log("done");
}

async function main() {
  console.log(`[setup-semantic-cache] model=${EMBEDDING_MODEL} dim=${EMBEDDING_DIM}`);

  await run("enable pgvector extension", `CREATE EXTENSION IF NOT EXISTS vector`);

  await run(
    "add embedding column",
    `ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS embedding vector(${EMBEDDING_DIM})`
  );

  await run(
    "add embedding_model column",
    `ALTER TABLE query_cache
       ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT '${EMBEDDING_MODEL}'`
  );

  await run(
    "create vector similarity index (ivfflat, cosine)",
    `CREATE INDEX IF NOT EXISTS query_cache_embedding_idx
       ON query_cache USING ivfflat (embedding vector_cosine_ops)
       WITH (lists = 100)`
  );

  // Confirm final schema back to the console.
  const { rows } = await pool.query(
    `SELECT column_name, data_type
     FROM information_schema.columns
     WHERE table_name = 'query_cache'
     ORDER BY ordinal_position`
  );

  console.log("\nquery_cache columns now:");
  for (const row of rows) {
    console.log(`  - ${row.column_name} (${row.data_type})`);
  }

  console.log("\n[setup-semantic-cache] done.");
}

main()
  .catch((error) => {
    console.error("[setup-semantic-cache] failed:", error.message);
    process.exitCode = 1;
  })
  .finally(() => pool.end());