"use strict";

// ======================================================
// End-to-end cache-flow tests for backend/server.js
//
// Boots the REAL Express backend in-process (same code the user runs) and
// replaces the Ollama daemon with an in-memory mock so the tests are fast and
// deterministic. PostgreSQL is the real database from DATABASE_URL; every row
// the suite creates is prefixed with `test::` and deleted afterwards.
// ======================================================

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { Client } = require("pg");

// ------------------------------------------------------------------
// Environment: server.js calls dotenv.config({ override: true }) which would
// force PORT=5000 from backend/.env. Parse that file ourselves, apply the
// values, then make dotenv.config a no-op so the test controls the process
// environment (port, warm-up, mock Ollama host).
// ------------------------------------------------------------------
const dotenv = require("dotenv");
const fileEnv = dotenv.parse(fs.readFileSync(path.join(__dirname, "..", ".env")));
dotenv.config = () => ({ parsed: {} });
Object.assign(process.env, fileEnv);

process.env.PORT = process.env.TEST_PORT || "5199";
process.env.OLLAMA_WARMUP = "false"; // no warm-up call in tests
process.env.CACHE_ENABLED = "true";
process.env.CACHE_DB_TIMEOUT_MS = "8000";

// ------------------------------------------------------------------
// Mock Ollama FIRST: server.js builds its client at require time.
// ------------------------------------------------------------------
const ollamaCalls = [];
const mockState = { cancelled: 0 };

const realFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = (async (url, opts = {}) => {
  const target = typeof url === "string" ? url : url?.url || "";
  if (target.includes("/api/chat")) {
    const body = JSON.parse(opts.body || "{}");
    ollamaCalls.push(body);

    const lastUser = [...(body.messages || [])].reverse().find((m) => m.role === "user");
    const promptText = String(lastUser?.content || "").replace(/\s+/g, " ");
    const answer = `MOCK::${promptText.slice(0, 60)}`;
    const encoder = new TextEncoder();

    // A query containing SLOW_STREAM holds the response open so a test can press Stop.
    const slow = promptText.includes("SLOW_STREAM");
    const chunk = (content, done = false) =>
      encoder.encode(
        JSON.stringify({
          model: body.model,
          message: { role: "assistant", content },
          done,
          ...(done ? { prompt_eval_count: 11, eval_count: 4 } : {}),
        }) + "\n"
      );

    let timer = null;
    let counted = false;
    let ctl = null;
    const markCancelled = () => {
      if (!counted) {
        counted = true;
        mockState.cancelled += 1;
      }
      if (timer) clearTimeout(timer);
    };

    const stream = new ReadableStream({
      start(controller) {
        ctl = controller;
        if (!slow) {
          controller.enqueue(chunk(answer));
          controller.enqueue(chunk("", true));
          controller.close();
          return;
        }

        let n = 0;
        const push = () => {
          try {
            controller.enqueue(chunk(`tok${n} `));
          } catch (_e) {
            markCancelled(); // consumer went away (Stop pressed)
            return;
          }
          n += 1;
          if (n >= 60) {
            try {
              controller.enqueue(chunk("", true));
              controller.close();
            } catch (_e) { /* already errored */ }
            return;
          }
          timer = setTimeout(push, 100);
        };
        push();
      },
      cancel() { markCancelled(); },
      error() { markCancelled(); },
    });

    // Model a real HTTP abort: the SDK's AbortableAsyncIterator.abort() aborts the
    // fetch it handed us, which tears the response body down just like a socket close.
    const signal = opts.signal;
    if (signal) {
      const onAbort = () => {
        markCancelled();
        try { ctl?.error(new Error("mock: request aborted")); } catch (_e) { /* already closed */ }
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    return new Response(stream, { status: 200, headers: { "content-type": "application/x-ndjson" } });
  }

  return realFetch(url, opts);
});

const { PROMPT_VERSION } = require("../server.js");

const BASE = `http://127.0.0.1:${Number(process.env.PORT)}`;
const MARK = "test::";

let db;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

async function waitForHealth(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE}/health`);
      if (res.ok) return;
    } catch (_e) { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("backend did not become healthy in time");
}

async function search(query) {
  const res = await fetch(`${BASE}/api/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await res.json();
  return { res, data };
}

async function assist(query, history) {
  const res = await fetch(`${BASE}/api/assist`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, history }),
  });
  const body = await res.text();
  return { res, body };
}

async function flushMemory() {
  await fetch(`${BASE}/api/cache/flush`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ memory: true, database: false }),
  });
}

async function stats() {
  const res = await fetch(`${BASE}/api/cache/stats`);
  return res.json();
}

async function seedCache({ norm, answer, query, ttlSec = 3600, promptVersion = PROMPT_VERSION }) {
  await db.query(
    `INSERT INTO query_cache (query_norm, query, answer, model, prompt_version, latency_ms, created_at, expires_at)
     VALUES ($1, $2, $3, 'test-model', $4, 1, NOW(), NOW() + ($5 || ' seconds')::interval)
     ON CONFLICT (query_norm) DO UPDATE
       SET answer = EXCLUDED.answer,
           query = EXCLUDED.query,
           prompt_version = EXCLUDED.prompt_version,
           created_at = NOW(),
           expires_at = EXCLUDED.expires_at`,
    [norm, query ?? norm, answer, promptVersion, String(ttlSec)]
  );
}

async function getRow(norm) {
  const { rows } = await db.query("SELECT * FROM query_cache WHERE query_norm = $1", [norm]);
  return rows[0] || null;
}

// Audit writes are fire-and-forget; poll briefly so the assertion is not racy.
async function getAuditRows(query, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  let rows = [];
  for (;;) {
    ({ rows } = await db.query("SELECT * FROM queries WHERE query = $1 ORDER BY id", [query]));
    if (rows.length || Date.now() >= deadline) return rows;
    await new Promise((r) => setTimeout(r, 50));
  }
}

async function waitForAuditCount(query, expected, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { rows } = await db.query("SELECT COUNT(*)::int AS n FROM queries WHERE query = $1", [query]);
    if (rows[0].n === expected) return rows[0].n;
    await new Promise((r) => setTimeout(r, 50));
  }
  const { rows } = await db.query("SELECT COUNT(*)::int AS n FROM queries WHERE query = $1", [query]);
  return rows[0].n;
}

// ------------------------------------------------------------------
// lifecycle
// ------------------------------------------------------------------

test.before(async () => {
  await waitForHealth();
  db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();
});

test.after(async () => {
  try {
    await db.query("DELETE FROM query_cache WHERE query_norm LIKE $1", [`${MARK}%`]);
    await db.query("DELETE FROM queries WHERE query LIKE $1", [`${MARK}%`]);
  } finally {
    await db.end().catch(() => {});
  }
  process.exit(0); // stop the heartbeat interval / listener
});

test.beforeEach(async () => {
  await flushMemory();
});

// ------------------------------------------------------------------
// Bug 1: cache MISS even though the row exists in PostgreSQL
// ------------------------------------------------------------------

test("BUG1: a matching PostgreSQL row is served as an L2 hit, before any LLM call", async () => {
  const norm = `${MARK}l2 hit probe`;
  await seedCache({ norm, answer: "CACHED-ANSWER-A" });

  const llmBefore = ollamaCalls.length;
  const { res, data } = await search("TEST::L2 HIT probe???");

  assert.equal(res.headers.get("x-cache"), "L2_POSTGRES");
  assert.equal(data.cache.hit, true);
  assert.equal(data.answer, "CACHED-ANSWER-A");
  assert.equal(ollamaCalls.length, llmBefore, "an L2 hit must not call the LLM");
});

test("BUG1: lookup is normalized (case / whitespace / trailing punctuation)", async () => {
  const norm = `${MARK}normalized lookup`;
  await seedCache({ norm, answer: "CACHED-ANSWER-B" });

  const { res, data } = await search("   TeSt::NORMALIZED   lookup!!   ");
  assert.equal(res.headers.get("x-cache"), "L2_POSTGRES");
  assert.equal(data.answer, "CACHED-ANSWER-B");
});

test("BUG1: an expired row is correctly a MISS (TTL is still enforced)", async () => {
  const norm = `${MARK}expired probe`;
  await seedCache({ norm, answer: "STALE", ttlSec: -60 });

  assert.equal(await getRow(norm) !== null, true, "row should exist but be expired");
  const llmBefore = ollamaCalls.length;
  const { res } = await search(norm);
  assert.equal(res.headers.get("x-cache"), "MISS");
  assert.equal(ollamaCalls.length, llmBefore + 1);
});

test("BUG1: a MISS is persisted and served from L2 on the next request", async () => {
  const norm = `${MARK}miss then hit`;
  await db.query("DELETE FROM query_cache WHERE query_norm = $1", [norm]);

  const first = await search(norm);
  assert.equal(first.res.headers.get("x-cache"), "MISS");
  assert.ok(await getRow(norm), "MISS must write the row");

  await flushMemory();
  const second = await search(norm);
  assert.equal(second.res.headers.get("x-cache"), "L2_POSTGRES");
  assert.equal(second.data.answer, first.data.answer);
});

// ------------------------------------------------------------------
// Bug 2: conversation history
// ------------------------------------------------------------------

test("BUG2: the last 3 turns are sent in order and the current message is not duplicated", async () => {
  const history = [
    { role: "user", content: "turn1 user" },
    { role: "assistant", content: "turn1 assistant" },
    { role: "user", content: "turn2 user" },
    { role: "assistant", content: "turn2 assistant" },
    { role: "user", content: "turn3 user" },
    { role: "assistant", content: "turn3 assistant" },
  ];
  const query = `${MARK}history probe`;

  const before = ollamaCalls.length;
  await assist(query, history);
  assert.equal(ollamaCalls.length, before + 1, "uncached follow-up must reach the model");

  const messages = ollamaCalls[ollamaCalls.length - 1].messages;
  assert.equal(messages[0].role, "system");

  const conversation = messages.slice(1);
  assert.deepEqual(
    conversation.map((m) => m.content),
    [...history.map((m) => m.content), query],
    "history must be chronological and the current message appended exactly once"
  );
  assert.equal(
    conversation.filter((m) => m.content === query).length,
    1,
    "the current user message must not be duplicated"
  );
});

test("BUG2: history is preserved on a cache MISS", async () => {
  const history = [
    { role: "user", content: "prior q" },
    { role: "assistant", content: "prior a" },
  ];
  const query = `${MARK}history on miss`;

  const before = ollamaCalls.length;
  await assist(query, history);
  const messages = ollamaCalls[ollamaCalls.length - 1].messages;
  assert.equal(ollamaCalls.length, before + 1);
  assert.ok(messages.some((m) => m.content === "prior q"));
  assert.ok(messages.some((m) => m.content === "prior a"));
});

test("BUG2: only the latest 3 turns survive even if the client sends more", async () => {
  const history = [];
  for (let i = 1; i <= 5; i += 1) {
    history.push({ role: "user", content: `turn${i} user` });
    history.push({ role: "assistant", content: `turn${i} assistant` });
  }
  const query = `${MARK}history cap`;

  await assist(query, history);
  const conversation = ollamaCalls[ollamaCalls.length - 1].messages.slice(1);

  assert.deepEqual(conversation.map((m) => m.content), [
    "turn3 user", "turn3 assistant",
    "turn4 user", "turn4 assistant",
    "turn5 user", "turn5 assistant",
    query,
  ]);
});

test("BUG2: an echoed copy of the current message in history is not duplicated", async () => {
  const query = `${MARK}echoed current`;
  const history = [
    { role: "user", content: "earlier q" },
    { role: "assistant", content: "earlier a" },
    { role: "user", content: query }, // client accidentally echoes the current message
  ];

  await assist(query, history);
  const conversation = ollamaCalls[ollamaCalls.length - 1].messages.slice(1);

  assert.equal(conversation.filter((m) => m.content === query).length, 1);
  assert.deepEqual(conversation.map((m) => m.content), ["earlier q", "earlier a", query]);
});

// ------------------------------------------------------------------
// Bug 3: hits must not write; concurrent misses must not duplicate writes
// ------------------------------------------------------------------

test("BUG3: an L2 hit does not write to the cache table", async () => {
  const norm = `${MARK}hit no write`;
  await seedCache({ norm, answer: "CACHED-ANSWER-C" });
  const rowBefore = await getRow(norm);

  const writesBefore = (await stats()).dbWrites;
  const { res } = await search(norm);
  const writesAfter = (await stats()).dbWrites;

  assert.equal(res.headers.get("x-cache"), "L2_POSTGRES");
  assert.equal(writesAfter, writesBefore, "a cache hit must not perform a cache write");

  const rowAfter = await getRow(norm);
  assert.equal(rowAfter.answer, rowBefore.answer, "cached answer must be untouched");
  assert.equal(
    new Date(rowAfter.created_at).getTime(),
    new Date(rowBefore.created_at).getTime(),
    "cached row must not be re-written"
  );
  assert.equal(
    Number(rowAfter.hits),
    Number(rowBefore.hits),
    "a hit must not UPDATE the cached row (no writes at all on a hit)"
  );
});

test("BUG3: an L1 hit does not write to the cache table", async () => {
  const norm = `${MARK}l1 no write`;
  await seedCache({ norm, answer: "CACHED-ANSWER-D" });

  await search(norm); // populates L1
  const writesBefore = (await stats()).dbWrites;
  const { res } = await search(norm); // L1
  const writesAfter = (await stats()).dbWrites;

  assert.equal(res.headers.get("x-cache"), "L1_MEMORY");
  assert.equal(writesAfter, writesBefore, "an L1 hit must not perform a cache write");
});

test("BUG3: concurrent identical MISS requests share one LLM call and one write", async () => {
  const norm = `${MARK}concurrent coalesce`;
  await db.query("DELETE FROM query_cache WHERE query_norm = $1", [norm]);

  const llmBefore = ollamaCalls.length;
  const writesBefore = (await stats()).dbWrites;

  const results = await Promise.all(Array.from({ length: 8 }, () => search(norm)));

  const misses = results.filter((r) => r.res.headers.get("x-cache") === "MISS").length;
  const nonMisses = results.length - misses;

  assert.equal(misses, 1, `exactly one request should be a MISS (saw ${misses})`);
  assert.equal(nonMisses, 7, "the other concurrent requests must be COALESCED / cache hits");
  assert.equal(ollamaCalls.length - llmBefore, 1, "concurrent identical requests must share one LLM call");
  assert.equal((await stats()).dbWrites - writesBefore, 1, "concurrent identical requests must share one write");
});

test("BUG3: a cache hit inserts no audit row", async () => {
  const query = `${MARK}audit on hit`;
  await db.query("DELETE FROM queries WHERE query = $1", [query]);
  await seedCache({ norm: query, answer: "CACHED-ANSWER-E" });

  const { res } = await search(query);
  assert.equal(res.headers.get("x-cache"), "L2_POSTGRES");

  // Give the (fire-and-forget) audit write time to appear if it were going to.
  assert.equal(await waitForAuditCount(query, 1, 600), 0, "a cache hit must not write an audit row");
});

test("BUG3: repeated misses keep exactly one audit row per query (upsert)", async () => {
  const query = `${MARK}audit unique`;
  await db.query("DELETE FROM queries WHERE query = $1", [query]);
  await db.query("DELETE FROM query_cache WHERE query_norm = $1", [query]);

  const first = await search(query); // MISS -> one audit row
  assert.equal(first.res.headers.get("x-cache"), "MISS");

  // Force a second real MISS for the same query, then assert no duplicate row.
  await db.query("DELETE FROM query_cache WHERE query_norm = $1", [query]);
  await flushMemory();
  const second = await search(query);
  assert.equal(second.res.headers.get("x-cache"), "MISS");

  const rows = await getAuditRows(query);
  assert.equal(rows.length, 1, "the same question must never accumulate audit rows");
  assert.equal(rows[0].cached, false, "only genuine misses are audited");
});

// ------------------------------------------------------------------
// Stop button: cancelling the stream must cancel generation downstream
// ------------------------------------------------------------------

test("STOP: aborting the stream cancels generation and caches nothing", async () => {
  const query = `${MARK}SLOW_STREAM stop probe`;
  const norm = query.toLowerCase();
  await db.query("DELETE FROM query_cache WHERE query_norm = $1", [norm]);
  await db.query("DELETE FROM queries WHERE query = $1", [query]);
  await flushMemory();

  const controller = new AbortController();
  const cancelledBefore = mockState.cancelled;

  const res = await fetch(`${BASE}/api/assist`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, history: [] }),
    signal: controller.signal,
  });

  const reader = res.body.getReader();
  const first = await reader.read();
  assert.equal(first.done, false, "stream must start before Stop is pressed");

  controller.abort(); // <-- the user presses Stop

  // The backend must propagate this into the Ollama request.
  const deadline = Date.now() + 3000;
  while (Date.now() < deadline && mockState.cancelled === cancelledBefore) {
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.ok(
    mockState.cancelled > cancelledBefore,
    "the model stream must be cancelled after Stop"
  );

  await new Promise((r) => setTimeout(r, 400)); // let any (wrong) writes land

  assert.equal(await getRow(norm), null, "a stopped request must not write query_cache");
  assert.equal(
    await waitForAuditCount(query, 1, 500),
    0,
    "a stopped request must not write an audit row"
  );
});
