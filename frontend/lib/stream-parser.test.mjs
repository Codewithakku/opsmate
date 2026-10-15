// frontend/lib/stream-parser.test.mjs
// Run with:  npm test   (from frontend/)
//
// The assist stream interleaves answer text with in-band markers and a chunk
// boundary can fall ANYWHERE, so the parser is verified by replaying one exact
// backend stream at every chunking (1 char, 7 random sizes, cuts inside the
// marker itself). This is the part of the tracing pipeline that is easiest to
// get subtly wrong and hardest to notice in the browser.

import { test } from "node:test";
import assert from "node:assert/strict";

import { StreamMarkerParser } from "./stream-parser.ts";

const trace = (n, stage, title, detail = "", ms = 3) =>
  `__OPSMATE_TRACE__:${JSON.stringify({ n, stage, status: "ok", title, detail, ms })}\n`;
const META = (fields) => `\n__OPSMATE_STREAM_META__:${JSON.stringify(fields)}`;
// The backend writes the error marker with no leading separator (the preceding
// trace line already ended with a newline).
const ERROR = (message) => `__OPSMATE_STREAM_ERROR__:${message}`;

const ANSWER =
  "Order ORD-1001 is SHIPPED. Next, check the carrier tracking information.";

// Exactly what the backend writes for "what is the status of order 1001":
// trace steps -> answer text -> final trace step -> meta marker.
const FULL_STREAM =
  trace(1, "understand", "Query understood", "order question · ORD-1001") +
  trace(2, "route", "Cache bypassed", "order data is live") +
  trace(3, "tool", "Tool selected by the model", "get_order") +
  trace(4, "answer", "Answer streaming", "first token 420ms", 420) +
  ANSWER +
  trace(5, "done", "Answered from BYPASS_ORDER", "93 char(s) · tool get_order", 812) +
  META({
    layer: "BYPASS_ORDER",
    provider: "postgres",
    model: "order-fast-path (postgres, no LLM)",
    tool: "get_order (SQL fast-path)",
    serverMs: 812,
  });

function consume(chunks, Parser = StreamMarkerParser) {
  const parser = new Parser();
  let text = "";
  const traces = [];

  for (const chunk of chunks) {
    const part = parser.push(chunk);
    text += part.text;
    traces.push(...part.traces);
  }

  const end = parser.finish();
  text += end.text;

  return { text, traces, meta: end.meta, error: end.error };
}

function assertFullStream(result, label) {
  // The answer is followed only by the separator that puts the terminal meta
  // marker on its own line — nothing of the protocol leaks into the text.
  assert.equal(result.text, ANSWER + "\n", `answer text (${label})`);
  assert.equal(result.traces.length, 5, `trace count (${label})`);
  assert.equal(
    result.traces.map((step) => step.stage).join(","),
    "understand,route,tool,answer,done",
    `stage order (${label})`
  );
  assert.equal(result.traces[3].ms, 420, `step timing preserved (${label})`);
  assert.equal(
    result.traces[4].title,
    "Answered from BYPASS_ORDER",
    `final step title (${label})`
  );
  assert.deepEqual(
    result.meta,
    {
      layer: "BYPASS_ORDER",
      provider: "postgres",
      model: "order-fast-path (postgres, no LLM)",
      tool: "get_order (SQL fast-path)",
      serverMs: 812,
    },
    `meta marker (${label})`
  );
}

test("whole stream in a single chunk", () => {
  assertFullStream(consume([FULL_STREAM]), "single chunk");
});

test("stream split one character at a time", () => {
  assertFullStream(consume([...FULL_STREAM]), "char by char");
});

test("stream split at 50 pseudo-random chunk sizes", () => {
  let seed = 42;
  const next = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);

  for (let pass = 0; pass < 50; pass += 1) {
    const chunks = [];

    for (let i = 0; i < FULL_STREAM.length; ) {
      const size = 1 + Math.floor(next() * 7);
      chunks.push(FULL_STREAM.slice(i, i + size));
      i += size;
    }

    assertFullStream(consume(chunks), `random pass ${pass}`);
  }
});

test("chunk boundary inside the trace marker itself", () => {
  const cut = FULL_STREAM.indexOf("__OPSMATE_TRACE__:") + 8;

  assertFullStream(
    consume([FULL_STREAM.slice(0, cut), FULL_STREAM.slice(cut)]),
    "inside marker"
  );
});

test("trace step split so its newline arrives in the next chunk", () => {
  const marker = "__OPSMATE_TRACE__:";
  const first = FULL_STREAM.indexOf(marker);
  const newline = FULL_STREAM.indexOf("\n", first);

  assertFullStream(
    consume([FULL_STREAM.slice(0, newline), FULL_STREAM.slice(newline)]),
    "newline in next chunk"
  );
});

test("cached answer with no trace steps still parses", () => {
  const stream =
    "Cached answer." +
    META({
      layer: "L1_MEMORY",
      embeddingStatus: "SKIPPED",
      semanticStatus: "SKIPPED",
      similarity: null,
    });

  const result = consume([...stream]);

  // "\n" is the separator that puts the meta marker on its own line.
  assert.equal(result.text, "Cached answer.\n");
  assert.equal(result.traces.length, 0);
  assert.equal(result.meta.layer, "L1_MEMORY");
});

test("error marker is reported and produces no answer text", () => {
  const result = consume([
    ...(trace(1, "understand", "Query understood") + ERROR("Ollama is not running.")),
  ]);

  assert.equal(result.text, "");
  assert.equal(result.traces.length, 1);
  assert.equal(result.error, "Ollama is not running.");
});

test("a partial marker prefix in the answer is released, not dropped", () => {
  const parser = new StreamMarkerParser();

  const first = parser.push("See __OPS");
  const second = parser.push("MATE lives");
  const end = parser.finish();

  assert.equal(first.text + second.text + end.text, "See __OPSMATE lives");
});

test("a malformed trace step never breaks the surrounding answer", () => {
  const stream = "__OPSMATE_TRACE__:not-json\nHello" + META({ layer: "MISS" });
  const result = consume([...stream]);

  // The newline that puts the meta marker on its own line belongs to the text;
  // it is whitespace inside <pre> and was already there before the trace tier.
  assert.equal(result.text, "Hello\n");
  assert.equal(result.traces.length, 0);
  assert.equal(result.meta.layer, "MISS");
});
