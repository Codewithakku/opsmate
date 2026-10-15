// backend/trace.js
// ======================================================
// PIPELINE TRACER
//
// One tracer per request. Every pipeline step produces:
//   1. ONE concise, numbered, aligned console line  ->  "proper staged" logging
//      (no JSON dumps, no per-layer chatter — the console reads as a short
//      list of steps per question)
//   2. ONE trace event, encoded in-band on the assist stream, so the UI can
//      render the very same pipeline live while the answer is still generating.
//
// Wire format (see server.js handleAssist):
//   __OPSMATE_TRACE__:{"n":3,"stage":"tool",...}\n
// JSON.stringify() never emits a raw newline, so the first "\n" after the
// prefix always terminates the event — several trace lines can therefore ride
// along inside a single answer stream, interleaved with the answer text.
// ======================================================

"use strict";

const TRACE_PREFIX = "__OPSMATE_TRACE__:";

// Fixed width keeps the console columns aligned, so a request reads as a block:
//   [k3f9] 1 ▸ understand  Query understood  ·  order question · ORD-1001
const STAGE_WIDTH = 10;

const STATUS_GLYPH = { run: "▸", ok: "✓", skip: "⏭", error: "✗" };

const clamp = (value, max) => {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

const newRequestId = () => Math.random().toString(36).slice(2, 6);

/**
 * @param {{id?: string, onEvent?: (event: object) => void, logger?: Console}} opts
 */
function createTracer({ id, onEvent, logger = console } = {}) {
  const requestId = id || newRequestId();
  const startedAt = Date.now();
  let step = 0;

  /**
   * Records ONE pipeline step: one console line, one streamed event.
   * @param {{stage: string, status?: "run"|"ok"|"skip"|"error", title?: string, detail?: string, ms?: number}} input
   */
  const stage = ({ stage: name, status = "ok", title = "", detail = "", ms } = {}) => {
    step += 1;

    const event = {
      n: step,
      stage: String(name || "step"),
      status,
      title: clamp(title, 72),
      detail: clamp(detail, 110),
      ms: Math.max(0, Math.round(ms ?? Date.now() - startedAt)),
    };

    const glyph = STATUS_GLYPH[status] || STATUS_GLYPH.ok;
    const line =
      `[${requestId}] ${event.n} ${glyph} ${event.stage.padEnd(STAGE_WIDTH)} ${event.title}` +
      (event.detail ? `  ·  ${event.detail}` : "");

    if (status === "error") logger.error(line);
    else logger.log(line);

    if (onEvent) {
      try {
        onEvent(event);
      } catch (_e) {
        // The client is gone; the console line above is already recorded.
      }
    }

    return event;
  };

  return { id: requestId, stage, elapsedMs: () => Date.now() - startedAt };
}

// Self-delimited trace line. No leading newline: the marker is located by
// index, so it can sit directly against the surrounding answer text without
// leaking blank lines into the displayed answer.
const encodeTrace = (event) => `${TRACE_PREFIX}${JSON.stringify(event)}\n`;

// Short human-readable tool args: order_id="ORD-1001", quantity=-5
const formatToolArgs = (args) =>
  Object.entries(args || {})
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(", ");

// One-line summary of what a tool actually returned, so the trace reads like
// the model's own reasoning loop ("find tool -> call it -> what came back").
function describeToolResult(name, result) {
  if (!result) return "no result";
  if (result.error) return String(result.error);
  if (result.message && result.isError) return String(result.message);

  const parts = (...values) => values.filter((v) => v !== null && v !== undefined && v !== "").join(" · ");

  switch (name) {
    case "get_order":
      return parts(
        result.order_id,
        result.status,
        Array.isArray(result.items) ? `${result.items.length} item(s)` : null
      );
    case "list_orders":
      return Array.isArray(result) ? `${result.length} order(s)` : parts(result.message);
    case "search_products":
      return Array.isArray(result)
        ? parts(
            `${result.length} match(es)`,
            result[0] ? `${result[0].name} · price ${result[0].price}` : null
          )
        : parts(result.message);
    case "check_stock":
    case "update_stock":
      return parts(result.sku, result.stock_quantity != null ? `stock ${result.stock_quantity}` : null);
    default:
      return "";
  }
}

module.exports = {
  TRACE_PREFIX,
  createTracer,
  encodeTrace,
  formatToolArgs,
  describeToolResult,
};
