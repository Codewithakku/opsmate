// frontend/lib/stream-parser.ts
//
// Parser for the OpsMate `/api/assist` stream.
//
// The backend rides status information along with the answer text using three
// in-band markers:
//
//   __OPSMATE_TRACE__:        one pipeline step, JSON, terminated by "\n"
//                             (can appear MANY times, anywhere in the stream —
//                             before the first token and after the last one)
//   __OPSMATE_STREAM_META__:  one JSON object, always the very last thing sent
//   __OPSMATE_STREAM_ERROR__: a plain message, always the very last thing sent
//
// A chunk boundary can fall anywhere — even inside a marker or its payload — so
// a single chunk can never be judged on its own:
//   * text is only safe to display once it can no longer be the start of a marker
//     (that is what heldBackLength() buffers),
//   * a trace step is only emitted once its terminating newline has arrived,
//   * meta/error are terminal: everything after them is their payload, and they
//     are only interpreted once the stream has ended.

export interface TraceStep {
  n: number;
  stage: string;
  status: "run" | "ok" | "skip" | "error";
  title: string;
  detail: string;
  ms: number;
}

export interface StreamMeta {
  layer: string;
  provider?: string | null;
  model?: string | null;
  tool?: string | null;
  serverMs?: number | null;
  embeddingStatus: string | null;
  semanticStatus: string | null;
  similarity: number | null;
}

export interface StreamParseResult {
  text: string;
  traces: TraceStep[];
}

export interface StreamFinishResult {
  text: string;
  meta: StreamMeta | null;
  error: string | null;
}

export const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";
export const STREAM_META_PREFIX = "__OPSMATE_STREAM_META__:";
export const STREAM_TRACE_PREFIX = "__OPSMATE_TRACE__:";

// Meta and error are terminal; a trace marker is not (parsing resumes after it).
const TERMINAL_MARKERS = [STREAM_ERROR_PREFIX, STREAM_META_PREFIX];
const STREAM_MARKERS = [STREAM_TRACE_PREFIX, ...TERMINAL_MARKERS];

// First marker in `text`, if any.
function findStreamMarker(
  text: string
): { index: number; marker: string } | null {
  let found: { index: number; marker: string } | null = null;

  for (const marker of STREAM_MARKERS) {
    const index = text.indexOf(marker);

    if (index !== -1 && (found === null || index < found.index)) {
      found = { index, marker };
    }
  }

  return found;
}

// Length of the longest suffix of `text` that must stay buffered because it
// could still turn out to be the beginning of a marker split across chunks.
function heldBackLength(text: string): number {
  let longest = 0;

  for (const marker of STREAM_MARKERS) {
    const max = Math.min(marker.length - 1, text.length);

    for (let length = max; length > 0; length -= 1) {
      if (text.endsWith(marker.slice(0, length))) {
        if (length > longest) longest = length;
        break;
      }
    }
  }

  return longest;
}

function parseTraceStep(payload: string): TraceStep | null {
  try {
    const value = JSON.parse(payload) as Partial<TraceStep>;

    if (!value || typeof value.title !== "string") return null;

    return {
      n: typeof value.n === "number" ? value.n : 0,
      stage: typeof value.stage === "string" ? value.stage : "step",
      status:
        value.status === "run" ||
        value.status === "skip" ||
        value.status === "error"
          ? value.status
          : "ok",
      title: value.title,
      detail: typeof value.detail === "string" ? value.detail : "",
      ms: typeof value.ms === "number" ? value.ms : 0,
    };
  } catch {
    // A malformed step must never break the answer that surrounds it.
    return null;
  }
}

export class StreamMarkerParser {
  private buffer = "";

  // Set once a terminal marker (meta/error) is seen: everything after it is payload.
  private terminal: { marker: string; payload: string } | null = null;

  // Set while a trace step's JSON is being collected.
  private inTrace = false;
  private tracePayload = "";

  push(chunk: string): StreamParseResult {
    const result: StreamParseResult = { text: "", traces: [] };

    if (this.terminal) {
      this.terminal.payload += chunk;
      return result;
    }

    this.buffer += chunk;

    for (;;) {
      // ---- inside a trace payload: the step ends at the first newline ----
      if (this.inTrace) {
        const newline = this.buffer.indexOf("\n");

        if (newline === -1) {
          this.tracePayload += this.buffer;
          this.buffer = "";
          break;
        }

        this.tracePayload += this.buffer.slice(0, newline);
        this.buffer = this.buffer.slice(newline + 1);
        this.inTrace = false;

        const step = parseTraceStep(this.tracePayload);
        this.tracePayload = "";
        if (step) result.traces.push(step);
        continue;
      }

      const hit = findStreamMarker(this.buffer);

      if (!hit) {
        // Hold back a suffix that could still be the start of a marker.
        const hold = heldBackLength(this.buffer);
        const safeLength = this.buffer.length - hold;

        result.text += this.buffer.slice(0, safeLength);
        this.buffer = this.buffer.slice(safeLength);
        break;
      }

      result.text += this.buffer.slice(0, hit.index);
      this.buffer = this.buffer.slice(hit.index + hit.marker.length);

      if (hit.marker === STREAM_TRACE_PREFIX) {
        this.inTrace = true;
        this.tracePayload = "";
        continue;
      }

      // meta / error: terminal. Everything left is this marker's payload.
      this.terminal = { marker: hit.marker, payload: this.buffer };
      this.buffer = "";
      break;
    }

    return result;
  }

  // Releases the buffered tail and interprets a terminal marker. Safe to call
  // once the stream has ended (and only then).
  finish(): StreamFinishResult {
    const result: StreamFinishResult = { text: "", meta: null, error: null };

    if (this.terminal) {
      const payload = this.terminal.payload.trim();

      if (this.terminal.marker === STREAM_META_PREFIX) {
        try {
          result.meta = JSON.parse(payload) as StreamMeta;
        } catch {
          // Malformed metadata: keep whatever the response headers reported.
        }
      } else {
        result.error =
          payload ||
          "Something went wrong while getting a response. Please try again.";
      }

      this.terminal = null;
      this.buffer = "";
      return result;
    }

    // No terminal marker: the held-back tail can no longer be a marker.
    result.text = this.buffer;
    this.buffer = "";
    this.inTrace = false;
    this.tracePayload = "";

    return result;
  }
}
