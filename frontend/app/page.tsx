
"use client";

import {
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";

// ============================================================
// Types
// ============================================================

interface UsageStats {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface CacheInfo {
  layer: string;
  provider: string;
  hit: boolean;
  tool: string | null;
  embeddingStatus: string | null;
  semanticStatus: string | null;
  similarity: number | null;
}

interface StreamMeta {
  layer: string;
  embeddingStatus: string | null;
  semanticStatus: string | null;
  similarity: number | null;
}

interface LatencyInfo {
  firstTokenMs: number | null;
  serverMs: number | null;
  totalMs: number;
}

interface ErrorInfo {
  icon: string;
  title: string;
  body: string;
}

interface ActiveRequest {
  id: number;
  controller: AbortController;
  reader: ReadableStreamDefaultReader<Uint8Array> | null;
  cancelled: boolean;
}

const MAX_HISTORY_MESSAGES = 3;
const MAX_HISTORY_ITEMS = MAX_HISTORY_MESSAGES * 2;

const LAYER_LABELS: Record<string, string> = {
  L1_MEMORY: "In-memory cache",
  L2_POSTGRES: "PostgreSQL exact match",
  L2_SEMANTIC: "PostgreSQL semantic match",
  COALESCED: "Shared in-flight request",
  MISS: "Live LLM response",
  BYPASS: "Cache bypassed",
  BYPASS_ORDER: "Direct order lookup",
  GREETING: "Greeting handler",
  OUT_OF_SCOPE: "Out-of-scope handler",
};

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  postgres: "PostgreSQL",
  cache: "Cache",
  "scope-gate": "Scope gate",
};

// Only stored cache layers count as cache hits.
// COALESCED is shared in-flight work, not a stored cache hit.
const HIT_LAYERS = new Set([
  "L1_MEMORY",
  "L2_POSTGRES",
  "L2_SEMANTIC",
]);

const isCacheHit = (layer: string) => HIT_LAYERS.has(layer);

const cacheLayerLabel = (layer: string) =>
  LAYER_LABELS[layer] ?? layer;

const cacheProviderLabel = (provider: string) =>
  PROVIDER_LABELS[provider] ?? provider;

function cacheOutcome(layer: string): string {
  if (isCacheHit(layer)) return "CACHE HIT";
  if (layer === "COALESCED") return "SHARED REQUEST";
  if (layer === "BYPASS_ORDER") return "ORDER LOOKUP";
  if (layer === "GREETING") return "GREETING";
  if (layer === "OUT_OF_SCOPE") return "OUT OF SCOPE";
  if (layer === "BYPASS") return "CACHE BYPASSED";
  return "CACHE MISS";
}

function embeddingStatusDisplay(
  status: string | null
): { icon: string; text: string } | null {
  switch (status) {
    case "HIT":
      return { icon: "🧬", text: "Embedding generated" };
    case "MISS":
      return { icon: "🧬❌", text: "Embedding failed" };
    case "SKIPPED":
      return { icon: "🧬⏭", text: "Embedding skipped" };
    case "SHARED":
      return { icon: "🧬🔗", text: "Embedding shared" };
    default:
      return null;
  }
}

function semanticStatusDisplay(
  status: string | null,
  similarity: number | null
): { icon: string; text: string } | null {
  switch (status) {
    case "HIT":
      return {
        icon: "🎯",
        text:
          similarity !== null
            ? `Semantic HIT (${Math.round(similarity * 100)}% similarity)`
            : "Semantic cache HIT",
      };
    case "MISS":
      return { icon: "🔍", text: "Semantic cache MISS" };
    case "SKIPPED":
      return { icon: "⏭", text: "Semantic cache skipped" };
    case "NOT_CHECKED":
      return { icon: "➖", text: "Semantic cache not checked" };
    case "SHARED":
      return { icon: "🔗", text: "Semantic cache shared" };
    default:
      return null;
  }
}

function latencyText(latency: LatencyInfo): string {
  if (latency.firstTokenMs !== null) {
    return `First token ${latency.firstTokenMs}ms · total ${latency.totalMs}ms`;
  }

  if (latency.serverMs !== null) {
    return `Server ${latency.serverMs}ms · total ${latency.totalMs}ms`;
  }

  return `Total ${latency.totalMs}ms`;
}

function parseServerTiming(
  header: string | null
): Record<string, number> {
  const out: Record<string, number> = {};

  if (!header) return out;

  for (const part of header.split(",")) {
    const match = part.trim().match(/^([\w-]+);dur=([\d.]+)$/);

    if (match) {
      out[match[1]] = Number(match[2]);
    }
  }

  return out;
}

// ============================================================
// Streaming marker parser
//
// The backend rides status information along with the answer text
// (`__OPSMATE_STREAM_META__:` / `__OPSMATE_STREAM_ERROR__:`). A chunk boundary
// can fall anywhere — even inside a marker or its payload — so a single chunk
// can never be judged on its own: text is only safe to display once it can no
// longer be the start of a marker, and a marker is only acted on once the
// stream has ended (both markers always trail the answer).
// ============================================================

// [stream-marker-parser]
const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";
const STREAM_META_PREFIX = "__OPSMATE_STREAM_META__:";

const STREAM_MARKERS = [STREAM_ERROR_PREFIX, STREAM_META_PREFIX];

interface StreamParseResult {
  text: string;
  meta: StreamMeta | null;
  error: string | null;
}

// First complete marker in `text`, if any.
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

class StreamMarkerParser {
  private buffer = "";
  private marker: string | null = null;
  private payload = "";

  push(chunk: string): StreamParseResult {
    const result: StreamParseResult = { text: "", meta: null, error: null };

    // Already inside a marker payload: every further byte belongs to it.
    if (this.marker !== null) {
      this.payload += chunk;
      return result;
    }

    this.buffer += chunk;

    const hit = findStreamMarker(this.buffer);

    if (hit) {
      result.text = this.buffer.slice(0, hit.index);
      this.marker = hit.marker;
      this.payload = this.buffer.slice(hit.index + hit.marker.length);
      this.buffer = "";
      return result;
    }

    const hold = heldBackLength(this.buffer);

    result.text = hold > 0 ? this.buffer.slice(0, this.buffer.length - hold) : this.buffer;
    this.buffer = hold > 0 ? this.buffer.slice(this.buffer.length - hold) : "";

    return result;
  }

  // Releases the buffered tail and interprets a marker that completed. Safe to
  // call once the stream has ended (and only then).
  finish(): StreamParseResult {
    const result: StreamParseResult = { text: "", meta: null, error: null };

    if (this.marker === null) {
      result.text = this.buffer;
      this.buffer = "";
      return result;
    }

    const payload = this.payload.trim();

    if (this.marker === STREAM_META_PREFIX) {
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

    this.marker = null;
    this.payload = "";

    return result;
  }
}
// [/stream-marker-parser]

// ============================================================
// Error handling
// ============================================================

function extractErrorMessage(raw: string): string {
  try {
    const parsed = JSON.parse(raw);

    if (typeof parsed === "string") return parsed;

    return parsed?.message || parsed?.error || raw;
  } catch {
    return raw;
  }
}

const CREDIT_ERROR_RE = /credit|quota|billing|insufficient/i;
const PROVIDER_LIST_RE = /^AI Service Unavailable\s*/i;

function creditError(): ErrorInfo {
  return {
    icon: "💳",
    title: "AI Credit Limit Reached",
    body: "You've reached your AI credit limit. Add credits to the AI provider account or switch to a provider with available credits, then try again.",
  };
}

function parseErrorInfo(raw: string): ErrorInfo {
  const msg = raw.toLowerCase();

  if (CREDIT_ERROR_RE.test(msg)) {
    return creditError();
  }

  if (PROVIDER_LIST_RE.test(raw)) {
    const reasons = raw
      .replace(PROVIDER_LIST_RE, "")
      .replace(/Please try again later\.?\s*$/i, "")
      .trim();

    const bullets = reasons
      .split("\n")
      .filter((line) => line.trim().startsWith("•"))
      .map((line) => line.replace(/^[\s•]+/, "").trim());

    if (
      bullets.length > 0 &&
      bullets.every((item) => CREDIT_ERROR_RE.test(item))
    ) {
      return creditError();
    }

    return {
      icon: "⛔",
      title: "All AI Providers Failed",
      body: bullets.length > 0 ? bullets.join("\n") : reasons,
    };
  }

  if (
    msg.includes("rate limit") ||
    msg.includes("too many requests") ||
    msg.includes("429")
  ) {
    return {
      icon: "⏳",
      title: "Too Many Requests",
      body: "The AI service is receiving too many requests. Wait a moment, then try again.",
    };
  }

  if (
    msg.includes("api key") ||
    msg.includes("authentication") ||
    msg.includes("unauthorized") ||
    msg.includes("forbidden") ||
    msg.includes("401") ||
    msg.includes("403")
  ) {
    return {
      icon: "🔑",
      title: "Authentication Error",
      body: "The AI service rejected the API key. Check the key configuration, then try again.",
    };
  }

  if (
    msg.includes("timeout") ||
    msg.includes("timed out") ||
    msg.includes("etimedout")
  ) {
    return {
      icon: "⌛",
      title: "Request Timed Out",
      body: "The AI took too long to respond. Check your connection and try again.",
    };
  }

  if (
    msg.includes("network") ||
    msg.includes("fetch") ||
    msg.includes("econnrefused") ||
    msg.includes("failed to fetch")
  ) {
    return {
      icon: "📡",
      title: "Network Error",
      body: "Could not reach the assistant service. Check that the backend is running and your connection is stable.",
    };
  }

  if (
    msg.includes("500") ||
    msg.includes("502") ||
    msg.includes("503") ||
    msg.includes("504") ||
    msg.includes("service unavailable") ||
    msg.includes("temporarily unavailable") ||
    msg.includes("overloaded") ||
    msg.includes("unavailable")
  ) {
    return {
      icon: "🔧",
      title: "Service Temporarily Unavailable",
      body: "The AI service is having issues right now. Try again in a few minutes.",
    };
  }

  if (
    msg.includes("please type a question") ||
    msg.includes("please enter a question")
  ) {
    return {
      icon: "✏️",
      title: "Empty Question",
      body: "Type a question first, then press Ask.",
    };
  }

  return {
    icon: "⚠️",
    title: "Something Went Wrong",
    body: raw || "An unexpected error occurred. Please try again.",
  };
}

// ============================================================
// Main component
// ============================================================

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [wsStatus, setWsStatus] = useState("Connecting WebSocket...");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [cacheInfo, setCacheInfo] = useState<CacheInfo | null>(null);
  const [latency, setLatency] = useState<LatencyInfo | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [requestNumber, setRequestNumber] = useState<number | null>(null);

  const [currentUsage, setCurrentUsage] =
    useState<UsageStats | null>(null);

  const [totalUsage, setTotalUsage] = useState<UsageStats>({
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  });

  const activeRequestRef = useRef<ActiveRequest | null>(null);
  const requestIdRef = useRef(0);
  const streamedAnswerRef = useRef("");
  const animationFrameRef = useRef<number | null>(null);
  const streamMetaRef = useRef<StreamMeta | null>(null);

  function isCurrentRequest(request: ActiveRequest): boolean {
    return (
      activeRequestRef.current === request &&
      !request.cancelled &&
      !request.controller.signal.aborted
    );
  }

  function cancelAnswerFrame() {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }

  function scheduleAnswerUpdate(request: ActiveRequest) {
    if (!isCurrentRequest(request)) return;
    if (animationFrameRef.current !== null) return;

    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;

      if (isCurrentRequest(request)) {
        setAnswer(streamedAnswerRef.current);
      }
    });
  }

  function flushAnswerUpdate(request: ActiveRequest) {
    if (!isCurrentRequest(request)) return;

    cancelAnswerFrame();
    setAnswer(streamedAnswerRef.current);
  }

  // ============================================================
  // Stop request
  // ============================================================

  function cancelActiveRequest(preservePartialAnswer: boolean) {
    const request = activeRequestRef.current;

    if (!request) return;

    request.cancelled = true;
    activeRequestRef.current = null;

    console.log(`[Q-${request.id}] CANCEL | User stopped request`);

    request.controller.abort();

    if (request.reader) {
      void request.reader.cancel().catch(() => {});
    }

    cancelAnswerFrame();

    if (preservePartialAnswer) {
      setAnswer(streamedAnswerRef.current);
    } else {
      streamedAnswerRef.current = "";
      setAnswer("");
    }

    setLoading(false);
  }

  // ============================================================
  // Cleanup
  // ============================================================

  useEffect(() => {
    return () => {
      const request = activeRequestRef.current;

      if (request) {
        request.cancelled = true;
        activeRequestRef.current = null;
        request.controller.abort();

        if (request.reader) {
          void request.reader.cancel().catch(() => {});
        }
      }

      cancelAnswerFrame();
    };
  }, []);

  // ============================================================
  // WebSocket status
  // ============================================================

  useEffect(() => {
    let ws: WebSocket | null = null;

    try {
      const wsUrl =
        process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:5000";

      ws = new WebSocket(wsUrl);

      ws.onopen = () => setWsStatus("WebSocket Connected");

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "status") {
            console.log(`[WS] ${data.message}`);
          }
        } catch {
          // Ignore invalid WebSocket messages.
        }
      };

      ws.onerror = () => setWsStatus("WebSocket Standby");
      ws.onclose = () => setWsStatus("WebSocket Standby");
    } catch {
      queueMicrotask(() => setWsStatus("WebSocket Standby"));
    }

    return () => ws?.close();
  }, []);

  // ============================================================
  // Main request
  // ============================================================

  async function runSearch(e?: FormEvent) {
    e?.preventDefault();

    const q = query.trim();

    if (!q) {
      setError("Please enter a question first.");
      return;
    }

    if (activeRequestRef.current) return;

    const request: ActiveRequest = {
      id: ++requestIdRef.current,
      controller: new AbortController(),
      reader: null,
      cancelled: false,
    };

    activeRequestRef.current = request;

    setRequestNumber(request.id);
    setCurrentQuestion(q);
    setLoading(true);
    setError("");
    setAnswer("");
    setModel("");
    setCacheInfo(null);
    setLatency(null);
    setCurrentUsage(null);

    streamedAnswerRef.current = "";
    streamMetaRef.current = null;

    const startedAt = performance.now();
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

    console.log(`[Q-${request.id}] START | ${q}`);

    try {
      const requestHistory = history.slice(-MAX_HISTORY_ITEMS);

      const res = await fetch("/api/assist", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: q,
          history: requestHistory,
        }),
        signal: request.controller.signal,
      });

      if (!isCurrentRequest(request)) return;

      if (!res.ok || !res.body) {
        const raw = await res.text().catch(() => "");

        if (!isCurrentRequest(request)) return;

        throw new Error(
          extractErrorMessage(raw) || `Request failed (${res.status})`
        );
      }

      const providerName = res.headers.get("x-provider") || "";
      const providerModel = res.headers.get("x-model") || "";
      const headerLayer = res.headers.get("x-cache") || "MISS";
      const toolName = res.headers.get("x-tool");
      const timing = parseServerTiming(
        res.headers.get("server-timing")
      );

      const embeddingStatus = res.headers.get("x-embedding-cache");
      const semanticStatus = res.headers.get("x-semantic-cache");
      const similarityHeader = res.headers.get("x-similarity");

      setModel(
        providerModel
          ? `${providerName} · ${providerModel}`
          : providerName
      );

      setCacheInfo({
        layer: headerLayer,
        provider: providerName,
        hit: isCacheHit(headerLayer),
        tool: toolName,
        embeddingStatus,
        semanticStatus,
        similarity:
          similarityHeader !== null ? Number(similarityHeader) : null,
      });

      console.log(
        `[Q-${request.id}] CACHE | ${cacheOutcome(headerLayer)} | ${headerLayer} | ${cacheLayerLabel(headerLayer)}`
      );

      reader = res.body.getReader();
      request.reader = reader;

      const decoder = new TextDecoder();
      const parser = new StreamMarkerParser();

      const appendAnswerText = (text: string) => {
        if (!text) return;
        streamedAnswerRef.current += text;
        scheduleAnswerUpdate(request);
      };

      const applyParsed = (parsed: StreamParseResult) => {
        appendAnswerText(parsed.text);
        if (parsed.meta) streamMetaRef.current = parsed.meta;
      };

      while (true) {
        const { done, value } = await reader.read();

        if (!isCurrentRequest(request)) return;
        if (done) break;

        applyParsed(parser.push(decoder.decode(value, { stream: true })));
      }

      if (!isCurrentRequest(request)) return;

      // Flush the decoder, then let the parser release its buffered tail (it can
      // no longer be part of a marker) and interpret a trailing marker.
      applyParsed(parser.push(decoder.decode()));

      const ending = parser.finish();

      if (!isCurrentRequest(request)) return;

      if (ending.error !== null) {
        cancelAnswerFrame();

        streamedAnswerRef.current = "";

        setAnswer("");
        setModel("");
        setCacheInfo(null);
        setLatency(null);
        setCurrentUsage(null);

        setError(ending.error);

        console.error(`[Q-${request.id}] ERROR | ${ending.error}`);

        return;
      }

      if (ending.meta) streamMetaRef.current = ending.meta;

      appendAnswerText(ending.text);

      if (!isCurrentRequest(request)) return;

      flushAnswerUpdate(request);

      if (streamMetaRef.current) {
        const meta = streamMetaRef.current;

        setCacheInfo((prev) =>
          prev
            ? {
                ...prev,
                layer: meta.layer || prev.layer,
                hit: isCacheHit(meta.layer || prev.layer),
                embeddingStatus:
                  meta.embeddingStatus ?? prev.embeddingStatus,
                semanticStatus:
                  meta.semanticStatus ?? prev.semanticStatus,
                similarity: meta.similarity ?? prev.similarity,
              }
            : prev
        );
      }

      const finalAnswer = streamedAnswerRef.current;
      const totalMs = Math.round(performance.now() - startedAt);
      const finalLayer = streamMetaRef.current?.layer || headerLayer;

      setLatency({
        firstTokenMs: timing["first-token"] ?? null,
        serverMs: timing["total"] ?? null,
        totalMs,
      });

      console.log(
        `[Q-${request.id}] RESULT | ${cacheOutcome(finalLayer)} | ${finalLayer} | ${cacheProviderLabel(providerName)} | ${totalMs}ms`
      );

      // Estimated token usage
      const historyCharacters = requestHistory.reduce(
        (total, message) => total + message.content.length,
        0
      );

      const promptEstimate = Math.ceil(
        (q.length + historyCharacters) / 4
      );

      const completionEstimate = Math.ceil(finalAnswer.length / 4);

      const usage: UsageStats = {
        promptTokens: promptEstimate,
        completionTokens: completionEstimate,
        totalTokens: promptEstimate + completionEstimate,
      };

      if (!isCurrentRequest(request)) return;

      setCurrentUsage(usage);

      setTotalUsage((prev) => ({
        promptTokens: prev.promptTokens + usage.promptTokens,
        completionTokens: prev.completionTokens + usage.completionTokens,
        totalTokens: prev.totalTokens + usage.totalTokens,
      }));

      const completedTurn: ChatMessage[] = [
        { role: "user", content: q },
        {
          role: "assistant",
          content: finalAnswer || "(empty response)",
        },
      ];

      if (!isCurrentRequest(request)) return;

      setHistory((prev) =>
        [...prev, ...completedTurn].slice(-MAX_HISTORY_ITEMS)
      );
    } catch (err) {
      if (!isCurrentRequest(request)) {
        console.log(`[Q-${request.id}] CANCELLED | Ignored stale request`);
        return;
      }

      if (
        request.controller.signal.aborted ||
        (err instanceof Error && err.name === "AbortError")
      ) {
        console.log(`[Q-${request.id}] CANCELLED | Request aborted`);
        return;
      }

      const message =
        err instanceof Error ? err.message : "Something went wrong";

      setError(message);
      console.error(`[Q-${request.id}] ERROR | ${message}`);
    } finally {
      if (reader) {
        try {
          reader.releaseLock();
        } catch {
          // Reader may still have a pending read.
        }
      }

      if (activeRequestRef.current === request) {
        activeRequestRef.current = null;
        setLoading(false);
      }
    }
  }

  // ============================================================
  // Reset
  // ============================================================

  function stopResponse(event: ReactMouseEvent<HTMLButtonElement>) {
    // Stop and Ask share this slot, so React rewrites the same <button> node
    // while the click is still being dispatched. Cancelling the click's default
    // action is the second half of keeping a Stop click out of the form (the
    // first half is that neither button is a submit button any more).
    event.preventDefault();

    cancelActiveRequest(true);
  }

  function handleInputKeyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Enter") return;

    // The form deliberately has no submit button (its controls are plain
    // buttons), so Enter must not rely on the browser's implicit submission:
    // submit here instead — exactly once, even while the input is disabled.
    event.preventDefault();

    void runSearch();
  }

  function resetSession() {
    cancelActiveRequest(false);
    cancelAnswerFrame();

    streamedAnswerRef.current = "";
    streamMetaRef.current = null;

    setHistory([]);
    setCurrentUsage(null);

    setTotalUsage({
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    });

    setCacheInfo(null);
    setLatency(null);
    setAnswer("");
    setQuery("");
    setError("");
    setModel("");
    setCurrentQuestion("");
    setRequestNumber(null);
  }

  // ============================================================
  // UI values
  // ============================================================

  const embeddingBadge = cacheInfo
    ? embeddingStatusDisplay(cacheInfo.embeddingStatus)
    : null;

  const semanticBadge = cacheInfo
    ? semanticStatusDisplay(
        cacheInfo.semanticStatus,
        cacheInfo.similarity
      )
    : null;

  const isShared = cacheInfo?.layer === "COALESCED";

  const isBypass =
    cacheInfo?.layer === "BYPASS_ORDER" ||
    cacheInfo?.layer === "BYPASS" ||
    cacheInfo?.layer === "GREETING" ||
    cacheInfo?.layer === "OUT_OF_SCOPE";

  const visibleCacheStatus = cacheInfo
    ? cacheOutcome(cacheInfo.layer)
    : loading
      ? "CHECKING CACHE..."
      : "WAITING FOR QUESTION";

  const cacheBadgeClass =
    cacheInfo && isCacheHit(cacheInfo.layer)
      ? "hit"
      : cacheInfo && (isShared || isBypass)
        ? "stats"
        : "miss";

  return (
    <main className="page">
      <div className="shell">
        <header className="header">
          <div className="header-top">
            <div className="brand-group">
              <p className="brand">OpsMate</p>

              <span
                className={`ws-badge ${
                  wsStatus.includes("Connected") ? "connected" : ""
                }`}
              >
                ● {wsStatus}
              </span>
            </div>

            {/* Tracked with state, not the request ref: reading a ref during
                render would leave this button stuck at its first value. */}
            {(history.length > 0 || loading) && (
              <button
                type="button"
                onClick={resetSession}
                className="reset-btn"
              >
                Reset Context
              </button>
            )}
          </div>

          <h1>Test the AI model</h1>

          <p className="sub">
            Ask a question. Cached answers return quickly; cache misses are
            generated by the model.
          </p>

          <p className="sub">
            Conversation history: last {history.length / 2} of{" "}
            {MAX_HISTORY_MESSAGES} turns retained.
          </p>
        </header>

        {/* Search */}
        <form className="search" onSubmit={runSearch}>
          <label htmlFor="q" className="sr-only">
            Search
          </label>

          <input
            id="q"
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              if (error) setError("");
            }}
            placeholder="Ask the model something…"
            autoComplete="off"
            disabled={loading}
            onKeyDown={handleInputKeyDown}
          />

          {loading ? (
            <button
              type="button"
              onClick={stopResponse}
              className="stop-btn"
              aria-label="Stop generating response"
            >
              ■ Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void runSearch()}
              className="ask-btn"
            >
              Ask
            </button>
          )}
        </form>

        {/* Permanent response panel: position remains stable */}
        <section
          className="response-panel"
          aria-live="polite"
          aria-label="AI response"
        >
          <div className="response-panel-header">
            <div>
              <h2>AI Response</h2>

              {requestNumber !== null && (
                <p className="response-question">
                  Q-{requestNumber}: {currentQuestion}
                </p>
              )}
            </div>

            <span className={`cache-badge ${cacheBadgeClass}`}>
              {visibleCacheStatus}
            </span>
          </div>

          <div className="response-cache-info">
            <span>
              <strong>Layer:</strong>{" "}
              {cacheInfo
                ? cacheLayerLabel(cacheInfo.layer)
                : "Not determined yet"}
            </span>

            {model && (
              <span>
                <strong>Provider:</strong> {model}
              </span>
            )}

            {latency && (
              <span>
                <strong>Time:</strong> {latencyText(latency)}
              </span>
            )}

            {cacheInfo?.tool && (
              <span>
                <strong>Tool:</strong> {cacheInfo.tool}
              </span>
            )}
          </div>

          {loading && (
            <div className="response-loading">
              <span className="loading-spinner" />
              <span>
                {answer
                  ? "Generating response…"
                  : "Waiting for the first token…"}
              </span>
            </div>
          )}

          <div className="response-body">
            {answer ? (
              <pre>{answer}</pre>
            ) : loading ? (
              <p className="response-placeholder">
                The answer will appear here while it streams.
              </p>
            ) : error ? (
              <p className="response-placeholder">
                The response could not be completed. See the error below.
              </p>
            ) : (
              <p className="response-placeholder">
                Your answer will appear here.
              </p>
            )}
          </div>

          {cacheInfo && (
            <div className="response-details">
              {embeddingBadge && (
                <span className="meta-chip">
                  {embeddingBadge.icon} {embeddingBadge.text}
                </span>
              )}

              {semanticBadge && (
                <span className="meta-chip">
                  {semanticBadge.icon} {semanticBadge.text}
                </span>
              )}

              {cacheInfo.layer === "L1_MEMORY" && (
                <p className="response-explanation">
                  This answer came from the in-memory cache. The LLM did not
                  need to generate it again.
                </p>
              )}

              {cacheInfo.layer === "L2_POSTGRES" && (
                <p className="response-explanation">
                  An exact match was found in PostgreSQL cache.
                </p>
              )}

              {cacheInfo.layer === "L2_SEMANTIC" && (
                <p className="response-explanation">
                  A semantically similar cached answer was reused.
                </p>
              )}

              {cacheInfo.layer === "MISS" && (
                <p className="response-explanation">
                  No usable cache answer was found; the model generated a new
                  response.
                </p>
              )}

              {cacheInfo.layer === "BYPASS_ORDER" && (
                <p className="response-explanation">
                  Order details were looked up directly in PostgreSQL. This
                  is not a cache hit.
                </p>
              )}

              {isShared && (
                <p className="response-explanation">
                  This request shared work with another identical request; it
                  is not a stored-cache hit.
                </p>
              )}

              {isBypass &&
                cacheInfo.layer !== "BYPASS_ORDER" && (
                  <p className="response-explanation">
                    This request bypassed the regular cache path.
                  </p>
                )}
            </div>
          )}
        </section>

        {/* Error */}
        {error &&
          (() => {
            const { icon, title, body } = parseErrorInfo(error);

            return (
              <div
                className="error-card"
                role="alert"
                aria-live="assertive"
              >
                <div className="error-card-header">
                  <span className="error-card-icon" aria-hidden="true">
                    {icon}
                  </span>

                  <p className="error-card-title">{title}</p>
                </div>

                <p className="error-card-body">{body}</p>

                <button
                  type="button"
                  className="error-card-retry"
                  onClick={() => runSearch()}
                >
                  🔄 Try Again
                </button>
              </div>
            );
          })()}

        {/* Context usage */}
        {currentUsage && (
          <section className="context-usage" aria-label="Context Usage">
            <h2>📊 Context Usage</h2>

            <div className="usage-grid">
              <div className="usage-card current">
                <div className="card-header">
                  <span className="icon">⚡</span>
                  <h3>Current Request Context</h3>
                </div>

                <p className="description">
                  Estimated tokens for the latest request
                </p>

                <div className="token-metrics">
                  <div className="metric">
                    <span className="label">Prompt (Input):</span>
                    <span className="value">
                      {currentUsage.promptTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric">
                    <span className="label">Response (Output):</span>
                    <span className="value">
                      {currentUsage.completionTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric total">
                    <span className="label">Current Request Total:</span>
                    <span className="value">
                      {currentUsage.totalTokens.toLocaleString()} tokens
                    </span>
                  </div>
                </div>
              </div>

              <div className="usage-card total">
                <div className="card-header">
                  <span className="icon">📈</span>
                  <h3>Total Context (Accumulated)</h3>
                </div>

                <p className="description">
                  Estimated tokens across requests in this session
                </p>

                <div className="token-metrics">
                  <div className="metric">
                    <span className="label">Total Prompt Tokens:</span>
                    <span className="value">
                      {totalUsage.promptTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric">
                    <span className="label">Total Response Tokens:</span>
                    <span className="value">
                      {totalUsage.completionTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric total">
                    <span className="label">
                      Total Accumulated Context:
                    </span>
                    <span className="value">
                      {totalUsage.totalTokens.toLocaleString()} tokens
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
