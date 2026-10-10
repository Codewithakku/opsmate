"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

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
  layer: string; // raw X-Cache value: "HIT" | "MISS" | "BYPASS"
  provider: string; // raw X-Provider value: SMALLTALK | ORDER_DB | CACHE | SEMANTIC_CACHE | ollama
  hit: boolean;
  tool: string | null;
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

// ============================================================
// Cache / latency presentation
//
// This backend has no in-memory / L1 cache and no multi-tier
// (L2 Postgres / L2 Redis / L3 semantic / coalesced) setup — it
// is DB-only caching plus a deterministic small-talk layer and a
// live order lookup. The badge below is built directly from the
// X-Provider header the backend actually sends, so it can never
// drift out of sync with reality again.
// ============================================================

const PROVIDER_LABELS: Record<string, string> = {
  SMALLTALK: "small talk",
  ORDER_DB: "live order lookup",
  CACHE: "database cache",
  SEMANTIC_CACHE: "semantic cache (Postgres)",
  ollama: "live model",
};

const MISS_LAYERS = new Set(["MISS", "BYPASS", ""]);

const isCacheHit = (layer: string) => !MISS_LAYERS.has(layer);

const cacheProviderLabel = (provider: string) =>
  PROVIDER_LABELS[provider] ?? provider;

function latencyText(latency: LatencyInfo): string {
  if (latency.firstTokenMs !== null) {
    return `first token ${latency.firstTokenMs}ms · total ${latency.totalMs}ms`;
  }

  if (latency.serverMs !== null) {
    return `answered in ${latency.serverMs}ms`;
  }

  return `total ${latency.totalMs}ms`;
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
// Error presentation
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
// Component
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

  const [currentUsage, setCurrentUsage] =
    useState<UsageStats | null>(null);

  const [totalUsage, setTotalUsage] = useState<UsageStats>({
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  });

  // Synchronous request ownership prevents overlapping requests.
  const activeRequestRef = useRef<ActiveRequest | null>(null);
  const requestIdRef = useRef(0);

  const streamedAnswerRef = useRef("");
  const animationFrameRef = useRef<number | null>(null);

  // ============================================================
  // Request ownership / rendering helpers
  // ============================================================

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

      if (!isCurrentRequest(request)) return;

      setAnswer(streamedAnswerRef.current);
    });
  }

  function flushAnswerUpdate(request: ActiveRequest) {
    if (!isCurrentRequest(request)) return;

    cancelAnswerFrame();
    setAnswer(streamedAnswerRef.current);
  }

  // ============================================================
  // Cancel active request
  // ============================================================

  function cancelActiveRequest(preservePartialAnswer: boolean) {
    const request = activeRequestRef.current;

    if (!request) return;

    // Invalidate ownership before aborting. Old async callbacks
    // cannot update the UI or affect a new request.
    request.cancelled = true;
    activeRequestRef.current = null;

    console.log(`[assist:${request.id}] cancellation requested`);

    request.controller.abort();

    if (request.reader) {
      void request.reader.cancel().catch(() => {
        // The stream may already be closed or errored.
      });
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
  // Cleanup on unmount
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
            console.log("WebSocket status:", data.message);
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
  // Main AI request
  // ============================================================

  async function runSearch(e?: FormEvent) {
    e?.preventDefault();

    const q = query.trim();

    if (!q) {
      setError("Please enter a question first.");
      return;
    }

    // Ref guard works synchronously, unlike the loading state.
    if (activeRequestRef.current) return;

    const request: ActiveRequest = {
      id: ++requestIdRef.current,
      controller: new AbortController(),
      reader: null,
      cancelled: false,
    };

    activeRequestRef.current = request;

    setLoading(true);
    setError("");
    setAnswer("");
    setModel("");
    setCacheInfo(null);
    setLatency(null);
    setCurrentUsage(null);

    streamedAnswerRef.current = "";

    const startedAt = performance.now();
    let chunkCount = 0;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

    console.log(`[assist:${request.id}] → POST /api/assist`);

    try {
      const requestHistory = history.slice(-MAX_HISTORY_ITEMS);

      console.log(
        `[assist:${request.id}] history: ${requestHistory.length} messages`
      );

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

      // Response metadata
      const providerName = res.headers.get("x-provider") || "";
      const providerModel = res.headers.get("x-model") || "";
      const cacheLayer = res.headers.get("x-cache") || "MISS";
      const toolName = res.headers.get("x-tool");
      const timing = parseServerTiming(
        res.headers.get("server-timing")
      );

      if (!isCurrentRequest(request)) return;

      setModel(
        providerModel ? `${providerName} · ${providerModel}` : providerName
      );

      setCacheInfo({
        layer: cacheLayer,
        provider: providerName,
        hit: isCacheHit(cacheLayer),
        tool: toolName,
      });

      // Stream tokens
      reader = res.body.getReader();
      request.reader = reader;

      const decoder = new TextDecoder();
      const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";

      while (true) {
        const { done, value } = await reader.read();

        // Stop may happen while reader.read() is pending.
        if (!isCurrentRequest(request)) return;

        if (done) break;

        const delta = decoder.decode(value, { stream: true });

        if (!delta) continue;

        if (delta.includes(STREAM_ERROR_PREFIX)) {
          const friendlyMessage = delta
            .split(STREAM_ERROR_PREFIX)[1]
            ?.trim();

          if (!isCurrentRequest(request)) return;

          streamedAnswerRef.current = "";
          cancelAnswerFrame();

          setAnswer("");
          setModel("");
          setCacheInfo(null);
          setLatency(null);
          setCurrentUsage(null);

          setError(
            friendlyMessage ||
              "Something went wrong while getting a response. Please try again."
          );

          console.error(
            `[assist:${request.id}] provider error:`,
            friendlyMessage
          );

          return;
        }

        streamedAnswerRef.current += delta;
        chunkCount++;

        scheduleAnswerUpdate(request);
      }

      // Flush any incomplete UTF-8 characters left in the decoder.
      const decoderTail = decoder.decode();

      if (!isCurrentRequest(request)) return;

      if (decoderTail) {
        streamedAnswerRef.current += decoderTail;
      }

      // Only a normally completed stream reaches this section.
      if (!isCurrentRequest(request)) return;

      flushAnswerUpdate(request);

      const finalAnswer = streamedAnswerRef.current;
      const totalMs = Math.round(performance.now() - startedAt);

      setLatency({
        firstTokenMs: timing["first-token"] ?? null,
        serverMs: timing["total"] ?? null,
        totalMs,
      });

      console.log(
        `[assist:${request.id}] ← ${cacheLayer} (${providerName}) | ${chunkCount} chunks | ${finalAnswer.length} chars | ${totalMs}ms`
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

      // Save history only after successful stream completion.
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
      // An old/cancelled request must not touch current UI state.
      if (!isCurrentRequest(request)) {
        console.log(`[assist:${request.id}] cancelled request ignored`);
        return;
      }

      if (
        request.controller.signal.aborted ||
        (err instanceof Error && err.name === "AbortError")
      ) {
        console.log(`[assist:${request.id}] request aborted`);
        return;
      }

      setError(
        err instanceof Error ? err.message : "Something went wrong"
      );
    } finally {
      if (reader) {
        // releaseLock can throw if a read is still pending.
        try {
          reader.releaseLock();
        } catch {
          // The reader may already be released or still have a pending read.
        }
      }

      // An old request must never clear a newer request's state.
      if (activeRequestRef.current === request) {
        activeRequestRef.current = null;
        setLoading(false);
      }

      console.log(`[assist:${request.id}] request cleanup completed`);
    }
  }

  // ============================================================
  // Stop / reset
  // ============================================================

  function stopResponse() {
    cancelActiveRequest(true);
  }

  function resetSession() {
    cancelActiveRequest(false);

    cancelAnswerFrame();

    streamedAnswerRef.current = "";

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
  }

  // ============================================================
  // UI
  // ============================================================

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

            {(history.length > 0 || activeRequestRef.current) && (
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
            Ask anything — cached answers return instantly, everything else
            streams from the fastest healthy provider.
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
            <button type="submit" className="ask-btn">
              Ask
            </button>
          )}
        </form>

        {/* Loading */}
        {loading && (
          <div className="loading-card" aria-live="polite">
            <div className="loading-spinner" />
            <span>Streaming tokens from AI model…</span>
          </div>
        )}

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

        {/* Result */}
        {answer && (
          <section className="result" aria-live="polite">
            <div className="result-meta">
              {cacheInfo && (
                <span
                  className={`cache-badge ${
                    cacheInfo.hit ? "hit" : "miss"
                  }`}
                >
                  {cacheInfo.hit ? "⚡ Cache HIT" : "❄ Cache MISS"}
                  <span className="badge-detail">
                    · {cacheProviderLabel(cacheInfo.provider)}
                  </span>
                </span>
              )}

              {cacheInfo?.tool && (
                <span
                  className="meta-chip"
                  title="The assistant called a tool to answer this"
                >
                  🔧 {cacheInfo.tool}
                </span>
              )}

              {model && <span className="meta-chip">🤖 {model}</span>}

              {latency && (
                <span className="meta-chip" title="Response latency">
                  ⏱ {latencyText(latency)}
                </span>
              )}
            </div>

            <pre>{answer}</pre>
          </section>
        )}

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
                    <span className="label">Total Accumulated Context:</span>
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