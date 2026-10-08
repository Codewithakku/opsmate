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

/** Cache status reported by the /api/assist response headers. */
interface CacheInfo {
  layer: string; // HIT | MISS | COALESCED | L1_MEMORY | L2_REDIS | L3_SEMANTIC ...
  hit: boolean;
  tool: string | null; // e.g. "order-lookup" when the backend ran a tool call
}

/** Latency reported by Server-Timing + measured in the browser. */
interface LatencyInfo {
  firstTokenMs: number | null; // server: time to first streamed token
  serverMs: number | null; // server: total time (instant answers)
  totalMs: number; // browser: full round trip
}

interface CacheCounters {
  hits: number;
  misses: number;
}

// ============================================================
// Cache / latency presentation
// ============================================================

const CACHE_LAYER_LABELS: Record<string, string> = {
  HIT: "answer cache",
  MISS: "live model",
  COALESCED: "shared in-flight",
  L1_MEMORY: "L1 memory",
  L2_REDIS: "L2 Redis",
  L3_SEMANTIC: "L3 semantic",
  BACKEND: "backend",
  BYPASS: "bypassed · live data",
};

const MISS_LAYERS = new Set(["MISS", "BYPASS", ""]);

const isCacheHit = (layer: string) => !MISS_LAYERS.has(layer);

const cacheLayerLabel = (layer: string) => CACHE_LAYER_LABELS[layer] ?? layer;

function latencyText(latency: LatencyInfo): string {
  if (latency.firstTokenMs !== null) {
    return `first token ${latency.firstTokenMs}ms · total ${latency.totalMs}ms`;
  }
  if (latency.serverMs !== null) {
    return `answered in ${latency.serverMs}ms`;
  }
  return `total ${latency.totalMs}ms`;
}

function parseServerTiming(header: string | null): Record<string, number> {
  const out: Record<string, number> = {};
  if (!header) return out;

  for (const part of header.split(",")) {
    const match = part.trim().match(/^([\w-]+);dur=([\d.]+)$/);
    if (match) out[match[1]] = Number(match[2]);
  }
  return out;
}

// ============================================================
// Error presentation — specific, actionable messages only
// ============================================================

interface ErrorInfo {
  icon: string;
  title: string;
  body: string;
}

/** Pulls a human message out of a JSON (or plain text) error body. */
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
    body: "You've reached your AI credit limit. Add credits to the AI provider account (or switch AI_PROVIDER to a provider with credits), then try again.",
  };
}

function parseErrorInfo(raw: string): ErrorInfo {
  const msg = raw.toLowerCase();

  // 1. Credit / quota exhaustion — always checked first so a billing problem
  //    is never mislabelled as a rate limit (providers send both as 429).
  if (CREDIT_ERROR_RE.test(msg)) return creditError();

  // 2. Every provider failed — the body already lists per-provider reasons.
  if (PROVIDER_LIST_RE.test(raw)) {
    const reasons = raw
      .replace(PROVIDER_LIST_RE, "")
      .replace(/Please try again later\.?\s*$/i, "")
      .trim();

    const bullets = reasons
      .split("\n")
      .filter((line) => line.trim().startsWith("•"))
      .map((line) => line.replace(/^[\s•]+/, "").trim());

    // If every provider failed for the same billing reason, say that plainly.
    if (bullets.length > 0 && bullets.every((b) => CREDIT_ERROR_RE.test(b.toLowerCase()))) {
      return creditError();
    }

    return {
      icon: "⛔",
      title: "All AI Providers Failed",
      body: bullets.length > 0 ? bullets.join("\n") : reasons,
    };
  }

  // 3. Rate limiting
  if (msg.includes("rate limit") || msg.includes("too many requests") || msg.includes("429")) {
    return {
      icon: "⏳",
      title: "Too Many Requests",
      body: "The AI service is receiving too many requests right now. Wait a moment, then try again.",
    };
  }

  // 4. API key / authentication
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

  // 5. Timeout
  if (msg.includes("timeout") || msg.includes("timed out") || msg.includes("etimedout")) {
    return {
      icon: "⌛",
      title: "Request Timed Out",
      body: "The AI took too long to respond. Check your connection and try again.",
    };
  }

  // 6. Network / connection
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

  // 7. Server-side problems
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

  // 8. Empty input
  if (msg.includes("please type a question") || msg.includes("please enter a question")) {
    return {
      icon: "✏️",
      title: "Empty Question",
      body: "Type a question first, then press Ask.",
    };
  }

  // 9. Fallback — still readable, never a raw stack trace
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
  const [cacheCounters, setCacheCounters] = useState<CacheCounters>({ hits: 0, misses: 0 });

  const [currentUsage, setCurrentUsage] = useState<UsageStats | null>(null);
  const [totalUsage, setTotalUsage] = useState<UsageStats>({
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  });

  // Request cancellation
  const abortRef = useRef<AbortController | null>(null);

  // Holds the full streamed answer without a React re-render per chunk;
  // the UI is flushed once per animation frame instead.
  const streamedAnswerRef = useRef("");
  const animationFrameRef = useRef<number | null>(null);

  // ----------------------------------------------------------
  // Batched answer rendering
  // ----------------------------------------------------------

  function scheduleAnswerUpdate() {
    if (animationFrameRef.current !== null) return;

    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      setAnswer(streamedAnswerRef.current);
    });
  }

  function flushAnswerUpdate() {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setAnswer(streamedAnswerRef.current);
  }

  // ----------------------------------------------------------
  // Cleanup on unmount
  // ----------------------------------------------------------

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, []);

  // ----------------------------------------------------------
  // WebSocket status
  // ----------------------------------------------------------

  useEffect(() => {
    let ws: WebSocket | null = null;

    try {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:5000";
      ws = new WebSocket(wsUrl);

      ws.onopen = () => setWsStatus("WebSocket Connected");

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "status") console.log("WebSocket status:", data.message);
        } catch {
          // Ignore invalid WebSocket messages.
        }
      };

      ws.onerror = () => setWsStatus("WebSocket Standby");
      ws.onclose = () => setWsStatus("WebSocket Standby");
    } catch {
      // Deferred so the effect body itself stays free of synchronous setState.
      queueMicrotask(() => setWsStatus("WebSocket Standby"));
    }

    return () => ws?.close();
  }, []);

  // ----------------------------------------------------------
  // Main AI request
  // ----------------------------------------------------------

  async function runSearch(e?: FormEvent) {
    if (e) e.preventDefault();

    const q = query.trim();

    if (!q) {
      setError("Please enter a question first.");
      return;
    }
    if (loading) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setModel("");
    setCacheInfo(null);
    setLatency(null);

    streamedAnswerRef.current = "";
    const controller = new AbortController();
    abortRef.current = controller;

    const startedAt = performance.now();
    let chunkCount = 0;

    console.log(`[assist] → POST /api/assist "${q}"`);

    try {
      const res = await fetch("/api/assist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, history }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const raw = await res.text().catch(() => "");
        throw new Error(extractErrorMessage(raw) || `Request failed (${res.status})`);
      }

      // ---- Response metadata: cache status, provider, latency, tool ----
      const providerName = res.headers.get("x-provider") || "";
      const providerModel = res.headers.get("x-model") || "";
      const cacheLayer = res.headers.get("x-cache") || "MISS";
      const toolName = res.headers.get("x-tool");
      const timing = parseServerTiming(res.headers.get("server-timing"));

      setModel(providerModel ? `${providerName} · ${providerModel}` : providerName);
      setCacheInfo({ layer: cacheLayer, hit: isCacheHit(cacheLayer), tool: toolName });

      // ---- Stream tokens ----
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const delta = decoder.decode(value, { stream: true });
        if (!delta) continue;

        if (delta.includes(STREAM_ERROR_PREFIX)) {
          const friendlyMessage = delta.split(STREAM_ERROR_PREFIX)[1]?.trim();

          streamedAnswerRef.current = "";
          flushAnswerUpdate();
          setModel("");
          setCacheInfo(null);
          setLatency(null);
          setCurrentUsage(null);
          setError(
            friendlyMessage || "Something went wrong while getting a response. Please try again."
          );

          console.error("[assist] provider error:", friendlyMessage);
          return;
        }

        streamedAnswerRef.current += delta;
        chunkCount++;
        scheduleAnswerUpdate();
      }

      flushAnswerUpdate();

      const finalAnswer = streamedAnswerRef.current;
      const totalMs = Math.round(performance.now() - startedAt);

      setLatency({
        firstTokenMs: timing["first-token"] ?? null,
        serverMs: timing["total"] ?? null,
        totalMs,
      });

      const hit = isCacheHit(cacheLayer);
      setCacheCounters((prev) =>
        hit ? { ...prev, hits: prev.hits + 1 } : { ...prev, misses: prev.misses + 1 }
      );

      console.log(
        `[assist] ← ${cacheLayer} | ${chunkCount} chunks | ${finalAnswer.length} chars | ${totalMs}ms`
      );

      // ---- Token usage (estimated client-side) ----
      const historyCharacters = history.reduce((total, m) => total + m.content.length, 0);
      const promptEstimate = Math.ceil((q.length + historyCharacters) / 4);
      const completionEstimate = Math.ceil(finalAnswer.length / 4);

      const usage: UsageStats = {
        promptTokens: promptEstimate,
        completionTokens: completionEstimate,
        totalTokens: promptEstimate + completionEstimate,
      };

      setCurrentUsage(usage);
      setTotalUsage((prev) => ({
        promptTokens: prev.promptTokens + usage.promptTokens,
        completionTokens: prev.completionTokens + usage.completionTokens,
        totalTokens: prev.totalTokens + usage.totalTokens,
      }));

      // ---- Conversation history ----
      setHistory((prev) => [
        ...prev,
        { role: "user", content: q },
        { role: "assistant", content: finalAnswer || "(empty response)" },
      ]);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        console.log("[assist] ⛔ request aborted by user");
      } else {
        setError(err instanceof Error ? err.message : "Something went wrong");
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  // ----------------------------------------------------------
  // Stop / reset
  // ----------------------------------------------------------

  function stopResponse() {
    if (!abortRef.current) return;
    console.log("[assist] 🛑 stop requested — cancelling stream");
    abortRef.current.abort();
  }

  function resetSession() {
    abortRef.current?.abort();
    abortRef.current = null;

    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    streamedAnswerRef.current = "";
    setHistory([]);
    setCurrentUsage(null);
    setTotalUsage({ promptTokens: 0, completionTokens: 0, totalTokens: 0 });
    setCacheInfo(null);
    setLatency(null);
    setCacheCounters({ hits: 0, misses: 0 });
    setAnswer("");
    setQuery("");
    setError("");
    setModel("");
  }

  const totalQueries = cacheCounters.hits + cacheCounters.misses;
  const hitRate =
    totalQueries > 0 ? Math.round((cacheCounters.hits / totalQueries) * 100) : 0;

  // ----------------------------------------------------------
  // UI
  // ----------------------------------------------------------

  return (
    <main className="page">
      <div className="shell">
        {/* ================= HEADER ================= */}
        <header className="header">
          <div className="header-top">
            <div className="brand-group">
              <p className="brand">OpsMate</p>

              <span className={`ws-badge ${wsStatus.includes("Connected") ? "connected" : ""}`}>
                ● {wsStatus}
              </span>

              {totalQueries > 0 && (
                <span className="cache-badge stats" title="Session cache hit rate">
                  ⚡ {cacheCounters.hits}/{totalQueries} cache hits ({hitRate}%)
                </span>
              )}
            </div>

            {history.length > 0 && (
              <button type="button" onClick={resetSession} className="reset-btn">
                Reset Context
              </button>
            )}
          </div>

          <h1>Test the AI model</h1>

          <p className="sub">
            Ask anything — cached answers return instantly, everything else streams from the
            fastest healthy provider.
          </p>
        </header>

        {/* ================= SEARCH ================= */}
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

        {/* ================= LOADING ================= */}
        {loading && (
          <div className="loading-card" aria-live="polite">
            <div className="loading-spinner" />
            <span>Streaming tokens from AI model…</span>
          </div>
        )}

        {/* ================= ERROR ================= */}
        {error &&
          (() => {
            const { icon, title, body } = parseErrorInfo(error);

            return (
              <div className="error-card" role="alert" aria-live="assertive">
                <div className="error-card-header">
                  <span className="error-card-icon" aria-hidden="true">
                    {icon}
                  </span>
                  <p className="error-card-title">{title}</p>
                </div>

                <p className="error-card-body">{body}</p>

                <button type="button" className="error-card-retry" onClick={() => runSearch()}>
                  🔄 Try Again
                </button>
              </div>
            );
          })()}

        {/* ================= RESULT ================= */}
        {answer && (
          <section className="result" aria-live="polite">
            <div className="result-meta">
              {cacheInfo && (
                <span className={`cache-badge ${cacheInfo.hit ? "hit" : "miss"}`}>
                  {cacheInfo.hit ? "⚡ Cache HIT" : "❄ Cache MISS"}
                  <span className="badge-detail">· {cacheLayerLabel(cacheInfo.layer)}</span>
                </span>
              )}

              {cacheInfo?.tool && (
                <span className="meta-chip" title="The assistant called a tool to answer this">
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

        {/* ================= CONTEXT USAGE ================= */}
        {currentUsage && (
          <section className="context-usage" aria-label="Context Usage">
            <h2>📊 Context Usage</h2>

            <div className="usage-grid">
              <div className="usage-card current">
                <div className="card-header">
                  <span className="icon">⚡</span>
                  <h3>Current Request Context</h3>
                </div>

                <p className="description">Tokens used in the latest API request</p>

                <div className="token-metrics">
                  <div className="metric">
                    <span className="label">Prompt (Input):</span>
                    <span className="value">{currentUsage.promptTokens.toLocaleString()} tokens</span>
                  </div>

                  <div className="metric">
                    <span className="label">Response (Output):</span>
                    <span className="value">
                      {currentUsage.completionTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric total">
                    <span className="label">Current Request Total:</span>
                    <span className="value">{currentUsage.totalTokens.toLocaleString()} tokens</span>
                  </div>
                </div>
              </div>

              <div className="usage-card total">
                <div className="card-header">
                  <span className="icon">📈</span>
                  <h3>Total Context (Accumulated)</h3>
                </div>

                <p className="description">Total tokens consumed across all turns</p>

                <div className="token-metrics">
                  <div className="metric">
                    <span className="label">Total Prompt Tokens:</span>
                    <span className="value">{totalUsage.promptTokens.toLocaleString()} tokens</span>
                  </div>

                  <div className="metric">
                    <span className="label">Total Response Tokens:</span>
                    <span className="value">
                      {totalUsage.completionTokens.toLocaleString()} tokens
                    </span>
                  </div>

                  <div className="metric total">
                    <span className="label">Total Accumulated Context:</span>
                    <span className="value">{totalUsage.totalTokens.toLocaleString()} tokens</span>
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
