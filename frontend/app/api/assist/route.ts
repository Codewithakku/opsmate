import { createHash } from "crypto";
import { streamText } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { Index } from "@upstash/vector";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// ============================================================
// VECTOR DB (shared with backend /api/search cache) — writes are best-effort
// and non-blocking so a missing/stoped vector DB never breaks the answer.
// ============================================================

const VECTOR_URL = process.env.UPSTASH_VECTOR_REST_URL;
const VECTOR_TOKEN = process.env.UPSTASH_VECTOR_REST_TOKEN;

let vectorIndex: Index | null = null;
let embeddingGenerator: ((text: string) => Promise<number[]>) | null = null;

if (VECTOR_URL && VECTOR_TOKEN) {
  vectorIndex = new Index({
    url: VECTOR_URL,
    token: VECTOR_TOKEN,
  });

  // Same model the backend /api/search cache uses: text-embedding-3-small
  // (OpenAI embedding model). Produces the raw vector array for upsert.
  embeddingGenerator = async (text: string): Promise<number[]> => {
    const apiKey = process.env.OPENAI_API_KEY || process.env.OPEN_AI_API_KEY;
    if (!apiKey) return [];

    const res = await fetch(
      `https://api.openai.com/v1/embeddings?model=text-embedding-3-small&dimensions=1536`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: text, dimensions: 1536 }),
      }
    );

    if (!res.ok) return [];
    const data = (await res.json()) as { data?: { embedding?: number[] }[] };
    return data.data?.[0]?.embedding ?? [];
  };
}

function getVectorIndex(): Index | null {
  return vectorIndex;
}

function getEmbeddingGenerator(): ((text: string) => Promise<number[]>) | null {
  return embeddingGenerator;
}

// Deterministic upsert id (same as backend exact-key hashing) so re-runs
// update the existing vector entry instead of doubling it.
function vectorUpsertId(normalizedQuery: string): string {
  return `sem-assist-${createHash("sha1").update(normalizedQuery).digest("hex").slice(0, 32)}`;
}

// Best-effort store of a successful answer into the shared vector index.
async function storeVectorResponse(query: string, answer: string): Promise<void> {
  if (!vectorIndex || !embeddingGenerator) return;

  try {
    const normalized = query
      .toLowerCase()
      .replace(/\s+/g, " ")
      .replace(/[?!.]+$/g, "")
      .trim();

    if (!normalized) return;

    const embedding = await embeddingGenerator(normalized);

    await vectorIndex.upsert({
      id: vectorUpsertId(normalized),
      vector: embedding,
      metadata: {
        question: query,
        response: answer,
        promptVersion: PROMPT_VERSION,
        embeddingModel: "text-embedding-3-small",
        createdAt: Date.now(),
      },
    });
  } catch (err) {
    console.warn(`[assist] vector write-back skipped:`, getRawErrorMessage(err));
  }
}

// ============================================================
// LATENCY CONFIG (all overridable from .env)
// ============================================================

// How long a provider may take to emit its first token before we fall back.
const FIRST_TOKEN_TIMEOUT_MS = Number(process.env.FIRST_TOKEN_TIMEOUT_MS || 6000);

// Fewer output tokens = faster response.
const MAX_OUTPUT_TOKENS = Number(process.env.AI_MAX_TOKENS || 400);

// Longer history = more input tokens = slower first token.
const MAX_HISTORY_TURNS = Number(process.env.ASSIST_MAX_HISTORY || 8);

// Answer cache (in-memory L1, single-turn, non-dynamic queries only).
const CACHE_TTL_MS = Number(process.env.ASSIST_CACHE_TTL_SEC || 300) * 1000;
const CACHE_MAX_ENTRIES = Number(process.env.ASSIST_CACHE_MAX || 300);

// Single-flight: how long an identical in-flight query waits for the first
// attempt's answer before starting its own model call.
const COALESCE_WAIT_MS = Number(process.env.ASSIST_COALESCE_WAIT_MS || 15000);

// How long the last "all providers failed" message is replayed for.
const FAILURE_REPLAY_MS = Number(process.env.ASSIST_FAILURE_REPLAY_MS || 30000);

// Optional: Express backend (server.js) -> order lookup + fast-path + its caches.
const BACKEND_URL = (process.env.BACKEND_URL || "").replace(/\/$/, "");
const BACKEND_TIMEOUT_MS = Number(process.env.BACKEND_TIMEOUT_MS || 5000);

// Circuit breaker: providers that just failed are parked for a short window so
// a request never has to wait on known-dead providers. This is the difference
// between ~1s responses and ~15s ones when a provider is out of credits.
const PROVIDER_COOLDOWN_MS = Number(process.env.PROVIDER_COOLDOWN_MS || 20000);
const PROVIDER_COOLDOWN_HARD_MS = Number(process.env.PROVIDER_COOLDOWN_HARD_MS || 45000);

// Short system prompt: grounded + concise (fewer output tokens = faster).
const SYSTEM_PROMPT =
  process.env.ASSIST_SYSTEM_PROMPT ||
  "You are OpsMate, an AI assistant for support agents of an e-commerce order-management platform. " +
    "Answer in 1-3 short sentences, starting with the direct answer. " +
    "Never invent order details such as statuses, amounts, dates or tracking numbers; " +
    "if live order data is not provided, say which order ID or field is needed.";

const PROMPT_VERSION = createHash("sha1").update(SYSTEM_PROMPT).digest("hex").slice(0, 8);

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

function estTokens(text: string) {
  return Math.max(0, Math.ceil(text.length / 4));
}

// ============================================================
// ERROR CLASSIFICATION + FRIENDLY MESSAGES
// ============================================================

type ProviderErrorType =
  | "CREDITS_EXHAUSTED"
  | "RATE_LIMITED"
  | "TEMPORARILY_UNAVAILABLE"
  | "AUTH_ERROR"
  | "TIMEOUT"
  | "CLIENT_DISCONNECTED"
  | "UNKNOWN";

interface ProviderFailure {
  provider: string;
  model: string;
  type: ProviderErrorType;
  message: string;
}

/** Reads a nested property from an unknown value without casting to `any`. */
function readPath(source: unknown, path: string[]): unknown {
  let current: unknown = source;
  for (const key of path) {
    if (current === null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function getRawErrorMessage(err: unknown): string {
  const candidates: unknown[] = [
    readPath(err, ["message"]),
    readPath(err, ["cause", "message"]),
    readPath(err, ["cause", "error", "message"]),
    readPath(err, ["data", "error", "message"]),
    readPath(err, ["responseBody"]),
    err,
  ];

  const found = candidates.find((c) => typeof c === "string" && c.length > 0);
  if (typeof found === "string") return found;

  return err === undefined || err === null ? "" : String(err);
}

function classifyProviderError(err: unknown): ProviderErrorType {
  const message = getRawErrorMessage(err).toLowerCase();

  const rawStatus =
    readPath(err, ["statusCode"]) ??
    readPath(err, ["status"]) ??
    readPath(err, ["cause", "statusCode"]) ??
    readPath(err, ["cause", "status"]);
  const statusCode = typeof rawStatus === "number" ? rawStatus : undefined;

  if (
    message.includes("responseaborted") ||
    message.includes("aborted") ||
    message.includes("aborterror") ||
    message.includes("client disconnected")
  ) {
    return "CLIENT_DISCONNECTED";
  }

  // Credit / quota problems must win over the generic 429 branch: providers
  // report "insufficient_quota" WITH status 429, but the fix is billing, not waiting.
  if (
    message.includes("credit balance") ||
    message.includes("credits remaining") ||
    message.includes("insufficient credits") ||
    message.includes("credit_balance_exhausted") ||
    message.includes("insufficient_quota") ||
    message.includes("quota exceeded") ||
    message.includes("exceeded your current quota") ||
    message.includes("billing") ||
    message.includes("resource_exhausted")
  ) {
    return "CREDITS_EXHAUSTED";
  }

  if (
    statusCode === 429 ||
    message.includes("rate limit") ||
    message.includes("rate_limit") ||
    message.includes("too many requests")
  ) {
    return "RATE_LIMITED";
  }

  if (
    statusCode === 401 ||
    statusCode === 403 ||
    message.includes("api key") ||
    message.includes("apikey") ||
    message.includes("authentication") ||
    message.includes("unauthorized") ||
    message.includes("permission denied") ||
    message.includes("forbidden")
  ) {
    return "AUTH_ERROR";
  }

  if (
    statusCode === 500 ||
    statusCode === 502 ||
    statusCode === 503 ||
    statusCode === 504 ||
    message.includes("503") ||
    message.includes("502") ||
    message.includes("500") ||
    message.includes("504") ||
    message.includes("service unavailable") ||
    message.includes("temporarily unavailable") ||
    message.includes("high demand") ||
    message.includes("overloaded") ||
    message.includes("unavailable")
  ) {
    return "TEMPORARILY_UNAVAILABLE";
  }

  if (
    message.includes("timeout") ||
    message.includes("timed out") ||
    message.includes("etimedout") ||
    message.includes("deadline exceeded")
  ) {
    return "TIMEOUT";
  }

  return "UNKNOWN";
}

function getProviderFriendlyMessage(provider: string, type: ProviderErrorType): string {
  switch (type) {
    case "CREDITS_EXHAUSTED":
      return `You've reached the ${provider} AI credit limit (no API credits left).`;
    case "RATE_LIMITED":
      return `${provider} has temporarily reached its request limit.`;
    case "TEMPORARILY_UNAVAILABLE":
      return `${provider} is temporarily unavailable or experiencing high demand.`;
    case "AUTH_ERROR":
      return `${provider} rejected the API key (invalid or expired).`;
    case "TIMEOUT":
      return `${provider} took too long to respond.`;
    case "CLIENT_DISCONNECTED":
      return `The connection to ${provider} was interrupted.`;
    default:
      return `${provider} could not generate a response.`;
  }
}

function getAllProvidersFailedMessage(failures: ProviderFailure[]): string {
  if (failures.length === 0) {
    return (
      "AI Service Unavailable\n\n" +
      "All available AI providers failed to generate a response. " +
      "Please try again later."
    );
  }

  const lines = failures.map((f) => `• ${getProviderFriendlyMessage(f.provider, f.type)}`);

  return (
    "AI Service Unavailable\n\n" +
    "All available AI providers failed to generate a response.\n\n" +
    lines.join("\n") +
    "\n\nPlease try again later."
  );
}

function getFriendlyErrorMessage(model: string, err: unknown): string {
  const type = classifyProviderError(err);

  switch (type) {
    case "CREDITS_EXHAUSTED":
      return `You've reached the ${model} AI credit limit. Add credits to the ${model} account (or switch AI_PROVIDER), then try again.`;
    case "RATE_LIMITED":
      return `The ${model} AI service has temporarily reached its request limit. Please wait a moment and try again.`;
    case "TEMPORARILY_UNAVAILABLE":
      return `The ${model} AI service is temporarily unavailable or experiencing high demand. Please try again in a moment.`;
    case "AUTH_ERROR":
      return `The ${model} AI service rejected the API key. Please check the API configuration.`;
    case "TIMEOUT":
      return `The ${model} AI service took too long to respond. Please try again.`;
    case "CLIENT_DISCONNECTED":
      return `The connection to the ${model} AI service was interrupted.`;
    default:
      return `Sorry, the ${model} AI service could not generate a response right now. Please try again in a moment.`;
  }
}

// ============================================================
// CACHES + COUNTERS (stored on globalThis to survive dev hot-reload)
// ============================================================

declare global {
  // Cached data survives Next.js dev hot-reloads.
  var __opsmateAnswerCache: Map<string, { text: string; exp: number }>;
  var __opsmateCacheStats: {
    hits: number;
    misses: number;
    coalesced: number;
    backendHits: number;
  };
  var __opsmateInflight: Map<string, Promise<string>>;
  var __opsmateProviderCooldown: Map<string, number>;
  var __opsmateLastFailure: { message: string; at: number };
}

interface CacheStats {
  hits: number;
  misses: number;
  coalesced: number;
  backendHits: number;
}

const answerCache = (globalThis.__opsmateAnswerCache ||= new Map());

const stats: CacheStats = (globalThis.__opsmateCacheStats ||= {
  hits: 0,
  misses: 0,
  coalesced: 0,
  backendHits: 0,
});

// Identical in-flight queries -> one shared model call, everyone gets the answer.
const inflight = (globalThis.__opsmateInflight ||= new Map());

function normalizeQuery(q: string) {
  return q
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[?!.]+$/g, "")
    .trim();
}

function cacheKeyFor(query: string, history: ChatMessage[]): string | null {
  if (history.length > 0) return null;
  const n = normalizeQuery(query);
  if (!n || n.length > 500 || /\d/.test(n) || n.includes("@")) return null;
  return `${PROMPT_VERSION}|${n}`;
}

function cacheGet(key: string): string | null {
  const entry = answerCache.get(key);
  if (!entry) return null;
  if (entry.exp <= Date.now()) {
    answerCache.delete(key);
    return null;
  }
  answerCache.delete(key); // refresh recency
  answerCache.set(key, entry);
  return entry.text;
}

function cacheSet(key: string, text: string) {
  answerCache.delete(key);
  answerCache.set(key, { text, exp: Date.now() + CACHE_TTL_MS });
  if (answerCache.size > CACHE_MAX_ENTRIES) {
    const oldest = answerCache.keys().next().value;
    if (oldest !== undefined) answerCache.delete(oldest);
  }
}

function getCacheStats() {
  const served = stats.hits + stats.coalesced + stats.backendHits;
  const total = served + stats.misses;
  return {
    ...stats,
    hitRate: total ? Number((served / total).toFixed(3)) : 0,
    entries: answerCache.size,
    inFlight: inflight.size,
    promptVersion: PROMPT_VERSION,
  };
}

// ============================================================
// PROVIDER CIRCUIT BREAKER
//   - a provider that just failed is skipped for a short window
//   - when every provider is parked, the last failure is replayed
//     instantly instead of re-racing dead providers
// ============================================================

const providerCooldown = (globalThis.__opsmateProviderCooldown ||= new Map());
const providerLastFailure = (globalThis.__opsmateLastFailure ||= { message: "", at: 0 });

function providerKey(label: string, model: string) {
  return `${label}:${model}`;
}

function providerIsCoolingDown(label: string, model: string): boolean {
  const key = providerKey(label, model);
  const until = providerCooldown.get(key);

  if (until === undefined) return false;

  if (until <= Date.now()) {
    providerCooldown.delete(key);
    return false;
  }

  return true;
}

function cooldownProvider(label: string, model: string, type: ProviderErrorType) {
  // Hard failures (no credits, bad key) stay parked longer than transient ones.
  const hard = type === "CREDITS_EXHAUSTED" || type === "AUTH_ERROR";
  const ms = hard ? PROVIDER_COOLDOWN_HARD_MS : PROVIDER_COOLDOWN_MS;

  providerCooldown.set(providerKey(label, model), Date.now() + ms);
  console.warn(`[assist] 🧯 ${label} (${model}) parked for ${ms}ms after ${type}`);
}

function lastFailureFresh(): boolean {
  return (
    Boolean(providerLastFailure.message) &&
    Date.now() - providerLastFailure.at < FAILURE_REPLAY_MS
  );
}

// ============================================================
// HISTORY SANITISING (smaller prompt + safer)
// ============================================================

function sanitizeHistory(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];

  const cleaned: ChatMessage[] = raw
    .filter((m): m is { role: "user" | "assistant"; content: string } => {
      if (typeof m !== "object" || m === null) return false;
      const candidate = m as { role?: unknown; content?: unknown };
      return (
        (candidate.role === "user" || candidate.role === "assistant") && // client cannot inject "system"
        typeof candidate.content === "string" &&
        candidate.content.trim().length > 0
      );
    })
    .slice(-MAX_HISTORY_TURNS)
    .map((m) => ({
      role: m.role,
      content: m.content.slice(0, 4000),
    }));

  // Anthropic needs the conversation to start with a user turn.
  while (cleaned.length && cleaned[0].role !== "user") cleaned.shift();

  return cleaned;
}

// ============================================================
// BACKEND (server.js) ORDER LOOKUP — lookupOrder tool + fast path
// ============================================================

const ORDER_ID_RE =
  /\bord[-_\s]?\d{1,10}\b|\border\s*(?:#|no\.?|number|id)?\s*[:#]?\s*\d{3,10}\b|#\d{3,10}\b/i;

interface BackendAnswer {
  text: string;
  layer: string;
  model: string;
  tool: string | null;
}

async function tryBackendAnswer(query: string): Promise<BackendAnswer | null> {
  if (!BACKEND_URL || !ORDER_ID_RE.test(query)) return null;

  try {
    const res = await fetch(`${BACKEND_URL}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });

    if (!res.ok) return null;

    const data = (await res.json().catch(() => null)) as {
      answer?: unknown;
      model?: unknown;
      usedTool?: unknown;
      cache?: { layer?: unknown };
    } | null;

    if (!data?.answer) return null;

    return {
      text: String(data.answer),
      layer: typeof data.cache?.layer === "string" ? data.cache.layer : "BACKEND",
      model: typeof data.model === "string" ? data.model : "backend",
      tool: data.usedTool ? "order-lookup" : null,
    };
  } catch (err: unknown) {
    console.warn(
      "[assist] backend order lookup failed, falling back to LLM:",
      getRawErrorMessage(err)
    );
    return null;
  }
}

// ============================================================
// STREAM LOGGING + SUPABASE PERSISTENCE
// ============================================================

interface FinishInfo {
  text?: string;
  usage?: Record<string, unknown>;
}

function pickToken(usage: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = usage[key];
    if (typeof value === "number") return value;
  }
  return undefined;
}

function handleFinish(label: string, model: string, query: string, info: FinishInfo) {
  const usage = info.usage ?? {};
  const text = info.text ?? "";

  const promptTokens = pickToken(usage, "promptTokens", "inputTokens") ?? estTokens(query);
  const completionTokens =
    pickToken(usage, "completionTokens", "outputTokens") ?? estTokens(text);

  console.log(
    `[assist] ✅ ${label} (${model}) finished | prompt=${promptTokens} completion=${completionTokens} chars=${text.length}`
  );

  logQuery(query, text).catch(() => {});

  // Best-effort write-back: store the successful (query, answer) pair in the
  // shared Upstash Epsilla vector index so /api/search's L3 semantic lookup
  // can reuse it. Skipped on failure/timeout/429/cancel automatically because
  // this only runs after a complete successful stream.
  storeVectorResponse(query, text).catch(() => {});
}

async function logQuery(query: string, answer: string) {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

  if (!url || !key) return;

  try {
    await fetch(`${url}/rest/v1/queries`, {
      method: "POST",
      headers: {
        apikey: key,
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
        Prefer: "return=minimal",
      },
      body: JSON.stringify({
        query,
        answer: answer || null,
        created_at: new Date().toISOString(),
      }),
    }).catch(() => {});
  } catch {
    // Non-blocking
  }
}

// ============================================================
// RESPONSE BUILDERS
// ============================================================

// Marker used to carry a friendly error inside the text stream.
const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";

interface StreamOptions {
  signal: AbortSignal;
  cacheKey?: string | null;
  ttftMs?: number;
  /** Called once with the full text on success, or "" on failure/cancel. */
  onComplete?: (text: string) => void;
}

function toTextResponse(
  gen: AsyncGenerator<string>,
  providerLabel: string,
  model: string,
  opts: StreamOptions
): Response {
  const encoder = new TextEncoder();
  let cancelled = false;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const parts: string[] = [];
      let failed = false;

      try {
        for await (const delta of gen) {
          if (cancelled) break;
          parts.push(delta);
          controller.enqueue(encoder.encode(delta));
        }
      } catch (err: unknown) {
        failed = true;
        const errorType = classifyProviderError(err);

        if (errorType === "CLIENT_DISCONNECTED" || cancelled) {
          console.warn(`[assist] ⛔ ${providerLabel} stream aborted (client disconnected).`);
        } else {
          console.error(
            `[assist] ❌ ${providerLabel} stream error [${errorType}]:`,
            getRawErrorMessage(err)
          );
          try {
            controller.enqueue(
              encoder.encode(`\n\n${STREAM_ERROR_PREFIX}${getFriendlyErrorMessage(model, err)}`)
            );
          } catch {
            // Controller already closed.
          }
        }
      } finally {
        const text = parts.join("");

        // Cache only complete, successful, non-aborted answers.
        if (!failed && !cancelled && !opts.signal.aborted && opts.cacheKey && text.trim()) {
          cacheSet(opts.cacheKey, text);
        }

        // Release single-flight waiters ("" => they run their own attempt).
        opts.onComplete?.(!failed && !cancelled && !opts.signal.aborted ? text : "");

        try {
          controller.close();
        } catch {
          // Already closed.
        }
      }
    },

    cancel() {
      cancelled = true;
    },
  });

  const headers: Record<string, string> = {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "X-Model": model,
    "X-Provider": providerLabel,
    // No cache key (order ids, e-mail, multi-turn) => the query was never a
    // cache candidate; otherwise this answer is fresh and will be stored.
    "X-Cache": opts.cacheKey ? "MISS" : "BYPASS",
  };

  if (opts.ttftMs !== undefined) {
    headers["Server-Timing"] = `first-token;dur=${opts.ttftMs}`;
  }

  return new Response(stream, { status: 200, headers });
}

// Instant (non-streamed) response for cache / backend / coalesced hits.
function toInstantResponse(
  text: string,
  provider: string,
  model: string,
  cacheLayer: string,
  ms: number,
  extraHeaders: Record<string, string> = {}
): Response {
  return new Response(text, {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Model": model,
      "X-Provider": provider,
      "X-Cache": cacheLayer,
      "Server-Timing": `total;dur=${ms}`,
      ...extraHeaders,
    },
  });
}

// ============================================================
// SDK CLIENTS (created once, reused across requests)
// ============================================================

let anthropicClient: ReturnType<typeof createAnthropic> | null = null;
let openaiClient: ReturnType<typeof createOpenAI> | null = null;

const ANTHROPIC_MODEL = () =>
  process.env.ANTHROPIC_MODEL || process.env.AI_MODEL || "claude-haiku-4-5-20251001";
const OPENAI_MODEL = () =>
  process.env.OPEN_AI_MODEL || process.env.OPENAI_MODEL || "gpt-4o-mini";
const GEMINI_MODEL = () =>
  process.env.GEMINI_MODEL || process.env.AI_MODEL || "gemini-2.5-flash";

// streamText() swallows errors into onError — this helper rethrows them so
// fallback + classification still work.
interface SdkStreamResult {
  textStream: AsyncIterable<string>;
}

async function* iterateSdkStream(
  result: SdkStreamResult,
  state: { error: unknown }
): AsyncGenerator<string> {
  for await (const chunk of result.textStream) {
    yield chunk;
  }
  if (state.error) throw state.error;
}

type StreamTextOptions = Parameters<typeof streamText>[0];

function sdkOptions(base: Record<string, unknown>, state: { error: unknown }) {
  return {
    ...base,
    system: SYSTEM_PROMPT,
    temperature: 0.3,
    // v4 => maxTokens, v5 => maxOutputTokens (unknown keys are ignored)
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    maxTokens: MAX_OUTPUT_TOKENS,
    onError: (e: unknown) => {
      state.error = readPath(e, ["error"]) ?? e;
    },
  } as unknown as StreamTextOptions;
}

// ============================================================
// PROVIDER GENERATORS
// ============================================================

async function* genAnthropic(
  messages: ChatMessage[],
  signal: AbortSignal,
  query: string
): AsyncGenerator<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Anthropic API key is not configured");

  const model = ANTHROPIC_MODEL();
  anthropicClient ||= createAnthropic({ apiKey });

  const state = { error: null as unknown };

  const result = streamText(
    sdkOptions(
      {
        model: anthropicClient(model),
        messages,
        abortSignal: signal,
        onFinish: (info: FinishInfo) => handleFinish("anthropic", model, query, info),
      },
      state
    )
  );

  yield* iterateSdkStream(result, state);
}

async function* genOpenAI(
  messages: ChatMessage[],
  signal: AbortSignal,
  query: string
): AsyncGenerator<string> {
  const apiKey = process.env.OPENAI_API_KEY || process.env.OPEN_AI_API_KEY;
  if (!apiKey) throw new Error("OpenAI API key is not configured");

  const model = OPENAI_MODEL();
  openaiClient ||= createOpenAI({ apiKey });

  const state = { error: null as unknown };

  const result = streamText(
    sdkOptions(
      {
        model: openaiClient(model),
        messages,
        abortSignal: signal,
        onFinish: (info: FinishInfo) => handleFinish("openai", model, query, info),
      },
      state
    )
  );

  yield* iterateSdkStream(result, state);
}

// Gemini: full history + system prompt, API key in header (never in the URL).
async function* genGemini(
  messages: ChatMessage[],
  signal: AbortSignal,
  query: string
): AsyncGenerator<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Gemini API key is not configured");

  const model = GEMINI_MODEL();

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: messages.map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      })),
      generationConfig: {
        temperature: 0.3,
        maxOutputTokens: Math.max(MAX_OUTPUT_TOKENS, 256),
      },
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw Object.assign(new Error(`Gemini stream failed (${response.status}) ${detail}`), {
      statusCode: response.status,
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;

      const jsonStr = trimmed.slice(5).trim();
      if (!jsonStr || jsonStr === "[DONE]") continue;

      try {
        const data = JSON.parse(jsonStr);
        const parts = (data?.candidates?.[0]?.content?.parts ?? []) as Array<{
          text?: unknown;
        }>;
        const text = parts
          .map((p) => p.text)
          .filter(Boolean)
          .join("");

        if (text) {
          full += text;
          yield text;
        }
      } catch {
        // Ignore partial SSE frames.
      }
    }
  }

  handleFinish("gemini", model, query, { text: full });
}

async function* genFallback(): AsyncGenerator<string> {
  const msg =
    "I'm unable to generate an AI response right now because no AI provider API keys are configured. Please try again later.";

  const chunks = msg.match(/\S+\s*/g) ?? [msg];
  for (const word of chunks) yield word;
}

// ============================================================
// PROVIDER CHAIN + FIRST-TOKEN RACE
// ============================================================

interface ProviderConfig {
  label: string;
  model: string;
  make: (signal: AbortSignal) => AsyncGenerator<string>;
}

function buildProviderChain(messages: ChatMessage[], query: string): ProviderConfig[] {
  const primary = (process.env.AI_PROVIDER || "anthropic").toLowerCase();

  const ordered = [primary, ...["anthropic", "gemini", "openai"].filter((p) => p !== primary)];

  const chain: ProviderConfig[] = [];

  for (const p of ordered) {
    if (p === "anthropic" && process.env.ANTHROPIC_API_KEY) {
      chain.push({
        label: "anthropic",
        model: ANTHROPIC_MODEL(),
        make: (signal) => genAnthropic(messages, signal, query),
      });
    } else if (p === "gemini" && process.env.GEMINI_API_KEY) {
      chain.push({
        label: "gemini",
        model: GEMINI_MODEL(),
        make: (signal) => genGemini(messages, signal, query),
      });
    } else if (p === "openai" && (process.env.OPENAI_API_KEY || process.env.OPEN_AI_API_KEY)) {
      chain.push({
        label: "openai",
        model: OPENAI_MODEL(),
        make: (signal) => genOpenAI(messages, signal, query),
      });
    }
  }

  return chain;
}

async function* rewindGen(
  firstChunk: string,
  rest: AsyncGenerator<string>
): AsyncGenerator<string> {
  if (firstChunk) yield firstChunk;
  yield* rest;
}

type ResolvedProvider =
  | { ok: true; label: string; model: string; gen: AsyncGenerator<string>; ttftMs: number }
  | { ok: false; failures: ProviderFailure[] };

interface RaceAttempt {
  index: number;
  ok: boolean;
  value: string;
  error: unknown;
}

// All providers are asked for their first token at the same time and the
// fastest one wins. Sequential fallbacks paid the full timeout (or error
// round-trip) of every dead provider before reaching the healthy one.
async function resolveProvider(
  chain: ProviderConfig[],
  parentSignal: AbortSignal
): Promise<ResolvedProvider> {
  const failures: ProviderFailure[] = [];

  if (chain.length === 0 || parentSignal.aborted) {
    return { ok: false, failures };
  }

  const startedAt = performance.now();

  // One AbortController per provider so losers can be cancelled without
  // touching the winner or the client's connection.
  const controllers = chain.map(() => new AbortController());
  const abortAll = () => controllers.forEach((c) => c.abort());
  parentSignal.addEventListener("abort", abortAll, { once: true });

  const gens: AsyncGenerator<string>[] = chain.map((cfg, i) =>
    cfg.make(controllers[i].signal)
  );

  const timers: Array<ReturnType<typeof setTimeout>> = [];

  const attempts = chain.map(
    (cfg, index) =>
      new Promise<RaceAttempt>((resolve) => {
        let settled = false;
        const settle = (result: RaceAttempt) => {
          if (settled) return;
          settled = true;
          resolve(result);
        };

        // Hard ceiling on time-to-first-token for this attempt.
        timers.push(
          setTimeout(() => {
            controllers[index].abort();
            settle({
              index,
              ok: false,
              value: "",
              error: new Error(`${cfg.label} timed out waiting for first token`),
            });
          }, FIRST_TOKEN_TIMEOUT_MS)
        );

        gens[index]
          .next()
          .then(({ value, done }) => {
            if (done) {
              settle({
                index,
                ok: false,
                value: "",
                error: new Error(`${cfg.label} returned an empty stream`),
              });
            } else {
              settle({ index, ok: true, value: value ?? "", error: null });
            }
          })
          .catch((error) => settle({ index, ok: false, value: "", error }));
      })
  );

  // Resolve as soon as any provider yields a token; otherwise wait for every
  // failure so they can be reported together.
  const results = await new Promise<RaceAttempt[]>((resolve) => {
    const settled: RaceAttempt[] = [];
    let remaining = attempts.length;

    attempts.forEach((attempt) => {
      attempt.then((result) => {
        settled[result.index] = result;
        remaining -= 1;
        if (result.ok || remaining === 0) resolve(settled);
      });
    });
  });

  timers.forEach(clearTimeout);

  const winner = results.find((r) => r && r.ok);

  if (winner) {
    controllers.forEach((c, i) => {
      if (i !== winner.index) {
        c.abort();
        gens[i].return("").catch(() => {});
      }
    });

    const cfg = chain[winner.index];
    const ttftMs = Math.round(performance.now() - startedAt);

    console.log(
      `[assist] ✅ Provider resolved → ${cfg.label} (${cfg.model}) first token in ${ttftMs}ms (raced ${chain.length})`
    );

    // "abortAll" stays attached so a client disconnect still cancels upstream.
    return {
      ok: true,
      label: cfg.label,
      model: cfg.model,
      gen: rewindGen(winner.value, gens[winner.index]),
      ttftMs,
    };
  }

  // ---------- every provider failed ----------
  controllers.forEach((c) => c.abort());
  gens.forEach((gen) => gen.return("").catch(() => {}));

  chain.forEach((cfg, index) => {
    const error = results[index]?.error;
    const type = classifyProviderError(error);

    if (type === "CLIENT_DISCONNECTED") return;

    cooldownProvider(cfg.label, cfg.model, type);
    failures.push({
      provider: cfg.label,
      model: cfg.model,
      type,
      message: getRawErrorMessage(error),
    });
  });

  return { ok: false, failures };
}

function createErrorResponse(message: string, status = 200): Response {
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(`${STREAM_ERROR_PREFIX}${message}`));
      controller.close();
    },
  });

  return new Response(stream, {
    status,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}

// ============================================================
// MAIN /assist HANDLER
//   1. answer cache             (instant)
//   2. backend order lookup     (instant for status queries)
//   3. single-flight coalescing (shared in-flight model call)
//   4. LLM stream with fast fallback
// ============================================================

async function handleAssist(
  request: Request,
  query: string,
  rawHistory: unknown
): Promise<Response> {
  const t0 = performance.now();
  const trimmed = query?.trim() || "";

  if (!trimmed) {
    return new Response("Please type a question first.", {
      status: 400,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  const signal = request.signal;
  const history = sanitizeHistory(rawHistory);

  // ---------- 1. Answer cache ----------
  const cacheKey = cacheKeyFor(trimmed, history);

  if (cacheKey) {
    const cached = cacheGet(cacheKey);
    if (cached) {
      stats.hits++;
      console.log(`[assist] ⚡ cache HIT (${stats.hits} hits / ${stats.misses} misses)`);
      return toInstantResponse(cached, "cache", "cache", "HIT", Math.round(performance.now() - t0));
    }
  }

  // ---------- 2. Backend order lookup (lookupOrder tool + fast path) ----------
  const backend = await tryBackendAnswer(trimmed);

  if (backend) {
    stats.backendHits++;
    console.log(`[assist] ⚡ backend answer (${backend.layer})`);
    return toInstantResponse(
      backend.text,
      "backend",
      backend.model,
      backend.layer,
      Math.round(performance.now() - t0),
      backend.tool ? { "X-Tool": backend.tool } : {}
    );
  }

  // ---------- 3. Single-flight: share an identical in-flight model call ----------
  let releaseCoalesce: (text: string) => void = () => {};

  if (cacheKey) {
    const pending = inflight.get(cacheKey);

    if (pending) {
      const timeout = new Promise<string>((resolve) => {
        const timer = setTimeout(() => resolve(""), COALESCE_WAIT_MS);
        (timer as unknown as { unref?: () => void }).unref?.();
      });

      const shared = await Promise.race([pending, timeout]);

      if (shared) {
        stats.coalesced++;
        console.log("[assist] ⚡ coalesced onto identical in-flight query");
        return toInstantResponse(
          shared,
          "shared",
          "cache",
          "COALESCED",
          Math.round(performance.now() - t0)
        );
      }
      // First attempt failed or timed out — run our own model call (without
      // registering, so the original entry keeps ownership of the key).
    } else {
      let resolveShared!: (text: string) => void;
      const promise = new Promise<string>((resolve) => {
        resolveShared = resolve;
      });
      inflight.set(cacheKey, promise);

      releaseCoalesce = (text) => {
        if (inflight.get(cacheKey) === promise) inflight.delete(cacheKey);
        resolveShared(text);
      };
    }
  }

  try {
    // ---------- 4. LLM stream ----------
    const messages: ChatMessage[] = [...history, { role: "user", content: trimmed }];

    const providers = buildProviderChain(messages, trimmed);

    if (providers.length === 0) {
      console.warn("[assist] ⚠️ No API keys configured — using built-in fallback");
      releaseCoalesce("");
      return toTextResponse(genFallback(), "none", "none", { signal });
    }

    // Skip providers that recently failed. If every provider is parked, replay
    // the last failure instantly instead of re-racing dead providers.
    const healthy = providers.filter((c) => !providerIsCoolingDown(c.label, c.model));

    if (healthy.length === 0) {
      if (lastFailureFresh()) {
        console.warn("[assist] 🧯 all providers parked — serving last failure instantly");
        releaseCoalesce("");
        return createErrorResponse(providerLastFailure.message);
      }
      // Nothing healthy and no fresh remembered failure: race once to populate it.
    } else if (healthy.length < providers.length) {
      console.warn(
        `[assist] 🧯 parked: ${providers
          .filter((c) => providerIsCoolingDown(c.label, c.model))
          .map((c) => c.label)
          .join(", ")}`
      );
    }

    const chain = healthy.length > 0 ? healthy : providers;

    console.log(
      "[assist] 🔀 chain:",
      chain.map((c) => `${c.label}(${c.model})`).join(" | "),
      `| history=${history.length} query=${trimmed.length} chars`
    );

    const resolved = await resolveProvider(chain, signal);

    if (!resolved.ok) {
      for (const f of resolved.failures) {
        console.error(`[assist]    ${f.provider} (${f.model}) → ${f.type}`);
      }

      const message = getAllProvidersFailedMessage(resolved.failures);
      providerLastFailure.message = message;
      providerLastFailure.at = Date.now();

      releaseCoalesce("");
      return createErrorResponse(message);
    }

    // A provider answered — clear the remembered failure.
    providerLastFailure.message = "";
    providerLastFailure.at = 0;

    stats.misses++;

    return toTextResponse(resolved.gen, resolved.label, resolved.model, {
      signal,
      cacheKey,
      ttftMs: Math.round(performance.now() - t0),
      onComplete: releaseCoalesce,
    });
  } catch (err) {
    releaseCoalesce("");
    throw err;
  }
}

// ============================================================
// GET /api/assist?q=...   |   GET /api/assist?stats=1
// ============================================================

export async function GET(request: Request) {
  const url = new URL(request.url);

  if (url.searchParams.has("stats")) {
    return Response.json(getCacheStats());
  }

  return handleAssist(request, url.searchParams.get("q") ?? "", []);
}

// ============================================================
// POST /api/assist
// ============================================================

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));

    return handleAssist(request, body.query ?? body.q ?? "", body.history);
  } catch (err) {
    console.error(
      `[assist] ❌ Unexpected POST error [${classifyProviderError(err)}]:`,
      getRawErrorMessage(err)
    );

    return createErrorResponse(getFriendlyErrorMessage("AI", err));
  }
}
