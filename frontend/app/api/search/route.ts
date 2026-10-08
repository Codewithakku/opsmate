import { generateText, ModelMessage } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";

export const runtime = "nodejs";

/** Human-readable message from any thrown value. */
function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (err && typeof err === "object" && "message" in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
}

/** Reads a token count from a usage object across AI SDK versions. */
function pickToken(usage: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = usage[key];
    if (typeof value === "number") return value;
  }
  return undefined;
}

function usageRecord(usage: unknown): Record<string, unknown> {
  if (usage && typeof usage === "object") return usage as Record<string, unknown>;
  return {};
}

function traceRequest(messages: ModelMessage[], modelName: string) {
  console.log("\n========== 🧠 LLM REQUEST TRACE ==========");
  console.log(`Model: ${modelName}`);
  console.log(`Total messages: ${messages.length}`);

  messages.forEach((msg, i) => {
    const contentStr =
      typeof msg.content === "string"
        ? msg.content
        : JSON.stringify(msg.content);

    console.log(
      `[${i}] role=${msg.role} | length=${contentStr.length} chars | preview: ${contentStr.slice(
        0,
        150
      )}`
    );
  });

  console.log("===========================================\n");
}

// Vercel AI SDK — Anthropic
async function askAnthropic(messages: ModelMessage[]) {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  const modelName =
    process.env.ANTHROPIC_MODEL ||
    process.env.AI_MODEL ||
    "claude-3-5-haiku-latest";

  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY not configured in .env.local");
  }

  const anthropicProvider = createAnthropic({ apiKey });
  const model = anthropicProvider(modelName);

  traceRequest(messages, modelName);

  const result = await generateText({
    model,
    messages,
  });

  const usage = usageRecord(result.usage);

  const promptTokens =
    pickToken(usage, "promptTokens", "inputTokens") ??
    Math.ceil(JSON.stringify(messages).length / 4);

  const completionTokens =
    pickToken(usage, "completionTokens", "outputTokens") ??
    Math.ceil(result.text.length / 4);

  const totalTokens = pickToken(usage, "totalTokens") ?? promptTokens + completionTokens;

  return {
    answer: result.text,
    model: modelName,
    usage: {
      promptTokens,
      completionTokens,
      totalTokens,
    },
  };
}

// Vercel AI SDK — OpenAI
async function askOpenAI(messages: ModelMessage[]) {
  const apiKey =
    process.env.OPENAI_API_KEY ||
    process.env.OPEN_AI_API_KEY;

  const modelName =
    process.env.OPEN_AI_MODEL ||
    process.env.OPENAI_MODEL ||
    "gpt-4o-mini";

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY not configured in .env.local");
  }

  const openaiProvider = createOpenAI({ apiKey });
  const model = openaiProvider(modelName);

  traceRequest(messages, modelName);

  const result = await generateText({
    model,
    messages,
  });

  const usage = usageRecord(result.usage);

  const promptTokens =
    pickToken(usage, "promptTokens", "inputTokens") ??
    Math.ceil(JSON.stringify(messages).length / 4);

  const completionTokens =
    pickToken(usage, "completionTokens", "outputTokens") ??
    Math.ceil(result.text.length / 4);

  const totalTokens = pickToken(usage, "totalTokens") ?? promptTokens + completionTokens;

  return {
    answer: result.text,
    model: modelName,
    usage: {
      promptTokens,
      completionTokens,
      totalTokens,
    },
  };
}

// Google Gemini API
async function askGemini(query: string) {
  const apiKey = process.env.GEMINI_API_KEY;

  const modelName =
    process.env.GEMINI_MODEL ||
    process.env.AI_MODEL ||
    "gemini-3.8-flash";

  if (!apiKey) {
    throw new Error("GEMINI_API_KEY not configured");
  }

  const endpoint =
    `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;

  const body = {
    contents: [
      {
        role: "user",
        parts: [{ text: query }],
      },
    ],
    generationConfig: {
      temperature: 0.7,
      maxOutputTokens: 1024,
    },
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
        `Gemini request failed (${response.status})`
    );
  }

  const answer =
    data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

  const meta = data?.usageMetadata || {};

  const promptTokens =
    meta.promptTokenCount ??
    Math.max(1, Math.ceil(query.length / 4));

  const completionTokens =
    meta.candidatesTokenCount ??
    Math.max(1, Math.ceil(answer.length / 4));

  return {
    answer,
    model: modelName,
    usage: {
      promptTokens,
      completionTokens,
      totalTokens:
        meta.totalTokenCount ??
        promptTokens + completionTokens,
    },
  };
}

// Save query to Supabase
async function logQuery(query: string, answer: string) {
  const url =
    process.env.SUPABASE_URL ||
    process.env.NEXT_PUBLIC_SUPABASE_URL;

  const key =
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_ANON_KEY;

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

// Main handler
async function handleQuery(
  query: string | null,
  history: ModelMessage[] = []
) {
  const trimmed = query?.trim() || "";

  if (!trimmed) {
    console.warn("[1] ❌ Empty request received");

    return Response.json(
      {
        error: "Please type a question first.",
      },
      { status: 400 }
    );
  }

  const provider = (
    process.env.AI_PROVIDER || "anthropic"
  ).toLowerCase();

  const messages: ModelMessage[] = [
    ...history,
    {
      role: "user",
      content: trimmed,
    },
  ];

  // =========================================================
  // REQUEST TRACE START
  // =========================================================

  console.log("\n");
  console.log("========== 🧠 OPSMATE REQUEST TRACE ==========");

  console.log("[1] 📥 Request received");
  console.log("    Message count:", messages.length);

  console.log("[2] 🔀 Provider configuration");
  console.log("    Primary provider:", provider);

  // =========================================================
  // TIER 1 — ANTHROPIC
  // =========================================================

  if (provider === "anthropic") {
    console.log("\n[3] 🤖 Trying Anthropic");

    try {
      if (process.env.ANTHROPIC_API_KEY) {
        console.log(
          "    Model:",
          process.env.ANTHROPIC_MODEL || "default"
        );

        console.log("    Sending request to Anthropic...");

        const resData = await askAnthropic(messages);

        console.log("    ✅ Anthropic response received");
        console.log("    💬 Final answer generated");

        await logQuery(trimmed, resData.answer);

        console.log("========== ✅ REQUEST COMPLETE ==========\n");

        return Response.json(
          {
            query: trimmed,
            ...resData,
          },
          { status: 200 }
        );
      }

      console.warn(
        "    ⚠️ Anthropic API key is not configured"
      );
    } catch (anthropicErr) {
      console.warn(
        "    ❌ Anthropic request failed"
      );

      console.warn(
        "    Reason:",
        getErrorMessage(anthropicErr)
      );

      console.log(
        "    🔀 Moving to next AI provider..."
      );
    }
  }

  // =========================================================
  // TIER 2 — GEMINI
  // =========================================================

  if (
    provider === "gemini" ||
    process.env.GEMINI_API_KEY
  ) {
    console.log("\n[4] 🤖 Trying Gemini");

    try {
      if (process.env.GEMINI_API_KEY) {
        console.log(
          "    Model:",
          process.env.GEMINI_MODEL || "default"
        );

        console.log(
          "    Sending request to Gemini..."
        );

        const resData = await askGemini(trimmed);

        console.log(
          "    ✅ Gemini response received"
        );

        console.log(
          "    💬 Final answer generated"
        );

        await logQuery(trimmed, resData.answer);

        console.log(
          "========== ✅ REQUEST COMPLETE ==========\n"
        );

        return Response.json(
          {
            query: trimmed,
            ...resData,
          },
          { status: 200 }
        );
      }

      console.warn(
        "    ⚠️ Gemini API key is not configured"
      );
    } catch (geminiErr) {
      console.warn(
        "    ❌ Gemini request failed"
      );

      console.warn(
        "    Reason:",
        getErrorMessage(geminiErr)
      );

      console.log(
        "    🔀 Moving to next AI provider..."
      );
    }
  }

  // =========================================================
  // TIER 3 — OPENAI
  // =========================================================

  console.log("\n[5] 🤖 Trying OpenAI");

  try {
    const apiKey =
      process.env.OPENAI_API_KEY ||
      process.env.OPEN_AI_API_KEY;

    if (apiKey) {
      console.log(
        "    Model:",
        process.env.OPENAI_MODEL || "default"
      );

      console.log(
        "    Sending request to OpenAI..."
      );

      const resData = await askOpenAI(messages);

      console.log(
        "    ✅ OpenAI response received"
      );

      console.log(
        "    💬 Final answer generated"
      );

      await logQuery(trimmed, resData.answer);

      console.log("========== ✅ REQUEST COMPLETE ==========\n");

      return Response.json(
        {
          query: trimmed,
          ...resData,
        },
        { status: 200 }
      );
    }

    console.warn(
      "    ⚠️ OpenAI API key is not configured"
    );
  } catch (openAiErr) {
    console.warn(
      "    ❌ OpenAI request failed"
    );

    console.warn(
      "    Reason:",
      getErrorMessage(openAiErr)
    );

    console.log(
      "    🔀 Moving to backend fallback..."
    );
  }

  // =========================================================
  // TIER 4 — EXPRESS BACKEND
  // =========================================================

  console.log("\n[6] 🔌 Trying Express backend fallback");

  try {
    // Server-side fallback call: go DIRECTLY to the backend. (This handler owns
    // /api/search, so routing through the frontend origin would loop back here.)
    const backendBase =
      process.env.BACKEND_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:5000";

    console.log(
      "    Backend URL:",
      backendBase
    );
    console.log(
      "    Sending request to Express backend..."
    );

    const res = await fetch(
      `${backendBase}/api/search`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmed,
        }),
      }
    );

    if (res.ok) {
      const data = await res.json();

      console.log(
        "    ✅ Express backend response received"
      );

      console.log(
        "    💬 Final answer returned by backend"
      );

      console.log(
        "========== ✅ REQUEST COMPLETE ==========\n"
      );

      return Response.json(
        data,
        { status: 200 }
      );
    }

    console.warn(
      "    ❌ Express backend returned HTTP",
      res.status
    );
  } catch (backendErr) {
    console.warn(
      "    ❌ Express backend request failed"
    );

    console.warn(
      "    Reason:",
      getErrorMessage(backendErr)
    );
  }

  // =========================================================
  // TIER 5 — FRIENDLY LOCAL FALLBACK
  // =========================================================

  console.warn(
    "\n[7] ⚠️ All AI services are currently unavailable"
  );

  console.log(
    "    Returning user-friendly fallback message"
  );

  console.log(
    "========== ⚠️ REQUEST FINISHED WITH FALLBACK ==========\n"
  );

  return Response.json(
    {
      query: trimmed,

      answer:
        "I'm unable to generate an AI response right now because the AI services are temporarily unavailable. Please try again in a moment.",

      model: "fallback",

      usage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
    },
    { status: 200 }
  );
}
// GET
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  return handleQuery(
    searchParams.get("q")
  );
}

// POST
export async function POST(request: Request) {
  const body = await request
    .json()
    .catch(() => ({}));

  return handleQuery(
    body.query ?? body.q ?? null,
    body.history ?? []
  );
}