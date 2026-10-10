// frontend/app/api/search/route.ts
// Thin proxy to the Express backend (Ollama + PostgreSQL cache).
// All model access, caching and DB write-back live in the backend; this route
// only forwards the request so the browser can stay on a single origin.
//
// No Vercel AI SDK, no OpenAI / Anthropic / Gemini, no Supabase.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = (
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:5000"
).replace(/\/$/, "");

function backendUnreachable(detail: string) {
  return Response.json(
    {
      error: `OpsMate backend unreachable at ${BACKEND_URL}. Start it with "cd backend && npm run dev".`,
      title: "Backend unavailable",
      code: "BACKEND_UNAVAILABLE",
      degraded: true,
      detail,
    },
    { status: 503 }
  );
}

async function forward(query: string | null) {
  const trimmed = (query ?? "").trim();

  if (!trimmed) {
    return Response.json({ error: "Please type a question first." }, { status: 400 });
  }

  try {
    const res = await fetch(`${BACKEND_URL}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed }),
      cache: "no-store",
    });

    const data = await res.json().catch(() => null);

    if (!data) {
      return Response.json(
        {
          error: `Backend returned a non-JSON response (HTTP ${res.status}).`,
          code: "BACKEND_BAD_RESPONSE",
          degraded: true,
        },
        { status: 502 }
      );
    }

    const headers = new Headers();
    for (const header of ["x-cache", "x-model", "x-provider", "x-tool", "server-timing"]) {
      const value = res.headers.get(header);
      if (value) headers.set(header, value);
    }

    return Response.json(data, { status: res.status, headers });
  } catch (err) {
    return backendUnreachable(err instanceof Error ? err.message : String(err));
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  return forward(searchParams.get("q"));
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  return forward(body.query ?? body.q ?? null);
}