// frontend/app/api/assist/route.ts

// Streaming proxy: Next.js -> Express -> Ollama/PostgreSQL.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = (
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:5000"
).replace(/\/+$/, "");

const STREAM_ERROR_PREFIX = "__OPSMATE_STREAM_ERROR__:";

const textHeaders = (extra: Record<string, string> = {}) => ({
  "Content-Type": "text/plain; charset=utf-8",
  "Cache-Control": "no-cache, no-transform",
  "X-Accel-Buffering": "no",
  ...extra,
});

function errorStream(
  message: string,
  extra: Record<string, string> = {}
): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(`${STREAM_ERROR_PREFIX}${message}`)
      );
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: textHeaders(extra),
  });
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "CanceledError")
  );
}

async function handleAssist(
  request: Request,
  query: string,
  history: unknown
): Promise<Response> {
  const trimmed = (query ?? "").trim();

  if (!trimmed) {
    return new Response("Please type a question first.", {
      status: 400,
      headers: textHeaders(),
    });
  }

  // One controller owns the upstream Express request.
  const upstreamController = new AbortController();

  const abortUpstream = () => {
    if (!upstreamController.signal.aborted) {
      upstreamController.abort(request.signal.reason);
    }
  };

  // Propagate incoming request cancellation to Express. This single
  // listener is enough: once upstreamController aborts, the fetch's
  // response body read in the pull() loop below throws on its own,
  // so there is no need for a second "abort" listener that duplicates
  // this logic (the previous version registered both abortUpstream
  // and a near-identical onRequestAbort for the same event).
  if (request.signal.aborted) {
    abortUpstream();
    return new Response(null, { status: 499 });
  }

  request.signal.addEventListener("abort", abortUpstream, {
    once: true,
  });

  let backend: Response;

  try {
    backend = await fetch(`${BACKEND_URL}/api/assist`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/plain",
      },
      body: JSON.stringify({
        query: trimmed,
        history: Array.isArray(history) ? history : [],
      }),
      signal: upstreamController.signal,
      cache: "no-store",
    });
  } catch (error) {
    request.signal.removeEventListener("abort", abortUpstream);

    if (upstreamController.signal.aborted || isAbortError(error)) {
      console.log("[assist] upstream request cancelled");

      return new Response(null, { status: 499 });
    }

    const detail =
      error instanceof Error ? error.message : String(error);

    console.error("[assist] backend unreachable:", detail);

    return errorStream(
      `OpsMate backend is not reachable at ${BACKEND_URL}. Start the backend and try again.`,
      {
        "X-Cache": "ERROR",
        "X-Provider": "none",
      }
    );
  }

  if (!backend.ok || !backend.body) {
    const detail = await backend.text().catch(() => "");

    request.signal.removeEventListener("abort", abortUpstream);

    console.error(
      "[assist] backend error:",
      backend.status,
      detail.slice(0, 200)
    );

    return errorStream(
      detail || `The backend failed to answer (HTTP ${backend.status}).`,
      {
        "X-Cache": "ERROR",
      }
    );
  }

  // Forward response metadata to the UI.
  const headers = new Headers(
    textHeaders({
      "X-Provider": backend.headers.get("x-provider") || "ollama",
      "X-Model": backend.headers.get("x-model") || "",
      "X-Cache": backend.headers.get("x-cache") || "MISS",
    })
  );

  for (const header of ["x-tool", "server-timing"]) {
    const value = backend.headers.get(header);

    if (value) {
      headers.set(header, value);
    }
  }

  // Wrap the backend stream so cancelling the frontend reader also
  // cancels the upstream request and releases the reader lock.
  const backendReader = backend.body.getReader();

  let streamFinished = false;

  const outputStream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        if (upstreamController.signal.aborted) {
          streamFinished = true;
          controller.close();
          return;
        }

        const { done, value } = await backendReader.read();

        if (upstreamController.signal.aborted) {
          streamFinished = true;
          controller.close();
          return;
        }

        if (done) {
          streamFinished = true;
          request.signal.removeEventListener("abort", abortUpstream);

          try {
            backendReader.releaseLock();
          } catch {
            // Reader may already be released.
          }

          controller.close();
          return;
        }

        if (value) {
          controller.enqueue(value);
        }
      } catch (error) {
        streamFinished = true;
        request.signal.removeEventListener("abort", abortUpstream);

        try {
          controller.error(error);
        } catch {
          // Stream may already be closed or cancelled.
        }
      }
    },

    async cancel(reason) {
      streamFinished = true;

      request.signal.removeEventListener("abort", abortUpstream);

      if (!upstreamController.signal.aborted) {
        upstreamController.abort(reason);
      }

      try {
        await backendReader.cancel(reason);
      } catch {
        // Upstream may already have stopped.
      }

      try {
        backendReader.releaseLock();
      } catch {
        // Reader may still be finalizing.
      }
    },
  });

  return new Response(outputStream, {
    status: 200,
    headers,
  });
}

export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url);

  if (url.searchParams.has("stats")) {
    try {
      const controller = new AbortController();

      if (request.signal.aborted) {
        return Response.json(
          { error: "Request cancelled" },
          { status: 499 }
        );
      }

      const onAbort = () => controller.abort();

      request.signal.addEventListener("abort", onAbort, {
        once: true,
      });

      try {
        const response = await fetch(
          `${BACKEND_URL}/api/cache/stats`,
          {
            cache: "no-store",
            signal: controller.signal,
          }
        );

        return Response.json(await response.json(), {
          status: response.status,
        });
      } finally {
        request.signal.removeEventListener("abort", onAbort);
      }
    } catch (error) {
      if (request.signal.aborted || isAbortError(error)) {
        return new Response(null, { status: 499 });
      }

      return Response.json(
        {
          error: `OpsMate backend unreachable at ${BACKEND_URL}.`,
          detail: error instanceof Error ? error.message : String(error),
        },
        { status: 503 }
      );
    }
  }

  return handleAssist(request, url.searchParams.get("q") ?? "", []);
}

export async function POST(request: Request): Promise<Response> {
  let body: {
    query?: unknown;
    q?: unknown;
    history?: unknown;
  };

  try {
    body = await request.json();
  } catch {
    return new Response("Invalid JSON request body.", {
      status: 400,
      headers: textHeaders(),
    });
  }

  const query =
    typeof body.query === "string"
      ? body.query
      : typeof body.q === "string"
        ? body.q
        : "";

  return handleAssist(request, query, body.history);
}