// backend/test-ollama-embedding.js
require("dotenv").config();

const startTime = Date.now();

async function testEmbedding() {
  console.log("\n========== OLLAMA EMBEDDING TEST ==========\n");

  try {
    const response = await fetch(
      "http://localhost:11434/api/embed",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "nomic-embed-text:latest",
          input: "What is the order status of order ID 1001?"
        }),
        signal: AbortSignal.timeout(60000)
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(JSON.stringify(data));
    }

    const embedding = data.embeddings?.[0];

    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error("Embedding vector missing in response");
    }

    if (!embedding.every(Number.isFinite)) {
      throw new Error("Embedding contains invalid numeric values");
    }

    console.log("Status: SUCCESS");
    console.log("Model:", "nomic-embed-text:latest");
    console.log("Vector dimensions:", embedding.length);
    console.log("Valid numeric values:", true);
    console.log("First 5 values:", embedding.slice(0, 5));
    console.log("Latency:", Date.now() - startTime, "ms");

  } catch (error) {
    console.error("Status: FAILED");
    console.error("Reason:", error.message);
    console.log("Latency:", Date.now() - startTime, "ms");
    process.exitCode = 1;
  }

  console.log("\n===========================================\n");
}

testEmbedding();

