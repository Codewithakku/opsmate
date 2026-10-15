
require('dotenv').config({ path: require('path').join(__dirname, '.env') });

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const pool = require('./db');
const { createTools } = require('./tools');

// ======================================================
// STDIO-SAFE LOGGING
// STDOUT is reserved for MCP JSON-RPC messages.
// All diagnostic logs must go to STDERR.
// ======================================================

const logLine = (line) => process.stderr.write(`${line}\n`);
const newCallId = () => Math.random().toString(36).slice(2, 6);

const oneLine = (value, max = 100) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

const formatArgs = (args) =>
  Object.entries(args || {})
    .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
    .join(', ');

function summarizeResult(name, result) {
  const text = result?.content?.[0]?.text || '';

  if (result?.isError) {
    return oneLine(text) || 'error';
  }

  if (!text) {
    return 'ok';
  }

  try {
    const data = JSON.parse(text);

    if (Array.isArray(data)) {
      return `${data.length} row(s)`;
    }

    if (name === 'get_order') {
      return [
        data.order_id,
        data.status,
        Array.isArray(data.items) ? `${data.items.length} item(s)` : null,
      ]
        .filter(Boolean)
        .join(' · ');
    }

    if (data.sku) {
      return [
        data.sku,
        data.stock_quantity != null ? `stock ${data.stock_quantity}` : null,
      ]
        .filter(Boolean)
        .join(' · ');
    }

    return oneLine(text);
  } catch (_error) {
    return oneLine(text);
  }
}

function registerTracedTool(server, name, description, inputSchema, handler) {
  server.tool(name, description, inputSchema, async (args) => {
    const callId = newCallId();
    const startedAt = Date.now();
    const label = name.padEnd(16);

    logLine(`[${callId}] ▸ ${label} ${oneLine(formatArgs(args), 96)}`);

    try {
      const result = await handler(args || {});
      const ms = Date.now() - startedAt;
      const ok = result?.isError !== true;

      logLine(
        `[${callId}] ${ok ? '✓' : '✗'} ${label} ${ms}ms · ${summarizeResult(name, result)}`
      );

      return result;
    } catch (error) {
      logLine(
        `[${callId}] ✗ ${label} ${Date.now() - startedAt}ms · failed: ${oneLine(error?.message || error)}`
      );
      throw error;
    }
  });
}

// ======================================================
// CREATE TOOLS
// ======================================================

const tools = createTools(pool);

const server = new McpServer({
  name: 'opsmate-order-inventory',
  version: '1.0.0',
});

registerTracedTool(
  server,
  'get_order',
  tools.getOrderDescription,
  tools.getOrderInputSchema,
  tools.getOrderHandler
);

registerTracedTool(
  server,
  'list_orders',
  tools.listOrdersDescription,
  tools.listOrdersInputSchema,
  tools.listOrdersHandler
);

registerTracedTool(
  server,
  'check_stock',
  tools.checkStockDescription,
  tools.checkStockInputSchema,
  tools.checkStockHandler
);

registerTracedTool(
  server,
  'search_products',
  tools.searchProductsDescription,
  tools.searchProductsInputSchema,
  tools.searchProductsHandler
);

registerTracedTool(
  server,
  'update_stock',
  tools.updateStockDescription,
  tools.updateStockInputSchema,
  tools.updateStockHandler
);

// ======================================================
// START MCP SERVER
// ======================================================

async function main() {
  // Validate that the configured database is reachable before connecting MCP.
  await pool.query('SELECT 1');

  logLine('[mcp] PostgreSQL connection successful');

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logLine(
    '[mcp] OpsMate order/inventory server ready (stdio) — get_order, list_orders, check_stock, search_products, update_stock'
  );
}

main().catch((error) => {
  logLine(
    `[mcp] MCP server failed to start: ${oneLine(error?.stack || error, 500)}`
  );
  process.exitCode = 1;
});
