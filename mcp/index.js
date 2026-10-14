// index.js
require('dotenv').config();
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const {
  getOrderInputSchema,
  getOrderHandler,
  listOrdersInputSchema,
  listOrdersHandler,
  checkStockInputSchema,
  checkStockHandler,
  updateStockInputSchema,
  updateStockHandler,
} = require('./tools');

const server = new McpServer({
  name: 'opsmate-order-inventory',
  version: '1.0.0',
});

server.tool(
  'get_order',
  'Get full details of an order (header + line items) by its business order_id',
  getOrderInputSchema,
  getOrderHandler
);

server.tool(
  'list_orders',
  'List orders, optionally filtered by status',
  listOrdersInputSchema,
  listOrdersHandler
);

server.tool(
  'check_stock',
  'Check current stock quantity for a product by SKU',
  checkStockInputSchema,
  checkStockHandler
);

server.tool(
  'update_stock',
  'Adjust a product stock by a signed delta (positive adds, negative removes)',
  updateStockInputSchema,
  updateStockHandler
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error('MCP server failed to start:', err);
  process.exit(1);
});