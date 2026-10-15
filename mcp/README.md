# OpsMate — Order & Inventory MCP Server

A standalone [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes five order and inventory tools over the existing OpsMate PostgreSQL database.

It runs as a separate process and can connect to Claude Desktop, MCP Inspector, and other MCP-compatible clients that support stdio transport.

**Repository:** https://github.com/aakash-raval43x/opsmate

## Features

- Retrieve complete order details.
- List orders with optional status filtering.
- Check current product stock.
- Search products by name or SKU.
- Increase or decrease product stock.
- PostgreSQL integration using a connection pool.
- Input validation and string sanitization.
- Stdio transport for MCP client communication.

## Project Structure

```text
opsmate/
├── backend/                   # Express chat backend
├── frontend/                  # Next.js frontend
├── mcp/
│   ├── .env                   # Database configuration (not committed)
│   ├── index.js               # MCP entry point and tool registration
│   ├── db.js                  # PostgreSQL connection pool
│   ├── tools.js               # Five MCP tool handlers
│   ├── advanced-test.js       # MCP tool tests
│   ├── package.json           # MCP dependencies
│   ├── package-lock.json
│   └── README.md              # MCP server documentation
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Requirements

- Node.js and npm
- PostgreSQL
- The OpsMate database and required tables
- Claude Desktop or MCP Inspector for testing

## Installation and Configuration

### 1. Open the MCP directory

Run these commands in PowerShell:

```powershell
cd "D:\Raval Aakash\intership-task\opsmate\mcp"
npm install
```

### 2. Configure the database

Create a `.env` file inside the `mcp/` directory:

```env
DATABASE_URL=postgresql://opsmate:opsmate_password@127.0.0.1:5432/opsmate
```

Replace the connection values with the configuration for your PostgreSQL database if they differ.

**Important:** Keep `.env` private. Never commit database passwords or other secrets to GitHub.

### 3. Start the MCP server

```powershell
node index.js
```

The server initializes its PostgreSQL connection and starts using stdio transport.

When running through Claude Desktop or MCP Inspector, the client launches the MCP process automatically. You do not need to start a separate HTTP server for this MCP connection.

## MCP Tools

| Tool | Inputs | Description |
|---|---|---|
| `get_order` | `order_id: string` | Retrieves an order, customer details, status, total, and line items. |
| `list_orders` | `status?: string`, `limit?: number` | Lists orders, optionally filtered by status. |
| `check_stock` | `sku: string` | Retrieves current stock and product active status. |
| `search_products` | `query: string`, `limit?: number` | Searches products by full or partial name, or SKU. |
| `update_stock` | `sku: string`, `quantity: number` | Adjusts stock using a signed quantity: positive adds stock, negative removes stock. |

String inputs are sanitized by the helper in `tools.js` to handle common formatting issues, such as surrounding quotes and stray key prefixes.

## Testing with MCP Inspector

MCP Inspector provides a browser interface for connecting to the MCP server and testing its tools.

From the `mcp/` directory, run:

```powershell
npx @modelcontextprotocol/inspector node index.js
```

Use the Inspector interface to connect to the server and invoke each tool with its required inputs.

### Example tool inputs

**Get an order**

```json
{
  "order_id": "ORD-1001"
}
```

**List orders**

```json
{
  "limit": 20
}
```

**Check stock**

```json
{
  "sku": "SKU-001"
}
```

**Search products**

```json
{
  "query": "Wireless Mouse"
}
```

**Increase stock by one unit**

```json
{
  "sku": "SKU-001",
  "quantity": 1
}
```

**Important:** `update_stock` changes the database. Use it only when you intend to change inventory.

### Run automated tests

```powershell
node --test advanced-test.js
```

Run this command from the `mcp/` directory. Test results depend on the current test file, dependencies, and database configuration.

## Claude Desktop Configuration

The MCP server has been connected and tested with Claude Desktop.

Open the Claude Desktop configuration file:

```text
C:\Users\alian\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json
```

Add the following entry inside the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "opsmate-orders": {
      "command": "node",
      "args": [
        "D:\\Raval Aakash\\intership-task\\opsmate\\mcp\\index.js"
      ]
    }
  }
}
```

**Configuration notes:**

- Merge this entry into the existing configuration; do not delete unrelated settings.
- Use the actual absolute path to `mcp/index.js` on your computer.
- Restart Claude Desktop after saving the configuration.
- Confirm that the server connects, then test its tools in a conversation.

## Database Schema

The MCP server uses these tables in the OpsMate PostgreSQL database:

| Table | Purpose |
|---|---|
| `orders` | Order details, customer, status, total amount, and creation date. |
| `products` | Product SKU, name, description, price, stock, and active status. |
| `order_items` | Products and quantities associated with each order. |

The chat backend's `queries` and `query_cache` tables are separate from the MCP tools' main order and inventory operations.

## Example Results

The examples below illustrate the tool response fields. Stock quantities can change after an inventory update.

**`get_order`**

```json
{
  "order_id": "ORD-1001",
  "customer_name": "Aakash",
  "status": "shipped",
  "total_amount": "1499.00",
  "items": [
    {
      "sku": "SKU-001",
      "name": "Wireless Mouse",
      "quantity": 2,
      "unit_price": "499.00"
    },
    {
      "sku": "SKU-002",
      "name": "Mechanical Keyboard",
      "quantity": 1,
      "unit_price": "2499.00"
    }
  ]
}
```

**`check_stock`**

The last verified result for `SKU-001` was:

```json
{
  "sku": "SKU-001",
  "name": "Wireless Mouse",
  "stock_quantity": 56,
  "is_active": true
}
```

**`search_products`**

```json
[
  {
    "sku": "SKU-001",
    "name": "Wireless Mouse",
    "price": "499.00",
    "stock_quantity": 56,
    "is_active": true
  }
]
```

**`update_stock`**

The tool accepts a signed quantity. For example, adding one unit to a stock of 55 should result in 56 units if the update succeeds.

```json
{
  "sku": "SKU-001",
  "quantity": 1
}
```

Always check the returned result or call `check_stock` again to verify the actual database value.

**Data consistency note:** The sample `ORD-1001` has a recorded total of ₹1,499, but its listed line items add up to ₹3,497. This is an inconsistency in the sample order data that should be corrected separately; it does not by itself indicate an MCP connection failure.

## Logging and Troubleshooting

The server uses stderr for diagnostic logging. With stdio transport, stdout is reserved for MCP protocol messages and must not contain ordinary debug logs.

If the server fails to connect:

1. Verify that PostgreSQL is running and the database is reachable.
2. Check the `DATABASE_URL` in `mcp/.env`.
3. Confirm that `index.js` and `db.js` load the correct `.env` file.
4. Confirm that Claude Desktop points to `mcp/index.js`, not the Express backend's `server.js`.
5. Restart Claude Desktop after editing its configuration.
6. Run the MCP Inspector to isolate connection or tool errors.

## Scope and Limitations

- **Authentication and authorization:** Not implemented in this phase.
- **Inventory audit history:** Not implemented. Stock is updated directly; no stock-change history is maintained.
- **Order creation and stock reservation:** Outside the scope of the current tools.
- **Transport:** The current server uses stdio; it is not configured as a Streamable HTTP server.

## Current Status

- [x] PostgreSQL connection configured
- [x] Five MCP tools implemented
- [x] MCP Inspector connection and tool testing completed
- [x] Claude Desktop connection verified
- [x] `get_order`, `list_orders`, `check_stock`, and `search_products` tested
- [x] `update_stock` tested and the updated stock verified
- [x] Input sanitization implemented

**Project goal:** Expose existing OpsMate order and inventory operations as reusable MCP tools that compatible AI clients can discover and invoke.
