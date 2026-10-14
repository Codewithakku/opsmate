# OpsMate — Order & Inventory MCP Server

A standalone [MCP](https://modelcontextprotocol.io) server exposing 4 tools —
`get_order`, `list_orders`, `check_stock`, `update_stock` — over the existing
OpsMate PostgreSQL database. Runs as a separate process, connectable from
Claude Desktop, MCP Inspector, or any MCP-compatible client.

Repo: https://github.com/aakash-raval43x/opsmate
Branches: `main`, `master`, `ollama` (active), `backup-before-author-fix`

---

## Project Structure

```
opsmate/
├─ backend/                  # Node.js/Express chat backend (Ollama + Postgres cache)
│  ├─ Dockerfile
│  ├─ package.json
│  ├─ schema.js
│  ├─ server.js
│  └─ test-embedding.js
│
├─ frontend/                 # Next.js frontend
│  ├─ app/
│  ├─ public/
│  ├─ Dockerfile
│  ├─ next.config.ts
│  ├─ package.json
│  └─ README.md
│
├─ mcp/                      # ← this MCP server (Order & Inventory tools)
│  ├─ db.js                  # Postgres connection pool
│  ├─ index.js                # MCP server entry point (stdio transport, tool registration)
│  ├─ tools.js                # All 4 tool handlers + input sanitizer
│  ├─ package.json
│  └─ .env                   # DATABASE_URL (not committed)
│
├─ .env
├─ .gitignore
├─ Caddyfile
├─ docker-compose.yml
├─ README.md                  # (this repo-level file)
└─ README.local.md
```

*(`node_modules`, `.freebuff`, `.next`, lockfiles, and build artifacts omitted
for clarity — see `.gitignore`.)*

---

## `mcp/` — Setup & Run

### 1. Install dependencies
```
cd mcp
npm install
```

### 2. Configure environment
Create `mcp\.env`:
```
DATABASE_URL=postgresql://opsmate:opsmate_password@127.0.0.1:5432/opsmate
```

### 3. Run the server directly
```
node index.js
```

### 4. Run with MCP Inspector (for testing/dev)
```
npx @modelcontextprotocol/inspector node index.js
```
This opens a local browser UI where each tool can be called interactively.

### 5. Connect from an MCP client (e.g. Claude Desktop)
Add to the client's MCP config:
```jsonc
{
  "mcpServers": {
    "opsmate-order-inventory": {
      "type": "stdio",
      "command": "node",
      "args": ["index.js"],
      "cwd": "D:\\Raval Aakash\\intership-task\\opsmate\\mcp"
    }
  }
}
```

> All commands above are Windows (cmd/PowerShell) commands.

---

## Database Schema Used

| Table | Purpose |
|---|---|
| `orders` | Order header — `id`, `order_id`, `customer_name`, `status`, `total_amount`, `created_at` |
| `products` | Product catalog — `id`, `sku`, `name`, `description`, `price`, `stock_quantity`, `is_active` |
| `order_items` | Line items per order — `order_id` (FK), `product_id` (FK), `quantity`, `unit_price` |

`queries` and `query_cache` (used by the chat `backend/`) are untouched by this
MCP server.

---

## Tools

| Tool | Input | Description |
|---|---|---|
| `get_order` | `order_id: string` | Full order detail: header + line items (SKU, product name, qty, unit price) |
| `list_orders` | `status?: string`, `limit?: number` | List orders, optionally filtered by status (default limit 20) |
| `check_stock` | `sku: string` | Current stock quantity and active status for a product |
| `update_stock` | `sku: string`, `quantity: number` | Adjust stock by a signed delta (positive = add, negative = remove) |

All string inputs pass through a `sanitizeString` helper (`tools.js`) that
strips stray `key:` prefixes and surrounding quotes, so inputs like
`sku: "SKU-001"` resolve to `SKU-001`.

### Example requests & responses

**`get_order`**
```json
// input
{ "order_id": "ORD-1001" }

// output
{
  "order_id": "ORD-1001",
  "customer_name": "Aakash",
  "status": "shipped",
  "total_amount": "1499.00",
  "created_at": "2026-09-29T01:51:52.903Z",
  "items": [
    { "sku": "SKU-001", "name": "Wireless Mouse", "quantity": 2, "unit_price": "499.00" },
    { "sku": "SKU-002", "name": "Mechanical Keyboard", "quantity": 1, "unit_price": "2499.00" }
  ]
}
```

**`list_orders`**
```json
// input
{ "status": "shipped" }

// output
[
  {
    "order_id": "ORD-1001",
    "customer_name": "Aakash",
    "status": "shipped",
    "total_amount": "1499.00",
    "created_at": "2026-09-29T01:51:52.903Z"
  }
]
```

**`check_stock`**
```json
// input
{ "sku": "SKU-001" }

// output
{ "sku": "SKU-001", "name": "Wireless Mouse", "stock_quantity": 55, "is_active": true }
```

**`update_stock`**
```json
// input
{ "sku": "SKU-001", "quantity": -5 }

// output
{ "sku": "SKU-001", "name": "Wireless Mouse", "stock_quantity": 45 }
```

Error cases:
```json
// invalid sku
{ "error": "Product not found: SKU-999" }

// insufficient stock
{ "error": "Insufficient stock for SKU-001. Current: 45, requested change: -1000" }
```

---

## Scope Notes

- **No authentication/authorization** in this phase — the goal was to verify
  the core concept (schema + 4 tools) works end-to-end. Auth can be layered
  on later if needed.
- **No inventory audit/log table** — `update_stock` adjusts `stock_quantity`
  directly via a single atomic `UPDATE`; only the latest value is tracked,
  no change history.
- **Order creation / stock reservation** are out of scope for these 4 tools.

## Status

✅ Schema created · ✅ All 4 tools implemented · ✅ All 4 tools tested via MCP
Inspector · ✅ Input sanitization added and retested.
