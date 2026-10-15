
const { z } = require('zod');

// --------------------------------------------------
// Input sanitizer
// --------------------------------------------------
function sanitizeString(value) {
  if (typeof value !== 'string') return value;

  let v = value.trim();

  // Remove prefixes such as "sku: SKU-001"
  const colonMatch = v.match(/^[a-zA-Z_]+\s*:\s*(.+)$/);
  if (colonMatch) v = colonMatch[1].trim();

  // Remove surrounding single or double quotes
  if (
    (v.startsWith('"') && v.endsWith('"')) ||
    (v.startsWith("'") && v.endsWith("'"))
  ) {
    v = v.slice(1, -1);
  }

  return v.trim();
}

// --------------------------------------------------
// Tool descriptions
// These help an LLM-powered MCP client understand
// when each tool should be selected.
// --------------------------------------------------

const getOrderDescription = `
Retrieve complete details for one specific order using its exact business order ID.

Use this tool when the user asks about a particular order's details, status,
customer name, total amount, creation date, purchased products, product SKUs,
item quantities, or item prices recorded for that order.

Examples:
- "Show order ORD-1001."
- "Where is my order ORD-1001?"
- "What is the status of ORD-1001?"
- "Who placed this order?"
- "What products are in ORD-1001?"
- "How much did this order cost?"

Extract the actual order ID from the user's message and pass it as order_id.
Do not invent an order ID. If the user wants multiple orders or orders filtered
by status, use list_orders instead.

This tool reads one order and its items. It does not create, update, cancel,
or provide live courier tracking for orders.
`.trim();

const listOrdersDescription = `
Retrieve a list of orders, optionally filtered by status and limited by result count.

Use this tool when the user asks to list, browse, find, or review multiple orders,
including all orders or orders with a specified status such as shipped,
processing, or delivered.

Examples:
- "Show all orders."
- "List shipped orders."
- "Which orders are processing?"
- "Show the latest 10 orders."
- "Find recently created orders."
- "List delivered orders with customer names and totals."

Pass status only when the user specifies a status filter.
Pass limit when the user requests a result count. The maximum limit is 100.
If omitted, the server uses its default limit.

This tool returns order summaries, not complete order items.
For details of one specific order, use get_order instead.
Do not claim that a limited result contains every order.

This tool is read-only and does not create, update, or cancel orders.
`.trim();

const checkStockDescription = `
Check current inventory information for one product using its exact SKU.

Use this tool when the user asks how much stock is available, the current
inventory quantity, or whether a product is active, and the exact SKU is known.

Examples:
- "Check stock for SKU-001."
- "How many units are available for SKU-001?"
- "Is SKU-001 active?"
- "Tell me the current inventory of SKU-001."

Pass the exact product SKU as sku.
If the user only provides a product name, partial name, or uncertain SKU,
use search_products first to identify the matching product.
If multiple products match, clarify which product the user means.

This tool only reads inventory. It does not change stock, reserve inventory,
or calculate stock availability after pending orders.
`.trim();

const searchProductsDescription = `
Search the product catalog by full or partial product name, exact SKU, or partial SKU.

Use this tool when the user refers to a product by name or SKU, does not know
the exact SKU, or asks for catalog information such as product price,
stock quantity, or active status.

Examples:
- "Find the USB-C Hub."
- "What is the price of the USB-C Hub?"
- "How much does a hub cost?"
- "Search for SKU-003."
- "Find products containing keyboard."
- "Show this product's price and stock."
- "I do not know the SKU; find the product for me."

Pass a concise product name, distinctive product words, or SKU as query.
Avoid passing the entire conversational question when a short search term
can be extracted. Use limit when the user requests a specific number of results.
The maximum limit is 50.

The tool can return multiple matches. Examine the returned product names
and SKUs before answering. If there is no reliable match, ask the user
to clarify instead of guessing.

Use this tool to find a product SKU before calling check_stock or update_stock
when the SKU is unknown.

This tool reads product information. It does not modify stock or prices.
`.trim();

const updateStockDescription = `
Change the inventory quantity for one product identified by its exact SKU.

Use this tool only when the user explicitly requests an inventory change,
such as adding stock, receiving new stock, removing units, or reducing inventory.

The quantity parameter is a signed integer representing the CHANGE,
not the desired final stock quantity:
- Positive quantity adds stock. Example: +10 adds 10 units.
- Negative quantity removes stock. Example: -5 removes 5 units.
- Zero is invalid.

Examples:
- "Add 10 units to SKU-001."
- "Increase SKU-001 stock by 20."
- "Remove 5 units from SKU-001."
- "Reduce inventory of SKU-001 by 3."
- "We received 15 new units of this product."

Extract the exact SKU and requested change.
If the product is known only by name, use search_products first.
If the intended product or quantity is ambiguous, ask for clarification
before making a change.

Never use this tool for requests that only ask to check, show, find,
or report stock. Those requests are read-only and should use check_stock
or search_products.

The server prevents stock from becoming negative. It returns an error
if the product does not exist or the requested removal exceeds current stock.
Only report success when the tool confirms the update.

This tool changes inventory in the database. It does not create orders,
reserve stock, change product prices, or create an audit entry.
`.trim();

// --------------------------------------------------
// Input schemas
// --------------------------------------------------

const getOrderInputSchema = {
  order_id: z.string()
    .min(1)
    .describe(
      'Exact business order ID extracted from the user request, e.g. ORD-1001. Required to retrieve one order.'
    ),
};

const listOrdersInputSchema = {
  status: z.string()
    .optional()
    .describe(
      'Optional exact order status filter, e.g. shipped, processing, or delivered. Omit when the user wants orders of any status.'
    ),

  limit: z.number()
    .int()
    .positive()
    .max(100)
    .optional()
    .describe(
      'Optional maximum number of order summaries to return. Must be an integer from 1 to 100. Default is 20.'
    ),
};

const checkStockInputSchema = {
  sku: z.string()
    .min(1)
    .describe(
      'Exact product SKU, e.g. SKU-001. Use search_products first if the user only provides a product name or an uncertain SKU.'
    ),
};

const searchProductsInputSchema = {
  query: z.string()
    .min(1)
    .describe(
      'Product full name, partial name, exact SKU, or partial SKU. Extract a concise search term from the user request, e.g. USB-C Hub, hub, or SKU-003.'
    ),

  limit: z.number()
    .int()
    .positive()
    .max(50)
    .optional()
    .describe(
      'Optional maximum number of matching products to return. Must be an integer from 1 to 50. Default is 10.'
    ),
};

const updateStockInputSchema = {
  sku: z.string()
    .min(1)
    .describe(
      'Exact SKU of the product whose inventory must change, e.g. SKU-001. Use search_products first if the SKU is unknown.'
    ),

  quantity: z.number()
    .int()
    .refine((value) => value !== 0, {
      message: 'quantity must not be 0',
    })
    .describe(
      'Signed integer stock change, not final stock: positive adds units, negative removes units, and zero is invalid. Only call when the user explicitly requests a stock change.'
    ),
};

// --------------------------------------------------
// Tool handlers
// --------------------------------------------------

function createTools(pool) {
  // Get one order and its line items
  async function getOrderHandler({ order_id }) {
    order_id = sanitizeString(order_id);

    const orderResult = await pool.query(
      'SELECT id, order_id, customer_name, status, total_amount, created_at FROM orders WHERE order_id = $1',
      [order_id]
    );

    if (orderResult.rows.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Order not found: ${order_id}`,
        }],
        isError: true,
      };
    }

    const order = orderResult.rows[0];

    const itemsResult = await pool.query(
      `SELECT p.sku, p.name, oi.quantity, oi.unit_price
       FROM order_items oi
       JOIN products p ON p.id = oi.product_id
       WHERE oi.order_id = $1`,
      [order.id]
    );

    const response = {
      order_id: order.order_id,
      customer_name: order.customer_name,
      status: order.status,
      total_amount: order.total_amount,
      created_at: order.created_at,
      items: itemsResult.rows,
    };

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(response, null, 2),
      }],
    };
  }

  // List orders, optionally filtering by status
  async function listOrdersHandler({ status, limit }) {
    const maxRows = limit || 20;
    const cleanStatus = status ? sanitizeString(status) : null;
    let result;

    if (cleanStatus) {
      result = await pool.query(
        'SELECT order_id, customer_name, status, total_amount, created_at FROM orders WHERE status = $1 ORDER BY created_at DESC LIMIT $2',
        [cleanStatus, maxRows]
      );
    } else {
      result = await pool.query(
        'SELECT order_id, customer_name, status, total_amount, created_at FROM orders ORDER BY created_at DESC LIMIT $1',
        [maxRows]
      );
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result.rows, null, 2),
      }],
    };
  }

  // Check stock for an exact SKU
  async function checkStockHandler({ sku }) {
    sku = sanitizeString(sku);

    const result = await pool.query(
      'SELECT sku, name, stock_quantity, is_active FROM products WHERE sku = $1',
      [sku]
    );

    if (result.rows.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Product not found: ${sku}`,
        }],
        isError: true,
      };
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result.rows[0], null, 2),
      }],
    };
  }

  // Search products by name or SKU
  async function searchProductsHandler({ query, limit }) {
    const term = sanitizeString(query);
    const maxRows = limit || 10;

    if (!term) {
      return {
        content: [{
          type: 'text',
          text: 'A product name or SKU is required.',
        }],
        isError: true,
      };
    }

    // Escape SQL LIKE wildcards in user input
    const escapeLike = (value) =>
      value.replace(/([\\%_])/g, '\\$1');

    const pattern = `%${escapeLike(term)}%`;

    let result = await pool.query(
      `SELECT sku, name, price, stock_quantity, is_active
       FROM products
       WHERE name ILIKE $1 OR sku ILIKE $1
       ORDER BY (LOWER(name) = LOWER($2)) DESC,
                (LOWER(sku) = LOWER($2)) DESC,
                name
       LIMIT $3`,
      [pattern, term, maxRows]
    );

    // Fallback: search individual words if the full phrase has no match
    if (result.rows.length === 0) {
      const words = term
        .split(/[^a-z0-9+#.-]+/i)
        .map((word) => word.replace(/^[.-]+|[.-]+$/g, ''))
        .filter((word) => word.length >= 3);

      if (words.length) {
        const wordPatterns = words.map(
          (word) => `%${escapeLike(word)}%`
        );

        result = await pool.query(
          `SELECT sku, name, price, stock_quantity, is_active
           FROM products
           WHERE name ILIKE ANY($1::text[])
              OR sku ILIKE ANY($1::text[])
           ORDER BY name
           LIMIT $2`,
          [wordPatterns, maxRows]
        );
      }
    }

    if (result.rows.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Product not found: ${term}`,
        }],
        isError: true,
      };
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result.rows, null, 2),
      }],
    };
  }

  // Add or remove stock using a signed quantity
  async function updateStockHandler({ sku, quantity }) {
    sku = sanitizeString(sku);

    const result = await pool.query(
      `UPDATE products
       SET stock_quantity = stock_quantity + $1,
           updated_at = now()
       WHERE sku = $2
         AND stock_quantity + $1 >= 0
       RETURNING sku, name, stock_quantity`,
      [quantity, sku]
    );

    if (result.rows.length === 0) {
      const exists = await pool.query(
        'SELECT stock_quantity FROM products WHERE sku = $1',
        [sku]
      );

      if (exists.rows.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `Product not found: ${sku}`,
          }],
          isError: true,
        };
      }

      return {
        content: [{
          type: 'text',
          text: `Insufficient stock for ${sku}. Current: ${exists.rows[0].stock_quantity}, requested change: ${quantity}`,
        }],
        isError: true,
      };
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result.rows[0], null, 2),
      }],
    };
  }

  // Expose descriptions, schemas, and handlers
  return {
    getOrderDescription,
    getOrderInputSchema,
    getOrderHandler,

    listOrdersDescription,
    listOrdersInputSchema,
    listOrdersHandler,

    checkStockDescription,
    checkStockInputSchema,
    checkStockHandler,

    searchProductsDescription,
    searchProductsInputSchema,
    searchProductsHandler,

    updateStockDescription,
    updateStockInputSchema,
    updateStockHandler,
  };
}

module.exports = {
  createTools,
  sanitizeString,

  getOrderDescription,
  listOrdersDescription,
  checkStockDescription,
  searchProductsDescription,
  updateStockDescription,
};
