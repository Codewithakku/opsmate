// tools.js
const { z } = require('zod');
const pool = require('./db');

// Robust input sanitizer — "sku: \"SKU-001\"" jevi values ne pan clean kari aape
function sanitizeString(value) {
  if (typeof value !== 'string') return value;
  let v = value.trim();

  // "key: value" jevo prefix hoy to value j kadhi lo
  const colonMatch = v.match(/^[a-zA-Z_]+\s*:\s*(.+)$/);
  if (colonMatch) v = colonMatch[1].trim();

  // surrounding quotes (single ke double) hoy to kadhi lo
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
    v = v.slice(1, -1);
  }

  return v.trim();
}

// ---- get_order ----
const getOrderInputSchema = {
  order_id: z.string().describe('Business order_id, e.g. ORD-1001'),
};

async function getOrderHandler({ order_id }) {
  order_id = sanitizeString(order_id);

  const orderResult = await pool.query(
    'SELECT id, order_id, customer_name, status, total_amount, created_at FROM orders WHERE order_id = $1',
    [order_id]
  );

  if (orderResult.rows.length === 0) {
    return {
      content: [{ type: 'text', text: `Order not found: ${order_id}` }],
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
    content: [{ type: 'text', text: JSON.stringify(response, null, 2) }],
  };
}

// ---- list_orders ----
const listOrdersInputSchema = {
  status: z.string().optional().describe('Filter by status, e.g. shipped, processing, delivered'),
  limit: z.number().int().positive().max(100).optional().describe('Max rows to return (default 20)'),
};

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
    content: [{ type: 'text', text: JSON.stringify(result.rows, null, 2) }],
  };
}

// ---- check_stock ----
const checkStockInputSchema = {
  sku: z.string().describe('Product SKU, e.g. SKU-001'),
};

async function checkStockHandler({ sku }) {
  sku = sanitizeString(sku);

  const result = await pool.query(
    'SELECT sku, name, stock_quantity, is_active FROM products WHERE sku = $1',
    [sku]
  );

  if (result.rows.length === 0) {
    return {
      content: [{ type: 'text', text: `Product not found: ${sku}` }],
      isError: true,
    };
  }

  return {
    content: [{ type: 'text', text: JSON.stringify(result.rows[0], null, 2) }],
  };
}

// ---- update_stock ----
const updateStockInputSchema = {
  sku: z.string().describe('Product SKU, e.g. SKU-001'),
  quantity: z.number().int().refine((val) => val !== 0, {
    message: 'quantity must not be 0',
  }).describe('Signed delta: positive adds stock, negative removes stock'),
};

async function updateStockHandler({ sku, quantity }) {
  sku = sanitizeString(sku);

  const result = await pool.query(
    `UPDATE products
     SET stock_quantity = stock_quantity + $1, updated_at = now()
     WHERE sku = $2 AND stock_quantity + $1 >= 0
     RETURNING sku, name, stock_quantity`,
    [quantity, sku]
  );

  if (result.rows.length === 0) {
    const exists = await pool.query('SELECT stock_quantity FROM products WHERE sku = $1', [sku]);

    if (exists.rows.length === 0) {
      return {
        content: [{ type: 'text', text: `Product not found: ${sku}` }],
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
    content: [{ type: 'text', text: JSON.stringify(result.rows[0], null, 2) }],
  };
}

module.exports = {
  getOrderInputSchema,
  getOrderHandler,
  listOrdersInputSchema,
  listOrdersHandler,
  checkStockInputSchema,
  checkStockHandler,
  updateStockInputSchema,
  updateStockHandler,
};