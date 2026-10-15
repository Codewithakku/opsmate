
const { test } = require('node:test');
const assert = require('node:assert/strict');
require('dotenv').config();

const pool = require('./db');
const { createTools } = require('./tools');

const {
  getOrderInputSchema,
  getOrderHandler,
  listOrdersInputSchema,
  listOrdersHandler,
  checkStockInputSchema,
  checkStockHandler,
  searchProductsInputSchema,
  searchProductsHandler,
  updateStockInputSchema,
  updateStockHandler,
} = createTools(pool);

const TEST_ORDER_ID = process.env.TEST_ORDER_ID || 'ORD-1001';
const TEST_SKU = process.env.TEST_SKU || 'SKU-001';

// Stock-changing tests run only with explicit opt-in and a dedicated test SKU.
const MUTATION_TESTS_ENABLED =
  process.env.RUN_MUTATION_TESTS === 'true' &&
  Boolean(process.env.TEST_SKU);

function getText(result) {
  assert.ok(result?.content?.length, 'Expected MCP content in result');
  return result.content[0].text;
}

function getJson(result) {
  return JSON.parse(getText(result));
}

async function readStock(sku) {
  const result = await checkStockHandler({ sku });
  assert.equal(result.isError, undefined, getText(result));
  return getJson(result).stock_quantity;
}

async function withStockRestored(sku, callback) {
  const initialStock = await readStock(sku);

  try {
    return await callback(initialStock);
  } finally {
    const currentStock = await readStock(sku);
    const difference = initialStock - currentStock;

    if (difference !== 0) {
      const restoreResult = await updateStockHandler({
        sku,
        quantity: difference,
      });

      assert.equal(
        restoreResult.isError,
        undefined,
        `Could not restore test stock: ${getText(restoreResult)}`
      );
    }

    assert.equal(await readStock(sku), initialStock, 'Stock was not restored');
  }
}

// 1. Database connection
test('Database connection works', async () => {
  const result = await pool.query('SELECT 1 AS connected');
  assert.equal(Number(result.rows[0].connected), 1);
});

// 2. get_order
test('get_order returns the expected order', async () => {
  const result = await getOrderHandler({ order_id: TEST_ORDER_ID });

  assert.equal(result.isError, undefined, getText(result));

  const order = getJson(result);
  assert.equal(order.order_id, TEST_ORDER_ID);
  assert.ok(Array.isArray(order.items));
});

test('get_order sanitizes prefixed and quoted input', async () => {
  const result = await getOrderHandler({
    order_id: `order_id: "${TEST_ORDER_ID}"`,
  });

  assert.equal(result.isError, undefined, getText(result));
  assert.equal(getJson(result).order_id, TEST_ORDER_ID);
});

test('get_order handles a missing order', async () => {
  const result = await getOrderHandler({
    order_id: 'TEST-ORDER-DOES-NOT-EXIST-999999',
  });

  assert.equal(result.isError, true);
  assert.match(getText(result), /Order not found/i);
});

// 3. list_orders
test('list_orders returns an array', async () => {
  const result = await listOrdersHandler({});
  const orders = getJson(result);

  assert.ok(Array.isArray(orders));
});

test('list_orders respects the limit', async () => {
  const result = await listOrdersHandler({ limit: 1 });
  assert.ok(getJson(result).length <= 1);
});

test('list_orders filters by status', async () => {
  const status = 'SHIPPED';
  const result = await listOrdersHandler({ status, limit: 100 });
  const orders = getJson(result);

  assert.ok(orders.every((order) => order.status === status));
});

test('list_orders schema rejects invalid limits', () => {
  assert.equal(listOrdersInputSchema.limit.safeParse(0).success, false);
  assert.equal(listOrdersInputSchema.limit.safeParse(-1).success, false);
  assert.equal(listOrdersInputSchema.limit.safeParse(101).success, false);
  assert.equal(listOrdersInputSchema.limit.safeParse(1.5).success, false);
  assert.equal(listOrdersInputSchema.limit.safeParse(10).success, true);
});

// 4. check_stock
test('check_stock returns a valid product', async () => {
  const result = await checkStockHandler({ sku: TEST_SKU });

  assert.equal(result.isError, undefined, getText(result));

  const product = getJson(result);
  assert.equal(product.sku, TEST_SKU);
  assert.ok(Number.isInteger(Number(product.stock_quantity)));
});

test('check_stock sanitizes prefixed and quoted input', async () => {
  const result = await checkStockHandler({
    sku: `sku: "${TEST_SKU}"`,
  });

  assert.equal(result.isError, undefined, getText(result));
  assert.equal(getJson(result).sku, TEST_SKU);
});

test('check_stock handles an unknown SKU', async () => {
  const result = await checkStockHandler({
    sku: 'TEST-SKU-DOES-NOT-EXIST-999999',
  });

  assert.equal(result.isError, true);
  assert.match(getText(result), /Product not found/i);
});

test('check_stock safely handles SQL-injection-like input', async () => {
  const result = await checkStockHandler({
    sku: "' OR '1'='1",
  });

  assert.equal(result.isError, true);
  assert.match(getText(result), /Product not found/i);
});

// 5. search_products
//
// These are the questions the other four tools could not answer: every one of
// them is keyed by an order id or an EXACT sku, so a product named in plain
// words ("USB-C Hub") had no lookup path at all.
test('search_products finds a product by name, ignoring case', async () => {
  const skuResult = await checkStockHandler({ sku: TEST_SKU });
  const { name } = getJson(skuResult);

  const result = await searchProductsHandler({ query: name.toUpperCase() });
  assert.equal(result.isError, undefined, getText(result));

  const matches = getJson(result);
  assert.ok(Array.isArray(matches));
  assert.ok(
    matches.some((product) => product.sku === TEST_SKU),
    `expected ${TEST_SKU} among ${JSON.stringify(matches)}`
  );
  assert.ok(Number.isFinite(Number(matches[0].price)), 'price must be returned');
});

test('search_products finds a product by partial name', async () => {
  const skuResult = await checkStockHandler({ sku: TEST_SKU });
  const name = getJson(skuResult).name;
  const fragment = name.slice(0, Math.max(3, Math.floor(name.length / 2))).toLowerCase();

  const result = await searchProductsHandler({ query: fragment });
  assert.equal(result.isError, undefined, getText(result));
  assert.ok(getJson(result).some((product) => product.sku === TEST_SKU));
});

test('search_products also matches by SKU and returns price + stock', async () => {
  const result = await searchProductsHandler({ query: TEST_SKU });
  assert.equal(result.isError, undefined, getText(result));

  const [product] = getJson(result);
  assert.equal(product.sku, TEST_SKU);
  assert.ok(Number.isFinite(Number(product.price)));
  assert.ok(Number.isInteger(Number(product.stock_quantity)));
});

test('search_products sanitizes prefixed and quoted input', async () => {
  const result = await searchProductsHandler({ query: `name: "${TEST_SKU}"` });
  assert.equal(result.isError, undefined, getText(result));
  assert.equal(getJson(result)[0].sku, TEST_SKU);
});

test('search_products falls back to word matching when the whole phrase has no match', async () => {
  const skuResult = await checkStockHandler({ sku: TEST_SKU });
  const { name } = getJson(skuResult);

  // Exactly how a model echoes the user's question instead of the bare name
  // ("tell me prize of product name USB-C Hub") — the full phrase matches
  // nothing, and without the fallback the caller is told "not found" for a
  // product that exists.
  const result = await searchProductsHandler({ query: `product name ${name}` });

  assert.equal(result.isError, undefined, getText(result));
  assert.ok(
    getJson(result).some((product) => product.sku === TEST_SKU),
    `expected ${TEST_SKU} among ${JSON.stringify(getJson(result))}`
  );
});

test('search_products handles an unknown product name', async () => {
  const result = await searchProductsHandler({
    query: 'TEST-PRODUCT-DOES-NOT-EXIST-999999',
  });

  assert.equal(result.isError, true);
  assert.match(getText(result), /Product not found/i);
});

test('search_products escapes LIKE wildcards instead of matching everything', async () => {
  const result = await searchProductsHandler({ query: '%' });

  // An unescaped "%" would match every product; a literal one matches none of
  // the seeded names.
  assert.equal(result.isError, true);
  assert.match(getText(result), /Product not found/i);
});

test('search_products schema requires a query and bounds the limit', () => {
  assert.equal(searchProductsInputSchema.query.safeParse('USB-C Hub').success, true);
  assert.equal(searchProductsInputSchema.query.safeParse(5).success, false);
  assert.equal(searchProductsInputSchema.limit.safeParse(0).success, false);
  assert.equal(searchProductsInputSchema.limit.safeParse(51).success, false);
  assert.equal(searchProductsInputSchema.limit.safeParse(10).success, true);
});

// 6. Input schemas
test('get_order schema validates order_id type', () => {
  assert.equal(getOrderInputSchema.order_id.safeParse('ORD-1001').success, true);
  assert.equal(getOrderInputSchema.order_id.safeParse(1001).success, false);
});

test('check_stock schema validates SKU type', () => {
  assert.equal(checkStockInputSchema.sku.safeParse('SKU-001').success, true);
  assert.equal(checkStockInputSchema.sku.safeParse(123).success, false);
});

test('update_stock schema rejects zero and non-integer quantities', () => {
  assert.equal(updateStockInputSchema.quantity.safeParse(0).success, false);
  assert.equal(updateStockInputSchema.quantity.safeParse(1.5).success, false);
  assert.equal(updateStockInputSchema.quantity.safeParse('5').success, false);
  assert.equal(updateStockInputSchema.quantity.safeParse(5).success, true);
  assert.equal(updateStockInputSchema.quantity.safeParse(-5).success, true);
});

// 7. Database error propagation
test('handlers propagate database errors', async () => {
  const originalQuery = pool.query;

  try {
    pool.query = async () => {
      throw new Error('TEST_DATABASE_FAILURE');
    };

    await assert.rejects(
      getOrderHandler({ order_id: TEST_ORDER_ID }),
      /TEST_DATABASE_FAILURE/
    );

    await assert.rejects(
      checkStockHandler({ sku: TEST_SKU }),
      /TEST_DATABASE_FAILURE/
    );
  } finally {
    pool.query = originalQuery;
  }
});

// 8. Stock mutation tests: explicit opt-in required
test(
  'update_stock adds and removes stock correctly, then restores it',
  { skip: !MUTATION_TESTS_ENABLED },
  async () => {
    await withStockRestored(TEST_SKU, async (initialStock) => {
      const addResult = await updateStockHandler({
        sku: TEST_SKU,
        quantity: 1,
      });

      assert.equal(addResult.isError, undefined, getText(addResult));
      assert.equal(await readStock(TEST_SKU), initialStock + 1);

      const removeResult = await updateStockHandler({
        sku: TEST_SKU,
        quantity: -1,
      });

      assert.equal(removeResult.isError, undefined, getText(removeResult));
      assert.equal(await readStock(TEST_SKU), initialStock);
    });
  }
);

test(
  'update_stock rejects removal beyond available stock',
  { skip: !MUTATION_TESTS_ENABLED },
  async () => {
    await withStockRestored(TEST_SKU, async (initialStock) => {
      const result = await updateStockHandler({
        sku: TEST_SKU,
        quantity: -(initialStock + 1),
      });

      assert.equal(result.isError, true);
      assert.match(getText(result), /Insufficient stock/i);
      assert.equal(await readStock(TEST_SKU), initialStock);
    });
  }
);

test(
  'concurrent stock updates preserve the original total',
  { skip: !MUTATION_TESTS_ENABLED },
  async () => {
    await withStockRestored(TEST_SKU, async (initialStock) => {
      assert.ok(initialStock >= 1, 'Test SKU must have at least 1 unit');

      const results = await Promise.all([
        updateStockHandler({ sku: TEST_SKU, quantity: 1 }),
        updateStockHandler({ sku: TEST_SKU, quantity: -1 }),
      ]);

      for (const result of results) {
        assert.equal(result.isError, undefined, getText(result));
      }

      assert.equal(await readStock(TEST_SKU), initialStock);
    });
  }
);

test('update_stock handles an unknown SKU', async () => {
  const result = await updateStockHandler({
    sku: 'TEST-SKU-DOES-NOT-EXIST-999999',
    quantity: 1,
  });

  assert.equal(result.isError, true);
  assert.match(getText(result), /Product not found/i);
});

test('update_stock schema requires a non-zero integer', () => {
  assert.equal(updateStockInputSchema.quantity.safeParse(0).success, false);
  assert.equal(updateStockInputSchema.quantity.safeParse(2.5).success, false);
});

test.after(async () => {
  await pool.end();
});
