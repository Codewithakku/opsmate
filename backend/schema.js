
// schema.js
// Standalone database schema setup for OpsMate + Order/Inventory MCP
// Run: node schema.js

require("dotenv").config();

const { Pool } = require("pg");

const pool = new Pool({
  connectionString:
    process.env.DATABASE_URL ||
    "postgresql://opsmate:opsmate_password@127.0.0.1:5432/opsmate",
});

const schemaSQL = `
-- =========================================================
-- 1. EXTENSIONS
-- =========================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- =========================================================
-- 2. USERS
-- Individual login + User/Admin roles
-- =========================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT users_role_check
        CHECK (role IN ('user', 'admin'))
);

CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_uniq
    ON users (LOWER(email));


-- =========================================================
-- 3. PRODUCTS
-- =========================================================

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(150) NOT NULL,
    description TEXT,
    price NUMERIC(10, 2) NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT products_price_check
        CHECK (price >= 0),

    CONSTRAINT products_stock_quantity_check
        CHECK (stock_quantity >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS products_sku_uniq
    ON products (sku);


-- =========================================================
-- 4. ORDERS
-- user_id connects each order to its owner.
-- Existing orders remain in the database.
-- =========================================================

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    customer_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    total_amount NUMERIC(10, 2),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,

    CONSTRAINT orders_user_id_fkey
        FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL
);

-- Add ownership column to an existing orders table.
ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS user_id INTEGER;

-- Add the foreign key safely if it does not already exist.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'orders_user_id_fkey'
          AND conrelid = 'public.orders'::regclass
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_user_id_fkey
            FOREIGN KEY (user_id)
            REFERENCES users(id)
            ON DELETE SET NULL;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS orders_user_id_idx
    ON orders (user_id);

CREATE UNIQUE INDEX IF NOT EXISTS orders_order_id_uniq
    ON orders (order_id);


-- =========================================================
-- 5. ORDER ITEMS
-- =========================================================

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL
        REFERENCES orders(id),
    product_id INTEGER NOT NULL
        REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT order_items_quantity_check
        CHECK (quantity > 0),

    CONSTRAINT order_items_unit_price_check
        CHECK (unit_price >= 0),

    CONSTRAINT order_items_order_id_product_id_key
        UNIQUE (order_id, product_id)
);

CREATE INDEX IF NOT EXISTS order_items_order_id_idx
    ON order_items (order_id);

CREATE INDEX IF NOT EXISTS order_items_product_id_idx
    ON order_items (product_id);


-- =========================================================
-- 6. QUERIES
-- Application query history / logs
-- =========================================================

CREATE TABLE IF NOT EXISTS queries (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT,
    model TEXT,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS queries_query_uniq
    ON queries (query);


-- =========================================================
-- 7. QUERY CACHE
-- Exact cache + semantic embedding cache
-- =========================================================

CREATE TABLE IF NOT EXISTS query_cache (
    query_norm TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT,
    prompt_version TEXT,
    hits INTEGER NOT NULL DEFAULT 0,
    latency_ms NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    embedding VECTOR(768),
    embedding_model TEXT DEFAULT 'nomic-embed-text:latest'
);

-- Ensure embedding fields exist in an older query_cache table.
ALTER TABLE query_cache
    ADD COLUMN IF NOT EXISTS embedding VECTOR(768);

ALTER TABLE query_cache
    ADD COLUMN IF NOT EXISTS embedding_model TEXT
    DEFAULT 'nomic-embed-text:latest';

CREATE INDEX IF NOT EXISTS query_cache_expires_at_idx
    ON query_cache (expires_at);

CREATE INDEX IF NOT EXISTS query_cache_embedding_idx
    ON query_cache
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- =========================================================
-- 8. ENSURE EXISTING TABLE COLUMNS
-- Adds any missing expected columns without dropping data.
-- =========================================================

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS sku VARCHAR(50),
    ADD COLUMN IF NOT EXISTS name VARCHAR(150),
    ADD COLUMN IF NOT EXISTS description TEXT,
    ADD COLUMN IF NOT EXISTS price NUMERIC(10, 2),
    ADD COLUMN IF NOT EXISTS stock_quantity INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS order_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS customer_name VARCHAR(100),
    ADD COLUMN IF NOT EXISTS status VARCHAR(50),
    ADD COLUMN IF NOT EXISTS total_amount NUMERIC(10, 2),
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE
        DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE order_items
    ADD COLUMN IF NOT EXISTS order_id INTEGER,
    ADD COLUMN IF NOT EXISTS product_id INTEGER,
    ADD COLUMN IF NOT EXISTS quantity INTEGER,
    ADD COLUMN IF NOT EXISTS unit_price NUMERIC(10, 2),
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

ALTER TABLE queries
    ADD COLUMN IF NOT EXISTS query TEXT,
    ADD COLUMN IF NOT EXISTS answer TEXT,
    ADD COLUMN IF NOT EXISTS model TEXT,
    ADD COLUMN IF NOT EXISTS cached BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS latency_ms NUMERIC,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

`;

async function main() {
  const client = await pool.connect();

  try {
    await client.query("BEGIN");
    await client.query(schemaSQL);
    await client.query("COMMIT");

    console.log("Database schema created/updated successfully.");
    console.log("Tables: users, products, orders, order_items, queries, query_cache");
    console.log("Order ownership column: orders.user_id");
    console.log("Semantic cache: vector(768) + IVFFlat index");
  } catch (error) {
    await client.query("ROLLBACK");

    console.error("Database schema setup failed:", error.message);
    process.exitCode = 1;
  } finally {
    client.release();
    await pool.end();
  }
}

main().catch(async (error) => {
  console.error("Database connection/setup failed:", error.message);
  await pool.end();
  process.exitCode = 1;
});