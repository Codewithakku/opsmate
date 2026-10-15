
require('dotenv').config({ path: require('path').join(__dirname, '.env') });

const { Pool } = require('pg');

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL is missing from the MCP .env file');
}

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

pool.on('error', (err) => {
  console.error('[db] Unexpected PostgreSQL pool error:', err.message);
});

module.exports = pool;
